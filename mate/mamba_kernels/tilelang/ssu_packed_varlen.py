"""Native MUSA tilelang kernel for the packed multi-token selective state update.

This is the MTP / packed-variable-length decode recurrence behind
``mate.mamba.selective_state_update``: one call carries several tokens per
sequence, and every token may write its own state slot.

Why a second kernel rather than a loop inside the one-token one: the one-token
kernel is the plain-decode path, one CTA per ``(dim block, head, sequence)`` with
no loop -- that shape is the whole point of it. A dynamic token loop plus three
device-tensor reads (``cu_seqlens``, ``num_accepted_tokens``, the slot tables)
would tax a path that cannot use them, so the two stay separate and
``mate.mamba`` picks by call shape.

Contract, as the stock Triton ``selective_state_update`` defines it (see
``vllm/model_executor/layers/mamba/ops/mamba_ssm.py``) and as
``mate.testing.operators.mamba`` mirrors it:

- ``N`` sequences are packed into the row axis of ``x``/``dt``/``B``/``C``/``z``/
  ``out``: sequence ``b`` owns rows ``[cu_seqlens[b], cu_seqlens[b + 1])``. A
  sequence of length zero is skipped entirely -- no read, no write;
- the initial state is read once, from ``src_slots[b, max(num_accepted[b] - 1, 0)]``
  when acceptance counts are given (``src_slots[b, 0]`` otherwise), so a step
  resumes from the state the accepted position left behind;
- the slot tables are addressed by **strides**, not by a declared shape. Triton
  reaches a flat table by unsqueezing it to ``[rows, 1]``, which makes both strides
  1, so sequence ``b``'s position ``t`` is entry ``b + t`` for a flat table and
  ``b * steps + t`` for a contiguous 2-D ``[sequences, steps]`` one. vLLM's MTP6
  call passes the flat form -- ``state_batch_indices`` and
  ``dst_state_batch_indices`` are ``[rows]``, one entry per packed row;
- with acceptance counts, **every** token stores its evolving state to
  ``dst_slots[b, i_t]``, or to ``src_slots[b, i_t]`` when no destination table is
  given (the Triton default), skipping slots equal to ``pad_slot_id``. That chain is
  what lets the next call start mid-sequence instead of recomputing the accepted
  prefix, so it is semantics, not an optimization;
- without acceptance counts, the state is stored once after the token loop, to
  ``dst_slots[b, 0]``, or to the read slot when no destination table is given;
- ``disable_state_update`` suppresses every store.

Numerics are the same recurrence as the one-token kernel, in the same order and at
the same rounding points as the oracle: ``state = state * dA + (dt * B) * x``, then
``out = sum_n(state * C)``, then ``+ x * D``, then the z gate, with a plain
(non-stochastic) rounding on the store. Under MTP the state feeds the draft model,
so a reassociation here moves acceptance length and the served output, not just a
tolerance: it would have to be justified with evidence rather than assumed
harmless. The state therefore stays in an fp32 register fragment across the whole
token loop and is rounded only where it is written, exactly like the one-token
kernel.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["prewarm_ssu_packed_varlen", "ssu_packed_launch"]

SOFTPLUS_THRESHOLD = 20.0

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}

_COMPILE_FLAGS = [
    "-O3",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


#: Elements per 16-byte vector -- the widest load the target issues -- by dtype.
_ALIGN_ELEMS = {torch.float16: 8, torch.bfloat16: 8, torch.float32: 4}


def _align_elems(dtype: torch.dtype, extent: int) -> int:
    """The stride alignment, in elements, the kernel asserts for one input.

    ``extent`` is the length of the axis a strided input is walked along. A 16-byte
    load is the widest this target issues and a load is never wider than the axis it
    walks, so the alignment a row start has to keep is the narrower of the two: a
    bf16 axis of 12 elements is read in 8-byte steps, not 16-byte ones.
    """
    align = _ALIGN_ELEMS.get(dtype)
    if align is None:
        raise RuntimeError(
            f"a strided input of dtype {dtype} has no defined 16-byte alignment."
        )
    return min(align, (extent & -extent) or 1)


def _require_aligned(name: str, tensor: torch.Tensor) -> None:
    """Refuse an outer stride that the kernel's ``T.assume`` lines would deny.

    ``T.assume`` is a compiler hint rather than a check, and a hint that does not
    hold is undefined behaviour, so the alignment the kernel asserts for ``tensor``'s
    outer strides has to be enforced here. A strided input is walked along its last
    axis, so that axis supplies the extent. An axis of extent 1 is exempt: nothing is
    ever addressed along it, so its stride cannot misalign a row start. Aligned
    strides over a misaligned base would still issue misaligned vector loads, so
    the storage offset is checked too.
    """
    align = _align_elems(tensor.dtype, tensor.shape[-1])
    if tensor.storage_offset() % align != 0:
        raise RuntimeError(
            f"{name} must start at a storage offset that is a multiple of {align} "
            f"elements ({align * tensor.element_size()}-byte alignment for "
            f"{tensor.dtype}); got offset {tensor.storage_offset()}."
        )
    for axis, stride in enumerate(tensor.stride()[:-1]):
        if tensor.shape[axis] != 1 and stride % align != 0:
            raise RuntimeError(
                f"{name} stride({axis}) must be a multiple of {align} elements "
                f"({align * tensor.element_size()}-byte alignment for "
                f"{tensor.dtype}); got {stride}."
            )


@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def tilelang_ssu_packed_varlen(
    state_dtype,
    io_dtype,
    slot_dtype,
    meta_dtype,
    dim,
    dstate,
    head_ratio,
    rows_per_cta,
    lanes_per_row,
    dt_dtype=None,
    dt_dim=None,
    a_dim=None,
    d_dim=None,
    b_dim=None,
):
    """Build the packed SSU kernel for one (dtype, shape, config) tuple.

    ``dt_dim``/``a_dim``/``d_dim``/``b_dim`` carry the same meaning as in
    ``ssu_one_token``: ``dt_dim == 1`` is the consumers' per-head step (broadcast
    over ``dim``), and a channel dimension of 1 is the tied per-head parameter
    form, so one compiled kernel covers both layouts.
    """
    if dt_dtype is None:
        dt_dtype = torch.float32
    if dt_dim is None:
        dt_dim = dim
    if a_dim is None:
        a_dim = 1
    if d_dim is None:
        d_dim = 1
    if b_dim is None:
        b_dim = 1
    rows = T.dynamic("rows")
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    groups = T.dynamic("groups")
    slots = T.dynamic("slots")
    # Each slot table is a flat buffer addressed by two strides the caller passes,
    # not by a declared shape: Triton reaches a flat table by unsqueezing it to
    # [rows, 1], which makes both strides 1, and that is the form vLLM's MTP call
    # hands over. The two tables may have different lengths, so each one gets its own
    # extent symbol for the declaration and an integer argument for the bounds check.
    src_elems = T.dynamic("src_elems")
    dst_elems = T.dynamic("dst_elems")
    # The query-start metadata is declared with its own dynamic length rather than
    # `batch + 1`: the arithmetic relation is a caller contract the launcher checks
    # on the shapes, and keeping every axis an independent dynamic symbol avoids
    # depending on how shape expressions survive lowering.
    meta_rows = T.dynamic("meta_rows")

    vec = tilelang.cdiv(dstate, lanes_per_row)
    num_shuffles = lanes_per_row.bit_length() - 1
    threads = rows_per_cta * lanes_per_row
    dim_blocks = tilelang.cdiv(dim, rows_per_cta)

    #: ``x``, ``dt``, ``B``, ``C`` and ``z`` are read-only inputs, so they are declared
    #: strided: their row axis may be a strided view of a wider buffer. ``B``/``C`` are
    #: the tensors the lane-vectorized loads walk, so their outer strides also carry
    #: the alignment the kernel assumes; ``x``, ``dt`` and ``z`` are read one element
    #: per thread, so no vector load depends on their strides and a hint would only
    #: narrow which views are legal. ``state`` is read and written in place by the pool
    #: update, ``A``/``Dv``/``dt_bias`` are the per-head matrices this launcher
    #: materializes itself, the slot tables plus ``cu_seqlens``/``num_accepted`` are
    #: caller-built metadata and ``out`` is the store target: all stay dense.
    b_stride_group = T.dynamic("b_stride_group")
    b_stride_row = T.dynamic("b_stride_row")
    c_stride_group = T.dynamic("c_stride_group")
    c_stride_row = T.dynamic("c_stride_row")
    x_strides = (T.dynamic("x_stride_row"), T.dynamic("x_stride_head"), 1)
    dt_strides = (T.dynamic("dt_stride_row"), T.dynamic("dt_stride_head"), 1)
    b_strides = (b_stride_row, b_stride_group, 1)
    c_strides = (c_stride_row, c_stride_group, 1)
    z_strides = (T.dynamic("z_stride_row"), T.dynamic("z_stride_head"), 1)
    io_align = _align_elems(io_dtype, dstate)

    @T.prim_func
    def tilelang_ssu_packed_varlen_kernel(
        state: T.Tensor((slots, heads, dim, dstate), state_dtype),
        x: T.StridedTensor((rows, heads, dim), x_strides, io_dtype),
        dt: T.StridedTensor((rows, heads, dt_dim), dt_strides, dt_dtype),
        A: T.Tensor((heads, a_dim), "float32"),
        B: T.StridedTensor((rows, groups, dstate), b_strides, io_dtype),
        C: T.StridedTensor((rows, groups, dstate), c_strides, io_dtype),
        Dv: T.Tensor((heads, d_dim), "float32"),
        dt_bias: T.Tensor((heads, b_dim), "float32"),
        z: T.StridedTensor((rows, heads, dim), z_strides, io_dtype),
        src_slots: T.Tensor((src_elems,), slot_dtype),
        src_seq_stride: T.int32,
        src_step_stride: T.int32,
        src_length: T.int32,
        dst_slots: T.Tensor((dst_elems,), slot_dtype),
        dst_seq_stride: T.int32,
        dst_step_stride: T.int32,
        dst_length: T.int32,
        cu_seqlens: T.Tensor((meta_rows,), meta_dtype),
        num_accepted: T.Tensor((batch,), meta_dtype),
        out: T.Tensor((rows, heads, dim), io_dtype),
        pad_slot_id: T.int32,
        use_dt_bias: T.int32,
        use_z: T.int32,
        has_D: T.int32,
        dt_softplus: T.int32,
        write_state: T.int32,
        use_accepted: T.int32,
        use_dst_table: T.int32,
    ):
        # Same grid as the one-token kernel: a CTA owns `rows_per_cta` consecutive
        # `dim` rows of one head of one sequence, and iterates that sequence's
        # tokens. Parallelism therefore scales with the decode batch, not with the
        # speculative depth.
        with T.Kernel(dim_blocks, heads, batch, threads=threads) as (
            d_block,
            head,
            batch_idx,
        ):
            T.assume(b_stride_row % io_align == 0)
            T.assume(b_stride_group % io_align == 0)
            T.assume(c_stride_row % io_align == 0)
            T.assume(c_stride_group % io_align == 0)

            tid = T.get_thread_binding()
            lane = tid % lanes_per_row
            row_in_block = tid // lanes_per_row
            d = d_block * rows_per_cta + row_in_block
            group = head // head_ratio

            state_row = T.alloc_local((vec,), dtype="float32")
            b_val = T.alloc_local((vec,), dtype="float32")
            c_val = T.alloc_local((vec,), dtype="float32")
            out_acc = T.alloc_local((1,), dtype="float32")
            dt_value = T.alloc_var("float32")
            decay = T.alloc_var("float32")
            x_value = T.alloc_var("float32")
            row_begin = T.alloc_var("int32")
            seq_len = T.alloc_var("int32")
            seed = T.alloc_var("int32")
            src = T.alloc_var("int32")
            dst = T.alloc_var("int32")
            slot_index = T.alloc_var("int32")
            token_dst = T.alloc_var("int32")

            if d < dim:
                row_begin = T.cast(cu_seqlens[batch_idx], "int32")
                seq_len = T.cast(cu_seqlens[batch_idx + 1], "int32") - row_begin

                # An empty sequence has no state to read and none to write; the
                # Triton kernel returns before touching anything, and a store here
                # would invent an update the caller never asked for.
                if seq_len > 0:
                    seed = 0
                    if use_accepted == 1:
                        # The accepted token is where the previous step stopped, so
                        # its own state is the starting point: count - 1, floored at
                        # the zero-th slot because a count of 0 is a legal "nothing
                        # accepted" sentinel.
                        seed = T.max(T.cast(num_accepted[batch_idx], "int32") - 1, 0)

                    # The table is addressed by strides rather than by a shape: this
                    # sequence's starting position is entry
                    # ``batch_idx * seq_stride + seed * step_stride``. vLLM's MTP call
                    # passes a flat [rows] table, whose strides are both 1 (that is
                    # what Triton's unsqueeze(1) produces), so the entry is simply
                    # ``row + seed`` -- and with one sequence in the batch, the
                    # accepted position itself.
                    #
                    # Branch around the read instead of selecting: a select evaluates
                    # both sides, so the never-taken side would still form the
                    # (possibly out-of-range) address. A caller can pass a table that
                    # does not cover every (sequence, position) pair; reading past it
                    # is a memory fault with bounds checks off, so a slot with no
                    # entry reads as "no state" instead.
                    src = pad_slot_id
                    slot_index = batch_idx * src_seq_stride + seed * src_step_stride
                    if slot_index < src_length:
                        src = T.cast(src_slots[slot_index], "int32")

                    if use_dst_table == 1:
                        dst = pad_slot_id
                        slot_index = batch_idx * dst_seq_stride
                        if slot_index < dst_length:
                            dst = T.cast(dst_slots[slot_index], "int32")
                    else:
                        # No destination table: the state stays where it was read
                        # from. For a single token per sequence that is the
                        # in-place update the one-token kernel performs.
                        dst = src

                    if src == pad_slot_id:
                        for i in T.vectorized(vec):
                            state_row[i] = 0.0
                    else:
                        for i in T.vectorized(vec):
                            state_row[i] = T.cast(
                                state[src, head, d, lane * vec + i], "float32"
                            )

                    for i_t in T.serial(seq_len):
                        row = row_begin + i_t

                        # dt -> decay, shared by every dstate lane of this row.
                        dt_value = T.cast(
                            dt[row, head, 0 if dt_dim == 1 else d], "float32"
                        )
                        if use_dt_bias == 1:
                            dt_value = dt_value + dt_bias[head, 0 if b_dim == 1 else d]
                        if dt_softplus == 1:
                            dt_value = T.if_then_else(
                                dt_value > SOFTPLUS_THRESHOLD,
                                dt_value,
                                T.log(1.0 + T.exp(dt_value)),
                            )
                        # A is either per-head (a_dim == 1, the tied form) or
                        # per-(head, dim): the consumers pass the latter as a
                        # [heads, dim, dstate] view whose last stride is zero, so
                        # the column read here is exact for both.
                        decay = T.exp(A[head, 0 if a_dim == 1 else d] * dt_value)
                        x_value = T.cast(x[row, head, d], "float32")

                        # B/C are shared by every (dim, dstate) element of this head.
                        for i in T.vectorized(vec):
                            b_val[i] = T.cast(B[row, group, lane * vec + i], "float32")
                            c_val[i] = T.cast(C[row, group, lane * vec + i], "float32")

                        out_acc[0] = 0.0
                        # Operand order copied from the one-token kernel and from
                        # the oracle, which is the contract for MTP numerics.
                        for i in T.unroll(vec):
                            state_row[i] = (
                                state_row[i] * decay + dt_value * b_val[i] * x_value
                            )
                            out_acc[0] += state_row[i] * c_val[i]

                        for offset in T.unroll(num_shuffles):
                            out_acc[0] += T.shfl_xor(
                                out_acc[0], (lanes_per_row // 2) >> offset
                            )

                        if use_accepted == 1:
                            # Every token publishes its own state slot, so the next
                            # call can resume from the accepted position. With no
                            # destination table the target is the read table itself,
                            # which is the default the Triton kernel applies when it
                            # aliases ``dst_state_batch_indices`` onto
                            # ``state_batch_indices``. A table that does not cover the
                            # position stores nothing rather than writing an unrelated
                            # slot.
                            token_dst = pad_slot_id
                            if use_dst_table == 1:
                                slot_index = (
                                    batch_idx * dst_seq_stride + i_t * dst_step_stride
                                )
                                if slot_index < dst_length:
                                    token_dst = T.cast(dst_slots[slot_index], "int32")
                            else:
                                slot_index = (
                                    batch_idx * src_seq_stride + i_t * src_step_stride
                                )
                                if slot_index < src_length:
                                    token_dst = T.cast(src_slots[slot_index], "int32")
                            if write_state == 1 and token_dst != pad_slot_id:
                                for i in T.vectorized(vec):
                                    state[token_dst, head, d, lane * vec + i] = T.cast(
                                        state_row[i], state_dtype
                                    )

                        if lane == 0:
                            if has_D == 1:
                                out_acc[0] += Dv[head, 0 if d_dim == 1 else d] * x_value
                            if use_z == 1:
                                out_acc[0] = (
                                    out_acc[0]
                                    * T.cast(z[row, head, d], "float32")
                                    * T.sigmoid(T.cast(z[row, head, d], "float32"))
                                )
                            out[row, head, d] = T.cast(out_acc[0], io_dtype)

                    if use_accepted == 0:
                        if write_state == 1 and dst != pad_slot_id:
                            for i in T.vectorized(vec):
                                state[dst, head, d, lane * vec + i] = T.cast(
                                    state_row[i], state_dtype
                                )

    return tilelang_ssu_packed_varlen_kernel


def _validate_launch(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    num_accepted: torch.Tensor | None,
    dstate: int,
    rows_per_cta: int,
    lanes_per_row: int,
) -> None:
    if dstate % lanes_per_row != 0:
        raise RuntimeError(
            f"dstate {dstate} must be divisible by lanes_per_row {lanes_per_row}."
        )
    if lanes_per_row != 32:
        raise RuntimeError(
            "the native SSU kernel assigns one dim row per warp so that the "
            "`d < dim` guard stays warp-uniform while the lane group shuffles; "
            f"lanes_per_row must be 32, got {lanes_per_row}. A sub-warp variant "
            "must hoist the reduction out of the guard first."
        )
    if rows_per_cta < 1 or rows_per_cta * lanes_per_row > 1024:
        raise RuntimeError(
            "the native SSU kernel requires 1 <= rows_per_cta * lanes_per_row <= 1024."
        )
    if x.stride(-1) != 1:
        raise RuntimeError(
            "x must have a dense dim axis (stride(-1) == 1); its outer axes may be "
            "strided."
        )
    # dt's last axis is either the dim it is read along or the extent-1 axis of the
    # per-head broadcast form, which is never advanced and so is exempt.
    if dt.shape[-1] != 1 and dt.stride(-1) != 1:
        raise RuntimeError(
            "dt must have a dense dim axis (stride(-1) == 1); the per-head broadcast "
            "form keeps its extent-1 axis."
        )
    # The pool is read and written in place, so it must be the dense
    # [slots, heads, dim, dstate] span the pool update addresses.
    if state.stride() != (
        state.shape[1] * state.shape[2] * dstate,
        state.shape[2] * dstate,
        dstate,
        1,
    ):
        raise RuntimeError(
            "state must be a contiguous [slots, heads, dim, dstate] pool."
        )
    if dt.shape != (x.shape[0], x.shape[1], dt.shape[2]) or dt.shape[2] not in (
        1,
        x.shape[2],
    ):
        raise RuntimeError(
            "dt must be [rows, heads, dim] or the per-head broadcast [rows, heads, 1]; "
            f"got {tuple(dt.shape)} for x={tuple(x.shape)}."
        )
    if dt.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise RuntimeError(f"dt must be fp32, bf16 or fp16; got {dt.dtype}.")
    if A.dtype != torch.float32:
        raise RuntimeError("A must be fp32 for the native SSU kernel.")
    if B.shape[1] == 0 or state.shape[1] % B.shape[1] != 0:
        raise RuntimeError("B/C groups must divide heads for the native SSU kernel.")
    # The kernel walks B/C along the pool's dstate axis, which is also the extent the
    # stride alignment is derived from: a row that is shorter than the pool's would
    # read past its own row, and one that is longer would make the hint too strong.
    if B.shape[0] != x.shape[0] or B.shape[2] != dstate:
        raise RuntimeError(
            f"B/C must be [rows={x.shape[0]}, groups, dstate={dstate}], got "
            f"{tuple(B.shape)}."
        )
    # The tables are flat buffers plus two caller-supplied strides, so a 1-D table
    # carries no sequence count and cannot be checked against the batch: the split
    # comes from cu_seqlens, and how far a read or a store reaches is bounded by the
    # table's own length in the kernel.
    for name, tensor in (("src_slots", src_slots), ("dst_slots", dst_slots)):
        if tensor is not None and tensor.dim() not in (1, 2):
            raise RuntimeError(
                f"{name} must be the flat [rows] or the 2-D [sequences, steps] slot "
                f"table; got {tuple(tensor.shape)}."
            )
    batch = cu_seqlens.numel() - 1
    if batch < 1:
        raise RuntimeError(
            "cu_seqlens must hold one start per sequence plus the total; got "
            f"{cu_seqlens.numel()} entries."
        )
    if cu_seqlens.numel() != batch + 1:
        raise RuntimeError(
            "cu_seqlens must hold one start per sequence plus the total: expected "
            f"{batch + 1} entries, got {cu_seqlens.numel()}."
        )
    if num_accepted is not None and num_accepted.numel() != batch:
        raise RuntimeError(
            f"num_accepted_tokens must hold one count per sequence ({batch}), got "
            f"{num_accepted.numel()}."
        )
    # All four are caller-built metadata that the kernel reads as dense spans.
    for name, tensor in (
        ("src_slots", src_slots),
        ("dst_slots", dst_slots),
        ("cu_seqlens", cu_seqlens),
        ("num_accepted", num_accepted),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous for the packed SSU kernel.")


def _channel_dim(tensor: torch.Tensor | None, dim: int) -> int:
    """1 for a per-head vector, ``dim`` for the per-(head, dim) matrix form."""
    if tensor is None:
        return 1
    if tensor.dim() == 1:
        return 1
    if tensor.dim() == 2 and tensor.shape[1] == 1:
        # [heads, 1] is the per-head form kept as a broadcast view; the wrapper
        # passes it instead of materializing a [heads, dim] copy.
        return 1
    if tensor.dim() == 2 and tensor.shape[1] == dim:
        return dim
    raise RuntimeError(
        f"per-head parameter must be [heads] or [heads, {dim}], got {tuple(tensor.shape)}."
    )


def _kernel_for(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    src_slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    rows_per_cta: int,
    lanes_per_row: int,
    a_dim: int = 1,
    d_dim: int = 1,
    b_dim: int = 1,
):
    return tilelang_ssu_packed_varlen(
        state_dtype=state.dtype,
        io_dtype=x.dtype,
        slot_dtype=src_slots.dtype,
        meta_dtype=cu_seqlens.dtype,
        dim=state.shape[2],
        dstate=state.shape[3],
        head_ratio=state.shape[1] // B.shape[1],
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
        dt_dtype=dt.dtype,
        dt_dim=dt.shape[2],
        a_dim=a_dim,
        d_dim=d_dim,
        b_dim=b_dim,
    )


def prewarm_ssu_packed_varlen(
    *,
    state_dtype: torch.dtype,
    io_dtype: torch.dtype,
    slot_dtype: torch.dtype,
    meta_dtype: torch.dtype,
    dim: int,
    dstate: int,
    head_ratio: int,
    rows_per_cta: int = 4,
    lanes_per_row: int = 32,
    dt_dtype: torch.dtype = torch.float32,
    dt_dim: int | None = None,
    a_dim: int = 1,
    d_dim: int = 1,
    b_dim: int = 1,
) -> None:
    """Compile one packed kernel configuration without launching it."""
    tilelang_ssu_packed_varlen(
        state_dtype=state_dtype,
        io_dtype=io_dtype,
        slot_dtype=slot_dtype,
        meta_dtype=meta_dtype,
        dim=dim,
        dstate=dstate,
        head_ratio=head_ratio,
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
        dt_dtype=dt_dtype,
        dt_dim=dt_dim,
        a_dim=a_dim,
        d_dim=d_dim,
        b_dim=b_dim,
    )


_DUMMY_CACHE: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}


def _dummy(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return a zeroed placeholder for an absent optional input.

    Zeroed rather than ``empty``: a select compiles to a blend, so the
    never-taken side is evaluated, and garbage from an uninitialized buffer can
    reach the result -- NaN included. One cached buffer per (shape, dtype,
    device) keeps the op at a single launch and allocation.
    """
    key = (shape, dtype, str(device))
    tensor = _DUMMY_CACHE.get(key)
    if tensor is None:
        tensor = torch.zeros(shape, device=device, dtype=dtype)
        _DUMMY_CACHE[key] = tensor
    return tensor


def ssu_packed_launch(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    Dv: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    z: torch.Tensor | None,
    src_slots: torch.Tensor,
    src_seq_stride: int,
    src_step_stride: int,
    dst_slots: torch.Tensor | None,
    dst_seq_stride: int,
    dst_step_stride: int,
    cu_seqlens: torch.Tensor,
    num_accepted: torch.Tensor | None,
    out: torch.Tensor,
    *,
    dt_softplus: bool,
    pad_slot_id: int,
    disable_state_update: bool,
    rows_per_cta: int = 4,
    lanes_per_row: int = 32,
) -> torch.Tensor:
    """Run the packed native SSU kernel.

    ``mate.mamba.selective_state_update`` has already validated the contract:
    packed rows, per-head (tied) ``A``/``D``/``dt_bias`` in fp32, per-head ``dt``,
    contiguous metadata, and a flat or 2-D slot table per side. The two strides say
    how the kernel reaches sequence ``b``'s position ``t`` in each table: from vLLM's
    flat ``state_batch_indices`` it is entry ``b + t`` (strides 1 and 1), from a
    contiguous 2-D ``[sequences, steps]`` table it is ``b * steps + t``.

    ``x``, ``dt``, ``B``, ``C`` and ``z`` may be strided views of a wider buffer;
    ``B`` and ``C`` additionally have to keep the row alignment the kernel's
    vectorized loads assume.
    """
    heads, dim = x.shape[1], x.shape[2]
    slots, state_heads, state_dim, dstate = state.shape
    if state_heads != heads or state_dim != dim:
        raise RuntimeError("state must match x as [slots, heads, dim, dstate].")
    if dst_slots is not None and dst_slots.dtype != src_slots.dtype:
        raise RuntimeError("source and destination slot tables must share a dtype.")
    # B and C are the lane-vectorized operands, so their outer strides also carry the
    # alignment the kernel assumes; z is read one element per thread.
    if B.stride(-1) != 1 or C.stride(-1) != 1:
        raise RuntimeError(
            "B and C must have a dense dstate axis (stride(-1) == 1); their row axis "
            "may be strided."
        )
    if z is not None and z.stride(-1) != 1:
        raise RuntimeError(
            "z must have a dense dim axis (stride(-1) == 1); its outer axes may be "
            "strided."
        )
    if not out.is_contiguous():
        # The kernel declares out as a dense [batch, heads, dim] store target, so a
        # strided view would be written at the wrong addresses rather than refused.
        raise RuntimeError(
            "out must be contiguous: the SSU kernel writes it as a dense "
            "[batch, heads, dim] buffer."
        )
    _require_aligned("B", B)
    _require_aligned("C", C)
    if any(
        tensor is not None and not tensor.is_contiguous()
        for tensor in (Dv, dt_bias, src_slots, dst_slots, cu_seqlens, num_accepted)
    ):
        raise RuntimeError(
            "D, dt_bias, the slot tables and the sequence metadata must be contiguous."
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}."
        )
    if num_accepted is not None and num_accepted.dtype != cu_seqlens.dtype:
        raise RuntimeError(
            "cu_seqlens and num_accepted_tokens must share a dtype for the packed "
            f"SSU kernel; got {cu_seqlens.dtype} and {num_accepted.dtype}."
        )
    _validate_launch(
        state,
        x,
        dt,
        A,
        B,
        src_slots,
        dst_slots,
        cu_seqlens,
        num_accepted,
        dstate,
        rows_per_cta,
        lanes_per_row,
    )
    A = A.reshape(heads, _channel_dim(A, dim)).contiguous()
    if Dv is not None:
        Dv = Dv.reshape(heads, _channel_dim(Dv, dim)).contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.reshape(heads, _channel_dim(dt_bias, dim)).contiguous()

    batch = cu_seqlens.numel() - 1
    kernel = _kernel_for(
        state,
        x,
        dt,
        B,
        src_slots,
        cu_seqlens,
        a_dim=_channel_dim(A, dim),
        d_dim=_channel_dim(Dv, dim),
        b_dim=_channel_dim(dt_bias, dim),
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
    )

    kernel(
        state,
        x,
        dt,
        A,
        B,
        C,
        Dv
        if Dv is not None
        else _dummy((heads, _channel_dim(Dv, dim)), torch.float32, x.device),
        dt_bias
        if dt_bias is not None
        else _dummy((heads, _channel_dim(dt_bias, dim)), torch.float32, x.device),
        z if z is not None else _dummy((x.shape[0], heads, dim), x.dtype, x.device),
        src_slots,
        int(src_seq_stride),
        int(src_step_stride),
        int(src_slots.numel()),
        dst_slots
        if dst_slots is not None
        else _dummy(tuple(src_slots.shape), src_slots.dtype, x.device),
        int(dst_seq_stride),
        int(dst_step_stride),
        int(src_slots.numel()) if dst_slots is None else int(dst_slots.numel()),
        cu_seqlens,
        num_accepted
        if num_accepted is not None
        else _dummy((batch,), cu_seqlens.dtype, x.device),
        out,
        int(pad_slot_id),
        1 if dt_bias is not None else 0,
        1 if z is not None else 0,
        1 if Dv is not None else 0,
        1 if dt_softplus else 0,
        0 if disable_state_update else 1,
        1 if num_accepted is not None else 0,
        1 if dst_slots is not None else 0,
    )
    return out
