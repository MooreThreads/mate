"""Native MUSA tilelang kernel for the one-token selective state update (SSU).

This is the S5000 decode recurrence behind ``mate.mamba.selective_state_update``
for the shapes the Mamba2/Nemotron consumers use: state ``[slots, heads, dim,
dstate]``, one token per sequence, and per-head ``A``, ``D`` and ``dt_bias``
broadcasting over ``dim``/``dstate``.

Dataflow (one CTA owns ``rows_per_cta`` consecutive ``dim`` rows of one head):
each lane owns ``dstate / lanes_per_row`` state values, stages ``B``/``C`` once
per head, and keeps the whole recurrence in fp32. The per-row output reduction is
a warp shuffle tree, so no shared-memory staging or cross-warp synchronization is
involved.

Optional inputs are selected by runtime flags rather than compile-time
specialization, so one compiled kernel covers every flag combination and
prewarming stays meaningful. The ``pad_slot_id``/``disable_state_update``
contract is honoured in kernel: a padded source reads as zero state and a padded
or disabled destination is not written.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["prewarm_ssu_one_token", "ssu_one_token_launch"]

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
def tilelang_ssu_one_token(
    state_dtype,
    io_dtype,
    slot_dtype,
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
    """Build the one-token SSU kernel for one (dtype, shape, config) tuple.

    ``dt_dim`` is ``1`` when the caller passes the per-head step instead of a
    [batch, heads, dim] tensor. vLLM expands it across head_dim, and the two
    forms carry identical values, so reading one element per head lets the caller
    hand over the unexpanded buffer and skip the copy that a widening cast on the
    decode path would otherwise cost per mixer per step.
    """
    if dt_dtype is None:
        dt_dtype = torch.float32
    if dt_dim is None:
        dt_dim = dim
    # A, D and dt_bias are per-head for a tied model and per-(head, dim) for a
    # per-channel one. The consumers signal the second form with a [heads, dim,
    # dstate] view whose last stride is zero, and the first with a [heads] vector,
    # so a dim of 1 here means "one value per head, broadcast over head_dim".
    if a_dim is None:
        a_dim = 1
    if d_dim is None:
        d_dim = 1
    if b_dim is None:
        b_dim = 1
    batch = T.dynamic("batch")
    heads = T.dynamic("heads")
    groups = T.dynamic("groups")
    slots = T.dynamic("slots")

    vec = tilelang.cdiv(dstate, lanes_per_row)
    num_shuffles = lanes_per_row.bit_length() - 1
    threads = rows_per_cta * lanes_per_row
    dim_blocks = tilelang.cdiv(dim, rows_per_cta)

    #: ``x``, ``dt``, ``B``, ``C`` and ``z`` are read-only inputs, so they are declared
    #: strided: their batch axis may be a strided view of a wider buffer. ``B``/``C``
    #: are the tensors the lane-vectorized loads walk, so their outer strides also
    #: carry the alignment the kernel assumes; ``x``, ``dt`` and ``z`` are read one
    #: element per thread, so no vector load depends on their strides and a hint would
    #: only narrow which views are legal. ``state`` is read and written in place by the
    #: pool update, ``A``/``Dv``/``dt_bias`` are the per-head matrices this launcher
    #: materializes itself, ``src_slots``/``dst_slots`` are caller-built metadata and
    #: ``out`` is the store target: all stay dense ``T.Tensor``.
    b_stride_batch = T.dynamic("b_stride_batch")
    b_stride_group = T.dynamic("b_stride_group")
    c_stride_batch = T.dynamic("c_stride_batch")
    c_stride_group = T.dynamic("c_stride_group")
    x_strides = (T.dynamic("x_stride_batch"), T.dynamic("x_stride_head"), 1)
    dt_strides = (T.dynamic("dt_stride_batch"), T.dynamic("dt_stride_head"), 1)
    b_strides = (b_stride_batch, b_stride_group, 1)
    c_strides = (c_stride_batch, c_stride_group, 1)
    z_strides = (T.dynamic("z_stride_batch"), T.dynamic("z_stride_head"), 1)
    io_align = _align_elems(io_dtype, dstate)

    @T.prim_func
    def tilelang_ssu_one_token_kernel(
        state: T.Tensor((slots, heads, dim, dstate), state_dtype),
        x: T.StridedTensor((batch, heads, dim), x_strides, io_dtype),
        dt: T.StridedTensor((batch, heads, dt_dim), dt_strides, dt_dtype),
        A: T.Tensor((heads, a_dim), "float32"),
        B: T.StridedTensor((batch, groups, dstate), b_strides, io_dtype),
        C: T.StridedTensor((batch, groups, dstate), c_strides, io_dtype),
        Dv: T.Tensor((heads, d_dim), "float32"),
        dt_bias: T.Tensor((heads, b_dim), "float32"),
        z: T.StridedTensor((batch, heads, dim), z_strides, io_dtype),
        src_slots: T.Tensor((batch,), slot_dtype),
        dst_slots: T.Tensor((batch,), slot_dtype),
        out: T.Tensor((batch, heads, dim), io_dtype),
        pad_slot_id: T.int32,
        use_dt_bias: T.int32,
        use_z: T.int32,
        has_D: T.int32,
        dt_softplus: T.int32,
        write_state: T.int32,
    ):
        with T.Kernel(dim_blocks, heads, batch, threads=threads) as (
            d_block,
            head,
            batch_idx,
        ):
            T.assume(b_stride_batch % io_align == 0)
            T.assume(b_stride_group % io_align == 0)
            T.assume(c_stride_batch % io_align == 0)
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
            src = T.alloc_var("int32")
            dst = T.alloc_var("int32")

            if d < dim:
                src = T.cast(src_slots[batch_idx], "int32")
                dst = T.cast(dst_slots[batch_idx], "int32")

                # dt -> decay, shared by every dstate lane of this row.
                # dt_dim == 1 is the broadcast form: every d of this head reads
                # the same value, so the unexpanded buffer can be used as-is.
                dt_value = T.cast(dt[batch_idx, head, 0 if dt_dim == 1 else d], "float32")
                if use_dt_bias == 1:
                    dt_value = dt_value + dt_bias[head, 0 if b_dim == 1 else d]
                if dt_softplus == 1:
                    dt_value = T.if_then_else(
                        dt_value > SOFTPLUS_THRESHOLD,
                        dt_value,
                        T.log(1.0 + T.exp(dt_value)),
                    )
                # A is either per-head (a_dim == 1, the tied form) or per-(head, dim):
                # the consumers pass the latter as a [heads, dim, dstate] view whose
                # last stride is zero, so the column read here is exact for both.
                decay = T.exp(A[head, 0 if a_dim == 1 else d] * dt_value)
                x_value = T.cast(x[batch_idx, head, d], "float32")

                # B/C are shared by every (dim, dstate) element of this head.
                for i in T.vectorized(vec):
                    b_val[i] = T.cast(B[batch_idx, group, lane * vec + i], "float32")
                    c_val[i] = T.cast(C[batch_idx, group, lane * vec + i], "float32")

                # A padded source slot has no state: branch around the load instead
                # of relying on a select, so no out-of-range address is formed and
                # the load stays vectorized.
                if src == pad_slot_id:
                    for i in T.vectorized(vec):
                        state_row[i] = 0.0
                else:
                    for i in T.vectorized(vec):
                        state_row[i] = T.cast(
                            state[src, head, d, lane * vec + i], "float32"
                        )

                out_acc[0] = 0.0
                for i in T.unroll(vec):
                    state_row[i] = state_row[i] * decay + dt_value * b_val[i] * x_value
                    out_acc[0] += state_row[i] * c_val[i]

                for offset in T.unroll(num_shuffles):
                    out_acc[0] += T.shfl_xor(out_acc[0], (lanes_per_row // 2) >> offset)

                if write_state == 1 and dst != pad_slot_id:
                    for i in T.vectorized(vec):
                        state[dst, head, d, lane * vec + i] = T.cast(
                            state_row[i], state_dtype
                        )

                if lane == 0:
                    if has_D == 1:
                        out_acc[0] += Dv[head, 0 if d_dim == 1 else d] * x_value
                    if use_z == 1:
                        out_acc[0] = (
                            out_acc[0]
                            * T.cast(z[batch_idx, head, d], "float32")
                            * T.sigmoid(T.cast(z[batch_idx, head, d], "float32"))
                        )
                    out[batch_idx, head, d] = T.cast(out_acc[0], io_dtype)

    return tilelang_ssu_one_token_kernel


def _validate_launch(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
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
            "dt must be [batch, heads, dim] or the per-head broadcast [batch, heads, 1]; "
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


def _channel_dim(tensor: torch.Tensor | None, dim: int) -> int:
    """1 for a per-head vector, ``dim`` for the per-(head, dim) matrix form."""
    if tensor is None:
        return 1
    if tensor.dim() == 1:
        return 1
    if tensor.dim() == 2 and tensor.shape[1] == 1:
        # [heads, 1] is the per-head form: the wrapper keeps the caller's broadcast
        # view instead of materializing a [heads, dim] copy, and says so with a
        # trailing axis of one.
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
    *,
    rows_per_cta: int,
    lanes_per_row: int,
    a_dim: int = 1,
    d_dim: int = 1,
    b_dim: int = 1,
):
    return tilelang_ssu_one_token(
        state_dtype=state.dtype,
        io_dtype=x.dtype,
        slot_dtype=src_slots.dtype,
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


def prewarm_ssu_one_token(
    *,
    state_dtype: torch.dtype,
    io_dtype: torch.dtype,
    slot_dtype: torch.dtype,
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
    """Compile one kernel configuration without launching it."""
    tilelang_ssu_one_token(
        state_dtype=state_dtype,
        io_dtype=io_dtype,
        slot_dtype=slot_dtype,
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


def ssu_one_token_launch(
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
    dst_slots: torch.Tensor | None,
    out: torch.Tensor,
    *,
    dt_softplus: bool,
    pad_slot_id: int,
    disable_state_update: bool,
    rows_per_cta: int = 4,
    lanes_per_row: int = 32,
) -> torch.Tensor:
    """Run the native one-token SSU kernel.

    ``mate.mamba.selective_state_update`` has already validated the contract:
    single-token inputs, per-head (tied) ``A``/``D``/``dt_bias`` in fp32,
    contiguous slots and outputs. ``x``, ``dt``, ``B``, ``C`` and ``z`` may be
    strided views of a wider buffer; ``B`` and ``C`` additionally have to keep the
    row alignment the kernel's vectorized loads assume.
    """
    batch, heads, dim = x.shape
    slots, state_heads, state_dim, dstate = state.shape
    if state_heads != heads or state_dim != dim:
        raise RuntimeError("state must match x as [slots, heads, dim, dstate].")
    if dst_slots is not None and dst_slots.dtype != src_slots.dtype:
        raise RuntimeError("source and destination slots must share a dtype.")
    # B and C are the lane-vectorized operands, so their outer strides also carry the
    # alignment the kernel assumes; z is read one element per thread.
    if B.stride(-1) != 1 or C.stride(-1) != 1:
        raise RuntimeError(
            "B and C must have a dense dstate axis (stride(-1) == 1); their batch axis "
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
        for tensor in (Dv, dt_bias, src_slots, dst_slots)
    ):
        raise RuntimeError("D, dt_bias and the slot tensors must be contiguous.")
    _validate_launch(state, x, dt, A, B, dstate, rows_per_cta, lanes_per_row)
    # The kernel declares these as [heads, a_dim]-style matrices, so the per-head
    # vector form is reshaped rather than duplicated: reshape on a contiguous
    # [heads] tensor is a view.
    heads = state.shape[1]
    A = A.reshape(heads, _channel_dim(A, dim)).contiguous()
    if Dv is not None:
        Dv = Dv.reshape(heads, _channel_dim(Dv, dim)).contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.reshape(heads, _channel_dim(dt_bias, dim)).contiguous()

    kernel = _kernel_for(
        state,
        x,
        dt,
        B,
        src_slots,
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
        Dv if Dv is not None else _dummy((heads, _channel_dim(Dv, dim)), torch.float32, x.device),
        dt_bias if dt_bias is not None else _dummy((heads, _channel_dim(dt_bias, dim)), torch.float32, x.device),
        z if z is not None else _dummy((batch, heads, dim), x.dtype, x.device),
        src_slots,
        dst_slots if dst_slots is not None else src_slots,
        out,
        int(pad_slot_id),
        1 if dt_bias is not None else 0,
        1 if z is not None else 0,
        1 if Dv is not None else 0,
        1 if dt_softplus else 0,
        0 if disable_state_update else 1,
    )
    return out
