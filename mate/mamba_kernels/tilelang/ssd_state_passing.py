"""Native MUSA tilelang kernel for the SSD inter-chunk state passing (stage 3).

This is stage 3 of the Mamba2/SSD chunked prefill. The intra-chunk states of the
previous stage are scanned **across** chunks, sequentially within a sequence and
in parallel across (sequence, head):

    running = entering_state(sequence, c)        # fp32
    running = exp(dA_cumsum[h, c, L-1]) * running + states[c, h]
    out[c, h] = running                          # rounded to state_dtype on store

Two boundary rules are supported, selected by a runtime flag so one compiled
kernel covers both:

* **``seq_idx``** (``seq_idx is not None``), the rule the device oracle uses:
  chunks are visited in index order, a sequence's first chunk enters from
  ``initial_states[seq_idx[c]]`` and every later chunk with that ``seq_idx``
  continues the sequence's running state. Chunks assigned to other sequences in
  between are skipped without decaying it, so one sequence's physical chunks need
  not be contiguous -- and ``initial_states`` is indexed by the **value** of
  ``seq_idx``, not by sequence position.
* **``last_chunk_indices``** arithmetic (``seq_idx is None``), the rule the
  shipped Triton kernel uses: ``chunk_end = last_chunk_indices[b] + 1`` and
  ``chunk_start = last_chunk_indices[b-1] + 1``, with ``b == 0`` starting at 0; a
  sequence with zero chunks (``last_chunk_indices[b] == -1``) iterates zero times
  and ``initial_states`` is indexed by sequence position.

The two agree whenever the caller's metadata is contiguous and consistent, which
is what the packed prefill path produces.

Both ``states`` and ``out`` are ``[nchunks, heads, dim, dstate]``; ``states`` is
fp32 and ``out`` is written in ``state_dtype``. ``initial_states`` is
``[batch, heads, dim, dstate]`` in any float dtype -- it is upcast to fp32 on
load -- and ``None`` means zero, selected by a runtime flag so one compiled kernel
covers both cases.

The decay is ``exp2(dA_cumsum * log2(e))``, the ``fast_exp`` of the shipped
implementation, and there is no ``min(..., 0)`` clip here: the cumsum is
non-positive under the documented ``A <= 0`` sign convention. The whole
recurrence stays in fp32 registers; only the store rounds, so the values written
for chunk ``c`` are the state **entering chunk ``c+1``**.

Dataflow (one CTA owns ``rows_per_cta`` consecutive ``dim`` rows of one head and
one sequence, and one warp lane group owns ``dstate / lanes_per_row`` consecutive
state columns of a row): the recurrence lives in per-thread registers, so the
kernel needs no shared memory and no cross-warp synchronization. Loads past the
``dim`` boundary are masked by branching around the address rather than
selecting, because the kernel is compiled with bounds checking disabled and an
out-of-range address must never be formed.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["prewarm_ssd_state_passing", "ssd_state_passing_launch"]

#: log2(e), the constant that turns an exp2 into an exp. Same value as the
#: ``fast_exp`` helper of the shipped MUSA implementation.
_LOG2E = 1.4426950408889634

_DEFAULT_ROWS_PER_CTA = 4
_DEFAULT_VEC = 4

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
def tilelang_ssd_state_passing(
    dim,
    dstate,
    chunk_size,
    state_dtype,
    in_dtype,
    init_dtype,
    seqlen_dtype,
    rows_per_cta=_DEFAULT_ROWS_PER_CTA,
    vec=_DEFAULT_VEC,
    accum_dtype="float32",
):
    """Build the inter-chunk state-passing kernel for one (shape, dtype, tile) set."""
    nchunks = T.dynamic("nchunks")
    heads = T.dynamic("heads")
    batch = T.dynamic("batch")

    lanes_per_row = dstate // vec
    threads = rows_per_cta * lanes_per_row
    dim_blocks = tilelang.cdiv(dim, rows_per_cta)

    states_shape = (nchunks, heads, dim, dstate)
    init_shape = (batch, heads, dim, dstate)
    da_shape = (heads, nchunks, chunk_size)

    #: ``initial_states`` is a read-only input, so it is declared strided: its sequence
    #: axis may be a view of a wider pool while the dstate axis the vectorized loads
    #: walk is pinned to stride 1, which the launcher enforces. ``states`` is read and
    #: written in place (``out`` defaults to it), ``out`` is the store target,
    #: ``dA_cumsum`` is the cumsum stage's dense output and ``last_chunk_indices`` /
    #: ``seq_idx`` are caller-built metadata: all stay ``T.Tensor`` as dense spans.
    init_stride_dim = T.dynamic("init_stride_dim")
    init_stride_head = T.dynamic("init_stride_head")
    init_stride_seq = T.dynamic("init_stride_seq")
    init_strides = (
        init_stride_seq,
        init_stride_head,
        init_stride_dim,
        1,
    )
    init_align = _align_elems(init_dtype, dstate)

    @T.prim_func
    def tilelang_ssd_state_passing_kernel(
        states: T.Tensor(states_shape, dtype=in_dtype),
        dA_cumsum: T.Tensor(da_shape, dtype=accum_dtype),
        last_chunk_indices: T.Tensor((batch,), dtype=seqlen_dtype),
        initial_states: T.StridedTensor(init_shape, init_strides, init_dtype),
        seq_idx: T.Tensor((nchunks,), dtype=seqlen_dtype),
        out: T.Tensor(states_shape, dtype=state_dtype),
        use_initial_states: T.int32,
        use_seq_idx: T.int32,
    ):
        with T.Kernel(dim_blocks, batch, heads, threads=threads) as (
            d_block,
            b_idx,
            head,
        ):
            T.assume(init_stride_seq % init_align == 0)
            T.assume(init_stride_head % init_align == 0)
            T.assume(init_stride_dim % init_align == 0)

            tid = T.get_thread_binding()
            lane = tid % lanes_per_row
            row_in_block = tid // lanes_per_row
            row = d_block * rows_per_cta + row_in_block

            chunk_start = T.alloc_var("int32")
            chunk_end = T.alloc_var("int32")
            num_chunks = T.alloc_var("int32")
            chunk = T.alloc_var("int32")
            opened = T.alloc_var("int32")
            decay = T.alloc_var(accum_dtype)

            running = T.alloc_local((vec,), dtype=accum_dtype)

            if row < dim:
                if use_seq_idx == 1:
                    # varlen walk keyed on seq_idx, the rule the device oracle
                    # uses: chunks are visited in index order, a sequence's first
                    # chunk enters from initial_states[seq_idx[c]] and every later
                    # chunk of that seq_idx continues the sequence's running
                    # state. Chunks assigned to other sequences in between are
                    # skipped without decaying it, so a sequence's physical
                    # chunks need not be contiguous.
                    opened = 0
                    for c in T.serial(nchunks):
                        if seq_idx[c] == b_idx:
                            if opened == 0:
                                # First chunk of this sequence: seed the state
                                # once, with the same decayed update below.
                                if use_initial_states == 1:
                                    for i in T.vectorized(vec):
                                        running[i] = T.cast(
                                            initial_states[
                                                b_idx, head, row, lane * vec + i
                                            ],
                                            accum_dtype,
                                        )
                                else:
                                    for i in T.unroll(vec):
                                        running[i] = 0.0
                                opened = 1
                            # exp(dA_cs) with exp2, exactly like the shipped
                            # kernel.
                            decay = T.exp2(dA_cumsum[head, c, chunk_size - 1] * _LOG2E)
                            for i in T.vectorized(vec):
                                running[i] = decay * running[i] + T.cast(
                                    states[c, head, row, lane * vec + i], accum_dtype
                                )
                            for i in T.vectorized(vec):
                                out[c, head, row, lane * vec + i] = T.cast(
                                    running[i], state_dtype
                                )
                else:
                    # The sequence's chunk range, from last_chunk_indices alone.
                    # The `b_idx - 1` entry is only touched behind the guard, so
                    # no out-of-range address is formed for the first sequence.
                    chunk_end = last_chunk_indices[b_idx] + 1
                    chunk_start = 0
                    if b_idx > 0:
                        chunk_start = last_chunk_indices[b_idx - 1] + 1
                    num_chunks = T.max(chunk_end - chunk_start, 0)

                    # initial_states is indexed by sequence position, not by
                    # seq_idx: the CTA is already the sequence.
                    if use_initial_states == 1:
                        for i in T.vectorized(vec):
                            running[i] = T.cast(
                                initial_states[b_idx, head, row, lane * vec + i],
                                accum_dtype,
                            )
                    else:
                        for i in T.unroll(vec):
                            running[i] = 0.0

                    for step in T.serial(num_chunks):
                        chunk = chunk_start + step
                        # exp(dA_cs) with exp2, exactly like the shipped kernel.
                        decay = T.exp2(dA_cumsum[head, chunk, chunk_size - 1] * _LOG2E)
                        for i in T.vectorized(vec):
                            running[i] = decay * running[i] + T.cast(
                                states[chunk, head, row, lane * vec + i], accum_dtype
                            )
                        for i in T.vectorized(vec):
                            out[chunk, head, row, lane * vec + i] = T.cast(
                                running[i], state_dtype
                            )

    symbol = (
        f"tilelang_ssd_state_passing_d{dim}_n{dstate}_cs{chunk_size}"
        f"_r{rows_per_cta}_v{vec}"
        f"_o{str(state_dtype).replace('torch.', '')}"
    )
    return tilelang_ssd_state_passing_kernel.with_attr("global_symbol", symbol)


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


def ssd_state_passing_launch(
    states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    initial_states: torch.Tensor | None = None,
    last_chunk_indices: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    *,
    chunk_size: int | None = None,
    state_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    rows_per_cta: int = _DEFAULT_ROWS_PER_CTA,
    vec: int = _DEFAULT_VEC,
) -> torch.Tensor:
    """Run the native inter-chunk state passing.

    ``states``/``out`` are ``[nchunks, heads, dim, dstate]`` and ``dA_cumsum`` is
    the head-major ``[heads, nchunks, chunk_size]`` fp32 output of the cumsum
    stage; ``chunk_size`` may be omitted and is read off that tensor.

    ``seq_idx`` is ``[nchunks]`` int32, one sequence id per physical chunk, and
    selects the ``seq_idx`` walk. Pass it whenever the caller has it: it is the
    rule the device oracle grades and it stays correct when a sequence's chunks
    are not contiguous. Without ``seq_idx`` the walk uses ``last_chunk_indices``
    ranges instead, so one of the two must be given.

    ``initial_states`` is ``[batch, heads, dim, dstate]`` and supplies ``batch``
    when ``last_chunk_indices`` is absent. ``out`` defaults to ``states`` --
    the op is in-place by default and each element is read before it is written,
    so a captured graph allocates nothing -- and a fresh tensor is only allocated
    when ``state_dtype`` differs from ``states.dtype``. Pass ``out`` to override.

    ``state_dtype`` resolves the way the reference does: the argument wins, then
    ``initial_states.dtype``, then ``states.dtype``. The reference's last-resort
    fallback is the activation dtype rather than the (always fp32) stage-3 input,
    so an orchestration that has no ``state_dtype`` and no ``initial_states``
    must pass the activation dtype explicitly to reproduce it.
    """
    if states.dim() != 4:
        raise RuntimeError("states must be [nchunks, heads, dim, dstate].")
    # Read and written in place -- `out` defaults to it -- so it stays a dense span
    # and keeps its contiguity gate.
    if not states.is_contiguous():
        raise RuntimeError("states must be contiguous.")
    nchunks, heads, dim, dstate = states.shape
    if dA_cumsum.dim() != 3:
        raise RuntimeError("dA_cumsum must be [heads, nchunks, chunk_size].")
    if dA_cumsum.shape != (heads, nchunks, dA_cumsum.shape[2]):
        raise RuntimeError(
            f"dA_cumsum must be [heads={heads}, nchunks={nchunks}, chunk_size], "
            f"got {tuple(dA_cumsum.shape)}."
        )
    if chunk_size is None:
        chunk_size = dA_cumsum.shape[2]
    if dA_cumsum.shape[2] != chunk_size:
        raise RuntimeError(
            f"dA_cumsum's last axis must be chunk_size {chunk_size}, got "
            f"{dA_cumsum.shape[2]}."
        )
    if dA_cumsum.dtype != torch.float32:
        raise RuntimeError("dA_cumsum must be fp32.")
    # The cumsum stage's dense output, read here as a dense span.
    if not dA_cumsum.is_contiguous():
        raise RuntimeError("dA_cumsum must be contiguous.")

    # Both boundary vectors are caller-built metadata, read as dense spans.
    if last_chunk_indices is not None:
        if last_chunk_indices.dim() != 1 or not last_chunk_indices.is_contiguous():
            raise RuntimeError("last_chunk_indices must be a contiguous [batch].")
        if last_chunk_indices.dtype != torch.int32:
            raise RuntimeError("last_chunk_indices must be int32.")
    if seq_idx is not None:
        if seq_idx.shape != (nchunks,) or not seq_idx.is_contiguous():
            raise RuntimeError("seq_idx must be a contiguous [nchunks].")
        if seq_idx.dtype != torch.int32:
            raise RuntimeError("seq_idx must be int32.")
    if seq_idx is None and last_chunk_indices is None:
        raise RuntimeError(
            "pass seq_idx or last_chunk_indices: without either there is no "
            "sequence boundary metadata."
        )

    if last_chunk_indices is not None:
        batch = last_chunk_indices.numel()
    elif initial_states is not None:
        batch = initial_states.shape[0]
    else:
        raise RuntimeError(
            "last_chunk_indices is needed to size the sequence grid when "
            "initial_states is absent (the chunk count alone does not name the "
            "sequences)."
        )
    if batch < 1:
        raise RuntimeError("batch must be at least 1.")

    if initial_states is not None:
        if initial_states.shape != (batch, heads, dim, dstate):
            raise RuntimeError(
                f"initial_states must be {(batch, heads, dim, dstate)}, got "
                f"{tuple(initial_states.shape)}."
            )
        if initial_states.stride(-1) != 1:
            raise RuntimeError(
                "initial_states must have a dense dstate axis (stride(-1) == 1); its "
                "sequence axis may be strided."
            )
        _require_aligned("initial_states", initial_states)
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be positive.")
    if dstate % vec != 0:
        raise RuntimeError(f"dstate {dstate} must be divisible by vec {vec}.")
    lanes_per_row = dstate // vec
    if rows_per_cta < 1 or rows_per_cta * lanes_per_row > 1024:
        raise RuntimeError(
            "the native state-passing kernel requires "
            "1 <= rows_per_cta * lanes_per_row <= 1024."
        )

    if state_dtype is None and initial_states is not None:
        state_dtype = initial_states.dtype
    if state_dtype is None:
        state_dtype = states.dtype
    if state_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError(
            "state_dtype must be fp32, fp16 or bf16 for the native "
            "state-passing kernel."
        )

    out_shape = (nchunks, heads, dim, dstate)
    if out is None:
        # In-place by default: every element is read before it is written, so
        # reusing the stage input costs nothing and keeps a captured graph
        # allocation-free.
        out = (
            states
            if state_dtype == states.dtype
            else torch.empty(out_shape, device=states.device, dtype=state_dtype)
        )
    if out.shape != out_shape:
        raise RuntimeError(f"out must be {out_shape}, got {tuple(out.shape)}.")
    if out.dtype != state_dtype:
        raise RuntimeError(f"out must be {state_dtype}, got {out.dtype}.")
    # The store target is written cell by cell, so it stays dense.
    if not out.is_contiguous():
        raise RuntimeError("out must be contiguous.")

    kernel = tilelang_ssd_state_passing(
        dim,
        dstate,
        chunk_size,
        state_dtype,
        states.dtype,
        states.dtype if initial_states is None else initial_states.dtype,
        torch.int32,
        rows_per_cta,
        vec,
    )
    device = states.device
    kernel(
        states,
        dA_cumsum,
        last_chunk_indices
        if last_chunk_indices is not None
        else _dummy((batch,), torch.int32, device),
        initial_states
        if initial_states is not None
        else _dummy((batch, heads, dim, dstate), states.dtype, device),
        seq_idx if seq_idx is not None else _dummy((nchunks,), torch.int32, device),
        out,
        1 if initial_states is not None else 0,
        1 if seq_idx is not None else 0,
    )
    return out


def prewarm_ssd_state_passing(
    dim: int,
    dstate: int,
    chunk_size: int,
    state_dtype: torch.dtype,
    in_dtype: torch.dtype = torch.float32,
    init_dtype: torch.dtype | None = None,
    rows_per_cta: int = _DEFAULT_ROWS_PER_CTA,
    vec: int = _DEFAULT_VEC,
) -> None:
    """Compile the kernel for one shape outside any captured graph.

    Call during warmup for every ``(dim, dstate, chunk_size, in dtype, state
    dtype)`` the serving configuration will use: a first-call compile inside a
    captured graph stalls the worker.
    """
    tilelang_ssd_state_passing(
        dim,
        dstate,
        chunk_size,
        state_dtype,
        in_dtype,
        in_dtype if init_dtype is None else init_dtype,
        torch.int32,
        rows_per_cta,
        vec,
    )
