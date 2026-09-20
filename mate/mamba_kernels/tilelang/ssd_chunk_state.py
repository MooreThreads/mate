"""Native MUSA tilelang kernel for the SSD intra-chunk state stage (stage 2).

This is stage 2 of the Mamba2/SSD chunked prefill. For every logical chunk ``c``
and head ``h`` it produces the chunk-local state

    S_c[h, d, n] = sum_{t < chunk_size_limit}
                   exp(min(dA_cs[h,c,L-1] - dA_cs[h,c,t], 0)) * dt_out[h,c,t]
                   * x[t,h,d] * B[t,g(h),n]

with ``L = chunk_size``, ``g(h) = h // head_ratio`` and ``head_ratio = H // G``.
``x`` is token-major ``[tokens, heads, dim]``, ``B`` is token-major
``[tokens, groups, dstate]``, the decay/step tensors are the head-major
``[heads, nchunks, chunk_size]`` fp32 outputs of the cumsum stage, and the result
is ``[nchunks, heads, dim, dstate]`` fp32.

Rounding contract, which the oracle mirrors: ``scale_t = exp(min(delta, 0)) *
dt_t`` stays fp32, the scaled ``B`` is rounded to ``x``'s dtype **before** the
dot, and the dot accumulates in fp32. The exponential is evaluated as
``exp2(delta * log2(e))``, the MUSA `fast_exp` the shipped implementation uses.

Padding is part of the contract, not slack space: ``dt_out`` is exactly zero past
a chunk's end, so those lanes contribute nothing even though their decay factor
is ``exp(0) = 1``. ``dA_cumsum`` and ``dt_out`` are read at the padded row's last
position unconditionally (that entry holds the chunk's total decay), and every
global access that can fall outside the valid token range or the tensor is masked
by hand because the kernel is compiled with bounds checking disabled.

Dataflow (one CTA owns one ``[block_m, block_n]`` tile of one chunk and head):
the whole chunk is staged as the GEMM reduction axis -- ``x`` as ``[L, block_m]``
and the already-scaled and already-rounded ``B`` as ``[L, block_n]`` -- so a
single ``T.gemm`` with ``transpose_A=True`` produces the fp32 accumulator. The
accumulator is materialized through shared memory before the masked store: the
GEMM accumulator register order on MUSA is not the logical order, and routing it
through a ``T.copy`` keeps the store index correct without a hand-written
permutation.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["prewarm_ssd_chunk_state", "ssd_chunk_state_launch"]

#: log2(e), the constant that turns an exp2 into an exp. Same value as the
#: ``fast_exp`` helper of the shipped MUSA implementation.
_LOG2E = 1.4426950408889634

_DEFAULT_BLOCK_M = 64
_DEFAULT_BLOCK_N = 64
_THREADS = 128

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
def tilelang_ssd_chunk_state(
    dim,
    dstate,
    chunk_size,
    head_ratio,
    x_dtype,
    b_dtype,
    out_dtype,
    seqlen_dtype,
    block_m=_DEFAULT_BLOCK_M,
    block_n=_DEFAULT_BLOCK_N,
    accum_dtype="float32",
):
    """Build the intra-chunk state kernel for one (shape, dtype, tile) set."""
    nchunks = T.dynamic("nchunks")
    num_tokens = T.dynamic("num_tokens")
    heads = T.dynamic("heads")
    groups = T.dynamic("groups")

    num_m_blocks = tilelang.cdiv(dim, block_m)
    num_n_blocks = tilelang.cdiv(dstate, block_n)
    num_blocks = num_m_blocks * num_n_blocks

    x_shape = (num_tokens, heads, dim)
    b_shape = (num_tokens, groups, dstate)
    da_shape = (heads, nchunks, chunk_size)
    states_shape = (nchunks, heads, dim, dstate)

    #: ``x`` and ``b`` are read-only inputs, so they are declared strided: their token
    #: and head/group axes may be views of a wider buffer while the axis each staged
    #: copy walks (``dim`` for ``x``, ``dstate`` for ``b``) is pinned to stride 1,
    #: which the launcher enforces. ``dt_out``/``dA_cumsum`` are written dense by the
    #: cumsum stage and read here as dense spans, ``states`` is written in full by
    #: this kernel, and ``cu_chunk_seqlens`` is caller-built metadata: all stay
    #: ``T.Tensor``.
    b_stride_group = T.dynamic("b_stride_group")
    b_stride_token = T.dynamic("b_stride_token")
    x_stride_head = T.dynamic("x_stride_head")
    x_stride_token = T.dynamic("x_stride_token")
    x_strides = (x_stride_token, x_stride_head, 1)
    b_strides = (b_stride_token, b_stride_group, 1)
    x_align = _align_elems(x_dtype, dim)
    b_align = _align_elems(b_dtype, dstate)

    @T.prim_func
    def tilelang_ssd_chunk_state_kernel(
        x: T.StridedTensor(x_shape, x_strides, x_dtype),
        b: T.StridedTensor(b_shape, b_strides, b_dtype),
        dt_out: T.Tensor(da_shape, dtype=accum_dtype),
        dA_cumsum: T.Tensor(da_shape, dtype=accum_dtype),
        cu_chunk_seqlens: T.Tensor((nchunks + 1,), dtype=seqlen_dtype),
        states: T.Tensor(states_shape, dtype=out_dtype),
    ):
        with T.Kernel(num_blocks, nchunks, heads, threads=_THREADS) as (
            block,
            chunk,
            head,
        ):
            T.assume(x_stride_token % x_align == 0)
            T.assume(x_stride_head % x_align == 0)
            T.assume(b_stride_token % b_align == 0)
            T.assume(b_stride_group % b_align == 0)

            m_block = block // num_n_blocks
            n_block = block % num_n_blocks
            row = m_block * block_m
            col = n_block * block_n
            group = head // head_ratio

            chunk_start = T.alloc_var("int32")
            chunk_end = T.alloc_var("int32")
            chunk_start = cu_chunk_seqlens[chunk]
            chunk_end = cu_chunk_seqlens[chunk + 1]

            # The chunk's total decay: the padded row's last position, read
            # unconditionally. It is the saturated cumsum because the padded
            # dt is exactly zero.
            dA_last = T.alloc_var(accum_dtype)
            dA_last = dA_cumsum[head, chunk, chunk_size - 1]

            x_shared = T.alloc_shared((chunk_size, block_m), dtype=x_dtype)
            b_shared = T.alloc_shared((chunk_size, block_n), dtype=x_dtype)
            acc_shared = T.alloc_shared((block_m, block_n), dtype=accum_dtype)
            acc = T.alloc_fragment((block_m, block_n), dtype=accum_dtype)

            # The chunk is the GEMM reduction axis. Lanes past the chunk's end
            # are staged as zero, which is what the reference's `t < limit` sum
            # leaves out.
            for k, i in T.Parallel(chunk_size, block_m):
                if chunk_start + k < chunk_end and row + i < dim:
                    x_shared[k, i] = x[chunk_start + k, head, row + i]
                else:
                    x_shared[k, i] = T.cast(0.0, x_dtype)
            for k, j in T.Parallel(chunk_size, block_n):
                if chunk_start + k < chunk_end and col + j < dstate:
                    # The fp32 scale (exp2 of delta times log2(e), the shipped
                    # kernel's fast_exp) multiplied into B and rounded to x's
                    # dtype: the two documented rounding points of this stage.
                    b_shared[k, j] = T.cast(
                        T.cast(b[chunk_start + k, group, col + j], accum_dtype)
                        * (
                            T.exp2(
                                T.min(dA_last - dA_cumsum[head, chunk, k], 0.0) * _LOG2E
                            )
                            * dt_out[head, chunk, k]
                        ),
                        x_dtype,
                    )
                else:
                    b_shared[k, j] = T.cast(0.0, x_dtype)
            T.sync_threads()

            # acc[d, n] += sum_k x[k, d] * b[k, n]
            T.gemm(x_shared, b_shared, acc, transpose_A=True, clear_accum=True)

            # The accumulator's register order is not the logical order, so hand
            # the tile over through shared memory before indexing it by hand.
            T.copy(acc, acc_shared)
            T.sync_threads()
            for i, j in T.Parallel(block_m, block_n):
                if row + i < dim and col + j < dstate:
                    states[chunk, head, row + i, col + j] = acc_shared[i, j]

    symbol = (
        f"tilelang_ssd_chunk_state_d{dim}_n{dstate}_cs{chunk_size}_r{head_ratio}"
        f"_x{str(x_dtype).replace('torch.', '')}_bm{block_m}_bn{block_n}"
    )
    return tilelang_ssd_chunk_state_kernel.with_attr("global_symbol", symbol)


def ssd_chunk_state_launch(
    x: torch.Tensor,
    b: torch.Tensor,
    dt_out: torch.Tensor,
    dA_cumsum: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    *,
    states: torch.Tensor | None = None,
    state_dtype: torch.dtype | None = None,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
) -> torch.Tensor:
    """Run the native intra-chunk state kernel.

    Returns ``states``, ``[nchunks, heads, dim, dstate]``, in ``state_dtype`` when
    given, else in the dtype of the ``states`` buffer passed in, else fp32. The
    accumulator is fp32 either way; the dtype only decides where the result is
    stored, which is the rounding point the production path applies between this
    stage and the state passing that follows. Pass ``states`` to reuse
    caller-owned scratch (the orchestration does, so a captured graph allocates
    nothing).
    """
    if x.dim() != 3:
        raise RuntimeError("x must be [tokens, heads, dim].")
    if b.dim() != 3:
        raise RuntimeError("b must be [tokens, groups, dstate].")
    if b.shape[0] != x.shape[0]:
        raise RuntimeError("x and b must share the token axis.")
    if x.stride(-1) != 1 or b.stride(-1) != 1:
        raise RuntimeError(
            "x and b must have a dense innermost axis (stride(-1) == 1); their outer "
            "axes may be strided."
        )
    _require_aligned("x", x)
    _require_aligned("b", b)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            "x must be fp16, bf16 or fp32 for the native chunk-state kernel."
        )
    num_tokens, heads, dim = x.shape
    groups, dstate = b.shape[1], b.shape[2]
    if groups == 0 or heads % groups != 0:
        raise RuntimeError("b's groups must divide x's heads.")
    head_ratio = heads // groups
    if dt_out.dim() != 3 or dt_out.shape != dA_cumsum.shape:
        raise RuntimeError("dt_out and dA_cumsum must share [heads, nchunks, L].")
    if dt_out.dtype != torch.float32 or dA_cumsum.dtype != torch.float32:
        raise RuntimeError("dt_out and dA_cumsum must be fp32.")
    # The cumsum stage writes both of them dense and this kernel reads them as dense
    # spans, so they keep their contiguity gate.
    if not dt_out.is_contiguous() or not dA_cumsum.is_contiguous():
        raise RuntimeError("dt_out and dA_cumsum must be contiguous.")
    if dt_out.shape[0] != heads:
        raise RuntimeError("dt_out's head axis must match x.")
    if dt_out.shape[2] != chunk_size:
        raise RuntimeError("dt_out's last axis must be chunk_size.")
    if cu_chunk_seqlens.dtype != torch.int32:
        raise RuntimeError("cu_chunk_seqlens must be int32.")
    # Caller-built chunk offsets, read as a dense span: metadata stays dense.
    if cu_chunk_seqlens.dim() != 1 or not cu_chunk_seqlens.is_contiguous():
        raise RuntimeError("cu_chunk_seqlens must be a contiguous [nchunks + 1].")
    nchunks = cu_chunk_seqlens.numel() - 1
    if nchunks < 1:
        raise RuntimeError("cu_chunk_seqlens must have at least two entries.")
    if nchunks != dt_out.shape[1]:
        raise RuntimeError("cu_chunk_seqlens must have nchunks + 1 entries.")
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be positive.")
    if block_m < 1 or block_n < 1:
        raise RuntimeError("block_m and block_n must be positive.")

    out_shape = (nchunks, heads, dim, dstate)
    if states is None:
        states = torch.empty(
            out_shape, device=x.device, dtype=state_dtype or torch.float32
        )
    if states.shape != out_shape:
        raise RuntimeError(f"states must be {out_shape}, got {tuple(states.shape)}.")
    if states.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise RuntimeError("states must be fp32, bf16 or fp16.")
    # This kernel writes every cell of a chunk's tile, so the buffer stays dense and
    # keeps its contiguity gate.
    if not states.is_contiguous():
        raise RuntimeError("states must be contiguous.")

    kernel = tilelang_ssd_chunk_state(
        dim,
        dstate,
        chunk_size,
        head_ratio,
        x.dtype,
        b.dtype,
        states.dtype,
        cu_chunk_seqlens.dtype,
        block_m,
        block_n,
    )
    kernel(x, b, dt_out, dA_cumsum, cu_chunk_seqlens, states)
    return states


def prewarm_ssd_chunk_state(
    dim: int,
    dstate: int,
    chunk_size: int,
    head_ratio: int,
    x_dtype: torch.dtype = torch.bfloat16,
    b_dtype: torch.dtype | None = None,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
) -> None:
    """Compile the kernel for one shape outside any captured graph.

    Call during warmup for every ``(dim, dstate, chunk_size, head_ratio, dtype)``
    the serving configuration will use: a first-call compile inside a captured
    graph stalls the worker.
    """
    tilelang_ssd_chunk_state(
        dim,
        dstate,
        chunk_size,
        head_ratio,
        x_dtype,
        x_dtype if b_dtype is None else b_dtype,
        torch.float32,
        torch.int32,
        block_m,
        block_n,
    )
