"""Native MUSA tilelang kernel for the SSD chunk-level ``C @ B^T`` (stage 4).

This is stage 4 of the Mamba2/SSD chunked prefill. For every logical chunk ``c``
and group ``g`` it materializes the **full, non-causal** ``L x L`` inner-product
matrix

    CB[c, g, i, j] = sum_n C[t_i, g, n] * B[t_j, g, n]

with ``t_i`` the chunk's ``i``-th token, ``L = chunk_size``, ``C``/``B`` token
major ``[tokens, groups, dstate]`` and the result ``[nchunks, groups, L, L]``
fp32. Non-causal means the strict upper triangle is computed too: the chunk-scan
stage applies the causal mask afterwards.

Rounding contract: both operands are rounded to ``dot_dtype`` before the dot --
``dot_dtype`` is the activation dtype when either operand is 16-bit and fp32
otherwise -- and the dot accumulates in fp32.

Masking: the chunk's valid length is ``chunk_size_limit = cu_chunk_seqlens[c+1] -
cu_chunk_seqlens[c]``. Rows ``i >= limit`` of ``C`` and rows ``j >= limit`` of
``B`` are staged as zero, so **entries at or beyond the partial-chunk boundary
are exactly zero**, while the whole ``L x L`` tile is still stored (the output is
uninitialized scratch, so the tail must be written, not skipped). Every load is
masked by hand because the kernel is compiled with bounds checking disabled.

Dataflow (one CTA owns one ``[block_m, block_n]`` tile of one chunk and group):
``C`` is staged into ``[block_m, block_k]`` and ``B`` into ``[block_n, block_k]``,
and a single ``T.gemm`` with ``transpose_B=True`` accumulates ``[block_m,
block_n]`` in fp32 over the ``block_k`` reduction tiles. The accumulator is
materialized through shared memory before the masked store: the GEMM accumulator
register order on MUSA is not the logical order, and routing it through a
``T.copy`` keeps the store index correct without a hand-written permutation.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["prewarm_ssd_bmm", "ssd_bmm_launch"]

_DEFAULT_BLOCK_M = 64
_DEFAULT_BLOCK_N = 64
_DEFAULT_BLOCK_K = 64
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


def _dot_dtype(a_dtype: torch.dtype, b_dtype: torch.dtype) -> torch.dtype:
    """The dot's activation dtype: 16-bit wins, fp32 otherwise."""
    if torch.bfloat16 in (a_dtype, b_dtype):
        return torch.bfloat16
    if torch.float16 in (a_dtype, b_dtype):
        return torch.float16
    return torch.float32


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
def tilelang_ssd_bmm(
    chunk_size,
    dstate,
    dot_dtype,
    a_dtype,
    b_dtype,
    out_dtype,
    seqlen_dtype,
    block_m=_DEFAULT_BLOCK_M,
    block_n=_DEFAULT_BLOCK_N,
    block_k=_DEFAULT_BLOCK_K,
    accum_dtype="float32",
):
    """Build the chunk-level ``C @ B^T`` kernel for one (shape, dtype, tile) set."""
    nchunks = T.dynamic("nchunks")
    num_tokens = T.dynamic("num_tokens")
    groups = T.dynamic("groups")

    num_m_blocks = tilelang.cdiv(chunk_size, block_m)
    num_n_blocks = tilelang.cdiv(chunk_size, block_n)
    num_blocks = num_m_blocks * num_n_blocks
    num_k_blocks = tilelang.cdiv(dstate, block_k)

    ab_shape = (num_tokens, groups, dstate)
    cb_shape = (nchunks, groups, chunk_size, chunk_size)

    #: ``cmat`` and ``bmat`` are read-only inputs, so they are declared strided: their
    #: outer axes may be strided views into a larger buffer. The innermost axis (dstate)
    #: is the one the block loads walk, so it is pinned to stride 1 and the launcher
    #: enforces it -- the one layout requirement this kernel cannot relax. ``cb`` is the
    #: output it writes and ``cu_chunk_seqlens`` is caller-built metadata; both are used
    #: as dense spans, so they stay ``T.Tensor``.
    bmat_stride_group = T.dynamic("bmat_stride_group")
    bmat_stride_token = T.dynamic("bmat_stride_token")
    cmat_stride_group = T.dynamic("cmat_stride_group")
    cmat_stride_token = T.dynamic("cmat_stride_token")
    cmat_strides = (cmat_stride_token, cmat_stride_group, 1)
    bmat_strides = (bmat_stride_token, bmat_stride_group, 1)
    #: The operand loads walk dstate with a vectorized copy, so every outer stride has
    #: to keep the start of a row aligned to the vector those loads can use. That is
    #: what the ``T.assume`` lines below tell the compiler, and what
    #: ``ssd_bmm_launch`` validates before launching.
    cmat_align = _align_elems(a_dtype, dstate)
    bmat_align = _align_elems(b_dtype, dstate)

    @T.prim_func
    def tilelang_ssd_bmm_kernel(
        cmat: T.StridedTensor(ab_shape, cmat_strides, a_dtype),
        bmat: T.StridedTensor(ab_shape, bmat_strides, b_dtype),
        cu_chunk_seqlens: T.Tensor((nchunks + 1,), dtype=seqlen_dtype),
        cb: T.Tensor(cb_shape, dtype=out_dtype),
    ):
        with T.Kernel(num_blocks, nchunks, groups, threads=_THREADS) as (
            block,
            chunk,
            group,
        ):
            T.assume(cmat_stride_token % cmat_align == 0)
            T.assume(cmat_stride_group % cmat_align == 0)
            T.assume(bmat_stride_token % bmat_align == 0)
            T.assume(bmat_stride_group % bmat_align == 0)

            m_block = block // num_n_blocks
            n_block = block % num_n_blocks
            row = m_block * block_m
            col = n_block * block_n

            chunk_start = T.alloc_var("int32")
            chunk_end = T.alloc_var("int32")
            limit = T.alloc_var("int32")
            chunk_start = cu_chunk_seqlens[chunk]
            chunk_end = cu_chunk_seqlens[chunk + 1]
            limit = chunk_end - chunk_start

            c_shared = T.alloc_shared((block_m, block_k), dtype=dot_dtype)
            b_shared = T.alloc_shared((block_n, block_k), dtype=dot_dtype)
            acc_shared = T.alloc_shared((block_m, block_n), dtype=accum_dtype)
            acc = T.alloc_fragment((block_m, block_n), dtype=accum_dtype)

            # Start from a fresh product. The zero fill is the identity for any
            # register order, so it is safe on a GEMM accumulator fragment.
            for i, j in T.Parallel(block_m, block_n):
                acc[i, j] = 0.0

            for k_block in T.serial(num_k_blocks):
                k0 = k_block * block_k
                # Rows past the chunk's end stay zero, so both the partial
                # chunk's tail and the tail of the reduction tile contribute
                # nothing.
                for i, kk in T.Parallel(block_m, block_k):
                    if (row + i < limit) and (k0 + kk < dstate):
                        c_shared[i, kk] = T.cast(
                            cmat[chunk_start + row + i, group, k0 + kk], dot_dtype
                        )
                    else:
                        c_shared[i, kk] = T.cast(0.0, dot_dtype)
                for j, kk in T.Parallel(block_n, block_k):
                    if (col + j < limit) and (k0 + kk < dstate):
                        b_shared[j, kk] = T.cast(
                            bmat[chunk_start + col + j, group, k0 + kk], dot_dtype
                        )
                    else:
                        b_shared[j, kk] = T.cast(0.0, dot_dtype)
                T.sync_threads()

                # acc[i, j] += sum_kk c[i, kk] * b[j, kk]
                T.gemm(c_shared, b_shared, acc, transpose_B=True, clear_accum=False)
                # The next tile overwrites the staged operands.
                T.sync_threads()

            # The accumulator's register order is not the logical order, so hand
            # the tile over through shared memory before indexing it by hand.
            T.copy(acc, acc_shared)
            T.sync_threads()
            for i, j in T.Parallel(block_m, block_n):
                if (row + i < chunk_size) and (col + j < chunk_size):
                    cb[chunk, group, row + i, col + j] = acc_shared[i, j]

    symbol = (
        f"tilelang_ssd_bmm_cs{chunk_size}_n{dstate}"
        f"_bm{block_m}_bn{block_n}_bk{block_k}"
        f"_d{str(dot_dtype).replace('torch.', '')}"
    )
    return tilelang_ssd_bmm_kernel.with_attr("global_symbol", symbol)


def ssd_bmm_launch(
    cmat: torch.Tensor,
    bmat: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    *,
    cb: torch.Tensor | None = None,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
) -> torch.Tensor:
    """Run the native chunk-level ``C @ B^T``.

    Returns ``cb``, ``[nchunks, groups, chunk_size, chunk_size]`` fp32. Pass
    ``cb`` to reuse caller-owned scratch (the orchestration does, so a captured
    graph allocates nothing).

    ``cmat`` and ``bmat`` may be strided views of a wider buffer as long as their
    dstate axis is dense and their outer strides keep the alignment the kernel's
    vector loads assume -- see ``_align_elems``.
    """
    if cmat.dim() != 3 or bmat.dim() != 3:
        raise RuntimeError("cmat and bmat must be [tokens, groups, dstate].")
    if bmat.shape != cmat.shape:
        raise RuntimeError(
            f"cmat and bmat must share a shape, got {tuple(cmat.shape)} and "
            f"{tuple(bmat.shape)}."
        )
    if cmat.stride(-1) != 1 or bmat.stride(-1) != 1:
        raise RuntimeError(
            "cmat and bmat must have a dense dstate axis (stride(-1) == 1); their "
            "outer axes may be strided."
        )
    _require_aligned("cmat", cmat)
    _require_aligned("bmat", bmat)
    num_tokens, groups, dstate = cmat.shape
    if groups < 1 or dstate < 1:
        raise RuntimeError("cmat's group and dstate axes must be non-empty.")
    if cu_chunk_seqlens.dtype != torch.int32:
        raise RuntimeError("cu_chunk_seqlens must be int32.")
    if cu_chunk_seqlens.dim() != 1 or not cu_chunk_seqlens.is_contiguous():
        raise RuntimeError("cu_chunk_seqlens must be a contiguous [nchunks + 1].")
    nchunks = cu_chunk_seqlens.numel() - 1
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be positive.")
    if block_m < 1 or block_n < 1 or block_k < 1:
        raise RuntimeError("block_m, block_n and block_k must be positive.")

    out_shape = (nchunks, groups, chunk_size, chunk_size)
    if cb is None:
        cb = torch.empty(out_shape, device=cmat.device, dtype=torch.float32)
    if cb.shape != out_shape:
        raise RuntimeError(f"cb must be {out_shape}, got {tuple(cb.shape)}.")
    if cb.dtype != torch.float32:
        raise RuntimeError("cb must be fp32.")
    if not cb.is_contiguous():
        raise RuntimeError("cb must be contiguous.")

    kernel = tilelang_ssd_bmm(
        chunk_size,
        dstate,
        _dot_dtype(cmat.dtype, bmat.dtype),
        cmat.dtype,
        bmat.dtype,
        torch.float32,
        cu_chunk_seqlens.dtype,
        block_m,
        block_n,
        block_k,
    )
    kernel(cmat, bmat, cu_chunk_seqlens, cb)
    return cb


def prewarm_ssd_bmm(
    chunk_size: int,
    dstate: int,
    a_dtype: torch.dtype = torch.bfloat16,
    b_dtype: torch.dtype | None = None,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
) -> None:
    """Compile the kernel for one shape outside any captured graph.

    Call during warmup for every ``(chunk_size, dstate, dtype)`` the serving
    configuration will use: a first-call compile inside a captured graph stalls
    the worker.
    """
    other_dtype = a_dtype if b_dtype is None else b_dtype
    tilelang_ssd_bmm(
        chunk_size,
        dstate,
        _dot_dtype(a_dtype, other_dtype),
        a_dtype,
        other_dtype,
        torch.float32,
        torch.int32,
        block_m,
        block_n,
        block_k,
    )
