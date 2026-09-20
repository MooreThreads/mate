"""Native MUSA tilelang kernel for the SSD chunk-local cumsum stage.

This is stage 1 of the Mamba2/SSD chunked prefill behind
``mate.mamba.ssd_combined_fwd_varlen``. For every chunk it produces

* the **processed dt** -- ``softplus(dt + dt_bias)`` clamped to ``dt_limit`` --
  which the chunk-state stage consumes, and
* ``dA_cumsum`` -- the **inclusive** prefix sum of ``processed_dt * A`` over the
  chunk's tokens, which the state-passing and chunk-scan stages consume.

Layout contract (matches the production MUSA implementation, see the notes at the
bottom): ``dt`` is token-major ``[tokens, heads]`` while **both outputs are
head-major** ``[heads, nchunks, chunk_size]``. A chunk never spans two sequences
(the caller guarantees it), so the scan is per chunk and `cu_seqlens` is not
needed -- only `cu_chunk_seqlens`, the chunk start offsets in token units.

Chunks are padded rows: chunk ``c`` covers tokens
``[cu_chunk_seqlens[c], cu_chunk_seqlens[c+1])`` and lives at ``[0:heads, c, 0:]``
of the output. The **whole row is written**, and padding is part of the contract
rather than slack space: ``dt_out`` is exactly zero past the chunk's end, so the
scan saturates and ``dA_cumsum`` holds the chunk's total decay in the tail --
which is what the downstream stages read, unconditionally, from the padded row's
last position ``dA_cumsum[:, c, chunk_size - 1]``. The kernel is compiled with
bounds checking disabled, so every ``dt`` load sits under the chunk-length test
and its row index is also clamped into the chunk: no load leaves
``[cu_chunk_seqlens[c], cu_chunk_seqlens[c+1])``, and a chunk with no tokens
issues none.

The grid is ``(nchunks, heads // HB)``: each CTA owns ``HB`` heads of one chunk
and scans its ``(HB, chunk_size)`` fp32 fragment with ``T.cumsum``, mirroring the
warp-scan in the shipped native implementation but at tile granularity. The
per-row arithmetic does not depend on ``HB``, so every ``HB`` produces
bit-identical outputs. Unless the caller fixes it, the launcher takes the largest
``HB`` in ``HEADS_PER_BLOCK_CHOICES`` that divides ``heads`` and still leaves
``MIN_GRID_CTAS`` CTAs, and one head per CTA when none does.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["chunk_cumsum_launch", "prewarm_ssd_chunk_cumsum"]

#: Softplus switches to the identity above this bound. Same constant as the SSU
#: kernel and its oracle, so the family stays self-consistent.
SOFTPLUS_THRESHOLD = 20.0

_THREADS = 128

#: Heads per CTA the launcher chooses from, largest first. At four, each of the
#: four warps owns one row of the scan.
HEADS_PER_BLOCK_CHOICES = (4, 2, 1)

#: CTAs the grid keeps before heads are packed into fewer, fuller CTAs. On S5000
#: at 64 heads, one head per CTA is fastest from one to four chunks (64-256 CTAs)
#: and four per CTA at 32 chunks (512 CTAs).
MIN_GRID_CTAS = 256

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    # Thread storage sync stays ON: ``dtT_shared`` is written by one loop and
    # read by another, so the barriers around that handoff have to be the
    # compiler's. With the pass disabled the transpose loop reads stale cells
    # (observed as whole heads of ``dA_cumsum`` off by O(1) on MUSA).
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


def _heads_per_block(heads: int, nchunks: int, requested: int | None = None) -> int:
    """The heads each CTA owns.

    An explicit ``requested`` must divide ``heads``. Otherwise the largest of
    ``HEADS_PER_BLOCK_CHOICES`` that divides ``heads`` while leaving at least
    ``MIN_GRID_CTAS`` CTAs, and one head per CTA when none does.
    """
    if requested is not None:
        if requested <= 0 or heads % requested != 0:
            raise RuntimeError(
                f"heads_per_block must be a positive divisor of heads={heads}; "
                f"got {requested}."
            )
        return requested
    for hb in HEADS_PER_BLOCK_CHOICES:
        if heads % hb == 0 and nchunks * (heads // hb) >= MIN_GRID_CTAS:
            return hb
    return 1


@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def tilelang_ssd_chunk_cumsum(
    heads,
    chunk_size,
    heads_per_block,
    dt_dtype,
    param_dtype,
    out_dtype,
    seqlen_dtype,
    accum_dtype="float32",
):
    """Build the chunk-local cumsum kernel for one (heads, chunk_size, HB, dtype) set."""
    if heads_per_block <= 0 or heads % heads_per_block != 0:
        raise ValueError(
            f"heads_per_block={heads_per_block} must be a positive divisor of "
            f"heads={heads}."
        )
    HB = heads_per_block
    nchunks = T.dynamic("nchunks")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    dt_shape = (num_tokens, heads)
    out_shape = (heads, nchunks, block_S)

    #: ``dt`` is a read-only input, so it is declared strided: its token axis may be a
    #: strided view of a wider buffer while the head axis the staged copy walks is
    #: pinned to stride 1, which the launcher enforces. ``A`` and ``dt_bias`` are
    #: per-head vectors -- there is no outer axis to stride, so their single axis
    #: carries the literal stride the kernel reads them with. ``dA_cumsum`` and
    #: ``dt_out`` are written in full and ``cu_chunk_seqlens`` is caller-built
    #: metadata: both stay ``T.Tensor`` as dense spans.
    dt_stride_token = T.dynamic("dt_stride_token")
    dt_strides = (dt_stride_token, 1)
    dt_align = _align_elems(dt_dtype, heads)

    @T.prim_func
    def tilelang_ssd_chunk_cumsum_kernel(
        dt: T.StridedTensor(dt_shape, dt_strides, dt_dtype),
        A: T.StridedTensor((heads,), (1,), param_dtype),
        dt_bias: T.StridedTensor((heads,), (1,), param_dtype),
        cu_chunk_seqlens: T.Tensor((nchunks + 1,), dtype=seqlen_dtype),
        dA_cumsum: T.Tensor(out_shape, dtype=out_dtype),
        dt_out: T.Tensor(out_shape, dtype=out_dtype),
        use_dt_bias: T.int32,
        dt_softplus: T.int32,
        dt_min: T.float32,
        dt_max: T.float32,
    ):
        with T.Kernel(nchunks, heads // HB, threads=_THREADS) as (bc, hb):
            T.assume(dt_stride_token % dt_align == 0)
            chunk_start = T.alloc_var("int32")
            chunk_end = T.alloc_var("int32")
            chunk_len = T.alloc_var("int32")
            last_row = T.alloc_var("int32")
            chunk_start = cu_chunk_seqlens[bc]
            chunk_end = cu_chunk_seqlens[bc + 1]
            chunk_len = chunk_end - chunk_start
            # The last row of the chunk any load may address.
            last_row = T.max(chunk_len - 1, 0)

            # Token-major staging of this CTA's heads, then transposed into the
            # (HB, chunk_size) fragment the scan works on. dt is staged in its own
            # dtype under the chunk test and widened afterwards: a fused
            # load-and-widen is vectorised ahead of its guard.
            dt_staged = T.alloc_fragment((block_S, HB), dtype=dt_dtype)
            dtT_fragment = T.alloc_fragment((block_S, HB), dtype=accum_dtype)
            dtT_shared = T.alloc_shared((block_S, HB + 1), dtype=accum_dtype)
            scaled = T.alloc_fragment((HB, block_S), dtype=accum_dtype)
            processed = T.alloc_fragment((HB, block_S), dtype=accum_dtype)

            T.clear(dt_staged)
            if chunk_len > 0:
                for j, i in T.Parallel(block_S, HB):
                    if j < chunk_len:
                        dt_staged[j, i] = dt[
                            chunk_start + T.min(j, last_row), hb * HB + i
                        ]
            for j, i in T.Parallel(block_S, HB):
                dtT_fragment[j, i] = T.cast(dt_staged[j, i], accum_dtype)
            T.copy(dtT_fragment, dtT_shared[:, 0:HB])

            for i, j in T.Parallel(HB, block_S):
                value = T.alloc_var(accum_dtype)
                value = dtT_shared[j, i]
                if use_dt_bias != 0:
                    value = value + T.cast(dt_bias[hb * HB + i], accum_dtype)
                if dt_softplus != 0:
                    value = T.if_then_else(
                        value > SOFTPLUS_THRESHOLD,
                        value,
                        T.log(1.0 + T.exp(value)),
                    )
                value = T.max(value, dt_min)
                value = T.min(value, dt_max)
                if chunk_start + j < chunk_end:
                    processed[i, j] = value
                    scaled[i, j] = value * T.cast(A[hb * HB + i], accum_dtype)
                else:
                    # dt is zero beyond the chunk's end, so the scan saturates at
                    # the chunk total -- the value the downstream stages read
                    # from the padded row's last position.
                    processed[i, j] = T.cast(0.0, accum_dtype)
                    scaled[i, j] = T.cast(0.0, accum_dtype)

            T.cumsum(scaled, dim=1)

            # The row total is the one cross-thread value the padded tail needs,
            # so its owner publishes it into a small shared broadcast. The scan
            # itself only reaches that total to within a rounding of its own last
            # cell (a tile-level scan never adds the tail's zeros the way a
            # sequential one does), and the padding contract is the total itself:
            # downstream stages read it back from index ``block_S - 1``.
            total_shared = T.alloc_shared((HB,), dtype=accum_dtype)
            for i, j in T.Parallel(HB, block_S):
                if j == block_S - 1:
                    total_shared[i] = scaled[i, j]

            # Every row is written in full: zero dt in the tail, saturated total
            # of dA_cumsum in the tail.
            for i, j in T.Parallel(HB, block_S):
                if chunk_start + j < chunk_end:
                    dA_cumsum[hb * HB + i, bc, j] = T.cast(scaled[i, j], out_dtype)
                else:
                    dA_cumsum[hb * HB + i, bc, j] = T.cast(total_shared[i], out_dtype)
                dt_out[hb * HB + i, bc, j] = T.cast(processed[i, j], out_dtype)

    # HB is part of the symbol, ahead of the chunk size, so no variant's name is a
    # substring of another's and no two variants share a compiled entry.
    symbol = (
        f"tilelang_ssd_chunk_cumsum_h{heads}_hb{HB}_cs{chunk_size}"
        f"_dt{str(dt_dtype).replace('torch.', '')}_o{str(out_dtype).replace('torch.', '')}"
    )
    return tilelang_ssd_chunk_cumsum_kernel.with_attr("global_symbol", symbol)


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


def chunk_cumsum_launch(
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor | None,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    *,
    dt_softplus: bool,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    dA_cumsum: torch.Tensor | None = None,
    dt_out: torch.Tensor | None = None,
    heads_per_block: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the native chunk-local cumsum.

    Returns ``(dA_cumsum, dt_out)``, both head-major
    ``[heads, nchunks, chunk_size]`` fp32. Pass ``dA_cumsum``/``dt_out`` to
    reuse caller-owned scratch (the orchestration does, so a captured graph
    allocates nothing).

    ``dt`` may be fp32 or low precision; the scan itself is fp32. It may be a
    strided view of a wider buffer as long as its head axis is dense and its token
    stride keeps the alignment the kernel's staged copy assumes.

    ``heads_per_block`` fixes the heads each CTA owns; ``None`` picks it from the
    number of chunks, as the module notes describe. The outputs do not depend on
    it.
    """
    if dt.dim() != 2:
        raise RuntimeError("dt must be [tokens, heads].")
    if dt.stride(-1) != 1:
        raise RuntimeError(
            "dt must have a dense head axis (stride(-1) == 1); its token axis may be "
            "strided."
        )
    _require_aligned("dt", dt)
    heads = dt.shape[1]
    if A.numel() != heads or A.dim() != 1:
        raise RuntimeError("A must be [heads].")
    if A.stride(-1) != 1:
        raise RuntimeError("A must be a dense [heads] vector (stride(-1) == 1).")
    if A.dtype != torch.float32:
        raise RuntimeError("A must be fp32.")
    if dt_bias is not None and (dt_bias.numel() != heads or dt_bias.dtype != torch.float32):
        raise RuntimeError("dt_bias must be fp32 [heads].")
    if dt_bias is not None and dt_bias.stride(-1) != 1:
        raise RuntimeError("dt_bias must be a dense [heads] vector (stride(-1) == 1).")
    if cu_chunk_seqlens.dtype != torch.int32:
        raise RuntimeError("cu_chunk_seqlens must be int32.")
    if cu_chunk_seqlens.dim() != 1 or cu_chunk_seqlens.numel() < 2:
        raise RuntimeError("cu_chunk_seqlens must be [nchunks + 1].")
    # The caller builds the chunk offsets dense, and the kernel reads them as a dense
    # span, so this one is checked with is_contiguous() rather than a stride gate.
    if not cu_chunk_seqlens.is_contiguous():
        raise RuntimeError("cu_chunk_seqlens must be contiguous.")
    nchunks = cu_chunk_seqlens.numel() - 1
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be positive.")
    hb = _heads_per_block(heads, nchunks, heads_per_block)

    out_shape = (heads, nchunks, chunk_size)
    if dA_cumsum is None:
        dA_cumsum = torch.empty(out_shape, device=dt.device, dtype=torch.float32)
    if dt_out is None:
        dt_out = torch.empty(out_shape, device=dt.device, dtype=torch.float32)
    if dA_cumsum.shape != out_shape or dt_out.shape != out_shape:
        raise RuntimeError(
            f"outputs must be {out_shape}, got {tuple(dA_cumsum.shape)} / "
            f"{tuple(dt_out.shape)}."
        )
    if dA_cumsum.dtype != torch.float32 or dt_out.dtype != torch.float32:
        raise RuntimeError("dA_cumsum and dt_out must be fp32.")
    # Both are written in full -- every cell of the padded chunk -- so they stay dense
    # buffers and keep their contiguity gate.
    if not dA_cumsum.is_contiguous() or not dt_out.is_contiguous():
        raise RuntimeError("dA_cumsum and dt_out must be contiguous.")

    kernel = tilelang_ssd_chunk_cumsum(
        heads,
        chunk_size,
        hb,
        dt.dtype,
        torch.float32,
        torch.float32,
        cu_chunk_seqlens.dtype,
    )
    dt_min, dt_max = dt_limit
    kernel(
        dt,
        A,
        dt_bias if dt_bias is not None else _dummy((heads,), torch.float32, dt.device),
        cu_chunk_seqlens,
        dA_cumsum,
        dt_out,
        1 if dt_bias is not None else 0,
        1 if dt_softplus else 0,
        float(dt_min),
        float(dt_max),
    )
    return dA_cumsum, dt_out


def prewarm_ssd_chunk_cumsum(
    heads: int,
    chunk_size: int,
    dt_dtype: torch.dtype = torch.float32,
    *,
    heads_per_block: int | None = None,
) -> None:
    """Compile the kernel for one shape outside any captured graph.

    Call during warmup for every ``(heads, chunk_size, dt dtype)`` the serving
    configuration will use: a first-call compile inside a captured graph stalls
    the worker. With ``heads_per_block=None`` the launcher picks ``HB`` per call
    from the number of chunks, so every variant it can pick is compiled here.
    """
    if heads_per_block is None:
        variants = [hb for hb in HEADS_PER_BLOCK_CHOICES if heads % hb == 0]
    else:
        variants = [_heads_per_block(heads, 0, heads_per_block)]
    for hb in variants:
        tilelang_ssd_chunk_cumsum(
            heads, chunk_size, hb, dt_dtype, torch.float32, torch.float32, torch.int32
        )
