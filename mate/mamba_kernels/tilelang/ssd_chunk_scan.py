"""Native MUSA tilelang kernel for the SSD chunk scan (stage 5).

This is the expensive stage of the Mamba2/SSD chunked prefill: it consumes the
chunk states from state passing, the `CB = C·Bᵀ` products from the BMM stage and
the stage-1 outputs, and produces the layer output in place.

Per (chunk, head), with `g = g(h)`:

```
y[t,d] = sum_n C[t,g,n] * ( exp(dA_cs[t]) * S_prev_c[d,n]
                          + sum_{j<=t} exp(min(dA_cs[t]-dA_cs[j], 0)) * dt_out[j]
                            * x[j,d] * B[j,g,n] )
       + D[d] * x[t,d]                 # only when HAS_D
       then y *= z * sigmoid(z)        # only when HAS_Z
```

evaluated the way production evaluates it, in this order and at the same rounding
points:

1. ``acc = dot(C, S_prev)``, then ``acc *= exp(dA_cs)`` -- the decay is applied
   **after** the past-state dot, not folded into ``C`` before it;
2. ``acc += dot(round_to_io(CB * exp(min(dA_cs_t - dA_cs_j, 0)) * dt_j), x)`` over
   the causal triangle, so the scaled ``CB`` is rounded to the activation dtype
   before the dot;
3. ``acc += x * D`` (per-dimension ``D`` or a per-head scalar);
4. ``acc *= z * sigmoid(z)``;
5. masked store, so tokens at or past a chunk's logical end are never written.

``S_prev`` is the state entering the chunk: ``initial_states[seq_idx[c]]`` when the
chunk opens a sequence, else ``states[c-1]``. That choice is an address decision
inside the kernel -- materializing it on the host would copy the whole
``(C, H, D, N)`` state buffer.

One deliberate deviation from the MUSA production path: the past-state dot is a
tensor-core dot in the activation dtype instead of an fp32 dot. With 16-bit
inputs the two agree to summation order (the products are exact in fp32 either
way) while an fp32 dot cannot use tensor cores at all. With the production bf16
SSM cache the operands are already 16-bit, so no precision is given up; an fp32
``state_dtype`` is rounded once on load, which the documented tolerance covers.
"""

import tilelang
import tilelang.language as T
import torch

__all__ = ["chunk_scan_launch", "prewarm_ssd_chunk_scan"]

LOG2E = 1.4426950408889634

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


@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def tilelang_ssd_chunk_scan(
    heads,
    groups,
    dim,
    dstate,
    chunk_size,
    io_dtype,
    state_dtype,
    block_M,
    seqlen_dtype,
    accum_dtype="float32",
):
    """Build the chunk-scan kernel for one shape.

    One CTA owns ``block_M`` consecutive rows of one (chunk, head) pair, so the
    grid is ``(ceildiv(chunk_size, block_M), nchunks, heads)``.
    """
    nchunks = T.dynamic("nchunks")
    num_tokens = T.dynamic("num_tokens")
    num_seqs = T.dynamic("num_seqs")
    block_S = chunk_size
    m_tiles = T.ceildiv(block_S, block_M)
    head_ratio = heads // groups
    threads = 128

    #: ``x``, ``C``, ``z``, ``D`` and ``initial_states`` stay dense, unlike the other
    #: stages: their stride contract was tried here and dropped on the device's evidence.
    #: With the declaration and its hints in place, a strided view returned different
    #: values than the same data read contiguously, while the identical pattern leaves the
    #: other kernels bit-identical - so the conversion is not safe for this kernel and the
    #: launcher's contiguity gate is the contract.
    #:
    #: Separately, and not caused by that conversion: this kernel's ``T.gemm`` cannot be
    #: lowered at small test shapes under the pinned TileLang ("m_warp * n_warp must equal
    #: num_warps, m_warp: 1, n_warp: 1, num_warps: 4", gemm.cc:196). The dense declaration
    #: fails the same way, which is why the contract is asserted without executing the
    #: kernel at that shape: tests/test_mamba.py::test_ssd_chunk_scan_kernel_keeps_its_dense_contract.
    #: The production shapes compile and serve - the campaign's rounds all ran this kernel -
    #: so the limitation is the test shape's, not this change's.
    @T.prim_func
    def tilelang_ssd_chunk_scan_kernel(
        x: T.Tensor((num_tokens, heads, dim), dtype=io_dtype),
        C: T.Tensor((num_tokens, groups, dstate), dtype=io_dtype),
        CB: T.Tensor((nchunks, groups, block_S, block_S), dtype=accum_dtype),
        dt_out: T.Tensor((heads, nchunks, block_S), dtype=accum_dtype),
        dA_cumsum: T.Tensor((heads, nchunks, block_S), dtype=accum_dtype),
        states: T.Tensor((nchunks, heads, dim, dstate), dtype=state_dtype),
        initial_states: T.Tensor((num_seqs, heads, dim, dstate), dtype=state_dtype),
        seq_idx: T.Tensor((nchunks,), dtype=seqlen_dtype),
        cu_chunk_seqlens: T.Tensor((nchunks + 1,), dtype=seqlen_dtype),
        D_param: T.Tensor((heads, dim), dtype=accum_dtype),
        z: T.Tensor((num_tokens, heads, dim), dtype=io_dtype),
        out: T.Tensor((num_tokens, heads, dim), dtype=io_dtype),
        use_initial_states: T.int32,
        has_d: T.int32,
        has_z: T.int32,
    ):
        with T.Kernel(m_tiles, nchunks, heads, threads=threads) as (bm, bc, bh):
            row0 = bm * block_M
            chunk_start = cu_chunk_seqlens[bc]
            limit = cu_chunk_seqlens[bc + 1] - chunk_start
            # A chunk opens a sequence when its owning request differs from the
            # previous chunk's; then the entering state is an initial state.
            prev_chunk = T.max(bc - 1, 0)
            opens_sequence = (bc == 0) | (seq_idx[bc] != seq_idx[prev_chunk])
            takes_initial_state = opens_sequence & (use_initial_states != 0)
            # A chunk that opens a sequence must not read the previous chunk's
            # state: that state belongs to the previous request. Without initial
            # states its entering state is zero, so `opens_sequence` -- not
            # `bc > 0` -- has to be the outer condition. Reading the previous
            # sequence's final state here is silent, plausible-looking corruption.
            carries_state = (opens_sequence == 0) & (bc > 0)
            head_group = bh // head_ratio

            c_shared = T.alloc_shared((block_M, dstate), dtype=io_dtype)
            s_shared = T.alloc_shared((dstate, dim), dtype=io_dtype)
            cb_shared = T.alloc_shared((block_M, block_S), dtype=io_dtype)
            x_shared = T.alloc_shared((block_S, dim), dtype=io_dtype)
            z_shared = T.alloc_shared((block_M, dim), dtype=io_dtype)
            acc = T.alloc_fragment((block_M, dim), dtype=accum_dtype)
            acc_shared = T.alloc_shared((block_M, dim), dtype=accum_dtype)

            # --- load the two operands of the past-state dot
            # Statement-level guards, not `T.if_then_else`: the select form only
            # masks the *value*, so the address is still formed and dereferenced
            # for the lanes past the chunk's end. On a partial last chunk that
            # reads up to `block_S` tokens past the packed tensor, which faults
            # as "misaligned address" *after* the kernel returns, and the async
            # report then lands on whichever stage runs next.
            for i, n in T.Parallel(block_M, dstate):
                if row0 + i < limit:
                    c_shared[i, n] = C[chunk_start + row0 + i, head_group, n]
                else:
                    c_shared[i, n] = T.cast(0, io_dtype)
            for d, n in T.Parallel(dim, dstate):
                # transposed into (dstate, dim) for the gemm. Both addresses stay
                # inside their tensors whichever way the condition goes -- a
                # sequence id and a chunk index are always in range -- so this is
                # a genuine value select, unlike the two loads above.
                s_shared[n, d] = T.if_then_else(
                    takes_initial_state,
                    T.cast(initial_states[seq_idx[bc], bh, d, n], io_dtype),
                    T.if_then_else(
                        carries_state,
                        T.cast(states[prev_chunk, bh, d, n], io_dtype),
                        T.cast(0, io_dtype),
                    ),
                )

            # TL_DISABLE_THREAD_STORAGE_SYNC is set (family convention), so the
            # barrier between a distributed shared-memory write and its consumer
            # is not inserted automatically. Without it the gemm reads operand
            # tiles that other threads have not finished writing yet: the result
            # is a sparse, data-dependent, run-to-run varying set of wrong rows
            # that looks like a math bug and is not one.
            T.sync_threads()

            T.gemm(c_shared, s_shared, acc, clear_accum=True)

            # --- the decay is applied to the past-state term after its dot
            for i, d in T.Parallel(block_M, dim):
                if row0 + i < limit:
                    acc[i, d] = acc[i, d] * T.exp(dA_cumsum[bh, bc, row0 + i])
                else:
                    acc[i, d] = 0.0

            # --- diagonal term: scaled CB rounded to the activation dtype
            for i, j in T.Parallel(block_M, block_S):
                if (row0 + i < limit) and (j <= row0 + i):
                    cb_shared[i, j] = T.cast(
                        CB[bc, head_group, row0 + i, j]
                        * T.exp(
                            T.min(
                                dA_cumsum[bh, bc, row0 + i] - dA_cumsum[bh, bc, j],
                                0.0,
                            )
                        )
                        * dt_out[bh, bc, j],
                        io_dtype,
                    )
                else:
                    cb_shared[i, j] = T.cast(0, io_dtype)

            for j, d in T.Parallel(block_S, dim):
                if j < limit:
                    x_shared[j, d] = x[chunk_start + j, bh, d]
                else:
                    x_shared[j, d] = T.cast(0, io_dtype)

            T.sync_threads()

            T.gemm(cb_shared, x_shared, acc, clear_accum=False)

            # --- epilogue: D skip, then the z gate, then a masked store
            if has_d != 0:
                for i, d in T.Parallel(block_M, dim):
                    acc[i, d] = acc[i, d] + T.cast(
                        x_shared[row0 + i, d], accum_dtype
                    ) * D_param[bh, d]

            if has_z != 0:
                for i, d in T.Parallel(block_M, dim):
                    if row0 + i < limit:
                        z_shared[i, d] = z[chunk_start + row0 + i, bh, d]
                    else:
                        z_shared[i, d] = T.cast(0, io_dtype)
                T.sync_threads()
                for i, d in T.Parallel(block_M, dim):
                    z_value = T.alloc_var(accum_dtype)
                    z_value = T.cast(z_shared[i, d], accum_dtype)
                    acc[i, d] = acc[i, d] * z_value / (1.0 + T.exp(-z_value))

            # Store straight from the fragment. The shared round-trip this replaces
            # redistributes the accumulator for coalesced stores, but its read-back
            # is not covered by a barrier that the copy reliably completes under
            # TL_DISABLE_THREAD_STORAGE_SYNC: repeating one recorded 512-token call
            # in one process gave agreement, 1e29, agreement -- single-element
            # corruption at the top of the bf16 range, which is what reading shared
            # memory that was never written looks like. The same call is stable with
            # the store taken directly from the fragment.
            for i, d in T.Parallel(block_M, dim):
                if row0 + i < limit:
                    out[chunk_start + row0 + i, bh, d] = T.cast(acc[i, d], io_dtype)

    symbol = (
        f"tilelang_ssd_chunk_scan_h{heads}_g{groups}_d{dim}_n{dstate}"
        f"_cs{chunk_size}_bm{block_M}"
    )
    return tilelang_ssd_chunk_scan_kernel.with_attr("global_symbol", symbol)


_DUMMY_CACHE: dict[tuple[tuple[int, ...], torch.dtype, str], torch.Tensor] = {}


def _dummy(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return a zeroed placeholder for an absent optional input.

    Zeroed, not ``empty``: the kernels reach a placeholder through the same
    expression as the real operand, and a select compiles to a blend, so the
    never-taken side is still evaluated. With an uninitialized buffer the blend
    can carry that garbage -- including NaN -- into the result, which is how a
    call whose input has no initial states at all ends up with every output
    element NaN.
    """
    key = (shape, dtype, str(device))
    tensor = _DUMMY_CACHE.get(key)
    if tensor is None:
        tensor = torch.zeros(shape, device=device, dtype=dtype)
        _DUMMY_CACHE[key] = tensor
    return tensor


def chunk_scan_launch(
    x: torch.Tensor,
    C: torch.Tensor,
    CB: torch.Tensor,
    dt_out: torch.Tensor,
    dA_cumsum: torch.Tensor,
    states: torch.Tensor,
    initial_states: torch.Tensor | None,
    seq_idx: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    out: torch.Tensor,
    *,
    D_param: torch.Tensor | None,
    z: torch.Tensor | None,
    block_M: int = 32,
) -> torch.Tensor:
    """Run the native chunk scan, writing ``out`` in place and returning it.

    ``out`` is written in place by contract: tokens at or past a chunk's logical
    end are never touched, so the caller may pass a view of a larger buffer.
    """
    if x.dim() != 3 or x.shape[0] != out.shape[0]:
        raise RuntimeError("x and out must be [tokens, heads, dim] with equal tokens.")
    tokens, heads, dim = x.shape
    if out.shape != x.shape:
        raise RuntimeError("out must match x's shape.")
    if C.dim() != 3 or C.shape[0] != tokens:
        raise RuntimeError("C must be [tokens, groups, dstate].")
    groups, dstate = C.shape[1], C.shape[2]
    if heads % groups:
        raise RuntimeError("heads must be divisible by groups.")
    if dt_out.shape != dA_cumsum.shape or dt_out.dim() != 3:
        raise RuntimeError("dt_out and dA_cumsum must share [heads, nchunks, chunk_size].")
    if dt_out.shape[0] != heads:
        raise RuntimeError("dt_out's head axis must match x.")
    nchunks, chunk_size = dt_out.shape[1], dt_out.shape[2]
    if CB.shape != (nchunks, groups, chunk_size, chunk_size):
        raise RuntimeError(
            f"CB must be {(nchunks, groups, chunk_size, chunk_size)}, "
            f"got {tuple(CB.shape)}."
        )
    if states.shape != (nchunks, heads, dim, dstate):
        raise RuntimeError(
            f"states must be {(nchunks, heads, dim, dstate)}, got {tuple(states.shape)}."
        )
    if initial_states is not None and tuple(initial_states.shape[1:]) != (
        heads,
        dim,
        dstate,
    ):
        raise RuntimeError("initial_states must be [sequences, heads, dim, dstate].")
    if seq_idx.shape != (nchunks,):
        raise RuntimeError("seq_idx must be [nchunks].")
    if cu_chunk_seqlens.shape != (nchunks + 1,):
        raise RuntimeError("cu_chunk_seqlens must be [nchunks + 1].")
    if cu_chunk_seqlens.dtype != torch.int32 or seq_idx.dtype != torch.int32:
        raise RuntimeError("seq_idx and cu_chunk_seqlens must be int32.")
    if D_param is not None and D_param.shape != (heads, dim):
        raise RuntimeError("D must be [heads, dim].")
    if D_param is not None and D_param.dtype != torch.float32:
        # The kernel declares D fp32; a lower-precision D would be read as fp32.
        raise RuntimeError("D must be fp32.")
    if z is not None and z.shape != x.shape:
        raise RuntimeError("z must match x's shape.")
    # ``z``, ``D`` and ``initial_states`` are read as dense spans too, so they are gated with
    # the rest instead of being trusted: a strided ``z`` used to pass this check and was then
    # addressed as if it were dense. The optional ones are skipped when absent.
    for tensor in (
        x,
        C,
        z,
        D_param,
        initial_states,
        CB,
        dt_out,
        dA_cumsum,
        states,
        seq_idx,
        cu_chunk_seqlens,
        out,
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise RuntimeError(
                "x, C, z, D, initial_states, CB, dt_out, dA_cumsum, states, metadata and "
                "out must be contiguous."
            )
    if out.dtype != x.dtype:
        raise RuntimeError("out must share x's dtype.")
    if block_M <= 0 or chunk_size % block_M:
        raise RuntimeError("block_M must divide chunk_size.")

    kernel = tilelang_ssd_chunk_scan(
        heads,
        groups,
        dim,
        dstate,
        chunk_size,
        x.dtype,
        states.dtype,
        block_M,
        seq_idx.dtype,
    )
    kernel(
        x,
        C,
        CB,
        dt_out,
        dA_cumsum,
        states,
        initial_states
        if initial_states is not None
        else _dummy((1, heads, dim, dstate), states.dtype, x.device),
        seq_idx,
        cu_chunk_seqlens,
        D_param if D_param is not None else _dummy((heads, dim), torch.float32, x.device),
        z if z is not None else _dummy(x.shape, x.dtype, x.device),
        out,
        1 if initial_states is not None else 0,
        1 if D_param is not None else 0,
        1 if z is not None else 0,
    )
    return out


def prewarm_ssd_chunk_scan(
    heads: int,
    groups: int,
    dim: int,
    dstate: int,
    chunk_size: int,
    io_dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = torch.bfloat16,
    block_M: int = 32,
) -> None:
    """Compile the kernel for one shape outside any captured graph."""
    tilelang_ssd_chunk_scan(
        heads,
        groups,
        dim,
        dstate,
        chunk_size,
        io_dtype,
        state_dtype,
        block_M,
        torch.int32,
    )
