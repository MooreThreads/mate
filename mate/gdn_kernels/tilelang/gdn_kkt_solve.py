from typing import Optional

import torch
from mate.gdn_kernels.tilelang.gdn_utils import prepare_chunk_indices
from mate.utils import cosize
import tilelang
import tilelang.language as T

__all__ = ["kkt_solve"]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    },
    compile_flags=[
        "-Od3",
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
    ],
)
def tilelang_kkt_solve(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)
    heads_per_k = H // Hg

    @T.macro
    def perm_s(i):
        stride = block_S // 8
        return (i % 8) * stride + (i // 8)

    @T.macro
    def kernel_body(
        bc,
        bhg,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S
        data_batch_idx = 0 if is_varlen else batch_idx

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

        a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
        a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
        a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)

        a32i0_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o2_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        T.annotate_layout(
            {
                a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
                a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
            }
        )

        k_bytes = 4 if qkva_dtype in (torch.float32, "float32") else 2
        k_robust_desc = T.make_robust_desc(
            T.address_of(k[data_batch_idx, seq_start_idx, 0, 0]),
            cosize((seq_end_idx - seq_start_idx, Hg, DK)) * k_bytes,
        )

        diag_input_is_ready = T.alloc_barrier(arrive_count=128)
        diag_inverse_is_ready = T.alloc_barrier(arrive_count=128)
        tx = T.get_thread_binding()
        for j_s, j_k in T.Parallel(block_S, DK):
            T.copy(
                k[data_batch_idx, left + perm_s(j_s), bhg, j_k],
                k_shared[j_s, j_k],
                src_robust_desc=k_robust_desc,
            )
        # A = K @ K^T
        T.gemm(k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if perm_s(j_s) < perm_s(j_t):
                a64_fragment[j_s, j_t] = 0.0

        for j_h in T.serial(heads_per_k):
            bh = bhg * heads_per_k + j_h

            # Load b
            if right <= seq_end_idx:
                for j_s in T.Parallel(block_S):
                    b_shared[j_s] = b[data_batch_idx, left + j_s, bh]
            else:
                for j_s in T.Parallel(block_S):
                    if left + j_s < seq_end_idx:
                        b_shared[j_s] = b[data_batch_idx, left + j_s, bh]
                    else:
                        b_shared[j_s] = 0
            # Prepare inversion input
            for j_s, j_t in T.Parallel(block_S, block_S):
                perm_j_s = perm_s(j_s)
                perm_j_t = perm_s(j_t)
                if (perm_j_s // 16) == (perm_j_t // 16) + 1:
                    a16o_shared[perm_j_s // 32, perm_j_s % 16, perm_j_t % 16] = (
                        -a64_fragment[j_s, j_t] * b_shared[perm_j_s]
                    )
                elif (perm_j_s // 16) == (perm_j_t // 16):
                    a16i_shared[perm_j_s // 16, perm_j_s % 16, perm_j_t % 16] = (
                        T.if_then_else(
                            perm_j_s == perm_j_t,
                            1.0,
                            a64_fragment[j_s, j_t] * b_shared[perm_j_s],
                        )
                    )
            T.barrier_arrive(diag_input_is_ready)
            T.barrier_wait(diag_input_is_ready, j_h % 2)

            # Invert the four 16x16 diagonal blocks directly. Each 16-lane
            # subwarp owns one block, with one lane carrying one row.
            if tx < 64:
                diag_block = tx // 16
                diag_row_idx = tx % 16
                diag_row = T.alloc_local([16], accum_dtype)

                for k_t in T.vectorized(16):
                    diag_row[k_t] = a16i_shared[diag_block, diag_row_idx, k_t]

                for src_row in T.unroll(15):
                    row_scale = -diag_row[src_row]
                    for k_t in T.unroll(src_row):
                        src_row_value = T.shfl_sync(
                            0xFFFFFFFF, diag_row[k_t], src_row, 16
                        )
                        if diag_row_idx > src_row:
                            diag_row[k_t] += row_scale * src_row_value
                    if diag_row_idx > src_row:
                        diag_row[src_row] = row_scale

                for k_t in T.vectorized(16):
                    a16i_shared[diag_block, diag_row_idx, k_t] = diag_row[k_t]
                if tx < 32:
                    for k_t in T.vectorized(16):
                        a32i0_shared[
                            diag_block * 16 + diag_row_idx, diag_block * 16 + k_t
                        ] = diag_row[k_t]
                else:
                    for k_t in T.vectorized(16):
                        a32i1_shared[
                            (diag_block - 2) * 16 + diag_row_idx,
                            (diag_block - 2) * 16 + k_t,
                        ] = diag_row[k_t]

            T.barrier_arrive(diag_inverse_is_ready)
            T.barrier_wait(diag_inverse_is_ready, j_h % 2)

            # First level 2x16x16
            T.clear(a16o_fragment)
            for k_r in T.unroll(16):
                for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
                    )
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
            T.clear(a16o_fragment)
            for k_r in T.unroll(16):
                for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                    a16o_fragment[j_s, k_s, k_t] += (
                        a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
                    )
            T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])
            T.sync_threads()

            # Second level 1x32x32
            for k_s, k_t in T.Parallel(16, 16):
                a32i0_shared[k_s, 16 + k_t] = 0
                a32i1_shared[k_s, 16 + k_t] = 0
            for j_s, k_s, k_t in T.Parallel(2, 16, 16):
                if j_s == 0:
                    a32i0_shared[16 + k_s, k_t] = a16o_shared[j_s, k_s, k_t]
                else:
                    a32i1_shared[16 + k_s, k_t] = a16o_shared[j_s, k_s, k_t]
            for j_s, j_t in T.Parallel(block_S, block_S):
                perm_j_s = perm_s(j_s)
                perm_j_t = perm_s(j_t)
                if perm_j_s >= 32 and perm_j_t < 32:
                    a32o1_shared[perm_j_s - 32, perm_j_t] = (
                        -a64_fragment[j_s, j_t] * b_shared[perm_j_s]
                    )
            T.sync_threads()

            valid_seqs = T.min(seq_end_idx - left, block_S)
            T.gemm(
                a32i1_shared, a32o1_shared, a32o_fragment, clear_accum=True, wg_wait=-1
            )
            T.warpgroup_commit_batch()
            for k_s, k_t in T.Parallel(32, 32):
                if k_s < valid_seqs:
                    a[data_batch_idx, left + k_s, bh, k_t] = a32i0_shared[k_s, k_t]
            T.warpgroup_wait(0)
            T.copy(a32o_fragment, a32o2_shared)
            T.sync_threads()
            T.gemm(
                a32o2_shared, a32i0_shared, a32o_fragment, clear_accum=True, wg_wait=-1
            )
            T.warpgroup_commit_batch()
            for k_s, k_t in T.Parallel(32, 32):
                if 32 + k_s < valid_seqs:
                    a[data_batch_idx, left + 32 + k_s, bh, 32 + k_t] = a32i1_shared[
                        k_s, k_t
                    ]
            T.warpgroup_wait(0)
            for k_s, k_t in T.Parallel(32, 32):
                if 32 + k_s < valid_seqs:
                    a[data_batch_idx, left + 32 + k_s, bh, k_t] = a32o_fragment[
                        k_s, k_t
                    ]
            for k_s, k_t in T.Parallel(32, 32):
                if k_s < valid_seqs:
                    a[data_batch_idx, left + k_s, bh, 32 + k_t] = 0

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * Hg, threads=128) as (bchg,):
                bc, bhg = bchg // Hg, bchg % Hg

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bc,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * Hg, threads=128) as (bchg,):
                bc, bhg = bchg // Hg, bchg % Hg

                batch_idx = T.alloc_var("int32")
                chunk_idx = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")

                batch_idx = bc % data_batch_size
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bc,
                    bhg,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    return tilelang_kkt_solve_kernel


def kkt_solve(
    k: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    assert K == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )

    tilelang_kkt_solve_kernel = tilelang_kkt_solve(
        H,
        Hg,
        K,
        chunk_size,
        qkva_dtype=k.dtype,
        b_dtype=b.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        tilelang_kkt_solve_kernel(k, b, cu_seqlens, chunk_indices, a)
    else:
        tilelang_kkt_solve_kernel(k, b, a, num_chunks)

    return a
