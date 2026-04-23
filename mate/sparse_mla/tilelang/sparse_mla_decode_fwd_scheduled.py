# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tvm import tir
import math


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        # tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    },
    verbose=True,
    compile_flags=[
        # "-Od3",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-fno-strict-aliasing",
        "-mllvm",
        "-misched=mtgpu-max-ilp",
        "-mllvm",
        "-mtgpu-tiny-offset-hint=1",
        "-mllvm",
        "-misched-recompute-slotindex=1",
        # "-mllvm",
        # "-mtgpu-combine-instr-with-burst=1",
        "-mllvm",
        "-mtgpu-combine-fop-instr=1",
    ],
)
def sparse_attention_fwd_kernel(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    block_H=64,
    block_I=64,
    threads=640,
    max_nums_splits=32,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_mp_parts = T.dynamic("num_mp_parts")

    head_kv = num_heads // kv_group
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), block_H)
    if padded_H != H:
        assert kv_group == 1
    kv_latent_dtype = "float8_e4m3"
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    L = block_I // 8
    dim_bytes = 656
    if head_kv > block_H:
        assert head_kv % block_H == 0, "head_kv should be a multiple of block_H"
        REPLICATE_H = head_kv // block_H
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else block_H
    PV_MMA_N = 64 if block_H == 32 else 128

    @T.macro
    def dsa_decode_split(
        Q: T.Tensor([batch, seq_len, num_heads, dim + tail_dim], dtype),  # type: ignore
        KV: T.Tensor([seq_len_kv, kv_group, dim_bytes], kv_latent_dtype),  # type: ignore
        K_pe: T.Tensor([seq_len_kv, kv_group, dim_bytes // 2], dtype),  # type: ignore
        Quant_scales: T.Tensor([seq_len_kv, kv_group, dim_bytes // 4], T.float32),  # type: ignore
        Indices: T.Tensor([batch, seq_len, kv_group, topk], indices_dtype),  # type: ignore
        tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        NumSplits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor([batch + num_mp_parts, seq_len, num_heads], T.float32),  # type: ignore
        Output_partial: T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        ),  # type: ignore
        Output: T.Tensor([batch, seq_len, num_heads, dim], dtype),  # type: ignore
        Lse: T.Tensor([batch, seq_len, num_heads], T.float32),  # type: ignore
    ):
        with T.Kernel(
            seq_len * REPLICATE_H, kv_group, num_mp_parts, threads=threads
        ) as (bx, by, bz):
            KV_shared_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_r = T.alloc_shared([BI, D // 2], dtype)
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            V_shared_0 = T.alloc_shared([BI, D // 4], dtype)
            V_shared_1 = T.alloc_shared([BI, D // 4], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sum_exp_inv_shared = T.alloc_shared([H_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([BI], "bool", scope="shared")
            kv_indices = T.alloc_shared([BI], "int32", scope="shared")
            Quant_shared = T.alloc_shared([BI, 4], "float32")
            bar_kv_mask_ready = T.alloc_barrier(arrive_count=128)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=256)
            bar_q = T.alloc_barrier(arrive_count=512)
            # bar_q_free = T.alloc_barrier(arrive_count=256)
            bar_indices_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_lma_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_lma_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_free = T.alloc_barrier(arrive_count=256)
            bar_kv1_free = T.alloc_barrier(arrive_count=256)
            bar_vl0_ready = T.alloc_barrier(arrive_count=256)
            bar_vl1_ready = T.alloc_barrier(arrive_count=256)
            bar_vr0_ready = T.alloc_barrier(arrive_count=256)
            bar_vr1_ready = T.alloc_barrier(arrive_count=256)
            # bar_vr0_free = T.alloc_barrier(arrive_count=256)
            # bar_vr1_free = T.alloc_barrier(arrive_count=256)
            bar_vl0_free = T.alloc_barrier(arrive_count=256)
            bar_vl1_free = T.alloc_barrier(arrive_count=256)
            # bar_p_free = T.alloc_barrier(arrive_count=256)
            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_final = T.alloc_barrier(arrive_count=256)
            # bar_final_free = T.alloc_barrier(arrive_count=256)
            T.sync_threads()

            begin_idx = T.alloc_var(T.int32)
            sched_begin_block_idx = T.alloc_var(T.int32)
            end_idx = T.alloc_var(T.int32)
            sched_end_block_idx = T.alloc_var(T.int32)
            begin_n_split_idx = T.alloc_var(T.int32)
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)
            begin_idx = tile_scheduler_metadata[bz, 0]
            sched_begin_block_idx = tile_scheduler_metadata[bz, 1]
            end_idx = tile_scheduler_metadata[bz, 2]
            sched_end_block_idx = tile_scheduler_metadata[bz, 3]
            begin_n_split_idx = tile_scheduler_metadata[bz, 4]

            q_robust_desc = T.make_robust_desc(
                T.address_of(Q[0, 0, 0, 0]),
                (batch * seq_len * num_heads * (dim + tail_dim)) * 2,
            )
            kv_robust_desc = T.make_robust_desc(
                T.address_of(KV[0, 0, 0]), (seq_len_kv * kv_group * (dim_bytes))
            )

            g_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block
            tid = T.get_thread_binding()
            for b_i in range(begin_idx, end_idx + 1, 1):
                tir.call_extern("void", "__musa_loop_transparent_outermost")
                start_block_idx = T.alloc_var(T.int32)
                end_block_idx = T.alloc_var(T.int32)
                n_split_idx = T.alloc_var(T.int32)
                is_split = T.alloc_var("bool")
                start_block_idx = T.if_then_else(
                    b_i == begin_idx, sched_begin_block_idx, 0
                )
                end_block_idx = T.if_then_else(
                    b_i == end_idx, sched_end_block_idx, T.ceildiv(topk, BI)
                )
                n_split_idx = T.if_then_else(b_i == begin_idx, begin_n_split_idx, 0)
                is_split = (NumSplits[b_i + 1] - NumSplits[b_i]) == 1
                if tid < 512:
                    # T.barrier_wait(bar_q_free, (b_i - begin_idx+1) & 1)
                    T.copy(
                        Q[b_i, s_i, H0:H1, 0 : D // 2],
                        Q_shared_l,
                        force_async_copy=True,
                        src_robust_desc=q_robust_desc,
                    )
                    T.copy(
                        Q[b_i, s_i, H0:H1, D // 2 : D],
                        Q_shared_r,
                        force_async_copy=True,
                        src_robust_desc=q_robust_desc,
                    )
                    T.copy(
                        Q[b_i, s_i, H0:H1, D:],
                        Q_tail_shared,
                        force_async_copy=True,
                        src_robust_desc=q_robust_desc,
                    )

                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, (b_i - begin_idx) & 1)
                if tid < 256:
                    sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([H_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                    acc_s_cast = T.alloc_fragment([H_per_block, BI], dtype)
                    acc_o_l_0 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                    acc_o_l_1 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                    kv_reg_l = T.alloc_local([64], dtype)
                    kv_reg_l_fp16 = T.view(kv_reg_l, [64], T.float16)
                    kv_reg_l_bf16_load = T.alloc_local([32], T.bfloat16)
                    kv_reg_l_fp8 = T.view(kv_reg_l_bf16_load, [64], kv_latent_dtype)
                    quant_local_l = T.alloc_local([2, 2], T.float32)
                    ldg_tx = (tid) % 8
                    ldg_ty = (tid) // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))
                        T.copy(Quant_shared[ldg_ty, 0:2], quant_local_l[0, :])
                        T.copy(Quant_shared[ldg_ty + 32, 0:2], quant_local_l[1, :])
                        T.annotate_layout(
                            {
                                KV_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_l[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        # KV_l fp8 quant
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                        KV_shared_l[
                                            (ldg_ty + r * 32),
                                            64 * u + ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_l_fp16[idx] * quant_local_l[r, 0],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    KV_shared_l[
                                        ldg_ty + r * 32, 64 * u + ldg_tx * 8 + v
                                    ] = kv_reg_l[r * 32 + u * 8 + v]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_l_bf16_load[r * 16 + (u + 2) * 4 + v] = (
                                        KV_shared_l[
                                            (ldg_ty + r * 32),
                                            64 * (u + 2) + ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l_fp16[idx] = kv_reg_l_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_l[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_l_fp16[idx] * quant_local_l[r, 1],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    KV_shared_l[
                                        ldg_ty + r * 32, 64 * (u + 2) + ldg_tx * 8 + v
                                    ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)
                        T.barrier_wait(bar_kv_mask_ready, (phase_count[0] & 1))
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i % 8 * 8 + bi_i // 8], 0, -(2**30)
                            )
                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_free)

                        T.barrier_wait(bar_kv0_quant_ready, (phase_count[0] & 1))
                        T.gemm(
                            Q_shared_l[:, :],
                            KV_shared_l[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv0_free)

                        T.barrier_wait(bar_kv1_quant_ready, (phase_count[0] & 1))
                        T.annotate_layout(
                            {
                                KV_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_r[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            Q_shared_r,
                            KV_shared_r[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.annotate_layout(
                            {
                                K_tail_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    K_tail_shared[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared[:, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        # T.barrier_arrive(bar_q_free)
                        T.barrier_arrive(bar_kv1_free)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2(
                                (m_i_prev[h_i] - m_i[h_i]) * sm_scale
                            )
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )

                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 4):
                            acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                            acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                        T.copy(alpha_local, alpha_shared)
                        T.copy(acc_s, acc_s_cast)
                        for i, t in T.Parallel(H_per_block, 8):
                            base = t * L
                            for l in T.vectorized(L):
                                S_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        T.annotate_layout(
                            {
                                V_shared_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_0[:, :], continuity=PV_MMA_N, k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    V_shared_0[
                                        ((ldg_ty + r * 32) % 8) * (block_I // 8)
                                        + (ldg_ty + r * 32) // 8,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + u * 8 + v]
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        T.gemm(
                            S_shared,
                            V_shared_0,
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.warpgroup_commit_batch()
                        T.annotate_layout(
                            {
                                V_shared_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_1[:, :], continuity=PV_MMA_N, k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    V_shared_1[
                                        ((ldg_ty + r * 32) % 8) * (block_I // 8)
                                        + (ldg_ty + r * 32) // 8,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl0_free)

                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, (phase_count[0] & 1))

                        T.gemm(
                            S_shared,
                            V_shared_1,
                            acc_o_l_1,
                            transpose_B=False,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl1_free)
                        phase_count[0] = phase_count[0] ^ 1
                    for h_i in T.Parallel(H_per_block):
                        sumexp_inv[h_i] = 1 / (sumexp[h_i] + 1e-8)
                    for h_i in T.Parallel(H_per_block):
                        sum_exp_inv_shared[h_i] = sumexp_inv[h_i]
                    T.lma_wait()
                    # T.barrier_wait(bar_final_free, (b_i - begin_idx+1) & 1)
                    T.barrier_arrive(bar_final)
                    for h_i, d_i in T.Parallel(H_per_block, D // 4):
                        acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                        acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

                    if is_split:
                        T.copy(acc_o_l_0, Output[b_i, s_i, H0:H1, 0 : D // 4])
                        T.copy(acc_o_l_1, Output[b_i, s_i, H0:H1, D // 4 : D // 2])
                        T.copy(sumexp, Lse[b_i, s_i, H0:H1])
                    else:
                        T.copy(
                            acc_o_l_0,
                            Output_partial[
                                n_split_idx + NumSplits[b_i], s_i, H0:H1, 0 : D // 4
                            ],
                        )
                        T.copy(
                            acc_o_l_1,
                            Output_partial[
                                n_split_idx + NumSplits[b_i],
                                s_i,
                                H0:H1,
                                D // 4 : D // 2,
                            ],
                        )
                        T.copy(sumexp, glse[n_split_idx + NumSplits[b_i], s_i, H0:H1])
                elif tid >= 256 and tid < 512:
                    acc_o_r_0 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                    acc_o_r_1 = T.alloc_fragment([H_per_block, D // 4], accum_dtype)
                    kv_reg_r = T.alloc_local([64], dtype)
                    kv_reg_r_fp16 = T.view(kv_reg_r, [64], T.float16)
                    kv_reg_r_fp8 = T.alloc_local([64], kv_latent_dtype)
                    kv_reg_r_bf16 = T.view(kv_reg_r_fp8, [32], T.bfloat16)
                    quant_local_r = T.alloc_local([2, 2], T.float32)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    ldg_tx = (tid - 256) % 8
                    ldg_ty = (tid - 256) // 8
                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv1_ready, (phase_count[0] & 1))
                        T.copy(Quant_shared[ldg_ty, 2:4], quant_local_r[0, :])
                        T.copy(Quant_shared[ldg_ty + 32, 2:4], quant_local_r[1, :])

                        T.annotate_layout(
                            {
                                KV_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_r[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        # KV_r fp8 quant
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_r_bf16[r * 16 + u * 4 + v] = KV_shared_r[
                                        ldg_ty + r * 32,
                                        64 * u + ldg_tx * 8 + v,
                                    ]
                        T.lma_wait()

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + u * 8 + v
                                    kv_reg_r[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_r_fp16[idx] * quant_local_r[r, 0],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    KV_shared_r[
                                        ldg_ty + r * 32, 64 * u + ldg_tx * 8 + v
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_r_bf16[r * 16 + (u + 2) * 4 + v] = (
                                        KV_shared_r[
                                            (ldg_ty + r * 32),
                                            64 * (u + 2) + ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_reg_r[idx] = T.Cast(
                                        "bfloat16",
                                        kv_reg_r_fp16[idx] * quant_local_r[r, 1],
                                    )

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    KV_shared_r[
                                        ldg_ty + r * 32, 64 * (u + 2) + ldg_tx * 8 + v
                                    ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]

                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_quant_ready)

                        T.barrier_wait(bar_vl0_free, (phase_count[0] & 1))

                        T.annotate_layout(
                            {
                                V_shared_0[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_0[:, :], continuity=PV_MMA_N, k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    V_shared_0[
                                        ((ldg_ty + r * 32) % 8) * (block_I // 8)
                                        + (ldg_ty + r * 32) // 8,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, (phase_count[0] & 1))

                        # compute v4-v7
                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 4):
                            acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                            acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                        # bar arrive & wait
                        T.gemm(
                            S_shared,
                            V_shared_0,
                            acc_o_r_0,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_vl1_free, (phase_count[0] & 1))

                        T.annotate_layout(
                            {
                                V_shared_1[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    V_shared_1[:, :], continuity=PV_MMA_N, k_major=False
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    V_shared_1[
                                        ((ldg_ty + r * 32) % 8) * (block_I // 8)
                                        + (ldg_ty + r * 32) // 8,
                                        64 * u + ldg_tx * 8 + v,
                                    ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                        T.lma_wait()
                        T.warpgroup_wait(0)
                        # T.barrier_arrive(bar_vr0_free)
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, (phase_count[0] & 1))

                        # compute v4-v7
                        T.gemm(
                            S_shared,
                            V_shared_1,
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        # T.barrier_arrive(bar_p_free)
                        # T.barrier_arrive(bar_vr1_free)
                        phase_count[0] = phase_count[0] ^ 1
                    T.barrier_wait(bar_final, (b_i - begin_idx) & 1)
                    for h_i, d_i in T.Parallel(H_per_block, D // 4):
                        acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]
                    T.lma_wait()
                    # T.barrier_arrive(bar_final_free)
                    if is_split:
                        T.copy(
                            acc_o_r_0, Output[b_i, s_i, H0:H1, D // 2 : D // 2 + D // 4]
                        )
                        T.copy(acc_o_r_1, Output[b_i, s_i, H0:H1, D // 2 + D // 4 : D])
                    else:
                        T.copy(
                            acc_o_r_0,
                            Output_partial[
                                n_split_idx + NumSplits[b_i],
                                s_i,
                                H0:H1,
                                D // 2 : D // 2 + D // 4,
                            ],
                        )
                        T.copy(
                            acc_o_r_1,
                            Output_partial[
                                n_split_idx + NumSplits[b_i],
                                s_i,
                                H0:H1,
                                D // 2 + D // 4 : D,
                            ],
                        )
                elif tid >= 512:
                    mask_local = T.alloc_local([4], "bool")
                    indices_local = T.alloc_local([4], indices_dtype)
                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], "int32")
                    # producer: 128 ldg_ty 16
                    ldg_tx = (tid - 512) % 8
                    ldg_ty = (tid - 512) // 8
                    ldg_scale_tx = (tid - 512) % 2
                    ldg_scale_ty = (tid - 512) // 2
                    for i_i in range(start_block_idx, end_block_idx):
                        for r in T.unroll(4):
                            kperm_indices_local[r] = Indices[
                                b_i,
                                s_i,
                                g_i,
                                (i_i) * block_I
                                + ((r * 16 + ldg_ty) % 8) * (block_I // 8)
                                + (r * 16 + ldg_ty) // 8,
                            ]
                        for r in T.unroll(4):
                            kperm_mask_local[r] = (
                                kperm_indices_local[r] >= 0
                                and kperm_indices_local[r] < seq_len_kv
                            )
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r],
                                kperm_indices_local[r],
                                (seq_len_kv),
                            )
                        T.barrier_wait(bar_kv_mask_free, (phase_count[0] & 1) ^ 1)
                        if ldg_tx == 0:
                            for r in T.unroll(4):
                                is_kv_valid[
                                    ((r * 16 + ldg_ty) % 8) * (block_I // 8)
                                    + (r * 16 + ldg_ty) // 8
                                ] = kperm_mask_local[r]
                                kv_indices[(r * 16 + ldg_ty)] = kperm_indices_local[r]
                        T.barrier_wait(bar_kv0_free, (phase_count[0] & 1) ^ 1)
                        T.annotate_layout(
                            {
                                KV_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_l[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(4):
                                    pass
                                    T.copy(
                                        K_pe[
                                            kperm_indices_local[r],
                                            g_i,
                                            32 * u + ldg_tx * 4 + v,
                                        ],
                                        KV_shared_l[
                                            r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )

                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_ready)
                        T.barrier_arrive(bar_indices_ready)
                        T.barrier_wait(bar_indices_ready, (phase_count[0] & 1))

                        for c in T.vectorized(2):
                            T.copy(
                                Quant_scales[
                                    kv_indices[ldg_scale_ty],
                                    g_i,
                                    128 + ldg_scale_tx * 2 + c,
                                ],
                                Quant_shared[ldg_scale_ty, ldg_scale_tx * 2 + c],
                                src_robust_desc=kv_robust_desc,
                            )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (phase_count[0] & 1) ^ 1)

                        # load k rope
                        T.annotate_layout(
                            {
                                K_tail_shared[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    K_tail_shared[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for v in T.vectorized(8):
                                pass
                                T.copy(
                                    K_pe[
                                        kperm_indices_local[r],
                                        g_i,
                                        D // 2 + 8 + ldg_tx * 8 + v,
                                    ],
                                    K_tail_shared[r * 16 + ldg_ty, ldg_tx * 8 + v],
                                    force_async_copy=True,
                                    src_robust_desc=kv_robust_desc,
                                )

                        # load k4-k7
                        # T.annotate_layout(
                        #         { KV_shared_r[:, :]: tilelang.layout.make_sqmma_swizzled_layout(KV_shared_r[:, :], k_major=True) },
                        #         allow_reannotation=True,
                        #         allow_buffer_region=True)
                        T.annotate_layout(
                            {
                                KV_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    KV_shared_r[:, :], k_major=True
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(4):
                                    pass
                                    T.copy(
                                        K_pe[
                                            kperm_indices_local[r],
                                            g_i,
                                            D // 4 + 32 * u + ldg_tx * 4 + v,
                                        ],
                                        KV_shared_r[
                                            r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)
                        phase_count[0] = phase_count[0] ^ 1

    Block_M = 8
    NUM_THREADS = Block_M * 32
    HEAD_DIM_V = 512
    ELEMS_PER_THREAD = HEAD_DIM_V // 32
    NUM_LSE_PER_THREAD = T.ceildiv(max_nums_splits, 32)

    @T.macro
    def dsa_combine(
        NumSplits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor([batch + num_mp_parts, seq_len, num_heads], accum_dtype),  # type: ignore
        # Shape: [total_num_splits, num_heads, seq_len, dim] - aligned with mate
        Output_partial: T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        ),  # type: ignore
        Output: T.Tensor([batch, seq_len, num_heads, dim], dtype),  # type: ignore
        Lse: T.Tensor([batch, seq_len, num_heads], accum_dtype),  # type: ignore
    ):
        # Each block handles one (batch, seq, head) triplet
        # Grid: [batch, seq_len, num_heads]
        with T.Kernel(
            batch, T.ceildiv(seq_len * num_heads, Block_M), threads=NUM_THREADS
        ) as (bx, by):
            batch_idx = bx
            m_block_idx = by
            sLseScale = T.alloc_shared([Block_M, max_nums_splits + 1], accum_dtype)
            tid = T.get_thread_binding()
            warp_idx = tid // 32
            lane_idx = tid % 32
            glse = T.view(
                glse, [batch + num_mp_parts, seq_len * num_heads], dtype=accum_dtype
            )
            Output_partial = T.view(
                Output_partial,
                [batch + num_mp_parts, seq_len * num_heads, dim],
                dtype=accum_dtype,
            )
            Output = T.view(Output, [batch, seq_len * num_heads, dim], dtype=dtype)
            Lse = T.view(Lse, [batch, seq_len * num_heads], dtype=accum_dtype)
            # Get split range for this batch
            split_start = NumSplits[batch_idx]
            split_end = NumSplits[batch_idx + 1]
            my_num_splits = split_end - split_start
            if my_num_splits > 1:
                num_cur_valid_q_seqs = T.alloc_var(T.int32)
                num_q_seqs = seq_len * num_heads
                num_cur_valid_q_seqs = T.min(
                    num_q_seqs - m_block_idx * Block_M, Block_M
                )
                # Shared memory for LSE values from all splits
                for loop in range(tid, my_num_splits * Block_M, NUM_THREADS):
                    split_idx = loop // Block_M
                    seq_idx = loop % Block_M
                    if seq_idx < num_cur_valid_q_seqs:
                        sLseScale[seq_idx, split_idx] = glse[
                            split_start + split_idx, seq_idx + m_block_idx * Block_M
                        ]
                    else:
                        sLseScale[seq_idx, split_idx] = -T.infinity(accum_dtype)
                T.sync_threads()

                if warp_idx < num_cur_valid_q_seqs:
                    sLse = T.alloc_local(
                        [NUM_LSE_PER_THREAD], accum_dtype
                    )  # Max 32 splits
                    # Step 1: Load all LSE values for this (batch, seq, head)
                    for i in T.unroll(NUM_LSE_PER_THREAD):
                        if i * 32 + lane_idx < my_num_splits:
                            sLse[i] = sLseScale[warp_idx, i * 32 + lane_idx]
                        else:
                            sLse[i] = -T.infinity(accum_dtype)

                    # Step 2: Compute max LSE
                    max_lse = T.alloc_local([1], accum_dtype)
                    max_lse[0] = -(2**30) * sm_scale

                    for i in T.unroll(NUM_LSE_PER_THREAD):
                        max_lse[0] = T.max(max_lse[0], sLse[i])
                    # T.reduce_max(max_lse[0], max_lse[0], dim=-1,clear=True)
                    # for offset in [16, 8, 4, 2, 1]:
                    #     temp = T.shfl_xor(0xFFFFFFFF, max_lse[0], offset)
                    #     max_lse[0] = T.max(max_lse[0], temp)
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 16))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 8))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 4))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 2))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 1))

                    # Step 3: Compute sum of exp(lse - max_lse)
                    sum_exp = T.alloc_local([1], accum_dtype)
                    sum_exp[0] = 0.0
                    temp = T.alloc_local([NUM_LSE_PER_THREAD], accum_dtype)
                    for i in T.unroll(NUM_LSE_PER_THREAD):
                        temp[i] = T.exp2(sLse[i] - max_lse[0])
                    for i in T.unroll(NUM_LSE_PER_THREAD):
                        sum_exp[0] += temp[i]
                    # T.reduce_sum(sum_exp[0], sum_exp[0], dim=-1,clear=True)
                    # for offset in [16, 8, 4, 2, 1]:
                    #     temp = T.shfl_xor(0xFFFFFFFF, sum_exp[0], offset)
                    #     sum_exp[0] += temp
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 16)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 8)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 4)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 2)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 1)
                    # Step 4: Compute global LSE
                    # LOG2E = 1.4426950408889634
                    global_lse = T.alloc_local([1], accum_dtype)
                    if sum_exp[0] == 0.0 or sum_exp[0] != sum_exp[0]:
                        global_lse[0] = T.infinity(accum_dtype)
                    else:
                        global_lse[0] = T.log2(sum_exp[0]) + max_lse[0]

                    if lane_idx == 0:
                        Lse[batch_idx, warp_idx + m_block_idx * Block_M] = global_lse[0]

                    # Step 5: Compute scales and accumulate output
                    for i in T.unroll(NUM_LSE_PER_THREAD):
                        if i * 32 + lane_idx < my_num_splits:
                            sLseScale[warp_idx, i * 32 + lane_idx] = T.exp2(
                                sLse[i] - global_lse[0]
                            )
                    T.sync_threads()

                    # Accumulate weighted output
                    # Each thread handles dim/128 elements
                    result = T.alloc_local([ELEMS_PER_THREAD], accum_dtype)
                    T.clear(result)

                    for split in T.serial(my_num_splits):
                        scale = sLseScale[warp_idx, split]
                        if scale != 0.0:
                            for i in T.unroll(ELEMS_PER_THREAD):
                                partial_val = Output_partial[
                                    split_start + split,
                                    warp_idx + m_block_idx * Block_M,
                                    lane_idx + i * 32,
                                ]
                                result[i] += scale * partial_val

                    # Write output
                    for i in T.unroll(ELEMS_PER_THREAD):
                        Output[
                            batch_idx,
                            warp_idx + m_block_idx * Block_M,
                            lane_idx + i * 32,
                        ] = T.Cast(dtype, result[i])

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, seq_len, num_heads, dim + tail_dim], dtype),  # type: ignore
        KV: T.Tensor([seq_len_kv, kv_group, dim_bytes], kv_latent_dtype),  # type: ignore
        K_pe: T.Tensor([seq_len_kv, kv_group, dim_bytes // 2], dtype),  # type: ignore
        Quant_scales: T.Tensor([seq_len_kv, kv_group, dim_bytes // 4], T.float32),  # type: ignore
        Indices: T.Tensor([batch, seq_len, kv_group, topk], indices_dtype),  # type: ignore
        tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        NumSplits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor([batch + num_mp_parts, seq_len, num_heads], accum_dtype),  # type: ignore
        Output_partial: T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        ),  # type: ignore
        Output: T.Tensor([batch, seq_len, num_heads, dim], dtype),  # type: ignore
        Lse: T.Tensor([batch, seq_len, num_heads], accum_dtype),  # type: ignore
    ):
        dsa_decode_split(
            Q,
            KV,
            K_pe,
            Quant_scales,
            Indices,
            tile_scheduler_metadata,
            NumSplits,
            glse,
            Output_partial,
            Output,
            Lse,
        )
        dsa_combine(NumSplits, glse, Output_partial, Output, Lse)

    return main_split


def tilelang_flashmla_interface(
    q,
    kv,
    indices,
    tile_scheduler_metadata,
    num_splits,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    b, seq_len, heads, dim_plus_tail_dim = q.shape
    seq_len_kv, kv_group, _ = kv.shape
    #  In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
    #         - The shape of the tensor `k_cache` is (num_blocks*page_block_size*num_heads_k, head_dim), and num_heads_k must be 1.
    #         - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
    #         - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
    #         - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.
    # assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v
    dim_bytes = 656
    assert kv.shape[-1] == dim_bytes
    tail_dim = dim_plus_tail_dim - dim
    _, _, _, topk = indices.shape
    num_mp_parts, _ = tile_scheduler_metadata.shape
    assert indices.shape == (b, seq_len, kv_group, topk)
    assert tile_scheduler_metadata.shape == (num_mp_parts, 8)
    assert num_splits.shape == (b + 1,)
    glse = torch.empty(
        [num_mp_parts + b, seq_len, heads], dtype=torch.float32, device=q.device
    )
    out_partial = torch.empty(
        [num_mp_parts + b, seq_len, heads, d_v], dtype=torch.float32, device=q.device
    )
    out = torch.empty([b, seq_len, heads, d_v], dtype=q.dtype, device=q.device)
    lse = torch.empty([b, seq_len, heads], dtype=torch.float32, device=q.device)
    # kernel = sparse_attention_fwd_kernel_v1(
    threads = 640
    if num_mp_parts <= 32:
        kernel = sparse_attention_fwd_kernel(
            heads,
            dim,
            tail_dim,
            topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            threads=threads,
            max_nums_splits=32,
        )
    elif num_mp_parts <= 64:
        kernel = sparse_attention_fwd_kernel(
            heads,
            dim,
            tail_dim,
            topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            threads=threads,
            max_nums_splits=64,
        )
    if verbose:
        kernel.show_source()
    kv_latent_f8 = kv.view(torch.float8_e4m3fn)
    k_rope = kv.view(torch.bfloat16)
    scales = kv.view(torch.float32)
    kernel(
        q,
        kv_latent_f8,
        k_rope,
        scales,
        indices,
        tile_scheduler_metadata,
        num_splits,
        glse,
        out_partial,
        out,
        lse,
    )
    return out, lse
