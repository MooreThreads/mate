# ruff: noqa
"""
MODEL1 sparse MLA prefill with a pipelined producer/consumer schedule.

Compared with the DeepSeek V3.2 sparse prefill kernel:
- q/k/v head_dim is 512, so there is no 512 + 64 tail split.
- The sparse candidates can come from both main and extra kv tensors.
- Each source has its own dynamic effective topk length.
- This remains a bf16 prefill-only research kernel.
"""

import torch
import tilelang
from tilelang import language as T
from tvm import tir

from ...utils import cosize
from .sparse_mla_prefill_common import (
    SPARSE_PREFILL_COMPILE_FLAGS,
    SPARSE_PREFILL_PASS_CONFIGS,
    optional_prefill_attn_sink,
    require_token_lengths,
)
from ...execution_context import raise_complete_if_dry_run


# tilelang.disable_cache()


@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs=SPARSE_PREFILL_PASS_CONFIGS,
    compile_flags=SPARSE_PREFILL_COMPILE_FLAGS,
)
def sparse_attention_fwd_kernel_model1(
    num_heads,
    dim,
    topk,
    *,
    extra_topk=0,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_i=64,
    threads=640,
    has_attn_sink=False,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert topk % block_i == 0, "topk must be a multiple of block_i"
    if extra_topk > 0:
        assert extra_topk % block_i == 0, "extra_topk must be a multiple of block_i"
    if sm_scale is None:
        logits_scale = (1.0 / dim) ** 0.5
    else:
        logits_scale = sm_scale
    sm_scale = logits_scale * 1.44269504
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    seq_len_kv_extra = T.dynamic("seq_len_kv_extra")

    head_kv = num_heads // kv_group
    q_shape = [seq_len, num_heads, dim]
    kv_shape = [seq_len_kv, kv_group, dim]
    o_shape = [seq_len, num_heads, dim]
    lse_shape = [seq_len, num_heads]
    max_logits_shape = [seq_len, num_heads]
    indices_shape = [seq_len, kv_group, topk]
    extra_indices_shape = [seq_len, kv_group, extra_topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    dtype_bytes = 2
    q_cosize = cosize(q_shape)
    kv_cosize = cosize(kv_shape)
    extra_kv_cosize = cosize([seq_len_kv_extra, kv_group, dim])

    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_head_kv != head_kv:
        assert kv_group == 1
    num_i_orig = tilelang.cdiv(topk, block_i)
    num_i_extra = tilelang.cdiv(extra_topk, block_i) if extra_topk > 0 else 0
    total_num_i = num_i_orig + num_i_extra
    dim_qk = dim
    lanes_per_vec = block_i // 8

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        head_repeats = head_kv // 64
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else 64

    @T.prim_func
    def dsa_prefill(
        q: T.Tensor(q_shape, dtype),
        kv: T.Tensor(kv_shape, dtype),
        indices: T.Tensor(indices_shape, indices_dtype),
        topk_length: T.Tensor([seq_len], indices_dtype),
        extra_kv: T.Tensor([seq_len_kv_extra, kv_group, dim], dtype),
        extra_indices: T.Tensor(extra_indices_shape, indices_dtype),
        extra_topk_length: T.Tensor([seq_len], indices_dtype),
        attn_sink: T.Tensor([num_heads], accum_dtype),
        output: T.Tensor(o_shape, dtype),
        max_logits_out: T.Tensor(max_logits_shape, accum_dtype),
        lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * head_repeats, kv_group, threads=threads) as (bx, by):
            q_shared_l = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, dim_qk // 2], dtype)
            kv_shared_l = T.alloc_shared([block_i, dim_qk // 2], dtype)
            kv_shared_r = T.alloc_shared([block_i, dim_qk // 2], dtype)
            v_shared_0 = T.alloc_shared([block_i, dim_qk // 4], dtype)
            v_shared_1 = T.alloc_shared([block_i, dim_qk // 4], dtype)

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            lse_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")

            bar_q = T.alloc_barrier(arrive_count=512)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_read_ready = T.alloc_barrier(arrive_count=256)
            bar_kv0_free = T.alloc_barrier(arrive_count=256)
            bar_kv1_free = T.alloc_barrier(arrive_count=256)

            bar_vl0_ready = T.alloc_barrier(arrive_count=256)
            bar_vl1_ready = T.alloc_barrier(arrive_count=256)
            bar_vr0_ready = T.alloc_barrier(arrive_count=256)
            bar_vr1_ready = T.alloc_barrier(arrive_count=256)
            bar_vl0_free = T.alloc_barrier(arrive_count=256)
            bar_vl1_free = T.alloc_barrier(arrive_count=256)

            bar_p_ready = T.alloc_barrier(arrive_count=256)
            bar_final = T.alloc_barrier(arrive_count=256)

            q_robust_desc = T.make_robust_desc(
                T.address_of(q[0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            kv_robust_desc = T.make_robust_desc(
                T.address_of(kv[0, 0, 0]),
                kv_cosize * dtype_bytes,
            )
            extra_kv_robust_desc = T.make_robust_desc(
                T.address_of(extra_kv[0, 0, 0]),
                extra_kv_cosize * dtype_bytes,
            )

            T.sync_threads()

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)
            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * 64
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()

            if tid < 512:
                T.copy(
                    q[s_i, h0:h1, 0 : dim_qk // 2],
                    q_shared_l,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc,
                )
                T.copy(
                    q[s_i, h0:h1, dim_qk // 2 : dim_qk],
                    q_shared_r,
                    force_async_copy=True,
                    src_robust_desc=q_robust_desc,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                T.barrier_arrive(bar_q)
                T.barrier_wait(bar_q, 0)

            if tid < 256:
                sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                max_logits = T.alloc_fragment([heads_per_block], accum_dtype)
                acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                acc_o_l_0 = T.alloc_fragment(
                    [heads_per_block, dim_qk // 4], accum_dtype
                )
                acc_o_l_1 = T.alloc_fragment(
                    [heads_per_block, dim_qk // 4], accum_dtype
                )
                kv_reg_l = T.alloc_local([64], dtype)
                consumer0_ldg_tx = tid % 8
                consumer0_ldg_ty = tid // 8
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))
                T.fill(acc_o_l_0, 0)
                T.fill(acc_o_l_1, 0)
                has_any_valid = T.alloc_var("bool")
                has_any_valid = False

                for i_i in range(total_num_i):
                    T.barrier_wait(bar_kv0_ready, (i_i & 1))
                    block_has_valid = T.alloc_var("bool")
                    block_has_valid = False
                    for valid_i in range(block_i):
                        block_has_valid = block_has_valid or is_kv_valid[valid_i]
                    has_any_valid = has_any_valid or block_has_valid

                    for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            is_kv_valid[bi_i % 8 * 8 + bi_i // 8],
                            0,
                            -(2**30),
                        )

                    T.annotate_layout(
                        {
                            kv_shared_l[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                kv_shared_l[:, :],
                                k_major=True,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    T.gemm(
                        q_shared_l,
                        kv_shared_l[:, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )

                    for r in T.unroll(2):
                        for u in T.unroll(4):
                            for v in T.vectorized(8):
                                kv_reg_l[r * 32 + u * 8 + v] = kv_shared_l[
                                    ((consumer0_ldg_ty + r * 32) % 8) * (block_i // 8)
                                    + (consumer0_ldg_ty + r * 32) // 8,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ]
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.lma_wait()
                    T.barrier_arrive(bar_kv0_free)

                    T.barrier_wait(bar_kv1_ready, (i_i & 1))
                    T.annotate_layout(
                        {
                            kv_shared_r[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                kv_shared_r[:, :],
                                k_major=True,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    T.gemm(
                        q_shared_r,
                        kv_shared_r[:, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.barrier_arrive(bar_kv1_read_ready)
                    T.copy(m_i, m_i_prev)
                    T.warpgroup_wait(0)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(heads_per_block):
                        m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
                    for h_i in T.Parallel(heads_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )

                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(heads_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
                        acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, acc_s_cast)
                    for i, t in T.Parallel(heads_per_block, 8):
                        base = t * lanes_per_vec
                        for l in T.vectorized(lanes_per_vec):
                            scores_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

                    T.lma_wait()
                    T.barrier_arrive(bar_p_ready)

                    T.annotate_layout(
                        {
                            v_shared_0[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                v_shared_0[:, :],
                                k_major=False,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_0[
                                    r * 32 + consumer0_ldg_ty,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + u * 8 + v]
                    T.lma_wait()
                    T.barrier_arrive(bar_vl0_ready)
                    T.barrier_wait(bar_vl0_ready, (i_i & 1))

                    T.gemm(
                        scores_shared,
                        v_shared_0,
                        acc_o_l_0,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.annotate_layout(
                        {
                            v_shared_1[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                v_shared_1[:, :],
                                k_major=False,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_1[
                                    r * 32 + consumer0_ldg_ty,
                                    64 * u + consumer0_ldg_tx * 8 + v,
                                ] = kv_reg_l[r * 32 + (u + 2) * 8 + v]

                    T.warpgroup_wait(0)
                    T.barrier_arrive(bar_vl0_free)

                    T.lma_wait()
                    T.barrier_arrive(bar_vl1_ready)
                    T.barrier_wait(bar_vl1_ready, (i_i & 1))

                    T.gemm(
                        scores_shared,
                        v_shared_1,
                        acc_o_l_1,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(bar_vl1_free)

                for h_i in T.Parallel(heads_per_block):
                    if sumexp[h_i] > 0 and has_any_valid:
                        sumexp_inv[h_i] = 1 / sumexp[h_i]
                        max_logits[h_i] = m_i[h_i] * logits_scale
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    else:
                        sumexp_inv[h_i] = 0
                        max_logits[h_i] = -T.infinity(accum_dtype)
                        sumexp[h_i] = T.infinity(accum_dtype)
                    sum_exp_inv_shared[h_i] = sumexp_inv[h_i]
                    lse_shared[h_i] = sumexp[h_i]
                T.barrier_arrive(bar_final)
                for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                    acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
                    acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]

                if has_attn_sink:
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        if sumexp_inv[h_i] > 0:
                            sink_scale = 1 / (
                                1
                                + T.exp2(
                                    attn_sink[h0 + h_i] * 1.4426950408889634
                                    - sumexp[h_i]
                                )
                            )
                            acc_o_l_0[h_i, d_i] *= sink_scale
                            acc_o_l_1[h_i, d_i] *= sink_scale

                T.copy(acc_o_l_0, output[s_i, h0:h1, 0 : dim_qk // 4])
                T.copy(acc_o_l_1, output[s_i, h0:h1, dim_qk // 4 : dim_qk // 2])
                T.copy(max_logits, max_logits_out[s_i, h0:h1])
                for h_i in T.Parallel(heads_per_block):
                    lse[s_i, h0 + h_i] = sumexp[h_i] * 0.6931471805599453
            elif tid < 512:
                acc_o_r_0 = T.alloc_fragment(
                    [heads_per_block, dim_qk // 4], accum_dtype
                )
                acc_o_r_1 = T.alloc_fragment(
                    [heads_per_block, dim_qk // 4], accum_dtype
                )
                kv_reg_r = T.alloc_local([64], dtype)
                T.fill(acc_o_r_0, 0)
                T.fill(acc_o_r_1, 0)

                consumer1_ldg_tx = (tid - 256) % 8
                consumer1_ldg_ty = (tid - 256) // 8

                for i_i in range(total_num_i):
                    T.barrier_wait(bar_kv1_read_ready, (i_i & 1))
                    for r in T.unroll(2):
                        for u in T.unroll(4):
                            for v in T.vectorized(8):
                                kv_reg_r[r * 32 + u * 8 + v] = kv_shared_r[
                                    ((consumer1_ldg_ty + r * 32) % 8) * (block_i // 8)
                                    + (consumer1_ldg_ty + r * 32) // 8,
                                    64 * u + consumer1_ldg_tx * 8 + v,
                                ]

                    T.lma_wait()
                    T.barrier_arrive(bar_kv1_free)
                    T.barrier_wait(bar_vl0_free, (i_i & 1))
                    T.annotate_layout(
                        {
                            v_shared_0[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                v_shared_0[:, :],
                                k_major=False,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_0[
                                    r * 32 + consumer1_ldg_ty,
                                    64 * u + consumer1_ldg_tx * 8 + v,
                                ] = kv_reg_r[r * 32 + u * 8 + v]

                    T.lma_wait()
                    T.barrier_arrive(bar_vr0_ready)
                    T.barrier_wait(bar_vr0_ready, (i_i & 1))

                    T.barrier_wait(bar_p_ready, (i_i & 1))
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                        acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                    T.gemm(
                        scores_shared,
                        v_shared_0,
                        acc_o_r_0,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.wait_wgmma(0)

                    T.barrier_wait(bar_vl1_free, (i_i & 1))
                    T.annotate_layout(
                        {
                            v_shared_1[
                                :, :
                            ]: tilelang.layout.make_sqmma_swizzled_layout(
                                v_shared_1[:, :],
                                k_major=False,
                            )
                        },
                        allow_reannotation=True,
                        allow_buffer_region=True,
                    )
                    for r in T.unroll(2):
                        for u in T.unroll(2):
                            for v in T.vectorized(8):
                                v_shared_1[
                                    r * 32 + consumer1_ldg_ty,
                                    64 * u + consumer1_ldg_tx * 8 + v,
                                ] = kv_reg_r[r * 32 + (u + 2) * 8 + v]
                    T.lma_wait()
                    T.barrier_arrive(bar_vr1_ready)
                    T.barrier_wait(bar_vr1_ready, (i_i & 1))

                    T.gemm(
                        scores_shared,
                        v_shared_1,
                        acc_o_r_1,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1,
                    )
                    T.wait_wgmma(0)

                T.barrier_wait(bar_final, 0)
                for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                    acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
                    acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]

                if has_attn_sink:
                    for h_i, d_i in T.Parallel(heads_per_block, dim_qk // 4):
                        if sum_exp_inv_shared[h_i] > 0:
                            sink_scale = 1 / (
                                1
                                + T.exp2(
                                    attn_sink[h0 + h_i] * 1.4426950408889634
                                    - lse_shared[h_i]
                                )
                            )
                            acc_o_r_0[h_i, d_i] *= sink_scale
                            acc_o_r_1[h_i, d_i] *= sink_scale

                T.copy(
                    acc_o_r_0,
                    output[s_i, h0:h1, dim_qk // 2 : dim_qk // 2 + dim_qk // 4],
                )
                T.copy(
                    acc_o_r_1, output[s_i, h0:h1, dim_qk // 2 + dim_qk // 4 : dim_qk]
                )
            else:
                kperm_mask_local = T.alloc_local([4], "bool")
                kperm_indices_local = T.alloc_local([4], indices_dtype)
                topk_len_local = T.alloc_local([1], indices_dtype)
                extra_topk_len_local = T.alloc_local([1], indices_dtype)
                producer_ldg_tx = (tid - 512) % 8
                producer_ldg_ty = (tid - 512) // 8
                topk_len_local[0] = topk_length[s_i]
                extra_topk_len_local[0] = extra_topk_length[s_i]

                for i_i in range(total_num_i):
                    if i_i < num_i_orig:
                        orig_block_index = i_i
                        for r in T.unroll(4):
                            token_pos = (
                                orig_block_index * block_i
                                + ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + producer_ldg_ty) // 8
                            )
                            kperm_indices_local[r] = indices[s_i, g_i, token_pos]
                            kperm_mask_local[r] = (
                                kperm_indices_local[r] >= 0
                                and kperm_indices_local[r] < seq_len_kv
                                and token_pos < topk_len_local[0]
                            )
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r],
                                kperm_indices_local[r],
                                0,
                            )

                        T.barrier_wait(bar_kv0_free, (i_i & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        kv_shared_l[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        for r in T.unroll(4):
                            is_kv_valid[
                                ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + producer_ldg_ty) // 8
                            ] = kperm_mask_local[r]
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (i_i & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    T.copy(
                                        kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            dim_qk // 2
                                            + 64 * u
                                            + producer_ldg_tx * 8
                                            + v,
                                        ],
                                        kv_shared_r[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)
                    else:
                        extra_block_index = i_i - num_i_orig
                        for r in T.unroll(4):
                            token_pos = (
                                extra_block_index * block_i
                                + ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + producer_ldg_ty) // 8
                            )
                            kperm_indices_local[r] = extra_indices[s_i, g_i, token_pos]
                            kperm_mask_local[r] = (
                                kperm_indices_local[r] >= 0
                                and kperm_indices_local[r] < seq_len_kv_extra
                                and token_pos < extra_topk_len_local[0]
                            )
                            kperm_indices_local[r] = T.if_then_else(
                                kperm_mask_local[r],
                                kperm_indices_local[r],
                                0,
                            )

                        T.barrier_wait(bar_kv0_free, (i_i & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared_l[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_l[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    T.copy(
                                        extra_kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        kv_shared_l[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=extra_kv_robust_desc,
                                    )
                        for r in T.unroll(4):
                            is_kv_valid[
                                ((r * 16 + producer_ldg_ty) % 8) * (block_i // 8)
                                + (r * 16 + producer_ldg_ty) // 8
                            ] = kperm_mask_local[r]
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_ready)

                        T.barrier_wait(bar_kv1_free, (i_i & 1) ^ 1)
                        T.annotate_layout(
                            {
                                kv_shared_r[
                                    :, :
                                ]: tilelang.layout.make_sqmma_swizzled_layout(
                                    kv_shared_r[:, :],
                                    k_major=True,
                                )
                            },
                            allow_reannotation=True,
                            allow_buffer_region=True,
                        )
                        for r in T.unroll(4):
                            for u in T.unroll(4):
                                for v in T.vectorized(8):
                                    T.copy(
                                        extra_kv[
                                            kperm_indices_local[r],
                                            g_i,
                                            dim_qk // 2
                                            + 64 * u
                                            + producer_ldg_tx * 8
                                            + v,
                                        ],
                                        kv_shared_r[
                                            r * 16 + producer_ldg_ty,
                                            64 * u + producer_ldg_tx * 8 + v,
                                        ],
                                        force_async_copy=True,
                                        src_robust_desc=extra_kv_robust_desc,
                                    )
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        T.barrier_arrive(bar_kv1_ready)

    return dsa_prefill


def sparse_mla_fwd_interface_model1(
    q,
    kv,
    indices,
    extra_kv=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
    return_max_logits: bool = False,
):
    is_causal = True
    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv.dtype == torch.bfloat16, "kv must be bfloat16"
    assert indices.dtype == torch.int32, "indices must be int32"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    seq_len, heads, dim_q = q.shape
    seq_len_kv, kv_group, _ = kv.shape

    dim = d_v
    assert kv_group == 1, (
        "Only MQA (kv_group == 1) is validated for model1 sparse prefill"
    )
    assert dim_q == dim, (
        f"MODEL1 expects q last_dim == kv last_dim == {dim}, got q={dim_q}, kv={kv.shape[-1]}"
    )
    assert kv.shape[-1] == dim, (
        f"MODEL1 expects kv last_dim == {dim}, got {kv.shape[-1]}"
    )
    assert dim == 512, f"MODEL1 kernel currently expects dim == d_v == 512, got {dim}"

    _, _, topk = indices.shape
    assert indices.shape == (seq_len, kv_group, topk)

    extra_topk = 0
    if extra_kv is not None:
        assert extra_indices is not None
        assert extra_kv.dtype == torch.bfloat16, "extra_kv must be bfloat16"
        assert extra_indices.dtype == torch.int32, "extra_indices must be int32"
        assert extra_kv.is_contiguous() and extra_indices.is_contiguous()
        assert extra_kv.shape[-1] == dim
        _, _, extra_topk = extra_indices.shape
        assert extra_indices.shape == (seq_len, kv_group, extra_topk)
    else:
        assert extra_indices is None, "extra_indices requires extra_kv"
        assert extra_topk_length is None, "extra_topk_length requires extra_kv"

    topk_length = require_token_lengths(
        topk_length, seq_len, topk, q.device, "topk_length"
    )
    if extra_kv is not None:
        extra_topk_length = require_token_lengths(
            extra_topk_length,
            seq_len,
            extra_topk,
            q.device,
            "extra_topk_length",
        )

    if extra_kv is None:
        # Keep one TileLang body. With extra_topk == 0 the extra branch is
        # compile-time dead; these alias arguments are not touched by math.
        extra_kv = kv
        extra_indices = indices[:, :, :0]
        extra_topk_length = topk_length

    kernel_kwargs = {
        "extra_topk": extra_topk,
        "kv_group": kv_group,
        "sm_scale": sm_scale,
        "is_causal": is_causal,
        "threads": threads,
        "has_attn_sink": attn_sink is not None,
    }
    kernel = sparse_attention_fwd_kernel_model1(heads, dim, topk, **kernel_kwargs)
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()

    attn_sink_arg, _ = optional_prefill_attn_sink(attn_sink, heads, q.device)
    out = kernel(
        q,
        kv,
        indices,
        topk_length,
        extra_kv,
        extra_indices,
        extra_topk_length,
        attn_sink_arg,
    )

    out_tensor, max_logits, lse_tensor = out
    if return_max_logits:
        return out_tensor, max_logits, lse_tensor
    return out_tensor, lse_tensor
