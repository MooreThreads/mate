# ruff: noqa

import torch

if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int16
if not hasattr(torch, "uint32"):
    torch.uint32 = torch.int32
if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64
import tilelang
from tilelang import language as T
from tvm import tir

from ...utils import cosize
from .sparse_mla_decode_scheduled_common import (
    SCHEDULED_DECODE_COMPILE_FLAGS,
    SCHEDULED_DECODE_PASS_CONFIGS,
    make_scheduled_decode_combine,
    make_scheduled_decode_finalize_left,
    make_scheduled_decode_finalize_right,
    make_scheduled_decode_indices_loader,
    make_scheduled_decode_online_softmax,
    make_scheduled_decode_stage_value_shared,
    require_batch_lengths,
    prepare_scheduled_decode_runtime,
)

from ...execution_context import raise_complete_if_dry_run


@tilelang.jit(
    out_idx=[],
    pass_configs=SCHEDULED_DECODE_PASS_CONFIGS,
    compile_flags=SCHEDULED_DECODE_COMPILE_FLAGS,
)
def sparse_attention_decode_fwd_scheduled_kernel_model1(
    num_heads,
    dim,
    *,
    has_extra=False,
    kv_group=1,
    sm_scale=None,
    block_i=64,
    threads=640,
    max_nums_splits=32,
    has_attn_sink=False,
    page_block_size=64,
    extra_page_block_size=64,
    page_stride_bytes=None,
    extra_page_stride_bytes=None,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504
    topk = T.dynamic("topk")
    extra_topk = T.dynamic("extra_topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    dim_bytes = 584
    rope_bytes_offset = 448
    scale_bytes_offset = 576
    if page_stride_bytes is None:
        page_stride_bytes = (
            (page_block_size * dim_bytes + scale_bytes_offset - 1)
            // scale_bytes_offset
            * scale_bytes_offset
        )
    if extra_page_stride_bytes is None:
        extra_page_stride_bytes = (
            (extra_page_block_size * dim_bytes + scale_bytes_offset - 1)
            // scale_bytes_offset
            * scale_bytes_offset
        )
    page_block_bytes = page_stride_bytes
    page_block_bytes_extra = extra_page_stride_bytes

    num_blocks = T.dynamic("num_blocks")
    num_blocks_extra = T.dynamic("num_blocks_extra")
    page_block_size_extra = extra_page_block_size
    seq_len_kv = num_blocks * page_block_size
    seq_len_kv_extra = num_blocks_extra * page_block_size_extra
    num_mp_parts = T.dynamic("num_mp_parts")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim]
    o_shape = [batch, seq_len, num_heads, dim]
    lse_shape = [batch, num_heads, seq_len]
    indices_shape = [batch, seq_len, kv_group, topk]
    extra_indices_shape = [batch, seq_len, kv_group, extra_topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    kv_latent_dtype = "float8_e4m3"
    dtype_bytes = 2
    q_cosize = cosize(q_shape)
    kv_nope_cosize = cosize([num_blocks, page_block_bytes])
    kv_rope_cosize = cosize([num_blocks, page_block_bytes // 2])
    quant_scales_cosize = cosize([num_blocks, page_block_bytes])
    extra_kv_nope_cosize = cosize([num_blocks_extra, page_block_bytes_extra])
    extra_kv_rope_cosize = cosize([num_blocks_extra, page_block_bytes_extra // 2])
    extra_quant_scales_cosize = cosize([num_blocks_extra, page_block_bytes_extra])

    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 64)
    if padded_head_kv != head_kv:
        assert kv_group == 1

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        head_repeats = head_kv // 64
    else:
        head_repeats = 1

    heads_per_block = padded_head_kv if head_repeats == 1 else 64
    dsa_combine = make_scheduled_decode_combine(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        dim=dim,
        num_mp_parts=num_mp_parts,
        dtype=dtype,
        accum_dtype=accum_dtype,
        max_nums_splits=max_nums_splits,
        has_attn_sink=has_attn_sink,
        max_lse_init=-(2**30),
    )

    update_online_softmax = make_scheduled_decode_online_softmax(
        h_per_block=heads_per_block,
        block_i=block_i,
        out_width=128,
        accum_dtype=accum_dtype,
        sm_scale=sm_scale,
    )
    finalize_left = make_scheduled_decode_finalize_left(
        h_per_block=heads_per_block,
        out_width=128,
        accum_dtype=accum_dtype,
        sm_scale=sm_scale,
        has_attn_sink=has_attn_sink,
        use_strict_valid=True,
        add_denominator_epsilon=False,
        sink_only_unsplit=True,
        sink_invalid_zero=True,
        wait_before_final=False,
        l0_start=0,
        l1_start=128,
    )
    finalize_right = make_scheduled_decode_finalize_right(
        h_per_block=heads_per_block,
        out_width=128,
        has_attn_sink=has_attn_sink,
        wait_after_scale=False,
        r0_start=256,
        r1_start=384,
    )
    stage_value_shared = make_scheduled_decode_stage_value_shared(
        block_i=block_i,
        continuity=128,
    )
    load_indices = make_scheduled_decode_indices_loader(block_i=block_i)

    @T.macro
    def load_model1_paged_kv_block(
        kv_rope,
        quant_scales,
        rope_robust_desc_arg,
        scale_robust_desc_arg,
        page_size,
        kperm_indices_local,
        kv_indices,
        kv_shared_l,
        kv_shared_r,
        quant_shared,
        ldg_ty,
        ldg_tx,
        ldg_scale_ty,
        ldg_scale_tx,
        phase,
        bar_kv0_free,
        bar_kv_mask_ready,
        bar_indices_ready,
        bar_kv0_ready,
        bar_kv1_free,
        bar_kv1_ready,
    ):
        T.barrier_wait(bar_kv0_free, (phase & 1) ^ 1)
        for r in T.unroll(4):
            for u in T.unroll(4):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + 32 * u
                            + ldg_tx * 4
                            + v,
                        ],
                        kv_shared_l[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )
        T.lma_wait()
        T.barrier_arrive(bar_kv_mask_ready)
        T.barrier_arrive(bar_indices_ready)
        T.barrier_wait(bar_indices_ready, phase & 1)

        for c in T.vectorized(4):
            T.copy(
                quant_scales[
                    kv_indices[ldg_scale_ty] // page_size,
                    page_size * scale_bytes_offset
                    + (kv_indices[ldg_scale_ty] % page_size) * 8
                    + ldg_scale_tx * 4
                    + c,
                ],
                quant_shared[ldg_scale_ty, ldg_scale_tx * 4 + c],
                force_async_copy=True,
                src_robust_desc=scale_robust_desc_arg,
            )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.barrier_arrive(bar_kv0_ready)

        T.barrier_wait(bar_kv1_free, (phase & 1) ^ 1)
        for r in T.unroll(4):
            for u in T.unroll(3):
                for v in T.vectorized(4):
                    T.copy(
                        kv_rope[
                            kperm_indices_local[r] // page_size,
                            (kperm_indices_local[r] % page_size)
                            * (scale_bytes_offset // 2)
                            + dim // 4
                            + 32 * u
                            + ldg_tx * 4
                            + v,
                        ],
                        kv_shared_r[r * 16 + ldg_ty, 64 * u + ldg_tx * 8 + v],
                        force_async_copy=True,
                        src_robust_desc=rope_robust_desc_arg,
                    )

        for r in T.unroll(4):
            for v in T.vectorized(8):
                T.copy(
                    kv_rope[
                        kperm_indices_local[r] // page_size,
                        (kperm_indices_local[r] % page_size) * (scale_bytes_offset // 2)
                        + rope_bytes_offset // 2
                        + ldg_tx * 8
                        + v,
                    ],
                    kv_shared_r[r * 16 + ldg_ty, 64 * 3 + ldg_tx * 8 + v],
                    force_async_copy=True,
                    src_robust_desc=rope_robust_desc_arg,
                )
        T.ptx_commit_group()
        T.ptx_wait_group(0)
        T.barrier_arrive(bar_kv1_ready)

    @T.prim_func
    def dsa_decode(
        q: T.Tensor(q_shape, dtype),  # type: ignore
        kv_nope: T.Tensor([num_blocks, page_block_bytes], kv_latent_dtype),  # type: ignore
        kv_rope: T.Tensor([num_blocks, page_block_bytes // 2], dtype),  # type: ignore
        quant_scales: T.Tensor([num_blocks, page_block_bytes], "uint8"),  # type: ignore
        indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        topk_length: T.Tensor([batch], indices_dtype),  # type: ignore
        extra_kv_nope: T.Tensor(
            [num_blocks_extra, page_block_bytes_extra], kv_latent_dtype
        ),  # type: ignore
        extra_kv_rope: T.Tensor([num_blocks_extra, page_block_bytes_extra // 2], dtype),  # type: ignore
        extra_quant_scales: T.Tensor(
            [num_blocks_extra, page_block_bytes_extra], "uint8"
        ),  # type: ignore
        extra_indices: T.Tensor(extra_indices_shape, indices_dtype),  # type: ignore
        extra_topk_length: T.Tensor([batch], indices_dtype),  # type: ignore
        attn_sink: T.Tensor([num_heads], accum_dtype),  # type: ignore
        tile_scheduler_metadata: T.Tensor([num_mp_parts, 8], T.int32),  # type: ignore
        num_splits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor([batch + num_mp_parts, seq_len, num_heads], accum_dtype),  # type: ignore
        output_partial: T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        ),  # type: ignore
        output: T.Tensor(o_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        # MODEL1 scheduled split main kernel. It follows the FlashMLA scheduler
        # contract: each program consumes one metadata part and either writes the
        # final output for unsplit batches or partial output/LSE for split ones.
        with T.Kernel(
            seq_len * head_repeats, kv_group, num_mp_parts, threads=threads
        ) as (bx, by, bz):
            q_shared_l = T.alloc_shared([heads_per_block, 256], dtype)
            q_shared_r = T.alloc_shared([heads_per_block, 256], dtype)
            kv_shared_l = T.alloc_shared([block_i, 256], dtype)
            kv_shared_r = T.alloc_shared([block_i, 256], dtype)
            v_shared_0 = T.alloc_shared([block_i, 128], dtype)
            v_shared_1 = T.alloc_shared([block_i, 128], dtype)

            scores_shared = T.alloc_shared([heads_per_block, block_i], dtype)
            sum_exp_inv_shared = T.alloc_shared([heads_per_block], accum_dtype)
            sink_scale_shared = T.alloc_shared([heads_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([heads_per_block], accum_dtype)
            is_kv_valid = T.alloc_shared([block_i], "bool", scope="shared")
            kv_indices = T.alloc_shared([block_i], "int32", scope="shared")
            quant_shared = T.alloc_shared([block_i, 8], "uint8")

            bar_kv_mask_ready = T.alloc_barrier(arrive_count=128)
            bar_kv_mask_free = T.alloc_barrier(arrive_count=256)
            bar_q = T.alloc_barrier(arrive_count=512)
            bar_indices_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_ready = T.alloc_barrier(arrive_count=128)
            bar_kv1_ready = T.alloc_barrier(arrive_count=128)
            bar_kv0_quant_ready = T.alloc_barrier(arrive_count=256)
            bar_kv1_quant_ready = T.alloc_barrier(arrive_count=256)

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
                T.address_of(q[0, 0, 0, 0]),
                q_cosize * dtype_bytes,
            )
            nope_robust_desc = T.make_robust_desc(
                T.address_of(kv_nope[0, 0]),
                kv_nope_cosize,
            )
            rope_robust_desc = T.make_robust_desc(
                T.address_of(kv_rope[0, 0]),
                kv_rope_cosize * dtype_bytes,
            )
            scale_robust_desc = T.make_robust_desc(
                T.address_of(quant_scales[0, 0]),
                quant_scales_cosize,
            )
            extra_nope_robust_desc = T.make_robust_desc(
                T.address_of(extra_kv_nope[0, 0]),
                extra_kv_nope_cosize,
            )
            extra_rope_robust_desc = T.make_robust_desc(
                T.address_of(extra_kv_rope[0, 0]),
                extra_kv_rope_cosize * dtype_bytes,
            )
            extra_scale_robust_desc = T.make_robust_desc(
                T.address_of(extra_quant_scales[0, 0]),
                extra_quant_scales_cosize,
            )

            T.sync_threads()

            g_i = by
            s_i = bx if head_repeats == 1 else (bx // head_repeats)
            h0 = g_i * padded_head_kv + (
                0 if head_repeats == 1 else (bx % head_repeats) * 64
            )
            h1 = h0 + heads_per_block
            tid = T.get_thread_binding()

            begin_idx = tile_scheduler_metadata[bz, 0]
            sched_begin_block_idx = tile_scheduler_metadata[bz, 1]
            end_idx = tile_scheduler_metadata[bz, 2]
            sched_end_block_idx = tile_scheduler_metadata[bz, 3]
            begin_n_split_idx = tile_scheduler_metadata[bz, 4]
            phase_count = T.alloc_local([1], T.int32)
            T.fill(phase_count, 0)

            for b_i in range(begin_idx, end_idx + 1, 1):
                tir.call_extern("void", "__musa_loop_transparent_outermost")
                start_block_idx = T.alloc_var(T.int32)
                end_block_idx = T.alloc_var(T.int32)
                n_split_idx = T.alloc_var(T.int32)
                dynamic_main_blocks = T.alloc_var(T.int32)
                dynamic_total_blocks = T.alloc_var(T.int32)
                dynamic_main_blocks = T.max(T.ceildiv(topk_length[b_i], block_i), 1)
                dynamic_total_blocks = dynamic_main_blocks
                if has_extra:
                    dynamic_total_blocks += T.ceildiv(extra_topk_length[b_i], block_i)
                start_block_idx = T.if_then_else(
                    b_i == begin_idx, sched_begin_block_idx, 0
                )
                end_block_idx = T.if_then_else(
                    b_i == end_idx, sched_end_block_idx, dynamic_total_blocks
                )
                n_split_idx = T.if_then_else(b_i == begin_idx, begin_n_split_idx, 0)
                is_unsplit = (num_splits[b_i + 1] - num_splits[b_i]) == 1

                if tid < 512:
                    T.copy(
                        q[b_i, s_i, h0:h1, 0:256],
                        q_shared_l,
                        barrier=bar_q,
                    )
                    T.copy(
                        q[b_i, s_i, h0:h1, 256:512],
                        q_shared_r,
                        barrier=bar_q,
                    )
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.barrier_arrive(bar_q)
                    T.barrier_wait(bar_q, (b_i - begin_idx) & 1)

                if tid < 256:
                    sumexp = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    sumexp_inv = T.alloc_fragment([heads_per_block], accum_dtype)
                    alpha_local = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i = T.alloc_fragment([heads_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([heads_per_block], accum_dtype)
                    acc_s = T.alloc_fragment([heads_per_block, block_i], accum_dtype)
                    acc_s_cast = T.alloc_fragment([heads_per_block, block_i], dtype)
                    acc_o_l_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_l_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_l = T.alloc_local([64], dtype)
                    kv_reg_l_fp16 = T.view(kv_reg_l, [64], T.float16)
                    kv_reg_l_bf16_load = T.alloc_local([32], T.bfloat16)
                    kv_reg_l_fp8 = T.view(kv_reg_l_bf16_load, [64], kv_latent_dtype)
                    quant_u8_l = T.alloc_local([2, 4], "uint8")
                    quant_local_l = T.alloc_local([2, 4], T.float32)
                    c0_ldg_tx = tid % 8
                    c0_ldg_ty = tid // 8
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l_0, 0)
                    T.fill(acc_o_l_1, 0)

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv0_ready, (phase_count[0] & 1))

                        T.copy(quant_shared[c0_ldg_ty, 0:4], quant_u8_l[0, :])
                        T.copy(quant_shared[c0_ldg_ty + 32, 0:4], quant_u8_l[1, :])

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

                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_l_bf16_load[r * 16 + u * 4 + v] = (
                                        kv_shared_l[
                                            c0_ldg_ty + r * 32,
                                            64 * u + c0_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(2):
                            for u in T.unroll(4):
                                quant_local_l[r, u] = T.reinterpret(
                                    "float32",
                                    T.Cast("int32", quant_u8_l[r, u]) << 23,
                                )
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
                                        dtype,
                                        kv_reg_l_fp16[idx] * quant_local_l[r, u],
                                    )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_shared_l[
                                        c0_ldg_ty + r * 32, 64 * u + c0_ldg_tx * 8 + v
                                    ] = kv_reg_l[r * 32 + u * 8 + v]
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_l_bf16_load[r * 16 + (u + 2) * 4 + v] = (
                                        kv_shared_l[
                                            c0_ldg_ty + r * 32,
                                            64 * (u + 2) + c0_ldg_tx * 8 + v,
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
                                        dtype,
                                        kv_reg_l_fp16[idx] * quant_local_l[r, u + 2],
                                    )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    idx = r * 32 + (u + 2) * 8 + v
                                    kv_shared_l[
                                        c0_ldg_ty + r * 32,
                                        64 * (u + 2) + c0_ldg_tx * 8 + v,
                                    ] = kv_reg_l[idx]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv0_quant_ready)

                        T.barrier_wait(bar_kv_mask_ready, (phase_count[0] & 1))
                        for h_i, bi_i in T.Parallel(heads_per_block, block_i):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                is_kv_valid[bi_i % 8 * 8 + bi_i // 8],
                                0,
                                -(2**30),
                            )
                        T.lma_wait()
                        T.barrier_arrive(bar_kv_mask_free)

                        T.barrier_wait(bar_kv0_quant_ready, (phase_count[0] & 1))
                        T.gemm(
                            q_shared_l,
                            kv_shared_l[:, :],
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
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_kv1_free)

                        update_online_softmax(
                            acc_s,
                            acc_s_cast,
                            scores_shared,
                            m_i,
                            m_i_prev,
                            sumexp,
                            sumexp_i,
                            alpha_local,
                            alpha_shared,
                            acc_o_l_0,
                            acc_o_l_1,
                        )

                        T.lma_wait()
                        T.barrier_arrive(bar_p_ready)
                        stage_value_shared(
                            v_shared_0, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 0
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vl0_ready)
                        T.barrier_wait(bar_vl0_ready, (phase_count[0] & 1))

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_l_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        stage_value_shared(
                            v_shared_1, kv_reg_l, c0_ldg_ty, c0_ldg_tx, 2
                        )
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vl0_free)

                        T.lma_wait()
                        T.barrier_arrive(bar_vl1_ready)
                        T.barrier_wait(bar_vl1_ready, (phase_count[0] & 1))

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
                        phase_count[0] = phase_count[0] ^ 1

                    finalize_left(
                        b_i,
                        s_i,
                        h0,
                        h1,
                        is_unsplit,
                        m_i,
                        sumexp,
                        sumexp_inv,
                        sum_exp_inv_shared,
                        sink_scale_shared,
                        acc_o_l_0,
                        acc_o_l_1,
                        output,
                        output_partial,
                        lse,
                        glse,
                        n_split_idx,
                        num_splits,
                        attn_sink,
                        bar_final,
                    )
                elif tid < 512:
                    acc_o_r_0 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    acc_o_r_1 = T.alloc_fragment([heads_per_block, 128], accum_dtype)
                    kv_reg_r = T.alloc_local([64], dtype)
                    kv_reg_r_fp16 = T.view(kv_reg_r, [64], T.float16)
                    kv_reg_r_bf16_load = T.alloc_local([32], T.bfloat16)
                    kv_reg_r_fp8 = T.view(kv_reg_r_bf16_load, [64], kv_latent_dtype)
                    quant_u8_r = T.alloc_local([2, 3], "uint8")
                    quant_local_r = T.alloc_local([2, 3], T.float32)
                    T.fill(acc_o_r_0, 0)
                    T.fill(acc_o_r_1, 0)
                    c1_ldg_tx = (tid - 256) % 8
                    c1_ldg_ty = (tid - 256) // 8

                    for i_i in range(start_block_idx, end_block_idx):
                        T.barrier_wait(bar_kv1_ready, (phase_count[0] & 1))

                        T.copy(quant_shared[c1_ldg_ty, 4:7], quant_u8_r[0, :])
                        T.copy(quant_shared[c1_ldg_ty + 32, 4:7], quant_u8_r[1, :])

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
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(4):
                                    kv_reg_r_bf16_load[r * 16 + u * 4 + v] = (
                                        kv_shared_r[
                                            c1_ldg_ty + r * 32,
                                            64 * u + c1_ldg_tx * 8 + v,
                                        ]
                                    )
                        T.lma_wait()
                        for r in T.unroll(2):
                            for u in T.unroll(3):
                                quant_local_r[r, u] = T.reinterpret(
                                    "float32",
                                    T.Cast("int32", quant_u8_r[r, u]) << 23,
                                )
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
                                        dtype,
                                        kv_reg_r_fp16[idx] * quant_local_r[r, u],
                                    )
                        for r in T.unroll(2):
                            for u in T.unroll(2):
                                for v in T.vectorized(8):
                                    kv_shared_r[
                                        c1_ldg_ty + r * 32, 64 * u + c1_ldg_tx * 8 + v
                                    ] = kv_reg_r[r * 32 + u * 8 + v]

                        for r in T.unroll(2):
                            for v in T.vectorized(4):
                                kv_reg_r_bf16_load[r * 16 + 8 + v] = kv_shared_r[
                                    c1_ldg_ty + r * 32,
                                    128 + c1_ldg_tx * 8 + v,
                                ]
                        T.lma_wait()
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + 16 + v
                                kv_reg_r_fp16[idx] = kv_reg_r_fp8[idx]
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + 16 + v
                                kv_reg_r[idx] = T.Cast(
                                    dtype,
                                    kv_reg_r_fp16[idx] * quant_local_r[r, 2],
                                )
                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + 16 + v
                                kv_shared_r[
                                    c1_ldg_ty + r * 32, 128 + c1_ldg_tx * 8 + v
                                ] = kv_reg_r[idx]

                        for r in T.unroll(2):
                            for v in T.vectorized(8):
                                idx = r * 32 + 24 + v
                                kv_reg_r[idx] = kv_shared_r[
                                    c1_ldg_ty + r * 32, 192 + c1_ldg_tx * 8 + v
                                ]
                        T.lma_wait()
                        T.barrier_arrive(bar_kv1_quant_ready)
                        T.barrier_wait(bar_vl0_free, (phase_count[0] & 1))
                        stage_value_shared(
                            v_shared_0, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 0
                        )
                        T.lma_wait()
                        T.barrier_arrive(bar_vr0_ready)
                        T.barrier_wait(bar_vr0_ready, (phase_count[0] & 1))

                        T.barrier_wait(bar_p_ready, (phase_count[0] & 1))
                        for h_i, d_i in T.Parallel(heads_per_block, 128):
                            acc_o_r_0[h_i, d_i] *= alpha_shared[h_i]
                            acc_o_r_1[h_i, d_i] *= alpha_shared[h_i]

                        T.gemm(
                            scores_shared,
                            v_shared_0,
                            acc_o_r_0,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(bar_vl1_free, (phase_count[0] & 1))
                        stage_value_shared(
                            v_shared_1, kv_reg_r, c1_ldg_ty, c1_ldg_tx, 2
                        )
                        T.lma_wait()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(bar_vr1_ready)
                        T.barrier_wait(bar_vr1_ready, (phase_count[0] & 1))

                        T.gemm(
                            scores_shared,
                            v_shared_1,
                            acc_o_r_1,
                            policy=T.GemmWarpPolicy.FullRow,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        phase_count[0] = phase_count[0] ^ 1

                    finalize_right(
                        b_i,
                        s_i,
                        h0,
                        h1,
                        is_unsplit,
                        acc_o_r_0,
                        acc_o_r_1,
                        sum_exp_inv_shared,
                        sink_scale_shared,
                        output,
                        output_partial,
                        n_split_idx,
                        num_splits,
                        bar_final,
                        (b_i - begin_idx) & 1,
                    )
                else:
                    kperm_mask_local = T.alloc_local([4], "bool")
                    kperm_indices_local = T.alloc_local([4], indices_dtype)
                    topk_len_local = T.alloc_local([1], indices_dtype)
                    extra_topk_len_local = T.alloc_local([1], indices_dtype)
                    ldg_tx = (tid - 512) % 8
                    ldg_ty = (tid - 512) // 8
                    ldg_scale_tx = (tid - 512) % 2
                    ldg_scale_ty = (tid - 512) // 2
                    topk_len_local[0] = topk_length[b_i]
                    extra_topk_len_local[0] = extra_topk_length[b_i]

                    for i_i in range(start_block_idx, end_block_idx):
                        main_block_count = T.max(
                            T.ceildiv(topk_len_local[0], block_i), 1
                        )
                        use_extra = i_i >= main_block_count
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
                        if use_extra:
                            block_index = i_i - main_block_count
                            load_indices(
                                extra_indices,
                                b_i,
                                s_i,
                                g_i,
                                block_index,
                                extra_topk_len_local[0],
                                seq_len_kv_extra,
                                ldg_ty,
                                ldg_tx,
                                phase_count[0],
                                kperm_indices_local,
                                kperm_mask_local,
                                is_kv_valid,
                                kv_indices,
                                bar_kv_mask_free,
                            )

                            load_model1_paged_kv_block(
                                extra_kv_rope,
                                extra_quant_scales,
                                extra_rope_robust_desc,
                                extra_scale_robust_desc,
                                page_block_size_extra,
                                kperm_indices_local,
                                kv_indices,
                                kv_shared_l,
                                kv_shared_r,
                                quant_shared,
                                ldg_ty,
                                ldg_tx,
                                ldg_scale_ty,
                                ldg_scale_tx,
                                phase_count[0],
                                bar_kv0_free,
                                bar_kv_mask_ready,
                                bar_indices_ready,
                                bar_kv0_ready,
                                bar_kv1_free,
                                bar_kv1_ready,
                            )
                            phase_count[0] = phase_count[0] ^ 1

                        else:
                            block_index = i_i
                            load_indices(
                                indices,
                                b_i,
                                s_i,
                                g_i,
                                block_index,
                                topk_len_local[0],
                                seq_len_kv,
                                ldg_ty,
                                ldg_tx,
                                phase_count[0],
                                kperm_indices_local,
                                kperm_mask_local,
                                is_kv_valid,
                                kv_indices,
                                bar_kv_mask_free,
                            )
                            load_model1_paged_kv_block(
                                kv_rope,
                                quant_scales,
                                rope_robust_desc,
                                scale_robust_desc,
                                page_block_size,
                                kperm_indices_local,
                                kv_indices,
                                kv_shared_l,
                                kv_shared_r,
                                quant_shared,
                                ldg_ty,
                                ldg_tx,
                                ldg_scale_ty,
                                ldg_scale_tx,
                                phase_count[0],
                                bar_kv0_free,
                                bar_kv_mask_ready,
                                bar_indices_ready,
                                bar_kv0_ready,
                                bar_kv1_free,
                                bar_kv1_ready,
                            )
                            phase_count[0] = phase_count[0] ^ 1

        # MODEL1 scheduled combine kernel. Only batches with more than one split
        # enter this stage; unsplit batches are already written by the split kernel.
        dsa_combine(num_splits, glse, output_partial, attn_sink, output, lse)

    return dsa_decode


def sparse_mla_decode_fwd_scheduled_interface_model1(
    q,
    kv_nope,
    kv_rope,
    kv_scales,
    indices,
    tile_scheduler_metadata,
    num_splits,
    *,
    extra_kv_nope=None,
    extra_kv_rope=None,
    extra_kv_scales=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
    sm_scale=None,
    attn_sink=None,
    return_p_sum: bool = False,
    d_v=512,
    threads=640,
    verbose=False,
    page_block_size=64,
    extra_page_block_size=64,
):
    assert return_p_sum is False, "This kernel file is for decode only"
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert kv_nope.dtype == torch.float8_e4m3fn, "kv_nope must be float8_e4m3fn"
    assert kv_rope.dtype == torch.bfloat16, "kv_rope must be bfloat16"
    assert kv_scales.dtype == torch.uint8, "kv_scales must be uint8"
    assert indices.dtype == torch.int32, "indices must be int32"
    assert q.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_q = q.shape
    assert int(num_splits.numel()) == batch + 1
    num_blocks, page_block_bytes = kv_nope.shape
    kv_group = 1
    assert dim_q == d_v == 512
    assert kv_rope.shape == (num_blocks, page_block_bytes // 2)
    assert kv_scales.shape == (num_blocks, page_block_bytes)
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    if extra_indices is None:
        extra_indices = indices[:, :, :, :0]
    extra_topk = extra_indices.shape[-1]
    assert extra_indices.dtype == torch.int32, "extra_indices must be int32"
    assert extra_indices.shape == (batch, seq_len, kv_group, extra_topk)
    if extra_kv_nope is None:
        extra_kv_nope, extra_kv_rope, extra_kv_scales = kv_nope, kv_rope, kv_scales
        extra_page_block_size = page_block_size
    assert extra_kv_nope.dtype == torch.float8_e4m3fn, (
        "extra_kv_nope must be float8_e4m3fn"
    )
    assert extra_kv_rope.dtype == torch.bfloat16, "extra_kv_rope must be bfloat16"
    assert extra_kv_scales.dtype == torch.uint8, "extra_kv_scales must be uint8"
    extra_num_blocks, extra_page_block_bytes = extra_kv_nope.shape
    assert extra_kv_rope.shape == (extra_num_blocks, extra_page_block_bytes // 2)
    assert extra_kv_scales.shape == (extra_num_blocks, extra_page_block_bytes)

    extra_topk_length_b = require_batch_lengths(
        extra_topk_length, batch, extra_topk, q.device, "extra_topk_length"
    )
    runtime = prepare_scheduled_decode_runtime(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=d_v,
        topk=topk,
        topk_length=topk_length,
        attn_sink=attn_sink,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        out_dtype=q.dtype,
        device=q.device,
        variant_name="MODEL1",
    )

    kernel = sparse_attention_decode_fwd_scheduled_kernel_model1(
        heads,
        d_v,
        has_extra=extra_topk > 0,
        kv_group=kv_group,
        sm_scale=sm_scale,
        threads=threads,
        max_nums_splits=runtime.max_nums_splits,
        has_attn_sink=runtime.has_attn_sink,
        page_block_size=page_block_size,
        extra_page_block_size=extra_page_block_size,
        page_stride_bytes=page_block_bytes,
        extra_page_stride_bytes=extra_page_block_bytes,
    )
    if verbose:
        kernel.show_source()
    raise_complete_if_dry_run()
    kernel(
        q,
        kv_nope,
        kv_rope,
        kv_scales,
        indices,
        runtime.topk_length,
        extra_kv_nope,
        extra_kv_rope,
        extra_kv_scales,
        extra_indices,
        extra_topk_length_b,
        runtime.attn_sink,
        tile_scheduler_metadata,
        num_splits,
        runtime.glse,
        runtime.out_partial,
        runtime.out,
        runtime.lse,
    )

    return runtime.out, runtime.lse
