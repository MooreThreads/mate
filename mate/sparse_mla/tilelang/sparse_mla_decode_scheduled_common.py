# ruff: noqa
"""Shared scheduled sparse decode plumbing.

The MODEL1 and V3.2 scheduled decode kernels intentionally stay in separate
TileLang factories for now: their kv layouts and producer load/dequant paths are
different enough that forcing one giant kernel body is hard to maintain and can
make TileLang compilation fragile.  This module is the first convergence layer:
shared compile options, host-side ABI helpers, split allocation policy, and
small wrappers for feature flags.  Future refactors should move only genuinely
common TileLang macros here and keep layout-specific branches behind
compile-time constants.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import tilelang
from tilelang import language as T


SCHEDULED_DECODE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}


SCHEDULED_DECODE_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def make_scheduled_decode_combine(
    *,
    batch,
    seq_len,
    num_heads,
    dim,
    num_mp_parts,
    dtype,
    accum_dtype,
    max_nums_splits,
    has_attn_sink,
    max_lse_init,
):
    """Build the shared scheduled sparse decode combine macro.

    The split producer stays layout-specific, but V3.2 and MODEL1 combine
    the scheduled partials with the same LSE reduction and output rescale.
    """

    block_m = 8
    num_threads = block_m * 32
    elems_per_thread = dim // 32
    num_lse_per_thread = T.ceildiv(max_nums_splits, 32)

    @T.macro
    def dsa_combine(
        num_splits: T.Tensor([batch + 1], T.int32),  # type: ignore
        glse: T.Tensor([batch + num_mp_parts, seq_len, num_heads], accum_dtype),  # type: ignore
        output_partial: T.Tensor(
            [batch + num_mp_parts, seq_len, num_heads, dim], accum_dtype
        ),  # type: ignore
        attn_sink: T.Tensor([num_heads], accum_dtype),  # type: ignore
        output: T.Tensor([batch, seq_len, num_heads, dim], dtype),  # type: ignore
        lse: T.Tensor([batch, num_heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(
            batch, T.ceildiv(seq_len * num_heads, block_m), threads=num_threads
        ) as (cb, cm):
            batch_idx = cb
            m_block_idx = cm
            lse_scale_shared = T.alloc_shared(
                [block_m, max_nums_splits + 1], accum_dtype
            )
            tid = T.get_thread_binding()
            warp_idx = tid // 32
            lane_idx = tid % 32
            split_start = num_splits[batch_idx]
            split_end = num_splits[batch_idx + 1]
            my_num_splits = split_end - split_start
            if my_num_splits > 1:
                num_q_seqs = seq_len * num_heads
                num_cur_valid_q_seqs = T.alloc_var(T.int32)
                num_cur_valid_q_seqs = T.min(
                    num_q_seqs - m_block_idx * block_m, block_m
                )
                for loop in range(tid, my_num_splits * block_m, num_threads):
                    split_idx = loop // block_m
                    seq_idx = loop % block_m
                    if seq_idx < num_cur_valid_q_seqs:
                        flat_idx = seq_idx + m_block_idx * block_m
                        lse_scale_shared[seq_idx, split_idx] = glse[
                            split_start + split_idx,
                            flat_idx // num_heads,
                            flat_idx % num_heads,
                        ]
                    else:
                        lse_scale_shared[seq_idx, split_idx] = -T.infinity(accum_dtype)
                T.sync_threads()

                if warp_idx < num_cur_valid_q_seqs:
                    lse_local = T.alloc_local([num_lse_per_thread], accum_dtype)
                    for i in T.unroll(num_lse_per_thread):
                        if i * 32 + lane_idx < my_num_splits:
                            loaded_lse = T.alloc_var(accum_dtype)
                            loaded_lse = lse_scale_shared[warp_idx, i * 32 + lane_idx]
                            lse_local[i] = T.if_then_else(
                                loaded_lse == T.infinity(accum_dtype),
                                -T.infinity(accum_dtype),
                                loaded_lse,
                            )
                        else:
                            lse_local[i] = -T.infinity(accum_dtype)

                    max_lse = T.alloc_local([1], accum_dtype)
                    max_lse[0] = max_lse_init
                    for i in T.unroll(num_lse_per_thread):
                        max_lse[0] = T.max(max_lse[0], lse_local[i])
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 16))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 8))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 4))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 2))
                    max_lse[0] = T.max(max_lse[0], T.shfl_xor(max_lse[0], 1))

                    sum_exp = T.alloc_local([1], accum_dtype)
                    sum_exp[0] = 0.0
                    temp = T.alloc_local([num_lse_per_thread], accum_dtype)
                    for i in T.unroll(num_lse_per_thread):
                        temp[i] = T.exp2(lse_local[i] - max_lse[0])
                    for i in T.unroll(num_lse_per_thread):
                        sum_exp[0] += temp[i]
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 16)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 8)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 4)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 2)
                    sum_exp[0] += T.shfl_xor(sum_exp[0], 1)

                    global_lse = T.alloc_local([1], accum_dtype)
                    if sum_exp[0] == 0.0 or sum_exp[0] != sum_exp[0]:
                        global_lse[0] = T.infinity(accum_dtype)
                    else:
                        global_lse[0] = T.log2(sum_exp[0]) + max_lse[0]

                    if lane_idx == 0:
                        flat_idx = warp_idx + m_block_idx * block_m
                        lse[
                            batch_idx,
                            flat_idx % num_heads,
                            flat_idx // num_heads,
                        ] = global_lse[0] * 0.6931471805599453

                    for i in T.unroll(num_lse_per_thread):
                        if i * 32 + lane_idx < my_num_splits:
                            lse_scale_shared[warp_idx, i * 32 + lane_idx] = (
                                T.if_then_else(
                                    global_lse[0] == T.infinity(accum_dtype),
                                    0.0,
                                    T.exp2(lse_local[i] - global_lse[0]),
                                )
                            )
                    T.sync_threads()

                    result = T.alloc_local([elems_per_thread], accum_dtype)
                    T.clear(result)
                    for split in T.serial(my_num_splits):
                        scale = lse_scale_shared[warp_idx, split]
                        if scale != 0.0:
                            for i in T.unroll(elems_per_thread):
                                flat_idx = warp_idx + m_block_idx * block_m
                                result[i] += (
                                    scale
                                    * output_partial[
                                        split_start + split,
                                        flat_idx // num_heads,
                                        flat_idx % num_heads,
                                        lane_idx + i * 32,
                                    ]
                                )

                    for i in T.unroll(elems_per_thread):
                        flat_idx = warp_idx + m_block_idx * block_m
                        sink_scale = T.alloc_var(accum_dtype)
                        sink_scale = 1.0
                        if has_attn_sink:
                            sink_scale = T.if_then_else(
                                global_lse[0] == T.infinity(accum_dtype),
                                1.0,
                                1.0
                                / (
                                    1.0
                                    + T.exp2(
                                        attn_sink[flat_idx % num_heads]
                                        * 1.4426950408889634
                                        - global_lse[0]
                                    )
                                ),
                            )
                        output[
                            batch_idx,
                            flat_idx // num_heads,
                            flat_idx % num_heads,
                            lane_idx + i * 32,
                        ] = T.Cast(dtype, result[i] * sink_scale)

    return dsa_combine


def make_scheduled_decode_online_softmax(
    *,
    h_per_block,
    block_i,
    out_width,
    accum_dtype,
    sm_scale,
):
    """Build the shared QK -> online softmax -> score staging macro."""

    score_swizzle = block_i // 8

    @T.macro
    def update_online_softmax(
        acc_s,
        acc_s_cast,
        s_shared,
        m_i,
        m_i_prev,
        sumexp,
        sumexp_i,
        alpha_local,
        alpha_shared,
        acc_o_l_0,
        acc_o_l_1,
    ):
        T.copy(m_i, m_i_prev)
        T.reduce_max(acc_s, m_i, dim=1, clear=False)
        for h_i in T.Parallel(h_per_block):
            m_i[h_i] = T.max(m_i_prev[h_i], m_i[h_i])
        for h_i in T.Parallel(h_per_block):
            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
        for h_i, bi_i in T.Parallel(h_per_block, block_i):
            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)

        T.reduce_sum(acc_s, sumexp_i, dim=1)
        for h_i in T.Parallel(h_per_block):
            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
        for h_i, d_i in T.Parallel(h_per_block, out_width):
            acc_o_l_0[h_i, d_i] *= alpha_local[h_i]
            acc_o_l_1[h_i, d_i] *= alpha_local[h_i]

        T.copy(alpha_local, alpha_shared)
        T.copy(acc_s, acc_s_cast)
        for i, t in T.Parallel(h_per_block, 8):
            base = t * score_swizzle
            for l in T.vectorized(score_swizzle):
                s_shared[i, base + l] = acc_s_cast[i, l * 8 + t]

    return update_online_softmax


def make_scheduled_decode_stage_value_shared(
    *,
    block_i,
    continuity,
):
    """Build the shared register -> swizzled V tile staging macro."""

    score_swizzle = block_i // 8

    @T.macro
    def stage_value_shared(v_shared, kv_reg, ldg_ty, ldg_tx, u_start):
        T.annotate_layout(
            {
                v_shared[:, :]: tilelang.layout.make_sqmma_swizzled_layout(
                    v_shared[:, :], continuity=continuity, k_major=False
                )
            },
            allow_reannotation=True,
            allow_buffer_region=True,
        )
        for r in T.unroll(2):
            for u in T.unroll(2):
                for v in T.vectorized(8):
                    v_shared[
                        ((ldg_ty + r * 32) % 8) * score_swizzle
                        + (ldg_ty + r * 32) // 8,
                        64 * u + ldg_tx * 8 + v,
                    ] = kv_reg[r * 32 + (u + u_start) * 8 + v]

    return stage_value_shared


def make_scheduled_decode_indices_loader(*, block_i):
    """Build the shared producer index/mask loader for scheduled decode."""

    score_swizzle = block_i // 8

    @T.macro
    def load_indices(
        indices,
        b_i,
        s_i,
        g_i,
        block_index,
        topk_length,
        seq_len_kv,
        ldg_ty,
        ldg_tx,
        phase,
        kperm_indices_local,
        kperm_mask_local,
        is_kv_valid,
        kv_indices,
        bar_kv_mask_free,
    ):
        for r in T.unroll(4):
            token_pos = T.alloc_var(T.int32)
            token_pos = (
                block_index * block_i
                + ((r * 16 + ldg_ty) % 8) * score_swizzle
                + (r * 16 + ldg_ty) // 8
            )
            kperm_indices_local[r] = indices[b_i, s_i, g_i, token_pos]
            kperm_mask_local[r] = (
                kperm_indices_local[r] >= 0
                and kperm_indices_local[r] < seq_len_kv
                and token_pos < topk_length
            )
            kperm_indices_local[r] = T.if_then_else(
                kperm_mask_local[r],
                kperm_indices_local[r],
                seq_len_kv,
            )

        T.barrier_wait(bar_kv_mask_free, (phase & 1) ^ 1)
        if ldg_tx == 0:
            for r in T.unroll(4):
                row = ((r * 16 + ldg_ty) % 8) * score_swizzle + (r * 16 + ldg_ty) // 8
                is_kv_valid[row] = kperm_mask_local[r]
                kv_indices[r * 16 + ldg_ty] = kperm_indices_local[r]

    return load_indices


def make_scheduled_decode_finalize_left(
    *,
    h_per_block,
    out_width,
    accum_dtype,
    sm_scale,
    has_attn_sink,
    use_strict_valid,
    add_denominator_epsilon,
    sink_only_unsplit,
    sink_invalid_zero,
    wait_before_final,
    l0_start,
    l1_start,
):
    """Build the shared left-half split writeback macro."""

    @T.macro
    def finalize_left(
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
    ):
        for h_i in T.Parallel(h_per_block):
            if use_strict_valid:
                if m_i[h_i] != -(2**30):
                    sumexp_inv[h_i] = 1 / sumexp[h_i]
                else:
                    sumexp_inv[h_i] = 0
            else:
                if add_denominator_epsilon:
                    sumexp_inv[h_i] = T.if_then_else(
                        m_i[h_i] > -(2**29), 1 / (sumexp[h_i] + 1e-8), 0.0
                    )
                else:
                    sumexp_inv[h_i] = T.if_then_else(
                        m_i[h_i] > -(2**29), 1 / sumexp[h_i], 0.0
                    )
            sum_exp_inv_shared[h_i] = sumexp_inv[h_i]

        for h_i in T.Parallel(h_per_block):
            if use_strict_valid:
                if m_i[h_i] != -(2**30):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                else:
                    sumexp[h_i] = T.infinity(accum_dtype)
            else:
                sumexp[h_i] = T.if_then_else(
                    m_i[h_i] > -(2**29),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                    T.infinity(accum_dtype),
                )

        if has_attn_sink:
            if sink_only_unsplit:
                if is_unsplit:
                    for h_i in T.Parallel(h_per_block):
                        if sink_invalid_zero:
                            sink_scale_shared[h_i] = T.if_then_else(
                                m_i[h_i] != -(2**30),
                                1
                                / (
                                    1
                                    + T.exp2(
                                        attn_sink[h0 + h_i] * 1.4426950408889634
                                        - sumexp[h_i]
                                    )
                                ),
                                0.0,
                            )
                        else:
                            sink_scale_shared[h_i] = T.if_then_else(
                                sumexp[h_i] == T.infinity(accum_dtype),
                                1.0,
                                1.0
                                / (
                                    1.0
                                    + T.exp2(
                                        attn_sink[h0 + h_i] * 1.4426950408889634
                                        - sumexp[h_i]
                                    )
                                ),
                            )
            else:
                for h_i in T.Parallel(h_per_block):
                    sink_scale_shared[h_i] = T.if_then_else(
                        sumexp[h_i] == T.infinity(accum_dtype),
                        1.0,
                        1.0
                        / (
                            1.0
                            + T.exp2(
                                attn_sink[h0 + h_i] * 1.4426950408889634 - sumexp[h_i]
                            )
                        ),
                    )

        if wait_before_final:
            T.lma_wait()
        T.barrier_arrive(bar_final)

        for h_i, d_i in T.Parallel(h_per_block, out_width):
            acc_o_l_0[h_i, d_i] *= sumexp_inv[h_i]
            acc_o_l_1[h_i, d_i] *= sumexp_inv[h_i]

        if is_unsplit:
            if has_attn_sink:
                for h_i, d_i in T.Parallel(h_per_block, out_width):
                    acc_o_l_0[h_i, d_i] *= sink_scale_shared[h_i]
                    acc_o_l_1[h_i, d_i] *= sink_scale_shared[h_i]
            T.copy(acc_o_l_0, output[b_i, s_i, h0:h1, l0_start : l0_start + out_width])
            T.copy(acc_o_l_1, output[b_i, s_i, h0:h1, l1_start : l1_start + out_width])
            for h_i in T.Parallel(h_per_block):
                lse[b_i, h0 + h_i, s_i] = sumexp[h_i] * 0.6931471805599453
        else:
            T.copy(
                acc_o_l_0,
                output_partial[
                    n_split_idx + num_splits[b_i],
                    s_i,
                    h0:h1,
                    l0_start : l0_start + out_width,
                ],
            )
            T.copy(
                acc_o_l_1,
                output_partial[
                    n_split_idx + num_splits[b_i],
                    s_i,
                    h0:h1,
                    l1_start : l1_start + out_width,
                ],
            )
            T.copy(sumexp, glse[n_split_idx + num_splits[b_i], s_i, h0:h1])

    return finalize_left


def make_scheduled_decode_finalize_right(
    *,
    h_per_block,
    out_width,
    has_attn_sink,
    wait_after_scale,
    r0_start,
    r1_start,
):
    """Build the shared right-half split writeback macro."""

    @T.macro
    def finalize_right(
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
        final_phase,
    ):
        T.barrier_wait(bar_final, final_phase)
        for h_i, d_i in T.Parallel(h_per_block, out_width):
            acc_o_r_0[h_i, d_i] *= sum_exp_inv_shared[h_i]
            acc_o_r_1[h_i, d_i] *= sum_exp_inv_shared[h_i]
        if wait_after_scale:
            T.lma_wait()

        if is_unsplit:
            if has_attn_sink:
                for h_i, d_i in T.Parallel(h_per_block, out_width):
                    acc_o_r_0[h_i, d_i] *= sink_scale_shared[h_i]
                    acc_o_r_1[h_i, d_i] *= sink_scale_shared[h_i]
            T.copy(acc_o_r_0, output[b_i, s_i, h0:h1, r0_start : r0_start + out_width])
            T.copy(acc_o_r_1, output[b_i, s_i, h0:h1, r1_start : r1_start + out_width])
        else:
            T.copy(
                acc_o_r_0,
                output_partial[
                    n_split_idx + num_splits[b_i],
                    s_i,
                    h0:h1,
                    r0_start : r0_start + out_width,
                ],
            )
            T.copy(
                acc_o_r_1,
                output_partial[
                    n_split_idx + num_splits[b_i],
                    s_i,
                    h0:h1,
                    r1_start : r1_start + out_width,
                ],
            )

    return finalize_right


def scheduled_max_num_splits(num_mp_parts: int, name: str) -> int:
    assert num_mp_parts <= 64, (
        f"{name} scheduled combine currently supports at most 64 MP parts"
    )
    return 32 if num_mp_parts <= 32 else 64


def require_batch_lengths(
    lengths: Optional[torch.Tensor],
    batch: int,
    fill: int,
    device,
    name: str,
) -> torch.Tensor:
    if lengths is None:
        return torch.full((batch,), fill, dtype=torch.int32, device=device)
    assert lengths.dtype == torch.int32, f"{name} must be int32"
    assert lengths.shape == (batch,), f"{name} must have shape [batch]"
    return lengths.contiguous()


def optional_attn_sink(attn_sink: Optional[torch.Tensor], heads: int, device):
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty((heads,), dtype=torch.float32, device=device)
    assert attn_sink.dtype == torch.float32
    assert attn_sink.shape == (heads,)
    return attn_sink.contiguous(), has_attn_sink


def allocate_scheduled_decode_outputs(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    num_mp_parts: int,
    dtype: torch.dtype,
    device,
):
    glse = torch.empty(
        (batch + num_mp_parts, seq_len, heads),
        dtype=torch.float32,
        device=device,
    )
    out_partial = torch.empty(
        (batch + num_mp_parts, seq_len, heads, dim),
        dtype=torch.float32,
        device=device,
    )
    out = torch.empty((batch, seq_len, heads, dim), dtype=dtype, device=device)
    lse = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=device)
    return glse, out_partial, out, lse


@dataclass
class ScheduledDecodeRuntime:
    """Host-side objects shared by scheduled sparse decode variants."""

    topk_length: torch.Tensor
    attn_sink: torch.Tensor
    has_attn_sink: bool
    max_nums_splits: int
    glse: torch.Tensor
    out_partial: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor


def prepare_scheduled_decode_runtime(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    topk: int,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    out_dtype: torch.dtype,
    device,
    variant_name: str,
) -> ScheduledDecodeRuntime:
    num_mp_parts = int(tile_scheduler_metadata.shape[0])
    assert tile_scheduler_metadata.shape == (num_mp_parts, 8)
    assert num_splits.shape == (batch + 1,)

    topk_length_arg = require_batch_lengths(
        topk_length, batch, topk, device, "topk_length"
    )
    attn_sink_arg, has_attn_sink = optional_attn_sink(attn_sink, heads, device)
    max_nums_splits = scheduled_max_num_splits(num_mp_parts, variant_name)
    glse, out_partial, out, lse = allocate_scheduled_decode_outputs(
        batch, seq_len, heads, dim, num_mp_parts, out_dtype, device
    )
    return ScheduledDecodeRuntime(
        topk_length=topk_length_arg,
        attn_sink=attn_sink_arg,
        has_attn_sink=has_attn_sink,
        max_nums_splits=max_nums_splits,
        glse=glse,
        out_partial=out_partial,
        out=out,
        lse=lse,
    )
