#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::attention::msa {

// Device-side builder for the per-MP persistent-scheduler metadata.  The
// metadata ABI is intentionally small: every MP slot receives
// {q_work_begin, q_work_length}.  Partition ownership is derived by the main
// scheduler from the one-dimensional CTA slot, so this kernel does not need to
// know about the KV layout, element type, or MMA pipeline.
template <class TileShape_, int HeadRatio_, bool ParallelKTiles_, bool IsVarlen_, bool IsCausal_>
struct MsaMaxScoreMetadataKernel {
  using TileShape = TileShape_;

  static constexpr int  TileQ          = mute::get<0>(TileShape{});
  static constexpr int  TileKV         = mute::get<1>(TileShape{});
  static constexpr int  HeadRatio      = HeadRatio_;
  static constexpr bool ParallelKTiles = ParallelKTiles_;
  static constexpr bool IsVarlen       = IsVarlen_;
  static constexpr bool IsCausal       = IsCausal_;
  static constexpr int  NumThreads     = 128;

  static_assert(HeadRatio > 0);
  static_assert(TileQ > 0 && TileKV > 0);
  static_assert(TileQ % HeadRatio == 0);

  static MUTLASS_DEVICE int schedule_q_work_cost(int            q_work_idx,
                                                 int            batch_size,
                                                 int            num_kv_heads,
                                                 int            max_seqlen_q,
                                                 int            max_seqlen_k,
                                                 int32_t const* cu_seqlens_q,
                                                 int32_t const* cu_seqlens_k,
                                                 int32_t const* qo_offset) {
    constexpr int q_tokens_per_tile = TileQ / HeadRatio;
    const int     q_tiles           = (max_seqlen_q + q_tokens_per_tile - 1) / q_tokens_per_tile;
    const int     work_per_batch    = q_tiles * num_kv_heads;
    const int     batch_idx         = q_work_idx / work_per_batch;
    const int     rem               = q_work_idx - batch_idx * work_per_batch;
    const int     q_tile_idx        = rem % q_tiles;
    const int     q_len             = IsVarlen ? cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx] : max_seqlen_q;
    const int     kv_len            = IsVarlen ? cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx] : max_seqlen_k;
    const int     q_local_begin     = q_tile_idx * q_tokens_per_tile;
    const int     q_count0          = q_len - q_local_begin;
    const int     q_count           = q_count0 > 0 ? (q_count0 < q_tokens_per_tile ? q_count0 : q_tokens_per_tile) : 0;
    if (q_count == 0) {
      return 0;
    }
    int visible_k = kv_len;
    if constexpr (IsCausal) {
      const int last_q         = q_local_begin + q_count - 1 + qo_offset[batch_idx];
      const int causal_visible = last_q + 1 > 0 ? last_q + 1 : 0;
      visible_k                = kv_len < causal_visible ? kv_len : causal_visible;
    }
    return visible_k > 0 ? (visible_k + TileKV - 1) / TileKV : 0;
  }

  static MUTLASS_DEVICE void run(int32_t const* cu_seqlens_q,
                                 int32_t const* cu_seqlens_k,
                                 int32_t const* qo_offset,
                                 int32_t*       schedule_metadata,
                                 int            batch_size,
                                 int            num_kv_heads,
                                 int            max_seqlen_q,
                                 int            max_seqlen_k,
                                 int            num_mps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
      return;
    }

    constexpr int q_tokens_per_tile = TileQ / HeadRatio;
    const int     q_tiles           = (max_seqlen_q + q_tokens_per_tile - 1) / q_tokens_per_tile;
    const int     q_work_count      = batch_size * q_tiles * num_kv_heads;

    int64_t total_cost = 0;
    for (int q_work_idx = 0; q_work_idx < q_work_count; ++q_work_idx) {
      total_cost += schedule_q_work_cost(
          q_work_idx, batch_size, num_kv_heads, max_seqlen_q, max_seqlen_k, cu_seqlens_q, cu_seqlens_k, qo_offset);
    }

    const int     k_tiles       = (max_seqlen_k + TileKV - 1) / TileKV;
    constexpr int ctas_per_mp   = ParallelKTiles ? 3 : 1;
    const int     target_ctas   = num_mps * ctas_per_mp;
    const int     q_works       = q_work_count > 0 ? q_work_count : 1;
    const int     k_partitions0 = (target_ctas + q_works / 2) / q_works;
    const int     k_partitions =
        k_tiles < (k_partitions0 > 1 ? k_partitions0 : 1) ? k_tiles : (k_partitions0 > 1 ? k_partitions0 : 1);
    const int  partition_groups   = (k_partitions + ctas_per_mp - 1) / ctas_per_mp;
    const int  rows_per_group     = num_mps / partition_groups;
    const int  extra_groups       = num_mps - rows_per_group * partition_groups;
    const bool one_q_work_per_row = q_work_count <= rows_per_group;

    // Keep rows grouped by partition group. Scan each group's weighted Q
    // prefix once instead of restarting from Q work zero for every MP row.
    int mp_idx = 0;
    for (int partition_group = 0; partition_group < partition_groups; ++partition_group) {
      const int rows = rows_per_group + (partition_group < extra_groups ? 1 : 0);
      if (one_q_work_per_row) {
        for (int row = 0; row < rows; ++row, ++mp_idx) {
          const int q_begin                 = row < q_work_count ? row : q_work_count;
          schedule_metadata[2 * mp_idx]     = q_begin;
          schedule_metadata[2 * mp_idx + 1] = row < q_work_count ? 1 : 0;
        }
        continue;
      }

      int     q_begin      = 0;
      int64_t group_prefix = 0;
      for (int row = 0; row < rows; ++row, ++mp_idx) {
        int q_end = q_work_count;
        if (row + 1 < rows && total_cost > 0) {
          const int64_t target = (total_cost * (row + 1) + rows - 1) / rows;
          q_end                = q_begin;
          while (q_end < q_work_count && group_prefix < target) {
            group_prefix += schedule_q_work_cost(
                q_end, batch_size, num_kv_heads, max_seqlen_q, max_seqlen_k, cu_seqlens_q, cu_seqlens_k, qo_offset);
            ++q_end;
          }
        }
        schedule_metadata[2 * mp_idx]     = q_begin;
        schedule_metadata[2 * mp_idx + 1] = q_end > q_begin ? q_end - q_begin : 0;
        q_begin                           = q_end;
      }
    }
  }
};

template <class MetadataKernel>
__global__ __launch_bounds__(128, 1) void build_msa_maxscore_schedule(int32_t const* cu_seqlens_q,
                                                                      int32_t const* cu_seqlens_k,
                                                                      int32_t const* qo_offset,
                                                                      int32_t*       schedule_metadata,
                                                                      int            batch_size,
                                                                      int            num_kv_heads,
                                                                      int            max_seqlen_q,
                                                                      int            max_seqlen_k,
                                                                      int            num_mps) {
  MetadataKernel::run(cu_seqlens_q,
                      cu_seqlens_k,
                      qo_offset,
                      schedule_metadata,
                      batch_size,
                      num_kv_heads,
                      max_seqlen_q,
                      max_seqlen_k,
                      num_mps);
}

}  // namespace mate::attention::msa
