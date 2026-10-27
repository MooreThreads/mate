/*
 * MSA sparse top-k block selector.
 *
 * The indexerTopK histogram/insertion-sort structure follows the MSA
 * sparse_topk_select path, adapted for MATE's maxscore layout
 * (total_q, num_qo_heads, max_k_tiles), so no transpose workspace is needed.
 */
#pragma once

#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>

namespace mate::attention::msa::sparse_topk {

static constexpr uint32_t kMaxTopK                  = 32;
static constexpr int64_t  kSparseBlockSize          = 128;
static constexpr uint32_t kSmallTopkMaxK            = 256;
static constexpr uint32_t kSparseTopkMaxK           = 64;
static constexpr size_t   kSmemAlignmentBytes       = 128;
static constexpr int      kSmallTopkWarpsPerBlock   = 8;
static constexpr int      kSmallTopkThreadsPerBlock = kSmallTopkWarpsPerBlock * 32;
static constexpr int      kIndexerThreadsPerBlock   = 64;
// Match the MSA indexerTopK shape: 10-bit histogram bins and a bounded
// threshold-bin candidate buffer before the final insertion-sort step.
static constexpr int kIndexerNumBins       = 1024;
static constexpr int kIndexerNumFinalItems = 256;

inline bool is_supported_topk(uint32_t topk) {
  return topk == 4 || topk == 8 || topk == 16;
}

constexpr size_t round_up_smem_bytes(size_t bytes) {
  return ((bytes + kSmemAlignmentBytes - 1) / kSmemAlignmentBytes) * kSmemAlignmentBytes;
}

__device__ __forceinline__ uint32_t ordered_score_bits(float x) {
  uint32_t bits = __float_as_uint(x);
  // The strict sign-boundary comparison gives both signed zeros the same key.
  return bits > 0x80000000u ? bits : ~bits & 0x7fffffffu;
}

template <int Step>
__device__ __forceinline__ uint32_t extract_bin_idx(float x) {
  if constexpr (Step == 0) {
    __half   hx   = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits          = bits > 0x8000 ? bits : ~bits & 0x7fff;
    return bits >> 6;
  } else {
    uint32_t bits = ordered_score_bits(x);
    if constexpr (Step == 1) {
      return bits >> 22;
    } else if constexpr (Step == 2) {
      return (bits >> 12) & 0x3ff;
    } else {
      return (bits >> 2) & 0x3ff;
    }
  }
}

template <int Shift>
__device__ __forceinline__ bool is_partial_match(float x, uint32_t pattern) {
  if constexpr (Shift == 0) {
    return true;
  }
  uint32_t bits = ordered_score_bits(x);
  return (bits ^ pattern) >> Shift == 0;
}

__device__ __forceinline__ bool is_topk_candidate(uint32_t idx, uint32_t force_begin, uint32_t force_end_start) {
  // The end of the unforced interval never exceeds the row's valid-page count.
  return idx >= force_begin && idx < force_end_start;
}

__device__ __forceinline__ int32_t forced_slot_to_block(
    uint32_t slot, uint32_t force_begin, uint32_t force_end, uint32_t force_end_start, uint32_t num_valid_pages) {
  if (slot < force_begin) {
    return slot < num_valid_pages ? static_cast<int32_t>(slot) : -1;
  }
  const uint32_t end_slot = slot - force_begin;
  if (end_slot >= force_end) {
    return -1;
  }
  const uint32_t idx = force_end_start + end_slot;
  // When the sequence is short, begin/end ranges can overlap.  Emit the
  // duplicate only once; the output contract uses -1 padding for the rest.
  if (idx >= num_valid_pages || idx < force_begin) {
    return -1;
  }
  return static_cast<int32_t>(idx);
}

__device__ __forceinline__ uint32_t get_row_num_valid_pages(uint32_t       q_abs,
                                                            uint32_t       default_num_valid_pages,
                                                            int64_t const* query_positions) {
  if (query_positions == nullptr) {
    return default_num_valid_pages;
  }
  int64_t position = query_positions[q_abs];
  if (position < 0) {
    return 0;
  }
  uint64_t visible_pages = static_cast<uint64_t>(position) / static_cast<uint64_t>(kSparseBlockSize) + 1;
  return static_cast<uint32_t>(visible_pages < default_num_valid_pages ? visible_pages : default_num_valid_pages);
}

__device__ __forceinline__ bool topk_score_index_better(int lhs_idx, float lhs_score, int rhs_idx, float rhs_score) {
  if (lhs_idx < 0) {
    return false;
  }
  if (rhs_idx < 0) {
    return true;
  }
  if (lhs_score > rhs_score) {
    return true;
  }
  if (lhs_score < rhs_score) {
    return false;
  }
  return lhs_idx < rhs_idx;
}

__device__ __forceinline__ void warp_bitonic_sort_asc32(uint32_t& key, uint32_t lane) {
  constexpr uint32_t kMask = 0xffffffffu;
#pragma unroll
  for (int k = 2; k <= 32; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      uint32_t partner    = __shfl_xor_sync(kMask, key, j);
      bool     asc_pair   = (lane & k) == 0;
      bool     lower_lane = (lane & j) == 0;
      if (lower_lane) {
        key = asc_pair ? min(key, partner) : max(key, partner);
      } else {
        key = asc_pair ? max(key, partner) : min(key, partner);
      }
    }
  }
}

__device__ __forceinline__ void warp_bitonic_sort_asc64(uint32_t* keys, uint32_t lane) {
  constexpr uint32_t kMask        = 0xffffffffu;
  constexpr int      kKeysPerLane = 2;
  constexpr int      kNumKeys     = 64;
#pragma unroll
  for (int k = 2; k <= kNumKeys; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      const bool asc_pair = k < kNumKeys ? ((lane & (k / 2)) == 0) : true;
      if (j >= kKeysPerLane) {
        const int  mask       = j / kKeysPerLane;
        const bool lower_lane = (lane & mask) == 0;
#pragma unroll
        for (int slot = 0; slot < kKeysPerLane; ++slot) {
          const uint32_t mine  = keys[slot];
          const uint32_t other = __shfl_xor_sync(kMask, mine, mask);
          keys[slot]           = lower_lane ? (asc_pair ? min(mine, other) : max(mine, other))
                                            : (asc_pair ? max(mine, other) : min(mine, other));
        }
      } else {
        if (asc_pair) {
          if (keys[0] > keys[1]) {
            uint32_t tmp = keys[0];
            keys[0]      = keys[1];
            keys[1]      = tmp;
          }
        } else {
          if (keys[0] < keys[1]) {
            uint32_t tmp = keys[0];
            keys[0]      = keys[1];
            keys[1]      = tmp;
          }
        }
      }
    }
  }
}

template <uint32_t TopK, uint32_t MaxK, bool ForceBlocksCountInTopK>
__global__ __launch_bounds__(kSmallTopkThreadsPerBlock) void sparse_topk_small_k_kernel(float const* __restrict__ in,
                                                                                        int32_t* __restrict__ out,
                                                                                        uint32_t       total_q,
                                                                                        uint32_t       num_qo_heads,
                                                                                        uint32_t       max_k_tiles,
                                                                                        uint32_t       num_valid_pages,
                                                                                        int64_t const* query_positions,
                                                                                        uint32_t       output_width,
                                                                                        uint32_t       force_begin,
                                                                                        uint32_t       force_end) {
  static_assert(TopK <= kMaxTopK, "TopK must fit in one warp for final index sorting.");
  static_assert(MaxK % 32 == 0, "MaxK must be a whole number of warp lanes.");
  constexpr uint32_t kWarpSize     = 32;
  constexpr uint32_t kItemsPerLane = MaxK / kWarpSize;

  uint32_t lane     = threadIdx.x;
  uint32_t warp_id  = threadIdx.y;
  uint32_t row_idx  = static_cast<uint32_t>(blockIdx.x) * kSmallTopkWarpsPerBlock + warp_id;
  uint32_t num_rows = total_q * num_qo_heads;
  if (row_idx >= num_rows) {
    return;
  }

  uint32_t     q_abs      = row_idx / num_qo_heads;
  uint32_t     head_idx   = row_idx - q_abs * num_qo_heads;
  size_t       row_offset = (static_cast<size_t>(q_abs) * num_qo_heads + head_idx) * max_k_tiles;
  float const* logits     = in + row_offset;
  int32_t*     row_out    = out + (static_cast<size_t>(q_abs) * num_qo_heads + head_idx) * output_width;

  uint32_t row_num_valid_pages = get_row_num_valid_pages(q_abs, num_valid_pages, query_positions);
  if (row_num_valid_pages <= output_width) {
    for (uint32_t idx = lane; idx < output_width; idx += kWarpSize) {
      row_out[idx] = idx < row_num_valid_pages ? static_cast<int32_t>(idx) : static_cast<int32_t>(-1);
    }
    return;
  }
  uint32_t force_end_start = force_end <= row_num_valid_pages ? row_num_valid_pages - force_end : 0;

  int   local_idx[kItemsPerLane];
  float local_score[kItemsPerLane];

#pragma unroll
  for (uint32_t item = 0; item < kItemsPerLane; ++item) {
    uint32_t idx      = lane + item * kWarpSize;
    bool     valid    = idx < max_k_tiles && is_topk_candidate(idx, force_begin, force_end_start);
    local_idx[item]   = valid ? static_cast<int>(idx) : -1;
    local_score[item] = valid ? logits[idx] : -FLT_MAX;
  }

  uint32_t ordinary_topk = ForceBlocksCountInTopK ? TopK - force_begin - force_end : TopK;
  uint32_t key32         = ~0u;
  uint32_t keys64[2]     = {~0u, ~0u};
#pragma unroll
  for (uint32_t rank = 0; rank < TopK; ++rank) {
    if (rank >= ordinary_topk) {
      break;
    }
    int   best_idx   = local_idx[0];
    float best_score = local_score[0];

#pragma unroll
    for (uint32_t item = 1; item < kItemsPerLane; ++item) {
      if (topk_score_index_better(local_idx[item], local_score[item], best_idx, best_score)) {
        best_idx   = local_idx[item];
        best_score = local_score[item];
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      int   other_idx   = __shfl_xor_sync(0xffffffffu, best_idx, offset);
      float other_score = __shfl_xor_sync(0xffffffffu, best_score, offset);
      if (topk_score_index_better(other_idx, other_score, best_idx, best_score)) {
        best_idx   = other_idx;
        best_score = other_score;
      }
    }

    int picked_idx = __shfl_sync(0xffffffffu, best_idx, 0);
    if (lane == rank) {
      key32 = picked_idx >= 0 ? static_cast<uint32_t>(picked_idx) : ~0u;
    }
    if (lane == rank / 2) {
      keys64[rank & 1] = picked_idx >= 0 ? static_cast<uint32_t>(picked_idx) : ~0u;
    }

#pragma unroll
    for (uint32_t item = 0; item < kItemsPerLane; ++item) {
      if (local_idx[item] == picked_idx) {
        local_idx[item]   = -1;
        local_score[item] = -FLT_MAX;
      }
    }
  }

  if (lane >= ordinary_topk && lane < output_width) {
    const int32_t idx =
        forced_slot_to_block(lane - ordinary_topk, force_begin, force_end, force_end_start, row_num_valid_pages);
    key32 = idx >= 0 ? static_cast<uint32_t>(idx) : ~0u;
  }
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    const uint32_t pos = lane * 2 + slot;
    if (pos >= ordinary_topk && pos < output_width) {
      const int32_t idx =
          forced_slot_to_block(pos - ordinary_topk, force_begin, force_end, force_end_start, row_num_valid_pages);
      keys64[slot] = idx >= 0 ? static_cast<uint32_t>(idx) : ~0u;
    }
  }

  if (ForceBlocksCountInTopK || output_width <= 32) {
    warp_bitonic_sort_asc32(key32, lane);
    if (lane < output_width) {
      row_out[lane] = key32 == ~0u ? static_cast<int32_t>(-1) : static_cast<int32_t>(key32);
    }
  } else {
    warp_bitonic_sort_asc64(keys64, lane);
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const uint32_t pos = lane * 2 + slot;
      if (pos < output_width) {
        const uint32_t key = keys64[slot];
        row_out[pos]       = key == ~0u ? static_cast<int32_t>(-1) : static_cast<int32_t>(key);
      }
    }
  }
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, IdxT len, Func f) {
  constexpr int kWarpSize = 32;
  using WideT             = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int kItemsPerScalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T     array[kItemsPerScalar];
    } wide;

    int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                       ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                       : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }

    WideT const* in_cast  = reinterpret_cast<WideT const*>(in + skip_cnt);
    IdxT const   len_cast = (len - skip_cnt) / kItemsPerScalar;
    for (IdxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      IdxT real_i = skip_cnt + i * kItemsPerScalar;
#pragma unroll
      for (int j = 0; j < kItemsPerScalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(kWarpSize >= kItemsPerScalar);
    if (thread_rank < static_cast<size_t>(skip_cnt)) {
      f(in[thread_rank], thread_rank);
    }
    IdxT remain_i = skip_cnt + len_cast * kItemsPerScalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <int      Step,
          int      NumThreads,
          int      NumBins,
          int      NumFinalItems,
          uint32_t TopK,
          bool     ForceBlocksCountInTopK,
          typename SmemFinalType,
          typename SmemOutputType>
__device__ bool process_histogram_step(float const*   logits,
                                       int            row_end,
                                       uint32_t&      logit_pattern,
                                       int&           threshold_bin_idx,
                                       SmemOutputType smem_output,
                                       int*           smem_threshold_bin_idx,
                                       int*           smem_final_dst_idx,
                                       int*           smem_final_bin_size,
                                       int*           smem_found_topk_values,
                                       SmemFinalType& smem_final,
                                       int            row_start,
                                       uint32_t       force_begin,
                                       uint32_t       force_end_start,
                                       uint32_t       num_valid_pages) {
  if constexpr (Step == 1) {
    // The FP32 pass rescans the ordinary interval, not only the FP16 threshold bin.
    // Discard coarse selections while retaining the reserved forced slots.
    if (threadIdx.x == 0) {
      smem_found_topk_values[0] = ForceBlocksCountInTopK ? force_begin + num_valid_pages - force_end_start : 0;
    }
  }
  // One histogram pass narrows the top-k threshold. Later passes refine the
  // float-bit prefix only when the current threshold bin is still too large.
  static_assert(NumBins % 4 == 0, "Histogram must contain whole int4 groups.");
  int4* histogram_vectors = reinterpret_cast<int4*>(smem_final.histo.data);
#pragma unroll
  for (int idx = threadIdx.x; idx < NumBins / 4; idx += NumThreads) {
    histogram_vectors[idx] = make_int4(0, 0, 0, 0);
  }
  __syncthreads();

  constexpr auto pattern_shift = Step < 2 ? 0 : Step == 2 ? 22 : 12;
  if constexpr (Step == 2) {
    logit_pattern = static_cast<uint32_t>(threshold_bin_idx & 0x3ff) << pattern_shift;
  } else if constexpr (Step == 3) {
    logit_pattern |= static_cast<uint32_t>(threshold_bin_idx & 0x3ff) << pattern_shift;
  }

  int  thread_first_bin   = NumBins;
  auto distribute_to_bins = [&](float score, int) {
    if (is_partial_match<pattern_shift>(score, logit_pattern)) {
      uint32_t bin_idx = extract_bin_idx<Step>(score);
      thread_first_bin = min(thread_first_bin, static_cast<int>(bin_idx));
      atomicAdd(&smem_final.histo.data[bin_idx], 1);
    }
  };
  vectorized_process(threadIdx.x, NumThreads, logits + row_start, row_end - row_start, distribute_to_bins);
  using Reduce  = cub::BlockReduce<int, NumThreads>;
  int first_bin = Reduce(smem_final.histo.reduce).Reduce(thread_first_bin, [](int a, int b) { return min(a, b); });
  // Publish the initial scan bound; the same slot later holds the threshold.
  if (threadIdx.x == 0) {
    smem_threshold_bin_idx[0] = first_bin;
  }
  __syncthreads();

  int last_value  = smem_found_topk_values[0];
  int first_round = smem_threshold_bin_idx[0] / NumThreads;
  for (int round = first_round; round < NumBins / NumThreads; ++round) {
    int idx       = threadIdx.x + NumThreads * round;
    int bin_count = smem_final.histo.data[idx];

    int prefix_sum = 0;
    int total_sum  = 0;
    using Scan     = cub::BlockScan<int, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    Scan(smem_final.histo.scan).ExclusiveSum(bin_count, prefix_sum, total_sum);

    prefix_sum += last_value;
    total_sum += last_value;
    if (prefix_sum < static_cast<int>(TopK) && prefix_sum + bin_count >= static_cast<int>(TopK)) {
      smem_threshold_bin_idx[0] = idx;
      smem_final_bin_size[0]    = bin_count;
    }
    // Publish the threshold and protect scan storage before a uniform exit.
    __syncthreads();
    if (total_sum >= static_cast<int>(TopK)) {
      break;
    }
    last_value = total_sum;
  }

  // The final scan-round barrier already publishes the threshold and protects
  // storage reuse. The block-wide aggregate makes the loop exit uniform.
  threshold_bin_idx      = smem_threshold_bin_idx[0];
  const bool bounded_bin = smem_final_bin_size[0] <= NumFinalItems;

  int  first_terminal_bin = 4;
  auto process_bins       = [&](float score, int idx) {
    idx += row_start;
    if (is_partial_match<pattern_shift>(score, logit_pattern)) {
      uint32_t bin_idx = extract_bin_idx<Step>(score);
      if (static_cast<int>(bin_idx) < threshold_bin_idx) {
        int dst_idx          = atomicAdd(&smem_found_topk_values[0], 1);
        smem_output[dst_idx] = idx;
      }
      if (static_cast<int>(bin_idx) == threshold_bin_idx && bounded_bin) {
        int dst_idx                       = atomicAdd(&smem_final_dst_idx[0], 1);
        smem_final.items.logits[dst_idx]  = score;
        smem_final.items.indices[dst_idx] = idx;
      }
      if constexpr (Step == 3) {
        if (static_cast<int>(bin_idx) == threshold_bin_idx && !bounded_bin) {
          first_terminal_bin = min(first_terminal_bin, static_cast<int>(ordered_score_bits(score) & 3u));
        }
      }
    }
  };
  vectorized_process(threadIdx.x, NumThreads, logits + row_start, row_end - row_start, process_bins);
  if constexpr (Step == 3) {
    if (!bounded_bin) {
      int first = Reduce(smem_final.histo.reduce).Reduce(first_terminal_bin, [](int a, int b) { return min(a, b); });
      if (threadIdx.x == 0) {
        // The threshold is cached; reuse its slot to publish the terminal start.
        smem_threshold_bin_idx[0] = first;
      }
    }
  }
  __syncthreads();

  if constexpr (Step == 3) {
    // A bounded final bin can use the same exact FP32 ranking as earlier passes.
    if (bounded_bin) {
      return false;
    }
    // The last histogram leaves two score bits unresolved. Refine those bits,
    // then scan in logical-index order so ties never depend on atomic arrival.
    int selected = smem_found_topk_values[0];
    for (uint32_t low_bin = smem_threshold_bin_idx[0]; low_bin < 4 && selected < static_cast<int>(TopK); ++low_bin) {
      for (int base = row_start; base < row_end && selected < static_cast<int>(TopK); base += NumThreads) {
        int idx       = base + threadIdx.x;
        int candidate = 0;
        if (idx < row_end) {
          float    score = logits[idx];
          uint32_t bits  = ordered_score_bits(score);
          candidate      = is_partial_match<pattern_shift>(score, logit_pattern) &&
                      static_cast<int>(extract_bin_idx<Step>(score)) == threshold_bin_idx && (bits & 3u) == low_bin;
        }
        int prefix = 0;
        int count  = 0;
        using Scan = cub::BlockScan<int, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
        Scan(smem_final.histo.scan).ExclusiveSum(candidate, prefix, count);
        if (candidate && selected + prefix < static_cast<int>(TopK)) {
          smem_output[selected + prefix] = idx;
        }
        selected += count;
        __syncthreads();
      }
    }
  }

  return !bounded_bin;
}

template <uint32_t TopK, bool ForceBlocksCountInTopK>
__global__ __launch_bounds__(kIndexerThreadsPerBlock) void sparse_topk_indexer_kernel(float const* __restrict__ in,
                                                                                      int32_t* __restrict__ out,
                                                                                      uint32_t       total_q,
                                                                                      uint32_t       num_qo_heads,
                                                                                      uint32_t       max_k_tiles,
                                                                                      uint32_t       num_valid_pages,
                                                                                      int64_t const* query_positions,
                                                                                      uint32_t       output_width,
                                                                                      uint32_t       force_begin,
                                                                                      uint32_t       force_end) {
  static_assert(TopK <= kMaxTopK, "TopK must fit in one warp for final index sorting.");
  constexpr int kNumThreads    = kIndexerThreadsPerBlock;
  constexpr int kNumBins       = kIndexerNumBins;
  constexpr int kNumFinalItems = kIndexerNumFinalItems;

  using Scan   = cub::BlockScan<int, kNumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  using Reduce = cub::BlockReduce<int, kNumThreads>;
  struct FinalItems {
    int   indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };
  struct Histogram {
    union {
      typename Scan::TempStorage   scan;
      typename Reduce::TempStorage reduce;
    };
    alignas(16) int data[kNumBins];
  };
  __shared__ union {
    FinalItems items;
    Histogram  histo;
  } smem_final;

  alignas(kSmemAlignmentBytes) extern __shared__ int32_t smem_output[];
  __shared__ int                                         smem_threshold_bin_idx[1];
  __shared__ int                                         smem_final_dst_idx[1];
  __shared__ int                                         smem_final_bin_size[1];
  __shared__ int                                         smem_found_topk_values[1];

  uint32_t q_abs    = static_cast<uint32_t>(blockIdx.x);
  uint32_t head_idx = static_cast<uint32_t>(blockIdx.y);
  // MATE maxscore is stored as [T, H, K], so each (token, head) row is already
  // contiguous along K and does not need MSA's transpose workspace.
  size_t       row_offset = (static_cast<size_t>(q_abs) * num_qo_heads + head_idx) * max_k_tiles;
  float const* logits     = in + row_offset;
  int32_t*     row_out    = out + (static_cast<size_t>(q_abs) * num_qo_heads + head_idx) * output_width;

  uint32_t row_num_valid_pages = get_row_num_valid_pages(q_abs, num_valid_pages, query_positions);
  if (row_num_valid_pages <= output_width) {
    for (uint32_t idx = threadIdx.x; idx < output_width; idx += kNumThreads) {
      row_out[idx] = idx < row_num_valid_pages ? static_cast<int32_t>(idx) : static_cast<int32_t>(-1);
    }
    return;
  }
  // Nontrivial rows have disjoint forced ranges and a nonempty ordinary interval,
  // which excludes both forced pages and the future/padded tail.
  uint32_t  force_end_start = force_end <= row_num_valid_pages ? row_num_valid_pages - force_end : 0;
  int const row_start       = static_cast<int>(force_begin);
  int const row_end         = static_cast<int>(force_end_start);
  uint32_t  forced_topk     = ForceBlocksCountInTopK ? force_begin + force_end : 0;

  for (uint32_t idx = threadIdx.x; idx < output_width; idx += kNumThreads) {
    smem_output[idx] = -1;
  }
  if (threadIdx.x == 0) {
    smem_final_dst_idx[0]     = 0;
    smem_found_topk_values[0] = forced_topk;
  }
  __syncthreads();

  int      threshold_bin_idx = -1;
  uint32_t logit_pattern     = 0;
  bool     continue_to_next_step =
      forced_topk < TopK &&
      process_histogram_step<0, kNumThreads, kNumBins, kNumFinalItems, TopK, ForceBlocksCountInTopK>(
          logits,
          row_end,
          logit_pattern,
          threshold_bin_idx,
          smem_output,
          smem_threshold_bin_idx,
          smem_final_dst_idx,
          smem_final_bin_size,
          smem_found_topk_values,
          smem_final,
          row_start,
          force_begin,
          force_end_start,
          row_num_valid_pages);
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<1, kNumThreads, kNumBins, kNumFinalItems, TopK, ForceBlocksCountInTopK>(
            logits,
            row_end,
            logit_pattern,
            threshold_bin_idx,
            smem_output,
            smem_threshold_bin_idx,
            smem_final_dst_idx,
            smem_final_bin_size,
            smem_found_topk_values,
            smem_final,
            row_start,
            force_begin,
            force_end_start,
            row_num_valid_pages);
  }
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<2, kNumThreads, kNumBins, kNumFinalItems, TopK, ForceBlocksCountInTopK>(
            logits,
            row_end,
            logit_pattern,
            threshold_bin_idx,
            smem_output,
            smem_threshold_bin_idx,
            smem_final_dst_idx,
            smem_final_bin_size,
            smem_found_topk_values,
            smem_final,
            row_start,
            force_begin,
            force_end_start,
            row_num_valid_pages);
  }
  if (continue_to_next_step) {
    continue_to_next_step =
        process_histogram_step<3, kNumThreads, kNumBins, kNumFinalItems, TopK, ForceBlocksCountInTopK>(
            logits,
            row_end,
            logit_pattern,
            threshold_bin_idx,
            smem_output,
            smem_threshold_bin_idx,
            smem_final_dst_idx,
            smem_final_bin_size,
            smem_found_topk_values,
            smem_final,
            row_start,
            force_begin,
            force_end_start,
            row_num_valid_pages);
  }

  if (!continue_to_next_step) {
    int base_idx    = smem_found_topk_values[0];
    int final_count = smem_final_dst_idx[0];
    // Rank only candidates in the threshold bin; bins above it were already
    // copied directly into smem_output.
    for (int i = threadIdx.x; i < final_count; i += kNumThreads) {
      int   out_index     = 0;
      float logit         = smem_final.items.logits[i];
      int   logical_index = smem_final.items.indices[i];
      for (int j = 0; j < final_count; ++j) {
        float other_logit = smem_final.items.logits[j];
        if (logit < other_logit || (logit == other_logit && logical_index > smem_final.items.indices[j])) {
          ++out_index;
        }
      }
      if (out_index + base_idx < static_cast<int>(TopK)) {
        smem_output[out_index + base_idx] = logical_index;
      }
    }
  }

  const uint32_t forced_slots  = force_begin + force_end;
  const uint32_t forced_offset = ForceBlocksCountInTopK ? 0 : TopK;
  for (uint32_t slot = threadIdx.x; slot < forced_slots; slot += kNumThreads) {
    smem_output[forced_offset + slot] =
        forced_slot_to_block(slot, force_begin, force_end, force_end_start, row_num_valid_pages);
  }
  __syncthreads();
  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane    = threadIdx.x & 31;
  if (warp_id != 0) {
    return;
  }

  // Sparse FMHA consumes block indexes in ascending order; invalid/OOB entries
  // become -1 and sort to the tail via the all-ones sentinel key.
  if (output_width <= 32) {
    uint32_t key = ~0u;
    if (lane < output_width) {
      const int32_t idx = smem_output[lane];
      key = idx >= 0 && static_cast<uint32_t>(idx) < row_num_valid_pages ? static_cast<uint32_t>(idx) : ~0u;
    }
    warp_bitonic_sort_asc32(key, lane);
    if (lane < output_width) {
      row_out[lane] = key == ~0u ? static_cast<int32_t>(-1) : static_cast<int32_t>(key);
    }
  } else {
    uint32_t keys[2] = {~0u, ~0u};
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const uint32_t pos = lane * 2 + slot;
      if (pos < output_width) {
        const int32_t idx = smem_output[pos];
        keys[slot] = idx >= 0 && static_cast<uint32_t>(idx) < row_num_valid_pages ? static_cast<uint32_t>(idx) : ~0u;
      }
    }
    warp_bitonic_sort_asc64(keys, lane);
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const uint32_t pos = lane * 2 + slot;
      if (pos < output_width) {
        row_out[pos] = keys[slot] == ~0u ? static_cast<int32_t>(-1) : static_cast<int32_t>(keys[slot]);
      }
    }
  }
}

__global__ void sparse_topk_identity_fill_kernel(int32_t* __restrict__ out,
                                                 uint32_t total_q,
                                                 uint32_t num_qo_heads,
                                                 uint32_t max_k_tiles,
                                                 uint32_t num_valid_pages,
                                                 uint32_t output_width) {
  uint32_t q_abs       = static_cast<uint32_t>(blockIdx.x);
  uint32_t head_idx    = static_cast<uint32_t>(blockIdx.y);
  int32_t* row         = out + (static_cast<size_t>(q_abs) * num_qo_heads + head_idx) * output_width;
  uint32_t valid_count = num_valid_pages < max_k_tiles ? num_valid_pages : max_k_tiles;
  for (uint32_t i = threadIdx.x; i < output_width; i += blockDim.x) {
    row[i] = i < valid_count ? static_cast<int32_t>(i) : static_cast<int32_t>(-1);
  }
}

template <uint32_t TopK, bool ForceBlocksCountInTopK>
inline musaError_t launch_sparse_topk_select_typed(float const*   max_score,
                                                   int32_t*       output_indices,
                                                   uint32_t       total_q,
                                                   uint32_t       num_qo_heads,
                                                   uint32_t       max_k_tiles,
                                                   uint32_t       num_valid_pages,
                                                   int64_t const* query_positions,
                                                   uint32_t       force_begin,
                                                   uint32_t       force_end,
                                                   musaStream_t   stream) {
  static_assert(TopK <= kMaxTopK, "TopK must fit in one warp for final index sorting.");
  if (total_q == 0 || num_qo_heads == 0) {
    return musaSuccess;
  }
  const uint32_t output_width = ForceBlocksCountInTopK ? TopK : TopK + force_begin + force_end;
  if (query_positions == nullptr && num_valid_pages <= output_width) {
    // Trivial selector: all real blocks fit in top-k, so emit identity indexes
    // and pad the rest with -1.
    dim3 grid(total_q, num_qo_heads);
    dim3 block(output_width);
    sparse_topk_identity_fill_kernel<<<grid, block, 0, stream>>>(
        output_indices, total_q, num_qo_heads, max_k_tiles, num_valid_pages, output_width);
    return musaGetLastError();
  }
  if (max_k_tiles <= kSmallTopkMaxK) {
    uint32_t num_rows = total_q * num_qo_heads;
    dim3     grid((num_rows + kSmallTopkWarpsPerBlock - 1) / kSmallTopkWarpsPerBlock);
    dim3     block(32, kSmallTopkWarpsPerBlock);
    if (max_k_tiles <= 64) {
      sparse_topk_small_k_kernel<TopK, 64, ForceBlocksCountInTopK><<<grid, block, 0, stream>>>(max_score,
                                                                                               output_indices,
                                                                                               total_q,
                                                                                               num_qo_heads,
                                                                                               max_k_tiles,
                                                                                               num_valid_pages,
                                                                                               query_positions,
                                                                                               output_width,
                                                                                               force_begin,
                                                                                               force_end);
      return musaGetLastError();
    } else if (max_k_tiles <= 128) {
      sparse_topk_small_k_kernel<TopK, 128, ForceBlocksCountInTopK><<<grid, block, 0, stream>>>(max_score,
                                                                                                output_indices,
                                                                                                total_q,
                                                                                                num_qo_heads,
                                                                                                max_k_tiles,
                                                                                                num_valid_pages,
                                                                                                query_positions,
                                                                                                output_width,
                                                                                                force_begin,
                                                                                                force_end);
      return musaGetLastError();
    } else {
      sparse_topk_small_k_kernel<TopK, 256, ForceBlocksCountInTopK><<<grid, block, 0, stream>>>(max_score,
                                                                                                output_indices,
                                                                                                total_q,
                                                                                                num_qo_heads,
                                                                                                max_k_tiles,
                                                                                                num_valid_pages,
                                                                                                query_positions,
                                                                                                output_width,
                                                                                                force_begin,
                                                                                                force_end);
      return musaGetLastError();
    }
  }
  if (max_k_tiles >= 12288) {
    return musaErrorNotSupported;
  }

  size_t dyn_smem_bytes = round_up_smem_bytes(output_width * sizeof(int32_t));
  dim3   grid(total_q, num_qo_heads);
  sparse_topk_indexer_kernel<TopK, ForceBlocksCountInTopK>
      <<<grid, kIndexerThreadsPerBlock, dyn_smem_bytes, stream>>>(max_score,
                                                                  output_indices,
                                                                  total_q,
                                                                  num_qo_heads,
                                                                  max_k_tiles,
                                                                  num_valid_pages,
                                                                  query_positions,
                                                                  output_width,
                                                                  force_begin,
                                                                  force_end);
  return musaGetLastError();
}

inline musaError_t launch_sparse_topk_select(float const*   max_score,
                                             int32_t*       output_indices,
                                             uint32_t       total_q,
                                             uint32_t       num_qo_heads,
                                             uint32_t       max_k_tiles,
                                             uint32_t       topk,
                                             uint32_t       num_valid_pages,
                                             int64_t const* query_positions,
                                             uint32_t       force_begin,
                                             uint32_t       force_end,
                                             bool           force_blocks_count_in_topk,
                                             musaStream_t   stream) {
  switch (topk) {
    case 4:
      return force_blocks_count_in_topk ? launch_sparse_topk_select_typed<4, true>(max_score,
                                                                                   output_indices,
                                                                                   total_q,
                                                                                   num_qo_heads,
                                                                                   max_k_tiles,
                                                                                   num_valid_pages,
                                                                                   query_positions,
                                                                                   force_begin,
                                                                                   force_end,
                                                                                   stream)
                                        : launch_sparse_topk_select_typed<4, false>(max_score,
                                                                                    output_indices,
                                                                                    total_q,
                                                                                    num_qo_heads,
                                                                                    max_k_tiles,
                                                                                    num_valid_pages,
                                                                                    query_positions,
                                                                                    force_begin,
                                                                                    force_end,
                                                                                    stream);
    case 8:
      return force_blocks_count_in_topk ? launch_sparse_topk_select_typed<8, true>(max_score,
                                                                                   output_indices,
                                                                                   total_q,
                                                                                   num_qo_heads,
                                                                                   max_k_tiles,
                                                                                   num_valid_pages,
                                                                                   query_positions,
                                                                                   force_begin,
                                                                                   force_end,
                                                                                   stream)
                                        : launch_sparse_topk_select_typed<8, false>(max_score,
                                                                                    output_indices,
                                                                                    total_q,
                                                                                    num_qo_heads,
                                                                                    max_k_tiles,
                                                                                    num_valid_pages,
                                                                                    query_positions,
                                                                                    force_begin,
                                                                                    force_end,
                                                                                    stream);
    case 16:
      return force_blocks_count_in_topk ? launch_sparse_topk_select_typed<16, true>(max_score,
                                                                                    output_indices,
                                                                                    total_q,
                                                                                    num_qo_heads,
                                                                                    max_k_tiles,
                                                                                    num_valid_pages,
                                                                                    query_positions,
                                                                                    force_begin,
                                                                                    force_end,
                                                                                    stream)
                                        : launch_sparse_topk_select_typed<16, false>(max_score,
                                                                                     output_indices,
                                                                                     total_q,
                                                                                     num_qo_heads,
                                                                                     max_k_tiles,
                                                                                     num_valid_pages,
                                                                                     query_positions,
                                                                                     force_begin,
                                                                                     force_end,
                                                                                     stream);
    default:
      return musaErrorNotSupported;
  }
}

}  // namespace mate::attention::msa::sparse_topk
