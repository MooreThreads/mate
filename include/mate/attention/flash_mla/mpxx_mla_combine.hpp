#pragma once

#include <mutlass/array.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>

#include <mute/tensor.hpp>

#include "mpxx_params.hpp"

namespace mate::flash_mla {

using namespace mute;

template <typename ElementT,
          int  HEAD_DIM_V,
          int  BLOCK_SIZE_M,
          int  D_PARTS,
          int  MAX_SPLITS,
          int  NUM_THREADS,
          bool is_varlen_q,
          int  HeadRatio    = 0,
          int  SPLIT_GROUPS = (D_PARTS + 3) / 4,
          int  VECTOR_WIDTH =
              (HEAD_DIM_V * SPLIT_GROUPS / (32 * D_PARTS) < 8 ? HEAD_DIM_V * SPLIT_GROUPS / (32 * D_PARTS) : 8)>
__global__ void __launch_bounds__(NUM_THREADS, 1) mpxx_mla_combine_kernel(const MlaCombineParams params) {
  // grid_shape: [batch_size, num_q_heads*s_q / BLOCK_SIZE_M * D_PARTS]
  // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result
  static_assert(NUM_THREADS / 32 == BLOCK_SIZE_M);  // The number of warps == block_size_m
  const int batch_idx   = blockIdx.x;
  const int m_block_idx = blockIdx.y / D_PARTS;
  const int d_part_idx  = blockIdx.y % D_PARTS;
  const int warp_idx    = threadIdx.x / 32;
  const int lane_idx    = threadIdx.x % 32;

  const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
  const int end_split_idx   = __ldg(params.num_splits_ptr + batch_idx + 1);
  const int my_num_splits   = end_split_idx - start_split_idx;
  // TODO: device assert under DEBUG macro
  // FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);
  if (my_num_splits == 1) {
    return;
  }

  const int     h_r            = HeadRatio > 0 ? HeadRatio : params.h_r;
  const int     h_k            = HeadRatio > 0 ? 1 : params.h_k;
  const int64_t lse_row_stride = HeadRatio > 0 ? 1 : params.lse_row_stride;
  const int     max_num_q_seqs = params.max_q_seq_per_hk * h_k;
  int           q_seq_start    = 0;
  int           q_seq_end      = 0;
  int           num_q_seqs     = max_num_q_seqs;
  if constexpr (is_varlen_q) {
    q_seq_start = *(params.seqlens_q_ptr + batch_idx);
    q_seq_end   = *(params.seqlens_q_ptr + batch_idx + 1);
    num_q_seqs  = (q_seq_end - q_seq_start) * h_r * h_k;
  }
  const int num_cur_valid_q_seqs = min(BLOCK_SIZE_M, num_q_seqs - m_block_idx * BLOCK_SIZE_M);
  Tensor gLseAccum = make_tensor(make_gmem_ptr((float*)params.softmax_lseaccum_ptr + start_split_idx * max_num_q_seqs +
                                               m_block_idx * BLOCK_SIZE_M),
                                 Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
                                 make_stride(max_num_q_seqs, _1{}));
  if (warp_idx >= num_cur_valid_q_seqs) return;
  const int q_seq_idx  = m_block_idx * BLOCK_SIZE_M + warp_idx;
  const int k_head_idx = HeadRatio > 0 ? 0 : q_seq_idx / params.max_q_seq_per_hk;
  const int row_idx    = HeadRatio > 0 ? q_seq_idx : q_seq_idx % params.max_q_seq_per_hk;
  const int q_idx      = is_varlen_q ? q_seq_idx / h_r : 0;
  const int h_idx      = is_varlen_q ? q_seq_idx % h_r : 0;

  constexpr int NUM_LSE_PER_THREAD = mute::ceil_div(MAX_SPLITS, 32);
  float         local_lse[NUM_LSE_PER_THREAD];
  // Warp #i gathers LseAccum for seq #i
  {
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
      const int split_idx = i * 32 + lane_idx;
      local_lse[i]        = split_idx < my_num_splits ? gLseAccum(split_idx, warp_idx) : -INFINITY;
    }

    float max_lse = -INFINITY;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) max_lse = max(max_lse, local_lse[i]);
    MUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2)
      max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

    float sum_lse = 0;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
    MUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

    float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : logf(sum_lse) + max_lse;
    if constexpr (is_varlen_q) {
      // B H Q
      // H, TOTAL_Q
      // m_block_idx --> max_q_per_hk
      float* out_lse = (float*)params.softmax_lse_ptr;

      bool is_valid_q = HeadRatio > 0 || q_seq_start + q_idx < q_seq_end;
      if (is_valid_q && lane_idx == 0 && d_part_idx == 0) {
        out_lse[h_idx * params.lse_head_stride + (q_seq_start + q_idx) * lse_row_stride] = global_lse;
      }
    } else {
      if (lane_idx == 0 && d_part_idx == 0) {
        const int64_t lse_offset =
            batch_idx * params.lse_batch_stride + k_head_idx * params.lse_head_stride + row_idx * lse_row_stride;
        reinterpret_cast<float*>(params.softmax_lse_ptr)[lse_offset] = global_lse;
      }
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) local_lse[i] = expf(local_lse[i] - global_lse);
  }

  // Warp #i accumulates activation for seq #i
  {
    const int64_t row_offset_oaccum =
        (int64_t)(start_split_idx * max_num_q_seqs + m_block_idx * BLOCK_SIZE_M + warp_idx) * HEAD_DIM_V;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<MAX_SPLITS>, Int<HEAD_DIM_V>>{},
                                 make_stride(max_num_q_seqs * HEAD_DIM_V, _1{}));

    static_assert(HEAD_DIM_V % (32 * D_PARTS) == 0);
    const int d_offset = d_part_idx * HEAD_DIM_V / D_PARTS;

    constexpr int ELEMS_PER_THREAD = HEAD_DIM_V * SPLIT_GROUPS / (32 * D_PARTS);
    static_assert(SPLIT_GROUPS > 0 && 32 % SPLIT_GROUPS == 0);
    static_assert(VECTOR_WIDTH > 0 && ELEMS_PER_THREAD % VECTOR_WIDTH == 0);
    const int d_lane  = lane_idx / SPLIT_GROUPS;
    auto      d_index = [&](int i) {
      return d_offset + d_lane * VECTOR_WIDTH + (i / VECTOR_WIDTH) * (32 / SPLIT_GROUPS) * VECTOR_WIDTH +
             i % VECTOR_WIDTH;
    };
    float result[ELEMS_PER_THREAD];
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) result[i] = 0.0f;

    // The weight provider is in the same split residue group, including the tail iteration.
    constexpr uint32_t residue_mask = uint32_t(uint64_t{0xffffffffu} / ((uint64_t{1} << SPLIT_GROUPS) - 1));
    const uint32_t     weight_mask  = residue_mask << (lane_idx % SPLIT_GROUPS);
#pragma unroll 1
    for (int split = lane_idx % SPLIT_GROUPS; split < my_num_splits; split += SPLIT_GROUPS) {
      float weight = local_lse[0];
      MUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < NUM_LSE_PER_THREAD; ++i)
        if (split >= i * 32) weight = local_lse[i];
      float lse_scale = __shfl_sync(weight_mask, weight, split % 32);
      if (lse_scale != 0.f) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) result[i] += lse_scale * gOaccum(split, d_index(i));
      }
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      MUTLASS_PRAGMA_UNROLL
      for (int offset = SPLIT_GROUPS / 2; offset > 0; offset /= 2)
        result[i] += __shfl_xor_sync(uint32_t(-1), result[i], offset);
    }
    if (lane_idx % SPLIT_GROUPS != 0) return;

    int64_t o_offset =
        batch_idx * params.o_batch_stride + k_head_idx * params.o_head_stride + row_idx * params.o_row_stride;
    if constexpr (is_varlen_q) {
      // TOTAL_Q, H, D
      o_offset = (q_seq_start + q_idx) * params.o_row_stride + h_idx * params.o_head_stride;
    }

    auto o_ptr_local = reinterpret_cast<ElementT*>(params.o_ptr) + o_offset;

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; i += VECTOR_WIDTH) {
      mutlass::Array<ElementT, VECTOR_WIDTH> packed;
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < VECTOR_WIDTH; ++j) packed[j] = (ElementT)result[i + j];
      __builtin_memcpy(o_ptr_local + d_index(i), packed.data(), VECTOR_WIDTH * sizeof(ElementT));
    }
  }
}

}  // namespace mate::flash_mla
