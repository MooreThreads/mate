// Adapted from https://github.com/deepseek-ai/DeepGEMM
#pragma once

#include <mutlass/mutlass.h>

#include <mute/tensor.hpp>

namespace mate::deep_gemm {

template <uint32_t kNextN, uint32_t BLOCK_KV, uint32_t kNumWarps, class ElementLogits = float>
__global__ __launch_bounds__(kNumWarps * 32, 1) void mpxx_clean_logits(const uint32_t  seq_len,
                                                                       const uint32_t  seq_len_kv,
                                                                       const uint64_t  stride_kv,
                                                                       const uint32_t* cu_seq_len_k_start,
                                                                       const uint32_t* cu_seq_len_k_end,
                                                                       ElementLogits*  logits) {
  const uint32_t&     num_mps   = gridDim.x;
  const uint32_t&     mp_idx    = blockIdx.x;
  const uint32_t&     warp_idx  = mutlass::canonical_warp_idx_sync();
  const ElementLogits neg_inf   = static_cast<ElementLogits>(-mute::numeric_limits<float>::infinity());
  constexpr uint32_t  kVecElems = 16 / sizeof(ElementLogits);

  // Allocate filled `-inf` shared memory
  extern __shared__ __align__(1024) ElementLogits smem_buffer[];
  MUTE_UNROLL
  for (uint32_t i = threadIdx.x; i < BLOCK_KV; i += kNumWarps * 32) {
    smem_buffer[i] = neg_inf;
  }
  __syncthreads();

  // Assign sequence to each warp
  const auto& assign_task = [&](const uint32_t& num,
                                const uint32_t& idx,
                                const uint32_t& start,
                                const uint32_t& total) -> mute::tuple<uint32_t, uint32_t> {
    const auto &per = total / num, rem = total % num;
    return {start + idx * per + min(idx, rem), per + (idx < rem)};
  };

  auto [mp_seq_start, mp_seq_len]     = assign_task(num_mps, mp_idx, 0, seq_len);
  auto [warp_seq_start, warp_seq_len] = assign_task(kNumWarps, warp_idx, mp_seq_start, mp_seq_len);

  // TME stores are issued collectively by one warp.
  if (warp_idx == 0) {
    for (uint32_t i = mp_seq_start; i < mp_seq_start + mp_seq_len; ++i) {
      const auto& ks         = cu_seq_len_k_start == nullptr ? 0 : __ldg(cu_seq_len_k_start + i / kNextN);
      const auto& ke         = __ldg(cu_seq_len_k_end + i / kNextN) - kNextN + i % kNextN + 1;
      const auto &aligned_ks = ks / kVecElems * kVecElems, aligned_ke = (ke + kVecElems - 1) / kVecElems * kVecElems;

      for (uint32_t left = 0; left < seq_len_kv; left += BLOCK_KV) {
        const auto& right = min(left + BLOCK_KV, static_cast<uint32_t>(stride_kv));
        if (right <= ks or ke <= left) {
          mute::MP31_BLK_COPY_S2G::copy(
              smem_buffer, logits + i * stride_kv + left, (right - left) * sizeof(ElementLogits));
          mute::tme_store_arrive();
        } else {
          if (left < aligned_ks) {
            mute::MP31_BLK_COPY_S2G::copy(
                smem_buffer, logits + i * stride_kv + left, (aligned_ks - left) * sizeof(ElementLogits));
            mute::tme_store_arrive();
          }
          if (aligned_ke < right) {
            mute::MP31_BLK_COPY_S2G::copy(
                smem_buffer, logits + i * stride_kv + aligned_ke, (right - aligned_ke) * sizeof(ElementLogits));
            mute::tme_store_arrive();
          }
        }
      }
    }
  }

  mute::tme_store_wait();
  __syncthreads();

  for (uint32_t i = warp_seq_start; i < warp_seq_start + warp_seq_len; ++i) {
    const auto& ks         = cu_seq_len_k_start == nullptr ? 0 : __ldg(cu_seq_len_k_start + i / kNextN);
    const auto& ke         = __ldg(cu_seq_len_k_end + i / kNextN) - kNextN + i % kNextN + 1;
    const auto &aligned_ks = ks / kVecElems * kVecElems, aligned_ke = (ke + kVecElems - 1) / kVecElems * kVecElems;
    for (uint32_t j = aligned_ks; j < ks; ++j) {
      logits[i * stride_kv + j] = neg_inf;
    }
    for (uint32_t j = ke; j < aligned_ke; ++j) {
      logits[i * stride_kv + j] = neg_inf;
    }
  }
}

}  // namespace mate::deep_gemm
