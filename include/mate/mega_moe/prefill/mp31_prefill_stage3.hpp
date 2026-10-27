#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>

#include "mate/mega_moe/mega_moe_sym_buffer.hpp"
#include "mate/mega_moe/mega_moe_unified_layout.hpp"

namespace mega_moe::prefill {

using PrefillStage3Float4 = float __attribute__((vector_size(16)));

template <uint32_t kNumMaxTokensPerRank,
          uint32_t kHidden,
          uint32_t kIntermediateHidden,
          uint32_t kNumExperts,
          uint32_t kNumTopk,
          uint32_t kNumMaxPoolTokens,
          uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumL1RingTokens,
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          uint32_t kNumThreads = 256>
__global__ __launch_bounds__(kNumThreads,
                             1) void mp31_prefill_stage3_impl(const layout::SymBuffer<kNumRanks> sym_buffer,
                                                              const uint32_t                     num_tokens,
                                                              mutlass::bfloat16_t*               output) {
  const auto buffer_layout             = unified::BufferLayout(sym_buffer.get_base_ptr(),
                                                   kNumRanks,
                                                   kNumExperts,
                                                   kNumMaxTokensPerRank,
                                                   kNumTopk,
                                                   kHidden,
                                                   kIntermediateHidden,
                                                   kNumMaxPoolTokens,
                                                   kNumL1RingTokens);
  const auto input_topk_idx_buffer     = buffer_layout.get_input_topk_idx_buffer();
  const auto input_topk_weights_buffer = buffer_layout.get_input_topk_weights_buffer();
  const auto combine_token_buffer      = buffer_layout.get_prefill_combine_token_buffer();

  constexpr uint32_t kElemsPerPackedLoad = sizeof(uint4) / sizeof(mutlass::bfloat16_t);
  static_assert(kHidden % kElemsPerPackedLoad == 0, "Stage 3 hidden must pack into uint4 loads");
  static_assert(kElemsPerPackedLoad == 8, "Stage 3 vector conversion expects eight BF16 elements per load");
  constexpr uint32_t kNumPackedLoads = kHidden / kElemsPerPackedLoad;
  constexpr uint32_t kNumWarps       = kNumThreads / mutlass::NumThreadsPerWarp;
  constexpr uint32_t kNumHiddenWarpTiles =
      (kNumPackedLoads + mutlass::NumThreadsPerWarp - 1) / mutlass::NumThreadsPerWarp;
  const uint32_t warp_idx = threadIdx.x / mutlass::NumThreadsPerWarp;
  const uint32_t lane_idx = threadIdx.x % mutlass::NumThreadsPerWarp;

  for (uint32_t token_idx = blockIdx.x * kNumWarps + warp_idx; token_idx < num_tokens;
       token_idx += kNumMPs * kNumWarps) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
    __musa_loop_transparent_outermost();
#endif
    const auto  topk_idx_data     = input_topk_idx_buffer.get_data_buffer(token_idx);
    const auto  topk_weights_data = input_topk_weights_buffer.get_data_buffer(token_idx);
    const auto* topk_idx_ptr      = topk_idx_data.get_base_ptr<int64_t>();
    const auto* topk_weights_ptr  = topk_weights_data.get_base_ptr<float>();

    for (uint32_t hidden_tile_idx = 0; hidden_tile_idx < kNumHiddenWarpTiles; ++hidden_tile_idx) {
      const uint32_t packed_idx = hidden_tile_idx * mutlass::NumThreadsPerWarp + lane_idx;
      if (packed_idx >= kNumPackedLoads) {
        continue;
      }

      int4                packed_output = make_int4(0, 0, 0, 0);
      PrefillStage3Float4 reduced_lo    = {};
      PrefillStage3Float4 reduced_hi    = {};
#pragma unroll
      for (uint32_t topk_idx = 0; topk_idx < kNumTopk; ++topk_idx) {
        const int64_t expert_idx = __ldg(topk_idx_ptr + topk_idx);
        if (expert_idx < 0) {
          continue;
        }
        const float  route_weight = __ldg(topk_weights_ptr + topk_idx);
        const auto   partial      = combine_token_buffer.get_rank_buffer(topk_idx).get_data_buffer(token_idx);
        const auto*  packed_ptr   = reinterpret_cast<const int4*>(partial.get_base_ptr()) + packed_idx;
        const int4   packed       = __lsu_ld_cache_hint(packed_ptr, 3, 3, 0, 0);
        const auto*  packed_bf16  = reinterpret_cast<const __mt_bfloat164*>(&packed);
        const float4 partial_lo4  = __bfloat1642float4(packed_bf16[0]);
        const float4 partial_hi4  = __bfloat1642float4(packed_bf16[1]);
        const auto&  partial_lo   = reinterpret_cast<const PrefillStage3Float4&>(partial_lo4);
        const auto&  partial_hi   = reinterpret_cast<const PrefillStage3Float4&>(partial_hi4);
        reduced_lo                = __musa_fma_f_bst4_svv(route_weight, partial_lo, reduced_lo);
        reduced_hi                = __musa_fma_f_bst4_svv(route_weight, partial_hi, reduced_hi);
      }

      auto* packed_output_bf16 = reinterpret_cast<__mt_bfloat164*>(&packed_output);
      packed_output_bf16[0]    = __float42bfloat164_rn(reinterpret_cast<const float4&>(reduced_lo));
      packed_output_bf16[1]    = __float42bfloat164_rn(reinterpret_cast<const float4&>(reduced_hi));

      auto* output_ptr = reinterpret_cast<int4*>(output + static_cast<uint64_t>(token_idx) * kHidden) + packed_idx;
      __lsu_st_cache_hint(output_ptr, packed_output, 4, 2, 1, 1);
    }
  }
}

}  // namespace mega_moe::prefill
