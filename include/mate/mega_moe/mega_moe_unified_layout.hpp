#pragma once

#include <cstdint>

#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/prefill/prefill_workspace.hpp"

namespace mega_moe::unified {

constexpr uint32_t kDecodeMaxTokensPerRank   = 128;
constexpr uint32_t kDecodeBlockM             = 32;
constexpr uint32_t kPathRegionAlignmentBytes = 4096;

// One symmetric allocation serves both execution paths.  Control state is
// path-local so a decode cleanup cannot invalidate prefill metadata (and vice
// versa), while inputs and the large activation scratch are shared.
struct BufferLayout {
  void*    base;
  uint32_t num_ranks;
  uint32_t num_experts;
  uint32_t num_max_tokens_per_rank;
  uint32_t num_topk;
  uint32_t hidden;
  uint32_t intermediate_hidden;
  uint32_t prefill_pool_capacity_rows;
  uint32_t prefill_l1_ring_tokens;

  __host__ __device__ BufferLayout(void*    base,
                                   uint32_t num_ranks,
                                   uint32_t num_experts,
                                   uint32_t num_max_tokens_per_rank,
                                   uint32_t num_topk,
                                   uint32_t hidden,
                                   uint32_t intermediate_hidden,
                                   uint32_t prefill_pool_capacity_rows = 0,
                                   uint32_t prefill_l1_ring_tokens     = 0)
      : base(base),
        num_ranks(num_ranks),
        num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_topk(num_topk),
        hidden(hidden),
        intermediate_hidden(intermediate_hidden),
        prefill_pool_capacity_rows(prefill_pool_capacity_rows),
        prefill_l1_ring_tokens(prefill_l1_ring_tokens) {
  }

  __host__ __device__ prefill::PrefillSplitWorkspace get_prefill_workspace() const {
    return prefill::PrefillSplitWorkspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
  }

  __host__ __device__ layout::Workspace get_decode_workspace() const {
    const auto decode_workspace_addr = math::align(reinterpret_cast<uint64_t>(get_prefill_workspace().get_end_ptr()),
                                                   static_cast<uint64_t>(kPathRegionAlignmentBytes));
    return layout::Workspace(reinterpret_cast<void*>(decode_workspace_addr),
                             num_ranks,
                             num_experts,
                             kDecodeMaxTokensPerRank,
                             num_topk,
                             kDecodeBlockM);
  }

  __host__ __device__ uint32_t get_decode_num_max_pool_tokens() const {
    return layout::get_num_max_pool_tokens(
        num_ranks, kDecodeMaxTokensPerRank, num_topk, num_experts / num_ranks, kDecodeBlockM);
  }

  __host__ __device__ uint32_t get_decode_num_padded_sf_pool_tokens() const {
    return layout::get_num_padded_sf_pool_tokens(get_decode_num_max_pool_tokens(), kDecodeBlockM);
  }

  __host__ __device__ uint32_t get_prefill_num_max_pool_tokens() const {
    if (prefill_pool_capacity_rows != 0) {
      return prefill_pool_capacity_rows;
    }
    return layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks, prefill::kBlockM);
  }

  __host__ __device__ uint32_t get_prefill_num_padded_sf_pool_tokens() const {
    return layout::get_num_padded_sf_pool_tokens(get_prefill_num_max_pool_tokens(), prefill::kBlockM);
  }

  // The routed L1 activation pool is recycled through a physical ring. Metadata
  // and arrival masks stay indexed by the logical pool block, only the token
  // and scale data are addressed modulo the ring capacity.
  __host__ __device__ uint32_t get_prefill_num_l1_ring_tokens() const {
    return prefill_l1_ring_tokens != 0 ? prefill_l1_ring_tokens : get_prefill_num_max_pool_tokens();
  }

  __host__ __device__ uint32_t get_prefill_num_padded_sf_l1_ring_tokens() const {
    return layout::get_num_padded_sf_pool_tokens(get_prefill_num_l1_ring_tokens(), prefill::kBlockM);
  }

  __host__ __device__ layout::Buffer get_input_token_buffer() const {
    return layout::Buffer(layout::Data(hidden), 1, num_max_tokens_per_rank, get_decode_workspace().get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_input_sf_buffer() const {
    const auto input_token_buffer = get_input_token_buffer();
    return layout::Buffer(
        layout::Data(hidden / 32, false), 1, num_max_tokens_per_rank, input_token_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_input_topk_idx_buffer() const {
    const auto input_sf_buffer = get_input_sf_buffer();
    return layout::Buffer(
        layout::Data(num_topk * sizeof(int64_t), false), 1, num_max_tokens_per_rank, input_sf_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_input_topk_weights_buffer() const {
    const auto input_topk_idx_buffer = get_input_topk_idx_buffer();
    return layout::Buffer(
        layout::Data(num_topk * sizeof(float), false), 1, num_max_tokens_per_rank, input_topk_idx_buffer.get_end_ptr());
  }

  __host__ __device__ void* get_scratch_base_ptr() const {
    return get_input_topk_weights_buffer().get_end_ptr();
  }

  __host__ __device__ layout::Buffer get_decode_l1_token_buffer() const {
    return layout::Buffer(layout::Data(hidden), 1, get_decode_num_max_pool_tokens(), get_scratch_base_ptr());
  }

  __host__ __device__ layout::Buffer get_decode_l1_sf_buffer() const {
    const auto l1_token_buffer = get_decode_l1_token_buffer();
    return layout::Buffer(
        layout::Data(hidden / 32, false), 1, get_decode_num_padded_sf_pool_tokens(), l1_token_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_decode_l1_topk_weights_buffer() const {
    const auto l1_sf_buffer = get_decode_l1_sf_buffer();
    return layout::Buffer(
        layout::Data(sizeof(float), false), 1, get_decode_num_max_pool_tokens(), l1_sf_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_decode_l2_token_buffer() const {
    const auto l1_topk_weights_buffer = get_decode_l1_topk_weights_buffer();
    return layout::Buffer(
        layout::Data(intermediate_hidden), 1, get_decode_num_max_pool_tokens(), l1_topk_weights_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_decode_l2_sf_buffer() const {
    const auto l2_token_buffer = get_decode_l2_token_buffer();
    return layout::Buffer(layout::Data(intermediate_hidden / 32),
                          1,
                          get_decode_num_padded_sf_pool_tokens(),
                          l2_token_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_decode_combine_token_buffer() const {
    const auto l2_sf_buffer = get_decode_l2_sf_buffer();
    return layout::Buffer(
        layout::Data(hidden * sizeof(uint16_t)), num_topk, kDecodeMaxTokensPerRank, l2_sf_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_prefill_l1_token_buffer() const {
    return layout::Buffer(layout::Data(hidden), 1, get_prefill_num_l1_ring_tokens(), get_scratch_base_ptr());
  }

  __host__ __device__ layout::Buffer get_prefill_l1_sf_buffer() const {
    const auto l1_token_buffer = get_prefill_l1_token_buffer();
    return layout::Buffer(
        layout::Data(hidden / 32, false), 1, get_prefill_num_padded_sf_l1_ring_tokens(), l1_token_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_prefill_l2_token_buffer() const {
    const auto l1_sf_buffer = get_prefill_l1_sf_buffer();
    return layout::Buffer(
        layout::Data(intermediate_hidden), 1, get_prefill_num_max_pool_tokens(), l1_sf_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_prefill_l2_sf_buffer() const {
    const auto l2_token_buffer = get_prefill_l2_token_buffer();
    return layout::Buffer(layout::Data(intermediate_hidden / 32),
                          1,
                          get_prefill_num_padded_sf_pool_tokens(),
                          l2_token_buffer.get_end_ptr());
  }

  __host__ __device__ layout::Buffer get_prefill_combine_token_buffer() const {
    const auto l2_sf_buffer = get_prefill_l2_sf_buffer();
    return layout::Buffer(
        layout::Data(hidden * sizeof(uint16_t)), num_topk, num_max_tokens_per_rank, l2_sf_buffer.get_end_ptr());
  }

  __host__ __device__ void* get_end_ptr() const {
    void* decode_scratch_end  = get_decode_combine_token_buffer().get_end_ptr();
    void* prefill_scratch_end = get_prefill_combine_token_buffer().get_end_ptr();
    return reinterpret_cast<uint64_t>(decode_scratch_end) > reinterpret_cast<uint64_t>(prefill_scratch_end)
               ? decode_scratch_end
               : prefill_scratch_end;
  }

  __host__ __device__ uint64_t get_num_bytes() const {
    return reinterpret_cast<uint64_t>(get_end_ptr()) - reinterpret_cast<uint64_t>(base);
  }
};

struct SymmBufferLayoutInfo {
  uint64_t num_bytes;
  uint64_t decode_workspace_num_bytes;
  uint64_t prefill_workspace_num_bytes;
  uint32_t decode_num_max_pool_tokens;
  uint32_t decode_num_padded_sf_pool_tokens;
  uint32_t prefill_num_max_pool_tokens;
  uint32_t prefill_num_padded_sf_pool_tokens;
  uint32_t prefill_l1_ring_tokens;
  uint32_t prefill_l1_ring_padded_sf_tokens;
  uint64_t input_token_offset;
  uint64_t input_sf_offset;
  uint64_t input_topk_idx_offset;
  uint64_t input_topk_weights_offset;
  uint64_t decode_l1_token_offset;
  uint64_t decode_l1_sf_offset;
  uint64_t decode_l1_topk_weights_offset;
  uint64_t decode_l2_token_offset;
  uint64_t decode_l2_sf_offset;
  uint64_t decode_combine_token_offset;
  uint64_t prefill_l1_token_offset;
  uint64_t prefill_l1_sf_offset;
  uint64_t prefill_l2_token_offset;
  uint64_t prefill_l2_sf_offset;
  uint64_t prefill_combine_token_offset;
};

__host__ inline SymmBufferLayoutInfo get_symm_buffer_layout_info(const uint32_t num_ranks,
                                                                 const uint32_t num_experts,
                                                                 const uint32_t num_max_tokens_per_rank,
                                                                 const uint32_t num_topk,
                                                                 const uint32_t hidden,
                                                                 const uint32_t intermediate_hidden,
                                                                 const uint32_t prefill_pool_capacity_rows = 0,
                                                                 const uint32_t prefill_l1_ring_tokens     = 0) {
  constexpr uint64_t kSyntheticBase = 1ull << 20;
  auto*              base           = reinterpret_cast<void*>(kSyntheticBase);
  const auto         offset_of      = [](const void* ptr) { return reinterpret_cast<uint64_t>(ptr) - kSyntheticBase; };
  const auto         layout         = BufferLayout(base,
                                   num_ranks,
                                   num_experts,
                                   num_max_tokens_per_rank,
                                   num_topk,
                                   hidden,
                                   intermediate_hidden,
                                   prefill_pool_capacity_rows,
                                   prefill_l1_ring_tokens);

  return {
      layout.get_num_bytes(),
      layout.get_decode_workspace().get_num_bytes(),
      layout.get_prefill_workspace().get_num_bytes(),
      layout.get_decode_num_max_pool_tokens(),
      layout.get_decode_num_padded_sf_pool_tokens(),
      layout.get_prefill_num_max_pool_tokens(),
      layout.get_prefill_num_padded_sf_pool_tokens(),
      layout.get_prefill_num_l1_ring_tokens(),
      layout.get_prefill_num_padded_sf_l1_ring_tokens(),
      offset_of(layout.get_input_token_buffer().get_base_ptr()),
      offset_of(layout.get_input_sf_buffer().get_base_ptr()),
      offset_of(layout.get_input_topk_idx_buffer().get_base_ptr()),
      offset_of(layout.get_input_topk_weights_buffer().get_base_ptr()),
      offset_of(layout.get_decode_l1_token_buffer().get_base_ptr()),
      offset_of(layout.get_decode_l1_sf_buffer().get_base_ptr()),
      offset_of(layout.get_decode_l1_topk_weights_buffer().get_base_ptr()),
      offset_of(layout.get_decode_l2_token_buffer().get_base_ptr()),
      offset_of(layout.get_decode_l2_sf_buffer().get_base_ptr()),
      offset_of(layout.get_decode_combine_token_buffer().get_base_ptr()),
      offset_of(layout.get_prefill_l1_token_buffer().get_base_ptr()),
      offset_of(layout.get_prefill_l1_sf_buffer().get_base_ptr()),
      offset_of(layout.get_prefill_l2_token_buffer().get_base_ptr()),
      offset_of(layout.get_prefill_l2_sf_buffer().get_base_ptr()),
      offset_of(layout.get_prefill_combine_token_buffer().get_base_ptr()),
  };
}

}  // namespace mega_moe::unified
