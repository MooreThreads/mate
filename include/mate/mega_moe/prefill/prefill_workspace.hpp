#pragma once

#include <cstdint>

#include "mate/mega_moe/mega_moe_layout.hpp"

namespace mega_moe::prefill {

constexpr uint32_t kBlockM                 = 256;
constexpr uint32_t kBlockK                 = 128;
constexpr uint32_t kMaxIntermediateHidden  = 8192;
constexpr uint32_t kMaxL2ArrivalMaskBlocks = 64;

// Stage 2 uses this metadata to direct-store each expert partial back to the
// originating rank/token/top-k lane. The remaining upper bits are reserved.
struct PackedTokenSrcMetadata {
  uint64_t value;

  static constexpr uint32_t kRankShift   = 32;
  static constexpr uint32_t kTopkShift   = 40;
  static constexpr uint32_t kExpertShift = 48;
  static constexpr uint64_t kValidBit    = 1ull << 63;

  __host__ __device__ static constexpr PackedTokenSrcMetadata pack(uint32_t rank_idx,
                                                                   uint32_t token_idx,
                                                                   uint32_t topk_idx,
                                                                   uint32_t local_expert_idx = 0) {
    return {static_cast<uint64_t>(token_idx) | (static_cast<uint64_t>(rank_idx) << kRankShift) |
            (static_cast<uint64_t>(topk_idx) << kTopkShift) |
            (static_cast<uint64_t>(local_expert_idx) << kExpertShift) | kValidBit};
  }

  __host__ __device__ constexpr uint32_t rank_idx() const {
    return static_cast<uint32_t>((value >> kRankShift) & 0xffu);
  }

  __host__ __device__ constexpr uint32_t token_idx() const {
    return static_cast<uint32_t>(value);
  }

  __host__ __device__ constexpr uint32_t topk_idx() const {
    return static_cast<uint32_t>((value >> kTopkShift) & 0xffu);
  }

  __host__ __device__ constexpr uint32_t local_expert_idx() const {
    return static_cast<uint32_t>((value >> kExpertShift) & 0x3ffu);
  }

  __host__ __device__ constexpr bool valid() const {
    return (value & kValidBit) != 0;
  }
};
static_assert(sizeof(PackedTokenSrcMetadata) == sizeof(uint64_t));

// Preserve the base Workspace control region so the existing grid/MTLink
// barriers can accept this object through a Workspace reference. Replace only
// the decode-oriented, 64B-per-item metadata tail with compact prefill arrays.
struct PrefillSplitWorkspace : public layout::Workspace {
  __host__ __device__ PrefillSplitWorkspace(
      void* base, uint32_t num_ranks, uint32_t num_experts, uint32_t num_max_tokens_per_rank, uint32_t num_topk)
      : layout::Workspace(base, num_ranks, num_experts, num_max_tokens_per_rank, num_topk, kBlockM) {
  }

  __host__ __device__ uint64_t get_num_bytes() const {
    return math::align(get_token_src_metadata_offset() + get_num_token_src_metadata_bytes(),
                       layout::kBufferAlignmentBytes);
  }

  __host__ __device__ void* get_end_ptr() const {
    return math::advance_ptr(base, get_num_bytes());
  }

  __host__ __device__ uint32_t* get_src_token_topk_idx_ptr(uint32_t local_expert_idx = 0,
                                                           uint32_t source_rank_idx  = 0,
                                                           uint32_t source_token_idx = 0) const {
    const uint64_t index =
        (static_cast<uint64_t>(local_expert_idx) * num_ranks + source_rank_idx) * num_max_tokens_per_rank +
        source_token_idx;
    return math::advance_ptr<uint32_t>(base, get_src_token_topk_idx_offset() + index * sizeof(uint32_t));
  }

  __host__ __device__ PackedTokenSrcMetadata* get_token_src_metadata_ptr(uint32_t pool_token_idx = 0) const {
    return math::advance_ptr<PackedTokenSrcMetadata>(
        base, get_token_src_metadata_offset() + static_cast<uint64_t>(pool_token_idx) * sizeof(PackedTokenSrcMetadata));
  }

  // One bit per L1 N block of a logical pool block. The L1 consumer sets its
  // bit once the pool block's activation source and scales are fully read, so
  // the dispatcher can recycle the physical ring slot.
  __host__ __device__ uint64_t* get_l1_consumed_mask_ptr(uint32_t pool_block_idx = 0) const {
    return math::advance_ptr<uint64_t>(
        base, get_l1_consumed_mask_offset() + static_cast<uint64_t>(pool_block_idx) * kLogicBufferSlotBytes);
  }

  __host__ __device__ uint64_t get_control_region_num_bytes() const {
    return get_l1_consumed_mask_offset() + static_cast<uint64_t>(num_max_pool_blocks) * kLogicBufferSlotBytes;
  }

  __host__ __device__ uint64_t get_l1_consumed_mask_offset() const {
    uint64_t offset = kNumBarrierSignalBytes;
    offset += static_cast<uint64_t>(num_experts) * kLogicBufferSlotBytes * 2;
    offset += static_cast<uint64_t>(num_experts_per_rank) * kLogicBufferSlotBytes;
    offset += static_cast<uint64_t>(math::align(num_max_pool_blocks, 2u)) * kLogicBufferSlotBytes;
    offset += static_cast<uint64_t>(num_max_pool_blocks) * kLogicBufferSlotBytes;
    return offset;
  }

  __host__ __device__ uint64_t get_src_token_topk_idx_offset() const {
    return math::align(get_control_region_num_bytes(), layout::kBufferAlignmentBytes);
  }

  __host__ __device__ uint64_t get_token_src_metadata_offset() const {
    return math::align(get_src_token_topk_idx_offset() + get_num_src_token_topk_idx_bytes(),
                       layout::kBufferAlignmentBytes);
  }

 private:
  __host__ __device__ uint64_t get_num_src_token_topk_idx_bytes() const {
    return static_cast<uint64_t>(num_experts_per_rank) * num_ranks * num_max_tokens_per_rank * sizeof(uint32_t);
  }

  __host__ __device__ uint64_t get_num_token_src_metadata_bytes() const {
    return static_cast<uint64_t>(num_max_pool_tokens) * sizeof(PackedTokenSrcMetadata);
  }
};

}  // namespace mega_moe::prefill
