#pragma once

#include <cstdint>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/chunk_kda_checkpoint.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// Recurrence work item: the shared work descriptor plus the row base of this
// sequence's checkpoint entries.
//
// `checkpoint_base` belongs here because the compact layout's base is a prefix
// sum over the *preceding* sequences' `ceil(len / EveryNTokens)`, and the work
// decode is the one point that sees all of them.  The store path runs once per
// cadence boundary, so recomputing the sum there would re-walk `cu_seqlens` in
// device memory for every stored row; filling it here makes that walk once per
// CTA.  It stays 0 for the packed chunk-slot layout.
template <class Components>
struct ChunkKdaRecurrenceWorkDesc : ChunkKdaWorkDesc<Components::IsVarlen,
                                                     typename Components::LogicalShape,
                                                     std::conditional_t<Components::IsVarlen, int64_t, int32_t>> {
  using Base      = ChunkKdaWorkDesc<Components::IsVarlen,
                                     typename Components::LogicalShape,
                                     std::conditional_t<Components::IsVarlen, int64_t, int32_t>>;
  using IndexType = typename Base::IndexType;

  IndexType checkpoint_base = 0;

  ChunkKdaRecurrenceWorkDesc() = default;
  MUTLASS_HOST_DEVICE ChunkKdaRecurrenceWorkDesc(Base const& base, IndexType checkpoint_base_)
      : Base(base), checkpoint_base(checkpoint_base_) {
  }
};

// A CTA owns one complete recurrence chain. Value slices vary fastest, then
// heads, then sequences; there is no scheduler state or replicated work.
template <class Components>
MUTLASS_HOST_DEVICE ChunkKdaRecurrenceWorkDesc<Components> make_recurrence_work(
    typename Components::ProblemShape const& problem_size, uint32_t block_idx) {
  using WorkDesc           = ChunkKdaRecurrenceWorkDesc<Components>;
  using Base               = typename WorkDesc::Base;
  using IndexType          = typename WorkDesc::IndexType;
  constexpr int kValueTile = Components::kRecurrenceValueTile;
  Base const    base =
      decode_recurrence_work<Components::IsVarlen, typename Components::LogicalShape, IndexType, kValueTile>(
          problem_size, block_idx);
  if constexpr (Components::HasCheckpointCadence) {
    // `is_valid()` is device-only, so spell the two conditions out to keep this
    // host/device helper callable from either side.
    if (base.seq_idx >= 0 && base.head_idx >= 0) {
      return WorkDesc{
          base,
          static_cast<IndexType>(
              chunk_kda_checkpoint_base<Components::IsVarlen, Components::CheckpointEveryNTokens>(problem_size, base))};
    }
  }
  return WorkDesc{base, 0};
}

}  // namespace mate::flat::kda
