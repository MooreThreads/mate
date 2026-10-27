#pragma once

#include <type_traits>

#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_mp31_policy.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_recurrence_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/kernel/chunk_kda_recurrence_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// K2 (recurrence) builder.  This release line is MP31-only: the persistent
// scheduler assigns each CTA its recurrence work, and the problem/workspace
// contract stays shared.
template <class ArchTag_,
          class Element_,
          class StateElement_,
          class CuSeqlensElement_,
          class LogicalShape_,
          class TileShape_,
          class StrideV_,
          class StrideO_,
          class... Options_>
struct ChunkKdaRecurrenceBuilder {
  using ArchTag        = ArchTag_;
  using SchedulePolicy = ChunkKdaMp31SchedulePolicy;
  using Components = ChunkKdaPrefillComponents<Element_, CuSeqlensElement_, LogicalShape_, SchedulePolicy, Options_...>;
  using CollectiveRecurrence = recurrence::
      Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized<Components, TileShape_, StateElement_, StrideV_, StrideO_>;
  using TileScheduler = ChunkKdaRecurrenceScheduler<typename Components::LogicalShape,
                                                    typename Components::CuSeqlensElement,
                                                    typename Components::SchedulePolicy,
                                                    Options_...>;
  using Kernel        = ChunkKdaRecurrenceKernel<CollectiveRecurrence, TileScheduler>;
};

}  // namespace mate::flat::kda
