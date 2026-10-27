#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/chunk_kda_problem.hpp"
#include "mate/flat/kda/chunk_kda_workspace.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// Keep optional parameter payloads out of the ABI when a feature is disabled.
// The empty specialization is deliberately an empty base so the surrounding
// Params object retains the same offsets as the feature-off layout.
template <bool Enabled, class Payload>
struct FeatureGatedParams {};

template <class Payload>
struct FeatureGatedParams<true, Payload> : Payload {};

// Both architecture-specific recurrence collectives expose the same host-side
// argument contract.  Keep its field order in one place; the collective Params
// may still add architecture-specific descriptors after this contract.
template <class Element, class StrideV, class StrideO, class Workspace>
struct ChunkKdaRecurrenceArguments {
  Element const*                    ptr_v;
  void const*                       ptr_initial_state;
  Element*                          ptr_out;
  void*                             ptr_final_state;
  void*                             ptr_chunk_states;
  int32_t const*                    ptr_state_indices;
  StrideV                           stride_v;
  StrideO                           stride_out;
  typename Workspace::ConstPointers workspace;
  // Row count of `ptr_chunk_states` as the caller allocated it, or 0 when there
  // is no pool.  It is part of this contract rather than of the kernel shell
  // because it describes a collective operand: the TME descriptor's outer
  // extent.  `to_underlying_arguments` resolves 0 to the packed workspace
  // record count and stores that in the collective's Params.
  int32_t checkpoint_pool_rows;
};

// Architecture collectives provide their own SQMMA/TME layouts and pipelines,
// but K1 and K2 must agree on this problem/workspace/scheduling contract.
// Keeping the contract free of an ArchTag lets MP31 and MP32 builders reuse it
// without pretending that their collectives are interchangeable.
template <class Element_, class CuSeqlensElement_, class LogicalShape_, class SchedulePolicy_, class... Options_>
struct ChunkKdaPrefillComponents {
  using Element             = Element_;
  using CuSeqlensElement    = CuSeqlensElement_;
  using LogicalShape        = LogicalShape_;
  using SchedulePolicy      = SchedulePolicy_;
  using ProblemShape        = ChunkKdaProblemShape<CuSeqlensElement>;
  using PrepareProblemShape = ChunkKdaPrepareProblemShape<CuSeqlensElement>;

  static constexpr bool HasStateIn =
      mate::flat::find_option_t<mate::flat::Tag::HasStateIn, std::false_type, Options_...>::value;
  static constexpr bool HasStateOut =
      mate::flat::find_option_t<mate::flat::Tag::HasStateOut, std::true_type, Options_...>::value;
  static constexpr bool HasChunkStatesOut =
      mate::flat::find_option_t<mate::flat::Tag::HasChunkStatesOut, std::false_type, Options_...>::value;
  // Host-selected mode: the pool uses the compact per-sequence checkpoint rows
  // of `chunk_kda(checkpoint_every_n_tokens=...)` instead of packed chunk slots.
  // The kernel branches on this flag, never on the cadence value: a non-positive
  // cadence is rejected by `mate.jit.kda_ops` before anything is rendered.
  static constexpr bool HasCheckpointCadence =
      mate::flat::find_option_t<mate::flat::Tag::HasCheckpointCadence, std::false_type, Options_...>::value;
  // Only read when `HasCheckpointCadence`; this is the divisor of the row
  // arithmetic, and the assert below is the compile-time half of the contract
  // the host enforces at the API boundary.
  static constexpr int CheckpointEveryNTokens = mate::flat::
      find_option_t<mate::flat::Tag::CheckpointEveryNTokens, std::integral_constant<int, 0>, Options_...>::value;
  static constexpr bool HasGateParams =
      mate::flat::find_option_t<mate::flat::Tag::HasGateParams, std::true_type, Options_...>::value;
  static constexpr bool IsVarlen =
      mate::flat::find_option_t<mate::flat::Tag::IsVarlen, std::false_type, Options_...>::value;
  static constexpr bool NormalizeQK =
      mate::flat::find_option_t<mate::flat::Tag::NormalizeQK, std::true_type, Options_...>::value;

  static constexpr int kChunk               = LogicalShape::Chunk;
  static constexpr int kHeadDim             = LogicalShape::HeadDim;
  static constexpr int kValueDim            = LogicalShape::ValueDim;
  static constexpr int kRecurrenceValueTile = SchedulePolicy::RecurrenceValueTile;
  static constexpr int kCheckpointStride    = CheckpointEveryNTokens / kChunk;

  static_assert(std::is_integral_v<CuSeqlensElement>);
  static_assert(kValueDim % kRecurrenceValueTile == 0);
  static_assert(!HasCheckpointCadence || (CheckpointEveryNTokens > 0 && CheckpointEveryNTokens % kChunk == 0),
                "a checkpoint-cadence build needs a positive multiple of the kernel chunk");

  using Workspace = ChunkKdaWorkspace<Element, kChunk, kHeadDim>;
  using PrepareTileScheduler =
      ChunkKdaPrepareTileScheduler<LogicalShape, CuSeqlensElement, SchedulePolicy, Options_...>;
  using RecurrenceTileScheduler =
      ChunkKdaRecurrenceScheduler<LogicalShape, CuSeqlensElement, SchedulePolicy, Options_...>;

  MUTLASS_HOST_DEVICE static int workspace_records(ProblemShape const& problem_shape) {
    return Workspace::template record_count<IsVarlen>(problem_shape);
  }

  MUTLASS_HOST_DEVICE static int prepare_upper_bound_works(ProblemShape const& problem_shape) {
    return workspace_records(problem_shape) * problem_shape.H;
  }

  MUTLASS_HOST_DEVICE static int prepare_partition_count(ProblemShape const& problem_shape, int mp_count) {
    int target_ctas = (mp_count > 0 ? mp_count : 1) * SchedulePolicy::PrepareCtasPerMp;
    int total_works = prepare_upper_bound_works(problem_shape);
    int partitions  = total_works < target_ctas ? total_works : target_ctas;
    return partitions > 0 ? partitions : 1;
  }
};

}  // namespace mate::flat::kda
