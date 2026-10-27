#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/kernel/chunk_kda_recurrence_work.hpp"
#include "mate/flat/simd_helper.hpp"

namespace mate::flat::kda {

template <class CollectiveRecurrence_, class TileScheduler_>
struct ChunkKdaRecurrenceKernel {
  using CollectiveRecurrence = CollectiveRecurrence_;
  using TileScheduler        = TileScheduler_;
  using Components           = typename CollectiveRecurrence::Components;
  using ArchTag              = typename CollectiveRecurrence::ArchTag;
  using CuSeqlensElement     = typename TileScheduler::CuSeqlensElement;
  using ProblemSize          = typename Components::ProblemShape;
  using SharedStorage        = typename CollectiveRecurrence::SharedStorage;

  static_assert(std::is_same_v<ProblemSize, typename TileScheduler::ProblemShape>);

  static constexpr int SharedStorageSize  = CollectiveRecurrence::SharedStorageSize;
  static constexpr int SmemAlignmentBytes = CollectiveRecurrence::SmemAlignmentBytes;
  // Occupancy comes from the schedule policy (RecurrenceCtasPerMp) so the kernel
  // header does not keep its own copy of the number.
  static constexpr uint32_t MinBlocksPerMultiprocessor =
      CollectiveRecurrence::Components::SchedulePolicy::RecurrenceCtasPerMp;
  static constexpr int MaxThreadsPerBlock = CollectiveRecurrence::NumProducerThreads +
                                            CollectiveRecurrence::NumStateThreads +
                                            CollectiveRecurrence::NumOutputThreads;

  struct Arguments {
    ProblemSize                              problem_size;
    typename CollectiveRecurrence::Arguments recurrence;
  };

  struct Params {
    ProblemSize                           problem_size;
    typename CollectiveRecurrence::Params recurrence;
  };

  template <class Problem, class WorkDesc>
  MUTLASS_DEVICE static void copy_empty_state(typename CollectiveRecurrence::Params const& params,
                                              Problem const&                               problem,
                                              WorkDesc const&                              work,
                                              int                                          local_tid) {
    if constexpr (Components::HasStateOut) {
      if constexpr (Components::HasStateIn) {
        if (params.ptr_initial_state == params.ptr_final_state) {
          return;
        }
      }
      int state_idx = params.ptr_state_indices[work.seq_idx];
      if (state_idx < 0) {
        return;
      }
      using StateElement          = typename CollectiveRecurrence::StateElement;
      constexpr int VecSize       = 8;
      using StateVec              = mutlass::Array<StateElement, VecSize>;
      constexpr int StateElements = Components::kRecurrenceValueTile * Components::kHeadDim;
      static_assert(StateElements % VecSize == 0);
      int64_t offset = (int64_t(state_idx) * problem.H + work.head_idx) * Components::kValueDim * Components::kHeadDim +
                       int64_t(work.value_start) * Components::kHeadDim;
      auto dst = reinterpret_cast<StateVec*>(static_cast<StateElement*>(params.ptr_final_state) + offset);
      // S_final = S_initial for an empty sequence. Keep the original state dtype
      // instead of rounding FP32 through the activation-typed shared tile.
      for (int i = local_tid; i < StateElements / VecSize; i += CollectiveRecurrence::NumProducerThreads) {
        StateVec value{};
        if constexpr (Components::HasStateIn) {
          auto src =
              reinterpret_cast<StateVec const*>(static_cast<StateElement const*>(params.ptr_initial_state) + offset);
          value = src[i];
        }
        dst[i] = value;
      }
    }
  }

  template <class Work, class Problem>
  MUTLASS_DEVICE static void run_role(SharedStorage&                               shared,
                                      typename CollectiveRecurrence::Params const& params,
                                      Problem const&                               problem) {
    auto work = make_recurrence_work<Components>(problem, uint32_t(blockIdx.x));
    // Every role skips the empty work's Seeded/Drained handshake together.
    if (work.n_chunks == 0) {
      if (int(threadIdx.x) < CollectiveRecurrence::NumProducerThreads) {
        copy_empty_state(params, problem, work, int(threadIdx.x));
      }
      return;
    }
    int sequence = 0;
    Work{}(shared, params, problem, work, sequence);
  }

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{
        args.problem_size,
        CollectiveRecurrence::to_underlying_arguments(args.problem_size, args.recurrence),
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.problem_size);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    CollectiveRecurrence::init_barriers(int(threadIdx.x));
    CollectiveRecurrence::init_shared_storage(shared_storage, int(threadIdx.x));
    __syncthreads();

    using WarpSpecRole   = typename CollectiveRecurrence::WarpSpecRole;
    using OutputConsumer = typename CollectiveRecurrence::OutputConsumer;
    using StateConsumer  = typename CollectiveRecurrence::StateConsumer;
    using ProducerWarp0  = typename CollectiveRecurrence::ProducerWarp0;
    using ProducerWarp1  = typename CollectiveRecurrence::ProducerWarp1;
    using ProducerWarp2  = typename CollectiveRecurrence::ProducerWarp2;
    using ProducerWarp3  = typename CollectiveRecurrence::ProducerWarp3;

    int warp_squad_idx = mutlass::canonical_warp_squad_idx();
    int warp_idx       = mutlass::canonical_warp_idx();
    if (warp_squad_idx == WarpSpecRole::OutputSquad) {
      run_role<OutputConsumer>(shared_storage, params.recurrence, params.problem_size);
      return;
    }
    if (warp_squad_idx == WarpSpecRole::StateSquad) {
      run_role<StateConsumer>(shared_storage, params.recurrence, params.problem_size);
      return;
    }
    if (warp_idx == WarpSpecRole::DecayedWarp) {
      run_role<ProducerWarp0>(shared_storage, params.recurrence, params.problem_size);
    } else if (warp_idx == WarpSpecRole::RestoredWarp) {
      run_role<ProducerWarp1>(shared_storage, params.recurrence, params.problem_size);
    } else if (warp_idx == WarpSpecRole::InverseWarp) {
      run_role<ProducerWarp2>(shared_storage, params.recurrence, params.problem_size);
    } else {
      run_role<ProducerWarp3>(shared_storage, params.recurrence, params.problem_size);
    }
  }
};

}  // namespace mate::flat::kda
