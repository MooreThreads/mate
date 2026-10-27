#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/pipeline/pipeline.hpp>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/kda_arch_traits.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

template <class CollectivePrepare_, class TileScheduler_>
struct ChunkKdaPrepareKernel {
  using CollectivePrepare = CollectivePrepare_;
  using TileScheduler     = TileScheduler_;
  using Components        = typename CollectivePrepare::Components;
  using ArchTag           = typename CollectivePrepare::ArchTag;
  using CuSeqlensElement  = typename TileScheduler::CuSeqlensElement;
  using ProblemSize       = typename Components::PrepareProblemShape;
  using SharedStorage     = typename CollectivePrepare::SharedStorage;
  using BarrierStorage    = typename CollectivePrepare::BarrierStorage;
  using Pipeline          = typename CollectivePrepare::Pipeline;

  static_assert(std::is_same_v<ProblemSize, typename TileScheduler::ProblemShape>);

  static constexpr int SharedStorageSize  = CollectivePrepare::SharedStorageSize;
  static constexpr int SmemAlignmentBytes = CollectivePrepare::SmemAlignmentBytes;
  // Resident prepare CTAs per MP. The persistent K1 launch sizes its metadata
  // workspace at KDA_MP{31,32}_PREPARE_CTAS_PER_MP * num_mps and hands one range
  // descriptor to each resident prepare CTA, but declaring that occupancy here
  // is counter-productive: __launch_bounds__' second argument is a scheduler
  // target, not a cap, and naming a target makes the scheduler reshuffle into a
  // schedule that spills (asking MP32 for eight CTAs cost 16 bytes of spill).
  // Left unconstrained the kernel lands on the resident count anyway: 75
  // registers / 6 CTAs on MP31 (capped by shared memory) and 62 registers /
  // 8 CTAs on MP32, nothing spilled on either.
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int      MaxThreadsPerBlock         = CollectivePrepare::NumThreadsPerBlock;

  struct Arguments {
    ProblemSize                           problem_size;
    typename CollectivePrepare::Arguments prepare;
  };

  struct Params {
    ProblemSize                        problem_size;
    typename CollectivePrepare::Params prepare;
    typename TileScheduler::Params     scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{
        args.problem_size,
        CollectivePrepare::to_underlying_arguments(args.problem_size, args.prepare),
        TileScheduler::to_underlying_arguments(args.problem_size),
    };
  }

  template <auto Runner>
  MUTLASS_DEVICE static void run_role(SharedStorage& shared_storage,
                                      Params const&  params,
                                      TileScheduler& tile_scheduler,
                                      Pipeline&      pipeline,
                                      int            thread_idx) {
    using WorkDesc    = typename TileScheduler::WorkDesc;
    using ProblemSize = typename TileScheduler::ProblemShape;
    auto work_cursor  = tile_scheduler.get_initial_work(params.scheduler, params.problem_size);
    while (work_cursor.remaining > 0) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      auto work_desc = tile_scheduler.get_work(params.scheduler, params.problem_size, work_cursor);
      for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
        Runner(shared_storage, params.prepare, params.problem_size, work_desc, pipeline, thread_idx, chunk_idx);
      }
      tile_scheduler.advance(params.scheduler, params.problem_size, work_cursor);
    }
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    int thread_idx = int(threadIdx.x);
    int warp_idx   = mutlass::canonical_warp_idx();

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    TileScheduler tile_scheduler;

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);
    CollectivePrepare::init_stage_barriers(barrier_storage, int(threadIdx.x));
    Pipeline pipeline(barrier_storage);

    __syncthreads();

    // MP31 assigns the producer squad (Q, K, G, inverse) to warps 0-3 and the
    // SQMMA/epilogue squad to warps 4-7, matching the recurrence kernel's
    // `thread_idx - NumProducerThreads` convention.
    if (warp_idx >= CollectivePrepare::ConsumerWarpBase) {
      run_role<&CollectivePrepare::template consumer_loop<ProblemSize, typename TileScheduler::WorkDesc>>(
          shared_storage,
          params,
          tile_scheduler,
          pipeline,
          thread_idx - CollectivePrepare::ConsumerWarpBase * mutlass::NumThreadsPerWarp);
    } else if (warp_idx == CollectivePrepare::NormalizeQWarp) {
      run_role<&CollectivePrepare::template producer_normalize_q_loop<ProblemSize, typename TileScheduler::WorkDesc>>(
          shared_storage, params, tile_scheduler, pipeline, thread_idx);
    } else if (warp_idx == CollectivePrepare::NormalizeKWarp) {
      run_role<&CollectivePrepare::template producer_normalize_k_loop<ProblemSize, typename TileScheduler::WorkDesc>>(
          shared_storage, params, tile_scheduler, pipeline, thread_idx);
    } else if (warp_idx == CollectivePrepare::GateWarp) {
      run_role<&CollectivePrepare::template producer_gate_loop<ProblemSize, typename TileScheduler::WorkDesc>>(
          shared_storage, params, tile_scheduler, pipeline, thread_idx);
    } else {
      run_role<&CollectivePrepare::template producer_inverse_loop<ProblemSize, typename TileScheduler::WorkDesc>>(
          shared_storage, params, tile_scheduler, pipeline, thread_idx);
    }
  }
};

}  // namespace mate::flat::kda
