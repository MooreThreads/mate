#pragma once

#include <mute/tensor.hpp>
#include <mutlass/kernel_hardware_info.hpp>
#include <mutlass/pipeline/pipeline.hpp>

#include "named_barrier.hpp"
#include "pipeline_ws.hpp"

namespace mate::attention::fmha {

using namespace mute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
struct FmhaFwdKernelWarpSpecialized {
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using TileScheduler      = TileScheduler_;

  static constexpr bool IsAppendKV = CollectiveMainloop::IsAppendKV;

  static constexpr int NumLoadWarpSquads = CollectiveMainloop::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = CollectiveMainloop::NumMmaWarpSquads;

  static constexpr int NumQKMmaWarps = CollectiveMainloop::NumQKConsumers * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumPVMmaWarps = CollectiveMainloop::NumPVConsumers * mutlass::NumWarpsPerWarpSquad;

  static constexpr int      SmemAlignmentBytes         = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad;

  using SharedStorage                    = typename CollectiveMainloop::SharedStorage;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static constexpr bool UseLSULoadQ        = CollectiveMainloop::UseLSULoadQ;
  static constexpr bool SingleProducerWarp = CollectiveMainloop::NumProducerThreads == mutlass::NumThreadsPerWarp;

  static constexpr int NumLoadWarps = !SingleProducerWarp ? NumLoadWarpSquads * mutlass::NumWarpsPerWarpSquad : 1;

  static constexpr bool UseLSULoadK = CollectiveMainloop::UseLSULoadK;
  static constexpr bool UseLSULoadV = CollectiveMainloop::UseLSULoadV;

  using MainloopPipelineQ           = typename CollectiveMainloop::MainloopPipelineQ;
  using MainloopPipelineK           = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV           = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineKVNew       = typename CollectiveMainloop::MainloopPipelineKVNew;
  using MainloopPipelineQState      = typename CollectiveMainloop::PipelineQState;
  using MainloopPipelineKState      = typename CollectiveMainloop::PipelineKState;
  using MainloopPipelineVState      = typename CollectiveMainloop::PipelineVState;
  using MainloopPipelineKVNewState  = typename CollectiveMainloop::PipelineKVNewState;
  using MainloopPipelineQParams     = typename MainloopPipelineQ::Params;
  using MainloopPipelineKParams     = typename MainloopPipelineK::Params;
  using MainloopPipelineVParams     = typename MainloopPipelineV::Params;
  using MainloopPipelineKVNewParams = typename MainloopPipelineKVNew::Params;

  static constexpr int StagesQ = CollectiveMainloop::StagesQ;
  static constexpr int StagesK = CollectiveMainloop::StagesK;
  static constexpr int StagesV = CollectiveMainloop::StagesV;

  using SeqlenInfo = typename CollectiveMainloop::SeqlenInfo;

  using NamedBarrier = mutlass::arch::AsyncBarrier;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t NamedBarriers[static_cast<int32_t>(FwdNamedBarriers::NumFwdNamedBarriers)];
    uint8_t PipelineQ[MainloopPipelineQ::NumBarriers];
    uint8_t PipelineK[MainloopPipelineK::NumBarriers];
    uint8_t PipelineV[MainloopPipelineV::NumBarriers];
    uint8_t PipelineKNew[MainloopPipelineKVNew::NumBarriers];
    uint8_t PipelineVNew[MainloopPipelineKVNew::NumBarriers];
    uint8_t ConsumerSync[1];
  };

  struct Arguments {
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    typename TileScheduler::Arguments      scheduler;
    mutlass::KernelHardwareInfo            hw_info{};
  };

  struct Params {
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params      scheduler;
    mutlass::KernelHardwareInfo         hw_info{};
  };

  static Params to_underlying_arguments(const Arguments& args) {
    MUTLASS_TRACE_HOST("to_underlying_arguments():");

    int mp_count = args.hw_info.sm_count;
    if (mp_count <= 0) {
      MUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid MP count.\n"
          " For optimal performance, popluate the arguments KernelHardwareInfo struct with the MP count.");
      mp_count = mutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    mutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, mp_count};

    return {CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            TileScheduler::to_underlying_arguments(args.scheduler),
            hw_info};
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.mp_count);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  MUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    // if (thread0()) {
    //   printf("hello world JIT attention\n");
    // }

    int            thread_idx               = threadIdx.x;
    int            warp_idx                 = mutlass::canonical_warp_idx();
    int            lane_idx                 = thread_idx % mutlass::NumThreadsPerWarp;
    int            warp_squad_idx           = mutlass::canonical_warp_squad_idx();
    int            warp_idx_in_warp_squad   = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int            consumer_warp_squad_idx  = warp_squad_idx - 1;
    int            thread_idx_in_warp_squad = thread_idx % mutlass::NumThreadsPerWarpSquad;
    SharedStorage& shared_storage           = *reinterpret_cast<SharedStorage*>(smem_buf);

    static constexpr int MmaThreadOffset = NumLoadWarpSquads * mutlass::NumThreadsPerWarpSquad;

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    MainloopPipelineQParams pipeline_params_q;
    if constexpr (UseLSULoadQ) {
      pipeline_params_q.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_q.consumer_arv_count = NumQKMmaWarps;
    } else {
      pipeline_params_q.transaction_bytes = CollectiveMainloop::TmeTransactionBytesQ;
      pipeline_params_q.num_consumers     = NumQKMmaWarps;
    }
    MainloopPipelineQ      pipeline_q(pipeline_params_q, reinterpret_cast<uint64_t>(&barrier_storage->PipelineQ));
    MainloopPipelineQState mainloop_pipe_q_producer_state;
    // MainloopPipelineQState mainloop_pipe_q_producer_state = mutlass::make_producer_start_state<MainloopPipelineQ>();
    MainloopPipelineQState mainloop_pipe_q_consumer_state;

    MainloopPipelineKParams pipeline_params_k;
    if constexpr (UseLSULoadK) {
      pipeline_params_k.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_k.consumer_arv_count = NumQKMmaWarps;
    } else {
      pipeline_params_k.transaction_bytes = CollectiveMainloop::TmeTransactionBytesK;
      pipeline_params_k.num_consumers     = NumQKMmaWarps;
    }
    MainloopPipelineK      pipeline_k(pipeline_params_k, reinterpret_cast<uint64_t>(&barrier_storage->PipelineK));
    MainloopPipelineKState mainloop_pipe_k_producer_state;
    // MainloopPipelineKState mainloop_pipe_k_producer_state = mutlass::make_producer_start_state<MainloopPipelineK>();
    MainloopPipelineKState mainloop_pipe_k_consumer_state;

    MainloopPipelineVParams pipeline_params_v;
    if constexpr (UseLSULoadV) {
      pipeline_params_v.producer_arv_count = CollectiveMainloop::NumProducerThreads / mutlass::NumThreadsPerWarp;
      pipeline_params_v.consumer_arv_count = NumPVMmaWarps;
    } else {
      pipeline_params_v.transaction_bytes = CollectiveMainloop::TmeTransactionBytesV;
      pipeline_params_v.num_consumers     = NumPVMmaWarps;
    }
    MainloopPipelineV      pipeline_v(pipeline_params_v, reinterpret_cast<uint64_t>(&barrier_storage->PipelineV));
    MainloopPipelineVState mainloop_pipe_v_producer_state;
    // MainloopPipelineVState mainloop_pipe_v_producer_state = mutlass::make_producer_start_state<MainloopPipelineV>();
    MainloopPipelineVState mainloop_pipe_v_consumer_state;

    MainloopPipelineKVNewParams pipeline_params_k_new;
    pipeline_params_k_new.transaction_bytes = CollectiveMainloop::TmeTransactionBytesK;
    pipeline_params_k_new.num_consumers     = NumQKMmaWarps;
    MainloopPipelineKVNew      pipeline_k_new(pipeline_params_k_new,
                                         reinterpret_cast<uint64_t>(&barrier_storage->PipelineKNew));
    MainloopPipelineKVNewState mainloop_pipe_write_new = mutlass::make_producer_start_state<MainloopPipelineKVNew>();
    MainloopPipelineKVNewState mainloop_pipe_read_new;

    MainloopPipelineKVNewParams pipeline_params_v_new;
    pipeline_params_v_new.transaction_bytes = CollectiveMainloop::TmeTransactionBytesV;
    pipeline_params_v_new.num_consumers     = NumPVMmaWarps;
    MainloopPipelineKVNew pipeline_v_new(pipeline_params_v_new,
                                         reinterpret_cast<uint64_t>(&barrier_storage->PipelineVNew));

    NamedBarrier consumer_sync(reinterpret_cast<uint64_t>(&barrier_storage->ConsumerSync));
    NamedBarrier appendkv(
        reinterpret_cast<uint64_t>(&barrier_storage->NamedBarriers[static_cast<int32_t>(FwdNamedBarriers::AppendKV)]));
    if (warp_idx == 0) {
      consumer_sync.init(NumQKMmaWarps);            // QK Warps == PV Warps
      appendkv.init(NumLoadWarps + NumQKMmaWarps);  // QK Warps == PV Warps
    }

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;
    TileScheduler      scheduler;

    __syncthreads();

    int work_idx = 0;
    if (warp_squad_idx == 0) {
      // Producer
      if constexpr (SingleProducerWarp) {
        if (warp_idx_in_warp_squad != 0) {
          return;
        }
      }

      MUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile_info = scheduler.get_initial_work(params.scheduler);
           work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto [block_idx, head_idx, batch_idx, split_idx] = work_tile_info.get_block_coord(params.scheduler);

        int num_splits = 1;  // Number of splits for the batch of current work_tile.

        if constexpr (scheduler.HasMetadata) {
          num_splits       = params.scheduler.num_splits_dynamic_ptr[batch_idx];
          int num_m_blocks = params.scheduler.num_m_blocks_ptr[batch_idx];
          if (split_idx >= num_splits || block_idx >= num_m_blocks) {
            continue;
          }
          batch_idx = params.scheduler.batch_table_ptr[batch_idx];
        }

        auto block_coord = make_shape(block_idx, head_idx, batch_idx, split_idx);

        SeqlenInfo seqlen_info{
            static_cast<uint32_t>(get<2>(block_coord)) /* batch_idx */,
            static_cast<uint32_t>(get<0>(params.mainloop.shape_Q)) /* seqlen_q_static */,
            static_cast<uint32_t>(!CollectiveMainloop::IsPagedKV /* seqlen_k_static */
                                      ? size<0>(params.mainloop.shape_K)
                                      : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable)),
            static_cast<uint32_t>(get<0>(params.mainloop.shape_K_new)) /* shape_K_new_0 */,
            params.mainloop.cu_seqlens_q /* cu_seqlens_q */,
            params.mainloop.cu_seqlens_k /* cu_seqlens_k */,
            params.mainloop.cu_seqlens_k_new /* cu_seqlens_k_new */,
            params.mainloop.seqused_q /* seqused_q */,
            params.mainloop.seqused_k /* seqused_k */,
            // params.mainloop.ptr_leftpad_k /* ptr_leftpad_k */,
            // params.mainloop.seqlens_rotary /* seqlens_rotary */,
            params.mainloop.cp_world_size /* cp_world_size */,
            params.mainloop.cp_rank /* cp_rank */,
            params.mainloop.cp_tot_seqused_k /* cp_tot_seqused_k */};

        if constexpr (IsAppendKV) {
          bool is_valid_new = mainloop.load_kv_new(params.mainloop,
                                                   pipeline_k_new,
                                                   pipeline_v_new,
                                                   mainloop_pipe_write_new,
                                                   shared_storage,
                                                   seqlen_info,
                                                   block_coord,
                                                   warp_idx_in_warp_squad,
                                                   work_idx,
                                                   num_splits);
          if (is_valid_new) {
            named_barrier_sync(static_cast<uint32_t>(FwdNamedBarriers::AppendKV));
          }
        }

        mainloop.load(params.mainloop,
                      pipeline_q,
                      pipeline_k,
                      pipeline_v,
                      mainloop_pipe_q_producer_state,
                      mainloop_pipe_k_producer_state,
                      mainloop_pipe_v_producer_state,
                      shared_storage,
                      barrier_storage,
                      seqlen_info,
                      block_coord,
                      work_idx,
                      num_splits);
      }
    } else {
      // dummy
      auto state_q = mainloop_pipe_q_consumer_state;
      auto state_k = mainloop_pipe_k_consumer_state;
      auto state_v = mainloop_pipe_v_consumer_state;

      MUTE_UNROLL
      for (int i = 0; i < StagesQ; ++i) {
        pipeline_q.consumer_release(state_q);
        ++state_q;
      }
      MUTE_UNROLL
      for (int i = 0; i < StagesK; ++i) {
        pipeline_k.consumer_release(state_k);
        ++state_k;
      }
      MUTE_UNROLL
      for (int i = 0; i < StagesV; ++i) {
        pipeline_v.consumer_release(state_v);
        ++state_v;
      }

      // Consumer
      MUTLASS_PRAGMA_NO_UNROLL
      for (auto work_tile_info = scheduler.get_initial_work(params.scheduler);
           work_tile_info.is_valid(params.scheduler);
           work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto [block_idx, head_idx, batch_idx, split_idx] = work_tile_info.get_block_coord(params.scheduler);

        int num_splits = 1;  // Number of splits for the batch of current work_tile.

        if constexpr (scheduler.HasMetadata) {
          num_splits       = params.scheduler.num_splits_dynamic_ptr[batch_idx];
          int num_m_blocks = params.scheduler.num_m_blocks_ptr[batch_idx];
          if (split_idx >= num_splits || block_idx >= num_m_blocks) {
            continue;
          }
          batch_idx = params.scheduler.batch_table_ptr[batch_idx];
        }
        auto block_coord = make_shape(block_idx, head_idx, batch_idx, split_idx);

        SeqlenInfo seqlen_info{
            static_cast<uint32_t>(get<2>(block_coord)) /* batch_idx */,
            static_cast<uint32_t>(get<0>(params.mainloop.shape_Q)) /* seqlen_q_static */,
            static_cast<uint32_t>(!CollectiveMainloop::IsPagedKV /* seqlen_k_static */
                                      ? size<0>(params.mainloop.shape_K)
                                      : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable)),
            static_cast<uint32_t>(get<0>(params.mainloop.shape_K_new)) /* shape_K_new_0 */,
            params.mainloop.cu_seqlens_q /* cu_seqlens_q */,
            params.mainloop.cu_seqlens_k /* cu_seqlens_k */,
            params.mainloop.cu_seqlens_k_new /* cu_seqlens_k_new */,
            params.mainloop.seqused_q /* seqused_q */,
            params.mainloop.seqused_k /* seqused_k */,
            // params.mainloop.ptr_leftpad_k /* ptr_leftpad_k */,
            // params.mainloop.seqlens_rotary /* seqlens_rotary */,
            params.mainloop.cp_world_size /* cp_world_size */,
            params.mainloop.cp_rank /* cp_rank */,
            params.mainloop.cp_tot_seqused_k /* cp_tot_seqused_k */};

        if constexpr (IsAppendKV) {
          bool is_valid_new = mainloop.store_kv_new(params.mainloop,
                                                    pipeline_k_new,
                                                    pipeline_v_new,
                                                    mainloop_pipe_read_new,
                                                    threadIdx.x - MmaThreadOffset,
                                                    shared_storage,
                                                    seqlen_info,
                                                    block_coord,
                                                    num_splits);
          if (is_valid_new) {
            named_barrier_arrive(static_cast<uint32_t>(FwdNamedBarriers::AppendKV));
          }
        }

        auto results = mainloop.mma(params.mainloop,
                                    pipeline_q,
                                    pipeline_k,
                                    pipeline_v,
                                    mainloop_pipe_q_consumer_state,
                                    mainloop_pipe_k_consumer_state,
                                    mainloop_pipe_v_consumer_state,
                                    shared_storage,
                                    barrier_storage,
                                    seqlen_info,
                                    block_coord,
                                    threadIdx.x - MmaThreadOffset,
                                    work_idx,
                                    num_splits);

        // In situations where input tiles are valid but there is no valid input data (e.g., due to causal)
        // we directly write 0 to output and -inf to lse, as they are still needed in combine phase.
        auto  is_valid = get<0>(results);
        auto  values   = get<1>(results);
        auto& acc      = get<0>(values);
        auto& lse      = get<1>(values);
        auto  do_store = [&](auto store_zero, auto split_kv) {
          static constexpr bool StoreZero = decltype(store_zero)::value;
          static constexpr bool SplitKV   = decltype(split_kv)::value;
          epilogue.template store<StoreZero, SplitKV>(params.epilogue,
                                                      acc,
                                                      lse,
                                                      typename CollectiveMainloop::TiledMmaPV{},
                                                      threadIdx.x - MmaThreadOffset,
                                                      seqlen_info,
                                                      block_coord);
        };

        if (!is_valid) {
          if (num_splits > 1) {
            do_store(/* StoreZero */ mute::true_type{}, /* SplitKV */ mute::true_type{});
          } else {
            do_store(/* StoreZero */ mute::true_type{}, /* SplitKV */ mute::false_type{});
          }
        } else {
          if (num_splits > 1) {
            do_store(/* StoreZero */ mute::false_type{}, /* SplitKV */ mute::true_type{});
          } else {
            do_store(/* StoreZero */ mute::false_type{}, /* SplitKV */ mute::false_type{});
          }
        }
        consumer_sync.sync();
      }
    }
  }
};

}  // namespace mate::attention::fmha
