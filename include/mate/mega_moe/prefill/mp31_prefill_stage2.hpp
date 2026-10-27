#pragma once

#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>

#include <cstdint>
#include <mute/algorithm/clear.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/int_tuple.hpp>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/gemm/deep_gemm/scaling_accumulation.hpp"
#include "mate/mega_moe/mega_barrier.hpp"
#include "mate/mega_moe/mega_moe_sym_buffer.hpp"
#include "mate/mega_moe/mega_moe_unified_layout.hpp"
#include "mate/mega_moe/stage2_scheduler.hpp"
#include "mate/mega_moe/utils.hpp"

namespace mega_moe::prefill {

using PrefillStage2LsuStoreFragment = int32_t __attribute__((ext_vector_type(16)));

extern "C" __device__ void prefill_stage2_lsu_store_2d_i32_i16(
    PrefillStage2LsuStoreFragment, uint16_t*, int, int, int) __asm__("llvm.musa.lsu.st.2d.v16i32.i16");

template <uint32_t kNumExperts, uint32_t kNumRanks, uint32_t kNumMPs, uint32_t kNumThreads>
__device__ __forceinline__ void clear_prefill_split_workspace(const PrefillSplitWorkspace& workspace) {
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
  const uint32_t     global_thread_idx  = blockIdx.x * kNumThreads + threadIdx.x;
  constexpr uint32_t kNumGlobalThreads  = kNumMPs * kNumThreads;

  for (uint32_t expert_idx = global_thread_idx; expert_idx < kNumExperts; expert_idx += kNumGlobalThreads) {
    st_relaxed_sys_global(workspace.get_expert_send_count_ptr(expert_idx), uint64_t{0});
  }
  for (uint32_t slot_idx = global_thread_idx; slot_idx < kNumExperts; slot_idx += kNumGlobalThreads) {
    const uint32_t rank_idx         = slot_idx / kNumExpertsPerRank;
    const uint32_t local_expert_idx = slot_idx % kNumExpertsPerRank;
    st_relaxed_sys_global(workspace.get_expert_recv_count_ptr(rank_idx, local_expert_idx), uint64_t{0});
  }
  for (uint32_t local_expert_idx = global_thread_idx; local_expert_idx < kNumExpertsPerRank;
       local_expert_idx += kNumGlobalThreads) {
    st_relaxed_sys_global(workspace.get_expert_recv_count_sum_ptr(local_expert_idx), uint64_t{0});
  }
  for (uint32_t pool_block_idx = global_thread_idx; pool_block_idx < workspace.num_max_pool_blocks;
       pool_block_idx += kNumGlobalThreads) {
    st_relaxed_sys_global(workspace.get_l1_arrival_count_ptr(pool_block_idx), uint32_t{0});
    st_relaxed_sys_global(workspace.get_l2_arrival_mask_ptr(pool_block_idx), uint64_t{0});
    st_relaxed_sys_global(workspace.get_l1_consumed_mask_ptr(pool_block_idx), uint64_t{0});
  }
}

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
          uint32_t kNumStages        = 2,
          uint32_t kNumMmaWarpSquads = 4,
          uint32_t kNumThreads       = (1 + kNumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad>
__global__ __launch_bounds__(kNumThreads,
                             1) void mp31_prefill_stage2_impl(const layout::SymBuffer<kNumRanks> sym_buffer,
                                                              const float* __restrict__ l2_acts_sf,
                                                              const float* __restrict__ l2_weights_sf,
                                                              const MUtensorDescriptor tensor_map_l2_acts,
                                                              const MUtensorDescriptor tensor_map_l2_weights) {
  using namespace mute;

  constexpr uint32_t kBlockM            = 256;
  constexpr uint32_t kBlockN            = 256;
  constexpr uint32_t kBlockK            = 128;
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
  constexpr uint32_t kNumKBlocks        = kIntermediateHidden / kBlockK;
  constexpr uint32_t kNumNBlocks        = kHidden / kBlockN;
  constexpr int      kScaleMsPerTile    = kBlockM;
  constexpr int      kScaleNsPerTile    = kBlockN / 128;
  static_assert(kHidden % kBlockN == 0 && kIntermediateHidden % kBlockK == 0);
  static_assert(kNumKBlocks <= 64, "Stage 1 L2 arrival mask is uint64");

  using ElementA            = mutlass::float_e4m3_t;
  using ElementB            = mutlass::float_e4m3_t;
  using ElementD            = mutlass::bfloat16_t;
  using ElementAccumulator  = float;
  using TileShape           = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
  using StrideA             = decltype(make_stride(
      Int<kIntermediateHidden>{}, _1{}, C<static_cast<uint64_t>(kNumMaxPoolTokens) * kIntermediateHidden>{}));
  using StrideB             = decltype(make_stride(
      Int<kIntermediateHidden>{}, _1{}, C<static_cast<uint64_t>(kHidden) * kIntermediateHidden>{}));
  constexpr int kAlignmentA = 32 / sizeof_bits_v<ElementA>;
  constexpr int kAlignmentB = 32 / sizeof_bits_v<ElementB>;
  using CollectiveMma       = typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                                                    mutlass::arch::OpClassTensorOp,
                                                                                    ElementA,
                                                                                    StrideA,
                                                                                    kAlignmentA,
                                                                                    ElementB,
                                                                                    StrideB,
                                                                                    kAlignmentB,
                                                                                    ElementAccumulator,
                                                                                    TileShape,
                                                                                    Shape<_1, _1, _1>,
                                                                                    Int<kNumStages>,
                                                                                    mutlass::gemm::KernelTme>::CollectiveOp;
  using TiledMma            = typename CollectiveMma::TiledMma;
  using SmemLayoutA         = typename CollectiveMma::SmemLayoutA;
  using SmemLayoutB         = typename CollectiveMma::SmemLayoutB;

  constexpr int kWarpsPerWarpSquad = mutlass::NumThreadsPerWarpSquad / mutlass::NumThreadsPerWarp;
  constexpr int kNumMmaWarps       = kNumMmaWarpSquads * kWarpsPerWarpSquad;
  using PipelineAB                 = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<kNumStages, 2>;
  using PipelineABParams           = typename PipelineAB::Params;
  using PipelineABState            = typename PipelineAB::PipelineState;
  using PipelineSeq                = mutlass::OrderedSequenceBarrier<1, kNumMmaWarpSquads>;
  using PipelineSeqParams          = typename PipelineSeq::Params;
  using ConsumerBarrier            = mutlass::arch::AsyncBarrier;

  constexpr int kTmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  constexpr int kTmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t PipelineSeq[PipelineSeq::NumBarriers];
    uint8_t Consumer[2];
  };
  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, 256> smem_a;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, 256> smem_b;
  };

  extern __shared__ __align__(256) uint8_t smem_buffer[];
  auto&                                    shared          = *reinterpret_cast<SharedStorage*>(smem_buffer);
  auto*                                    barrier_storage = reinterpret_cast<BarrierStorage*>(0);
  mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount);

  const auto buffer_layout        = unified::BufferLayout(sym_buffer.get_base_ptr(),
                                                   kNumRanks,
                                                   kNumExperts,
                                                   kNumMaxTokensPerRank,
                                                   kNumTopk,
                                                   kHidden,
                                                   kIntermediateHidden,
                                                   kNumMaxPoolTokens,
                                                   kNumL1RingTokens);
  const auto workspace            = buffer_layout.get_prefill_workspace();
  const auto l2_token_buffer      = buffer_layout.get_prefill_l2_token_buffer();
  const auto l2_sf_buffer         = buffer_layout.get_prefill_l2_sf_buffer();
  const auto combine_token_buffer = buffer_layout.get_prefill_combine_token_buffer();

  PipelineABParams pipeline_params;
  pipeline_params.transaction_bytes = kTmeTransactionBytesA + kTmeTransactionBytesB;
  pipeline_params.num_consumers     = kNumMmaWarps;
  pipeline_params.num_producers     = 1;
  PipelineAB        pipeline(pipeline_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));
  const int         warp_idx          = mutlass::canonical_warp_idx_sync();
  const int         squad             = mutlass::canonical_warp_squad_idx();
  const int         warp_in_squad     = warp_idx & (kWarpsPerWarpSquad - 1);
  const bool        is_producer_squad = squad == 0;
  const bool        is_consumer_squad = squad >= 1 && squad < int(1 + kNumMmaWarpSquads);
  const bool        is_tme_issue_warp = is_producer_squad && warp_in_squad == 0;
  PipelineSeqParams seq_params{};
  seq_params.group_size = kWarpsPerWarpSquad;
  seq_params.group_id   = is_consumer_squad ? squad - 1 : 0;
  PipelineSeq    pipeline_seq(seq_params,
                           static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->PipelineSeq)));
  const uint32_t consumer_barrier_base = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Consumer)) +
                                         ConsumerBarrier::ReservedAsyncBarrierCount;
  if (threadIdx.x == 0) {
    ConsumerBarrier::init(consumer_barrier_base, kNumMmaWarps, 0);
    ConsumerBarrier::init(consumer_barrier_base + 1, kNumMmaWarps, 0);
  }
  __syncthreads();

  Tensor          sA         = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
  Tensor          sB         = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});
  PipelineABState pipe_write = mutlass::make_producer_start_state_warpspecialized<PipelineAB>();
  PipelineABState pipe_read;
  using Scheduler =
      mate_stage2::WorkspaceTileScheduler<kNumExpertsPerRank, kNumNBlocks, kNumMPs, kNumRanks, kBlockM, true>;
  using TmeLoad = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                         mute::TME::SmemSwizzleStride::B256,
                                         mute::TME::SmemSwizzleLine::B256,
                                         mute::TME::CacheHint::CACHE_NORMAL,
                                         mute::TME::CacheHint::CACHE_NONE,
                                         mute::PrefetchSize::B128>;

  if (is_tme_issue_warp) {
    Scheduler scheduler;
    auto      work = scheduler.initial_work_tile_info(workspace);
    while (work.valid) {
      for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
        const int stage = int(pipe_write.index());
        pipeline.producer_acquire(pipe_write);
        const uint32_t barrier_id = pipeline.producer_get_barrier_id(pipe_write);
        TmeLoad::copy(&tensor_map_l2_acts,
                      barrier_id,
                      raw_pointer_cast(sA(_, _, stage).data()),
                      static_cast<int32_t>(kb * kBlockK),
                      static_cast<int32_t>(work.pool_block_idx * kBlockM),
                      0,
                      static_cast<int32_t>(kBlockK),
                      static_cast<int32_t>(kBlockM),
                      1);
        TmeLoad::copy(&tensor_map_l2_weights,
                      barrier_id,
                      raw_pointer_cast(sB(_, _, stage).data()),
                      static_cast<int32_t>(kb * kBlockK),
                      static_cast<int32_t>(work.n_block_idx * kBlockN),
                      static_cast<int32_t>(work.expert_idx),
                      static_cast<int32_t>(kBlockK),
                      static_cast<int32_t>(kBlockN),
                      1);
        ++pipe_write;
      }
      scheduler.advance_to_next_work();
      work = scheduler.get_work_tile_info();
    }
  } else if (is_consumer_squad) {
    const int consumer_thread_idx       = int(threadIdx.x) - mutlass::NumThreadsPerWarpSquad;
    uint32_t  consumer_barrier_call_idx = 0;
    auto      consumer_sync             = [&]() {
      const uint32_t barrier_idx = consumer_barrier_base + (consumer_barrier_call_idx & 1u);
      const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
      ConsumerBarrier::wait(barrier_idx, phase);
      ++consumer_barrier_call_idx;
    };
    Scheduler scheduler;
    auto      work = scheduler.initial_work_tile_info(workspace);
    while (work.valid) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      TiledMma tiled_mma;
      auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);
      auto     rAcc    = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
      fill(rAcc, ElementAccumulator(0));
      Tensor tCsA = thr_mma.partition_A(sA);
      Tensor tCsB = thr_mma.partition_B(sB);
      Tensor tCrA = thr_mma.make_fragment_A(tCsA);
      Tensor tCrB = thr_mma.make_fragment_B(tCsB);

      Tensor mScaleA    = make_tensor(make_gmem_ptr(l2_acts_sf),
                                   make_shape(Int<kNumPaddedSFPoolTokens>{}, Int<kNumKBlocks>{}),
                                   make_stride(Int<kNumKBlocks>{}, _1{}));
      Tensor mScaleB    = make_tensor(make_gmem_ptr(l2_weights_sf),
                                   make_shape(Int<kHidden / 128>{}, Int<kNumKBlocks>{}, Int<kNumExpertsPerRank>{}),
                                   make_stride(Int<kNumKBlocks>{}, _1{}, Int<(kHidden / 128) * kNumKBlocks>{}));
      Tensor gScaleA    = local_tile(mScaleA, make_tile(Int<kScaleMsPerTile>{}), make_coord(work.pool_block_idx, _));
      Tensor gScaleB_nk = mScaleB(_, _, work.expert_idx);
      Tensor gScaleB    = local_tile(gScaleB_nk, make_tile(Int<kScaleNsPerTile>{}), make_coord(work.n_block_idx, _));
      using ScaleAViewAsCLayout =
          Layout<Shape<Shape<_1, Int<kScaleMsPerTile>>, Int<kBlockN>>, Stride<Stride<_0, _1>, _0>>;
      using ScaleBViewAsCLayout =
          Layout<Shape<Int<kBlockM>, Shape<Int<128>, Int<kScaleNsPerTile>>>, Stride<_0, Stride<_0, _1>>>;
      Tensor tCgScaleA0   = thr_mma.partition_C(gScaleA(_, 0).compose(ScaleAViewAsCLayout{}));
      Tensor tCgScaleB0   = thr_mma.partition_C(gScaleB(_, 0).compose(ScaleBViewAsCLayout{}));
      Tensor tCrScaleA    = make_tensor_like<float>(tCgScaleA0);
      Tensor tCrScaleB    = make_tensor_like<float>(tCgScaleB0);
      using ScaleCopyAtom = Copy_Atom<MP31_ROBUST_LOAD<float>, float>;
      auto scale_copy_a =
          ScaleCopyAtom{}.with(make_robust_desc(l2_acts_sf, static_cast<size_t>(cosize(mScaleA.layout()))));
      auto scale_copy_b =
          ScaleCopyAtom{}.with(make_robust_desc(l2_weights_sf, static_cast<size_t>(cosize(mScaleB.layout()))));
      using ScalingAcc = ::mate::deep_gemm::ScalingAccumulationIterative<typename decltype(rAcc)::engine_type,
                                                                         typename decltype(rAcc)::layout_type,
                                                                         typename decltype(tCrScaleA)::engine_type,
                                                                         typename decltype(tCrScaleA)::layout_type,
                                                                         typename decltype(tCrScaleB)::engine_type,
                                                                         typename decltype(tCrScaleB)::layout_type>;
      const uint32_t mma_per_iter = static_cast<uint32_t>(size<2>(tCrA));
      ScalingAcc     accumulation(rAcc, mma_per_iter, mma_per_iter, tCrScaleA, tCrScaleB);
      accumulation.initializeA(scale_copy_a, tCgScaleA0, tCrScaleA);
      accumulation.initializeB(scale_copy_b, tCgScaleB0, tCrScaleB);
      for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
        Tensor tCgScaleA = thr_mma.partition_C(gScaleA(_, kb).compose(ScaleAViewAsCLayout{}));
        Tensor tCgScaleB = thr_mma.partition_C(gScaleB(_, kb).compose(ScaleBViewAsCLayout{}));
        accumulation.copyA(scale_copy_a, tCgScaleA, tCrScaleA);
        accumulation.copyB(scale_copy_b, tCgScaleB, tCrScaleB);
        if (accumulation.prepare_if_needed()) {
          accumulation.update_iterationAB(tCrScaleA, tCrScaleB);
          accumulation.div_if_needAB();
        }
        pipeline.consumer_wait(pipe_read);
        pipeline_seq.wait();
        const int stage = int(pipe_read.index());
        MUTE_UNROLL
        for (int mma_k = 0; mma_k < size<2>(tCrA); ++mma_k) {
          gemm(tiled_mma, tCrA(_, _, mma_k, stage), tCrB(_, _, mma_k, stage), accumulation());
        }
        ::mate::warpsquad_commit_batch();
        pipeline_seq.arrive();
        accumulation.advance();
        ::mate::warpsquad_wait();
        pipeline.consumer_release(pipe_read);
        ++pipe_read;
      }
      accumulation.scale_residue_if_needed(tCrScaleA, tCrScaleB);

      auto          cD               = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
      auto          tCcD             = thr_mma.partition_C(cD);
      constexpr int kAccumPerThread  = 128;
      constexpr int kAccumColsPerRow = 16;
      constexpr int kAccumRowGroups  = kAccumPerThread / kAccumColsPerRow;
      constexpr int kAccumColStride  = 8;
      static_assert(size(rAcc) == kAccumPerThread, "Unexpected M256 C fragment layout");
      static_assert(sizeof(ElementD) == 2, "Stage 2 2D store expects 16-bit output");
      MUTE_UNROLL
      for (int row_group = 0; row_group < kAccumRowGroups; ++row_group) {
        const int      first_i     = row_group * kAccumColsPerRow;
        const auto     first_coord = tCcD(first_i);
        const uint32_t row         = static_cast<uint32_t>(get<0>(first_coord));
        if (row >= work.valid_m) {
          continue;
        }
        const uint32_t pool_token_idx = work.pool_block_idx * kBlockM + row;
        const auto     metadata       = *workspace.get_token_src_metadata_ptr(pool_token_idx);
        const auto     destination =
            combine_token_buffer.get_rank_buffer(metadata.topk_idx()).get_data_buffer(metadata.token_idx());
        auto* destination_ptr =
            sym_buffer.map(destination.get_base_ptr<ElementD>() + work.n_block_idx * kBlockN, metadata.rank_idx());
        // The MP31 CLayout is first_col + col_group * 8. One B512 2D
        // store covers all 16 accumulator columns in the row group.
        constexpr int kStoreCols = 16;
        static_assert(kAccumColsPerRow % kStoreCols == 0);
        const uint32_t first_col = static_cast<uint32_t>(get<1>(first_coord));
        MUTE_UNROLL
        for (int store_group = 0; store_group < kAccumColsPerRow / kStoreCols; ++store_group) {
          PrefillStage2LsuStoreFragment store_fragment;
          MUTE_UNROLL
          for (int store_col = 0; store_col < kStoreCols; ++store_col) {
            const int i               = first_i + store_group * kStoreCols + store_col;
            store_fragment[store_col] = static_cast<int32_t>(static_cast<ElementD>(rAcc(i)).storage);
          }
          prefill_stage2_lsu_store_2d_i32_i16(
              store_fragment,
              reinterpret_cast<uint16_t*>(destination_ptr + first_col + store_group * kStoreCols * kAccumColStride),
              sizeof(ElementD),
              kStoreCols,
              kAccumColStride * sizeof(ElementD));
        }
      }
      consumer_sync();
      scheduler.advance_to_next_work();
      work = scheduler.get_work_tile_info();
    }
  }

  __threadfence_system_noflush();
  mtlink_barrier<kNumRanks, kNumMPs, kNumThreads, 0, 1, true>(
      workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);
  clear_prefill_split_workspace<kNumExperts, kNumRanks, kNumMPs, kNumThreads>(workspace);
  __threadfence_system_noflush();
  mtlink_barrier<kNumRanks, kNumMPs, kNumThreads, 1, 2, true>(
      workspace, sym_buffer, blockIdx.x, threadIdx.x, [&]() { __syncthreads_lm(); }, true, true);
}

}  // namespace mega_moe::prefill
