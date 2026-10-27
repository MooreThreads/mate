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
#include "mate/mega_moe/mp31_activation.hpp"
#include "mate/mega_moe/stage1_scheduler.hpp"
#include "mate/mega_moe/utils.hpp"

namespace mega_moe::prefill {

union MUTE_ALIGNAS(64) PrefillStage1B512 {
  int4 b128[4];
};

template <uint32_t kNumStages>
class PrefillStage1TmePipeline {
 public:
  static constexpr uint32_t Stages           = kNumStages;
  using Pipeline                             = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<kNumStages, 2>;
  static constexpr uint32_t BarPerStageRatio = Pipeline::BarPerStageRatio;
  using PipelineState                        = typename Pipeline::PipelineState;

  struct Storage {
    uint8_t barriers[Pipeline::NumBarriers];
  };

  __device__ PrefillStage1TmePipeline(Storage* storage, const uint32_t transaction_bytes, const uint32_t num_consumers)
      : pipeline_(make_params(transaction_bytes, num_consumers),
                  static_cast<uint32_t>(reinterpret_cast<uint64_t>(&storage->barriers))) {
  }

  __device__ __forceinline__ void producer_acquire(const PipelineState state) {
    pipeline_.producer_acquire(state);
  }

  __device__ __forceinline__ uint32_t producer_get_barrier_id(const PipelineState state) {
    return pipeline_.producer_get_barrier_id(state);
  }

  __device__ __forceinline__ void consumer_wait(const PipelineState state, const int) {
    pipeline_.consumer_wait(state);
  }

  __device__ __forceinline__ void consumer_release(const PipelineState state, const int) {
    pipeline_.consumer_release(state);
  }

 private:
  __device__ __forceinline__ static typename Pipeline::Params make_params(const uint32_t transaction_bytes,
                                                                          const uint32_t num_consumers) {
    typename Pipeline::Params params;
    params.transaction_bytes = transaction_bytes;
    params.num_consumers     = num_consumers;
    params.num_producers     = 1;
    return params;
  }

  Pipeline pipeline_;
};

template <uint32_t kNumMaxTokensPerRank,
          uint32_t kHidden,
          uint32_t kIntermediateHidden,
          uint32_t kNumExperts,
          uint32_t kNumTopk,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumMaxPoolTokens,
          uint32_t kNumL1RingBlocks,
          uint32_t kNumMPs,
          uint32_t kNumRanks,
          bool     kUseSitu,
          bool     kUseSituLinear,
          bool     kDispatchOnly      = false,
          bool     kComputeOnly       = false,
          uint32_t kNumOwnerCtas      = 4,
          uint32_t kNumStages         = 2,
          uint32_t kNumMmaWarpSquads  = 4,
          uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks,
          uint32_t kL1ShapeN          = kIntermediateHidden * 2,
          uint32_t kL1ShapeK          = kHidden,
          uint32_t kNumLoadWarpSquads = 1,
          uint32_t kNumThreads        = (kNumLoadWarpSquads + kNumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad>
__global__ __launch_bounds__(kNumThreads,
                             1) void mp31_prefill_stage1_impl(const uint32_t                     num_tokens,
                                                              const layout::SymBuffer<kNumRanks> sym_buffer,
                                                              const float* __restrict__ l1_weights_sf,
                                                              mutlass::float_e4m3_t*   l1_acts,
                                                              float*                   l1_acts_sf,
                                                              mutlass::float_e4m3_t*   l2_acts,
                                                              float*                   l2_acts_sf,
                                                              const MUtensorDescriptor tensor_map_l1_acts,
                                                              const MUtensorDescriptor tensor_map_l1_weights,
                                                              const float              activation_clamp,
                                                              const float              activation_beta,
                                                              const float              activation_linear_beta) {
  using namespace mute;

  static_assert(kBlockM == 256 && kBlockK == 128);
  static_assert(kNumStages == 2, "Prefill Stage 1 uses a 2-stage A/B pipeline");
  static_assert(kNumMmaWarpSquads == 4, "M256 uses four MMA warp squads");
  static_assert(kL1ShapeN % 256 == 0 && kL1ShapeK % kBlockK == 0);
  static_assert(kIntermediateHidden <= kMaxIntermediateHidden);
  static_assert(kNumOwnerCtas >= 1, "Owner CTA group must be positive");
  static_assert(kNumMPs % kNumOwnerCtas == 0, "Owner CTA group must divide the launched MP count");

  constexpr uint32_t kBlockN                    = 256;
  constexpr uint32_t kSwigluGran                = 64;
  constexpr uint32_t kSwigluPairCols            = 128;
  constexpr uint32_t kOutputColsPerTile         = kBlockN / 2;
  constexpr uint32_t kSwigluColumnGroupsPerTile = kBlockN / kSwigluPairCols;
  constexpr uint32_t kNumKBlocks                = kL1ShapeK / kBlockK;
  // Routed L1 activation data lives in a power-of-two ring; all routing and
  // arrival metadata stays indexed by the logical pool block.
  constexpr uint32_t kNumL1BlockNs         = kL1ShapeN / kBlockN;
  constexpr uint32_t kWorstCasePoolBlocks  = kNumMaxPoolTokens / kBlockM;
  constexpr bool     kL1RingEnabled        = kNumL1RingBlocks != 0 && kNumL1RingBlocks < kWorstCasePoolBlocks;
  constexpr uint32_t kL1RingBlocks         = kL1RingEnabled ? kNumL1RingBlocks : kWorstCasePoolBlocks;
  constexpr uint32_t kL1RingTokens         = kL1RingBlocks * kBlockM;
  constexpr uint32_t kL1RingPaddedSFTokens = kL1RingTokens;
  constexpr uint64_t kL1ConsumedFullMask   = kNumL1BlockNs == 64 ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
  static_assert(kNumL1BlockNs > 0 && kNumL1BlockNs <= 64, "L1 consumed mask is uint64");
  static_assert(kL1RingTokens <= kNumMaxPoolTokens, "L1 ring must not exceed the worst-case pool");
  if constexpr (kL1RingEnabled) {
    static_assert((kL1RingBlocks & (kL1RingBlocks - 1)) == 0, "L1 ring capacity must be a power of two");
    // Top-k routes must contain each expert at most once per source token.
    // Grouped swizzle can visit all M blocks of one expert before releasing
    // its first block. Reserve an entire worst-case expert to guarantee
    // progress independently of the routing distribution. Once preceding
    // experts have drained, every slot of this expert is writable: its
    // previous occupant is strictly before this expert's logical block range.
    // Each owner group writes increasing block indices, so its producer order
    // cannot block publication of those writable blocks behind a later block.
    constexpr uint64_t kMaxExpertBlocks = (uint64_t(kNumRanks) * kNumMaxTokensPerRank + kBlockM - 1) / kBlockM;
    static_assert(kL1RingBlocks >= kMaxExpertBlocks,
                  "Unsafe L1 ring capacity for grouped-swizzle scheduling: ring must hold "
                  "ceil(num_ranks * num_max_tokens_per_rank / 256) blocks. "
                  "Disable recycling with prefill_l1_ring=False or increase the ring capacity.");
  }
  constexpr int kScaleGranularityM = 1;
  constexpr int kScaleGranularityN = kSwigluGran;
  constexpr int kScaleMsPerTile    = kBlockM / kScaleGranularityM;
  constexpr int kScaleNsPerTile    = kBlockN / kScaleGranularityN;

  using ElementA           = mutlass::float_e4m3_t;
  using ElementB           = mutlass::float_e4m3_t;
  using ElementD           = mutlass::bfloat16_t;
  using ElementAccumulator = float;
  using TileShape          = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
  // The pool batch stride exceeds int32 for the 8K/H7168 prefill shape.
  using StrideA = decltype(make_stride(Int<kL1ShapeK>{}, _1{}, uint64_t{}));
  using StrideB = decltype(make_stride(Int<kL1ShapeK>{}, _1{}, Int<kL1ShapeN * kL1ShapeK>{}));

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

  constexpr int kNumThreadsPerWarp          = mutlass::NumThreadsPerWarp;
  constexpr int kNumThreadsPerWarpSquad     = mutlass::NumThreadsPerWarpSquad;
  constexpr int kNumMmaThreads              = kNumMmaWarpSquads * kNumThreadsPerWarpSquad;
  constexpr int kNumMmaWarps                = kNumMmaThreads / kNumThreadsPerWarp;
  constexpr int kWarpsPerWarpSquad          = kNumThreadsPerWarpSquad / kNumThreadsPerWarp;
  constexpr int kNumEffectiveLoadWarpSquads = kNumLoadWarpSquads;
  constexpr int kTotalWarpSquads            = kNumEffectiveLoadWarpSquads + kNumMmaWarpSquads;
  constexpr int kNumDispatchWarps           = 3;
  constexpr int kNumDispatchSetupWarps      = 8;

  using PipelineAB        = PrefillStage1TmePipeline<kNumStages>;
  using PipelineABState   = typename PipelineAB::PipelineState;
  using PipelineSeq       = mutlass::OrderedSequenceBarrier<1, kNumMmaWarpSquads>;
  using PipelineSeqParams = typename PipelineSeq::Params;
  using ConsumerBarrier   = mutlass::arch::AsyncBarrier;

  constexpr int kTmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  constexpr int kTmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  struct MUTE_ALIGNAS(1) BarrierStorage {
    typename PipelineAB::Storage PipelineAB;
    uint8_t                      PipelineSeq[PipelineSeq::NumBarriers];
    uint8_t                      Consumer[2];
    uint8_t                      TileDesc[2];
    uint8_t                      TokenPull[kNumDispatchWarps];
    uint8_t                      Dispatch[2];
    uint8_t                      DispatchSetup[2];
    uint8_t                      DispatchReady[1];
  };
  static_assert(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount <=
                    ConsumerBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "Stage 1 async barrier storage exceeds hardware capacity");

  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, 256> smem_a;
    mute::array_aligned<ElementB, cosize_v<SmemLayoutB>, 256> smem_b;
    // One amax per row and per 64-column SwiGLU group.  The two groups are
    // merged into the per-128 output scale below.
    mute::array_aligned<float, kSwigluColumnGroupsPerTile * kBlockM, 256> smem_output_amax;
  };

  extern __shared__ __align__(256) uint8_t smem_buffer[];
  __shared__ uint32_t                      consumer_sync_count;
  __shared__ uint32_t                      consumer_sync_phase;
  __shared__ uint32_t                      smem_l1_tile_desc[stage1_sched::L1TileDesc::kNumWords];
  auto&                                    shared          = *reinterpret_cast<SharedStorage*>(smem_buffer);
  auto*                                    barrier_storage = reinterpret_cast<BarrierStorage*>(0);
  mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage) + ConsumerBarrier::ReservedAsyncBarrierCount);

  const auto buffer_layout = unified::BufferLayout(sym_buffer.get_base_ptr(),
                                                   kNumRanks,
                                                   kNumExperts,
                                                   kNumMaxTokensPerRank,
                                                   kNumTopk,
                                                   kHidden,
                                                   kIntermediateHidden,
                                                   kNumMaxPoolTokens,
                                                   kL1RingTokens);
  const auto workspace     = buffer_layout.get_prefill_workspace();

  constexpr uint32_t kPipelineTransactionBytes = kTmeTransactionBytesA + kTmeTransactionBytesB;
  constexpr uint32_t kPipelineNumConsumers     = kNumMmaWarps;
  PipelineAB         pipeline(&barrier_storage->PipelineAB, kPipelineTransactionBytes, kPipelineNumConsumers);

  constexpr int kComputeThreadOffset = 0;
  const int     warp_idx             = mutlass::canonical_warp_idx_sync();
  const int     compute_thread_idx   = int(threadIdx.x) - kComputeThreadOffset;
  const int     compute_warp_idx     = compute_thread_idx >= 0 ? compute_thread_idx / kNumThreadsPerWarp : 0;
  const int     squad                = compute_thread_idx >= 0 ? compute_thread_idx / kNumThreadsPerWarpSquad : 0;
  const int     warp_idx_in_squad    = compute_warp_idx & (kWarpsPerWarpSquad - 1);
  const bool    is_producer_squad    = compute_thread_idx >= 0 && squad == 0;
  const bool    is_consumer_squad =
      compute_thread_idx >= 0 && squad >= kNumEffectiveLoadWarpSquads && squad < kTotalWarpSquads;
  constexpr int kTmeIssueWarpInSquad = kNumDispatchWarps;
  const bool    is_tme_issue_warp    = is_producer_squad && warp_idx_in_squad == kTmeIssueWarpInSquad;

  Tensor sA = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shared.smem_b.data()), SmemLayoutB{});

  PipelineABState pipe_write = mutlass::make_producer_start_state_warpspecialized<PipelineAB>();
  PipelineABState pipe_read;

  PipelineSeqParams seq_params{};
  seq_params.group_size = kWarpsPerWarpSquad;
  seq_params.group_id   = is_consumer_squad ? squad - kNumEffectiveLoadWarpSquads : 0;
  PipelineSeq    pipeline_seq(seq_params,
                           static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->PipelineSeq)));
  const uint32_t consumer_barrier_base = static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Consumer)) +
                                         ConsumerBarrier::ReservedAsyncBarrierCount;
  const uint32_t tile_desc_barrier_base =
      static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->TileDesc)) +
      ConsumerBarrier::ReservedAsyncBarrierCount;

  if (threadIdx.x == 0) {
    consumer_sync_count = 0;
    consumer_sync_phase = 0;
    ConsumerBarrier::init(consumer_barrier_base, kNumMmaWarps, 0);
    ConsumerBarrier::init(consumer_barrier_base + 1, kNumMmaWarps, 0);
    ConsumerBarrier::init(tile_desc_barrier_base, 1 + kNumMmaWarps, 0);
    ConsumerBarrier::init(tile_desc_barrier_base + 1, 1 + kNumMmaWarps, 0);
    const uint32_t dispatch_barrier_base =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Dispatch)) +
        ConsumerBarrier::ReservedAsyncBarrierCount;
    const uint32_t token_pull_barrier_base =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->TokenPull)) +
        ConsumerBarrier::ReservedAsyncBarrierCount;
    for (uint32_t i = 0; i < kNumDispatchWarps; ++i) {
      ConsumerBarrier::init(token_pull_barrier_base + i, 1, 0);
    }
    ConsumerBarrier::init(dispatch_barrier_base, kNumDispatchWarps, 0);
    ConsumerBarrier::init(dispatch_barrier_base + 1, kNumDispatchWarps, 0);
    const uint32_t dispatch_setup_barrier =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchSetup)) +
        ConsumerBarrier::ReservedAsyncBarrierCount;
    ConsumerBarrier::init(dispatch_setup_barrier, kNumDispatchSetupWarps, 0);
    ConsumerBarrier::init(dispatch_setup_barrier + 1, kNumDispatchSetupWarps, 0);
    const uint32_t dispatch_ready_barrier =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchReady)) +
        ConsumerBarrier::ReservedAsyncBarrierCount;
    ConsumerBarrier::init(dispatch_ready_barrier, kNumDispatchWarps, 0);
  }
  __syncthreads();

  uint32_t tile_desc_barrier_call_idx = 0;
  auto     tile_desc_sync             = [&]() {
    const uint32_t barrier_idx = tile_desc_barrier_base + (tile_desc_barrier_call_idx & 1u);
    const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
    ConsumerBarrier::wait(barrier_idx, phase);
    ++tile_desc_barrier_call_idx;
  };

  constexpr uint32_t kNumDispatchThreads      = kNumDispatchWarps * mutlass::NumThreadsPerWarp;
  constexpr uint32_t kNumDispatchSetupThreads = kNumDispatchSetupWarps * mutlass::NumThreadsPerWarp;
  const bool         is_dispatch_warp         = warp_idx < int(kNumDispatchWarps);
  const bool         is_dispatch_setup_warp   = warp_idx < int(kNumDispatchSetupWarps);
  if constexpr (!kComputeOnly) {
    if (is_dispatch_setup_warp) {
      const uint32_t     thread_idx        = threadIdx.x;
      const uint32_t     lane_idx          = mutlass::canonical_lane_idx();
      const uint32_t     mp_idx            = blockIdx.x;
      const uint32_t     dispatch_warp_idx = warp_idx;
      constexpr uint32_t kDispatchScratchOffset =
          math::constexpr_align<uint32_t>(static_cast<uint32_t>(sizeof(SharedStorage)), 4096u);
      constexpr uint32_t kDispatchCountScratch =
          math::constexpr_align<uint32_t>(2 * kNumExperts * sizeof(uint32_t), 4096u);
      auto*              smem_expert_count    = reinterpret_cast<uint32_t*>(smem_buffer + kDispatchScratchOffset);
      auto*              smem_rank_prefix     = smem_expert_count + kNumExperts;
      constexpr uint32_t kDispatchTokenStride = math::constexpr_align<uint32_t>(kHidden, 4096u);
      auto*              smem_token_pull =
          reinterpret_cast<int4*>(smem_buffer + kDispatchScratchOffset + kDispatchCountScratch +
                                  (is_dispatch_warp ? dispatch_warp_idx : 0) * kDispatchTokenStride);

      uint32_t       dispatch_barrier_call_idx = 0;
      const uint32_t dispatch_barrier_base =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->Dispatch)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      auto dispatch_sync = [&]() {
        const uint32_t barrier_idx = dispatch_barrier_base + (dispatch_barrier_call_idx & 1u);
        const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
        ConsumerBarrier::wait(barrier_idx, phase);
        ++dispatch_barrier_call_idx;
      };
      const uint32_t dispatch_setup_barrier =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchSetup)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      uint32_t dispatch_setup_barrier_call_idx = 0;
      auto     dispatch_setup_sync             = [&]() {
        const uint32_t barrier_idx = dispatch_setup_barrier + (dispatch_setup_barrier_call_idx & 1u);
        const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
        ConsumerBarrier::wait(barrier_idx, phase);
        ++dispatch_setup_barrier_call_idx;
      };

      const auto input_token_buffer        = buffer_layout.get_input_token_buffer();
      const auto input_sf_buffer           = buffer_layout.get_input_sf_buffer();
      const auto input_topk_idx_buffer     = buffer_layout.get_input_topk_idx_buffer();
      const auto input_topk_weights_buffer = buffer_layout.get_input_topk_weights_buffer();
      (void)input_topk_weights_buffer;

      for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumDispatchSetupThreads) {
        smem_expert_count[expert_idx] = 0;
      }
      dispatch_setup_sync();

      const auto for_each_source_token = [&](const auto& process) {
        for (uint32_t token_idx = (mp_idx * kNumDispatchSetupWarps + dispatch_warp_idx); token_idx < num_tokens;
             token_idx += kNumMPs * kNumDispatchSetupWarps) {
          process(token_idx);
          __syncwarp();
        }
      };
      for_each_source_token([&](uint32_t token_idx) {
        int32_t expert_idx = -1;
        if (lane_idx < kNumTopk) {
          expert_idx = static_cast<int32_t>(
              __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx));
        }
        if (expert_idx >= 0 && expert_idx < static_cast<int32_t>(kNumExperts)) {
          atomic_add_shared_u32(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
        }
      });
      dispatch_setup_sync();

      for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumDispatchSetupThreads) {
        const uint64_t send_value     = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[expert_idx]);
        smem_expert_count[expert_idx] = static_cast<uint32_t>(atomic_add_global_u64(
            reinterpret_cast<unsigned long long*>(workspace.get_expert_send_count_ptr(expert_idx)), send_value));
      }
      dispatch_setup_sync();

      __threadfence_system_noflush();
      grid_sync<kNumMPs, 0>(workspace, mp_idx, thread_idx, dispatch_setup_sync);

      for_each_source_token([&](uint32_t token_idx) {
        int32_t expert_idx = -1;
        if (lane_idx < kNumTopk) {
          expert_idx = static_cast<int32_t>(
              __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx));
        }
        const bool     routed           = expert_idx >= 0 && expert_idx < static_cast<int32_t>(kNumExperts);
        const uint32_t dst_rank_idx     = routed ? static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank : 0;
        const uint32_t local_expert_idx = routed ? static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank : 0;

        uint32_t dst_slot_idx = 0;
        if (routed) {
          dst_slot_idx = atomic_add_shared_u32(smem_expert_count + static_cast<uint32_t>(expert_idx), 1);
          *sym_buffer.map(workspace.get_src_token_topk_idx_ptr(local_expert_idx, sym_buffer.rank_idx, dst_slot_idx),
                          dst_rank_idx) = token_idx * kNumTopk + lane_idx;
        }
      });

      __syncwarp();
      if (lane_idx == 0) {
        __threadfence_system_noflush();
      }
      __syncwarp();

      grid_sync<kNumMPs, 0>(workspace, mp_idx, thread_idx, dispatch_setup_sync);
      if (mp_idx == 0) {
        for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumDispatchSetupThreads) {
          const uint32_t dst_rank_idx         = expert_idx / kNumExpertsPerRank;
          const uint32_t dst_local_expert_idx = expert_idx % kNumExpertsPerRank;
          const uint64_t expert_status        = ld_volatile_global(workspace.get_expert_send_count_ptr(expert_idx));
          st_relaxed_sys_global(
              sym_buffer.map(workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                             dst_rank_idx),
              expert_status & 0xffffffffull);
          atomic_add_global_u64(reinterpret_cast<unsigned long long*>(sym_buffer.map(
                                    workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx)),
                                static_cast<unsigned long long>(expert_status));
        }
      }

      __threadfence_system();
      mtlink_barrier<kNumRanks, kNumMPs, kNumDispatchSetupThreads, 0, 1, true>(
          workspace, sym_buffer, mp_idx, thread_idx, dispatch_setup_sync, true, true);
      const uint32_t dispatch_ready_barrier =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchReady)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      if (is_dispatch_warp) {
        ConsumerBarrier::arrive<false>(dispatch_ready_barrier);
      }

      if (is_dispatch_warp) {
        // The route loop below is latency-sensitive.  Build one CTA-wide prefix
        // table with a single warp, using all 32 lanes to avoid serial HBM loads.
        if (dispatch_warp_idx == 0) {
          uint32_t pool_block_prefix = 0;
          for (uint32_t expert_base = 0; expert_base < kNumExperts; expert_base += mutlass::NumThreadsPerWarp) {
            const uint32_t expert_idx = expert_base + lane_idx;
            const uint32_t expert_tokens =
                static_cast<uint32_t>(ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(expert_idx)));
            const uint32_t expert_blocks = math::ceil_div(expert_tokens, kBlockM);
            uint32_t       inclusive     = expert_blocks;
#pragma unroll
            for (uint32_t offset = 1; offset < mutlass::NumThreadsPerWarp; offset <<= 1) {
              const uint32_t previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
              if (lane_idx >= offset) {
                inclusive += previous;
              }
            }
            smem_expert_count[expert_idx] = pool_block_prefix + inclusive - expert_blocks;
            pool_block_prefix += __shfl_sync(0xffffffffu, inclusive, mutlass::NumThreadsPerWarp - 1);
          }
          for (uint32_t local_expert_idx = lane_idx; local_expert_idx < kNumExpertsPerRank;
               local_expert_idx += mutlass::NumThreadsPerWarp) {
            uint32_t rank_token_prefix = 0;
            for (uint32_t source_rank_idx = 0; source_rank_idx < kNumRanks; ++source_rank_idx) {
              smem_rank_prefix[local_expert_idx * kNumRanks + source_rank_idx] = rank_token_prefix;
              rank_token_prefix +=
                  ld_volatile_global(workspace.get_expert_recv_count_ptr(source_rank_idx, local_expert_idx));
            }
          }
        }
        dispatch_sync();

        constexpr uint32_t kNumSF = kHidden / 128;
        // kNumOwnerCtas CTAs own each pool block. Their copy warps fill
        // disjoint rows, making complete M tiles available without waiting
        // for the whole active-token stripe. Publish only after every lane's
        // stores are system-visible; each warp contributes its actual row count.
        constexpr uint32_t kNumCopyWarpsPerBlock = kNumOwnerCtas * kNumDispatchWarps;
        uint32_t           local_expert_idx      = 0;
        const uint32_t     last_expert           = kNumExpertsPerRank - 1;
        const uint32_t     pool_blocks =
            smem_expert_count[last_expert] +
            math::ceil_div(
                static_cast<uint32_t>(ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(last_expert))),
                kBlockM);
        for (uint32_t pool_block = mp_idx / kNumOwnerCtas; pool_block < pool_blocks;
             pool_block += kNumMPs / kNumOwnerCtas) {
          while (local_expert_idx + 1 < kNumExpertsPerRank && smem_expert_count[local_expert_idx + 1] <= pool_block) {
            ++local_expert_idx;
          }
          const uint32_t expert_tokens =
              static_cast<uint32_t>(ld_volatile_global(workspace.get_expert_recv_count_sum_ptr(local_expert_idx)));
          const uint32_t expert_row_base = (pool_block - smem_expert_count[local_expert_idx]) * kBlockM;
          const uint32_t valid_m         = min(kBlockM, expert_tokens - expert_row_base);
          // The physical slot is only overwritten once every L1 N block of the
          // previous occupant has released it.
          if constexpr (kL1RingEnabled && !kDispatchOnly) {
            if (pool_block >= kL1RingBlocks) {
              // One lane per CTA polls on behalf of all dispatch warps.  Besides
              // avoiding atomic-CAS traffic, this prevents the idle copy warps
              // from competing with the consumer that must release the slot.
              if (dispatch_warp_idx == 0 && lane_idx == 0) {
                const auto* consumed_ptr = reinterpret_cast<const volatile uint64_t*>(
                    __musa_ptr_gen_to_global(workspace.get_l1_consumed_mask_ptr(pool_block - kL1RingBlocks)));
                while (*consumed_ptr != kL1ConsumedFullMask) {
                  __nanosleep(1000);
                }
              }
              dispatch_sync();
              __threadfence_system_noflush();
            }
          }
          const uint32_t ring_pool_block = kL1RingEnabled ? (pool_block & (kL1RingBlocks - 1)) : pool_block;
          uint32_t       copied          = 0;
          for (uint32_t row = (mp_idx % kNumOwnerCtas) * kNumDispatchWarps + dispatch_warp_idx;
               dispatch_warp_idx < kNumDispatchWarps && row < valid_m;
               row += kNumCopyWarpsPerBlock) {
            const uint32_t expert_row      = expert_row_base + row;
            uint32_t       source_rank_idx = 0;
            for (uint32_t r = 1; r < kNumRanks; ++r) {
              if (smem_rank_prefix[local_expert_idx * kNumRanks + r] <= expert_row) source_rank_idx = r;
            }
            const uint32_t source_slot = expert_row - smem_rank_prefix[local_expert_idx * kNumRanks + source_rank_idx];
            const uint32_t token_topk  = ld_volatile_global(
                workspace.get_src_token_topk_idx_ptr(local_expert_idx, source_rank_idx, source_slot));
            const uint32_t source_token_idx = token_topk / kNumTopk;
            const uint32_t topk_idx         = token_topk % kNumTopk;
            const uint32_t pool_token       = pool_block * kBlockM + row;
            const uint32_t ring_pool_token  = ring_pool_block * kBlockM + row;
            if (lane_idx == 0) {
              *workspace.get_token_src_metadata_ptr(pool_token) =
                  PackedTokenSrcMetadata::pack(source_rank_idx, source_token_idx, topk_idx, local_expert_idx);
            }
            const auto* src_token =
                sym_buffer.map(input_token_buffer.get_data_buffer(source_token_idx).get_base_ptr(), source_rank_idx);
            const auto* src_sf = sym_buffer.map(
                input_sf_buffer.get_data_buffer(source_token_idx).get_base_ptr<uint32_t>(), source_rank_idx);
            for (uint32_t sf = lane_idx; sf < kNumSF; sf += mutlass::NumThreadsPerWarp) {
              reinterpret_cast<uint32_t*>(l1_acts_sf)[static_cast<uint64_t>(ring_pool_token) * kNumSF + sf] =
                  src_sf[sf];
            }
            __syncwarp();
            auto* dst =
                reinterpret_cast<PrefillStage1B512*>(l1_acts + static_cast<uint64_t>(ring_pool_token) * kHidden);
            for (uint32_t i = lane_idx; i < kHidden / sizeof(PrefillStage1B512); i += mutlass::NumThreadsPerWarp) {
              dst[i] = reinterpret_cast<const PrefillStage1B512*>(src_token)[i];
            }
            ++copied;
          }
          __syncwarp();
          __threadfence_system_noflush();
          __syncwarp();
          if (lane_idx == 0 && copied != 0)
            atomic_add_global_u32(workspace.get_l1_arrival_count_ptr(pool_block), copied);
        }

        mtlink_barrier<kNumRanks, kNumMPs, kNumDispatchThreads, 1, 2, true>(
            workspace, sym_buffer, mp_idx, thread_idx, dispatch_sync, true, false);
      }
    }
  }
  if constexpr (!kDispatchOnly) {
    if (!is_dispatch_warp) {
      const uint32_t dispatch_ready_barrier =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->DispatchReady)) +
          ConsumerBarrier::ReservedAsyncBarrierCount;
      if constexpr (!kComputeOnly) {
        ConsumerBarrier::wait(dispatch_ready_barrier, 0);
      }
      using Scheduler = stage1_sched::L1TileScheduler<kBlockM,
                                                      kBlockN,
                                                      kBlockK,
                                                      kL1ShapeN,
                                                      kL1ShapeK,
                                                      kNumExpertsPerRank,
                                                      kNumExpertsPerWave,
                                                      kNumMPs,
                                                      kNumRanks,
                                                      true>;

      using TmeLoadA = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                              mute::TME::SmemSwizzleStride::B256,
                                              mute::TME::SmemSwizzleLine::B256,
                                              mute::TME::CacheHint::CACHE_NORMAL,
                                              mute::TME::CacheHint::CACHE_NONE,
                                              mute::PrefetchSize::B128>;
      using TmeLoadB = mute::MP31_TME_LOAD_3D<mute::TME::SmemSwizzleGranularity::B16,
                                              mute::TME::SmemSwizzleStride::B256,
                                              mute::TME::SmemSwizzleLine::B256,
                                              mute::TME::CacheHint::CACHE_NORMAL,
                                              mute::TME::CacheHint::CACHE_NONE,
                                              mute::PrefetchSize::B128>;
      if (is_tme_issue_warp) {
        Scheduler scheduler(workspace);
        scheduler.init_l1_dispatch_count_scheduler();
        if constexpr (kL1RingEnabled) {
          // Preserve the weight-friendly grouped swizzle while the routed pool
          // fits in the ring.  Under ring pressure, consume all N tiles of an M
          // block together so its physical slot can be released promptly.
          scheduler.set_grouped_swizzle(scheduler.get_total_num_m_blocks() <= kL1RingBlocks);
        }
        while (true) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
          __musa_loop_transparent_outermost();
#endif
          stage1_sched::L1TileDesc next_desc;
          const bool               has_block = scheduler.fetch_next_l1_tile_desc(next_desc);
          if (mutlass::canonical_lane_idx() == 0) {
            if (has_block) {
              const auto* arrival_ptr = workspace.get_l1_arrival_count_ptr(next_desc.pool_block_idx);
              while (ld_volatile_global(arrival_ptr) != next_desc.valid_m) {
                __nanosleep(20);
              }
            }
            next_desc.store_to_smem(smem_l1_tile_desc);
          }
          __syncwarp();
          tile_desc_sync();
          const stage1_sched::L1TileDesc desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc);
          if (!desc.is_valid()) {
            break;
          }
          const uint32_t expert_idx  = desc.local_expert_idx;
          const uint32_t n_block_idx = desc.n_block_idx;
          const uint32_t m_base =
              (kL1RingEnabled ? (desc.pool_block_idx & (kL1RingBlocks - 1u)) : desc.pool_block_idx) * kBlockM;

          MUTLASS_PRAGMA_NO_UNROLL
          for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
            const int stage = int(pipe_write.index());
            pipeline.producer_acquire(pipe_write);
            const uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);
            auto           tAsA   = sA(_, _, stage);
            TmeLoadA::copy(&tensor_map_l1_acts,
                           bar_id,
                           raw_pointer_cast(tAsA.data()),
                           static_cast<int32_t>(kb * kBlockK),
                           static_cast<int32_t>(m_base),
                           0,
                           static_cast<int32_t>(kBlockK),
                           static_cast<int32_t>(kBlockM),
                           1);
            auto tBsB = sB(_, _, stage);
            TmeLoadB::copy(&tensor_map_l1_weights,
                           bar_id,
                           raw_pointer_cast(tBsB.data()),
                           static_cast<int32_t>(kb * kBlockK),
                           static_cast<int32_t>(n_block_idx * kBlockN),
                           static_cast<int32_t>(expert_idx),
                           static_cast<int32_t>(kBlockK),
                           static_cast<int32_t>(kBlockN),
                           1);
            ++pipe_write;
          }
        }
      } else if (is_consumer_squad) {
        const int consumer_thread_idx = compute_thread_idx - int(kNumEffectiveLoadWarpSquads * kNumThreadsPerWarpSquad);
        const int mma_squad           = squad - kNumEffectiveLoadWarpSquads;
        uint32_t  consumer_local_phase      = 0;
        uint32_t  consumer_barrier_call_idx = 0;
        auto      consumer_sync_async       = [&]() {
          const uint32_t barrier_idx = consumer_barrier_base + (consumer_barrier_call_idx & 1u);
          const unsigned phase       = ConsumerBarrier::arrive<true>(barrier_idx);
          ConsumerBarrier::wait(barrier_idx, phase);
          ++consumer_barrier_call_idx;
        };
        auto consumer_sync = [&]() {
          __threadfence_block();
          __syncwarp();
          if ((consumer_thread_idx % int(mutlass::NumThreadsPerWarp)) == 0) {
            const uint32_t old_count = atomic_add_shared_u32(&consumer_sync_count, 1);
            if (old_count == kNumMmaWarps - 1) {
              atomicExch(reinterpret_cast<uint32_t*>(__musa_ptr_gen_to_shared(&consumer_sync_count)), 0u);
              __threadfence_block();
              atomicExch(reinterpret_cast<uint32_t*>(__musa_ptr_gen_to_shared(&consumer_sync_phase)),
                         consumer_local_phase ^ 1u);
            }
          }
          __syncwarp();
          auto* phase_ptr = reinterpret_cast<volatile uint32_t*>(__musa_ptr_gen_to_shared(&consumer_sync_phase));
          while (*phase_ptr != (consumer_local_phase ^ 1u)) {
            __nanosleep(20);
          }
          __threadfence_block();
          __syncwarp();
          consumer_local_phase ^= 1u;
        };

        while (true) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
          __musa_loop_transparent_outermost();
#endif
          tile_desc_sync();
          const stage1_sched::L1TileDesc desc = stage1_sched::L1TileDesc::load_from_smem(smem_l1_tile_desc);
          if (!desc.is_valid()) {
            break;
          }
          const uint32_t expert_idx     = desc.local_expert_idx;
          const uint32_t n_block_idx    = desc.n_block_idx;
          const uint32_t pool_block_idx = desc.pool_block_idx;
          const uint32_t valid_m        = desc.valid_m;
          const uint32_t l1_ring_block  = kL1RingEnabled ? (pool_block_idx & (kL1RingBlocks - 1u)) : pool_block_idx;

          TiledMma tiled_mma;
          auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);
          auto     rAcc    = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
          fill(rAcc, ElementAccumulator(0));

          Tensor tCsA = thr_mma.partition_A(sA);
          Tensor tCsB = thr_mma.partition_B(sB);
          Tensor tCrA = thr_mma.make_fragment_A(tCsA);
          Tensor tCrB = thr_mma.make_fragment_B(tCsB);

          Tensor mScaleA = make_tensor(make_gmem_ptr(l1_acts_sf),
                                       make_shape(Int<kL1RingPaddedSFTokens>{}, Int<kNumKBlocks>{}),
                                       make_stride(Int<kNumKBlocks>{}, _1{}));
          Tensor mScaleB = make_tensor(
              make_gmem_ptr(l1_weights_sf),
              make_shape(Int<kL1ShapeN / kScaleGranularityN>{}, Int<kNumKBlocks>{}, Int<kNumExpertsPerRank>{}),
              make_stride(Int<kNumKBlocks>{}, _1{}, Int<(kL1ShapeN / kScaleGranularityN) * kNumKBlocks>{}));
          Tensor gScaleA = local_tile(mScaleA, make_tile(Int<kScaleMsPerTile>{}), make_coord(l1_ring_block, _));
          Tensor gScaleB =
              local_tile(mScaleB(_, _, expert_idx), make_tile(Int<kScaleNsPerTile>{}), make_coord(n_block_idx, _));

          using ScaleAViewAsCLayout = Layout<Shape<Shape<Int<kScaleGranularityM>, Int<kScaleMsPerTile>>, Int<kBlockN>>,
                                             Stride<Stride<_0, _1>, _0>>;
          using ScaleBViewAsCLayout = Layout<Shape<Int<kBlockM>, Shape<Int<kScaleGranularityN>, Int<kScaleNsPerTile>>>,
                                             Stride<_0, Stride<_0, _1>>>;
          Tensor tCgScaleA0         = thr_mma.partition_C(gScaleA(_, 0).compose(ScaleAViewAsCLayout{}));
          Tensor tCgScaleB0         = thr_mma.partition_C(gScaleB(_, 0).compose(ScaleBViewAsCLayout{}));
          Tensor tCrScaleA          = make_tensor_like<float>(tCgScaleA0);
          Tensor tCrScaleB          = make_tensor_like<float>(tCgScaleB0);

          using ScaleCopyAtom = Copy_Atom<MP31_ROBUST_LOAD<float>, float>;
          auto scale_copy_a =
              ScaleCopyAtom{}.with(make_robust_desc(l1_acts_sf, static_cast<size_t>(cosize(mScaleA.layout()))));
          auto scale_copy_b =
              ScaleCopyAtom{}.with(make_robust_desc(l1_weights_sf, static_cast<size_t>(cosize(mScaleB.layout()))));

          using AccumTensor  = decltype(rAcc);
          using ScaleATensor = decltype(tCrScaleA);
          using ScaleBTensor = decltype(tCrScaleB);
          using IterAccum    = ::mate::deep_gemm::ScalingAccumulationIterative<typename AccumTensor::engine_type,
                                                                               typename AccumTensor::layout_type,
                                                                               typename ScaleATensor::engine_type,
                                                                               typename ScaleATensor::layout_type,
                                                                               typename ScaleBTensor::engine_type,
                                                                               typename ScaleBTensor::layout_type>;
          const uint32_t mma_per_iter = static_cast<uint32_t>(size<2>(tCrA));
          IterAccum      accumulation(rAcc, mma_per_iter, mma_per_iter, tCrScaleA, tCrScaleB);
          accumulation.initializeA(scale_copy_a, tCgScaleA0, tCrScaleA);
          accumulation.initializeB(scale_copy_b, tCgScaleB0, tCrScaleB);

          MUTLASS_PRAGMA_NO_UNROLL
          for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
            Tensor tCgScaleA = thr_mma.partition_C(gScaleA(_, kb).compose(ScaleAViewAsCLayout{}));
            Tensor tCgScaleB = thr_mma.partition_C(gScaleB(_, kb).compose(ScaleBViewAsCLayout{}));
            accumulation.copyA(scale_copy_a, tCgScaleA, tCrScaleA);
            accumulation.copyB(scale_copy_b, tCgScaleB, tCrScaleB);
            if (accumulation.prepare_if_needed()) {
              accumulation.update_iterationAB(tCrScaleA, tCrScaleB);
              accumulation.div_if_needAB();
            }

            pipeline.consumer_wait(pipe_read, mma_squad);
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
            pipeline.consumer_release(pipe_read, mma_squad);
            ++pipe_read;
          }
          accumulation.scale_residue_if_needed(tCrScaleA, tCrScaleB);

          if (consumer_thread_idx < int(kSwigluColumnGroupsPerTile * kBlockM)) {
            shared.smem_output_amax[consumer_thread_idx] = 0.0f;
          }
          consumer_sync_async();
          // All MMA warps have drained the tile's A/scales, so the logical pool
          // block's physical ring slot can be recycled by the dispatcher.
          if constexpr (kL1RingEnabled) {
            if (consumer_thread_idx == 0) {
              __threadfence_system_noflush();
              atomicOr(reinterpret_cast<unsigned long long*>(
                           __musa_ptr_gen_to_global(workspace.get_l1_consumed_mask_ptr(pool_block_idx))),
                       static_cast<unsigned long long>(1ull << n_block_idx));
            }
          }

          constexpr int kAccumPerThread      = 128;
          constexpr int kAccumColsPerRow     = 16;
          constexpr int kGateAccumColsPerRow = kAccumColsPerRow / 2;
          constexpr int kPairOffset          = kGateAccumColsPerRow;
          constexpr int kAccumRowGroups      = kAccumPerThread / kAccumColsPerRow;
          static_assert(size(rAcc) == kAccumPerThread, "Unexpected M256 C fragment layout");
          // Direct SiTU needs two tanh chains in addition to sigmoid.  Keeping
          // the complete C fragment live while evaluating them makes MP31 spill
          // all 128 accumulators repeatedly.  Once MMA has drained the A/B
          // pipeline, stage the fragment in the unused A/B shared buffers so
          // all 128 accumulator registers die before activation.  Pack four
          // BF16 values per transaction to keep this fixed staging pass cheap.
          uint64_t* activation_scratch = reinterpret_cast<uint64_t*>(shared.smem_a.data());
          static_assert(
              sizeof(shared.smem_a) + sizeof(shared.smem_b) >= kAccumPerThread * kNumMmaThreads * sizeof(ElementD),
              "Stage 1 A/B shared storage is too small for SiTU activation staging");
          if constexpr (kUseSitu) {
            MUTE_UNROLL
            for (int group = 0; group < kAccumPerThread / 4; ++group) {
              const int i                                                      = group * 4;
              activation_scratch[group * kNumMmaThreads + consumer_thread_idx] = mp31::pack_bf16x4(
                  ElementD(rAcc(i + 0)), ElementD(rAcc(i + 1)), ElementD(rAcc(i + 2)), ElementD(rAcc(i + 3)));
            }
          }
          const int      lane_idx = consumer_thread_idx % int(mutlass::NumThreadsPerWarp);
          const uint32_t lane_col = static_cast<uint32_t>(consumer_thread_idx) & 7u;
          // MP31 M128N128 CLayout is a four-quadrant mapping.  Each thread owns
          // one column lane and 16-row strides inside its 128x128 quadrant;
          // this is the exact mapping emitted by TiledMma::partition_C for the
          // M256/N256 tile used here.  Keeping it explicit avoids constructing
          // and evaluating an identity C tensor twice in the epilogue.
          const uint32_t quadrant        = static_cast<uint32_t>(mma_squad);
          const uint32_t quadrant_thread = static_cast<uint32_t>(consumer_thread_idx) & 127u;
          const uint32_t row_base        = (quadrant >> 1) * 128u + (quadrant_thread >> 3);
          const uint32_t col_base        = (quadrant & 1u) * 128u + (quadrant_thread & 7u);
          MUTE_UNROLL
          for (int row_group = 0; row_group < kAccumRowGroups; ++row_group) {
            const int      first_gate_i = row_group * kAccumColsPerRow;
            const uint32_t row          = row_base + static_cast<uint32_t>(row_group * 16);
            const uint32_t first_col    = col_base;
            const uint32_t swiglu_group = first_col / kSwigluPairCols;
            const bool     group_valid  = row < valid_m;
            float          local_amax   = 0.0f;
            if (group_valid) {
              static_assert(kGateAccumColsPerRow % 4 == 0,
                            "Burst4 activation requires groups of four accumulator columns");
              MUTE_UNROLL
              for (int col_group = 0; col_group < kGateAccumColsPerRow; col_group += 4) {
                const int gate_i = first_gate_i + col_group;
                if constexpr (kUseSitu) {
                  // Stage 1 is already at the MP31 register limit.  Finish and
                  // write the gate activation before loading the up values so
                  // the two transcendental chains do not overlap their live
                  // ranges and spill to private memory.
                  float4 gate =
                      mp31::unpack_bf16x4(activation_scratch[(gate_i / 4) * kNumMmaThreads + consumer_thread_idx]);
                  const float4 gate_sigmoid = mp31::activation_sigmoid_via_tanh_bst4(gate);
                  gate                      = mul(__frcp_rn(activation_beta), gate);
                  gate                      = mp31::activation_tanh_bst4(gate);
                  gate                      = mul(activation_beta, gate);
                  gate                      = mul(gate, gate_sigmoid);
                  float4 up                 = mp31::unpack_bf16x4(
                      activation_scratch[((gate_i + kPairOffset) / 4) * kNumMmaThreads + consumer_thread_idx]);
                  if constexpr (kUseSituLinear) {
                    up = mul(activation_linear_beta,
                             mp31::activation_tanh_bst4(mul(__frcp_rn(activation_linear_beta), up)));
                  }
                  const float y0 = static_cast<float>(ElementD(up.x * gate.x));
                  local_amax     = fmaxf(local_amax, fabsf(y0));
                  const float y1 = static_cast<float>(ElementD(up.y * gate.y));
                  local_amax     = fmaxf(local_amax, fabsf(y1));
                  const float y2 = static_cast<float>(ElementD(up.z * gate.z));
                  local_amax     = fmaxf(local_amax, fabsf(y2));
                  const float y3 = static_cast<float>(ElementD(up.w * gate.w));
                  local_amax     = fmaxf(local_amax, fabsf(y3));
                  activation_scratch[(gate_i / 4) * kNumMmaThreads + consumer_thread_idx] =
                      mp31::pack_bf16x4(ElementD(y0), ElementD(y1), ElementD(y2), ElementD(y3));
                } else {
                  float gate_values[4];
                  float up_values[4];
                  MUTE_UNROLL
                  for (int i = 0; i < 4; ++i) {
                    ElementD gate_bf16 = ElementD(rAcc(gate_i + i));
                    float    up        = static_cast<float>(ElementD(rAcc(gate_i + kPairOffset + i)));
                    if (activation_clamp >= 0.0f) {
                      gate_bf16 = ElementD(fminf(static_cast<float>(gate_bf16), activation_clamp));
                      up        = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                    }
                    gate_values[i] = static_cast<float>(gate_bf16);
                    up_values[i]   = up;
                  }
                  const float4 gate       = make_float4(gate_values[0], gate_values[1], gate_values[2], gate_values[3]);
                  const float4 up         = make_float4(up_values[0], up_values[1], up_values[2], up_values[3]);
                  float4       gate_act   = mul(gate, mp31::activation_sigmoid_bst4(gate));
                  gate_act.x              = static_cast<float>(ElementD(gate_act.x));
                  gate_act.y              = static_cast<float>(ElementD(gate_act.y));
                  gate_act.z              = static_cast<float>(ElementD(gate_act.z));
                  gate_act.w              = static_cast<float>(ElementD(gate_act.w));
                  const float4 output     = mul(up, gate_act);
                  const float  outputs[4] = {
                      static_cast<float>(ElementD(output.x)),
                      static_cast<float>(ElementD(output.y)),
                      static_cast<float>(ElementD(output.z)),
                      static_cast<float>(ElementD(output.w)),
                  };
                  MUTE_UNROLL
                  for (int i = 0; i < 4; ++i) {
                    rAcc(gate_i + i) = outputs[i];
                    local_amax       = fmaxf(local_amax, fabsf(outputs[i]));
                  }
                }
              }
            }
            const float reduced_amax = octet_reduce_max(local_amax, lane_idx);
            if (group_valid && lane_col == 0) {
              shared.smem_output_amax[swiglu_group * kBlockM + row] = reduced_amax;
            }
          }
          consumer_sync();

          if (consumer_thread_idx < int(valid_m)) {
            const uint32_t row        = static_cast<uint32_t>(consumer_thread_idx);
            const uint64_t row_global = static_cast<uint64_t>(pool_block_idx) * kBlockM + row;
            const float    amax  = fmaxf(fmaxf(shared.smem_output_amax[row], shared.smem_output_amax[kBlockM + row]),
                                     kSGLangSwiGLUFP8AmaxFloor);
            const float    scale = amax / kFinfoAmaxE4M3;
            atomicExch(reinterpret_cast<unsigned int*>(__musa_ptr_gen_to_shared(shared.smem_output_amax.data() + row)),
                       __float_as_uint(scale));
            l2_acts_sf[row_global * (kIntermediateHidden / kBlockK) + n_block_idx] = scale;
          }
          consumer_sync_async();

          MUTE_UNROLL
          for (int row_group = 0; row_group < kAccumRowGroups; ++row_group) {
            const int      first_gate_i = row_group * kAccumColsPerRow;
            const uint32_t row          = row_base + static_cast<uint32_t>(row_group * 16);
            if (row >= valid_m) {
              continue;
            }
            const volatile float* scale_ptr =
                reinterpret_cast<const volatile float*>(__musa_ptr_gen_to_shared(shared.smem_output_amax.data() + row));
            const float    scale      = *scale_ptr;
            const float    scale_inv  = __frcp_rn(scale);
            const uint64_t row_global = static_cast<uint64_t>(pool_block_idx) * kBlockM + row;
            if constexpr (kUseSitu) {
              MUTE_UNROLL
              for (int col_group = 0; col_group < kGateAccumColsPerRow; col_group += 4) {
                const int    gate_i = first_gate_i + col_group;
                const float4 values =
                    mp31::unpack_bf16x4(activation_scratch[(gate_i / 4) * kNumMmaThreads + consumer_thread_idx]);
                const float unpacked[4] = {values.x, values.y, values.z, values.w};
                MUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                  const uint32_t col = col_base + static_cast<uint32_t>((col_group + i) * 8);
                  const uint32_t output_col =
                      n_block_idx * kOutputColsPerTile + (col / kSwigluPairCols) * kSwigluGran + col % kSwigluGran;
                  const auto quantized = static_cast<mutlass::float_e4m3_t>(clamp_fp8_e4m3(unpacked[i] * scale_inv));
                  l2_acts[row_global * static_cast<uint64_t>(kIntermediateHidden) + output_col] = quantized;
                }
              }
            } else {
              MUTE_UNROLL
              for (int col_group = 0; col_group < kGateAccumColsPerRow; ++col_group) {
                const int      gate_i = first_gate_i + col_group;
                const uint32_t col    = col_base + static_cast<uint32_t>(col_group * 8);
                const uint32_t output_col =
                    n_block_idx * kOutputColsPerTile + (col / kSwigluPairCols) * kSwigluGran + col % kSwigluGran;
                const float y         = rAcc(gate_i);
                const auto  quantized = static_cast<mutlass::float_e4m3_t>(clamp_fp8_e4m3(y * scale_inv));
                l2_acts[row_global * static_cast<uint64_t>(kIntermediateHidden) + output_col] = quantized;
              }
            }
          }
          consumer_sync_async();

          if (consumer_thread_idx == 0) {
            __threadfence_system_noflush();
            atomicOr(reinterpret_cast<unsigned long long*>(
                         __musa_ptr_gen_to_global(workspace.get_l2_arrival_mask_ptr(pool_block_idx))),
                     1ull << n_block_idx);
          }
          consumer_sync_async();
        }
      }
    }
  }
}

}  // namespace mega_moe::prefill
