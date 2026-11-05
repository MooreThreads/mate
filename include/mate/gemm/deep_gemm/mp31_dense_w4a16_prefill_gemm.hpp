#pragma once

#include <mutlass/bfloat16.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <type_traits>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"

namespace mate::gemm::dense_w4a16_prefill_gemm {

using namespace mute;

// Large-M W4A16: fp16 activations x signed INT4 weights, one fp16 scale per (K128 group, n).
//
// The K tile is exactly one scale group, so each consumer squad accumulates one K tile into a
// partial fragment and folds it into the output with a single FMA per element. A is loaded by TME
// straight into the SQMMA operand layout, and each consumer squad owns one 128x64 atom of the
// 128xBlockN tile, so no warp squad restages anything.
//
// With zero points the dequantized weight is s * (q + 8 - zp) for the stored signed nibble q, so a
// group contributes s * (A q) + (8 - zp) * s * sum(A) to the output. The per-group row sums of A do
// not depend on N, so the caller computes them once and they ride the stage like the scales: the
// scale operand carries s and (8 - zp) * s as two rows per group, and the consumer folds both terms
// in with two FMAs.
template <class ElementD_, int BlockN_, uint32_t Stages_, bool ZeroPoints_ = false>
struct Mp31DenseW4A16PrefillGemm {
  using ElementA     = mutlass::half_t;
  using ElementB     = mute::int4_t;
  using ElementScale = mutlass::half_t;
  using ElementD     = ElementD_;

  static constexpr int BlockM = 128;
  static constexpr int BlockN = BlockN_;
  static constexpr int BlockK = 128;
  static constexpr int Stages = int(Stages_);
  using TileShape             = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;

  static constexpr bool IsZeroPoints   = ZeroPoints_;
  static constexpr int  ScaleRows      = IsZeroPoints ? 2 : 1;  // per group: s, then (8 - zp) * s

  static_assert(BlockN % 64 == 0, "each consumer squad owns one 64-wide atom of the tile");
  static_assert(Stages >= 2, "the pipeline needs at least two stages");

  using StrideA     = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideB     = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideScale = decltype(make_stride(_1{}, int64_t{}, int64_t{}));  // (n, group): N-contiguous
  using StrideD     = decltype(make_stride(int64_t{}, _1{}, int64_t{}));

  using SqmmaOp = decltype(MP31::SQMMA::
                               ss_op_selector<ElementA, ElementB, float, TileShape, TCE::Major::K, TCE::Major::K, _128, _64>());
  static constexpr int NumMmaWarpSquads = BlockN / 64;
  using AtomLayout                      = decltype(make_layout(Shape<_1, Int<NumMmaWarpSquads>, _1>{}, LayoutRight{}));
  using TiledMma                        = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomA = decltype(mutlass::gemm::collective::detail::
                                       ss_smem_selector_A<TCE::Major::K, ElementA, SqmmaOp, TileShape>());
  using SmemLayoutAtomB = decltype(mutlass::gemm::collective::detail::
                                       ss_smem_selector_B<TCE::Major::K, ElementB, SqmmaOp, TileShape>());
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<Stages>{})));
  using SmemLayoutScale =
      Layout<Shape<Int<BlockN>, Int<ScaleRows>, Int<Stages>>, Stride<_1, Int<BlockN>, Int<BlockN * ScaleRows>>>;
  using SmemLayoutRowSum = Layout<Shape<Int<BlockM>, _1, Int<Stages>>, Stride<_1, Int<BlockM>, Int<BlockM>>>;
  using ElementRowSum    = float;
  using StrideRowSum     = decltype(make_stride(_1{}, int64_t{}, int64_t{}));  // (m, group): M-contiguous

  static constexpr TME::CacheHint TmeAInnerHint     = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeAOuterHint     = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBInnerHint     = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBOuterHint     = TME::CacheHint::CACHE_NONE;
  static constexpr TME::CacheHint TmeScaleInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeScaleOuterHint = TME::CacheHint::CACHE_PERSIST;

  using TME_A = decltype(make_tme_copy<TmeAInnerHint, TmeAOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{})));
  using TME_B = decltype(make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{}),
      make_shape(Int<BlockN>{}, Int<BlockK>{})));
  using TME_Scale = decltype(make_tme_copy<TmeScaleInnerHint, TmeScaleOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementScale const*>(nullptr)),
                  repeat_like(StrideScale{}, int64_t(0)),
                  StrideScale{}),
      take<0, 2>(SmemLayoutScale{})));

  using TME_RowSum = decltype(make_tme_copy<TME::CacheHint::CACHE_NORMAL, TME::CacheHint::CACHE_NORMAL>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementRowSum const*>(nullptr)),
                  repeat_like(StrideRowSum{}, int64_t(0)),
                  StrideRowSum{}),
      take<0, 2>(SmemLayoutRowSum{})));

  static constexpr int TmeTransactionBytesRowSum =
      IsZeroPoints ? mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutRowSum{})) * sizeof_bits_v<ElementRowSum>) : 0;

  static constexpr int TmeTransactionBytes = TmeTransactionBytesRowSum +
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>) +
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>) +
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutScale{})) * sizeof_bits_v<ElementScale>);

  static constexpr int NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr int WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  static constexpr int NumLoadWarpSquads      = 1;
  static constexpr int NumMmaWarps            = NumMmaWarpSquads * WarpsPerWarpSquad;

  static_assert(NumMmaWarpSquads * NumThreadsPerWarpSquad == int(size(TiledMma{})),
                "one 128x64 atom per consumer squad");

  static constexpr int      SmemAlignmentBytes         = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock = (NumLoadWarpSquads + NumMmaWarpSquads) * NumThreadsPerWarpSquad;

  using PipelineAB       = mutlass::Mp31PipelineTmeAsync<Stages>;
  using PipelineABParams = typename PipelineAB::Params;
  using PipelineABState  = typename PipelineAB::PipelineState;

  using PipelineSeq       = mutlass::OrderedSequenceBarrier<1, NumMmaWarpSquads>;
  using PipelineSeqParams = typename PipelineSeq::Params;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineAB[PipelineAB::NumBarriers];
    uint8_t PipelineSeq[PipelineSeq::NumBarriers];
  };

  struct SharedStorage {
    mute::array_aligned<ElementA, cosize_v<SmemLayoutA>, 256>         smem_a;
    MUTE_ALIGNAS(256) ArrayEngine<ElementB, cosize_v<SmemLayoutB>>    smem_b;  // packed, two values per byte
    mute::array_aligned<ElementScale, cosize_v<SmemLayoutScale>, 256> smem_scale;
    mute::array_aligned<ElementRowSum, IsZeroPoints ? cosize_v<SmemLayoutRowSum> : 1, 256> smem_row_sum;
  };

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= 192 * 1024, "shared storage exceeds the MP31 per-block limit");

  struct Arguments {
    ElementA const*     ptr_a     = nullptr;
    ElementB const*     ptr_b     = nullptr;
    ElementScale const*  ptr_scale   = nullptr;
    ElementRowSum const* ptr_row_sum = nullptr;  // zero points only: (K/128, M) per-group row sums of A
    ElementD*            ptr_d       = nullptr;

    StrideA      stride_a;
    StrideB      stride_b;
    StrideScale  stride_scale;
    StrideRowSum stride_row_sum;
    StrideD      stride_d;

    int m       = 0;
    int n       = 0;
    int k       = 0;
    int num_mps = 0;
  };

  struct Params {
    TME_A     tme_a;
    TME_B     tme_b;
    TME_Scale  tme_scale;
    TME_RowSum tme_row_sum;
    ElementD*  ptr_d;
    StrideD   stride_d;
    int       m, n, k;
    int       num_mps;
    int       k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_blocks = ceil_div(args.k, BlockK);

    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k, 1), args.stride_a);
    TME_A tme_a = make_tme_copy<TmeAInnerHint, TmeAOuterHint>(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutA{}));

    auto gB = make_tensor(
        make_gmem_ptr<ElementB>(static_cast<void const*>(args.ptr_b)), make_shape(args.n, args.k, 1), args.stride_b);
    TME_B tme_b = make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
        MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}), make_shape(Int<BlockN>{}, Int<BlockK>{}));

    auto gScale =
        make_tensor(make_gmem_ptr(args.ptr_scale), make_shape(args.n, ScaleRows * k_blocks, 1), args.stride_scale);
    TME_Scale tme_scale =
        make_tme_copy<TmeScaleInnerHint, TmeScaleOuterHint>(MP31_TME_LOAD{}, gScale, take<0, 2>(SmemLayoutScale{}));

    TME_RowSum tme_row_sum{};
    if constexpr (IsZeroPoints) {
      auto gRowSum =
          make_tensor(make_gmem_ptr(args.ptr_row_sum), make_shape(args.m, k_blocks, 1), args.stride_row_sum);
      tme_row_sum = make_tme_copy<TME::CacheHint::CACHE_NORMAL, TME::CacheHint::CACHE_NORMAL>(
          MP31_TME_LOAD{}, gRowSum, take<0, 2>(SmemLayoutRowSum{}));
    }

    return Params{tme_a,
                  tme_b,
                  tme_scale,
                  tme_row_sum,
                  args.ptr_d,
                  args.stride_d,
                  args.m,
                  args.n,
                  args.k,
                  args.num_mps,
                  k_blocks};
  }

  static dim3 get_grid_shape_persistent(Params const& params) {
    const int num_mps = params.num_mps > 0 ? params.num_mps : 1;
    return dim3(static_cast<uint32_t>(num_mps), 1u, 1u);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineABParams pipe_params;
    pipe_params.transaction_bytes = TmeTransactionBytes;
    pipe_params.num_consumers     = NumMmaWarps;
    pipe_params.num_producers     = 1;
    PipelineAB pipeline(pipe_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineAB));

    const int  squad             = mutlass::canonical_warp_squad_idx();
    const int  warp_idx_in_squad = mutlass::canonical_warp_idx_sync() & (WarpsPerWarpSquad - 1);
    const bool is_producer_squad = squad == 0;
    const bool is_tme_issue_warp = is_producer_squad && warp_idx_in_squad == 0;

    PipelineSeqParams seq_params{};
    seq_params.group_size = WarpsPerWarpSquad;
    seq_params.group_id   = is_producer_squad ? 0 : squad - NumLoadWarpSquads;
    PipelineSeq pipeline_seq(seq_params, PipelineAB::NumBarriers);

    Tensor sA     = make_tensor(make_smem_ptr(shared.smem_a.data()), SmemLayoutA{});
    Tensor sB     = make_tensor(make_smem_ptr(shared.smem_b.begin()), SmemLayoutB{});
    Tensor sScale = make_tensor(make_smem_ptr(shared.smem_scale.data()), SmemLayoutScale{});
    Tensor sRowSum = make_tensor(make_smem_ptr(shared.smem_row_sum.data()), SmemLayoutRowSum{});

    using Scheduler =
        mate::deep_gemm::detail::Mp31PersistentTileScheduler<mate::deep_gemm::GemmType::Normal, BlockM, BlockN, 4>;
    Scheduler scheduler(static_cast<uint32_t>(params.m), static_cast<uint32_t>(params.n), 1u, nullptr, 0u);

    const int k_blocks = params.k_blocks;

    PipelineABState pipe_write = mutlass::make_producer_start_state<PipelineAB>();
    PipelineABState pipe_read;

    __syncthreads();

    if (is_producer_squad) {
      auto cta_tme_a     = params.tme_a.get_slice(0);
      auto cta_tme_b     = params.tme_b.get_slice(0);
      auto cta_tme_scale = params.tme_scale.get_slice(0);

      auto mA      = params.tme_a.get_tme_tensor(make_shape(params.m, params.k, 1))(_, _, _0{});
      auto gA_full = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(_, _));
      auto mB      = params.tme_b.get_tme_tensor(make_shape(params.n, params.k, 1))(_, _, _0{});
      auto gB_full = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(_, _));
      auto mScale  = params.tme_scale.get_tme_tensor(make_shape(params.n, ScaleRows * k_blocks, 1))(_, _, _0{});
      auto gS_full = local_tile(mScale, make_shape(Int<BlockN>{}, Int<ScaleRows>{}), make_coord(_, _));
      auto cta_tme_row_sum = params.tme_row_sum.get_slice(0);
      auto mRowSum   = params.tme_row_sum.get_tme_tensor(make_shape(params.m, k_blocks, 1))(_, _, _0{});
      auto gRS_full  = local_tile(mRowSum, make_shape(Int<BlockM>{}, _1{}), make_coord(_, _));

      auto work = scheduler.initial_work_tile_info();
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        const int m_block_idx = int(work.M_idx);
        const int n_block_idx = int(work.N_idx);
        MUTLASS_PRAGMA_NO_UNROLL
        for (int k_block = 0; k_block < k_blocks; ++k_block) {
          if (is_tme_issue_warp) {
            pipeline.producer_acquire(pipe_write);
            const uint32_t bar_id = pipeline.producer_get_barrier_id(pipe_write);
            const int      stage  = int(pipe_write.index());
            copy(params.tme_a.with(bar_id),
                 cta_tme_a.partition_S(gA_full(_, _, m_block_idx, k_block)),
                 cta_tme_a.partition_D(sA(_, _, stage)));
            copy(params.tme_b.with(bar_id),
                 cta_tme_b.partition_S(gB_full(_, _, n_block_idx, k_block)),
                 cta_tme_b.partition_D(sB(_, _, stage)));
            copy(params.tme_scale.with(bar_id),
                 cta_tme_scale.partition_S(gS_full(_, _, n_block_idx, k_block)),
                 cta_tme_scale.partition_D(sScale(_, _, stage)));
            if constexpr (IsZeroPoints) {
              copy(params.tme_row_sum.with(bar_id),
                   cta_tme_row_sum.partition_S(gRS_full(_, _, m_block_idx, k_block)),
                   cta_tme_row_sum.partition_D(sRowSum(_, _, stage)));
            }
          }
          ++pipe_write;
        }
        scheduler.advance_to_next_work();
        work = scheduler.get_work_tile_info();
      }
      return;
    }

    const int consumer_thread_idx = int(threadIdx.x) - NumLoadWarpSquads * NumThreadsPerWarpSquad;
    TiledMma  tiled_mma;
    auto      thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    // Tile coordinate of each accumulator element: its column selects the scale, its row the store.
    Tensor tCcTile = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{})));

    auto work = scheduler.initial_work_tile_info();
    MUTLASS_PRAGMA_NO_UNROLL
    for (; work.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
      __musa_loop_transparent_outermost();
#endif
      auto rAcc     = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
      auto rPartial = make_fragment_like(rAcc);
      fill(rAcc, 0.f);

      MUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = 0; k_block < k_blocks; ++k_block) {
        pipeline.consumer_wait(pipe_read);
        pipeline_seq.wait();
        const int stage = int(pipe_read.index());

        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
        MUTE_UNROLL
        for (int kb = 0; kb < size<2>(tCrA); ++kb) {
          mute::gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage), rPartial);
          tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
        }
        mate::warpsquad_commit_batch();
        pipeline_seq.arrive();

        mate::warpsquad_wait();

        // The scales live in the stage, so they are read before it is released.
        if constexpr (IsZeroPoints) {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(rAcc); ++i) {
            const int col = int(get<1>(tCcTile(i)));
            rAcc(i)       = fmaf(rPartial(i), float(sScale(col, 0, stage)), rAcc(i));
            rAcc(i)       = fmaf(float(sScale(col, 1, stage)), sRowSum(int(get<0>(tCcTile(i))), 0, stage), rAcc(i));
          }
        } else {
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(rAcc); ++i) {
            rAcc(i) = fmaf(rPartial(i), float(sScale(int(get<1>(tCcTile(i))), 0, stage)), rAcc(i));
          }
        }
        pipeline.consumer_release(pipe_read);
        ++pipe_read;
      }

      const int m0 = int(work.M_idx) * BlockM;
      const int n0 = int(work.N_idx) * BlockN;
      auto      mD = make_tensor(make_gmem_ptr(params.ptr_d), make_shape(params.m, params.n, 1), params.stride_d);
      auto      gD = local_tile(mD(_, _, _0{}), make_shape(Int<BlockM>{}, Int<BlockN>{}), make_coord(int(work.M_idx), int(work.N_idx)));
      auto      tCgD = thr_mma.partition_C(gD);
      if (m0 + BlockM <= params.m && n0 + BlockN <= params.n) {
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(rAcc); ++i) {
          tCgD(i) = ElementD(rAcc(i));
        }
      } else {
        // Fragment element 8r + j is the j-th column of the thread's r-th row, so rows are checked
        // once each and columns only on the tile at the N edge.
        static_assert(decltype(size(rAcc))::value == 64, "one 128x64 atom: 8 rows x 8 columns per thread");
        const int  rows_left  = params.m - m0;
        const bool full_width = n0 + BlockN <= params.n;
        MUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < 8; ++r) {
          if (int(get<0>(tCcTile(r * 8))) < rows_left) {
            MUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 8; ++j) {
              if (full_width || n0 + int(get<1>(tCcTile(r * 8 + j))) < params.n) {
                tCgD(r * 8 + j) = ElementD(rAcc(r * 8 + j));
              }
            }
          }
        }
      }

      scheduler.advance_to_next_work();
      work = scheduler.get_work_tile_info();
    }
  }
};

}  // namespace mate::gemm::dense_w4a16_prefill_gemm
