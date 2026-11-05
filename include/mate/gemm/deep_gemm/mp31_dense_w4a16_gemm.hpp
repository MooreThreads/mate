#pragma once

#include <musa_burst.h>
#include <mutlass/bfloat16.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/algorithm/clear.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/gemm/deep_gemm/gemm_type.hpp"
#include "mate/gemm/deep_gemm/mp31_scheduler.hpp"

namespace mate::gemm::dense_w4a16_gemm {

using namespace mute;

// Stored weight nibbles are signed INT4 whose value is `unsigned_nibble - kWeightNibbleBias`, so the
// atom's signed INT4 product already carries the shared bias of every group. One signed INT4 operand
// therefore covers the symmetric case, where the dequant is exactly `scale * q` because the zero
// point equals the bias, and the asymmetric case, where the remaining offset
// `(kWeightNibbleBias - zero_point)` is a per-group rank-1 correction on the reduction of A.
inline constexpr int kWeightNibbleBias = 8;

// Act order (g_idx) permutes K, which smem load cannot express: a K tile is copied contiguously with
// no per-element index to hook. The permutation must be materialized into the packed B operand
// before launch; this type is the single device-side marker of that contract.
struct GidxKPermuteHook {
  static constexpr bool     IsIdentityOnly = true;
  MUTLASS_DEVICE static int apply(int k_index) {
    return k_index;
  }
};

// What a split launch needs on the device. It is a *base* of `Params` only in the split
// instantiation: measured on the whole-K path, 24 unused bytes of kernel argument cost 0.1% on the
// K=8960 shapes and 0.6-1.0% on the K=1536 ones, which is most of the regression this switch removes.
struct Mp31SplitParamsFields {
  float*  ptr_partial;
  int64_t partial_split_stride;
  int     k_split;
};

// Empty, and an empty base occupies no space, so the whole-K `Params` is the original struct.
struct Mp31NoSplitParamsFields {};

// fp16 activations, signed INT4 weights, one fp16 scale per K128 weight group. Zero points are
// optional: without them the kernel is the symmetric fast path, which computes no correction.
template <class ElementA_,
          class ElementB_,
          class ElementScaleB_,
          class ElementZeroB_,
          class ElementRowSum_,
          class ElementD_,
          class TileShape_,
          uint32_t Stages_,
          bool     ZeroPoints_,
          bool     kSplit_    = false,
          class GidxPermute_ = GidxKPermuteHook>
struct Mp31DenseW4A16Gemm {
  using ElementA      = ElementA_;
  using ElementB      = ElementB_;
  using ElementScaleB = ElementScaleB_;
  using ElementZeroB  = ElementZeroB_;
  using ElementRowSum = ElementRowSum_;
  using ElementMmaA   = ElementA_;
  using ElementD      = ElementD_;
  using TileShape     = TileShape_;
  using GidxPermute   = GidxPermute_;

  static constexpr bool IsZeroPoints = ZeroPoints_;

  // Compile-time K-split switch. `false` is the whole-K launch this kernel always was: no K range
  // indirection, no split branch, no second store path, and the persistent scheduler walk verbatim.
  // A split launch costs a couple of percent on the whole-K path when the two share one instantiation
  // (measured through a captured graph at cc=8: 933 vs 960 tok/s), so the two do not share one.
  static constexpr bool KSplit = kSplit_;

  // Act order is a property of the packed operand, not of this kernel's indexing: the K extent of a
  // tile is copied contiguously, so a permutation has to be materialized before launch.
  static_assert(GidxPermute::IsIdentityOnly, "Act order (g_idx) must be applied to the packed weights");

  using StrideA      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideB      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideScaleB = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideZeroB  = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideRowSum = decltype(make_stride(int64_t{}, _1{}, int64_t{}));
  using StrideD      = decltype(make_stride(int64_t{}, _1{}, int64_t{}));

  static constexpr int BlockM             = size<0>(TileShape{});
  static constexpr int BlockN             = size<1>(TileShape{});
  static constexpr int BlockK             = size<2>(TileShape{});
  static constexpr int ScaleBlockK        = 128;
  static constexpr int ScaleBlocksPerTile = BlockK / ScaleBlockK;
  static constexpr int TmeScaleKBlocks    = 8;
  static constexpr int SqmmasPerScale     = 2;
  static constexpr int Stages             = int(Stages_);

  static_assert(BlockK == 256, "This kernel requires a K256 tile");
  static_assert(ScaleBlocksPerTile == 2, "Each K256 mainloop tile must contain two K128 weight-scale blocks");
  static_assert(TmeScaleKBlocks * int(sizeof(ElementScaleB)) >= 16,
                "Scale-B TME loads require at least 16 contiguous bytes");
  static_assert(TmeScaleKBlocks * int(sizeof(ElementZeroB)) >= 16,
                "Zero-B TME loads require at least 16 contiguous bytes");
  // The stage count is bounded by shared memory and the async-barrier budget, both asserted below.
  static_assert(Stages >= 2, "The W4A16 K256 pipeline needs at least two stages");
  static_assert(is_same_v<ElementA, mutlass::half_t>, "The A operand must use FP16 elements end to end");
  static_assert(sizeof_bits_v<ElementMmaA> == 16, "The A operand must use FP16 elements end to end");
  static_assert(is_same_v<ElementB, int4_t>, "The B operand must use signed INT4 elements");
  static_assert(is_same_v<ElementScaleB, mutlass::half_t>, "B scales must use FP16 elements");
  static_assert(!IsZeroPoints || is_same_v<ElementZeroB, mutlass::half_t>,
                "Zero-point corrections must use FP16 elements");
  static_assert(is_same_v<ElementRowSum, float>, "The A reduction must accumulate in FP32");

  static constexpr TCE::Major SqmmaMajorA = TCE::Major::K;
  static constexpr TCE::Major SqmmaMajorB = TCE::Major::K;

  using SqmmaOp =
      decltype(MP31::SQMMA::
                   ss_op_selector<ElementMmaA, ElementB, float, TileShape, SqmmaMajorA, SqmmaMajorB, _32, _128>());
  // Two atoms side by side along N only when the tile is at least two atoms wide. A tile exactly one
  // atom wide would otherwise hand the second consumer squad the same A, B and C partitions as the
  // first, so it would issue every SQMMA and store every output a second time.
  static constexpr int AtomN       = size<1>(typename MMA_Traits<SqmmaOp>::Shape_MNK{});
  static constexpr int AtomsAlongN = BlockN >= 2 * AtomN ? 2 : 1;
  static_assert(BlockN % (AtomN * AtomsAlongN) == 0, "BlockN must be a whole number of atoms");
  using AtomLayout = decltype(make_layout(Shape<_1, Int<AtomsAlongN>, _1>{}, LayoutRight{}));
  using TiledMma   = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomMmaA =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<SqmmaMajorA, ElementMmaA, SqmmaOp, TileShape>());
  using SmemLayoutAtomB =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<SqmmaMajorB, ElementB, SqmmaOp, TileShape>());
  using SmemLayoutMmaA =
      decltype(tile_to_shape(SmemLayoutAtomMmaA{}, make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<Stages>{})));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<Stages>{})));

  using SmemLayoutA =
      Layout<Shape<Int<BlockM>, Int<BlockK>, Int<Stages>>, Stride<Int<BlockK>, _1, Int<BlockM * BlockK>>>;

  static constexpr TME::CacheHint TmeAInnerHint      = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeAOuterHint      = TME::CacheHint::CACHE_PERSIST;
  static constexpr TME::CacheHint TmeBInnerHint      = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeBOuterHint      = TME::CacheHint::CACHE_NONE;
  static constexpr TME::CacheHint TmeScaleBInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeScaleBOuterHint = TME::CacheHint::CACHE_PERSIST;

  using TME_A = decltype(make_tme_copy<TmeAInnerHint, TmeAOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{})));

  using TME_B = decltype(make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
      take<0, 2>(SmemLayoutB{}),
      make_shape(Int<BlockN>{}, Int<BlockK>{})));

  static constexpr int      SmemAlignmentBytes           = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor   = 1;
  static constexpr uint32_t Num1DBlocksPerSchedulerGroup = 4;

  static constexpr uint32_t NumProducerWarpSquads  = 1;
  static constexpr uint32_t NumCasterWarpSquads    = 2;
  static constexpr uint32_t NumConsumerWarpSquads  = AtomsAlongN;
  static constexpr uint32_t NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr uint32_t NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t WarpsPerWarpSquad      = NumThreadsPerWarpSquad / NumThreadsPerWarp;
  static constexpr uint32_t NumCasterThreads       = NumCasterWarpSquads * NumThreadsPerWarpSquad;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumProducerWarpSquads + NumCasterWarpSquads + NumConsumerWarpSquads) * NumThreadsPerWarpSquad;

  static_assert(int(NumConsumerWarpSquads * NumThreadsPerWarpSquad) == int(size(TiledMma{})),
                "The MMA thread layout must cover every consumer warp squad");

  static constexpr int MaxBarriersPerStage = 2;
  using PipelineBarrierRatio =
      mutlass::Mp31PipelineWarpSpecializedBarrierRatio<MaxBarriersPerStage, 0, Stages, Stages, Stages>;
  static constexpr int BarriersPerStage = PipelineBarrierRatio::value;
  static_assert(BarriersPerStage > 0, "A, B, cast, and Scale-B pipelines exceed the MP31 async-barrier budget");

  using PipelineLoadA       = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineLoadB       = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineCastA       = mutlass::Mp31PipelineAsyncWarpSpecialized<Stages, BarriersPerStage>;
  using PipelineLoadAParams = typename PipelineLoadA::Params;
  using PipelineLoadBParams = typename PipelineLoadB::Params;
  using PipelineCastAParams = typename PipelineCastA::Params;
  using PipelineLoadAState  = typename PipelineLoadA::PipelineState;
  using PipelineLoadBState  = typename PipelineLoadB::PipelineState;
  using PipelineCastAState  = typename PipelineCastA::PipelineState;

  static constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr int TmeTransactionBytesB =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * sizeof_bits_v<ElementB>);
  static constexpr int MaxSharedStorageBytes = 192 * 1024;
  // Per-m operand of the scale step: the reduction of A over each K128 weight group, produced by the
  // A prepare stage, so it is staged on the cast pipeline rather than fetched from global memory.
  using SmemLayoutRowSum = Layout<Shape<Int<BlockM>, Int<ScaleBlocksPerTile>, Int<Stages>>,
                                  Stride<Int<ScaleBlocksPerTile>, _1, Int<BlockM * ScaleBlocksPerTile>>>;
  using SmemLayoutScaleB = Layout<Shape<Int<BlockN>, Int<TmeScaleKBlocks>, Int<Stages>>,
                                  Stride<Int<TmeScaleKBlocks>, _1, Int<BlockN * TmeScaleKBlocks>>>;
  using TME_ScaleB       = decltype(make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(nullptr)),
                  make_shape(int64_t{}, int64_t{}, int64_t{}),
                  make_stride(int64_t{}, _1{}, int64_t{})),
      take<0, 2>(SmemLayoutScaleB{}),
      make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{})));
  static constexpr int TmeTransactionBytesScaleB =
      mutlass::bits_to_bytes(BlockN * TmeScaleKBlocks * sizeof_bits_v<ElementScaleB>);
  using SmemLayoutZeroB = SmemLayoutScaleB;
  using TME_ZeroB       = decltype(make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementZeroB const*>(nullptr)),
                  make_shape(int64_t{}, int64_t{}, int64_t{}),
                  make_stride(int64_t{}, _1{}, int64_t{})),
      take<0, 2>(SmemLayoutZeroB{}),
      make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{})));
  static constexpr int TmeTransactionBytesZeroB =
      IsZeroPoints ? mutlass::bits_to_bytes(BlockN * TmeScaleKBlocks * sizeof_bits_v<ElementZeroB>) : 0;

  using Scheduler = mate::deep_gemm::detail::
      Mp31PersistentTileScheduler<mate::deep_gemm::GemmType::Normal, BlockM, BlockN, Num1DBlocksPerSchedulerGroup>;
  using WorkTileInfo = typename Scheduler::WorkTileInfo;

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineLoadA[PipelineLoadA::NumBarriers];
    uint8_t PipelineLoadB[PipelineLoadB::NumBarriers];
    uint8_t PipelineCastA[PipelineCastA::NumBarriers];
  };
  static_assert(sizeof(BarrierStorage) + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount <=
                    mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers,
                "Async barrier storage exceeds the MP31 hardware limit");

  // The rank-1 zero-point correction stages its own operands: the per-K128 reductions of A and the
  // folded correction that multiplies them. The symmetric build has no correction, so it must not pay
  // shared memory for one. The empty alternative keeps the member names present without storage, which
  // turns any access that escapes its `if constexpr` into a compile error instead of a write into a
  // placeholder.
  struct ZeroPointStorage {
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementRowSum, cosize_v<SmemLayoutRowSum>> smem_row_sum;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementZeroB, cosize_v<SmemLayoutZeroB>> smem_zero_b;
  };
  struct EmptyZeroPointStorage {
    struct Unavailable {
      MUTLASS_DEVICE void* begin() = delete;
    };
    Unavailable smem_row_sum;
    Unavailable smem_zero_b;
  };
  using ZeroPointStorageT = std::conditional_t<IsZeroPoints, ZeroPointStorage, EmptyZeroPointStorage>;

  struct SharedStorage {
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementMmaA, cosize_v<SmemLayoutMmaA>> smem_mma_a;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> smem_b;
    MUTE_ALIGNAS(SmemAlignmentBytes) ZeroPointStorageT zero_point;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementScaleB, cosize_v<SmemLayoutScaleB>> smem_scale_b;
    MUTE_ALIGNAS(SmemAlignmentBytes) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> smem_a;
  };

  struct Arguments {
    ElementA const*      ptr_a       = nullptr;
    ElementB const*      ptr_b       = nullptr;
    ElementScaleB const* ptr_scale_b = nullptr;
    // Per-(n, K128 group) `(kWeightNibbleBias - zero_point) * scale`, already folded into the
    // magnitude of the scale it corrects. Unused when IsZeroPoints is false.
    ElementZeroB const* ptr_zero_b = nullptr;
    ElementD*           ptr_d      = nullptr;

    StrideA      stride_a;
    StrideB      stride_b;
    StrideScaleB stride_scale_b;
    StrideZeroB  stride_zero_b;
    StrideD      stride_d;

    // K-split output. When `k_split > 1` each split writes its own fp32 partial accumulator here and
    // `ptr_d` is left untouched, so the reduction across splits stays outside the kernel and needs no
    // atomics. The buffer is `[k_split, m_pad, n]` fp32, n-contiguous, so a split's plane starts
    // `partial_split_stride` elements after the previous one. Both are unused when `k_split == 1`.
    float*  ptr_partial          = nullptr;
    int64_t partial_split_stride = 0;
    int     k_split              = 1;

    int m       = 0;
    int n       = 0;
    int k       = 0;
    int num_mps = 0;
  };

  // The split payload rides in the base, which is empty for the whole-K instantiation: every field
  // below keeps its offset and the argument block is the one this kernel always took.
  struct Params : std::conditional_t<KSplit, Mp31SplitParamsFields, Mp31NoSplitParamsFields> {
    TME_A      tme_a;
    TME_B      tme_b;
    TME_ScaleB tme_scale_b;
    TME_ZeroB  tme_zero_b;
    ElementD*  ptr_d;

    StrideD stride_d;

    int m;
    int n;
    int k;
    int num_mps;
    int k_blocks;
    int scale_k_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int k_blocks       = ceil_div(args.k, BlockK);
    const int scale_k_blocks = k_blocks * ScaleBlocksPerTile;
    const int k_split        = args.k_split > 0 ? args.k_split : 1;

    auto  gA    = make_tensor(make_gmem_ptr(args.ptr_a), make_shape(args.m, args.k, 1), args.stride_a);
    TME_A tme_a = make_tme_copy<TmeAInnerHint, TmeAOuterHint>(MP31_TME_LOAD{}, gA, take<0, 2>(SmemLayoutA{}));

    auto gB = make_tensor(
        make_gmem_ptr<ElementB>(static_cast<void const*>(args.ptr_b)), make_shape(args.n, args.k, 1), args.stride_b);
    TME_B tme_b = make_tme_copy<TmeBInnerHint, TmeBOuterHint, uint8_t>(
        MP31_TME_LOAD{}, gB, take<0, 2>(SmemLayoutB{}), make_shape(Int<BlockN>{}, Int<BlockK>{}));

    auto       scale_b_layout = make_layout(make_shape(args.n, scale_k_blocks, 1), args.stride_scale_b);
    auto       gScaleB        = make_tensor(make_gmem_ptr(args.ptr_scale_b), scale_b_layout);
    TME_ScaleB tme_scale_b    = make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
        MP31_TME_LOAD{}, gScaleB, take<0, 2>(SmemLayoutScaleB{}), make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}));

    auto      zero_b_layout = make_layout(make_shape(args.n, scale_k_blocks, 1), args.stride_zero_b);
    auto      gZeroB        = make_tensor(make_gmem_ptr(args.ptr_zero_b), zero_b_layout);
    TME_ZeroB tme_zero_b    = make_tme_copy<TmeScaleBInnerHint, TmeScaleBOuterHint>(
        MP31_TME_LOAD{}, gZeroB, take<0, 2>(SmemLayoutZeroB{}), make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}));

    // The base initialises first, and there are two of them, so the list is spelled once per
    // instantiation. Everything after the base is identical.
    if constexpr (KSplit) {
      return Params{
          Mp31SplitParamsFields{k_split > 1 ? args.ptr_partial : nullptr, args.partial_split_stride, k_split},
          tme_a,
          tme_b,
          tme_scale_b,
          tme_zero_b,
          args.ptr_d,
          args.stride_d,
          args.m,
          args.n,
          args.k,
          args.num_mps,
          k_blocks,
          scale_k_blocks,
      };
    } else {
      return Params{
          Mp31NoSplitParamsFields{},
          tme_a,
          tme_b,
          tme_scale_b,
          tme_zero_b,
          args.ptr_d,
          args.stride_d,
          args.m,
          args.n,
          args.k,
          args.num_mps,
          k_blocks,
          scale_k_blocks,
      };
    }
  }

  static dim3 get_grid_shape_persistent(Params const& params) {
    const int num_mps = params.num_mps > 0 ? params.num_mps : 1;
    // A split launch has `tiles * k_split` units of work, each of which streams a fixed K range,
    // so the grid is the number of units when the machine is wider than the problem. That is the
    // whole point: without the split, N=1536 at M=8 leaves 24 of the 60 MPs with nothing to do.
    if constexpr (KSplit) {
      const int tiles     = ceil_div(params.m, BlockM) * ceil_div(params.n, BlockN);
      const int num_units = tiles * params.k_split;
      return dim3(static_cast<uint32_t>(num_units < num_mps ? num_units : num_mps), 1u, 1u);
    }
    return dim3(static_cast<uint32_t>(num_mps), 1u, 1u);
  }

  // The slice of the K axis one split owns. Every split takes `chunk` K blocks of the shared axis, so
  // the last one may be short (and empty if `k_split` exceeds the block count). The load, cast and
  // compute pipelines all walk exactly this range, which is what keeps them in lockstep; the scale and
  // zero-point indexing inside the loop stays absolute, so each partial is a valid partial sum.
  struct KRange {
    int begin;
    int end;
  };

  MUTLASS_DEVICE static KRange k_range_for(Params const& params, int split_id) {
    const int k_split = params.k_split > 0 ? params.k_split : 1;
    const int chunk   = ceil_div(params.k_blocks, k_split);
    const int begin   = split_id * chunk;
    const int end     = begin + chunk < params.k_blocks ? begin + chunk : params.k_blocks;
    return KRange{begin, end};
  }

  // A split launch numbers its work units itself, because the tile scheduler has no split axis. The
  // tile index covers the output tiles in the same row-major (m block, n block) order the scheduler's
  // flat block index uses, and the record is the one the scheduler builds for a Normal-type problem.
  MUTLASS_DEVICE static WorkTileInfo unit_work_tile(Params const& params, int tile_idx) {
    const int      num_m_blocks = ceil_div(params.m, BlockM);
    const int      m_block_idx  = tile_idx % num_m_blocks;
    const int      n_block_idx  = tile_idx / num_m_blocks;
    const uint32_t m_row_offset = uint32_t(m_block_idx * BlockM);
    return WorkTileInfo{int32_t(m_block_idx), int32_t(n_block_idx), 0, m_row_offset, true, true};
  }

  MUTLASS_DEVICE
  static void load(Params const&       params,
                   SharedStorage&      shared_storage,
                   PipelineLoadA&      pipeline_load_a,
                   PipelineLoadB&      pipeline_load_b,
                   PipelineLoadAState& pipe_a_write,
                   PipelineLoadBState& pipe_b_write,
                   WorkTileInfo const& work_tile,
                   int                 warp_idx_in_squad,
                   int                 split_id) {
    const bool is_tme_issue_warp = warp_idx_in_squad == 0;

    Tensor sA      = make_tensor(make_smem_ptr(shared_storage.smem_a.begin()), SmemLayoutA{});
    Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_b.begin()), SmemLayoutB{});
    Tensor sScaleB = make_tensor(make_smem_ptr(shared_storage.smem_scale_b.begin()), SmemLayoutScaleB{});

    auto cta_tme_a       = params.tme_a.get_slice(0);
    auto cta_tme_b       = params.tme_b.get_slice(0);
    auto cta_tme_scale_b = params.tme_scale_b.get_slice(0);
    auto cta_tme_zero_b  = params.tme_zero_b.get_slice(0);

    const uint32_t m_block_idx = static_cast<uint32_t>(work_tile.M_idx);
    const uint32_t n_block_idx = static_cast<uint32_t>(work_tile.N_idx);

    auto mA  = params.tme_a.get_tme_tensor(make_shape(params.m, params.k, 1))(_, _, _0{});
    auto gA_ = local_tile(mA, make_shape(Int<BlockM>{}, Int<BlockK>{}), make_coord(_, _));

    auto mB  = params.tme_b.get_tme_tensor(make_shape(params.n, params.k, 1))(_, _, _0{});
    auto gB_ = local_tile(mB, make_shape(Int<BlockN>{}, Int<BlockK>{}), make_coord(_, _));

    Tensor gA = gA_(_, _, m_block_idx, _);
    Tensor gB = gB_(_, _, n_block_idx, _);

    Tensor tAgA = cta_tme_a.partition_S(gA);
    Tensor tAsA = cta_tme_a.partition_D(sA);
    Tensor tBgB = cta_tme_b.partition_S(gB);
    Tensor tBsB = cta_tme_b.partition_D(sB);

    auto   mScaleB   = params.tme_scale_b.get_tme_tensor(make_shape(params.n, params.scale_k_blocks, 1))(_, _, _0{});
    Tensor tSsScaleB = cta_tme_scale_b.partition_D(sScaleB);

    auto mZeroB = params.tme_zero_b.get_tme_tensor(make_shape(params.n, params.scale_k_blocks, 1))(_, _, _0{});

    // The whole-K instantiation folds these back into the `0 .. params.k_blocks` loop it has always
    // run; the split instantiation walks the slice its split owns.
    int k_begin = 0;
    int k_end   = params.k_blocks;
    if constexpr (KSplit) {
      const KRange slice = k_range_for(params, split_id);
      k_begin            = slice.begin;
      k_end              = slice.end;
    }
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_begin; k_block < k_end; ++k_block) {
      if (is_tme_issue_warp) {
        pipeline_load_a.producer_acquire(pipe_a_write);
        const uint32_t barrier_id_a = pipeline_load_a.producer_get_barrier_id(pipe_a_write);
        copy(params.tme_a.with(barrier_id_a), tAgA(_, _, _, k_block), tAsA(_, _, _, int(pipe_a_write.index())));

        pipeline_load_b.producer_acquire(pipe_b_write);
        const uint32_t barrier_id_b = pipeline_load_b.producer_get_barrier_id(pipe_b_write);
        auto           mScaleBAtK   = domain_offset(make_coord(_0{}, k_block * ScaleBlocksPerTile), mScaleB);
        Tensor         gScaleB =
            local_tile(mScaleBAtK, make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}), make_coord(n_block_idx, _0{}));
        Tensor tSgScaleB = cta_tme_scale_b.partition_S(gScaleB);
        copy(params.tme_scale_b.with(barrier_id_b), tSgScaleB, tSsScaleB(_, _, _, int(pipe_b_write.index())));

        if constexpr (IsZeroPoints) {
          Tensor sZeroB = make_tensor(make_smem_ptr(shared_storage.zero_point.smem_zero_b.begin()), SmemLayoutZeroB{});
          Tensor tSsZeroB  = cta_tme_zero_b.partition_D(sZeroB);
          auto   mZeroBAtK = domain_offset(make_coord(_0{}, k_block * ScaleBlocksPerTile), mZeroB);
          Tensor gZeroB =
              local_tile(mZeroBAtK, make_shape(Int<BlockN>{}, Int<TmeScaleKBlocks>{}), make_coord(n_block_idx, _0{}));
          Tensor tSgZeroB = cta_tme_zero_b.partition_S(gZeroB);
          copy(params.tme_zero_b.with(barrier_id_b), tSgZeroB, tSsZeroB(_, _, _, int(pipe_b_write.index())));
        }

        copy(params.tme_b.with(barrier_id_b), tBgB(_, _, _, k_block), tBsB(_, _, _, int(pipe_b_write.index())));

        ++pipe_a_write;
        ++pipe_b_write;
      }
    }
  }

  // A is fp16 end to end, so this stage only restages the tile into the MMA smem layout and produces
  // the per-K128 reductions of A. The reductions ride the cast pipeline, so the consumer observes them
  // for the same stage as the tile they belong to.
  MUTLASS_DEVICE
  static void prepare_a(Params const&       params,
                        SharedStorage&      shared_storage,
                        PipelineLoadA&      pipeline_load_a,
                        PipelineCastA&      pipeline_cast_a,
                        PipelineLoadAState& pipe_a_read,
                        PipelineCastAState& pipe_cast_a_write,
                        int                 caster_thread_idx,
                        int                 split_id) {
    Tensor sA    = make_tensor(make_smem_ptr(shared_storage.smem_a.begin()), SmemLayoutA{});
    Tensor sMmaA = make_tensor(make_smem_ptr(shared_storage.smem_mma_a.begin()), SmemLayoutMmaA{});

    using StoreVector               = uint128_t;
    constexpr int ElementsPerVector = sizeof(StoreVector) / sizeof(ElementA);
    constexpr int ThreadsPerRow     = int(NumCasterThreads) / BlockM;
    constexpr int VectorsPerRow     = BlockK / ElementsPerVector;
    constexpr int VectorsPerThread  = VectorsPerRow / ThreadsPerRow;
    constexpr int ElementsPerThread = VectorsPerThread * ElementsPerVector;
    constexpr int ThreadsPerGroup   = ScaleBlockK / ElementsPerThread;
    static_assert(is_same_v<ElementA, ElementMmaA>, "The A prepare stage performs no type conversion");
    static_assert(int(NumCasterThreads) % BlockM == 0);
    static_assert(VectorsPerRow % ThreadsPerRow == 0);
    static_assert(sizeof(StoreVector) / sizeof(ElementMmaA) == ElementsPerVector);
    // A thread run that straddles a K128 boundary would need two reductions in one accumulator.
    static_assert(ScaleBlockK < ElementsPerThread || ScaleBlockK % ElementsPerThread == 0);
    static_assert(ThreadsPerRow % ThreadsPerGroup == 0, "The threads of a row must tile the K128 groups");
    static_assert(ScaleBlocksPerTile == ThreadsPerRow / ThreadsPerGroup,
                  "The row reduction must produce exactly one value per K128 weight group");

    using StoreCopyAtom     = Copy_Atom<UniversalCopy<StoreVector>, ElementMmaA>;
    using StoreThreadLayout = Layout<Shape<Int<BlockM>, Int<ThreadsPerRow>>, Stride<Int<ThreadsPerRow>, _1>>;
    using StoreValueLayout  = Layout<Shape<_1, Int<ElementsPerThread>>>;
    using StoreTiledCopy    = decltype(make_tiled_copy(StoreCopyAtom{}, StoreThreadLayout{}, StoreValueLayout{}));

    const int m_idx               = caster_thread_idx / ThreadsPerRow;
    const int thread_idx_in_row   = caster_thread_idx % ThreadsPerRow;
    const int first_vector_in_row = thread_idx_in_row * VectorsPerThread;

    StoreTiledCopy tiled_copy_store;
    auto           thr_copy_store = tiled_copy_store.get_thread_slice(caster_thread_idx);

    Tensor rA                  = make_tensor<StoreVector>(Shape<Int<VectorsPerThread>>{});
    Tensor rAArray             = recast<ElementMmaA>(rA);
    auto   tCsMmaAStorePattern = thr_copy_store.partition_D(sMmaA(_, _, _0{}));
    auto   tCrMmaAStore        = make_fragment_like<ElementMmaA>(tCsMmaAStorePattern);
    Tensor rMmaAArray          = recast<StoreVector>(tCrMmaAStore);
    MUTE_STATIC_ASSERT_V(size(rMmaAArray) == Int<VectorsPerThread>{});

    // The whole-K instantiation folds these back into the `0 .. params.k_blocks` loop it has always
    // run; the split instantiation walks the slice its split owns.
    int k_begin = 0;
    int k_end   = params.k_blocks;
    if constexpr (KSplit) {
      const KRange slice = k_range_for(params, split_id);
      k_begin            = slice.begin;
      k_end              = slice.end;
    }
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_begin; k_block < k_end; ++k_block) {
      pipeline_load_a.consumer_wait(pipe_a_read);

      Tensor sAVec = recast<StoreVector>(sA(_, _, int(pipe_a_read.index())));
      MUTE_UNROLL
      for (int vector_idx = 0; vector_idx < VectorsPerThread; ++vector_idx) {
        UniversalCopy<StoreVector>::copy(sAVec(m_idx, first_vector_in_row + vector_idx), rA(vector_idx));
      }

      __syncwarp();
      pipeline_load_a.consumer_release(pipe_a_read);
      ++pipe_a_read;

      MUTE_UNROLL
      for (int vector_idx = 0; vector_idx < VectorsPerThread; ++vector_idx) {
        rMmaAArray(vector_idx) = rA(vector_idx);
      }

      pipeline_cast_a.producer_acquire(pipe_cast_a_write);

      const int stage        = int(pipe_cast_a_write.index());
      auto      tCsMmaAStore = thr_copy_store.partition_D(sMmaA(_, _, stage));
      copy(tiled_copy_store, tCrMmaAStore, tCsMmaAStore);

      // The reduction feeds the zero-point correction alone, so the symmetric build neither computes nor
      // stages it. It is published before the stage is committed, so the consumer reads it together with
      // the tile it belongs to.
      if constexpr (IsZeroPoints) {
        float row_sum = 0.0f;
        MUTE_UNROLL
        for (int element_idx = 0; element_idx < ElementsPerThread; ++element_idx) {
          row_sum += float(rAArray(element_idx));
        }
        MUTLASS_PRAGMA_UNROLL
        for (int offset = 1; offset < ThreadsPerGroup; offset <<= 1) {
          row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
        }
        if (thread_idx_in_row % ThreadsPerGroup == 0) {
          Tensor sRowSum =
              make_tensor(make_smem_ptr(shared_storage.zero_point.smem_row_sum.begin()), SmemLayoutRowSum{});
          sRowSum(m_idx, thread_idx_in_row / ThreadsPerGroup, stage) = row_sum;
        }
      }

      __syncwarp();
      pipeline_cast_a.producer_commit(pipe_cast_a_write);
      ++pipe_cast_a_write;
    }
  }

  // The one store shape every path shares: rAcc into `dst` (fp16/bf16 for a whole-K launch, fp32 for a
  // split partial) with the same M/N indexing, bounds test and tail masking. `dst` and its stride are
  // the only things that differ between the two, so a split partial is stored by exactly the code the
  // final output is stored by.
  template <class AccumTensor, class ElementStore, class StrideStore>
  MUTLASS_DEVICE static void store_tile(ElementStore*       dst,
                                        StrideStore const&  stride_dst,
                                        Params const&       params,
                                        AccumTensor const&  rAcc,
                                        WorkTileInfo const& work_tile,
                                        int                 consumer_thread_idx) {
    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    auto mD  = make_tensor(make_gmem_ptr(dst), make_shape(params.m, params.n, 1), stride_dst);
    auto gD_ = local_tile(mD, make_shape(Int<BlockM>{}, Int<BlockN>{}, _1{}), make_coord(_, _, _));
    auto gD  = gD_(_, _, _0{}, int(work_tile.M_idx), int(work_tile.N_idx), _0{});

    auto tCgD = thr_mma.partition_C(gD);
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCgD));

    const uint32_t m0 = uint32_t(work_tile.M_idx) * uint32_t(BlockM);
    const uint32_t n0 = uint32_t(work_tile.N_idx) * uint32_t(BlockN);
    if (m0 + uint32_t(BlockM) <= uint32_t(params.m) && n0 + uint32_t(BlockN) <= uint32_t(params.n)) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        tCgD(i) = ElementStore(rAcc(i));
      }
    } else {
      auto cD   = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{}));
      auto tCcD = thr_mma.partition_C(cD);

      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(rAcc); ++i) {
        const auto     coord   = tCcD(i);
        const uint32_t coord_m = m0 + uint32_t(get<0>(coord));
        const uint32_t coord_n = n0 + uint32_t(get<1>(coord));
        if (coord_m < uint32_t(params.m) && coord_n < uint32_t(params.n)) {
          tCgD(i) = ElementStore(rAcc(i));
        }
      }
    }
  }

  template <class AccumTensor>
  MUTLASS_DEVICE static void epilogue(Params const&       params,
                                      AccumTensor const&  rAcc,
                                      WorkTileInfo const& work_tile,
                                      int                 consumer_thread_idx,
                                      int                 split_id) {
    // A split launch accumulates fp32 partials only: the cross-split reduction happens after the
    // kernel, so `ptr_d` is not touched at all. The whole-K instantiation compiles just the store it
    // always had -- one path, not a branch between two copies of the tail logic.
    if constexpr (KSplit) {
      store_tile(params.ptr_partial + int64_t(split_id) * params.partial_split_stride,
                 make_stride(params.n, _1{}, int64_t(0)),
                 params,
                 rAcc,
                 work_tile,
                 consumer_thread_idx);
    } else {
      (void)split_id;
      store_tile(params.ptr_d, params.stride_d, params, rAcc, work_tile, consumer_thread_idx);
    }
  }

  // One packed word carries a 16-bit group value for each K128 weight block of the K256 tile: the low
  // half is the first block, the high half the second.
  template <class ElementPacked, int ScaleIndex, class PackedTensor>
  MUTLASS_DEVICE static float4 unpack_packed_bst4(PackedTensor const& packed, int i) {
    constexpr int Shift = ScaleIndex * 16;
    return make_float4(float(ElementPacked::bitcast(uint16_t(packed(i + 0) >> Shift))),
                       float(ElementPacked::bitcast(uint16_t(packed(i + 1) >> Shift))),
                       float(ElementPacked::bitcast(uint16_t(packed(i + 2) >> Shift))),
                       float(ElementPacked::bitcast(uint16_t(packed(i + 3) >> Shift))));
  }

  template <int ScaleIndex, class ScaleTensor, class PackedScaleBTensor>
  MUTLASS_DEVICE static void unpack_scale_bst4(ScaleTensor& scale, PackedScaleBTensor const& scale_b_packed) {
    static_assert(ScaleIndex == 0 || ScaleIndex == 1);
    static_assert(size(ScaleTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(scale); i += 4) {
      const float4 scale_vec = unpack_packed_bst4<ElementScaleB, ScaleIndex>(scale_b_packed, i);
      scale(i + 0)           = scale_vec.x;
      scale(i + 1)           = scale_vec.y;
      scale(i + 2)           = scale_vec.z;
      scale(i + 3)           = scale_vec.w;
    }
  }

  // Rank-1 term of the asymmetric dequant, `(kWeightNibbleBias - zero_point) * reduce_k(a)`, for the
  // K128 group selected by ScaleIndex. `row_sum` is that reduction for the single tile row this
  // thread's C fragment covers, and the zero-point stream holds `(bias - zero_point) * scale` per
  // output column, so the product is the whole correction for every element of the group. The atom
  // already produced the signed INT4 product of the same group, so adding it is exact.
  template <int ScaleIndex, class CorrTensor, class RowSumTensor, class PackedZeroBTensor>
  MUTLASS_DEVICE static void calculate_correction_bst4(CorrTensor&              corr,
                                                       RowSumTensor const&      row_sum,
                                                       PackedZeroBTensor const& zero_b_packed) {
    static_assert(ScaleIndex == 0 || ScaleIndex == 1);
    static_assert(size(CorrTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");
    static_assert(size(CorrTensor{}) % size(RowSumTensor{}) == 0,
                  "the reduction must divide the accumulator fragment it corrects");

    // `row_sum` holds one reduction per row of the C fragment this thread owns, and the fragment
    // covers several columns of each of those rows, so each reduction is repeated across its columns.
    constexpr int ColumnsPerRow = size(CorrTensor{}) / size(RowSumTensor{});

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(corr); i += 4) {
      const float4 zero_vec = unpack_packed_bst4<ElementZeroB, ScaleIndex>(zero_b_packed, i);
      corr(i + 0)           = float(row_sum((i + 0) / ColumnsPerRow)) * zero_vec.x;
      corr(i + 1)           = float(row_sum((i + 1) / ColumnsPerRow)) * zero_vec.y;
      corr(i + 2)           = float(row_sum((i + 2) / ColumnsPerRow)) * zero_vec.z;
      corr(i + 3)           = float(row_sum((i + 3) / ColumnsPerRow)) * zero_vec.w;
    }
  }

  template <class AccumTensor, class ScaleTensor, class CorrTensor>
  MUTLASS_DEVICE static void scale_accumulate_bst4(AccumTensor&       accum,
                                                   AccumTensor const& accum_temp,
                                                   ScaleTensor const& scale,
                                                   CorrTensor const&  correction) {
    static_assert(size(AccumTensor{}) % 4 == 0, "BST4 scaling requires groups of four accumulators");

    auto accum_bst4      = recast<f4>(accum);
    auto accum_temp_bst4 = recast<f4>(accum_temp);
    auto scale_bst4      = recast<f4>(scale);
    auto corr_bst4       = recast<f4>(correction);

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_bst4); ++i) {
      if constexpr (IsZeroPoints) {
        accum_bst4(i) += accum_temp_bst4(i) * scale_bst4(i) + corr_bst4(i);
      } else {
        accum_bst4(i) += accum_temp_bst4(i) * scale_bst4(i);
      }
    }
  }

  MUTLASS_DEVICE
  static auto compute(Params const&       params,
                      SharedStorage&      shared_storage,
                      PipelineLoadB&      pipeline_load_b,
                      PipelineCastA&      pipeline_cast_a,
                      PipelineLoadBState& pipe_b_read,
                      PipelineCastAState& pipe_cast_a_read,
                      WorkTileInfo const& work_tile,
                      int                 consumer_thread_idx,
                      int                 split_id) {
    TiledMma tiled_mma;
    auto     thr_mma = tiled_mma.get_thread_slice(consumer_thread_idx);

    Tensor sMmaA   = make_tensor(make_smem_ptr(shared_storage.smem_mma_a.begin()), SmemLayoutMmaA{});
    Tensor sB      = make_tensor(make_smem_ptr(shared_storage.smem_b.begin()), SmemLayoutB{});
    Tensor sScaleB = make_tensor(make_smem_ptr(shared_storage.smem_scale_b.begin()), SmemLayoutScaleB{});

    auto rAcc           = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    auto rAccTempFirst  = make_fragment_like(rAcc);
    auto rAccTempSecond = make_fragment_like(rAcc);
    auto rScaleFirst    = make_fragment_like(rAcc);
    auto rScaleSecond   = make_fragment_like(rAcc);
    auto rCorrFirst     = make_fragment_like(rAcc);
    auto rCorrSecond    = make_fragment_like(rAcc);
    fill(rAcc, float(0));

    Tensor tCsA = thr_mma.partition_A(sMmaA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    MUTE_STATIC_ASSERT_V(size<2>(tCrA) == _4{});
    MUTE_STATIC_ASSERT_V(size<2>(tCrB) == _4{});

    using RowSumViewAsCLayout = Layout<Shape<Shape<_1, Int<BlockM>>, Int<BlockN>>, Stride<Stride<_0, _1>, _0>>;
    using ScaleBViewAsCLayout = Layout<Shape<Int<BlockM>, Shape<_1, Int<BlockN>>>, Stride<_0, Stride<_0, _1>>>;

    Tensor tCsScaleBTemplate       = thr_mma.partition_C(sScaleB(_, _0{}, _0{}).compose(ScaleBViewAsCLayout{}));
    Tensor tCrScaleBPacked         = make_tensor_like<uint32_t>(tCsScaleBTemplate);
    auto   tCrScaleBPackedFlat     = filter_zeros(tCrScaleBPacked);

    // The correction stream has the scale stream's shape, so its register state is derived from the
    // scale state and needs no shared-memory view of its own.
    Tensor tCrZeroBPacked         = make_tensor_like<uint32_t>(tCrScaleBPacked);
    auto   tCrZeroBPackedFlat     = filter_zeros(tCrZeroBPacked);

    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrScaleBPacked));
    MUTE_STATIC_ASSERT_V(size(rAcc) == size(tCrZeroBPacked));

    // The whole-K instantiation folds these back into the `0 .. params.k_blocks` loop it has always
    // run; the split instantiation walks the slice its split owns.
    int k_begin = 0;
    int k_end   = params.k_blocks;
    if constexpr (KSplit) {
      const KRange slice = k_range_for(params, split_id);
      k_begin            = slice.begin;
      k_end              = slice.end;
    }
    MUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_begin; k_block < k_end; ++k_block) {
      pipeline_cast_a.consumer_wait(pipe_cast_a_read);
      pipeline_load_b.consumer_wait(pipe_b_read);

      const int stage_a = int(pipe_cast_a_read.index());
      const int stage_b = int(pipe_b_read.index());

      tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      MUTLASS_PRAGMA_UNROLL
      for (int mma_k = 0; mma_k < SqmmasPerScale; ++mma_k) {
        mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAccTempFirst);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
      }
      ::mate::warpsquad_commit_batch();

      tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      MUTLASS_PRAGMA_UNROLL
      for (int mma_k = SqmmasPerScale; mma_k < ScaleBlocksPerTile * SqmmasPerScale; ++mma_k) {
        mute::gemm(tiled_mma, tCrA(_, _, mma_k, stage_a), tCrB(_, _, mma_k, stage_b), rAccTempSecond);
        tiled_mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
      }
      ::mate::warpsquad_commit_batch();

      // The scales (and the zero-point corrections) of this stage are read while the tensor engine
      // works on it, so waiting for them never holds back the SQMMAs of the stage itself.
      Tensor tCsScaleB     = thr_mma.partition_C(sScaleB(_, _0{}, stage_b).compose(ScaleBViewAsCLayout{}));
      auto   tCsScaleBFlat = filter_zeros(tCsScaleB);
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrScaleBPackedFlat); ++i) {
        UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsScaleBFlat(i)), tCrScaleBPackedFlat(i));
      }
      if constexpr (IsZeroPoints) {
        Tensor sZeroB = make_tensor(make_smem_ptr(shared_storage.zero_point.smem_zero_b.begin()), SmemLayoutZeroB{});
        Tensor tCsZeroB     = thr_mma.partition_C(sZeroB(_, _0{}, stage_b).compose(ScaleBViewAsCLayout{}));
        auto   tCsZeroBFlat = filter_zeros(tCsZeroB);
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrZeroBPackedFlat); ++i) {
          UniversalCopy<uint32_t>::copy(reinterpret_cast<uint32_t const&>(tCsZeroBFlat(i)), tCrZeroBPackedFlat(i));
        }
        // The reduction lives in shared memory, so its broadcast into the accumulator fragment is
        // expressed as a C-shaped view: filtering the zero strides leaves one reduction per row that
        // this thread's fragment covers.
        Tensor sRowSum = make_tensor(make_smem_ptr(shared_storage.zero_point.smem_row_sum.begin()), SmemLayoutRowSum{});
        Tensor tCsRowSumFirst      = thr_mma.partition_C(sRowSum(_, _0{}, stage_a).compose(RowSumViewAsCLayout{}));
        Tensor tCsRowSumSecond     = thr_mma.partition_C(sRowSum(_, _1{}, stage_a).compose(RowSumViewAsCLayout{}));
        auto   tCsRowSumFirstFlat  = filter_zeros(tCsRowSumFirst);
        auto   tCsRowSumSecondFlat = filter_zeros(tCsRowSumSecond);
        MUTE_STATIC_ASSERT_V(size(rCorrFirst) % size(tCsRowSumFirstFlat) == _0{});
        calculate_correction_bst4<0>(rCorrFirst, tCsRowSumFirstFlat, tCrZeroBPacked);
        calculate_correction_bst4<1>(rCorrSecond, tCsRowSumSecondFlat, tCrZeroBPacked);
      }
      unpack_scale_bst4<0>(rScaleFirst, tCrScaleBPacked);
      unpack_scale_bst4<1>(rScaleSecond, tCrScaleBPacked);

      ::mate::warpsquad_wait<1>();
      scale_accumulate_bst4(rAcc, rAccTempFirst, rScaleFirst, rCorrFirst);
      ::mate::warpsquad_wait<0>();
      scale_accumulate_bst4(rAcc, rAccTempSecond, rScaleSecond, rCorrSecond);

      pipeline_cast_a.consumer_release(pipe_cast_a_read);
      pipeline_load_b.consumer_release(pipe_b_read);
      ++pipe_cast_a_read;
      ++pipe_b_read;
    }

    return rAcc;
  }

  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static_assert(SharedStorageSize <= MaxSharedStorageBytes, "Shared storage exceeds the MP31 per-block limit");

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineLoadAParams pipeline_a_params;
    pipeline_a_params.transaction_bytes = TmeTransactionBytesA;
    pipeline_a_params.num_consumers     = NumCasterWarpSquads * WarpsPerWarpSquad;
    pipeline_a_params.num_producers     = 1;
    PipelineLoadA pipeline_load_a(pipeline_a_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineLoadA));

    PipelineLoadBParams pipeline_b_params;
    pipeline_b_params.transaction_bytes = TmeTransactionBytesB + TmeTransactionBytesScaleB + TmeTransactionBytesZeroB;
    pipeline_b_params.num_consumers     = NumConsumerWarpSquads * WarpsPerWarpSquad;
    pipeline_b_params.num_producers     = 1;
    PipelineLoadB pipeline_load_b(pipeline_b_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineLoadB));

    PipelineCastAParams pipeline_cast_a_params;
    pipeline_cast_a_params.producer_arv_count = NumCasterWarpSquads * WarpsPerWarpSquad;
    pipeline_cast_a_params.consumer_arv_count = NumConsumerWarpSquads * WarpsPerWarpSquad;
    PipelineCastA pipeline_cast_a(pipeline_cast_a_params, reinterpret_cast<uint64_t>(&barrier_storage->PipelineCastA));

    const int squad_idx           = mutlass::canonical_warp_squad_idx();
    const int warp_idx_in_squad   = mutlass::canonical_warp_idx_sync() % int(WarpsPerWarpSquad);
    const int thread_idx_in_squad = int(threadIdx.x) % int(NumThreadsPerWarpSquad);

    constexpr int FirstCasterSquad   = int(NumProducerWarpSquads);
    constexpr int FirstConsumerSquad = int(NumProducerWarpSquads + NumCasterWarpSquads);
    constexpr int TotalWarpSquads    = int(NumProducerWarpSquads + NumCasterWarpSquads + NumConsumerWarpSquads);

    const bool is_producer_squad = squad_idx == 0;
    const bool is_caster_squad   = squad_idx >= FirstCasterSquad && squad_idx < FirstConsumerSquad;
    const bool is_consumer_squad = squad_idx >= FirstConsumerSquad && squad_idx < TotalWarpSquads;
    const int  caster_thread_idx = (squad_idx - FirstCasterSquad) * int(NumThreadsPerWarpSquad) + thread_idx_in_squad;
    const int  consumer_thread_idx =
        (squad_idx - FirstConsumerSquad) * int(NumThreadsPerWarpSquad) + thread_idx_in_squad;

    PipelineLoadAState pipe_a_write      = mutlass::make_producer_start_state_warpspecialized<PipelineLoadA>();
    PipelineLoadBState pipe_b_write      = mutlass::make_producer_start_state_warpspecialized<PipelineLoadB>();
    PipelineCastAState pipe_cast_a_write = mutlass::make_producer_start_state_warpspecialized<PipelineCastA>();

    Scheduler scheduler(static_cast<uint32_t>(params.m), static_cast<uint32_t>(params.n), 1u, nullptr, 0u);

    __syncthreads();

    // Split-launch state. Both loads are compile-time-absent from the whole-K instantiation, so that
    // instantiation is the kernel it was before the split existed.
    int k_split   = 1;
    int num_units = 0;
    if constexpr (KSplit) {
      k_split   = params.k_split > 0 ? params.k_split : 1;
      num_units = ceil_div(params.m, BlockM) * ceil_div(params.n, BlockN) * k_split;
    }

    WorkTileInfo work_tile = WorkTileInfo::invalid_work_tile();
    if constexpr (!KSplit) {
      work_tile = scheduler.initial_work_tile_info();
    }

    if (is_producer_squad) {
      if constexpr (KSplit) {
        MUTLASS_PRAGMA_NO_UNROLL
        for (int unit = int(blockIdx.x); unit < num_units; unit += int(gridDim.x)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
          __musa_loop_transparent_outermost();
#endif
          load(params,
               shared_storage,
               pipeline_load_a,
               pipeline_load_b,
               pipe_a_write,
               pipe_b_write,
               unit_work_tile(params, unit / k_split),
               warp_idx_in_squad,
               unit % k_split);
        }
        return;
      }
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        load(params,
             shared_storage,
             pipeline_load_a,
             pipeline_load_b,
             pipe_a_write,
             pipe_b_write,
             work_tile,
             warp_idx_in_squad,
             0);
        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
      return;
    }

    PipelineLoadAState pipe_a_read;
    PipelineLoadBState pipe_b_read;
    PipelineCastAState pipe_cast_a_read;

    if (is_caster_squad) {
      if constexpr (KSplit) {
        MUTLASS_PRAGMA_NO_UNROLL
        for (int unit = int(blockIdx.x); unit < num_units; unit += int(gridDim.x)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
          __musa_loop_transparent_outermost();
#endif
          prepare_a(params,
                    shared_storage,
                    pipeline_load_a,
                    pipeline_cast_a,
                    pipe_a_read,
                    pipe_cast_a_write,
                    caster_thread_idx,
                    unit % k_split);
        }
        return;
      }
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        prepare_a(params,
                  shared_storage,
                  pipeline_load_a,
                  pipeline_cast_a,
                  pipe_a_read,
                  pipe_cast_a_write,
                  caster_thread_idx,
                  0);
        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
      return;
    }

    if (is_consumer_squad) {
      if constexpr (KSplit) {
        MUTLASS_PRAGMA_NO_UNROLL
        for (int unit = int(blockIdx.x); unit < num_units; unit += int(gridDim.x)) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
          __musa_loop_transparent_outermost();
#endif
          const int          split_id = unit % k_split;
          const WorkTileInfo tile     = unit_work_tile(params, unit / k_split);
          auto               rAcc     = compute(params,
                                  shared_storage,
                                  pipeline_load_b,
                                  pipeline_cast_a,
                                  pipe_b_read,
                                  pipe_cast_a_read,
                                  tile,
                                  consumer_thread_idx,
                                  split_id);
          epilogue(params, rAcc, tile, consumer_thread_idx, split_id);
        }
        return;
      }
      MUTLASS_PRAGMA_NO_UNROLL
      for (; work_tile.is_valid();) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
        __musa_loop_transparent_outermost();
#endif
        auto rAcc = compute(params,
                            shared_storage,
                            pipeline_load_b,
                            pipeline_cast_a,
                            pipe_b_read,
                            pipe_cast_a_read,
                            work_tile,
                            consumer_thread_idx,
                            0);
        epilogue(params, rAcc, work_tile, consumer_thread_idx, 0);

        scheduler.advance_to_next_work();
        work_tile = scheduler.get_work_tile_info();
      }
    }
  }
};

}  // namespace mate::gemm::dense_w4a16_gemm
