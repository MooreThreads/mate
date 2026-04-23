#pragma once

#include <mutlass/fast_math.h>
#include <mutlass/mutlass.h>

#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "../../common/mma_mp31_sqmma.hpp"
#include "block_info.hpp"
#include "load_primitive_builder.hpp"
#include "mask.hpp"
#include "pack_gqa.hpp"
#include "paged_kv.hpp"
#include "pipeline_ws.hpp"
#include "seqlen.hpp"
#include "softmax.hpp"
#include "utils.hpp"

#define SHOW(x)   \
  print(#x ": "); \
  print(x);       \
  print("\n")

namespace mate::attention::fmha {

using namespace mute;

template <class Element_,
          class ElementAccumulator_,
          class TileShape_,
          int  StagesK_,
          int  StagesV_,
          int  HeadDimV_,
          int  NumQKConsumers_,
          bool HasCuseqlensQ_,
          bool HasCuseqlensK_,
          bool HasCuseqlensKNew_,
          bool HasKvBatchIdx_,
          bool HasSequsedQ_,
          bool HasSequsedK_,
          bool IsPagedKV_,
          bool ForceLSULoadKV_,
          bool IsCausal_,
          bool IsLocal_,
          bool HasLearnableSink_,
          bool HasSoftcap_,
          bool IsAppendKV_,
          int  HeadRatio_,
          bool IsPackGQA_,
          bool Split_,
          bool EnableCP_,
          int  NumPVConsumers_ = NumQKConsumers_>
struct Mp31FmhaFwdTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementSink        = mutlass::bfloat16_t;

  using TileShape                    = TileShape_;
  static constexpr int TileM         = get<0>(TileShape{});
  static constexpr int TileN         = get<1>(TileShape{});
  static constexpr int TileK         = get<2>(TileShape{});  // Entry for future headdim tiling
  static constexpr int HeadDimQK     = get<2>(TileShape{});
  static constexpr int HeadDimVO     = HeadDimV_;
  static constexpr int HeadRatio     = HeadRatio_;
  static constexpr int TileHeadRatio = std::min(HeadRatio, TileM);

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumQKConsumers    = NumQKConsumers_;
  static constexpr int NumPVConsumers    = NumPVConsumers_;
  // For Pinghu, the consumer granularity is WarpSquad
  static constexpr int NumMmaWarpSquads = std::max(NumQKConsumers_, NumPVConsumers_);
  static constexpr int NumMmaThreads    = NumMmaWarpSquads * mutlass::NumThreadsPerWarpSquad;

  static_assert(TileM % NumQKConsumers == 0);
  static_assert(TileM % NumPVConsumers == 0);

  static constexpr bool HasCuseqlensQ    = HasCuseqlensQ_;
  static constexpr bool HasCuseqlensK    = HasCuseqlensK_;
  static constexpr bool HasCuseqlensKNew = HasCuseqlensKNew_;
  static constexpr bool HasKvBatchIdx    = HasKvBatchIdx_;
  static constexpr bool HasSequsedQ      = HasSequsedQ_;
  static constexpr bool HasSequsedK      = HasSequsedK_;

  static constexpr bool IsPackGQA = IsPackGQA_;
  static constexpr bool IsPagedKV = IsPagedKV_;
  static constexpr bool IsCausal  = IsCausal_;
  static constexpr bool IsLocal   = IsLocal_;
  static constexpr bool Split     = Split_;

  static constexpr bool HasLearnableSink = HasLearnableSink_;
  static constexpr bool HasSoftcap       = HasSoftcap_;

  static constexpr bool EnableCP   = EnableCP_;
  static constexpr bool IsAppendKV = IsAppendKV_;

  static constexpr bool SameHeadDim = HeadDimQK == HeadDimVO;

  static constexpr int UsePackGQATMELoad =
      IsPackGQA_ && mutlass::is_pow2<TileHeadRatio>::value && TileM % TileHeadRatio == 0;
  static constexpr bool UseTMELoadQ = !IsPackGQA_ || UsePackGQATMELoad;
  static constexpr bool UseLSULoadQ = !UseTMELoadQ;

  static constexpr bool ForceLSULoadKV = ForceLSULoadKV_ && IsPagedKV;
  static constexpr bool UseLSULoadK    = ForceLSULoadKV || (!IsPagedKV && HasCuseqlensK);
  static constexpr bool UseLSULoadV    = ForceLSULoadKV;

  static_assert(IsPagedKV || (!IsPagedKV && !ForceLSULoadKV), "ForceLSULoadKV is only supported for paged KV");
  static_assert(!IsPagedKV || (UseLSULoadK && UseLSULoadV) || (!UseLSULoadK && !UseLSULoadV),
                "KV Load methods must be same if paged KV is enabled!");
  static_assert(IsPagedKV || (HasCuseqlensK && UseLSULoadK) || !HasCuseqlensK,
                "Load K support LSU Only if ragged KV is enabled!");
  static_assert(IsPagedKV || !UseLSULoadV, "Load V support TME Only if paged kv is disabled!");

  static constexpr int NumProducerThreads =
      UseLSULoadQ || UseLSULoadK || UseLSULoadV ? mutlass::NumThreadsPerWarpSquad : mutlass::NumThreadsPerWarp;
  static constexpr bool SingleProducerWarp = NumProducerThreads == mutlass::NumThreadsPerWarp;

  static constexpr bool IntraWarpSquadOverlap = true;

  static constexpr bool IsMmaPvRS = false;

  static constexpr int StagesQ = 1;
  static constexpr int StagesK = StagesK_;
  static constexpr int StagesV = StagesV_;

  static constexpr int Alignment = 32 / sizeof_bits_v<Element>;

  using ShapeQKV  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
  using StrideQKV = Stride<int64_t, _1, int64_t, int64_t>;

  // ((head_ratio, seqlen_q), d, nheads_kv, batch)
  using ShapeQPacked_  = Shape<Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>;
  using StrideQPacked_ = Stride<Stride<int64_t, int64_t>, _1, int64_t, int64_t>;

  using ShapeQPacked  = std::conditional_t<!IsPackGQA, ShapeQKV, ShapeQPacked_>;
  using StrideQPacked = std::conditional_t<!IsPackGQA, StrideQKV, StrideQPacked_>;

  using ShapePageTable  = Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
  using StridePageTable = Stride<int64_t, _1>;

  using ShapeRotary  = Shape<int32_t, int32_t>;
  using StrideRotary = Stride<int64_t, _1>;

  using StrideDescale = Stride<int64_t, int64_t>;

  using SeqlenInfo = SeqlenInfoQK<HasCuseqlensQ,
                                  HasSequsedQ,
                                  HasCuseqlensK,
                                  HasSequsedK,
                                  false /* HasLeftpadK */,
                                  IsAppendKV,
                                  HasCuseqlensKNew,
                                  EnableCP>;
  using BlockInfo  = BlockInfo<SeqlenInfo, TileM, TileN, HeadRatio, IsCausal, IsLocal, IsPackGQA, Split, EnableCP>;

  // NOTE: RoPE not implemented yet.
  // using Rotary = Rotary<TileN, TileK, NumMmaThreads, Element>;

  using PackGQAManager   = PackGQAManager<Element, HeadRatio, TileM, HeadDimQK, NumProducerThreads>;
  using TileMPack        = Shape<Int<TileHeadRatio>, Int<TileM / TileHeadRatio>>;
  using LayoutMPack      = decltype(make_layout(TileMPack{}));
  using PackGQATileShape = Shape<TileMPack, Int<HeadDimQK>>;

  // Tile View
  using TileShapeQKD = Shape<Int<TileM>, Int<TileN>, Int<HeadDimQK>>;
  using TileShapePDV = Shape<Int<TileM>, Int<HeadDimVO>, Int<TileN>>;

  using AtomLayoutQK = Layout<Shape<Int<NumQKConsumers>, _1, _1>>;

  using TiledMmaQK = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapeQKD,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::K,
                                                                                     Int<TileM / NumQKConsumers>>(),
                                                   AtomLayoutQK{}));

  using AtomLayoutPV = Layout<Shape<Int<NumPVConsumers>, _1, _1>>;

  using TiledMmaPV = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapePDV,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::MN,
                                                                                     Int<TileM / NumPVConsumers>>(),
                                                   AtomLayoutPV{}));

  // TODO: refine ss_smem_selector
  using SmemAtomLayoutQ =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQKD>());
  using SmemAtomLayoutK =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQKD>());

  using SmemAtomLayoutP =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePDV>());
  using SmemAtomLayoutV =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::MN, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePDV>());

  using SmemLayoutQ        = decltype(tile_to_shape(SmemAtomLayoutQ{}, select<0, 2>(TileShapeQKD{})));
  using SmemLayoutQPack_   = decltype(composition(SmemLayoutQ{}, make_tile(LayoutMPack{}, Underscore{})));
  using SmemLayoutTMELoadQ = std::conditional_t<!IsPackGQA, SmemLayoutQ, SmemLayoutQPack_>;
  using SmemLayoutK        = decltype(tile_to_shape(
      SmemAtomLayoutK{}, make_shape(shape<1>(TileShapeQKD{}), shape<2>(TileShapeQKD{}), Int<StagesK>{})));
  using SmemLayoutP        = decltype(tile_to_shape(SmemAtomLayoutP{}, select<0, 2>(TileShapePDV{})));
  using SmemLayoutV        = decltype(tile_to_shape(
      SmemAtomLayoutV{}, make_shape(shape<1>(TileShapePDV{}), shape<2>(TileShapePDV{}), Int<StagesV>{})));

  using SmemAtomLayoutVLsu =
      decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG32_Atom<get<2>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               get<1>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               Element,
                                                               TCE::Major::K>{});
  using SmemLayoutVLsu = decltype(tile_to_shape(
      SmemAtomLayoutVLsu{}, make_shape(shape<2>(TileShapePDV{}), shape<1>(TileShapePDV{}), Int<StagesV>{})));

  static constexpr TME::CacheHint TmeQInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKNewInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKNewOuterHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVNewInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVNewOuterHint = TME::CacheHint::CACHE_NORMAL;

  using TmeLoadKeyBuilder = Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutK, StrideQKV, TmeKInnerHint, TmeKOuterHint>;
  using TmeLoadKeyNewBuilder =
      Mp31FmhaTmeLoadKeyBuilderNoPermute<Element, SmemLayoutK, StrideQKV, TmeKNewInnerHint, TmeKNewOuterHint>;

  static constexpr int FragmentSize = TmeLoadKeyBuilder::Fragment;

  using FragmentTypeR2S = typename TmeLoadKeyBuilder::FragmentType;
  using PermuteTileR2S  = Tile<Underscore, typename TmeLoadKeyBuilder::PermuteTileN, Underscore>;

  using PermutedShapeK = decltype(TmeLoadKeyBuilder::get_permuted_shape(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{})));

  using TmeKTileShape = typename TmeLoadKeyBuilder::TmeKTileShape;

  using BarrierQ = mutlass::arch::AsyncTransactionBarrier;

  using TileShapeQ = std::conditional_t<!IsPackGQA, Shape<Int<TileM>, Int<HeadDimQK>>, PackGQATileShape>;
  using TME_Q      = decltype(make_tme_copy<TmeQInnerHint, TmeQOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQPacked{}, int32_t(0)),
                  StrideQPacked{}),
      SmemLayoutTMELoadQ{},
      TileShapeQ{}));
  using TME_K      = typename TmeLoadKeyBuilder::TME_K;
  using TME_KNew   = typename TmeLoadKeyNewBuilder::TME_K;
  using TME_V      = decltype(make_tme_copy<TmeVInnerHint, TmeVOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQKV{}, int32_t(0)),
                  select<1, 0, 2, 3>(StrideQKV{})),
      take<0, 2>(SmemLayoutV{})));

  using PermuteTiledMmaQK = decltype(convert_to_permuted_mma(TiledMmaQK{}, PermuteTileR2S{}));
  using R2SCopyAtom       = Copy_Atom<UniversalCopy<FragmentTypeR2S>, Element>;
  using R2STiledCopy      = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaQK{}));

  using LsuLoadKeyBuilder =
      Mp31FmhaLsuLoadKeyBuilder<Element, TileShapeQKD, SmemAtomLayoutK, NumProducerThreads, StrideQKV>;
  using GmemTiledCopyK = typename LsuLoadKeyBuilder::GmemTiledCopy;
  using LsuKTile       = typename LsuLoadKeyBuilder::LsuKTile;
  using PermuteTileK   = Tile<Underscore, typename LsuLoadKeyBuilder::PermuteTileN, Underscore>;
  using FragmentTypeK  = typename LsuLoadKeyBuilder::FragmentType;

  static constexpr int GmemElemsPerLoad  = 128 / sizeof_bits_v<Element>;
  static constexpr int GmemThreadsPerRow = 8;
  using GmemCopyAtomAppedKV              = mute::Copy_Atom<MP31_ROBUST_STORE<mute::uint128_t>, Element>;
  using GmemLayoutAtomAppedKV =
      Layout<Shape<Int<NumMmaThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>, Stride<Int<GmemThreadsPerRow>, _1>>;
  using GmemTiledCopyAppendKV = decltype(make_tiled_copy(
      GmemCopyAtomAppedKV{}, GmemLayoutAtomAppedKV{}, Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));

  using MainloopPipelineQ     = std::conditional_t<!UseTMELoadQ,
                                                   mutlass::Mp31PipelineAsyncWarpsepcialized<StagesQ>,
                                                   mutlass::Mp31PipelineTmeAsyncWarpsepcialized<StagesQ>>;
  using MainloopPipelineK     = std::conditional_t<UseLSULoadK,
                                                   mutlass::Mp31PipelineAsyncWarpsepcialized<StagesK>,
                                                   mutlass::Mp31PipelineTmeAsyncWarpsepcialized<StagesK>>;
  using MainloopPipelineV     = std::conditional_t<UseLSULoadV,
                                                   mutlass::Mp31PipelineAsyncWarpsepcialized<StagesV>,
                                                   mutlass::Mp31PipelineTmeAsyncWarpsepcialized<StagesV>>;
  using MainloopPipelineKVNew = mutlass::Mp31PipelineTmeAsync<StagesK>;  // Always use TME for new KV

  using PipelineQState     = typename MainloopPipelineQ::PipelineState;
  using PipelineKState     = typename MainloopPipelineK::PipelineState;
  using PipelineVState     = typename MainloopPipelineV::PipelineState;
  using PipelineKVNewState = typename MainloopPipelineKVNew::PipelineState;

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>> smem_k;
    mute::array_aligned<Element, cosize_v<SmemLayoutP>> smem_p;
    mute::array_aligned<Element, cosize_v<SmemLayoutV>> smem_v;
  };

  static constexpr int TmeTransactionBytesQ = mutlass::bits_to_bytes(size(SmemLayoutQ{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesV =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutV{})) * sizeof_bits_v<Element>);

  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const       shape_Q;
    StrideQKV const      stride_Q;
    Element* const       ptr_K;
    ShapeQKV const       shape_K;
    StrideQKV const      stride_K;
    Element* const       ptr_V;
    int32_t const        headdim_V;
    StrideQKV const      stride_V;

    // AppendKV
    Element const* const ptr_K_new;
    ShapeQKV const       shape_K_new;
    StrideQKV const      stride_K_new;
    Element const* const ptr_V_new;
    StrideQKV const      stride_V_new;

    // Qv
    // Element const* const ptr_Qv;
    // StrideQK const stride_Qv;

    // Rotary
    Element const* const ptr_rotary_cos;
    ShapeRotary const    shape_rotary;
    StrideRotary const   stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const   stride_rotary_sin;
    // bool const is_rotary_interleaved;

    // PageTable
    int const* const      ptr_pagetable;
    ShapePageTable const  shape_pagetable;
    StridePageTable const stride_pagetable;

    // Scale
    float const         softmax_scale;
    float const*        ptr_q_descale;
    StrideDescale const stride_q_descale;
    float const*        ptr_k_descale;
    StrideDescale const stride_k_descale;
    float const*        ptr_v_descale;
    StrideDescale const stride_v_descale;

    // Local
    int const window_size_left  = -1;
    int const window_size_right = -1;

    // Chunk
    // int const attention_chunk = 0;

    // Learnable Sink
    ElementSink const* ptr_learnable_sink;

    // Softcap
    float const softcap_val;

    int const num_splits;

    // Aux tensors
    uint32_t const* const kv_batch_idx     = nullptr;
    uint32_t const* const cu_seqlens_q     = nullptr;
    uint32_t const* const cu_seqlens_k     = nullptr;
    uint32_t const* const cu_seqlens_k_new = nullptr;
    uint32_t const* const seqused_q        = nullptr;
    uint32_t const* const seqused_k        = nullptr;
    // uint32_t const* const leftpad_k        = nullptr;
    uint32_t const* const seqlens_rotary = nullptr;

    // CP
    int             cp_world_size    = 1;
    int             cp_rank          = 0;
    uint32_t const* cp_tot_seqused_k = nullptr;
  };

  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const       shape_Q;
    StrideQKV const      stride_Q;
    ShapeQPacked const   shape_Q_packed;
    StrideQPacked const  stride_Q_packed;
    Element* const       ptr_K;
    ShapeQKV const       shape_K;
    PermutedShapeK const permuted_shape_K;
    StrideQKV const      stride_K;
    Element* const       ptr_V;
    ShapeQKV const       shape_V;
    int32_t const        headdim_V;
    StrideQKV const      stride_V;

    // AppendKV
    Element const* const ptr_K_new;
    ShapeQKV const       shape_K_new;
    StrideQKV const      stride_K_new;
    Element const* const ptr_V_new;
    StrideQKV const      stride_V_new;

    // Qv
    // Element const* const ptr_Qv;
    // StrideV const stride_Qv;
    // ShapeQPacked const shape_Qv_packed;
    // StrideQPacked const stride_Qv_packed;

    // Rotary
    Element const* const ptr_rotary_cos;
    ShapeRotary const    shape_rotary;
    StrideRotary const   stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const   stride_rotary_sin;
    // bool const is_rotary_interleaved;

    // PageTable
    int const* const      ptr_pagetable;
    ShapePageTable const  shape_pagetable;
    StridePageTable const stride_pagetable;

    mutlass::FastDivmod const page_size_divmod;
    // mutlass::FastDivmod const blockN_per_page_size_divmod;
    // mutlass::FastDivmod const qhead_per_khead_divmod;

    // TiledCopy
    TME_Q    tme_load_Q;
    TME_K    tme_load_K;
    TME_V    tme_load_V;
    TME_KNew tme_load_K_new;
    TME_V    tme_load_V_new;
    // TME_Qv tme_load_Qv;

    // Robust Desc
    RobustDescriptor const desc_Q;
    RobustDescriptor const desc_K;
    RobustDescriptor const desc_V;
    RobustDescriptor const desc_page_table;
    RobustDescriptor const desc_K_new;
    RobustDescriptor const desc_V_new;

    // Scale
    float const         softmax_scale;
    float const         softmax_scale_log2;
    float const*        ptr_q_descale;
    StrideDescale const stride_q_descale;
    float const*        ptr_k_descale;
    StrideDescale const stride_k_descale;
    float const*        ptr_v_descale;
    StrideDescale const stride_v_descale;

    // Local
    int const window_size_left  = -1;
    int const window_size_right = -1;

    // Sink
    // Unused
    int const sink_token_length = 0;

    // Learnable Sink
    ElementSink const* ptr_learnable_sink = nullptr;

    float const softcap_val = 0.f;

    // Chunk
    // mutlass::FastDivmod attention_chunk_divmod;

    // Aux tensors
    uint32_t const* const kv_batch_idx     = nullptr;
    uint32_t const* const cu_seqlens_q     = nullptr;
    uint32_t const* const cu_seqlens_k     = nullptr;
    uint32_t const* const cu_seqlens_k_new = nullptr;
    uint32_t const* const seqused_q        = nullptr;
    uint32_t const* const seqused_k        = nullptr;
    // uint32_t const* const leftpad_k        = nullptr;
    uint32_t const* const seqlens_rotary = nullptr;

    // CP
    int             cp_world_size    = 1;
    int             cp_rank          = 0;
    uint32_t const* cp_tot_seqused_k = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // If IsPackGQA, reshape Q to be ((head_ratio, seqlen_q), head_size, num_head_kv, batch_size)

    int const  qhead_per_k_head = !IsPackGQA ? 1 : HeadRatio;
    auto const shape_Q_packed_  = make_shape(make_shape(qhead_per_k_head, get<0>(args.shape_Q)),
                                            get<1>(args.shape_Q),
                                            get<2>(args.shape_K),
                                            get<3>(args.shape_Q));
    auto const shape_Q_packed   = mute::conditional_return<!IsPackGQA>(args.shape_Q, shape_Q_packed_);

    auto const stride_Q_packed_ = make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)),
                                              get<1>(args.stride_Q),
                                              get<2>(args.stride_Q) * qhead_per_k_head,
                                              get<3>(args.stride_Q));
    auto const stride_Q_packed  = mute::conditional_return<!IsPackGQA>(args.stride_Q, stride_Q_packed_);

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), shape_Q_packed, stride_Q_packed);
    TME_Q  tme_load_Q =
        make_tme_copy<TmeQInnerHint, TmeQOuterHint>(MP31_TME_LOAD{}, mQ, SmemLayoutTMELoadQ{}, TileShapeQ{});

    int  cosize_q = get<0>(args.shape_Q) == 0 ? 0 : cosize(make_layout(shape_Q_packed, stride_Q_packed));
    auto desc_Q   = make_robust_desc(args.ptr_Q, cosize_q);

    // print("TME_Q:");
    // print(tme_load_Q);
    // print("\n");

    Tensor         mK               = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TME_K          tme_load_K       = TmeLoadKeyBuilder::make_tme_copy(mK);
    PermutedShapeK permuted_shape_K = TmeLoadKeyBuilder::get_permuted_shape(mK);
    int            cosize_k         = get<0>(args.shape_K) == 0 ? 0 : cosize(make_layout(args.shape_K, args.stride_K));

    RobustDescriptor desc_K = make_robust_desc(args.ptr_K, cosize_k);

    // print("TME_K:");
    // print(tme_load_K);
    // print("\n");

    auto   shape_V    = make_shape(args.headdim_V, get<0>(args.shape_K), get<2>(args.shape_K), get<3>(args.shape_K));
    Tensor mV         = make_tensor(make_gmem_ptr(args.ptr_V), shape_V, select<1, 0, 2, 3>(args.stride_V));
    TME_V  tme_load_V = make_tme_copy<TmeVInnerHint, TmeVOuterHint>(MP31_TME_LOAD{}, mV, take<0, 2>(SmemLayoutV{}));

    RobustDescriptor desc_V = make_robust_desc(args.ptr_V, cosize_k);

    // print("TME_V:");
    // print(tme_load_V);
    // print("\n");

    // AppendKV
    Tensor   mKnew          = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
    TME_KNew tme_load_K_new = TmeLoadKeyNewBuilder::make_tme_copy(conditional_return<IsAppendKV>(mKnew, mK));
    int cosize_k_new = get<0>(args.shape_K_new) == 0 ? 0 : cosize(make_layout(args.shape_K_new, args.stride_K_new));

    RobustDescriptor desc_K_new = make_robust_desc(args.ptr_K_new, cosize_k_new);

    Tensor mVnew = make_tensor(
        make_gmem_ptr(args.ptr_V_new),
        make_shape(args.headdim_V, get<0>(args.shape_K_new), get<2>(args.shape_K_new), get<3>(args.shape_K_new)),
        select<1, 0, 2, 3>(args.stride_V_new));
    TME_V tme_load_V_new = make_tme_copy<TmeVNewInnerHint, TmeVNewOuterHint>(
        MP31_TME_LOAD{}, conditional_return<IsAppendKV>(mVnew, mV), take<0, 2>(SmemLayoutV{}));

    RobustDescriptor desc_V_new = make_robust_desc(args.ptr_V_new, cosize_k_new);

    // Qv

    float const log2e = std::log2(std::exp(1.0f));

    int const page_size = IsPagedKV ? get<0>(args.shape_K) : 1;

    RobustDescriptor desc_page_table = make_robust_desc(args.ptr_pagetable, cosize(make_layout(args.shape_pagetable)));

    // SHOW(shape_Q_packed);
    // SHOW(stride_Q_packed);

    // SHOW(shape_Q_packed);
    // SHOW(stride_Q_packed);

    // SHOW(TiledMmaQK{});
    // SHOW(TiledMmaPV{});

    // SHOW(SmemLayoutQ{});
    // SHOW(SmemLayoutK{});
    // SHOW(SmemLayoutP{});
    // SHOW(SmemLayoutV{});

    // SHOW(R2STiledCopy{});

    // SHOW(SmemAtomLayoutV{});
    // SHOW(SmemAtomLayoutVLsu{});
    // SHOW(SmemLayoutVLsu{});

    return Params{
        .ptr_Q            = args.ptr_Q,
        .shape_Q          = args.shape_Q,
        .stride_Q         = args.stride_Q,
        .shape_Q_packed   = shape_Q_packed,
        .stride_Q_packed  = stride_Q_packed,
        .ptr_K            = args.ptr_K,
        .shape_K          = args.shape_K,
        .permuted_shape_K = permuted_shape_K,
        .stride_K         = args.stride_K,
        .ptr_V            = args.ptr_V,
        .shape_V          = shape_V,
        .headdim_V        = args.headdim_V,
        .stride_V         = args.stride_V,

        // AppendKV
        .ptr_K_new    = args.ptr_K_new,
        .shape_K_new  = args.shape_K_new,
        .stride_K_new = args.stride_K_new,
        .ptr_V_new    = args.ptr_V_new,
        .stride_V_new = args.stride_V_new,

        // QV

        // Rotary

        // PageTable
        .ptr_pagetable    = args.ptr_pagetable,
        .shape_pagetable  = args.shape_pagetable,
        .stride_pagetable = args.stride_pagetable,
        .page_size_divmod = mutlass::FastDivmod(page_size),

        // TiledCopy
        .tme_load_Q     = tme_load_Q,
        .tme_load_K     = tme_load_K,
        .tme_load_V     = tme_load_V,
        .tme_load_K_new = tme_load_K_new,
        .tme_load_V_new = tme_load_V_new,
        //.tme_load_Qv = tme_load_Qv,

        .desc_Q          = desc_Q,
        .desc_K          = desc_K,
        .desc_V          = desc_V,
        .desc_page_table = desc_page_table,
        .desc_K_new      = desc_K_new,
        .desc_V_new      = desc_V_new,

        .softmax_scale      = args.softmax_scale,
        .softmax_scale_log2 = args.softmax_scale * log2e,

        .window_size_left  = args.window_size_left,
        .window_size_right = args.window_size_right,

        .ptr_learnable_sink = args.ptr_learnable_sink,

        .softcap_val = args.softcap_val,

        // Chunk

        // Aux tensors
        .kv_batch_idx     = args.kv_batch_idx,
        .cu_seqlens_q     = args.cu_seqlens_q,
        .cu_seqlens_k     = args.cu_seqlens_k,
        .cu_seqlens_k_new = args.cu_seqlens_k_new,
        .seqused_q        = args.seqused_q,
        .seqused_k        = args.seqused_k,
        //.leftpad_k    = args.leftpad_k,
        //.seqlens_rotary = args.seqlens_rotary,

        // CP
        .cp_world_size    = args.cp_world_size,
        .cp_rank          = args.cp_rank,
        .cp_tot_seqused_k = args.cp_tot_seqused_k,
    };
  }

  template <class BarrierStorage, class BlockCoord>
  MUTE_DEVICE void load(Params const&      params,
                        MainloopPipelineQ& pipeline_q,
                        MainloopPipelineK& pipeline_k,
                        MainloopPipelineV& pipeline_v,
                        PipelineQState&    smem_pipe_write_q,
                        PipelineKState&    smem_pipe_write_k,
                        PipelineVState&    smem_pipe_write_v,
                        SharedStorage&     shared_storage,
                        BarrierStorage*    barrier_storage,
                        SeqlenInfo const&  seqlen_info,
                        BlockCoord         blk_coord,
                        int&               work_idx,
                        int const          num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(
        seqlen_info, m_block, split_idx, /*splits*/ num_splits, params.window_size_left, params.window_size_right);

    // If no valid block, no need to load.
    if (n_block_min >= n_block_max) {
      return;
    }

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutTMELoadQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    // Used for LSU
    Tensor sK_pi = mute::as_position_independent_swizzle_tensor(sK);
    Tensor sV_pi = mute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{}));

    int const thread_idx             = threadIdx.x % NumProducerThreads;
    int const bidh_kv                = !IsPackGQA ? bidh / HeadRatio : bidh;
    int const bidb_kv                = !HasKvBatchIdx ? bidb : params.kv_batch_idx[bidb];
    int const warp_idx               = mutlass::canonical_warp_idx();
    int const warp_idx_in_warp_squad = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int const seqlen_k               = seqlen_info.seqlen_k;

    auto offset_coord_q = [&]() {
      if constexpr (!IsPackGQA) {
        return make_coord(seqlen_info.offset_q, _0{});
      } else {
        return make_coord(make_coord(_0{}, seqlen_info.offset_q), _0{});
      }
    }();
    Tensor mQ = params.tme_load_Q.get_tme_tensor(params.shape_Q_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
    Tensor mK = params.tme_load_K.get_tme_tensor(params.permuted_shape_K)(_, _, bidh_kv, _);
    Tensor mV = params.tme_load_V.get_tme_tensor(params.shape_V)(_, _, bidh_kv, _);

    Tensor gQ = local_tile(domain_offset(offset_coord_q, mQ), TileShapeQ{}, make_coord(m_block, _0{}));
    Tensor gK = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}, _0{}), mK), TmeKTileShape{}, make_coord(_, _0{}, _));
    Tensor gV = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k, _0{}), mV),
                           select<1, 2>(TileShapePDV{}),
                           make_coord(_0{}, _, _));

    auto   cta_tme_Q = params.tme_load_Q.get_slice(0);
    Tensor tQgQ      = group_modes<0, 3>(cta_tme_Q.partition_S(gQ));
    Tensor tQsQ      = group_modes<0, 3>(cta_tme_Q.partition_D(sQ));

    auto   cta_tme_K = params.tme_load_K.get_slice(0);
    Tensor tKgK      = group_modes<0, 3>(cta_tme_K.partition_S(gK));  // (TME, n, l)
    Tensor tKsK      = group_modes<0, 3>(cta_tme_K.partition_D(sK));  // (TME, pipe)

    auto   cta_tme_V = params.tme_load_V.get_slice(0);
    Tensor tVgV      = group_modes<0, 3>(cta_tme_V.partition_S(gV));  // (TME, n, l)
    Tensor tVsV      = group_modes<0, 3>(cta_tme_V.partition_D(sV));  // (TME, pipe)

    int const bidb_kv_idx = !HasCuseqlensK && !IsPagedKV ? bidb_kv : 0;

    using KVManager =
        PagedKVManager<IsPagedKV, Element, NumProducerThreads, TileN, HeadDimQK, HeadDimVO, !IntraWarpSquadOverlap>;
    GmemTiledCopyK tiled_copy_kv;
    auto           thr_copy_kv = tiled_copy_kv.get_thread_slice(threadIdx.x);

    Tensor mK_lsu = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh_kv, HasCuseqlensK ? 0 : bidb_kv);
    Tensor gK_lsu =
        local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK_lsu), LsuKTile{}, make_coord(_, _0{}));

    KVManager paged_kv_manager{params.ptr_pagetable,
                               params.shape_pagetable,
                               params.stride_pagetable,
                               params.desc_page_table,
                               params.ptr_K,
                               params.shape_K,
                               params.stride_K,
                               params.desc_K,
                               params.ptr_V,
                               params.headdim_V,
                               params.stride_V,
                               params.desc_V,
                               params.page_size_divmod,
                               seqlen_k,
                               0,  // seqlen_info.leftpad_k,
                               thread_idx,
                               bidb_kv,
                               bidh_kv,
                               bidb_kv_idx};

    auto load_K = [&](int const n_block, auto& smem_pipe_write) {
      pipeline_k.producer_acquire(smem_pipe_write);
      if constexpr (IsPagedKV) {
        if constexpr (UseLSULoadK) {
          paged_kv_manager.load_K(n_block, sK_pi(_, _, smem_pipe_write.index()));
          mute::ldgsts_wait();
          pipeline_k.producer_commit(smem_pipe_write);
          ++smem_pipe_write;
        } else {
          uint32_t bar_id = pipeline_k.producer_get_barrier_id(smem_pipe_write);

          auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_tme_k();

          copy(params.tme_load_K.with(bar_id), tKgK(_, n_block_idx, bidb_kv_idx), tKsK(_, smem_pipe_write.index()));
          ++smem_pipe_write;
        }
      } else {
        // NOT Paged KV
        // Got BSHD or Ragged KV
        if constexpr (UseLSULoadK) {
          Tensor tKgK_lsu = group_modes<0, 3>(thr_copy_kv.partition_S(gK_lsu(_, _, n_block)));
          Tensor tKsK_lsu = group_modes<0, 3>(thr_copy_kv.partition_D(sK));
          copy(tiled_copy_kv.with(params.desc_K), tKgK_lsu, tKsK_lsu(_, smem_pipe_write.index()));

          mute::ldgsts_wait();
          pipeline_k.producer_commit(smem_pipe_write);
          ++smem_pipe_write;

        } else {
          uint32_t bar_id = pipeline_k.producer_get_barrier_id(smem_pipe_write);

          copy(params.tme_load_K.with(bar_id), tKgK(_, n_block, bidb_kv_idx), tKsK(_, smem_pipe_write.index()));
          ++smem_pipe_write;
        }
      }
    };

    auto load_V = [&](int const n_block, auto& smem_pipe_write) {
      pipeline_v.producer_acquire(smem_pipe_write);
      if constexpr (IsPagedKV) {
        if constexpr (UseLSULoadV) {
          paged_kv_manager.load_V(n_block, sV_pi(_, _, smem_pipe_write.index()));
          mute::ldgsts_wait();
          pipeline_v.producer_commit(smem_pipe_write);
          ++smem_pipe_write;
        } else {
          uint32_t bar_id = pipeline_v.producer_get_barrier_id(smem_pipe_write);

          auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_tme_v();

          copy(params.tme_load_V.with(bar_id), tVgV(_, n_block_idx, bidb_kv_idx), tVsV(_, smem_pipe_write.index()));
          ++smem_pipe_write;
        }
      } else {
        // NOT Paged KV
        // Got BSHD or Ragged KV
        uint32_t bar_id = pipeline_v.producer_get_barrier_id(smem_pipe_write);
        copy(params.tme_load_V.with(bar_id), tVgV(_, n_block, bidb_kv_idx), tVsV(_, smem_pipe_write.index()));
        ++smem_pipe_write;
      }
    };

    bool should_load_K = UseLSULoadK || SingleProducerWarp || warp_idx_in_warp_squad == 0;
    bool should_load_V = UseLSULoadV || SingleProducerWarp || warp_idx_in_warp_squad == 0;

    int n_block = n_block_max - 1;

    if constexpr (UseTMELoadQ) {
      // (Non-)PackGQA TME load Q
      if (SingleProducerWarp || warp_idx_in_warp_squad == 0) {
        pipeline_q.producer_acquire(smem_pipe_write_q);
        uint32_t bar_id = pipeline_q.producer_get_barrier_id(smem_pipe_write_q);

        copy(params.tme_load_Q.with(bar_id), tQgQ, tQsQ);
        ++smem_pipe_write_q;
      }
    } else {
      // PackGQA LSU Load Q
      pipeline_q.producer_acquire(smem_pipe_write_q);
      Tensor mQPack =
          make_tensor(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q),
                      make_layout(params.shape_Q_packed, params.stride_Q_packed))(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
      Tensor sQPack = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
      PackGQAManager::load_Q(params, mQPack, sQPack, thread_idx, m_block);
    }

    // Prologue load from page_table + load K
    if (should_load_K) {
      if constexpr (IsPagedKV) {
        // NOTE: force use same load method (LSU or TME) for both K and V if IsPagedKV
        if constexpr (UseLSULoadK || UseLSULoadV) {
          paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ true>(n_block);
        } else {
          paged_kv_manager.template load_page_table_for_tme</* FirstIter */ true>(n_block);
        }
      }
      load_K(n_block, smem_pipe_write_k);
    }

    if constexpr (!UseTMELoadQ) {
      mute::ldgsts_wait();
      pipeline_q.producer_commit(smem_pipe_write_q);
      ++smem_pipe_write_q;
    }

    if constexpr (!IntraWarpSquadOverlap) {
      if (should_load_V) {
        load_V(n_block, smem_pipe_write_v);
      }
    }

    int n_block_prev = n_block;
    --n_block;
    for (; n_block >= n_block_min; --n_block) {
      if constexpr (IsPagedKV) {
        // NOTE: using same load method (LSU or TME) for both K and V if IsPagedKV
        if (should_load_K) {
          if constexpr (UseLSULoadK || UseLSULoadV) {
            paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ false>(n_block);
          } else {
            paged_kv_manager.template load_page_table_for_tme</* FirstIter */ false>(n_block);
          }
        }
      }
      if (should_load_K) {
        load_K(n_block, smem_pipe_write_k);
      }

      if (should_load_V) {
        if constexpr (IntraWarpSquadOverlap) {
          load_V(n_block_prev, smem_pipe_write_v);
        } else {
          load_V(n_block, smem_pipe_write_v);
        }
      }
      n_block_prev = n_block;
    }

    if constexpr (IntraWarpSquadOverlap) {
      if (should_load_V) {
        load_V(n_block_prev, smem_pipe_write_v);
      }
    }
  }

  template <class BarrierStorage, class BlockCoord>
  MUTE_DEVICE auto mma(Params const&      params,
                       MainloopPipelineQ& pipeline_q,
                       MainloopPipelineK& pipeline_k,
                       MainloopPipelineV& pipeline_v,
                       PipelineQState&    smem_pipe_read_q,
                       PipelineKState&    smem_pipe_read_k,
                       PipelineVState&    smem_pipe_read_v,
                       SharedStorage&     shared_storage,
                       BarrierStorage*    barrier_storage,
                       SeqlenInfo const&  seqlen_info,
                       BlockCoord         blk_coord,
                       int const          thread_idx,
                       int&               work_idx,
                       int const          num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(
        seqlen_info, m_block, split_idx, /*splits*/ num_splits, params.window_size_left, params.window_size_right);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    Tensor acc_pv    = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePDV{}));
    auto   acc_pv_mn = make_tensor(acc_pv.data(), layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    constexpr int Rows = size<0>(layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    float const effective_scale      = HasSoftcap ? 1.0f : params.softmax_scale;
    float const effective_scale_log2 = HasSoftcap ? static_cast<float>(M_LOG2E) : params.softmax_scale_log2;
    Softmax<Rows, HasLearnableSink> softmax{effective_scale, effective_scale_log2};

    // If invalid, return empty result.
    if (n_block_min >= n_block_max) {
      auto lse = make_fragment_like(softmax.row_sum);
      return mute::make_tuple(false, mute::make_tuple(acc_pv, lse));
    }

    int const thread_idx_in_warpgroup = thread_idx % mutlass::NumThreadsPerWarpSquad;
    int const warpgroup_idx           = thread_idx / mutlass::NumThreadsPerWarpSquad;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    // TiledMmaQK tiled_mma_qk;
    // TiledMmaPV tiled_mma_pv;

    ThrMMA thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tOrP = thr_mma_pv.partition_fragment_A(sP);
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);

    R2STiledCopy tiled_copy_r2s;
    ThrCopy      thr_copy_r2s = tiled_copy_r2s.get_thread_slice(thread_idx);
    Tensor       tPsP         = thr_copy_r2s.partition_D(mute::as_position_independent_swizzle_tensor(sP));

    auto write_P_to_smem = [&](auto& accum_cvt) {
      Tensor tPrP = thr_copy_r2s.retile_S(accum_cvt);
      copy(tiled_copy_r2s, tPrP, tPsP);
      // TODO: remote sync
    };

    auto arrive_on_P_write_barrier = [&] {
      __syncwarp();
      // TODO: remote sync
    };

    auto apply_softcap = [&](auto& acc_qk) {
      if constexpr (HasSoftcap) {
        float const scale_times_inv_softcap = params.softmax_scale / params.softcap_val;
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); ++i) {
          acc_qk(i) = params.softcap_val * mutlass::fast_tanh(acc_qk(i) * scale_times_inv_softcap);
        }
      }
    };

    // Q is ready
    pipeline_q.consumer_wait(smem_pipe_read_q);

    // Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePDV{}));

    clear(acc_pv);

    // if (thread_idx == 0) {
    //   SHOW(tPsP);
    // }

    // constexpr int                   Rows = size<0>(layout_acc_mn(tiled_mma_pv, acc_pv.layout()));
    // Softmax<Rows, HasLearnableSink> softmax{params.softmax_scale, params.softmax_scale_log2};
    Mask<PermuteTiledMmaQK, SeqlenInfo, TileM, TileN, HeadRatio, IsPackGQA, IsCausal, IsLocal, EnableCP> mask(
        thread_idx, seqlen_info, params.window_size_left, params.window_size_right, params.sink_token_length);

    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;
    int       n_block  = n_block_max - 1;

    if constexpr (IntraWarpSquadOverlap) {
      Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      pipeline_k.consumer_wait(smem_pipe_read_k);

      // MMA QK
      mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), acc_qk);

      mate::warpsquad_commit_batch();
      mate::warpsquad_wait();

      pipeline_k.consumer_release(smem_pipe_read_k);
      ++smem_pipe_read_k;

      apply_softcap(acc_qk);

      // Mask Mode
      mask.template apply</*SeqlenMask*/ true>(acc_qk, m_block, n_block);

      // Softmax
      Tensor correction_scales = softmax.template online_softmax<true, true>(acc_qk, tiled_mma_qk);

      Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
      convert_type<FragmentSize>(acc_qk, accum_cvt);

      if constexpr (!IsMmaPvRS) {
        write_P_to_smem(accum_cvt);
      }
      if constexpr (!IsMmaPvRS) {
        arrive_on_P_write_barrier();
      }

      --n_block;

      // Each step does gemm0 for iter n_block, gemm1 for iter n_block+1, and softmax for iter n_block
      auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
        static constexpr bool CheckInf = decltype(check_inf_type)::value;

        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
        // trigger zero init sqmma
        clear(acc_qk);

        pipeline_k.consumer_wait(smem_pipe_read_k);
        mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), acc_qk);

        mate::warpsquad_commit_batch();

        pipeline_v.consumer_wait(smem_pipe_read_v);
        mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), acc_pv);

        mate::warpsquad_commit_batch();

        // wait QK done
        mate::warpsquad_wait<1>();

        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        apply_softcap(acc_qk);

        // Mask Mode
        mask_fn(acc_qk, n_block);

        // Softmax
        mute::copy(softmax.template online_softmax<false, CheckInf>(acc_qk, tiled_mma_qk), correction_scales);

        // wait PV done
        mate::warpsquad_wait<0>();

        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;

        convert_type<FragmentSize>(acc_qk, accum_cvt);

        if constexpr (!IsMmaPvRS) {
          write_P_to_smem(accum_cvt);
        }
        softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
        if constexpr (!IsMmaPvRS) {
          arrive_on_P_write_barrier();
        }
      };

      // Causal/Local Masking
      if constexpr (IsCausal || IsLocal) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*seqlenk mask*/ false>(tSrS, m_block, n_block);
        };
        int const n_block_min_causal_local_mask =
            BlockInfo::get_n_block_min_causal_local_mask(seqlen_info, m_block, n_block_min, params.window_size_right);

        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, /* CheckInf */ mute::true_type{});
        }
      }
      // No mask iterations
      int const n_block_min_before_local_mask =
          BlockInfo::get_n_block_min_before_local_mask(seqlen_info, m_block, n_block_min, params.window_size_left);
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, /* CheckInf */ mute::false_type{});
      }

      // Local mask iterations
      if constexpr (IsLocal) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*SeqlenKMask*/ false>(tSrS, m_block, n_block);
        };
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, /* CheckInf */ mute::true_type{});
        }
      }

      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;

      // Last PV MMA
      pipeline_v.consumer_wait(smem_pipe_read_v);
      mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), acc_pv);

      mate::warpsquad_commit_batch();
      mate::warpsquad_wait();
      pipeline_v.consumer_release(smem_pipe_read_v);
      ++smem_pipe_read_v;
    } else {
      auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
        static constexpr bool IsFirstIter = decltype(is_first_iter_type)::value;
        static constexpr bool CheckInf    = decltype(check_inf_type)::value;

        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
        // trigger zero init sqmma
        clear(acc_qk);

        pipeline_k.consumer_wait(smem_pipe_read_k);

        // MMA QK
        mute::gemm(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), acc_qk);

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        pipeline_k.consumer_release(smem_pipe_read_k);
        ++smem_pipe_read_k;

        apply_softcap(acc_qk);

        // Mask Mode
        mask_fn(acc_qk, n_block);

        // Softmax
        Tensor correction_scales = softmax.template online_softmax<IsFirstIter, CheckInf>(acc_qk, tiled_mma_qk);

        Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
        convert_type<FragmentSize>(acc_qk, accum_cvt);
        if constexpr (!IsMmaPvRS) {
          write_P_to_smem(accum_cvt);
        }
        if constexpr (!IsFirstIter) {
          softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
        }
        if constexpr (!IsMmaPvRS) {
          arrive_on_P_write_barrier();
        }

        pipeline_v.consumer_wait(smem_pipe_read_v);

        // MMA PV
        mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), acc_pv);

        mate::warpsquad_commit_batch();
        mate::warpsquad_wait();
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
      };

      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply</*seqlenk mask*/ true>(tSrS, m_block, n_block);
      };
      fwd_step(n_block, first_iter_mask_fn, /*IsFirstIter*/ mute::true_type{}, /*CheckInf*/ mute::true_type{});
      --n_block;

      // Causal/Local Masking
      if constexpr (IsCausal || IsLocal) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*seqlenk mask*/ false>(tSrS, m_block, n_block);
        };

        int const n_block_min_causal_local_mask =
            BlockInfo::get_n_block_min_causal_local_mask(seqlen_info, m_block, n_block_min, params.window_size_right);

        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, /* IsFirstIter */ mute::false_type{}, /* CheckInf */ mute::true_type{});
        }
      }

      // No mask iterations
      int const n_block_min_before_local_mask =
          BlockInfo::get_n_block_min_before_local_mask(seqlen_info, m_block, n_block_min, params.window_size_left);
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, /* IsFirstIter */ mute::false_type{}, /* CheckInf */ mute::false_type{});
      }

      // Local mask iterations
      if constexpr (IsLocal) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*SeqlenKMask*/ false>(tSrS, m_block, n_block);
        };
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, /*IsFirstIter*/ mute::false_type{}, /* CheckInf */ mute::true_type{});
        }
      }

      // release q
      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;
    }
    ++work_idx;

    auto sink_vals = [&]() {
      if constexpr (IsPackGQA) {
        return make_tensor<ElementAccumulator>(Shape<Int<Rows>>{});
      } else {
        return make_tensor<ElementAccumulator>(Layout<Shape<Int<Rows>>, Stride<Int<0>>>{});
      }
    }();
    if constexpr (HasLearnableSink) {
      if constexpr (IsPackGQA) {
        int const  head_idx_qo_base = bidh * HeadRatio;
        auto const tScS_m           = mask.tScS_mn(_, _0{});
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Rows; ++i) {
          int const head_idx_qo = head_idx_qo_base + (m_block * TileM + get<0>(tScS_m(i))) % HeadRatio;
          sink_vals(i)          = static_cast<ElementAccumulator>(params.ptr_learnable_sink[head_idx_qo]);
        }
      } else {
        int const head_idx_qo = bidh;
        sink_vals(0)          = static_cast<ElementAccumulator>(params.ptr_learnable_sink[head_idx_qo]);
      }
      // if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      //   MUTLASS_PRAGMA_UNROLL)
      //   for (int i = 0; i < Rows; ++i) {
      //     printf("sink_val: %f\n", float(sink_vals(i)));
      //   }
      // }
    }

    // TODO: without store_zero, we get invalid lse here
    Tensor lse = softmax.tail(acc_pv, tiled_mma_pv, sink_vals);

    return mute::make_tuple(true, mute::make_tuple(acc_pv, lse));
  }

  template <class BlockCoord>
  MUTE_DEVICE bool load_kv_new(Params const&          params,
                               MainloopPipelineKVNew& pipeline_k_new,
                               MainloopPipelineKVNew& pipeline_v_new,
                               PipelineKVNewState&    smem_pipe_write_kv_new,
                               SharedStorage&         shared_storage,
                               SeqlenInfo const&      seqlen_info,
                               BlockCoord             blk_coord,
                               int const              warp_idx_in_warp_squad,
                               int&                   work_idx,
                               int const              num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_new_min, n_block_new_max] = BlockInfo::get_n_block_k_new_min_max(
        seqlen_info, m_block, bidb, split_idx, num_splits, params.window_size_left, params.window_size_right);
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("MP=%d, bidm=%d, bidb=%d, bidh=%d, bids=%d, num_splits=%d, n_block_new_min=%d, n_block_new_max=%d\n",
    //          blockIdx.x,
    //          m_block,
    //          bidb,
    //          bidh,
    //          split_idx,
    //          num_splits,
    //          n_block_new_min,
    //          n_block_new_max);
    // }
    if (n_block_new_max <= n_block_new_min) {
      return false;
    }

    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{});

    int const bidh_kv = !IsPackGQA ? bidh / HeadRatio : bidh;

    Tensor mKnew =
        params.tme_load_K_new.get_tme_tensor(params.shape_K_new)(_, _, bidh_kv, !HasCuseqlensKNew ? bidb : 0);
    auto shape_Vnew = make_shape(
        params.headdim_V, get<0>(params.shape_K_new), get<2>(params.shape_K_new), get<3>(params.shape_K_new));
    Tensor mVnew = params.tme_load_V_new.get_tme_tensor(shape_Vnew)(_, _, bidh_kv, !HasCuseqlensKNew ? bidb : 0);

    Tensor gKnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew),
                              select<1, 2>(TileShapeQKD{}),
                              make_coord(_, _0{}));  // (N, K, _)
    Tensor gVnew = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k_new), mVnew),
                              select<1, 2>(TileShapePDV{}),
                              make_coord(_0{}, _));  // (K, N, _)

    auto   cta_tme_K_new = params.tme_load_K_new.get_slice(0);
    Tensor tKgKnew       = group_modes<0, 3>(cta_tme_K_new.partition_S(gKnew));  // (TME, k)
    Tensor tKsKnew       = group_modes<0, 3>(cta_tme_K_new.partition_D(sK));     // (TME, pipe)

    auto   cta_tme_V_new = params.tme_load_V_new.get_slice(0);
    Tensor tVgVnew       = group_modes<0, 3>(cta_tme_V_new.partition_S(gVnew));  // (TME, k)
    Tensor tVsVnew       = group_modes<0, 3>(cta_tme_V_new.partition_D(sV));     // (TME, pipe)

    auto load_K_new = [&](int const n_block, auto const& smem_pipe_write) {
      pipeline_k_new.producer_acquire(smem_pipe_write);
      auto bar_id = pipeline_k_new.producer_get_barrier_id(smem_pipe_write);
      copy(params.tme_load_K_new.with(bar_id), tKgKnew(_, n_block), tKsKnew(_, smem_pipe_write.index()));
    };

    auto load_V_new = [&](int const n_block, auto const& smem_pipe_write) {
      pipeline_v_new.producer_acquire(smem_pipe_write);
      auto bar_id = pipeline_v_new.producer_get_barrier_id(smem_pipe_write);
      copy(params.tme_load_V_new.with(bar_id), tVgVnew(_, n_block), tVsVnew(_, smem_pipe_write.index()));
    };

    bool should_load_kv = SingleProducerWarp || warp_idx_in_warp_squad == 0;

    // pipeline_kv_guard.producer_acquire(smem_pipe_write_kv_guard);

    int n_block = n_block_new_max - 1;
    // Unlike the Hopper kernel, we don't need barrier_O here.
    // This kernel doesn't have the async O-side epilogue / cluster handoff that keeps
    // shared memory alive across stages, so load_kv_new doesn't need an extra recycle
    // barrier before reusing smem_k and smem_v.
    // Note: TME copies are issued by a producer warp, not a single elected thread,
    // so we intentionally don't use elect_one_sync() here.
    if (should_load_kv) {
      load_K_new(n_block, smem_pipe_write_kv_new);
      load_V_new(n_block, smem_pipe_write_kv_new);
    }
    ++smem_pipe_write_kv_new;
    --n_block;
    for (; n_block >= n_block_new_min; --n_block) {
      if (should_load_kv) {
        load_K_new(n_block, smem_pipe_write_kv_new);
        load_V_new(n_block, smem_pipe_write_kv_new);
      }
      ++smem_pipe_write_kv_new;
    }

    return true;
  }

  template <class BlockCoord>
  MUTLASS_DEVICE bool store_kv_new(Params const&          params,
                                   MainloopPipelineKVNew& pipeline_k_new,
                                   MainloopPipelineKVNew& pipeline_v_new,
                                   PipelineKVNewState&    smem_pipe_read_kv_new,
                                   int const              thread_idx,
                                   SharedStorage&         shared_storage,
                                   SeqlenInfo const&      seqlen_info,
                                   BlockCoord             blk_coord,
                                   int const              num_splits) {
    int const m_block                       = get<0>(blk_coord);
    int const bidh                          = get<1>(blk_coord);
    int const bidb                          = get<2>(blk_coord);
    int const split_idx                     = get<3>(blk_coord);
    auto [n_block_new_min, n_block_new_max] = BlockInfo::get_n_block_k_new_min_max(
        seqlen_info, m_block, bidb, split_idx, num_splits, params.window_size_left, params.window_size_right);

    if (n_block_new_max <= n_block_new_min) {
      return false;
    }

    Tensor sK = mute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{}));
    // We want to use SmemLayoutVt to have shape (TileN, kHeadDim) instead of (kHeadDim, TileN)
    Tensor sV = mute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{}));

    int const bidh_kv = !IsPackGQA ? bidh / HeadRatio : bidh;
    int const bidb_kv = !HasKvBatchIdx ? bidb : params.kv_batch_idx[bidb];

    Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh_kv, !HasCuseqlensKNew ? bidb_kv : 0);
    auto shape_V = make_shape(params.headdim_V, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
    Tensor mV    = make_tensor(make_gmem_ptr(params.ptr_V), shape_V, params.stride_V)(
        _, _, bidh_kv, !HasCuseqlensKNew ? bidb_kv : 0);

    int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;

    Tensor gK = local_tile(
        domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShapeQKD{}), make_coord(_, _0{}));  // (N, K, _)
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV),
                           select<2, 1>(TileShapePDV{}),
                           make_coord(_, _0{}));  // (N, K_v, _)

    int const seqlen_k_new = seqlen_info.seqlen_k_new;

    // Rope not implemented yet.
    // Rotary rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
    //               params.ptr_rotary_sin, params.stride_rotary_sin,
    //               params.is_rotary_interleaved, thread_idx, seqlen_k_new,
    //               seqlen_info.seqlen_rotary);

    // This is used to index into the batch dimension of mK and mV
    int const bidb_kv_idx = !HasCuseqlensKNew && !IsPagedKV ? bidb_kv : 0;

    using KVManager =
        PagedKVManager<IsPagedKV, Element, NumMmaThreads, TileN, HeadDimQK, HeadDimVO, true /* IsKVSameIter */>;

    // passing offset_k instead of leftpad_k will move the PageTable pointer to the right position
    KVManager paged_kv_manager{params.ptr_pagetable,
                               params.shape_pagetable,
                               params.stride_pagetable,
                               params.desc_page_table,
                               params.ptr_K,
                               params.shape_K,
                               params.stride_K,
                               params.desc_K,
                               params.ptr_V,
                               params.headdim_V,
                               params.stride_V,
                               params.desc_V,
                               params.page_size_divmod,
                               seqlen_k_new,
                               offset_k,  // seqlen_info.offset_k + seqlen_info.seqlen_k_og
                               thread_idx,
                               bidb_kv,
                               bidh_kv,
                               bidb_kv_idx};

    GmemTiledCopyAppendKV gmem_tiled_copy_kv;

    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);

    Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
    Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);

    Tensor cK   = make_identity_tensor(select<1, 2>(TileShapeQKD{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
    MUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tKpK); ++k) {
      tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K);
    }

    Tensor cV    = make_identity_tensor(select<2, 1>(TileShapePDV{}));  // (BLK_N,BLK_K_V) -> (blk_n,blk_k_v)
    Tensor tVcV  = conditional_return<SameHeadDim>(tKcK, gmem_thr_copy_kv.partition_D(cV));
    Tensor tVpV_ = make_tensor<bool>(make_shape(size<2>(tVsV)));
    MUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tVpV_); ++k) {
      tVpV_(k) = get<1>(tVcV(_0{}, _0{}, k)) < params.headdim_V;
    }
    Tensor tVpV = conditional_return<SameHeadDim>(tKpK, tVpV_);

    auto store_K = [&](int const n_block, auto const& smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * TileN, TileN);

      pipeline_k_new.consumer_wait(smem_pipe_read);
      Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_read.index());
      Tensor tKrK     = make_fragment_like(tKsK_cur);  // ((_8,_1),_4,_2):((_1,_0),_8,_32)
      Tensor tKrK_src = gmem_thr_copy_kv.retile_S(tKrK);
      copy(tKsK_cur, tKrK);
      if constexpr (!IsPagedKV) {
        Tensor tKgK_cur = tKgK(_, _, _, n_block);
        MUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tKgK_cur); ++k) {
          copy(gmem_tiled_copy_kv.with(params.desc_K_new).with(tKpK(k)), tKrK_src(_, _, k), tKgK_cur(_, _, k));
        }
      } else {
        paged_kv_manager.store_K(n_block, tKrK_src);
      }
      pipeline_k_new.consumer_release(smem_pipe_read);
    };

    auto store_V = [&](int const n_block, auto const& smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * TileN, TileN);

      pipeline_v_new.consumer_wait(smem_pipe_read);
      Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_read.index());
      Tensor tVrV     = make_fragment_like(tVsV_cur);
      Tensor tVrV_src = gmem_thr_copy_kv.retile_S(tVrV);
      copy(tVsV_cur, tVrV);
      if constexpr (!IsPagedKV) {
        Tensor tVgV_cur = tVgV(_, _, _, n_block);
        MUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tVgV_cur); ++k) {
          copy(gmem_tiled_copy_kv.with(params.desc_V_new).with(tVpV(k)), tVrV_src(_, _, k), tVgV_cur(_, _, k));
        }
      } else {
        paged_kv_manager.store_V(n_block, tVrV_src);
      }
      pipeline_v_new.consumer_release(smem_pipe_read);
    };

    // int n_block = 0;  // DEBUG ONLY
    int n_block = n_block_new_max - 1;
    if constexpr (IsPagedKV) {
      paged_kv_manager.template load_page_table_for_lsu<true /* FirstIter */, false /* PermuteK */>(n_block);
    }
    store_K(n_block, smem_pipe_read_kv_new);
    store_V(n_block, smem_pipe_read_kv_new);
    ++smem_pipe_read_kv_new;
    --n_block;

    for (; n_block >= n_block_new_min; --n_block) {
      if constexpr (IsPagedKV) {
        paged_kv_manager.template load_page_table_for_lsu<false /* FirstIter */, false /* PermuteK */>(n_block);
      }
      store_K(n_block, smem_pipe_read_kv_new);
      store_V(n_block, smem_pipe_read_kv_new);
      ++smem_pipe_read_kv_new;
    }

    return true;
  }
};

}  // namespace mate::attention::fmha
