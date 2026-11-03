#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mute/algorithm/prefetch.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/simd_mp31.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>

#include "mate/attention/fmha/pipeline_ws.hpp"
#include "mate/attention/fmha/utils.hpp"
#include "mate/attention/msa/msa_options.hpp"
#include "mate/attention/msa/paged_kv.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"

namespace mate::attention::msa::collective {

using namespace mute;

template <class Element_, class TileShape_, int HeadRatio_, class... Options_>
struct Mp31MsaMaxScoreCollectiveTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = float;
  using TileShape          = TileShape_;

  static constexpr bool IsPagedKV = find_option_t<Tag::IsPagedKV, std::false_type, Options_...>::value;
  static constexpr bool IsCausal  = find_option_t<Tag::IsCausal, std::false_type, Options_...>::value;
  static constexpr int  HeadRatio = HeadRatio_;

  static constexpr int TileQ    = get<0>(TileShape{});
  static constexpr int TileKV   = get<1>(TileShape{});
  static constexpr int HeadDim  = get<2>(TileShape{});
  static constexpr int PageSize = find_option_t<Tag::PageSize, std::integral_constant<int, TileKV>, Options_...>::value;
  static constexpr PageTableKind PageTableMode =
      find_option_t<Tag::PageTable,
                    std::integral_constant<PageTableKind, IsPagedKV ? PageTableKind::Batched2D : PageTableKind::Dense>,
                    Options_...>::value;
  static constexpr int QStages = find_option_t<Tag::QStages, std::integral_constant<int, 1>, Options_...>::value;
  static constexpr int KStages =
      find_option_t<Tag::KStages,
                    std::integral_constant<int, is_same_v<Element, mutlass::float_e4m3_t> ? 2 : 1>,
                    Options_...>::value;
  static constexpr int  QTokensPerTile     = TileQ / HeadRatio;
  static constexpr int  SmemAlignmentBytes = 256;
  static constexpr bool EnableKPrefetch =
      find_option_t<Tag::EnableKPrefetch,
                    std::bool_constant<(TileQ > 16 && !is_same_v<Element, mutlass::bfloat16_t>)>,
                    Options_...>::value;

  static constexpr int MmaAlignment = 32 / sizeof_bits_v<Element>;
  static constexpr int MmaTileQ =
      find_option_t<Tag::MmaTileQ, std::integral_constant<int, (TileQ >= 32 ? 32 : 16)>, Options_...>::value;
  using BuilderTileShape = Shape<Int<MmaTileQ>, Int<TileKV>, Int<HeadDim>>;

  using BuilderCollective = typename mutlass::gemm::collective::CollectiveBuilder<
      mutlass::arch::Mp31,
      mutlass::arch::OpClassTensorOp,
      Element,
      mutlass::layout::RowMajor,
      MmaAlignment,
      Element,
      mutlass::layout::ColumnMajor,
      MmaAlignment,
      ElementAccumulator,
      BuilderTileShape,
      Shape<_1, _1, _1>,
      mutlass::gemm::collective::StageCount<(KStages < 2 ? 2 : KStages)>,
      mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;

  using BuilderTiledMma = typename BuilderCollective::TiledMma;
  using BuilderMmaOp    = typename BuilderTiledMma::Atom::MMA_Op;
  using AtomLayoutQK    = Layout<Shape<Int<TileQ / MmaTileQ>, _1, _1>>;
  using TiledMmaQK      = decltype(make_tiled_mma(BuilderMmaOp{}, AtomLayoutQK{}));

  static constexpr int NumThreadsPerWarp      = mutlass::NumThreadsPerWarp;
  static constexpr int NumThreadsPerWarpSquad = mutlass::NumThreadsPerWarpSquad;
  static constexpr int WarpsPerWarpSquad      = mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumProducerWarpSquads  = 1;
  static constexpr int NumMmaWarpSquads       = TileQ / MmaTileQ;
  static constexpr int NumProducerWarps       = NumProducerWarpSquads * WarpsPerWarpSquad;
  static constexpr int NumConsumerWarps       = NumMmaWarpSquads * WarpsPerWarpSquad;
  static constexpr int NumProducerThreads     = NumProducerWarps * NumThreadsPerWarp;
  static constexpr int NumConsumerThreads     = NumConsumerWarps * NumThreadsPerWarp;
  static constexpr int NumThreads             = NumProducerThreads + NumConsumerThreads;
  static constexpr int QLoadWarpInProducer    = 0;
  static constexpr int KLoadWarpInProducer    = 1;

  static_assert(HeadRatio > 0);
  static_assert(TileQ > 0 && TileKV > 0 && HeadDim > 0);
  static_assert(PageSize == TileKV, "max-score currently requires one K tile per KV page");
  static_assert(IsPagedKV == (PageTableMode != PageTableKind::Dense));
  static_assert(TileQ % HeadRatio == 0);
  static_assert(TileQ % MmaTileQ == 0);
  static_assert(TileQ % NumMmaWarpSquads == 0);
  static_assert(QTokensPerTile <= TileKV, "one Q work must not span more than one K tile");
  static_assert(QStages > 0 && KStages > 0);
  static_assert(HeadDim % 8 == 0);

  static_assert(decltype(size(TiledMmaQK{}))::value == NumConsumerThreads);

  using SmemAtomLayoutQ = typename BuilderCollective::SmemLayoutAtomA;
  using SmemAtomLayoutK = typename BuilderCollective::SmemLayoutAtomB;
  using SmemLayoutQ     = decltype(tile_to_shape(SmemAtomLayoutQ{},
                                             make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<QStages>{})));
  using SmemLayoutK     = decltype(tile_to_shape(SmemAtomLayoutK{},
                                             make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<KStages>{})));

  using QRowTile       = Layout<Shape<Int<HeadRatio>, Int<QTokensPerTile>>, Stride<_1, Int<HeadRatio>>>;
  using SmemLayoutQTme = decltype(flatten(composition(SmemLayoutQ{}, make_tile(QRowTile{}, _, _))));

  using ShapeQTme       = Shape<int32_t, int32_t, int32_t, int32_t>;  // (head_local, token_q, d, head_kv)
  using StrideQTme      = Stride<int64_t, int64_t, _1, int64_t>;
  using ShapeKTme       = Shape<int32_t, int32_t, int32_t, int32_t>;  // (token_k, d, head_kv, page)
  using StrideKTme      = Stride<int64_t, _1, int64_t, int64_t>;
  using ShapePageTable  = Shape<int32_t, int32_t>;
  using StridePageTable = Stride<int64_t, _1>;
  using ShapeMaxScore   = Shape<int32_t, int32_t, int32_t>;  // (token_q, head_q, k_tile)
  using StrideMaxScore  = Stride<int64_t, int64_t, _1>;

  using TmeTileShapeQ = Shape<Int<HeadRatio>, Int<QTokensPerTile>, Int<HeadDim>>;
  using TmeTileShapeK = Shape<Int<TileKV>, Int<HeadDim>>;

  using TME_Q = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQTme{}, int32_t(0)), StrideQTme{}),
      take<0, 3>(SmemLayoutQTme{}),
      TmeTileShapeQ{}));

  using TME_K          = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideKTme{}, int32_t(0)), StrideKTme{}),
      take<0, 2>(SmemLayoutK{})));
  using PagedKvManager = mate::attention::msa::KVManager<IsPagedKV, PageSize, PageTableMode>;

  // Rotate two async barrier pairs per shared-memory stage, matching master.
  using MainloopPipelineQ                = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<QStages, 2>;
  using MainloopPipelineK                = mutlass::Mp31PipelineTmeAsyncWarpSpecialized<KStages, 2>;
  static constexpr int AsyncBarrierCount = static_cast<int>(mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount) +
                                           static_cast<int>(MainloopPipelineQ::NumBarriers) +
                                           static_cast<int>(MainloopPipelineK::NumBarriers);
  static_assert(AsyncBarrierCount <=
                    static_cast<int>(mutlass::arch::AsyncBarrier::HardwareMaxNumAsyncTransactionBarriers),
                "MSA max-score async barrier configuration exceeds the MP31 hardware limit");
  using PipelineQParams = typename MainloopPipelineQ::Params;
  using PipelineKParams = typename MainloopPipelineK::Params;
  using PipelineQState  = typename MainloopPipelineQ::PipelineState;
  using PipelineKState  = typename MainloopPipelineK::PipelineState;

  static constexpr int TmeTransactionBytesQ =
      mutlass::bits_to_bytes(size(take<0, 3>(SmemLayoutQTme{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits_v<Element>);

  struct SharedStorage {
    mute::array_aligned<Element, cosize_v<SmemLayoutQ>, SmemAlignmentBytes> smem_q;
    mute::array_aligned<Element, cosize_v<SmemLayoutK>, SmemAlignmentBytes> smem_k;
  };

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t pipeline_q[MainloopPipelineQ::NumBarriers];
    uint8_t pipeline_k[MainloopPipelineK::NumBarriers];
  };

  struct Arguments {
    Element const* ptr_q;
    Element const* ptr_k;
    int32_t const* ptr_page_table;
    int32_t const* ptr_kv_page_indptr;
    int32_t const* ptr_cu_seqlens_q;
    int32_t const* ptr_cu_seqlens_k;
    int32_t const* ptr_qo_offset;
    float*         ptr_max_score;
  };

  struct TensorParams {
    Element const*  ptr_q;
    Element const*  ptr_k;
    int32_t const*  ptr_page_table;
    int32_t const*  ptr_kv_page_indptr;
    ShapePageTable  shape_page_table;
    StridePageTable stride_page_table;
    int32_t const*  ptr_cu_seqlens_q;
    int32_t const*  ptr_cu_seqlens_k;
    int32_t const*  ptr_qo_offset;
    float*          ptr_max_score;
    int             batch_size;
    int             total_q;
    int             total_k;
    int             num_qo_heads;
    int             num_kv_heads;
    int             max_seqlen_q;
    int             max_seqlen_k;
    int             max_k_tiles;
  };

  struct TmeLoadQParams {
    TME_Q     tme_load;
    ShapeQTme shape;
  };

  struct TmeLoadKParams {
    TME_K     tme_load;
    ShapeKTme shape;
  };

  struct Params {
    TensorParams   args;
    TmeLoadQParams load_q;
    TmeLoadKParams load_k;
  };

  static constexpr int MaxStoreRowsPerThread = 2;
  struct RowmaxStoreParams {
    int64_t output_row_base[MaxStoreRowsPerThread];
  };

  struct Pipeline {
    MainloopPipelineQ q;
    MainloopPipelineK k;
    PipelineQState    q_read;
    PipelineQState    q_write;
    PipelineKState    k_read;
    PipelineKState    k_write;

    static MUTLASS_DEVICE PipelineQParams make_q_params() {
      PipelineQParams params{};
      params.transaction_bytes = TmeTransactionBytesQ;
      params.num_consumers     = NumConsumerWarps;
      return params;
    }

    static MUTLASS_DEVICE PipelineKParams make_k_params() {
      PipelineKParams params{};
      params.transaction_bytes = TmeTransactionBytesK;
      params.num_consumers     = NumConsumerWarps;
      return params;
    }

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* barrier_storage)
        : q(make_q_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_q))),
          k(make_k_params(), static_cast<uint32_t>(reinterpret_cast<uint64_t>(&barrier_storage->pipeline_k))),
          q_read{},
          q_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineQ>()),
          k_read{},
          k_write(mutlass::make_producer_start_state_warpspecialized<MainloopPipelineK>()) {
    }
  };

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    int max_pages_per_batch = mutlass::ceil_div(problem_size.max_seqlen_k, PageSize);
    int page_extent         = IsPagedKV ? mutlass::ceil_div(problem_size.total_k, PageSize) : 1;

    ShapeQTme  shape_q = make_shape(HeadRatio, problem_size.total_q, HeadDim, problem_size.num_kv_heads);
    StrideQTme stride_q =
        make_stride(int64_t(HeadDim), int64_t(problem_size.num_qo_heads) * HeadDim, _1{}, int64_t(HeadRatio) * HeadDim);

    ShapeKTme  shape_k{};
    StrideKTme stride_k{};
    if constexpr (IsPagedKV) {
      shape_k  = make_shape(TileKV, HeadDim, problem_size.num_kv_heads, page_extent);
      stride_k = make_stride(int64_t(problem_size.num_kv_heads) * HeadDim,
                             _1{},
                             int64_t(HeadDim),
                             int64_t(PageSize) * problem_size.num_kv_heads * HeadDim);
    } else {
      shape_k  = make_shape(problem_size.total_k, HeadDim, problem_size.num_kv_heads, 1);
      stride_k = make_stride(int64_t(problem_size.num_kv_heads) * HeadDim, _1{}, int64_t(HeadDim), int64_t(0));
    }

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_q), shape_q, stride_q);
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_k), shape_k, stride_k);

    ShapePageTable  shape_page_table  = make_shape(problem_size.batch_size, max_pages_per_batch);
    StridePageTable stride_page_table = make_stride(int64_t(max_pages_per_batch), _1{});

    TensorParams tensor_args{
        args.ptr_q,
        args.ptr_k,
        IsPagedKV ? args.ptr_page_table : nullptr,
        IsPagedKV ? args.ptr_kv_page_indptr : nullptr,
        shape_page_table,
        stride_page_table,
        args.ptr_cu_seqlens_q,
        args.ptr_cu_seqlens_k,
        args.ptr_qo_offset,
        args.ptr_max_score,
        problem_size.batch_size,
        problem_size.total_q,
        problem_size.total_k,
        problem_size.num_qo_heads,
        problem_size.num_kv_heads,
        problem_size.max_seqlen_q,
        problem_size.max_seqlen_k,
        problem_size.max_k_tiles,
    };

    TmeLoadQParams load_q{
        make_tme_copy(MP31_TME_LOAD{}, mQ, take<0, 3>(SmemLayoutQTme{}), TmeTileShapeQ{}),
        shape_q,
    };
    TmeLoadKParams load_k{
        make_tme_copy(MP31_TME_LOAD{}, mK, take<0, 2>(SmemLayoutK{})),
        shape_k,
    };
    return {tensor_args, load_q, load_k};
  }

  template <class WorkTile>
  static MUTLASS_DEVICE int safe_page_idx(Params const& params, WorkTile const& work_tile) {
    if constexpr (IsPagedKV) {
      if (work_tile.k_tile_begin < work_tile.kv_len && work_tile.k_tile_idx < get<1>(params.args.shape_page_table)) {
        PagedKvManager manager = PagedKvManager::create(params.args.ptr_page_table,
                                                        params.args.ptr_kv_page_indptr,
                                                        get<0>(params.args.stride_page_table),
                                                        params.args.ptr_cu_seqlens_k,
                                                        nullptr);
        return manager.physical_block_index(work_tile.batch_idx, work_tile.k_tile_idx);
      }
    }
    return 0;
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_q_gmem(TmeLoadQParams const& load_q, WorkTile const& work_tile) {
    Tensor mQ_head = load_q.tme_load.get_tme_tensor(load_q.shape)(_, _, _, work_tile.head_kv);
    Tensor mQ_tile = domain_offset(make_coord(_0{}, work_tile.q_abs_begin, _0{}), mQ_head);
    return local_tile(mQ_tile, TmeTileShapeQ{}, make_coord(_0{}, _0{}, _0{}));
  }

  template <class WorkTile>
  static MUTLASS_DEVICE auto make_tme_k_gmem(TmeLoadKParams const& load_k, WorkTile const& work_tile, int page_idx) {
    if constexpr (IsPagedKV) {
      Tensor mK_paged = load_k.tme_load.get_tme_tensor(load_k.shape)(_, _, work_tile.head_kv, page_idx);
      return local_tile(mK_paged, TmeTileShapeK{}, make_coord(_0{}, _0{}));
    } else {
      Tensor mK_contiguous = load_k.tme_load.get_tme_tensor(load_k.shape)(_, _, work_tile.head_kv, _0{});
      Tensor mK_tile = domain_offset(make_coord(work_tile.k_batch_begin + work_tile.k_tile_begin, _0{}), mK_contiguous);
      return local_tile(mK_tile, TmeTileShapeK{}, make_coord(_0{}, _0{}));
    }
  }

  template <class WorkTile>
  static MUTLASS_DEVICE int k_tile_count(Params const& params, WorkTile const& work_tile) {
    int k_tiles = work_tile.valid_k_tiles;
    if constexpr (IsCausal) {
      // A Q tile only needs keys visible to its last valid query.  The prior
      // implementation computed every K tile and masked roughly half of the
      // scores back to -inf during causal prefill.  The output is initialized
      // to -inf by the caller, so future K tiles can be skipped altogether.
      int last_q_local   = work_tile.q_local_begin + work_tile.q_count - 1;
      int visible_keys   = last_q_local + params.args.ptr_qo_offset[work_tile.batch_idx] + 1;
      int causal_k_tiles = visible_keys > 0 ? mutlass::ceil_div(visible_keys, TileKV) : 0;
      if (causal_k_tiles < k_tiles) {
        k_tiles = causal_k_tiles;
      }
    }
    return k_tiles;
  }

  template <class WorkTile>
  static MUTLASS_DEVICE int first_masked_k_tile(Params const& params, WorkTile const& work_tile) {
    // A partial KV tail starts at floor(kv_len / TileKV).  When kv_len is
    // aligned there is no tail tile, so begin after the final valid tile.
    int first_masked = work_tile.valid_k_tiles;
    if (work_tile.kv_len % TileKV != 0) {
      first_masked = work_tile.kv_len / TileKV;
    }
    if constexpr (IsCausal) {
      // Count the keys visible to the first Q row. Any K tile starting at or
      // after this quotient needs the per-row causal predicate. A Q work spans
      // at most TileKV rows, so this covers at most two boundary tiles.
      int visible_to_first = work_tile.q_local_begin + params.args.ptr_qo_offset[work_tile.batch_idx] + 1;
      visible_to_first     = visible_to_first > 0 ? visible_to_first : 0;
      int first_causal     = visible_to_first / TileKV;
      if (first_causal < first_masked) {
        first_masked = first_causal;
      }
    }
    return first_masked;
  }

  template <class WorkTile>
  MUTLASS_DEVICE void load_q(Params const&   params,
                             Pipeline&       pipeline,
                             SharedStorage&  shared_storage,
                             WorkTile const& work_tile) const {
    pipeline.q.producer_acquire(pipeline.q_write);
    uint32_t bar_id  = pipeline.q.producer_get_barrier_id(pipeline.q_write);
    auto     cta_tme = params.load_q.tme_load.get_slice(0);
    Tensor   gQ      = make_tme_q_gmem(params.load_q, work_tile);
    Tensor   sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQTme{})(_, _, _, pipeline.q_write.index());
    copy(params.load_q.tme_load.with(bar_id), cta_tme.partition_S(gQ), cta_tme.partition_D(sQ));
    ++pipeline.q_write;
  }

  template <class WorkTile>
  MUTLASS_DEVICE void load_k(Params const&   params,
                             Pipeline&       pipeline,
                             SharedStorage&  shared_storage,
                             WorkTile const& work_tile,
                             int             page_idx) const {
    pipeline.k.producer_acquire(pipeline.k_write);
    uint32_t bar_id  = pipeline.k.producer_get_barrier_id(pipeline.k_write);
    auto     cta_tme = params.load_k.tme_load.get_slice(0);
    Tensor   gK      = make_tme_k_gmem(params.load_k, work_tile, page_idx);
    Tensor   sK      = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    copy(params.load_k.tme_load.with(bar_id),
         cta_tme.partition_S(gK),
         cta_tme.partition_D(sK(_, _, pipeline.k_write.index())));
    ++pipeline.k_write;
  }

  template <class WorkTile>
  MUTLASS_DEVICE void prefetch_k(Params const& params, WorkTile const& work_tile, int page_idx) const {
    Tensor gK      = make_tme_k_gmem(params.load_k, work_tile, page_idx);
    auto   cta_tme = params.load_k.tme_load.get_slice(0);
    mute::prefetch(params.load_k.tme_load, cta_tme.partition_S(gK));
  }

  MUTLASS_DEVICE void wait_q(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    if (lane_idx == 0) {
      pipeline.q.consumer_wait(pipeline.q_read);
    }
    __syncwarp();
  }

  MUTLASS_DEVICE void release_q(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    __syncwarp();
    if (lane_idx == 0) {
      pipeline.q.consumer_release(pipeline.q_read);
    }
    ++pipeline.q_read;
  }

  MUTLASS_DEVICE void wait_k(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    if (lane_idx == 0) {
      pipeline.k.consumer_wait(pipeline.k_read);
    }
    __syncwarp();
  }

  MUTLASS_DEVICE void release_k(Pipeline& pipeline, int consumer_thread_idx) const {
    int lane_idx = consumer_thread_idx % NumThreadsPerWarp;
    __syncwarp();
    if (lane_idx == 0) {
      pipeline.k.consumer_release(pipeline.k_read);
    }
    ++pipeline.k_read;
  }

  template <class WorkTile>
  MUTLASS_DEVICE RowmaxStoreParams make_rowmax_store_params(Params const&   params,
                                                            WorkTile const& work_tile,
                                                            int             consumer_thread_idx) const {
    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    Tensor     cQK        = make_identity_tensor(make_shape(Int<TileQ>{}, Int<TileKV>{}));
    Tensor     tCcQK      = thr_mma_qk.partition_C(cQK);
    Tensor     tCcQK_mn = make_tensor(tCcQK.data(), mate::attention::fmha::layout_acc_mn(tiled_mma_qk, tCcQK.layout()));
    static_assert(size<0>(tCcQK_mn) <= MaxStoreRowsPerThread, "unexpected max-score rows per thread");

    RowmaxStoreParams store_params{{-1, -1}};
    constexpr int     reduction_size = size(mate::attention::fmha::reduction_target_n(TiledMmaQK{}));
    int               lane_idx       = consumer_thread_idx % NumThreadsPerWarp;
    if ((lane_idx % reduction_size) == 0) {
      MUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0>(tCcQK_mn); ++m) {
        int row             = int(get<0>(tCcQK_mn(m, 0)));
        int token_row_local = row / HeadRatio;
        if (token_row_local < work_tile.q_count) {
          int head_local = row - token_row_local * HeadRatio;
          int q_abs      = work_tile.q_abs_begin + token_row_local;
          int head_q     = work_tile.head_kv * HeadRatio + head_local;
          store_params.output_row_base[m] =
              (int64_t(q_abs) * params.args.num_qo_heads + head_q) * params.args.max_k_tiles;
        }
      }
    }
    return store_params;
  }

  template <class AccQK, class WorkTile>
  MUTLASS_DEVICE void apply_score_mask(Params const&   params,
                                       AccQK&          acc_qk,
                                       WorkTile const& work_tile,
                                       int             consumer_thread_idx) const {
    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    Tensor acc_qk_mn = make_tensor(acc_qk.data(), mate::attention::fmha::layout_acc_mn(tiled_mma_qk, acc_qk.layout()));
    Tensor cQK       = make_identity_tensor(make_shape(Int<TileQ>{}, Int<TileKV>{}));
    Tensor tCcQK     = thr_mma_qk.partition_C(cQK);
    Tensor tCcQK_mn  = make_tensor(tCcQK.data(), mate::attention::fmha::layout_acc_mn(tiled_mma_qk, tCcQK.layout()));

    int q_token_begin = work_tile.q_local_begin;
    int k_begin       = work_tile.k_tile_begin;
    int causal_offset = IsCausal ? params.args.ptr_qo_offset[work_tile.batch_idx] : 0;

    // Consume each column predicate immediately for every accumulator row.
    // This keeps the causal/tail predicate live range short after the fully
    // unrolled loops, matching the source structure of the TileLang kernel.
    MUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(acc_qk_mn); ++n) {
      int col     = int(get<1>(tCcQK_mn(0, n)));
      int k_local = k_begin + col;
      MUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
        if constexpr (IsCausal) {
          int row        = int(get<0>(tCcQK_mn(m, n)));
          int token_row  = row / HeadRatio;
          int q_local    = q_token_begin + token_row;
          int last_valid = std::min(work_tile.kv_len - 1, q_local + causal_offset);
          if (k_local > last_valid) {
            acc_qk_mn(m, n) = -std::numeric_limits<float>::infinity();
          }
        } else if (k_local >= work_tile.kv_len) {
          acc_qk_mn(m, n) = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }

  template <class AccQK, class WorkTile>
  MUTLASS_DEVICE void store_rowmax(Params const&            params,
                                   AccQK&                   acc_qk,
                                   WorkTile const&          work_tile,
                                   RowmaxStoreParams const& store_params) const {
    TiledMmaQK tiled_mma_qk;

    auto          reduction_target_qk = mate::attention::fmha::reduction_target_n(tiled_mma_qk);
    constexpr int red_rank            = decltype(rank(reduction_target_qk))::value;
    Tensor acc_qk_mn = make_tensor(acc_qk.data(), mate::attention::fmha::layout_acc_mn(tiled_mma_qk, acc_qk.layout()));

    static_assert(size<1>(acc_qk_mn) % 4 == 0, "N must be a multiple of 4");
    static_assert(size<0>(acc_qk_mn) <= MaxStoreRowsPerThread, "unexpected max-score rows per thread");

    float4 row_max_cur[MaxStoreRowsPerThread];
    float  row_max[MaxStoreRowsPerThread];

    MUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(acc_qk_mn); n += 4) {
      MUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
        float4 vec_acc = make_float4(
            float(acc_qk_mn(m, n)), float(acc_qk_mn(m, n + 1)), float(acc_qk_mn(m, n + 2)), float(acc_qk_mn(m, n + 3)));
        if (n == 0) {
          row_max_cur[m] = vec_acc;
        } else {
          mute::max(row_max_cur[m], vec_acc, row_max_cur[m]);
        }
      }
    }

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
      float2 row_max_pair;
      mute::max(row_max_pair,
                make_float2(row_max_cur[m].x, row_max_cur[m].y),
                make_float2(row_max_cur[m].z, row_max_cur[m].w));
      row_max[m] = max(row_max_pair.x, row_max_pair.y);
    }

    for_each(make_seq<red_rank>{}, [&](auto r) {
      MUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < shape<r>(reduction_target_qk); j *= 2) {
        MUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
          row_max[m] = max(row_max[m], __shfl_xor_sync(uint32_t(-1), row_max[m], stride<r>(reduction_target_qk) * j));
        }
      }
    });

    MUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(acc_qk_mn); ++m) {
      int64_t output_row_base = store_params.output_row_base[m];
      if (output_row_base >= 0) {
        params.args.ptr_max_score[output_row_base + work_tile.k_tile_idx] = row_max[m];
      }
    }
  }

  template <class AccQK, class WorkTile>
  MUTLASS_DEVICE void mask_and_store_rowmax(Params const&            params,
                                            AccQK&                   acc_qk,
                                            WorkTile const&          work_tile,
                                            int                      first_masked_k_tile,
                                            RowmaxStoreParams const& store_params,
                                            int                      consumer_thread_idx) const {
    // The Q-work prologue has already found the first possible causal/length
    // boundary. Keep the common path to one comparison and evaluate the full
    // per-score predicate only for the one or two boundary tiles.
    if (work_tile.k_tile_idx >= first_masked_k_tile) {
      apply_score_mask(params, acc_qk, work_tile, consumer_thread_idx);
    }
    store_rowmax(params, acc_qk, work_tile, store_params);
  }

  template <class WorkTile>
  MUTLASS_DEVICE void compute_k_tile(Params const&            params,
                                     Pipeline&                pipeline,
                                     SharedStorage&           shared_storage,
                                     WorkTile const&          work_tile,
                                     int                      first_masked_k_tile,
                                     RowmaxStoreParams const& store_params,
                                     int                      consumer_thread_idx) const {
    wait_k(pipeline, consumer_thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{})(_, _, pipeline.q_read.index());
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{})(_, _, pipeline.k_read.index());

    TiledMmaQK tiled_mma_qk;
    auto       thr_mma_qk = tiled_mma_qk.get_thread_slice(consumer_thread_idx);
    Tensor     tSrQ       = thr_mma_qk.partition_fragment_A(sQ);
    Tensor     tSrK       = thr_mma_qk.partition_fragment_B(sK);
    Tensor     acc_qk     = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShape{}));
    clear(acc_qk);

    mute::gemm(tiled_mma_qk, tSrQ, tSrK, acc_qk);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();
    release_k(pipeline, consumer_thread_idx);

    mask_and_store_rowmax(params, acc_qk, work_tile, first_masked_k_tile, store_params, consumer_thread_idx);
  }
};

}  // namespace mate::attention::msa::collective
