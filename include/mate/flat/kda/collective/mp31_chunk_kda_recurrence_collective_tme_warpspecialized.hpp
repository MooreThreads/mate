#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mute/arch/copy_mp31.hpp>
#include <mute/swizzle_layout.hpp>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/collective/flat_collective_inverse_nxn.hpp"
#include "mate/flat/collective/flat_collective_load_tme.hpp"
#include "mate/flat/collective/flat_collective_mma_tme_sqmma.hpp"
#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/kda_permutes.hpp"
#include "mate/flat/kda/named_barrier.hpp"
#include "mate/flat/simd_helper.hpp"

namespace mate::flat::kda {

namespace recurrence {

using namespace mute;
using mate::flat::simd::pack_from;
using mate::flat::simd::unpack_to;

template <bool UsePermute, class TiledMma, class PermuteTile>
struct MaybePermutedSqmma {
  using Type = TiledMma;
};

template <class TiledMma, class PermuteTile>
struct MaybePermutedSqmma<true, TiledMma, PermuteTile> {
  using Type = decltype(mate::convert_to_permuted_sqmma(TiledMma{}, PermuteTile{}));
};

template <class Components_, class TileShape_, class StateElement_, class StrideV_, class StrideO_>
struct Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized {
  using ArchTag                     = mutlass::arch::Mp31;
  using Components                  = Components_;
  using Element                     = typename Components::Element;
  using StateElement                = StateElement_;
  using TileShape                   = TileShape_;
  using StrideQK                    = ChunkKdaQkStride;
  using StrideV                     = StrideV_;
  using StrideO                     = StrideO_;
  static constexpr bool HasStateIn  = Components::HasStateIn;
  static constexpr bool HasStateOut = Components::HasStateOut;
  // Per-chunk boundary snapshots.  Off by default: the whole publish path is
  // `if constexpr`-gated so the default (feature-off) build stays byte-for-byte
  // unchanged.
  static constexpr bool HasChunkStatesOut = Components::HasChunkStatesOut;
  static constexpr bool IsVarlen          = Components::IsVarlen;

  static constexpr int kChunk     = Components::kChunk;
  static constexpr int kHeadDim   = Components::kHeadDim;
  static constexpr int kValueDim  = Components::kValueDim;
  static constexpr int kValueTile = Components::kRecurrenceValueTile;
  // AlignmentA/AlignmentB for every CollectiveBuilder below: the operands
  // are 16-byte aligned.
  static constexpr int kOperandAlignment = 8;
  using Workspace                        = typename Components::Workspace;
  // The inbound tile shape is a contract only -- nothing below builds from it.
  static constexpr int TileM = get<0>(TileShape{});
  static constexpr int TileN = get<1>(TileShape{});
  static constexpr int TileK = get<2>(TileShape{});
  static_assert(TileM == kChunk && TileN == kChunk && TileK == kHeadDim);
  static_assert(kChunk % 2 == 0);
  static_assert(TileM == TileN);
  static_assert(kHeadDim == 128);

  // K2 owns the recurrent traversal.  Every input ring is ONE stage deep and is
  // driven by explicit named barriers -- the same machinery the MP32 recurrence
  // uses -- so nothing here builds a mutlass pipeline.  A second TME in flight
  // only takes smem bandwidth away from the SQMMAs reading the first one; the
  // producers never stall waiting for a stage to come back either way.
  static constexpr int TmeStages = 1;
  // The warp-specialized MP31 mainloop the collectives are built from asserts
  // `Stages >= 2`, so the collectives keep a two-stage policy they never run: the
  // recurrence re-stages every operand with its own depth (TmeStages) below and
  // only takes the instruction, the atom and the operand tile off them.
  using DummyStageCount                  = mutlass::gemm::collective::StageCount<2>;
  static constexpr int NumProducerSquads = 1;
  // NumStateSquads is how many warp squads run the state MMA.  With a single
  // (M=64, N=128) atom the whole headDim fits one 4-warp squad.
  static constexpr int NumStateSquads     = 1;
  static constexpr int NumOutputSquads    = 1;
  static constexpr int NumProducerThreads = NumProducerSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumStateThreads    = NumStateSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumOutputThreads   = NumOutputSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumProducerWarps   = NumProducerSquads * mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumStateWarps      = NumStateSquads * mutlass::NumWarpsPerWarpSquad;
  // The value axis is covered by the grid (kValueDim / kValueTile
  // tiles), not by a CTA cluster, so only divisibility is required here.
  static_assert(kValueDim % kValueTile == 0);
  static constexpr int NumOutputWarps = NumOutputSquads * mutlass::NumWarpsPerWarpSquad;
  struct WarpSpecRole {
    static constexpr int Producer      = 0;
    static constexpr int Output        = 1;
    static constexpr int State         = 2;
    static constexpr int ProducerSquad = Producer;
    static constexpr int OutputSquad   = Output;
    static constexpr int StateSquad    = State;
    static constexpr int DecayedWarp   = 0;
    static constexpr int RestoredWarp  = 1;
    static constexpr int InverseWarp   = 2;
    static constexpr int ScaleWarp     = 3;
  };
  // The concatenated decayed tile is laid out as
  //   K[0:k/2], Q[0:k/2], K[k/2:k], Q[k/2:k].
  static_assert(mutlass::NumWarpsPerWarpSquad % 2 == 0);
  // ---- the named-barrier map ------------------------------------------------
  // The ids are RecurrenceBarriers' (named_barrier.hpp); this arch stages, per
  // input ring, a full/empty pair, and consumes them as:
  //   Decayed   decayed workspace -> OUTPUT squad (DS A operand)
  //   Restored  k_restored        -> STATE squad (UK B operand)
  //   V         v                 -> OUTPUT squad (residual source)
  //   InvP      inverse/P/total   -> OUTPUT (inverse, P) + STATE (total)
  //   BarrierMap::Local     one rendezvous per consumer squad
  //   BarrierMap::Seeded    producer -> both consumer squads, once per work item
  //   BarrierMap::Drained   state squad -> producer, once per work item
  //   BarrierMap::UReady          OUTPUT squad -> STATE squad: U is in smem, run UK
  //   BarrierMap::StateCommitted  STATE squad  -> OUTPUT squad: the updated state is in smem
  using BarrierMap = RecurrenceBarriers<TmeStages,
                                        NumStateSquads + NumOutputSquads,
                                        /*HasVRing=*/true,
                                        /*HasStateSnapshot=*/true>;
  static_assert(BarrierMap::NumBarriers == 15 && BarrierMap::StateSnapshot == 14,
                "the MP31 recurrence barrier map size moved");
  static constexpr int SmemAlignmentBytes = 256;

  using TileShapeDS = Shape<Int<2 * kChunk>, Int<kValueTile>, Int<kHeadDim>>;
  // K1 permutes the k_restored workspace's headDim axis, which lets a state
  // squad lane hold 16 *consecutive* logical k when headDim is the UK
  // accumulator's N axis.  Hence the swap: A = U^T (M = value), B = kr
  // (N = headDim).  The atom stays 128 threads wide (64x64) and two of them
  // tile N, so both state squads stay busy and the accumulator stays 32/lane.
  using TileShapeKU = Shape<Int<kValueTile>, Int<kHeadDim>, Int<kChunk>>;
  using TileShapeQS = Shape<Int<kChunk>, Int<kValueTile>, Int<kHeadDim>>;
  using TileShapePU = Shape<Int<kChunk>, Int<kValueTile>, Int<kChunk>>;
  using TileShapeV  = Shape<Int<kChunk>, Int<kChunk>, Int<kValueTile>>;

  // Built the way MP32 builds its collectives: the builder derives the SQMMA
  // instruction and the smem layouts from the tile shape, and the cluster is
  // trivial (one CTA owns one kValueTile-wide value tile).
  using CollectiveMmaDS =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQK,
                                                            kOperandAlignment,
                                                            Element,
                                                            StrideQK,
                                                            kOperandAlignment,
                                                            float,
                                                            TileShapeDS,
                                                            Shape<_1, _1, _1>,
                                                            DummyStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;

  using CollectiveMmaKU =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideV,
                                                            kOperandAlignment,
                                                            Element,
                                                            StrideV,
                                                            kOperandAlignment,
                                                            float,
                                                            TileShapeKU,
                                                            Shape<_1, _1, _1>,
                                                            DummyStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;
  using CollectiveMmaQS =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQK,
                                                            kOperandAlignment,
                                                            Element,
                                                            StrideV,
                                                            kOperandAlignment,
                                                            float,
                                                            TileShapeQS,
                                                            Shape<_1, _1, _1>,
                                                            DummyStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;
  using CollectiveMmaPU =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideQK,
                                                            kOperandAlignment,
                                                            Element,
                                                            StrideV,
                                                            kOperandAlignment,
                                                            float,
                                                            TileShapePU,
                                                            Shape<_1, _1, _1>,
                                                            DummyStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;

  using TiledMmaDS = typename CollectiveMmaDS::TiledMma;
  using TiledMmaKU = typename CollectiveMmaKU::TiledMma;
  using TiledMmaQS = typename CollectiveMmaQS::TiledMma;
  using TiledMmaPU = typename CollectiveMmaPU::TiledMma;
  static_assert(DummyStageCount::value == CollectiveMmaDS::DispatchPolicy::Stages);
  static_assert(decltype(size(TiledMmaDS{}))::value == NumOutputThreads);
  static_assert(decltype(size(TiledMmaKU{}))::value == NumStateThreads);
  static_assert(decltype(size(TiledMmaQS{}))::value == NumOutputThreads);
  static_assert(decltype(size(TiledMmaPU{}))::value == NumOutputThreads);

  using SmemLayoutDecayed =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaDS::SmemLayoutA{}, Int<TmeStages>{}));
  using SmemLayoutDecayedTile = decltype(take<0, 2>(SmemLayoutDecayed{}));
  // V is only a TME destination and a residual source, so its operand layout
  // must NOT follow the UK swap.  Pin it to the pre-swap UK collective's B
  // layout (probe-verified byte-identical to the old SmemLayoutV).
  using CollectiveMmaUK =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideV,
                                                            kOperandAlignment,
                                                            Element,
                                                            StrideV,
                                                            kOperandAlignment,
                                                            float,
                                                            Shape<Int<kHeadDim>, Int<kValueTile>, Int<kChunk>>,
                                                            Shape<_1, _1, _1>,
                                                            DummyStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;
  using SmemLayoutKRestored =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaKU::SmemLayoutB{}, Int<TmeStages>{}));
  using SmemLayoutKRestoredTile = decltype(take<0, 2>(SmemLayoutKRestored{}));
  // The collectives are built with a two-stage policy (DummyStageCount); U,
  // inverse and P are single tiles here, so each is re-staged with the
  // recurrence's own depth -- the tile is take<0,2>, and one stage is all the
  // single-buffered handshakes need.
  using SmemLayoutU = decltype(mate::unstage_smem_layout(typename CollectiveMmaPU::SmemLayoutB{}, Int<TmeStages>{}));
  using SmemLayoutV = decltype(mate::unstage_smem_layout(typename CollectiveMmaUK::SmemLayoutB{}, Int<TmeStages>{}));
  // The recurrent state is one live snapshot, not a double-buffered TME
  // operand.  BarrierMap::StateCommitted also orders reuse of the U stage: the state
  // squad publishes the next state only after completing the prior U update.
  using SmemLayoutState = decltype(take<0, 2>(typename CollectiveMmaDS::SmemLayoutB{}));
  using SmemLayoutInverseA =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaPU::SmemLayoutA{}, Int<TmeStages>{}));
  using SmemLayoutP = SmemLayoutInverseA;

  // ---- chunk_states: pool rows written by a single TME store ---------------
  // `SmemLayoutState` is the canonical row-major (64,128):(128,1) frame carrying
  // Sw<4,4,4>; the v-axis permute lives in the *writers*, not in the layout.
  // Re-expose the permuted v axis as two 8-row sub-modes so the 8x8 block
  // transpose becomes a pure stride exchange inside one TME descriptor (the
  // mapping is unchanged: 8*r_hi + r_lo == r, so 1024*r_hi + 128*r_lo == 128*r).
  using SmemLayoutStateBlocked = decltype(composition(
      SmemLayoutState{}.layout_a(),
      SmemLayoutState{}.offset(),
      make_layout(make_shape(make_shape(_8{}, _8{}), _128{}), make_stride(make_stride(_1024{}, _128{}), _1{}))));
  // smem coord ((r_hi,r_lo),k) -> flat coord of the *natural* pool row, i.e.
  // b*128 + a*1024 + k with the pool row viewed as (b:8, a:16, k:128):(128,1024,1).
  using TilerStateChunkStates =
      decltype(make_layout(make_shape(make_shape(_8{}, _8{}), _128{}), make_stride(make_stride(_1{}, _8{}), _128{})));
  // One rank-4 descriptor covers the whole pool: dim3 is the pool row and `a`
  // selects the 64-row half (value_start / 8), so a per-chunk store only changes
  // coordinates.  The inner 128 elements are the contiguous k axis, which is what
  // makes dim0 = 128 legal (TME wants dim0 to be a multiple of 8 elements).
  //
  // The pool is `Element`-typed (the activation), not `StateElement`: the source
  // tile is the activation-typed snapshot, and the checkpoint cadence allows an
  // fp32 recurrent state whose pool stays io-dtype (cudnn's `state_checkpoints`).
  using TMEChunkStates = decltype(make_tme_copy(MP31_TME_STORE{},
                                                make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                                                            make_layout(make_shape(_8{}, _16{}, _128{}, int32_t{}),
                                                                        make_stride(_128{}, _1024{}, _1{}, _16384{}))),
                                                SmemLayoutStateBlocked{},
                                                TilerStateChunkStates{}));

  static constexpr int R2SVectorBits    = kValueTile == 32 ? 16 : 128;
  static constexpr int R2SMmaAtomN      = 8;
  static constexpr int R2SFragmentSize  = R2SVectorBits / sizeof_bits_v<Element>;
  static constexpr int R2SGranularity   = R2SMmaAtomN * R2SFragmentSize;
  static constexpr int SmemTileNForR2S  = decltype(size<0>(SmemLayoutU{}))::value;
  static constexpr int R2SPermuteRepeat = kValueTile == 32 ? 1 : SmemTileNForR2S / R2SGranularity;
  static_assert(kValueTile == 32 || SmemTileNForR2S % R2SGranularity == 0);

  using R2SFragmentType   = mute::uint_bit_t<R2SVectorBits>;
  using UMmaPermuteTile   = decltype(filter(
      make_ordered_layout(Shape<Int<R2SMmaAtomN>, Int<R2SFragmentSize>, Int<R2SPermuteRepeat>>{}, Step<_2, _1, _3>{})));
  using PermuteTileForU   = Tile<Underscore, UMmaPermuteTile, Underscore>;
  using PermuteTiledMmaPU = typename MaybePermutedSqmma<kValueTile != 32, TiledMmaPU, PermuteTileForU>::Type;
  // The SQMMA enumerates V in 8x8-transposed blocks and this tile stores the
  // logical order, so an atom coordinate reaches the tile row through MP31's
  // PermuteV; see kda_permutes.hpp.
  using ValuePermute        = PermuteV<mutlass::arch::Mp31, kValueTile>;
  using TileRowFromInternal = typename ValuePermute::TileRowFromInternal;
  using R2SCopyAtom         = Copy_Atom<UniversalCopy<R2SFragmentType>, Element>;
  using R2STiledCopy        = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaPU{}));

  // Q@State is produced and consumed entirely by the output squad before it
  // advances to the next chunk.  It therefore needs one bridge tile, not a
  // TME-staged allocation.
  static_assert(cosize_v<SmemLayoutDecayed> == 2 * kChunk * kHeadDim * TmeStages);
  static_assert(cosize_v<SmemLayoutP> == kChunk * kChunk * TmeStages);
  static_assert(cosize_v<SmemLayoutState> >= kHeadDim * kValueTile);
  // V is pinned to the pre-swap collective's B layout, so its TME type comes
  // from that collective too (the swapped UK's B operand is kr).
  using TME_V = typename CollectiveMmaUK::Params::TME_B;

  using TMEWorkspaceDecayed   = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), typename Workspace::DecayedLayout{}),
      SmemLayoutDecayedTile{}));
  using TMEWorkspaceKRestored = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), typename Workspace::KRestoredLayout{}),
      SmemLayoutKRestoredTile{}));
  // inverse / P / total are brought in by the inverse+P producer warp rather
  // than TME: the tiles are tiny, and staging the total in smem removes the
  // per-chunk strided gmem reads in scale_state.  inverse/P land in the SQMMA
  // operand layout with plain cooperative loads (a swizzled tile is not a
  // row-major copy target); only the flat total row is a robust LDGSTS copy.
  // The total row is kHeadDim Elements and the producer warp is 32 lanes, so a
  // 64-bit atom (4 Elements) covers it in exactly one pass: 32 * 4 == kHeadDim.
  static constexpr int kTotalVectorBits = 64;
  static constexpr int kTotalFragment   = kTotalVectorBits / sizeof_bits_v<Element>;
  static_assert(kHeadDim % kTotalFragment == 0);
  // See mate/attention/fmha/load_primitive_builder.hpp
  // (Mp31FmhaLsuLoadKeyBuilder): the atom width follows from the element width
  // and the per-lane vector width, and the tiled copy's alignment argument is
  // that same per-lane element count.
  using GmemCopyAtomTotal = mute::MP31_ROBUST_LDGSTS<mute::uint_bit_t<sizeof_bits_v<Element> * kTotalFragment>>;
  using StrideTotalTile   = Stride<Int<kHeadDim>, _1>;
  using GmemTiledCopyTotal =
      decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<mutlass::NumThreadsPerWarp,
                                                                       Element,
                                                                       kTotalFragment,
                                                                       StrideTotalTile,
                                                                       1,
                                                                       kHeadDim,
                                                                       GmemCopyAtomTotal>());
  using SmemLayoutTotal =
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<TmeStages>{}), make_stride(_1{}, Int<kHeadDim>{})));
  // inverse/P staging: one 16 B vector per lane (eight contiguous bf16), the
  // same (16,16) row-major shape the prepare kernel writes, so the gmem side of
  // the copy is coalesced.
  static constexpr int kInvPVectorBits = 128;
  static constexpr int kInvPFragment   = kInvPVectorBits / sizeof_bits_v<Element>;
  static_assert(kChunk % kInvPFragment == 0);
  using GmemCopyAtomInvP  = mute::MP31_ROBUST_LDGSTS<mute::uint_bit_t<kInvPVectorBits>>;
  using StrideInvPTile    = Stride<Int<kChunk>, _1>;
  using GmemTiledCopyInvP = decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<mutlass::NumThreadsPerWarp,
                                                                                             Element,
                                                                                             kInvPFragment,
                                                                                             StrideInvPTile,
                                                                                             kChunk,
                                                                                             kChunk,
                                                                                             GmemCopyAtomInvP>());

  // The four input rings are explicit named-barrier pairs, not pipelines, so
  // there is no PipelineQ/K/G/V, no PipelineState and no mutlass::Mp31Pipeline*.
  //
  // V keeps the proven partition_SD/step pair from CollectiveLoadTme.  That helper
  // asks its state for the stage it is about to fill -- both to select the smem
  // stage and to derive the barrier id -- so the state is seeded from VRing each
  // chunk rather than counting on its own, and the one hook below serves it.
  struct VRingState {
    int                stage = 0;
    MUTLASS_DEVICE int index() const {
      return stage;
    }
    MUTLASS_DEVICE VRingState& operator++() {
      if (++stage == TmeStages) {
        stage = 0;
      }
      return *this;
    }
  };
  struct VRingHooks {
    using PipelineState = VRingState;
    MUTLASS_DEVICE uint32_t producer_get_barrier_id(VRingState const& state) const {
      return named_barrier_id(BarrierMap::VReady + state.index());
    }
  };
  using LoadV = collective::
      CollectiveLoadTme<collective::LoadKind::kV, VRingHooks, Element, SmemLayoutV, TME_V, SmemAlignmentBytes>;

  template <class StorageElement, class SmemLayout>
  using SmemArray = array_aligned<StorageElement, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SharedStorage {
    SmemArray<Element, SmemLayoutDecayed>   smem_decayed;
    typename LoadV::SharedStorage           smem_v_raw;
    SmemArray<Element, SmemLayoutKRestored> smem_k_restored;
    // U is already a builder-staged GEMM operand (the stage is part of
    // CollectiveMmaKU::SmemLayoutB); keep one allocation and select the live
    // stage in the tensor view.
    SmemArray<Element, SmemLayoutU>        smem_u;
    SmemArray<Element, SmemLayoutInverseA> smem_inverse;
    SmemArray<Element, SmemLayoutP>        smem_p;
    SmemArray<Element, SmemLayoutState>    smem_state;
    SmemArray<Element, SmemLayoutTotal>    smem_total;
  };

  static constexpr int     SharedStorageSize  = sizeof(SharedStorage);
  static constexpr int64_t KTileBytes         = int64_t(kChunk) * kHeadDim * sizeof(Element);
  static constexpr int64_t VTileBytes         = int64_t(kChunk) * kValueTile * sizeof(Element);
  static constexpr int64_t KRestoredTileBytes = KTileBytes;

  // `ptr_chunk_states` only exists when the feature is on.  Putting it (and the
  // pool's TME descriptor) in a base class keeps the feature-off Params
  // byte-identical to the pre-chunk_states layout: an unconditional member
  // shifts the kernel parameter block and perturbs the feature-off register
  // allocation (Shared 204 -> 205).
  //
  // `checkpoint_pool_rows` mirrors the contract field of the same name: unlike
  // MP32, this Params does not inherit the collective's `Arguments`, so the
  // resolved descriptor extent has to be copied in here.  The row *mapping* is
  // not here: the checkpoint cadence is a compile-time specialization axis
  // (`Components::CheckpointEveryNTokens`), so this payload carries only the
  // runtime extent.
  //
  // The int32 leads the payload on purpose: trailing it after the TME descriptor
  // could give the struct tail padding, and a derived `Params` is then allowed
  // to place its first member inside that padding, which the
  // `offsetof(Params, ptr_v)` guard below rejects.
  struct ChunkStatesPtrStorage {
    int32_t        checkpoint_pool_rows;
    void*          ptr_chunk_states;
    TMEChunkStates tme_chunk_states;
  };
  using ChunkStatesPtrBase = FeatureGatedParams<HasChunkStatesOut, ChunkStatesPtrStorage>;

  using Arguments = ChunkKdaRecurrenceArguments<Element, StrideV, StrideO, Workspace>;

  struct Params : ChunkStatesPtrBase {
    Element const*         ptr_v;
    void const*            ptr_initial_state;
    Element*               ptr_out;
    void*                  ptr_final_state;
    int32_t const*         ptr_state_indices;
    StrideV                stride_v;
    StrideO                stride_out;
    TMEWorkspaceDecayed    tme_workspace_decayed;
    TMEWorkspaceKRestored  tme_workspace_k_restored;
    Element const*         ptr_workspace_inverse;
    Element const*         ptr_workspace_p;
    Element const*         ptr_workspace_total;
    mute::RobustDescriptor desc_inverse;
    mute::RobustDescriptor desc_p;
    mute::RobustDescriptor desc_total;
    TME_V                  tme_v;
  };

  static_assert(sizeof(FeatureGatedParams<false, ChunkStatesPtrStorage>) == 1);
  static_assert(alignof(FeatureGatedParams<false, ChunkStatesPtrStorage>) == 1);
  static_assert(offsetof(Params, ptr_v) == (HasChunkStatesOut ? sizeof(ChunkStatesPtrStorage) : 0),
                "feature-gated MP31 Params prefix moved");

  // Only the lane that issued a TME store can wait on it, so the chunk_states
  // publish retires its store behind a state-squad-only rendezvous.  The id is
  // reserved unconditionally (see BarrierMap::StateSnapshot) so the feature-off build keeps
  // the same barrier map.
  static constexpr uint32_t kStateSnapshotBarrier = uint32_t(BarrierMap::StateSnapshot);

  // The four staged input rings.  Each is a full/empty barrier pair whose ids are
  // BarrierMap's and whose stage, phase and consumer/producer halves are
  // NamedBarrierRing's (named_barrier.hpp).
  //
  // NOTE: the delivered configuration is `TmeStages == 1`.  The map, the smem
  // staging and the init loop all follow `TmeStages`, and a two-stage build has
  // been run, but only the one-stage build has a correctness result behind it.
  using DecayedRing  = NamedBarrierRing<TmeStages, BarrierMap::DecayedReady, BarrierMap::DecayedFree>;
  using RestoredRing = NamedBarrierRing<TmeStages, BarrierMap::RestoredReady, BarrierMap::RestoredFree>;
  using VRing        = NamedBarrierRing<TmeStages, BarrierMap::VReady, BarrierMap::VFree>;
  using InvPRing     = NamedBarrierRing<TmeStages, BarrierMap::InvPReady, BarrierMap::InvPFree>;

  static MUTLASS_DEVICE constexpr int k_state_row(int physical_row) {
    return (physical_row / kChunk) * (kChunk / 2) + (physical_row % kChunk);
  }

  // The named-barrier map, initialized once by thread 0.  A ring's `full`
  // barrier takes exactly one arrival -- the TME producers arm theirs with
  // arrive_and_expect_tx and the LDGSTS producer arrives after ldgsts_wait -- and
  // its `empty` mate takes one arrival per consuming warp, so a ring read by both
  // consumer squads counts both squads' warps.
  static MUTLASS_DEVICE void init_stage_barriers(int tid) {
    if (tid != 0) {
      return;
    }
    MUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < TmeStages; ++s) {
      named_barrier_init(BarrierMap::DecayedReady + s, 1);
      named_barrier_init(BarrierMap::DecayedFree + s, NumOutputWarps);
      named_barrier_init(BarrierMap::RestoredReady + s, 1);
      named_barrier_init(BarrierMap::RestoredFree + s, NumStateWarps);
      named_barrier_init(BarrierMap::VReady + s, 1);
      named_barrier_init(BarrierMap::VFree + s, NumOutputWarps);
      // inverse/P and the chunk total ride one ring: the OUTPUT squad retires
      // inverse and P, the STATE squad retires the total.
      named_barrier_init(BarrierMap::InvPReady + s, 1);
      named_barrier_init(BarrierMap::InvPFree + s, NumOutputWarps + NumStateWarps);
    }
    // One rendezvous per consumer squad: BarrierMap::Local + 0 is the state squad's, the
    // output squad's is the next slot.
    for (int s = 0; s < NumStateSquads + NumOutputSquads; ++s) {
      named_barrier_init(BarrierMap::Local + s, mutlass::NumWarpsPerWarpSquad);
    }
    // The four producer warps seed the whole state tile between them, so four
    // arrivals publish it; both consumer squads wait on it.
    named_barrier_init(BarrierMap::Seeded, NumProducerWarps);
    // The STATE squad's four warps publish the item's last state once; the four
    // producer warps wait on it before exporting their rows.
    named_barrier_init(BarrierMap::Drained, NumStateWarps);
    // The two per-chunk cross-squad handshakes, each arrived once per warp of the
    // squad that publishes it.
    named_barrier_init(BarrierMap::UReady, NumOutputWarps);
    named_barrier_init(BarrierMap::StateCommitted, NumStateWarps);
    // State-squad-only; the count is the number of participating warps.
    named_barrier_init(kStateSnapshotBarrier, NumStateWarps);
  }

  static MUTLASS_DEVICE void init_barriers(int tid) {
    mutlass::arch::allocate_async_barriers(BarrierMap::NumBarriers);
    init_stage_barriers(tid);
  }

  static MUTLASS_DEVICE void init_shared_storage(SharedStorage&, int) {
    // MP31's ring producers fully initialize their tiles before publication.
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    // V's tile is pinned to the PRE-swap collective's B layout (SmemLayoutV), so
    // its TME descriptor must be built from that same collective: the swapped
    // UK's B operand is kr and would describe a (128, 16) box instead of V's.
    auto params_v = CollectiveMmaUK::to_underlying_arguments(
        make_shape(Int<kHeadDim>{}, Int<kHeadDim>{}, problem_size.T, make_shape(problem_size.H, problem_size.B)),
        typename CollectiveMmaUK::Arguments{
            args.ptr_v,
            args.stride_v,
            args.ptr_v,
            args.stride_v,
        },
        nullptr);
    int32_t records = Components::workspace_records(problem_size);
    int32_t outer   = records * problem_size.H;

    auto workspace_decayed = Workspace::make_decayed_tensor(args.workspace.decayed, outer);
    auto workspace_kr      = Workspace::make_k_restored_tensor(args.workspace.k_restored, outer);
    auto workspace_inverse = Workspace::make_matrix_tensor(args.workspace.inverse, outer);
    auto workspace_p       = Workspace::make_matrix_tensor(args.workspace.p, outer);
    auto workspace_total   = Workspace::make_total_tensor(args.workspace.total, outer);
    // The chunk-state pool is [outer, kHeadDim, kHeadDim] with the k axis
    // contiguous, i.e. one row per (checkpoint entry, head).  The descriptor is
    // built host-side once; the device side only changes its coordinates.  The
    // extent is the caller's pool row count, which for the checkpoint cadence is
    // the compact bound (smaller than the packed chunk-slot space).
    //
    // The pool is `Element`-typed (the activation), not `StateElement`: the
    // source tile is the activation-typed snapshot, and the checkpoint cadence
    // deliberately allows an fp32 recurrent state (its pool stays io-dtype,
    // matching cudnn's `state_checkpoints`).  The legacy `chunk_states` pool
    // cannot reach this with an fp32 state -- the host gate refuses it.
    ChunkStatesPtrBase chunk_states_base{};
    if constexpr (HasChunkStatesOut) {
      int32_t const pool_rows                = args.checkpoint_pool_rows > 0 ? args.checkpoint_pool_rows : records;
      auto          pool                     = make_tensor(make_gmem_ptr(static_cast<Element*>(args.ptr_chunk_states)),
                              make_layout(make_shape(_8{}, _16{}, _128{}, pool_rows * problem_size.H),
                                          make_stride(_128{}, _1024{}, _1{}, _16384{})));
      chunk_states_base.ptr_chunk_states     = args.ptr_chunk_states;
      chunk_states_base.checkpoint_pool_rows = pool_rows;
      chunk_states_base.tme_chunk_states =
          make_tme_copy(MP31_TME_STORE{}, pool, SmemLayoutStateBlocked{}, TilerStateChunkStates{});
    }
    return Params{
        chunk_states_base,
        args.ptr_v,
        args.ptr_initial_state,
        args.ptr_out,
        args.ptr_final_state,
        args.ptr_state_indices,
        args.stride_v,
        args.stride_out,
        make_tme_copy(MP31_TME_LOAD{}, workspace_decayed, SmemLayoutDecayedTile{}),
        make_tme_copy(MP31_TME_LOAD{}, workspace_kr, SmemLayoutKRestoredTile{}),
        static_cast<Element const*>(args.workspace.inverse),
        static_cast<Element const*>(args.workspace.p),
        args.workspace.total,
        // The robust-copy descriptors bound-check every access; span the whole
        // chunk-major field with the header layout rather than a field-major
        // `outer * field_size` count.
        mute::make_robust_desc(args.workspace.inverse, static_cast<size_t>(mute::cosize(workspace_inverse.layout()))),
        mute::make_robust_desc(args.workspace.p, static_cast<size_t>(mute::cosize(workspace_p.layout()))),
        mute::make_robust_desc(args.workspace.total, static_cast<size_t>(mute::cosize(workspace_total.layout()))),
        params_v.tme_load_b,
    };
  }

  // Publish S_{i+1} to the state tile.  With the single (M=64, N=128) atom each
  // lane owns four rows and, per row, exactly the 16 *consecutive* logical columns
  // 16 * (lane % 8) .. + 15 -- positions the K1 permute was designed to line up --
  // so this is one 32 B store per row, no shuffles and no scalar stores.
  template <class TensorStateSmem, class TensorState>
  static MUTLASS_DEVICE void commit_state(TensorStateSmem& sState, TensorState const& rState, int local_tid) {
    static_assert(kValueTile == 64 && kHeadDim == 128);
    static_assert(decltype(size(rState))::value == 64);

    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);
    auto       cState     = make_identity_tensor(make_shape(Int<kValueTile>{}, Int<kHeadDim>{}));
    auto       tCcState   = thr_mma_ku.partition_C(cState);

    using FloatVec8   = mutlass::Array<float, 8>;
    using ElementVec8 = mutlass::Array<Element, 8>;
    auto rStateVec    = recast<FloatVec8 const>(rState);

    int const lane = local_tid % mutlass::NumThreadsPerWarp;
    int const k0   = 16 * (lane % 8);

    MUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 8; ++j) {
      int const row = int(get<0>(tCcState(8 * j)));
      // The address must come from the layout accessor: the 32-column groups are
      // reordered for some rows, so raw pointer arithmetic is wrong.
      *reinterpret_cast<ElementVec8*>(&sState(make_coord(row, k0 + 8 * (j % 2)))) = pack_from<Element, 8>(rStateVec(j));
    }
  }

  // Snapshot the state ENTERING chunk `chunk_idx` (S_i, not yet decayed by this
  // chunk) into the caller's chunk-state pool.
  //
  // The pool row is the exact gmem layout `initial_state` / `final_state` use:
  // [chunk_row, Hv, kHeadDim, kHeadDim] with the k axis contiguous, so a row is
  // by construction usable as an `initial_state`.  The row index is the same
  // packed chunk slot the prepare workspace uses (`Workspace::record`):
  // workspace_base + chunk_idx, i.e. seq_idx*chunks + i dense and
  // ceil(bos/kChunk) + seq_idx + i varlen.  Sequences we do not own are never
  // touched, so packed-slot holes keep the caller's initial values.
  //
  // Source of truth is the smem state tile, which holds S_i at the top of the
  // chunk loop (state_seed wrote S_0, commit_state wrote S_i at the end of the
  // previous iteration).  The tile is 16 KB of bf16, so the whole row goes out as
  // **one TME store issued by a single lane**: the swizzled tile is described by
  // `SmemLayoutStateBlocked`, the descriptor's `a` coordinate picks the
  // value_start half and its `dim3` the pool row.
  //
  // The store must have *read* the tile before any lane overwrites it, but only
  // the issuing lane can wait on it, so both call sites pair the wait with the
  // state-squad rendezvous `kStateSnapshotBarrier` (see kda_recurrence_state).
  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE void store_chunk_states_for_chunk(Params const&      params,
                                                          ProblemSize const& problem_size,
                                                          WorkDesc const&    work_desc,
                                                          SharedStorage&     shared_storage,
                                                          int                chunk_idx,
                                                          int                local_tid) {
    if constexpr (HasChunkStatesOut) {
      // One lane, one store: four warps would write the same 16 KB four times.
      if (local_tid != 0) {
        return;
      }

      // The host selects the row mapping; the kernel only picks between the two
      // compile-time layouts.  `kStride` is a constant, so the cadence modulus
      // below folds away when the cadence equals the chunk (every chunk stores).
      constexpr int kStride = Components::kCheckpointStride;
      int64_t       chunk_row;
      if constexpr (Components::HasCheckpointCadence) {
        // Checkpoint cadence: only the chunk whose state enters token
        // `j * every_n_tokens`, packed per sequence.  Skipping the store here
        // is safe because the caller's post-store rendezvous is unconditional;
        // only the TME issue is conditional.
        if ((chunk_idx % kStride) != 0) {
          return;
        }
        // The work descriptor already holds this sequence's checkpoint row base
        // (see `make_recurrence_work`), so nothing is scanned here.
        chunk_row = int64_t(work_desc.checkpoint_base) + chunk_idx / kStride;
      } else {
        chunk_row = int64_t(work_desc.workspace_base) + chunk_idx;
      }
      int64_t const pool_row = chunk_row * problem_size.H + work_desc.head_idx;
      // Same extent the host built the descriptor with (== Params' pool rows).
      int32_t const pool_rows = params.checkpoint_pool_rows * problem_size.H;

      auto sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutStateBlocked{});
      auto gcoord = params.tme_chunk_states.get_tme_tensor(make_shape(Int<8>{}, Int<16>{}, Int<128>{}, pool_rows));
      // The half index must ride the *coordinate tensor*: slicing it out of the
      // partitioned destination does not reach the instruction (device-verified
      // on S5000: value_start = 64 then silently writes the a = 0 half).
      auto gc = domain_offset(make_coord(_0{}, int32_t(work_desc.value_start / 8), _0{}, int32_t(pool_row)), gcoord);
      auto cta_tme = params.tme_chunk_states.get_slice(_0{});
      auto tS      = group_modes<0, 3>(cta_tme.partition_S(sState));
      auto tG      = group_modes<0, 3>(cta_tme.partition_D(gc));
      copy(params.tme_chunk_states, tS, tG(_, _0{}, _0{}));
      tme_store_arrive();
    }
  }

  // Re-apply the chunk total from the smem staging the inverse/P warp filled with
  // LDGSTS.  K1 writes the total row in headDim order, and the K1 permute makes a
  // lane's sixteen register columns the sixteen consecutive headDim values
  // 16 * (lane % 8) .. + 15, so the lane's factor is sixteen consecutive floats:
  // four vector loads, reused by the four rows it holds.
  template <class TensorState>
  static MUTLASS_DEVICE void scale_state(TensorState& rState, Element const* total_smem, int local_tid) {
    using FloatVec4     = mutlass::Array<float, 4>;
    using ElementVec16  = mutlass::Array<Element, 16>;
    auto      rStateVec = recast<FloatVec4>(rState);
    int const r         = local_tid % mutlass::NumThreadsPerWarp % 8;
    // This lane's sixteen consecutive headDim columns: one Element vector load,
    // widened once and reused by the four rows the lane holds.  The workspace
    // total is an Element row, so the widening happens here.
    auto total_f32 = mutlass::NumericArrayConverter<float, Element, 16>{}(
        *reinterpret_cast<ElementVec16 const*>(total_smem + 16 * r));
    auto const* total_v4 = reinterpret_cast<float4 const*>(&total_f32);

    // The lane's sixteen registers are row-major: group 4 * row + c is row
    // `row` and headDim columns 16 * (lane % 8) + 4 * c .. + 3, and the four rows
    // share the same four column groups.
    MUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      MUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        auto& values = reinterpret_cast<float4&>(rStateVec(4 * row + c));
        values       = simd::vmul(values, total_v4[c]);
      }
    }
  }

  static MUTLASS_DEVICE void inverse_residual(SharedStorage& shared_storage,
                                              int            stage,
                                              int            inverse_stage,
                                              int            local_tid) {
    auto sInverse =
        make_tensor(make_smem_ptr(shared_storage.smem_inverse.data()), SmemLayoutInverseA{})(_, _, inverse_stage);
    auto sU   = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
    auto sUMn = make_tensor(sU.data(), mate::select_layout<1, 0>(sU.layout()));

    TiledMmaPU tiled_mma_pu;
    auto       thr_mma_pu = tiled_mma_pu.get_thread_slice(local_tid);
    auto       acc_u      = partition_fragment_C(tiled_mma_pu, make_shape(Int<kChunk>{}, Int<kValueTile>{}));
    clear(acc_u);
    auto tSsInverse = thr_mma_pu.partition_A(sInverse);
    auto tSsU       = thr_mma_pu.partition_B(sU);
    auto tSrInverse = thr_mma_pu.make_fragment_A(tSsInverse);
    auto tSrU       = thr_mma_pu.make_fragment_B(tSsU);
    gemm(tiled_mma_pu, acc_u, tSrInverse, tSrU, acc_u);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();

    R2STiledCopy tiled_copy_r2s;
    auto         thr_copy_r2s = tiled_copy_r2s.get_thread_slice(local_tid);
    Tensor       tUsU         = thr_copy_r2s.partition_D(as_position_independent_swizzle_tensor(sUMn));
    Tensor       accum_cvt    = make_fragment_like<Element>(acc_u);
    Tensor       tUrU         = thr_copy_r2s.retile_S(accum_cvt);
    Tensor       tCvt_frg     = recast<mutlass::Array<Element, R2SFragmentSize>>(accum_cvt);
    Tensor       tAcc_frg     = recast<mutlass::Array<float, R2SFragmentSize>>(acc_u);

    MUTE_UNROLL
    for (int i = 0; i < size(tCvt_frg); ++i) {
      tCvt_frg(i) = pack_from<Element, R2SFragmentSize>(tAcc_frg(i));
    }
    copy(tiled_copy_r2s, tUrU, tUsU);
    // U is published through BarrierMap::UReady; the state squad's UK gemm waits on it.  The
    // arrive is per warp, so the barrier also tells this squad that all four of
    // its warps have finished writing U.
    arrive_warp(BarrierMap::UReady);
  }

  // The state tile's home is smem and the producer squad owns its gmem traffic.
  // Measured on the known-good kernel: the state gmem's k axis is the k_restored
  // *workspace index* space (not the logical headDim), and its value axis is
  // indexed by INL.  The shared tile, however, is consumed by the DS against the
  // decayed tile, so it carries the LOGICAL headDim in its columns.
  // One state-copy vector: eight natural headDim values (16 B at bf16), which
  // the swizzled state tile keeps contiguous within each 8-column group.
  static constexpr int kStateVec           = 8;
  static constexpr int kStateVecsPerRow    = kHeadDim / kStateVec;
  static constexpr int kStateVecsPerThread = kValueTile * kStateVecsPerRow / NumProducerThreads;
  static_assert(kHeadDim % kStateVec == 0);
  static_assert(kValueTile * kStateVecsPerRow % NumProducerThreads == 0);

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void state_seed(SharedStorage&     shared_storage,
                                        Params const&      params,
                                        ProblemSize const& problem_size,
                                        WorkDesc const&    work_desc,
                                        int                local_tid) {
    auto                sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});
    StateElement const* g      = nullptr;
    if constexpr (HasStateIn) {
      int32_t state_idx    = params.ptr_state_indices[work_desc.seq_idx];
      int64_t state_offset = (int64_t(state_idx) * problem_size.H + work_desc.head_idx) * kHeadDim * kHeadDim +
                             int64_t(work_desc.value_start) * kHeadDim;
      g = static_cast<StateElement const*>(params.ptr_initial_state) + state_offset;
    }
    // The permute is already baked into the operand layout, so the seed is a
    // plain copy in natural headDim order, eight values (16 B) at a time; only
    // the smem swizzle is left to the tile's layout accessor.
    using StateVec   = mutlass::Array<StateElement, kStateVec>;
    using ElementVec = mutlass::Array<Element, kStateVec>;
    for (int i = 0; i < kStateVecsPerThread; ++i) {
      int const vec_idx = local_tid + i * NumProducerThreads;
      int const v       = vec_idx / (kHeadDim / kStateVec);
      int const h0      = (vec_idx % (kHeadDim / kStateVec)) * kStateVec;
      StateVec  packed  = HasStateIn ? *reinterpret_cast<StateVec const*>(g + v * kHeadDim + h0) : StateVec{};
      *reinterpret_cast<ElementVec*>(&sState(make_coord(int(TileRowFromInternal{}(v)), h0))) =
          pack_from<Element, kStateVec>(packed);
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void state_drain(SharedStorage&     shared_storage,
                                         Params const&      params,
                                         ProblemSize const& problem_size,
                                         WorkDesc const&    work_desc,
                                         int                local_tid) {
    if constexpr (HasStateOut) {
      auto    sState       = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});
      int32_t state_idx    = params.ptr_state_indices[work_desc.seq_idx];
      int64_t state_offset = (int64_t(state_idx) * problem_size.H + work_desc.head_idx) * kHeadDim * kHeadDim +
                             int64_t(work_desc.value_start) * kHeadDim;
      auto g           = static_cast<StateElement*>(params.ptr_final_state) + state_offset;
      using StateVec   = mutlass::Array<StateElement, kStateVec>;
      using ElementVec = mutlass::Array<Element, kStateVec>;
      for (int i = 0; i < kStateVecsPerThread; ++i) {
        int const  vec_idx = local_tid + i * NumProducerThreads;
        int const  v       = vec_idx / (kHeadDim / kStateVec);
        int const  h0      = (vec_idx % (kHeadDim / kStateVec)) * kStateVec;
        ElementVec packed =
            *reinterpret_cast<ElementVec const*>(&sState(make_coord(int(TileRowFromInternal{}(v)), h0)));
        *reinterpret_cast<StateVec*>(g + v * kHeadDim + h0) = unpack_to<StateElement, kStateVec>(packed);
      }
    }
  }

  // K1 publishes k_restored with the headDim axis in the order MP31's
  // PermuteN states (see kda_permutes.hpp),
  // so B's N coordinate s carries logical headDim k = 16 * (s % 8) + s / 8.

  template <class TensorState>
  static MUTLASS_DEVICE void issue_state_update(
      TensorState& rState, SharedStorage& shared_storage, int stage, int k_stage, int local_tid, uint32_t phase) {
    auto sKRestored =
        make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{})(_, _, k_stage);
    auto sU = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);

    TiledMmaKU tiled_mma_ku;
    auto       thr_mma_ku = tiled_mma_ku.get_thread_slice(local_tid);

    auto tSsU         = thr_mma_ku.partition_A(sU);          // A = U^T (M = value)
    auto tSsKRestored = thr_mma_ku.partition_B(sKRestored);  // B = kr  (N = headDim)
    auto tSrU         = thr_mma_ku.make_fragment_A(tSsU);
    auto tSrKRestored = thr_mma_ku.make_fragment_B(tSsKRestored);

    // U_i is the output squad's; UK cannot read it until UReady_i.  The phase is
    // the chunk parity, not the (single) stage index.
    wait_warp(BarrierMap::UReady, int(phase));
    gemm(tiled_mma_ku, rState, tSrU, tSrKRestored, rState);
    mate::warpsquad_commit_batch();
  }

  // The four producer warps, one ring each.  A ring is published by its own
  // producer warp and retired by the consumer squad(s) that read it, so the
  // producer's `empty` wait is the fence that stops it overwriting a tile while a
  // squad still reads it.  `sequence` is the role's own chunk counter: it runs
  // across work items, so the barrier phases stay in step with the consumers.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void load(SharedStorage&     shared_storage,
                                  Params const&      params,
                                  ProblemSize const& problem_size,
                                  WorkDesc const&    work_desc,
                                  int&               sequence,
                                  int                local_tid) {
    int const lane          = mutlass::canonical_lane_idx();
    int const producer_warp = local_tid / mutlass::NumThreadsPerWarp;
    if (producer_warp == 0) {
      // decayed (Q): one 2*kChunk x kHeadDim TME tile per chunk, issued by lane 0.
      int32_t records = Components::workspace_records(problem_size);
      int32_t outer   = records * problem_size.H;
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx, ++sequence) {
        int64_t record = Workspace::record(problem_size, work_desc, chunk_idx);
        // Take the stage back from the output squad: it arrives on empty once it
        // has retired the tile.
        DecayedRing::wait_empty(sequence);
        auto g_decayed =
            params.tme_workspace_decayed.get_tme_tensor(make_shape(Int<2 * kChunk>{}, Int<kHeadDim>{}, outer));
        auto s_decayed = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{});
        auto cta_tme   = params.tme_workspace_decayed.get_slice(_0{});
        auto tDgD      = group_modes<0, 3>(cta_tme.partition_S(g_decayed));
        auto tDsD      = group_modes<0, 3>(cta_tme.partition_D(s_decayed));
        if (lane == 0) {
          // Announce the TME transaction -- it has to be on the barrier BEFORE the
          // copy is issued -- then issue it.
          int const full = DecayedRing::full(sequence);
          named_barrier_arrive_and_expect_tx(full, uint32_t(2 * KTileBytes));
          copy(params.tme_workspace_decayed.with(named_barrier_id(full)),
               tDgD(_, int32_t(record)),
               tDsD(_, DecayedRing::stage(sequence)));
        }
      }
    } else if (producer_warp == 1) {
      // k_restored: one kHeadDim x kChunk TME tile per chunk, issued by lane 0.
      int32_t records = Components::workspace_records(problem_size);
      int32_t outer   = records * problem_size.H;
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx, ++sequence) {
        int64_t record = Workspace::record(problem_size, work_desc, chunk_idx);
        RestoredRing::wait_empty(sequence);
        auto g_kr = params.tme_workspace_k_restored.get_tme_tensor(make_shape(Int<kHeadDim>{}, Int<kChunk>{}, outer));
        auto s_kr = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestored{});
        auto cta_tme_kr = params.tme_workspace_k_restored.get_slice(_0{});
        auto tKRgKR     = group_modes<0, 3>(cta_tme_kr.partition_S(g_kr));
        auto tKRsKR     = group_modes<0, 3>(cta_tme_kr.partition_D(s_kr));
        if (lane == 0) {
          int const full = RestoredRing::full(sequence);
          named_barrier_arrive_and_expect_tx(full, uint32_t(KRestoredTileBytes));
          copy(params.tme_workspace_k_restored.with(named_barrier_id(full)),
               tKRgKR(_, int32_t(record)),
               tKRsKR(_, RestoredRing::stage(sequence)));
        }
      }
    } else if (producer_warp == 2) {
      // V: one kValueTile x kChunk TME tile per chunk.  The ring is one deep and
      // the tiling is V's own, so the proven partition_SD is reused; the only
      // thing it asks its `pipeline` for is the barrier id it fills.
      VRingHooks v_hooks;
      LoadV      load_v(params.tme_v, v_hooks, shared_storage.smem_v_raw);
      auto       v_src_dst = load_v.partition_SD(problem_size, TileShapeV{}, work_desc);
      VRingState v_state;
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx, ++sequence) {
        VRing::wait_empty(sequence);
        if (lane == 0) {
          named_barrier_arrive_and_expect_tx(VRing::full(sequence), uint32_t(VTileBytes));
        }
        __syncwarp();
        // `step` takes BOTH the smem stage and the barrier id from the state it is
        // handed, so that state and the ring have to name the same stage.  It is
        // built once per work item and starts at 0, while `sequence` deliberately
        // survives across them, so the two agreed only while the ring was one
        // stage deep.  Seed it from the ring, which is the one owner of the stage.
        v_state.stage = VRing::stage(sequence);
        load_v.template step<false>(v_src_dst, chunk_idx, v_state);
      }
    } else {
      // inverse, P and the chunk total: all three are tiny, so they ride async
      // LDGSTS and one ldgsts_wait retires the copies.  The whole warp takes part;
      // the arrive publishes them to both consumer squads.
      MUTLASS_PRAGMA_NO_UNROLL
      for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx, ++sequence) {
        int64_t record = Workspace::record(problem_size, work_desc, chunk_idx);
        InvPRing::wait_empty(sequence);
        int const     g_stage = InvPRing::stage(sequence);
        int64_t const rec     = int64_t(record);
        // Chunk-major: inverse and P live at the same in-record offset, but the
        // records are strided by Workspace::RecordStride, not by the 16x16 field.
        int64_t const  base      = rec * Workspace::RecordStride;
        auto           s_inverse = make_tensor(make_smem_ptr(shared_storage.smem_inverse.data()), SmemLayoutInverseA{});
        auto           s_p       = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
        Element const* g_inverse = params.ptr_workspace_inverse + base;
        Element const* g_p       = params.ptr_workspace_p + base;
        // The staging tile is a swizzled SQMMA operand, so it is filled at
        // layout coordinates; inverse and P share one pass.  Each lane owns
        // eight contiguous elements (half a row), so one in-flight 16 B LDGSTS
        // per tensor replaces eight dependent 2 B load/store pairs.
        {
          GmemTiledCopyInvP tiled_invp;
          auto              thr_invp = tiled_invp.get_thread_slice(lane);
          auto              g_invp   = make_tensor(make_gmem_ptr(g_inverse),
                                    make_layout(make_shape(Int<kChunk>{}, Int<kChunk>{}), StrideInvPTile{}));
          auto              g_pt =
              make_tensor(make_gmem_ptr(g_p), make_layout(make_shape(Int<kChunk>{}, Int<kChunk>{}), StrideInvPTile{}));
          copy(tiled_invp.with(params.desc_inverse),
               thr_invp.partition_S(g_invp),
               thr_invp.partition_D(s_inverse(_, _, g_stage)));
          copy(tiled_invp.with(params.desc_p), thr_invp.partition_S(g_pt), thr_invp.partition_D(s_p(_, _, g_stage)));
        }
        // The total is one flat 128-wide row per chunk: a plain robust LDGSTS.
        {
          GmemTiledCopyTotal tiled_total;
          auto               thr_total = tiled_total.get_thread_slice(lane);
          auto g_total = make_tensor(make_gmem_ptr(params.ptr_workspace_total + rec * Workspace::RecordStride),
                                     make_layout(make_shape(_1{}, Int<kHeadDim>{}), StrideTotalTile{}));
          auto s_total = make_tensor(make_smem_ptr(shared_storage.smem_total.data() + g_stage * kHeadDim),
                                     make_layout(make_shape(_1{}, Int<kHeadDim>{}), StrideTotalTile{}));
          copy(tiled_total.with(params.desc_total), thr_total.partition_S(g_total), thr_total.partition_D(s_total));
        }
        // The LDGSTS copies are fire-and-forget: wait them out, then hand the
        // stage to the consumers.
        mute::ldgsts_wait();
        __syncwarp();
        InvPRing::publish(sequence);
      }
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void kda_recurrence_producer(SharedStorage&     shared_storage,
                                                     Params const&      params,
                                                     ProblemSize const& problem_size,
                                                     WorkDesc const&    work_desc,
                                                     int&               sequence,
                                                     int                local_tid,
                                                     uint32_t           item_idx) {
    // Keep the ring phase across chunks of this work item.
    // Once-per-item handshakes: the work-item parity is the phase (the state tile
    // is single-stage, so these two barriers are not stage-slotted).
    uint32_t const sd_phase = item_idx & 1u;
    if (work_desc.n_chunks > 0) {
      state_seed(shared_storage, params, problem_size, work_desc, local_tid);
      // state_seed is written by all NumProducerWarps warps, but the first
      // consumer of the state tile (the output squad's chunk-0 DS) is released by
      // the decayed ring alone, so nothing on the consumer side waits here.  The
      // producers therefore have to hold each other: the warp that finishes its
      // slice must not arm the decayed ring before the other three wrote theirs.
      //
      // The arrival has to be gated.  `..._arrive_and_wait` counts per thread, so
      // an unguarded call has all 128 producer threads arrive on a barrier whose
      // count is NumProducerWarps (4): it completes 32 times per work item, and
      // 32 is even, so the net phase looks right while the rendezvous does
      // nothing.  `sync_warp_squad` arrives once per warp and waits every lane.
      sync_warp_squad(BarrierMap::Seeded);
      load(shared_storage, params, problem_size, work_desc, sequence, local_tid);
      // The state squad owns the state export; wait for its drain before writing
      // this item's final rows back to gmem.
      wait_warp(BarrierMap::Drained, int(sd_phase));
      state_drain(shared_storage, params, problem_size, work_desc, local_tid);
    }
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void kda_recurrence_state(SharedStorage&     shared_storage,
                                                  Params const&      params,
                                                  ProblemSize const& problem_size,
                                                  WorkDesc const&    work_desc,
                                                  int&               sequence,
                                                  int                local_tid,
                                                  uint32_t           item_idx) {
    auto       sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});
    TiledMmaKU tiled_mma_ku;
    auto       rState = partition_fragment_C(tiled_mma_ku, take<0, 2>(TileShapeKU{}));

    uint32_t const sd_phase = item_idx & 1u;

    // ---- S_0 comes from the producer's tile (it owns the state gmem traffic) --
    wait_warp(BarrierMap::Seeded, int(sd_phase));
    {
      auto thr_mma_ku     = tiled_mma_ku.get_thread_slice(local_tid);
      auto cState         = make_identity_tensor(make_shape(Int<kValueTile>{}, Int<kHeadDim>{}));
      auto tCcState       = thr_mma_ku.partition_C(cState);
      using ElementVec8   = mutlass::Array<Element, 8>;
      using FloatVec8     = mutlass::Array<float, 8>;
      auto      rStateVec = recast<FloatVec8>(rState);
      int const k0        = 16 * (local_tid % mutlass::NumThreadsPerWarp % 8);
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 8; ++j) {
        int const         row    = int(get<0>(tCcState(8 * j)));
        ElementVec8 const packed = *reinterpret_cast<ElementVec8 const*>(&sState(make_coord(row, k0 + 8 * (j % 2))));
        rStateVec(j)             = unpack_to<float, 8>(packed);
      }
    }

    MUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx) {
      // The smem tile holds exactly S_i (the state entering this chunk) here:
      // state_seed wrote S_0, the previous iteration's commit_state wrote S_i.
      // Snapshot it before this iteration's overwrite.  Compile-time gated, so
      // the default (feature-off) instantiation has no trace of it.
      if constexpr (HasChunkStatesOut) {
        store_chunk_states_for_chunk(params, problem_size, work_desc, shared_storage, chunk_idx, local_tid);
      }

      // The chunk's decay total arrives on the inverse/P ring (LDGSTS).
      InvPRing::wait(sequence);
      scale_state(rState, shared_storage.smem_total.data() + InvPRing::stage(sequence) * kHeadDim, local_tid);
      // The total is this squad's half of the ring; the output squad retires
      // inverse and P, so `empty` only flips once both halves have arrived.
      InvPRing::release(sequence);

      RestoredRing::wait(sequence);
      int const k_stage = RestoredRing::stage(sequence);

      // BarrierMap::UReady is a per-chunk handshake (the output squad arrives once per chunk),
      // so its phase is the chunk parity, not the ring phase.
      issue_state_update(rState, shared_storage, k_stage, k_stage, local_tid, uint32_t(sequence & 1));
      mate::warpsquad_wait<0>();

      RestoredRing::release(sequence);

      // This overwrites the tile the snapshot TME store is still reading, so the
      // store has to be retired first.  Only its issuing lane can wait on it;
      // the rendezvous is what tells the other three warps the tile is free.
      // The store was issued before a full SQMMA, so its read is normally already
      // done by now and the wait is nearly free.
      if constexpr (HasChunkStatesOut) {
        if (local_tid == 0) {
          tme_store_wait();
        }
        __syncwarp();
        named_barrier_arrive_and_wait(kStateSnapshotBarrier);
      }
      commit_state(sState, rState, local_tid);
      // Publish S_{i+1} as soon as it is in the tile: the output squad reads it
      // for the next chunk's DS.  The squad rendezvous first, so the publish means
      // "all four warps have written", not "this warp has".
      sync_warp_squad(BarrierMap::Local + 0);
      arrive_warp(BarrierMap::StateCommitted);
      ++sequence;
    }

    if constexpr (HasChunkStatesOut) {
      // The producer drains the tile as soon as StateDrained arrives and the
      // caller may reuse the pool right after the kernel, so the last snapshot
      // must be retired before we hand the tile over.
      if (local_tid == 0) {
        tme_store_wait();
      }
      __syncwarp();
    }
    arrive_warp(BarrierMap::Drained);
  }

  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void kda_recurrence_output(SharedStorage&     shared_storage,
                                                   Params const&      params,
                                                   ProblemSize const& problem_size,
                                                   WorkDesc const&    work_desc,
                                                   int&               sequence,
                                                   int                local_tid) {
    MUTLASS_PRAGMA_NO_UNROLL
    for (int chunk_idx = 0; chunk_idx < work_desc.n_chunks; ++chunk_idx, ++sequence) {
      // The decayed tile for this chunk, published by producer warp 0.
      DecayedRing::wait(sequence);
      int const stage = DecayedRing::stage(sequence);

      // The DS accumulator is [K rows | Q rows] per lane.  The Q half is
      // q_decayed @ S, which is exactly this chunk's output accumulator: both
      // MMAs give a lane the same row and the same value columns, so P@U
      // accumulates straight into those registers later on.
      TiledMmaDS tiled_mma_ds;
      auto       acc_ds = partition_fragment_C(tiled_mma_ds, take<0, 2>(TileShapeDS{}));

      {
        // The state tile holds S_i, which the state squad published at the end of
        // the previous chunk -- one phase earlier.  The very first chunk of the
        // CTA reads the item's seed and must not wait.
        //
        // The wait is keyed on the GLOBAL chunk index, not on `chunk_idx`: the U
        // tile is single-buffered, and this is also what stops the first chunk of
        // a later work item from overwriting the U a still-running state squad is
        // reading at the tail of the previous item.  With two stages the stage
        // index covered that for free; with one it has to be a handshake.
        if (sequence > 0) {
          wait_warp(BarrierMap::StateCommitted, (sequence - 1) & 1);
        }

        auto sDecayed =
            make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, stage);
        auto sState = make_tensor(make_smem_ptr(shared_storage.smem_state.data()), SmemLayoutState{});

        // DS is an output-squad MMA.  Its K rows are consumed immediately for
        // the residual; its Q rows are scattered into the small QS-layout bridge
        // because DS and QS intentionally have different warp ownership.
        // For the singleton DS atom, physical_row = 4*warp + (lane>>3) +
        // 16*(slot>>3); hence the first two warps are K and the last two Q.
        auto thr_mma_ds = tiled_mma_ds.get_thread_slice(local_tid);
        auto tSsDecayed = thr_mma_ds.partition_A(sDecayed);
        auto tSsState   = thr_mma_ds.partition_B(sState);
        auto tSrDecayed = thr_mma_ds.make_fragment_A(tSsDecayed);
        auto tSrState   = thr_mma_ds.make_fragment_B(tSsState);
        clear(acc_ds);
        gemm(tiled_mma_ds, acc_ds, tSrDecayed, tSrState, acc_ds);
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait<0>();

        // Decayed is dead as soon as the DS SQMMA completes.  Hand its ring back
        // before the residual/QState scatter so producer warp 0 can prefetch the
        // next chunk while output still consumes V and the register fragment.
        DecayedRing::release(sequence);

        // V is not consumed by Decayed@State.  Defer its wait until the DS
        // fragment is ready so producer warp 2 can overlap the V TME load with
        // both DS SQMMA instructions.
        VRing::wait(sequence);
        auto sU      = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
        int  v_stage = VRing::stage(sequence);
        auto sV      = make_tensor(make_smem_ptr(shared_storage.smem_v_raw.data()), SmemLayoutV{})(_, _, v_stage);

        auto cDS  = make_identity_tensor(take<0, 2>(TileShapeDS{}));
        auto tDcD = thr_mma_ds.partition_C(cDS);
        // DS C-fragment ownership, measured on MP31: every lane holds eight K
        // rows (elements 0..7, physical row lane / 8) followed by eight Q rows
        // (elements 8..15, physical row 16 + lane / 8).  The split is inside the
        // lane, not between warps, so vector 0 is the residual and vector 1 the
        // Q@State bridge for every lane.
        using AccumResidualVec = mutlass::Array<float, R2SFragmentSize>;
        using ElementVec       = mutlass::Array<Element, R2SFragmentSize>;
        using QStateVec        = mutlass::Array<float, R2SFragmentSize>;
        auto acc_ds_vec        = recast<AccumResidualVec const>(acc_ds);
        static_assert(decltype(size(acc_ds_vec))::value == 2);
        {
          // residual = V - K_decayed @ S, straight into U
          constexpr int    base            = 0;
          int              physical_row    = int(get<0>(tDcD(base)));
          int              col             = int(get<1>(tDcD(base)));
          int              state_row       = k_state_row(physical_row);
          int              tile_row        = int(TileRowFromInternal{}(col));
          ElementVec       v               = *reinterpret_cast<ElementVec const*>(&sV(tile_row, state_row));
          AccumResidualVec residual        = unpack_to<float, R2SFragmentSize>(v);
          auto*            residual_float4 = reinterpret_cast<float4*>(&residual);
          auto*            acc_ds_float4   = reinterpret_cast<float4 const*>(&acc_ds_vec(0));
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < R2SFragmentSize / 4; ++i) {
            residual_float4[i] = simd::vsub(residual_float4[i], acc_ds_float4[i]);
          }
          *reinterpret_cast<ElementVec*>(&sU(tile_row, state_row)) = pack_from<Element, R2SFragmentSize>(residual);
        }
        // (the Q half stays in acc_ds and becomes the P@U accumulator below)
        VRing::release(sequence);
        // The residual tile is written by all four warps and read back by all of
        // them, so this is a squad-internal rendezvous before inverse_residual.
        sync_warp_squad(BarrierMap::Local + 1);
      }

      {
        InvPRing::wait(sequence);
        int const g_stage = InvPRing::stage(sequence);
        auto      sP      = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{})(_, _, g_stage);
        auto      sU      = make_tensor(make_smem_ptr(shared_storage.smem_u.data()), SmemLayoutU{})(_, _, stage);
        // Inverse(residual) remains in the output squad.  It overwrites the
        // residual in smem_u with U and signals the state squads to consume it.
        inverse_residual(shared_storage, stage, g_stage, local_tid);
        // BarrierMap::UReady is arrived once per CHUNK, not once per ring stage, so its phase
        // is the chunk parity even when a ring is deeper than one.
        wait_warp(BarrierMap::UReady, sequence & 1);

        TiledMmaPU tiled_mma_pu;
        TiledMmaQS tiled_mma_qs;
        auto       thr_mma_pu = tiled_mma_pu.get_thread_slice(local_tid);
        auto       thr_mma_qs = tiled_mma_qs.get_thread_slice(local_tid);
        auto       tSsP       = thr_mma_pu.partition_A(sP);
        auto       tSsU       = thr_mma_pu.partition_B(sU);
        auto       tSrP       = thr_mma_pu.make_fragment_A(tSsP);
        auto       tSrU       = thr_mma_pu.make_fragment_B(tSsU);
        // No clear and no copy: the DS Q half already holds q_decayed @ S, and
        // P@U accumulates on top of it (q@S first, as the reference does).
        // Handing the DS Q half to the P@U gemm as its accumulator is only valid
        // while both per-thread layouts are contiguous flat register runs of the
        // same shape -- checked here so a layout change is caught by the compiler
        // instead of by a wrong result.
        using DsCFrag = decltype(partition_fragment_C(TiledMmaDS{}, take<0, 2>(TileShapeDS{})));
        using QsCFrag = decltype(partition_fragment_C(TiledMmaQS{}, take<0, 2>(TileShapeQS{})));
        static_assert(decltype(size(DsCFrag{}))::value == 2 * R2SFragmentSize, "DS C fragment size");
        static_assert(decltype(size(QsCFrag{}))::value == R2SFragmentSize, "QS C fragment size");
        // Both layouts use the same (2,4) inner register pattern with strides
        // (1,2); the DS accumulator adds the third mode = the K/Q half with
        // stride 8, so DS registers 8..15 are exactly a QS accumulator and the
        // P@U gemm can accumulate into them verbatim.
        static_assert(
            std::is_same_v<
                decltype(DsCFrag{}.layout()),
                Layout<mute::tuple<mute::tuple<_2, _4, _2>, _1, _1>, mute::tuple<mute::tuple<_1, _2, _8>, _0, _0>>>,
            "DS C fragment layout changed: its Q half no longer matches the QS accumulator");
        static_assert(
            std::is_same_v<
                decltype(QsCFrag{}.layout()),
                Layout<mute::tuple<mute::tuple<_2, _4, _1>, _1, _1>, mute::tuple<mute::tuple<_1, _2, _0>, _0, _0>>>,
            "QS/P@U C fragment layout changed: the DS Q half no longer matches it");
        auto acc_ds_vec = recast<mutlass::Array<float, R2SFragmentSize>>(acc_ds);
        auto acc_q      = make_tensor(reinterpret_cast<float*>(acc_ds_vec.data() + 1),
                                 partition_fragment_C(tiled_mma_qs, take<0, 2>(TileShapeQS{})).layout());
        gemm(tiled_mma_pu, acc_q, tSrP, tSrU, acc_q);
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait<0>();
        // This squad's half of the inverse/P/total ring: P (and inverse) is dead
        // once P@U has been issued; the state squad separately retires the total.
        InvPRing::release(sequence);
        auto cO   = make_identity_tensor(take<0, 2>(TileShapeQS{}));
        auto tOcO = thr_mma_qs.partition_C(cO);
        static_assert(decltype(size(acc_q))::value == R2SFragmentSize);

        int     actual_len = work_desc.actual_len(chunk_idx);
        int64_t token_base = work_desc.tme_token(chunk_idx);
        auto    out_outer  = get<2>(params.stride_out);
        int     row        = int(get<0>(tOcO(0)));
        if (row < actual_len) {
          using AccumOutputVec         = mutlass::Array<float, R2SFragmentSize>;
          using ElementOutVec          = mutlass::Array<Element, R2SFragmentSize>;
          auto          acc_output_vec = recast<AccumOutputVec const>(acc_q);
          ElementOutVec packed         = pack_from<Element, R2SFragmentSize>(acc_output_vec(0));
          int           internal_col   = int(get<1>(tOcO(0)));
          int           natural_col    = int(TileRowFromInternal{}(internal_col)) + work_desc.value_start;
          int64_t       offset = int64_t(natural_col) + (token_base + row) * int64_t(get<1>(params.stride_out)) +
                           int64_t(work_desc.head_idx) * int64_t(get<0>(out_outer)) +
                           int64_t(work_desc.tme_batch()) * int64_t(get<1>(out_outer));
          *reinterpret_cast<ElementOutVec*>(params.ptr_out + offset) = packed;
        }
      }
    }
  }

  struct ProducerRole {
    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      kda_recurrence_producer(shared, params, problem, work, sequence, int(threadIdx.x), 0);
    }
  };
  struct OutputConsumer {
    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      kda_recurrence_output(shared, params, problem, work, sequence, int(threadIdx.x) - NumProducerThreads);
    }
  };
  struct StateConsumer {
    template <class Problem, class WorkDesc>
    MUTLASS_DEVICE void operator()(SharedStorage&  shared,
                                   Params const&   params,
                                   Problem const&  problem,
                                   WorkDesc const& work,
                                   int&            sequence) const {
      kda_recurrence_state(
          shared, params, problem, work, sequence, int(threadIdx.x) - NumProducerThreads - NumOutputThreads, 0);
    }
  };
  using ProducerWarp0 = ProducerRole;
  using ProducerWarp1 = ProducerRole;
  using ProducerWarp2 = ProducerRole;
  using ProducerWarp3 = ProducerRole;
};

}  // namespace recurrence

}  // namespace mate::flat::kda
