#pragma once

#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/tensor.hpp>

namespace mate::flat::kda {

// The prepare kernel (K1) writes this record layout and the recurrence kernel
// (K2) reads it.  Keep the layout in one place: changing one tile size or
// pointer order must not silently change the K1/K2 ABI.
//
// Single-allocation, chunk-major record layout
// -------------------------------------------
// `mate/kda.py` allocates ONE buffer of shape (records, H, RecordStride) and
// hands the kernels five views into it, so all five fields of a record are
// adjacent, in this order (Element is a 2-byte type -- bf16/fp16):
//
//   field          elements   bytes   byte offset in the record block
//   decayed            4096    8192        0
//   k_restored         2048    4096     8192
//   p                   256     512    12288
//   inverse             256     512    12800
//   total               128     256    13312
//   ---------------------------------------------------------------
//   RecordStride       6784   13568
//
// Every field offset and the record stride are multiples of 256 B, so every
// field of every record is 256 B aligned -- the alignment the TME and robust
// descriptors need.  All five fields are Element rows, so the record stride is
// simply the sum of the field sizes: 4096 + 2048 + 256 + 256 + 128.
template <class Element_, int Chunk_ = 16, int HeadDim_ = 128>
struct ChunkKdaWorkspace {
  using Element = Element_;

  static constexpr int kChunk   = Chunk_;
  static constexpr int kHeadDim = HeadDim_;

  // Per-field element counts, one record each.  These stay the *field* sizes:
  // K1 also uses them as loop/vector counts (e.g. the MP32 prepare drain), so
  // they must never be reused as a record stride.
  static constexpr int64_t DecayedElements   = int64_t(2 * kChunk) * kHeadDim;
  static constexpr int64_t KRestoredElements = int64_t(kChunk) * kHeadDim;
  static constexpr int64_t InverseElements   = int64_t(kChunk) * kChunk;
  static constexpr int64_t PElements         = int64_t(kChunk) * kChunk;
  // `total` is one Element row per record.
  static constexpr int64_t ScaleElements = kHeadDim;

  // Field offsets inside one record block, in Element units.  `p` precedes
  // `inverse` so the MP32 recurrence can read both as ONE 32-row tile: rows
  // 0..15 are P (the C rows owned by warps 0/1, which also own the decayed Q
  // rows, i.e. the output path) and rows 16..31 the inverse (warps 2/3, the K
  // rows, i.e. the U path).  Both fields are kChunk x kChunk, so the order swap
  // leaves every offset and the record stride unchanged.
  static constexpr int64_t DecayedOffset   = 0;
  static constexpr int64_t KRestoredOffset = DecayedOffset + DecayedElements;
  static constexpr int64_t POffset         = KRestoredOffset + KRestoredElements;
  static constexpr int64_t InverseOffset   = POffset + PElements;
  static constexpr int64_t TotalOffset     = InverseOffset + InverseElements;  // byte 13312

  // Chunk-major record stride: the five field sizes in Element units, summed.
  // `total` used to be an fp32 row -- ScaleElements fp32 elements, i.e. two
  // Element slots each -- which made the stride 128 Element slots longer than
  // the field sizes add up to.  It is an Element row now, so the stride is just
  // the sum and the fp32 view of these numbers is gone:
  //   RecordStride 6784 Element units (13568 B)
  static constexpr int64_t TotalOffsetBytes  = TotalOffset * int64_t(sizeof(Element));
  static constexpr int64_t RecordStrideBytes = TotalOffsetBytes + ScaleElements * int64_t(sizeof(Element));
  static constexpr int64_t RecordStride      = RecordStrideBytes / int64_t(sizeof(Element));

  // mute::Int<> takes an int: assert the narrowing is lossless instead of
  // silently truncating the ABI.
  static constexpr int RecordStrideInt = static_cast<int>(RecordStride);
  static_assert(RecordStrideBytes % int64_t(sizeof(Element)) == 0,
                "record stride must be a whole number of Element slots");
  static_assert(RecordStride == int64_t(RecordStrideInt), "RecordStride does not fit in int");

  // Every field must land on a 256 B boundary inside every record.  Record 0
  // starts at a 256 B aligned allocation base, so it is enough that every
  // field offset and the record stride are multiples of 256 B.
  static_assert(DecayedOffset * int64_t(sizeof(Element)) % 256 == 0, "decayed is not 256 B aligned");
  static_assert(KRestoredOffset * int64_t(sizeof(Element)) % 256 == 0, "k_restored is not 256 B aligned");
  static_assert(InverseOffset * int64_t(sizeof(Element)) % 256 == 0, "inverse is not 256 B aligned");
  static_assert(POffset * int64_t(sizeof(Element)) % 256 == 0, "p is not 256 B aligned");
  static_assert(TotalOffsetBytes % 256 == 0, "total is not 256 B aligned");
  static_assert(RecordStrideBytes % 256 == 0, "the record block is not 256 B aligned");

  using DecayedLayout =
      decltype(mute::make_layout(mute::make_shape(mute::Int<2 * kChunk>{}, mute::Int<kHeadDim>{}, int32_t{}),
                                 mute::make_stride(mute::Int<kHeadDim>{}, mute::_1{}, mute::Int<RecordStrideInt>{})));
  using KRestoredLayout =
      decltype(mute::make_layout(mute::make_shape(mute::Int<kHeadDim>{}, mute::Int<kChunk>{}, int32_t{}),
                                 mute::make_stride(mute::_1{}, mute::Int<kHeadDim>{}, mute::Int<RecordStrideInt>{})));
  using MatrixLayout =
      decltype(mute::make_layout(mute::make_shape(mute::Int<kChunk>{}, mute::Int<kChunk>{}, int32_t{}),
                                 mute::make_stride(mute::Int<kChunk>{}, mute::_1{}, mute::Int<RecordStrideInt>{})));
  // `total` is one kHeadDim Element row per record, strided like every other
  // field.  Before the chunk-major layout the collectives built this view ad hoc
  // as `record * kHeadDim`; that arithmetic now comes from here.
  using TotalLayout = decltype(mute::make_layout(mute::make_shape(mute::Int<kHeadDim>{}, int32_t{}),
                                                 mute::make_stride(mute::_1{}, mute::Int<RecordStrideInt>{})));

  struct MutablePointers {
    Element* decayed;
    Element* k_restored;
    Element* inverse;
    Element* p;
    Element* total;
  };

  struct ConstPointers {
    Element const* decayed;
    Element const* k_restored;
    Element const* inverse;
    Element const* p;
    Element const* total;
  };

  template <bool IsVarlen, class ProblemShape>
  MUTLASS_HOST_DEVICE static int record_count(ProblemShape const& problem_shape) {
    if constexpr (IsVarlen) {
      return problem_shape.template max_chunks<kChunk>() + problem_shape.N;
    } else {
      return problem_shape.N * problem_shape.workspace_chunks;
    }
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_decayed_tensor(Pointer pointer, int records) {
    return mute::make_tensor(
        mute::make_gmem_ptr(pointer),
        mute::make_layout(mute::make_shape(mute::Int<2 * kChunk>{}, mute::Int<kHeadDim>{}, records),
                          mute::make_stride(mute::Int<kHeadDim>{}, mute::_1{}, mute::Int<RecordStrideInt>{})));
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_k_restored_tensor(Pointer pointer, int records) {
    return mute::make_tensor(
        mute::make_gmem_ptr(pointer),
        mute::make_layout(mute::make_shape(mute::Int<kHeadDim>{}, mute::Int<kChunk>{}, records),
                          mute::make_stride(mute::_1{}, mute::Int<kHeadDim>{}, mute::Int<RecordStrideInt>{})));
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_matrix_tensor(Pointer pointer, int records) {
    return mute::make_tensor(
        mute::make_gmem_ptr(pointer),
        mute::make_layout(mute::make_shape(mute::Int<kChunk>{}, mute::Int<kChunk>{}, records),
                          mute::make_stride(mute::Int<kChunk>{}, mute::_1{}, mute::Int<RecordStrideInt>{})));
  }

  template <class Pointer>
  MUTLASS_HOST_DEVICE static auto make_total_tensor(Pointer pointer, int records) {
    return mute::make_tensor(mute::make_gmem_ptr(pointer),
                             mute::make_layout(mute::make_shape(mute::Int<kHeadDim>{}, records),
                                               mute::make_stride(mute::_1{}, mute::Int<RecordStrideInt>{})));
  }

  // Workspace records are allocated per value head (the second dimension of
  // the Python workspace tensors), so use the value-head index rather than
  // the grouped Q/K head index stored in WorkDesc::qk_head_idx.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_HOST_DEVICE static typename WorkDesc::IndexType record(ProblemSize const& problem_size,
                                                                 WorkDesc const&    work_desc,
                                                                 int                chunk_idx) {
    using IndexType = typename WorkDesc::IndexType;
    return (IndexType(work_desc.workspace_base) + IndexType(chunk_idx)) * IndexType(problem_size.H) +
           IndexType(work_desc.head_idx);
  }
};

}  // namespace mate::flat::kda
