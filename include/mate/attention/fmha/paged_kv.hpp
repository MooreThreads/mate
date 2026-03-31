#pragma once

#include <mute/tensor.hpp>

namespace mate::attention::fmha {

using namespace mute;

template <bool IsPagedKV,
          class Element,
          int  NumThreads,
          int  TileN,
          int  HeadDimQK,
          int  HeadDimVO,
          bool IsKVSameIter = false>
struct PagedKVManager {
  using ShapePageTable  = Shape<int32_t, int32_t>;
  using StridePageTable = Stride<int64_t, _1>;

  using TensorPageTable = decltype(make_tensor(
      make_gmem_ptr(static_cast<int32_t const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int32_t(0), _));

  using ShapeKV  = Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideKV = Stride<int64_t, _1, int64_t, int64_t>;

  using TensorKV =
      decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, 0, _));

  // For Lsu Paged Load
  static constexpr int GmemThreadsPerRow = 8;
  static constexpr int ElementsPerLoad   = 128 / sizeof_bits_v<Element>;
  using FragmentType                     = mute::uint_bit_t<128>;
  using GmemCopyAtom                     = Copy_Atom<MP31_ROBUST_LDGSTS<FragmentType>, Element>;
  using GmemTiledCopy                    = decltype(make_tiled_copy(
      GmemCopyAtom{},
      make_ordered_layout(make_shape(Int<NumThreads / GmemThreadsPerRow>{}, Int<GmemThreadsPerRow>{}), Step<_1, _0>{}),
      make_layout(make_shape(_1{}, Int<ElementsPerLoad>{}))));
  using ThrGmemTiledCopy                 = decltype(GmemTiledCopy{}.get_thread_slice(0));

  using TensortKcK = decltype(GmemTiledCopy{}.get_thread_slice(0).partition_S(
      make_identity_tensor(Shape<Int<TileN>, Int<HeadDimQK>>{})));
  using TensortVcV = decltype(GmemTiledCopy{}.get_thread_slice(0).partition_S(
      make_identity_tensor(Shape<Int<TileN>, Int<HeadDimVO>>{})));

  static_assert(MUTE_STATIC_V(size<1>(TensortKcK{})) == MUTE_STATIC_V(size<1>(TensortVcV{})));

  static constexpr int PageEntryPerThread = ceil_div(size<1>(TensortKcK{}), GmemThreadsPerRow);
  using TensorKVPtr                       = decltype(make_tensor<Element*>(Shape<Int<PageEntryPerThread>>{}));

  using TensorPageOffset = decltype(make_tensor<mute::tuple<int32_t, int32_t>>(Shape<Int<PageEntryPerThread>>{}));

  // Permutation traits
  static constexpr int MmaAtomN = 8;
  static constexpr int Fragment = 128 / sizeof_bits_v<Element>;  // Sts vector width
  static constexpr int Repeats  = TileN / (MmaAtomN * Fragment);
  static_assert(TileN % 64 == 0);
  using PermuteTile =
      decltype(filter(make_ordered_layout(Shape<Int<MmaAtomN>, Int<Fragment>, Int<Repeats>>{}, Step<_2, _1, _3>{})));

  int32_t const* const ptr_page_table;
  TensorPageTable      mPageTable;

  int                    bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;
  int const              thread_idx;
  TensorKV               mK, mV;
  RobustDescriptor       desc_K, desc_V, desc_page_table;
  TensorPageOffset       tPrPageOffsetK, tPrPageOffsetV;
  ThrGmemTiledCopy const gmem_thr_copy_kv;
  TensorKVPtr            tPrVPtr;

  mutlass::FastDivmod const& page_size_divmod;

  MUTLASS_DEVICE
  PagedKVManager(int const* const           ptr_page_table,
                 ShapePageTable const&      shape_page_table,
                 StridePageTable const&     stride_page_table,
                 RobustDescriptor const&    desc_page_table,
                 Element*                   ptr_K,
                 ShapeKV const&             shape_K,
                 StrideKV const&            stride_K,
                 RobustDescriptor const&    desc_K,
                 Element*                   ptr_V,
                 int const&                 headdim_v,
                 StrideKV const&            stride_V,
                 RobustDescriptor const&    desc_V,
                 mutlass::FastDivmod const& page_size_divmod,
                 int const                  thread_idx,
                 int const                  bidb_kv,
                 int const                  bidh_kv,
                 int                        bidb_kv_idx)
      : ptr_page_table(ptr_page_table),
        desc_page_table(desc_page_table),
        bidb_kv_idx(bidb_kv_idx),
        bidb_kv_idx_prev(bidb_kv_idx),
        desc_K(desc_K),
        desc_V(desc_V),
        gmem_thr_copy_kv(GmemTiledCopy{}.get_thread_slice(thread_idx)),
        thread_idx(thread_idx),
        page_size_divmod(page_size_divmod) {
    mPageTable   = make_tensor(make_gmem_ptr(ptr_page_table), shape_page_table, stride_page_table)(bidb_kv, _);
    mK           = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh_kv, _);
    auto shape_V = make_shape(get<0>(shape_K), headdim_v, get<2>(shape_K), get<3>(shape_K));
    mV           = make_tensor(make_gmem_ptr(ptr_V), shape_V, stride_V)(_, _, bidh_kv, _);
  }

  template <bool FirstIter = false>
  MUTLASS_DEVICE void load_page_table_for_tme(int const n_block) {
    if constexpr (IsPagedKV) {
      bidb_kv_idx = mPageTable(n_block);
    } else {
      n_block_idx = n_block;
    }
    if constexpr (FirstIter && !IsKVSameIter) {
      bidb_kv_idx_prev = bidb_kv_idx;
      n_block_idx_prev = n_block_idx;
    }
  }

  template <bool FirstIter = false>
  MUTLASS_DEVICE void load_page_table_for_lsu(int const n_block) {
    if constexpr (IsPagedKV) {
      MUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < PageEntryPerThread; ++i) {
        int32_t const row = i * NumThreads + (NumThreads / GmemThreadsPerRow) * (thread_idx % GmemThreadsPerRow) +
                            (thread_idx / GmemThreadsPerRow);
        int32_t const permute_row = PermuteTile{}(row);

        int32_t const row_idx         = n_block * TileN + row;
        int32_t const permute_row_idx = n_block * TileN + permute_row;

        // printf("tid:%d row:%d permute_row:%d\n", thread_idx, row, permute_row);

        // K
        {
          int32_t page_idx, page_offset;
          page_idx = page_size_divmod.divmod(page_offset, permute_row_idx);
          int32_t page;
          mute::MP31_ROBUST_LOAD<int32_t>::copy(mPageTable(page_idx), page, true, desc_page_table);
          tPrPageOffsetK(i) = {page, page_offset};
        }

        // V
        {
          int32_t page_idx, page_offset;
          page_idx = page_size_divmod.divmod(page_offset, row_idx);
          int32_t page;
          mute::MP31_ROBUST_LOAD<int32_t>::copy(mPageTable(page_idx), page, true, desc_page_table);
          tPrPageOffsetV(i) = {page, page_offset};
        }
      }
      if constexpr (FirstIter && !IsKVSameIter) {
        compute_V_ptr();
      }
    }
  }

  MUTLASS_DEVICE
  auto compute_K_ptr() {
    TensorKVPtr tPrKPtr;

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PageEntryPerThread; ++i) {
      auto [page_idx, page_offset] = tPrPageOffsetK(i);
      tPrKPtr(i)                   = &mK(page_offset, _0{}, page_idx);
    }
    return tPrKPtr;
  }

  MUTLASS_DEVICE
  auto compute_V_ptr() {
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < PageEntryPerThread; ++i) {
      auto [page_idx, page_offset] = tPrPageOffsetV(i);
      tPrVPtr(i)                   = &mV(page_offset, _0{}, page_idx);
    }
  }

  template <class TensorK>
  MUTLASS_DEVICE void load_K(int const n_block, TensorK&& sK) {
    Tensor cK   = make_identity_tensor(Shape<Int<TileN>, Int<HeadDimQK>>{});
    Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
    Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);  // cpy, cpy_seq, cpy_hd

    Tensor tPrKPtr = compute_K_ptr();

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tKsK); ++i) {
      // TODO: check use mutlass address space
      Element const* k_ptr = (Element const*)__musa_ptr_gen_to_global(
          (void*)(__shfl_sync(0xffffffff,
                              reinterpret_cast<uint64_t>(tPrKPtr(i / GmemThreadsPerRow)),
                              i % GmemThreadsPerRow,
                              GmemThreadsPerRow)));
      Tensor mK_paged_cur      = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<HeadDimQK>>{});
      Tensor mK_paged_cur_copy = mute::tiled_divide(mK_paged_cur, Shape<Int<ElementsPerLoad>>{});

      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tKsK); ++j) {
        int const j_idx = get<1>(tKcK(_0{}, _0{}, j)) / ElementsPerLoad;
        mute::copy(GmemTiledCopy{}.with(desc_K).with(true), mK_paged_cur_copy(_, j_idx), tKsK(_, i, j));
      }
    }
  }

  template <class TensorV>
  MUTLASS_DEVICE void load_V(int const n_block, TensorV&& sV) {
    Tensor cV   = make_identity_tensor(Shape<Int<TileN>, Int<HeadDimVO>>{});
    Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
    Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);  // cpy, cpy_seq, cpy_hd

    if constexpr (IsKVSameIter) {
      compute_V_ptr();
    }

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tVsV); ++i) {
      // TODO: check use mutlass address space
      Element const* v_ptr = (Element const*)__musa_ptr_gen_to_global(
          (void*)(__shfl_sync(0xffffffff,
                              reinterpret_cast<uint64_t>(tPrVPtr(i / GmemThreadsPerRow)),
                              i % GmemThreadsPerRow,
                              GmemThreadsPerRow)));
      Tensor mV_paged_cur      = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<HeadDimVO>>{});
      Tensor mV_paged_cur_copy = mute::tiled_divide(mV_paged_cur, Shape<Int<ElementsPerLoad>>{});

      // TODO: clear oob if needed
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tVsV); ++j) {
        int const j_idx = get<1>(tVcV(_0{}, _0{}, j)) / ElementsPerLoad;
        mute::copy(GmemTiledCopy{}.with(desc_V).with(true), mV_paged_cur_copy(_, j_idx), tVsV(_, i, j));
      }
    }
    if constexpr (!IsKVSameIter) {
      compute_V_ptr();
    }
  }

  MUTLASS_DEVICE
  mute::tuple<int, int> get_indices_for_tme_k() {
    return {n_block_idx, bidb_kv_idx};
  }

  MUTLASS_DEVICE
  mute::tuple<int, int> get_indices_for_tme_v() {
    if constexpr (IsKVSameIter) {
      return {n_block_idx, bidb_kv_idx};
    } else {
      mute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};

      bidb_kv_idx_prev = bidb_kv_idx;
      n_block_idx_prev = n_block_idx;
      return indices;
    }
  }
};

}  // namespace mate::attention::fmha
