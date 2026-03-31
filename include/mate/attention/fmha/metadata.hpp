#include <mutlass/fast_math.h>

#include "fmha.hpp"

template <typename T, typename Compare>
__device__ __forceinline__ int merge_path_find_i(const T* A, int m, const T* B, int n, int d, Compare comp) {
  int lo = max(0, d - n);
  int hi = min(d, m);
  while (lo < hi) {
    int i = (lo + hi) >> 1;
    int j = d - i;

    // Want: A[i-1] not before B[j]  <=> B[j] before A[i-1]
    bool a_im1_after_bj = false;
    if (i > 0 && j < n) {
      // if comp(B[j], A[i-1]) then A[i-1] should be after B[j]
      a_im1_after_bj = comp(B[j], A[i - 1]);
    }

    // Want: B[j-1] before A[i]  (strict) to avoid crossing.
    // If NOT (B[j-1] before A[i]) then we need larger i.
    bool b_jm1_not_before_ai = false;
    if (j > 0 && i < m) {
      b_jm1_not_before_ai = !comp(B[j - 1], A[i]);
    }

    if (a_im1_after_bj) {
      hi = i;
    } else if (b_jm1_not_before_ai) {
      lo = i + 1;
    } else {
      return i;
    }
  }
  return lo;
}

// Block cooperative merge of two runs in shared memory.
// Each thread writes a contiguous output segment.
template <int BLOCK_THREADS, typename T, typename Compare>
__device__ __forceinline__ void block_merge_two_runs(const T* in,
                                                     T*       out,
                                                     int      run0_begin,
                                                     int      run0_len,
                                                     int      run1_begin,
                                                     int      run1_len,
                                                     int      out_begin,
                                                     int      out_len,
                                                     Compare  comp) {
  int tid = (int)threadIdx.x;

  // Partition output across threads.
  int chunk = (out_len + BLOCK_THREADS - 1) / BLOCK_THREADS;
  int o0    = tid * chunk;
  int o1    = min(out_len, o0 + chunk);
  if (o0 >= o1) return;

  const T* A = in + run0_begin;
  const T* B = in + run1_begin;
  int      m = run0_len, n = run1_len;

  int d0 = o0;
  int d1 = o1;
  int i0 = merge_path_find_i(A, m, B, n, d0, comp);
  int j0 = d0 - i0;
  int i1 = merge_path_find_i(A, m, B, n, d1, comp);
  int j1 = d1 - i1;

  int ia = i0, ib = j0;
  int out_idx = out_begin + o0;

  // Stable merge: if neither is strictly before the other, take from left (A).
  while (ia < i1 || ib < j1) {
    bool takeA;
    if (ib >= j1)
      takeA = true;
    else if (ia >= i1)
      takeA = false;
    else {
      // take A unless B should come before A
      takeA = !comp(B[ib], A[ia]);
    }
    out[out_idx++] = takeA ? A[ia++] : B[ib++];
  }
}

template <typename T, int BLOCK_THREADS>
struct BlockMergeSortLike_1Item {
  static constexpr int ITEMS_PER_THREAD = 1;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS;

  struct TempStorage {
    T buf0[TILE_ITEMS];
    T buf1[TILE_ITEMS];
  };

  TempStorage& temp;

  __device__ __forceinline__ explicit BlockMergeSortLike_1Item(TempStorage& s) : temp(s) {
  }

  template <typename Compare>
  __device__ __forceinline__ void Sort(T (&items)[1], Compare comp) {
    int tid = (int)threadIdx.x;

    // Write to shared
    temp.buf0[tid] = items[0];
    __syncthreads();

    T* in  = temp.buf0;
    T* out = temp.buf1;

    // Merge passes: run length doubles
    MUTLASS_PRAGMA_UNROLL
    for (int run = 1; run < TILE_ITEMS; run <<= 1) {
      int pair_len = run << 1;

      // Merge all pairs in this pass
      MUTLASS_PRAGMA_UNROLL
      for (int base = 0; base < TILE_ITEMS; base += pair_len) {
        int len0 = min(run, TILE_ITEMS - base);
        int len1 = min(run, TILE_ITEMS - (base + run));
        if (len1 <= 0) {
          // copy leftover run
          int chunk = (len0 + BLOCK_THREADS - 1) / BLOCK_THREADS;
          int o0    = tid * chunk;
          int o1    = min(len0, o0 + chunk);
          for (int k = o0; k < o1; ++k) out[base + k] = in[base + k];
        } else {
          block_merge_two_runs<BLOCK_THREADS>(in, out, base, len0, base + run, len1, base, len0 + len1, comp);
        }
      }

      __syncthreads();
      T* tmp = in;
      in     = out;
      out    = tmp;
      __syncthreads();
    }

    items[0] = in[tid];
  }
};

namespace mate::attention::fmha {
// Sort in descending order
template <typename T>
struct PrepareSortOp {
  __device__ __forceinline__ bool operator()(T const& lhs, T const& rhs) {
    return lhs > rhs;
  }
};

template <>
struct PrepareSortOp<int2> {
  __device__ __forceinline__ bool operator()(int2 const& lhs, int2 const& rhs) const {
    return lhs.x > rhs.x;
  }
};

template <>
struct PrepareSortOp<int4> {
  __device__ __forceinline__ bool operator()(int4 const& lhs, int4 const& rhs) const {
    return lhs.x > rhs.x;
  }
};

template <int  NumWarps,
          bool Sort,
          bool PackGQA,
          bool Causal,
          bool L2Swizzle        = false,
          bool HasCuSeqlensQ    = false,
          bool HasSequsedQ      = false,
          bool HasCuSeqlensK    = false,
          bool HasSequsedK      = false,
          bool HasCuSeqlensKNew = false,
          bool HasLeftpadK      = false>
__global__ void get_metadata_kernel(int                   seqlen_q_static,
                                    int                   seqlen_k_static,
                                    int                   seqlen_k_new_static,
                                    uint32_t const* const cu_seqlens_q,
                                    uint32_t const* const cu_seqlens_k,
                                    uint32_t const* const cu_seqlens_k_new,
                                    uint32_t const* const seqused_q,
                                    uint32_t const* const seqused_k,
                                    uint32_t const* const leftpad_k_ptr,
                                    int                   num_batch,
                                    int                   num_head,
                                    int                   qhead_per_khead,
                                    int                   num_mp,
                                    int                   num_splits_static,
                                    mutlass::FastDivmod   blockm_divmod,
                                    mutlass::FastDivmod   blockn_divmod,
                                    // int* const tile_count_semaphore,
                                    int* const num_m_blocks_ptr,
                                    int* const num_splits_dynamic_ptr,
                                    int* const varlen_batch_idx_ptr,
                                    // int* const num_n_blocks_ptr,
                                    int* const num_nheads_in_l2_ptr,
                                    int        max_kvblocks_in_l2) {
  static constexpr int kNumBatchPerWarp = mutlass::NumThreadsPerWarp - 1;  // Each warp processes 31 batches
  static constexpr int BATCH_PER_THREAD = 1;
  static constexpr int BLOCK_DIM_X      = NumWarps * mutlass::NumThreadsPerWarp;
  static_assert(BLOCK_DIM_X * BATCH_PER_THREAD == NumWarps * mutlass::NumThreadsPerWarp);
  using BlockMergeSort = BlockMergeSortLike_1Item<int4, BLOCK_DIM_X>;

  __shared__ int                                  total_blocks_smem[1];
  __shared__ typename BlockMergeSort::TempStorage temp_storage;

  if (threadIdx.x == 0) {
    total_blocks_smem[0] = 0;
  }
  __syncthreads();

  int lane                 = threadIdx.x % mutlass::NumThreadsPerWarp;
  int warp_idx             = threadIdx.x / mutlass::NumThreadsPerWarp;
  int batch_cta_idx_offset = static_cast<int>(blockIdx.x) * kNumBatchPerWarp * mutlass::NumThreadsPerWarp;
  int batch_offset         = batch_cta_idx_offset + warp_idx * kNumBatchPerWarp;
  int batch_idx            = batch_offset + lane;

  auto get_num_m_blocks = [&](int batch_idx) {
    int seqlen;
    if constexpr (HasSequsedQ) {
      seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
    } else if constexpr (HasCuSeqlensQ) {
      int cur_cu_seqlen  = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
      int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
      seqlen             = next_cu_seqlen - cur_cu_seqlen;
    } else {
      seqlen = seqlen_q_static;
    }
    if constexpr (PackGQA) {
      seqlen *= qhead_per_khead;
    }
    return batch_idx < num_batch && lane < kNumBatchPerWarp ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
  };

  auto get_num_n_blocks = [&](int batch_idx) {
    int leftpad_k;
    if constexpr (HasLeftpadK) {
      leftpad_k = batch_idx < num_batch ? leftpad_k_ptr[batch_idx] : 0;
    } else {
      leftpad_k = 0;
    }
    int seqlen;
    if constexpr (HasSequsedK) {
      seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
    } else if constexpr (HasCuSeqlensK) {
      int cur_cu_seqlen  = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
      int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
      seqlen             = next_cu_seqlen - cur_cu_seqlen;
    } else {
      seqlen = seqlen_k_static;
    }
    int seqlen_new;
    if constexpr (HasCuSeqlensKNew) {
      int cur_cu_seqlen_new  = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
      int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
      seqlen_new             = next_cu_seqlen_new - cur_cu_seqlen_new;
    } else {
      seqlen_new = seqlen_k_new_static;
    }
    seqlen = seqlen - leftpad_k + seqlen_new;
    return batch_idx < num_batch && lane < kNumBatchPerWarp ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
  };

  auto get_nheads_in_l2 = [&](int n_blocks) {
    int nheads_in_l2 = n_blocks * 16 <= max_kvblocks_in_l2  ? 16
                       : n_blocks * 8 <= max_kvblocks_in_l2 ? 8
                       : n_blocks * 4 <= max_kvblocks_in_l2 ? 4
                       : n_blocks * 2 <= max_kvblocks_in_l2 ? 2
                                                            : 1;
    if constexpr (!PackGQA) {
      nheads_in_l2 *= qhead_per_khead;
    }
    return min(nheads_in_l2, num_head);
  };

  int num_m_blocks = get_num_m_blocks(batch_idx);
  int num_n_blocks = get_num_n_blocks(batch_idx);

  int num_splits_dynamic;
  if (num_splits_static == 1 || static_cast<int>(gridDim.x) != 1) {
    num_splits_dynamic = 1;
  } else {
    int total_blocks = num_m_blocks * num_n_blocks;
    // BlockSum
    MUTLASS_PRAGMA_UNROLL
    for (int i = mutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
      total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
    }
    if (lane == 0) atomicAdd(total_blocks_smem, total_blocks);
    __syncthreads();
    total_blocks = total_blocks_smem[0];

    int blocks_per_mp =
        static_cast<int>(ceilf(static_cast<float>(total_blocks) * 1.1f * static_cast<float>(num_head) / num_mp));
    num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_mp - 1) / blocks_per_mp, num_splits_static), 1);
    int n_blocks_per_tile = mutlass::ceil_div(num_n_blocks, num_splits_dynamic);
    int extra_n_blocks    = n_blocks_per_tile * num_splits_dynamic - num_n_blocks;
    // if(threadIdx.x==0){
    //   printf("n_blocks_per_tile: %d num_splits_dyn: %d num_n_blocks: %d\n",
    //     n_blocks_per_tile, num_splits_dynamic, num_n_blocks);
    //   printf("extra_n_blocks: %d\n", extra_n_blocks);
    // }
    while (extra_n_blocks >= n_blocks_per_tile) {
      // if(threadIdx.x==0)
      //   printf("n_blocks_per_tile: %d num_splits_dyn: %d \n", n_blocks_per_tile, num_splits_dynamic);
      num_splits_dynamic--;
      n_blocks_per_tile = mutlass::ceil_div(num_n_blocks, num_splits_dynamic);
      extra_n_blocks    = n_blocks_per_tile * num_splits_dynamic - num_n_blocks;
      // if(threadIdx.x==0)
      //   printf("extra_n_blocks: %d\n", extra_n_blocks);
    }
    // if(threadIdx.x==0){
    // printf("new_num_n_blocks: %d num_splits_dyn: %d \n", n_blocks_per_tile, num_splits_dynamic);
    // printf("extra_n_blocks: %d\n", extra_n_blocks);}
    num_n_blocks = n_blocks_per_tile;
  }

  if constexpr (Sort) {
    // thread 31 is invalid
    if (lane == kNumBatchPerWarp || batch_idx >= num_batch) {
      num_n_blocks = INT_MIN;
    } else if constexpr (Causal) {
      num_n_blocks =
          num_n_blocks * blockn_divmod.divisor - num_m_blocks * blockm_divmod.divisor;  // n = n * bn - m * bm
    }
    int4 batch_coords[BATCH_PER_THREAD];
    batch_coords[0] = make_int4(num_n_blocks, num_m_blocks, num_splits_dynamic, batch_idx);
    // Sort
    BlockMergeSort(temp_storage).Sort(batch_coords, PrepareSortOp<int4>());

    if constexpr (Causal) {
      batch_coords[0].x =
          blockn_divmod.div(batch_coords[0].x + batch_coords[0].y * blockm_divmod.divisor);  // recover num_n_blocks
    }
    // After sorting, only first 31 * 32 threads contains valid data
    batch_idx = batch_cta_idx_offset + threadIdx.x;
    if (batch_idx < num_batch && threadIdx.x < kNumBatchPerWarp * 32) {
      if constexpr (L2Swizzle) {
        num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(batch_coords[0].x, 1));
      }
      num_m_blocks_ptr[batch_idx]       = batch_coords[0].y;
      num_splits_dynamic_ptr[batch_idx] = batch_coords[0].z;
      varlen_batch_idx_ptr[batch_idx]   = batch_coords[0].w;
    }
  } else {
    if (batch_idx < num_batch && lane < kNumBatchPerWarp) {
      if constexpr (L2Swizzle) {
        num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(num_n_blocks, 1));
      }
      num_splits_dynamic_ptr[batch_idx] = num_splits_dynamic;
      num_m_blocks_ptr[batch_idx]       = num_m_blocks;
    }
  }
}

inline int num_splits_heuristic(int  total_mblocks,
                                int  num_mp,
                                int  num_n_blocks,
                                int  num_m_blocks,
                                int  size_one_kv_head,
                                bool is_causal_or_local,
                                int  max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  // However, in the case of super long seqlen where each head of KV doesn't even fit into
  // L2 (we assume that L2 size is 50MB), we want to split.
  if (total_mblocks >= 0.8f * num_mp) {
    int const size_l2 = 1.5 * 1024 * 1024;  // 1.5 MB
    // Only split if there are enough queries to go over the KV at least twice
    // Don't split if causal
    if (size_one_kv_head > size_l2 && num_m_blocks >= num_mp * 2 && !is_causal_or_local) {
      return std::min((size_one_kv_head + size_l2 - 1) / size_l2, max_splits);
    } else {
      return 1;
    }
  }
  // If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
  if (num_n_blocks <= 4) {
    return 1;
  }
  max_splits                        = std::min({max_splits, num_mp, num_n_blocks});
  float              max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    float n_waves = float(total_mblocks * num_splits) / num_mp;
    float eff     = n_waves / ceil(n_waves);
    // printf("num_splits = %d, eff = %f\n", num_splits, eff);
    if (eff > max_efficiency) {
      max_efficiency = eff;
    }
    efficiency.push_back(eff);
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

inline int get_num_splits(FmhaFwdParams const& params, int tile_m, int tile_n) {
  // TODO: leftpad_k
  bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k;

  int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
  // TODO: metadata kernel need to set params for local.
  int const seqlen_k_loaded = params.seqlen_k;
  // int const seqlen_k_loaded = !params.is_local
  //   ? params.seqlen_k
  //   : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + tile_m));
  int const num_n_blocks = (seqlen_k_loaded + tile_n - 1) / tile_n;
  int const num_m_blocks = (seqlen_q_packgqa + tile_m - 1) / tile_m;
  // TODO: Other dtype size (fp8, etc.)
  int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (2);
  // TODO: for non-scheduled, this should be batch
  int total_mblocks = params.b * params.h_k * num_m_blocks;

  return num_splits_heuristic(total_mblocks,
                              params.num_mp,
                              num_n_blocks,
                              num_m_blocks,
                              size_one_kv_head,
                              params.is_causal || params.is_local,
                              128);
}

}  // namespace mate::attention::fmha
