/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <musa.h>

#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;

  // The number of heads.
  int h, h_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  using index_t = int64_t;

  // The O matrix (output).
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  // For FP8 scaling
  float* __restrict__ q_descale_ptr;
  float* __restrict__ k_descale_ptr;
  float* __restrict__ v_descale_ptr;
  index_t q_descale_batch_stride;
  index_t q_descale_head_stride;
  index_t k_descale_batch_stride;
  index_t k_descale_head_stride;
  index_t v_descale_batch_stride;
  index_t v_descale_head_stride;

  // The dimensions.
  int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
  int total_q, total_k, total_knew;
  int b_k;             // When having KV cache and with cache_batch_idx, K & V might have larger batch size
                       // than Q
  int dv, dv_rounded;  // For the case where V headdim is different from Q/K headdim

  // The scaling factors for the kernel.
  float scale_softmax;
  float softcap;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;
  int* __restrict__ cu_seqlens_knew;
  int* __restrict__ leftpad_k;

  // If provided, the actual length of each q/k sequence.
  int* __restrict__ seqused_q;
  int* __restrict__ seqused_k;

  // The stride between rows of Oaccum.
  index_t oaccum_split_stride;
  index_t oaccum_batch_stride;
  index_t oaccum_row_stride;
  index_t oaccum_head_stride;

  // The stride between rows of LSEaccum.
  index_t lseaccum_split_stride;
  index_t lseaccum_batch_stride;
  index_t lseaccum_head_stride;

  index_t lse_batch_stride;
  index_t lse_head_stride;

  // The K_new and V_new matrices.
  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;

  void* __restrict__ qv_ptr;
  index_t qv_batch_stride;
  index_t qv_row_stride;
  index_t qv_head_stride;

  // The cos and sin matrices for rotary embedding.
  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;
  int* __restrict__ seqlens_rotary;

  // The indices to index into the KV cache.
  int* __restrict__ kv_batch_idx;

  // Paged KV cache
  int* __restrict__ page_table;
  index_t page_table_batch_stride;
  int     page_size;
  int     num_pages;
  bool    pagedkv_tma;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  // uint32_t p_dropout_in_uint;
  // uint16_t p_dropout_in_uint16_t;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;

  // Local window size
  int window_size_left, window_size_right;
  int attention_chunk;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;

  bool is_rotary_interleaved;

  int  num_splits;  // For split-KV version
  bool pack_gqa;

  int* __restrict__ tile_count_semaphore;
  int* __restrict__ num_m_blocks_ptr;
  // int * __restrict__ num_n_blocks_ptr;
  int* __restrict__ num_splits_dynamic_ptr;
  int* __restrict__ varlen_batch_idx_ptr;  // virtual -> actual
  int* __restrict__ num_nheads_in_l2_ptr;
  bool skip_scheduler_metadata_computation;
  bool varlen_sort_batches;
  int  tile_count_semaphore_offset;
  bool head_swizzle;
  bool prepare_varlen_pdl;

  int  arch;
  int  num_sm;
  bool is_varlen_q;
  bool is_varlen_k;
};

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split,
          bool PackGQA>
void dispatch_fmha_kernel(Flash_fwd_params params);
