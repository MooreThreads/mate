// Adapted from https://github.com/deepseek-ai/FlashMLA
#pragma once

#include <cstdint>

namespace mate::flash_mla {

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx (inclusive), begin_block_idx (inclusive), end_idx (inclusive), end_block_idx (exclusive),
// begin_n_split_idx, _, _, _]

struct GetDecodingMetadataParams {
  int* __restrict__ seqlens_k_ptr;
  int* __restrict__ tile_scheduler_metadata_ptr;
  int* __restrict__ num_splits_ptr;
  int batch_size;
  int block_size_n;
  int fixed_overhead_num_blocks;
  int num_mp_parts;
  int topk;
};

struct MlaCombineParams {
  using index_t = int64_t;

  void* __restrict__ o_ptr;
  void* __restrict__ softmax_lse_ptr;

  void* __restrict__ softmax_lseaccum_ptr;
  void* __restrict__ oaccum_ptr;
  int* __restrict__ num_splits_ptr;

  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  int q_seq_per_hk;
  int h_k;
  int batch_size;
  int num_mp_parts;
};

}  // namespace mate::flash_mla
