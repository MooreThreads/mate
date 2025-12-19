#include <torch/all.h>
#include <torch/torch.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include "mate/gemm/deep_gemm/mp31_fp8_paged_mqa_logits.hpp"
#include "mate/gemm/deep_gemm/mpxx_clean_logits.hpp"
#include "mate_utils.muh"
#include "mutlass/device_kernel.h"

using namespace mute;

namespace {

void mpxx_clean_logits(const torch::Tensor&                logits,
                       const std::optional<torch::Tensor>& cu_seq_len_k_start,
                       const torch::Tensor&                cu_seq_len_k_end,
                       const int&                          next_n,
                       const int&                          seq_len,
                       const int&                          seq_len_kv,
                       const uint64_t&                     stride_kv) {
  constexpr int BLOCK_KV  = 8192;
  constexpr int NUM_WARPS = 8;
  constexpr int SMEM_SIZE = BLOCK_KV * sizeof(float);

  const at::musa::OptionalMUSAGuard device_guard(device_of(logits));
  musaStream_t                      stream = at::musa::getCurrentMUSAStream();

  auto      dprops  = at::musa::getCurrentDeviceProperties();
  const int num_mps = dprops->multiProcessorCount;

  if (next_n == 1) {
    constexpr int kNextN = 1;
    mate::deep_gemm::mpxx_clean_logits<kNextN, BLOCK_KV, NUM_WARPS><<<num_mps, NUM_WARPS * 32, SMEM_SIZE, stream>>>(
        seq_len,
        seq_len_kv,
        stride_kv,
        cu_seq_len_k_start.has_value() ? static_cast<uint32_t*>(cu_seq_len_k_start.value().data_ptr()) : nullptr,
        static_cast<uint32_t*>(cu_seq_len_k_end.data_ptr()),
        logits.data_ptr<float>());
  } else if (next_n == 2) {
    constexpr int kNextN = 2;
    mate::deep_gemm::mpxx_clean_logits<kNextN, BLOCK_KV, NUM_WARPS><<<num_mps, NUM_WARPS * 32, SMEM_SIZE, stream>>>(
        seq_len,
        seq_len_kv,
        stride_kv,
        cu_seq_len_k_start.has_value() ? static_cast<uint32_t*>(cu_seq_len_k_start.value().data_ptr()) : nullptr,
        static_cast<uint32_t*>(cu_seq_len_k_end.data_ptr()),
        logits.data_ptr<float>());
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

}  // namespace

torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int64_t block_kv, int64_t num_mps = 0) {
  const at::musa::OptionalMUSAGuard device_guard(device_of(context_lens));

  int const batch_size = context_lens.size(0);
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(context_lens.is_contiguous());

  auto dprops = at::musa::getCurrentDeviceProperties();
  num_mps     = num_mps <= 0 ? dprops->multiProcessorCount : num_mps;

  auto schedule_metadata = torch::empty({num_mps + 1, 2}, context_lens.options());

  constexpr int num_math_warpsquads = 4;
  constexpr int num_threads         = 32;
  const int     aligned_batch_size  = round_up(batch_size, 32);
  const int     split_kv            = block_kv * num_math_warpsquads;

  const int smem_size = aligned_batch_size * static_cast<int>(sizeof(int));

  TORCH_CHECK(smem_size <= 192 * 1024);

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  auto launch_paged_mqa_logits_metadata = [&](auto AlignedBatchSize) {
    constexpr int kAlignedBatch = decltype(AlignedBatchSize)::value;
    mate::deep_gemm::mpxx_paged_mqa_logits_metadata<kAlignedBatch>
        <<<1, num_threads, smem_size, stream>>>(batch_size,
                                                static_cast<uint32_t*>(context_lens.data_ptr()),
                                                static_cast<uint32_t*>(schedule_metadata.data_ptr()),
                                                split_kv,
                                                num_mps);
  };

  if (aligned_batch_size == 32) {
    launch_paged_mqa_logits_metadata(mute::_32{});
  } else if (aligned_batch_size == 64) {
    launch_paged_mqa_logits_metadata(mute::_64{});
  } else if (aligned_batch_size == 96) {
    launch_paged_mqa_logits_metadata(mute::_96{});
  } else if (aligned_batch_size == 128) {
    launch_paged_mqa_logits_metadata(mute::_128{});
  } else {
    TORCH_CHECK(false, "Unsupported batch size: ", aligned_batch_size);
  }
  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());

  return schedule_metadata;
}

torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q,
                                   const torch::Tensor& fused_kv_cache,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& context_lens,
                                   const torch::Tensor& block_table,
                                   const torch::Tensor& schedule_meta,
                                   const int64_t&       max_context_len,
                                   const bool&          clean_logits) {
  c10::musa::OptionalMUSAGuard device_guard(device_of(q));

  const int batch_size = q.size(0);
  const int next_n     = q.size(1);
  const int num_heads  = q.size(2);
  const int head_dim   = q.size(3);

  const int num_kv_blocks    = fused_kv_cache.size(0);
  const int block_kv         = fused_kv_cache.size(1);
  const int num_heads_kv     = fused_kv_cache.size(2);
  const int head_dim_with_sf = fused_kv_cache.size(3);

  const int batch_size_next_n = weights.size(0);

  const int max_block_len = block_table.size(1);

  const int schedule_meta_size = schedule_meta.size(0);
  const int meta_info_size     = schedule_meta.size(1);

  const int kv_cache_stride_bytes = fused_kv_cache.stride(0);
  const int block_table_stride    = block_table.stride(0);

  auto      dprops  = at::musa::getCurrentDeviceProperties();
  const int num_mps = dprops->multiProcessorCount;

  TORCH_CHECK(batch_size == context_lens.size(0) && batch_size == block_table.size(0));
  TORCH_CHECK(batch_size_next_n == batch_size * next_n);
  TORCH_CHECK(num_heads == weights.size(1) && num_heads_kv == 1);
  TORCH_CHECK(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
  TORCH_CHECK(schedule_meta_size == num_mps + 1 && meta_info_size == 2);

  // TODO: Relax these restrictions in the furture
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(next_n == 1 or next_n == 2);
  TORCH_CHECK(block_kv == 64);
  TORCH_CHECK(num_heads == 64);
  TORCH_CHECK(kv_cache_stride_bytes % sizeof(float) == 0);
  TORCH_CHECK(fused_kv_cache.stride(1) == head_dim_with_sf);
  TORCH_CHECK(fused_kv_cache.stride(2) == head_dim_with_sf);
  TORCH_CHECK(fused_kv_cache.stride(3) == 1);
  TORCH_CHECK(weights.is_contiguous());
  TORCH_CHECK(context_lens.is_contiguous());
  TORCH_CHECK(block_table.stride(1) == 1);
  TORCH_CHECK(schedule_meta.is_contiguous());

  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(fused_kv_cache.scalar_type() == torch::kByte);
  TORCH_CHECK(weights.scalar_type() == torch::kFloat);
  TORCH_CHECK(context_lens.scalar_type() == torch::kInt);
  TORCH_CHECK(block_table.scalar_type() == torch::kInt);
  TORCH_CHECK(schedule_meta.scalar_type() == torch::kInt);

  // Derive FP8 values and SF tensor from KV cache
  const auto& kv_cache        = torch::from_blob(fused_kv_cache.data_ptr(),
                                                 {num_kv_blocks, block_kv, head_dim},
                                                 {kv_cache_stride_bytes, head_dim, 1},
                                          torch::TensorOptions().dtype(torch::kFloat8_e4m3fn));
  const auto& kv_cache_scales = torch::from_blob(fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
                                                 {num_kv_blocks, block_kv},
                                                 {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
                                                 torch::TensorOptions().dtype(torch::kFloat32));

  constexpr int num_math_warp_groups = 4;
  constexpr int BLOCK_KV             = 64;
  constexpr int kNumHeads            = 64;

  // Allocate output
  const auto& aligned_max_context_len = round_up(max_context_len, num_math_warp_groups * block_kv);
  auto        logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
  logits             = logits.slice(-1, 0, max_context_len);

  using Element = mutlass::float_e4m3_t;

  // View q as 2D tensor
  auto q_mn_view          = q.view({-1, head_dim});
  auto stride_q           = make_stride(q_mn_view.stride(0), _1{});
  auto stride_k           = make_stride(kv_cache.stride(1), _1{}, kv_cache.stride(0));
  auto stride_sfk         = make_stride(_1{}, kv_cache_scales.stride(0));
  auto weights_mn_view    = weights.view({batch_size, -1});
  auto stride_w           = make_stride(_1{}, weights_mn_view.stride(0));
  auto stride_block_table = make_stride(block_table.stride(0), _1{});
  auto stride_logits      = make_stride(logits.stride(0), _1{});

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // TODO: dispatch different nextn & headdim
  auto launch_mqa_logits = [&](auto NextN, auto HeadDim) {
    constexpr int kNextN   = decltype(NextN)::value;
    constexpr int kHeadDim = decltype(HeadDim)::value;

    using Fp8PagedMqaLogitsKernel = mate::deep_gemm::Mp31Fp8PagedMqaLogits<kNextN,
                                                                           kNumHeads,
                                                                           kHeadDim,
                                                                           BLOCK_KV,
                                                                           decltype(stride_q),
                                                                           decltype(stride_k),
                                                                           decltype(stride_sfk),
                                                                           decltype(stride_w),
                                                                           decltype(stride_block_table),
                                                                           decltype(stride_logits)>;

    typename Fp8PagedMqaLogitsKernel::Arguments arguments = {
        .ptr_q              = static_cast<Element const*>(q.data_ptr()),
        .stride_q           = stride_q,
        .ptr_k              = static_cast<Element const*>(kv_cache.data_ptr()),
        .stride_k           = stride_k,
        .ptr_sfk            = kv_cache_scales.data_ptr<float>(),
        .stride_sfk         = stride_sfk,
        .ptr_weights        = weights.data_ptr<float>(),
        .stride_weights     = stride_w,
        .ptr_context_lens   = static_cast<uint32_t*>(context_lens.data_ptr()),
        .ptr_schedule_meta  = static_cast<uint32_t*>(schedule_meta.data_ptr()),
        .ptr_block_table    = block_table.data_ptr<int32_t>(),
        .stride_block_table = stride_block_table,
        .ptr_logits         = logits.data_ptr<float>(),
        .stride_logits      = stride_logits,
        .batch_size         = batch_size,
        .num_kv_blocks      = num_kv_blocks};

    typename Fp8PagedMqaLogitsKernel::Params params = Fp8PagedMqaLogitsKernel::to_underlying_arguments(arguments);

    mutlass::device_kernel<Fp8PagedMqaLogitsKernel>
        <<<num_mps, Fp8PagedMqaLogitsKernel::MaxThreadsPerBlock, Fp8PagedMqaLogitsKernel::SharedStorageSize, stream>>>(
            params);

    MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
  };

  if (next_n == 1) {
    if (head_dim == 32) {
      launch_mqa_logits(mute::_1{}, mute::_32{});
    } else if (head_dim == 64) {
      launch_mqa_logits(mute::_1{}, mute::_64{});
    } else if (head_dim == 128) {
      launch_mqa_logits(mute::_1{}, mute::_128{});
    } else {
      TORCH_CHECK(false, "Unsupported head dim");
    }
  } else if (next_n == 2) {
    if (head_dim == 32) {
      launch_mqa_logits(mute::_2{}, mute::_32{});
    } else if (head_dim == 64) {
      launch_mqa_logits(mute::_2{}, mute::_64{});
    } else if (head_dim == 128) {
      launch_mqa_logits(mute::_2{}, mute::_128{});
    } else {
      TORCH_CHECK(false, "Unsupported head dim");
    }
  } else {
    TORCH_CHECK(false, "Unsupported next n");
  }

  if (clean_logits) {
    mpxx_clean_logits(
        logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
  }

  return logits;
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "get_paged_mqa_logits_metadata("
      "Tensor context_lens,"
      "int block_kv,"
      "int num_mps = 0) -> Tensor");

  m.def(
      "fp8_paged_mqa_logits("
      "Tensor q,"
      "Tensor fused_kv_cache,"
      "Tensor weight,"
      "Tensor context_lens,"
      "Tensor block_table,"
      "Tensor schedule_meta,"
      "int max_context_len,"
      "bool clean_logits) -> Tensor");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata);
  m.impl("fp8_paged_mqa_logits", &fp8_paged_mqa_logits);
}
