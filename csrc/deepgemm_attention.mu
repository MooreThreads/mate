#include <torch/all.h>
#include <torch/torch.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include "mate/gemm/deep_gemm/mp31_fp8_mqa_logits.hpp"
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
  } else if (next_n == 4) {
    constexpr int kNextN = 4;
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

  TORCH_CHECK(context_lens.scalar_type() == torch::kInt32);
  TORCH_CHECK(context_lens.is_contiguous());
  TORCH_CHECK(context_lens.dim() == 1 || context_lens.dim() == 2,
              "context_lens must be 1D [batch] or 2D [batch, next_n]");

  const bool is_context_lens_2d = (context_lens.dim() == 2);
  const int  batch_size         = context_lens.size(0);
  const int  next_n             = is_context_lens_2d ? static_cast<int>(context_lens.size(1)) : 1;

  TORCH_CHECK(
      !is_context_lens_2d || next_n == 1 || next_n == 2 || next_n == 4,
      "2D context_lens: next_n must be 1, 2 or 4, got ",
      next_n);

  auto dprops = at::musa::getCurrentDeviceProperties();
  num_mps     = num_mps <= 0 ? dprops->multiProcessorCount : num_mps;

  auto schedule_metadata = torch::empty({num_mps + 1, 2}, context_lens.options());

  constexpr int num_math_warpsquads = 4;
  constexpr int num_threads         = 32;
  const int     aligned_batch_size  = round_up(batch_size, 32);
  const int     split_kv            = block_kv * num_math_warpsquads;
  const int     smem_size           = aligned_batch_size * static_cast<int>(sizeof(int));

  TORCH_CHECK(smem_size <= 192 * 1024);

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  auto* ctx_ptr  = static_cast<uint32_t*>(context_lens.data_ptr());
  auto* meta_ptr = static_cast<uint32_t*>(schedule_metadata.data_ptr());

  auto launch = [&](auto AlignedBatchSizeTag, auto IsContextLens2DTag, auto NextNTag) {
    constexpr int  kAlignedBatch = decltype(AlignedBatchSizeTag)::value;
    constexpr bool kIs2D         = decltype(IsContextLens2DTag)::value;
    constexpr int  kNextN        = decltype(NextNTag)::value;
    mate::deep_gemm::mpxx_paged_mqa_logits_metadata<kAlignedBatch, kIs2D, kNextN>
        <<<1, num_threads, smem_size, stream>>>(batch_size, ctx_ptr, meta_ptr, split_kv, num_mps);
  };

  auto dispatch_next_n = [&](auto AlignedBatchSizeTag, auto IsContextLens2DTag) {
    if (next_n == 1) {
      launch(AlignedBatchSizeTag, IsContextLens2DTag, mute::_1{});
    } else if (next_n == 2) {
      launch(AlignedBatchSizeTag, IsContextLens2DTag, mute::_2{});
    } else {
      launch(AlignedBatchSizeTag, IsContextLens2DTag, mute::_4{});
    }
  };

  auto dispatch_is_2d = [&](auto AlignedBatchSizeTag) {
    if (is_context_lens_2d) {
      dispatch_next_n(AlignedBatchSizeTag, std::true_type{});
    } else {
      dispatch_next_n(AlignedBatchSizeTag, std::false_type{});
    }
  };

  if (aligned_batch_size == 32) {
    dispatch_is_2d(mute::_32{});
  } else if (aligned_batch_size == 64) {
    dispatch_is_2d(mute::_64{});
  } else if (aligned_batch_size == 96) {
    dispatch_is_2d(mute::_96{});
  } else if (aligned_batch_size == 128) {
    dispatch_is_2d(mute::_128{});
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

  const int batch_size_next_n     = weights.size(0);
  const int schedule_meta_size    = schedule_meta.size(0);
  const int meta_info_size        = schedule_meta.size(1);
  const int kv_cache_stride_bytes = fused_kv_cache.stride(0);

  auto      dprops  = at::musa::getCurrentDeviceProperties();
  const int num_mps = dprops->multiProcessorCount;

  TORCH_CHECK(context_lens.dim() == 1 || context_lens.dim() == 2,
              "context_lens must be 1D [batch] or 2D [batch, next_n]");
  const bool is_context_lens_2d = (context_lens.dim() == 2);

  if (is_context_lens_2d) {
    TORCH_CHECK(context_lens.size(0) == batch_size && context_lens.size(1) == next_n,
                "2D context_lens shape must be [batch_size, next_n]");
  } else {
    TORCH_CHECK(context_lens.size(0) == batch_size);
  }

  TORCH_CHECK(batch_size == block_table.size(0));
  TORCH_CHECK(batch_size_next_n == batch_size * next_n);
  TORCH_CHECK(num_heads == weights.size(1) && num_heads_kv == 1);
  TORCH_CHECK(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
  TORCH_CHECK(schedule_meta_size == num_mps + 1 && meta_info_size == 2);

  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(next_n == 1 or next_n == 2 or next_n == 4);
  TORCH_CHECK(block_kv == 64);
  TORCH_CHECK(num_heads == 32 || num_heads == 64, "Unsupported num_heads=", num_heads);
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

  const auto& aligned_max_context_len = round_up(max_context_len, num_math_warp_groups * block_kv);
  auto        logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
  logits             = logits.slice(-1, 0, max_context_len);

  using Element = mutlass::float_e4m3_t;

  auto q_mn_view          = q.view({-1, head_dim});
  auto stride_q           = make_stride(q_mn_view.stride(0), _1{});
  auto stride_k           = make_stride(kv_cache.stride(1), _1{}, kv_cache.stride(0));
  auto stride_sfk         = make_stride(_1{}, kv_cache_scales.stride(0));
  auto weights_mn_view    = weights.view({batch_size, -1});
  auto stride_w           = make_stride(_1{}, weights_mn_view.stride(0));
  auto stride_block_table = make_stride(block_table.stride(0), _1{});
  auto stride_logits      = make_stride(logits.stride(0), _1{});

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  auto launch_mqa_logits = [&](auto NextNTag, auto NumHeadsTag, auto HeadDimTag, auto IsContextLens2DTag) {
    constexpr int  kNextN           = decltype(NextNTag)::value;
    constexpr int  kNumHeads        = decltype(NumHeadsTag)::value;
    constexpr int  kHeadDim         = decltype(HeadDimTag)::value;
    constexpr bool kIsContextLens2D = decltype(IsContextLens2DTag)::value;

    using Fp8PagedMqaLogitsKernel = mate::deep_gemm::Mp31Fp8PagedMqaLogits<kNextN,
                                                                           kNumHeads,
                                                                           kHeadDim,
                                                                           BLOCK_KV,
                                                                           kIsContextLens2D,
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
        .num_kv_blocks      = num_kv_blocks,
    };

    auto params = Fp8PagedMqaLogitsKernel::to_underlying_arguments(arguments);
    mutlass::device_kernel<Fp8PagedMqaLogitsKernel>
        <<<num_mps, Fp8PagedMqaLogitsKernel::MaxThreadsPerBlock, Fp8PagedMqaLogitsKernel::SharedStorageSize, stream>>>(
            params);

    MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
  };

  auto dispatch_num_heads = [&](auto NextNTag, auto HeadDimTag, auto IsContextLens2DTag) {
    if (num_heads == 32) {
      launch_mqa_logits(NextNTag, mute::_32{}, HeadDimTag, IsContextLens2DTag);
    } else if (num_heads == 64) {
      launch_mqa_logits(NextNTag, mute::_64{}, HeadDimTag, IsContextLens2DTag);
    } else {
      TORCH_CHECK(false, "Unsupported num_heads=", num_heads, " (supported: 32/64)");
    }
  };

  auto dispatch_head_dim = [&](auto NextNTag, auto IsContextLens2DTag) {
    if (head_dim == 32) {
      dispatch_num_heads(NextNTag, mute::_32{}, IsContextLens2DTag);
    } else if (head_dim == 64) {
      dispatch_num_heads(NextNTag, mute::_64{}, IsContextLens2DTag);
    } else if (head_dim == 128) {
      dispatch_num_heads(NextNTag, mute::_128{}, IsContextLens2DTag);
    } else {
      TORCH_CHECK(false, "Unsupported head_dim=", head_dim, " (supported: 32/64/128)");
    }
  };

  auto dispatch_is_2d = [&](auto NextNTag) {
    if (is_context_lens_2d) {
      dispatch_head_dim(NextNTag, std::true_type{});
    } else {
      dispatch_head_dim(NextNTag, std::false_type{});
    }
  };

  if (next_n == 1) {
    dispatch_is_2d(mute::_1{});
  } else if (next_n == 2) {
    dispatch_is_2d(mute::_2{});
  } else if (next_n == 4) {
    dispatch_is_2d(mute::_4{});
  } else {
    TORCH_CHECK(false, "Unsupported next_n=", next_n, " (supported: 1/2/4)");
  }

  if (clean_logits) {
    if (is_context_lens_2d) {
      auto context_lens_flat = context_lens.reshape({-1});  // [batch*next_n]
      mpxx_clean_logits(logits,
                        std::nullopt,
                        context_lens_flat,
                        /*next_n=*/1,
                        batch_size * next_n,
                        max_context_len,
                        aligned_max_context_len);
    } else {
      mpxx_clean_logits(
          logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    }
  }

  return logits;
}

at::Tensor fp8_mqa_logits(const at::Tensor&                q,
                          const at::Tensor&                kv,
                          const at::Tensor&                weights,
                          const at::Tensor&                cu_seq_len_k_start,
                          const at::Tensor&                cu_seq_len_k_end,
                          const std::optional<at::Tensor>& kv_scale_opt,
                          const bool                       clean_logits,
                          const int64_t                    max_seq_len_k) {
  c10::musa::OptionalMUSAGuard device_guard(device_of(q));

  musaDeviceProp device_prop{};
  const int      device = q.device().index();
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&device_prop, device));

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  TORCH_CHECK(q.dim() == 3);
  TORCH_CHECK(kv.dim() == 2);
  TORCH_CHECK(weights.dim() == 2);

  const int64_t seq_len    = q.size(0);
  const int64_t num_heads  = q.size(1);
  const int64_t head_dim   = q.size(2);
  const int64_t seq_len_kv = kv.size(0);

  TORCH_CHECK(kv.size(1) == head_dim);

  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(kv.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(weights.scalar_type() == torch::kFloat);

  TORCH_CHECK(q.stride(-1) == 1);
  TORCH_CHECK(kv.stride(-1) == 1);
  TORCH_CHECK(weights.is_contiguous());

  TORCH_CHECK(cu_seq_len_k_start.dim() == 1 && cu_seq_len_k_start.size(0) == seq_len);
  TORCH_CHECK(cu_seq_len_k_end.dim() == 1 && cu_seq_len_k_end.size(0) == seq_len);
  TORCH_CHECK(cu_seq_len_k_start.scalar_type() == torch::kInt && cu_seq_len_k_end.scalar_type() == torch::kInt);
  TORCH_CHECK(cu_seq_len_k_start.is_contiguous() && cu_seq_len_k_end.is_contiguous());
  TORCH_CHECK(cu_seq_len_k_start.device() == q.device() && cu_seq_len_k_end.device() == q.device());

  constexpr int kBlockKV = 128;
  constexpr int kBlockQ  = 2;

  auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

  at::Tensor kv_scale;
  if (kv_scale_opt.has_value()) {
    kv_scale = kv_scale_opt.value();
    TORCH_CHECK(kv_scale.scalar_type() == torch::kFloat);
    TORCH_CHECK(kv_scale.numel() == seq_len_kv);
    TORCH_CHECK(kv_scale.is_contiguous());
    TORCH_CHECK(kv_scale.device() == q.device());
  } else {
    kv_scale = torch::ones({seq_len_kv}, q.options().dtype(torch::kFloat));
  }

  const bool compressed_logits = (max_seq_len_k > 0);
  int32_t    max_seq_kv_param  = static_cast<int32_t>(seq_len_kv);

  if (compressed_logits) {
    TORCH_CHECK(!clean_logits);
    TORCH_CHECK(max_seq_len_k <= seq_len_kv);
    max_seq_kv_param = static_cast<int32_t>(max_seq_len_k);
  }

  const int64_t num_kv_blocks = ceil_div(seq_len_kv, kBlockKV);

  const int64_t out_cols_logical = static_cast<int64_t>(max_seq_kv_param);
  const int64_t out_cols_padded  = ceil_div(out_cols_logical, kBlockKV) * kBlockKV;

  const auto opts    = q.options().dtype(torch::kFloat);
  at::Tensor out_pad = torch::empty({seq_len, out_cols_padded}, opts);

  auto q_mn_view = q.view({seq_len * num_heads, head_dim});
  auto stride_q  = make_stride(int32_t(q_mn_view.stride(0)), _1{});
  using StrideQ  = decltype(stride_q);

  auto kv_mn_view   = kv.view({seq_len_kv, head_dim});
  auto stride_k_row = int32_t(kv_mn_view.stride(0));
  auto stride_k     = make_stride(stride_k_row, _1{}, int32_t(kBlockKV) * stride_k_row);
  using StrideK     = decltype(stride_k);

  auto stride_sfk = make_stride(_1{}, int32_t(kBlockKV) * int32_t(kv_scale.stride(0)));
  using StrideSFK = decltype(stride_sfk);

  auto weights_mn_view = weights.view({seq_len, num_heads});
  auto stride_w        = make_stride(_1{}, int32_t(num_heads * kBlockQ));
  using StrideW        = decltype(stride_w);

  auto stride_logits = make_stride(int32_t(out_pad.stride(0)), _1{});
  using StrideLogits = decltype(stride_logits);

  using Element = mutlass::float_e4m3_t;

  const int64_t q_groups = ceil_div(seq_len, kBlockQ);

  auto ptr_ks = cu_seq_len_k_start.data_ptr<int32_t>();
  auto ptr_ke = cu_seq_len_k_end.data_ptr<int32_t>();

  const int num_mps = device_prop.multiProcessorCount;
  TORCH_CHECK(num_mps > 0);

  dim3 grid(static_cast<uint32_t>(num_mps));

  auto launch_mqa_logits = [&](auto num_heads_tag, auto head_dim_tag, auto compressed_tag) {
    constexpr bool kIsCompressedLogits = decltype(compressed_tag)::value;
    constexpr int  kNumHeadsT          = decltype(num_heads_tag)::value;
    constexpr int  kHeadDimT           = decltype(head_dim_tag)::value;

    using Kernel = mate::deep_gemm::Mp31Fp8NonPagedMqaLogits<kIsCompressedLogits,
                                                             kBlockQ,
                                                             kNumHeadsT,
                                                             kHeadDimT,
                                                             kBlockKV,
                                                             StrideQ,
                                                             StrideK,
                                                             StrideSFK,
                                                             StrideW,
                                                             StrideLogits>;

    dim3 block(Kernel::MaxThreadsPerBlock);

    typename Kernel::Arguments arguments{
        .q_group        = static_cast<int32_t>(q_groups),
        .seq_kv         = static_cast<int32_t>(seq_len_kv),
        .num_heads      = static_cast<int32_t>(num_heads),
        .head_dim       = static_cast<int32_t>(head_dim),
        .total_seq_q    = static_cast<int32_t>(seq_len),
        .ptr_q          = static_cast<Element const*>(q_mn_view.data_ptr()),
        .stride_q       = stride_q,
        .ptr_k          = static_cast<Element const*>(kv_mn_view.data_ptr()),
        .stride_k       = stride_k,
        .ptr_sfk        = kv_scale.data_ptr<float>(),
        .stride_sfk     = stride_sfk,
        .ptr_weights    = weights_mn_view.data_ptr<float>(),
        .stride_weights = stride_w,
        .ptr_ks         = ptr_ks,
        .ptr_ke         = ptr_ke,
        .ptr_logits     = out_pad.data_ptr<float>(),
        .stride_logits  = stride_logits,
        .max_seq_kv     = max_seq_kv_param,
    };

    auto params = Kernel::to_underlying_arguments(arguments);
    mutlass::device_kernel<Kernel><<<grid, block, Kernel::SharedStorageSize, stream>>>(params);
  };

  auto dispatch_num_heads = [&](auto head_dim_tag, auto compressed_tag) {
    if (num_heads == 32) {
      launch_mqa_logits(mute::_32{}, head_dim_tag, compressed_tag);
    } else if (num_heads == 64) {
      launch_mqa_logits(mute::_64{}, head_dim_tag, compressed_tag);
    } else {
      TORCH_CHECK(false);
    }
  };

  auto dispatch_head_dim = [&](auto compressed_tag) {
    if (head_dim == 32) {
      dispatch_num_heads(mute::_32{}, compressed_tag);
    } else if (head_dim == 64) {
      dispatch_num_heads(mute::_64{}, compressed_tag);
    } else if (head_dim == 128) {
      dispatch_num_heads(mute::_128{}, compressed_tag);
    } else {
      TORCH_CHECK(false);
    }
  };

  if (compressed_logits) {
    dispatch_head_dim(std::true_type{});
  } else {
    dispatch_head_dim(std::false_type{});
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());

  at::Tensor out = out_pad.narrow(/*dim=*/1, /*start=*/0, /*length=*/out_cols_logical);

  if (!compressed_logits && clean_logits) {
    std::optional<at::Tensor> cu_start_opt =
        cu_seq_len_k_start.defined() ? std::optional<at::Tensor>(cu_seq_len_k_start) : std::nullopt;

    mpxx_clean_logits(out,
                      cu_start_opt,
                      cu_seq_len_k_end,
                      /*next_n=*/1,
                      /*seq_len=*/static_cast<int>(seq_len),
                      /*seq_len_kv=*/static_cast<int>(seq_len_kv),
                      /*stride_kv=*/static_cast<uint64_t>(out.stride(0)));
  }

  return out;
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

  m.def(
      "fp8_mqa_logits("
      "Tensor q,"
      "Tensor kv,"
      "Tensor weights,"
      "Tensor cu_seq_len_k_start,"
      "Tensor cu_seq_len_k_end,"
      "Tensor? kv_scale = None,"
      "bool   clean_logits = False,"
      "int    max_seqlen_k = 0"
      ") -> Tensor");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata);
  m.impl("fp8_paged_mqa_logits", &fp8_paged_mqa_logits);
  m.impl("fp8_mqa_logits", &fp8_mqa_logits);
}
