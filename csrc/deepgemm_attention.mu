#include <musa_runtime.h>

#include <cstdint>
#include <type_traits>

#include "mate/gemm/deep_gemm/mp31_fp8_mqa_logits.hpp"
#include "mate/gemm/deep_gemm/mp31_fp8_paged_mqa_logits.hpp"
#include "mate/gemm/deep_gemm/mpxx_clean_logits.hpp"
#include "mudnn_utils.hpp"
#include "mutlass/device_kernel.h"
#include "op_utils.hpp"

using namespace mute;

namespace {

inline int32_t* optional_int32_data_ptr(ffi::Optional<ffi::TensorView> maybe_tensor) {
  return maybe_tensor.has_value() ? static_cast<int32_t*>(maybe_tensor.value().data_ptr()) : nullptr;
}

void mpxx_clean_logits(ffi::TensorView                logits,
                       ffi::Optional<ffi::TensorView> cu_seq_len_k_start,
                       ffi::TensorView                cu_seq_len_k_end,
                       int64_t                        next_n,
                       int64_t                        seq_len,
                       int64_t                        seq_len_kv,
                       uint64_t                       stride_kv) {
  constexpr int BLOCK_KV  = 8192;
  constexpr int NUM_WARPS = 8;
  constexpr int SMEM_SIZE = BLOCK_KV * sizeof(float);

  CHECK_MUSA(logits);
  CHECK_INPUT_TYPE(logits, dl_float32);
  CHECK_MUSA(cu_seq_len_k_end);
  CHECK_CONTIGUOUS(cu_seq_len_k_end);
  CHECK_INPUT_TYPE(cu_seq_len_k_end, dl_int32);
  CHECK_DEVICE(cu_seq_len_k_end, logits);
  TVM_FFI_ICHECK(next_n == 1 || next_n == 2 || next_n == 4) << "next_n must be 1, 2 or 4";

  if (cu_seq_len_k_start.has_value()) {
    CHECK_MUSA(cu_seq_len_k_start.value());
    CHECK_CONTIGUOUS(cu_seq_len_k_start.value());
    CHECK_INPUT_TYPE(cu_seq_len_k_start.value(), dl_int32);
    CHECK_DEVICE(cu_seq_len_k_start.value(), logits);
  }

  ffi::MUSADeviceGuard device_guard(logits.device().device_id);

  musaDeviceProp dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, logits.device().device_id));
  const int num_mps = dprops.multiProcessorCount;
  TVM_FFI_ICHECK(num_mps > 0) << "No available MUSA MPs";

  musaStream_t stream = get_stream(logits.device());

  auto launch = [&](auto NextNTag) {
    constexpr int kNextN = decltype(NextNTag)::value;
    mate::deep_gemm::mpxx_clean_logits<kNextN, BLOCK_KV, NUM_WARPS><<<num_mps, NUM_WARPS * 32, SMEM_SIZE, stream>>>(
        static_cast<uint32_t>(seq_len),
        static_cast<uint32_t>(seq_len_kv),
        stride_kv,
        reinterpret_cast<uint32_t const*>(optional_int32_data_ptr(cu_seq_len_k_start)),
        static_cast<uint32_t*>(cu_seq_len_k_end.data_ptr()),
        static_cast<float*>(logits.data_ptr()));
  };

  if (next_n == 1) {
    launch(mute::_1{});
  } else if (next_n == 2) {
    launch(mute::_2{});
  } else {
    launch(mute::_4{});
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

}  // namespace

void get_paged_mqa_logits_metadata(ffi::TensorView context_lens, int64_t block_kv, ffi::TensorView schedule_meta) {
  CHECK_MUSA(context_lens);
  CHECK_MUSA(schedule_meta);
  CHECK_CONTIGUOUS(context_lens);
  CHECK_CONTIGUOUS(schedule_meta);
  CHECK_INPUT_TYPE(context_lens, dl_int32);
  CHECK_INPUT_TYPE(schedule_meta, dl_int32);
  CHECK_DEVICE(schedule_meta, context_lens);
  TVM_FFI_ICHECK(context_lens.dim() == 1 || context_lens.dim() == 2)
      << "context_lens must be 1D [batch] or 2D [batch, next_n]";
  CHECK_DIM(2, schedule_meta);
  TVM_FFI_ICHECK(block_kv > 0) << "block_kv must be positive";

  ffi::MUSADeviceGuard device_guard(context_lens.device().device_id);

  const bool is_context_lens_2d = (context_lens.dim() == 2);
  const int  batch_size         = static_cast<int>(context_lens.size(0));
  const int  next_n             = is_context_lens_2d ? static_cast<int>(context_lens.size(1)) : 1;

  TVM_FFI_ICHECK(!is_context_lens_2d || next_n == 1 || next_n == 2 || next_n == 4)
      << "2D context_lens: next_n must be 1, 2 or 4, got " << next_n;

  musaDeviceProp dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, context_lens.device().device_id));
  const int64_t num_mps = schedule_meta.size(0) - 1;
  TVM_FFI_ICHECK(schedule_meta.size(1) == 2) << "schedule_meta shape must be [num_mps + 1, 2]";
  TVM_FFI_ICHECK(num_mps > 0 && num_mps <= dprops.multiProcessorCount)
      << "num_mps must be in range [1, " << dprops.multiProcessorCount << "]";

  constexpr int num_math_warpsquads = 4;
  constexpr int num_threads         = 32;
  const int     aligned_batch_size  = round_up(batch_size, 32);
  const int     split_kv            = block_kv * num_math_warpsquads;
  const int     smem_size           = aligned_batch_size * static_cast<int>(sizeof(int));

  TVM_FFI_ICHECK(smem_size <= 192 * 1024) << "smem_size exceeds 192KB";

  musaStream_t stream = get_stream(context_lens.device());

  auto* ctx_ptr  = static_cast<uint32_t*>(context_lens.data_ptr());
  auto* meta_ptr = static_cast<uint32_t*>(schedule_meta.data_ptr());

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
    TVM_FFI_ICHECK(false) << "Unsupported batch size: " << aligned_batch_size;
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

ffi::Tensor fp8_paged_mqa_logits(ffi::TensorView q,
                                 ffi::TensorView fused_kv_cache,
                                 ffi::TensorView weights,
                                 ffi::TensorView context_lens,
                                 ffi::TensorView block_table,
                                 ffi::TensorView schedule_meta,
                                 int64_t         max_context_len,
                                 bool            clean_logits) {
  CHECK_INPUT(q);
  CHECK_MUSA(fused_kv_cache);
  CHECK_INPUT(weights);
  CHECK_MUSA(context_lens);
  CHECK_MUSA(block_table);
  CHECK_MUSA(schedule_meta);
  CHECK_DIM(4, q);
  CHECK_DIM(4, fused_kv_cache);
  CHECK_DIM(2, weights);
  CHECK_DIM(2, block_table);
  CHECK_DIM(2, schedule_meta);
  CHECK_INPUT_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(fused_kv_cache, dl_uint8);
  CHECK_INPUT_TYPE(weights, dl_float32);
  CHECK_INPUT_TYPE(context_lens, dl_int32);
  CHECK_INPUT_TYPE(block_table, dl_int32);
  CHECK_INPUT_TYPE(schedule_meta, dl_int32);
  CHECK_DEVICE(fused_kv_cache, q);
  CHECK_DEVICE(weights, q);
  CHECK_DEVICE(context_lens, q);
  CHECK_DEVICE(block_table, q);
  CHECK_DEVICE(schedule_meta, q);

  TVM_FFI_ICHECK(context_lens.dim() == 1 || context_lens.dim() == 2)
      << "context_lens must be 1D [batch] or 2D [batch, next_n]";
  CHECK_CONTIGUOUS(context_lens);
  CHECK_CONTIGUOUS(schedule_meta);
  TVM_FFI_ICHECK(block_table.stride(1) == 1) << "block_table must be contiguous at the last dimension";

  ffi::MUSADeviceGuard device_guard(q.device().device_id);

  const int64_t batch_size = q.size(0);
  const int64_t next_n     = q.size(1);
  const int64_t num_heads  = q.size(2);
  const int64_t head_dim   = q.size(3);

  const int64_t num_kv_blocks    = fused_kv_cache.size(0);
  const int64_t block_kv         = fused_kv_cache.size(1);
  const int64_t num_heads_kv     = fused_kv_cache.size(2);
  const int64_t head_dim_with_sf = fused_kv_cache.size(3);

  const bool is_context_lens_2d = (context_lens.dim() == 2);

  if (is_context_lens_2d) {
    TVM_FFI_ICHECK(context_lens.size(0) == batch_size && context_lens.size(1) == next_n)
        << "2D context_lens shape must be [batch_size, next_n]";
  } else {
    TVM_FFI_ICHECK(context_lens.size(0) == batch_size) << "1D context_lens shape must be [batch_size]";
  }

  const int64_t batch_size_next_n  = weights.size(0);
  const int64_t schedule_meta_size = schedule_meta.size(0);
  const int64_t meta_info_size     = schedule_meta.size(1);
  const int64_t kv_cache_stride    = fused_kv_cache.stride(0);
  const int     num_mps            = static_cast<int>(schedule_meta_size - 1);

  TVM_FFI_ICHECK(batch_size == block_table.size(0)) << "block_table.shape[0] must match batch_size";
  TVM_FFI_ICHECK(batch_size_next_n == batch_size * next_n) << "weights.shape[0] must match batch_size * next_n";
  TVM_FFI_ICHECK(num_heads == weights.size(1) && num_heads_kv == 1) << "Unsupported weights or kv head layout";
  TVM_FFI_ICHECK(head_dim_with_sf == head_dim + static_cast<int64_t>(sizeof(float)))
      << "fused_kv_cache last dim must be head_dim + 4";
  TVM_FFI_ICHECK(schedule_meta_size >= 2 && meta_info_size == 2) << "schedule_meta shape must be [num_mps + 1, 2]";
  TVM_FFI_ICHECK(next_n == 1 || next_n == 2 || next_n == 4) << "Unsupported next_n=" << next_n;
  TVM_FFI_ICHECK(block_kv == 64) << "Unsupported block_kv=" << block_kv;
  TVM_FFI_ICHECK(num_heads == 32 || num_heads == 64) << "Unsupported num_heads=" << num_heads;
  TVM_FFI_ICHECK(num_mps > 0) << "schedule_meta must encode at least one MP";
  TVM_FFI_ICHECK(max_context_len >= 0) << "max_context_len must be non-negative";
  TVM_FFI_ICHECK(kv_cache_stride % sizeof(float) == 0) << "fused_kv_cache row stride must be 4-byte aligned";
  TVM_FFI_ICHECK(fused_kv_cache.stride(1) == head_dim_with_sf);
  TVM_FFI_ICHECK(fused_kv_cache.stride(2) == head_dim_with_sf);
  TVM_FFI_ICHECK(fused_kv_cache.stride(3) == 1);

  constexpr int num_math_warp_groups    = 4;
  constexpr int BLOCK_KV                = 64;
  const int64_t aligned_max_context_len = round_up(max_context_len, int64_t(num_math_warp_groups * block_kv));

  ffi::Tensor logits_pad =
      alloc_tensor(ffi::Shape{batch_size * next_n, aligned_max_context_len}, dl_float32, q.device());
  ffi::Shape  logits_shape{batch_size * next_n, max_context_len};
  ffi::Shape  logits_strides{aligned_max_context_len, 1};
  ffi::Tensor logits = logits_pad.as_strided(logits_shape, logits_strides);

  using Element = mutlass::float_e4m3_t;

  auto stride_q   = make_stride(static_cast<int32_t>(q.stride(2)), _1{});
  auto stride_k   = make_stride(static_cast<int32_t>(head_dim), _1{}, static_cast<int32_t>(kv_cache_stride));
  auto stride_sfk = make_stride(_1{}, static_cast<int32_t>(kv_cache_stride / static_cast<int64_t>(sizeof(float))));
  auto stride_w   = make_stride(_1{}, static_cast<int32_t>(weights.size(1) * next_n));
  auto stride_block_table = make_stride(static_cast<int32_t>(block_table.stride(0)), _1{});
  auto stride_logits      = make_stride(static_cast<int32_t>(logits.stride(0)), _1{});

  auto* kv_cache_ptr = static_cast<Element const*>(fused_kv_cache.data_ptr());
  auto* kv_scale_ptr =
      reinterpret_cast<float const*>(static_cast<uint8_t*>(fused_kv_cache.data_ptr()) + block_kv * head_dim);

  musaStream_t stream = get_stream(q.device());

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
        .ptr_k              = kv_cache_ptr,
        .stride_k           = stride_k,
        .ptr_sfk            = kv_scale_ptr,
        .stride_sfk         = stride_sfk,
        .ptr_weights        = static_cast<float*>(weights.data_ptr()),
        .stride_weights     = stride_w,
        .ptr_context_lens   = static_cast<uint32_t*>(context_lens.data_ptr()),
        .ptr_schedule_meta  = static_cast<uint32_t*>(schedule_meta.data_ptr()),
        .ptr_block_table    = static_cast<int32_t*>(block_table.data_ptr()),
        .stride_block_table = stride_block_table,
        .ptr_logits         = static_cast<float*>(logits.data_ptr()),
        .stride_logits      = stride_logits,
        .batch_size         = static_cast<int32_t>(batch_size),
        .num_kv_blocks      = static_cast<int32_t>(num_kv_blocks),
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
      TVM_FFI_ICHECK(false) << "Unsupported num_heads=" << num_heads << " (supported: 32/64)";
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
      TVM_FFI_ICHECK(false) << "Unsupported head_dim=" << head_dim << " (supported: 32/64/128)";
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
    TVM_FFI_ICHECK(false) << "Unsupported next_n=" << next_n << " (supported: 1/2/4)";
  }

  if (clean_logits) {
    if (is_context_lens_2d) {
      ffi::Shape      context_lens_flat_shape{batch_size * next_n};
      ffi::Shape      context_lens_flat_strides{1};
      ffi::TensorView context_lens_flat = context_lens.as_strided(context_lens_flat_shape, context_lens_flat_strides);
      mpxx_clean_logits(logits,
                        ffi::Optional<ffi::TensorView>(),
                        context_lens_flat,
                        /*next_n=*/1,
                        batch_size * next_n,
                        max_context_len,
                        static_cast<uint64_t>(aligned_max_context_len));
    } else {
      mpxx_clean_logits(logits,
                        ffi::Optional<ffi::TensorView>(),
                        context_lens,
                        next_n,
                        batch_size * next_n,
                        max_context_len,
                        static_cast<uint64_t>(aligned_max_context_len));
    }
  }

  return logits;
}

ffi::Tensor fp8_mqa_logits(ffi::TensorView q,
                           ffi::TensorView kv,
                           ffi::TensorView weights,
                           ffi::TensorView cu_seq_len_k_start,
                           ffi::TensorView cu_seq_len_k_end,
                           ffi::TensorView kv_scale,
                           bool            clean_logits,
                           int64_t         max_seq_len_k) {
  CHECK_MUSA(q);
  CHECK_MUSA(kv);
  CHECK_INPUT(weights);
  CHECK_INPUT(cu_seq_len_k_start);
  CHECK_INPUT(cu_seq_len_k_end);
  CHECK_MUSA(kv_scale);
  CHECK_DIM(3, q);
  CHECK_DIM(2, kv);
  CHECK_DIM(2, weights);
  CHECK_DIM(1, cu_seq_len_k_start);
  CHECK_DIM(1, cu_seq_len_k_end);
  CHECK_DIM(1, kv_scale);
  CHECK_INPUT_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(kv, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(weights, dl_float32);
  CHECK_INPUT_TYPE(cu_seq_len_k_start, dl_int32);
  CHECK_INPUT_TYPE(cu_seq_len_k_end, dl_int32);
  CHECK_INPUT_TYPE(kv_scale, dl_float32);
  CHECK_DEVICE(kv, q);
  CHECK_DEVICE(weights, q);
  CHECK_DEVICE(cu_seq_len_k_start, q);
  CHECK_DEVICE(cu_seq_len_k_end, q);
  CHECK_DEVICE(kv_scale, q);

  ffi::MUSADeviceGuard device_guard(q.device().device_id);

  musaDeviceProp device_prop{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&device_prop, q.device().device_id));

  musaStream_t stream = get_stream(q.device());

  const int64_t seq_len    = q.size(0);
  const int64_t num_heads  = q.size(1);
  const int64_t head_dim   = q.size(2);
  const int64_t seq_len_kv = kv.size(0);

  TVM_FFI_ICHECK(kv.size(1) == head_dim) << "kv.shape[1] must match q.shape[2]";
  TVM_FFI_ICHECK(weights.size(0) == seq_len && weights.size(1) == num_heads)
      << "weights shape must be [seq_len, num_heads]";
  TVM_FFI_ICHECK(q.stride(-1) == 1) << "q must be contiguous at the last dimension";
  TVM_FFI_ICHECK(q.stride(1) == head_dim && q.stride(0) == num_heads * head_dim)
      << "q must be laid out as contiguous [seq_len, num_heads, head_dim]";
  TVM_FFI_ICHECK(kv.stride(-1) == 1) << "kv must be contiguous at the last dimension";
  TVM_FFI_ICHECK(kv_scale.numel() == seq_len_kv) << "kv_scale.numel() must match kv.shape[0]";

  constexpr int kBlockKV = 128;
  constexpr int kBlockQ  = 2;

  auto       ceil_div          = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
  const bool compressed_logits = (max_seq_len_k > 0);
  int32_t    max_seq_kv_param  = static_cast<int32_t>(seq_len_kv);

  if (compressed_logits) {
    TVM_FFI_ICHECK(!clean_logits) << "clean_logits is not supported with compressed logits";
    TVM_FFI_ICHECK(max_seq_len_k <= seq_len_kv) << "max_seq_len_k must be <= seq_len_kv";
    max_seq_kv_param = static_cast<int32_t>(max_seq_len_k);
  }

  const int64_t out_cols_logical = static_cast<int64_t>(max_seq_kv_param);
  const int64_t out_cols_padded  = ceil_div(out_cols_logical, int64_t(kBlockKV)) * kBlockKV;

  ffi::Tensor out_pad = alloc_tensor(ffi::Shape{seq_len, out_cols_padded}, dl_float32, q.device());
  ffi::Shape  out_shape{seq_len, out_cols_logical};
  ffi::Shape  out_strides{out_cols_padded, 1};
  ffi::Tensor out = out_pad.as_strided(out_shape, out_strides);

  auto stride_q      = make_stride(static_cast<int32_t>(q.stride(1)), _1{});
  using StrideQ      = decltype(stride_q);
  auto stride_k_row  = static_cast<int32_t>(kv.stride(0));
  auto stride_k      = make_stride(stride_k_row, _1{}, int32_t(kBlockKV) * stride_k_row);
  using StrideK      = decltype(stride_k);
  auto stride_sfk    = make_stride(_1{}, int32_t(kBlockKV) * static_cast<int32_t>(kv_scale.stride(0)));
  using StrideSFK    = decltype(stride_sfk);
  auto stride_w      = make_stride(_1{}, static_cast<int32_t>(num_heads * kBlockQ));
  using StrideW      = decltype(stride_w);
  auto stride_logits = make_stride(static_cast<int32_t>(out.stride(0)), _1{});
  using StrideLogits = decltype(stride_logits);

  using Element = mutlass::float_e4m3_t;

  const int64_t q_groups = ceil_div(seq_len, int64_t(kBlockQ));
  const int     num_mps  = device_prop.multiProcessorCount;
  TVM_FFI_ICHECK(num_mps > 0) << "No available MUSA MPs";

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
        .ptr_q          = static_cast<Element const*>(q.data_ptr()),
        .stride_q       = stride_q,
        .ptr_k          = static_cast<Element const*>(kv.data_ptr()),
        .stride_k       = stride_k,
        .ptr_sfk        = static_cast<float*>(kv_scale.data_ptr()),
        .stride_sfk     = stride_sfk,
        .ptr_weights    = static_cast<float*>(weights.data_ptr()),
        .stride_weights = stride_w,
        .ptr_ks         = static_cast<int32_t*>(cu_seq_len_k_start.data_ptr()),
        .ptr_ke         = static_cast<int32_t*>(cu_seq_len_k_end.data_ptr()),
        .ptr_logits     = static_cast<float*>(out.data_ptr()),
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
      TVM_FFI_ICHECK(false) << "Unsupported num_heads=" << num_heads;
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
      TVM_FFI_ICHECK(false) << "Unsupported head_dim=" << head_dim;
    }
  };

  if (compressed_logits) {
    dispatch_head_dim(std::true_type{});
  } else {
    dispatch_head_dim(std::false_type{});
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());

  if (!compressed_logits && clean_logits) {
    mpxx_clean_logits(out,
                      cu_seq_len_k_start,
                      cu_seq_len_k_end,
                      /*next_n=*/1,
                      seq_len,
                      seq_len_kv,
                      static_cast<uint64_t>(out.stride(0)));
  }

  return out;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_paged_mqa_logits_metadata, get_paged_mqa_logits_metadata);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_paged_mqa_logits, fp8_paged_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mqa_logits, fp8_mqa_logits);
