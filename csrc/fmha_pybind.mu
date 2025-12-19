#include <torch/all.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include "flash.hpp"
#include "mate_utils.muh"
#include "static_switch.hpp"
#include "torch_utils.hpp"

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor       out,
                      void*            cu_seqlens_q_d,
                      void*            cu_seqlens_k_d,
                      void*            seqused_q,
                      void*            seqused_k,
                      void*            softmax_lse_d,
                      float            p_dropout,
                      float            softmax_scale,
                      int              window_size_left,
                      int              window_size_right,
                      int              attention_chunk,
                      const float      softcap   = 0.f,
                      const int        sm_margin = 0) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride  = q.stride(-3);
  params.k_row_stride  = k.stride(-3);
  params.v_row_stride  = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride  = v.stride(-1);
  params.o_ptr         = out.data_ptr();
  params.o_row_stride  = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  // if (cu_seqlens_q_d == nullptr) {
  params.q_batch_stride = q.stride(0);
  params.o_batch_stride = out.stride(0);
  // }
  // if (cu_seqlens_k_d == nullptr) {
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  // }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_q    = static_cast<int*>(seqused_q);
  params.seqused_k    = static_cast<int*>(seqused_k);

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b                = b;
  params.h                = h;
  params.h_k              = h_k;
  params.seqlen_q         = seqlen_q;
  params.seqlen_k         = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d                = d;
  params.d_rounded        = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.softcap       = softcap;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout           = 1.f / params.p_dropout;
  TORCH_CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
#endif

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
  params.is_local  = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  if (attention_chunk > 0) {
    window_size_left  = std::min(window_size_left, attention_chunk - 1);
    window_size_right = std::min(window_size_right, attention_chunk - 1);
  }
  params.window_size_left  = window_size_left;
  params.window_size_right = window_size_right;
  params.attention_chunk   = attention_chunk;

  TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fmha_fwd(
    at::Tensor q,                      // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    at::Tensor k,                      // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or
                                       // (num_pages, page_size, h_k, d) if there is page_table.
    at::Tensor v,                      // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or
                                       // (num_pages, page_size, h_k, dv) if there is page_table.
    std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
    std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
    std::optional<at::Tensor> q_v_,    // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    std::optional<at::Tensor> out_,    // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    std::optional<at::Tensor> cu_seqlens_q_,      // b+1
    std::optional<at::Tensor> cu_seqlens_k_,      // b+1
    std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
    std::optional<at::Tensor> seqused_q_,         // b. If given, only this many elements of each batch
                                                  // element's queries and outputs are used.
    std::optional<at::Tensor>
                           seqused_k_,  // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<at::Tensor> page_table_,      // (b_k, max_num_pages_per_seq)
    std::optional<at::Tensor> kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<at::Tensor> leftpad_k_,       // b
    std::optional<at::Tensor> rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<at::Tensor> rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<at::Tensor> seqlens_rotary_,  // b
    std::optional<at::Tensor> q_descale_,       // (b, h_k), not (b, h)
    std::optional<at::Tensor> k_descale_,       // (b, h_k)
    std::optional<at::Tensor> v_descale_,       // (b, h_k)
    std::optional<double>     softmax_scale_,
    bool                      is_causal,
    int64_t                   window_size_left,
    int64_t                   window_size_right,
    int64_t                   attention_chunk,
    double                    softcap,
    bool                      is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 &
                                                      // rotary_dim / 2
    std::optional<at::Tensor> scheduler_metadata_,    // (b + 1)
    int64_t                   num_splits,
    std::optional<bool>       pack_gqa_,
    int64_t                   sm_margin) {
  musaDeviceProp dprops;
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, q.device().index()));
  TORCH_CHECK(dprops.major >= 3, "Fmha fwd only supports MP31 GPUs or newer.");

  Flash_fwd_params params;
  auto             q_type = q.scalar_type();
  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
              "Fmha fwd only supports fp16 and bf16 data type");
  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(v);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  at::Tensor page_table;
  const bool paged_KV = page_table_.has_value();
  if (paged_KV) {
    page_table = page_table_.value();
    CHECK_MUSA(page_table);
    TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
    TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
  }

  at::Tensor cu_seqlens_q;
  bool const is_varlen_q = cu_seqlens_q_.has_value();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.value();
    CHECK_MUSA(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
    TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
  }
  at::Tensor cu_seqlens_k;
  bool const is_varlen_k = cu_seqlens_k_.has_value();
  int        max_seqlen_k;
  void*      cu_seqlens_k_ptr = nullptr;
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.value();
    CHECK_MUSA(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
    TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
    TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
    TORCH_CHECK(!kv_batch_idx_.has_value(), "If cu_seqlens_k is passed in, then page table is not supported");
    max_seqlen_k     = max_seqlen_k_.value();
    cu_seqlens_k_ptr = cu_seqlens_k.data_ptr();
  }

  auto const sizes                 = q.sizes();
  const int  batch_size            = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;
  int        seqlen_q              = !is_varlen_q ? sizes[1] : max_seqlen_q_.value();
  int        total_q               = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  int        num_heads             = q.size(-2);
  int const  head_size             = q.size(-1);
  int const  head_size_v           = v.size(-1);
  int const  max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
  int const  num_pages             = !paged_KV ? 0 : k.size(0);
  int const  page_size             = !paged_KV ? 1 : k.size(1);
  int const  seqlen_k =
      !is_varlen_k ? (!paged_KV ? k.size(1) : max_num_pages_per_seq * page_size) : max_seqlen_k_.value();
  int const total_k      = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
  int const num_heads_k  = k.size(-2);
  int const batch_size_k = !paged_KV ? (!is_varlen_k ? k.size(0) : cu_seqlens_k.size(0) - 1) : page_table.size(0);

  if (paged_KV) {
    TORCH_CHECK(page_size == 64, "page_size must be 64 for paged KV now");
  }

  double softmax_scale = 1.0 / sqrt(double(head_size));
  if (softmax_scale_.has_value()) {
    softmax_scale = softmax_scale_.value();
  }
  if (!kv_batch_idx_.has_value()) {
    TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k");
  }

  TORCH_CHECK(window_size_left == -1 && window_size_right == -1, "mha not supported Sliding-window attention yet");
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
    if ((head_size <= 64 || head_size > 128) || !paged_KV) {
      is_causal = false;
    }
  }

  if (head_size == 128 && head_size == 128) {
    TORCH_CHECK(q.stride(-2) == q.size(-1), "q must be contiguous in HD dim");
  }

  if (is_causal) {
    window_size_right = 0;
  }
  if (!is_varlen_q) {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  } else {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }
  if (!paged_KV) {
    if (!is_varlen_k) {
      CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
      CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k, total_k, num_heads_k, head_size);
      CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    }
  } else {
    CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
    CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
  }

  TORCH_CHECK(seqused_q_.has_value() == false, "mha not supported Specified seqused_q_ yet");
  if (seqused_q_.has_value()) {
    auto seqused_q = seqused_q_.value();
    TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
    CHECK_MUSA(seqused_q);
    CHECK_CONTIGUOUS(seqused_q);
    CHECK_SHAPE(seqused_q, batch_size);
  }
  if (seqused_k_.has_value()) {
    auto seqused_k = seqused_k_.value();
    TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    CHECK_MUSA(seqused_k);
    CHECK_CONTIGUOUS(seqused_k);
    CHECK_SHAPE(seqused_k, batch_size);
  }

  TORCH_CHECK(leftpad_k_.has_value() == false, "mha not supported leftpad_k_ yet");
  if (leftpad_k_.has_value()) {
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_MUSA(leftpad_k);
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  bool const is_varlen =
      is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value() || leftpad_k_.has_value();
  auto       opts     = q.options();
  auto       out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.scalar_type() == out_type,
                "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, "
                "output must have dtype BF16");
    CHECK_MUSA(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    if (!is_varlen_q) {
      CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
    } else {
      CHECK_SHAPE(out, total_q, num_heads, head_size_v);
    }
  } else {
    out = !is_varlen_q ? torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(out_type))
                       : torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type));
  }

  at::Tensor softmax_lse;
  if (!is_varlen_q) {
    softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  } else {
    softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
  }

  auto      round_multiple      = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded   = head_size;
  int const head_size_v_rounded = head_size_v;
  int const seqlen_q_rounded    = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded    = round_multiple(seqlen_k, 128);

  set_params_fprop(params,
                   batch_size,
                   seqlen_q,
                   seqlen_k,
                   seqlen_q_rounded,
                   seqlen_k_rounded,
                   num_heads,
                   num_heads_k,
                   head_size,
                   head_size_rounded,
                   q,
                   k,
                   v,
                   out,
                   !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                   !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
                   seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                   seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                   softmax_lse.data_ptr(),
                   /*p_dropout=*/0.f,
                   softmax_scale,
                   window_size_left,
                   window_size_right,
                   attention_chunk,
                   softcap,
                   sm_margin);

  params.lse_batch_stride = is_varlen_q ? 0 : softmax_lse.stride(0);
  params.lse_head_stride  = is_varlen_q ? softmax_lse.stride(0) : softmax_lse.stride(1);
  params.is_varlen_q      = is_varlen_q;
  params.is_varlen_k      = is_varlen_k;
  params.total_q          = total_q;
  params.total_k          = total_k;
  params.b_k              = batch_size_k;
  params.dv               = head_size_v;
  params.dv_rounded       = head_size_v_rounded;
  if (leftpad_k_.has_value()) {  // This needs to be set before get_pagedkv_tma
    params.leftpad_k = static_cast<int*>(leftpad_k_.value().data_ptr());
  }
  if (paged_KV) {
    params.page_table              = page_table.data_ptr<int>();
    params.page_table_batch_stride = page_table.stride(0);
  }
  params.page_size = page_size;
  params.num_pages = num_pages;

  TORCH_CHECK(k_new_.has_value() == false, "mha not supported k_new_ yet");
  if (k_new_.has_value()) {  // This needs to be set before get_pagedkv_tma
    at::Tensor k_new, v_new;
    TORCH_CHECK(v_new_.has_value(), "If k_new is supplied, v_new must also be passed in");
    TORCH_CHECK(seqused_k_.has_value(), "If k_new is supplied, seqlens_k must also be passed in");
    TORCH_CHECK(seqlen_q <= seqlen_k, "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache");
    at::Tensor cu_seqlens_k_new;
    bool const is_varlen_k_new = cu_seqlens_k_new_.has_value();
    if (is_varlen_k_new) {
      cu_seqlens_k_new = cu_seqlens_k_new_.value();
      CHECK_MUSA(cu_seqlens_k_new);
      CHECK_CONTIGUOUS(cu_seqlens_k_new);
      TORCH_CHECK(cu_seqlens_k_new.dtype() == torch::kInt32, "cu_seqlens_k_new must have dtype torch.int32");
    }
    k_new = k_new_.value();
    v_new = v_new_.value();
    TORCH_CHECK(k_new.dtype() == q_type, "k_new must have the same dtype as query");
    TORCH_CHECK(v_new.dtype() == q_type, "v_new must have the same dtype as query");
    CHECK_MUSA(k_new);
    CHECK_MUSA(v_new);
    TORCH_CHECK(k_new.stride(-1) == 1, "k_new tensor must have contiguous last dimension");
    TORCH_CHECK(v_new.stride(-1) == 1, "v_new tensor must have contiguous last dimension");
    // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when is_varlen_k_new
    int seqlen_k_new = !is_varlen_k_new ? k_new.size(1) : 0;
    int total_k_new  = !is_varlen_k_new ? batch_size * k_new.size(1) : k_new.size(0);
    if (!is_varlen_k_new) {
      CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
    }
    params.seqlen_knew = seqlen_k_new;
    params.total_knew  = total_k_new;
    params.knew_ptr    = k_new.data_ptr();
    params.vnew_ptr    = v_new.data_ptr();
    // All stride are in elements, not bytes.
    params.knew_row_stride  = k_new.stride(-3);
    params.vnew_row_stride  = v_new.stride(-3);
    params.knew_head_stride = k_new.stride(-2);
    params.vnew_head_stride = v_new.stride(-2);
    if (!is_varlen_k_new) {
      params.knew_batch_stride = k_new.stride(0);
      params.vnew_batch_stride = v_new.stride(0);
    }
    if (is_varlen_k_new) {
      params.cu_seqlens_knew = static_cast<int*>(cu_seqlens_k_new.data_ptr());
    }
  }
  TORCH_CHECK(rotary_cos_.has_value() == false, "mha not supported rotary yet");
  if (rotary_cos_.has_value()) {
    TORCH_CHECK(k_new_.has_value(),
                "If rotary cos/sin are provided, new key / value to be appended to KV cache must "
                "also be provided");
    auto rotary_cos = rotary_cos_.value();
    CHECK_MUSA(rotary_cos);
    CHECK_CONTIGUOUS(rotary_cos);
    params.rotary_dim = rotary_cos.size(1) * 2;
    TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
    TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    const int seqlen_ro = rotary_cos.size(0);
    if (paged_KV) {
      TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    }
    CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    auto rotary_sin = rotary_sin_.value();
    CHECK_MUSA(rotary_sin);
    CHECK_CONTIGUOUS(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
    TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
    params.rotary_cos_ptr        = rotary_cos.data_ptr();
    params.rotary_sin_ptr        = rotary_sin.data_ptr();
    params.is_rotary_interleaved = is_rotary_interleaved;
    if (seqlens_rotary_.has_value()) {
      at::Tensor seqlens_rotary = seqlens_rotary_.value();
      CHECK_MUSA(seqlens_rotary);
      CHECK_CONTIGUOUS(seqlens_rotary);
      TORCH_CHECK(seqlens_rotary.dtype() == torch::kInt32, "seqlens_rotary must have dtype torch.int32");
      CHECK_SHAPE(seqlens_rotary, batch_size);
      params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
    }
  } else {
    params.rotary_dim = 0;
  }

  TORCH_CHECK(kv_batch_idx_.has_value() == false, "mha not supported kv_batch_idx_ yet");
  if (kv_batch_idx_.has_value()) {
    auto kv_batch_idx = kv_batch_idx_.value();
    CHECK_MUSA(kv_batch_idx);
    CHECK_CONTIGUOUS(kv_batch_idx);
    TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
    params.kv_batch_idx = reinterpret_cast<int*>(kv_batch_idx.data_ptr());
  }

  at::Tensor out_accum, softmax_lse_accum;
  auto       outaccum_type = at::ScalarType::Float;
  //   TORCH_CHECK(int(params.num_splits) == 1, "mha does not support splits yet.");
  if (params.num_splits > 1) {
    TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
    if (!is_varlen_q) {
      out_accum =
          torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_v}, opts.dtype(outaccum_type));
      softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
      params.oaccum_batch_stride   = out_accum.stride(1);
      params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
    } else {
      out_accum         = torch::empty({params.num_splits, num_heads, total_q, head_size_v}, opts.dtype(outaccum_type));
      softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
    }
    params.is_fp32               = false;
    params.oaccum_ptr            = out_accum.data_ptr();
    params.softmax_lseaccum_ptr  = softmax_lse_accum.data_ptr();
    params.oaccum_split_stride   = out_accum.stride(0);
    params.oaccum_row_stride     = out_accum.stride(-2);
    params.oaccum_head_stride    = out_accum.stride(-3);
    params.lseaccum_split_stride = softmax_lse_accum.stride(0);
    params.lseaccum_head_stride  = softmax_lse_accum.stride(-2);
  }

  if (q_type == at::ScalarType::Float8_e4m3fn) {
    if (q_descale_.has_value()) {
      auto q_descale = q_descale_.value();
      CHECK_MUSA(q_descale);
      CHECK_SHAPE(q_descale, batch_size, num_heads_k);
      params.q_descale_ptr          = q_descale.data_ptr<float>();
      params.q_descale_batch_stride = q_descale.stride(0);
      params.q_descale_head_stride  = q_descale.stride(1);
    } else {
      params.q_descale_ptr = nullptr;
    }
    if (k_descale_.has_value()) {
      auto k_descale = k_descale_.value();
      CHECK_MUSA(k_descale);
      CHECK_SHAPE(k_descale, batch_size, num_heads_k);
      params.k_descale_ptr          = k_descale.data_ptr<float>();
      params.k_descale_batch_stride = k_descale.stride(0);
      params.k_descale_head_stride  = k_descale.stride(1);
    } else {
      params.k_descale_ptr = nullptr;
    }
    if (v_descale_.has_value()) {
      auto v_descale = v_descale_.value();
      CHECK_MUSA(v_descale);
      CHECK_SHAPE(v_descale, batch_size, num_heads_k);
      params.v_descale_ptr          = v_descale.data_ptr<float>();
      params.v_descale_batch_stride = v_descale.stride(0);
      params.v_descale_head_stride  = v_descale.stride(1);
    } else {
      params.v_descale_ptr = nullptr;
    }
  }

  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  TORCH_CHECK(k.stride(-2) == k.size(-1), "k (num_heads, head_size) dimention must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");

  const at::musa::OptionalMUSAGuard device_guard(device_of(q));

#define LAUNCH_KERNEL(QKV_TYPE, OUT_TYPE, IS_CAUSAL, IS_VARLEN, CTA_Q, CTA_KV, HEADDIM_QK, HEADDIM_V, kPagedKV) \
  [&] {                                                                                                         \
    dispatch_fmha_kernel<31,                                                                                    \
                         QKV_TYPE,                                                                              \
                         OUT_TYPE,                                                                              \
                         IS_CAUSAL,                                                                             \
                         IS_VARLEN,                                                                             \
                         CTA_Q,                                                                                 \
                         CTA_KV,                                                                                \
                         HEADDIM_QK,                                                                            \
                         HEADDIM_V,                                                                             \
                         false,                                                                                 \
                         kPagedKV>(params);                                                                     \
  }()

  FP16_SWITCH(q_type == at::ScalarType::Half, [&] {
    HEADDIM_SWITCH(head_size, head_size_v, [&] {
      BOOL_SWITCH(is_causal, kIsCausal, [&] {
        BOOL_SWITCH(is_varlen, kIsVarlen, [&] {
          BOOL_SWITCH(paged_KV, kPagedKV, [&] {
            constexpr auto cfg         = get_tile_config<kHeadSizeQK, kHeadSizeV, kPagedKV>();
            constexpr int  kCTA_Q      = cfg.kCTA_Q;
            constexpr int  kCTA_K      = cfg.kCTA_K;
            constexpr int  kHeadSizeQK = cfg.kHeadSizeQK;
            constexpr int  kHeadSizeV  = cfg.kHeadSizeV;

            if (kCTA_Q != 0) {
              LAUNCH_KERNEL(
                  elem_type, elem_type, kIsCausal, kIsVarlen, kCTA_Q, kCTA_K, kHeadSizeQK, kHeadSizeV, kPagedKV);
            } else {
              TORCH_CHECK(kCTA_Q != 0, "unsupported head size combination");
            }
          });
        });
      });
    });
  });

  //   return {out, softmax_lse};
  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "fmha_fwd("
      "Tensor q,"
      "Tensor k,"
      "Tensor v,"
      "Tensor(k_new!)? k_new = None,"
      "Tensor(v_new!)? v_new = None,"
      "Tensor? q_v = None,"
      "Tensor(out!)? out = None,"
      "Tensor? cu_seqlens_q = None,"
      "Tensor? cu_seqlens_k = None,"
      "Tensor? cu_seqlens_k_new = None,"
      "Tensor? seqused_q = None,"
      "Tensor? seqused_k = None,"
      "int? max_seqlen_q = None,"
      "int? max_seqlen_k = None,"
      "Tensor? page_table = None,"
      "Tensor? kv_batch_idx = None,"
      "Tensor? leftpad_k = None,"
      "Tensor? rotary_cos = None,"
      "Tensor? rotary_sin = None,"
      "Tensor? seqlens_rotary = None,"
      "Tensor? q_descale = None,"
      "Tensor? k_descale = None,"
      "Tensor? v_descale = None,"
      "float? softmax_scale = None,"
      "bool is_causal = False,"
      "int window_size_left = -1,"
      "int window_size_right = -1,"
      "int attention_chunk = 0,"
      "float softcap = 0.0,"
      "bool is_rotary_interleaved = False,"
      "Tensor? scheduler_metadata = None,"
      "int num_splits = 1,"
      "bool? pack_gqa = None,"
      "int sm_margin = 0) -> (Tensor(out!), Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("fmha_fwd", &fmha_fwd);
}
