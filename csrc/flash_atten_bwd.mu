#include <musa.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>
#include <torch/torch.h>

#include <string>

#include "mate_utils.muh"
#include "mudnn_utils.hpp"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_utils.hpp"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dnn_mha_varlen_bwd(
    at::Tensor                dout,  // total_q x num_heads, x head_size
    at::Tensor                q,     // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    at::Tensor                k,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    at::Tensor                v,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    at::Tensor                out,   // total_q x num_heads x head_size
    at::Tensor                softmax_lse,
    std::optional<at::Tensor> dq_,            // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor> dk_,            // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<at::Tensor> dv_,            // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    at::Tensor                cu_seqlens_q,   // b+1
    at::Tensor                cu_seqlens_k,   // b+1
    std::optional<at::Tensor> alibi_slopes_,  // num_heads or b x num_heads
    int64_t                   max_seqlen_q,
    int64_t                   max_seqlen_k,  // max sequence length to choose the kernel
    double                    p_dropout,     // probability to drop
    double                    softmax_scale,
    bool                      zero_tensors,
    bool                      is_causal,
    int64_t                   window_size_left,
    int64_t                   window_size_right,
    double                    softcap,
    bool                      deterministic,
    std::optional<at::Tensor> gen_,
    std::optional<at::Tensor> rng_state) {
  if (is_causal) {
    window_size_right = 0;
  }
  musaDeviceProp dprops;
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, q.device().index()));
  TORCH_CHECK(dprops.major >= 3, "FlashAttention bwd only supports MP31 GPUs or newer.");

  bool is_dropout = p_dropout > 0.0;
  auto q_dtype    = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  CHECK_MUSA(q);
  CHECK_MUSA(k);
  CHECK_MUSA(v);
  CHECK_MUSA(out);
  CHECK_MUSA(dout);
  CHECK_MUSA(softmax_lse);
  CHECK_MUSA(cu_seqlens_q);
  CHECK_MUSA(cu_seqlens_k);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const auto sizes = q.sizes();

  const int total_q     = sizes[0];
  const int batch_size  = cu_seqlens_q.numel() - 1;
  const int num_heads   = sizes[1];
  const int head_size   = sizes[2];
  const int total_k     = k.size(0);
  const int num_heads_k = k.size(1);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  TORCH_CHECK(softcap == 0.f, "Softcapping does not support dropout for now");

  if (window_size_left >= max_seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k) {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  CHECK_SHAPE(out, total_q, num_heads, head_size);
  CHECK_SHAPE(dout, total_q, num_heads, head_size);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    CHECK_MUSA(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, total_q, num_heads, head_size);
  } else {
    dq = torch::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_MUSA(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
  } else {
    dk = torch::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    CHECK_MUSA(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
  } else {
    dv = torch::empty_like(v);
  }
  auto opts      = q.options();
  auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
  if (zero_tensors) {
    dq.zero_();
    dk.zero_();
    dv.zero_();
  }
  musa::dnn::ScaledDotProductAttention op;
  musa::dnn::Handle&                   h = at::GetMudnnHandle();
  op.SetEmbedDim(num_heads * head_size);
  op.SetHeadsNum(num_heads);
  op.SetMaskMode(false);
  op.SetCausal(is_causal);
  op.SetBatchSize(batch_size);
  op.SetMaxSeqlenQ(max_seqlen_q);
  op.SetMaxSeqlenK(max_seqlen_k);
  op.SetScale(softmax_scale);
  op.SetComputeMode(static_cast<musa::dnn::ScaledDotProductAttention::ComputeMode>(0));

  auto params_type = q.scalar_type();

  musa::dnn::Tensor q_dnn = make_mudnn_tensor(
      q.data_ptr(), params_type, {total_q, num_heads, head_size}, {q.stride(0), q.stride(1), q.stride(2)});
  musa::dnn::Tensor k_dnn = make_mudnn_tensor(
      k.data_ptr(), params_type, {total_k, num_heads_k, head_size}, {k.stride(0), k.stride(1), k.stride(2)});
  musa::dnn::Tensor v_dnn = make_mudnn_tensor(
      v.data_ptr(), params_type, {total_k, num_heads_k, head_size}, {v.stride(0), v.stride(1), v.stride(2)});
  musa::dnn::Tensor out_dnn = make_mudnn_tensor(
      out.data_ptr(), params_type, {total_q, num_heads, head_size}, {out.stride(0), out.stride(1), out.stride(2)});
  musa::dnn::Tensor dout_dnn = make_mudnn_tensor(
      dout.data_ptr(), params_type, {total_q, num_heads, head_size}, {dout.stride(0), dout.stride(1), dout.stride(2)});
  musa::dnn::Tensor lse_dnn =
      make_mudnn_tensor(softmax_lse.data_ptr(),
                        at::kFloat,
                        {batch_size, num_heads, max_seqlen_q, 1},
                        {softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2), 1});
  musa::dnn::Tensor dq_dnn = make_mudnn_tensor(
      dq.data_ptr(), params_type, {total_q, num_heads, head_size}, {dq.stride(0), dq.stride(1), dq.stride(2)});
  musa::dnn::Tensor dk_dnn = make_mudnn_tensor(
      dk.data_ptr(), params_type, {total_k, num_heads_k, head_size}, {dk.stride(0), dk.stride(1), dk.stride(2)});
  musa::dnn::Tensor dv_dnn = make_mudnn_tensor(
      dv.data_ptr(), params_type, {total_k, num_heads_k, head_size}, {dv.stride(0), dv.stride(1), dv.stride(2)});
  musa::dnn::Tensor cu_seqlens_q_dnn =
      make_mudnn_tensor(cu_seqlens_q.data_ptr(), at::kInt, {cu_seqlens_q.size(0)}, {1});
  musa::dnn::Tensor cu_seqlens_k_dnn =
      make_mudnn_tensor(cu_seqlens_k.data_ptr(), at::kInt, {cu_seqlens_k.size(0)}, {1});
  musa::dnn::Tensor empty_dropout_mask;
  musa::dnn::Tensor empty_mask;
  MATE_MUDNN_STATUS_CHECK(op.RunFlashVarlenBwd(h,
                                               dq_dnn,
                                               dk_dnn,
                                               dv_dnn,
                                               dout_dnn,
                                               q_dnn,
                                               k_dnn,
                                               v_dnn,
                                               empty_mask,
                                               out_dnn,
                                               lse_dnn,
                                               empty_dropout_mask,
                                               cu_seqlens_q_dnn,
                                               cu_seqlens_k_dnn,
                                               at::musa::InternalMemAlloc));
  return {dq, dk, dv, softmax_d};
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "dnn_mha_varlen_bwd("
      "Tensor dout,"
      "Tensor q,"
      "Tensor k,"
      "Tensor v,"
      "Tensor out,"
      "Tensor softmax_lse,"
      "Tensor? dq,"
      "Tensor? dk,"
      "Tensor? dv,"
      "Tensor cu_seqlens_q,"
      "Tensor cu_seqlens_k,"
      "Tensor? alibi_slopes,"
      "int max_seqlen_q,"
      "int max_seqlen_k,"
      "float p_dropout,"
      "float softmax_scale,"
      "bool zero_tensors,"
      "bool is_causal,"
      "int window_size_left,"
      "int window_size_right,"
      "float softcap,"
      "bool deterministic,"
      "Tensor? gen,"
      "Tensor? rng_state) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("dnn_mha_varlen_bwd", &dnn_mha_varlen_bwd);
}
