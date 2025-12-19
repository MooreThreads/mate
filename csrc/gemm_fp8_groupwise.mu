
#include <mudnn_xmma.h>
#include <musa.h>
#include <musa_runtime.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <string>

#include "mate_utils.muh"
#include "mudnn_utils.hpp"
#include "torch_utils.hpp"

struct MatMulFP8ScaledParam {
  static constexpr int DIM_ABD = 2;

  TensorMajor major_a;
  TensorMajor major_b;
  TensorMajor major_scale_a;
  TensorMajor major_scale_b;

  MatMulScalingMode scale_mode;

  at::ScalarType type_a;
  at::ScalarType type_b;
  at::ScalarType type_d;

  int m;
  int n;
  int k;

  int scale_a_m;
  int scale_a_k;
  int scale_b_n;
  int scale_b_k;

  int scale_granularity_m;
  int scale_granularity_n;
  int scale_granularity_k;
  int quant_tile;

  int64_t stride_a[DIM_ABD];
  int64_t stride_b[DIM_ABD];
  int64_t stride_d[DIM_ABD];
  int64_t stride_scale_a[DIM_ABD];
  int64_t stride_scale_b[DIM_ABD];

  void* p_a;
  void* p_b;
  void* p_scale_a;
  void* p_scale_b;
  void* p_d;

};  // struct MatMulFP8ScaledParam

namespace {

MatMulScalingMode get_scaling_mode_gemm_fp8(const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk) {
  int scale_granularity_m = std::get<0>(scale_granularity_mnk);
  int scale_granularity_n = std::get<1>(scale_granularity_mnk);
  int scale_granularity_k = std::get<2>(scale_granularity_mnk);

  if (scale_granularity_k == -1) {
    if (scale_granularity_m == 1 && scale_granularity_n == -1) {
      return MatMulScalingMode::CHANNEL_TENSOR;

    } else if (scale_granularity_m == 1 && scale_granularity_n == 1) {
      return MatMulScalingMode::CHANNEL_CHANNEL;
    }
  }

  if (scale_granularity_k == 128 && scale_granularity_m == 1 && scale_granularity_n == 128) {
    return MatMulScalingMode::GROUP_BLOCK;
  }

  TORCH_CHECK(false, "gemm_fp8_nt_groupwise get unsupported scale_granularity_mnk!");
}

void gemm_fp8_mudnn_run(const MatMulFP8ScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::Tensor a;
  if (param.major_a == TensorMajor::K) {
    a = make_mudnn_tensor(param.p_a, param.type_a, {param.m, param.k}, {param.stride_a[0], param.stride_a[1]});
  } else {
    a = make_mudnn_tensor(param.p_a, param.type_a, {param.k, param.m}, {param.stride_a[0], param.stride_a[1]});
  }

  musa::dnn::Tensor b;
  if (param.major_b == TensorMajor::K) {
    b = make_mudnn_tensor(param.p_b, param.type_b, {param.n, param.k}, {param.stride_b[0], param.stride_b[1]});
  } else {
    b = make_mudnn_tensor(param.p_b, param.type_b, {param.k, param.n}, {param.stride_b[0], param.stride_b[1]});
  }

  musa::dnn::Tensor d =
      make_mudnn_tensor(param.p_d, param.type_d, {param.m, param.n}, {param.stride_d[0], param.stride_d[1]});

  musa::dnn::Tensor scale_a;
  if (param.major_scale_a == TensorMajor::K) {
    scale_a = make_mudnn_tensor(param.p_scale_a,
                                at::kFloat,
                                {param.scale_a_m, param.scale_a_k},
                                {param.stride_scale_a[0], param.stride_scale_a[1]});
  } else {
    scale_a = make_mudnn_tensor(param.p_scale_a,
                                at::kFloat,
                                {param.scale_a_k, param.scale_a_m},
                                {param.stride_scale_a[0], param.stride_scale_a[1]});
  }

  musa::dnn::Tensor scale_b;
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    // if b is per tensor scaling
    // scale_b is a scalar
    scale_b = make_mudnn_scalar_tensor(param.p_scale_b, at::kFloat);
  } else {
    if (param.major_scale_b == TensorMajor::K) {
      scale_b = make_mudnn_tensor(param.p_scale_b,
                                  at::kFloat,
                                  {param.scale_b_n, param.scale_b_k},
                                  {param.stride_scale_b[0], param.stride_scale_b[1]});
    } else {
      scale_b = make_mudnn_tensor(param.p_scale_b,
                                  at::kFloat,
                                  {param.scale_b_k, param.scale_b_n},
                                  {param.stride_scale_b[0], param.stride_scale_b[1]});
    }
  }

  musa::dnn::MatMulLtParam lt_param;

  const bool is_scale_mn_major = param.major_scale_a == TensorMajor::MN;
  MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(
      scale_a, scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, param.quant_tile, is_scale_mn_major));

  musa::dnn::BatchMatMul bmm;
  MATE_MUDNN_STATUS_CHECK(bmm.SetComputeMode(musa::dnn::BatchMatMul::ComputeMode::TENSOR));
  MATE_MUDNN_STATUS_CHECK(
      bmm.SetTranspose(param.major_a == TensorMajor::K ? false : true, param.major_b == TensorMajor::K ? true : false));

  MATE_MUDNN_STATUS_CHECK(bmm.RunLt(handle, d, a, b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, lt_param, nullptr));
}

}  // namespace

at::Tensor gemm_fp8_nt_groupwise(const at::Tensor&                            a,
                                 const at::Tensor&                            b,
                                 const at::Tensor&                            scale_a,
                                 const at::Tensor&                            scale_b,
                                 const std::string&                           scale_major_mode,
                                 const int64_t                                mma_sm,
                                 const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
                                 at::Tensor&                                  d,
                                 const std::string&                           backend) {
  // a shape: (m, k), k major
  // b shape: (n, k), k major
  // scale_a shape: (scale_a_m, scale_a_k), scale_a_k major
  // scale_b shape: (scale_b_n, scale_b_k), scale_b_k major
  // d shape: (m, n), n major

  CHECK_MP31("gemm_fp8_nt_groupwise");
  at::TensorArg targs[]{{a, "a", 0}, {b, "b", 1}, {scale_a, "scale_a", 2}, {scale_b, "scale_b", 3}, {d, "d", 4}};
  at::checkAllSameGPU(__func__, targs);

  constexpr int dim_abd = MatMulFP8ScaledParam::DIM_ABD;
  CHECK_TENSOR_AND_CONTIGUOUS(a, dim_abd);
  CHECK_TENSOR_AND_CONTIGUOUS(b, dim_abd);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(d, dim_abd);
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "gemm_fp8_nt_groupwise: scale_a must be float type");
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "gemm_fp8_nt_groupwise: scale_b must be float type");

  at::musa::OptionalMUSAGuard guard(a.device());
  MatMulFP8ScaledParam        param;

  param.major_a = TensorMajor::K;  // !trans_a
  param.major_b = TensorMajor::K;  // trans_b

  param.type_a = a.scalar_type();
  param.type_b = b.scalar_type();
  param.type_d = d.scalar_type();
  TORCH_CHECK(is_fp8_tensor_type(param.type_a), "gemm_fp8_nt_groupwise: a must be fp8 type");
  TORCH_CHECK(is_fp8_tensor_type(param.type_b), "gemm_fp8_nt_groupwise: b must be fp8 type");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(param.type_d), "gemm_fp8_nt_groupwise: d must be bf16 or fp16 type");

  param.m = a.size(0);
  param.n = b.size(0);
  param.k = a.size(1);

  param.scale_granularity_m = std::get<0>(scale_granularity_mnk);
  param.scale_granularity_n = std::get<1>(scale_granularity_mnk);
  param.scale_granularity_k = std::get<2>(scale_granularity_mnk);

  param.scale_mode = get_scaling_mode_gemm_fp8(scale_granularity_mnk);
  if (param.scale_mode == MatMulScalingMode::CHANNEL_CHANNEL || param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    param.quant_tile = 0;
  } else {
    //  group_block
    param.quant_tile = param.scale_granularity_k;
  }
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    CHECK_TENSOR_AND_CONTIGUOUS(scale_a, dim_abd);
    CHECK_SCALAR_TENSOR(scale_b);
  } else {
    CHECK_TENSOR_AND_CONTIGUOUS(scale_a, dim_abd);
    CHECK_TENSOR_AND_CONTIGUOUS(scale_b, dim_abd);
  }

  CHECK_SHAPE(b, param.n, param.k);
  CHECK_SHAPE(d, param.m, param.n);
  for (int i = 0; i < dim_abd; ++i) {
    param.stride_a[i]       = a.strides()[i];
    param.stride_b[i]       = b.strides()[i];
    param.stride_d[i]       = d.strides()[i];
    param.stride_scale_a[i] = scale_a.strides()[i];
  }

  if (param.scale_mode != MatMulScalingMode::CHANNEL_TENSOR) {
    for (int i = 0; i < dim_abd; ++i) {
      param.stride_scale_b[i] = scale_b.strides()[i];
    }
  }

  param.major_scale_a = scale_major_mode == "K" ? TensorMajor::K : TensorMajor::MN;
  param.major_scale_b = scale_major_mode == "K" ? TensorMajor::K : TensorMajor::MN;
  if (scale_major_mode == "K") {
    // Scale K Major

    if (param.scale_mode == MatMulScalingMode::CHANNEL_CHANNEL) {
      CHECK_SHAPE(scale_a, param.m, 1);
      CHECK_SHAPE(scale_b, param.n, 1);
    } else if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
      CHECK_SHAPE(scale_a, param.m, 1);
    } else {
      // Group_Block
      CHECK_SHAPE(scale_a, param.m, mutlass::ceil_div(param.k, param.scale_granularity_k));
      CHECK_SHAPE(scale_b,
                  mutlass::ceil_div(param.n, param.scale_granularity_n),
                  mutlass::ceil_div(param.k, param.scale_granularity_k));
    }

    param.scale_a_m = scale_a.size(0);
    param.scale_a_k = scale_a.size(1);
    if (param.scale_mode != MatMulScalingMode::CHANNEL_TENSOR) {
      param.scale_b_n = scale_b.size(0);
      param.scale_b_k = scale_b.size(1);
    } else {
      param.scale_b_n = 0;
      param.scale_b_k = 0;
    }

  } else if (scale_major_mode == "MN") {
    // Scale MN Major

    if (param.scale_mode == MatMulScalingMode::CHANNEL_CHANNEL) {
      CHECK_SHAPE(scale_a, 1, param.m);
      CHECK_SHAPE(scale_b, 1, param.n);
    } else if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
      CHECK_SHAPE(scale_a, 1, param.m);
    } else {
      // Group_Block
      CHECK_SHAPE(scale_a, mutlass::ceil_div(param.k, param.scale_granularity_k), param.m);
      CHECK_SHAPE(scale_b,
                  mutlass::ceil_div(param.k, param.scale_granularity_k),
                  mutlass::ceil_div(param.n, param.scale_granularity_n));
    }

    param.scale_a_m = scale_a.size(1);
    param.scale_a_k = scale_a.size(0);
    if (param.scale_mode != MatMulScalingMode::CHANNEL_TENSOR) {
      param.scale_b_n = scale_b.size(1);
      param.scale_b_k = scale_b.size(0);
    } else {
      param.scale_b_n = 0;
      param.scale_b_k = 0;
    }

  } else {
    throw std::runtime_error("gemm_fp8_nt_groupwise get unsupported scale_major_mode!");
  }

  param.p_a       = a.data_ptr();
  param.p_b       = b.data_ptr();
  param.p_scale_a = scale_a.data_ptr();
  param.p_scale_b = scale_b.data_ptr();
  param.p_d       = d.data_ptr();

  if (backend == "mudnn") {
    musa::dnn::Handle& h = at::GetMudnnHandle();
    gemm_fp8_mudnn_run(param, h);
  } else {
    throw std::runtime_error("gemm_fp8_nt_groupwise get unsupported backend!");
  }

  return d;
}
TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "gemm_fp8_nt_groupwise("
      "Tensor a,"
      "Tensor b,"
      "Tensor scale_a,"
      "Tensor scale_b,"
      "str scale_major_mode,"
      "int mma_sm,"
      "(int, int, int) scale_granularity_mnk,"
      "Tensor d,"
      "str backend"
      ")"
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("gemm_fp8_nt_groupwise", &gemm_fp8_nt_groupwise);
}
