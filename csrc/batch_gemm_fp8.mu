
#include <musa.h>
#include <mudnn_xmma.h>
#include <musa_runtime.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <string>

#include "mate_utils.muh"
#include "mudnn_utils.hpp"
#include "torch_utils.hpp"

struct BatchMatMulScaledParam {
  static constexpr int DIM_ABD = 3;

  TensorMajor major_a;
  TensorMajor major_b;

  MatMulScalingMode scale_mode;

  at::ScalarType type_a;
  at::ScalarType type_b;
  at::ScalarType type_d;

  int m;
  int n;
  int k;
  int nr_batch;

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

};  // struct BatchMatMulScaledParam

namespace {

MatMulScalingMode get_scaling_mode_bmm_fp8(const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk) {
  int scale_granularity_m = std::get<0>(scale_granularity_mnk);
  int scale_granularity_n = std::get<1>(scale_granularity_mnk);
  int scale_granularity_k = std::get<2>(scale_granularity_mnk);

  if (scale_granularity_k == -1) {
    if (scale_granularity_m == -1 && scale_granularity_n == -1) {
      return MatMulScalingMode::TENSOR_TENSOR;

    } else if (scale_granularity_m == 1 && scale_granularity_n == -1) {
      return MatMulScalingMode::CHANNEL_TENSOR;
    }
  }

  TORCH_CHECK(false, "bmm_fp8 get unsupported scale_granularity_mnk!");
}

void bmm_fp8_run(const BatchMatMulScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::Tensor a;
  if (param.major_a == TensorMajor::K) {
    a = make_mudnn_tensor(param.p_a,
                          param.type_a,
                          {param.nr_batch, param.m, param.k},
                          {param.stride_a[0], param.stride_a[1], param.stride_a[2]});
  } else {
    a = make_mudnn_tensor(param.p_a,
                          param.type_a,
                          {param.nr_batch, param.k, param.m},
                          {param.stride_a[0], param.stride_a[1], param.stride_a[2]});
  }

  musa::dnn::Tensor b;
  if (param.major_b == TensorMajor::K) {
    b = make_mudnn_tensor(param.p_b,
                          param.type_b,
                          {param.nr_batch, param.n, param.k},
                          {param.stride_b[0], param.stride_b[1], param.stride_b[2]});
  } else {
    b = make_mudnn_tensor(param.p_b,
                          param.type_b,
                          {param.nr_batch, param.k, param.n},
                          {param.stride_b[0], param.stride_b[1], param.stride_b[2]});
  }

  musa::dnn::Tensor d = make_mudnn_tensor(param.p_d,
                                          param.type_d,
                                          {param.nr_batch, param.m, param.n},
                                          {param.stride_d[0], param.stride_d[1], param.stride_d[2]});

  musa::dnn::Tensor scale_a;
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    scale_a = make_mudnn_tensor(param.p_scale_a,
                                at::kFloat,
                                {param.nr_batch, param.m, 1},
                                {param.stride_scale_a[0], param.stride_scale_a[1], param.stride_scale_a[2]});
  } else {
    // TENSOR_TENSOR
    scale_a = make_mudnn_scalar_tensor(param.p_scale_a, at::kFloat);
  }
  musa::dnn::Tensor scale_b = make_mudnn_scalar_tensor(param.p_scale_b, at::kFloat);

  musa::dnn::MatMulLtParam lt_param;
  MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(scale_a, scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}));

  musa::dnn::BatchMatMul bmm;
  MATE_MUDNN_STATUS_CHECK(bmm.SetComputeMode(musa::dnn::BatchMatMul::ComputeMode::TENSOR));
  MATE_MUDNN_STATUS_CHECK(
      bmm.SetTranspose(param.major_a == TensorMajor::K ? false : true, param.major_b == TensorMajor::K ? true : false));

  MATE_MUDNN_STATUS_CHECK(bmm.RunLt(handle, d, a, b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, lt_param, nullptr));
}

void bmm_fp16_run(const BatchMatMulScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::Tensor a;
  if (param.major_a == TensorMajor::K) {
    a = make_mudnn_tensor(param.p_a,
                          param.type_a,
                          {param.nr_batch, param.m, param.k},
                          {param.stride_a[0], param.stride_a[1], param.stride_a[2]});
  } else {
    a = make_mudnn_tensor(param.p_a,
                          param.type_a,
                          {param.nr_batch, param.k, param.m},
                          {param.stride_a[0], param.stride_a[1], param.stride_a[2]});
  }

  musa::dnn::Tensor b;
  if (param.major_b == TensorMajor::K) {
    b = make_mudnn_tensor(param.p_b,
                          param.type_b,
                          {param.nr_batch, param.n, param.k},
                          {param.stride_b[0], param.stride_b[1], param.stride_b[2]});
  } else {
    b = make_mudnn_tensor(param.p_b,
                          param.type_b,
                          {param.nr_batch, param.k, param.n},
                          {param.stride_b[0], param.stride_b[1], param.stride_b[2]});
  }

  musa::dnn::Tensor d = make_mudnn_tensor(param.p_d,
                                          param.type_d,
                                          {param.nr_batch, param.m, param.n},
                                          {param.stride_d[0], param.stride_d[1], param.stride_d[2]});

  musa::dnn::MatMulLtParam lt_param;
  musa::dnn::BatchMatMul   bmm;
  MATE_MUDNN_STATUS_CHECK(bmm.SetComputeMode(musa::dnn::BatchMatMul::ComputeMode::TENSOR));
  MATE_MUDNN_STATUS_CHECK(
      bmm.SetTranspose(param.major_a == TensorMajor::K ? false : true, param.major_b == TensorMajor::K ? true : false));
  MATE_MUDNN_STATUS_CHECK(bmm.RunLt(handle, d, a, b, musa::dnn::Tensor{}, musa::dnn::Tensor{}, lt_param, nullptr));
}

}  // namespace

at::Tensor bmm_fp8(const at::Tensor&                            a,
                   const at::Tensor&                            b,
                   const at::Tensor&                            scale_a,
                   const at::Tensor&                            scale_b,
                   at::Tensor&                                  d,
                   const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
                   const std::string&                           backend) {
  // a shape: (nr_batch, m, k), k major
  // b shape: (nr_batch, k, n), k major
  // d shape: (nr_batch, m, n), n major

  CHECK_MP31("bmm_fp8");
  at::TensorArg targs[]{{a, "a", 0}, {b, "b", 1}, {scale_a, "scale_a", 2}, {scale_b, "scale_b", 3}, {d, "d", 4}};
  at::checkAllSameGPU(__func__, targs);

  auto scale_mode = get_scaling_mode_bmm_fp8(scale_granularity_mnk);

  constexpr int dim_abd = BatchMatMulScaledParam::DIM_ABD;
  CHECK_TENSOR_AND_CONTIGUOUS(a, dim_abd);
  CHECK_TENSOR(b, dim_abd);
  TORCH_CHECK(b.strides()[1] == 1, "Tensor b must be contiguous at k dimension!");

  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "bmm_fp8: scale_a must be float type");
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "bmm_fp8: scale_b must be float type");

  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(d, dim_abd);

  at::musa::OptionalMUSAGuard guard(a.device());
  BatchMatMulScaledParam      param;

  param.major_a = TensorMajor::K;  // !trans_a
  param.major_b = TensorMajor::K;  // trans_b

  param.scale_mode = scale_mode;

  param.type_a = a.scalar_type();
  param.type_b = b.scalar_type();
  param.type_d = d.scalar_type();
  TORCH_CHECK(is_fp8_tensor_type(param.type_a), "bmm_fp8: a must be fp8 type");
  TORCH_CHECK(is_fp8_tensor_type(param.type_b), "bmm_fp8: b must be fp8 type");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(param.type_d), "bmm_fp8: d must be bf16 or fp16 type");

  param.nr_batch = a.size(0);
  param.m        = a.size(1);
  param.n        = b.size(2);
  param.k        = a.size(2);

  CHECK_SHAPE(b, param.nr_batch, param.k, param.n);
  CHECK_SHAPE(d, param.nr_batch, param.m, param.n);
  if (scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    CHECK_TENSOR_AND_CONTIGUOUS(scale_a, dim_abd);
    CHECK_SHAPE(scale_a, param.nr_batch, param.m, 1);
  } else {
    CHECK_SCALAR_TENSOR(scale_a);
  }
  CHECK_SCALAR_TENSOR(scale_b);

  // TODO: maybe fail when support MN-major
  at::Tensor trans_b = b.transpose(-2, -1);
  for (int i = 0; i < dim_abd; ++i) {
    param.stride_a[i] = a.strides()[i];
    param.stride_b[i] = trans_b.strides()[i];
    param.stride_d[i] = d.strides()[i];
  }

  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    for (int i = 0; i < dim_abd; ++i) {
      param.stride_scale_a[i] = scale_a.strides()[i];
    }
  }

  param.p_a       = a.data_ptr();
  param.p_b       = trans_b.data_ptr();
  param.p_scale_a = scale_a.data_ptr();
  param.p_scale_b = scale_b.data_ptr();
  param.p_d       = d.data_ptr();

  if (backend == "auto" || backend == "mudnn") {
    musa::dnn::Handle& h = at::GetMudnnHandle();
    bmm_fp8_run(param, h);
  }

  return d;
}

at::Tensor bmm_fp16(const at::Tensor& a, const at::Tensor& b, at::Tensor& d, const std::string& backend) {
  CHECK_MP31("bmm_fp16");
  at::TensorArg targs[]{{a, "a", 0}, {b, "b", 1}, {d, "d", 4}};
  at::checkAllSameGPU(__func__, targs);

  constexpr int dim_abd = BatchMatMulScaledParam::DIM_ABD;
  CHECK_TENSOR(a, dim_abd);
  TORCH_CHECK(a.strides()[2] == 1, "Tensor b must be contigouts at k dimension!");
  CHECK_TENSOR(b, dim_abd);
  TORCH_CHECK(b.strides()[1] == 1, "Tensor b must be contiguous at k dimension!");
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(d, dim_abd);

  at::musa::OptionalMUSAGuard guard(a.device());
  BatchMatMulScaledParam      param;

  param.major_a = TensorMajor::K;  // !trans_a
  param.major_b = TensorMajor::K;  // trans_b

  param.type_a = a.scalar_type();
  param.type_b = b.scalar_type();
  param.type_d = d.scalar_type();

  TORCH_CHECK(is_bf16_or_fp16_tensor_type(param.type_a), "bmm_fp8: a must be bf16 or fp16 type");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(param.type_b), "bmm_fp8: b must be bf16 or fp16 type");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(param.type_d), "bmm_fp8: d must be bf16 or fp16 type");

  param.nr_batch = a.size(0);
  param.m        = a.size(1);
  param.n        = b.size(2);
  param.k        = a.size(2);

  CHECK_SHAPE(a, param.nr_batch, param.m, param.k);
  CHECK_SHAPE(b, param.nr_batch, param.k, param.n);
  CHECK_SHAPE(d, param.nr_batch, param.m, param.n);

  at::Tensor trans_b = b.transpose(-2, -1);
  for (int i = 0; i < dim_abd; ++i) {
    param.stride_a[i] = a.strides()[i];
    param.stride_b[i] = trans_b.strides()[i];
    param.stride_d[i] = d.strides()[i];
  }

  param.p_a = a.data_ptr();
  param.p_b = b.data_ptr();
  param.p_d = d.data_ptr();

  if (backend == "auto" || backend == "mudnn") {
    musa::dnn::Handle& h = at::GetMudnnHandle();
    bmm_fp16_run(param, h);
  }

  return d;
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "bmm_fp8("
      "Tensor a,"
      "Tensor b,"
      "Tensor scale_a,"
      "Tensor scale_b,"
      "Tensor d,"
      "(int, int, int) scale_granularity_mnk,"
      "str backend"
      ")"
      "-> Tensor");
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "bmm_fp16("
      "Tensor a,"
      "Tensor b,"
      "Tensor d,"
      "str backend"
      ")"
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("bmm_fp8", &bmm_fp8);
}
TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("bmm_fp16", &bmm_fp16);
}
