#include <mudnn_xmma.h>
#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <string>
#include <tuple>

#include "gemm_mudnn_utils.hpp"

struct BatchMatMulScaledParam {
  static constexpr int DIM_ABD = 3;

  TensorMajor major_a;
  TensorMajor major_b;

  MatMulScalingMode scale_mode;

  DLDataType type_a;
  DLDataType type_b;
  DLDataType type_d;

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
};

namespace {

namespace gemm_common = mate::gemm::common;
namespace gemm_mudnn  = mate::gemm::mudnn;

MatMulScalingMode get_scaling_mode_bmm_fp8(const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk) {
  const int scale_granularity_m = static_cast<int>(std::get<0>(scale_granularity_mnk));
  const int scale_granularity_n = static_cast<int>(std::get<1>(scale_granularity_mnk));
  const int scale_granularity_k = static_cast<int>(std::get<2>(scale_granularity_mnk));

  if (scale_granularity_k == -1) {
    if (scale_granularity_m == -1 && scale_granularity_n == -1) {
      return MatMulScalingMode::TENSOR_TENSOR;
    }
    if (scale_granularity_m == 1 && scale_granularity_n == -1) {
      return MatMulScalingMode::CHANNEL_TENSOR;
    }
  }

  TVM_FFI_THROW(ValueError) << "bmm_fp8 got unsupported scale_granularity_mnk";
}

void run_bmm_mudnn_lt(const BatchMatMulScaledParam&   param,
                      musa::dnn::Handle&              handle,
                      const musa::dnn::MatMulLtParam& lt_param) {
  const int64_t a_dim0 = param.major_a == TensorMajor::K ? param.m : param.k;
  const int64_t a_dim1 = param.major_a == TensorMajor::K ? param.k : param.m;
  const int64_t b_dim0 = param.major_b == TensorMajor::K ? param.n : param.k;
  const int64_t b_dim1 = param.major_b == TensorMajor::K ? param.k : param.n;
  auto          a      = make_mudnn_tensor(param.p_a,
                             param.type_a,
                                           {param.nr_batch, a_dim0, a_dim1},
                                           {param.stride_a[0], param.stride_a[1], param.stride_a[2]});
  auto          b      = make_mudnn_tensor(param.p_b,
                             param.type_b,
                                           {param.nr_batch, b_dim0, b_dim1},
                                           {param.stride_b[0], param.stride_b[1], param.stride_b[2]});
  auto          d      = make_mudnn_tensor(param.p_d,
                             param.type_d,
                                           {param.nr_batch, param.m, param.n},
                                           {param.stride_d[0], param.stride_d[1], param.stride_d[2]});
  gemm_mudnn::run_mudnn_lt_matmul(handle, d, a, b, param.major_a, param.major_b, lt_param);
}

void bmm_fp8_run(const BatchMatMulScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::Tensor scale_a;
  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    scale_a = make_mudnn_tensor(param.p_scale_a,
                                dl_float32,
                                {param.nr_batch, param.m, 1},
                                {param.stride_scale_a[0], param.stride_scale_a[1], param.stride_scale_a[2]});
  } else {
    scale_a = make_mudnn_scalar_tensor(param.p_scale_a, dl_float32);
  }
  musa::dnn::Tensor scale_b = make_mudnn_scalar_tensor(param.p_scale_b, dl_float32);

  musa::dnn::MatMulLtParam lt_param;
  MATE_MUDNN_STATUS_CHECK(lt_param.SetScale(scale_a, scale_b, musa::dnn::Tensor{}, musa::dnn::Tensor{}));
  run_bmm_mudnn_lt(param, handle, lt_param);
}

void bmm_fp16_run(const BatchMatMulScaledParam& param, musa::dnn::Handle& handle) {
  musa::dnn::MatMulLtParam lt_param;
  run_bmm_mudnn_lt(param, handle, lt_param);
}

void dispatch_bmm_backend(
    const BatchMatMulScaledParam& param, const std::string& backend, int device_id, musaStream_t stream, bool is_fp8) {
  gemm_mudnn::validate_mudnn_backend(backend, is_fp8 ? "bmm_fp8" : "bmm_fp16");
  musa::dnn::Handle handle(device_id);
  gemm_mudnn::init_mudnn_handle(handle, stream);
  if (is_fp8) {
    bmm_fp8_run(param, handle);
  } else {
    bmm_fp16_run(param, handle);
  }
}

}  // namespace

void bmm_fp8(ffi::TensorView                              a,
             ffi::TensorView                              b,
             ffi::TensorView                              scale_a,
             ffi::TensorView                              scale_b,
             ffi::TensorView                              d,
             const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
             const std::string&                           backend) {
  check_mp31(a.device(), "bmm_fp8");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(d);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, d);
  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, d);
  CHECK_CONTIGUOUS(a);
  TVM_FFI_ICHECK_EQ(b.stride(1), 1) << "b must be contiguous at k dimension";
  TVM_FFI_ICHECK_EQ(d.stride(-1), 1) << "d must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(scale_a.dtype(), dl_float32) << "scale_a must be float32";
  TVM_FFI_ICHECK_EQ(scale_b.dtype(), dl_float32) << "scale_b must be float32";

  const auto scale_mode = get_scaling_mode_bmm_fp8(scale_granularity_mnk);

  BatchMatMulScaledParam param{};
  param.major_a    = TensorMajor::K;
  param.major_b    = TensorMajor::K;
  param.scale_mode = scale_mode;
  param.type_a     = a.dtype();
  param.type_b     = b.dtype();
  param.type_d     = d.dtype();

  TVM_FFI_ICHECK(is_fp8_dtype(param.type_a)) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(param.type_b)) << "b must be fp8";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_d)) << "d must be bf16 or fp16";

  param.nr_batch = static_cast<int>(a.size(0));
  param.m        = static_cast<int>(a.size(1));
  param.n        = static_cast<int>(b.size(2));
  param.k        = static_cast<int>(a.size(2));

  TVM_FFI_ICHECK_EQ(b.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(b.size(1), param.k);
  TVM_FFI_ICHECK_EQ(d.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(d.size(1), param.m);
  TVM_FFI_ICHECK_EQ(d.size(2), param.n);

  if (gemm_common::gemm_early_return(param.m, param.n, param.k, d)) {
    return;
  }

  if (scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    CHECK_DIM(3, scale_a);
    CHECK_CONTIGUOUS(scale_a);
    TVM_FFI_ICHECK_EQ(scale_a.size(0), param.nr_batch);
    TVM_FFI_ICHECK_EQ(scale_a.size(1), param.m);
    TVM_FFI_ICHECK_EQ(scale_a.size(2), 1);
  } else {
    TVM_FFI_ICHECK_EQ(scale_a.ndim(), 0) << "scale_a must be a scalar tensor";
  }
  TVM_FFI_ICHECK_EQ(scale_b.ndim(), 0) << "scale_b must be a scalar tensor";

  param.stride_a[0] = a.stride(0);
  param.stride_a[1] = a.stride(1);
  param.stride_a[2] = a.stride(2);
  param.stride_b[0] = b.stride(0);
  param.stride_b[1] = b.stride(2);
  param.stride_b[2] = b.stride(1);
  param.stride_d[0] = d.stride(0);
  param.stride_d[1] = d.stride(1);
  param.stride_d[2] = d.stride(2);

  if (param.scale_mode == MatMulScalingMode::CHANNEL_TENSOR) {
    param.stride_scale_a[0] = scale_a.stride(0);
    param.stride_scale_a[1] = scale_a.stride(1);
    param.stride_scale_a[2] = scale_a.stride(2);
  }

  param.p_a       = a.data_ptr();
  param.p_b       = b.data_ptr();
  param.p_scale_a = scale_a.data_ptr();
  param.p_scale_b = scale_b.data_ptr();
  param.p_d       = d.data_ptr();

  ffi::MUSADeviceGuard device_guard(a.device().device_id);
  dispatch_bmm_backend(param, backend, a.device().device_id, get_stream(a.device()), true);
}

void bmm_fp16(ffi::TensorView a, ffi::TensorView b, ffi::TensorView d, const std::string& backend) {
  check_mp31(a.device(), "bmm_fp16");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(d);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, d);
  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, d);
  TVM_FFI_ICHECK_EQ(a.stride(2), 1) << "a must be contiguous at k dimension";
  TVM_FFI_ICHECK_EQ(b.stride(1), 1) << "b must be contiguous at k dimension";
  TVM_FFI_ICHECK_EQ(d.stride(-1), 1) << "d must be contiguous at the last dimension";

  BatchMatMulScaledParam param{};
  param.major_a = TensorMajor::K;
  param.major_b = TensorMajor::K;
  param.type_a  = a.dtype();
  param.type_b  = b.dtype();
  param.type_d  = d.dtype();

  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_a)) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_b)) << "b must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(param.type_d)) << "d must be bf16 or fp16";

  param.nr_batch = static_cast<int>(a.size(0));
  param.m        = static_cast<int>(a.size(1));
  param.n        = static_cast<int>(b.size(2));
  param.k        = static_cast<int>(a.size(2));

  TVM_FFI_ICHECK_EQ(b.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(b.size(1), param.k);
  TVM_FFI_ICHECK_EQ(d.size(0), param.nr_batch);
  TVM_FFI_ICHECK_EQ(d.size(1), param.m);
  TVM_FFI_ICHECK_EQ(d.size(2), param.n);

  if (gemm_common::gemm_early_return(param.m, param.n, param.k, d)) {
    return;
  }

  param.stride_a[0] = a.stride(0);
  param.stride_a[1] = a.stride(1);
  param.stride_a[2] = a.stride(2);
  param.stride_b[0] = b.stride(0);
  param.stride_b[1] = b.stride(2);
  param.stride_b[2] = b.stride(1);
  param.stride_d[0] = d.stride(0);
  param.stride_d[1] = d.stride(1);
  param.stride_d[2] = d.stride(2);

  param.p_a = a.data_ptr();
  param.p_b = b.data_ptr();
  param.p_d = d.data_ptr();

  ffi::MUSADeviceGuard device_guard(a.device().device_id);
  dispatch_bmm_backend(param, backend, a.device().device_id, get_stream(a.device()), false);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp8, bmm_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm_fp16, bmm_fp16);
