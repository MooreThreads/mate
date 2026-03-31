
#pragma once

#include <mudnn.h>

#include "mate_utils.muh"

#ifndef MATE_MUDNN_STATUS_CHECK
#define MATE_MUDNN_STATUS_CHECK(error)                                                       \
  do {                                                                                       \
    const auto& e = (error);                                                                 \
    if (e != musa::dnn::Status::SUCCESS) {                                                   \
      std::cout << "MUDNN API Failed at " << __FILE__ << " Line: " << __LINE__ << std::endl; \
      exit(EXIT_FAILURE);                                                                    \
    }                                                                                        \
  } while (0)
#endif

inline musa::dnn::Tensor::Type to_mudnn_tensor_type(const at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::kFloat: {
      return musa::dnn::Tensor::Type::FLOAT;
    }
    case at::kHalf: {
      return musa::dnn::Tensor::Type::HALF;
    }
    case at::kBFloat16: {
      return musa::dnn::Tensor::Type::BFLOAT16;
    }
    case at::kFloat8_e4m3fn: {
      return musa::dnn::Tensor::Type::FP8_E4M3;
    }
    case at::kFloat8_e5m2: {
      return musa::dnn::Tensor::Type::FP8_E5M2;
    }
    case at::kInt: {
      return musa::dnn::Tensor::Type::INT32;
    }
    default: {
      throw std::runtime_error("to_mudnn_tensor_type get unsupported data type!");
    }
  }
}

inline auto make_mudnn_tensor(void*                          ptr,
                              const at::ScalarType           datatype,
                              std::initializer_list<int64_t> shape,
                              std::initializer_list<int64_t> stride) {
  musa::dnn::Tensor tensor;

  tensor.SetType(to_mudnn_tensor_type(datatype));
  tensor.SetAddr(ptr);
  tensor.SetNdInfo(shape, stride);

  return tensor;
}

inline auto make_mudnn_scalar_tensor(void* ptr, const at::ScalarType datatype) {
  musa::dnn::Tensor tensor;

  tensor.SetType(to_mudnn_tensor_type(datatype));
  tensor.SetAddr(ptr);
  tensor.SetNdInfo({1}, {1});

  return tensor;
}
