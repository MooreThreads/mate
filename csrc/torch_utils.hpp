#pragma once

#include <torch/torch.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#include <stdexcept>
#include <string>

#define CHECK_MP31(f) TORCH_CHECK(at::musa::getMUSAArch() >= 310, #f " requires MP31 or above!");
#define CHECK_DEFINED(x) TORCH_CHECK(x.defined(), "Tensor " #x " must be defined!");
#define CHECK_MUSA(x) TORCH_CHECK(at::musa::is_musa(x), "Tensor " #x " must be on MUSA!");
#define CHECK_TYPE(x, t) TORCH_CHECK(x.scalar_type() == t, "Tensor " #x " must has type: " #t);
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous!");
#define CHECK_LAST_DIM_CONTIGUOUS(x)                    \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, \
              "Tensor " #x                              \
              "must be contiguous at last "             \
              "dimension!")
#define CHECK_DIM(x, d) \
  TORCH_CHECK(x.dim() == (d), "Tensor " #x " must be a ", (d), "-D tensor! Got ", x.dim(), "-D tensor instead.");

#define CHECK_SHAPE(x, ...) \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), "Tensor " #x " must have shape (" #__VA_ARGS__ ")")

#define CHECK_TENSOR(x, d) \
  CHECK_DEFINED(x);        \
  CHECK_MUSA(x);           \
  CHECK_DIM(x, d);

#define CHECK_SCALAR_TENSOR(x) \
  CHECK_DEFINED(x);            \
  CHECK_MUSA(x);               \
  CHECK_DIM(x, 0);

#define CHECK_TENSOR_AND_CONTIGUOUS(x, d) \
  CHECK_TENSOR(x, d);                     \
  CHECK_CONTIGUOUS(x);

#define CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(x, d) \
  CHECK_TENSOR(x, d);                              \
  CHECK_LAST_DIM_CONTIGUOUS(x);

inline MUtensorDescriptorDataType torch_type_to_tme_type(const at::ScalarType scalar_type) {
  switch (scalar_type) {
    // float
    case at::kDouble: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64;
    }
    case at::kFloat: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32;
    }
    case at::kHalf: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16;
    }
    case at::kBFloat16: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
    }
    case at::kFloat8_e4m3fn: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
    }
    case at::kFloat8_e5m2: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
    }

    // int
    case at::kLong: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64;
    }
    case at::kInt: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32;
    }
    case at::kShort: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT16;
    }
    case at::kChar: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8;
    }

    // uint
    case at::kUInt64: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64;
    }
    case at::kUInt32: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
    }
    case at::kUInt16: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
    }
    case at::kByte: {
      return MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
    }

    default: {
      throw std::runtime_error("torch_type_to_tme_type() get INVALID Input tensor type!");
    }
  }
}

inline bool is_bf16_or_fp16_tensor_type(const at::ScalarType scalar_type) {
  return scalar_type == at::kBFloat16 || scalar_type == at::kHalf;
}

inline bool is_fp8_tensor_type(const at::ScalarType scalar_type) {
  return scalar_type == at::kFloat8_e4m3fn || scalar_type == at::kFloat8_e5m2;
}
