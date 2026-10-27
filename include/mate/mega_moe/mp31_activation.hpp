#pragma once

#include <musa_burst.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>

#include <cstdint>

namespace mega_moe::mp31 {

constexpr float kActivationLog2e = 1.4426950408889634f;

__device__ __forceinline__ float4 activation_tanh_bst4(const float4 value) {
  return make_float4(__musa_tanh_f(value.x), __musa_tanh_f(value.y), __musa_tanh_f(value.z), __musa_tanh_f(value.w));
}

__device__ __forceinline__ float4 activation_rcp_bst4(const float4 value) {
#if __has_builtin(__musa_rcp_f_bst4)
  f4 result = __musa_rcp_f_bst4(*(const f4*)(&value));
  return *(const float4*)(&result);
#else
  return make_float4(__frcp_rn(value.x), __frcp_rn(value.y), __frcp_rn(value.z), __frcp_rn(value.w));
#endif
}

__device__ __forceinline__ float4 activation_sigmoid_bst4(const float4 value) {
  return activation_rcp_bst4(add(1.0f, __exp2f(mul(-kActivationLog2e, value))));
}

__device__ __forceinline__ float4 activation_sigmoid_via_tanh_bst4(const float4 value) {
  return mul(0.5f, add(1.0f, activation_tanh_bst4(mul(0.5f, value))));
}

__device__ __forceinline__ uint64_t pack_bf16x4(const mutlass::bfloat16_t x0,
                                                const mutlass::bfloat16_t x1,
                                                const mutlass::bfloat16_t x2,
                                                const mutlass::bfloat16_t x3) {
  return static_cast<uint64_t>(x0.raw()) | (static_cast<uint64_t>(x1.raw()) << 16) |
         (static_cast<uint64_t>(x2.raw()) << 32) | (static_cast<uint64_t>(x3.raw()) << 48);
}

__device__ __forceinline__ float4 unpack_bf16x4(const uint64_t bits) {
  return make_float4(static_cast<float>(mutlass::bfloat16_t::bitcast(static_cast<uint16_t>(bits))),
                     static_cast<float>(mutlass::bfloat16_t::bitcast(static_cast<uint16_t>(bits >> 16))),
                     static_cast<float>(mutlass::bfloat16_t::bitcast(static_cast<uint16_t>(bits >> 32))),
                     static_cast<float>(mutlass::bfloat16_t::bitcast(static_cast<uint16_t>(bits >> 48))));
}

}  // namespace mega_moe::mp31
