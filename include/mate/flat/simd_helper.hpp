#pragma once

#include <musa_runtime.h>
#include <mutlass/bfloat16.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <mute/arch/simd_mp31.hpp>
#include <type_traits>

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310))
#define MATE_FLAT_SIMD_MATH_ENABLED
#endif

namespace mate::flat::detail {

MUTLASS_DEVICE float fast_exp2(float x) {
  return __musa_exp2_f(x);
}
}  // namespace mate::flat::detail

namespace mate::flat::simd {

MUTLASS_DEVICE float4 fast_exp2(float4 x) {
  float4 y;
  mute::fast_exp2(y, x);
  return y;
}

MUTLASS_DEVICE float4 splat4(float x) {
  return make_float4(x, x, x, x);
}

// Vector reciprocal, 1/x per lane.
//
// The ext_vector_type division below is not a style choice. Writing this as
// make_float4(1.0f / x.x, 1.0f / x.y, 1.0f / x.z, 1.0f / x.w) puts four
// unrelated scalars in the IR, so MTGPUISelLowering's performFDIVCombine --
// which rewrites fdiv(splat 1.0, v4f32) into a v4f32 MTGPUISD::RCP -- can
// never match it, and the backend emits four scalar AF32.RCP. Expressed as a
// real <4 x float> division it selects a single AF32.RCP.BST4 instead.
//
// Measured on the KDA prepare kernel: the unrolled gate-cumsum loop went from
// 64 scalar AF32.RCP to 16 AF32.RCP.BST4 (same lane coverage), the loop body
// shrank 280 -> 233 instructions, p50 improved 3.4%, and registers, spill and
// the numerical results were all unchanged (bit-identical output).
MUTLASS_DEVICE float4 vrcp(float4 const& x) {
  using F4 = float __attribute__((ext_vector_type(4)));
  F4 xv    = *reinterpret_cast<F4 const*>(&x);
  F4 one   = {1.0f, 1.0f, 1.0f, 1.0f};
  F4 rv    = one / xv;
  return *reinterpret_cast<float4*>(&rv);
}

MUTLASS_DEVICE void vadd(float4& c, float4 const& a, float b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::add(b, a);
#else
  c = make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
#endif
}

MUTLASS_DEVICE float4 vadd(float4 const& a, float b) {
  float4 c;
  vadd(c, a, b);
  return c;
}

MUTLASS_DEVICE void vadd(float4& c, float4 const& a, float4 const& b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::add(a, b);
#else
  c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
#endif
}

MUTLASS_DEVICE float4 vadd(float4 const& a, float4 const& b) {
  float4 c;
  vadd(c, a, b);
  return c;
}

MUTLASS_DEVICE void vmul(float4& c, float4 const& a, float b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::mul(b, a);
#else
  c = make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
#endif
}

MUTLASS_DEVICE float4 vmul(float4 const& a, float b) {
  float4 c;
  vmul(c, a, b);
  return c;
}

MUTLASS_DEVICE void vmul(float4& c, float4 const& a, float4 const& b) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  c = ::mul(a, b);
#else
  c = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
#endif
}

MUTLASS_DEVICE float4 vmul(float4 const& a, float4 const& b) {
  float4 c;
  vmul(c, a, b);
  return c;
}

MUTLASS_DEVICE void vsub(float4& c, float4 const& a, float b) {
  vadd(c, a, -b);
}

MUTLASS_DEVICE float4 vsub(float4 const& a, float b) {
  float4 c;
  vsub(c, a, b);
  return c;
}

MUTLASS_DEVICE void vsub(float4& c, float4 const& a, float4 const& b) {
  vadd(c, a, vmul(b, -1.0f));
}

MUTLASS_DEVICE float4 vsub(float4 const& a, float4 const& b) {
  float4 c;
  vsub(c, a, b);
  return c;
}

MUTLASS_DEVICE void dot4(float4 const& a, float4 const& b, float& c) {
  c = fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, fmaf(a.x, b.x, c))));
}

MUTLASS_DEVICE void vfma(float4& d, float4 const& a, float b, float c) {
#if defined(MATE_FLAT_SIMD_MATH_ENABLED)
  d = ::fma(b, a, c);
#else
  d = make_float4(a.x * b + c, a.y * b + c, a.z * b + c, a.w * b + c);
#endif
}

MUTLASS_DEVICE float4 vfma(float4 const& a, float b, float c) {
  float4 d;
  vfma(d, a, b, c);
  return d;
}

}  // namespace mate::flat::simd

namespace mate::flat::simd {

// This vector surface uses `mutlass::Array` directly.  The two representations
// the KDA kernels care about -- one 32-bit register word per element versus one
// element per array slot for 16-bit types -- are chosen by `Array`'s own
// `RegisterSized` default, which reads `sizeof_bits<T> >= 32` again; a local
// packed alias would be an identity, so there is none.  The conversion helpers
// below keep the caller-facing contract: values in, one word per element out
// (`pack_from`), and back (`unpack_to`).

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
template <class T>
inline constexpr bool kBf16BurstEnabled = std::is_same_v<T, mutlass::bfloat16_t>;
#else
template <class T>
inline constexpr bool kBf16BurstEnabled = false;
#endif

// MUSA's burst converts operate on four values at a time.  Every bf16 result
// element occupies one word, which is what lets the caller issue one 16-byte
// access for every eight bf16 values.
template <int N, class SrcVec>
MUTLASS_DEVICE mutlass::Array<mutlass::bfloat16_t, N> burst_pack_bf16(SrcVec const& src) {
  static_assert(N % 4 == 0, "burst converts operate on 4-element groups");
  mutlass::Array<mutlass::bfloat16_t, N> out;
  auto const*                            in4  = reinterpret_cast<float4 const*>(&src);
  __mt_bfloat164*                        out4 = reinterpret_cast<__mt_bfloat164*>(&out);
  MUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N / 4; ++i) {
    out4[i] = __float42bfloat164_rn(in4[i]);
  }
  return out;
}

template <int N>
MUTLASS_DEVICE mutlass::Array<float, N> burst_unpack_bf16(mutlass::Array<mutlass::bfloat16_t, N> const& packed) {
  static_assert(N % 4 == 0, "burst converts operate on 4-element groups");
  mutlass::Array<float, N> out;
  auto const*              in4  = reinterpret_cast<__mt_bfloat164 const*>(&packed);
  float4*                  out4 = reinterpret_cast<float4*>(&out);
  MUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N / 4; ++i) {
    out4[i] = __bfloat1642float4(in4[i]);
  }
  return out;
}

// Values -> Dst, one word per element.  The bf16<->float case goes through the
// burst converts; all other conversions keep mutlass' generic semantics.
template <class Dst, int N, class SrcVec>
MUTLASS_DEVICE mutlass::Array<Dst, N> pack_from(SrcVec const& src) {
  using Src = typename SrcVec::value_type;
  if constexpr (std::is_same_v<Dst, Src>) {
    return src;
  } else if constexpr (kBf16BurstEnabled<Dst> && std::is_same_v<Src, float> && (N % 4 == 0)) {
    return burst_pack_bf16<N>(src);
  } else {
    return mutlass::NumericArrayConverter<Dst, Src, N, mutlass::FloatRoundStyle::round_to_nearest>{}(src);
  }
}

// Words -> the values used by the rest of the kernel.
template <class Dst, int N, class Src>
MUTLASS_DEVICE mutlass::Array<Dst, N> unpack_to(mutlass::Array<Src, N> const& packed) {
  if constexpr (std::is_same_v<Dst, Src>) {
    return packed;
  } else if constexpr (kBf16BurstEnabled<Src> && std::is_same_v<Dst, float> && (N % 4 == 0)) {
    return burst_unpack_bf16<N>(packed);
  } else {
    mutlass::NumericConverter<Dst, Src> convert_element;
    mutlass::Array<Dst, N>              out;
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      out[i] = convert_element(packed[i]);
    }
    return out;
  }
}

// Packed4 helpers widen/narrow four contiguous 16-bit elements through float4.
MUTLASS_DEVICE float4 load_packed4_as_float4(mutlass::bfloat16_t const* ptr) {
  auto packed = *reinterpret_cast<__mt_bfloat164 const*>(ptr);
  return __bfloat1642float4(packed);
}

MUTLASS_DEVICE void store_float4_to_packed4_rn(mutlass::bfloat16_t* ptr, float4 value) {
  auto packed                             = __float42bfloat164_rn(value);
  *reinterpret_cast<__mt_bfloat164*>(ptr) = packed;
}

MUTLASS_DEVICE float4 load_packed4_as_float4(mutlass::half_t const* ptr) {
  auto packed = *reinterpret_cast<__half4 const*>(ptr);
  return __half42float4(packed);
}

MUTLASS_DEVICE void store_float4_to_packed4_rn(mutlass::half_t* ptr, float4 value) {
  auto packed                      = __float42half4_rn(value);
  *reinterpret_cast<__half4*>(ptr) = packed;
}

MUTLASS_DEVICE float4 load_float4(float const* ptr) {
  return *reinterpret_cast<float4 const*>(ptr);
}

MUTLASS_DEVICE void store_float4(float* ptr, float4 value) {
  *reinterpret_cast<float4*>(ptr) = value;
}

}  // namespace mate::flat::simd
