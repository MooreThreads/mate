#pragma once

#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "dtype_utils.hpp"
#include "mp31_sage_attention_mubin.hpp"

#define STRING(s) #s
#define STRINGIFY(a) STRING(a)
#define REGISTER_SAGE_FA_ASM_KERNEL(IS_CAUSAL, DTYPE, HEADDIM_QK, IS_KV_CACHE, QUANT_MODE, IS_QK_INT8, KERN_NAME) \
  {                                                                                                               \
    auto           fa_id = FlashAttenAsmID{IS_CAUSAL, 0, DTYPE, HEADDIM_QK, IS_KV_CACHE, QUANT_MODE, IS_QK_INT8}; \
    unsigned char* ptr   = KERN_NAME;                                                                             \
    std::pair<unsigned char*, const char*> p = std::make_pair(ptr, STRINGIFY(KERN_NAME));                         \
    fa_asm_kern_registry[fa_id]              = p;                                                                 \
  }

struct FlashAttenAsmID {
  int is_causal;
  int is_varlen;
  int dtype;
  int headdim_qk;
  int is_kv_cache;
  int quant_mode;
  int is_qk_int8;

  bool operator==(const FlashAttenAsmID& other) const {
    return is_causal == other.is_causal && is_varlen == other.is_varlen && dtype == other.dtype &&
           headdim_qk == other.headdim_qk && is_kv_cache == other.is_kv_cache && quant_mode == other.quant_mode &&
           is_qk_int8 == other.is_qk_int8;
  }
};

template <>
struct std::hash<FlashAttenAsmID> {
  std::size_t operator()(const FlashAttenAsmID& id) const {
    return static_cast<std::size_t>(id.is_causal) | static_cast<std::size_t>(id.is_varlen) << 1 |
           static_cast<std::size_t>(id.dtype) << 2 | static_cast<std::size_t>(id.headdim_qk) << 3 |
           static_cast<std::size_t>(id.is_kv_cache) << 4 | static_cast<std::size_t>(id.quant_mode) << 5 |
           static_cast<std::size_t>(id.is_qk_int8) << 9;
  }
};

enum DataTypeEnum {
  DT_BF16     = 0,
  DT_FP16     = 1,
  DT_FP8_E4M3 = 2,
};

inline int getDataTypeEnum(DLDataType dtype) {
  switch (encode_dlpack_dtype(dtype)) {
    case bfloat16_code:
      return DataTypeEnum::DT_BF16;
    case float16_code:
      return DataTypeEnum::DT_FP16;
    case float8_e4m3fn_code:
      return DataTypeEnum::DT_FP8_E4M3;
    default:
      throw std::runtime_error("Unsupported dtype for FA ASM registry");
  }
}

namespace {

using ASMPair = std::pair<unsigned char*, const char*>;
static std::unordered_map<FlashAttenAsmID, ASMPair> fa_asm_kern_registry;
static std::once_flag                               fa_asm_kern_registry_init_flag;

inline void init_fa_asm_kern_registry() {
  std::call_once(fa_asm_kern_registry_init_flag, []() {
    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 0, 0, e4m3tce_flash_atten_quant_mode_0_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 1, 0, e4m3tce_flash_atten_quant_mode_1_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 2, 0, e4m3tce_flash_atten_quant_mode_2_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 6, 0, e4m3tce_flash_atten_quant_mode_6_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 7, 0, e4m3tce_flash_atten_quant_mode_7_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(1, 2, 0, 0, 0, 0, e4m3tce_flash_atten_quant_mode_0_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(1, 2, 0, 0, 1, 0, e4m3tce_flash_atten_quant_mode_1_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(1, 2, 0, 0, 2, 0, e4m3tce_flash_atten_quant_mode_2_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(1, 2, 0, 0, 6, 0, e4m3tce_flash_atten_quant_mode_6_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(1, 2, 0, 0, 7, 0, e4m3tce_flash_atten_quant_mode_7_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 1, 0, 0, e4m3tce_flash_atten_quant_mode_0_512_256x128x128_kvcache)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 1, 1, 0, e4m3tce_flash_atten_quant_mode_1_512_256x128x128_kvcache)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 1, 2, 0, e4m3tce_flash_atten_quant_mode_2_512_256x128x128_kvcache)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 1, 6, 0, e4m3tce_flash_atten_quant_mode_6_512_256x128x128_kvcache)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 1, 0, 0, e4m3tce_flash_atten_quant_mode_0_512_256x128x128_kvcache_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 1, 1, 0, e4m3tce_flash_atten_quant_mode_1_512_256x128x128_kvcache_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 1, 2, 0, e4m3tce_flash_atten_quant_mode_2_512_256x128x128_kvcache_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 1, 6, 0, e4m3tce_flash_atten_quant_mode_6_512_256x128x128_kvcache_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 0, 1, e4m3tce_flash_atten_qk_int8_quant_mode_0_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 1, 1, e4m3tce_flash_atten_qk_int8_quant_mode_1_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 2, 1, e4m3tce_flash_atten_qk_int8_quant_mode_2_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 6, 1, e4m3tce_flash_atten_qk_int8_quant_mode_6_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(0, 2, 0, 0, 7, 1, e4m3tce_flash_atten_qk_int8_quant_mode_7_512_256x128x128)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 0, 0, 1, e4m3tce_flash_atten_qk_int8_quant_mode_0_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 0, 1, 1, e4m3tce_flash_atten_qk_int8_quant_mode_1_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 0, 2, 1, e4m3tce_flash_atten_qk_int8_quant_mode_2_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 0, 6, 1, e4m3tce_flash_atten_qk_int8_quant_mode_6_512_256x128x128_causal_persistence)

    REGISTER_SAGE_FA_ASM_KERNEL(
        1, 2, 0, 0, 7, 1, e4m3tce_flash_atten_qk_int8_quant_mode_7_512_256x128x128_causal_persistence)
  });
}

}  // namespace

#undef STRING
#undef STRINGIFY
#undef REGISTER_SAGE_FA_ASM_KERNEL
