

#pragma once

#include <mutlass/fast_math.h>

#include <stdexcept>
#include <unordered_map>

#include "mp31_flash_atten_mubin.hpp"

#define STRING(s) #s
#define STRINGIFY(a) STRING(a)
#define REGISTER_FA_ASM_KERNEL(IS_CAUSAL, IS_VARLEN, DTYPE, HEADDIM_QK, KERN_NAME)                           \
  {                                                                                                          \
    auto                                   fa_id = FlashAttenAsmID{IS_CAUSAL, IS_VARLEN, DTYPE, HEADDIM_QK}; \
    unsigned char*                         ptr   = KERN_NAME;                                                \
    std::pair<unsigned char*, const char*> p     = std::make_pair(ptr, STRINGIFY(KERN_NAME));                \
    fa_asm_kern_registry[fa_id]                  = p;                                                        \
  }

struct FlashAttenAsmID {
  int is_causal;
  int is_varlen;

  int dtype;

  int headdim_qk;

  bool operator==(const FlashAttenAsmID& other) const {
    return is_causal == other.is_causal && is_varlen == other.is_varlen && dtype == other.dtype &&
           headdim_qk == other.headdim_qk;
  }
};

template <>
struct std::hash<FlashAttenAsmID> {
  std::size_t operator()(const FlashAttenAsmID& id) const {
    return static_cast<std::size_t>(id.is_causal) | static_cast<std::size_t>(id.is_varlen) << 1 |
           static_cast<std::size_t>(id.dtype) << 2 | static_cast<std::size_t>(id.headdim_qk) << 3;
  }
};

namespace {

using ASMPair = std::pair<unsigned char*, const char*>;
static std::unordered_map<FlashAttenAsmID, ASMPair> fa_asm_kern_registry;
static std::once_flag                               fa_asm_kern_registry_init_flag;

inline void init_fa_asm_kern_registry() {
  std::call_once(fa_asm_kern_registry_init_flag, []() {
    // clang-format off

    REGISTER_FA_ASM_KERNEL(
      1, 1,
      0,
      0,
      bf16tce_flash_atten_512_256x128x128_varlen_causal
    )

    REGISTER_FA_ASM_KERNEL(
      1, 0,
      0,
      0,
      bf16tce_flash_atten_512_256x128x128_causal_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      0, 1,
      0,
      0,
      bf16tce_flash_atten_512_256x128x128_varlen
    )

    REGISTER_FA_ASM_KERNEL(
      0, 0,
      0,
      0,
      bf16tce_flash_atten_512_256x128x128_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      1, 1,
      1,
      0,
      htce_flash_atten_512_256x128x128_varlen_causal
    )

    REGISTER_FA_ASM_KERNEL(
      1, 0,
      1,
      0,
      htce_flash_atten_512_256x128x128_causal_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      0, 1,
      1,
      0,
      htce_flash_atten_512_256x128x128_varlen
    )

    REGISTER_FA_ASM_KERNEL(
      0, 0,
      1,
      0,
      htce_flash_atten_512_256x128x128_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      1, 1,
      0,
      1,
      bf16tce_flash_atten_512_256x64_192_128_varlen_causal
    )

    REGISTER_FA_ASM_KERNEL(
      1, 0,
      0,
      1,
      bf16tce_flash_atten_512_256x64_192_128_causal_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      0, 1,
      0,
      1,
      bf16tce_flash_atten_512_256x64_192_128_varlen_nomask
    )

    REGISTER_FA_ASM_KERNEL(
      0, 0,
      0,
      1,
      bf16tce_flash_atten_512_256x64_192_128_nomask_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      1, 1,
      1,
      1,
      htce_flash_atten_512_256x64_192_128_varlen_causal
    )

    REGISTER_FA_ASM_KERNEL(
      1, 0,
      1,
      1,
      htce_flash_atten_512_256x64_192_128_causal_persistence
    )

    REGISTER_FA_ASM_KERNEL(
      0, 1,
      1,
      1,
      htce_flash_atten_512_256x64_192_128_varlen_nomask
    )

    REGISTER_FA_ASM_KERNEL(
      0, 0,
      1,
      1,
      htce_flash_atten_512_256x64_192_128_nomask_persistence
    )

    // clang-format on
  });  // call_once

}  // init_fa_asm_kern_registry

}  // namespace

#undef STRING
#undef STRINGIFY
#undef REGISTER_FA_ASM_KERNEL
