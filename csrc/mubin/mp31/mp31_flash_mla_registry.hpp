

#pragma once

#include <mutlass/fast_math.h>

#include <stdexcept>
#include <unordered_map>

#include "mp31_flash_mla_mubin.hpp"

#define STRING(s) #s
#define STRINGIFY(a) STRING(a)
#define REGISTER_FLASH_MLA_ASM_KERNEL(IS_CAUSAL, IS_VARLEN_Q, DTYPE, KERN_NAME)                   \
  {                                                                                               \
    auto                                   mla_id = FlashMLAAsmID{IS_CAUSAL, IS_VARLEN_Q, DTYPE}; \
    unsigned char*                         ptr    = KERN_NAME;                                    \
    std::pair<unsigned char*, const char*> p      = std::make_pair(ptr, STRINGIFY(KERN_NAME));    \
    flash_mla_asm_kern_registry[mla_id]           = p;                                            \
  }

struct FlashMLAAsmID {
  int is_causal;
  int is_varlen_q;

  int dtype;

  bool operator==(const FlashMLAAsmID& other) const {
    return is_causal == other.is_causal && is_varlen_q == other.is_varlen_q && dtype == other.dtype;
  }
};

template <>
struct std::hash<FlashMLAAsmID> {
  std::size_t operator()(const FlashMLAAsmID& id) const {
    return static_cast<std::size_t>(id.is_causal) | static_cast<std::size_t>(id.is_varlen_q) << 1 |
           static_cast<std::size_t>(id.dtype) << 2;
  }
};

namespace {

using ASMPair = std::pair<unsigned char*, const char*>;
static std::unordered_map<FlashMLAAsmID, ASMPair> flash_mla_asm_kern_registry;
static std::once_flag                             flash_mla_asm_kern_registry_init_flag;

inline void init_flash_mla_asm_kern_registry() {
  std::call_once(flash_mla_asm_kern_registry_init_flag, []() {
    // clang-format off

    REGISTER_FLASH_MLA_ASM_KERNEL(
      1, 1,
      0,
      bf16tce_flash_mla_512_128x64x512_ws_causal_varlen_q
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      1, 0,
      0,
      bf16tce_flash_mla_512_128x64x512_ws_causal
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      0, 1,
      0,
      bf16tce_flash_mla_512_128x64x512_ws_varlen_q
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      0, 0,
      0,
      bf16tce_flash_mla_512_128x64x512_ws
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      1, 1,
      1,
      htce_flash_mla_512_128x64x512_ws_causal_varlen_q
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      1, 0,
      1,
      htce_flash_mla_512_128x64x512_ws_causal
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      0, 1,
      1,
      htce_flash_mla_512_128x64x512_ws_varlen_q
    )

    REGISTER_FLASH_MLA_ASM_KERNEL(
      0, 0,
      1,
      htce_flash_mla_512_128x64x512_ws
    )

    // clang-format on
  });  // call_once

}  // init_flash_mla_asm_kern_registry

}  // namespace

#undef STRING
#undef STRINGIFY
#undef REGISTER_FLASH_MLA_ASM_KERNEL
