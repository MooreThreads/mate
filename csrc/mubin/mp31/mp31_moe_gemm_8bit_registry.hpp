

#pragma once

#include <mutlass/fast_math.h>

#include <stdexcept>
#include <unordered_map>

#include "mp31_gemm_mubin.hpp"

#define STRING(s) #s
#define STRINGIFY(a) STRING(a)
#define REGISTER_MOE_GEMM_ASM_KERNEL(                                                                             \
    SRC_A, SRC_B, DST, GROUP_MODE, TILE_M_ID, TILE_N_ID, TILE_K_BYTE_ID, HINT_A, HINT_B, FSL_A, FSL_B, KERN_NAME) \
  {                                                                                                               \
    auto gemm_id = MoeGemm8bitAsmID{                                                                              \
        SRC_A, SRC_B, DST, GROUP_MODE, TILE_M_ID, TILE_N_ID, TILE_K_BYTE_ID, HINT_A, HINT_B, FSL_A, FSL_B};       \
    unsigned char*                         ptr = KERN_NAME;                                                       \
    std::pair<unsigned char*, const char*> p   = std::make_pair(ptr, STRINGIFY(KERN_NAME));                       \
    moe_gemm_asm_kern_registry[gemm_id]        = p;                                                               \
  }

struct MoeGemm8bitAsmID {
  int a_type;  // 0: e4m3 1: e5m2
  int b_type;  // 0: e4m3 1: e5m2
  int d_type;  // 0: bf16 1: half

  int mode;  // 0: ragged 1: masked

  int tile_m;       // 0: 128 1: 256
  int tile_n;       // 0: 128 1: 160 2: 256
  int tile_k_byte;  // 0: 128 1: 256

  int a_tmenc{};  // 0: None 1: enable
  int b_tmenc{};  // 0: None 1: enable
  int a_fsl;      // 0: None 1: enable
  int b_fsl{};    // 0: None 1: enable

  bool operator==(const MoeGemm8bitAsmID& other) const {
    return a_type == other.a_type && b_type == other.b_type && d_type == other.d_type && mode == other.mode &&
           tile_m == other.tile_m && tile_n == other.tile_n && tile_k_byte == other.tile_k_byte &&
           a_tmenc == other.a_tmenc && b_tmenc == other.b_tmenc && a_fsl == other.a_fsl && b_fsl == other.b_fsl;
  }
};  // struct MoeGemm8bitAsmID

template <>
struct std::hash<MoeGemm8bitAsmID> {
  std::size_t operator()(const MoeGemm8bitAsmID& id) const {
    return static_cast<std::size_t>(id.a_type) | static_cast<std::size_t>(id.b_type) << 1 |
           static_cast<std::size_t>(id.d_type) << 2 | static_cast<std::size_t>(id.mode) << 3 |
           static_cast<std::size_t>(id.a_tmenc) << 4 | static_cast<std::size_t>(id.b_tmenc) << 5 |
           static_cast<std::size_t>(id.a_fsl) << 6 | static_cast<std::size_t>(id.b_fsl) << 7 |
           static_cast<std::size_t>(id.tile_m) << 9 | static_cast<std::size_t>(id.tile_n) << 11 |
           static_cast<std::size_t>(id.tile_k_byte) << 13;
  }
};

namespace {

float membound_score(int group, int m, int n, int blk_m, int blk_n, int nr_mp) {
  int   total_blk = group * mutlass::ceil_div(m, blk_m) * mutlass::ceil_div(n, blk_n);
  float score     = (float)(total_blk % nr_mp) / (float)nr_mp;
  score           = score == 0 ? 1.f : score;
  return score;
}

using ASMPair = std::pair<unsigned char*, const char*>;
static std::unordered_map<MoeGemm8bitAsmID, ASMPair> moe_gemm_asm_kern_registry;
static std::once_flag                                moe_gemm_asm_kern_registry_flag;

inline void init_moe_gemm_asm_kern_registry() {
  std::call_once(moe_gemm_asm_kern_registry_flag, []() {
    // clang-format off

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      0,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      0,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      1, 2, 0,
      0, 0,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      1, 2, 0,
      0, 1,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 0,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 0, 1,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 1,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 0,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 0,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      0,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      0,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      0, 2, 0,
      0, 0,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 1, 1,
      1,
      0, 2, 0,
      0, 1,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    // clang-format on
  });  // call_once

}  // init_moe_gemm_asm_kern_registry

}  // namespace

#undef STRING
#undef STRINGIFY
#undef REGISTER_MOE_GEMM_ASM_KERNEL
