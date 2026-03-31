



#pragma once

#include <mutlass/fast_math.h>

#include <stdexcept>
#include <unordered_map>

#include "mp31_gemm_mubin.hpp"

#define STRING(s) #s
#define STRINGIFY(a) STRING(a)
#define REGISTER_MOE_GEMM_ASM_KERNEL(SRC_A, SRC_B, DST, GROUP_MODE, TILE_M_ID, TILE_N_ID,      \
                                       TILE_K_BYTE_ID, TransA, TransB, \
                                       HINT_A, HINT_B, FSL_A, FSL_B, KERN_NAME)  \
  {                                                                                              \
    auto gemm_id =                                                                               \
        MoeGemmAsmID{SRC_A,          SRC_B,  DST,    GROUP_MODE, TILE_M_ID, TILE_N_ID, \
                               TILE_K_BYTE_ID,TransA, TransB, HINT_A, HINT_B, FSL_A,      FSL_B};               \
\
    unsigned char* ptr = KERN_NAME;                                                              \
    std::pair<unsigned char*, const char*> p = std::make_pair(ptr, STRINGIFY(KERN_NAME));        \
    moe_gemm_asm_kern_registry[gemm_id] = p;                                                             \
  }


struct MoeGemmAsmID {

  int a_type;
  int b_type;
  int d_type;

  int mode;

  int tile_m;
  int tile_n;
  int tile_k_byte;

  int trans_a; // 0: NonTrans 1: Trans
  int trans_b; // 0: NonTrans 1: Trans

  int a_tmenc{};  // 0: None 1: enable
  int b_tmenc{};  // 0: None 1: enable

  int a_fsl;      // 0: None 1: enable
  int b_fsl{};    // 0: None 1: enable

  bool operator==(const MoeGemmAsmID& other) const {
    return a_type == other.a_type && b_type == other.b_type && d_type == other.d_type &&
           mode == other.mode && tile_m == other.tile_m && tile_n == other.tile_n &&
           tile_k_byte == other.tile_k_byte && trans_a == other.trans_a && trans_b == other.trans_b
           && a_tmenc == other.a_tmenc && b_tmenc == other.b_tmenc
           && a_fsl == other.a_fsl && b_fsl == other.b_fsl;
  }
};  // struct MoeGemmAsmID

template <>
struct std::hash<MoeGemmAsmID> {
  std::size_t operator()(const MoeGemmAsmID& id) const {

    return static_cast<std::size_t>(id.trans_a) | static_cast<std::size_t>(id.trans_b) << 1 |
           static_cast<std::size_t>(id.a_tmenc) << 2 | static_cast<std::size_t>(id.b_tmenc) << 3 |
           static_cast<std::size_t>(id.a_fsl) << 4 | static_cast<std::size_t>(id.b_fsl) << 5 |
           static_cast<std::size_t>(id.a_type) << 6 | static_cast<std::size_t>(id.b_type) << 9 | static_cast<std::size_t>(id.d_type) << 12 |
           static_cast<std::size_t>(id.mode) << 14 |
           static_cast<std::size_t>(id.tile_m) << 17 | static_cast<std::size_t>(id.tile_n) << 20 | static_cast<std::size_t>(id.tile_k_byte) << 23;
  }
};

namespace {

float membound_score(int group, int m, int n, int blk_m, int blk_n, int nr_mp) {
  int total_blk = group * mutlass::ceil_div(m, blk_m) * mutlass::ceil_div(n, blk_n);
  float score = (float)(total_blk % nr_mp) / (float)nr_mp;
  score = score == 0 ? 1.f : score;
  return score;
}

using ASMPair = std::pair<unsigned char*, const char*>;
static std::unordered_map<MoeGemmAsmID, ASMPair> moe_gemm_asm_kern_registry;
static std::once_flag moe_gemm_asm_kern_registry_flag;

static std::unordered_map<int, int> moe_gemm_asm_tile_m_to_id;
static std::unordered_map<int, int> moe_gemm_asm_tile_n_to_id;
static std::unordered_map<int, int> moe_gemm_asm_tile_k_to_id;
static std::unordered_map<at::ScalarType, int> moe_gemm_asm_src_type_to_id;
static std::unordered_map<at::ScalarType, int> moe_gemm_asm_dst_type_to_id;

inline void init_moe_gemm_asm_kern_registry() {

  std::call_once(moe_gemm_asm_kern_registry_flag, []() {


    moe_gemm_asm_src_type_to_id[at::kHalf] = 0;
    moe_gemm_asm_src_type_to_id[at::kBFloat16] = 1;
    moe_gemm_asm_src_type_to_id[at::kFloat8_e4m3fn] = 2;
    moe_gemm_asm_src_type_to_id[at::kFloat8_e5m2] = 3;

    moe_gemm_asm_dst_type_to_id[at::kBFloat16] = 0;
    moe_gemm_asm_dst_type_to_id[at::kHalf] = 1;
    moe_gemm_asm_dst_type_to_id[at::kFloat] = 2;

    moe_gemm_asm_tile_m_to_id[128] = 0;
    moe_gemm_asm_tile_m_to_id[256] = 1;
    moe_gemm_asm_tile_m_to_id[320] = 2;
    moe_gemm_asm_tile_m_to_id[384] = 3;

    moe_gemm_asm_tile_n_to_id[128] = 0;
    moe_gemm_asm_tile_n_to_id[160] = 1;
    moe_gemm_asm_tile_n_to_id[256] = 2;
    moe_gemm_asm_tile_n_to_id[384] = 3;

    moe_gemm_asm_tile_k_to_id[128] = 0;

// clang-format off

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      e5m2e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      e5m2e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      1, 0,
      e5m2e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      1, 0,
      e5m2e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      e5m2e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      e5m2e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      1, 0,
      e5m2e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      1, 0,
      e5m2e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      e4m3e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      e4m3e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      1, 0,
      e4m3e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      1, 0,
      e4m3e4m3ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_512_256x256B128_epilogue_group_block_128_fsla_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nn_tce_512_256x256B128_epilogue_group_block_128_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      e4m3e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      e4m3e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      1, 0,
      e4m3e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      1, 0,
      e4m3e5m2ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_group_group_128_fsla_persis_stage2_btmenc_scaledualbuffer
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e5m2e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e4m3bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e4m3hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e5m2bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm3_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      1, 0,
      e4m3e5m2hhssgemm_gm4_nt_tce_256_128x256B128_epilogue_group_block_128_fsla_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nn_tce_256_128x256B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e4m3bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 2, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e4m3hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e5m2bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      3, 3, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e5m2e5m2hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e4m3bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 2, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e4m3hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e5m2bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nt_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      2, 3, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      e4m3e5m2hhssgemm_gm1_nn_tce_256_256x128B128_epilogue_group_block_128_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm1_nt_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm3_nt_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm4_nt_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm1_nt_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm3_nt_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm4_nt_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      hhhhssgemm_gm1_nn_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      hhhhssgemm_gm1_nn_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nt_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      3,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm3_nt_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      4,
      1, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm4_nt_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nn_tce_512_256x256B128_epilogue_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      bf16bf16ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      bf16bf16ssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      hhssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_persis_stage2
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 2,
      6,
      1, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      hhssssgemm_gm6_tn_tce_512_256x256B128_epilogue_mldz_persis_stage2_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm1_nt_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm3_nt_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm4_nt_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm1_nt_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm3_nt_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm4_nt_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      hhhhssgemm_gm1_nn_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      hhhhssgemm_gm1_nn_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nt_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      3,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm3_nt_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      4,
      0, 2, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm4_nt_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      0, 2, 0,
      0, 0,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nn_tce_256_128x256B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 2,
      6,
      0, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      bf16bf16ssssgemm_gm6_tn_tce_256_128x256B128_epilogue_mldz_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 2,
      6,
      0, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      bf16bf16ssssgemm_gm6_tn_tce_256_128x256B128_epilogue_mldz_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 2,
      6,
      0, 2, 0,
      1, 0,
      0, 0,
      0, 0,
      hhssssgemm_gm6_tn_tce_256_128x256B128_epilogue_mldz_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 2,
      6,
      0, 2, 0,
      1, 0,
      0, 1,
      0, 0,
      hhssssgemm_gm6_tn_tce_256_128x256B128_epilogue_mldz_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      hhhhssgemm_gm1_nt_tce_256_256x128B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      hhhhssgemm_gm1_nt_tce_256_256x128B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      hhhhssgemm_gm1_nn_tce_256_256x128B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      0, 0, 1,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      hhhhssgemm_gm1_nn_tce_256_256x128B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 0, 0,
      0, 1,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nt_tce_256_256x128B128_epilogue_persis_stage4_btmenc
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 0,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_persis_stage4
    )

    REGISTER_MOE_GEMM_ASM_KERNEL(
      1, 1, 0,
      1,
      1, 0, 0,
      0, 0,
      0, 1,
      0, 0,
      bf16bf16bf16bf16ssgemm_gm1_nn_tce_256_256x128B128_epilogue_persis_stage4_btmenc
    )

// clang-format on


  }
  ); // call_once

} // init_moe_gemm_asm_kern_registry


}  // namespace

#undef STRING
#undef STRINGIFY
#undef REGISTER_MOE_GEMM_ASM_KERNEL

