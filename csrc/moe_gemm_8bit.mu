#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mutlass/fast_math.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <iostream>
#include <mute/algorithm/tuple_algorithms.hpp>
#include <mute/arch/copy_mp31_desc.hpp>
#include <tuple>
#include <vector>

#include "asm_common.hpp"
#include "mate_utils.muh"
#include "mubin/mp31/mp31_moe_gemm_8bit_registry.hpp"
#include "torch_utils.hpp"

namespace mate::moe_gemm {

enum class MoeGemmMode {
  Ragged = 0,
  Masked,
};  // enum class MoeGemmMode

struct MoeGemm8bitArgs {
  at::ScalarType type_a;
  at::ScalarType type_b;
  at::ScalarType type_d;

  // for deepgemm contiguous, m == sum(align(group_m[i]))
  // for deepgemm masked,     m == align(max_m) * num_expert
  int m;
  int n;
  int k;
  int num_expert;

  int quant_tile;
  int alignment_m;
  int expected_m;  // only for masked
  int total_mp_count;
  int target_mp_count;  // used by masked group gemm kernel selection

  size_t stride_m_a;
  size_t stride_batch_a;
  size_t stride_n_b;
  size_t stride_batch_b;
  size_t stride_m_out;
  size_t stride_batch_out;

  TensorMajor major_scale_a;
  TensorMajor major_scale_b;

  int scale_a_m;
  int scale_a_k;
  int scale_a_nr_elem;
  int scale_b_n;
  int scale_b_k;
  int scale_b_nr_elem;

  // deivce memory ptr
  void* p_a;
  void* p_b;
  void* p_d;

  void* p_scale_a;
  void* p_scale_b;

  void* p_m_indices;
  void* p_signal;

};  // struct MoeGemm8bitArgs
namespace mubin {

template <MoeGemmMode mode>
struct MoeGemm8bitAsmDispatcher {
  MoeGemm8bitAsmDispatcher() {
    // register all moe gemm mubin kernels
    // do only once
    init_moe_gemm_asm_kern_registry();
  }

  struct Config {
    int nr_thr = 512;
    int tile_m = 256;
    int tile_n = 256;
    int tile_k = 128;

    int tile_a_dim0 = 128;
    int tile_a_dim1 = 128;
    int tile_a_dim2 = 1;

    int tile_b_dim0 = 128;
    int tile_b_dim1 = 128;
    int tile_b_dim2 = 1;

    int tile_c_dim0 = 128;
    int tile_c_dim1 = 4;
    int tile_c_dim2 = 1;

    int tile_b1_dim0 = 160 % 128;
    int tile_b1_dim1 = 128;
    int tile_b1_dim2 = 1;

    int scale_a_mode = 4;
    int scale_b_mode = 3;
    int scale_d_mode = 0;

    int macro_tile_x   = 4;
    int switch_swizzle = 0;

    int blk_per_mp = 1;

    MUfunction* asm_func;

  };  // struct Config

  Config get_kernel_config(const MoeGemm8bitArgs& args) {
    Config config;

    int b_tmenc_mode = 0;

    if constexpr (mode == MoeGemmMode::Ragged) {
      if (args.alignment_m == 256) {
        config.tile_m = 256;
        config.nr_thr = 512;
      } else if (args.alignment_m == 128) {
        config.tile_m = 128;
        config.nr_thr = 256;
      }
      float m_per_group = float(args.m) / args.num_expert;

      if (m_per_group <= 192) {
        // use optimized version
        b_tmenc_mode = 1;
      }
    } else if constexpr (mode == MoeGemmMode::Masked) {
      int select_m = -1;

      if (args.alignment_m == 0) {
        select_m = args.expected_m <= 128 ? 128 : 256;
      } else {
        select_m = args.alignment_m;
      }

      if (select_m == 256 && args.expected_m <= 256) {
        // use optimized version
        b_tmenc_mode = 1;
      }

      constexpr int select_n = -1;
      if (select_m == 128) {
        b_tmenc_mode = 1;
      }

      if (select_m == 256) {
        config.tile_m = 256;
        config.nr_thr = 512;
      } else if (select_m == 128 && select_n == 160) {
        config.tile_m     = 128;
        config.tile_n     = 160;
        config.nr_thr     = 256;
        config.blk_per_mp = 2;
      } else if (select_m == 128) {
        config.tile_m = 128;
        config.nr_thr = 256;
      }
    } else {
      throw std::runtime_error("MoeGemm8bitAsmDispatcher mode not supported!");
    }

    MoeGemm8bitAsmID id;
    id.a_type = args.type_a == at::kFloat8_e4m3fn ? 0 : 1;
    id.b_type = args.type_b == at::kFloat8_e4m3fn ? 0 : 1;
    id.d_type = args.type_d == at::kBFloat16 ? 0 : 1;
    id.mode   = static_cast<int>(mode);

    id.tile_m = config.tile_m == 128 ? 0 : 1;
    if (config.tile_n == 128) {
      id.tile_n = 0;
    } else if (config.tile_n == 160) {
      id.tile_n = 1;
    } else if (config.tile_n == 256) {
      id.tile_n = 2;
    }
    id.tile_k_byte = config.tile_k == 128 ? 0 : 1;

    id.a_fsl   = static_cast<int>(args.major_scale_a == TensorMajor::MN);
    id.b_tmenc = b_tmenc_mode;

    // checking if we have this kernel
    auto res = kernel_map.find(id);
    if (res != kernel_map.end()) {
      // the kernel is already called once
      // we don't need to load it again

      config.asm_func = &res->second.asm_func;

      return config;

    } else {
      auto asm_meta = moe_gemm_asm_kern_registry.find(id);
      if (asm_meta == moe_gemm_asm_kern_registry.end()) {
        throw std::runtime_error("MoeGemm8bitAsmDispatcher() mubin kernel not found!");
      }

      // Call the kernel first time
      // we need to load it and cache it

      auto [curr_asm_meta, success] = kernel_map.try_emplace(id, asm_meta->second.first, asm_meta->second.second);

      config.asm_func = &curr_asm_meta->second.asm_func;

      return config;
    }
  }

  std::unordered_map<MoeGemm8bitAsmID, MateAsmKernel> kernel_map;

};  // struct MoeGemm8bitAsmDispatcher

template <MoeGemmMode mode, class Dispatcher = MoeGemm8bitAsmDispatcher<mode>>
class MoeGemm8bitAsmKernel {
 public:
  using Config       = typename Dispatcher::Config;
  using LaunchConfig = MUlaunchConfig;

  struct Params {
    void* output_ptr{};
    void* amax_ptr{};

    MUtensorDescriptor a_desc{};
    MUtensorDescriptor b_desc{};
    MUtensorDescriptor c_desc{};

    mute::RobustReg robust_input_bias{};
    mute::RobustReg robust_input_z{};
    mute::RobustReg robust_input_a_scale{};
    mute::RobustReg robust_input_b_scale{};
    mute::RobustReg robust_input_d_scale{};
    mute::RobustReg robust_group_idx{};
    mute::RobustReg robust_group_scale_idx{};

    void*    k_part_buffer{};
    int32_t* k_part_flag{};

    int32_t* group_scale_a_robust_vec{};
    int32_t* group_scale_b_robust_vec{};

    int32_t* sync_out_flag_ptr{};

    float rcp_scale_o{1.0f};

    int32_t mask_blky_per_group{};

    float alpha{1.0f};
    float beta{0.0f};
    float gamma{0.0f};

    int32_t scale_a_mode{};
    int32_t scale_b_mode{};
    int32_t scale_d_mode{};

    int32_t scale_a_m{};
    int32_t scale_a_k{};
    int32_t scale_b_k{};
    int32_t scale_b_n{};

    int32_t batch_stride_scale_a{};
    int32_t batch_stride_scale_b{};

    int32_t quant_tile{};
    int32_t amax_mode{};
    int32_t act_mode{};

    int32_t m{};
    int32_t n{};
    int32_t k{};
    int32_t num_expert{};

    int32_t ldd{};
    int32_t ldc{};
    int32_t total_tile{};

    int32_t mask_m_max{};
    int32_t macro_bidx_max{};

    int64_t batch_stride_d{};
    int64_t batch_stride_c{};

    int32_t k_part{};
    int32_t k_part_block{};
    int32_t ldkpart{};
    int64_t kpart_batch_stride_c{};

    int32_t fast_xytile_ori{};
    int32_t fast_xytile_mul{};
    int32_t fast_xytile_sft{};

    int32_t fast_batch_ori{};
    int32_t fast_batch_mul{};
    int32_t fast_batch_sft{};

    int32_t fast_nr_block_group_ori{};
    int32_t fast_nr_block_group_mul{};
    int32_t fast_nr_block_group_sft{};

    int32_t fast_gridx_ori{};
    int32_t fast_gridx_mul{};
    int32_t fast_gridx_sft{};

    int32_t cluster_dimx{0};
    int32_t macro_bid_max{};
    int32_t switch_swizzle{};
    int32_t max_tile_y{};

    int32_t  fast_macro_tile_x_ori{};
    uint32_t fast_macro_tile_x_mul{};
    uint32_t fast_macro_tile_x_sft{};

    int32_t  fast_res_tile_x_ori{};
    uint32_t fast_res_tile_x_mul{};
    uint32_t fast_res_tile_x_sft{};

    int32_t  fast_macro_tile_size_ori{};
    uint32_t fast_macro_tile_size_mul{};
    uint32_t fast_macro_tile_size_sft{};

    int32_t tile_a_dim0{};
    int32_t tile_a_dim1{};
    int32_t tile_a_dim2{};
    int32_t tile_b_dim0{};
    int32_t tile_b_dim1{};
    int32_t tile_b_dim2{};
    int32_t tile_c_dim0{};
    int32_t tile_c_dim1{};
    int32_t tile_c_dim2{};
    int32_t tile_b1_dim0{};
    int32_t tile_b1_dim1{};
    int32_t tile_b1_dim2{};
  };  // struct Params

  MoeGemm8bitAsmKernel(const Params& in_params, const Config& in_config, const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const MoeGemm8bitArgs& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    constexpr int batch = 1;
    if constexpr (mode == MoeGemmMode::Ragged) {
      TmeDesc tensor_desc_a(mute::make_tuple(args.k, args.m, batch),
                            mute::make_tuple(args.stride_m_a, args.stride_batch_a),
                            args.p_a,
                            MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8);
      TmeDesc tensor_desc_b(mute::make_tuple(args.k, args.n, args.num_expert),
                            mute::make_tuple(args.stride_n_b, args.stride_batch_b),
                            args.p_b,
                            MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8);

      int grid_y         = mutlass::ceil_div(args.m, config.tile_m);
      int grid_x         = mutlass::ceil_div(args.n, config.tile_n);
      int grid_z         = 1;
      int gridy_in_group = grid_y;

      int swizzle_dim0 = 0;
      int swizzle_dim1 = 0;
      if (config.switch_swizzle == 0) {
        swizzle_dim0 = grid_x;
        swizzle_dim1 = grid_y;
      } else {
        swizzle_dim0 = grid_y;
        swizzle_dim1 = grid_x;
      }
      int macro_tile_size = config.macro_tile_x * swizzle_dim1;
      int res_tile_x      = swizzle_dim0 % config.macro_tile_x;

      auto fast_macro_tile_x    = mutlass::FastDivmod(config.macro_tile_x);
      auto fast_res_tile_x      = get_fast_div_mod(res_tile_x);
      auto fast_macro_tile_size = mutlass::FastDivmod(macro_tile_size);
      auto fast_batch           = mutlass::FastDivmod(batch);
      auto fast_xytile          = mutlass::FastDivmod(grid_y * grid_x);
      auto fast_gridx           = mutlass::FastDivmod(grid_x);
      auto fast_nr_block_group  = mutlass::FastDivmod(gridy_in_group * grid_x);

      params.output_ptr = args.p_d;

      params.a_desc = tensor_desc_a.desc;
      params.b_desc = tensor_desc_b.desc;

      params.robust_input_a_scale =
          mute::make_robust_desc(static_cast<float*>(args.p_scale_a), args.scale_a_nr_elem).reg;
      params.robust_input_b_scale =
          mute::make_robust_desc(static_cast<float*>(args.p_scale_b), args.scale_b_nr_elem).reg;

      params.robust_group_idx = mute::make_robust_desc(static_cast<int*>(args.p_m_indices), args.m).reg;

      params.scale_a_mode = config.scale_a_mode;
      params.scale_b_mode = config.scale_b_mode;
      params.scale_d_mode = config.scale_d_mode;

      params.scale_a_m            = args.scale_a_m;
      params.scale_a_k            = args.scale_a_k;
      params.scale_b_k            = args.scale_b_k;
      params.scale_b_n            = args.scale_b_n;
      params.batch_stride_scale_a = params.scale_a_m * params.scale_a_k;
      params.batch_stride_scale_b = params.scale_b_k * params.scale_b_n;

      params.m          = args.m;
      params.n          = args.n;
      params.k          = args.k;
      params.num_expert = args.num_expert;

      params.ldd = args.n;
      params.ldc = args.n;

      params.total_tile     = grid_x * grid_y * grid_z;
      params.mask_m_max     = args.m / args.num_expert;
      params.macro_bidx_max = swizzle_dim0 / config.macro_tile_x * config.macro_tile_x;
      params.macro_bid_max  = swizzle_dim0 / config.macro_tile_x * config.macro_tile_x * swizzle_dim1;

      params.batch_stride_d = args.stride_batch_out;
      params.batch_stride_c = params.batch_stride_d;

      params.k_part       = 0;
      params.k_part_block = args.k;
      params.ldkpart      = mutlass::ceil_div(args.n, config.tile_n) * config.tile_n;
      params.kpart_batch_stride_c =
          static_cast<size_t>(mutlass::ceil_div(args.m, config.tile_m)) * config.tile_m * params.ldkpart;

      params.fast_xytile_ori = fast_xytile.divisor;
      params.fast_xytile_mul = fast_xytile.multiplier;
      params.fast_xytile_sft = fast_xytile.shift_right;

      params.fast_batch_ori = fast_batch.divisor;
      params.fast_batch_mul = fast_batch.multiplier;
      params.fast_batch_sft = fast_batch.shift_right;

      params.fast_nr_block_group_ori = fast_nr_block_group.divisor;
      params.fast_nr_block_group_mul = fast_nr_block_group.multiplier;
      params.fast_nr_block_group_sft = fast_nr_block_group.shift_right;

      params.fast_gridx_ori = fast_gridx.divisor;
      params.fast_gridx_mul = fast_gridx.multiplier;
      params.fast_gridx_sft = fast_gridx.shift_right;

      params.max_tile_y = swizzle_dim1 - 1;
      params.quant_tile = args.quant_tile;

      params.fast_macro_tile_x_ori = fast_macro_tile_x.divisor;
      params.fast_macro_tile_x_mul = fast_macro_tile_x.multiplier;
      params.fast_macro_tile_x_sft = fast_macro_tile_x.shift_right;

      params.fast_res_tile_x_ori = fast_res_tile_x.divisor;
      params.fast_res_tile_x_mul = fast_res_tile_x.multiplier;
      params.fast_res_tile_x_sft = fast_res_tile_x.shift_right;

      params.fast_macro_tile_size_ori = fast_macro_tile_size.divisor;
      params.fast_macro_tile_size_mul = fast_macro_tile_size.multiplier;
      params.fast_macro_tile_size_sft = fast_macro_tile_size.shift_right;

      params.tile_a_dim0 = config.tile_a_dim0;
      params.tile_a_dim1 = config.tile_a_dim1;
      params.tile_a_dim2 = config.tile_a_dim2;
      params.tile_b_dim0 = config.tile_b_dim0;
      params.tile_b_dim1 = config.tile_b_dim1;
      params.tile_b_dim2 = config.tile_b_dim2;
      params.tile_c_dim0 = config.tile_c_dim0;
      params.tile_c_dim1 = config.tile_c_dim1;
      params.tile_c_dim2 = config.tile_c_dim2;

      launch_config.blockDimX      = config.nr_thr;
      launch_config.blockDimY      = 1;
      launch_config.blockDimZ      = 1;
      launch_config.gridDimX       = std::min(args.total_mp_count, params.total_tile);
      launch_config.gridDimY       = 1;
      launch_config.gridDimZ       = 1;
      launch_config.hStream        = reinterpret_cast<MUstream>(stream);
      launch_config.sharedMemBytes = 0;
      launch_config.attrs          = NULL;
      launch_config.numAttrs       = 0;

    } else if constexpr (mode == MoeGemmMode::Masked) {
      int max_m = args.m / args.num_expert;

      TmeDesc tensor_desc_a(mute::make_tuple(args.k, args.m, batch),
                            mute::make_tuple(args.stride_m_a, args.stride_batch_a),
                            args.p_a,
                            MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8);
      TmeDesc tensor_desc_b(mute::make_tuple(args.k, args.n, args.num_expert),
                            mute::make_tuple(args.stride_n_b, args.stride_batch_b),
                            args.p_b,
                            MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8);

      int gridy_in_group = mutlass::ceil_div(max_m, config.tile_m);
      int grid_x         = mutlass::ceil_div(args.n, config.tile_n);
      int grid_y         = gridy_in_group * args.num_expert;
      int grid_z         = 1;

      int swizzle_dim0 = 0;
      int swizzle_dim1 = 0;
      if (config.switch_swizzle == 0) {
        swizzle_dim0 = grid_x;
        swizzle_dim1 = grid_y;
      } else {
        swizzle_dim0 = grid_y;
        swizzle_dim1 = grid_x;
      }
      int macro_tile_size = config.macro_tile_x * swizzle_dim1;
      int res_tile_x      = swizzle_dim0 % config.macro_tile_x;

      auto fast_macro_tile_x    = mutlass::FastDivmod(config.macro_tile_x);
      auto fast_res_tile_x      = get_fast_div_mod(res_tile_x);
      auto fast_macro_tile_size = mutlass::FastDivmod(macro_tile_size);
      auto fast_batch           = mutlass::FastDivmod(batch);
      auto fast_xytile          = mutlass::FastDivmod(grid_y * grid_x);
      auto fast_gridx           = mutlass::FastDivmod(grid_x);
      auto fast_nr_block_group  = mutlass::FastDivmod(gridy_in_group * grid_x);

      params.output_ptr = args.p_d;

      params.a_desc = tensor_desc_a.desc;
      params.b_desc = tensor_desc_b.desc;

      params.robust_input_a_scale =
          mute::make_robust_desc(static_cast<float*>(args.p_scale_a), args.scale_a_nr_elem).reg;
      params.robust_input_b_scale =
          mute::make_robust_desc(static_cast<float*>(args.p_scale_b), args.scale_b_nr_elem).reg;

      params.robust_group_idx = mute::make_robust_desc(static_cast<int*>(args.p_m_indices), args.num_expert).reg;

      params.sync_out_flag_ptr   = static_cast<int32_t*>(args.p_signal);
      params.mask_blky_per_group = gridy_in_group;

      params.scale_a_mode = config.scale_a_mode;
      params.scale_b_mode = config.scale_b_mode;
      params.scale_d_mode = config.scale_d_mode;

      params.scale_a_m            = args.scale_a_m;
      params.scale_a_k            = args.scale_a_k;
      params.scale_b_k            = args.scale_b_k;
      params.scale_b_n            = args.scale_b_n;
      params.batch_stride_scale_a = params.scale_a_m * params.scale_a_k;
      params.batch_stride_scale_b = params.scale_b_k * params.scale_b_n;

      params.m          = args.m;
      params.n          = args.n;
      params.k          = args.k;
      params.num_expert = args.num_expert;

      params.ldd = args.n;
      params.ldc = args.n;

      params.total_tile     = grid_x * grid_y * grid_z;
      params.mask_m_max     = args.m / args.num_expert;
      params.macro_bidx_max = swizzle_dim0 / config.macro_tile_x * config.macro_tile_x;
      params.macro_bid_max  = swizzle_dim0 / config.macro_tile_x * config.macro_tile_x * swizzle_dim1;

      params.batch_stride_d = args.stride_batch_out;
      params.batch_stride_c = params.batch_stride_d;

      params.k_part       = 0;
      params.k_part_block = args.k;
      params.ldkpart      = mutlass::ceil_div(args.n, config.tile_n) * config.tile_n;
      params.kpart_batch_stride_c =
          static_cast<size_t>(mutlass::ceil_div(args.m, config.tile_m)) * config.tile_m * params.ldkpart;

      params.fast_xytile_ori = fast_xytile.divisor;
      params.fast_xytile_mul = fast_xytile.multiplier;
      params.fast_xytile_sft = fast_xytile.shift_right;

      params.fast_batch_ori = fast_batch.divisor;
      params.fast_batch_mul = fast_batch.multiplier;
      params.fast_batch_sft = fast_batch.shift_right;

      params.fast_nr_block_group_ori = fast_nr_block_group.divisor;
      params.fast_nr_block_group_mul = fast_nr_block_group.multiplier;
      params.fast_nr_block_group_sft = fast_nr_block_group.shift_right;

      params.fast_gridx_ori = fast_gridx.divisor;
      params.fast_gridx_mul = fast_gridx.multiplier;
      params.fast_gridx_sft = fast_gridx.shift_right;

      params.max_tile_y = swizzle_dim1 - 1;
      params.quant_tile = args.quant_tile;

      params.fast_macro_tile_x_ori = fast_macro_tile_x.divisor;
      params.fast_macro_tile_x_mul = fast_macro_tile_x.multiplier;
      params.fast_macro_tile_x_sft = fast_macro_tile_x.shift_right;

      params.fast_res_tile_x_ori = fast_res_tile_x.divisor;
      params.fast_res_tile_x_mul = fast_res_tile_x.multiplier;
      params.fast_res_tile_x_sft = fast_res_tile_x.shift_right;

      params.fast_macro_tile_size_ori = fast_macro_tile_size.divisor;
      params.fast_macro_tile_size_mul = fast_macro_tile_size.multiplier;
      params.fast_macro_tile_size_sft = fast_macro_tile_size.shift_right;

      params.tile_a_dim0  = config.tile_a_dim0;
      params.tile_a_dim1  = config.tile_a_dim1;
      params.tile_a_dim2  = config.tile_a_dim2;
      params.tile_b_dim0  = config.tile_b_dim0;
      params.tile_b_dim1  = config.tile_b_dim1;
      params.tile_b_dim2  = config.tile_b_dim2;
      params.tile_c_dim0  = config.tile_c_dim0;
      params.tile_c_dim1  = config.tile_c_dim1;
      params.tile_c_dim2  = config.tile_c_dim2;
      params.tile_b1_dim0 = config.tile_b1_dim0;
      params.tile_b1_dim1 = config.tile_b1_dim1;
      params.tile_b1_dim2 = config.tile_b1_dim2;

      int max_blk_cnt =
          args.total_mp_count == args.target_mp_count ? args.total_mp_count * config.blk_per_mp : args.target_mp_count;
      launch_config.blockDimX      = config.nr_thr;
      launch_config.blockDimY      = 1;
      launch_config.blockDimZ      = 1;
      launch_config.gridDimX       = std::min(max_blk_cnt, params.total_tile);
      launch_config.gridDimY       = 1;
      launch_config.gridDimZ       = 1;
      launch_config.hStream        = reinterpret_cast<MUstream>(stream);
      launch_config.sharedMemBytes = 0;
      launch_config.attrs          = NULL;
      launch_config.numAttrs       = 0;

    } else {
      throw std::runtime_error("Get Unsupported MoeGemmMode!");
    }

    return std::make_tuple(params, config, launch_config);
  }  // to_underlying_arguments

  void run() {
    launch_asm(*config.asm_func, launch_config, params);
  }

 protected:
  Params       params;
  Config       config;
  LaunchConfig launch_config;

};  // class MoeGemm8bitAsmKernel

}  // namespace mubin
}  // namespace mate::moe_gemm

void ragged_moe_gemm_8bit(const std::tuple<at::Tensor, at::Tensor>&    input_a,             // (a, scale_a)
                          const std::tuple<at::Tensor, at::Tensor>&    input_b,             // (b, scale_b)
                          const at::Tensor&                            ragged_tokens_info,  // m_indices
                          const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
                          at::Tensor&                                  out,
                          const int64_t                                alignment_m) {
  const auto& [a, scale_a] = input_a;
  const auto& [b, scale_b] = input_b;

  CHECK_MP31("ragged_moe_gemm_8bit");
  at::TensorArg targs[]{{a, "a", 0},
                        {b, "b", 1},
                        {scale_a, "scale_a", 2},
                        {scale_b, "scale_b", 3},
                        {ragged_tokens_info, "ragged_tokens_info", 4},
                        {out, "out", 5}};
  at::checkAllSameGPU(__func__, targs);

  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(a, 2);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(b, 3);
  CHECK_TENSOR_AND_CONTIGUOUS(out, 2);
  CHECK_TENSOR_AND_CONTIGUOUS(scale_a, 2);
  CHECK_TENSOR_AND_CONTIGUOUS(scale_b, 3);
  CHECK_TENSOR_AND_CONTIGUOUS(ragged_tokens_info, 1);
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "ragged_moe_gemm_8bit() scale_a must be float type");
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "ragged_moe_gemm_8bit() scale_b must be float type");
  TORCH_CHECK(ragged_tokens_info.scalar_type() == at::kInt,
              "ragged_moe_gemm_8bit() ragged_tokens_info must be int32 type");

  TORCH_CHECK(is_fp8_tensor_type(a.scalar_type()), "ragged_moe_gemm_8bit() a must be fp8 type");
  TORCH_CHECK(is_fp8_tensor_type(b.scalar_type()), "ragged_moe_gemm_8bit() b must be fp8 type");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(out.scalar_type()), "ragged_moe_gemm_8bit() out must be bf16 or fp16 type");

  at::musa::OptionalMUSAGuard guard(a.device());
  auto                        dprops = at::musa::getCurrentDeviceProperties();

  mate::moe_gemm::MoeGemm8bitArgs args;

  args.type_a = a.scalar_type();
  args.type_b = b.scalar_type();
  args.type_d = out.scalar_type();

  args.m          = a.size(0);
  args.n          = b.size(1);
  args.k          = a.size(1);
  args.num_expert = b.size(0);

  args.quant_tile = std::get<2>(scale_granularity_mnk);
  if (args.quant_tile != 128) {
    throw std::runtime_error("ragged_moe_gemm_8bit() quant_tile (scale_granularity_k) must be 128!");
  }
  args.alignment_m = alignment_m;
  if (args.alignment_m != 128 && args.alignment_m != 256) {
    throw std::runtime_error("ragged_moe_gemm_8bit() alignment_m must be 128 or 256!");
  }
  args.total_mp_count  = dprops->multiProcessorCount;
  args.target_mp_count = dprops->multiProcessorCount;

  const auto batch_a    = a.unsqueeze(0);
  const auto batch_d    = out.unsqueeze(0);
  args.stride_m_a       = batch_a.stride(1);
  args.stride_batch_a   = batch_a.stride(0);
  args.stride_n_b       = b.stride(1);
  args.stride_batch_b   = b.stride(0);
  args.stride_m_out     = batch_d.stride(1);
  args.stride_batch_out = batch_d.stride(0);

  args.major_scale_a = TensorMajor::K;
  args.major_scale_b = TensorMajor::K;

  CHECK_SHAPE(a, args.m, args.k);
  CHECK_SHAPE(b, args.num_expert, args.n, args.k);
  CHECK_SHAPE(scale_a, args.m, mutlass::ceil_div(args.k, args.quant_tile));
  CHECK_SHAPE(
      scale_b, args.num_expert, mutlass::ceil_div(args.n, args.quant_tile), mutlass::ceil_div(args.k, args.quant_tile));
  CHECK_SHAPE(out, args.m, args.n);
  CHECK_SHAPE(ragged_tokens_info, args.m);

  args.scale_a_m       = scale_a.size(0);
  args.scale_a_k       = scale_a.size(1);
  args.scale_a_nr_elem = scale_a.numel();
  args.scale_b_n       = scale_b.size(1);
  args.scale_b_k       = scale_b.size(2);
  args.scale_b_nr_elem = scale_b.numel();

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = scale_a.data_ptr();
  args.p_scale_b   = scale_b.data_ptr();
  args.p_m_indices = ragged_tokens_info.data_ptr();
  args.p_signal    = nullptr;

  musaStream_t stream = at::musa::getCurrentMUSAStream().stream();

  using Kernel = mate::moe_gemm::mubin::MoeGemm8bitAsmKernel<mate::moe_gemm::MoeGemmMode::Ragged>;
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();
}

std::optional<std::tuple<int64_t, int64_t>> masked_moe_gemm_8bit(
    const std::tuple<at::Tensor, at::Tensor>&    input_a,  // (a, scale_a)
    const std::tuple<at::Tensor, at::Tensor>&    input_b,  // (b, scale_b)
    const at::Tensor&                            masked_tokens_info,
    const std::tuple<int64_t, int64_t, int64_t>& scale_granularity_mnk,
    at::Tensor&                                  out,
    const int64_t                                expect_tokens,
    const std::optional<at::Tensor>&             signal) {
  const auto& [a, scale_a] = input_a;
  const auto& [b, scale_b] = input_b;

  CHECK_MP31("masked_moe_gemm_8bit");
  at::TensorArg targs[]{{a, "a", 0},
                        {b, "b", 1},
                        {scale_a, "scale_a", 2},
                        {scale_b, "scale_b", 3},
                        {masked_tokens_info, "masked_tokens_info", 4},
                        {out, "out", 5}};
  at::checkAllSameGPU(__func__, targs);

  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(a, 3);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(b, 3);

  CHECK_TENSOR_AND_CONTIGUOUS(out, 3);
  // CHECK_TENSOR_AND_CONTIGUOUS(scale_a, 3);
  CHECK_TENSOR(scale_a, 3);
  TORCH_CHECK(scale_a.stride(-1) == 1 || scale_a.stride(-2) == 1, "masked_moe_gemm_8bit() scale_a must be contiguous")
  CHECK_TENSOR_AND_CONTIGUOUS(scale_b, 3);
  CHECK_TENSOR_AND_CONTIGUOUS(masked_tokens_info, 1);
  if (signal.has_value()) {
    CHECK_TENSOR_AND_CONTIGUOUS(signal.value(), 1);
  }

  TORCH_CHECK(scale_a.scalar_type() == at::kFloat, "masked_moe_gemm_8bit() scale_a must be float type");
  TORCH_CHECK(scale_b.scalar_type() == at::kFloat, "masked_moe_gemm_8bit() scale_b must be float type");
  TORCH_CHECK(masked_tokens_info.scalar_type() == at::kInt,
              "masked_moe_gemm_8bit() masked_tokens_info must be int32 type");
  TORCH_CHECK(is_fp8_tensor_type(a.scalar_type()), "masked_moe_gemm_8bit() a must be fp8 type");
  TORCH_CHECK(is_fp8_tensor_type(b.scalar_type()), "masked_moe_gemm_8bit() b must be fp8 type");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(out.scalar_type()), "masked_moe_gemm_8bit() out must be bf16 or fp16 type");

  at::musa::OptionalMUSAGuard guard(a.device());
  auto                        dprops = at::musa::getCurrentDeviceProperties();

  mate::moe_gemm::MoeGemm8bitArgs args;
  args.type_a = a.scalar_type();
  args.type_b = b.scalar_type();
  args.type_d = out.scalar_type();

  const int max_m = a.size(1);
  args.num_expert = a.size(0);
  args.m          = max_m * args.num_expert;
  args.n          = b.size(1);
  args.k          = b.size(2);

  args.quant_tile = std::get<2>(scale_granularity_mnk);
  if (args.quant_tile != 128) {
    throw std::runtime_error("masked_moe_gemm_8bit() quant_tile (scale_granularity_k) must 128!");
  }
  args.alignment_m     = 0;
  args.expected_m      = expect_tokens;
  args.total_mp_count  = dprops->multiProcessorCount;
  args.target_mp_count = dprops->multiProcessorCount;

  auto batch_a          = a.unsqueeze(0);
  auto batch_d          = out.unsqueeze(0);
  args.stride_m_a       = batch_a.stride(2);
  args.stride_batch_a   = batch_a.stride(0);
  args.stride_n_b       = b.stride(1);
  args.stride_batch_b   = b.stride(0);
  args.stride_m_out     = batch_d.stride(2);
  args.stride_batch_out = batch_d.stride(0);

  args.major_scale_a = scale_a.stride(-1) != 1 ? TensorMajor::MN : TensorMajor::K;
  args.major_scale_b = TensorMajor::K;
  if (args.major_scale_a == TensorMajor::MN) {
    TORCH_CHECK(max_m == scale_a.stride(-1),
                "masked_moe_gemm_8bit() uncontig scale a only support max_m_per_group "
                "== scale_a.stride(-1)");
  }

  CHECK_SHAPE(a, args.num_expert, max_m, args.k);
  CHECK_SHAPE(b, args.num_expert, args.n, args.k);
  CHECK_SHAPE(scale_a, args.num_expert, max_m, mutlass::ceil_div(args.k, args.quant_tile));
  CHECK_SHAPE(
      scale_b, args.num_expert, mutlass::ceil_div(args.n, args.quant_tile), mutlass::ceil_div(args.k, args.quant_tile));
  CHECK_SHAPE(out, args.num_expert, max_m, args.n);
  CHECK_SHAPE(masked_tokens_info, args.num_expert);
  if (signal.has_value()) {
    constexpr int tile_signal = 64;
    CHECK_SHAPE(signal.value(), args.num_expert * mutlass::ceil_div(max_m, tile_signal));
  }

  args.scale_a_m       = args.major_scale_a == TensorMajor::MN ? max_m : scale_a.size(0) * scale_a.size(1);
  args.scale_a_k       = scale_a.size(2);
  args.scale_a_nr_elem = scale_a.numel();
  args.scale_b_n       = scale_b.size(1);
  args.scale_b_k       = scale_b.size(2);
  args.scale_b_nr_elem = scale_b.numel();

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = scale_a.data_ptr();
  args.p_scale_b   = scale_b.data_ptr();
  args.p_m_indices = masked_tokens_info.data_ptr();
  args.p_signal    = signal.has_value() ? signal.value().data_ptr() : nullptr;

  musaStream_t stream = at::musa::getCurrentMUSAStream().stream();

  using Kernel = mate::moe_gemm::mubin::MoeGemm8bitAsmKernel<mate::moe_gemm::MoeGemmMode::Masked>;
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();

  if (signal.has_value()) {
    return std::make_pair(config.tile_m, mutlass::ceil_div(args.n, config.tile_n));
  } else {
    return std::nullopt;
  }
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "ragged_moe_gemm_8bit("
      "(Tensor, Tensor) input_a,"
      "(Tensor, Tensor) input_b,"
      "Tensor ragged_tokens_info,"
      "(int, int, int) scale_granularity_mnk,"
      "Tensor out,"
      "int alignment_m"
      ")"
      "-> ()");
  m.def(
      "masked_moe_gemm_8bit("
      "(Tensor, Tensor) input_a,"
      "(Tensor, Tensor) input_b,"
      "Tensor masked_tokens_info,"
      "(int, int, int) scale_granularity_mnk,"
      "Tensor out,"
      "int expect_tokens,"
      "Tensor? signal=None"
      ")"
      "-> ((int, int)?)");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("ragged_moe_gemm_8bit", &ragged_moe_gemm_8bit);
  m.impl("masked_moe_gemm_8bit", &masked_moe_gemm_8bit);
}
