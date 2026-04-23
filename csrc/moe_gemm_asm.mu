#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mutlass/fast_math.h>

#include <iostream>
#include <mute/algorithm/tuple_algorithms.hpp>
#include <mute/arch/copy_mp31_desc.hpp>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "asm_common.hpp"
#include "gemm_common_utils.hpp"
#include "mubin/mp31/mp31_moe_gemm_8bit_registry.hpp"
#include "mudnn_utils.hpp"
#include "op_utils.hpp"

namespace mate::moe_gemm {

enum class MoeGemmMode {
  Reserve0 = 0,
  RaggedPSumLayout,
  Reserve1,
  Ragged,
  Masked,
  ByteML_Ragged,
  KContig
};  // enum class MoeGemmMode

struct MoeGemmArgs {
  DLDataType type_a;
  DLDataType type_b;
  DLDataType type_d;

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
  size_t stride_k_a{1};
  size_t stride_batch_a;
  size_t stride_n_b;
  size_t stride_k_b{1};
  size_t stride_batch_b;
  size_t stride_m_out;
  size_t stride_batch_out;

  TensorMajor major_a;
  TensorMajor major_b;

  TensorQuantMode quant_mode_a;
  TensorQuantMode quant_mode_b;

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

};  // struct MoeGemmArgs

namespace mubin {

template <MoeGemmMode mode>
struct MoeGemmMubinDispatcher {
  MoeGemmMubinDispatcher() {
    // register all moe gemm mubin kernels
    // do only once
    init_moe_gemm_asm_kern_registry();
  }

  struct Block {
    int num_thread = 512;
    int tile_m     = 256;
    int tile_n     = 256;
    int tile_k     = 128;

    float base_score = 1.0;

    int blk_per_mp = 1;

    float estimate(const int nr_groups, const int m, const int n, const int k, const int nr_mp) const {
      if (base_score == 0.f) return base_score;
      int average_group_m         = m / nr_groups;
      int upper_aver_group_m      = (double)average_group_m * 1.04;
      int lower_aver_group_m      = (double)average_group_m * 0.96;
      int deviate_group_m_nr      = nr_groups / 3;  // assume that 1/3 of group has lower m and 1/3 has more m.
      int lower_aver_group_m_tile = mutlass::ceil_div(lower_aver_group_m, tile_m);
      int upper_aver_group_m_tile = mutlass::ceil_div(upper_aver_group_m, tile_m);
      int aver_group_m_tile       = mutlass::ceil_div(upper_aver_group_m, tile_m);
      int m_tile = lower_aver_group_m_tile * deviate_group_m_nr + upper_aver_group_m_tile * deviate_group_m_nr +
                   aver_group_m_tile * (nr_groups - deviate_group_m_nr * 2);
      int   n_tile       = mutlass::ceil_div(n, tile_n);
      int   wave         = mutlass::ceil_div(m_tile * n_tile, nr_mp);
      float wave_ratio   = (float)m * n / ((float)tile_m * tile_n * wave * nr_mp);
      float normal_score = base_score * wave_ratio;
      return normal_score;
      // TODO: consider splitk
    }

  };  // struct Block

  struct Config {
    Block block;

    int num_squad_m = 2;
    int num_squad_n = 2;

    int macro_tile_x   = 2;
    int switch_swizzle = 0;

    MUfunction* asm_func;

  };  // struct Config

  void get_block_tile(const std::vector<Block>& block_map, Config& config, const MoeGemmArgs& args) {
    float        best_score = -1.0;
    const Block* best_block = nullptr;
    for (auto& block : block_map) {
      float rst = block.estimate(args.num_expert, args.m, args.n, args.k, args.total_mp_count);
      if (rst > best_score) {
        best_score = rst;
        best_block = &block;
      }
    }
    if (best_block == nullptr) {
      throw std::runtime_error("No block to find!");
    }
    config.block.tile_m     = best_block->tile_m;
    config.block.tile_n     = best_block->tile_n;
    config.block.num_thread = best_block->num_thread;
    config.block.tile_k     = best_block->tile_k;
    config.block.blk_per_mp = best_block->blk_per_mp;
    // now use static kernel tile
  }

  Config get_kernel_config(const MoeGemmArgs& args) {
    static std::once_flag                                      group_mode_to_block_map_init_flag;
    static std::unordered_map<MoeGemmMode, std::vector<Block>> group_mode_to_block_map;

    // TODO: for new dispatching method
    std::call_once(group_mode_to_block_map_init_flag, [&]() {
      group_mode_to_block_map[MoeGemmMode::Ragged].push_back({512, 128, 256, 128, 1.0, 1});
      group_mode_to_block_map[MoeGemmMode::Ragged].push_back({512, 256, 256, 128, 1.0, 1});

      group_mode_to_block_map[MoeGemmMode::Masked].push_back({512, 128, 256, 128, 1.0, 1});
      group_mode_to_block_map[MoeGemmMode::Masked].push_back({512, 256, 256, 128, 1.0, 1});

      group_mode_to_block_map[MoeGemmMode::KContig].push_back({512, 128, 256, 128, 1.0, 1});
      group_mode_to_block_map[MoeGemmMode::KContig].push_back({512, 256, 256, 128, 1.0, 1});

      group_mode_to_block_map[MoeGemmMode::RaggedPSumLayout].push_back({512, 256, 256, 128, 1.0, 1});
      group_mode_to_block_map[MoeGemmMode::RaggedPSumLayout].push_back({256, 128, 256, 128, 0.8, 1});
      group_mode_to_block_map[MoeGemmMode::RaggedPSumLayout].push_back({256, 256, 128, 128, 0.8, 1});
    });

    Config config;

    int b_tmenc_mode = 0;

    if constexpr (mode == MoeGemmMode::Ragged) {
      if (args.alignment_m == 256) {
        config.block.tile_m     = 256;
        config.block.num_thread = 512;
      } else if (args.alignment_m == 128) {
        config.block.tile_m     = 128;
        config.block.num_thread = 256;
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

      if (select_m == 128 || (select_m == 256 && args.expected_m <= 256)) {
        // use optimized version
        b_tmenc_mode = 1;
      }

      constexpr int select_n = -1;
      if (select_m == 256) {
        config.block.tile_m     = 256;
        config.block.num_thread = 512;
      } else if (select_m == 128 && select_n == 160) {
        config.block.tile_m     = 128;
        config.block.tile_n     = 160;
        config.block.num_thread = 256;
        config.block.blk_per_mp = 2;
      } else if (select_m == 128) {
        config.block.tile_m     = 128;
        config.block.num_thread = 256;
      }
    } else if constexpr (mode == MoeGemmMode::KContig) {
      config.block.tile_m     = 256;
      config.block.tile_n     = 256;
      config.block.num_thread = 512;
      config.block.tile_k     = 128;
    } else if constexpr (mode == MoeGemmMode::RaggedPSumLayout) {
      get_block_tile(group_mode_to_block_map[MoeGemmMode::RaggedPSumLayout], config, args);
    } else {
      throw std::runtime_error("MoeGemmMubinDispatcher mode not supported!");
    }

    config.num_squad_m = config.block.tile_m / 128;
    config.num_squad_n = config.block.tile_n / 128;

    MoeGemmAsmID id;
    id.a_type = moe_gemm_asm_src_type_to_id.at(encode_dlpack_dtype(args.type_a));
    id.b_type = moe_gemm_asm_src_type_to_id.at(encode_dlpack_dtype(args.type_b));
    id.d_type = moe_gemm_asm_dst_type_to_id.at(encode_dlpack_dtype(args.type_d));
    id.mode   = static_cast<int>(mode);

    id.trans_a = static_cast<int>(args.major_a == TensorMajor::MN);
    id.trans_b = static_cast<int>(args.major_b == TensorMajor::K);

    id.tile_m      = moe_gemm_asm_tile_m_to_id.at(config.block.tile_m);
    id.tile_n      = moe_gemm_asm_tile_n_to_id.at(config.block.tile_n);
    id.tile_k_byte = moe_gemm_asm_tile_k_to_id.at(config.block.tile_k);

    id.b_tmenc = b_tmenc_mode;
    id.a_fsl   = is_fp8_dtype(args.type_a) && static_cast<int>(args.major_scale_a == TensorMajor::MN);

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
        throw std::runtime_error("MoeGemmMubinDispatcher() mubin kernel not found!");
      }

      // Call the kernel first time
      // we need to load it and cache it

      auto [curr_asm_meta, success] = kernel_map.try_emplace(id, asm_meta->second.first, asm_meta->second.second);

      config.asm_func = &curr_asm_meta->second.asm_func;

      return config;
    }
  }

  static float calc_block_score(
      const Block& block, const int num_expert, const int m, const int n, const int k, const int num_mp) {
    // TODO: for new dispatching method
    int average_group_m         = m / num_expert;
    int upper_aver_group_m      = (double)average_group_m * 1.04;
    int lower_aver_group_m      = (double)average_group_m * 0.96;
    int lower_aver_group_m_tile = mutlass::ceil_div(lower_aver_group_m, block.tile_m);
    int upper_aver_group_m_tile = mutlass::ceil_div(upper_aver_group_m, block.tile_m);

    int aver_group_m_tile = mutlass::ceil_div(upper_aver_group_m, block.tile_m);

    int deviate_group_m_nr = num_expert / 3;
    int m_tile = lower_aver_group_m_tile * deviate_group_m_nr + upper_aver_group_m_tile * deviate_group_m_nr +
                 aver_group_m_tile * (num_expert - deviate_group_m_nr * 2);
    int n_tile = mutlass::ceil_div(n, block.tile_n);

    constexpr float epilog_loss_ratio = 0.5;

    int   wave         = mutlass::ceil_div(m_tile * n_tile, num_mp);
    float wave_ratio   = (float)m * n / ((float)block.tile_m * block.tile_n * wave * num_mp);
    float epilog_loss  = (float)block.tile_k / max(k, block.tile_k) * epilog_loss_ratio;
    float epilog_ratio = 1.f - epilog_loss;
    float normal_score = block.base_score * wave_ratio * epilog_ratio;

    return normal_score;
  }

  std::unordered_map<MoeGemmAsmID, MateAsmKernel> kernel_map;

};  // struct MoeGemmMubinDispatcher

template <MoeGemmMode mode, class Dispatcher = MoeGemmMubinDispatcher<mode>>
class MoeGemmAsmKernel {
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
    mute::RobustReg robust_input_a_scale1{};
    mute::RobustReg robust_input_b_scale1{};
    mute::RobustReg robust_input_d_scale{};
    float           rcp_scale_o{1.0f};
    void*           k_part_buffer{};
    int32_t*        k_part_flag{};

    mute::RobustReg robust_group_idx{};
    mute::RobustReg robust_group_scale_idx{};

    int32_t* group_scale_a_robust_vec{};
    int32_t* group_scale_b_robust_vec{};
    int32_t* sync_out_flag_ptr{};

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

    int32_t quant_tile{128};
    int32_t amax_mode{};
    int32_t act_mode{};

    int32_t m{};
    int32_t n{};
    int32_t k{};
    int32_t num_expert{};

    int32_t ldd{};
    int32_t ldc{};

    int32_t out_trans_stride{0};
    int32_t fast_out_trans_ori{0};
    int32_t fast_out_trans_mul{0};
    int32_t fast_out_trans_sft{0};

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
    int32_t tile_x1_dim0{};
    int32_t tile_x1_dim1{};
    int32_t tile_x1_dim2{};
  };  // struct Params

  MoeGemmAsmKernel(const Params& in_params, const Config& in_config, const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const MoeGemmArgs& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    constexpr int batch   = 1;
    int           batch_b = mode == MoeGemmMode::KContig ? 1 : args.num_expert;

    size_t tme_a_dim0, tme_a_dim1, tme_a_dim2, tme_a_stride0, tme_a_stride1;
    size_t tme_b_dim0, tme_b_dim1, tme_b_dim2, tme_b_stride0, tme_b_stride1;
    if (args.major_a == TensorMajor::MN) {
      tme_a_dim0    = args.m;
      tme_a_dim1    = args.k;
      tme_a_dim2    = batch;
      tme_a_stride0 = args.stride_k_a;
      tme_a_stride1 = args.stride_batch_a;
    } else {
      tme_a_dim0    = args.k;
      tme_a_dim1    = args.m;
      tme_a_dim2    = batch;
      tme_a_stride0 = args.stride_m_a;
      tme_a_stride1 = args.stride_batch_a;
    }
    if (args.major_b == TensorMajor::MN) {
      tme_b_dim0    = args.n;
      tme_b_dim1    = args.k;
      tme_b_dim2    = batch_b;
      tme_b_stride0 = args.stride_k_b;
      tme_b_stride1 = args.stride_batch_b;
    } else {
      tme_b_dim0    = args.k;
      tme_b_dim1    = args.n;
      tme_b_dim2    = batch_b;
      tme_b_stride0 = args.stride_n_b;
      tme_b_stride1 = args.stride_batch_b;
    }
    TmeDesc tensor_desc_a(mute::make_tuple(tme_a_dim0, tme_a_dim1, tme_a_dim2),
                          mute::make_tuple(tme_a_stride0, tme_a_stride1),
                          args.p_a,
                          dl_dtype_to_tme_type(args.type_a));
    TmeDesc tensor_desc_b(mute::make_tuple(tme_b_dim0, tme_b_dim1, tme_b_dim2),
                          mute::make_tuple(tme_b_stride0, tme_b_stride1),
                          args.p_b,
                          dl_dtype_to_tme_type(args.type_b));

    int gridy_in_group;
    int grid_y;
    int grid_x;
    int grid_z;
    if constexpr (mode == MoeGemmMode::Ragged) {
      grid_x         = mutlass::ceil_div(args.n, config.block.tile_n);
      grid_y         = mutlass::ceil_div(args.m, config.block.tile_m);
      grid_z         = 1;
      gridy_in_group = grid_y;
    } else if constexpr (mode == MoeGemmMode::Masked) {
      int max_m      = args.m / args.num_expert;
      gridy_in_group = mutlass::ceil_div(max_m, config.block.tile_m);
      grid_x         = mutlass::ceil_div(args.n, config.block.tile_n);
      grid_y         = gridy_in_group * args.num_expert;
      grid_z         = 1;
    } else if constexpr (mode == MoeGemmMode::KContig) {
      grid_x = mutlass::ceil_div(args.n, config.block.tile_n);
      grid_y = mutlass::ceil_div(args.m, config.block.tile_m);
      grid_z = mutlass::ceil_div(args.k, config.block.tile_k / dl_dtype_size(args.type_a));
    } else if constexpr (mode == MoeGemmMode::RaggedPSumLayout) {
      grid_x         = mutlass::ceil_div(args.n, config.block.tile_n);
      grid_y         = mutlass::ceil_div(args.m, config.block.tile_m) + args.num_expert - 1;
      grid_z         = 1;
      gridy_in_group = grid_y;
    } else {
      throw std::runtime_error("Get Unsupported MoeGemmMode!");
    }

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

    params.robust_input_a_scale = mute::make_robust_desc(static_cast<float*>(args.p_scale_a), args.scale_a_nr_elem).reg;
    params.robust_input_b_scale = mute::make_robust_desc(static_cast<float*>(args.p_scale_b), args.scale_b_nr_elem).reg;

    if constexpr (mode == MoeGemmMode::Ragged) {
      params.robust_group_idx = mute::make_robust_desc(static_cast<int*>(args.p_m_indices), args.m).reg;
    } else if constexpr (mode == MoeGemmMode::Masked) {
      params.robust_group_idx    = mute::make_robust_desc(static_cast<int*>(args.p_m_indices), args.num_expert).reg;
      params.sync_out_flag_ptr   = static_cast<int32_t*>(args.p_signal);
      params.mask_blky_per_group = gridy_in_group;
    } else if constexpr (mode == MoeGemmMode::KContig) {
      params.robust_group_idx = mute::make_robust_desc(static_cast<int*>(args.p_m_indices), args.num_expert).reg;
      params.beta             = 1.0;
      params.robust_input_z =
          mute::make_robust_desc(static_cast<int*>(args.p_d), args.m * args.n * args.num_expert).reg;
    } else if constexpr (mode == MoeGemmMode::RaggedPSumLayout) {
      params.robust_group_idx = mute::make_robust_desc(static_cast<int*>(args.p_m_indices), args.num_expert).reg;
    } else {
      throw std::runtime_error("Get Unsupported MoeGemmMode!");
    }

    params.scale_a_mode = static_cast<int>(args.quant_mode_a);
    params.scale_b_mode = static_cast<int>(args.quant_mode_b);
    params.scale_d_mode = 0;

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
    params.ldkpart      = mutlass::ceil_div(args.n, config.block.tile_n) * config.block.tile_n;
    params.kpart_batch_stride_c =
        static_cast<size_t>(mutlass::ceil_div(args.m, config.block.tile_m)) * config.block.tile_m * params.ldkpart;

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

    if (args.major_a == TensorMajor::MN) {
      params.tile_a_dim0 = config.block.tile_m / config.num_squad_m;
      params.tile_a_dim1 = config.block.tile_k / dl_dtype_size(args.type_a);
    } else {
      params.tile_a_dim0 = config.block.tile_k / dl_dtype_size(args.type_a);
      params.tile_a_dim1 = config.block.tile_m / config.num_squad_m;
    }
    params.tile_a_dim2 = 1;

    if (args.major_b == TensorMajor::MN) {
      params.tile_b_dim0 = config.block.tile_n / config.num_squad_n;
      params.tile_b_dim1 = config.block.tile_k / dl_dtype_size(args.type_b);
    } else {
      params.tile_b_dim0 = config.block.tile_k / dl_dtype_size(args.type_b);
      params.tile_b_dim1 = config.block.tile_n / config.num_squad_n;
    }
    params.tile_b_dim2 = 1;

    params.tile_c_dim0 = 128 / dl_dtype_size(args.type_d);
    params.tile_c_dim1 = 4;
    params.tile_c_dim2 = 1;

    // TODO: for unalign cases
    params.tile_x1_dim0 = 160 % 128;
    params.tile_x1_dim1 = 128;
    params.tile_x1_dim2 = 1;

    launch_config.blockDimX      = config.block.num_thread;
    launch_config.blockDimY      = 1;
    launch_config.blockDimZ      = 1;
    launch_config.gridDimX       = std::min(args.total_mp_count, params.total_tile);
    launch_config.gridDimY       = 1;
    launch_config.gridDimZ       = 1;
    launch_config.hStream        = reinterpret_cast<MUstream>(stream);
    launch_config.sharedMemBytes = 0;
    launch_config.attrs          = NULL;
    launch_config.numAttrs       = 0;

    if constexpr (mode == MoeGemmMode::Masked) {
      int max_blk_cnt = args.total_mp_count == args.target_mp_count ? args.total_mp_count * config.block.blk_per_mp
                                                                    : args.target_mp_count;
      launch_config.gridDimX = std::min(max_blk_cnt, params.total_tile);
    }

    return std::make_tuple(params, config, launch_config);
  }  // to_underlying_arguments

  void run() {
    void* kernel_params[] = {
        // 1-2: 输出指针
        &params.output_ptr,
        &params.amax_ptr,

        // 3-5: Tensor描述符 (对象地址)
        &params.a_desc,
        &params.b_desc,
        &params.c_desc,

        // 6-12: RobustReg输入参数 (对象地址)
        &params.robust_input_bias,
        &params.robust_input_z,
        &params.robust_input_a_scale,
        &params.robust_input_b_scale,
        &params.robust_input_a_scale1,
        &params.robust_input_b_scale1,
        &params.robust_input_d_scale,

        // 13-15: 浮点参数与缓冲区指针
        &params.rcp_scale_o,
        &params.k_part_buffer,
        &params.k_part_flag,

        // 16-17: Group索引RobustReg (对象地址)
        &params.robust_group_idx,
        &params.robust_group_scale_idx,

        // 18-20: Group缩放向量指针
        &params.group_scale_a_robust_vec,
        &params.group_scale_b_robust_vec,
        &params.sync_out_flag_ptr,

        // 21-24: 基础掩码与缩放系数
        &params.mask_blky_per_group,
        &params.alpha,
        &params.beta,
        &params.gamma,

        // 25-27: 缩放模式
        &params.scale_a_mode,
        &params.scale_b_mode,
        &params.scale_d_mode,

        // 28-31: 缩放维度
        &params.scale_a_m,
        &params.scale_a_k,
        &params.scale_b_k,
        &params.scale_b_n,

        // 32-33: 缩放步幅
        &params.batch_stride_scale_a,
        &params.batch_stride_scale_b,

        // 34-36: 量化与激活配置
        &params.quant_tile,
        &params.amax_mode,
        &params.act_mode,

        // 37-40: 核心矩阵维度
        &params.m,
        &params.n,
        &params.k,
        &params.num_expert,

        // 41-42: leading dimension
        &params.ldd,
        &params.ldc,

        // 43-46: 输出转置快速计算参数
        &params.out_trans_stride,
        &params.fast_out_trans_ori,
        &params.fast_out_trans_mul,
        &params.fast_out_trans_sft,

        // 47-49: Tile与掩码配置
        &params.total_tile,
        &params.mask_m_max,
        &params.macro_bidx_max,

        // 50-51: Batch步幅 (int64)
        &params.batch_stride_d,
        &params.batch_stride_c,

        // 52-55: K-partition相关
        &params.k_part,
        &params.k_part_block,
        &params.ldkpart,
        &params.kpart_batch_stride_c,

        // 56-58: XY-Tile快速计算
        &params.fast_xytile_ori,
        &params.fast_xytile_mul,
        &params.fast_xytile_sft,

        // 59-61: Batch快速计算
        &params.fast_batch_ori,
        &params.fast_batch_mul,
        &params.fast_batch_sft,

        // 62-64: NR Block Group快速计算
        &params.fast_nr_block_group_ori,
        &params.fast_nr_block_group_mul,
        &params.fast_nr_block_group_sft,

        // 65-67: Grid X快速计算
        &params.fast_gridx_ori,
        &params.fast_gridx_mul,
        &params.fast_gridx_sft,

        // 68-71: 集群与Swizzle配置
        &params.cluster_dimx,
        &params.macro_bid_max,
        &params.switch_swizzle,
        &params.max_tile_y,

        // 72-74: Macro Tile X快速计算
        &params.fast_macro_tile_x_ori,
        &params.fast_macro_tile_x_mul,
        &params.fast_macro_tile_x_sft,

        // 75-77: Res Tile X快速计算
        &params.fast_res_tile_x_ori,
        &params.fast_res_tile_x_mul,
        &params.fast_res_tile_x_sft,

        // 78-80: Macro Tile Size快速计算
        &params.fast_macro_tile_size_ori,
        &params.fast_macro_tile_size_mul,
        &params.fast_macro_tile_size_sft,

        // 81-83: Tile A维度
        &params.tile_a_dim0,
        &params.tile_a_dim1,
        &params.tile_a_dim2,

        // 84-86: Tile B维度
        &params.tile_b_dim0,
        &params.tile_b_dim1,
        &params.tile_b_dim2,

        // 87-89: Tile C维度
        &params.tile_c_dim0,
        &params.tile_c_dim1,
        &params.tile_c_dim2,

        // 90-92: Tile X1维度
        &params.tile_x1_dim0,
        &params.tile_x1_dim1,
        &params.tile_x1_dim2,
    };
    MATE_MUSA_DRIVER_CHECK(muLaunchKernelEx(&launch_config, *config.asm_func, kernel_params, nullptr));
  }

 protected:
  Params       params;
  Config       config;
  LaunchConfig launch_config;

};  // class MoeGemmAsmKernel

}  // namespace mubin
}  // namespace mate::moe_gemm

namespace {

namespace gemm_common = mate::gemm::common;

int current_num_mps(DLDevice device) {
  musaDeviceProp dprops{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, device.device_id));
  return dprops.multiProcessorCount;
}

size_t leading_unsqueezed_batch_stride(ffi::TensorView tensor) {
  TVM_FFI_ICHECK_GT(tensor.ndim(), 0) << "tensor must have rank greater than 0";
  return static_cast<size_t>(tensor.stride(0) * tensor.size(0));
}

template <typename Kernel>
void run_moe_gemm_kernel(const mate::moe_gemm::MoeGemmArgs& args, DLDevice device) {
  ffi::MUSADeviceGuard device_guard(device.device_id);
  musaStream_t         stream         = get_stream(device);
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();
}

}  // namespace

void ragged_moe_gemm_8bit(const std::tuple<ffi::TensorView, ffi::TensorView>& input_a,
                          const std::tuple<ffi::TensorView, ffi::TensorView>& input_b,
                          ffi::TensorView                                     ragged_tokens_info,
                          const std::tuple<int64_t, int64_t, int64_t>&        scale_granularity_mnk,
                          ffi::TensorView                                     out,
                          int64_t                                             alignment_m) {
  const auto& [a, scale_a] = input_a;
  const auto& [b, scale_b] = input_b;

  check_mp31(a.device(), "ragged_moe_gemm_8bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(ragged_tokens_info);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, ragged_tokens_info);
  CHECK_DEVICE(a, out);
  CHECK_DIM(2, a);
  CHECK_DIM(3, b);
  CHECK_DIM(2, scale_a);
  CHECK_DIM(3, scale_b);
  CHECK_DIM(1, ragged_tokens_info);
  CHECK_DIM(2, out);
  TVM_FFI_ICHECK_EQ(a.stride(-1), 1) << "a must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(b.stride(-1), 1) << "b must be contiguous at the last dimension";
  CHECK_CONTIGUOUS(scale_a);
  CHECK_CONTIGUOUS(scale_b);
  CHECK_CONTIGUOUS(ragged_tokens_info);
  CHECK_CONTIGUOUS(out);
  TVM_FFI_ICHECK(dtype_equal(scale_a.dtype(), dl_float32)) << "scale_a must be float32";
  TVM_FFI_ICHECK(dtype_equal(scale_b.dtype(), dl_float32)) << "scale_b must be float32";
  TVM_FFI_ICHECK(dtype_equal(ragged_tokens_info.dtype(), dl_int32)) << "ragged_tokens_info must be int32";
  TVM_FFI_ICHECK(is_fp8_dtype(a.dtype())) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(b.dtype())) << "b must be fp8";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(out.dtype())) << "out must be bf16 or fp16";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a = a.dtype();
  args.type_b = b.dtype();
  args.type_d = out.dtype();

  args.m          = static_cast<int>(a.size(0));
  args.n          = static_cast<int>(b.size(1));
  args.k          = static_cast<int>(a.size(1));
  args.num_expert = static_cast<int>(b.size(0));

  TVM_FFI_ICHECK_EQ(b.size(2), args.k);
  TVM_FFI_ICHECK_EQ(scale_a.size(0), args.m);
  TVM_FFI_ICHECK_EQ(scale_a.size(1), mutlass::ceil_div(args.k, 128));
  TVM_FFI_ICHECK_EQ(scale_b.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(scale_b.size(1), mutlass::ceil_div(args.n, 128));
  TVM_FFI_ICHECK_EQ(scale_b.size(2), mutlass::ceil_div(args.k, 128));
  TVM_FFI_ICHECK_EQ(out.size(0), args.m);
  TVM_FFI_ICHECK_EQ(out.size(1), args.n);
  TVM_FFI_ICHECK_EQ(ragged_tokens_info.size(0), args.m);

  if (gemm_common::gemm_early_return(args.m, args.n, args.k, out)) {
    return;
  }

  args.quant_tile = static_cast<int>(std::get<2>(scale_granularity_mnk));
  TVM_FFI_ICHECK_EQ(args.quant_tile, 128) << "quant_tile must be 128";
  args.alignment_m = static_cast<int>(alignment_m);
  TVM_FFI_ICHECK(args.alignment_m == 128 || args.alignment_m == 256) << "alignment_m must be 128 or 256";
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = args.total_mp_count;

  args.stride_m_a       = a.stride(0);
  args.stride_batch_a   = leading_unsqueezed_batch_stride(a);
  args.stride_n_b       = b.stride(1);
  args.stride_batch_b   = b.stride(0);
  args.stride_m_out     = out.stride(0);
  args.stride_batch_out = leading_unsqueezed_batch_stride(out);

  args.major_a       = TensorMajor::K;
  args.major_b       = TensorMajor::K;
  args.quant_mode_a  = TensorQuantMode::GROUP;
  args.quant_mode_b  = TensorQuantMode::BLOCK;
  args.major_scale_a = TensorMajor::K;
  args.major_scale_b = TensorMajor::K;

  args.scale_a_m       = static_cast<int>(scale_a.size(0));
  args.scale_a_k       = static_cast<int>(scale_a.size(1));
  args.scale_a_nr_elem = static_cast<int>(scale_a.numel());
  args.scale_b_n       = static_cast<int>(scale_b.size(1));
  args.scale_b_k       = static_cast<int>(scale_b.size(2));
  args.scale_b_nr_elem = static_cast<int>(scale_b.numel());

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = scale_a.data_ptr();
  args.p_scale_b   = scale_b.data_ptr();
  args.p_m_indices = ragged_tokens_info.data_ptr();
  args.p_signal    = nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::Ragged>;
  run_moe_gemm_kernel<Kernel>(args, a.device());
}

std::optional<std::tuple<int64_t, int64_t>> masked_moe_gemm_8bit(
    const std::tuple<ffi::TensorView, ffi::TensorView>& input_a,
    const std::tuple<ffi::TensorView, ffi::TensorView>& input_b,
    ffi::TensorView                                     masked_tokens_info,
    const std::tuple<int64_t, int64_t, int64_t>&        scale_granularity_mnk,
    ffi::TensorView                                     out,
    int64_t                                             expect_tokens,
    std::optional<ffi::TensorView>                      signal) {
  const auto& [a, scale_a] = input_a;
  const auto& [b, scale_b] = input_b;

  check_mp31(a.device(), "masked_moe_gemm_8bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(masked_tokens_info);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, masked_tokens_info);
  CHECK_DEVICE(a, out);
  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(3, scale_a);
  CHECK_DIM(3, scale_b);
  CHECK_DIM(1, masked_tokens_info);
  CHECK_DIM(3, out);
  TVM_FFI_ICHECK_EQ(a.stride(-1), 1) << "a must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(b.stride(-1), 1) << "b must be contiguous at the last dimension";
  TVM_FFI_ICHECK(scale_a.stride(-1) == 1 || scale_a.stride(-2) == 1) << "scale_a must be contiguous";
  CHECK_CONTIGUOUS(scale_b);
  CHECK_CONTIGUOUS(masked_tokens_info);
  CHECK_CONTIGUOUS(out);
  if (signal.has_value()) {
    CHECK_MUSA(signal.value());
    CHECK_CONTIGUOUS(signal.value());
    CHECK_DEVICE(a, signal.value());
    CHECK_DIM(1, signal.value());
  }
  TVM_FFI_ICHECK(dtype_equal(scale_a.dtype(), dl_float32)) << "scale_a must be float32";
  TVM_FFI_ICHECK(dtype_equal(scale_b.dtype(), dl_float32)) << "scale_b must be float32";
  TVM_FFI_ICHECK(dtype_equal(masked_tokens_info.dtype(), dl_int32)) << "masked_tokens_info must be int32";
  TVM_FFI_ICHECK(is_fp8_dtype(a.dtype())) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(b.dtype())) << "b must be fp8";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(out.dtype())) << "out must be bf16 or fp16";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a = a.dtype();
  args.type_b = b.dtype();
  args.type_d = out.dtype();

  const int max_m = static_cast<int>(a.size(1));
  args.num_expert = static_cast<int>(a.size(0));
  args.m          = max_m * args.num_expert;
  args.n          = static_cast<int>(b.size(1));
  args.k          = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_EQ(a.size(2), args.k);
  TVM_FFI_ICHECK_EQ(b.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(out.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(out.size(1), max_m);
  TVM_FFI_ICHECK_EQ(out.size(2), args.n);
  TVM_FFI_ICHECK_EQ(masked_tokens_info.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(scale_a.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(scale_a.size(1), max_m);
  TVM_FFI_ICHECK_EQ(scale_a.size(2), mutlass::ceil_div(args.k, 128));
  TVM_FFI_ICHECK_EQ(scale_b.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(scale_b.size(1), mutlass::ceil_div(args.n, 128));
  TVM_FFI_ICHECK_EQ(scale_b.size(2), mutlass::ceil_div(args.k, 128));

  if (gemm_common::gemm_early_return(max_m, args.n, args.k, out)) {
    return std::nullopt;
  }

  args.quant_tile = static_cast<int>(std::get<2>(scale_granularity_mnk));
  TVM_FFI_ICHECK_EQ(args.quant_tile, 128) << "quant_tile must be 128";
  args.alignment_m     = 0;
  args.expected_m      = static_cast<int>(expect_tokens);
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = args.total_mp_count;

  args.stride_m_a       = a.stride(1);
  args.stride_batch_a   = leading_unsqueezed_batch_stride(a);
  args.stride_n_b       = b.stride(1);
  args.stride_batch_b   = b.stride(0);
  args.stride_m_out     = out.stride(1);
  args.stride_batch_out = leading_unsqueezed_batch_stride(out);

  args.major_a       = TensorMajor::K;
  args.major_b       = TensorMajor::K;
  args.quant_mode_a  = TensorQuantMode::GROUP;
  args.quant_mode_b  = TensorQuantMode::BLOCK;
  args.major_scale_a = scale_a.stride(-1) != 1 ? TensorMajor::MN : TensorMajor::K;
  args.major_scale_b = TensorMajor::K;

  if (args.major_scale_a == TensorMajor::MN) {
    TVM_FFI_ICHECK_EQ(scale_a.stride(-1), max_m)
        << "uncontiguous scale_a only supports max_m_per_group == scale_a.stride(-1)";
  }

  if (signal.has_value()) {
    constexpr int tile_signal = 64;
    TVM_FFI_ICHECK_EQ(signal.value().size(0), args.num_expert * mutlass::ceil_div(max_m, tile_signal));
  }

  args.scale_a_m = args.major_scale_a == TensorMajor::MN ? max_m : static_cast<int>(scale_a.size(0) * scale_a.size(1));
  args.scale_a_k = static_cast<int>(scale_a.size(2));
  args.scale_a_nr_elem = static_cast<int>(scale_a.numel());
  args.scale_b_n       = static_cast<int>(scale_b.size(1));
  args.scale_b_k       = static_cast<int>(scale_b.size(2));
  args.scale_b_nr_elem = static_cast<int>(scale_b.numel());

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = scale_a.data_ptr();
  args.p_scale_b   = scale_b.data_ptr();
  args.p_m_indices = masked_tokens_info.data_ptr();
  args.p_signal    = signal.has_value() ? signal.value().data_ptr() : nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::Masked>;
  run_moe_gemm_kernel<Kernel>(args, a.device());

  if (signal.has_value()) {
    auto [_, config, __] = Kernel::to_underlying_arguments(args, get_stream(a.device()));
    return std::make_pair(config.block.tile_m, mutlass::ceil_div(args.n, config.block.tile_n));
  }
  return std::nullopt;
}

void m_grouped_contig_gemm_8bit(const std::tuple<ffi::TensorView, ffi::TensorView>& input_a,
                                const std::tuple<ffi::TensorView, ffi::TensorView>& input_b,
                                ffi::TensorView                                     group_m_idx,
                                const std::tuple<int64_t, int64_t, int64_t>&        scale_granularity_mnk,
                                ffi::TensorView                                     out,
                                const std::string&                                  major_mode_a,
                                const std::string&                                  major_mode_b,
                                std::optional<int64_t>                              num_mp) {
  const auto& [a, scale_a] = input_a;
  const auto& [b, scale_b] = input_b;

  check_mp31(a.device(), "m_grouped_contig_gemm_8bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(group_m_idx);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, group_m_idx);
  CHECK_DEVICE(a, out);
  CHECK_DIM(2, a);
  CHECK_DIM(1, group_m_idx);
  TVM_FFI_ICHECK(scale_a.stride(-1) == 1 || scale_a.stride(-2) == 1) << "scale_a must be contiguous";
  TVM_FFI_ICHECK(dtype_equal(scale_a.dtype(), dl_float32)) << "scale_a must be float32";
  TVM_FFI_ICHECK(dtype_equal(scale_b.dtype(), dl_float32)) << "scale_b must be float32";
  TVM_FFI_ICHECK(is_fp8_dtype(a.dtype())) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(b.dtype())) << "b must be fp8";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a = a.dtype();
  args.type_b = b.dtype();
  args.type_d = out.dtype();

  args.num_expert = static_cast<int>(group_m_idx.size(0));
  args.m          = static_cast<int>(a.size(0));
  args.n          = major_mode_b[0] == 'K' ? static_cast<int>(b.size(-2)) : static_cast<int>(b.size(-1));
  args.k          = static_cast<int>(a.size(1));

  if (gemm_common::gemm_early_return(args.m, args.n, args.k, out)) {
    return;
  }

  args.quant_tile = static_cast<int>(std::get<2>(scale_granularity_mnk));
  TVM_FFI_ICHECK_EQ(args.quant_tile, 128) << "quant_tile must be 128";
  args.alignment_m     = 0;
  args.expected_m      = 0;
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = static_cast<int>(num_mp.value_or(args.total_mp_count));

  args.stride_m_a       = args.k;
  args.stride_k_a       = args.m;
  args.stride_batch_a   = args.m * args.k;
  args.stride_n_b       = args.k;
  args.stride_k_b       = args.n;
  args.stride_batch_b   = args.n * args.k;
  args.stride_m_out     = args.n;
  args.stride_batch_out = args.m * args.n;

  args.major_a       = TensorMajor::K;
  args.major_b       = major_mode_b[0] == 'K' ? TensorMajor::K : TensorMajor::MN;
  args.quant_mode_a  = TensorQuantMode::GROUP;
  args.quant_mode_b  = TensorQuantMode::BLOCK;
  args.major_scale_a = TensorMajor::K;
  args.major_scale_b = major_mode_b[0] == 'K' ? TensorMajor::K : TensorMajor::MN;

  args.scale_a_m       = static_cast<int>(scale_a.size(0));
  args.scale_a_k       = static_cast<int>(scale_a.size(-1));
  args.scale_a_nr_elem = args.scale_a_m * args.scale_a_k;
  args.scale_b_n = major_mode_b[0] == 'K' ? static_cast<int>(scale_b.size(-2)) : static_cast<int>(scale_b.size(-1));
  args.scale_b_k = major_mode_b[0] == 'K' ? static_cast<int>(scale_b.size(-1)) : static_cast<int>(scale_b.size(-2));
  args.scale_b_nr_elem = args.scale_b_n * args.scale_b_k * args.num_expert;

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = scale_a.data_ptr();
  args.p_scale_b   = scale_b.data_ptr();
  args.p_m_indices = group_m_idx.data_ptr();
  args.p_signal    = nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::RaggedPSumLayout>;
  run_moe_gemm_kernel<Kernel>(args, a.device());
}

void k_grouped_contig_gemm_8bit(const std::tuple<ffi::TensorView, ffi::TensorView>& input_a,
                                const std::tuple<ffi::TensorView, ffi::TensorView>& input_b,
                                ffi::TensorView                                     group_k_idx,
                                const std::tuple<int64_t, int64_t, int64_t>&        scale_granularity_mnk,
                                ffi::TensorView                                     out,
                                std::optional<int64_t>                              num_mp) {
  const auto& [a, scale_a] = input_a;
  const auto& [b, scale_b] = input_b;

  check_mp31(a.device(), "k_grouped_contig_gemm_8bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(scale_a);
  CHECK_MUSA(scale_b);
  CHECK_MUSA(group_k_idx);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, scale_a);
  CHECK_DEVICE(a, scale_b);
  CHECK_DEVICE(a, group_k_idx);
  CHECK_DEVICE(a, out);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(2, scale_a);
  CHECK_DIM(2, scale_b);
  CHECK_DIM(1, group_k_idx);
  TVM_FFI_ICHECK_EQ(a.stride(-1), 1) << "a must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(b.stride(-1), 1) << "b must be contiguous at the last dimension";
  TVM_FFI_ICHECK(scale_a.stride(-1) == 1 || scale_a.stride(-2) == 1) << "scale_a must be contiguous";
  CHECK_CONTIGUOUS(scale_b);
  TVM_FFI_ICHECK(dtype_equal(scale_a.dtype(), dl_float32)) << "scale_a must be float32";
  TVM_FFI_ICHECK(dtype_equal(scale_b.dtype(), dl_float32)) << "scale_b must be float32";
  TVM_FFI_ICHECK(is_fp8_dtype(a.dtype())) << "a must be fp8";
  TVM_FFI_ICHECK(is_fp8_dtype(b.dtype())) << "b must be fp8";
  TVM_FFI_ICHECK(dtype_equal(out.dtype(), dl_float32)) << "out must be float32";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a = a.dtype();
  args.type_b = b.dtype();
  args.type_d = out.dtype();

  args.num_expert = static_cast<int>(group_k_idx.size(0));
  args.m          = static_cast<int>(a.size(1));
  args.n          = static_cast<int>(b.size(1));
  args.k          = static_cast<int>(b.size(0));

  if (args.k == 0) {
    return;
  }

  args.quant_tile = static_cast<int>(std::get<2>(scale_granularity_mnk));
  TVM_FFI_ICHECK_EQ(args.quant_tile, 128) << "quant_tile must be 128";
  args.alignment_m     = 0;
  args.expected_m      = 0;
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = static_cast<int>(num_mp.value_or(args.total_mp_count));

  args.stride_m_a       = a.stride(1);
  args.stride_k_a       = a.stride(0);
  args.stride_batch_a   = args.m * args.k;
  args.stride_n_b       = b.stride(1);
  args.stride_k_b       = b.stride(0);
  args.stride_batch_b   = args.n * args.k;
  args.stride_m_out     = args.n;
  args.stride_batch_out = args.m * args.n;

  args.major_a       = TensorMajor::MN;
  args.major_b       = TensorMajor::MN;
  args.quant_mode_a  = TensorQuantMode::GROUP;
  args.quant_mode_b  = TensorQuantMode::GROUP;
  args.major_scale_a = TensorMajor::MN;
  args.major_scale_b = TensorMajor::MN;

  args.scale_a_m       = static_cast<int>(scale_a.size(1));
  args.scale_a_k       = static_cast<int>(scale_a.size(0));
  args.scale_a_nr_elem = static_cast<int>(scale_a.numel());
  args.scale_b_n       = static_cast<int>(scale_b.size(1));
  args.scale_b_k       = static_cast<int>(scale_b.size(0));
  args.scale_b_nr_elem = static_cast<int>(scale_b.numel());

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = scale_a.data_ptr();
  args.p_scale_b   = scale_b.data_ptr();
  args.p_m_indices = group_k_idx.data_ptr();
  args.p_signal    = nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::KContig>;
  run_moe_gemm_kernel<Kernel>(args, a.device());
}

void ragged_moe_gemm_16bit(ffi::TensorView        a,
                           ffi::TensorView        b,
                           ffi::TensorView        ragged_tokens_info,
                           ffi::TensorView        out,
                           bool                   use_psum_layout,
                           std::optional<int64_t> expected_m_for_psum_layout,
                           int64_t                alignment_m) {
  check_mp31(a.device(), "ragged_moe_gemm_16bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(ragged_tokens_info);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, ragged_tokens_info);
  CHECK_DEVICE(a, out);
  CHECK_DIM(2, a);
  CHECK_DIM(3, b);
  CHECK_DIM(1, ragged_tokens_info);
  CHECK_DIM(2, out);
  TVM_FFI_ICHECK_EQ(a.stride(-1), 1) << "a must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(b.stride(-1), 1) << "b must be contiguous at the last dimension";
  CHECK_CONTIGUOUS(ragged_tokens_info);
  CHECK_CONTIGUOUS(out);
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(a.dtype())) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(b.dtype())) << "b must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(out.dtype())) << "out must be bf16 or fp16";
  TVM_FFI_ICHECK(dtype_equal(a.dtype(), b.dtype()) && dtype_equal(a.dtype(), out.dtype()))
      << "a, b, out must have the same dtype";
  TVM_FFI_ICHECK(dtype_equal(ragged_tokens_info.dtype(), dl_int32)) << "ragged_tokens_info must be int32";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a     = a.dtype();
  args.type_b     = b.dtype();
  args.type_d     = out.dtype();
  args.m          = static_cast<int>(a.size(0));
  args.n          = static_cast<int>(b.size(1));
  args.k          = static_cast<int>(a.size(1));
  args.num_expert = static_cast<int>(b.size(0));

  TVM_FFI_ICHECK_EQ(b.size(2), args.k);
  TVM_FFI_ICHECK_EQ(out.size(0), args.m);
  TVM_FFI_ICHECK_EQ(out.size(1), args.n);

  if (gemm_common::gemm_early_return(args.m, args.n, args.k, out)) {
    return;
  }

  TVM_FFI_ICHECK(!use_psum_layout) << "use_psum_layout must be false";
  if (use_psum_layout) {
    TVM_FFI_ICHECK_EQ(ragged_tokens_info.size(0), args.num_expert);
    TVM_FFI_ICHECK(expected_m_for_psum_layout.has_value()) << "expected_m_for_psum_layout must be set";
  } else {
    TVM_FFI_ICHECK_EQ(ragged_tokens_info.size(0), args.m);
  }

  args.alignment_m = static_cast<int>(alignment_m);
  TVM_FFI_ICHECK(args.alignment_m == 128 || args.alignment_m == 256) << "alignment_m must be 128 or 256";

  args.major_a         = TensorMajor::K;
  args.major_b         = TensorMajor::K;
  args.quant_mode_a    = TensorQuantMode::NO_QUANT;
  args.quant_mode_b    = TensorQuantMode::NO_QUANT;
  args.major_scale_a   = TensorMajor::K;
  args.major_scale_b   = TensorMajor::K;
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = args.total_mp_count;

  args.stride_m_a       = a.stride(0);
  args.stride_batch_a   = leading_unsqueezed_batch_stride(a);
  args.stride_n_b       = b.stride(1);
  args.stride_batch_b   = b.stride(0);
  args.stride_m_out     = out.stride(0);
  args.stride_batch_out = leading_unsqueezed_batch_stride(out);

  args.scale_a_m = args.scale_a_k = args.scale_a_nr_elem = 0;
  args.scale_b_n = args.scale_b_k = args.scale_b_nr_elem = 0;

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = nullptr;
  args.p_scale_b   = nullptr;
  args.p_m_indices = ragged_tokens_info.data_ptr();
  args.p_signal    = nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::Ragged>;
  run_moe_gemm_kernel<Kernel>(args, a.device());
}

std::optional<std::tuple<int64_t, int64_t>> masked_moe_gemm_16bit(ffi::TensorView                a,
                                                                  ffi::TensorView                b,
                                                                  ffi::TensorView                masked_tokens_info,
                                                                  ffi::TensorView                out,
                                                                  int64_t                        expect_tokens,
                                                                  std::optional<ffi::TensorView> signal) {
  check_mp31(a.device(), "masked_moe_gemm_16bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(masked_tokens_info);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, masked_tokens_info);
  CHECK_DEVICE(a, out);
  CHECK_DIM(3, a);
  CHECK_DIM(3, b);
  CHECK_DIM(1, masked_tokens_info);
  CHECK_DIM(3, out);
  TVM_FFI_ICHECK_EQ(a.stride(-1), 1) << "a must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(b.stride(-1), 1) << "b must be contiguous at the last dimension";
  CHECK_CONTIGUOUS(masked_tokens_info);
  CHECK_CONTIGUOUS(out);
  if (signal.has_value()) {
    CHECK_MUSA(signal.value());
    CHECK_CONTIGUOUS(signal.value());
    CHECK_DEVICE(a, signal.value());
    CHECK_DIM(1, signal.value());
  }
  TVM_FFI_ICHECK(dtype_equal(masked_tokens_info.dtype(), dl_int32)) << "masked_tokens_info must be int32";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(a.dtype())) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(b.dtype())) << "b must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(out.dtype())) << "out must be bf16 or fp16";
  TVM_FFI_ICHECK(dtype_equal(a.dtype(), b.dtype()) && dtype_equal(a.dtype(), out.dtype()))
      << "a, b, out must have the same dtype";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a = a.dtype();
  args.type_b = b.dtype();
  args.type_d = out.dtype();

  const int max_m = static_cast<int>(a.size(1));
  args.num_expert = static_cast<int>(a.size(0));
  args.m          = max_m * args.num_expert;
  args.n          = static_cast<int>(b.size(1));
  args.k          = static_cast<int>(b.size(2));

  TVM_FFI_ICHECK_EQ(a.size(2), args.k);
  TVM_FFI_ICHECK_EQ(b.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(out.size(0), args.num_expert);
  TVM_FFI_ICHECK_EQ(out.size(1), max_m);
  TVM_FFI_ICHECK_EQ(out.size(2), args.n);
  TVM_FFI_ICHECK_EQ(masked_tokens_info.size(0), args.num_expert);
  if (signal.has_value()) {
    constexpr int tile_signal = 64;
    TVM_FFI_ICHECK_EQ(signal.value().size(0), args.num_expert * mutlass::ceil_div(max_m, tile_signal));
  }

  if (gemm_common::gemm_early_return(max_m, args.n, args.k, out)) {
    return std::nullopt;
  }

  args.alignment_m     = 0;
  args.expected_m      = static_cast<int>(expect_tokens);
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = args.total_mp_count;

  args.stride_m_a       = a.stride(1);
  args.stride_batch_a   = leading_unsqueezed_batch_stride(a);
  args.stride_n_b       = b.stride(1);
  args.stride_batch_b   = b.stride(0);
  args.stride_m_out     = out.stride(1);
  args.stride_batch_out = leading_unsqueezed_batch_stride(out);

  args.major_a       = TensorMajor::K;
  args.major_b       = TensorMajor::K;
  args.quant_mode_a  = TensorQuantMode::NO_QUANT;
  args.quant_mode_b  = TensorQuantMode::NO_QUANT;
  args.major_scale_a = TensorMajor::K;
  args.major_scale_b = TensorMajor::K;
  args.scale_a_m = args.scale_a_k = args.scale_a_nr_elem = 0;
  args.scale_b_n = args.scale_b_k = args.scale_b_nr_elem = 0;

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_scale_a   = nullptr;
  args.p_scale_b   = nullptr;
  args.p_m_indices = masked_tokens_info.data_ptr();
  args.p_signal    = signal.has_value() ? signal.value().data_ptr() : nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::Masked>;
  run_moe_gemm_kernel<Kernel>(args, a.device());

  if (signal.has_value()) {
    auto [_, config, __] = Kernel::to_underlying_arguments(args, get_stream(a.device()));
    return std::make_pair(config.block.tile_m, mutlass::ceil_div(args.n, config.block.tile_n));
  }
  return std::nullopt;
}

void m_grouped_contig_gemm_16bit(ffi::TensorView        a,
                                 ffi::TensorView        b,
                                 ffi::TensorView        group_m_idx,
                                 ffi::TensorView        out,
                                 const std::string&     major_mode_a,
                                 const std::string&     major_mode_b,
                                 std::optional<int64_t> num_mp) {
  (void)major_mode_a;
  check_mp31(a.device(), "m_grouped_contig_gemm_16bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(group_m_idx);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, group_m_idx);
  CHECK_DEVICE(a, out);
  CHECK_DIM(2, a);
  CHECK_DIM(1, group_m_idx);
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(a.dtype())) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(b.dtype())) << "b must be bf16 or fp16";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a     = a.dtype();
  args.type_b     = b.dtype();
  args.type_d     = out.dtype();
  args.num_expert = static_cast<int>(group_m_idx.size(0));
  args.m          = static_cast<int>(a.size(0));
  args.n          = major_mode_b[0] == 'K' ? static_cast<int>(b.size(-2)) : static_cast<int>(b.size(-1));
  args.k          = static_cast<int>(a.size(1));

  if (gemm_common::gemm_early_return(args.m, args.n, args.k, out)) {
    return;
  }

  args.quant_tile      = 128;
  args.alignment_m     = 0;
  args.expected_m      = 0;
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = static_cast<int>(num_mp.value_or(args.total_mp_count));

  args.stride_m_a       = args.k;
  args.stride_k_a       = args.m;
  args.stride_batch_a   = args.m * args.k;
  args.stride_n_b       = args.k;
  args.stride_k_b       = args.n;
  args.stride_batch_b   = args.n * args.k;
  args.stride_m_out     = args.n;
  args.stride_batch_out = args.m * args.n;

  args.major_a       = TensorMajor::K;
  args.major_b       = major_mode_b[0] == 'K' ? TensorMajor::K : TensorMajor::MN;
  args.quant_mode_a  = TensorQuantMode::NO_QUANT;
  args.quant_mode_b  = TensorQuantMode::NO_QUANT;
  args.major_scale_a = TensorMajor::K;
  args.major_scale_b = major_mode_b[0] == 'K' ? TensorMajor::K : TensorMajor::MN;

  args.scale_a_m = args.scale_a_k = args.scale_a_nr_elem = 0;
  args.scale_b_n = args.scale_b_k = args.scale_b_nr_elem = 0;

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_m_indices = group_m_idx.data_ptr();
  args.p_scale_a   = nullptr;
  args.p_scale_b   = nullptr;
  args.p_signal    = nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::RaggedPSumLayout>;
  run_moe_gemm_kernel<Kernel>(args, a.device());
}

void k_grouped_contig_gemm_16bit(ffi::TensorView        a,
                                 ffi::TensorView        b,
                                 ffi::TensorView        group_k_idx,
                                 ffi::TensorView        out,
                                 std::optional<int64_t> num_mp) {
  check_mp31(a.device(), "k_grouped_contig_gemm_16bit");
  CHECK_MUSA(a);
  CHECK_MUSA(b);
  CHECK_MUSA(group_k_idx);
  CHECK_MUSA(out);
  CHECK_DEVICE(a, b);
  CHECK_DEVICE(a, group_k_idx);
  CHECK_DEVICE(a, out);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(1, group_k_idx);
  TVM_FFI_ICHECK_EQ(a.stride(-1), 1) << "a must be contiguous at the last dimension";
  TVM_FFI_ICHECK_EQ(b.stride(-1), 1) << "b must be contiguous at the last dimension";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(a.dtype())) << "a must be bf16 or fp16";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(b.dtype())) << "b must be bf16 or fp16";
  TVM_FFI_ICHECK(dtype_equal(out.dtype(), dl_float32)) << "out must be float32";

  mate::moe_gemm::MoeGemmArgs args{};
  args.type_a     = a.dtype();
  args.type_b     = b.dtype();
  args.type_d     = out.dtype();
  args.num_expert = static_cast<int>(group_k_idx.size(0));
  args.m          = static_cast<int>(a.size(1));
  args.n          = static_cast<int>(b.size(1));
  args.k          = static_cast<int>(b.size(0));

  if (args.k == 0) {
    return;
  }

  args.quant_tile      = 128;
  args.alignment_m     = 0;
  args.expected_m      = 0;
  args.total_mp_count  = current_num_mps(a.device());
  args.target_mp_count = static_cast<int>(num_mp.value_or(args.total_mp_count));

  args.stride_m_a       = a.stride(1);
  args.stride_k_a       = a.stride(0);
  args.stride_batch_a   = args.m * args.k;
  args.stride_n_b       = b.stride(1);
  args.stride_k_b       = b.stride(0);
  args.stride_batch_b   = args.n * args.k;
  args.stride_m_out     = args.n;
  args.stride_batch_out = args.m * args.n;

  args.major_a       = TensorMajor::MN;
  args.major_b       = TensorMajor::MN;
  args.quant_mode_a  = TensorQuantMode::NO_QUANT;
  args.quant_mode_b  = TensorQuantMode::NO_QUANT;
  args.major_scale_a = TensorMajor::MN;
  args.major_scale_b = TensorMajor::MN;
  args.scale_a_m = args.scale_a_k = args.scale_a_nr_elem = 0;
  args.scale_b_n = args.scale_b_k = args.scale_b_nr_elem = 0;

  args.p_a         = a.data_ptr();
  args.p_b         = b.data_ptr();
  args.p_d         = out.data_ptr();
  args.p_m_indices = group_k_idx.data_ptr();
  args.p_scale_a   = nullptr;
  args.p_scale_b   = nullptr;
  args.p_signal    = nullptr;

  using Kernel = mate::moe_gemm::mubin::MoeGemmAsmKernel<mate::moe_gemm::MoeGemmMode::KContig>;
  run_moe_gemm_kernel<Kernel>(args, a.device());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_moe_gemm_16bit, ragged_moe_gemm_16bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(masked_moe_gemm_16bit, masked_moe_gemm_16bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_grouped_contig_gemm_16bit, k_grouped_contig_gemm_16bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_contig_gemm_16bit, m_grouped_contig_gemm_16bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_moe_gemm_8bit, ragged_moe_gemm_8bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(masked_moe_gemm_8bit, masked_moe_gemm_8bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(k_grouped_contig_gemm_8bit, k_grouped_contig_gemm_8bit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_contig_gemm_8bit, m_grouped_contig_gemm_8bit);
