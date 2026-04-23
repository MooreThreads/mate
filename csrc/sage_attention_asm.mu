#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mutlass/fast_math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "asm_common.hpp"
#include "mubin/mp31/mp31_sage_attention_registry.hpp"
#include "op_utils.hpp"

namespace mate::sage_attention {

// ============================================================================
// Quantized Sage Attention Arguments
// ============================================================================

struct SageAttenQuantizedASMArgs {
  bool    is_causal{};
  bool    is_kv_cache{};
  bool    is_qk_int8{};
  int32_t quant_mode;  // 0/7: extra ASM modes, 1: unit-block recipe, 2: per-block 128, 6: per-block 128 + per-thread 16

  DLDataType q_data_type{};
  DLDataType k_data_type{};
  DLDataType v_data_type{};
  DLDataType out_data_type{};

  int32_t num_mp{};

  // shape (bshd layout)
  int32_t batch{};
  int32_t nr_heads{};
  int32_t seqlen_q{};
  int32_t headdim_qk{};

  int32_t seqlen_kv{};
  int32_t nr_heads_kv{};

  int32_t headdim_v{};

  // stride
  int32_t batch_stride_q{};
  int32_t head_stride_q{};
  int32_t seq_stride_q{};

  int32_t batch_stride_k{};
  int32_t head_stride_k{};
  int32_t seq_stride_k{};

  int32_t batch_stride_v{};
  int32_t head_stride_v{};
  int32_t seq_stride_v{};

  // scale strides
  int32_t batch_stride_q_scale{};
  int32_t head_stride_q_scale{};
  int32_t seq_stride_q_scale{};  // For per-block quantization

  int32_t batch_stride_k_scale{};
  int32_t head_stride_k_scale{};
  int32_t seq_stride_k_scale{};  // For per-block/per-thread quantization

  int32_t batch_stride_v_scale{};
  int32_t seq_stride_v_scale{};
  int32_t head_stride_v_scale{};

  // output strides
  int32_t batch_stride_out{};
  int32_t head_stride_out{};
  int32_t seq_stride_out{};

  // for kvcache (paged attention)
  int32_t num_blocks_stride_k_cache{};
  int32_t page_block_size_stride_k_cache{};
  int32_t nheads_k_stride_k_cache{};
  int32_t batch_stride_block_table{};
  int32_t page_block_size{};
  int32_t num_blocks{};

  int32_t nr_k{};
  int32_t nr_q_scale{};
  int32_t nr_k_scale{};
  int32_t nr_v_scale{};

  double softmax_scale{};

  // data pointers
  void* p_q{};
  void* p_k{};
  void* p_v{};

  // kvcache pointers
  void*   p_k_cache{};
  void*   p_v_cache{};
  void*   p_page_table{};
  int32_t nr_page_table{};
  void*   p_cache_seqlens{};
  int32_t nr_cache_seqlens{};

  // scale pointers
  void* p_q_scale{};
  void* p_k_scale{};
  void* p_v_scale{};

  // output pointers
  void* p_output{};
  void* p_lse_output{};

};  // struct SageAttenQuantizedASMArgs

// ============================================================================
// Sage Attention ASM Dispatcher
// ============================================================================

namespace mubin {

template <class Args>
struct SageAttenQuantizedASMDispatcher {
  SageAttenQuantizedASMDispatcher() {
    init_fa_asm_kern_registry();
  }

  struct Config {
    int nr_thr{};

    int tile_m{};
    int tile_n{};
    int tile_k{};
    int tile_hdim{};

    MUfunction* asm_func{};

  };  // struct Config

  Config get_kernel_config(const Args& args) {
    Config config;

    // Reuse the 128-dim kernel bucket for all supported head dims up to 128.
    const int headdim_qk = args.headdim_qk <= 128 ? 128 : args.headdim_qk;

    if (headdim_qk == 128) {
      config.nr_thr    = 512;
      config.tile_m    = 256;
      config.tile_n    = 128;
      config.tile_k    = 64;
      config.tile_hdim = 128;
    } else {
      config.nr_thr    = 512;
      config.tile_m    = 256;
      config.tile_n    = 64;
      config.tile_k    = 64;
      config.tile_hdim = 128;
    }

    FlashAttenAsmID id;
    id.is_causal   = static_cast<int>(args.is_causal);
    id.is_kv_cache = static_cast<int>(args.is_kv_cache);
    id.is_varlen   = 0;  // SageAttention quantized path currently does not support varlen
    id.headdim_qk  = headdim_qk == 128 ? 0 : 1;
    id.dtype       = getDataTypeEnum(args.v_data_type);
    id.quant_mode  = args.quant_mode;
    id.is_qk_int8  = static_cast<int>(args.is_qk_int8);

    auto res = kernel_map.find(id);
    if (res != kernel_map.end()) {
      config.asm_func = &res->second.asm_func;
      return config;
    } else {
      auto asm_meta = fa_asm_kern_registry.find(id);
      if (asm_meta == fa_asm_kern_registry.end()) {
        throw std::runtime_error("SageAttenQuantizedASMDispatcher() mubin kernel not found!");
      }

      auto [curr_asm_meta, success] = kernel_map.try_emplace(id, asm_meta->second.first, asm_meta->second.second);
      config.asm_func               = &curr_asm_meta->second.asm_func;
      return config;
    }
  }

  std::unordered_map<FlashAttenAsmID, MateAsmKernel> kernel_map;

};  // struct SageAttenQuantizedASMDispatcher

// ============================================================================
// Sage Attention Quantized Kernel (Non-KVCache)
// ============================================================================

template <class Args_ = SageAttenQuantizedASMArgs, class Dispatcher = SageAttenQuantizedASMDispatcher<Args_>>
class SageAttentionQuantizedAsmKernel {
 public:
  using Args         = Args_;
  using Config       = typename Dispatcher::Config;
  using LaunchConfig = MUlaunchConfig;

  struct Params {
    void* output_ptr;
    void* out_lse_ptr;

    mute::RobustReg robust_mask;  // Not used by the dense SageAttention path.
    mute::RobustReg robust_key;
    mute::RobustReg robust_q_scale;
    mute::RobustReg robust_k_scale;
    mute::RobustReg robust_v_scale;
    bool            is_qk_int8;

    MUtensorDescriptor q_desc;
    MUtensorDescriptor k_desc;
    MUtensorDescriptor k_desc_1;
    MUtensorDescriptor v_desc;
    MUtensorDescriptor out_desc;

    // Scale strides
    int32_t q_scale_batch_stride;
    int32_t q_scale_seq_stride;
    int32_t q_scale_head_stride;
    int32_t k_scale_batch_stride;
    int32_t k_scale_seq_stride;
    int32_t k_scale_head_stride;
    int32_t v_scale_batch_stride;
    int32_t v_scale_seq_stride;
    int32_t v_scale_head_stride;

    float scale;
    float rln2_scale;
    float nr_ln2;

    int32_t batch;
    int32_t nheads;
    int32_t kv_heads;
    int32_t seqlen_q;
    int32_t seqlen_kv;
    int32_t headdim_v;

    int32_t  head_group_ori;
    uint32_t head_group_mul;
    uint32_t head_group_sft;

    int32_t ldg_key_stride;

    int32_t batch_stride_k;
    int32_t head_stride_k;
    int32_t seq_stride_k;

    int64_t batch_stride_mask;
    int64_t head_stride_mask;
    int32_t seq_stride_mask;

    int32_t batch_stride_out;
    int32_t head_stride_out;
    int32_t seq_stride_out;

    int32_t tile_size_q;
    int32_t tile_size_k;
    int32_t tile_size_v;
    int32_t tile_size_key_scale;
    int32_t tile_num;

    int32_t  double_block_num_sub1;
    int32_t  double_block_num_div;
    uint32_t double_block_num_mul;
    uint32_t double_block_num_sft;

    int32_t  seqq_mul_heads_div;
    uint32_t seqq_mul_heads_mul;
    uint32_t seqq_mul_heads_sft;

    int32_t  seqq_div;
    uint32_t seqq_mul;
    uint32_t seqq_sft;

    int32_t tile_q_dim0;
    int32_t tile_q_dim1;
    int32_t tile_q_dim2;
    int32_t tile_q_dim3;

    int32_t tile_k_dim0;
    int32_t tile_k_dim1;
    int32_t tile_k_dim2;
    int32_t tile_k_dim3;

    int32_t tile_v_dim0;
    int32_t tile_v_dim1;
    int32_t tile_v_dim2;
    int32_t tile_v_dim3;

    int32_t tile_out_dim0;
    int32_t tile_out_dim1;
    int32_t tile_out_dim2;
    int32_t tile_out_dim3;

  };  // struct Params

  SageAttentionQuantizedAsmKernel(const Params&       in_params,
                                  const Config&       in_config,
                                  const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const SageAttenQuantizedASMArgs& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    // Create tensor descriptors (bshd layout)
    TmeDesc tensor_desc_q(mute::make_tuple(args.headdim_qk, args.seqlen_q, args.nr_heads, args.batch),
                          mute::make_tuple(args.seq_stride_q, args.head_stride_q, args.batch_stride_q),
                          args.p_q,
                          dl_dtype_to_tme_type(args.q_data_type));
    TmeDesc tensor_desc_k(mute::make_tuple(args.headdim_qk, args.seqlen_kv, args.nr_heads_kv, args.batch),
                          mute::make_tuple(args.seq_stride_k, args.head_stride_k, args.batch_stride_k),
                          args.p_k,
                          dl_dtype_to_tme_type(args.k_data_type));
    TmeDesc tensor_desc_v(mute::make_tuple(args.headdim_v, args.seqlen_kv, args.nr_heads_kv, args.batch),
                          mute::make_tuple(args.seq_stride_v, args.head_stride_v, args.batch_stride_v),
                          args.p_v,
                          dl_dtype_to_tme_type(args.v_data_type));
    TmeDesc tensor_desc_out(mute::make_tuple(args.headdim_v, args.seqlen_q, args.nr_heads, args.batch),
                            mute::make_tuple(args.seq_stride_out, args.head_stride_out, args.batch_stride_out),
                            args.p_output,
                            dl_dtype_to_tme_type(args.out_data_type));

    params.output_ptr  = args.p_output;
    params.out_lse_ptr = args.p_lse_output;

    params.q_desc     = tensor_desc_q.desc;
    params.k_desc     = tensor_desc_k.desc;
    params.k_desc_1   = tensor_desc_k.desc;
    params.v_desc     = tensor_desc_v.desc;
    params.out_desc   = tensor_desc_out.desc;
    params.is_qk_int8 = args.is_qk_int8;

    if (args.is_qk_int8) {
      params.robust_key = mute::make_robust_desc(static_cast<int8_t*>(args.p_k), args.nr_k).reg;
    } else {
      params.robust_key = mute::make_robust_desc(static_cast<mutlass::float_e4m3_t*>(args.p_k), args.nr_k).reg;
    }
    params.robust_mask = mute::make_robust_desc(static_cast<int32_t*>(nullptr), 0).reg;

    params.robust_q_scale = mute::make_robust_desc(static_cast<float*>(args.p_q_scale), args.nr_q_scale).reg;
    params.robust_k_scale = mute::make_robust_desc(static_cast<float*>(args.p_k_scale), args.nr_k_scale).reg;
    params.robust_v_scale = mute::make_robust_desc(static_cast<float*>(args.p_v_scale), args.nr_v_scale).reg;

    params.q_scale_batch_stride = args.batch_stride_q_scale;
    params.q_scale_seq_stride   = args.seq_stride_q_scale;
    params.q_scale_head_stride  = args.head_stride_q_scale;
    params.k_scale_batch_stride = args.batch_stride_k_scale;
    params.k_scale_seq_stride   = args.seq_stride_k_scale;
    params.k_scale_head_stride  = args.head_stride_k_scale;
    params.v_scale_batch_stride = args.batch_stride_v_scale;
    params.v_scale_seq_stride   = args.seq_stride_v_scale;
    params.v_scale_head_stride  = args.head_stride_v_scale;

    const double log2e = std::log2(std::exp(1.0));
    params.scale       = args.softmax_scale;
    params.rln2_scale  = args.softmax_scale * log2e;
    params.nr_ln2      = 1.0 / log2e;

    params.batch     = args.batch;
    params.nheads    = args.nr_heads;
    params.kv_heads  = args.nr_heads_kv;
    params.seqlen_q  = args.seqlen_q;
    params.seqlen_kv = args.seqlen_kv;
    params.headdim_v = args.headdim_v;

    int  head_group       = args.nr_heads / args.nr_heads_kv;
    auto fast_head_group  = mutlass::FastDivmod(head_group);
    params.head_group_ori = fast_head_group.divisor;
    params.head_group_mul = fast_head_group.multiplier;
    params.head_group_sft = fast_head_group.shift_right;

    const int qk_element_size =
        args.is_qk_int8 ? static_cast<int>(sizeof(int8_t)) : static_cast<int>(sizeof(mutlass::float_e4m3_t));
    params.ldg_key_stride = args.seq_stride_k * config.tile_n * qk_element_size;

    params.batch_stride_k = args.batch_stride_k;
    params.head_stride_k  = args.head_stride_k;
    params.seq_stride_k   = args.seq_stride_k;

    params.batch_stride_mask = 0;
    params.head_stride_mask  = 0;
    params.seq_stride_mask   = 0;

    params.batch_stride_out = args.batch_stride_out;
    params.head_stride_out  = args.head_stride_out;
    params.seq_stride_out   = args.seq_stride_out;

    params.tile_size_q         = config.tile_m * config.tile_k * qk_element_size;
    params.tile_size_k         = config.tile_n * config.tile_k * qk_element_size;
    params.tile_size_v         = config.tile_hdim * config.tile_k * sizeof(mutlass::float_e4m3_t);
    params.tile_size_key_scale = (args.quant_mode == 6 ? config.tile_n / 16 : config.tile_n) * sizeof(float);

    int seq_q_tile_num = mutlass::ceil_div(args.seqlen_q, config.tile_m);
    params.tile_num    = seq_q_tile_num * args.batch * args.nr_heads;

    // Match the non-persistent dense flash-attention wrapper in ComputeAsmKern.
    int block_num                = 1;
    params.double_block_num_sub1 = 2 * block_num - 1;
    int  double_block_num        = 2 * block_num;
    auto fast_double_block_num   = mutlass::FastDivmod(double_block_num);
    params.double_block_num_div  = fast_double_block_num.divisor;
    params.double_block_num_mul  = fast_double_block_num.multiplier;
    params.double_block_num_sft  = fast_double_block_num.shift_right;

    auto fast_seqq_mul_heads  = mutlass::FastDivmod(args.nr_heads * seq_q_tile_num);
    params.seqq_mul_heads_div = fast_seqq_mul_heads.divisor;
    params.seqq_mul_heads_mul = fast_seqq_mul_heads.multiplier;
    params.seqq_mul_heads_sft = fast_seqq_mul_heads.shift_right;

    auto fast_seqq  = mutlass::FastDivmod(seq_q_tile_num);
    params.seqq_div = fast_seqq.divisor;
    params.seqq_mul = fast_seqq.multiplier;
    params.seqq_sft = fast_seqq.shift_right;

    params.tile_q_dim0 = config.tile_k;
    params.tile_q_dim1 = config.tile_m;
    params.tile_q_dim2 = 1;
    params.tile_q_dim3 = 1;

    params.tile_k_dim0 = config.tile_k;
    params.tile_k_dim1 = config.tile_n;
    params.tile_k_dim2 = 1;
    params.tile_k_dim3 = 1;

    params.tile_v_dim0 = config.tile_hdim;
    params.tile_v_dim1 = config.tile_k;
    params.tile_v_dim2 = 1;
    params.tile_v_dim3 = 1;

    params.tile_out_dim0 = args.headdim_v;
    params.tile_out_dim1 = 4;
    params.tile_out_dim2 = 1;
    params.tile_out_dim3 = 1;

    // Match the dense flash-attention wrapper launch convention from ComputeAsmKern.
    launch_config.blockDimX = config.nr_thr;
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    if (args.is_causal) {
      launch_config.gridDimX = params.nheads;
      launch_config.gridDimY = params.batch;
      launch_config.gridDimZ = seq_q_tile_num;
    } else {
      launch_config.gridDimX = seq_q_tile_num;
      launch_config.gridDimY = params.nheads;
      launch_config.gridDimZ = params.batch;
    }
    launch_config.hStream        = reinterpret_cast<MUstream>(stream);
    launch_config.sharedMemBytes = 0;
    launch_config.attrs          = NULL;
    launch_config.numAttrs       = 0;

    return std::make_tuple(params, config, launch_config);

  }  // to_underlying_arguments

  void run() {
    void* kernel_params[] = {&params.output_ptr,
                             &params.out_lse_ptr,
                             &params.robust_mask,
                             &params.robust_key,
                             &params.robust_q_scale,
                             &params.robust_k_scale,
                             &params.robust_v_scale,
                             &params.q_desc,
                             &params.k_desc,
                             &params.v_desc,
                             &params.out_desc,
                             &params.q_scale_batch_stride,
                             &params.q_scale_seq_stride,
                             &params.q_scale_head_stride,
                             &params.k_scale_batch_stride,
                             &params.k_scale_seq_stride,
                             &params.k_scale_head_stride,
                             &params.v_scale_batch_stride,
                             &params.v_scale_seq_stride,
                             &params.v_scale_head_stride,
                             &params.scale,
                             &params.rln2_scale,
                             &params.nr_ln2,
                             &params.batch,
                             &params.nheads,
                             &params.kv_heads,
                             &params.seqlen_q,
                             &params.seqlen_kv,
                             &params.headdim_v,
                             &params.head_group_ori,
                             &params.head_group_mul,
                             &params.head_group_sft,
                             &params.ldg_key_stride,
                             &params.batch_stride_k,
                             &params.head_stride_k,
                             &params.seq_stride_k,
                             &params.batch_stride_mask,
                             &params.head_stride_mask,
                             &params.seq_stride_mask,
                             &params.batch_stride_out,
                             &params.head_stride_out,
                             &params.seq_stride_out,
                             &params.tile_size_q,
                             &params.tile_size_k,
                             &params.tile_size_v,
                             &params.tile_size_key_scale,
                             &params.tile_num,
                             &params.double_block_num_sub1,
                             &params.double_block_num_div,
                             &params.double_block_num_mul,
                             &params.double_block_num_sft,
                             &params.seqq_mul_heads_div,
                             &params.seqq_mul_heads_mul,
                             &params.seqq_mul_heads_sft,
                             &params.seqq_div,
                             &params.seqq_mul,
                             &params.seqq_sft,
                             &params.tile_q_dim0,
                             &params.tile_q_dim1,
                             &params.tile_q_dim2,
                             &params.tile_q_dim3,
                             &params.tile_k_dim0,
                             &params.tile_k_dim1,
                             &params.tile_k_dim2,
                             &params.tile_k_dim3,
                             &params.tile_v_dim0,
                             &params.tile_v_dim1,
                             &params.tile_v_dim2,
                             &params.tile_v_dim3,
                             &params.tile_out_dim0,
                             &params.tile_out_dim1,
                             &params.tile_out_dim2,
                             &params.tile_out_dim3};

    MATE_MUSA_DRIVER_CHECK(muLaunchKernelEx(&launch_config, *config.asm_func, kernel_params, nullptr));
  }

 protected:
  Params       params;
  Config       config;
  LaunchConfig launch_config;

};  // class SageAttentionQuantizedAsmKernel

// ============================================================================
// Sage Attention Quantized Kernel with KV Cache (Paged Attention)
// ============================================================================

template <class Args_ = SageAttenQuantizedASMArgs, class Dispatcher = SageAttenQuantizedASMDispatcher<Args_>>
class SageAttentionQuantizedWithKVCacheAsmKernel {
 public:
  using Args         = Args_;
  using Config       = typename Dispatcher::Config;
  using LaunchConfig = MUlaunchConfig;

  struct Params {
    void* output_ptr;
    void* out_lse_ptr;

    MUtensorDescriptor q_desc;
    MUtensorDescriptor k_desc;
    MUtensorDescriptor v_desc;
    MUtensorDescriptor out_desc;

    mute::RobustReg robust_mask;
    mute::RobustReg robust_key;
    mute::RobustReg robust_block_table;
    mute::RobustReg robust_cache_seqlens;
    // Keep these packed slots reserved so existing kernel parameter layouts stay stable.
    mute::RobustReg robust_reserved_0;
    mute::RobustReg robust_reserved_1;

    mute::RobustReg robust_q_scale;
    mute::RobustReg robust_k_scale;
    mute::RobustReg robust_v_scale;

    int32_t batch_stride_block_table;
    int32_t num_blocks_stride_k_cache;
    int32_t page_block_size_stride_k_cache;
    int32_t nheads_k_stride_k_cache;

    // Scale strides
    int32_t q_scale_batch_stride;
    int32_t q_scale_seq_stride;
    int32_t q_scale_head_stride;

    int32_t k_scale_block_stride;
    int32_t k_scale_page_stride;
    int32_t k_scale_head_stride;

    int32_t v_scale_batch_stride;
    int32_t v_scale_head_stride;

    float scale;
    float rln2_scale;
    float nr_ln2;

    int32_t batch;
    int32_t nheads;
    int32_t kv_heads;
    int32_t seqlen_q;
    int32_t seqlen_kv;
    int32_t headdim_v;

    int32_t  head_group_ori;
    uint32_t head_group_mul;
    uint32_t head_group_sft;

    int64_t batch_stride_mask;
    int64_t head_stride_mask;
    int32_t seq_stride_mask;

    int32_t batch_stride_out;
    int32_t head_stride_out;
    int32_t seq_stride_out;

    int32_t tile_size_q;
    int32_t tile_size_k;
    int32_t tile_size_v;
    int32_t tile_size_key_scale;

    int32_t  tile_num;
    int32_t  double_block_num_sub1;
    int32_t  double_block_num_div;
    uint32_t double_block_num_mul;
    uint32_t double_block_num_sft;

    int32_t  seqq_mul_heads_div;
    uint32_t seqq_mul_heads_mul;
    uint32_t seqq_mul_heads_sft;

    int32_t  seqq_div;
    uint32_t seqq_mul;
    uint32_t seqq_sft;

    int32_t tile_q_dim0;
    int32_t tile_q_dim1;
    int32_t tile_q_dim2;
    int32_t tile_q_dim3;

    int32_t tile_k_dim0;
    int32_t tile_k_dim1;
    int32_t tile_k_dim2;
    int32_t tile_k_dim3;

    int32_t tile_v_dim0;
    int32_t tile_v_dim1;
    int32_t tile_v_dim2;
    int32_t tile_v_dim3;

    int32_t tile_out_dim0;
    int32_t tile_out_dim1;
    int32_t tile_out_dim2;
    int32_t tile_out_dim3;

  };  // struct Params

  SageAttentionQuantizedWithKVCacheAsmKernel(const Params&       in_params,
                                             const Config&       in_config,
                                             const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const SageAttenQuantizedASMArgs& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    // Create tensor descriptors for paged KV cache
    TmeDesc tensor_desc_q(mute::make_tuple(args.headdim_qk, args.nr_heads, args.seqlen_q, args.batch),
                          mute::make_tuple(args.head_stride_q, args.seq_stride_q, args.batch_stride_q),
                          args.p_q,
                          dl_dtype_to_tme_type(args.q_data_type));
    TmeDesc tensor_desc_v(mute::make_tuple(args.headdim_v, args.nr_heads_kv, args.page_block_size, args.num_blocks),
                          mute::make_tuple(args.headdim_v,
                                           args.nr_heads_kv * args.headdim_v,
                                           args.nr_heads_kv * args.headdim_v * args.page_block_size),
                          args.p_v_cache,
                          dl_dtype_to_tme_type(args.v_data_type));
    TmeDesc tensor_desc_out(mute::make_tuple(args.headdim_v, args.nr_heads, args.seqlen_q, args.batch),
                            mute::make_tuple(args.head_stride_out, args.seq_stride_out, args.batch_stride_out),
                            args.p_output,
                            dl_dtype_to_tme_type(args.out_data_type));

    params.output_ptr  = args.p_output;
    params.out_lse_ptr = args.p_lse_output;

    params.q_desc   = tensor_desc_q.desc;
    params.v_desc   = tensor_desc_v.desc;
    params.out_desc = tensor_desc_out.desc;

    params.robust_key  = mute::make_robust_desc(static_cast<mutlass::float_e4m3_t*>(args.p_k_cache), args.nr_k).reg;
    params.robust_mask = mute::make_robust_desc(static_cast<int32_t*>(nullptr), 0).reg;
    params.robust_block_table =
        mute::make_robust_desc(static_cast<int32_t*>(args.p_page_table), args.nr_page_table).reg;
    params.robust_cache_seqlens =
        mute::make_robust_desc(static_cast<int32_t*>(args.p_cache_seqlens), args.nr_cache_seqlens).reg;

    params.robust_reserved_0 = mute::make_robust_desc(static_cast<int32_t*>(nullptr), 0).reg;
    params.robust_reserved_1 = mute::make_robust_desc(static_cast<int32_t*>(nullptr), 0).reg;

    params.robust_q_scale = mute::make_robust_desc(static_cast<float*>(args.p_q_scale), args.nr_q_scale).reg;
    params.robust_k_scale = mute::make_robust_desc(static_cast<float*>(args.p_k_scale), args.nr_k_scale).reg;
    params.robust_v_scale = mute::make_robust_desc(static_cast<float*>(args.p_v_scale), args.nr_v_scale).reg;

    params.q_scale_batch_stride = args.batch_stride_q_scale;
    params.q_scale_seq_stride   = args.seq_stride_q_scale;
    params.q_scale_head_stride  = args.head_stride_q_scale;
    params.k_scale_block_stride = args.batch_stride_k_scale;
    params.k_scale_page_stride  = args.seq_stride_k_scale;
    params.k_scale_head_stride  = args.head_stride_k_scale;
    params.v_scale_batch_stride = args.batch_stride_v_scale;
    params.v_scale_head_stride  = args.head_stride_v_scale;

    params.batch_stride_block_table = args.batch_stride_block_table;

    params.num_blocks_stride_k_cache      = args.num_blocks_stride_k_cache;
    params.page_block_size_stride_k_cache = args.page_block_size_stride_k_cache;
    params.nheads_k_stride_k_cache        = args.nheads_k_stride_k_cache;

    const double log2e = std::log2(std::exp(1.0));
    params.scale       = args.softmax_scale;
    params.rln2_scale  = args.softmax_scale * log2e;
    params.nr_ln2      = 1.0 / log2e;

    params.batch     = args.batch;
    params.nheads    = args.nr_heads;
    params.kv_heads  = args.nr_heads_kv;
    params.seqlen_q  = args.seqlen_q;
    params.seqlen_kv = args.seqlen_kv;
    params.headdim_v = args.headdim_v;

    int  head_group       = args.nr_heads / args.nr_heads_kv;
    auto fast_head_group  = mutlass::FastDivmod(head_group);
    params.head_group_ori = fast_head_group.divisor;
    params.head_group_mul = fast_head_group.multiplier;
    params.head_group_sft = fast_head_group.shift_right;

    params.batch_stride_mask = 0;
    params.head_stride_mask  = 0;
    params.seq_stride_mask   = 0;

    params.batch_stride_out = args.batch_stride_out;
    params.head_stride_out  = args.head_stride_out;
    params.seq_stride_out   = args.seq_stride_out;

    params.tile_size_q         = config.tile_m * config.tile_k * sizeof(mutlass::float_e4m3_t);
    params.tile_size_k         = config.tile_n * config.tile_k * sizeof(mutlass::float_e4m3_t);
    params.tile_size_v         = config.tile_hdim * config.tile_k * sizeof(mutlass::float_e4m3_t);
    params.tile_size_key_scale = (args.quant_mode == 6 ? config.tile_n / 16 : config.tile_n) * sizeof(float);

    int seq_q_tile_num = mutlass::ceil_div(args.seqlen_q, config.tile_m);
    params.tile_num    = seq_q_tile_num * args.batch * args.nr_heads;

    // Match the non-persistent dense flash-attention wrapper in ComputeAsmKern.
    int block_num                = 1;
    params.double_block_num_sub1 = 2 * block_num - 1;
    int  double_block_num        = 2 * block_num;
    auto fast_double_block_num   = mutlass::FastDivmod(double_block_num);
    params.double_block_num_div  = fast_double_block_num.divisor;
    params.double_block_num_mul  = fast_double_block_num.multiplier;
    params.double_block_num_sft  = fast_double_block_num.shift_right;

    auto fast_seqq_mul_heads  = mutlass::FastDivmod(args.nr_heads * seq_q_tile_num);
    params.seqq_mul_heads_div = fast_seqq_mul_heads.divisor;
    params.seqq_mul_heads_mul = fast_seqq_mul_heads.multiplier;
    params.seqq_mul_heads_sft = fast_seqq_mul_heads.shift_right;

    auto fast_seqq  = mutlass::FastDivmod(seq_q_tile_num);
    params.seqq_div = fast_seqq.divisor;
    params.seqq_mul = fast_seqq.multiplier;
    params.seqq_sft = fast_seqq.shift_right;

    params.tile_q_dim0 = config.tile_k;
    params.tile_q_dim1 = 1;
    params.tile_q_dim2 = config.tile_m;
    params.tile_q_dim3 = 1;

    params.tile_k_dim0 = config.tile_k;
    params.tile_k_dim1 = 1;
    params.tile_k_dim2 = config.tile_n;
    params.tile_k_dim3 = 1;

    params.tile_v_dim0 = config.tile_hdim;
    params.tile_v_dim1 = 1;
    params.tile_v_dim2 = config.tile_k;
    params.tile_v_dim3 = 1;

    params.tile_out_dim0 = args.headdim_v;
    params.tile_out_dim1 = 1;
    params.tile_out_dim2 = 4;
    params.tile_out_dim3 = 1;

    // Launch configuration
    launch_config.blockDimX = config.nr_thr;
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    if (args.is_causal) {
      launch_config.gridDimX = block_num;
      launch_config.gridDimY = 1;
      launch_config.gridDimZ = 1;
    } else {
      launch_config.gridDimX = mutlass::ceil_div(params.seqlen_q, config.tile_m);
      launch_config.gridDimY = params.nheads;
      launch_config.gridDimZ = params.batch;
    }
    launch_config.hStream        = reinterpret_cast<MUstream>(stream);
    launch_config.sharedMemBytes = 0;
    launch_config.attrs          = NULL;
    launch_config.numAttrs       = 0;

    return std::make_tuple(params, config, launch_config);

  }  // to_underlying_arguments

  void run() {
    launch_asm(*config.asm_func, launch_config, params);
  }

 protected:
  Params       params;
  Config       config;
  LaunchConfig launch_config;

};  // class SageAttentionQuantizedWithKVCacheAsmKernel

}  // namespace mubin

}  // namespace mate::sage_attention

// ============================================================================
// Dispatch Functions for Quantized Sage Attention
// ============================================================================

void sage_attn_quantized_asm(ffi::TensorView out,
                             ffi::TensorView out_lse,
                             ffi::TensorView q,
                             ffi::TensorView k,
                             ffi::TensorView v,
                             double          softmax_scale,
                             ffi::TensorView q_scale,
                             ffi::TensorView k_scale,
                             ffi::TensorView v_scale,
                             bool            is_causal,
                             int64_t         quant_mode) {
  check_mp31(q.device(), "sage_attn_quantized_asm");
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v);
  CHECK_INPUT(q_scale);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale);
  CHECK_INPUT(out);
  CHECK_INPUT(out_lse);

  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, v);
  CHECK_DEVICE(q, q_scale);
  CHECK_DEVICE(q, k_scale);
  CHECK_DEVICE(q, v_scale);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, out_lse);

  const DLDataType q_dtype = q.dtype();
  const DLDataType k_dtype = k.dtype();
  const DLDataType v_dtype = v.dtype();
  const bool qk_int8_path  = dtype_equal(q_dtype, dl_int8) && dtype_equal(k_dtype, dl_int8) && is_fp8_dtype(v_dtype);

  if (qk_int8_path) {
    TVM_FFI_ICHECK(dtype_equal(q_dtype, k_dtype)) << "q and k must have same dtype";
  } else {
    TVM_FFI_ICHECK(dtype_equal(q_dtype, k_dtype)) << "q and k must have same dtype";
    TVM_FFI_ICHECK(dtype_equal(q_dtype, v_dtype)) << "q and v must have same dtype";
    TVM_FFI_ICHECK(is_fp8_dtype(q_dtype)) << "qkv must be e4m3 or e5m2";
  }
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(out.dtype())) << "out must be bf16 or fp16";

  using Args = mate::sage_attention::SageAttenQuantizedASMArgs;
  Args args;
  args.is_causal     = is_causal;
  args.is_kv_cache   = false;
  args.is_qk_int8    = qk_int8_path;
  args.quant_mode    = quant_mode;
  args.q_data_type   = q_dtype;
  args.k_data_type   = k_dtype;
  args.v_data_type   = v_dtype;
  args.out_data_type = out.dtype();

  musaDeviceProp prop{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&prop, q.device().device_id));
  args.num_mp = prop.multiProcessorCount;

  args.batch       = q.size(0);
  args.seqlen_q    = q.size(1);
  args.nr_heads    = q.size(2);
  args.headdim_qk  = q.size(3);
  args.seqlen_kv   = k.size(1);
  args.nr_heads_kv = k.size(2);
  args.headdim_v   = v.size(3);

  TVM_FFI_ICHECK_GT(args.nr_heads_kv, 0) << "nheads_kv must be positive";
  TVM_FFI_ICHECK_EQ(args.nr_heads % args.nr_heads_kv, 0) << "nheads must be divisible by nheads_kv";
  TVM_FFI_ICHECK_GT(args.headdim_qk, 0) << "headdim_qk must be positive";
  TVM_FFI_ICHECK_LE(args.headdim_qk, 128) << "headdim_qk must be <= 128";
  TVM_FFI_ICHECK_GT(args.headdim_v, 0) << "headdim_v must be positive";
  TVM_FFI_ICHECK_LE(args.headdim_v, 128) << "headdim_v must be <= 128";

  int q_seq_scale_num = mutlass::ceil_div(args.seqlen_q, 128);
  int k_seq_scale_num = mutlass::ceil_div(args.seqlen_kv, 128);
  int v_seq_scale_num = 1;
  if (quant_mode == 6) {
    k_seq_scale_num = mutlass::ceil_div(args.seqlen_kv, 128) * 128 / 16;
  } else if (quant_mode == 1) {
    q_seq_scale_num = args.seqlen_q;
    k_seq_scale_num = args.seqlen_kv;
    v_seq_scale_num = args.seqlen_kv;
  } else if (quant_mode == 0) {
    q_seq_scale_num = 1;
    k_seq_scale_num = 1;
  } else if (quant_mode == 7) {
    v_seq_scale_num = mutlass::ceil_div(args.seqlen_kv, 128);
  } else if (quant_mode != 2) {
    TVM_FFI_ICHECK(false) << "Unsupported dense SageAttention quant_mode: " << quant_mode;
  }

  expect_shape(q, {args.batch, args.seqlen_q, args.nr_heads, args.headdim_qk}, "q");
  expect_shape(k, {args.batch, args.seqlen_kv, args.nr_heads_kv, args.headdim_qk}, "k");
  expect_shape(v, {args.batch, args.seqlen_kv, args.nr_heads_kv, args.headdim_v}, "v");
  if (quant_mode == 0) {
    expect_shape(q_scale, {1, 1, 1, 1}, "q_scale");
    expect_shape(k_scale, {1, 1, 1, 1}, "k_scale");
    expect_shape(v_scale, {1, 1, 1, 1}, "v_scale");
  } else {
    expect_shape(q_scale, {args.batch, q_seq_scale_num, args.nr_heads, 1}, "q_scale");
    expect_shape(k_scale, {args.batch, k_seq_scale_num, args.nr_heads_kv, 1}, "k_scale");
    if (quant_mode == 7) {
      expect_shape(v_scale, {args.batch, v_seq_scale_num, args.nr_heads_kv, 2}, "v_scale");
    } else if (quant_mode == 1) {
      expect_shape(v_scale, {args.batch, v_seq_scale_num, args.nr_heads_kv, args.headdim_v}, "v_scale");
    } else {
      expect_shape(v_scale, {args.batch, 1, args.nr_heads_kv, args.headdim_v}, "v_scale");
    }
  }
  expect_shape(out, {args.batch, args.seqlen_q, args.nr_heads, args.headdim_v}, "out");
  expect_shape(out_lse, {args.batch, args.nr_heads, args.seqlen_q}, "out_lse");

  args.batch_stride_q       = q.stride(0);
  args.seq_stride_q         = q.stride(1);
  args.head_stride_q        = q.stride(2);
  args.batch_stride_k       = k.stride(0);
  args.seq_stride_k         = k.stride(1);
  args.head_stride_k        = k.stride(2);
  args.batch_stride_v       = v.stride(0);
  args.seq_stride_v         = v.stride(1);
  args.head_stride_v        = v.stride(2);
  args.batch_stride_q_scale = q_scale.size(0) == 1 ? 0 : q_scale.stride(0);
  args.seq_stride_q_scale   = q_scale.size(1) == 1 ? 0 : q_scale.stride(1);
  args.head_stride_q_scale  = q_scale.size(2) == 1 ? 0 : q_scale.stride(2);
  args.batch_stride_k_scale = k_scale.size(0) == 1 ? 0 : k_scale.stride(0);
  args.seq_stride_k_scale   = k_scale.size(1) == 1 ? 0 : k_scale.stride(1);
  args.head_stride_k_scale  = k_scale.size(2) == 1 ? 0 : k_scale.stride(2);
  args.batch_stride_v_scale = v_scale.size(0) == 1 ? 0 : v_scale.stride(0);
  args.seq_stride_v_scale   = v_scale.size(1) == 1 ? 0 : v_scale.stride(1);
  args.head_stride_v_scale  = v_scale.size(2) == 1 ? 0 : v_scale.stride(2);
  args.batch_stride_out     = out.stride(0);
  args.seq_stride_out       = out.stride(1);
  args.head_stride_out      = out.stride(2);

  args.nr_k          = static_cast<int32_t>(k.numel());
  args.nr_q_scale    = static_cast<int32_t>(q_scale.numel());
  args.nr_k_scale    = static_cast<int32_t>(k_scale.numel());
  args.nr_v_scale    = static_cast<int32_t>(v_scale.numel());
  args.softmax_scale = softmax_scale;

  args.p_q          = q.data_ptr();
  args.p_k          = k.data_ptr();
  args.p_v          = v.data_ptr();
  args.p_q_scale    = q_scale.data_ptr();
  args.p_k_scale    = k_scale.data_ptr();
  args.p_v_scale    = v_scale.data_ptr();
  args.p_output     = out.data_ptr();
  args.p_lse_output = out_lse.data_ptr();

  musaStream_t stream                 = get_stream(q.device());
  using Kernel                        = mate::sage_attention::mubin::SageAttentionQuantizedAsmKernel<Args>;
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();
}

void sage_attn_quantized_with_kvcache_asm(ffi::TensorView out,
                                          ffi::TensorView out_lse,
                                          ffi::TensorView q,
                                          ffi::TensorView k_cache,
                                          ffi::TensorView v_cache,
                                          ffi::TensorView page_table,
                                          ffi::TensorView cache_seqlens,
                                          ffi::TensorView q_scale,
                                          ffi::TensorView k_scale,
                                          ffi::TensorView v_scale,
                                          double          softmax_scale,
                                          bool            is_causal,
                                          int64_t         quant_mode) {
  check_mp31(q.device(), "sage_attn_quantized_with_kvcache_asm");
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_scale);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_scale);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_scale);
  CHECK_INPUT(out);
  CHECK_INPUT(out_lse);
  CHECK_INPUT(page_table);
  CHECK_INPUT(cache_seqlens);

  CHECK_DEVICE(q, k_cache);
  CHECK_DEVICE(q, v_cache);
  CHECK_DEVICE(q, page_table);
  CHECK_DEVICE(q, cache_seqlens);
  CHECK_DEVICE(q, q_scale);
  CHECK_DEVICE(q, k_scale);
  CHECK_DEVICE(q, v_scale);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, out_lse);

  TVM_FFI_ICHECK(dtype_equal(page_table.dtype(), dl_int32)) << "page_table must be int32";
  TVM_FFI_ICHECK(dtype_equal(cache_seqlens.dtype(), dl_int32)) << "cache_seqlens must be int32";

  using Args = mate::sage_attention::SageAttenQuantizedASMArgs;
  Args args;
  args.is_causal     = is_causal;
  args.is_kv_cache   = true;
  args.is_qk_int8    = false;
  args.quant_mode    = quant_mode;
  args.q_data_type   = q.dtype();
  args.k_data_type   = k_cache.dtype();
  args.v_data_type   = v_cache.dtype();
  args.out_data_type = out.dtype();

  TVM_FFI_ICHECK(is_fp8_dtype(args.q_data_type)) << "q must be e4m3 or e5m2";
  TVM_FFI_ICHECK(is_fp8_dtype(args.k_data_type)) << "k_cache must be e4m3 or e5m2";
  TVM_FFI_ICHECK(is_fp8_dtype(args.v_data_type)) << "v_cache must be e4m3 or e5m2";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(args.out_data_type)) << "out must be bf16 or fp16";

  musaDeviceProp prop{};
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&prop, q.device().device_id));
  args.num_mp = prop.multiProcessorCount;

  args.batch           = q.size(0);
  args.seqlen_q        = q.size(1);
  args.nr_heads        = q.size(2);
  args.headdim_qk      = q.size(3);
  args.num_blocks      = k_cache.size(0);
  args.page_block_size = k_cache.size(1);
  args.nr_heads_kv     = k_cache.size(2);
  args.headdim_v       = v_cache.size(3);

  TVM_FFI_ICHECK_GT(args.nr_heads_kv, 0) << "nheads_kv must be positive";
  TVM_FFI_ICHECK_EQ(args.nr_heads % args.nr_heads_kv, 0) << "nheads must be divisible by nheads_kv";
  TVM_FFI_ICHECK_GT(args.headdim_qk, 0) << "headdim_qk must be positive";
  TVM_FFI_ICHECK_LE(args.headdim_qk, 128) << "headdim_qk must be <= 128";
  TVM_FFI_ICHECK_GT(args.headdim_v, 0) << "headdim_v must be positive";
  TVM_FFI_ICHECK_LE(args.headdim_v, 128) << "headdim_v must be <= 128";
  TVM_FFI_ICHECK(args.page_block_size == 128 || args.page_block_size == 64) << "page_block_size must be 64 or 128";

  int q_seq_scale_num    = mutlass::ceil_div(args.seqlen_q, 128);
  int k_scale_per_block  = mutlass::ceil_div(args.page_block_size, 128);
  int k_scale_per_thread = mutlass::ceil_div(args.page_block_size, 16);
  int k_seq_scale_num    = args.num_blocks * k_scale_per_block;
  if (quant_mode == 6) {
    k_seq_scale_num = args.num_blocks * k_scale_per_thread;
  } else if (quant_mode == 1) {
    q_seq_scale_num = args.seqlen_q;
    k_seq_scale_num = args.num_blocks * args.page_block_size;
  } else if (quant_mode == 0) {
    q_seq_scale_num = 1;
    k_seq_scale_num = 1;
  } else if (quant_mode != 2) {
    TVM_FFI_ICHECK(false) << "Unsupported KV-cache SageAttention quant_mode: " << quant_mode;
  }

  expect_shape(q, {args.batch, args.seqlen_q, args.nr_heads, args.headdim_qk}, "q");
  expect_shape(k_cache, {args.num_blocks, args.page_block_size, args.nr_heads_kv, args.headdim_qk}, "k_cache");
  expect_shape(v_cache, {args.num_blocks, args.page_block_size, args.nr_heads_kv, args.headdim_v}, "v_cache");
  if (quant_mode == 0) {
    expect_shape(q_scale, {1, 1, 1, 1}, "q_scale");
    expect_shape(k_scale, {1, 1, 1, 1}, "k_scale");
    expect_shape(v_scale, {1, 1, 1, 1}, "v_scale");
  } else {
    expect_shape(q_scale, {args.batch, q_seq_scale_num, args.nr_heads, 1}, "q_scale");
    expect_shape(k_scale, {args.num_blocks, k_seq_scale_num / args.num_blocks, args.nr_heads_kv, 1}, "k_scale");
    if (quant_mode == 1) {
      expect_shape(v_scale, {args.num_blocks, args.page_block_size, args.nr_heads_kv, args.headdim_v}, "v_scale");
    } else {
      expect_shape(v_scale, {args.num_blocks, 1, args.nr_heads_kv, args.headdim_v}, "v_scale");
    }
  }
  expect_shape(out, {args.batch, args.seqlen_q, args.nr_heads, args.headdim_v}, "out");
  expect_shape(out_lse, {args.batch, args.nr_heads, args.seqlen_q}, "out_lse");
  expect_shape(page_table, {args.batch, page_table.size(1)}, "page_table");
  expect_shape(cache_seqlens, {args.batch}, "cache_seqlens");

  args.batch_stride_q                 = q.stride(0);
  args.seq_stride_q                   = q.stride(1);
  args.head_stride_q                  = q.stride(2);
  args.num_blocks_stride_k_cache      = k_cache.stride(0);
  args.page_block_size_stride_k_cache = k_cache.stride(1);
  args.nheads_k_stride_k_cache        = k_cache.stride(2);
  args.batch_stride_block_table       = page_table.stride(0);
  args.batch_stride_out               = out.stride(0);
  args.seq_stride_out                 = out.stride(1);
  args.head_stride_out                = out.stride(2);
  args.batch_stride_q_scale           = q_scale.size(0) == 1 ? 0 : q_scale.stride(0);
  args.seq_stride_q_scale             = q_scale.size(1) == 1 ? 0 : q_scale.stride(1);
  args.head_stride_q_scale            = q_scale.size(2) == 1 ? 0 : q_scale.stride(2);
  args.batch_stride_k_scale           = k_scale.size(0) == 1 ? 0 : k_scale.stride(0);
  args.seq_stride_k_scale             = k_scale.size(1) == 1 ? 0 : k_scale.stride(1);
  args.head_stride_k_scale            = k_scale.size(2) == 1 ? 0 : k_scale.stride(2);
  args.batch_stride_v_scale           = v_scale.size(0) == 1 ? 0 : v_scale.stride(0);
  args.seq_stride_v_scale             = v_scale.size(1) == 1 ? 0 : v_scale.stride(1);
  args.head_stride_v_scale            = v_scale.size(2) == 1 ? 0 : v_scale.stride(2);

  args.nr_k          = static_cast<int32_t>(k_cache.numel());
  args.nr_q_scale    = static_cast<int32_t>(q_scale.numel());
  args.nr_k_scale    = static_cast<int32_t>(k_scale.numel());
  args.nr_v_scale    = static_cast<int32_t>(v_scale.numel());
  args.softmax_scale = softmax_scale;

  args.p_q              = q.data_ptr();
  args.p_k_cache        = k_cache.data_ptr();
  args.p_v_cache        = v_cache.data_ptr();
  args.p_page_table     = page_table.data_ptr();
  args.p_cache_seqlens  = cache_seqlens.data_ptr();
  args.nr_page_table    = static_cast<int32_t>(page_table.numel());
  args.nr_cache_seqlens = static_cast<int32_t>(cache_seqlens.numel());
  args.p_q_scale        = q_scale.data_ptr();
  args.p_k_scale        = k_scale.data_ptr();
  args.p_v_scale        = v_scale.data_ptr();
  args.p_output         = out.data_ptr();
  args.p_lse_output     = out_lse.data_ptr();

  musaStream_t stream                 = get_stream(q.device());
  using Kernel                        = mate::sage_attention::mubin::SageAttentionQuantizedWithKVCacheAsmKernel<Args>;
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sage_attn_quantized_asm, sage_attn_quantized_asm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sage_attn_quantized_with_kvcache_asm, sage_attn_quantized_with_kvcache_asm);
