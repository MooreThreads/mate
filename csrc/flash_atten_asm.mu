
#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mutlass/fast_math.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <optional>

#include "asm_common.hpp"
#include "mate_utils.muh"
#include "mubin/mp31/mp31_flash_atten_registry.hpp"
#include "torch_utils.hpp"

namespace mate::flash_attention {

struct FlashAttenASMArgs {
  bool is_causal{};

  at::ScalarType data_type;

  int32_t num_mp{};

  // shape
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

  // unused
  int32_t batch_stride_mask{};
  int32_t head_stride_mask{};
  int32_t seq_stride_mask{};

  int32_t batch_stride_out{};
  int32_t head_stride_out{};
  int32_t seq_stride_out{};

  int32_t nr_k{};
  int32_t nr_mask{};  // unused

  double softmax_scale{};

  void* p_q{};
  void* p_k{};
  void* p_v{};

  void* p_input_mask{};  // unused

  void* p_output{};
  void* p_lse_output{};

};  // struct FlashAttenASMArgs

struct FlashAttenVarlenASMArgs {
  bool is_causal{};

  at::ScalarType data_type;

  int32_t num_mp{};

  int32_t batch{};

  // shape
  int32_t total_seqlen_q{};
  int32_t nr_heads{};
  int32_t headdim_qk{};

  int32_t total_seqlen_kv{};
  int32_t nr_heads_kv{};

  int32_t headdim_v{};

  // stride
  int32_t total_seqlen_stride_q{};
  int32_t head_stride_q{};

  int32_t total_seqlen_stride_k{};
  int32_t head_stride_k{};

  int32_t total_seqlen_stride_v{};
  int32_t head_stride_v{};

  int32_t total_seqlen_stride_out{};
  int32_t head_stride_out{};

  int32_t nr_k{};
  int32_t max_seqlen_q{};
  int32_t max_seqlen_kv{};

  double softmax_scale{};

  void* p_q{};
  void* p_k{};
  void* p_v{};

  void* p_input_cu_seqlen_q{};
  void* p_input_cu_seqlen_k{};

  void* p_output{};
  void* p_lse_output{};

};  // struct FlashAttenVarlenASMArgs

namespace mubin {

template <class Args>
struct FlashAttenASMDispatcher {
  FlashAttenASMDispatcher() {
    // register all fa mubin kernels
    // do only once
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
    constexpr bool is_varlen = std::is_same_v<Args, FlashAttenVarlenASMArgs>;

    Config config;

    const int headdim_qk = args.headdim_qk <= 128 ? 128 : args.headdim_qk;

    if (headdim_qk == 128) {
      config.nr_thr    = 512;
      config.tile_m    = 256;
      config.tile_n    = 128;
      config.tile_k    = 64;
      config.tile_hdim = 128;
    } else {
      // headdim_qk == 192
      config.nr_thr    = 512;
      config.tile_m    = 256;
      config.tile_n    = 64;
      config.tile_k    = 64;
      config.tile_hdim = 128;
    }

    FlashAttenAsmID id;
    id.is_causal  = static_cast<int>(args.is_causal);
    id.is_varlen  = static_cast<int>(is_varlen);
    id.headdim_qk = headdim_qk == 128 ? 0 : 1;
    id.dtype      = args.data_type == at::kBFloat16 ? 0 : 1;

    // checking if we have this kernel
    auto res = kernel_map.find(id);
    if (res != kernel_map.end()) {
      // the kernel is already called once
      // we don't need to load it again

      config.asm_func = &res->second.asm_func;

      return config;

    } else {
      auto asm_meta = fa_asm_kern_registry.find(id);
      if (asm_meta == fa_asm_kern_registry.end()) {
        throw std::runtime_error("FlashAttenAsmDispatcher() mubin kernel not found!");
      }

      // Call the kernel first time
      // we need to load it and cache it

      auto [curr_asm_meta, success] = kernel_map.try_emplace(id, asm_meta->second.first, asm_meta->second.second);

      config.asm_func = &curr_asm_meta->second.asm_func;

      return config;
    }
  }

  std::unordered_map<FlashAttenAsmID, MateAsmKernel> kernel_map;

};  // struct FlashAttenASMDispatcher

template <class Args_ = FlashAttenASMArgs, class Dispatcher = FlashAttenASMDispatcher<Args_>>
class FlashAttentionAsmKernel {
 public:
  using Args         = Args_;
  using Config       = typename Dispatcher::Config;
  using LaunchConfig = MUlaunchConfig;

  struct Params {
    void* output_ptr;
    void* out_lse_ptr;

    MUtensorDescriptor q_desc;
    MUtensorDescriptor k_desc_0;
    MUtensorDescriptor k_desc_1;
    MUtensorDescriptor v_desc;
    MUtensorDescriptor out_desc;
    mute::RobustReg    robust_mask;
    mute::RobustReg    robust_key;

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
    int32_t tile_num;

    int32_t double_block_num_sub1;
    int32_t double_block_num;

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

  FlashAttentionAsmKernel(const Params& in_params, const Config& in_config, const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const FlashAttenASMArgs& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    // bhsd
    TmeDesc tensor_desc_q(mute::make_tuple(args.headdim_qk, args.seqlen_q, args.nr_heads, args.batch),
                          mute::make_tuple(args.seq_stride_q, args.head_stride_q, args.batch_stride_q),
                          args.p_q,
                          torch_type_to_tme_type(args.data_type));
    TmeDesc tensor_desc_v(mute::make_tuple(args.headdim_v, args.seqlen_kv, args.nr_heads_kv, args.batch),
                          mute::make_tuple(args.seq_stride_v, args.head_stride_v, args.batch_stride_v),
                          args.p_v,
                          torch_type_to_tme_type(args.data_type));
    TmeDesc tensor_desc_out(mute::make_tuple(args.headdim_v, args.seqlen_q, args.nr_heads, args.batch),
                            mute::make_tuple(args.seq_stride_out, args.head_stride_out, args.batch_stride_out),
                            args.p_output,
                            torch_type_to_tme_type(args.data_type));

    params.output_ptr  = args.p_output;
    params.out_lse_ptr = args.p_lse_output;

    params.q_desc   = tensor_desc_q.desc;
    params.v_desc   = tensor_desc_v.desc;
    params.out_desc = tensor_desc_out.desc;

    if (args.data_type == at::kHalf) {
      params.robust_key = mute::make_robust_desc(static_cast<mutlass::half_t*>(args.p_k), args.nr_k).reg;
    } else {
      params.robust_key = mute::make_robust_desc(static_cast<mutlass::bfloat16_t*>(args.p_k), args.nr_k).reg;
    }
    params.robust_mask = mute::make_robust_desc(static_cast<int32_t*>(args.p_input_mask), args.nr_mask).reg;

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

    // sizeof(half) == sizeof(bfloat16)
    params.ldg_key_stride = args.seq_stride_k * config.tile_n * sizeof(mutlass::half_t);

    params.batch_stride_k = args.batch_stride_k;
    params.head_stride_k  = args.head_stride_k;
    params.seq_stride_k   = args.seq_stride_k;

    params.batch_stride_mask = args.batch_stride_mask;
    params.head_stride_mask  = args.head_stride_mask;
    params.seq_stride_mask   = args.seq_stride_mask;

    params.batch_stride_out = args.batch_stride_out;
    params.head_stride_out  = args.head_stride_out;
    params.seq_stride_out   = args.seq_stride_out;

    // sizeof(half) == sizeof(bfloat16)
    params.tile_size_q = config.tile_m * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_k = config.tile_n * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_v = config.tile_hdim * config.tile_k * sizeof(mutlass::half_t);

    int seq_q_tile_num = mutlass::ceil_div(args.seqlen_q, config.tile_m);
    params.tile_num    = seq_q_tile_num * args.batch * args.nr_heads;

    int nr_persistence           = std::min(args.num_mp, params.tile_num);
    params.double_block_num_sub1 = 2 * nr_persistence - 1;
    params.double_block_num      = 2 * nr_persistence;

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

    // this is varlen == this is NOT persistence
    launch_config.blockDimX      = config.nr_thr;
    launch_config.blockDimY      = 1;
    launch_config.blockDimZ      = 1;
    launch_config.gridDimX       = nr_persistence;
    launch_config.gridDimY       = 1;
    launch_config.gridDimZ       = 1;
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

};  // class FlashAttentionAsmKernel

template <class Args_ = FlashAttenVarlenASMArgs, class Dispatcher = FlashAttenASMDispatcher<Args_>>
class FlashAttentionVarlenAsmKernel {
 public:
  using Args         = Args_;
  using Config       = typename Dispatcher::Config;
  using LaunchConfig = MUlaunchConfig;

  struct Params {
    void* output_ptr{};
    void* out_lse_ptr{};

    MUtensorDescriptor q_desc{};
    MUtensorDescriptor k_desc{};
    MUtensorDescriptor v_desc{};
    mute::RobustReg    robust_key{};

    int32_t* input_cu_seqlen_q_ptr{};
    int32_t* input_cu_seqlen_k_ptr{};

    float scale{};
    float rln2_scale{};
    float nr_ln2{};

    int32_t batch{};
    int32_t nheads{};
    int32_t kv_heads{};
    int32_t max_seqlen_q{};
    int32_t max_seqlen_kv{};
    int32_t headdim_v{};

    int32_t  head_group_ori{};
    uint32_t head_group_mul{};
    uint32_t head_group_sft{};

    int32_t ldg_key_stride{};

    int32_t total_seqlen_q{};

    int32_t total_seqlen_stride_k{};
    int32_t head_stride_k{};

    int32_t total_seqlen_stride_out{};
    int32_t head_stride_out{};

    int32_t tile_size_q{};
    int32_t tile_size_k{};
    int32_t tile_size_v{};

    int32_t tile_q_dim0{};
    int32_t tile_q_dim1{};
    int32_t tile_q_dim2{};

    int32_t tile_k_dim0{};
    int32_t tile_k_dim1{};
    int32_t tile_k_dim2{};

    int32_t tile_v_dim0{};
    int32_t tile_v_dim1{};
    int32_t tile_v_dim2{};

  };  // struct Params

  FlashAttentionVarlenAsmKernel(const Params& in_params, const Config& in_config, const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const Args& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    TmeDesc tensor_desc_q(mute::make_tuple(args.headdim_qk, args.nr_heads, args.total_seqlen_q),
                          mute::make_tuple(args.head_stride_q, args.total_seqlen_stride_q),
                          args.p_q,
                          torch_type_to_tme_type(args.data_type));
    TmeDesc tensor_desc_v(mute::make_tuple(args.headdim_v, args.nr_heads_kv, args.total_seqlen_kv),
                          mute::make_tuple(args.head_stride_v, args.total_seqlen_stride_v),
                          args.p_v,
                          torch_type_to_tme_type(args.data_type));

    params.output_ptr  = args.p_output;
    params.out_lse_ptr = args.p_lse_output;

    params.q_desc = tensor_desc_q.desc;
    params.v_desc = tensor_desc_v.desc;
    if (args.data_type == at::kHalf) {
      params.robust_key = mute::make_robust_desc(static_cast<mutlass::half_t*>(args.p_k), args.nr_k).reg;
    } else {
      params.robust_key = mute::make_robust_desc(static_cast<mutlass::bfloat16_t*>(args.p_k), args.nr_k).reg;
    }

    params.input_cu_seqlen_q_ptr = static_cast<int*>(args.p_input_cu_seqlen_q);
    params.input_cu_seqlen_k_ptr = static_cast<int*>(args.p_input_cu_seqlen_k);

    const double log2e = std::log2(std::exp(1.0));
    params.scale       = args.softmax_scale;
    params.rln2_scale  = args.softmax_scale * log2e;
    params.nr_ln2      = 1.0 / log2e;

    params.batch         = args.batch;
    params.nheads        = args.nr_heads;
    params.kv_heads      = args.nr_heads_kv;
    params.max_seqlen_q  = args.max_seqlen_q;
    params.max_seqlen_kv = args.max_seqlen_kv;
    params.headdim_v     = args.headdim_v;

    int  head_group       = args.nr_heads / args.nr_heads_kv;
    auto fast_head_group  = mutlass::FastDivmod(head_group);
    params.head_group_ori = fast_head_group.divisor;
    params.head_group_mul = fast_head_group.multiplier;
    params.head_group_sft = fast_head_group.shift_right;

    // sizeof(half) == sizeof(bfloat16)
    params.ldg_key_stride = args.total_seqlen_stride_k * config.tile_n * sizeof(mutlass::half_t);
    params.total_seqlen_q = args.total_seqlen_q;

    params.total_seqlen_stride_k = args.total_seqlen_stride_k;
    params.head_stride_k         = args.head_stride_k;

    params.total_seqlen_stride_out = args.total_seqlen_stride_out;
    params.head_stride_out         = args.head_stride_out;

    // sizeof(half) == sizeof(bfloat16)
    params.tile_size_q = config.tile_m * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_k = config.tile_n * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_v = config.tile_hdim * config.tile_k * sizeof(mutlass::half_t);

    params.tile_q_dim0 = config.tile_k;
    params.tile_q_dim1 = 1;
    params.tile_q_dim2 = config.tile_m;

    params.tile_k_dim0 = config.tile_k;
    params.tile_k_dim1 = 1;
    params.tile_k_dim2 = config.tile_n;

    params.tile_v_dim0 = config.tile_hdim;
    params.tile_v_dim1 = 1;
    params.tile_v_dim2 = config.tile_k;

    // this is varlen == this is NOT persistence
    launch_config.blockDimX = config.nr_thr;
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    if (args.is_causal) {
      launch_config.gridDimX = params.nheads;
      launch_config.gridDimY = params.batch;
      launch_config.gridDimZ = mutlass::ceil_div(params.max_seqlen_q, config.tile_m);

    } else {
      launch_config.gridDimX = mutlass::ceil_div(params.max_seqlen_q, config.tile_m);
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

};  // class FlashAttentionVarlenAsmKernel

}  // namespace mubin

}  // namespace mate::flash_attention

std::tuple<at::Tensor, at::Tensor> dispatch_flash_atten_asm(const at::Tensor& q,
                                                            const at::Tensor& k,
                                                            const at::Tensor& v,
                                                            const double      softmax_scale,
                                                            at::Tensor&       out,
                                                            at::Tensor&       out_lse,
                                                            const bool        is_causal) {
  CHECK_MP31("dispatch_flash_atten_asm");
  at::TensorArg targs[]{{q, "q", 0}, {k, "k", 1}, {v, "v", 2}, {out, "out", 3}, {out_lse, "out_lse", 4}};
  at::checkAllSameGPU(__func__, targs);

  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(q, 4);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(k, 4);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(v, 4);
  CHECK_TENSOR_AND_CONTIGUOUS(out, 4);
  CHECK_TENSOR_AND_CONTIGUOUS(out_lse, 3);

  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type() && out.scalar_type() == q.scalar_type(),
      "dispatch_flash_atten_asm() qkv and out must have same data type!")

  using Args = mate::flash_attention::FlashAttenASMArgs;
  Args args;

  args.is_causal = is_causal;

  args.data_type = q.scalar_type();
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(q.scalar_type()), "dispatch_flash_atten_asm() qkv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(k.scalar_type()), "dispatch_flash_atten_asm() qkv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(v.scalar_type()), "dispatch_flash_atten_asm() qkv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(out.scalar_type()), "dispatch_flash_atten_asm() out must be half or bf16!");

  auto dprops = at::musa::getCurrentDeviceProperties();
  args.num_mp = dprops->multiProcessorCount;

  // shape
  // bshd
  args.batch      = q.size(0);
  args.seqlen_q   = q.size(1);
  args.nr_heads   = q.size(2);
  args.headdim_qk = q.size(3);

  args.seqlen_kv   = k.size(1);
  args.nr_heads_kv = k.size(2);

  args.headdim_v = v.size(3);

  bool is_192_128         = args.headdim_qk == 192 && args.headdim_v == 128;
  bool is_128_128_or_less = args.headdim_qk == args.headdim_v && args.headdim_qk <= 128;
  TORCH_CHECK(is_192_128 || is_128_128_or_less, "HeadDim unsupported!");

  CHECK_SHAPE(q, args.batch, args.seqlen_q, args.nr_heads, args.headdim_qk);
  CHECK_SHAPE(k, args.batch, args.seqlen_kv, args.nr_heads_kv, args.headdim_qk);
  CHECK_SHAPE(v, args.batch, args.seqlen_kv, args.nr_heads_kv, args.headdim_v);
  CHECK_SHAPE(out, args.batch, args.seqlen_q, args.nr_heads, args.headdim_v);
  CHECK_SHAPE(out_lse, args.batch, args.nr_heads, args.seqlen_q);

  // stride
  // bshd
  args.batch_stride_q = q.stride(0);
  args.head_stride_q  = q.stride(2);
  args.seq_stride_q   = q.stride(1);

  args.batch_stride_k = k.stride(0);
  args.head_stride_k  = k.stride(2);
  args.seq_stride_k   = k.stride(1);

  args.batch_stride_v = v.stride(0);
  args.head_stride_v  = v.stride(2);
  args.seq_stride_v   = v.stride(1);

  args.batch_stride_out = out.stride(0);
  args.head_stride_out  = out.stride(2);
  args.seq_stride_out   = out.stride(1);

  args.nr_k          = k.numel();
  args.softmax_scale = softmax_scale;

  args.p_q = q.data_ptr();
  args.p_k = k.data_ptr();
  args.p_v = v.data_ptr();

  args.p_output     = out.data_ptr();
  args.p_lse_output = out_lse.data_ptr();

  musaStream_t stream         = at::musa::getCurrentMUSAStream().stream();
  int          current_device = at::musa::current_device();

  using Kernel                        = mate::flash_attention::mubin::FlashAttentionAsmKernel<Args>;
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();

  return std::make_tuple(out, out_lse);
}

std::tuple<at::Tensor, at::Tensor> dispatch_flash_atten_varlen_asm(const at::Tensor& q,
                                                                   const at::Tensor& k,
                                                                   const at::Tensor& v,
                                                                   const at::Tensor& input_cu_seqlen_q,
                                                                   const at::Tensor& input_cu_seqlen_k,
                                                                   const int64_t     max_seqlen_q,
                                                                   const int64_t     max_seqlen_kv,
                                                                   const double      softmax_scale,
                                                                   at::Tensor&       out,
                                                                   at::Tensor&       out_lse,
                                                                   const bool        is_causal) {
  at::TensorArg targs[]{{q, "q", 0},
                        {k, "k", 1},
                        {v, "v", 2},
                        {input_cu_seqlen_q, "input_cu_seqlen_q", 3},
                        {input_cu_seqlen_k, "input_cu_seqlen_k", 4},
                        {out, "out", 5},
                        {out_lse, "out_lse", 6}};
  at::checkAllSameGPU(__func__, targs);

  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(q, 3);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(k, 3);
  CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(v, 3);
  CHECK_TENSOR_AND_CONTIGUOUS(out, 3);
  CHECK_TENSOR_AND_CONTIGUOUS(out_lse, 2);
  CHECK_TENSOR_AND_CONTIGUOUS(input_cu_seqlen_q, 1);
  CHECK_TENSOR_AND_CONTIGUOUS(input_cu_seqlen_k, 1);
  TORCH_CHECK(input_cu_seqlen_q.scalar_type() == at::kInt,
              "dispatch_flash_atten_varlen_asm() input_cu_seqlen_q must be int!");
  TORCH_CHECK(input_cu_seqlen_k.scalar_type() == at::kInt,
              "dispatch_flash_atten_varlen_asm() input_cu_seqlen_k must be int!");
  TORCH_CHECK(input_cu_seqlen_k.size(0) == input_cu_seqlen_q.size(0),
              "dispatch_flash_atten_varlen_asm() input_cu_seqlen_k and input_cu_seqlen_q must have "
              "same size!");
  TORCH_CHECK(input_cu_seqlen_k.size(0) > 0,
              "dispatch_flash_atten_varlen_asm() input_cu_seqlen_k len must be greater than 0!");

  using Args = mate::flash_attention::FlashAttenVarlenASMArgs;
  Args args;

  args.is_causal = is_causal;

  args.data_type = q.scalar_type();
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(q.scalar_type()), "dispatch_flash_atten_asm() qkv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(k.scalar_type()), "dispatch_flash_atten_asm() qkv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(v.scalar_type()), "dispatch_flash_atten_asm() qkv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(out.scalar_type()), "dispatch_flash_atten_asm() out must be half or bf16!");

  args.batch = input_cu_seqlen_q.size(0) - 1;

  // bypass bs=1
  if (args.batch == 1) {
    auto q_view       = q.unsqueeze(0);
    auto k_view       = k.unsqueeze(0);
    auto v_view       = v.unsqueeze(0);
    auto out_view     = out.unsqueeze(0);
    auto out_lse_view = out_lse.unsqueeze(0);
    dispatch_flash_atten_asm(q_view, k_view, v_view, softmax_scale, out_view, out_lse_view, is_causal);
    return std::make_tuple(out, out_lse);
  }

  auto dprops = at::musa::getCurrentDeviceProperties();
  args.num_mp = dprops->multiProcessorCount;

  // shape
  args.total_seqlen_q = q.size(0);
  args.nr_heads       = q.size(1);
  args.headdim_qk     = q.size(2);

  args.total_seqlen_kv = k.size(0);
  args.nr_heads_kv     = k.size(1);

  args.headdim_v = v.size(2);

  bool is_192_128         = args.headdim_qk == 192 && args.headdim_v == 128;
  bool is_128_128_or_less = args.headdim_qk == args.headdim_v && args.headdim_qk <= 128;
  TORCH_CHECK(is_192_128 || is_128_128_or_less, "HeadDim unsupported!");
  TORCH_CHECK(args.nr_heads >= args.nr_heads_kv,
              "dispatch_flash_atten_varlen_asm() nr_heads must be greater than nr_heads_kv!")
  TORCH_CHECK(args.nr_heads % args.nr_heads_kv == 0,
              "dispatch_flash_atten_varlen_asm() nr_heads must be divisible by nr_heads_kv!")

  CHECK_SHAPE(q, args.total_seqlen_q, args.nr_heads, args.headdim_qk);
  CHECK_SHAPE(k, args.total_seqlen_kv, args.nr_heads_kv, args.headdim_qk);
  CHECK_SHAPE(v, args.total_seqlen_kv, args.nr_heads_kv, args.headdim_v);
  CHECK_SHAPE(out, args.total_seqlen_q, args.nr_heads, args.headdim_v);
  CHECK_SHAPE(out_lse, args.nr_heads, args.total_seqlen_q);

  // stride
  args.total_seqlen_stride_q = q.stride(0);
  args.head_stride_q         = q.stride(1);

  args.total_seqlen_stride_k = k.stride(0);
  args.head_stride_k         = k.stride(1);

  args.total_seqlen_stride_v = v.stride(0);
  args.head_stride_v         = v.stride(1);

  args.total_seqlen_stride_out = out.stride(0);
  args.head_stride_out         = out.stride(1);

  args.nr_k          = k.numel();
  args.max_seqlen_q  = max_seqlen_q;
  args.max_seqlen_kv = max_seqlen_kv;

  args.p_q = q.data_ptr();
  args.p_k = k.data_ptr();
  args.p_v = v.data_ptr();

  args.softmax_scale = softmax_scale;

  args.p_input_cu_seqlen_q = input_cu_seqlen_q.data_ptr();
  args.p_input_cu_seqlen_k = input_cu_seqlen_k.data_ptr();

  args.p_output     = out.data_ptr();
  args.p_lse_output = out_lse.data_ptr();

  musaStream_t stream         = at::musa::getCurrentMUSAStream().stream();
  int          current_device = at::musa::current_device();

  using Kernel                        = mate::flash_attention::mubin::FlashAttentionVarlenAsmKernel<Args>;
  auto [param, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                         = Kernel(param, config, launch_config);
  kernel.run();

  return std::make_tuple(out, out_lse);
}

std::tuple<at::Tensor, at::Tensor> flash_atten_varlen_asm(const at::Tensor&         q,
                                                          const at::Tensor&         k,
                                                          const at::Tensor&         v,
                                                          const double              softmax_scale,
                                                          at::Tensor&               out,
                                                          at::Tensor&               out_lse,
                                                          const bool                is_causal,
                                                          std::optional<at::Tensor> input_cu_seqlen_q,
                                                          std::optional<at::Tensor> input_cu_seqlen_k,
                                                          std::optional<int64_t>    max_seqlen_q,
                                                          std::optional<int64_t>    max_seqlen_kv) {
  CHECK_MP31("flash_atten_varlen_asm");

  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type() && out.scalar_type() == q.scalar_type(),
      "flash_atten_varlen_asm() qkv & out must have same data type!")

  c10::musa::OptionalMUSAGuard guard(q.device());

  bool is_varlen = input_cu_seqlen_q.has_value() || input_cu_seqlen_k.has_value();

  if (is_varlen) {
    TORCH_CHECK(input_cu_seqlen_q.has_value(),
                "flash_atten_varlen_asm() is_varlen is true! input_cu_seqlen_q must be provided!");
    TORCH_CHECK(input_cu_seqlen_k.has_value(),
                "flash_atten_varlen_asm() is_varlen is true! input_cu_seqlen_k must be provided!");
    TORCH_CHECK(max_seqlen_q.has_value(), "flash_atten_varlen_asm() is_varlen is true! max_seqlen_q must be provided!");
    TORCH_CHECK(max_seqlen_kv.has_value(),
                "flash_atten_varlen_asm() is_varlen is true! max_seqlen_kv must be provided!");
    return dispatch_flash_atten_varlen_asm(q,
                                           k,
                                           v,
                                           input_cu_seqlen_q.value(),
                                           input_cu_seqlen_k.value(),
                                           max_seqlen_q.value(),
                                           max_seqlen_kv.value(),
                                           softmax_scale,
                                           out,
                                           out_lse,
                                           is_causal);
  } else {
    // is NOT varlen
    return dispatch_flash_atten_asm(q, k, v, softmax_scale, out, out_lse, is_causal);
  }
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "flash_atten_varlen_asm("
      "Tensor q,"
      "Tensor k,"
      "Tensor v,"
      "float softmax_scale,"
      "Tensor out,"
      "Tensor out_lse,"
      "bool is_causal,"
      "Tensor? input_cu_seqlen_q = None,"
      "Tensor? input_cu_seqlen_k = None,"
      "int? max_seqlen_q = None,"
      "int? max_seqlen_kv = None"
      ")"
      "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("flash_atten_varlen_asm", &flash_atten_varlen_asm);
}
