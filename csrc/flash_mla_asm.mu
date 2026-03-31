#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mutlass/fast_math.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include <mute/algorithm/tuple_algorithms.hpp>
#include <mute/arch/copy_mp31_desc.hpp>

#include "asm_common.hpp"
#include "attention_combine.hpp"
#include "mate/attention/flash_mla/mpxx_params.hpp"
#include "mate_utils.muh"
#include "mubin/mp31/mp31_flash_mla_registry.hpp"
#include "torch_utils.hpp"

namespace mate::flash_mla {

struct MLAAsmArgs {
  bool is_causal{};
  bool is_varlen_q{};

  at::ScalarType data_type;

  // int32_t tile_k{};
  int32_t page_block_size{};
  int32_t nr_block{};

  // shape
  int32_t batch{};
  int32_t seqlen_q{};  // total_seqlen_q if is_varlen_q
  int32_t nr_heads{};
  int32_t headdim_qk{};

  int32_t seqlen_kv{};
  int32_t nr_heads_kv{};
  int32_t headdim_v{};

  int32_t total_q{};  // for varlen q

  // stride
  int32_t stride_q[4]{};
  int32_t stride_q_rope[4]{};
  int32_t stride_k[4]{};
  int32_t stride_v[4]{};

  int32_t batch_stride_out{};
  int32_t nosplit_batch_stride_out{};
  int32_t head_stride_out{};
  int32_t seq_stride_out{};

  int32_t batch_stride_lse{};
  int32_t nosplit_batch_stride_lse{};
  int32_t head_stride_lse{};
  int32_t seq_stride_lse{};

  int32_t block_table_stride0{};
  int32_t q_seq_per_hk{};
  int32_t max_q_seq_per_hk{};

  int32_t nr_mp_parts{};

  double softmax_scale{};

  void* p_q{};
  void* p_q_rope{};
  void* p_k{};
  void* p_v{};

  void* p_output{};
  void* p_lse_output{};

  void* out_accum_ptr;
  void* out_lseaccum_ptr;

  void* kv_cache_ptr{};
  void* block_table_ptr{};
  void* cache_seqlen_ptr{};
  void* tile_scheduler_metadata_ptr{};
  void* num_splits_ptr{};
  void* seqlens_q_ptr{};  // for varlen_q

};  // struct MLAAsmArgs

namespace mubin {

struct MLAAsmDispatcher {
  struct Config {
    int nr_thr = 512;

    int tile_m         = 128;
    int tile_n         = 64;
    int tile_k         = 64;
    int tile_hdim      = 512;
    int tile_rope_hdim = 64;

    MUfunction* asm_func;

  };  // struct Config

  MLAAsmDispatcher() {
    // register all mla mubin kernels
    // do only once
    init_flash_mla_asm_kern_registry();
  }

  Config get_kernel_config(const MLAAsmArgs& args) {
    Config config;

    FlashMLAAsmID id;

    id.is_causal   = args.is_causal;
    id.is_varlen_q = args.is_varlen_q;
    id.dtype       = args.data_type == at::kBFloat16 ? 0 : 1;

    // checking if we have this kernel
    auto res = kernel_map.find(id);
    if (res != kernel_map.end()) {
      // the kernel is already called once
      // we don't need to load it again

      config.asm_func = &res->second.asm_func;

      return config;

    } else {
      auto asm_meta = flash_mla_asm_kern_registry.find(id);
      if (asm_meta == flash_mla_asm_kern_registry.end()) {
        throw std::runtime_error("MLAAsmDispatcher() mubin kernel not found!");
      }

      // Call the kernel first time
      // we need to load it and cache it

      auto [curr_asm_meta, success] = kernel_map.try_emplace(id, asm_meta->second.first, asm_meta->second.second);

      config.asm_func = &curr_asm_meta->second.asm_func;

      return config;
    }

    return config;
  }

  std::unordered_map<FlashMLAAsmID, MateAsmKernel> kernel_map;

};  // struct MLAAsmDispatcher

struct MLAAsmKernel {
  using Dispatcher   = MLAAsmDispatcher;
  using Args         = MLAAsmArgs;
  using Config       = typename Dispatcher::Config;
  using LaunchConfig = MUlaunchConfig;

  struct Params {
    void* output_ptr;
    void* out_lse_ptr;

    void* out_accum_ptr;
    void* out_lseaccum_ptr;

    MUtensorDescriptor q_desc;
    MUtensorDescriptor q_rope_desc;
    MUtensorDescriptor k_desc;
    MUtensorDescriptor v_desc;

    void* kv_cache_ptr;
    void* block_table_ptr;
    void* cache_seqlen_ptr;
    void* abs_indices_ptr;
    void* indices_in_kvcache_ptr;
    void* tile_scheduler_metadata_ptr;
    void* num_splits_ptr;
    void* seqlens_q_ptr;

    float scale;
    float rln2_scale;
    float nr_ln2;

    int32_t batch;
    int32_t nheads;
    int32_t kv_heads;
    int32_t q_seq_per_hk;
    int32_t seqlen_q;
    int32_t total_q;
    int32_t seqlen_kv;
    int32_t headdim_qk;
    int32_t headdim_v;

    int32_t  head_group_ori;
    uint32_t head_group_mul;
    uint32_t head_group_sft;

    int32_t batch_stride_block_table;

    int32_t batch_stride_out;
    int32_t nosplit_batch_stride_out;
    int32_t seq_stride_out;
    int32_t seq_stride_lse;
    int32_t head_stride_out;

    int32_t batch_stride_lse;
    int32_t nosplit_batch_stride_lse;
    int32_t head_stride_lse;

    int32_t tile_size_q;
    int32_t tile_size_q_rope;
    int32_t tile_size_k;
    int32_t tile_size_v;

    int32_t tile_q_dim0;
    int32_t tile_q_dim1;
    int32_t tile_q_dim2;
    int32_t tile_q_dim3;
    int32_t tile_q_dim4;

    int32_t tile_q_rope_dim0;
    int32_t tile_q_rope_dim1;
    int32_t tile_q_rope_dim2;
    int32_t tile_q_rope_dim3;
    int32_t tile_q_rope_dim4;

    int32_t tile_k_dim0;
    int32_t tile_k_dim1;
    int32_t tile_k_dim2;
    int32_t tile_k_dim3;
    int32_t tile_k_dim4;

    int32_t tile_v_dim0;
    int32_t tile_v_dim1;
    int32_t tile_v_dim2;
    int32_t tile_v_dim3;
    int32_t tile_v_dim4;

  };  // struct Params

  MLAAsmKernel(const Params& in_params, const Config& in_config, const LaunchConfig& in_launch_config)
      : params(in_params), config(in_config), launch_config(in_launch_config) {
  }

  static auto to_underlying_arguments(const Args& args, musaStream_t stream) {
    static Dispatcher dispatcher;

    Config config = dispatcher.get_kernel_config(args);

    LaunchConfig launch_config;

    Params params;

    constexpr int tile_k = 64;

    TmeDesc tensor_desc_q;
    TmeDesc tensor_desc_q_rope;

    if (args.is_varlen_q) {
      TmeDesc tensor_desc_q_(mute::make_tuple(tile_k, args.nr_heads, args.total_q, args.headdim_v / tile_k),
                             mute::make_tuple(args.stride_q[0], args.stride_q[1], args.stride_q[2]),
                             args.p_q,
                             torch_type_to_tme_type(args.data_type));
      TmeDesc tensor_desc_q_rope_(mute::make_tuple(tile_k, args.nr_heads, args.total_q, 1),
                                  mute::make_tuple(args.stride_q_rope[0], args.stride_q_rope[1], args.stride_q_rope[2]),
                                  args.p_q_rope,
                                  torch_type_to_tme_type(args.data_type));

      tensor_desc_q      = tensor_desc_q_;
      tensor_desc_q_rope = tensor_desc_q_rope_;

    } else {
      // is NOT varlen Q
      TmeDesc tensor_desc_q_(
          mute::make_tuple(tile_k, args.nr_heads, args.seqlen_q, args.headdim_v / tile_k, args.batch),
          mute::make_tuple(args.stride_q[0], args.stride_q[1], args.stride_q[2], args.stride_q[3]),
          args.p_q,
          torch_type_to_tme_type(args.data_type));

      TmeDesc tensor_desc_q_rope_(
          mute::make_tuple(tile_k, args.nr_heads, args.seqlen_q, 1, args.batch),
          mute::make_tuple(args.stride_q_rope[0], args.stride_q_rope[1], args.stride_q_rope[2], args.stride_q_rope[3]),
          args.p_q_rope,
          torch_type_to_tme_type(args.data_type));

      tensor_desc_q      = tensor_desc_q_;
      tensor_desc_q_rope = tensor_desc_q_rope_;
    }

    TmeDesc tensor_desc_k(
        mute::make_tuple(args.headdim_qk * args.nr_heads_kv, 8, 8, args.page_block_size / 64, args.nr_block),
        mute::make_tuple(args.stride_k[0], args.stride_k[1], args.stride_k[2], args.stride_k[3]),
        args.p_k,
        torch_type_to_tme_type(args.data_type));
    TmeDesc tensor_desc_v(
        mute::make_tuple(args.headdim_qk * args.nr_heads_kv, 1, 1, args.page_block_size, args.nr_block),
        mute::make_tuple(args.stride_v[0], args.stride_v[1], args.stride_v[2], args.stride_v[3]),
        args.p_v,
        torch_type_to_tme_type(args.data_type));

    params.output_ptr  = args.p_output;
    params.out_lse_ptr = args.p_lse_output;

    params.out_accum_ptr    = args.out_accum_ptr;
    params.out_lseaccum_ptr = args.out_lseaccum_ptr;

    params.q_desc      = tensor_desc_q.desc;
    params.q_rope_desc = tensor_desc_q_rope.desc;
    params.k_desc      = tensor_desc_k.desc;
    params.v_desc      = tensor_desc_v.desc;

    params.kv_cache_ptr                = nullptr;
    params.block_table_ptr             = args.block_table_ptr;
    params.cache_seqlen_ptr            = args.cache_seqlen_ptr;
    params.abs_indices_ptr             = nullptr;
    params.indices_in_kvcache_ptr      = nullptr;
    params.tile_scheduler_metadata_ptr = args.tile_scheduler_metadata_ptr;
    params.num_splits_ptr              = args.num_splits_ptr;
    params.seqlens_q_ptr               = args.seqlens_q_ptr;

    const double log2e = std::log2(std::exp(1.0));
    params.scale       = args.softmax_scale;
    params.rln2_scale  = args.softmax_scale * log2e;
    params.nr_ln2      = 1.0 / log2e;

    params.batch        = args.batch;
    params.nheads       = args.nr_heads;
    params.kv_heads     = args.nr_heads_kv;
    params.seqlen_q     = args.seqlen_q;
    params.total_q      = args.total_q;
    params.seqlen_kv    = args.seqlen_kv;
    params.headdim_qk   = args.headdim_qk;
    params.headdim_v    = args.headdim_v;
    params.q_seq_per_hk = args.q_seq_per_hk;

    int  head_group       = args.nr_heads / args.nr_heads_kv;
    auto fast_head_group  = mutlass::FastDivmod(head_group);
    params.head_group_ori = fast_head_group.divisor;
    params.head_group_mul = fast_head_group.multiplier;
    params.head_group_sft = fast_head_group.shift_right;

    params.batch_stride_block_table = args.block_table_stride0;

    params.batch_stride_out         = args.batch_stride_out;
    params.nosplit_batch_stride_out = args.nosplit_batch_stride_out;
    params.head_stride_out          = args.head_stride_out;
    params.seq_stride_out           = args.seq_stride_out;
    params.seq_stride_lse           = args.seq_stride_lse;

    params.batch_stride_lse = args.batch_stride_lse;
    params.head_stride_lse  = args.head_stride_lse;

    // only support 2 byte datatype
    params.tile_size_q      = 8 * config.tile_m * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_q_rope = 1 * config.tile_m * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_k      = config.tile_n * config.tile_k * sizeof(mutlass::half_t);
    params.tile_size_v      = config.tile_n * config.tile_k * sizeof(mutlass::half_t);

    params.tile_q_dim0 = config.tile_k;
    params.tile_q_dim1 = config.tile_m;
    params.tile_q_dim2 = 1;
    params.tile_q_dim3 = 8;
    params.tile_q_dim4 = 1;

    params.tile_q_rope_dim0 = config.tile_k;
    params.tile_q_rope_dim1 = config.tile_m;
    params.tile_q_rope_dim2 = 1;
    params.tile_q_rope_dim3 = 1;
    params.tile_q_rope_dim4 = 1;

    params.tile_k_dim0 = config.tile_k;
    params.tile_k_dim1 = 8;
    params.tile_k_dim2 = 8;
    params.tile_k_dim3 = 1;
    params.tile_k_dim4 = 1;

    params.tile_v_dim0 = config.tile_k;
    params.tile_v_dim1 = 1;
    params.tile_v_dim2 = 1;
    params.tile_v_dim3 = config.tile_n;
    params.tile_v_dim4 = 1;

    launch_config.blockDimX      = config.nr_thr;
    launch_config.blockDimY      = 1;
    launch_config.blockDimZ      = 1;
    launch_config.gridDimX       = mutlass::ceil_div(args.seqlen_q * args.nr_heads, config.tile_m);
    launch_config.gridDimY       = args.nr_heads_kv;
    launch_config.gridDimZ       = args.nr_mp_parts;
    launch_config.hStream        = reinterpret_cast<MUstream>(stream);
    launch_config.sharedMemBytes = 0;
    launch_config.attrs          = NULL;
    launch_config.numAttrs       = 0;

    return std::make_tuple(params, config, launch_config);
  }

  void run() {
    void* kernel_params[] = {&params.output_ptr,
                             &params.out_lse_ptr,
                             &params.out_accum_ptr,
                             &params.out_lseaccum_ptr,
                             &params.q_desc,
                             &params.q_rope_desc,
                             &params.k_desc,
                             &params.v_desc,
                             &params.kv_cache_ptr,
                             &params.block_table_ptr,
                             &params.cache_seqlen_ptr,
                             &params.abs_indices_ptr,
                             &params.indices_in_kvcache_ptr,
                             &params.tile_scheduler_metadata_ptr,
                             &params.num_splits_ptr,
                             &params.seqlens_q_ptr,
                             &params.scale,
                             &params.rln2_scale,
                             &params.nr_ln2,
                             &params.batch,
                             &params.nheads,
                             &params.kv_heads,
                             &params.q_seq_per_hk,  // TODO:
                             &params.seqlen_q,
                             &params.total_q,
                             &params.seqlen_kv,
                             &params.headdim_qk,
                             &params.headdim_v,
                             &params.head_group_ori,
                             &params.head_group_mul,
                             &params.head_group_sft,
                             &params.batch_stride_block_table,
                             &params.batch_stride_out,
                             &params.seq_stride_out,
                             &params.seq_stride_lse,
                             &params.head_stride_out,
                             &params.batch_stride_lse,
                             &params.head_stride_lse,
                             &params.tile_size_q,
                             &params.tile_size_q_rope,
                             &params.tile_size_k,
                             &params.tile_size_v,
                             &params.tile_q_dim0,
                             &params.tile_q_dim1,
                             &params.tile_q_dim2,
                             &params.tile_q_dim3,
                             &params.tile_q_dim4,
                             &params.tile_q_rope_dim0,
                             &params.tile_q_rope_dim1,
                             &params.tile_q_rope_dim2,
                             &params.tile_q_rope_dim3,
                             &params.tile_q_rope_dim4,
                             &params.tile_k_dim0,
                             &params.tile_k_dim1,
                             &params.tile_k_dim2,
                             &params.tile_k_dim3,
                             &params.tile_k_dim4,
                             &params.tile_v_dim0,
                             &params.tile_v_dim1,
                             &params.tile_v_dim2,
                             &params.tile_v_dim3,
                             &params.tile_v_dim4};

    MATE_MUSA_DRIVER_CHECK(muLaunchKernelEx(&launch_config, *config.asm_func, kernel_params, nullptr));
  }

 protected:
  Params       params;
  Config       config;
  LaunchConfig launch_config;

};  // struct MLAAsmKernel

}  // namespace mubin

}  // namespace mate::flash_mla

std::vector<at::Tensor> flash_mla_asm(at::Tensor&               q_nope,       // bnhd
                                      at::Tensor&               q_pe,         // bnhd
                                      at::Tensor const&         ckv,          // num_block, page_size, h_k, headdim
                                      at::Tensor const&         kpe,          // num_block, page_size, h_k, headdim
                                      at::Tensor const&         seqlens_k,    // batch_size
                                      at::Tensor const&         block_table,  // batch_size, max_num_blocks_per_seq
                                      at::Tensor const&         tile_scheduler_metadata,
                                      at::Tensor const&         num_splits,
                                      double const              softmax_scale,
                                      bool const                is_causal,
                                      std::optional<at::Tensor> cu_seqlens_q,
                                      std::optional<int64_t>    max_seqlen_q) {
  CHECK_MP31("flash_mla_asm");
  at::TensorArg targs[]{{q_nope, "q_nope", 0},
                        {q_pe, "q_pe", 1},
                        {ckv, "ckv", 2},
                        {kpe, "kpe", 3},
                        {seqlens_k, "seqlens_k", 4},
                        {block_table, "block_table", 5},
                        {tile_scheduler_metadata, "tile_scheduler_metadata", 6},
                        {num_splits, "num_splits", 7}};
  at::checkAllSameGPU(__func__, targs);

  constexpr int headdim_latent = 512;
  constexpr int headdim_rope   = 64;

  TORCH_CHECK(tile_scheduler_metadata.scalar_type() == at::kInt,
              "flash_mla_asm() tile_scheduler_metadata must be int!");
  TORCH_CHECK(num_splits.scalar_type() == at::kInt, "flash_mla_asm() tile_scheduler_metadata must be int!");

  CHECK_TENSOR_AND_CONTIGUOUS(tile_scheduler_metadata, 2);
  CHECK_TENSOR_AND_CONTIGUOUS(num_splits, 1);

  const bool is_varlen_q = cu_seqlens_q.has_value();
  if (is_varlen_q) {
    CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(q_nope, 3);
    CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(q_pe, 3);
    TORCH_CHECK(max_seqlen_q.has_value(), "flash_mla_asm() varlen_q is enabled! max_seqlen_q must be provided!")
  } else {
    CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(q_nope, 4);
    CHECK_TENSOR_AND_LAST_DIM_CONTIGUOUS(q_pe, 4);
  }

  CHECK_MUSA(ckv);
  CHECK_MUSA(kpe);
  CHECK_MUSA(block_table);
  CHECK_MUSA(seqlens_k);

  const at::musa::OptionalMUSAGuard device_guard(q_nope.device());

  mate::flash_mla::MLAAsmArgs args;

  args.is_causal   = is_causal;
  args.is_varlen_q = is_varlen_q;

  args.data_type = q_nope.scalar_type();
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(q_nope.scalar_type()), "flash_mla_asm() q_nope must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(q_pe.scalar_type()), "flash_mla_asm() q_pe must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(ckv.scalar_type()), "flash_mla_asm() ckv must be half or bf16!");
  TORCH_CHECK(is_bf16_or_fp16_tensor_type(kpe.scalar_type()), "flash_mla_asm() kpe must be half or bf16!");

  // args.tile_k = 64;
  args.page_block_size = 64;
  args.nr_block        = ckv.size(0);

  args.batch      = !is_varlen_q ? q_nope.size(0) : cu_seqlens_q->size(0) - 1;
  args.seqlen_q   = !is_varlen_q ? q_nope.size(1) : max_seqlen_q.value();
  args.nr_heads   = q_nope.size(-2);
  args.headdim_qk = q_nope.size(-1) + q_pe.size(-1);
  args.total_q    = !is_varlen_q ? q_nope.size(0) * q_nope.size(1) : q_nope.size(0);

  args.seqlen_kv   = args.page_block_size;
  args.nr_heads_kv = 1;
  args.headdim_v   = ckv.size(-1);

  TORCH_CHECK(args.headdim_qk == 576, "flash_mla_asm() headdim_qk must be 576!");
  TORCH_CHECK(args.headdim_v == 512, "flash_mla_asm() headdim_v must be 576!");
  TORCH_CHECK(args.nr_heads_kv == 1, "flash_mla_asm() nr_heads_kv must be 1!");
  if (ckv.dim() == 4) {
    CHECK_SHAPE(ckv, args.nr_block, args.page_block_size, 1, 512);
    CHECK_SHAPE(kpe, args.nr_block, args.page_block_size, 1, 64);
    TORCH_CHECK(ckv.stride(-3) % 576 == 0, "flash_mla_asm() ckv stride -3 must be multiple of 576!");
    TORCH_CHECK(kpe.stride(-3) % 576 == 0, "flash_mla_asm() kpe stride -3 must be multiple of 576!");
    TORCH_CHECK(ckv.stride(0) % 576 == 0, "flash_mla_asm() ckv stride 0 must be multiple of 576!");
    TORCH_CHECK(kpe.stride(0) % 576 == 0, "flash_mla_asm() kpe stride 0 must be multiple of 576!");
  } else {
    CHECK_SHAPE(ckv, args.nr_block, args.page_block_size, 512);
    CHECK_SHAPE(kpe, args.nr_block, args.page_block_size, 64);
    TORCH_CHECK(ckv.stride(-2) % 576 == 0, "flash_mla_asm() ckv stride -2 must be 576!");
    TORCH_CHECK(kpe.stride(-2) % 576 == 0, "flash_mla_asm() kpe stride -2 must be 576!");
    TORCH_CHECK(ckv.stride(0) % 576 == 0, "flash_mla_asm() ckv stride 0 must be 576!");
    TORCH_CHECK(kpe.stride(0) % 576 == 0, "flash_mla_asm() kpe stride 0 must be 576!");
  }

  int max_num_blocks_per_seq = block_table.size(1);
  CHECK_SHAPE(seqlens_k, args.batch);
  CHECK_SHAPE(block_table, args.batch, max_num_blocks_per_seq);

  const int heads_ratio  = args.nr_heads / args.nr_heads_kv;
  const int q_seq_per_hk = args.seqlen_q * heads_ratio;

  if (!is_varlen_q) {
    q_nope = q_nope.view({args.batch, args.seqlen_q, args.nr_heads, 8, 64});
    // 0 3 2 1 4
    args.stride_q[0] = q_nope.stride(2);
    args.stride_q[1] = q_nope.stride(1);
    args.stride_q[2] = q_nope.stride(3);
    args.stride_q[3] = q_nope.stride(0);

    q_pe = q_pe.view({args.batch, args.seqlen_q, args.nr_heads, 1, 64});
    // 0 3 2 1 4
    args.stride_q_rope[0] = q_pe.stride(2);
    args.stride_q_rope[1] = q_pe.stride(1);
    args.stride_q_rope[2] = q_pe.stride(3);
    args.stride_q_rope[3] = q_pe.stride(0);
  } else {
    // varlen version

    q_nope = q_nope.view({args.total_q, args.nr_heads, 8, 64});

    args.stride_q[0] = q_nope.stride(1);
    args.stride_q[1] = q_nope.stride(0);
    args.stride_q[2] = q_nope.stride(2);

    q_pe = q_pe.view({args.total_q, args.nr_heads, 1, 64});

    args.stride_q_rope[0] = q_pe.stride(1);
    args.stride_q_rope[1] = q_pe.stride(0);
    args.stride_q_rope[2] = q_pe.stride(2);
  }

  auto ckv_stride_block     = ckv.stride(0);
  auto ckv_stride_page_size = ckv.stride(1);
  // TODO: opt here view + read instead of compute
  // [num_page, page_size, dim_qk * h_k]
  // [num_page, 8, 8, 1, dim_qk * h_k]
  // stride[stride_block, 8*stride_page, stride_page, 64*stride_page, 1]
  args.stride_k[0] = 8 * ckv_stride_page_size;
  args.stride_k[1] = ckv_stride_page_size;
  args.stride_k[2] = 64 * ckv_stride_page_size;
  args.stride_k[3] = ckv_stride_block;

  // TODO: opt here view + read instead of compute
  // [num_page, page_size, dim_qk * h_k]
  // [num_page, page_size, 1, 1, dim_qk*h_k]
  args.stride_v[0] = ckv_stride_page_size;
  args.stride_v[1] = ckv_stride_page_size;
  args.stride_v[2] = ckv_stride_page_size;
  args.stride_v[3] = ckv_stride_block;

  args.block_table_stride0 = block_table.stride(0);

  args.softmax_scale = softmax_scale;

  auto      opts     = q_nope.options();
  const int head_out = args.nr_heads_kv;

  at::Tensor out;
  at::Tensor out_lse;
  if (!is_varlen_q) {
    out     = at::empty({args.batch, q_seq_per_hk, head_out, headdim_latent}, opts);
    out_lse = at::empty({args.batch, head_out, q_seq_per_hk}, opts.dtype(at::kFloat));
  } else {
    out     = at::empty({args.total_q, args.nr_heads, headdim_latent}, opts);
    out_lse = at::empty({args.nr_heads, args.total_q}, opts.dtype(at::kFloat));
  }

  args.q_seq_per_hk = q_seq_per_hk;
  args.nr_mp_parts  = tile_scheduler_metadata.size(0);

  if (!is_varlen_q) {
    // [BSHD]
    args.batch_stride_out         = out.stride(0);
    args.nosplit_batch_stride_out = out.stride(0);
    args.seq_stride_out           = out.stride(1);  // not used in non-varlen, same as head_stride_out
    args.head_stride_out          = out.stride(2);  // == headdim_v

    args.batch_stride_lse         = out_lse.stride(0);
    args.nosplit_batch_stride_lse = out_lse.stride(0);
    args.head_stride_lse          = out_lse.stride(1);
  } else {
    // varlen
    args.seq_stride_out  = out.stride(-3);
    args.head_stride_out = out.stride(-2);

    args.head_stride_lse  = out_lse.stride(-2);
    args.batch_stride_lse = args.seqlen_q * 128;  // slplit_idx * max_sqlen_q * 128
    args.seq_stride_lse   = 128;                  // seqlen_q_idx * 128
  }

  // combine for load balance
  const int  total_num_splits  = args.batch + tile_scheduler_metadata.size(0);
  const int  num_mp_parts      = tile_scheduler_metadata.size(0);
  at::Tensor softmax_lse_accum = torch::empty({total_num_splits, head_out, q_seq_per_hk}, opts.dtype(at::kFloat));
  at::Tensor out_accum =
      torch::empty({total_num_splits, head_out, q_seq_per_hk, headdim_latent}, opts.dtype(at::kFloat));

  if (is_varlen_q) {
    args.batch_stride_out = out_accum.stride(0);
  }

  args.p_output     = out.data_ptr();
  args.p_lse_output = out_lse.data_ptr();

  args.out_accum_ptr    = out_accum.data_ptr();
  args.out_lseaccum_ptr = softmax_lse_accum.data_ptr();

  args.p_q      = q_nope.data_ptr();
  args.p_q_rope = q_pe.data_ptr();
  args.p_k      = ckv.data_ptr();
  args.p_v      = ckv.data_ptr();

  args.kv_cache_ptr                = nullptr;
  args.block_table_ptr             = block_table.data_ptr();
  args.cache_seqlen_ptr            = seqlens_k.data_ptr();
  args.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr();
  args.num_splits_ptr              = num_splits.data_ptr();
  args.seqlens_q_ptr               = !is_varlen_q ? nullptr : cu_seqlens_q.value().data_ptr();

  musaStream_t stream = at::musa::getCurrentMUSAStream().stream();

  using Kernel                         = mate::flash_mla::mubin::MLAAsmKernel;
  auto [params, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                          = Kernel(params, config, launch_config);
  kernel.run();

  mate::flash_mla::MlaCombineParams combine_params;
  combine_params.is_varlen_q          = args.is_varlen_q;
  combine_params.o_ptr                = out.data_ptr();
  combine_params.softmax_lse_ptr      = out_lse.data_ptr();
  combine_params.oaccum_ptr           = out_accum.data_ptr();
  combine_params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
  combine_params.num_splits_ptr       = num_splits.data_ptr<int>();
  combine_params.seqlens_q_ptr        = static_cast<int*>(args.seqlens_q_ptr);

  combine_params.o_batch_stride = !is_varlen_q ? out.stride(0) : q_seq_per_hk;
  combine_params.o_row_stride   = !is_varlen_q ? out.stride(1) : out.stride(1);
  combine_params.o_head_stride  = !is_varlen_q ? out.stride(2) : 0;

  combine_params.max_q_seq_per_hk = q_seq_per_hk;
  combine_params.total_q          = args.total_q;
  combine_params.h_r              = args.nr_heads;
  combine_params.h_k              = 1;
  combine_params.batch_size       = args.batch;
  combine_params.num_mp_parts     = num_mp_parts;

  if (args.data_type == at::kHalf) {
    if (is_varlen_q) {
      run_mla_combine_kernel<mutlass::half_t, true>(combine_params, stream);
    } else {
      run_mla_combine_kernel<mutlass::half_t, false>(combine_params, stream);
    }

  } else if (args.data_type == at::kBFloat16) {
    if (is_varlen_q) {
      run_mla_combine_kernel<mutlass::bfloat16_t, true>(combine_params, stream);
    } else {
      run_mla_combine_kernel<mutlass::bfloat16_t, false>(combine_params, stream);
    }

  } else {
    TORCH_CHECK(false, "flash_mla_asm() combine get unsupported data type!");
  }

  if (!is_varlen_q) {
    out     = out.view({args.batch, args.seqlen_q, args.nr_heads, headdim_latent});
    out_lse = out_lse.view({args.batch, args.seqlen_q, args.nr_heads});
  } else {
    out     = out.view({args.total_q, args.nr_heads, headdim_latent});
    out_lse = out_lse.view({args.nr_heads, args.total_q});
  }
  return {out, out_lse};
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "flash_mla_asm("
      "Tensor q_nope,"
      "Tensor q_pe,"
      "Tensor ckv,"
      "Tensor kpe,"
      "Tensor seqlens_k,"
      "Tensor block_table,"
      "Tensor tile_scheduler_metadata,"
      "Tensor num_splits,"
      "float sm_scale,"
      "bool is_causal = False,"
      "Tensor? cu_seqlens_q = None,"
      "int? max_seqlen_q = None)  -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("flash_mla_asm", &flash_mla_asm);
}
