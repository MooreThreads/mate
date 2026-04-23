#include <musa.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mutlass/fast_math.h>

#include <cmath>
#include <mute/algorithm/tuple_algorithms.hpp>
#include <mute/arch/copy_mp31_desc.hpp>
#include <mutex>
#include <optional>

#include "asm_common.hpp"
#include "attention_combine.hpp"
#include "mate/attention/flash_mla/mpxx_params.hpp"
#include "mubin/mp31/mp31_flash_mla_registry.hpp"
#include "op_utils.hpp"

namespace mate::flash_mla {

struct MLAAsmArgs {
  bool is_causal{};
  bool is_varlen_q{};

  DLDataType data_type;

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
    id.dtype       = dtype_equal(args.data_type, dl_bfloat16) ? 0 : 1;

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
                             dl_dtype_to_tme_type(args.data_type));
      TmeDesc tensor_desc_q_rope_(mute::make_tuple(tile_k, args.nr_heads, args.total_q, 1),
                                  mute::make_tuple(args.stride_q_rope[0], args.stride_q_rope[1], args.stride_q_rope[2]),
                                  args.p_q_rope,
                                  dl_dtype_to_tme_type(args.data_type));

      tensor_desc_q      = tensor_desc_q_;
      tensor_desc_q_rope = tensor_desc_q_rope_;

    } else {
      // is NOT varlen Q
      TmeDesc tensor_desc_q_(
          mute::make_tuple(tile_k, args.nr_heads, args.seqlen_q, args.headdim_v / tile_k, args.batch),
          mute::make_tuple(args.stride_q[0], args.stride_q[1], args.stride_q[2], args.stride_q[3]),
          args.p_q,
          dl_dtype_to_tme_type(args.data_type));

      TmeDesc tensor_desc_q_rope_(
          mute::make_tuple(tile_k, args.nr_heads, args.seqlen_q, 1, args.batch),
          mute::make_tuple(args.stride_q_rope[0], args.stride_q_rope[1], args.stride_q_rope[2], args.stride_q_rope[3]),
          args.p_q_rope,
          dl_dtype_to_tme_type(args.data_type));

      tensor_desc_q      = tensor_desc_q_;
      tensor_desc_q_rope = tensor_desc_q_rope_;
    }

    TmeDesc tensor_desc_k(
        mute::make_tuple(args.headdim_qk * args.nr_heads_kv, 8, 8, args.page_block_size / 64, args.nr_block),
        mute::make_tuple(args.stride_k[0], args.stride_k[1], args.stride_k[2], args.stride_k[3]),
        args.p_k,
        dl_dtype_to_tme_type(args.data_type));
    TmeDesc tensor_desc_v(
        mute::make_tuple(args.headdim_qk * args.nr_heads_kv, 1, 1, args.page_block_size, args.nr_block),
        mute::make_tuple(args.stride_v[0], args.stride_v[1], args.stride_v[2], args.stride_v[3]),
        args.p_v,
        dl_dtype_to_tme_type(args.data_type));

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

void flash_mla_asm(ffi::TensorView                q_nope,
                   ffi::TensorView                q_pe,
                   ffi::TensorView                ckv,
                   ffi::TensorView                kpe,
                   ffi::TensorView                seqlens_k,
                   ffi::TensorView                block_table,
                   ffi::TensorView                tile_scheduler_metadata,
                   ffi::TensorView                num_splits,
                   ffi::TensorView                out,
                   ffi::TensorView                out_lse,
                   double                         softmax_scale,
                   bool                           is_causal,
                   ffi::Optional<ffi::TensorView> cu_seqlens_q,
                   ffi::Optional<int64_t>         max_seqlen_q) {
  check_mp31(q_nope.device(), "flash_mla_asm");

  constexpr int headdim_latent = 512;
  constexpr int headdim_rope   = 64;

  CHECK_MUSA(q_nope);
  CHECK_MUSA(q_pe);
  CHECK_MUSA(ckv);
  CHECK_MUSA(kpe);
  CHECK_MUSA(seqlens_k);
  CHECK_MUSA(block_table);
  CHECK_MUSA(tile_scheduler_metadata);
  CHECK_MUSA(num_splits);
  CHECK_DEVICE(q_nope, q_pe);
  CHECK_DEVICE(q_nope, ckv);
  CHECK_DEVICE(q_nope, kpe);
  CHECK_DEVICE(q_nope, seqlens_k);
  CHECK_DEVICE(q_nope, block_table);
  CHECK_DEVICE(q_nope, tile_scheduler_metadata);
  CHECK_DEVICE(q_nope, num_splits);

  CHECK_CONTIGUOUS(tile_scheduler_metadata);
  CHECK_CONTIGUOUS(num_splits);
  CHECK_CONTIGUOUS(seqlens_k);
  CHECK_INPUT_TYPE(tile_scheduler_metadata, dl_int32);
  CHECK_INPUT_TYPE(num_splits, dl_int32);
  CHECK_INPUT_TYPE(seqlens_k, dl_int32);
  CHECK_INPUT_TYPE(block_table, dl_int32);
  CHECK_DIM(2, tile_scheduler_metadata);
  CHECK_DIM(1, num_splits);
  CHECK_DIM(1, seqlens_k);
  CHECK_DIM(2, block_table);
  TVM_FFI_ICHECK_EQ(block_table.stride(-1), 1) << "block_table must have contiguous last dimension";

  const bool is_varlen_q = cu_seqlens_q.has_value();
  if (is_varlen_q) {
    CHECK_HAS_VALUE_WITH_MSG(max_seqlen_q, "flash_mla_asm() varlen_q is enabled; max_seqlen_q must be provided");
    CHECK_DIM(3, q_nope);
    CHECK_DIM(3, q_pe);
    CHECK_MUSA(cu_seqlens_q.value());
    CHECK_CONTIGUOUS(cu_seqlens_q.value());
    CHECK_INPUT_TYPE(cu_seqlens_q.value(), dl_int32);
    CHECK_DEVICE(q_nope, cu_seqlens_q.value());
  } else {
    CHECK_DIM(4, q_nope);
    CHECK_DIM(4, q_pe);
  }
  TVM_FFI_ICHECK_EQ(q_nope.stride(-1), 1) << "q_nope must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(q_pe.stride(-1), 1) << "q_pe must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(ckv.stride(-1), 1) << "ckv must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(kpe.stride(-1), 1) << "kpe must have contiguous last dimension";

  TVM_FFI_ICHECK(dtype_equal(q_nope.dtype(), q_pe.dtype()) && dtype_equal(q_nope.dtype(), ckv.dtype()) &&
                 dtype_equal(q_nope.dtype(), kpe.dtype()))
      << "q_nope, q_pe, ckv and kpe must have the same dtype";
  TVM_FFI_ICHECK(is_bf16_or_fp16_dtype(q_nope.dtype())) << "flash_mla_asm only supports fp16 and bf16";

  mate::flash_mla::MLAAsmArgs args{};
  args.is_causal       = is_causal;
  args.is_varlen_q     = is_varlen_q;
  args.data_type       = q_nope.dtype();
  args.page_block_size = 64;
  args.nr_block        = static_cast<int32_t>(ckv.size(0));

  args.batch =
      !is_varlen_q ? static_cast<int32_t>(q_nope.size(0)) : static_cast<int32_t>(cu_seqlens_q.value().size(0) - 1);
  args.seqlen_q   = !is_varlen_q ? static_cast<int32_t>(q_nope.size(1)) : static_cast<int32_t>(max_seqlen_q.value());
  args.nr_heads   = static_cast<int32_t>(q_nope.size(-2));
  args.headdim_qk = static_cast<int32_t>(q_nope.size(-1) + q_pe.size(-1));
  args.total_q =
      !is_varlen_q ? static_cast<int32_t>(q_nope.size(0) * q_nope.size(1)) : static_cast<int32_t>(q_nope.size(0));
  args.seqlen_kv   = args.page_block_size;
  args.nr_heads_kv = 1;
  args.headdim_v   = static_cast<int32_t>(ckv.size(-1));

  TVM_FFI_ICHECK_EQ(args.headdim_qk, 576) << "flash_mla_asm() headdim_qk must be 576";
  TVM_FFI_ICHECK_EQ(args.headdim_v, 512) << "flash_mla_asm() headdim_v must be 512";
  TVM_FFI_ICHECK_EQ(args.nr_heads_kv, 1) << "flash_mla_asm() nr_heads_kv must be 1";
  TVM_FFI_ICHECK_EQ(q_pe.size(-1), headdim_rope) << "q_pe last dim must be 64";
  TVM_FFI_ICHECK_EQ(q_nope.size(-1), headdim_latent) << "q_nope last dim must be 512";
  TVM_FFI_ICHECK_EQ(q_nope.size(-2), q_pe.size(-2)) << "q_nope and q_pe head counts must match";
  if (!is_varlen_q) {
    TVM_FFI_ICHECK_EQ(q_nope.size(0), q_pe.size(0));
    TVM_FFI_ICHECK_EQ(q_nope.size(1), q_pe.size(1));
  } else {
    TVM_FFI_ICHECK_EQ(q_nope.size(0), q_pe.size(0));
  }

  if (ckv.dim() == 4) {
    expect_shape(ckv, {args.nr_block, args.page_block_size, 1, headdim_latent}, "ckv");
    expect_shape(kpe, {args.nr_block, args.page_block_size, 1, headdim_rope}, "kpe");
    TVM_FFI_ICHECK_EQ(kpe.dim(), 4) << "kpe must match ckv rank";
    TVM_FFI_ICHECK_EQ(ckv.stride(-3) % 576, 0) << "flash_mla_asm() ckv stride -3 must be multiple of 576";
    TVM_FFI_ICHECK_EQ(kpe.stride(-3) % 576, 0) << "flash_mla_asm() kpe stride -3 must be multiple of 576";
    TVM_FFI_ICHECK_EQ(ckv.stride(0) % 576, 0) << "flash_mla_asm() ckv stride 0 must be multiple of 576";
    TVM_FFI_ICHECK_EQ(kpe.stride(0) % 576, 0) << "flash_mla_asm() kpe stride 0 must be multiple of 576";
  } else {
    expect_shape(ckv, {args.nr_block, args.page_block_size, headdim_latent}, "ckv");
    expect_shape(kpe, {args.nr_block, args.page_block_size, headdim_rope}, "kpe");
    TVM_FFI_ICHECK_EQ(kpe.dim(), 3) << "kpe must match ckv rank";
    TVM_FFI_ICHECK_EQ(ckv.stride(-2) % 576, 0) << "flash_mla_asm() ckv stride -2 must be multiple of 576";
    TVM_FFI_ICHECK_EQ(kpe.stride(-2) % 576, 0) << "flash_mla_asm() kpe stride -2 must be multiple of 576";
    TVM_FFI_ICHECK_EQ(ckv.stride(0) % 576, 0) << "flash_mla_asm() ckv stride 0 must be multiple of 576";
    TVM_FFI_ICHECK_EQ(kpe.stride(0) % 576, 0) << "flash_mla_asm() kpe stride 0 must be multiple of 576";
  }

  const int max_num_blocks_per_seq = static_cast<int>(block_table.size(1));
  expect_shape(seqlens_k, {args.batch}, "seqlens_k");
  expect_shape(block_table, {args.batch, max_num_blocks_per_seq}, "block_table");
  TVM_FFI_ICHECK_EQ(tile_scheduler_metadata.size(1), mate::flash_mla::TileSchedulerMetaDataSize)
      << "tile_scheduler_metadata has unexpected second dimension";
  TVM_FFI_ICHECK_EQ(num_splits.size(0), args.batch + 1) << "num_splits must have shape [batch + 1]";

  const int heads_ratio = args.nr_heads / args.nr_heads_kv;
  TVM_FFI_ICHECK_EQ(args.nr_heads % args.nr_heads_kv, 0) << "nr_heads must be divisible by nr_heads_kv";
  const int q_seq_per_hk = args.seqlen_q * heads_ratio;

  CHECK_MUSA(out);
  CHECK_MUSA(out_lse);
  CHECK_DEVICE(out, q_nope);
  CHECK_DEVICE(out_lse, q_nope);
  TVM_FFI_ICHECK_EQ(out.stride(-1), 1) << "out must have contiguous last dimension";
  TVM_FFI_ICHECK_EQ(out_lse.stride(-1), 1) << "out_lse must have contiguous last dimension";
  TVM_FFI_ICHECK(dtype_equal(out.dtype(), q_nope.dtype())) << "out must have the same dtype as q_nope";
  TVM_FFI_ICHECK(dtype_equal(out_lse.dtype(), dl_float32)) << "out_lse must have dtype float32";
  if (!is_varlen_q) {
    expect_shape(out, {args.batch, args.seqlen_q, args.nr_heads, headdim_latent}, "out");
    expect_shape(out_lse, {args.batch, args.seqlen_q, args.nr_heads}, "out_lse");
    TVM_FFI_ICHECK_EQ(out.stride(1), out.size(-2) * out.stride(2))
        << "non-contiguous out layout is not representable for flash_mla_asm; materialize out first";
    TVM_FFI_ICHECK_EQ(out_lse.stride(1), out_lse.size(-1) * out_lse.stride(2))
        << "non-contiguous out_lse layout is not representable for flash_mla_asm; materialize out_lse first";
  } else {
    expect_shape(out, {args.total_q, args.nr_heads, headdim_latent}, "out");
    expect_shape(out_lse, {args.nr_heads, args.total_q}, "out_lse");
  }

  StridedTensorView q_nope_reshaped =
      !is_varlen_q
          ? StridedTensorView(
                q_nope,
                ffi::Shape{args.batch, args.seqlen_q, args.nr_heads, 8, 64},
                ffi::Shape{
                    q_nope.stride(0), q_nope.stride(1), q_nope.stride(2), 64 * q_nope.stride(-1), q_nope.stride(-1)})
          : StridedTensorView(
                q_nope,
                ffi::Shape{args.total_q, args.nr_heads, 8, 64},
                ffi::Shape{q_nope.stride(0), q_nope.stride(1), 64 * q_nope.stride(-1), q_nope.stride(-1)});
  StridedTensorView q_pe_reshaped =
      !is_varlen_q
          ? StridedTensorView(
                q_pe,
                ffi::Shape{args.batch, args.seqlen_q, args.nr_heads, 1, 64},
                ffi::Shape{q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), 64 * q_pe.stride(-1), q_pe.stride(-1)})
          : StridedTensorView(q_pe,
                              ffi::Shape{args.total_q, args.nr_heads, 1, 64},
                              ffi::Shape{q_pe.stride(0), q_pe.stride(1), 64 * q_pe.stride(-1), q_pe.stride(-1)});

  auto q_nope_work = q_nope_reshaped.view;
  auto q_pe_work   = q_pe_reshaped.view;
  if (!is_varlen_q) {
    args.stride_q[0]      = static_cast<int32_t>(q_nope_work.stride(2));
    args.stride_q[1]      = static_cast<int32_t>(q_nope_work.stride(1));
    args.stride_q[2]      = static_cast<int32_t>(q_nope_work.stride(3));
    args.stride_q[3]      = static_cast<int32_t>(q_nope_work.stride(0));
    args.stride_q_rope[0] = static_cast<int32_t>(q_pe_work.stride(2));
    args.stride_q_rope[1] = static_cast<int32_t>(q_pe_work.stride(1));
    args.stride_q_rope[2] = static_cast<int32_t>(q_pe_work.stride(3));
    args.stride_q_rope[3] = static_cast<int32_t>(q_pe_work.stride(0));
  } else {
    args.stride_q[0]      = static_cast<int32_t>(q_nope_work.stride(1));
    args.stride_q[1]      = static_cast<int32_t>(q_nope_work.stride(0));
    args.stride_q[2]      = static_cast<int32_t>(q_nope_work.stride(2));
    args.stride_q_rope[0] = static_cast<int32_t>(q_pe_work.stride(1));
    args.stride_q_rope[1] = static_cast<int32_t>(q_pe_work.stride(0));
    args.stride_q_rope[2] = static_cast<int32_t>(q_pe_work.stride(2));
  }

  const int64_t ckv_stride_block     = ckv.stride(0);
  const int64_t ckv_stride_page_size = ckv.stride(1);
  args.stride_k[0]                   = static_cast<int32_t>(8 * ckv_stride_page_size);
  args.stride_k[1]                   = static_cast<int32_t>(ckv_stride_page_size);
  args.stride_k[2]                   = static_cast<int32_t>(64 * ckv_stride_page_size);
  args.stride_k[3]                   = static_cast<int32_t>(ckv_stride_block);
  args.stride_v[0]                   = static_cast<int32_t>(ckv_stride_page_size);
  args.stride_v[1]                   = static_cast<int32_t>(ckv_stride_page_size);
  args.stride_v[2]                   = static_cast<int32_t>(ckv_stride_page_size);
  args.stride_v[3]                   = static_cast<int32_t>(ckv_stride_block);

  args.block_table_stride0 = static_cast<int32_t>(block_table.stride(0));
  args.softmax_scale       = softmax_scale;
  args.q_seq_per_hk        = q_seq_per_hk;
  args.nr_mp_parts         = static_cast<int32_t>(tile_scheduler_metadata.size(0));

  const int                        head_out = args.nr_heads_kv;
  std::optional<StridedTensorView> out_view;
  std::optional<StridedTensorView> out_lse_view;
  ffi::TensorView                  out_work     = out;
  ffi::TensorView                  out_lse_work = out_lse;
  if (!is_varlen_q) {
    out_view.emplace(out,
                     ffi::Shape{args.batch, q_seq_per_hk, head_out, headdim_latent},
                     ffi::Shape{out.stride(0), out.stride(2), out.stride(2), out.stride(3)});
    out_lse_view.emplace(
        out_lse, ffi::Shape{args.batch, head_out, q_seq_per_hk}, ffi::Shape{out_lse.stride(0), out_lse.stride(0), 1});
    out_work     = out_view->view;
    out_lse_work = out_lse_view->view;
  }

  if (!is_varlen_q) {
    args.batch_stride_out         = static_cast<int32_t>(out_work.stride(0));
    args.nosplit_batch_stride_out = static_cast<int32_t>(out_work.stride(0));
    args.seq_stride_out           = static_cast<int32_t>(out_work.stride(1));
    args.head_stride_out          = static_cast<int32_t>(out_work.stride(2));
    args.batch_stride_lse         = static_cast<int32_t>(out_lse_work.stride(0));
    args.nosplit_batch_stride_lse = static_cast<int32_t>(out_lse_work.stride(0));
    args.head_stride_lse          = static_cast<int32_t>(out_lse_work.stride(1));
  } else {
    args.seq_stride_out   = static_cast<int32_t>(out.stride(-3));
    args.head_stride_out  = static_cast<int32_t>(out.stride(-2));
    args.head_stride_lse  = static_cast<int32_t>(out_lse.stride(-2));
    args.batch_stride_lse = args.seqlen_q * 128;
    args.seq_stride_lse   = 128;
  }

  const int   total_num_splits = args.batch + static_cast<int>(tile_scheduler_metadata.size(0));
  const int   num_mp_parts     = static_cast<int>(tile_scheduler_metadata.size(0));
  ffi::Tensor softmax_lse_accum =
      alloc_tensor(ffi::Shape{total_num_splits, head_out, q_seq_per_hk}, dl_float32, q_nope.device());
  ffi::Tensor out_accum =
      alloc_tensor(ffi::Shape{total_num_splits, head_out, q_seq_per_hk, headdim_latent}, dl_float32, q_nope.device());

  if (is_varlen_q) {
    args.batch_stride_out = static_cast<int32_t>(out_accum.stride(0));
  }

  args.p_output                    = !is_varlen_q ? out_work.data_ptr() : out.data_ptr();
  args.p_lse_output                = !is_varlen_q ? out_lse_work.data_ptr() : out_lse.data_ptr();
  args.out_accum_ptr               = out_accum.data_ptr();
  args.out_lseaccum_ptr            = softmax_lse_accum.data_ptr();
  args.p_q                         = q_nope_work.data_ptr();
  args.p_q_rope                    = q_pe_work.data_ptr();
  args.p_k                         = ckv.data_ptr();
  args.p_v                         = ckv.data_ptr();
  args.kv_cache_ptr                = nullptr;
  args.block_table_ptr             = block_table.data_ptr();
  args.cache_seqlen_ptr            = seqlens_k.data_ptr();
  args.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr();
  args.num_splits_ptr              = num_splits.data_ptr();
  args.seqlens_q_ptr               = is_varlen_q ? cu_seqlens_q.value().data_ptr() : nullptr;

  musaStream_t stream = get_stream(q_nope.device());

  using Kernel                         = mate::flash_mla::mubin::MLAAsmKernel;
  auto [params, config, launch_config] = Kernel::to_underlying_arguments(args, stream);
  auto kernel                          = Kernel(params, config, launch_config);
  kernel.run();

  mate::flash_mla::MlaCombineParams combine_params{};
  combine_params.is_varlen_q          = args.is_varlen_q;
  combine_params.o_ptr                = !is_varlen_q ? out_work.data_ptr() : out.data_ptr();
  combine_params.softmax_lse_ptr      = !is_varlen_q ? out_lse_work.data_ptr() : out_lse.data_ptr();
  combine_params.oaccum_ptr           = out_accum.data_ptr();
  combine_params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
  combine_params.num_splits_ptr       = static_cast<int*>(num_splits.data_ptr());
  combine_params.seqlens_q_ptr        = static_cast<int*>(args.seqlens_q_ptr);
  combine_params.o_batch_stride       = !is_varlen_q ? static_cast<int>(out_work.stride(0)) : q_seq_per_hk;
  combine_params.o_row_stride  = !is_varlen_q ? static_cast<int>(out_work.stride(1)) : static_cast<int>(out.stride(1));
  combine_params.o_head_stride = !is_varlen_q ? static_cast<int>(out_work.stride(2)) : 0;
  combine_params.max_q_seq_per_hk = q_seq_per_hk;
  combine_params.total_q          = args.total_q;
  combine_params.h_r              = args.nr_heads;
  combine_params.h_k              = 1;
  combine_params.batch_size       = args.batch;
  combine_params.num_mp_parts     = num_mp_parts;

  if (dtype_equal(args.data_type, dl_float16)) {
    if (is_varlen_q) {
      run_mla_combine_kernel<mutlass::half_t, true>(combine_params, stream);
    } else {
      run_mla_combine_kernel<mutlass::half_t, false>(combine_params, stream);
    }
  } else if (dtype_equal(args.data_type, dl_bfloat16)) {
    if (is_varlen_q) {
      run_mla_combine_kernel<mutlass::bfloat16_t, true>(combine_params, stream);
    } else {
      run_mla_combine_kernel<mutlass::bfloat16_t, false>(combine_params, stream);
    }
  } else {
    TVM_FFI_ICHECK(false) << "flash_mla_asm() combine got unsupported data type";
  }

  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_mla_asm, flash_mla_asm);
