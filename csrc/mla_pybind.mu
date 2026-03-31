#include <mutlass/device_kernel.h>
#include <torch/all.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include "attention_combine.hpp"
#include "attention_scheduler.hpp"
#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_mla_collective_tme_warpspecialized.hpp"
#include "kernel/mla_kernel_tme_warpspecialzed.hpp"
#include "mate/attention/flash_mla/fmha_mla_collective_tme_warpspecialized.hpp"
#include "mate/attention/flash_mla/mla_collective_epilogue.hpp"
#include "mate/attention/flash_mla/mla_kernel_tme_warpspecialized.hpp"
#include "mate/attention/flash_mla/mla_tile_scheduler.hpp"
#include "mate/attention/flash_mla/mpxx_mla_combine.hpp"
#include "mate/attention/flash_mla/mpxx_params.hpp"
#include "mate_utils.muh"
#include "torch_utils.hpp"

using namespace mute;

// DecodingAttnImplMeta - A struct to hold metadata for Decoding Attention Implementation
struct DecodingAttnImplMeta {
  int num_mp_parts;
  int fixed_overhead_num_blocks;
  int k_block_size;
};

template <int TILE_M = 128>
DecodingAttnImplMeta get_attn_impl_meta(Arch               arch,
                                        int                mp_count,
                                        int                num_q_tokens_per_head_k,
                                        int                h_k,
                                        std::optional<int> h_q_,
                                        bool               is_fp8_kvcache,
                                        bool               is_sparse_attn) {
  if (arch.is_mp31()) {
    if (is_sparse_attn) {
      if (is_fp8_kvcache) {
        // FP8 + Sparse MLA
        TORCH_CHECK(false, "Sparse FP8 MLA is not supported on MP31");
      } else {
        // Sparse BF16 MLA
        TORCH_CHECK(false, "Sparse BF16 MLA is not supported on MP31");
      }
    } else {
      if (is_fp8_kvcache) {
        // Dense FP8 MLA
        TORCH_CHECK(false, "Dense FP8 MLA is not supported on MP31");
      } else {
        // Dense BF16 MLA
        return {std::max(mp_count / h_k / mutlass::ceil_div(num_q_tokens_per_head_k, TILE_M), 1), 5, 64};
      }
    }
  } else {
    TORCH_CHECK(false, "Unsupported GPU architecture");
  }
}

template <typename Element, typename ElementO, bool IsCausal>
void dispatch_mla_kernel(at::Tensor q_nope,
                         at::Tensor q_pe,
                         at::Tensor ckv,
                         at::Tensor kpe,
                         at::Tensor out,
                         at::Tensor softmax_lse,
                         at::Tensor page_table,
                         at::Tensor kv_len,

                         float   sm_scale,
                         int     batch,
                         int     page_count,
                         int     page_size,
                         int     head_num_q,
                         int     head_dim_ckv,
                         int     head_dim_kpe,
                         int     seqlen_q,
                         int64_t q_nope_stride_batch,
                         int64_t q_nope_stride_num_heads,
                         int64_t q_pe_stride_batch,
                         int64_t q_pe_stride_num_heads,
                         int64_t ckv_stride_page_num,
                         int64_t ckv_stride_num_heads,
                         int64_t kpe_stride_page_num,
                         int64_t kpe_stride_num_heads,
                         int32_t pt_stride_batch,
                         int64_t o_stride_batch,
                         int64_t o_stride_num_heads,
                         int32_t lse_stride_batch,
                         int32_t lse_stride_num_heads) {
  // unused q_stride_seqlen,
  auto stride_q_nope = make_stride(q_nope_stride_num_heads, _1{}, make_stride(_0{}, q_nope_stride_batch));
  auto stride_q_pe   = make_stride(q_pe_stride_num_heads, _1{}, make_stride(_0{}, q_pe_stride_batch));
  auto stride_ckv    = make_stride(ckv_stride_num_heads, _1{}, make_stride(_0{}, ckv_stride_page_num));
  auto stride_kpe    = make_stride(kpe_stride_num_heads, _1{}, make_stride(_0{}, kpe_stride_page_num));
  auto stride_pt     = make_stride(pt_stride_batch, _1{});
  auto stride_o      = make_stride(o_stride_num_heads, _1{}, make_stride(_0{}, o_stride_batch));
  auto stride_lse    = make_stride(_1{}, make_stride(lse_stride_num_heads, lse_stride_batch));

  using StrideQ = decltype(stride_q_nope);
  // using StrideQPe = decltype(stride_q_pe);
  using StrideC = decltype(stride_ckv);
  // using StrideKPe = decltype(stride_kpe);
  using StrideO   = decltype(stride_o);
  using StrideLse = decltype(stride_lse);

  // CTA_Q, CTA_KV, D_QK, D_VO
  // using TileShape = Shape<Int<CTA_Q>, Int<CTA_K>, Shape<Int<Latent>, Int<Rope>>>;
  using TileShape = Shape<_32, _64, Shape<_512, _64>>;

  using Fusion = conditional_t<IsCausal,
                               mutlass::fmha::collective::CausalFusion<false, true>,
                               mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::
      FmhaMlaMainloopTmeWarpSpecializedV2<Element, float, TileShape, StrideQ, StrideC, Fusion>;

  using CollectiveEpilogue =
      mutlass::fmha::collective::FmhaFwdEpilogue<Element, float, Shape<_32, _512>, StrideO, StrideLse>;

  using MlaKernel = mutlass::fmha::kernel::MlaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue>;

  using ProblemShapeType = typename MlaKernel::ProblemShape;
  ProblemShapeType problem_shape =
      make_shape(seqlen_q, page_size, make_shape(head_dim_ckv, head_dim_kpe), head_num_q, page_count, batch);
  typename MlaKernel::CollectiveMainloop::Fusion::Arguments fusion;
  int                                                       head_num_kv = 1;
  mutlass::FastDivmod                                       divmod_hr(head_num_q / head_num_kv);
  fusion.fast_divmod_hr = divmod_hr;

  typename MlaKernel::Arguments arguments{
      problem_shape,
      {static_cast<Element*>(q_nope.data_ptr()),
       stride_q_nope,
       static_cast<Element*>(q_pe.data_ptr()),
       stride_q_pe,
       static_cast<Element*>(ckv.data_ptr()),
       stride_ckv,
       static_cast<Element*>(kpe.data_ptr()),
       stride_kpe,
       static_cast<int*>(page_table.data_ptr()),
       stride_pt,
       static_cast<int*>(kv_len.data_ptr()),
       sm_scale,
       fusion},
      {
          static_cast<ElementO*>(out.data_ptr()),
          stride_o,
          static_cast<float*>(softmax_lse.data_ptr()),
          stride_lse,
      },
  };

  musaStream_t               stream     = at::musa::getCurrentMUSAStream();
  typename MlaKernel::Params params     = MlaKernel::to_underlying_arguments(arguments);
  int                        num_splits = 1;

  dim3 grid_dim{static_cast<uint32_t>(ceil_div(head_num_q * seqlen_q, get<0>(TileShape{}))),
                static_cast<uint32_t>(num_splits),
                static_cast<uint32_t>(get<5>(problem_shape))};
  mutlass::device_kernel<MlaKernel>
      <<<grid_dim, MlaKernel::MaxThreadsPerBlock, MlaKernel::SharedStorageSize, stream>>>(params);
  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

std::vector<at::Tensor> mla(at::Tensor                q_nope,
                            at::Tensor                q_pe,
                            at::Tensor                ckv,
                            at::Tensor                kpe,
                            at::Tensor                page_table,  // (batch, max_num_pages) int32
                            at::Tensor                kv_len,      // (batch) int32
                            double                    sm_scale,
                            std::optional<at::Tensor> out_,
                            std::optional<at::Tensor> lse_,
                            bool                      is_causal) {
  musaDeviceProp dprops;
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, q_nope.device().index()));
  TORCH_CHECK(dprops.major >= 3, "mla only supports MP31 GPUs or newer.");

  auto q_type = q_nope.scalar_type();

  const int batch        = kv_len.size(0);
  int       seqlen_q     = q_nope.size(1);
  int       head_num_q   = q_nope.size(-2);
  int       head_dim_ckv = q_nope.size(-1);
  int       head_dim_kpe = q_pe.size(-1);
  int       page_size    = ckv.size(1);
  int       page_count   = ckv.size(0);

  auto       opts     = q_nope.options();
  auto       out_type = q_type;
  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
  } else {
    out = torch::empty({batch, seqlen_q, head_num_q, head_dim_ckv}, opts.dtype(out_type));
  }

  auto       lse_type = at::ScalarType::Float;
  at::Tensor lse;
  if (lse_.has_value()) {
    lse = lse_.value();
  } else {
    lse = torch::empty({batch, seqlen_q, head_num_q}, opts.dtype(lse_type));
  }

  TORCH_CHECK(page_size == 64, "page size only support 64 now");
  TORCH_CHECK(head_dim_ckv == 512, "head dim ckv only support 512 now");
  TORCH_CHECK(head_dim_kpe == 64, "head dim kpe only support 64 now");

  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
              "mla only supports fp16 and bf16 data type");
  TORCH_CHECK(q_pe.scalar_type() == q_type && ckv.scalar_type() == q_type && kpe.scalar_type() == q_type,
              "input tensor must have the same dtype");
  TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs");
  TORCH_CHECK(lse.scalar_type() == lse_type, "lse must have dtype float");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int && kv_len.scalar_type() == at::ScalarType::Int,
              "page_table and kv_len must have dtype int32");

  CHECK_SHAPE(q_nope, batch, seqlen_q, head_num_q, head_dim_ckv);
  CHECK_SHAPE(q_pe, batch, seqlen_q, head_num_q, head_dim_kpe);
  if (ckv.dim() == 3) {
    CHECK_SHAPE(ckv, page_count, page_size, head_dim_ckv);
    CHECK_SHAPE(kpe, page_count, page_size, head_dim_kpe);
  } else {
    CHECK_SHAPE(ckv, page_count, page_size, 1, head_dim_ckv);
    CHECK_SHAPE(kpe, page_count, page_size, 1, head_dim_kpe);
  }
  CHECK_SHAPE(out, batch, seqlen_q, head_num_q, head_dim_ckv);
  CHECK_SHAPE(lse, batch, seqlen_q, head_num_q);
  CHECK_SHAPE(kv_len, batch);
  TORCH_CHECK(page_table.size(0) == batch, "page_table size(0) must equal batch size");

  TORCH_CHECK(q_nope.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(q_pe.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(ckv.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(kpe.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(page_table.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(kv_len.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");

  CHECK_MUSA(q_nope);
  CHECK_MUSA(q_pe);
  CHECK_MUSA(ckv);
  CHECK_MUSA(kpe);
  CHECK_MUSA(page_table);
  CHECK_MUSA(kv_len);
  CHECK_MUSA(out);

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_nope));

#define LAUNCH_KERNEL(QKV_TYPE, OUT_TYPE, IS_CAUSAL)                         \
  [&] {                                                                      \
    dispatch_mla_kernel<QKV_TYPE, OUT_TYPE, IS_CAUSAL>(q_nope,               \
                                                       q_pe,                 \
                                                       ckv,                  \
                                                       kpe,                  \
                                                       out,                  \
                                                       lse,                  \
                                                       page_table,           \
                                                       kv_len,               \
                                                       sm_scale,             \
                                                       batch,                \
                                                       page_count,           \
                                                       page_size,            \
                                                       head_num_q,           \
                                                       head_dim_ckv,         \
                                                       head_dim_kpe,         \
                                                       seqlen_q,             \
                                                       q_nope.stride(0),     \
                                                       q_nope.stride(2),     \
                                                       q_pe.stride(0),       \
                                                       q_pe.stride(2),       \
                                                       ckv.stride(0),        \
                                                       ckv.stride(1),        \
                                                       kpe.stride(0),        \
                                                       kpe.stride(1),        \
                                                       page_table.stride(0), \
                                                       out.stride(0),        \
                                                       out.stride(2),        \
                                                       lse.stride(0),        \
                                                       lse.stride(2));       \
  }()

  if (q_type == at::ScalarType::Half) {
    if (is_causal) {
      LAUNCH_KERNEL(mutlass::half_t, mutlass::half_t, true);
    } else {
      LAUNCH_KERNEL(mutlass::half_t, mutlass::half_t, false);
    }
  } else if (q_type == at::ScalarType::BFloat16) {
    if (is_causal) {
      LAUNCH_KERNEL(mutlass::bfloat16_t, mutlass::bfloat16_t, true);
    } else {
      LAUNCH_KERNEL(mutlass::bfloat16_t, mutlass::bfloat16_t, false);
    }
  }

  return {out, lse};
}

std::vector<at::Tensor> mla_with_kvcache(at::Tensor&       q_latent,   // (b, s_q, h, d) or (total_q, h, d) if varlen_q
                                         at::Tensor&       q_rope,     // (b, s_q, h, d) or (total_q, h, d) if varlen_q
                                         at::Tensor const& ckv,        // num_block, page_size, h_k, headdim
                                         at::Tensor const& kpe,        // num_block, page_size, h_k, headdim
                                         at::Tensor const& seqlens_k,  // batch_size
                                         std::optional<at::Tensor> const& cu_seqlens_q_,  // batch_size + 1
                                         std::optional<int64_t>           max_seqlen_q_,
                                         at::Tensor const& block_table,  // batch_size, max_num_blocks_per_seq
                                         at::Tensor const& tile_scheduler_metadata,
                                         at::Tensor const& num_splits,
                                         double const      softmax_scale,
                                         bool const        is_causal) {
  musaDeviceProp dprops;
  MATE_MUSA_RUNTIME_CHECK(musaGetDeviceProperties(&dprops, q_latent.device().index()));
  TORCH_CHECK(dprops.major >= 3, "mla only supports MP31 GPUs or newer.");

  int const mp_count = dprops.multiProcessorCount;

  auto q_type = q_latent.scalar_type();

  // Get is_varlen_q, and check
  at::Tensor cu_seqlens_q;
  bool const is_varlen_q = cu_seqlens_q_.has_value();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.value();
    CHECK_MUSA(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
  }

  // Q shape: (b, s_q, h, d) or (total_q, h, d) if varlen_q
  auto const sizes_latent    = q_latent.sizes();
  auto const sizes_rope      = q_rope.sizes();
  int const  batch_size      = !is_varlen_q ? sizes_latent[0] : cu_seqlens_q.size(0) - 1;
  int        seqlen_q        = !is_varlen_q ? sizes_latent[1] : max_seqlen_q_.value();
  int        total_q         = !is_varlen_q ? batch_size * sizes_latent[1] : sizes_latent[0];
  int        num_heads_q     = q_latent.size(-2);
  int const  head_dim_latent = q_latent.size(-1);
  int const  head_dim_rope   = q_rope.size(-1);

  int const num_blocks = ckv.size(0);
  int const page_size  = ckv.size(1);

  int constexpr num_heads_k = 1;
  int const num_mp_parts    = tile_scheduler_metadata.size(0);

  int const max_num_blocks_per_seq = block_table.size(1);

  Arch arch = {dprops.major, dprops.minor};

  TORCH_CHECK(page_size == 64, "page size only support 64 now");
  TORCH_CHECK(head_dim_latent == 512, "head dim latent only support 512 now");
  TORCH_CHECK(head_dim_rope == 64, "head dim rope only support 64 now");

  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
              "mla only supports fp16 and bf16 data type");
  TORCH_CHECK(q_rope.scalar_type() == q_type && ckv.scalar_type() == q_type && kpe.scalar_type() == q_type,
              "input tensor must have the same dtype");

  // Q shape: (b, s_q, h, d) or (total_q, h, d) if varlen_q
  if (!is_varlen_q) {
    CHECK_SHAPE(q_latent, batch_size, seqlen_q, num_heads_q, head_dim_latent);
    CHECK_SHAPE(q_rope, batch_size, seqlen_q, num_heads_q, head_dim_rope);
  } else {
    CHECK_SHAPE(q_latent, total_q, num_heads_q, head_dim_latent);
    CHECK_SHAPE(q_rope, total_q, num_heads_q, head_dim_rope);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }

  if (ckv.dim() == 3) {
    CHECK_SHAPE(ckv, num_blocks, page_size, head_dim_latent);
    CHECK_SHAPE(kpe, num_blocks, page_size, head_dim_rope);
  } else {
    CHECK_SHAPE(ckv, num_blocks, page_size, num_heads_k, head_dim_latent);
    CHECK_SHAPE(kpe, num_blocks, page_size, num_heads_k, head_dim_rope);
  }
  CHECK_SHAPE(seqlens_k, batch_size);
  CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);

  TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype int32");
  TORCH_CHECK(num_heads_k == 1, "The number of k heads must be 1");

  CHECK_MUSA(q_latent);
  CHECK_MUSA(q_rope);
  CHECK_MUSA(ckv);
  CHECK_MUSA(kpe);
  CHECK_MUSA(block_table);
  CHECK_MUSA(seqlens_k);

  const at::musa::OptionalMUSAGuard device_guard(device_of(q_latent));
  auto                              stream = at::musa::getCurrentMUSAStream().stream();

  auto opts = q_latent.options();

  int const heads_ratio  = num_heads_q / num_heads_k;
  int const q_seq_per_hk = seqlen_q * heads_ratio;

  if (!is_varlen_q) {
    q_latent = q_latent.view({batch_size, seqlen_q, num_heads_k, heads_ratio, head_dim_latent})
                   .transpose(2, 3)
                   .reshape({batch_size, q_seq_per_hk, num_heads_k, head_dim_latent});
    q_rope = q_rope.view({batch_size, seqlen_q, num_heads_k, heads_ratio, head_dim_rope})
                 .transpose(2, 3)
                 .reshape({batch_size, q_seq_per_hk, num_heads_k, head_dim_rope});
  } else {
    int const total_q_per_hk = total_q * heads_ratio;

    q_latent = q_latent.view({total_q, num_heads_k, heads_ratio, head_dim_latent})
                   .transpose(1, 2)
                   .reshape({total_q_per_hk, num_heads_k, head_dim_latent})
                   .view({total_q, num_heads_q, head_dim_latent});
    q_rope = q_rope.view({total_q, num_heads_k, heads_ratio, head_dim_rope})
                 .transpose(1, 2)
                 .reshape({total_q_per_hk, num_heads_k, head_dim_rope})
                 .view({total_q, num_heads_q, head_dim_rope});
  }

  // Meatadata relative
  TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
  TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");

  at::Tensor out         = !is_varlen_q ? torch::empty({batch_size, q_seq_per_hk, num_heads_k, head_dim_latent}, opts)
                                        : torch::empty({total_q, heads_ratio, num_heads_k, head_dim_latent}, opts);
  at::Tensor softmax_lse = !is_varlen_q ? torch::empty({batch_size, num_heads_k, q_seq_per_hk}, opts.dtype(at::kFloat))
                                        : torch::empty({num_heads_k, heads_ratio, total_q}, opts.dtype(at::kFloat));

  int const  total_num_splits  = batch_size + tile_scheduler_metadata.size(0);
  at::Tensor softmax_lse_accum = torch::empty({total_num_splits, num_heads_k, q_seq_per_hk}, opts.dtype(at::kFloat));
  at::Tensor out_accum =
      torch::empty({total_num_splits, num_heads_k, q_seq_per_hk, head_dim_latent}, opts.dtype(at::kFloat));
  CHECK_CONTIGUOUS(out_accum);
  CHECK_CONTIGUOUS(softmax_lse_accum);

  // launch kernel
  auto stride_q_latent = !is_varlen_q
                             ? make_stride(q_latent.stride(1), _1{}, make_stride(_0{}, int(q_latent.stride(0))))
                             : make_stride(q_latent.stride(1), _1{}, make_stride(_0{}, 0));
  auto stride_q_rope =
      !is_varlen_q ? make_stride(q_rope.stride(1), _1{}, make_stride(_0{}, int(q_rope.stride(0))))
                   : make_stride(q_rope.stride(1), _1{}, make_stride(_0{}, 0));  // q_rope shape: [total_q, h, d]
  auto stride_ckv = make_stride(ckv.stride(1), _1{}, make_stride(_0{}, ckv.stride(0)));
  auto stride_kpe = make_stride(kpe.stride(1), _1{}, make_stride(_0{}, kpe.stride(0)));
  auto stride_pt  = make_stride(int(block_table.stride(0)), _1{});

  auto stride_out = !is_varlen_q ? make_stride(out.stride(2), _1{}, make_stride(_0{}, int(out.stride(0))))
                                 : make_stride(out.stride(2), _1{}, make_stride(_0{}, 0));
  auto stride_lse = !is_varlen_q
                        ? make_stride(_1{}, make_stride(int(softmax_lse.stride(1)), int(softmax_lse.stride(0))))
                        : make_stride(_1{}, make_stride(int(softmax_lse.stride(0)), 0));

  auto run_mla = [&](auto element_type, auto use_trival_scheduler, auto causal, auto varlen_q) {
    static constexpr bool UseTrivalScheduler = decltype(use_trival_scheduler)::value;
    static constexpr bool Causal             = decltype(causal)::value;
    static constexpr bool VarlenQ            = decltype(varlen_q)::value;

    using StrideQ   = decltype(stride_q_latent);
    using StrideC   = decltype(stride_ckv);
    using StrideO   = decltype(stride_out);
    using StrideLse = decltype(stride_lse);

    using Element   = decltype(element_type);
    using TileShape = Shape<_128, _64, Shape<_512, _64>>;

    using Fusion = conditional_t<Causal,
                                 mutlass::fmha::collective::CausalFusion<false, true>,
                                 mutlass::fmha::collective::DefaultFusion>;

    using CollectiveMainloop = mate::flash_mla::
        FmhaMlaMainloopTmeWarpSpecialized<Element, float, TileShape, StrideQ, StrideC, Fusion, VarlenQ>;

    using CollectiveEpilogue =
        mate::flash_mla::MlaFwdEpilogue<Element, float, Shape<_32, _512>, StrideO, StrideLse, VarlenQ>;

    using TileScheduler = mate::flash_mla::FlashMlaTileScheduler<UseTrivalScheduler>;

    using MlaKernel =
        mate::flash_mla::MlaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

    typename MlaKernel::ProblemShape problem_shape;

    auto problem_size = make_shape(seqlen_q,
                                   page_size,
                                   make_shape(head_dim_latent, head_dim_rope),
                                   num_heads_q,
                                   num_blocks,
                                   batch_size,
                                   total_q,
                                   seqlen_q);

    get<0>(problem_shape) =
        mutlass::fmha::collective::VariableLength{!is_varlen_q ? static_cast<int>(get<0>(problem_size)) : total_q,
                                                  static_cast<int>(seqlen_q),
                                                  !is_varlen_q ? nullptr : static_cast<int*>(cu_seqlens_q.data_ptr())};
    get<1>(problem_shape) = get<1>(problem_size);  // page size
    get<2>(problem_shape) = get<2>(problem_size);  // D
    get<3>(problem_shape) = get<3>(problem_size);  // H_q
    get<4>(problem_shape) = get<4>(problem_size);  // num_blocks
    get<5>(problem_shape) = get<5>(problem_size);  // batch
    get<6>(problem_shape) = get<6>(problem_size);  // total_q (varlen q)
    get<7>(problem_shape) = get<7>(problem_size);  // max_q (varlen q)

    typename Fusion::Arguments fusion;
    int                        head_num_kv = 1;
    mutlass::FastDivmod        divmod_hr(heads_ratio);
    fusion.fast_divmod_hr = divmod_hr;

    typename MlaKernel::Arguments arguments{problem_shape,
                                            {static_cast<Element*>(q_latent.data_ptr()),
                                             stride_q_latent,
                                             static_cast<Element*>(q_rope.data_ptr()),
                                             stride_q_rope,
                                             static_cast<Element*>(ckv.data_ptr()),
                                             stride_ckv,
                                             static_cast<Element*>(kpe.data_ptr()),
                                             stride_kpe,
                                             static_cast<int32_t*>(block_table.data_ptr()),
                                             stride_pt,
                                             static_cast<int32_t*>(seqlens_k.data_ptr()),
                                             static_cast<float>(softmax_scale),
                                             fusion,
                                             VarlenQ ? static_cast<int64_t*>(cu_seqlens_q.data_ptr()) : nullptr},
                                            {static_cast<Element*>(out.data_ptr()),
                                             stride_out,
                                             static_cast<float*>(softmax_lse.data_ptr()),
                                             stride_lse,
                                             static_cast<float*>(out_accum.data_ptr()),
                                             static_cast<float*>(softmax_lse_accum.data_ptr()),
                                             q_seq_per_hk},
                                            {
                                                num_mp_parts,
                                                static_cast<int32_t*>(tile_scheduler_metadata.data_ptr()),
                                                static_cast<int32_t*>(num_splits.data_ptr()),
                                            }};

    typename MlaKernel::Params params   = MlaKernel::to_underlying_arguments(arguments);
    dim3                       grid_dim = TileScheduler::get_grid_shape(params.tile_scheduler);

    mutlass::device_kernel<MlaKernel>
        <<<grid_dim, MlaKernel::MaxThreadsPerBlock, MlaKernel::SharedStorageSize, stream>>>(params);

    // run combine
    if constexpr (!UseTrivalScheduler) {
      mate::flash_mla::MlaCombineParams combine_params;
      combine_params.batch_size           = batch_size;
      combine_params.num_mp_parts         = num_mp_parts;
      combine_params.o_ptr                = out.data_ptr();
      combine_params.softmax_lse_ptr      = softmax_lse.data_ptr();
      combine_params.oaccum_ptr           = out_accum.data_ptr();
      combine_params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
      combine_params.num_splits_ptr       = num_splits.data_ptr<int>();
      combine_params.seqlens_q_ptr        = !varlen_q ? nullptr : cu_seqlens_q.data_ptr<int>();
      combine_params.max_q_seq_per_hk     = q_seq_per_hk;
      combine_params.total_q              = total_q;
      combine_params.h_r                  = heads_ratio;
      combine_params.h_k                  = 1;
      combine_params.o_batch_stride       = out.stride(0);
      combine_params.o_row_stride         = out.stride(1);
      combine_params.o_head_stride        = out.stride(2);

      run_mla_combine_kernel<Element, varlen_q>(combine_params, stream);
    }

    MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
  };

  bool const need_load_balance = true;
  if (need_load_balance) {
    if (is_causal) {
      if (is_varlen_q) {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::true_type{}, mute::true_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::true_type{}, mute::true_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      } else {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::true_type{}, mute::false_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::true_type{}, mute::false_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      }
    } else {
      if (is_varlen_q) {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::false_type{}, mute::true_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::false_type{}, mute::true_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      } else {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::false_type{}, mute::false_type{}, mute::false_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::false_type{}, mute::false_type{}, mute::false_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      }
    }
  } else {
    if (is_causal) {
      if (is_varlen_q) {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::true_type{}, mute::true_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::true_type{}, mute::true_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      } else {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::true_type{}, mute::false_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::true_type{}, mute::false_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      }
    } else {
      if (is_varlen_q) {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::false_type{}, mute::true_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::false_type{}, mute::true_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      } else {
        if (q_type == at::ScalarType::Half) {
          run_mla(mutlass::half_t{}, mute::true_type{}, mute::false_type{}, mute::false_type{});
        } else if (q_type == at::ScalarType::BFloat16) {
          run_mla(mutlass::bfloat16_t{}, mute::true_type{}, mute::false_type{}, mute::false_type{});
        } else {
          TORCH_CHECK(false, "Unsupported element dtype");
        }
      }
    }
  }
  // just view since h_k=1
  if (!is_varlen_q) {
    out = out.view({batch_size, seqlen_q, heads_ratio, num_heads_k, head_dim_latent})
              .transpose(2, 3)
              .reshape({batch_size, seqlen_q, num_heads_q, head_dim_latent});
    softmax_lse = softmax_lse.view({batch_size, seqlen_q, heads_ratio, num_heads_k})
                      .transpose(2, 3)
                      .reshape({batch_size, seqlen_q, num_heads_q});
  } else {
    out = out.view({total_q, heads_ratio, num_heads_k, head_dim_latent})
              .transpose(1, 2)
              .reshape({total_q, num_heads_q, head_dim_latent});
    softmax_lse = softmax_lse.view({num_heads_k, heads_ratio, total_q}).transpose(0, 1).reshape({num_heads_q, total_q});
  }

  return {out, softmax_lse};
}

std::vector<at::Tensor> get_mla_decoding_metadata(const at::Tensor&            seqlens_k,
                                                  const int64_t                num_q_tokens_per_head_k,
                                                  const int64_t                h_k,
                                                  const std::optional<int64_t> h_q,
                                                  const bool                   is_fp8_kvcache,
                                                  const std::optional<int64_t> topk,
                                                  std::optional<at::Tensor>    tile_scheduler_metadata_,
                                                  std::optional<at::Tensor>    num_splits_) {
  bool is_sparse_attn = topk.has_value();
  CHECK_MUSA(seqlens_k);
  TORCH_CHECK(seqlens_k.is_contiguous());
  TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);
  if (is_sparse_attn) {
    TORCH_CHECK(h_q.has_value(), "num_heads_q must be provided when topk is provided");
  }

  int  batch_size    = seqlens_k.size(0);
  int* seqlens_k_ptr = seqlens_k.data_ptr<int>();
  auto options       = seqlens_k.options();

  auto dprops   = at::musa::getCurrentDeviceProperties();
  int  mp_count = dprops->multiProcessorCount;
  Arch arch     = {dprops->major, dprops->minor};
  TORCH_CHECK(arch.is_mp31(), "Only MP31 is supported");
  DecodingAttnImplMeta attn_impl_meta =
      get_attn_impl_meta(arch, mp_count, num_q_tokens_per_head_k, h_k, h_q, is_fp8_kvcache, is_sparse_attn);

  at::Tensor tile_scheduler_metadata;
  if (tile_scheduler_metadata_.has_value()) {
    tile_scheduler_metadata = tile_scheduler_metadata_.value();
  } else {
    tile_scheduler_metadata =
        torch::empty({attn_impl_meta.num_mp_parts, mate::flash_mla::TileSchedulerMetaDataSize}, options);
  }

  at::Tensor num_splits;
  if (num_splits_.has_value()) {
    num_splits = num_splits_.value();
  } else {
    num_splits = torch::empty({batch_size + 1}, options);
  }

  int* tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
  int* num_splits_ptr              = num_splits.data_ptr<int>();

  at::musa::OptionalMUSAGuard                device_guard(seqlens_k.device());
  auto                                       stream = at::musa::getCurrentMUSAStream().stream();
  mate::flash_mla::GetDecodingMetadataParams params = {};
  params.seqlens_k_ptr                              = seqlens_k_ptr;
  params.tile_scheduler_metadata_ptr                = tile_scheduler_metadata_ptr;
  params.num_splits_ptr                             = num_splits_ptr;
  params.batch_size                                 = batch_size;
  params.block_size_n                               = attn_impl_meta.k_block_size;
  params.fixed_overhead_num_blocks                  = attn_impl_meta.fixed_overhead_num_blocks;
  params.num_mp_parts                               = attn_impl_meta.num_mp_parts;
  params.topk                                       = is_sparse_attn ? topk.value() : -1;
  run_get_mla_metadata_kernel(params, stream);

  return {tile_scheduler_metadata, num_splits};
}

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
                                      std::optional<int64_t>    max_seqlen_q);

std::vector<at::Tensor> dispatch_mla_impl_for_fa_interface(at::Tensor&               q_nope,
                                                           at::Tensor&               q_pe,
                                                           at::Tensor const&         ckv,
                                                           at::Tensor const&         kpe,
                                                           at::Tensor const&         seqlens_k,
                                                           at::Tensor const&         block_table,
                                                           double const              softmax_scale,
                                                           bool const                is_causal,
                                                           std::optional<at::Tensor> cu_seqlens_q,
                                                           std::optional<int64_t>    max_seqlen_q,
                                                           std::optional<at::Tensor> workspace,
                                                           bool                      is_inited) {
  int        num_heads_q  = q_nope.size(-2);
  bool const is_varlen_q  = cu_seqlens_q.has_value();
  int        seqlen_q     = !is_varlen_q ? q_nope.size(1) : max_seqlen_q.value();
  int        num_heads_k  = ckv.dim() == 4 ? ckv.size(-2) : 1;
  int        q_seq_per_hk = seqlen_q * num_heads_q / num_heads_k;
  int        batch_size   = seqlens_k.size(0);

  auto                 dprops         = at::musa::getCurrentDeviceProperties();
  int                  mp_count       = dprops->multiProcessorCount;
  DecodingAttnImplMeta attn_impl_meta = get_attn_impl_meta(
      {dprops->major, dprops->minor}, mp_count, q_seq_per_hk, num_heads_k, std::nullopt, false, false);

  auto options = seqlens_k.options();

  if (is_varlen_q) {
    TORCH_CHECK(max_seqlen_q.has_value(), "max_seqlen_q must be provided when cu_seqlens_q is provided");
  }
  // Only for TP1
  TORCH_CHECK(workspace.has_value(), "workspace must be provided");
  at::Tensor tile_scheduler_metadata = torch::from_blob(
      workspace.value().data_ptr(), {attn_impl_meta.num_mp_parts, mate::flash_mla::TileSchedulerMetaDataSize}, options);
  ;
  at::Tensor num_splits =
      torch::from_blob(workspace.value().data_ptr<uint8_t>() +
                           attn_impl_meta.num_mp_parts * mate::flash_mla::TileSchedulerMetaDataSize * sizeof(int32_t),
                       {batch_size + 1},
                       options);
  if (!is_inited) {
    // Call get_metadata kernel
    auto mla_meta = get_mla_decoding_metadata(
        seqlens_k, q_seq_per_hk, num_heads_k, std::nullopt, false, std::nullopt, tile_scheduler_metadata, num_splits);
    tile_scheduler_metadata = mla_meta[0];
    num_splits              = mla_meta[1];
  }
  if (num_heads_q == 128) {
    // mla_asm
    return flash_mla_asm(q_nope,
                         q_pe,
                         ckv,
                         kpe,
                         seqlens_k,
                         block_table,
                         tile_scheduler_metadata,
                         num_splits,
                         softmax_scale,
                         is_causal,
                         cu_seqlens_q,
                         max_seqlen_q);
  } else {
    return mla_with_kvcache(q_nope,
                            q_pe,
                            ckv,
                            kpe,
                            seqlens_k,
                            cu_seqlens_q,
                            max_seqlen_q,
                            block_table,
                            tile_scheduler_metadata,
                            num_splits,
                            softmax_scale,
                            is_causal);
  }
}

TORCH_LIBRARY_FRAGMENT(mate, m) {
  m.def(
      "mla("
      "Tensor q_nope,"
      "Tensor q_pe,"
      "Tensor ckv,"
      "Tensor kpe,"
      "Tensor page_table,"
      "Tensor kv_len,"
      "float sm_scale,"
      "Tensor? out_ = None,"
      "Tensor? lse_ = None,"
      "bool is_causal = False) -> Tensor[]");

  m.def(
      "mla_with_kvcache("
      "Tensor q_nope,"
      "Tensor q_pe,"
      "Tensor ckv,"
      "Tensor kpe,"
      "Tensor seqlens_k,"
      "Tensor? cu_seqlens_q_,"
      "int? max_seqlen_q_,"
      "Tensor block_table,"
      "Tensor tile_scheduler_metadata,"
      "Tensor num_splits,"
      "float sm_scale,"
      "bool is_causal = False) -> Tensor[]");

  m.def(
      "get_mla_decoding_metadata("
      "Tensor seqlens_k,"
      "int num_q_tokens_per_head_k,"
      "int h_k,"
      "int? h_q = None,"
      "bool is_fp8_kvcache = False,"
      "int? topk = None,"
      "Tensor? tile_scheduler_metadata = None,"
      "Tensor? num_splits = None"
      ") -> Tensor[]");

  m.def(
      "dispatch_mla_impl_for_fa_interface("
      "Tensor q_nope,"
      "Tensor q_pe,"
      "Tensor ckv,"
      "Tensor kpe,"
      "Tensor kv_len,"
      "Tensor page_table,"
      "float sm_scale,"
      "bool is_causal,"
      "Tensor? cu_seqlens_q = None,"
      "int? max_seqlen_q = None,"
      "Tensor? workspace = None,"
      "bool is_inited = False) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(mate, PrivateUse1, m) {
  m.impl("mla", &mla);
  m.impl("mla_with_kvcache", &mla_with_kvcache);
  m.impl("get_mla_decoding_metadata", &get_mla_decoding_metadata);
  m.impl("dispatch_mla_impl_for_fa_interface", &dispatch_mla_impl_for_fa_interface);
}
