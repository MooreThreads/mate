#pragma once

#include <mutlass/device_kernel.h>
#include <torch/all.h>
#include <torch/torch.h>
#include <torch_musa/csrc/core/MUSAGuard.h>

#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_collective_tme_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_paged_collective_tme_warpspecialized.hpp"
#include "flash.hpp"
#include "fmha_options.hpp"
#include "kernel/fmha_kernel_tme_warpspecialzed.hpp"
#include "kernel/fmha_paged_kernel_tme_warpspecialzed.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "mate_utils.muh"
#include "torch_utils.hpp"

using namespace mute;

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split>
void fmha_kernel_launcher(Flash_fwd_params params) {
  int  seqlen_q   = params.is_varlen_q ? params.total_q : params.seqlen_q;
  int  seqlen_k   = params.is_varlen_k ? params.total_k : params.seqlen_k;
  auto stride_q   = make_stride(params.q_row_stride, _1{}, make_stride(params.q_head_stride, params.q_batch_stride));
  auto stride_k   = make_stride(params.k_row_stride, _1{}, make_stride(params.k_head_stride, params.k_batch_stride));
  auto stride_v   = make_stride(params.v_row_stride, _1{}, make_stride(params.v_head_stride, params.v_batch_stride));
  auto stride_o   = make_stride(params.o_row_stride, _1{}, make_stride(params.o_head_stride, params.o_batch_stride));
  auto stride_lse = make_stride(_1{}, make_stride(params.lse_head_stride, params.lse_batch_stride));

  using StrideQ   = decltype(stride_q);
  using StrideK   = decltype(stride_k);
  using StrideV   = decltype(stride_v);
  using StrideO   = decltype(stride_o);
  using StrideLse = decltype(stride_lse);

  constexpr int Consumers = CTA_Q / 64;

  // CTA_Q, CTA_KV, D_QK, D_VO

  using TileShape = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEADDIM_QK>, Int<HEADDIM_V>>;

  static constexpr bool UseFopMask = false;
  static constexpr bool UpperLeft  = false;

  using Fusion = std::conditional_t<Causal,
                                    mutlass::fmha::collective::CausalFusion<UpperLeft, false>,
                                    mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaMainloopTmeWarpSpecialized<
      Element,
      float,
      TileShape,
      StrideQ,
      StrideK,
      StrideV,
      Fusion,
      mutlass::fmha::Option<mutlass::fmha::Tag::NumMmaWarpSquads, Int<Consumers>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::Varlen, conditional_t<Varlen, true_type, false_type>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::VStage, Int<1>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::KStage, Int<1>>>;

  using EpilogueTileShape = Shape<Int<tuple_element_t<0, TileShape>{} / Consumers>, tuple_element_t<3, TileShape>>;

  using CollectiveEpilogue =
      mutlass::fmha::collective::FmhaFwdEpilogue<ElementO, float, EpilogueTileShape, StrideO, StrideLse>;

  using TileScheduler = mutlass::fmha::kernel::FmhaIndividualTileScheduler<Causal>;
  using Kernel =
      mutlass::fmha::kernel::FmhaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  auto problem_size = make_shape(seqlen_q, seqlen_k, HEADDIM_QK, HEADDIM_V, params.h, params.h_k, params.b);

  typename Kernel::ProblemShape problem_shape;  // run kernel

  get<0>(problem_shape) = mutlass::fmha::collective::VariableLength{static_cast<int>(get<0>(problem_size)),
                                                                    static_cast<int>(params.seqlen_q),
                                                                    static_cast<int*>(params.cu_seqlens_q)};
  get<1>(problem_shape) = mutlass::fmha::collective::VariableLength{static_cast<int>(get<1>(problem_size)),
                                                                    static_cast<int>(params.seqlen_k),
                                                                    static_cast<int*>(params.cu_seqlens_k)};
  get<2>(problem_shape) = get<2>(problem_size);
  get<3>(problem_shape) = get<3>(problem_size);
  get<4>(problem_shape) = get<4>(problem_size);
  get<5>(problem_shape) = get<5>(problem_size);
  get<6>(problem_shape) = get<6>(problem_size);

  typename Kernel::Arguments arguments{
      problem_shape,
      {
          static_cast<Element*>(params.q_ptr),
          stride_q,
          static_cast<Element*>(params.k_ptr),
          stride_k,
          static_cast<Element*>(params.v_ptr),
          stride_v,
          params.scale_softmax,
      },
      {
          static_cast<ElementO*>(params.o_ptr),
          stride_o,
          static_cast<float*>(params.softmax_lse_ptr),
          stride_lse,
      },
  };

  musaStream_t            stream        = at::musa::getCurrentMUSAStream();
  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(arguments);

  auto grid_dim = TileScheduler::get_grid_shape(kernel_params.scheduler);
  mutlass::device_kernel<Kernel>
      <<<grid_dim, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize, stream>>>(kernel_params);
  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split>
void fmha_paged_kernel_launcher(Flash_fwd_params params) {
  auto stride_q   = make_stride(params.q_row_stride, _1{}, make_stride(params.q_head_stride, params.q_batch_stride));
  auto stride_k   = make_stride(params.k_row_stride, _1{}, make_stride(params.k_head_stride, params.k_batch_stride));
  auto stride_v   = make_stride(params.v_row_stride, _1{}, make_stride(params.v_head_stride, params.v_batch_stride));
  auto stride_o   = make_stride(params.o_row_stride, _1{}, make_stride(params.o_head_stride, params.o_batch_stride));
  auto stride_lse = make_stride(_1{}, make_stride(params.lse_head_stride, params.lse_batch_stride));
  auto stride_pt  = make_stride(int(params.page_table_batch_stride), _1{});  // TODO:

  using StrideQ   = decltype(stride_q);
  using StrideK   = decltype(stride_k);
  using StrideV   = decltype(stride_v);
  using StrideO   = decltype(stride_o);
  using StrideLse = decltype(stride_lse);

  constexpr int Consumers = CTA_Q / 64;

  // CTA_Q, CTA_KV, D_QK, D_VO

  static constexpr bool UseFopMask = false;
  static constexpr bool UpperLeft  = false;
  static constexpr bool PackGQA    = false;

  using Fusion = std::conditional_t<Causal,
                                    mutlass::fmha::collective::CausalFusion<UpperLeft, PackGQA>,
                                    mutlass::fmha::collective::DefaultFusion>;

  using TileShape = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEADDIM_QK>, Int<HEADDIM_V>>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaPagedMainloopTmeWarpSpecialized<
      Element,
      float,
      TileShape,
      StrideQ,
      StrideK,
      StrideV,
      Fusion,
      mutlass::fmha::Option<mutlass::fmha::Tag::NumMmaWarpSquads, Int<Consumers>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::KStage, Int<2>>,
      mutlass::fmha::Option<mutlass::fmha::Tag::VStage, Int<2>>>;

  using EpilogueTileShape = Shape<Int<tuple_element_t<0, TileShape>{} / Consumers>, tuple_element_t<3, TileShape>>;

  using CollectiveEpilogue =
      mutlass::fmha::collective::FmhaFwdEpilogue<ElementO, float, EpilogueTileShape, StrideO, StrideLse>;

  using TileScheduler = mutlass::fmha::kernel::FmhaIndividualTileScheduler<Causal>;
  using Kernel =
      mutlass::fmha::kernel::PagedFmhaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  typename Kernel::ProblemShape problem_shape =
      make_shape(params.seqlen_q, params.seqlen_k, HEADDIM_QK, HEADDIM_V, params.h, params.h_k, params.b);

  typename Kernel::Arguments arguments{
      problem_shape,
      {
          static_cast<Element*>(params.q_ptr),
          stride_q,
          static_cast<Element*>(params.k_ptr),
          stride_k,
          static_cast<Element*>(params.v_ptr),
          stride_v,
          static_cast<int*>(params.page_table),
          stride_pt,
          static_cast<int*>(params.seqused_k),
          params.scale_softmax,
          params.page_size,
          params.num_pages,
      },
      {
          static_cast<ElementO*>(params.o_ptr),
          stride_o,
          static_cast<float*>(params.softmax_lse_ptr),
          stride_lse,
      },
  };

  musaStream_t            stream        = at::musa::getCurrentMUSAStream();
  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(arguments);

  auto grid_dim = TileScheduler::get_grid_shape(kernel_params.scheduler);
  mutlass::device_kernel<Kernel>
      <<<grid_dim, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize, stream>>>(kernel_params);
  MATE_MUSA_RUNTIME_CHECK(musaGetLastError());
}

template <int Arch,
          typename Element,
          typename ElementO,
          bool Causal,
          bool Varlen,
          int  CTA_Q,
          int  CTA_KV,
          int  HEADDIM_QK,
          int  HEADDIM_V,
          bool Split,
          bool PAGED_KV>
void dispatch_fmha_kernel(Flash_fwd_params params) {
  if constexpr (PAGED_KV) {
    fmha_paged_kernel_launcher<Arch, Element, ElementO, Causal, Varlen, CTA_Q, CTA_KV, HEADDIM_QK, HEADDIM_V, Split>(
        params);
  } else {
    fmha_kernel_launcher<Arch, Element, ElementO, Causal, Varlen, CTA_Q, CTA_KV, HEADDIM_QK, HEADDIM_V, Split>(params);
  }
}
