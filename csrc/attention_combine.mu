#include <mutlass/fast_math.h>

#include "attention_combine.hpp"
#include "mate/attention/flash_mla/mpxx_mla_combine.hpp"
#include "mate_utils.hpp"
#include "tvm/ffi/error.h"

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                     \
  [&] {                                                                  \
    if (NUM_SPLITS <= 32) {                                              \
      constexpr static int NAME = 32;                                    \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 64) {                                       \
      constexpr static int NAME = 64;                                    \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 96) {                                       \
      constexpr static int NAME = 96;                                    \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 128) {                                      \
      constexpr static int NAME = 128;                                   \
      return __VA_ARGS__();                                              \
    } else if (NUM_SPLITS <= 160) {                                      \
      constexpr static int NAME = 160;                                   \
      return __VA_ARGS__();                                              \
    } else {                                                             \
      TVM_FFI_THROW(ValueError) << "Invalid num_splits: " << NUM_SPLITS; \
    }                                                                    \
  }()

template <typename T, bool VarlenQ, int HeadRatio>
void run_mla_combine_kernel(const mate::flash_mla::MlaCombineParams& params, musaStream_t stream) {
  static constexpr int HEAD_DIM_V = 512;  // Since only this head dimension is supported by Flash MLA
  MLA_NUM_SPLITS_SWITCH(params.num_mp_parts, NUM_SPLITS, [&] {
    constexpr int BLOCK_SIZE_M = 8;
    constexpr int NUM_THREADS  = BLOCK_SIZE_M * 32;

    dim3 grid;
    grid.x = params.batch_size;
    grid.y = mutlass::ceil_div(params.h_k * params.max_q_seq_per_hk, BLOCK_SIZE_M);
    grid.z = 1;

    const int scheduler_ctas = params.num_mp_parts * params.h_k * mutlass::ceil_div(params.max_q_seq_per_hk, 128);
    int       d_parts        = 1;
    // Two to three Combine CTAs per scheduler CTA cover the measured bandwidth plateau. Once there are at least as
    // many batches as scheduler CTAs, D partitioning mostly multiplies CTAs that return for single-split batches.
    while (d_parts < 16 && params.batch_size < scheduler_ctas &&
           params.batch_size * grid.y * d_parts * 2 < scheduler_ctas * 5)
      d_parts *= 2;
    auto launch = [&](auto d_parts_c) {
      constexpr int D_PARTS = decltype(d_parts_c)::value;
      grid.y *= D_PARTS;
      mate::flash_mla::
          mpxx_mla_combine_kernel<T, HEAD_DIM_V, BLOCK_SIZE_M, D_PARTS, NUM_SPLITS, NUM_THREADS, VarlenQ, HeadRatio>
          <<<grid, NUM_THREADS, 0, stream>>>(params);
      MATE_CHECK_MUSA_KERNEL_LAUNCH();
    };
    if (d_parts == 1) return launch(mute::Int<1>{});
    if (d_parts == 2) return launch(mute::Int<2>{});
    if (d_parts == 4) return launch(mute::Int<4>{});
    if (d_parts == 8) return launch(mute::Int<8>{});
    return launch(mute::Int<16>{});
  });
}

template void run_mla_combine_kernel<mutlass::half_t, false>(const mate::flash_mla::MlaCombineParams& params,
                                                             musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::half_t, true>(const mate::flash_mla::MlaCombineParams& params,
                                                            musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::bfloat16_t, false>(const mate::flash_mla::MlaCombineParams& params,
                                                                 musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::bfloat16_t, true>(const mate::flash_mla::MlaCombineParams& params,
                                                                musaStream_t                             stream);

template void run_mla_combine_kernel<mutlass::half_t, false, 128>(const mate::flash_mla::MlaCombineParams& params,
                                                                  musaStream_t                             stream);
template void run_mla_combine_kernel<mutlass::half_t, true, 128>(const mate::flash_mla::MlaCombineParams& params,
                                                                 musaStream_t                             stream);
template void run_mla_combine_kernel<mutlass::bfloat16_t, false, 128>(const mate::flash_mla::MlaCombineParams& params,
                                                                      musaStream_t                             stream);
template void run_mla_combine_kernel<mutlass::bfloat16_t, true, 128>(const mate::flash_mla::MlaCombineParams& params,
                                                                     musaStream_t                             stream);

#undef MLA_NUM_SPLITS_SWITCH
