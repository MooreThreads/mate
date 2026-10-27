
#include "attention_scheduler.hpp"
#include "mate/attention/flash_mla/mpxx_get_mla_metadata.hpp"
#include "mate_utils.hpp"

void run_get_mla_metadata_kernel(mate::flash_mla::GetDecodingMetadataParams& params,
                                 musaStream_t                                stream,
                                 size_t                                      max_shared_memory_per_block) {
  const size_t smem_size = sizeof(int) * (static_cast<size_t>(params.batch_size) * 5 + 1);
  if (smem_size <= max_shared_memory_per_block) {
    mate::flash_mla::get_mla_metadata_kernel<<<1, 32, smem_size, stream>>>(params);
    MATE_CHECK_MUSA_KERNEL_LAUNCH();
  } else {
    const size_t compact_smem_size = sizeof(int) * static_cast<size_t>(params.batch_size);
    if (compact_smem_size <= max_shared_memory_per_block) {
      mate::flash_mla::get_mla_metadata_kernel_compact_fallback<<<1, 32, compact_smem_size, stream>>>(params);
      MATE_CHECK_MUSA_KERNEL_LAUNCH();
    } else {
      mate::flash_mla::get_mla_metadata_kernel_fallback<<<1, 32, 0, stream>>>(params);
      MATE_CHECK_MUSA_KERNEL_LAUNCH();
    }
  }
}
