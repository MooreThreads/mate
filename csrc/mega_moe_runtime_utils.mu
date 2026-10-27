#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "mate/mega_moe/mega_moe_layout.hpp"
#include "mate/mega_moe/mega_moe_unified_layout.hpp"
#include "mate/mega_moe/prefill/prefill_workspace.hpp"
#include "mate_utils.hpp"
#include "tvm_ffi_utils.hpp"

namespace {

void check_cpu_tensor(ffi::TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU) << name << " must be a CPU tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

// IPC requires a musaMalloc allocation, independently of PyTorch's allocator.
struct MegaMoEBufferAllocator {
  void AllocData(DLTensor* tensor) {
    ffi::MUSADeviceGuard guard(tensor->device.device_id);
    MATE_MUSA_RUNTIME_CHECK(musaMalloc(&tensor->data, ffi::GetDataSize(*tensor)));
  }

  void FreeData(DLTensor* tensor) {
    ffi::MUSADeviceGuard guard(tensor->device.device_id);
    MATE_MUSA_RUNTIME_CHECK(musaFree(tensor->data));
  }
};

}  // namespace

ffi::Tensor mega_moe_alloc_buffer(int64_t num_bytes, int device_id) {
  TVM_FFI_ICHECK_GT(num_bytes, 0);
  return ffi::Tensor::FromNDAlloc(
      MegaMoEBufferAllocator{}, ffi::Shape({num_bytes}), dl_int8, DLDevice{kDLExtDev, device_id});
}

int64_t mega_moe_get_ipc_handle_size() {
  return static_cast<int64_t>(sizeof(musaIpcMemHandle_t));
}

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

ffi::Tensor make_tensor_view(ffi::Tensor buffer, uint64_t byte_offset, std::vector<int64_t> shape, DLDataType dtype) {
  DLTensor prototype    = *buffer.GetDLTensorPtr();
  auto     strides      = make_contiguous_strides(shape);
  prototype.data        = static_cast<uint8_t*>(buffer.data_ptr()) + byte_offset;
  prototype.dtype       = dtype;
  prototype.ndim        = static_cast<int32_t>(shape.size());
  prototype.shape       = shape.data();
  prototype.strides     = strides.data();
  prototype.byte_offset = 0;

  TVMFFIObjectHandle out;
  auto*              obj_handle = ffi::details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(buffer);
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorCreateUnsafeView(obj_handle, &prototype, &out));
  return ffi::Tensor(ffi::details::ObjectUnsafe::ObjectPtrFromOwned<ffi::TensorObj>(static_cast<TVMFFIObject*>(out)));
}

void mega_moe_get_ipc_handle(ffi::TensorView tensor, ffi::TensorView handle_out) {
  CHECK_MUSA(tensor);
  check_cpu_tensor(handle_out, "handle_out");
  CHECK_INPUT_TYPE(handle_out, dl_uint8);
  TVM_FFI_ICHECK_GE(handle_out.numel(), static_cast<int64_t>(sizeof(musaIpcMemHandle_t)))
      << "handle_out is too small for musaIpcMemHandle_t";

  musaIpcMemHandle_t handle{};
  MATE_MUSA_RUNTIME_CHECK(musaIpcGetMemHandle(&handle, tensor.data_ptr()));
  std::memcpy(handle_out.data_ptr(), &handle, sizeof(handle));
}

void mega_moe_open_ipc_handles(ffi::TensorView handles,
                               ffi::TensorView buffer,
                               int             rank_idx,
                               ffi::TensorView ptrs_out) {
  check_cpu_tensor(handles, "handles");
  check_cpu_tensor(ptrs_out, "ptrs_out");
  CHECK_INPUT_TYPE(handles, dl_uint8);
  CHECK_INPUT_TYPE(ptrs_out, dl_int64);
  CHECK_MUSA(buffer);
  CHECK_DIM(2, handles);
  CHECK_DIM(1, ptrs_out);

  const int64_t num_ranks   = handles.size(0);
  const int64_t handle_size = handles.size(1);
  TVM_FFI_ICHECK_GE(handle_size, static_cast<int64_t>(sizeof(musaIpcMemHandle_t)))
      << "handles second dimension is too small for musaIpcMemHandle_t";
  TVM_FFI_ICHECK_EQ(ptrs_out.size(0), num_ranks) << "ptrs_out shape must match handles.shape[0]";
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, num_ranks);

  const auto* handle_bytes = static_cast<const uint8_t*>(handles.data_ptr());
  auto*       ptrs         = static_cast<int64_t*>(ptrs_out.data_ptr());
  for (int64_t i = 0; i < num_ranks; ++i) {
    if (i == rank_idx) {
      ptrs[i] = reinterpret_cast<int64_t>(buffer.data_ptr());
      continue;
    }
    musaIpcMemHandle_t handle{};
    std::memcpy(&handle, handle_bytes + i * handle_size, sizeof(handle));
    void* ptr = nullptr;
    MATE_MUSA_RUNTIME_CHECK(musaIpcOpenMemHandle(&ptr, handle, musaIpcMemLazyEnablePeerAccess));
    ptrs[i] = reinterpret_cast<int64_t>(ptr);
  }
}

void mega_moe_close_ipc_handles(ffi::TensorView ptrs, int rank_idx) {
  check_cpu_tensor(ptrs, "ptrs");
  CHECK_INPUT_TYPE(ptrs, dl_int64);
  CHECK_DIM(1, ptrs);

  const auto*   values    = static_cast<const int64_t*>(ptrs.data_ptr());
  const int64_t num_ranks = ptrs.size(0);
  TVM_FFI_ICHECK_GE(rank_idx, 0);
  TVM_FFI_ICHECK_LT(rank_idx, num_ranks);
  for (int64_t i = 0; i < num_ranks; ++i) {
    if (i == rank_idx || values[i] == 0) {
      continue;
    }
    MATE_MUSA_RUNTIME_CHECK(musaIpcCloseMemHandle(reinterpret_cast<void*>(values[i])));
  }
}

std::tuple<int64_t, ffi::Function> mega_moe_unified_get_buffer_layout(int num_ranks,
                                                                      int num_experts,
                                                                      int num_max_tokens_per_rank,
                                                                      int num_topk,
                                                                      int hidden,
                                                                      int intermediate_hidden,
                                                                      int prefill_pool_capacity_rows,
                                                                      int prefill_l1_ring_tokens) {
  TVM_FFI_ICHECK_GT(num_ranks, 0);
  TVM_FFI_ICHECK_LE(num_ranks, 8) << "MegaMoE supports only an intranode group of at most 8 ranks";
  TVM_FFI_ICHECK_GT(num_experts, 0);
  TVM_FFI_ICHECK_EQ(num_experts % num_ranks, 0);
  TVM_FFI_ICHECK_GT(num_max_tokens_per_rank, 0);
  TVM_FFI_ICHECK_EQ(num_max_tokens_per_rank % ::mega_moe::prefill::kBlockM, 0)
      << "unified MegaMoE capacity must be aligned to 256";
  TVM_FFI_ICHECK_LE(num_max_tokens_per_rank, 65536);
  TVM_FFI_ICHECK_GT(num_topk, 0);
  TVM_FFI_ICHECK_LE(num_topk, 32);
  TVM_FFI_ICHECK_LE(num_topk, num_experts);
  TVM_FFI_ICHECK_GT(hidden, 0);
  TVM_FFI_ICHECK_EQ(hidden % 128, 0) << "unified MegaMoE requires hidden divisible by 128";
  TVM_FFI_ICHECK_GT(intermediate_hidden, 0);
  TVM_FFI_ICHECK_EQ(intermediate_hidden % 128, 0) << "unified MegaMoE requires intermediate_hidden divisible by 128";
  TVM_FFI_ICHECK_LE(intermediate_hidden, ::mega_moe::prefill::kMaxIntermediateHidden);
  TVM_FFI_ICHECK_GT(prefill_pool_capacity_rows, 0);
  TVM_FFI_ICHECK_EQ(prefill_pool_capacity_rows % ::mega_moe::prefill::kBlockM, 0)
      << "prefill pool capacity must be aligned to 256 rows";
  const auto prefill_worst_case_rows =
      ::mega_moe::layout::get_num_max_pool_tokens(static_cast<uint32_t>(num_ranks),
                                                  static_cast<uint32_t>(num_max_tokens_per_rank),
                                                  static_cast<uint32_t>(num_topk),
                                                  static_cast<uint32_t>(num_experts / num_ranks),
                                                  ::mega_moe::prefill::kBlockM);
  TVM_FFI_ICHECK_LE(static_cast<uint32_t>(prefill_pool_capacity_rows), prefill_worst_case_rows)
      << "prefill pool capacity must not exceed the worst-case pool";
  TVM_FFI_ICHECK_GE(prefill_l1_ring_tokens, 0) << "prefill L1 ring tokens must not be negative";
  if (prefill_l1_ring_tokens != 0) {
    TVM_FFI_ICHECK_EQ(prefill_l1_ring_tokens % ::mega_moe::prefill::kBlockM, 0)
        << "prefill L1 ring must be aligned to 256 rows";
    TVM_FFI_ICHECK_LE(static_cast<uint32_t>(prefill_l1_ring_tokens), static_cast<uint32_t>(prefill_pool_capacity_rows))
        << "prefill L1 ring must not exceed the pool capacity";
    const uint32_t ring_blocks = static_cast<uint32_t>(prefill_l1_ring_tokens) / ::mega_moe::prefill::kBlockM;
    TVM_FFI_ICHECK_EQ(ring_blocks & (ring_blocks - 1), 0u) << "prefill L1 ring must use a power-of-two block count";
    // Grouped swizzle can visit every M block of one expert before releasing the
    // first, so a ring smaller than one worst-case expert can deadlock the
    // dispatcher. Legal deduplicated top-k caps one expert at
    // ceil(num_ranks * num_max_tokens_per_rank / 256) blocks.
    const uint32_t max_expert_blocks =
        (static_cast<uint32_t>(num_ranks) * static_cast<uint32_t>(num_max_tokens_per_rank) +
         ::mega_moe::prefill::kBlockM - 1) /
        ::mega_moe::prefill::kBlockM;
    TVM_FFI_ICHECK_GE(ring_blocks, max_expert_blocks)
        << "prefill L1 ring is too small for grouped-swizzle scheduling; it must hold at least "
           "ceil(num_ranks * num_max_tokens_per_rank / 256) blocks. Disable recycling with "
           "prefill_l1_ring=False.";
  }

  const auto layout =
      ::mega_moe::unified::get_symm_buffer_layout_info(static_cast<uint32_t>(num_ranks),
                                                       static_cast<uint32_t>(num_experts),
                                                       static_cast<uint32_t>(num_max_tokens_per_rank),
                                                       static_cast<uint32_t>(num_topk),
                                                       static_cast<uint32_t>(hidden),
                                                       static_cast<uint32_t>(intermediate_hidden),
                                                       static_cast<uint32_t>(prefill_pool_capacity_rows),
                                                       static_cast<uint32_t>(prefill_l1_ring_tokens));

  auto slice_buffer = ffi::Function::FromTyped([=](ffi::Tensor buffer) {
    CHECK_INPUT_TYPE(buffer, dl_int8);
    CHECK_MUSA(buffer);
    TVM_FFI_ICHECK_GE(buffer.numel() * get_element_size(buffer), static_cast<int64_t>(layout.num_bytes))
        << "MegaMoE symmetric buffer is smaller than the unified layout";

    return std::make_tuple(
        make_tensor_view(buffer, layout.input_token_offset, {num_max_tokens_per_rank, hidden}, dl_int8),
        make_tensor_view(buffer, layout.input_sf_offset, {num_max_tokens_per_rank, hidden / 128}, dl_float32),
        make_tensor_view(buffer, layout.input_topk_idx_offset, {num_max_tokens_per_rank, num_topk}, dl_int64),
        make_tensor_view(buffer, layout.input_topk_weights_offset, {num_max_tokens_per_rank, num_topk}, dl_float32),
        make_tensor_view(buffer,
                         layout.decode_l1_token_offset,
                         {static_cast<int64_t>(layout.decode_num_max_pool_tokens), hidden},
                         dl_int8),
        make_tensor_view(buffer,
                         layout.decode_l1_sf_offset,
                         {static_cast<int64_t>(layout.decode_num_padded_sf_pool_tokens), hidden / 128},
                         dl_float32),
        make_tensor_view(buffer,
                         layout.decode_l1_topk_weights_offset,
                         {static_cast<int64_t>(layout.decode_num_max_pool_tokens)},
                         dl_float32),
        make_tensor_view(buffer,
                         layout.decode_l2_token_offset,
                         {static_cast<int64_t>(layout.decode_num_max_pool_tokens), intermediate_hidden},
                         dl_int8),
        make_tensor_view(buffer,
                         layout.decode_l2_sf_offset,
                         {static_cast<int64_t>(layout.decode_num_padded_sf_pool_tokens), intermediate_hidden / 128},
                         dl_float32),
        make_tensor_view(buffer,
                         layout.decode_combine_token_offset,
                         {num_topk, ::mega_moe::unified::kDecodeMaxTokensPerRank, hidden},
                         dl_bfloat16),
        make_tensor_view(buffer,
                         layout.prefill_l1_token_offset,
                         {static_cast<int64_t>(layout.prefill_l1_ring_tokens), hidden},
                         dl_int8),
        make_tensor_view(buffer,
                         layout.prefill_l1_sf_offset,
                         {static_cast<int64_t>(layout.prefill_l1_ring_padded_sf_tokens), hidden / 128},
                         dl_float32),
        make_tensor_view(buffer,
                         layout.prefill_l2_token_offset,
                         {static_cast<int64_t>(layout.prefill_num_max_pool_tokens), intermediate_hidden},
                         dl_int8),
        make_tensor_view(buffer,
                         layout.prefill_l2_sf_offset,
                         {static_cast<int64_t>(layout.prefill_num_padded_sf_pool_tokens), intermediate_hidden / 128},
                         dl_float32),
        make_tensor_view(
            buffer, layout.prefill_combine_token_offset, {num_topk, num_max_tokens_per_rank, hidden}, dl_bfloat16));
  });

  return {static_cast<int64_t>(layout.num_bytes), slice_buffer};
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_alloc_buffer, mega_moe_alloc_buffer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_get_ipc_handle_size, mega_moe_get_ipc_handle_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_get_ipc_handle, mega_moe_get_ipc_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_open_ipc_handles, mega_moe_open_ipc_handles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_close_ipc_handles, mega_moe_close_ipc_handles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mega_moe_unified_get_buffer_layout, mega_moe_unified_get_buffer_layout);
