# FlashAttention3 Forward Compatibility

This document is a quick reference for the current FlashAttention-compatible **forward** coverage provided by MATE on MUSA.

## At a Glance

| Area | Status | Notes |
| --- | --- | --- |
| Q Mode | ✅ Supported | `Normal`, `Ragged`, `Padded` |
| KV Mode | ✅ Supported | `Normal`, `Ragged`, `Padded`, `Paged` |
| Mask Mode | ✅ Partially supported | `None`, `Causal`, `Local`; `Local + attention_chunk` is not supported |
| Score Mode | ✅ Supported | Standard softmax and `softcap` |
| Page Size | ✅ Supported | `1`, `16`, `64`, and arbitrary page sizes |
| Dtype | ✅ Partially supported | `bf16`, `fp16`; standard FMHA `fp8` is not supported |
| HeadDim | ✅ Supported | Any `headdim <= 512` |
| Optimization | ✅ Supported | `SplitKV`, `PackGQA`, `SchedulerMetadata` |
| Output | ✅ Supported | `out`, `softmax_lse` |

## MATE Extensions

| Extension | Status | Notes |
| --- | --- | --- |
| Context Parallel | ✅ Supported | `cp_world_size`, `cp_rank`, `cp_tot_seqused_k` |
| Learnable Sink | ✅ Supported | Supported on the local-attention path |

## Not Supported Yet

| Feature | Status | Notes |
| --- | --- | --- |
| Append New KV | ❌ Not supported | `Normal`, `Ragged` |
| RoPE Input | ❌ Not supported | `Interleaved`, `Non-interleaved` |
| Cache Index Options | ❌ Not supported | `cache_batch_idx`, `cache_leftpad` |
| Chunked Attention | ❌ Not supported | `attention_chunk > 0` |
| Standard FMHA FP8 Input | ❌ Not supported | Forward FMHA path only |

## Notes

- This page summarizes the compatibility surface, not every internal kernel detail.
- The statement `Any headdim <= 512` refers to the supported forward-path head-dimension range.
- For wrapper-level usage, see [../wrappers/flash-attention/README.md](../wrappers/flash-attention/README.md).
