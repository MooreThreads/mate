# GDN Support Matrix

This page summarizes the current GDN support surface in MATE.

## Decode

### Dispatch Matrix

| `state_layout` | `state.dtype` | `T` | Status | Notes |
| --- | --- | --- | --- | --- |
| `VK` | `bfloat16` | `1..4` | ❌ Not supported | BF16 state backend is not implemented yet |
| `VK` | `bfloat16` | `> 4` | ❌ Not supported | MTP path is not implemented yet |
| `VK` | `float32` | `1` | ✅ Supported | Current active decode path |
| `VK` | `float32` | `> 1` | ❌ Not supported | FP32 MTP backend is not implemented yet |
| `KV` | `float32` | `1` | ❌ Not supported | KV backend is not implemented yet |
| `KV` | anything else | any | ❌ Not supported | Unsupported combination |

### Current MATE Active Path

Current MATE decode support is intentionally narrow:

- API: `mate.gated_delta_rule_decode(...)`
- Supported combination: `state_layout="VK"`, `state.dtype=float32`, `T=1`
- Backend: TileLang
- State update: in place

### Current MATE Restrictions

| Item | MATE |
| --- | --- |
| `state_indices` | ❌ Not supported |
| BF16 state backend | ❌ Not supported |
| KV backend | ❌ Not supported |
| MTP (`T > 1`) | ❌ Not supported |
| `intermediate_states_buffer` | ❌ Not supported |
| `disable_state_update` | ❌ Not supported |

### Input Contract on the Active Path

On the currently supported MATE path:

- `q / k / v`: `float16` or `bfloat16`
- `A_log / dt_bias`: `float32`
- `a / b`: same dtype as `q`
- `output`: optional, supports `float16` / `bfloat16` / `float32`

## Prefill

### Prefill At a Glance

| Area | Status | Notes |
| --- | --- | --- |
| Backend | ✅ Supported | TileLang implementation on MUSA (`mate.gdn.tilelang.gdn_prefill`) |
| Sequence Mode | ✅ Supported | Varlen prefill with `cu_seqlens` |
| Head Layout | ✅ Supported | `GQA` and `GVA` |
| Dtype (Q/K/V) | ✅ Supported | `fp16`, `bf16`, `fp32` |
| Gate Inputs | ✅ Supported | `g` (alpha) and `beta` are float32 tensors; defaults to all-ones when omitted |
| Initial State | ✅ Supported | Optional `initial_state` with shape `[batch, head_sab, dim_v, dim_k]` (float32) |
| Final State Output | ✅ Supported | `output_final_state=True` returns `(output, final_state)` |
| Output Heads | ✅ Supported | `head_o = max(num_q_heads, num_v_heads)` |
| QK L2 Norm Option | ✅ Supported | `use_qk_l2norm_in_kernel=True` (wrapper-side normalize before kernel launch) |

### Prefill Shape Rules

| Item | Requirement |
| --- | --- |
| `q`, `k`, `v` rank | 3D tensors: `[total_tokens, heads, dim]` |
| Token count | `q.size(0) == k.size(0) == v.size(0)` |
| Q/K dim | `q.size(2) == k.size(2)` |
| Head layout | `GQA`: `num_v_heads == num_k_heads` and `num_q_heads % num_k_heads == 0`; `GVA`: `num_q_heads == num_k_heads` and `num_v_heads % num_q_heads == 0` |
| `cu_seqlens` | Required by public wrapper for varlen prefill |
| `chunk_size` | Must be positive |

### Current Kernel Constraints

| Item | Status | Notes |
| --- | --- | --- |
| `dim_k` tiling | ✅ Required | Current kernel launch uses `block_DK=64`, so `dim_k % 64 == 0` |
| `dim_v` tiling | ✅ Required | Current prefill path uses `block_DV=64` for prepare/output kernels, so `dim_v % 64 == 0` |
| Device | ✅ Required | MUSA |

### Not Supported Yet (Prefill)

| Feature | Status | Notes |
| --- | --- | --- |
| Non-GQA/GVA head mapping | ❌ Not supported | Only the two grouped layouts above are accepted |

### Notes

- Public API entry: `mate.gdn_prefill.chunk_gated_delta_rule`.
- Internal prefill path runs three kernels: prepare (`w/u/cu_g`), recurrent state update, and output projection.
- Final state layout is `k-last`: `[N, H, V, K]`.
