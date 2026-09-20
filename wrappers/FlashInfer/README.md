# FlashInfer Compatibility Wrapper (flashinfer)

`flashinfer` preserves a focused set of FlashInfer Python APIs while running
the operations on MUSA through MATE.

## Overview

The wrapper currently covers four integration areas:

| Area | Public surface |
| --- | --- |
| GEMM | BF16/FP16 BMM, FP8 BMM, groupwise FP8 GEMM, and grouped FP8 GEMM |
| Norm | RMSNorm, LayerNorm, fused add RMSNorm, quantized outputs, and diffusion-model fused norms |
| MLA RoPE | Fused RoPE and FP8 quantization for 512-value NoPE and 64-value RoPE components |
| Sparse MLA decode | FP8 sparse decode with optional reusable scheduler metadata |

The GEMM APIs align with official FlashInfer v0.6.17. This is a compatibility
wrapper rather than a complete FlashInfer distribution; only the APIs listed
below are provided.

## Package and import

- Distribution name: `flashinfer-python`
- Import path: `flashinfer`
- Runtime backend: MATE operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`. Use
`python -m pip show flashinfer-python` to distinguish this wrapper from the
native package.

```python
import flashinfer
import flashinfer.decode
import flashinfer.gemm
import flashinfer.norm
import flashinfer.rope
```

## Requirements

- TorchMUSA is installed and the MUSA runtime environment is configured.
- A matching MATE version is installed.
- Operator inputs are placed on a MUSA device.

## Build and installation

Run the following commands from `wrappers/FlashInfer`. Install the local MATE
checkout first, then install the wrapper in editable mode:

```bash
python -m pip install --no-build-isolation --no-deps -e ../.. -v
python -m pip install --no-build-isolation --no-deps -e .
```

Build and install a local wheel with:

```bash
python -m build --wheel
python -m pip install --no-deps dist/flashinfer_python-*.whl
```

## GEMM APIs

```python
from flashinfer.gemm import (
    batch_deepgemm_fp8_nt_groupwise,
    bmm_bf16,
    bmm_fp8,
    gemm_fp8_nt_groupwise,
    group_deepgemm_fp8_nt_groupwise,
)
```

`bmm_bf16` and `bmm_fp8` are also exported from the `flashinfer` package root.

| API | Supported behavior | Backend |
| --- | --- | --- |
| `bmm_bf16` | FP16 or BF16 inputs; BF16, FP16, or FP32 output | `auto`, `mudnn` |
| `bmm_fp8` | FP8 inputs with scalar or rank-3 scales; BF16 or FP16 output | `auto`, `mudnn` |
| `gemm_fp8_nt_groupwise` | Groupwise FP8 NT GEMM; optional FP8 output scaling | `auto`, `mudnn`, or `mubin` depending on output mode |
| `group_deepgemm_fp8_nt_groupwise` | Contiguous grouped FP8 NT GEMM with BF16 output | MATE grouped GEMM |
| `batch_deepgemm_fp8_nt_groupwise` | Masked batched FP8 NT GEMM with BF16 output | MATE grouped GEMM |

The BMM signatures follow FlashInfer's argument order:

```python
bmm_bf16(A, B, out=None, out_dtype=torch.bfloat16, backend="auto")
bmm_fp8(A, B, A_scale, B_scale, dtype, out=None, backend="auto")
```

`A` has shape `(batch, m, k)` and `B` has shape `(batch, k, n)`. The wrapper
passes a transpose view of `B` to MATE without making a layout copy. Scalar
FP8 scales select per-tensor scaling; rank-3 scales are a MATE extension for
channelwise scaling. When `out` is supplied, the wrapper writes into and
returns that tensor.

For `gemm_fp8_nt_groupwise`, `auto` selects `mudnn` for BF16 or FP16 output.
Supplying `output_scale` selects FP8 E4M3 output and makes `auto` select
`mubin`.

## Norm APIs

```python
from flashinfer.norm import (
    fused_add_rmsnorm,
    fused_add_rmsnorm_fp8_block_quant,
    fused_add_rmsnorm_quant,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    layernorm,
    layernorm_quant,
    rmsnorm,
    rmsnorm_quant,
)
```

The wrapper also exports the fused DiT LayerNorm, fused QK RMSNorm/RoPE, and
fused RMSNorm/SiLU APIs implemented by MATE. All supported Norm functions are
available both from `flashinfer.norm` and the `flashinfer` package root.

Quantized RMSNorm and LayerNorm functions follow FlashInfer's output-first
signatures and write into the supplied FP8 tensor. Python float scales are
converted to device-local FP32 scalar tensors. Fused add RMSNorm updates the
residual in place; its non-quantized variant also replaces the input with the
normalized output. The `enable_pdl` argument is accepted for compatibility and
has no effect on MUSA.

`fused_add_rmsnorm_fp8_block_quant` produces FP8 E4M3 activations together
with dynamic FP32 scales for groups of 128 consecutive hidden elements. MATE
uses a contiguous row-major scale tensor with shape
`[rows, hidden_size // 128]`; this intentionally differs from FlashInfer's
swizzled DeepGEMM scale layout.

## FP8 MLA APIs

```python
from flashinfer.decode import (
    get_batch_decode_metadata_mla,
    trtllm_batch_decode_with_kv_cache_mla,
)
from flashinfer.rope import mla_rope_quantize_fp8
```

### RoPE and FP8 quantization

`mla_rope_quantize_fp8` accepts BF16 query/key components and an FP32 or BF16
cosine/sine cache. It produces FP8 E4M3 outputs for a 512-value NoPE component
and a 64-value RoPE component. Optional output buffers may be strided in their
leading dimensions, so the outputs can be written directly into slices of a
merged `[..., 576]` tensor:

```python
q_nope_out = q_fp8[..., :512]
q_rope_out = q_fp8[..., 512:]
```

The quantization scales are multipliers applied before the FP8 cast:

```text
Q_fp8  = cast(Q  * quant_scale_q)
KV_fp8 = cast(KV * quant_scale_kv)
```

### Sparse MLA decode

`trtllm_batch_decode_with_kv_cache_mla` supports:

- A contiguous FP8 E4M3 query with shape `[batch, q_len, heads, 576]`.
- A contiguous FP8 E4M3 KV cache with shape `[pages, page_size, 576]` or
  `[pages, 1, page_size, 576]`.
- Contiguous int32 physical-token indices with shape
  `[batch, q_len, sparse_mla_top_k]`.
- A contiguous int32 `seq_lens` tensor with shape `[batch]`.
- Optional preallocated BF16 output and FP32 LSE buffers.

The decode path requires `kv_lora_rank=512`, `qk_rope_head_dim=64`, and a
positive `sparse_mla_top_k` that is divisible by 64. Unsupported shapes,
dtypes, layouts, devices, and tensor-valued decode scales fail explicitly.

`workspace_buffer` and FlashInfer implementation-selection arguments are kept
for call compatibility. MATE owns the runtime storage and always dispatches
its FP8 sparse MLA kernel, independent of the supplied decode backend name.

When RoPE quantization uses non-unit multipliers, fold their reciprocal scales
into the decode scales:

```python
q_descale = 1.0 / quant_scale_q
kv_descale = 1.0 / quant_scale_kv
bmm1_scale = attention_scale * q_descale * kv_descale
bmm2_scale = kv_descale
```

### Metadata reuse

Decode prepares scheduler metadata on every call unless a metadata object is
provided. Reuse metadata while `seq_lens`, query shape, and sparse top-k remain
unchanged:

```python
metadata = flashinfer.decode.get_batch_decode_metadata_mla(
    query=query,
    seq_lens=seq_lens,
    sparse_mla_top_k=topk,
)
out, lse = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
    ...,
    workspace_buffer=None,
    return_lse=True,
    metadata=metadata,
)
```

Prepare a new metadata object after changing `seq_lens`, query shape, or top-k.
Do not share one metadata object between overlapping decode calls.

## Mamba / SSD APIs

```python
from flashinfer.mamba import selective_state_update
from flashinfer.mamba.ssd_combined import ssd_combined_fwd_varlen
```

| Symbol | Use |
| --- | --- |
| `selective_state_update` | One-step SSM state update for Mamba/SSD decode. |
| `ssd_combined_fwd_varlen`, aliased as `mamba_chunk_scan_combined_varlen` | Packed variable-length SSD prefill. |
| `checkpointing_ssu`, `allocate_checkpointing_ssu_scratch` | Checkpointed SSU paths. |
| `replayssm_materialize` | ReplaySSM state materialization. |
| `CheckpointingSSURunner` | Exported only when the MATE backend provides it. |

Parameter lists here are frozen contracts rather than conveniences. Consumers
call the packed-varlen entry point by keyword and the state-update entry point by
keyword using upstream FlashInfer names, so do not rename parameters, reorder
positional ones, or add MUSA-specific knobs to these names; backend selection
and tuning belong to MATE or to the environment.

These upstream symbols are intentionally absent: `ssd_combined_fwd` and
`SSDCombined` (dense CUTLASS/CuTe entry points) and
`cake_selective_state_update` and `CakeSSDCombined` (Cake/CUDA paths).

Implementations are resolved from the MATE mamba family at call time
(`mate.mamba` by default). A symbol whose MATE implementation is missing raises
`NotImplementedError` naming the gap instead of returning a wrong result, and
capability-shaped symbols such as `CheckpointingSSURunner` are absent rather than
present-but-broken, so downstream probing stays honest.

## Example

[`examples/fp8_mla_decode.py`](examples/fp8_mla_decode.py) demonstrates the
full RoPE-to-FP8-to-sparse-decode path with non-unit quantization scales,
merged 576-value buffers, and reusable metadata:

```bash
python examples/fp8_mla_decode.py
```

For the native MATE API and its complete parameter documentation, see
[`mate.sparse_mla_interface`](../../docs/source/api/sparse_mla.rst).
