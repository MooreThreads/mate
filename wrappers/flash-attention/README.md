# flash-attention

`mate-flash-attention` is a compatibility wrapper package that preserves the `flash_attn` import path on top of MATE attention operators on MUSA.

## Overview

This wrapper is intended for projects that already target FlashAttention-style Python APIs and want to run on MUSA through MATE with smaller integration changes.

- Package name: `mate-flash-attention`
- Import path: `flash_attn`
- Runtime backend: MATE attention operators on MUSA

For the current compatibility scope and known limitations, see [../../docs/flash_attention.md](../../docs/flash_attention.md).

## Requirements

Before using this wrapper, make sure the following are already available:

- MATE is installed and importable
- TorchMUSA and the MUSA runtime environment are available
- The target workload is expected to run on MUSA devices

## Build

Build a wheel from the `wrappers/flash-attention` directory:

```bash
python -m build --wheel --no-isolation
```

The generated wheel will be placed under `dist/`.

## Installation

Install from source:

```bash
pip install --no-build-isolation -e .
```

Install a built wheel:

```bash
pip install dist/mate_flash_attention-*.whl
```

## Import

Import the package directly:

```python
import flash_attn
```

Import individual APIs:

```python
from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
```

## Public APIs

The wrapper currently exposes:

- `flash_attn_varlen_func`: FlashAttention-compatible varlen / ragged FMHA entry
- `flash_attn_with_kvcache`: FlashAttention-compatible KV-cache entry, including paged KV cache use cases
- `get_scheduler_metadata`: helper for split-KV scheduler metadata preparation

## Quick Start

Minimal varlen FMHA example:

```python
import torch
from flash_attn import flash_attn_varlen_func

device = "musa"
dtype = torch.bfloat16

num_heads_q = 32
num_heads_kv = 8
head_dim = 128

cu_seqlens_q = torch.tensor([0, 32, 96], device=device, dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 64, 160], device=device, dtype=torch.int32)

q = torch.randn((96, num_heads_q, head_dim), device=device, dtype=dtype)
k = torch.randn((160, num_heads_kv, head_dim), device=device, dtype=dtype)
v = torch.randn((160, num_heads_kv, head_dim), device=device, dtype=dtype)

out = flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=64,
    max_seqlen_k=96,
    causal=False,
)
```

Paged KV cache skeleton with scheduler metadata:

```python
from flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

metadata = get_scheduler_metadata(
    batch_size=batch_size,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    num_heads_q=num_heads_q,
    num_heads_kv=num_heads_kv,
    headdim=head_dim_qk,
    headdim_v=head_dim_v,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    page_size=page_size,
    num_splits=0,
    pack_gqa=True,
)

out, lse, *_ = flash_attn_with_kvcache(
    q=q_unpad,
    k_cache=k_cache_paged,
    v_cache=v_cache_paged,
    cache_seqlens=cache_seqlens,
    page_table=page_table,
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    scheduler_metadata=metadata,
    return_softmax_lse=True,
)
```

## Tests

Wrapper-level tests are available in:

```text
tests/test_flash_attn.py
```

Run them from the `wrappers/flash-attention` directory:

```bash
pytest tests/test_flash_attn.py
```

## Notes

- This wrapper preserves the FlashAttention-style Python surface, but execution is provided by MATE on MUSA
- Actual feature coverage is documented in [../../docs/flash_attention.md](../../docs/flash_attention.md)
- For the authoritative operator behavior, refer to the corresponding MATE APIs under `mate.mha_interface`
