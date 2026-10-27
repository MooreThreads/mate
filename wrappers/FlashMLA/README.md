# FlashMLA Compatibility Wrapper (flash_mla)

`flash_mla` is a compatibility wrapper package that preserves the official
`flash_mla` package name and import path while running on MUSA through MATE
Multi-head Latent Attention (MLA) operators.

## Overview

This wrapper is designed for projects that already target the FlashMLA Python
API. It lets you run MLA dense decode, sparse decode, and sparse prefill
workloads on MUSA with minimal integration changes.

The current compatibility scope includes `FlashMLASchedMeta`,
`get_mla_metadata`, `flash_mla_with_kvcache`, `flash_mla_sparse_fwd`, and
`fused_norm_rope_attn_rope_cast`.

## Package and import

- Package name: `flash_mla`
- Import path: `flash_mla`
- Runtime backend: MATE MLA operators on MUSA

MUSA wrapper releases use the PEP 440 local version suffix `+musa`, for
example `0.2.4+musa`. Use `python -m pip show flash_mla` to distinguish this
wrapper from the native package.

## Requirements

Before using this wrapper, make sure the following are available:

- TorchMUSA is installed and the MUSA runtime environment is configured.
- The target workload is configured to run on MUSA devices.

## Build

Build a wheel from the `wrappers/FlashMLA` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under:

```text
dist/
```

## Installation

For delivered packages, install from the external MUSA wheel source:

```bash
python -m pip install flash_mla \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

This installs the matching `mate` dependency automatically.

For local wrapper development, install from source:

```bash
python -m pip install --no-build-isolation --no-deps -e ../.. -v
python -m pip install --no-build-isolation --no-deps -e .
```

Install a built local wheel:

```bash
python -m pip install --no-deps dist/flash_mla-*.whl
```

## Import

Import the package directly:

```python
import flash_mla
```

Import individual APIs:

```python
from flash_mla import (
    FlashMLASchedMeta,
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_mla_sparse_fwd,
)
```

## Behavior

- `get_mla_metadata(...)` follows the current upstream FlashMLA Python interface and returns `(FlashMLASchedMeta(), None)`.
- The real scheduler tensors are initialized lazily on the first `flash_mla_with_kvcache(...)` call and cached inside `FlashMLASchedMeta`.
- Reusing the same `FlashMLASchedMeta` requires the same tensor shapes and the same values of `cache_seqlens`, `topk_length`, and `extra_topk_length`, following the upstream contract. Create a new object when those values change.
- `flash_mla_with_kvcache(...)` is the dense/sparse decode entry. The wrapper validates `FlashMLASchedMeta`, lazily materializes the real scheduler with `mate.flashmla.get_mla_metadata(...)`, and then forwards to `mate.flashmla.flash_mla_with_kvcache(...)`.
- `flash_mla_sparse_fwd(...)` is the sparse MLA prefill entry.

## Sparse cache formats

The ordinary decode API selects the following packed cache layouts from Q and
cache dimensions. Cache tensors have shape `[pages, page_size, 1, bytes_per_token]`.

| Q dimension | Bytes/token | Payload and scale placement |
| --- | --- | --- |
| 576 | 656 | V3.2: 512 E4M3, four inline FP32 group-128 scales, 64 BF16 RoPE values |
| 512 | 584 | V4: per-page 448 E4M3 + 64 BF16 data per token, then an 8-byte/token E8M0 scale footer |
| 512 | 528 | V4.1: per-page 512 E4M3 data per token, then a 16-byte/token group-32 E8M0 footer |
| 512 | 288 (extra only) | V4.1: per-page 256 packed E2M1 bytes per token, then 32 group-16 E4M3 scales per token |

Main/extra pairs are 584/584, 528/528, or 528/288. In FP4, the even element
occupies the low nibble. Main and extra caches may have independent page sizes
and padded page strides. `topk_length` bounds a candidate prefix including any
`-1` holes; it is not the number of valid indices. General prefill preserves each
query's candidates and does not infer private shared-index packing from shapes.

GLM NoPE uses a different 528-byte token layout. Use the explicit
`flashinfer.mla.SparseMLASm120Wrapper` with `kv_scale_format="arbitrary_fp32"`
for that layout; FlashMLA Q512/cache528 always denotes V4.1.

## Fused module

```python
from flash_mla import fused_norm_rope_attn_rope_cast
```

The module provides upstream-compatible `prefill`, `decode`, `permute_q_b_proj`
and `permute_wv_proj` entry points. Q_b weights must be permuted before projection;
ordinary attention expects canonical head order and does not use these weights.
The fused entry expects BF16 Q512, interleaved RoPE64, and group-32 FP8 output
with power-of-two scales packed into column-major int32 words. Q_b input scales
may describe groups of 32 or 128; Wv input scales describe groups of 32.

On MUSA, one kernel combines Q layout conversion, optional RMS normalization
and RoPE. Prefill and unsplit decode apply inverse RoPE, Wv layout conversion,
FP8 quantization and scale packing directly to the final FP32 attention
accumulators. Split decode keeps partial outputs in FP32 and performs this
epilogue after the combine kernel merges them and applies the attention sink.
These paths do not materialize a BF16 attention output or launch a separate
output epilogue. Output scales use `ceil_pow2(max(amax, 1e-4) / 448)`.

Q preparation remains a separate kernel. RMS normalization is rounded into
the prepared BF16 Q; results are validated with BF16 numerical tolerances
against upstream's logit-scale convention.


## Notes

- This wrapper keeps the official FlashMLA import surface, but execution is provided by MATE on MUSA.
- For the authoritative MLA operator behavior, refer to `mate.flashmla`.
