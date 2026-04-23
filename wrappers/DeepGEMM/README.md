# deep_gemm

`deep-gemm` is a compatibility wrapper package that preserves the `deep_gemm` import path on top of MATE GEMM operators on MUSA.

## Overview

This wrapper is intended for projects that already target DeepGEMM-style Python APIs and want to run on MUSA through MATE with smaller integration changes.

- Package name: `deep-gemm`
- Import path: `deep_gemm`
- Runtime backend: MATE GEMM and logits operators on MUSA

The package currently covers grouped GEMM, dense FP8 GEMM, and MQA-logits-related helper APIs.

## Requirements

Before using this wrapper, make sure the following are already available:

- MATE is installed and importable
- TorchMUSA and the MUSA runtime environment are available
- The target workload is expected to run on MUSA devices

## Build

Build a wheel from the `wrappers/DeepGEMM` directory:

```bash
python -m build --wheel
```

The generated wheel will be placed under:

```text
dist/
```

## Installation

Install from source:

```bash
pip install --no-build-isolation -e .
```

Install a built wheel:

```bash
pip install dist/deep_gemm-*.whl
```

If you previously installed the legacy `mate-deep-gemm` package, uninstall it
before installing `deep-gemm` so the environment does not keep stale wrapper
metadata.

```bash
pip uninstall -y mate-deep-gemm
pip install dist/deep_gemm-*.whl
```

## Import

Import the package directly:

```python
import deep_gemm
```

Import individual APIs:

```python
from deep_gemm import (
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    fp8_gemm_nt,
    get_paged_mqa_logits_metadata,
    fp8_paged_mqa_logits,
    fp8_mqa_logits,
)
```

## Public APIs

Grouped GEMM:

- `m_grouped_bf16_gemm_nt_contiguous`
- `m_grouped_bf16_gemm_nt_masked`
- `m_grouped_fp8_gemm_nt_contiguous`
- `m_grouped_fp8_gemm_nt_masked`
- Legacy aliases: `fp8_m_grouped_gemm_nt_masked`, `bf16_m_grouped_gemm_nt_masked`

Dense FP8 GEMM:

- `fp8_gemm_nt`

MQA logits helpers:

- `get_paged_mqa_logits_metadata`
- `fp8_paged_mqa_logits`
- `fp8_mqa_logits`

Utility helpers re-exported from `deep_gemm.utils`:

- `get_num_sms`, `set_num_sms`
- `get_tc_util`, `set_tc_util`
- `get_mk_alignment_for_contiguous_layout`
- `get_col_major_tma_aligned_tensor`
- `get_mn_major_tma_aligned_tensor`

## Quick Start

Minimal import example:

```python
import deep_gemm
```

An example script is provided at:

```text
examples/run_deep_gemm.py
```

## Examples

Run the bundled example:

```bash
python examples/run_deep_gemm.py
```

## Notes

- This wrapper preserves the DeepGEMM-style Python surface, but execution is provided by MATE on MUSA
- `get_paged_mqa_logits_metadata(..., block_kv, ...)` currently requires `block_kv == 64`
- The example script currently demonstrates the FP8 grouped GEMM path
