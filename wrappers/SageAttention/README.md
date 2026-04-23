# sageattention

`sageattention` is a compatibility wrapper package that preserves the
SageAttention Python package surface on top of MATE's dense quantized attention
operators on MUSA.

## Overview

This wrapper is intended for projects that already target SageAttention-style
Python APIs and want to run on MUSA through MATE with smaller integration
changes.

- Package name: `sageattention`
- Import path: `sageattention`
- Runtime backend: MATE dense quantized attention operators on MUSA

The wrapper currently exposes the supported public SageAttention-compatible
entries:

- `sageattn`
- `sageattn_qk_int8_pv_fp8_cuda_sm90`

## Requirements

Before using this wrapper, make sure the following are already available:

- MATE is installed and importable
- TorchMUSA and the MUSA runtime environment are available
- The target workload is expected to run on MUSA devices

## Build

Build a wheel from the `wrappers/SageAttention` directory:

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
pip install dist/sageattention-*.whl
```

## Import

Import the package directly:

```python
import sageattention
```

Import individual APIs:

```python
from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda_sm90
```

## Public APIs

The wrapper currently exposes:

- `sageattn`: primary SageAttention-compatible dense attention entry
- `sageattn_qk_int8_pv_fp8_cuda_sm90`: compatibility alias for the supported
  dense quantized attention path

## Quick Start

Minimal dense attention example:

```python
import torch
from sageattention import sageattn

device = "musa"
dtype = torch.bfloat16

q = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)
k = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)
v = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)

out = sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_dtype="int8",
)
```

## Tests

Wrapper-level tests are available in:

```text
tests/test_sageattn_interface.py
```

Run them from the `wrappers/SageAttention` directory:

```bash
pytest tests/test_sageattn_interface.py
```

## Notes

- This wrapper currently supports the dense SageAttention path only
- Input tensors must be on the same MUSA device, use `torch.float16` or
  `torch.bfloat16`, and share the same dtype
- Supported public `tensor_layout` values are `"HND"` and `"NHD"`
- Supported head dimensions are positive values up to `128`
- `qk_quant_dtype` supports `int8` and `fp8`
- The default quantization recipe is `(128, 16, -1, 1)`; passing
  `quant_recipe` overrides `qk_quant_gran`
- Only `qk_quant_gran="per_thread"` is supported as a shortcut; other
  granularities should be expressed via an explicit supported `quant_recipe`
- Unsupported in this wrapper package: varlen, KV-cache wrapper entrypoints,
  public INT8 wrapper entrypoints other than the SM90-compatible name, and
  low-level pre-quantized public APIs
