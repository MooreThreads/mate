# flash_mla

`flash_mla` is a compatibility wrapper package that preserves the official `flash_mla` package name and import path on top of MATE MLA operators on MUSA.

## Overview

This wrapper is intended for projects that already target the FlashMLA Python API and want to run MLA dense/sparse decode and sparse prefill on MUSA through MATE with smaller integration changes.

- Package name: `flash_mla`
- Import path: `flash_mla`
- Runtime backend: MATE MLA operators on MUSA

The package currently exposes the minimal FlashMLA-compatible surface:

- `FlashMLASchedMeta`
- `get_mla_metadata`
- `flash_mla_with_kvcache`
- `flash_mla_sparse_fwd`

## Requirements

Before using this wrapper, make sure the following are already available:

- MATE is installed and importable
- TorchMUSA and the MUSA runtime environment are available
- The target workload is expected to run on MUSA devices

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

Install from source:

```bash
pip install --no-build-isolation -e .
```

Install a built wheel:

```bash
pip install dist/flash_mla-*.whl
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
- Reusing the same `FlashMLASchedMeta` requires the same decode configuration across calls.
- `flash_mla_with_kvcache(...)` is the dense/sparse decode entry. The wrapper validates `FlashMLASchedMeta`, lazily materializes the real scheduler with `mate.flashmla.get_mla_metadata(...)`, and then forwards to `mate.flashmla.flash_mla_with_kvcache(...)`.
- `flash_mla_sparse_fwd(...)` is the sparse MLA prefill entry.


## Notes

- This wrapper keeps the official FlashMLA import surface, but execution is provided by MATE on MUSA.
- For the authoritative MLA operator behavior, refer to `mate.flashmla`.
