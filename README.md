# MUSA AI Tensor Engine

MATE (**M**USA **A**I **T**ensor **E**ngine) is a centralized library for Generative AI workloads on MUSA. It provides high-performance attention and GEMM operators, and compatibility wrappers for CUDA-oriented Python APIs.

## Highlights

- High-performance attention and GEMM operators for MUSA
- Compatibility wrappers for `flash_attn_3`, `sageattention`, `flash_mla`, and `deep_gemm`
- CLI tools for environment checks, configuration inspection, and replay

## Quick Links

- CLI documentation: [docs/mate_cli.md](docs/mate_cli.md)
- FlashAttention-3 compatibility summary: [docs/flash_attention.md](docs/flash_attention.md)
- FlashAttention-3 wrapper: [wrappers/flash-attention/README.md](wrappers/flash-attention/README.md)
- SageAttention wrapper: [wrappers/SageAttention/README.md](wrappers/SageAttention/README.md)
- FlashMLA wrapper: [wrappers/FlashMLA/README.md](wrappers/FlashMLA/README.md)
- DeepGEMM wrapper: [wrappers/DeepGEMM/README.md](wrappers/DeepGEMM/README.md)

## Requirements

| Component | Requirement |
| --- | --- |
| MUSA Toolkit | `4.3.6` or later |
| TorchMUSA | `2.7` or later |
| Architecture | `Pinghu (MP31)` |

## Build From Source

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
python -m build --wheel --no-isolation
```

For local development, install MATE in editable mode:

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
pip install --no-build-isolation -e . -v
```

If you forgot `--recursive` when cloning, initialize submodules before building or
installing:

```bash
git submodule update --init --recursive
```

Pre-build AOT kernels before packaging the wheel:

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
cd mate
MATE_MUSA_ARCH_LIST=3.1 python -m mate.aot
python -m build --wheel --no-isolation
```

Customize AOT coverage by operator family when needed:

```bash
python -m mate.aot --attention-aot-level 0 --add-gemm true --add-moe false
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `mate/` | Core Python package and public APIs |
| `wrappers/` | Compatibility wrapper packages for existing Python ecosystems |
| `docs/` | Markdown docs and Sphinx sources |
| `tests/` | Correctness and integration tests |
| `benchmarks/` | Performance and benchmarking scripts |

## MATE CLI

MATE provides a command-line interface for configuration, debugging, diagnostics, and replay.

| Command | Purpose |
| --- | --- |
| `mate check` | Validate the runtime environment |
| `mate show-config` | Display installation and runtime configuration |
| `mate env` | Show relevant environment variables |
| `mate replay --dir PATH` | Replay API calls from Level 10 dumps |
| `mate list-dumps PATH` | List recorded dump directories |

Example:

```bash
mate check
mate show-config
mate env
mate replay --dir mate_dumps/
mate list-dumps mate_dumps/
```

See [docs/mate_cli.md](docs/mate_cli.md) for full CLI documentation.

## Wrappers

MATE uses the packages under `wrappers/` as a compatibility layer for CUDA-oriented software stacks on MUSA. These wrappers preserve familiar package names and high-level APIs while routing execution to MATE operators and kernels on MUSA, which helps existing integrations migrate with smaller code changes.

| Wrapper | Package | Import Path | Purpose | Documentation |
| --- | --- | --- | --- | --- |
| `wrappers/flash-attention` | `flash_attn_3` | `flash_attn_interface` | FlashAttention-3-compatible APIs on top of MATE attention operators on MUSA | [wrapper README](wrappers/flash-attention/README.md), [compatibility summary](docs/flash_attention.md) |
| `wrappers/SageAttention` | `sageattention` | `sageattention` | SageAttention-compatible dense quantized attention wrapper on top of MATE on MUSA | [wrapper README](wrappers/SageAttention/README.md) |
| `wrappers/FlashMLA` | `flash_mla` | `flash_mla` | FlashMLA-compatible MLA dense/sparse decode and sparse prefill APIs on top of MATE MLA operators on MUSA | [wrapper README](wrappers/FlashMLA/README.md) |
| `wrappers/DeepGEMM` | `deep-gemm` | `deep_gemm` | DeepGEMM-compatible APIs on top of MATE GEMM operators on MUSA | [wrapper README](wrappers/DeepGEMM/README.md) |

## Build Documentation

After installing `mate`, build the Sphinx docs with:

```bash
pip install sphinx furo
cd docs
make html
```

## Acknowledgement

MATE is inspired by [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [cutlass](https://github.com/NVIDIA/cutlass), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), and [DeepGemm](https://github.com/deepseek-ai/DeepGEMM).
