# MUSA AI Tensor Engine

MATE (**M**USA **A**I **T**ensor **E**ngine) is a centralized library for Generative AI workloads on MUSA. It provides high-performance attention and GEMM operators, and compatibility wrappers for CUDA-oriented Python APIs.

## Highlights

- High-performance attention and GEMM operators for MUSA
- Compatibility wrappers for `flash_attn` and `deep_gemm`
- CLI tools for environment checks, configuration inspection, and replay

## Quick Links

- CLI documentation: [docs/mate_cli.md](docs/mate_cli.md)
- FlashAttention compatibility summary: [docs/flash_attention.md](docs/flash_attention.md)
- FlashAttention wrapper: [wrappers/flash-attention/README.md](wrappers/flash-attention/README.md)
- DeepGEMM wrapper: [wrappers/deep_gemm/README.md](wrappers/deep_gemm/README.md)

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
bash build.sh
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
| `wrappers/flash-attention` | `mate-flash-attention` | `flash_attn` | FlashAttention-compatible APIs on top of MATE attention operators on MUSA | [wrapper README](wrappers/flash-attention/README.md), [compatibility summary](docs/flash_attention.md) |
| `wrappers/deep_gemm` | `mate-deep_gemm` | `deep_gemm` | DeepGEMM-compatible APIs on top of MATE GEMM operators on MUSA | [wrapper README](wrappers/deep_gemm/README.md) |

## Build Documentation

After installing `mate`, build the Sphinx docs with:

```bash
pip install sphinx furo
cd docs
make html
```

## Acknowledgement

MATE is inspired by [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [cutlass](https://github.com/NVIDIA/cutlass), [FlashMLA](https://github.com/deepseek-ai/FlashMLA), and [DeepGemm](https://github.com/deepseek-ai/DeepGEMM).
