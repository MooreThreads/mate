# MUSA AI Tensor Engine

MATE(**M**USA **A**I **T**ensor **E**ngine) is a centralized library for Generative AI.

## Installation

### Requirements

MUSA Toolkits: 4.3.4 or later

TorchMUSA: 2.7.1 or later

Architecutre: Pinghu(MP31)

### Install From Source

```bash
git clone git@github.com:MooreThreads/mate.git --recursive
cd mate
pip install --no-build-isolation . -v
```

### Build From Source

```bash
git clone git@github.com:MooreThreads/mate.git --recursive
cd mate
python -m build --wheel --no-isolation
```

## Build Docs

**After installing mate**, run the following commands:

```bash
pip install sphinx furo
cd mate/docs
make html
```

## Acknowledgement

Mate is inspired by [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [cutlass](https://github.com/NVIDIA/cutlass), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) and [DeepGemm](https://github.com/deepseek-ai/DeepGEMM) projects.
