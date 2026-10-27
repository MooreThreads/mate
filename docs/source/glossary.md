# Glossary

This glossary defines key MATE terms for new users.

This page is part of the MATE 0.2.8 documentation set. See
{doc}`Documentation History <documentation_history>` for the release history.

## Core platform

| Term | Definition |
| --- | --- |
| MATE | MUSA AI Tensor Engine, an operator library for generative AI and large language model workloads on Moore Threads GPUs. |
| MUSA | Moore Threads' GPU software and hardware platform. |
| MUSA SDK | The compiler and runtime toolkit for MUSA development. |
| TorchMUSA | The PyTorch integration layer for MUSA devices. |
| GPU | Moore Threads GPU hardware. |

## Compatibility wrappers

| Term | Definition |
| --- | --- |
| wrapper | A compatibility package that preserves an upstream package surface while routing supported calls to MATE. |
| native MATE API | The direct `mate` Python APIs. Use these when no wrapper matches your workload or when you need lower-level control. |
| FlashAttention-3 wrapper (`flash_attn_3`) | Compatibility wrapper for FlashAttention-3 style attention APIs. |
| FlashMLA wrapper (`flash_mla`) | Compatibility wrapper for FlashMLA-style MLA APIs. |
| FlashKDA wrapper (`flash_kda`) | Compatibility wrapper for Kimi Delta Attention APIs. |
| FlashInfer wrapper (`flashinfer-python`) | Compatibility wrapper for selected FlashInfer GEMM, Norm, RoPE, and sparse decode APIs. |
| DeepGEMM wrapper (`deep-gemm`) | Compatibility wrapper for DeepGEMM-style GEMM and logits APIs. |
| MSA wrapper (`fmha_sm100`) | Compatibility wrapper for MiniMax Sparse Attention APIs. |

## Feature terms

| Term | Definition |
| --- | --- |
| FlashAttention | The MATE FMHA attention path, including dense, varlen, and KV-cache workflows. |
| FlashMLA | The MATE MLA path for metadata, decode, and sparse prefill workflows. |
| KDA | Kimi Delta Attention. The MATE KDA APIs cover chunked KDA and decode. |
| FlashInfer Norm | The FlashInfer-compatible RMSNorm, LayerNorm, fused add Norm, and quantization APIs. |
| DeepGEMM | The DeepGEMM-compatible GEMM and logits API surface. |
| MSA (MiniMax Sparse Attention) | The MATE attention path for dense, paged, and sparse MSA workflows. |
| GDN (Gated Delta Network) | The MATE APIs for GDN decode and prefill workflows. |
| SSU (selective state update) | The Mamba2 decode-step recurrence implemented by `mate.mamba.selective_state_update`. |
| SSD (State Space Duality) | The Mamba2 chunked prefill formulation implemented by `mate.mamba.ssd_combined_fwd_varlen`. |
| QSA (Qwen Sparse Attention) | The MATE MQA score and sparse Core Attention path for prepared Qwen-style inputs. |

## Installation and runtime

| Term | Definition |
| --- | --- |
| wheel | A Python distribution artifact that can be installed with `pip`. |
| package source | The `pip` index or repository that provides a package. |
| editable install | An install that points Python at the local checkout while you edit it. |
| source build | Building a package directly from the repository checkout. |
| local wheel | A wheel built from the local checkout. |
| build baseline | The minimum version needed to build or run a feature. |
| runtime | The stage where an installed package or binary is executed. |
| validation | The checks used to confirm that installation or behavior is correct. |
| environment variable | A shell or process setting that changes MATE behavior. |
| AOT | Ahead-of-time compilation. |
| JIT | Just-in-time compilation. |
| MUBIN | Pre-generated kernel metadata and binary artifacts used by selected operator paths. |
| `+musa` | The PEP 440 local version suffix used by MUSA wrapper packages. |
