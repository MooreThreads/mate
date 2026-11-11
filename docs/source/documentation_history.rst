Documentation History
=====================

This page records documentation changes by release.

For runtime, API, and compatibility changes, see
`GitHub Releases <https://github.com/MooreThreads/mate/releases>`_.

The current docs site history starts at ``0.2.2``.

0.2.8.post1
-----------

**Dense attention**

- Added the standalone backward entry point ``mate.fmha_backward`` to
  :doc:`Attention <api/attention>`, following the FA3 C++ ``mha_bwd``
  parameter order with independently optional gradient buffers, and exposed
  the matching ``_flash_attn_backward`` entry in the
  :doc:`FlashAttention wrapper <wrappers/flash_attention_wrapper>`.

**Linear attention**

- Documented packed GDN state pools in the :doc:`GDN support matrix <gdn>`:
  padded ``[conv | ssm | pad]`` page views, ``state_indices`` pool mapping,
  and strided initial/output states.

**Sparse attention**

- Clarified the QSA vector-load alignment, block-hint width, and decode
  ``page_size`` contracts in :doc:`QSA <api/qsa>`.

**Installation**

- Updated the :doc:`installation <install>` guidance: TileLang MUSA
  (``tilelang-musa==0.1.12+musa.5``) is now a required dependency instead of
  an optional extra.

0.2.8
-----

**Dense attention**

- Added max-logit output and backward backend selection to
  :doc:`Attention <api/attention>`, covering the supported forward backends, the
  MLA metadata restriction, and rank-local behavior under context parallelism.
- Noted the ``NaN`` limitation for FMHA with head dimensions below 128 and
  non-contiguous KV padded with ``NaN``/infinity.

**GEMM and MoE GEMM**

- Extended :doc:`Mega MoE <api/mega_moe>` with SiTU activation, the split
  prefill pipeline, the ``prefill_pool_capacity_ratio`` and
  ``prefill_l1_ring`` memory modes, and the token-alignment query.
- Documented BF16 logits output for the DeepGEMM indexer logits helpers in
  :doc:`GEMM <api/gemm>`.

**Linear attention**

- Added :doc:`KDA <api/kda>` notes for the prepare/recurrence split, the
  ``chunk_states`` pool, and ``checkpoint_every_n_tokens`` state checkpoints.
- Updated the :doc:`GDN support matrix <gdn>` for BF16-state decode and MTP.

**Sparse attention**

- Added :doc:`QSA <api/qsa>` coverage for MQA score and sparse Core Attention.
- Documented the sparse MLA packed KV cache formats (V3.2, V4, V4.1, and the
  NVFP4 extra cache), the extra-cache inputs, and the ``NaN`` limitation when a
  cached token contains ``NaN`` in :doc:`Attention <api/attention>`.
- Extended :doc:`Sparse MLA <api/sparse_mla>` and the
  :doc:`FlashInfer wrapper <wrappers/flashinfer_wrapper>` with GLM NoPE sparse
  decode.
- Documented exact sparse top-k selection in :doc:`MSA <api/msa>`.

**Miscellaneous**

- Documented GLM-5.3-Flash-style 288-expert routing in the
  :doc:`Python API index <api_reference>`.
- Added :doc:`Mamba / SSU <api/mamba>` coverage for native
  ``selective_state_update``, packed variable-length SSD prefill, and kernel
  prewarm/workspace control, plus the :doc:`FlashInfer wrapper
  <wrappers/flashinfer_wrapper>` Mamba/SSD adapter surface.
- Updated the :doc:`installation <install>` baseline to 0.2.8 version
  references.

**CLI**

- Added the MTCC and TileLang version rows reported by ``mate show-config`` to
  the :doc:`MATE CLI reference <mate_cli>`.

0.2.7
-----

- Updated the :doc:`installation and compatibility guidance <install>` for the
  MUSA SDK 5.2.0 build baseline.
- Added :doc:`FlashAttention forward support
  <wrappers/flash_attention_forward_compatibility>` for chunked large head
  dimensions, including Gemma 512-512 attention.
- Added :doc:`FlashMLA <api/attention>` metadata support for arbitrary batch
  sizes, including the additional multi-batch cases covered by the release
  tests.
- Added :doc:`KDA <api/kda>` state-pool support.
- Added :doc:`FlashInfer Norm API reference <api/flashinfer>`.
- Extended the :doc:`DeepGEMM wrapper <wrappers/deep_gemm_wrapper>` with
  cuBLASLt-style GEMM, M-grouped and K-grouped FP8 interfaces, scale-layout
  helpers, and ``disable_ue8m0_cast=True`` compatibility. The wrapper page
  documents its MUSA-specific behavior and limits.
- Fixed hangs in affected MUTLASS S4FP8 decode scenarios.
- Extended the :doc:`SageAttention wrapper
  <wrappers/sageattention_wrapper>` with dense FP8/FP16 compatibility names
  and a varlen entry point. The wrapper page documents which arguments are
  signature-compatible only and which MATE paths execute.
- Added :doc:`MSA <api/msa>` notes for indexer behavior and short-query
  metadata handling.
- Added the :doc:`Glossary <glossary>` page for core MATE terms.
- Documented the ``MATE_DRY_RUN`` developer and diagnostic option in the
  :doc:`environment-variable reference <environment_variables>`.

0.2.6
-----

- Updated :doc:`Installing MATE <install>` and the repository README with
  the 0.2.6 compatibility baseline and package-install guidance.
- Added MUTLASS S4FP8 masked GroupGEMM coverage to :doc:`GEMM <api/gemm>`.
- Expanded the :doc:`FlashAttention wrapper
  <wrappers/flash_attention_wrapper>` and :doc:`DeepGEMM wrapper
  <wrappers/deep_gemm_wrapper>`, including related :doc:`GEMM <api/gemm>` and
  :doc:`Python APIs <api_reference>` coverage.
- Added the :doc:`FlashInfer wrapper <wrappers/flashinfer_wrapper>` and
  :doc:`FlashInfer API reference <api/flashinfer>` for GEMM, FP8 MLA RoPE
  quantization, and sparse decode, together with the native
  :doc:`Sparse MLA APIs <api/sparse_mla>` and an end-to-end FP8 example.
- Added the :doc:`Development Environment <development_environment>` page and
  TileLang 0.1.12 installation guidance.

0.2.5
-----

- Added Kimi Delta Attention (KDA) decode coverage to
  :doc:`Python APIs <api_reference>` and :doc:`KDA <api/kda>`.
- Expanded :doc:`Attention <api/attention>` for FMHA ``only_qv`` and MLA
  head-ratio coverage.
- Added :doc:`MSA <api/msa>` direct APIs and the
  :doc:`MSA/fmha_sm100 wrapper <wrappers/fmha_sm100_wrapper>` for MiniMax
  Sparse Attention (MSA) workflows on MUSA.
- Updated :doc:`Installing MATE <install>` for the MUSA simple package index.
- Added MUBIN artifact management commands to :doc:`Command Line Interface
  <cli>`.
- Refined FlashMLA/DenseMLA support for multiple head-ratio configurations.

0.2.4
-----

- Updated :doc:`Installing MATE <install>` for wrapper-first ``pip``
  installation, including automatic ``mate`` dependency installation.
- Added W4A8 mixed-dtype MoE GEMM to :doc:`Python APIs <api_reference>` and
  :doc:`GEMM <api/gemm>`.
- Added Mega MoE to :doc:`Python APIs <api_reference>` and
  :doc:`Mega MoE <api/mega_moe>`.

0.2.3
-----

- Expanded the docs set around :doc:`Overview <overview>`,
  :doc:`Installing MATE <install>`, :doc:`Wrappers <wrapper_tutorials>`,
  :doc:`Diagnostic Overview <diagnostics>`, :doc:`Command Line Interface
  <cli>`, :doc:`Design and Architecture <design_and_architecture>`, and
  :doc:`Python APIs <api_reference>`.
- Added wrapper pages for :doc:`FlashAttention
  <wrappers/flash_attention_wrapper>`, :doc:`SageAttention
  <wrappers/sageattention_wrapper>`, :doc:`FlashMLA
  <wrappers/flash_mla_wrapper>`, :doc:`FlashKDA
  <wrappers/flash_kda_wrapper>`, and :doc:`DeepGEMM
  <wrappers/deep_gemm_wrapper>`.
- Added API pages for :doc:`Attention <api/attention>`,
  :doc:`GEMM <api/gemm>`, :doc:`HyperConnection <api/hyperconnection>`, and
  :doc:`KDA <api/kda>`.
- Moved :doc:`Command Line Interface <cli>`,
  :doc:`Environment Variables <environment_variables>`, and
  :doc:`GDN Support Matrix <gdn>` into the Sphinx site structure.

0.2.2
-----

- Established the initial public docs baseline in :doc:`the home page <index>`
  and the early API set.
- Added the :doc:`HyperConnection API page <api/hyperconnection>`.
- Updated :doc:`FlashAttention <api/attention>` and
  :doc:`GDN Support Matrix <gdn>`.
- Refreshed :doc:`the home page <index>` structure.
