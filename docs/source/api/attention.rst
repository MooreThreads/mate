.. _apiattention:

Attention
=========

Use the ``flash_attn_3`` or ``flash_mla`` wrapper packages first when your
integration already targets FlashAttention-3 or FlashMLA Python APIs. Use the
MATE APIs below when wrapper coverage is not enough.

FMHA
----

.. currentmodule:: mate.mha_interface

Forward dtype notes:

- ``flash_attn_varlen_func`` and ``flash_attn_with_kvcache`` accept
  ``torch.float16``, ``torch.bfloat16``, ``torch.float8_e4m3fn``, and
  ``torch.float8_e5m2`` inputs on the MATE FMHA forward path.
- For FP8 inputs, optional ``q_descale``, ``k_descale``, and ``v_descale``
  tensors are ``torch.float32`` scale factors with shape
  ``(batch_size, num_heads_kv)``.
- FP8 inputs produce ``torch.bfloat16`` outputs by default.
- The FMHA forward path also supports an optional ``qv`` input, including FP8
  inputs with ``qv``.
- When both ``q`` and ``qv`` are FP8, ``q_descale`` applies to both query
  tensors.
- ``flash_attn_with_kvcache(..., only_qv=True)`` skips the QK score path.
  In this mode, ``qv`` is required and ``q`` / ``k_cache`` may be ``None``.
- For best FP8 attention performance, use MUSA SDK 5.2.0 or newer when
  available.
- Chunked FMHA forward supports large head dimensions, including Gemma
  512-512 attention. When the optional ``qv`` input is provided, the Q/K and
  value head dimensions must each be divisible by 8.
- Without ``qv``, FP16/BF16 head dimensions must be divisible by 2 and FP8
  head dimensions must be divisible by 4. These alignment rules apply in
  addition to the ``headdim <= 512`` limit documented in the compatibility
  guide.
- Use ``flash_attn_combine`` to merge partial FMHA outputs and log-sum-exp
  buffers when your integration splits the attention computation.
- Use ``get_scheduler_metadata`` to precompute the ``scheduler_metadata`` input
  for ``flash_attn_with_kvcache`` when you call the FMHA path directly.

Max logits
~~~~~~~~~~

``flash_attn_varlen_func(..., return_max_logits=True)`` and
``flash_attn_with_kvcache(..., return_max_logits=True)`` additionally return
FP32, non-differentiable per-head maxima with shape ``(num_heads,)``. The
maxima are reduced over valid batch, query, and key positions after scale,
softcap, and masking, and they exclude learnable sinks. Heads without any
valid attention score are ``-inf``.

The call returns ``(out, max_logits)``, or ``(out, lse, max_logits)`` when
``return_softmax_lse=True``.

Restrictions:

- ``return_max_logits`` requires the ``mubin`` or ``mutlass`` forward backend.
  The TileLang backend rejects the argument.
- MLA ``scheduler_metadata`` tuples are not compatible with max logits.
- FP8 Q/K descales and, without softcap, a negative ``softmax_scale`` are not
  supported.
- Under context parallelism, each rank reduces its own local key shard, so the
  maxima are rank-local rather than global.
- Mark the result as non-differentiable in custom autograd integrations; the
  backward pass does not propagate through ``max_logits``.

Backward backend
~~~~~~~~~~~~~~~~

``flash_attn_varlen_func(..., bwd_backend=...)`` selects the backward
implementation independently of the forward backend:

- ``auto`` (default) prefers MUBIN on supported MP31 inputs and otherwise uses
  TileLang.
- ``mubin`` requires ASM support and rejects forward contracts that the ASM
  backward ABI does not cover: ``qv``, paged KV, learnable sinks, chunked
  attention, context parallelism, and FP8 descales. ``auto`` falls back to
  TileLang for those contracts instead of ignoring their inputs.
- The ASM backward path is available for the 128/128 and 192/128
  Q/K-to-V head-dimension pairs.
- ``tilelang`` explicitly selects the TileLang backward.

Known limitations
~~~~~~~~~~~~~~~~~

When the head dimension is below 128 and the KV inputs are non-contiguous
while padded positions contain ``NaN`` or infinity, the FMHA result can
contain ``NaN``. Keep the KV inputs contiguous or clear the padding.

.. autofunction:: flash_attn_combine
.. autofunction:: flash_attn_with_kvcache
.. autofunction:: get_scheduler_metadata
.. autofunction:: flash_attn_varlen_func
.. autofunction:: fmha_backward

Standalone backward follows the FA3 C++ ``mha_bwd`` parameter order. Gradient
buffers are optional independently; provided buffers are returned by identity.
``backend="auto"`` currently selects TileLang. Aligned TileLang outputs are
written directly, including strided views; head-dimension padding may require
temporary storage. The caller must keep gradient buffers disjoint from inputs
and from each other.

.. code-block:: python

   dq, dk, dv, softmax_d = mate.fmha_backward(
       dout, q, k, v, out, softmax_lse, dq, dk, dv,
       softmax_scale=q.shape[-1] ** -0.5,
       window_size_left=127, window_size_right=0,
       backend="tilelang",
   )


MLA
---

.. currentmodule:: mate.flashmla

FlashMLA supports multiple Q/K head-ratio configurations.

Minimal dense MLA example:

.. code-block:: python

   import torch
   from mate.flashmla import get_mla_metadata, flash_mla_with_kvcache

   q = torch.randn((1, 32, 8, 576), device="musa", dtype=torch.bfloat16)
   k_cache = torch.randn((4, 16, 2, 576), device="musa", dtype=torch.bfloat16)
   block_table = torch.zeros((1, 4), device="musa", dtype=torch.int32)
   cache_seqlens = torch.tensor([32], device="musa", dtype=torch.int32)

   tile_scheduler_metadata, num_splits = get_mla_metadata(
       cache_seqlens=cache_seqlens,
       num_q_tokens_per_head_k=q.shape[1] * q.shape[2] // k_cache.shape[2],
       num_heads_k=k_cache.shape[2],
       num_heads_q=q.shape[2],
       q=q,
       bs=q.shape[0],
   )

   out, lse = flash_mla_with_kvcache(
       q=q,
       k_cache=k_cache,
       block_table=block_table,
       cache_seqlens=cache_seqlens,
       head_dim_v=512,
       tile_scheduler_metadata=tile_scheduler_metadata,
       num_splits=num_splits,
   )

Packed KV cache formats
~~~~~~~~~~~~~~~~~~~~~~~

Sparse MLA decode selects a packed cache layout from the Q and cache
dimensions. Cache tensors use ``[pages, page_size, 1, bytes_per_token]``:

- ``656`` bytes: DeepSeek V3.2, with 512 E4M3 values, four inline FP32
  group-128 scales, and 64 BF16 RoPE values.
- ``584`` bytes: V4, with 448 E4M3 and 64 BF16 values per token followed by an
  8-byte E8M0 scale footer.
- ``528`` bytes: V4.1, with 512 E4M3 values per token followed by a 16-byte
  group-32 E8M0 footer.
- ``288`` bytes: V4.1 NVFP4, accepted only as an extra cache, with 256 packed
  E2M1 bytes and 32 group-16 E4M3 scales per token.

Pass an extra NVFP4 cache through ``extra_k_cache`` together with
``extra_indices_in_kvcache`` and ``extra_topk_length``. Main/extra pairs are
``584/584``, ``528/528``, or ``528/288``. ``extra_k_cache`` is valid only for
sparse 512-dim decode and is rejected otherwise.

GLM NoPE keeps a separate 528-byte token layout. Use
``flashinfer.mla.SparseMLASm120Wrapper`` with
``kv_scale_format="arbitrary_fp32"`` for it; a 528-byte FlashMLA cache always
means V4.1. See :doc:`Sparse MLA <sparse_mla>`.

Known limitations
~~~~~~~~~~~~~~~~~

If a packed MLA KV cache token contains ``NaN``, the MLA output for queries
that attend to that token also contains ``NaN``. Sanitize padding and unused
(token, scale, RoPE) slots before writing them to the cache.

.. autofunction:: get_mla_metadata
.. autofunction:: flash_mla_with_kvcache
.. autofunction:: flash_mla_sparse_fwd
.. autofunction:: flash_mla_sparse_fwd_pack8
