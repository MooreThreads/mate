.. _apisparsemla:

Sparse MLA
==========

Use the :doc:`FlashInfer wrapper <../wrappers/flashinfer_wrapper>` when your
framework expects the ``flashinfer.rope`` and ``flashinfer.decode`` package
surface. The native APIs below expose MATE's fused MLA RoPE quantization and
FP8 sparse decode path directly.

RoPE and FP8 quantization
-------------------------

.. currentmodule:: mate.sparse_mla_interface

``mla_rope_quantize_fp8`` applies rotary embedding to the 64-value RoPE tail,
quantizes the 512-value latent component and the rotated tail to FP8 E4M3, and
supports writing both components into slices of caller-owned 576-byte rows.

.. autofunction:: mla_rope_quantize_fp8

FP8 sparse decode
-----------------

Use ``mate.sparse_mla_interface.get_batch_decode_metadata_mla`` to prepare
scheduler metadata once while the query shape, sparse lengths, and top-k remain
unchanged, then pass it to subsequent decode calls.

.. autofunction:: sparse_mla_fp8_decode

GLM NoPE sparse decode
----------------------

``sparse_mla_nope_decode`` handles GLM's BF16 Q512 contract and its inline
528-byte cache rows containing 512 E4M3 values and four arbitrary FP32 scales.
The cache layout is distinct from FlashMLA's packed pages described in
:ref:`apiattention`; legacy 656-byte padded rows are also accepted, and both
row widths use the same inline payload so a single call works with either.

The cache is token-major with shape ``[capacity, 1, row_bytes]``. The first 512
bytes hold the FP8 E4M3 latent and bytes 512 to 527 hold the four FP32
group-128 scales of that token; 656-byte rows leave bytes 528 to 655 unused.
``slots`` selects the row to write, and negative or out-of-range slots are
skipped. The payload is inline, so Python fills it with
``mate.sparse_mla.glm_nope.write_glm_nope_cache(latent, cache, slots)``, which
quantizes BF16 ``latent`` rows of 512 values into E4M3 with per-group-128 FP32
scales and writes them to the ``slots`` rows of a contiguous uint8, int8, or
FP8 cache.

``softmax_scale`` is required because the caller applies the absorbed weight
before the call, so the operator cannot infer the scale. ``topk_length`` bounds
a prefix of the candidate list, including holes: index ``i`` attends to
``indices[..., :topk_length[i]]`` rather than to ``topk_length[i]`` valid IDs.
In the rare case where a valid index beyond ``topk_length[i]`` points at a token
containing NaN, the output for that query contains NaN, so keep padded
candidates free of NaN. When ``tile_scheduler_metadata`` and ``num_splits`` are
omitted, the operator builds them with ``get_mla_metadata``; pass them together
to reuse the metadata across calls.

.. autofunction:: mate.sparse_mla.glm_nope.sparse_mla_nope_decode
