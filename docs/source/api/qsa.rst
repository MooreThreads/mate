QSA (Qwen Sparse Attention)
===========================

QSA is the sparse attention layer used by Qwen's hybrid attention stack. It has
two stages. A lightweight MQA indexer scores compressed micro-blocks, the caller
keeps the top-scoring blocks, and core attention then runs on the original
tokens inside those selected blocks.

MQA names the indexer's structure, not a second attention mechanism: several
query heads share one compressed key head, which keeps block scoring cheap.
Core attention is the main attention over the selected tokens.

MATE implements the two compute stages for prepared inputs:

- ``qsa_indexer_mqa_score``, ``qsa_mqa_prefill`` (also exported as ``qsa_mqa``)
  and ``qsa_mqa_decode`` produce indexer scores.
- ``qsa_sparse_attention`` runs sparse core attention over the selected tokens.

Projection, RMSNorm, RoPE, cache updates, top-k selection and index expansion
belong to the caller. Native execution supports MP31 (S5000) with
TileLang-MUSA 0.1.12+musa.5; other architectures are rejected before
compilation.

MQA score
---------

The canonical score input is BF16 or FP16 query ``[B,Q,4,128]`` and compressed
key ``[B,N,128]``. Each compressed key represents four tokens. Scores sum the
four head-wise ReLU dot products, scaled by ``1/sqrt(128)``. Incomplete or
future blocks are masked.

``qsa_mqa_prefill`` takes packed queries, shared keys and per-row key intervals.
``qsa_mqa_decode`` takes paged compressed keys ``[pages, page_size, 1, dim]``, a
page table and compressed context lengths. Its native path supports only
``heads, dim == (4, 128)`` with BF16/FP16 and ``page_size`` in
``8, 16, 32, 64``; any other page size raises under ``backend="tilelang"`` and
requires ``backend="reference"``. ``scale`` is a multiplier; ``score_scale`` is
a denominator. Do not pass both. The caller supplies valid active pages; unused
entries may be ``-1``. Native decode masks missing or out-of-range pages to
``-inf``.

Native score and prefill kernels read Q and K through vector loads, so the same
three layout rules apply to ``query`` and ``block_keys`` in score and to
``query`` and ``keys`` in prefill:

- the feature stride (``stride(-1)``) must be ``1``;
- ``data_ptr()`` must be 16-byte aligned;
- every outer axis with more than one element must have a positive stride whose
  byte span (``stride * element_size``) is a multiple of 16.

For BF16 and FP16 this means outer strides must be positive multiples of 8
elements; axes of size 1 are exempt. A sliced view that satisfies the shape
rules but not these stride rules fails in the native path with
``vector loads require unit feature stride and 16-byte alignment``; pass a
contiguous copy or adjust the slice offsets instead. ``qsa_mqa_decode`` is not
subject to this check and accepts broadcast Q through its LSU load path.

Core Attention
--------------

``qsa_sparse_attention`` consumes framework-prepared request-local indices
shared by all Q heads. TileLang requires int32 indices, positions, lengths and
page tables; ``-1`` pads, and kernels mask invalid or future tokens. Token-only
selections may include padding beyond the canonical 2051 slots, and generic
paths use the actual runtime width, so the caller owns the selected-token
budget; block-hint widths follow the fixed formula below. Supplied metadata is
used without dtype conversion. Batch/query dimensions of query, indices, gate
and output must form zero-copy row views; otherwise pass framework-prepared
flattened tensors.

- Canonical geometry is 24 Q heads, 2 KV heads and D256. Contiguous,
  direct-paged and SplitKV kernels also accept positive divisible Hq/Hkv with
  group size up to 16. Global Q/O use logical heads, and dummy compute-tile rows
  are zero-filled in shared memory. Structured prefill and warp specialization
  require canonical geometry; generic paged prefill also supports local 3/1
  heads.
- Paged K/V use ``[pages, page_size, Hkv, 256]``; reference execution also
  accepts flattened varlen K/V. Native kernels mask missing or out-of-range
  pages. ``seq_lens_q`` counts query rows per request independently of cached
  causal positions, and padded rows produce zero output.
- Optional int32 ``block_indices`` describes the four-token blocks behind
  ``token_indices``. Structured paged prefill uses the hint only when the KV
  page size is divisible by four; otherwise it is ignored and the plain
  ``token_indices`` path runs. With a block hint the widths are tied:
  ``token_indices.shape[-1] == block_indices.shape[-1] * 4 + 3``, and
  ``block_indices.shape[-1]`` must be in ``[1, 512]``. The widest block-hint
  selection is therefore 2051 tokens, so wider selections such as 2055 must use
  the token-only path.
- ``num_splits=None`` selects the prefill/decode schedule, and a positive
  integer sets the split count explicitly; ``1`` disables SplitKV. Warp
  specialization is opt-in. ``gate`` applies sigmoid gating, while output
  projection remains external. ``return_lse=True`` exports native softmax
  statistics; paged execution combines SplitKV statistics without repeating QK
  or PV.
- Q/K/V, indices, metadata and optional ``out`` accept sliced views and storage
  offsets, with concrete strides passed as runtime int64. K/V require aligned
  unit feature strides, and the regular prefill path also requires contiguous
  K/V. MQA score and prefill enforce the aligned Q/K layout described in
  `MQA score`_, while MQA decode accepts broadcast Q through its LSU load path.
  Native paths bind ``out`` directly and reject input/output overlap.

The caller prepares valid request IDs, monotonic cumulative lengths ending at
the tensor row count, and sequence lengths within KV capacity. Native wrappers
check shape, dtype, device and layout without reading device contents.
Warmed-up native MQA and logical/physical Core APIs support MUSA Graph
capture/replay; replay consumes current metadata, and explicit reference
execution is outside this guarantee.

``selected_kv_indices=`` accepts flattened BF16 Q/K/V with physical token
indices instead of page translation or causal filtering. Its metadata-only
validation supports public MUSA Graph capture/replay after warmup. Optional
``out`` and natural-log ``lse`` buffers bind directly, and providing ``lse``
writes it regardless of ``return_lse``. Empty selections return zero output and
``-inf`` LSE. FP8 KV and FP8 compute remain unsupported. ``softmax_scale``
aliases ``scale``, and supplying both is an error. Physical indices cannot be
combined with logical request/page metadata or gating.

``backend='tilelang'`` (the default) uses native kernels for supported MUSA
layouts and head geometry and raises otherwise, with no automatic reference
fallback; ``reference`` selects the PyTorch oracle, and ``auto`` is not
supported.

Entrypoints
-----------

.. currentmodule:: mate

.. autofunction:: qsa_indexer_mqa_score
.. autofunction:: qsa_mqa
.. autofunction:: qsa_mqa_decode
.. autofunction:: qsa_mqa_prefill
.. autofunction:: qsa_sparse_attention
