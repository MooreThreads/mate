.. _apimarlin:

Marlin W4A16
============

MATE's Marlin wrapper, for frameworks that already speak upstream vLLM's Marlin vocabulary.
The **layouts are upstream's** — the repacked weight, the permuted scales and the packed zero
points all keep the byte layout NVIDIA Marlin uses, so a converted checkpoint stays usable and
upstream's Python utilities can keep producing these tensors. The **implementation is MATE's**:
:func:`mate.gemm.dense_w4a16_gemm` and the repack in :mod:`mate.marlin.layout`.

What a consumer calls
---------------------

Two names. Load-time work happens in the first and nowhere else:

* :func:`mate.marlin.prepare_marlin_w4a16` converts upstream's *already converted* tensors into the
  kernel frame MATE's FFI expects and returns a :class:`mate.marlin.MarlinW4A16KernelFrame`.
* :func:`mate.marlin.apply_gptq_marlin_linear` runs a forward with no branch on ``M``, so nothing is
  frozen at trace time.

The conversion is not a stride trick, for two independent reasons. MATE's int4 B operand is consumed
with consecutive ``k``, even ``k`` in the low nibble, while upstream's words are a 16x64 tile map; no
TMA stride descriptor expresses that. And the two sides disagree about what a nibble *means*: MATE's
declared operand frame decodes a presented nibble ``v`` as ``v - 8`` (``mate/gemm.py:1503-1507``,
``mate/jit/gemm/dense_w4a16_gemm.py:28-31,46-49``), while the measured atom reads two's complement,
so upstream's ``uint4b8`` codes must have their sign bit flipped — ``s(u ^ 0x8) = u - 8`` for all 16
codes. Presenting them verbatim is a uniform ``8 * scale`` error per element, which no shape-based
comparison detects. One load-time pass pays for both.

.. currentmodule:: mate.marlin

What a consumer calls
---------------------

.. autofunction:: prepare_marlin_w4a16
.. autoclass:: MarlinW4A16KernelFrame
   :members:
.. autofunction:: apply_gptq_marlin_linear

Frame conversion
----------------

The host path receives upstream's *output*, so these are inverses of upstream layouts, each one
stating its direction. :func:`marlin_scales_to_musa` and :func:`marlin_zero_points_to_musa` return the
plain ``(N, group)`` orientation the kernel's scale operand uses; upstream's frame is ``(group, N)``.

.. autofunction:: marlin_weight_to_musa
.. autofunction:: marlin_scales_to_musa
.. autofunction:: marlin_zero_points_to_musa

Upstream-mirrored surface
-------------------------

Same names, same argument order and same results as
``vllm/model_executor/layers/quantization/utils/marlin_utils.py``, implemented by MATE. Where a
signature says ``num_bits``, only 4 is supported: MATE's kernel is int4 x fp16, so the 8-bit
branches are refused with a reason rather than approximated.

.. autofunction:: gptq_marlin_repack
.. autofunction:: awq_marlin_repack
.. autofunction:: get_scale_perms
.. autofunction:: marlin_permute_scales
.. autofunction:: marlin_zero_points
.. autofunction:: awq_to_marlin_zero_points
.. autofunction:: marlin_padded_nk
.. autofunction:: marlin_repacked_nk
.. autofunction:: marlin_pad_qweight
.. autofunction:: marlin_pad_scales
.. autofunction:: marlin_pad_dim
.. autofunction:: marlin_unpad_output

MATE additions
--------------

``unpack_marlin_zero_points`` inverts both the packing and the permutation; upstream has no such
function because its kernel consumes the permuted frame directly. ``check_musa_marlin_supports_shape``
returns ``(ok, reason)`` for the shapes MATE's kernel can serve, since its requirements are stricter
than upstream's in one direction: the K256 mainloop tile carries two K128 scale blocks, so ``K % 256
== 0`` is structural, and a TP-sharded ``down`` projection at TP=2 (rank-local K=4480) is refused
rather than silently padded.

.. autofunction:: unpack_marlin_zero_points
.. autofunction:: marlin_is_tile_aligned
.. autofunction:: check_musa_marlin_supports_shape

Layout primitives
-----------------

``mate.marlin.layout`` is the repack that ships and the tile index maps; ``mate.marlin.quant`` holds
the torch forms of the upstream scale and zero-point layout; ``mate.marlin.reference`` is the
numpy-only oracle the gates compare against, and nothing at runtime imports it.

.. autofunction:: repack_gptq
.. autofunction:: unpack_gptq
