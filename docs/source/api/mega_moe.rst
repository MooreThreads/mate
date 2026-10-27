.. _apimega_moe:

Mega MoE
========

Mega MoE is MATE's distributed FP8 Mixture-of-Experts execution path for
multi-rank expert-parallel workloads. Use it when your integration already has
token activations, top-k expert indices, and top-k expert weights prepared and
you need MATE to run the expert compute path across a process group.

For integrations that already target DeepGEMM-style distributed MoE
execution, prefer the ``deep-gemm`` wrapper first. Use the direct MATE APIs
below when wrapper coverage is not enough.

The examples here use the ``mate.deep_gemm`` aliases. The same entrypoints are
also available from ``mate.mega_moe``.

Use Mega MoE when
-----------------

- you are integrating a distributed expert-parallel MoE path across multiple
  ranks
- your expert weights are stored as grouped FP8 tensors
- you need the prefill two-layer expert execution path, not just one grouped
  GEMM call

Use the :doc:`GEMM APIs <gemm>` instead when you only need grouped or masked
expert GEMM operators, including the ``0.2.4`` W4A8 mixed-dtype GEMM path.

Typical Flow
------------

Mega MoE uses a shared runtime workspace that holds FP8 inputs, routing
metadata, arrival masks, and intermediate activations for the serial split
``Stage 1 -> Stage 2 -> Stage 3`` execution path.

1. Allocate the symmetric runtime buffer with
   ``get_symm_buffer_for_mega_moe``.
2. Transform logical FP8 expert weights with
   ``transform_weights_for_mega_moe``.
3. Launch the kernel with ``fp8_fp8_mega_moe``.

Decode workloads of up to 128 tokens use the decode kernel. Larger token
counts use the split prefill pipeline, which runs three stages. Prefill
specializations are compiled per configuration; the token count is a runtime
dimension.

Activations
-----------

``activation="swiglu"`` is the default. ``activation="situ"`` is also
supported and takes two additional positive, finite coefficients:

- ``activation_beta`` — required for SiTU.
- ``activation_linear_beta`` — optional SiTU linear coefficient.

``activation_clamp`` applies to SwiGLU only and is rejected for SiTU. Pass the
same ``activation`` value to ``get_symm_buffer_for_mega_moe`` and
``fp8_fp8_mega_moe`` so the workspace layout matches the kernel.

Prefill memory options
----------------------

``get_symm_buffer_for_mega_moe`` accepts two optional prefill memory-saving
modes. Enable at most one at a time:

- ``prefill_pool_capacity_ratio`` (default ``1.0``) scales the L1/L2 activation
  pools; see the warning below before lowering it.
- ``prefill_l1_ring`` (default ``False``) recycles the routed L1 activation pool
  through an automatically sized ring while routing metadata stays at
  worst-case size.

.. warning::
   A ``prefill_pool_capacity_ratio`` below ``1.0`` is an unsafe capacity hint.
   If the actual sum of 256-row-aligned local expert counts exceeds the reduced
   capacity, kernels may access outside the allocation. Keep the default
   ``1.0`` unless you have verified the routing distribution.

Configuring a pool ratio below ``1.0`` together with ``prefill_l1_ring=True``
raises ``ValueError``.

Token alignment
---------------

``get_token_alignment_for_mega_moe()`` returns ``256``. Round token counts up
to this alignment before sizing caller-side buffers.

Current Limits
--------------

- Intranode process groups only, with up to ``8`` ranks.
- ``num_experts`` must be divisible by the process-group size.
- ``activation`` must be ``"swiglu"`` or ``"situ"``.
- ``use_fp8_dispatch=True`` only.
- ``hidden`` must be divisible by ``128``.
- ``intermediate_hidden`` must be divisible by ``128`` and no greater than
  ``8192``.
- The symmetric-buffer token capacity is rounded up to a multiple of ``256``.
- ``activation_clamp`` is not supported by the split-prefill path.
- ``fp8_fp8_mega_moe`` writes into a contiguous ``torch.bfloat16`` output
  tensor.

API
---

.. py:function:: mate.deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, use_fp8_dispatch=True, activation="swiglu", prefill_pool_capacity_ratio=1.0, prefill_l1_ring=False)

   Allocate the shared runtime workspace used by Mega MoE execution.

   ``prefill_pool_capacity_ratio`` may reduce the 256-row-aligned Prefill L1
   and L2 activation pools relative to their worst-case capacity. The default
   ``1.0`` is safe for every valid routing distribution. Values below ``1.0``
   are an unsafe capacity hint: if the actual sum of per-expert padded rows
   exceeds the configured pool, device memory may be overwritten.

   ``prefill_l1_ring`` recycles the routed Prefill L1 activation pool through
   an automatically sized ring. Routing metadata stays at the worst-case size
   and results are unchanged. It is off by default because it cannot be
   combined with a ``prefill_pool_capacity_ratio`` below ``1.0``.

.. py:function:: mate.deep_gemm.get_token_alignment_for_mega_moe()

   Return the split-prefill token-capacity alignment, currently ``256``.

.. py:function:: mate.deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights=None)

   Transform grouped FP8 expert weights into the layout expected by the Mega
   MoE runtime.

.. py:function:: mate.deep_gemm.fp8_fp8_mega_moe(y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu", activation_clamp=None, fast_math=True, activation_beta=None, activation_linear_beta=None)

   Run the FP8 Mega MoE expert compute path and write the result to ``y``.
   Workloads of up to ``128`` tokens use the decode kernel; larger workloads
   run serial split-prefill Stage 1, Stage 2, and Stage 3.
