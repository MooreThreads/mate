.. _apikda:

KDA
===

.. currentmodule:: mate.kda

For framework integrations that already target FlashKDA Python APIs, prefer
the ``flash_kda`` wrapper package first. Use the MATE API below when wrapper
coverage is not enough or when direct operator-level control is required.

KDA (Kimi Delta Attention) covers MATE's direct operator APIs for chunked KDA
and KDA decode on MUSA. Use this page when the ``flash_kda`` wrapper is not
enough, or when you need direct control over ``mate.kda`` entrypoints such as
``chunk_kda`` and ``gated_delta_rule_decode``.

KDA improves performance and adds state-pool mode to decode.

``chunk_kda`` is MATE's fused chunked KDA operator on MUSA, while
``gated_delta_rule_decode`` provides the direct KDA decode API.

Minimal KDA example:

.. code-block:: python

   import torch
   from mate.kda import chunk_kda

   q = torch.randn((1, 4, 2, 128), device="musa", dtype=torch.bfloat16)
   k = torch.randn((1, 4, 2, 128), device="musa", dtype=torch.bfloat16)
   v = torch.randn((1, 4, 4, 128), device="musa", dtype=torch.bfloat16)
   g = torch.randn((1, 4, 4, 128), device="musa", dtype=torch.bfloat16)
   beta = torch.randn((1, 4, 4), device="musa", dtype=torch.bfloat16)

   output = chunk_kda(q=q, k=k, v=v, g=g, beta=beta)

Chunk KDA at a glance
---------------------

- Public API: ``mate.chunk_kda`` / ``mate.kda.chunk_kda``
- Device: MUSA
- Input dtypes: ``torch.float16`` and ``torch.bfloat16``
- Head dimension: currently fixed to ``128``
- Sequence modes:
  - Dense: ``[B, T, H, 128]``
  - Varlen: ``[S, H, 128]`` or ``[1, S, H, 128]`` with ``cu_seqlens``
- Optional recurrent state input/output
- Optional preallocated ``output`` and ``final_state``
- Optional per-chunk ``chunk_states`` pool sized with
  ``mate.kda.kda_chunk_state_rows``
- Optional state checkpoints via ``checkpoint_every_n_tokens=C`` on MP31 and
  MP32

Chunk KDA toolchain requirements
--------------------------------

- The repository-wide install baseline still applies, but build the current
  fused chunk KDA path with MUSA SDK / MTCC 5.1.0 or newer.
- The 4.3.6 toolchain may fail to compile KDA kernels.

Chunk KDA shape contract
------------------------

For dense mode:

- ``q`` / ``k``: ``[B, T, Hqk, 128]``
- ``v`` / ``g``: ``[B, T, Hv, 128]``
- ``beta``: ``[B, T, Hv]``

For varlen mode:

- ``q`` / ``k``: ``[S, Hqk, 128]`` or ``[1, S, Hqk, 128]``
- ``v`` / ``g``: ``[S, Hv, 128]`` or ``[1, S, Hv, 128]``
- ``beta``: ``[S, Hv]`` or ``[1, S, Hv]``
- ``cu_seqlens``: cumulative sequence lengths with shape ``[num_seqs + 1]``

Additional constraints:

- ``k.shape == q.shape``
- ``g.shape == v.shape``
- ``beta.shape == v.shape[:3]``
- ``Hv`` must be divisible by ``Hqk``
- the last dimension of every tensor passed to the kernel must be contiguous

Chunk KDA state tensors
-----------------------

- ``initial_state`` is optional
- ``final_state`` is optional unless ``output_final_state=True``
- state shape: ``[num_seqs, Hv, 128, 128]``
- state dtype: same as value dtype or ``torch.float32``
- if both ``initial_state`` and ``final_state`` are provided, their dtypes must match

Chunk states pool
-----------------

``chunk_kda`` runs as a chunk-parallel prepare kernel followed by a recurrence
kernel. ``chunk_states`` is an optional caller-allocated pool that receives the
recurrent state at every chunk boundary:

- shape ``[chunk_rows, H, Dv, D]`` and contiguous, with the same dtype as the
  recurrent state
- size it with ``mate.kda.kda_chunk_state_rows``; over-provisioning is fine
  because the kernel writes only the rows it needs
- row ``workspace_slot + chunk_idx`` of a sequence receives the state entering
  that chunk, so row 0 is the state before the first token
- the state after the final chunk is reported only through ``final_state`` or
  the ``initial_state`` pool

``chunk_states`` cannot be combined with ``checkpoint_every_n_tokens`` because
both write chunk-boundary states with different row mappings.

State-pool mode
---------------

Both KDA entrypoints support reusable state pools, but they use different
parameter names:

- ``chunk_kda`` uses ``initial_state`` with
  ``initial_state_indices``. The index tensor has shape ``[num_seqs]`` and
  selects the pool row used by each input sequence. Selected rows are updated
  in place.
- ``gated_delta_rule_decode`` uses ``state`` with ``state_indices``. The index
  tensor has shape ``[B]`` or ``[B, T]`` and selects the pool row for each
  batch entry and decode step.

Chunk KDA gate parameters
-------------------------

When gate parameters are enabled:

- ``A_log`` and ``dt_bias`` must be provided together
- ``A_log`` shape: ``[Hv]``
- ``dt_bias`` shape: ``[Hv, 128]``
- ``lower_bound`` defaults to ``-5.0``

Chunk KDA outputs
-----------------

- default return: ``output``
- if ``output_final_state=True``: returns ``(output, final_state)``
- if ``checkpoint_every_n_tokens > 0``: returns
  ``(output, final_state, checkpoints)``, where ``final_state`` is a zero-size
  tensor if it was not requested

Chunk KDA state checkpoints
---------------------------

``checkpoint_every_n_tokens=C`` makes ``chunk_kda`` return the recurrent state
every ``C`` tokens instead of only at the end of a sequence, so a serving
framework can cache prefix states and restore one later. The operator allocates
the pool itself and returns it as the third element of the result; the layout
matches the cuDNN frontend GDN ``state_checkpoints`` output.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Item
     - Supported scope
   * - Architecture
     - MP31 and MP32
   * - Cadence
     - ``C`` is ``0`` (disabled) or a positive multiple of the kernel chunk,
       ``16``
   * - Checkpoint meaning
     - Entry ``j`` of sequence ``b`` holds the state entering token ``j * C``.
       Entry ``0`` is the incoming state; the state after the final token is
       ``final_state``, never a checkpoint
   * - Packing
     - Entries are packed per sequence in ``cu_seqlens`` order, so the valid
       rows are ``sum_b ((len_b - 1) // C + 1)`` and empty sequences contribute
       none
   * - Pool shape
     - Allocated by the operator as ``[capacity, Hv, 128, 128]`` with
       ``capacity = max(total_tokens // C + num_seqs, 1)``
   * - Pool dtype
     - The activation dtype (``torch.float16`` or ``torch.bfloat16``), even
       when the recurrent state is ``torch.float32``
   * - Not combinable with
     - ``chunk_states`` and ``initial_state_indices``

Decode support
--------------

The decode entry point is ``mate.kda.gated_delta_rule_decode``. It remains
module-scoped because ``mate.gated_delta_rule_decode`` is the separate GDN API.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - Supported KDA decode scope
   * - Device
     - MUSA through the TileLang backend
   * - Sequence layout
     - Dense ``[B, T, H, D]`` or varlen ``[1, total_tokens, H, D]`` with
       ``cu_seqlens``
   * - Q/K/V dtypes
     - Matching ``torch.float16`` or ``torch.bfloat16`` tensors
   * - Head dimensions
     - ``K == V`` and the dimension must be divisible by ``32``
   * - Head grouping
     - The number of value/state heads must be divisible by the number of Q/K
       heads
   * - State
     - Optional ``torch.float32`` or ``torch.bfloat16`` state in V-first
       ``[pool, HV, V, K]`` or K-first ``[pool, HV, K, V]`` layout
   * - Output dtype
     - ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``
   * - Serving modes
     - Fixed or variable-length decode, continuous batching through
       ``state_indices``, and speculative decode through
       ``num_accepted_tokens``
   * - State output
     - Separate final-state output or in-place state update

Decode inputs use ``a`` with shape ``[B, T, HV, K]`` and ``b`` with shape
``[B, T, HV]`` or ``[B, T, HV, V]``. ``A_log`` is optional float32 metadata
with shape ``[HV]``; ``dt_bias`` is optional float32 or bfloat16 data with
shape ``[HV, K]``.

API reference
-------------

.. autofunction:: chunk_kda

.. autofunction:: gated_delta_rule_decode
