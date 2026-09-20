.. _apimamba:

Mamba / SSU
===========

.. currentmodule:: mate.mamba

Mamba2 selective state update (SSU) on MUSA. ``selective_state_update`` applies
decode steps of the SSM recurrence, and is the MUSA-native implementation behind
the FlashInfer-shaped ``flashinfer.mamba.selective_state_update`` compatibility
surface. Two TileLang kernels share the entry point: a one-token kernel for the
plain decode step (one token per sequence, no slot table) and a packed kernel for
everything else -- packed variable-length rows, multi-token (MTP) decoding with
``num_accepted_tokens``, and the dense 4-D multi-token form.

Minimal SSU example:

.. code-block:: python

   import torch
   from mate.mamba import selective_state_update

   batch, heads, dim, dstate, groups, slots = 2, 8, 8, 128, 2, 8
   state = torch.zeros(slots, heads, dim, dstate, device="musa", dtype=torch.float16)
   x = torch.randn(batch, heads, dim, device="musa", dtype=torch.bfloat16)
   dt = torch.rand(batch, heads, dim, device="musa", dtype=torch.float32) * 0.1
   A = -torch.rand(heads, device="musa", dtype=torch.float32)
   B = torch.randn(batch, groups, dstate, device="musa", dtype=torch.bfloat16)
   C = torch.randn(batch, groups, dstate, device="musa", dtype=torch.bfloat16)
   slots_index = torch.tensor([0, 1], device="musa", dtype=torch.int32)

   y = selective_state_update(state, x, dt, A, B, C, None)
   y = selective_state_update(
       state,
       x,
       dt,
       A,
       B,
       C,
       None,
       state_batch_indices=slots_index,
       dst_state_batch_indices=slots_index,
   )

Speculative decoding (MTP) example, ``batch`` sequences of ``steps`` tokens:

.. code-block:: python

   steps = 7
   rows = batch * steps
   x = torch.randn(rows, heads, dim, device="musa", dtype=torch.bfloat16)
   # One state slot per (sequence, speculative position) on the destination side,
   # and the position the accepted token stopped at on the read side.
   read_slots = torch.arange(rows, device="musa", dtype=torch.int32).view(batch, steps)
   write_slots = (read_slots + rows).contiguous()
   cu_seqlens = torch.arange(0, rows + 1, steps, device="musa", dtype=torch.int32)
   accepted = torch.full((batch,), steps, device="musa", dtype=torch.int32)

   y = selective_state_update(
       state,
       x,
       dt,
       A,
       B,
       C,
       None,
       state_batch_indices=read_slots,
       dst_state_batch_indices=write_slots,
       num_accepted_tokens=accepted,
       cu_seqlens=cu_seqlens,
       pad_slot_id=0,
   )

vLLM's MTP6 decode passes the slot tables **flat** instead: one entry per packed
row, which the Triton kernel addresses by unsqueezing to ``[rows, 1]``. Both
forms are accepted, and with one sequence per call they name the same entries:

.. code-block:: python

   rows = 7
   slots = torch.arange(rows, device="musa", dtype=torch.int32)
   y = selective_state_update(
       state,
       x,
       dt,
       A,
       B,
       C,
       None,
       state_batch_indices=slots,
       dst_state_batch_indices=slots,
       num_accepted_tokens=torch.tensor([1], device="musa", dtype=torch.int32),
       cu_seqlens=torch.tensor([0, rows], device="musa", dtype=torch.int32),
       pad_slot_id=0,
   )

SSU at a glance
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - Supported scope
   * - Device
     - MUSA through the TileLang backend
   * - Sequence layout
     - One token per sequence (``x``, ``dt`` and ``z`` are ``[batch, heads, dim]``)
       on the one-token path; the packed ``[rows, heads, dim]`` layout with
       ``cu_seqlens``, or the dense ``[batch, steps, heads, dim]`` form, on the
       packed path
   * - State
     - ``[slots, heads, dim, dstate]``, contiguous, fp16/bf16/fp32, updated in
       place through the destination slots
   * - ``A`` / ``D`` / ``dt_bias``
     - per-head fp32, broadcasting over ``dim``/``dstate``
   * - ``B`` / ``C``
     - ``[batch, groups, dstate]`` with the ``x`` dtype; ``heads`` must be
       divisible by ``groups``
   * - Output
     - ``[batch, heads, dim]`` in the ``x`` dtype, optionally written into a
       caller-provided ``out``
   * - ``dt_softplus``
     - supported
   * - Slot semantics
     - ``state_batch_indices`` and ``dst_state_batch_indices`` may be the flat
       ``[rows]`` form or the 2-D ``[sequences, steps]`` table; both are addressed
       by strides, so a flat table puts sequence ``b``'s position ``t`` at entry
       ``b + t`` and a contiguous 2-D table at ``b * steps + t``. ``pad_slot_id``
       reads as a zero state and is never written; ``disable_state_update`` leaves
       the pool unchanged
   * - Variable-length and MTP decode
     - supported: ``cu_seqlens`` splits the packed rows between sequences and
       ``num_accepted_tokens`` seeds each sequence's read slot at ``count - 1``
       (floored at 0). Every token then publishes its state to its own destination
       entry — or to its own entry of the read table when no destination table is
       given, the default the Triton kernel applies. ``pad_slot_id`` destinations
       are never written; empty sequences are skipped
   * - Stochastic rounding
     - not implemented; ``rand_seed`` raises ``NotImplementedError``

SSU toolchain requirements
--------------------------

- TileLang for MUSA (``tilelang-musa``) is required; the kernel is compiled at
  run time for the shape and dtype combination it is first called with.
- The recurrence runs in fp32 with the state updated in the state dtype.

Graph capture
-------------

Call ``prewarm_selective_state_update`` during warmup, before any graph capture:
the TileLang kernel is compiled on first use, and a first-call compile inside a
captured graph stalls the worker. The packed path issues a single kernel launch and
performs no host/device scalar copies; the one-token path makes one comparison
against ``cu_seqlens`` (the check that decides which kernel a call takes, and the
only synchronization in the entry point).

SSD packed prefill
-------------------

``ssd_combined_fwd_varlen`` is the prefill-side counterpart of the decode
entrypoint: one packed call over a variable-length batch, running the five SSD
stages (chunk-local dt/decay cumsum, intra-chunk states, state passing, the
``C·Bᵀ`` products, and the chunk scan) as five TileLang launches. It is the
symbol the FlashInfer-shaped compatibility wrapper dispatches to for
``--mamba-backend flashinfer`` on MUSA, and it drives the same implementation for
every Mamba2/Nemotron-H prefill shape.

The per-shape intermediates come from a cached workspace, so a steady-state call
allocates nothing beyond the returned state tensor. ``cu_chunk_seqlens`` is the
only authority for the chunk-to-token mapping, and ``seq_idx`` -- not the
``last_chunk_indices`` arithmetic -- decides which chunks open a sequence and
therefore where their entering state comes from.

Everything outside the supported contract raises ``NotImplementedError`` or
``ValueError`` naming the missing capability rather than silently narrowing
semantics: the ``checkpoint_*`` replay arguments are not implemented.

API reference
-------------

.. autofunction:: selective_state_update

.. autofunction:: prewarm_selective_state_update

.. autofunction:: ssd_combined_fwd_varlen

.. autofunction:: prewarm_ssd_combined_fwd_varlen
