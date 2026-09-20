"""PyTorch oracle for the native mamba/SSU family.

The kernel tests compare against these functions, and the family falls back to
them only where documented: they are the definition of correct behaviour, not a
shipping fast path. The recurrence arithmetic is fp32 with the same operand order
as the kernels, so a mismatch points at the kernel rather than at rounding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "selective_state_update_multi_token_reference",
    "selective_state_update_one_token_reference",
    "ssd_bmm_reference",
    "ssd_chunk_cumsum_reference",
    "ssd_chunk_scan_reference",
    "ssd_chunk_state_reference",
    "ssd_state_passing_reference",
]

_SOFTPLUS_THRESHOLD = 20.0


def _dot_dtype(a_dtype: torch.dtype, b_dtype: torch.dtype) -> torch.dtype:
    """The SSD dot's activation dtype: 16-bit wins, fp32 otherwise."""
    if torch.bfloat16 in (a_dtype, b_dtype):
        return torch.bfloat16
    if torch.float16 in (a_dtype, b_dtype):
        return torch.float16
    return torch.float32


def selective_state_update_one_token_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    pad_slot_id: int,
    disable_state_update: bool,
) -> torch.Tensor:
    """Reference one-token selective state update.

    Per-head ``A``, ``D`` and ``dt_bias`` broadcast over the state dimensions.
    ``state`` is updated in place through the destination slots; the returned
    tensor is the per-token output ``y``.
    """
    slots, heads, dim, dstate = state.shape
    batch = x.shape[0]
    groups = B.shape[1]
    head_ratio = heads // groups

    dt_value = dt.to(torch.float32)
    if dt_bias is not None:
        dt_value = dt_value + dt_bias.to(torch.float32).view(1, heads, 1)
    if dt_softplus:
        dt_value = torch.where(
            dt_value > _SOFTPLUS_THRESHOLD,
            dt_value,
            torch.log1p(torch.exp(dt_value)),
        )

    y = torch.empty((batch, heads, dim), dtype=torch.float32, device=x.device)
    dst_slots = src_slots if dst_slots is None else dst_slots
    head_groups = torch.arange(heads, device=x.device) // head_ratio
    a_head = A.to(torch.float32).view(heads, 1, 1)

    for b in range(batch):
        src = int(src_slots[b])
        dst = int(dst_slots[b])
        old = (
            torch.zeros((heads, dim, dstate), dtype=torch.float32, device=x.device)
            if src == pad_slot_id
            else state[src].to(torch.float32)
        )
        xb = x[b].to(torch.float32)
        dtb = dt_value[b]
        decay = torch.exp(a_head * dtb[:, :, None])
        # Same operand order as the kernel: (dt * B) * x.
        update = (
            dtb[:, :, None] * B[b].to(torch.float32)[head_groups][:, None, :]
        ) * xb[:, :, None]
        new = old * decay + update
        y[b] = (new * C[b].to(torch.float32)[head_groups][:, None, :]).sum(dim=-1)
        if D is not None:
            y[b] = y[b] + D.to(torch.float32).view(heads, 1) * xb
        if z is not None:
            zb = z[b].to(torch.float32)
            y[b] = y[b] * zb * torch.sigmoid(zb)
        if not disable_state_update and dst != pad_slot_id:
            state[dst].copy_(new.to(state.dtype))

    return y.to(x.dtype)


def _slot_at(slots: torch.Tensor, batch_idx: int, step: int) -> int:
    """Read a slot id, addressed the way the kernels address it.

    A 2-D ``[sequences, steps]`` table is read at ``[batch_idx, step]``. A 1-D table
    is the flat form the Triton kernel reaches by unsqueezing it to ``[rows, 1]``:
    both of its strides are then 1, so sequence ``batch_idx``'s position ``step`` is
    entry ``batch_idx + step``. That is what vLLM's MTP6 call passes -- one entry per
    packed row -- and with a single sequence it is simply the position.
    """
    if slots.dim() == 1:
        return int(slots[batch_idx + step])
    return int(slots[batch_idx, step])


def selective_state_update_multi_token_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor | None,
    pad_slot_id: int,
    disable_state_update: bool,
    steps: Sequence[int],
    row_base: Sequence[int],
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference selective state update over several tokens per sequence.

    Covers both packed variable-length decode (``x`` is ``[total_tokens, heads,
    dim]`` and ``row_base[b]`` is ``cu_seqlens[b]``) and MTP decode (``x`` is the
    flattened ``[batch * steps, heads, dim]`` view of ``[batch, steps, heads,
    dim]`` with ``row_base[b] == b * steps``). The caller derives ``steps`` and
    ``row_base``; the oracle itself only needs the flattened layout.

    The state stays in registers across a sequence's tokens (a single read at the
    start), and the three write regimes follow the contract the consumers rely on:

    - plain packed varlen: one write after the sequence's token loop, to
      ``dst_slots[b, 0]`` (falling back to the read slot);
    - plain MTP without destination slots: one write after the token loop, back
      to the read slot;
    - speculative decoding (``num_accepted_tokens`` given): one write per token to
      ``dst_slots[b, t]``, or to ``src_slots[b, t]`` when no destination table is
      given, which is the default the Triton kernel applies by aliasing
      ``dst_state_batch_indices`` onto ``state_batch_indices``.

    A padded source slot reads as a zero state, a padded destination slot is never
    written, and ``disable_state_update`` suppresses every write. Empty sequences
    (``steps[b] == 0``) are skipped entirely, with no read and no write.

    One deliberate difference from the fork's reference: plain MTP that carries
    destination slots without ``num_accepted_tokens`` still writes its final state
    once, to ``dst_slots[b, 0]``, where the fork's ``elif`` chain happens to skip
    that write. No consumer configuration reaches that combination (the serving
    path passes destination slots only together with ``num_accepted_tokens``, which
    takes the slot-chain regime above).
    """
    heads, dim, dstate = state.shape[1:]
    groups = B.shape[1]
    head_ratio = heads // groups

    dt_value = dt.to(torch.float32)
    if dt_bias is not None:
        dt_value = dt_value + dt_bias.to(torch.float32).view(1, heads, 1)
    if dt_softplus:
        dt_value = torch.where(
            dt_value > _SOFTPLUS_THRESHOLD,
            dt_value,
            torch.log1p(torch.exp(dt_value)),
        )

    batch = len(steps)
    total_rows = x.shape[0]
    y = torch.empty((total_rows, heads, dim), dtype=torch.float32, device=x.device)
    head_groups = torch.arange(heads, device=x.device) // head_ratio
    a_head = A.to(torch.float32).view(heads, 1, 1)
    spec_decoding = num_accepted_tokens is not None

    for b in range(batch):
        n_steps = int(steps[b])
        if n_steps == 0:
            continue
        first_row = int(row_base[b])
        accepted = max(int(num_accepted_tokens[b]) - 1, 0) if spec_decoding else 0
        read_slot = _slot_at(src_slots, b, accepted)
        running = (
            torch.zeros((heads, dim, dstate), dtype=torch.float32, device=x.device)
            if read_slot == pad_slot_id
            else state[read_slot].to(torch.float32)
        )

        for token in range(n_steps):
            row = first_row + token
            xb = x[row].to(torch.float32)
            dtb = dt_value[row]
            decay = torch.exp(a_head * dtb[:, :, None])
            b_h = B[row].to(torch.float32)[head_groups]
            # Same operand order as the kernel: (dt * B) * x.
            running = (
                running * decay + (dtb[:, :, None] * b_h[:, None, :]) * xb[:, :, None]
            )
            c_h = C[row].to(torch.float32)[head_groups]
            y[row] = (running * c_h[:, None, :]).sum(dim=-1)
            if D is not None:
                y[row] = y[row] + D.to(torch.float32).view(heads, 1) * xb
            if z is not None:
                zb = z[row].to(torch.float32)
                y[row] = y[row] * zb * torch.sigmoid(zb)

            if disable_state_update:
                continue
            if spec_decoding:
                # No destination table: the write target is the read table's own
                # entry for this token, not the slot the sequence started from.
                write_slot = _slot_at(
                    src_slots if dst_slots is None else dst_slots, b, token
                )
                if write_slot != pad_slot_id:
                    state[write_slot].copy_(running.to(state.dtype))

        if disable_state_update or spec_decoding:
            continue
        if dst_slots is not None:
            final_slot = _slot_at(dst_slots, b, 0)
        else:
            # Plain MTP without destination slots updates the read slot in place.
            final_slot = read_slot
        if final_slot != pad_slot_id:
            state[final_slot].copy_(running.to(state.dtype))

    return y.to(x.dtype)


def ssd_chunk_cumsum_reference(
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor | None,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    dt_softplus: bool,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference SSD chunk-local cumsum.

    Returns ``(dA_cumsum, dt_out)``, both head-major
    ``[heads, nchunks, chunk_size]`` fp32, where ``dt_out`` is the processed
    ``dt`` (bias, softplus, clamp) and ``dA_cumsum`` is the *inclusive* prefix
    sum of ``dt_out * A`` over each chunk's tokens.

    Padding is part of the contract, not slack space:

    * ``dt_out`` is **exactly zero** beyond a chunk's end, so the scan saturates;
    * consequently ``dA_cumsum`` beyond a chunk's end holds the chunk's **total**
      decay, and downstream stages read the padded row's last position
      ``dA_cumsum[:, c, chunk_size - 1]`` unconditionally to obtain it.

    Chunks are scanned independently and never across a sequence boundary.
    """
    if dt.dim() != 2:
        raise ValueError("dt must be [tokens, heads].")
    heads = dt.shape[1]
    offsets = [int(x) for x in cu_chunk_seqlens.reshape(-1).tolist()]
    nchunks = len(offsets) - 1
    if nchunks < 1:
        raise ValueError("cu_chunk_seqlens must have at least two entries.")

    dt_min, dt_max = dt_limit
    dA_cumsum = torch.zeros(
        (heads, nchunks, chunk_size), dtype=torch.float32, device=dt.device
    )
    dt_out = torch.zeros_like(dA_cumsum)

    a_value = A.to(torch.float32)
    bias = None if dt_bias is None else dt_bias.to(torch.float32)

    for chunk in range(nchunks):
        lo, hi = offsets[chunk], offsets[chunk + 1]
        length = hi - lo
        if length < 0 or hi > dt.shape[0]:
            raise ValueError(
                f"chunk {chunk} spans [{lo}, {hi}) outside 0..{dt.shape[0]}."
            )
        if length == 0:
            continue
        segment = dt[lo:hi].to(torch.float32)
        value = segment if bias is None else segment + bias.view(1, heads)
        if dt_softplus:
            value = torch.where(
                value > _SOFTPLUS_THRESHOLD,
                value,
                torch.log1p(torch.exp(value)),
            )
        value = value.clamp(min=dt_min, max=dt_max)
        # Zero-pad the chunk's dt to the full row before scanning, so the prefix
        # sum saturates at the chunk total in the tail -- the value downstream
        # stages read from the row's last position.
        padded = torch.zeros(
            (chunk_size, heads), dtype=torch.float32, device=dt.device
        )
        padded[:length] = value
        prefix = torch.cumsum(padded * a_value.view(1, heads), dim=0)
        dA_cumsum[:, chunk, :] = prefix.transpose(0, 1)
        dt_out[:, chunk, :] = padded.transpose(0, 1)

    return dA_cumsum, dt_out


def ssd_chunk_state_reference(
    x: torch.Tensor,
    B: torch.Tensor,
    dt_out: torch.Tensor,
    dA_cumsum: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    round_operands: bool = True,
) -> torch.Tensor:
    """Reference intra-chunk states ``S_c[h, d, n]`` for the SSD prefill.

    Shapes: ``x`` is ``[tokens, heads, dim]``, ``B`` is ``[tokens, groups,
    dstate]``, and ``dt_out``/``dA_cumsum`` are the head-major ``[heads, nchunks,
    chunk_size]`` outputs of the cumsum stage. The result is ``[nchunks, heads,
    dim, dstate]`` fp32.

    ``round_operands=False`` disables the pre-dot rounding: the device path has no
    such choice, but turning it off is how the cost of that convention gets
    measured against an fp32 reference instead of asserted.

    The arithmetic mirrors the device path exactly, including its two rounding
    points: ``scale = exp(min(last - dA_cumsum[t], 0)) * dt`` stays fp32, and the
    scaled ``B`` is rounded to the activation dtype **before** the dot, which is
    accumulated in fp32.

    Padding carries the contract: because ``dt_out`` is zero past a chunk's end
    and ``dA_cumsum`` saturates there, the decay factor for those positions is
    zero and the masked-out tail contributes nothing.
    """
    if x.dim() != 3 or B.dim() != 3:
        raise ValueError("x and B must be [tokens, heads, dim] / [tokens, groups, dstate].")
    tokens, heads, dim = x.shape
    if B.shape[0] != tokens:
        raise ValueError("x and B must share the token axis.")
    groups, dstate = B.shape[1], B.shape[2]
    if heads % groups:
        raise ValueError("heads must be divisible by groups.")
    head_ratio = heads // groups
    _, nchunks, block = dA_cumsum.shape
    if dt_out.shape != dA_cumsum.shape:
        raise ValueError("dt_out and dA_cumsum must share a shape.")
    if dA_cumsum.shape[0] != heads:
        raise ValueError("dA_cumsum's head axis must match x.")

    offsets = [int(v) for v in cu_chunk_seqlens.reshape(-1).tolist()]
    if len(offsets) - 1 != nchunks:
        raise ValueError("cu_chunk_seqlens must have nchunks + 1 entries.")
    if block != chunk_size:
        raise ValueError("dA_cumsum's last axis must be chunk_size.")

    x_fp32 = x.to(torch.float32)
    b_fp32 = B.to(torch.float32)
    states = torch.zeros(
        (nchunks, heads, dim, dstate), dtype=torch.float32, device=x.device
    )

    for chunk in range(nchunks):
        start, end = offsets[chunk], offsets[chunk + 1]
        limit = end - start
        if limit <= 0:
            continue
        # The chunk's total decay lives in the padded row's last position.
        last = dA_cumsum[:, chunk, block - 1].to(torch.float32)
        delta = (last.unsqueeze(1) - dA_cumsum[:, chunk, :limit].to(torch.float32))
        delta = delta.clamp(max=0.0)
        scale = torch.exp(delta) * dt_out[:, chunk, :limit].to(torch.float32)
        x_chunk = x_fp32[start:end]
        for group in range(groups):
            head_slice = slice(group * head_ratio, (group + 1) * head_ratio)
            scaled = (
                b_fp32[start:end, group, :].unsqueeze(0)
                * scale[head_slice].unsqueeze(2)
            )
            if round_operands:
                scaled = scaled.to(x.dtype)
            # Contract the token axis exactly once: the scaled operand is
            # [head, token, dstate] and pairs token t with token t, not every
            # (t, l) pair.
            states[chunk, head_slice] = torch.einsum(
                "thd,htn->hdn", x_chunk[:, head_slice, :], scaled.to(torch.float32)
            )
    return states


def ssd_chunk_scan_reference(
    x: torch.Tensor,
    C: torch.Tensor,
    B: torch.Tensor,
    dt_out: torch.Tensor,
    dA_cumsum: torch.Tensor,
    states: torch.Tensor,
    initial_states: torch.Tensor | None,
    seq_idx: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    D_param: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    CB: torch.Tensor | None = None,
    round_operands: bool = True,
) -> torch.Tensor:
    """Reference SSD chunk scan.

    With ``CB`` unset the ``C·Bᵀ`` products are recomputed from ``B``, so this
    doubles as an independent check of the BMM stage; passing ``CB`` exercises the
    materialized products the device kernel consumes instead.
    ``out`` is written in place when supplied and returned; positions at or past a
    chunk's logical end are never touched, and rows opened by a sequence take
    their entering state from ``initial_states[seq_idx[chunk]]``.

    Rounding points mirror the device path: the past-state product accumulates in
    fp32, the decay ``exp(dA_cs[t])`` scales that term *after* the dot, and the
    scaled ``CB`` is rounded to the activation dtype **before** the diagonal dot.
    ``round_operands=False`` disables that rounding, which is how its cost is
    measured rather than asserted.
    """
    tokens, heads, dim = x.shape
    groups, dstate = C.shape[1], C.shape[2]
    head_ratio = heads // groups
    _, nchunks, block = dA_cumsum.shape
    if block != chunk_size:
        raise ValueError("dA_cumsum's last axis must be chunk_size.")
    offsets = [int(v) for v in cu_chunk_seqlens.reshape(-1).tolist()]
    if len(offsets) - 1 != nchunks:
        raise ValueError("cu_chunk_seqlens must have nchunks + 1 entries.")
    if out is None:
        out = torch.zeros_like(x)
    if not out.is_contiguous():
        raise ValueError("out must be contiguous to be filled in place.")

    seq = [int(v) for v in seq_idx.reshape(-1).tolist()]
    c_fp32 = C.to(torch.float32)
    b_fp32 = B.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    dA = dA_cumsum.to(torch.float32)
    dt = dt_out.to(torch.float32)
    st = states.to(torch.float32)

    for chunk in range(nchunks):
        start, end = offsets[chunk], offsets[chunk + 1]
        limit = end - start
        if limit <= 0:
            continue
        opens = chunk == 0 or seq[chunk] != seq[chunk - 1]
        if opens and initial_states is not None:
            prev = initial_states[seq[chunk]].to(torch.float32)
        elif opens:
            prev = torch.zeros((heads, dim, dstate), dtype=torch.float32, device=x.device)
        else:
            prev = st[chunk - 1]
        for head in range(heads):
            group = head // head_ratio
            dA_head = dA[head, chunk, :limit]
            dt_head = dt[head, chunk, :limit]
            c_rows = c_fp32[start:end, group, :]  # (limit, dstate)
            # past-state term, then the decay applied after the dot. The state
            # is [dim, dstate] and is transposed for the product, exactly as the
            # kernel's load transposes it into (dstate, dim).
            acc = c_rows @ prev[head].transpose(0, 1)  # (limit, dim)
            acc = acc * torch.exp(dA_head).unsqueeze(1)
            # diagonal term over the causal triangle
            decay = torch.exp(
                torch.clamp(
                    dA_head.unsqueeze(1) - dA_head.unsqueeze(0), max=0.0
                )
            )
            if CB is None:
                b_rows = b_fp32[start:end, group, :]  # (limit, dstate)
                cb = torch.einsum("in,jn->ij", c_rows, b_rows)  # (limit, limit)
            else:
                cb = CB[chunk, group, :limit, :limit].to(torch.float32)
            cb = cb * decay * dt_head.unsqueeze(0)
            causal = torch.tril(torch.ones(limit, limit, device=x.device, dtype=torch.bool))
            cb = torch.where(causal, cb, torch.zeros_like(cb))
            if round_operands:
                cb = cb.to(x.dtype)
            cb = cb.to(torch.float32)
            acc = acc + cb @ x_fp32[start:end, head, :]
            if D_param is not None:
                acc = acc + x_fp32[start:end, head, :] * D_param[head].to(torch.float32)
            if z is not None:
                z_head = z[start:end, head, :].to(torch.float32)
                acc = acc * z_head * torch.sigmoid(z_head)
            out[start:end, head, :] = acc.to(out.dtype)
    return out


def ssd_state_passing_reference(
    states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    initial_states: torch.Tensor | None = None,
    state_dtype: torch.dtype | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference inter-chunk state passing for the SSD prefill.

    Shapes: ``states`` is ``[nchunks, heads, dim, dstate]``, ``dA_cumsum`` is the
    head-major ``[heads, nchunks, chunk_size]`` output of the cumsum stage and
    ``last_chunk_indices`` is ``[batch]``. The result is ``[nchunks, heads, dim,
    dstate]`` in ``state_dtype``.

    ``out[c]`` is the state **entering chunk ``c+1``**: the running state after
    chunk ``c`` has been applied:
    ``out[c] = exp(dA_cumsum[:, c, L-1]) * running + states[c]``.

    Boundaries come from ``seq_idx`` when it is given -- the same rule the device
    oracle uses: chunks are visited in index order, a sequence's **first** chunk
    enters from ``initial_states[seq_idx[c]]`` and every later chunk of the same
    ``seq_idx`` continues that sequence's running state. Physical chunks that the
    caller assigned to other sequences in between are skipped without decaying
    it, so ``initial_states`` is indexed by the value of ``seq_idx``, not by
    sequence position.

    Without ``seq_idx`` the boundaries are ``last_chunk_indices`` arithmetic
    instead: sequence ``b`` owns chunks ``(last_chunk_indices[b-1],
    last_chunk_indices[b]]`` -- so its first chunk is ``last_chunk_indices[b-1] +
    1`` and its last is ``last_chunk_indices[b]``, with ``b == 0`` starting at
    chunk 0 -- and enters from ``initial_states[b]``, indexed by **sequence
    position**. A sequence with no chunks (``last_chunk_indices[b] == -1``, or an
    empty range between two neighbours) iterates zero times and so does not
    advance the running state. The two rules agree whenever the caller's metadata
    is contiguous and consistent, which is what the packed prefill path produces.

    Rounding point: ``running`` stays fp32 across every chunk of a sequence and is
    rounded to ``state_dtype`` only on the store. That matches the device kernel,
    which also rounds the state the chunk-scan stage reads back in.

    ``state_dtype`` resolves as ``state_dtype`` -> ``initial_states.dtype`` ->
    ``states.dtype``. The device orchestration's last-resort fallback is the
    activation dtype rather than the fp32 stage input, so a caller that supplies
    neither must pass ``state_dtype`` explicitly to reproduce it.

    Rows the metadata does not cover are left zero, so an empty sequence is
    observable as an unwritten range rather than as the input's contents.
    """
    if states.dim() != 4:
        raise ValueError("states must be [nchunks, heads, dim, dstate].")
    nchunks, heads, dim, dstate = states.shape
    if dA_cumsum.dim() != 3:
        raise ValueError("dA_cumsum must be [heads, nchunks, chunk_size].")
    chunk_size = dA_cumsum.shape[2]
    if dA_cumsum.shape != (heads, nchunks, chunk_size):
        raise ValueError("dA_cumsum must be [heads, nchunks, chunk_size].")
    if last_chunk_indices is not None and last_chunk_indices.dim() != 1:
        raise ValueError("last_chunk_indices must be [batch].")
    if seq_idx is not None and seq_idx.shape != (nchunks,):
        raise ValueError("seq_idx must contain one sequence id per chunk.")
    if last_chunk_indices is not None:
        batch = last_chunk_indices.shape[0]
    elif initial_states is not None:
        batch = initial_states.shape[0]
    elif seq_idx is not None:
        # The device oracle derives the sequence count this way when the caller
        # supplies only seq_idx.
        batch = max([int(value) for value in seq_idx.reshape(-1).tolist()] or [0]) + 1
    else:
        raise ValueError("pass seq_idx or last_chunk_indices for the boundaries.")
    if initial_states is not None and initial_states.shape != (
        batch,
        heads,
        dim,
        dstate,
    ):
        raise ValueError(
            "initial_states must be [batch, heads, dim, dstate] matching "
            "last_chunk_indices."
        )
    if seq_idx is not None:
        seq_idx_values = [int(value) for value in seq_idx.reshape(-1).tolist()]
        if any(value < 0 or value >= batch for value in seq_idx_values):
            raise ValueError("seq_idx values must index initial_states.")

    if state_dtype is None and initial_states is not None:
        state_dtype = initial_states.dtype
    if state_dtype is None:
        state_dtype = states.dtype

    out = torch.zeros(
        (nchunks, heads, dim, dstate), dtype=state_dtype, device=states.device
    )

    if seq_idx is not None:
        running_by_sequence: dict[int, torch.Tensor] = {}
        for chunk in range(nchunks):
            sequence = seq_idx_values[chunk]
            running = running_by_sequence.get(sequence)
            if running is None:
                running = (
                    torch.zeros(
                        (heads, dim, dstate),
                        dtype=torch.float32,
                        device=states.device,
                    )
                    if initial_states is None
                    else initial_states[sequence].to(torch.float32)
                )
            decay = torch.exp(dA_cumsum[:, chunk, chunk_size - 1].to(torch.float32))
            running = decay.view(heads, 1, 1) * running + states[chunk].to(
                torch.float32
            )
            running_by_sequence[sequence] = running
            out[chunk] = running.to(state_dtype)
        return out

    last = [int(value) for value in last_chunk_indices.reshape(-1).tolist()]

    for sequence in range(batch):
        chunk_end = last[sequence] + 1
        chunk_start = (last[sequence - 1] + 1) if sequence > 0 else 0
        if initial_states is None:
            running = torch.zeros(
                (heads, dim, dstate), dtype=torch.float32, device=states.device
            )
        else:
            running = initial_states[sequence].to(torch.float32)
        for chunk in range(chunk_start, chunk_end):
            # exp(dA_cs) at the padded row's last position, the chunk's total
            # decay. The kernel evaluates it as exp2(dA_cs * log2(e)).
            decay = torch.exp(dA_cumsum[:, chunk, chunk_size - 1].to(torch.float32))
            # Same operand order as the kernel: (decay * running) + states.
            running = decay.view(heads, 1, 1) * running + states[chunk].to(
                torch.float32
            )
            out[chunk] = running.to(state_dtype)

    return out


def ssd_bmm_reference(
    cmat: torch.Tensor,
    bmat: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    chunk_size: int,
    dot_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reference non-causal chunk-level ``C @ B^T`` for the SSD prefill.

    ``cmat``/``bmat`` are token-major ``[tokens, groups, dstate]`` and the result
    is ``[nchunks, groups, chunk_size, chunk_size]`` fp32:

        ``CB[c, g, i, j] = sum_n C[t_i, g, n] * B[t_j, g, n]``

    with ``t_i = cu_chunk_seqlens[c] + i``. Non-causal: the strict upper triangle
    is materialized as well, and the chunk-scan stage masks it afterwards.

    Rounding point: both operands are rounded to ``dot_dtype`` -- the activation
    dtype when either operand is 16-bit, fp32 otherwise -- **before** the product,
    and the reduction over ``dstate`` accumulates in fp32.

    Masking: the chunk's valid length is ``cu_chunk_seqlens[c+1] -
    cu_chunk_seqlens[c]``. Rows at or past that boundary of either operand are
    zero, so entries with ``i >= limit`` or ``j >= limit`` are exactly zero while
    the full ``chunk_size x chunk_size`` tile is still written.
    """
    if cmat.dim() != 3 or bmat.dim() != 3:
        raise ValueError("cmat and bmat must be [tokens, groups, dstate].")
    if bmat.shape != cmat.shape:
        raise ValueError("cmat and bmat must share a shape.")
    tokens, _, _ = cmat.shape
    offsets = [int(value) for value in cu_chunk_seqlens.reshape(-1).tolist()]
    nchunks = len(offsets) - 1
    if nchunks < 1:
        raise ValueError("cu_chunk_seqlens must have at least two entries.")
    if dot_dtype is None:
        dot_dtype = _dot_dtype(cmat.dtype, bmat.dtype)

    cb = torch.zeros(
        (nchunks, cmat.shape[1], chunk_size, chunk_size),
        dtype=torch.float32,
        device=cmat.device,
    )
    for chunk in range(nchunks):
        lo, hi = offsets[chunk], offsets[chunk + 1]
        if lo < 0 or hi > tokens or hi < lo:
            raise ValueError(f"chunk {chunk} spans [{lo}, {hi}) outside 0..{tokens}.")
        limit = hi - lo
        if limit <= 0:
            continue
        if limit > chunk_size:
            raise ValueError(f"chunk {chunk} is longer than chunk_size.")
        # The rounding happens before the product, on both operands.
        a = cmat[lo:hi].to(dot_dtype).to(torch.float32)
        b = bmat[lo:hi].to(dot_dtype).to(torch.float32)
        cb[chunk, :, :limit, :limit] = torch.einsum("igk,jgk->gij", a, b)
    return cb
