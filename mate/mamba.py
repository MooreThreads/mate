"""Public MUSA-native mamba/SSU API.

``selective_state_update`` is the decode-side entry point consumed by the
FlashInfer-shaped compatibility wrapper (``flashinfer.mamba``) in a MUSA build
whose provider is MATE. It implements the selective state update natively in
tilelang and mirrors the reference contract exactly for the supported
configuration.

Two kernels share this entry point, and the call shape picks between them:

- one token per sequence with no acceptance counts and no slot table
  (``ssu_one_token``) -- the plain decode step, which vLLM's Mamba2 mixer makes on
  every step, so it keeps the shape it was built for;
- everything else (``ssu_packed_varlen``): packed variable-length rows described by
  ``cu_seqlens``, one token per sequence per speculative position, a per-token
  destination slot table, and the acceptance count that seeds the read slot. This is
  the MTP decode call, and its state chain is what makes speculative decoding work.

Supported today:

- ``state`` fp16/bf16/fp32 ``[slots, heads, dim, dstate]``, contiguous
- ``x``/``B``/``C``/``z`` fp16 or bf16; ``dt``/``A``/``D``/``dt_bias`` fp32
- per-head (tied) ``A``, ``D`` and ``dt_bias``, i.e. ``A[h]`` broadcasting over
  ``dim``/``dstate``, which is the layout the Mamba2 consumers pass
- ``state_batch_indices``/``dst_state_batch_indices`` as a ``[sequences]`` vector or
  a ``[sequences, steps]`` table, ``cu_seqlens``, ``num_accepted_tokens``
- 3-D packed ``x`` ``[rows, heads, dim]`` and the dense 4-D
  ``[batch, steps, heads, dim]`` form of it
- ``pad_slot_id`` and ``disable_state_update`` honoured in kernel

Everything else raises ``NotImplementedError`` naming the missing capability
rather than silently narrowing semantics. Not implemented yet: stochastic rounding
(``rand_seed``), quantized state (``state_scale``), intermediate/replay state
capture (``intermediate_*``, and ``cache_steps`` outside the packed form where
FlashInfer defines it as the sequence length), and channel-wise ``D``/``dt_bias``.

Graph capture: the tilelang kernel is compiled on first use for each
(dtype, shape, flag) tuple. Call :func:`prewarm_selective_state_update` during
warmup, outside any capture context — a first-call compile inside a captured
graph stalls the worker (see `.claude/rules/tilelang-musa.md`). The op itself
issues a single kernel launch, allocates nothing when ``out`` is provided, and
performs no host/device scalar copies.
"""

from __future__ import annotations

import torch

import functools

from mate.api_logging import get_api_logger, mate_api

__all__ = [
    "prewarm_selective_state_update",
    "prewarm_ssd_combined_fwd_varlen",
    "reset_ssd_workspaces",
    "selective_state_update",
    "ssd_combined_fwd_varlen",
]

def _ssu_launch():
    """Import the SSU launcher on first use.

    ``mate.mamba`` is the symbol surface the FlashInfer-shaped compatibility
    wrapper probes, so importing it must not require tilelang: on a CPU-only node
    (or in a test environment) the probe still has to answer.
    """
    from mate.mamba_kernels.tilelang.ssu_one_token import ssu_one_token_launch

    return ssu_one_token_launch


def _ssu_packed_launch():
    """Import the packed (MTP/packed-varlen) SSU launcher on first use."""
    from mate.mamba_kernels.tilelang.ssu_packed_varlen import ssu_packed_launch

    return ssu_packed_launch


_SUPPORTED_STATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_SLOT_DTYPES = (torch.int32, torch.int64)
#: The values a consumer can hand us through ``algorithm``: vLLM spells them
#: ``vllm.config.mamba.MambaSSUAlgorithm`` (``auto``/``simple``/``vertical``/``horizontal``)
#: and passes its selection straight through. This family implements one algorithm, so
#: every accepted value runs the same kernels -- ``auto`` included -- and validation only
#: turns a typo into an error here instead of a silently ignored argument.
_SUPPORTED_ALGORITHMS = ("auto", "simple", "vertical", "horizontal")
#: Backend names this family answers to, named by implementation rather than by consumer:
#: ``tilelang`` is the kernel family under ``mate/mamba_kernels/tilelang``. A consumer's
#: own backend name is not part of this vocabulary -- vLLM's ``--mamba-backend flashinfer``
#: selects us where it dispatches and never reaches this argument.
_SUPPORTED_BACKENDS = ("auto", "tilelang")
_DEFAULT_LANES_PER_ROW = 32
_DEFAULT_ROWS_PER_CTA = 4

#: Dtype of the query-start metadata this module synthesizes. vLLM allocates its
#: ``query_start_loc`` as int32, and the synthesized tensor has to be comparable
#: with the caller's for the one-token fast-path check to mean anything.
_SEQ_DTYPE = torch.int32


def _per_head(
    tensor: torch.Tensor, heads: int, *, name: str, dim: int | None = None
) -> torch.Tensor:
    """Return ``tensor`` as a contiguous per-head vector, or explain the gap."""
    if tensor.dtype != torch.float32:
        raise NotImplementedError(
            f"native mamba SSU requires fp32 {name}, got {tensor.dtype}."
        )
    if tensor.dim() == 1:
        if tensor.shape[0] != heads:
            raise ValueError(
                f"{name} must have one entry per head, got {tensor.shape}."
            )
        return tensor.contiguous()
    if tensor.dim() == 3 and tensor.stride(1) == 0 and tensor.stride(2) == 0:
        # The consumers pass A as a tied strided view [heads, dim, dstate].
        if tensor.shape[0] != heads:
            raise ValueError(
                f"{name} must have one entry per head, got {tensor.shape}."
            )
        return tensor[:, 0, 0].contiguous()
    if tensor.dim() == 3 and tensor.stride(2) == 0 and _matches(tensor, heads, dim):
        # vLLM's decode path passes a per-channel A/D/dt_bias as a [heads, dim,
        # dstate] view whose last stride is zero, because the value does not vary
        # along dstate. Reading the first column is exact for that form.
        return tensor[:, :, 0].contiguous()
    if tensor.dim() == 2 and _matches(tensor, heads, dim):
        return tensor.contiguous()
    raise NotImplementedError(
        f"native mamba SSU implements per-head (tied) or per-(head, dim) {name} only; "
        f"got shape {tuple(tensor.shape)} with strides {tuple(tensor.stride())}."
    )


def _matches(tensor: torch.Tensor, heads: int, dim: int | None) -> bool:
    """Whether a matrix-shaped parameter covers one value per head, or per channel."""
    if tensor.shape[0] != heads:
        return False
    if tensor.shape[1] == 1:
        return True
    return dim is None or tensor.shape[1] == dim


def _slot_vector(
    indices: torch.Tensor | None,
    batch: int,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """The one-token path's ``[batch]`` slot vector, materialized when absent."""
    if indices is None:
        # One tensor op; the consumers pass explicit indices on the capture path.
        return torch.arange(batch, dtype=_SEQ_DTYPE, device=device)
    if indices.dim() != 1:
        raise ValueError(
            f"{name} must be [sequences] on the one-token path; got "
            f"{tuple(indices.shape)}."
        )
    if indices.shape[0] != batch:
        raise ValueError(
            f"{name} must hold one entry per sequence ({batch}), got "
            f"{indices.shape[0]}."
        )
    if indices.dtype not in _SUPPORTED_SLOT_DTYPES:
        raise ValueError(f"{name} must be int32 or int64, got {indices.dtype}.")
    return indices.contiguous()


def _slot_table(
    indices: torch.Tensor | None,
    *,
    name: str,
) -> tuple[torch.Tensor, int, int] | None:
    """Flatten a slot table and say how the packed kernel must address it.

    The kernel reads entry ``sequence * seq_stride + position * step_stride``, which
    is how the Triton kernel addresses these tensors: it reaches a flat table by
    unsqueezing it to ``[rows, 1]``, so both of its strides are 1 and sequence ``b``'s
    position ``t`` is entry ``b + t``. vLLM's MTP6 call passes exactly that form --
    ``state_batch_indices`` and ``dst_state_batch_indices`` are ``[rows]``, one entry
    per packed row -- while a contiguous 2-D ``[sequences, steps]`` table keeps
    strides ``(steps, 1)``. Only the addressing differs: with one sequence per call
    the two forms name the same entries.
    """
    if indices is None:
        return None
    if indices.dtype not in _SUPPORTED_SLOT_DTYPES:
        raise ValueError(f"{name} must be int32 or int64, got {indices.dtype}.")
    table = indices.contiguous()
    if table.dim() == 1:
        return table, 1, 1
    if table.dim() == 2:
        # Contiguous here, so the strides are the row length and 1 by construction;
        # asking the tensor keeps the two in step if that ever changes.
        return table.reshape(-1), table.stride(0), table.stride(1)
    raise ValueError(
        f"{name} must be the flat [rows] or the 2-D [sequences, steps] slot table; "
        f"got {tuple(indices.shape)}."
    )


_DENSE_SEQ_STARTS: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}


def _dense_seq_starts(
    batch: int, steps: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Regular query starts for a dense ``[batch, steps, ...]`` call.

    The packed kernel reads every sequence's row range from ``cu_seqlens``, and the
    dense layout is just equal-length sequences. Building the tensor once per
    (batch, steps) keeps a ``torch.arange`` out of every decode step and every
    captured graph replay, which is the family's wrapper contract.
    """
    key = (batch, steps, str(device), dtype)
    starts = _DENSE_SEQ_STARTS.get(key)
    if starts is None:
        starts = torch.arange(
            0, (batch + 1) * steps, steps, dtype=dtype, device=device
        ).contiguous()
        _DENSE_SEQ_STARTS[key] = starts
    return starts


def _flatten_steps(
    tensor: torch.Tensor | None, rows: int, *, name: str
) -> torch.Tensor | None:
    """Flatten a dense multi-token operand's ``(batch, steps)`` axes into rows."""
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise ValueError(
            f"a dense (4-D) x requires a 4-D {name} as well, got "
            f"{tuple(tensor.shape)}; pass every operand in one layout."
        )
    return tensor.reshape(rows, *tensor.shape[2:])


def _one_token_per_sequence(cu_seqlens: torch.Tensor | None, rows: int) -> bool:
    """Whether the query starts describe exactly one token per sequence.

    vLLM's Mamba2 mixer passes them on every decode step, and for plain
    (non-speculative) decoding they are ``arange(rows + 1)`` -- the call the
    one-token kernel exists for. Equal counts do not prove it (``[0, 0, 2]`` also
    has ``rows + 1`` entries), so the vector is compared rather than counted, and an
    empty sequence keeps the fast path out of the picture.

    This comparison is the one place the entry point synchronizes, and it is the
    check the one-token path already performed. The packed route never does: it
    reads ``cu_seqlens`` in kernel.
    """
    if cu_seqlens is None:
        return True
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() != rows + 1:
        return False
    if cu_seqlens.dtype not in _SUPPORTED_SLOT_DTYPES:
        raise ValueError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}.")
    expected = torch.arange(rows + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return bool(torch.equal(cu_seqlens, expected))


@functools.lru_cache(maxsize=1)
def _announce_native_ssd() -> None:
    """Log once per process that the native SSD prefill is the active path.

    A performance pair that only shows a latency difference cannot prove which
    implementation ran: the same argv can reach a different kernel. This line is
    the assertion -- visible with ``MATE_LOGLEVEL=1`` (the ``mate.api`` logger is
    silent otherwise, so it costs nothing on the default path), and absent when
    the wrapper resolves somewhere else.
    """
    get_api_logger().info(
        "mate.mamba: native SSD prefill active (TileLang chunked scan on MUSA); "
        "flashinfer.mamba.ssd_combined_fwd_varlen resolves here"
    )


def _packed_selective_state_update(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    a_head: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    d_head: torch.Tensor | None,
    bias_head: torch.Tensor | None,
    z: torch.Tensor | None,
    state_batch_indices: torch.Tensor | None,
    dst_state_batch_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    out: torch.Tensor,
    dt_softplus: bool,
    pad_slot_id: int,
    disable_state_update: bool,
) -> torch.Tensor:
    """Route a multi-token call to the packed kernel, validating its extra contract.

    Everything here is a device-tensor or shape decision: the packed route never
    copies a scalar back to the host, so a captured decode step can replay it.
    """
    rows = x.shape[0]
    # How many sequences the call describes. The row split answers this whenever
    # cu_seqlens is present, which does not depend on the slot tables at all -- and a
    # flat table carries one entry per packed row, not one per sequence, so its length
    # cannot answer it. Acceptance counts are one per sequence, and a 2-D table has a
    # row per sequence.
    if cu_seqlens is not None:
        sequences = int(cu_seqlens.numel()) - 1
    elif num_accepted_tokens is not None:
        sequences = int(num_accepted_tokens.numel())
    elif state_batch_indices is not None:
        # A flat table only reaches here without cu_seqlens, and then a row is a
        # sequence, so either form has one entry per sequence: the row split below
        # requires exactly that.
        sequences = int(state_batch_indices.shape[0])
    elif dst_state_batch_indices is not None:
        sequences = int(dst_state_batch_indices.shape[0])
    else:
        raise ValueError(
            "a packed multi-token call must carry per-sequence metadata: "
            "state_batch_indices, dst_state_batch_indices, num_accepted_tokens or "
            "cu_seqlens."
        )

    if num_accepted_tokens is not None:
        if num_accepted_tokens.dim() != 1:
            raise ValueError(
                "num_accepted_tokens must be [sequences], got "
                f"{tuple(num_accepted_tokens.shape)}."
            )
        if num_accepted_tokens.dtype not in _SUPPORTED_SLOT_DTYPES:
            raise ValueError(
                f"num_accepted_tokens must be int32 or int64, got "
                f"{num_accepted_tokens.dtype}."
            )
        # The Triton launcher asserts the same thing: an acceptance count picks an
        # entry of the read table, so without that table there is nothing to pick it
        # from. The table itself may be flat or 2-D -- the count is an index, not a
        # shape requirement -- and vLLM's MTP6 call passes the flat form.
        if state_batch_indices is None:
            raise ValueError(
                "num_accepted_tokens selects the read slot, so it needs "
                "state_batch_indices to read it from."
            )
        if cu_seqlens is not None and num_accepted_tokens.dtype != cu_seqlens.dtype:
            raise ValueError(
                "cu_seqlens and num_accepted_tokens must share a dtype for the packed "
                f"SSU kernel; got {cu_seqlens.dtype} and {num_accepted_tokens.dtype}."
            )

    if cu_seqlens is None:
        if rows != sequences:
            raise ValueError(
                "a packed multi-token call must pass cu_seqlens: "
                f"{rows} rows cannot be split across {sequences} sequences without "
                "it. A dense [batch, steps, heads, dim] x carries that layout."
            )
        cu_seqlens = _dense_seq_starts(sequences, 1, x.device, _SEQ_DTYPE)
    elif cu_seqlens.dim() != 1 or cu_seqlens.numel() != sequences + 1:
        raise ValueError(
            f"cu_seqlens must be the 1-D vector holding one start per sequence plus "
            f"the total: expected {sequences + 1} entries for {sequences} sequences, "
            f"got {tuple(cu_seqlens.shape)}."
        )
    if cu_seqlens.dtype not in _SUPPORTED_SLOT_DTYPES:
        raise ValueError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}.")

    src_table = _slot_table(state_batch_indices, name="state_batch_indices")
    if src_table is None:
        # With no slot table the state coordinate is the sequence's own row, which is
        # what the Triton kernel uses when HAS_STATE_BATCH_INDICES is false: one slot
        # per sequence, so stride 1 with a single position.
        src_table = (
            torch.arange(sequences, dtype=_SEQ_DTYPE, device=x.device),
            1,
            1,
        )
    dst_table = _slot_table(dst_state_batch_indices, name="dst_state_batch_indices")
    src_slots, src_seq_stride, src_step_stride = src_table
    dst_slots, dst_seq_stride, dst_step_stride = (
        (None, 0, 0) if dst_table is None else dst_table
    )

    return _ssu_packed_launch()(
        state,
        x,
        dt,
        a_head,
        B,
        C,
        d_head,
        bias_head,
        z,
        src_slots,
        src_seq_stride,
        src_step_stride,
        dst_slots,
        dst_seq_stride,
        dst_step_stride,
        cu_seqlens.contiguous(),
        None if num_accepted_tokens is None else num_accepted_tokens.contiguous(),
        out,
        dt_softplus=dt_softplus,
        pad_slot_id=pad_slot_id,
        disable_state_update=disable_state_update,
    )


@mate_api
def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    state_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    intermediate_state_scales: torch.Tensor | None = None,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Apply one decode-step selective state update and return the output ``y``.

    The argument order, defaults and semantics follow the FlashInfer-shaped
    contract the MUSA consumers already call, so this function can serve as the
    implementation behind that surface without an adapter layer.

    ``x`` is the packed ``[rows, heads, dim]`` form together with ``cu_seqlens``, or
    the dense ``[batch, steps, heads, dim]`` form of the same recurrence. Both are
    supported, and the call shape -- not a flag -- decides which kernel runs.
    """
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"native mamba SSU implements {_SUPPORTED_ALGORITHMS}, got algorithm={algorithm!r}."
        )
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"native mamba SSU accepts backends {_SUPPORTED_BACKENDS}, got backend={backend!r}."
        )
    if rand_seed is not None:
        raise NotImplementedError(
            "native mamba SSU does not implement stochastic rounding (rand_seed) yet."
        )
    if state_scale is not None:
        raise NotImplementedError(
            "native mamba SSU does not implement quantized state (state_scale) yet."
        )
    if (
        intermediate_states_buffer is not None
        or intermediate_state_indices is not None
        or intermediate_state_scales is not None
    ):
        raise NotImplementedError(
            "native mamba SSU does not implement intermediate/replay state capture yet."
        )
    if cache_steps and cu_seqlens is None:
        # FlashInfer defines cache_steps as the number of steps to cache -- the
        # sequence length in varlen mode, the depth of a dense multi-token call
        # otherwise -- and vLLM's SSU dispatch forwards it as the slot-table width
        # whenever it passes cu_seqlens. Refusing the packed form would leave the
        # whole --mamba-backend flashinfer arm unreachable, so only the dense form is
        # refused here: that one asks for intermediate-state capture, which is the
        # capability this module does not have.
        raise NotImplementedError(
            "native mamba SSU does not implement the dense multi-token "
            "intermediate-state cache (cache_steps without cu_seqlens) yet."
        )
    if x.dim() not in (3, 4):
        raise ValueError(
            f"x must be the packed [rows, heads, dim] or the dense "
            f"[batch, steps, heads, dim] form, got {tuple(x.shape)}."
        )
    dense_shape = None
    dense_out = None
    if x.dim() == 4:
        # The dense form is the same recurrence as packed varlen with one
        # equal-length sequence per batch row, so it is flattened -- a view on the
        # contiguous operands the consumers pass -- and the packed kernel is handed
        # the regular cu_seqlens that expresses it. Keeping one kernel for both
        # layouts is what makes the dense form free rather than a second code path.
        dense_batch, steps = x.shape[:2]
        dense_shape = tuple(x.shape)
        if cu_seqlens is None:
            cu_seqlens = _dense_seq_starts(dense_batch, steps, x.device, _SEQ_DTYPE)
        if out is not None:
            if not out.is_contiguous():
                raise ValueError(
                    "a dense (4-D) out must be contiguous: the kernel writes it "
                    "through a flattened view, and a copy would drop the result."
                )
            dense_out = out
            out = out.view(dense_batch * steps, out.shape[2], out.shape[3])
        rows = dense_batch * steps
        x = x.reshape(rows, x.shape[2], x.shape[3])
        dt = _flatten_steps(dt, rows, name="dt")
        B = _flatten_steps(B, rows, name="B")
        C = _flatten_steps(C, rows, name="C")
        z = _flatten_steps(z, rows, name="z")
    rows = x.shape[0]

    if state.dim() != 4:
        raise ValueError(
            f"state must be [slots, heads, dim, dstate], got {tuple(state.shape)}."
        )
    slots, heads, dim, dstate = state.shape
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        raise NotImplementedError(
            f"native mamba SSU supports state dtypes {_SUPPORTED_STATE_DTYPES}, "
            f"got {state.dtype}."
        )
    if x.dtype not in _SUPPORTED_IO_DTYPES:
        raise NotImplementedError(
            f"native mamba SSU supports x dtypes {_SUPPORTED_IO_DTYPES}, got {x.dtype}."
        )
    dt_broadcast = dt.dim() == 3 and dt.shape[:2] == (rows, heads) and dt.shape[2] == 1
    if not dt_broadcast and (x.shape != dt.shape or x.shape != (rows, heads, dim)):
        raise ValueError(
            f"x/dt must match the state head layout: x={tuple(x.shape)}, "
            f"dt={tuple(dt.shape)}, state={tuple(state.shape)}."
        )
    if B.dim() != 3 or C.shape != B.shape:
        raise ValueError(
            f"B/C must be [rows, groups, dstate]; got B={tuple(B.shape)}, "
            f"C={tuple(C.shape)}."
        )
    if B.dtype != x.dtype or C.dtype != x.dtype:
        raise ValueError("B and C must share the x dtype for the native SSU kernel.")

    a_head = _per_head(A, heads, dim=dim, name="A")
    d_head = _per_head(D, heads, dim=dim, name="D") if D is not None else None
    bias_head = (
        _per_head(dt_bias, heads, dim=dim, name="dt_bias")
        if dt_bias is not None
        else None
    )
    if z is not None and (z.shape != x.shape or z.dtype != x.dtype):
        raise ValueError("z must match x in shape and dtype for the native SSU kernel.")
    if out is None:
        out = torch.empty((rows, heads, dim), dtype=x.dtype, device=x.device)
    elif out.shape != (rows, heads, dim) or out.dtype != x.dtype:
        raise ValueError(
            f"out must be {x.dtype} with shape {tuple(x.shape)}, got "
            f"{out.dtype} {tuple(out.shape)}."
        )

    # The plain decode step -- one token per sequence, no acceptance counts, no slot
    # table -- stays on the kernel built for it, because that is the call vLLM's
    # Mamba2 mixer makes on every step of every decode. Multi-token information is
    # what moves a call to the packed kernel: acceptance counts, a slot table with a
    # column per speculative position, or query starts that are not one token per
    # row.
    if (
        num_accepted_tokens is None
        and _one_token_per_sequence(cu_seqlens, rows)
        and (state_batch_indices is None or state_batch_indices.dim() == 1)
        and (dst_state_batch_indices is None or dst_state_batch_indices.dim() == 1)
    ):
        result = _ssu_launch()(
            state,
            x,
            dt,
            a_head,
            B,
            C,
            d_head,
            bias_head,
            z,
            _slot_vector(
                state_batch_indices, rows, name="state_batch_indices", device=x.device
            ),
            None
            if dst_state_batch_indices is None
            else _slot_vector(
                dst_state_batch_indices,
                rows,
                name="dst_state_batch_indices",
                device=x.device,
            ),
            out,
            dt_softplus=bool(dt_softplus),
            pad_slot_id=int(pad_slot_id),
            disable_state_update=bool(disable_state_update),
        )
    else:
        result = _packed_selective_state_update(
            state=state,
            x=x,
            dt=dt,
            a_head=a_head,
            B=B,
            C=C,
            d_head=d_head,
            bias_head=bias_head,
            z=z,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            out=out,
            dt_softplus=bool(dt_softplus),
            pad_slot_id=int(pad_slot_id),
            disable_state_update=bool(disable_state_update),
        )

    if dense_out is not None:
        # The caller's own buffer, so `returned is out` holds for the dense form too.
        return dense_out
    if dense_shape is not None:
        # A dense call gets its rank back; shapes carry meaning in this family.
        return result.view(dense_shape)
    return result


@mate_api
def prewarm_selective_state_update(
    *,
    state_dtype: torch.dtype,
    io_dtype: torch.dtype,
    batch: int,
    heads: int,
    dim: int,
    dstate: int,
    groups: int,
    slot_dtype: torch.dtype,
    rows_per_cta: int = _DEFAULT_ROWS_PER_CTA,
    lanes_per_row: int = _DEFAULT_LANES_PER_ROW,
    packed: bool = False,
    meta_dtype: torch.dtype = _SEQ_DTYPE,
    dt_dim: int | None = None,
) -> None:
    """Compile the native kernel for one configuration, outside capture.

    Call this during warmup (before any graph capture) for each shape/dtype
    combination the serving configuration will use. Compiling inside a captured
    graph stalls the worker, so prewarming is required rather than optional.

    ``packed`` selects the kernel a multi-token serving configuration will actually
    run -- the packed/MTP one -- because that is a different compilation from the
    one-token fast path. ``dt_dim`` is what the configuration passes as ``dt``: ``1``
    for the per-head step the FlashInfer-shaped wrapper hands over (one value per
    head, broadcast over ``dim``), ``None`` for this module's default of a full
    ``[rows, heads, dim]`` dt. ``meta_dtype`` is the dtype of
    ``cu_seqlens``/``num_accepted_tokens``, which the packed kernel specializes on.
    """
    if packed:
        from mate.mamba_kernels.tilelang.ssu_packed_varlen import (
            prewarm_ssu_packed_varlen,
        )

        prewarm_ssu_packed_varlen(
            state_dtype=state_dtype,
            io_dtype=io_dtype,
            slot_dtype=slot_dtype,
            meta_dtype=meta_dtype,
            dim=dim,
            dstate=dstate,
            head_ratio=heads // groups,
            dt_dim=dt_dim,
            rows_per_cta=rows_per_cta,
            lanes_per_row=lanes_per_row,
        )
        return

    from mate.mamba_kernels.tilelang.ssu_one_token import prewarm_ssu_one_token

    prewarm_ssu_one_token(
        state_dtype=state_dtype,
        io_dtype=io_dtype,
        dim=dim,
        dstate=dstate,
        head_ratio=heads // groups,
        slot_dtype=slot_dtype,
        rows_per_cta=rows_per_cta,
        lanes_per_row=lanes_per_row,
        dt_dim=dt_dim,
    )


# --- SSD packed prefill (the `--mamba-backend flashinfer` prefill entry) ------

_SSD_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_SSD_SUPPORTED_DT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_SSD_SUPPORTED_STATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class _SsdWorkspace:
    """Per-shape scratch for the SSD stages.

    Allocating the intermediates per call is what makes a wrapper expensive: five
    kernels' worth of scratch inside a captured graph is allocation churn on every
    replay. The workspace is keyed by everything the buffers depend on, so a
    serving run allocates once per shape and replays allocation-free after that.
    """

    __slots__ = ("dt_out", "dA_cumsum", "states", "CB")

    def __init__(self, heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, device):
        self.dt_out = torch.empty(
            (heads, nchunks, chunk_size), device=device, dtype=torch.float32
        )
        self.dA_cumsum = torch.empty_like(self.dt_out)
        self.states = torch.empty(
            (nchunks, heads, dim, dstate), device=device, dtype=state_dtype
        )
        self.CB = torch.empty(
            (nchunks, groups, chunk_size, chunk_size), device=device, dtype=torch.float32
        )


_WORKSPACES: dict[tuple, _SsdWorkspace] = {}


def _workspace(
    heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, device
) -> _SsdWorkspace:
    key = (heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, str(device))
    space = _WORKSPACES.get(key)
    if space is None:
        space = _SsdWorkspace(
            heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, device
        )
        _WORKSPACES[key] = space
    return space


def reset_ssd_workspaces() -> None:
    """Drop cached scratch, e.g. before changing device or freeing memory."""
    _WORKSPACES.clear()


_STAGE_LAUNCHERS: dict | None = None


def _stage(name: str):
    """Resolve one stage launcher.

    Imported lazily so that a partial checkout fails with the name of the missing
    stage rather than an ImportError at module import, and so the stage table
    stays patchable for tests. The table is built once; each stage's compiled
    kernel is then reused across calls instead of being re-resolved by the JIT.
    """
    global _STAGE_LAUNCHERS
    if _STAGE_LAUNCHERS is None:
        from mate.mamba_kernels.tilelang import (
            ssd_bmm,
            ssd_chunk_cumsum,
            ssd_chunk_scan,
            ssd_chunk_state,
            ssd_state_passing,
        )
        from mate.mamba_kernels.tilelang._kernel_cache import reuse_compiled

        for module, factory in (
            (ssd_chunk_cumsum, "tilelang_ssd_chunk_cumsum"),
            (ssd_chunk_state, "tilelang_ssd_chunk_state"),
            (ssd_state_passing, "tilelang_ssd_state_passing"),
            (ssd_bmm, "tilelang_ssd_bmm"),
            (ssd_chunk_scan, "tilelang_ssd_chunk_scan"),
        ):
            reuse_compiled(module, factory)
        _STAGE_LAUNCHERS = {
            "cumsum": ssd_chunk_cumsum.chunk_cumsum_launch,
            "chunk_state": ssd_chunk_state.ssd_chunk_state_launch,
            "state_passing": ssd_state_passing.ssd_state_passing_launch,
            "bmm": ssd_bmm.ssd_bmm_launch,
            "chunk_scan": ssd_chunk_scan.chunk_scan_launch,
        }
    return _STAGE_LAUNCHERS[name]


@mate_api
def ssd_combined_fwd_varlen(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    seq_idx: torch.Tensor,
    out: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: torch.Tensor | None = None,
    return_intermediate_states: bool = False,
    state_dtype: torch.dtype | None = None,
    checkpoint_token_indices: torch.Tensor | None = None,
    checkpoint_state_slots: torch.Tensor | None = None,
    checkpoint_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed variable-length SSD prefill, native on MUSA.

    The five stages run as five tilelang launches with per-shape scratch taken
    from a cached workspace, so a steady-state call allocates nothing but the
    returned state tensor.

    Contract, matching vLLM's Mamba2 prefill caller:

    - ``x`` is ``[tokens, heads, dim]`` and ``dt`` is ``[tokens, heads]``, both
      packed over the flattened (variable-length) batch;
    - ``cu_chunk_seqlens`` is the only authority for the chunk-to-token mapping:
      chunk ``c`` covers ``[cu_chunk_seqlens[c], cu_chunk_seqlens[c+1])``. A chunk
      never spans two sequences, and every chunk but the last of a sequence spans
      ``chunk_size`` tokens;
    - ``last_chunk_indices[b]`` is the ordinal of sequence ``b``'s last logical
      chunk and ``seq_idx[c]`` is the request owning chunk ``c``. A chunk opens a
      sequence when ``seq_idx`` differs from the previous chunk's, and that
      transition -- not the ``last_chunk_indices`` arithmetic -- decides where the
      entering state comes from. The two can be made to disagree only by an
      inconsistent caller, and the two independent implementations of this
      contract resolve that disagreement differently, so ``seq_idx`` is the
      authority;
    - ``out`` is written in place when supplied;
    - the return value is a single tensor: the full ``[nchunks, heads, dim,
      dstate]`` state buffer when ``return_intermediate_states``, otherwise
      ``states[last_chunk_indices]`` with shape ``[batch, heads, dim, dstate]``,
      both in ``state_dtype`` (which defaults to ``initial_states.dtype``, else
      ``C.dtype``).

    ``checkpoint_*`` is not supported and raises rather than silently ignoring a
    capability the caller asked for.
    """
    if any(
        value is not None
        for value in (checkpoint_token_indices, checkpoint_state_slots, checkpoint_states)
    ):
        raise NotImplementedError(
            "ssd_combined_fwd_varlen does not support the checkpoint_* arguments; "
            "the caller must dispatch to its reference path for checkpointed runs."
        )
    if x.dim() != 3 or dt.dim() != 2 or B.dim() != 3 or C.dim() != 3:
        raise ValueError("packed SSD expects x [tokens, heads, dim] and dt [tokens, heads].")
    tokens, heads, dim = x.shape
    groups, dstate = B.shape[1], B.shape[2]
    if dt.shape != (tokens, heads):
        raise ValueError("dt must be [tokens, heads].")
    if x.dtype not in _SSD_SUPPORTED_IO_DTYPES or B.dtype != x.dtype or C.dtype != x.dtype:
        raise ValueError("x, B and C must share one of fp16/bf16.")
    if dt.dtype not in _SSD_SUPPORTED_DT_DTYPES:
        raise ValueError("dt must be fp32, fp16 or bf16.")
    if heads % groups:
        raise ValueError("heads must be divisible by groups.")
    if A.shape != (heads,):
        raise ValueError("A must be [heads].")
    if A.dtype != torch.float32:
        raise ValueError("A must be fp32.")
    if dt_bias is not None and dt_bias.shape != (heads,):
        raise ValueError("dt_bias must be [heads].")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    nchunks = cu_chunk_seqlens.numel() - 1
    _announce_native_ssd()
    if nchunks < 1:
        raise ValueError("cu_chunk_seqlens must have at least two entries.")
    if seq_idx.numel() != nchunks:
        raise ValueError("seq_idx must hold one request id per logical chunk.")
    if cu_seqlens.numel() != last_chunk_indices.numel() + 1:
        raise ValueError(
            "cu_seqlens and last_chunk_indices must describe the same batch."
        )
    batch = last_chunk_indices.numel()
    if initial_states is not None:
        if tuple(initial_states.shape[1:]) != (heads, dim, dstate):
            raise ValueError("initial_states must be [batch, heads, dim, dstate].")
        if int(initial_states.shape[0]) != batch:
            raise ValueError(
                "initial_states must hold one row per sequence in cu_seqlens."
            )
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("out must match x's shape and dtype.")
    if D is not None and D.shape not in ((heads, dim), (heads,)):
        raise ValueError("D must be [heads, dim] or [heads].")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must match x's shape.")
    if state_dtype is None:
        state_dtype = initial_states.dtype if initial_states is not None else C.dtype
    if state_dtype not in _SSD_SUPPORTED_STATE_DTYPES:
        raise ValueError("state_dtype must be fp16, bf16 or fp32.")
    for tensor in (x, dt, A, B, C, cu_seqlens, cu_chunk_seqlens, last_chunk_indices, seq_idx):
        if not tensor.is_contiguous():
            raise ValueError(
                "x, dt, A, B, C and the chunk metadata must be contiguous."
            )

    space = _workspace(
        heads, groups, dim, dstate, nchunks, chunk_size, state_dtype, x.device
    )

    _stage("cumsum")(
        dt,
        A,
        dt_bias,
        cu_chunk_seqlens,
        chunk_size,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        dA_cumsum=space.dA_cumsum,
        dt_out=space.dt_out,
    )
    _stage("chunk_state")(
        x,
        B,
        space.dt_out,
        space.dA_cumsum,
        cu_chunk_seqlens,
        chunk_size,
        states=space.states,
    )
    _stage("state_passing")(
        space.states,
        space.dA_cumsum,
        initial_states,
        last_chunk_indices,
        seq_idx,
        state_dtype=state_dtype,
    )
    _stage("bmm")(
        C,
        B,
        cu_chunk_seqlens,
        chunk_size,
        cb=space.CB,
    )
    _stage("chunk_scan")(
        x,
        C,
        space.CB,
        space.dt_out,
        space.dA_cumsum,
        space.states,
        initial_states,
        seq_idx,
        cu_chunk_seqlens,
        out,
        D_param=D,
        z=z,
    )
    if return_intermediate_states:
        return space.states
    # MUSA: index_select runs this gather about five times slower than indexing.
    return space.states[last_chunk_indices]


@mate_api
def prewarm_ssd_combined_fwd_varlen(
    heads: int,
    groups: int,
    dim: int,
    dstate: int,
    chunk_size: int,
    io_dtype: torch.dtype = torch.bfloat16,
    state_dtype: torch.dtype = torch.bfloat16,
    dt_dtype: torch.dtype = torch.float32,
    block_M: int = 32,
) -> None:
    """Compile every SSD stage for one shape, outside any captured graph.

    Call during warmup for each configuration the server will serve: a first-call
    compile inside a captured graph stalls the worker. The compile-time degrees of
    freedom are the four shape parameters, the three dtypes and the scan's
    ``block_M``; the token and chunk counts are runtime dimensions.
    """
    from mate.mamba_kernels.tilelang.ssd_bmm import prewarm_ssd_bmm
    from mate.mamba_kernels.tilelang.ssd_chunk_cumsum import prewarm_ssd_chunk_cumsum
    from mate.mamba_kernels.tilelang.ssd_chunk_scan import prewarm_ssd_chunk_scan
    from mate.mamba_kernels.tilelang.ssd_chunk_state import prewarm_ssd_chunk_state
    from mate.mamba_kernels.tilelang.ssd_state_passing import prewarm_ssd_state_passing

    # Each stage's prewarm takes its own shape vocabulary: cumsum and the scan key
    # on (heads, groups), the intra-chunk stages on (dim, dstate, chunk_size) plus
    # the head ratio, and the chunk matrix on (chunk_size, dstate) -- it is shared
    # across the heads of a group. Passing the orchestration's own argument order
    # through would put integers where dtypes belong.
    head_ratio = heads // groups
    prewarm_ssd_chunk_cumsum(heads, chunk_size, dt_dtype)
    prewarm_ssd_chunk_state(dim, dstate, chunk_size, head_ratio, io_dtype)
    prewarm_ssd_state_passing(
        dim, dstate, chunk_size, state_dtype, in_dtype=state_dtype,
        init_dtype=state_dtype,
    )
    prewarm_ssd_bmm(chunk_size, dstate, io_dtype)
    prewarm_ssd_chunk_scan(
        heads, groups, dim, dstate, chunk_size, io_dtype, state_dtype, block_M
    )
