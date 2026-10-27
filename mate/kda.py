"""Public chunk and decode APIs for KDA."""

from __future__ import annotations

import os

import torch

from mate.api_logging import mate_api
from mate.mate_runtime import get_physical_num_mps
from mate.jit.kda_ops import (
    KDA_PREFILL_CHUNK_SIZE,
    KDA_PREFILL_HEAD_DIM,
    get_kda_recurrence_ops_function_name,
    get_kda_recurrence_ops_module,
    get_kda_prepare_ops_function_name,
    get_kda_prepare_ops_module,
    make_kda_ops_config,
    resolve_kda_prepare_ctas_per_mp,
)
from mate.kda_kernels.tilelang import kda_decode as kda_decode_tilelang

_SUPPORTED_QKVA_DTYPES = (torch.float16, torch.bfloat16)


def _as_4d_varlen_input(
    x: torch.Tensor,
    *,
    name: str,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if cu_seqlens is None:
        if x.ndim != 4:
            raise ValueError(f"{name} must be a 4D tensor [B, T, H, 128].")
        return x, False
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}.shape[0] must be 1 when cu_seqlens is provided; "
                "flatten variable-length input as [S, H, 128]."
            )
        return x, False
    raise ValueError(f"{name} must be [S, H, 128] or [1, S, H, 128] for varlen.")


def _check_state_dtype(
    x: torch.Tensor,
    *,
    name: str,
    value_dtype: torch.dtype,
) -> None:
    if x.dtype not in (value_dtype, torch.float32):
        raise TypeError(f"{name} must have dtype {value_dtype} or torch.float32.")


def kda_chunk_state_rows(
    *,
    batch: int,
    num_tokens: int,
    cu_seqlens: torch.Tensor | None = None,
) -> int:
    """Rows a caller must allocate for ``chunk_kda(chunk_states=...)``.

    One row per (sequence slot, chunk) pair, following the same packed chunk
    slot space the KDA prepare workspace uses.  Callers may over-provision:
    the kernel only writes the rows it needs, so a fixed-capacity pool sized
    for the worst case is fine.
    """
    chunks = -(-num_tokens // KDA_PREFILL_CHUNK_SIZE)
    if cu_seqlens is None:
        return batch * chunks
    return chunks + int(cu_seqlens.numel() - 1)


def _kda_checkpoint_rows(
    *,
    batch: int,
    num_tokens: int,
    cu_seqlens: torch.Tensor | None = None,
    every_n_tokens: int,
) -> int:
    """Rows the checkpoint pool that ``chunk_kda`` allocates needs.

    Internal: the operator owns the pool, so this exists to size that allocation
    (and to let the tests predict its shape).  The capacity is the same bound
    cudnn-frontend allocates: ``max(total_tokens // every_n_tokens + N, 1)``,
    which covers the exact packed row count
    ``sum_b ((len_b - 1) // C + 1)``.  ``num_tokens`` is the per-sequence length
    for a dense batch and the packed total for varlen.
    """
    if every_n_tokens <= 0:
        raise ValueError("every_n_tokens must be positive.")
    if every_n_tokens % KDA_PREFILL_CHUNK_SIZE:
        raise ValueError(
            f"every_n_tokens must be a multiple of the kernel chunk "
            f"({KDA_PREFILL_CHUNK_SIZE}), got {every_n_tokens}."
        )
    sequences = batch if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    total_tokens = batch * num_tokens if cu_seqlens is None else num_tokens
    return max(total_tokens // every_n_tokens + sequences, 1)


# ---------------------------------------------------------------------------
# KDA prefill workspace: ONE allocation, chunk-major record layout.
#
# `chunk_kda` allocates a single buffer of shape (records, H, RecordStride) and
# hands the kernels five views into it, so for each (chunk, head) record the
# five fields are adjacent:
#
#   field          elements   bytes   byte offset in the record block
#   decayed            4096    8192        0
#   k_restored         2048    4096     8192
#   p                   256     512    12288
#   inverse             256     512    12800
#   total               128     256    13312
#   ---------------------------------------------------------------
#   RecordStride       6784   13568
#
# Every field offset and the record stride are multiples of 256 B, so every
# field of every record is 256 B aligned -- the alignment the TME descriptors
# and robust copies need.
#
# These numbers mirror ChunkKdaWorkspace in
# include/mate/flat/kda/chunk_kda_workspace.hpp; keep the two in sync.
# All five fields are Element rows, so RecordStride is exactly the sum of the
# five field sizes: 4096 + 2048 + 256 + 256 + 128 = 6784.
_KDA_WS_CHUNK = KDA_PREFILL_CHUNK_SIZE
_KDA_WS_HEAD_DIM = KDA_PREFILL_HEAD_DIM
_KDA_WS_DECAYED_ELEMENTS = 2 * _KDA_WS_CHUNK * _KDA_WS_HEAD_DIM
_KDA_WS_KRESTORED_ELEMENTS = _KDA_WS_CHUNK * _KDA_WS_HEAD_DIM
_KDA_WS_INVERSE_ELEMENTS = _KDA_WS_CHUNK * _KDA_WS_CHUNK
_KDA_WS_P_ELEMENTS = _KDA_WS_CHUNK * _KDA_WS_CHUNK
_KDA_WS_SCALE_ELEMENTS = _KDA_WS_HEAD_DIM  # elements in the `total` row
_KDA_WS_DECAYED_OFFSET = 0
_KDA_WS_KRESTORED_OFFSET = _KDA_WS_DECAYED_OFFSET + _KDA_WS_DECAYED_ELEMENTS
# `p` precedes `inverse` so the MP32 recurrence can read both as ONE 32-row tile:
# rows 0..15 are P (the C rows owned by warps 0/1, which also own the decayed Q
# rows, i.e. the output path) and rows 16..31 the inverse (warps 2/3, the K rows,
# i.e. the U path).  The fields are the same size, so the order costs nothing.
_KDA_WS_P_OFFSET = _KDA_WS_KRESTORED_OFFSET + _KDA_WS_KRESTORED_ELEMENTS
_KDA_WS_INVERSE_OFFSET = _KDA_WS_P_OFFSET + _KDA_WS_P_ELEMENTS
_KDA_WS_TOTAL_OFFSET = _KDA_WS_INVERSE_OFFSET + _KDA_WS_INVERSE_ELEMENTS
_KDA_WS_RECORD_STRIDE = _KDA_WS_TOTAL_OFFSET + _KDA_WS_SCALE_ELEMENTS

# The chunk-major ABI the kernels reconstruct from ChunkKdaWorkspace.
assert _KDA_WS_RECORD_STRIDE == 6784, _KDA_WS_RECORD_STRIDE
assert _KDA_WS_TOTAL_OFFSET == 6656, _KDA_WS_TOTAL_OFFSET


def _allocate_kda_workspace(tile_shape, device, dtype):
    """Allocate the chunk-major workspace and return its five field views.

    The views keep exactly the shapes the previous five separate allocations
    had, so nothing downstream changes.  Every field of every record is 256 B
    aligned and all five views alias the one buffer.
    """
    records, heads = int(tile_shape[0]), int(tile_shape[1])
    buffer = torch.empty(
        (records, heads, _KDA_WS_RECORD_STRIDE), device=device, dtype=dtype
    )
    assert buffer.element_size() == 2, (
        "the chunk-major workspace layout assumes 2-byte activation elements"
    )
    row_stride = heads * _KDA_WS_RECORD_STRIDE

    def view(offset, shape, inner_stride):
        return torch.as_strided(
            buffer,
            size=shape,
            stride=(row_stride, _KDA_WS_RECORD_STRIDE) + inner_stride,
            storage_offset=offset,
        )

    decayed = view(
        _KDA_WS_DECAYED_OFFSET,
        (records, heads, 2 * _KDA_WS_CHUNK, _KDA_WS_HEAD_DIM),
        (_KDA_WS_HEAD_DIM, 1),
    )
    k_restored = view(
        _KDA_WS_KRESTORED_OFFSET,
        (records, heads, _KDA_WS_CHUNK, _KDA_WS_HEAD_DIM),
        (_KDA_WS_HEAD_DIM, 1),
    )
    inverse = view(
        _KDA_WS_INVERSE_OFFSET,
        (records, heads, _KDA_WS_CHUNK, _KDA_WS_CHUNK),
        (_KDA_WS_CHUNK, 1),
    )
    p = view(
        _KDA_WS_P_OFFSET,
        (records, heads, _KDA_WS_CHUNK, _KDA_WS_CHUNK),
        (_KDA_WS_CHUNK, 1),
    )
    total = view(
        _KDA_WS_TOTAL_OFFSET,
        (records, heads, _KDA_WS_HEAD_DIM),
        (1,),
    )

    # Every field of every record must be 256 B aligned.  Each field offset and
    # the record stride is a multiple of 256 B and a fresh allocation is at
    # least 256 B aligned, so this holds by construction; assert it rather than
    # trusting it.
    base = buffer.data_ptr()
    assert base % 256 == 0, f"workspace base {base:#x} is not 256 B aligned"
    fields = (
        ("decayed", decayed),
        ("k_restored", k_restored),
        ("inverse", inverse),
        ("p", p),
        ("total", total),
    )
    for name, tensor in fields:
        assert tensor.data_ptr() % 256 == 0, (
            f"{name} view at {tensor.data_ptr():#x} is not 256 B aligned"
        )
    if os.environ.get("MATE_KDA_WS_PROBE") == "1":
        print(
            f"[ws-probe] buffer base={base:#x} bytes={buffer.numel() * buffer.element_size()} "
            f"records={records} heads={heads} RecordStride={_KDA_WS_RECORD_STRIDE}"
        )
        for name, tensor in fields:
            offset = tensor.data_ptr() - base
            print(
                f"[ws-probe] {name:11s} byte_offset={offset:7d} "
                f"record_stride_bytes={tensor.stride(1) * tensor.element_size():6d} "
                f"align256={offset % 256 == 0} shape={tuple(tensor.shape)}"
            )
    return decayed, k_restored, inverse, p, total


@mate_api
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    use_qk_l2norm_in_kernel: bool = True,
    output: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    chunk_states: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run chunk KDA as a chunk-parallel prepare kernel and a recurrence kernel.

    Args:
        q: Query tensor with shape ``[B, T, Hqk, 128]`` for dense mode or
            ``[S, Hqk, 128]`` / ``[1, S, Hqk, 128]`` for varlen mode.
        k: Key tensor with the same shape and dtype as ``q``.
        v: Value tensor with shape ``[B, T, Hv, 128]`` or varlen equivalent.
        g: Gate input tensor with the same shape as ``v``.
        beta: Beta logits tensor with shape ``[B, T, Hv]`` or varlen equivalent.
        scale: Optional QK scaling factor. Defaults to ``128**-0.5``.
        initial_state: Optional dense recurrent state or recurrent state pool.
        output_final_state: Whether to return the final recurrent state. In
            pool mode, selected rows are updated regardless of this flag; the
            flag only controls whether the pool is also returned.
        cu_seqlens: Optional cumulative sequence lengths for varlen mode.
        A_log: Optional per-head gate parameter tensor.
        dt_bias: Optional per-head, per-channel gate bias tensor.
        lower_bound: Gate lower bound used when gate parameters ``A_log`` and
            ``dt_bias`` are enabled. Defaults to ``-5.0``.
        use_qk_l2norm_in_kernel: Whether to normalize Q/K in the kernel.
        output: Optional preallocated output tensor.
        final_state: Optional preallocated dense final-state tensor. Pool mode
            writes ``initial_state`` directly and accepts only an alias here.
        initial_state_indices: Optional contiguous int32 mapping from input
            sequences to rows of ``initial_state``. When provided, the selected
            pool rows are read and written in place. Concurrent sequences must
            refer to distinct valid pool rows.
        chunk_states: Optional caller-allocated pool that receives the
            recurrent state at every chunk boundary, so a serving framework can
            cache prefix states and skip recomputing a shared prefix on a later
            request. Shape ``[chunk_rows, H, Dv, D]``, contiguous, with the same
            dtype as the recurrent state; size it with
            :func:`mate.kda.kda_chunk_state_rows` (over-provisioning is fine,
            the kernel writes only the rows it needs). Row
            ``workspace_slot + chunk_idx`` of each sequence receives the state
            entering ``chunk_idx``, i.e. row 0 is the state before the first
            token. The state after the final chunk is not written here -- that
            is ``final_state`` / the ``initial_state`` pool.
        checkpoint_every_n_tokens: Optional checkpoint cadence ``C``. When
            positive (and a multiple of the kernel chunk, 16), the call returns
            the recurrent state every ``C`` tokens instead of every chunk, in a
            compact pool that ``chunk_kda`` allocates itself. Entry ``j`` of
            sequence ``b`` holds the state entering token ``j * C`` of that
            sequence, for ``j = 0 .. (len_b - 1) // C`` -- entry 0 is the
            incoming state, and the state after the final token is only
            ``final_state``. Entries are packed per sequence in ``cu_seqlens``
            order, so the valid rows are ``sum_b ((len_b - 1) // C + 1)`` (empty
            sequences contribute none) and the allocated capacity is
            ``max(total_tokens // C + N, 1)``. The pool dtype is the
            activation dtype even when the recurrent state is fp32, matching
            cudnn's ``state_checkpoints``. Cannot be combined with
            ``chunk_states`` or ``initial_state_indices`` (cudnn's
            ``state_indices`` pool is mutually exclusive with checkpoints too).

    Returns:
        ``output``, or ``(output, final_state)`` when ``output_final_state`` is
        set, or ``(output, final_state, checkpoints)`` when
        ``checkpoint_every_n_tokens > 0`` -- where ``final_state`` is a
        zero-size tensor if it was not requested. ``checkpoints`` has shape
        ``[capacity, H, Dv, D]``, contiguous, in the activation dtype.
    """

    q, squeeze_varlen = _as_4d_varlen_input(q, name="q", cu_seqlens=cu_seqlens)
    k, _ = _as_4d_varlen_input(k, name="k", cu_seqlens=cu_seqlens)
    v, _ = _as_4d_varlen_input(v, name="v", cu_seqlens=cu_seqlens)
    g, _ = _as_4d_varlen_input(g, name="g", cu_seqlens=cu_seqlens)
    if beta.ndim == 2 and cu_seqlens is not None:
        beta = beta.unsqueeze(0)

    if (
        q.shape[-1] != 128
        or k.shape[-1] != 128
        or v.shape[-1] != 128
        or g.shape[-1] != 128
    ):
        raise ValueError("chunk_kda currently requires D=128.")
    if k.shape != q.shape:
        raise ValueError("k must have the same shape as q.")
    if q.shape[:2] != v.shape[:2] or q.shape[:2] != g.shape[:2]:
        raise ValueError("q, v and g must have matching [B, T] dimensions.")
    if g.shape != v.shape:
        raise ValueError("g must have the same shape as v.")
    if v.shape[2] % q.shape[2] != 0:
        raise ValueError("GVA requires v/g heads to be divisible by q/k heads.")
    if beta.shape != v.shape[:3]:
        raise ValueError("beta must have shape [B, T, Hv].")
    if q.dtype not in _SUPPORTED_QKVA_DTYPES:
        raise TypeError("chunk_kda supports torch.float16 and torch.bfloat16 inputs.")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or g.dtype != q.dtype
        or beta.dtype != q.dtype
    ):
        raise TypeError("k, v, g and beta must have the same dtype as q.")
    if (A_log is None) != (dt_bias is None):
        raise ValueError("A_log and dt_bias must be provided together.")
    if initial_state is not None:
        _check_state_dtype(initial_state, name="initial_state", value_dtype=q.dtype)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if cu_seqlens is not None:
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have dtype torch.int32 or torch.int64.")
        cu_seqlens = cu_seqlens.contiguous()

    if output is None:
        output = torch.empty_like(v)
    elif squeeze_varlen and output.ndim == 3:
        output = output.unsqueeze(0)
    if output.shape != v.shape:
        raise ValueError("output must have the same shape as v.")
    if output.dtype != q.dtype:
        raise TypeError("output must have the same dtype as q.")

    nseq = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else q.shape[0]
    expected_state_tail = (v.shape[2], 128, 128)
    if initial_state is not None:
        if (
            initial_state.dim() != 4
            or tuple(initial_state.shape[1:]) != expected_state_tail
        ):
            raise ValueError(
                "initial_state must have shape [N_or_pool, H, Dv, D], got "
                f"{tuple(initial_state.shape)}."
            )
        if initial_state.device != q.device:
            raise ValueError(
                f"initial_state must be on device {q.device}, got {initial_state.device}."
            )

    if initial_state_indices is None:
        if initial_state is not None and initial_state.shape[0] != nseq:
            raise ValueError(
                "Dense initial_state must have one row per sequence: expected "
                f"{nseq}, got {initial_state.shape[0]}."
            )
        state_indices = torch.arange(nseq, device=q.device, dtype=torch.int32)
        if output_final_state and final_state is None:
            if initial_state is not None:
                final_state = torch.empty_like(initial_state)
            else:
                final_state = torch.empty(
                    (nseq, *expected_state_tail),
                    device=q.device,
                    dtype=q.dtype,
                )
        elif not output_final_state:
            final_state = None
    else:
        if initial_state is None:
            raise ValueError(
                "initial_state must be provided when initial_state_indices is set."
            )
        if initial_state_indices.dim() != 1 or initial_state_indices.numel() != nseq:
            raise ValueError(
                "initial_state_indices must have shape [N], where N is the number "
                f"of sequences ({nseq}); got {tuple(initial_state_indices.shape)}."
            )
        if initial_state_indices.dtype != torch.int32:
            raise TypeError(
                "initial_state_indices must have dtype torch.int32, got "
                f"{initial_state_indices.dtype}."
            )
        if initial_state_indices.device != q.device:
            raise ValueError(
                "initial_state_indices must be on the same device as q, got "
                f"{initial_state_indices.device} and {q.device}."
            )
        if not initial_state_indices.is_contiguous():
            raise ValueError("initial_state_indices must be contiguous.")
        if (
            final_state is not None
            and final_state.data_ptr() != initial_state.data_ptr()
        ):
            raise ValueError(
                "Pool mode writes initial_state in place; final_state must be None "
                "or alias initial_state."
            )
        # Avoid a device-to-host value-range check, which would synchronize the stream.
        state_indices = initial_state_indices
        final_state = initial_state

    if final_state is not None:
        _check_state_dtype(final_state, name="final_state", value_dtype=q.dtype)
        if (
            final_state.dim() != 4
            or tuple(final_state.shape[1:]) != expected_state_tail
        ):
            raise ValueError(
                "final_state must have shape [N_or_pool, H, Dv, D], got "
                f"{tuple(final_state.shape)}."
            )
        if initial_state_indices is None and final_state.shape[0] != nseq:
            raise ValueError(
                "Dense final_state must have one row per sequence: expected "
                f"{nseq}, got {final_state.shape[0]}."
            )
        if final_state.device != q.device:
            raise ValueError(
                f"final_state must be on device {q.device}, got {final_state.device}."
            )

    if (
        initial_state is not None
        and final_state is not None
        and initial_state.dtype != final_state.dtype
    ):
        raise TypeError("initial_state and final_state must have the same dtype.")

    state_fp32 = any(
        tensor is not None and tensor.dtype == torch.float32
        for tensor in (initial_state, final_state, chunk_states)
    )
    state_dtype = (
        initial_state.dtype
        if initial_state is not None
        else final_state.dtype
        if final_state is not None
        else chunk_states.dtype
        if chunk_states is not None
        else q.dtype
    )
    checkpoints = None
    if checkpoint_every_n_tokens:
        if (
            checkpoint_every_n_tokens < 0
            or checkpoint_every_n_tokens % KDA_PREFILL_CHUNK_SIZE
        ):
            raise ValueError(
                "checkpoint_every_n_tokens must be 0 or a positive multiple of the "
                f"kernel chunk ({KDA_PREFILL_CHUNK_SIZE}), got "
                f"{checkpoint_every_n_tokens}."
            )
        if chunk_states is not None:
            raise ValueError(
                "chunk_states and checkpoint_every_n_tokens write the same "
                "chunk-boundary pool with different row mappings; pass only one."
            )
        if initial_state_indices is not None:
            # cudnn-frontend refuses this combination too: its `state_indices`
            # pool and `checkpoint_every_n_tokens` are mutually exclusive.
            raise ValueError(
                "initial_state_indices cannot be combined with "
                "checkpoint_every_n_tokens"
            )
        checkpoint_rows = _kda_checkpoint_rows(
            batch=q.shape[0],
            num_tokens=q.shape[1],
            cu_seqlens=cu_seqlens,
            every_n_tokens=checkpoint_every_n_tokens,
        )
        # The kernel publishes the activation-typed state tile, so the
        # checkpoint pool is activation-typed even when the recurrent state is
        # fp32 -- cudnn's `state_checkpoints` is io-dtype regardless of an fp32
        # `initial_state`.
        checkpoints = torch.empty(
            (checkpoint_rows, *expected_state_tail),
            device=q.device,
            dtype=q.dtype,
        )
    if chunk_states is not None:
        if chunk_states.dtype == torch.float32:
            # The snapshot is published from the activation-typed state tile, so
            # a fp32 pool cannot be filled exactly.  Fail loudly rather than
            # silently handing back activation-precision values in an fp32
            # buffer.
            raise NotImplementedError(
                "chunk_states does not support an fp32 pool: the kernel publishes "
                "the activation-typed state tile, so an fp32 pool cannot be filled "
                "exactly. Use a bf16/fp16 pool, or omit chunk_states."
            )
        # The pool is caller-owned; the kernel writes it as StateElement, so the
        # dtype has to agree with whatever the other state tensors selected.
        if chunk_states.dim() != 4 or tuple(chunk_states.shape[1:]) != (
            v.shape[2],
            128,
            128,
        ):
            raise ValueError(
                "chunk_states must have shape [chunk_rows, H, Dv, D], got "
                f"{tuple(chunk_states.shape)}."
            )
        if chunk_states.device != q.device:
            raise ValueError(
                f"chunk_states must be on device {q.device}, got {chunk_states.device}."
            )
        if not chunk_states.is_contiguous():
            raise ValueError("chunk_states must be contiguous.")
        if chunk_states.dtype != state_dtype:
            raise TypeError(
                "chunk_states must have the same dtype as the recurrent state "
                f"({state_dtype}), got {chunk_states.dtype}."
            )
        required_rows = kda_chunk_state_rows(
            batch=q.shape[0],
            num_tokens=q.shape[1],
            cu_seqlens=cu_seqlens,
        )
        if chunk_states.shape[0] < required_rows:
            raise ValueError(
                f"chunk_states needs at least {required_rows} rows for this input, "
                f"got {chunk_states.shape[0]}; see mate.kda.kda_chunk_state_rows."
            )
    # Exactly one of the two pools reaches the kernel.
    state_pool = chunk_states if chunk_states is not None else checkpoints
    kda_config = make_kda_ops_config(
        q.dtype,
        state_dtype=state_dtype,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        has_chunk_states_out=state_pool is not None,
        # Compile-time row mapping for the pool: 0 keeps the packed chunk-slot
        # rows of `chunk_states`, a positive multiple of the chunk selects the
        # compact checkpoint layout.  It is validated by `make_kda_ops_config`
        # (and again by the rendered kernel's static_asserts).
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        state_fp32=state_fp32,
        has_gate_params=A_log is not None,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(use_qk_l2norm_in_kernel),
    )
    chunks = (q.shape[1] + KDA_PREFILL_CHUNK_SIZE - 1) // KDA_PREFILL_CHUNK_SIZE
    workspace_records = chunks + nseq if cu_seqlens is not None else nseq * chunks
    workspace_tile_shape = (workspace_records, v.shape[2])
    # ONE chunk-major allocation; see _allocate_kda_workspace above.
    (
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
    ) = _allocate_kda_workspace(workspace_tile_shape, q.device, q.dtype)
    # Persistent K1 uses one int4 range descriptor per launched partition.
    # The MP31 launch policy consumes at most one descriptor per resident
    # prepare CTA.  MP32 can supply a different policy without changing the
    # workspace record ABI above.
    num_seqs = cu_seqlens.numel() - 1 if cu_seqlens is not None else 1
    # Sized by the launch policy the kernel was built with, so the arch comes
    # from the same resolver that renders its ArchTag -- not from the device,
    # which can disagree with MATE_MUSA_ARCH_LIST during a cross build.
    metadata_capacity = (
        resolve_kda_prepare_ctas_per_mp() * get_physical_num_mps(q.device)
        if cu_seqlens is not None and num_seqs > 1
        else 1
    )
    workspace_metadata = torch.empty(
        (metadata_capacity, 4), device=q.device, dtype=torch.int32
    )

    get_kda_prepare_ops_module(kda_config).get_function(
        get_kda_prepare_ops_function_name(kda_config)
    )(
        q,
        k,
        g,
        beta,
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
        workspace_metadata,
        cu_seqlens,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        bool(use_qk_l2norm_in_kernel),
    )
    get_kda_recurrence_ops_module(kda_config).get_function(
        get_kda_recurrence_ops_function_name(kda_config)
    )(
        v,
        output,
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
        state_indices,
        initial_state,
        final_state,
        cu_seqlens,
        state_pool,
    )

    if squeeze_varlen:
        output = output.squeeze(0)
    if checkpoints is not None:
        # Cadence mode: `checkpoint_every_n_tokens` is the caller's contract, so
        # the checkpoint series is always the third element; final_state is a
        # zero-size tensor when it was not requested (cudnn's convention).
        final_out = final_state if final_state is not None else output.new_empty(0)
        return output, final_out, checkpoints
    if output_final_state:
        assert final_state is not None
        return output, final_state
    return output


_SUPPORTED_GATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_QKV_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_STATE_DTYPES = (torch.float32, torch.bfloat16)
_SUPPORTED_OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_DT_BIAS_DTYPES = (torch.float32, torch.bfloat16)


def _check_same_device(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"Expected {name} to be on device {reference.device}, got {tensor.device}."
            )


def _validate_common_decode_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A_log: torch.Tensor | None,
    a: torch.Tensor,
    dt_bias: torch.Tensor | None,
    b: torch.Tensor,
    *,
    output: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int]:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "q, k, and v must each have shape [B, T, H, D] or [1, total_tokens, H, D]."
        )
    if a.dim() != 4:
        raise ValueError("a must have shape [B, T, HV, K].")
    if A_log is not None and A_log.dim() != 1:
        raise ValueError("A_log must have shape [HV].")
    if dt_bias is not None and dt_bias.dim() != 2:
        raise ValueError("dt_bias must have shape [HV, K] when provided.")

    B, T, H, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, HV, V = v.shape

    if (Bk, Tk, Hk, Kk) != (B, T, H, K):
        raise ValueError(
            f"k must match q shape [B, T, H, K], got q={tuple(q.shape)}, k={tuple(k.shape)}."
        )
    if (Bv, Tv) != (B, T):
        raise ValueError(
            f"v must match q in batch/time dims, got q={tuple(q.shape)}, v={tuple(v.shape)}."
        )
    if HV % H != 0:
        raise ValueError(f"Expected HV to be divisible by H, got HV={HV}, H={H}.")
    if a.shape != (B, T, HV, K):
        raise ValueError(f"Expected a shape {(B, T, HV, K)}, got a={tuple(a.shape)}.")
    if A_log is not None and A_log.numel() != HV:
        raise ValueError(f"A_log must have {HV} elements, got {A_log.numel()}.")
    if dt_bias is not None and dt_bias.shape != (HV, K):
        raise ValueError(
            f"dt_bias must have shape {(HV, K)} when provided, got {tuple(dt_bias.shape)}."
        )

    if q.dtype not in _SUPPORTED_QKV_DTYPES:
        raise NotImplementedError(
            f"q/k/v dtype must be float16 or bfloat16, got {q.dtype}."
        )
    if A_log is not None and A_log.dtype != torch.float32:
        raise ValueError(f"A_log must be float32, got A_log={A_log.dtype}.")
    if dt_bias is not None and dt_bias.dtype not in _SUPPORTED_DT_BIAS_DTYPES:
        raise ValueError(
            f"dt_bias must be float32 or bfloat16 when provided, got dt_bias={dt_bias.dtype}."
        )

    same_device_tensors = {
        "k": k,
        "v": v,
        "a": a,
        "b": b,
    }
    if A_log is not None:
        same_device_tensors["A_log"] = A_log
    if dt_bias is not None:
        same_device_tensors["dt_bias"] = dt_bias
    _check_same_device(q, **same_device_tensors)

    if output is not None:
        if output.shape != (B, T, HV, V):
            raise ValueError(
                f"Expected output shape {(B, T, HV, V)}, got {tuple(output.shape)}."
            )
        if output.dtype not in _SUPPORTED_OUTPUT_DTYPES:
            raise NotImplementedError(
                f"Unsupported output dtype {output.dtype}. "
                f"Supported dtypes: {_SUPPORTED_OUTPUT_DTYPES}."
            )
        if output.device != q.device:
            raise ValueError(
                f"Expected output to be on device {q.device}, got {output.device}."
            )

    return B, T, H, K, HV, V


def _run_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor | None,
    a: torch.Tensor,
    dt_bias: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    b: torch.Tensor,
    *,
    state_indices: torch.Tensor | None,
    scale: float,
    output: torch.Tensor | None,
    use_qk_l2norm: bool | None,
    lower_bound: float | None,
    use_gate_in_kernel: bool | None,
    use_lower_bound: bool | None,
    apply_beta_sigmoid: bool | None,
    allow_neg_eigval: bool | None,
    use_initial_state: bool,
    state_v_first: bool,
    is_varlen: bool,
    store_final_state: bool,
    inplace_final_state: bool,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size = q.shape[0]
    num_v_heads = v.shape[2]
    head_dim_v = v.shape[3]

    if output is None:
        output = torch.empty(
            (batch_size, q.shape[1], num_v_heads, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
    if use_initial_state or inplace_final_state:
        if state is None:
            raise ValueError(
                "state must be provided when use_initial_state or "
                "inplace_final_state is enabled."
            )
        state_is_contiguous = state.is_contiguous()
        state_kernel = state if state_is_contiguous else state.contiguous()
    else:
        state_is_contiguous = True
        pool_size = 1 if state_indices is None else int(state_indices.shape[0])
        if state_v_first:
            state_shape = (pool_size, num_v_heads, head_dim_v, q.shape[-1])
        else:
            state_shape = (pool_size, num_v_heads, q.shape[-1], head_dim_v)
        state_kernel = torch.empty(state_shape, dtype=state_dtype, device=q.device)
    output_result, final_state_result = (
        kda_decode_tilelang.run_gated_delta_rule_decode_vk_fp32(
            q=q,
            k=k,
            v=v,
            state=state_kernel,
            state_indices=state_indices,
            A_log=A_log,
            g=a,
            dt_bias=dt_bias,
            b=b,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            output=output,
            scale=float(scale),
            lower_bound=0.0 if lower_bound is None else float(lower_bound),
            use_qk_l2norm=True if use_qk_l2norm is None else bool(use_qk_l2norm),
            is_varlen=bool(is_varlen),
            inplace_final_state=bool(inplace_final_state),
            is_beta_headwise=b.dim() == 4,
            is_continuous_batching=state_indices is not None,
            is_spec_decoding=num_accepted_tokens is not None,
            store_final_state=bool(store_final_state),
            has_dt_bias=dt_bias is not None,
            use_gate_in_kernel=True
            if use_gate_in_kernel is None
            else bool(use_gate_in_kernel),
            use_lower_bound=False if use_lower_bound is None else bool(use_lower_bound),
            apply_beta_sigmoid=False
            if apply_beta_sigmoid is None
            else bool(apply_beta_sigmoid),
            allow_neg_eigval=False
            if allow_neg_eigval is None
            else bool(allow_neg_eigval),
            state_v_first=bool(state_v_first),
            use_initial_state=bool(use_initial_state),
        )
    )

    if not state_is_contiguous:
        state.copy_(state_kernel)
        if inplace_final_state and final_state_result is not None:
            final_state_result = state

    return output_result, final_state_result


@mate_api
def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor | None,
    A_log: torch.Tensor | None,
    a: torch.Tensor,
    dt_bias: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    b: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    scale: float | None = None,
    output: torch.Tensor | None = None,
    use_qk_l2norm: bool | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool | None = None,
    use_lower_bound: bool | None = None,
    apply_beta_sigmoid: bool | None = None,
    allow_neg_eigval: bool | None = None,
    use_initial_state: bool | None = None,
    state_v_first: bool | None = None,
    is_varlen: bool | None = None,
    store_final_state: bool = False,
    inplace_final_state: bool = False,
    state_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Unified Gated Delta Rule Decode API.

    Args:
        q (torch.Tensor):
            Query of shape ``[B, T, H, K] or [1, total_tokens, H, K]``.
        k (torch.Tensor):
            Key of shape ``[B, T, H, K] or [1, total_tokens, H, K]``. Must match ``q`` in shape and dtype.
        v (torch.Tensor):
            Value of shape ``[B, T, HV, V] or [1, total_tokens, HV, K]``. Must match ``q`` in batch/time dims.
        state (Optional[torch.Tensor]):
            State buffer in VK layout. Shape is ``[B_or_pool, HV, V, K]`` when
            `state_v_first` is `True`, otherwise
            ``[B_or_pool, HV, K, V]``. Supported dtypes are ``torch.float32``
            and ``torch.bfloat16``. It may be `None` when initial-state reads and
            in-place writes are disabled. The returned final state uses the same
            dtype. BF16 state is widened for arithmetic and rounded back to BF16
            after each decoded token, matching repeated single-token decode.
        A_log (Optional[torch.Tensor]):
            Optional log decay of shape ``[HV]``. When `None`, the backend uses zeros.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, T, HV, K]``.
        dt_bias (Optional[torch.Tensor]):
            Optional decay bias of shape ``[HV, K]``. When ``None``, the backend runs with
            `has_dt_bias=False` and supplies a dummy tensor internally. Supported dtypes are
            ``torch.float32`` and ``torch.bfloat16``.
        num_accepted_tokens (Optional[torch.Tensor]):
            Optional speculative-decoding accepted-token counts of shape ``[B]``.
        cu_seqlens (Optional[torch.Tensor]):
            Optional cumulative sequence lengths for varlen mode.
        b (torch.Tensor):
            Update gate of shape ``[B, T, HV]`` or ``[B, T, HV, V]``.
        state_indices (Optional[torch.Tensor]):
            Optional ``[B]`` or ``[B, T]`` int32/int64 mapping batch entries
            and decode steps to a state pool. Every non-negative index must be
            smaller than the state pool size.
        scale (Optional[float]):
            Query scale. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Optional pre-allocated output tensor of shape ``[B, T, HV, V]``.
        use_qk_l2norm (Optional[bool]):
            Whether to L2-normalize q and k in-kernel. ``None`` uses the current default.
        lower_bound (Optional[float]):
            Optional backend lower-bound value. ``None`` uses the current default.
        use_gate_in_kernel (Optional[bool]):
            Optional explicit control of in-kernel gate application.
        use_lower_bound (Optional[bool]):
            Optional explicit control of backend lower-bound logic.
        apply_beta_sigmoid (Optional[bool]):
            Optional explicit control of in-kernel sigmoid application on ``b``.
        allow_neg_eigval (Optional[bool]):
            Optional explicit control of backend negative-eigenvalue handling.
        use_initial_state (Optional[bool]):
            Optional explicit control of whether the backend reads the provided state.
        state_v_first (Optional[bool]):
            State matrix layout selector. ``None`` defaults to `True`.
        is_varlen (Optional[bool]):
            Whether to run the varlen path. ``None`` defaults to whether
            `cu_seqlens` is provided.
        store_final_state (bool):
            Whether to request final-state output from the backend.
        inplace_final_state (bool):
            Whether final-state writes should reuse `state`.
        state_dtype (Optional[torch.dtype]):
            State storage dtype. When omitted, it is inferred from `state`, or
            defaults to `torch.float32` when `state` is `None`. Supported dtypes
            are `torch.float32` and `torch.bfloat16`.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - output: backend output tensor
            - final_state: semantic final-state result, or `None` when not requested
    """
    resolved_state_v_first = True if state_v_first is None else bool(state_v_first)
    resolved_is_varlen = (
        (cu_seqlens is not None) if is_varlen is None else bool(is_varlen)
    )
    resolved_use_initial_state = (
        True if use_initial_state is None else bool(use_initial_state)
    )
    if state_dtype is None:
        resolved_state_dtype = torch.float32 if state is None else state.dtype
    else:
        resolved_state_dtype = state_dtype
    if resolved_state_dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError(
            "state_dtype must be torch.float32 or torch.bfloat16, "
            f"got {resolved_state_dtype}."
        )

    _, _, _, K, HV, V = _validate_common_decode_inputs(
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        output=output,
        num_accepted_tokens=num_accepted_tokens,
    )

    if resolved_is_varlen:
        if cu_seqlens is None or cu_seqlens.dim() != 1:
            raise ValueError("cu_seqlens must be a 1D tensor when is_varlen=True.")
        logical_batch_size = cu_seqlens.numel() - 1
    else:
        logical_batch_size = q.shape[0]

    if num_accepted_tokens is not None and (
        num_accepted_tokens.dim() != 1
        or num_accepted_tokens.shape[0] != logical_batch_size
    ):
        raise ValueError(
            "num_accepted_tokens must have shape "
            f"[B={logical_batch_size}], got {tuple(num_accepted_tokens.shape)}."
        )

    if state_indices is not None:
        if state_indices.dim() not in (1, 2):
            raise ValueError(
                "state_indices must have shape [B] or [B, T], "
                f"got {tuple(state_indices.shape)}."
            )
        if state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "state_indices must have dtype torch.int32 or torch.int64, "
                f"got {state_indices.dtype}."
            )
        if state_indices.shape[0] != logical_batch_size:
            raise ValueError(
                f"state_indices first dimension must be {logical_batch_size}, "
                f"got {state_indices.shape[0]}."
            )
        if state_indices.dim() == 1 and (
            num_accepted_tokens is not None
            or (inplace_final_state and (resolved_is_varlen or q.shape[1] > 1))
        ):
            raise ValueError(
                "1D state_indices only support non-speculative single-token "
                "in-place decode; use [B, T] state_indices for speculative, "
                "multi-token, or varlen in-place decode."
            )
        if state_indices.dim() == 2:
            if state_indices.shape[1] < 1:
                raise ValueError("state_indices must contain at least one column.")
            if (
                not resolved_is_varlen
                and (inplace_final_state or num_accepted_tokens is not None)
                and state_indices.shape[1] < q.shape[1]
            ):
                raise ValueError(
                    f"state_indices needs at least {q.shape[1]} columns for "
                    "fixed multi-token in-place or speculative decode, "
                    f"got {state_indices.shape[1]}."
                )

    if resolved_use_initial_state and state is None:
        raise ValueError("state must be provided when use_initial_state=True.")
    if inplace_final_state and state is None:
        raise ValueError("state must be provided when inplace_final_state=True.")
    if state is not None:
        if state.dtype not in _SUPPORTED_STATE_DTYPES:
            raise ValueError(
                "state must have dtype torch.float32 or torch.bfloat16, "
                f"got {state.dtype}."
            )
        if state.device != q.device:
            raise ValueError(
                f"Expected state to be on device {q.device}, got {state.device}."
            )
        if state.dtype != resolved_state_dtype:
            raise ValueError(
                f"state has dtype {state.dtype}, but state_dtype="
                f"{resolved_state_dtype}."
            )
        expected_tail = (HV, V, K) if resolved_state_v_first else (HV, K, V)
        if state.dim() != 4 or tuple(state.shape[1:]) != expected_tail:
            raise ValueError(
                "state must have shape [pool, HV, V, K] for V-first or "
                "[pool, HV, K, V] for K-first; "
                f"expected trailing dimensions {expected_tail}, got {tuple(state.shape)}."
            )
        if state_indices is None and state.shape[0] < logical_batch_size:
            raise ValueError(
                f"state pool must contain at least {logical_batch_size} slots, "
                f"got {state.shape[0]}."
            )
        if state_indices is not None and state.shape[0] == 0:
            raise ValueError("state pool must contain at least one slot.")

    return _run_kda_decode(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        b=b,
        state_indices=state_indices,
        scale=K**-0.5 if scale is None else float(scale),
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        use_lower_bound=use_lower_bound,
        apply_beta_sigmoid=apply_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        use_initial_state=resolved_use_initial_state,
        state_v_first=resolved_state_v_first,
        is_varlen=resolved_is_varlen,
        store_final_state=store_final_state,
        inplace_final_state=inplace_final_state,
        state_dtype=resolved_state_dtype,
    )
