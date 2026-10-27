"""Public QSA MQA-score and sparse-attention interfaces.

Core prefill and SplitKV decode share ``qsa_sparse_attention``.
``num_splits=None`` selects the schedule; a positive value overrides it.

The caller prepares valid metadata: request IDs lie in the request batch,
cumulative lengths start at zero, are monotonic, and end at the tensor's row
count, and sequence lengths fit the KV capacity. Shape/dtype/device checks do
not read device contents. Native kernels mask invalid selected IDs and missing
pages without synchronizing the host. Warm up the chosen specialization before
Graph capture; replay reads the current metadata buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from .api_logging import mate_api
from .qsa_kernels.core import physical_attention, sparse_attention
from .qsa_kernels.mqa import mqa_decode, mqa_prefill, mqa_score


@dataclass(frozen=True)
class _QueryMetadata:
    """Flattened query metadata shared by the indexer and core reference."""

    flat: torch.Tensor
    batch_ids: torch.Tensor
    positions: torch.Tensor
    batch_size: int
    row_valid: Optional[torch.Tensor]
    leading_shape: tuple[int, ...]

    def restore(self, flat_value: torch.Tensor) -> torch.Tensor:
        """Restore a value from ``[rows, ...]`` to the query's input layout."""

        if len(self.leading_shape) == 2:
            return flat_value.view(*self.leading_shape, *flat_value.shape[1:])
        return flat_value


def _check_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")


def _check_same_device(name: str, value: torch.Tensor, device: torch.device) -> None:
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}.")


def _validate_int_vector(value, *, name, length, device):
    _check_tensor(name, value)
    _check_same_device(name, value, device)
    if value.ndim != 1 or value.numel() != length:
        raise ValueError(f"{name} must have shape [{length}].")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be int32 or int64.")


def _validate_cu_seqlens(value, *, name, device):
    _check_tensor(name, value)
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError(f"{name} must be a vector with at least two entries.")
    _validate_int_vector(value, name=name, length=value.numel(), device=device)


def _query_metadata(
    query,
    *,
    cu_seqlens_q=None,
    token_to_req=None,
    num_requests=None,
    query_positions=None,
    seq_lens_q=None,
):
    """Use caller metadata directly; derive only omitted request/position data."""
    _check_tensor("query", query)
    if query.ndim not in (3, 4):
        raise ValueError("query must have shape [B,Q,H,D] or [total_q,H,D].")
    device = query.device
    if query.ndim == 4:
        batch, query_len = query.shape[:2]
        if cu_seqlens_q is not None or token_to_req is not None:
            raise ValueError("cu_seqlens_q/token_to_req require rank-3 queries.")
        if num_requests is not None and num_requests != batch:
            raise ValueError("num_requests does not match the query batch.")
        try:
            flat = query.view(batch * query_len, *query.shape[2:])
        except RuntimeError as error:
            raise ValueError(
                "query batch/query axes must form a zero-copy row view; "
                "pass a framework-prepared [total_q,H,D] query."
            ) from error
        batch_ids = torch.arange(
            batch, device=device, dtype=torch.int32
        ).repeat_interleave(query_len)
        leading_shape = (batch, query_len)
    else:
        rows = query.shape[0]
        if token_to_req is not None and cu_seqlens_q is not None:
            raise ValueError("token_to_req and cu_seqlens_q are mutually exclusive.")
        flat = query
        leading_shape = (rows,)
        if token_to_req is not None:
            _validate_int_vector(
                token_to_req, name="token_to_req", length=rows, device=device
            )
            if num_requests is None:
                raise ValueError("num_requests is required with token_to_req.")
            batch = int(num_requests)
            if batch <= 0:
                raise ValueError("num_requests must be positive.")
            batch_ids = token_to_req
        elif cu_seqlens_q is not None:
            if num_requests is not None:
                raise ValueError("num_requests is inferred from cu_seqlens_q.")
            _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", device=device)
            batch = cu_seqlens_q.numel() - 1
            counts = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            batch_ids = torch.repeat_interleave(
                torch.arange(batch, device=device, dtype=torch.int32),
                counts,
                output_size=rows,
            )
        else:
            if num_requests not in (None, 1):
                raise ValueError(
                    "multiple requests require token_to_req or cu_seqlens_q."
                )
            batch = 1
            batch_ids = torch.zeros(rows, device=device, dtype=torch.int32)

    rows = flat.shape[0]
    positions = None
    if query_positions is not None:
        _check_tensor("query_positions", query_positions)
        _check_same_device("query_positions", query_positions, device)
        if query_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("query_positions must be int32 or int64.")
        if tuple(query_positions.shape) not in (leading_shape, (rows,)):
            raise ValueError("query_positions must match the query rows.")
        try:
            positions = query_positions.view(rows)
        except RuntimeError as error:
            raise ValueError(
                "query_positions must form a zero-copy row view."
            ) from error
    if positions is None or seq_lens_q is not None:
        # Padding uses the row ordinal; cached causal positions have a KV offset.
        if query.ndim == 4:
            ordinals = torch.arange(query_len, device=device, dtype=torch.int32).repeat(
                batch
            )
        elif cu_seqlens_q is not None:
            ordinals = (
                torch.arange(rows, device=device, dtype=torch.int32)
                - cu_seqlens_q[batch_ids]
            )
        elif token_to_req is None:
            ordinals = torch.arange(rows, device=device, dtype=torch.int32)
        else:
            counts = torch.zeros(batch, device=device, dtype=torch.int32)
            counts.scatter_add_(
                0, batch_ids.long(), torch.ones_like(batch_ids, dtype=torch.int32)
            )
            order = torch.argsort(batch_ids, stable=True)
            starts = counts.cumsum(0) - counts
            ordinals = torch.empty(rows, device=device, dtype=torch.int32)
            ordinals[order] = (
                torch.arange(rows, device=device, dtype=torch.int32)
                - starts.repeat_interleave(counts, output_size=rows)
            ).to(torch.int32)
        if positions is None:
            positions = ordinals

    row_valid = None
    if seq_lens_q is not None:
        _validate_int_vector(seq_lens_q, name="seq_lens_q", length=batch, device=device)
        row_valid = (positions >= 0) & (ordinals < seq_lens_q[batch_ids])
    return _QueryMetadata(flat, batch_ids, positions, batch, row_valid, leading_shape)


def _qsa_arguments(operation, parameters):
    """Package arguments without preparing or reading device data."""
    args = SimpleNamespace(operation=operation, **parameters)
    if args.backend not in ("reference", "tilelang"):
        raise ValueError(f"unsupported QSA backend {args.backend!r}.")
    query = parameters.get("query", parameters.get("q_idx", parameters.get("q")))
    if (
        args.backend == "tilelang"
        and isinstance(query, torch.Tensor)
        and query.device.type == "musa"
        and torch.musa.get_device_capability(query.device) != (3, 1)
    ):
        raise ValueError("TileLang QSA currently supports MP31 only.")
    args._prepare = _prepare_qsa_arguments
    return args


def _prepare_qsa_arguments(args):
    """Validate the kernel ABI and create zero-copy views."""
    query = args.query if args.operation == "core" else getattr(args, "q_idx", None)
    if query is not None and args.backend == "tilelang":
        for name in (
            "token_to_req",
            "query_positions",
            "seq_lens_q",
            "seq_lens_k",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "block_table",
        ):
            tensor = getattr(args, name, None)
            if tensor is not None:
                _check_tensor(name, tensor)
                if tensor.dtype != torch.int32:
                    raise TypeError(
                        f"{name} must be int32 on the TileLang path; prepare it in the caller."
                    )
    operation = args.operation
    if operation in ("score", "core"):
        query = args.query if operation == "core" else args.q_idx
        num_requests = args.num_requests
        if args.token_to_req is not None and num_requests is None:
            # Request count is layout metadata, never a reduction of device IDs.
            if getattr(args, "seq_lens_k", None) is not None:
                num_requests = args.seq_lens_k.numel()
            elif args.block_table is not None:
                num_requests = args.block_table.shape[0]
            elif getattr(args, "cu_seqlens_k", None) is not None:
                num_requests = args.cu_seqlens_k.numel() - 1
            else:
                kv = args.key if operation == "core" else args.block_keys
                if operation == "core" and kv.ndim == 3:
                    raise ValueError(
                        "num_requests is required with flattened KV and token_to_req."
                    )
                num_requests = kv.shape[0] if kv.ndim >= 3 else 1
        args.metadata = _query_metadata(
            query,
            cu_seqlens_q=args.cu_seqlens_q,
            token_to_req=args.token_to_req,
            num_requests=num_requests,
            query_positions=args.query_positions,
            seq_lens_q=args.seq_lens_q,
        )
    if operation == "core":
        rows = int(args.metadata.flat.shape[0])
        args.gate_rows = args.gate
        if args.gate is not None:
            _check_tensor("gate", args.gate)
            _check_same_device("gate", args.gate, query.device)
            gate = args.gate
            expected = (*args.metadata.leading_shape, query.shape[-2])
            if tuple(gate.shape) in ((*expected, 1), (rows, query.shape[-2], 1)):
                gate = gate.squeeze(-1)
            if tuple(gate.shape) not in (expected, (rows, query.shape[-2])):
                raise ValueError("gate must have shape [B,Q,Hq] or [total_q,Hq].")
            if gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError("gate must be FP16, BF16, or FP32.")
            try:
                args.gate_rows = gate.view(rows, query.shape[-2])
            except RuntimeError as error:
                raise ValueError("gate must form a zero-copy row view.") from error
        for name in ("token_indices", "block_indices"):
            tensor = getattr(args, name)
            if tensor is None and name == "block_indices":
                args.blocks = None
                continue
            _check_tensor(name, tensor)
            _check_same_device(name, tensor, query.device)
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must be int32 or int64.")
            if tensor.dtype != torch.int32 and (
                args.backend == "tilelang" or name == "block_indices"
            ):
                raise TypeError(f"{name} must be int32 on the TileLang path.")
            if tensor.ndim not in (2, 3):
                raise ValueError(f"{name} must have rank 2 or 3.")
            if tensor.shape[-1] <= 0:
                raise ValueError(f"{name} must contain at least one slot per query.")
            if (
                tensor.ndim == 3
                and tuple(tensor.shape[:2]) == args.metadata.leading_shape
            ):
                try:
                    view = tensor.view(rows, tensor.shape[-1])
                except RuntimeError as error:
                    raise ValueError(
                        f"{name} batch/query axes must be viewable as rows; "
                        "pass a framework-prepared [total_q, K] tensor."
                    ) from error
            elif tensor.ndim == 2 and tensor.shape[0] == rows:
                view = tensor
            else:
                raise ValueError(f"{name} must match the query rows.")
            setattr(args, "indices" if name == "token_indices" else "blocks", view)
        if (
            args.blocks is not None
            and args.indices.shape[-1] != args.blocks.shape[-1] * 4 + 3
        ):
            raise ValueError(
                "token_indices width must equal block_indices width * 4 + 3."
            )
        key = args.key
        _check_tensor("key", key)
        _check_tensor("value", args.value)
        if key.ndim not in (3, 4):
            raise ValueError("key/value must be rank-3 or rank-4 tensors.")
        batch = args.metadata.batch_size
        if args.block_table is not None:
            _check_tensor("block_table", args.block_table)
            if key.ndim != 4 or args.block_table.ndim != 2:
                raise ValueError(
                    "paged KV requires rank-4 KV and a rank-2 block_table."
                )
            if args.page_size is None:
                args.page_size = int(key.shape[1])
            default_length = int(args.block_table.shape[1]) * int(args.page_size)
        elif key.ndim == 4:
            default_length = int(key.shape[1])
        elif args.cu_seqlens_k is not None:
            _validate_cu_seqlens(
                args.cu_seqlens_k,
                name="cu_seqlens_k",
                device=query.device,
            )
            if args.cu_seqlens_k.numel() != batch + 1:
                raise ValueError("cu_seqlens_k must contain B + 1 entries.")
            default_length = args.cu_seqlens_k[1:] - args.cu_seqlens_k[:-1]
        else:
            default_length = int(key.shape[0])
        if args.seq_lens_k is None:
            args.lengths_k = (
                default_length
                if isinstance(default_length, torch.Tensor)
                else torch.full(
                    (batch,), default_length, dtype=torch.int32, device=query.device
                )
            )
        else:
            _validate_int_vector(
                args.seq_lens_k, name="seq_lens_k", length=batch, device=query.device
            )
            args.lengths_k = args.seq_lens_k
    return args


@mate_api
def qsa_indexer_mqa_score(
    q_idx: torch.Tensor,
    block_keys: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    token_to_req: Optional[torch.Tensor] = None,
    num_requests: Optional[int] = None,
    query_positions: Optional[torch.Tensor] = None,
    seq_lens_q: Optional[torch.Tensor] = None,
    block_size: int = 4,
    scale: Optional[float] = None,
    score_scale: Optional[float] = None,
    backend: str = "tilelang",
) -> torch.Tensor:
    """Compute QSA MQA block scores.

    Parameters
    ----------
    q_idx:
        Projected indexer queries, ``[B, Q, 4, 128]``. Varlen
        ``[total_q, 4, 128]`` requires explicit reference execution; use
        ``qsa_mqa_prefill`` for native packed rows.
    block_keys:
        Logical compressed keys ``[B, N_B, 128]``.  With ``block_table`` this
        instead is a paged cache ``[pages, page_size, 1, 128]`` (the head axis
        may be omitted), supported only by the explicit reference. Native
        paged scoring uses ``qsa_mqa_decode``.
    query_positions:
        Local request positions.  If omitted, positions are inferred from the
        padded/varlen query layout.
    seq_lens_q:
        Optional request lengths.  Padded rows beyond the length are masked.
    score_scale:
        Optional packed-row denominator.  ``scale`` and
        ``score_scale`` are mutually exclusive; the default is
        ``scale=1/sqrt(D)``.
    backend:
        ``"reference"`` forces the FP32 PyTorch oracle; ``"tilelang"``
        requires the supported MUSA ABI (sliced inputs are accepted);
        TileLang is the default; unsupported native combinations raise an error.
        This entrypoint does not select Top-K.

    Returns
    -------
    torch.Tensor
        FP32 scores with shape ``[B, Q, N_B]`` or ``[total_q, N_B]``.  A block
        whose last token is not visible has score ``-inf``.

    Raises:
        ValueError: Inconsistent layouts, metadata, scaling, or backend options.
        TypeError: Unsupported argument types or metadata dtypes.
    """

    return mqa_score(_qsa_arguments("score", locals()))


@mate_api
def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    *,
    scale: Optional[float] = None,
    backend: str = "tilelang",
) -> torch.Tensor:
    """Compute packed-prefill MQA logits.

    Parameters:
        q: Queries with shape ``[rows,heads,dim]``.
        k: Shared keys with shape ``[keys,dim]`` or ``[keys,1,dim]``.
        row_starts: Inclusive visible-key start per query row.
        row_ends: Exclusive visible-key end per query row.
        score_scale: Positive score denominator, exclusive with ``scale``.
        scale: Positive score multiplier; defaults to ``1/sqrt(dim)``.
        backend: ``tilelang`` (default) or explicit ``reference``.

    Returns:
        torch.Tensor: FP32 ``[rows,keys]`` scores with ``-inf`` outside each
        visible interval. TileLang supports BF16/FP16 H4/D128 inputs on MUSA.

    Raises:
        ValueError: Invalid shape, scaling, or unsupported explicit backend.
        TypeError: Unsupported argument types or metadata dtypes.

    ``row_starts``/``row_ends`` identify the visible compressed-key interval
    for each query row.  ``score_scale`` uses the denominator
    convention; ``scale`` is an optional multiplier used by the MATE API.
    """

    return mqa_prefill(_qsa_arguments("prefill", locals()))


@mate_api
def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    *,
    scale: Optional[float] = None,
    backend: str = "tilelang",
) -> torch.Tensor:
    """Compute paged-decode MQA logits with packed-row metadata.

    Parameters:
        q: Queries, ``[B,heads,dim]`` or ``[B,1,heads,dim]``.
        k_cache: Compressed paged keys, ``[pages,page_size,1,dim]``; the shared
            head axis may be omitted.
        page_table: Logical-to-physical page IDs with shape ``[B,max_pages]``.
        context_lens: Visible compressed-key counts, one per request.
        max_model_len: Output score width in compressed-key units.
        score_scale: Positive score denominator, exclusive with ``scale``.
        scale: Positive score multiplier; defaults to ``1/sqrt(dim)``.
        backend: ``tilelang`` (default) or explicit ``reference``.

    Returns:
        torch.Tensor: FP32 ``[B,max_model_len]`` scores; invisible keys have
        score ``-inf``. TileLang supports BF16/FP16 H4/D128 inputs on MUSA.

    Raises:
        ValueError: Invalid shape, scaling, or backend options.
        TypeError: Unsupported argument types or metadata dtypes.
    """

    return mqa_decode(_qsa_arguments("decode", locals()))


qsa_mqa = qsa_mqa_prefill


@mate_api
def qsa_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: Optional[torch.Tensor] = None,
    *,
    block_indices: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    token_to_req: Optional[torch.Tensor] = None,
    num_requests: Optional[int] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seq_lens_q: Optional[torch.Tensor] = None,
    seq_lens_k: Optional[torch.Tensor] = None,
    query_positions: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    scale: Optional[float] = None,
    gate: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    return_lse: bool = False,
    backend: str = "tilelang",
    warp_specialized: bool = False,
    assume_compact_blocks: bool = False,
    out: Optional[torch.Tensor] = None,
    selected_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    compute_dtype: torch.dtype = torch.bfloat16,
    lse: Optional[torch.Tensor] = None,
):
    """Run QSA core attention over a per-query compact token set.

    Parameters:
        query: ``[B,Q,Hq,D]`` or flattened ``[total_q,Hq,D]`` queries.
        key: Padded or paged KV storage; flattened-varlen requires reference.
        value: Values with the same layout, dtype, and device as ``key``.
        token_indices: Framework-prepared int32 request-local token IDs; ``-1`` pads.
        selected_kv_indices: Alternative int32 ``[rows, selected_tokens]`` physical
            IDs into flattened BF16 K/V. Mutually exclusive with ``token_indices``.
            The caller handles causal filtering and page translation. Duplicates
            are retained; negative and out-of-capacity IDs are masked.
        softmax_scale: Alias for ``scale``; do not supply both.
        k_scale: Must be 1 for BF16 K/V.
        v_scale: Must be 1 for BF16 K/V.
        compute_dtype: BF16 for physical indices; FP8 is not enabled.
        lse: Optional FP32 ``[rows, Hq]`` natural-log LSE output for physical
            indices. Written even when ``return_lse=False``; empty rows are
            ``-inf``. The supplied buffer is bound directly to the merge kernel.
        block_indices: Optional four-token blocks that produced token indices.
        query_positions: Request-local causal positions; inferred if omitted.
        seq_lens_k: Actual KV lengths, bounded by the supplied cache capacity.
        block_table: Optional logical-to-physical page mapping.
        num_splits: ``None`` chooses a schedule; ``1`` disables SplitKV.
        gate: Optional per-query/head gate, applied through sigmoid.
        return_lse: Also return FP32 natural-log log-sum-exp. TileLang exports
            native softmax statistics; paged inputs use a SplitKV merge.
        backend: ``tilelang`` (default) or explicit ``reference``.
        out: Optional writable, nonoverlapping positive-stride output view
            matching the query shape, dtype, and device. Batch/query axes must
            form a zero-copy row view. Input/output overlap is rejected.

    Returns:
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Query-shaped output,
        or ``(output, lse)`` when requested. LSE has shape ``query.shape[:-1]``.
        The output is exactly ``out`` when supplied.

    Raises:
        ValueError: Invalid shape, output layout, scaling, incompatible hints,
            or unsupported explicit backend.
        TypeError: Unsupported argument types or metadata dtypes.

    TileLang consumes int32 indices, positions, lengths and page tables without
    dtype conversion. Padded batch/query axes of query, indices, gate and out
    must collapse to zero-copy row views; otherwise pass framework-prepared
    flattened tensors. Indices use ``-1`` padding. The same list is shared by
    all Q heads; keys/values are grouped
    by ``Hq // Hkv`` (12 for the Qwen configuration).  Contiguous padded KV,
    flattened varlen KV, and paged KV are accepted by the reference path.
    Native nonpaged KV requires padded rank-4 queries and ``num_splits=1``;
    native flattened queries and SplitKV require paged KV.

    ``out`` may be a nonoverlapping strided view with the query shape,
    dtype and device. The returned output aliases that exact view; unused
    storage around a slice is preserved. Strides are runtime kernel parameters.

    ``num_splits=None`` uses the measured prefill/decode heuristic. A positive
    integer selects that split count explicitly; ``1`` disables SplitKV.

    ``block_indices`` is an optional optimization hint for the canonical QSA
    layout.  It must be the 4-token block selection that produced
    ``token_indices``.  When both are supplied, official-width paged prefill
    transfers one encoded descriptor per block and derives the current tail
    from query metadata.  Omitting it preserves arbitrary selected-token
    semantics and the original dispatch path.
    ``assume_compact_blocks`` is an opt-in producer contract: set it only when
    the indexer guarantees valid block ids occupy a prefix of each row,
    followed by `-1` padding, and every selected block is causally visible. It
    selects compact dynamic prefill without scanning or reading back a proof.
    TileLang dispatch accepts BF16/FP16 Q/K/V with D256, positive divisible
    Hq/Hkv and GQA groups up to 16. Q, K, V, indices, metadata and ``out`` may
    be sliced views with storage offsets. Nonoverlapping output is required;
    the returned tensor is exactly ``out`` when supplied. Head geometry, dtype
    and algorithm tiles specialize the kernels; logical lengths and concrete
    strides are runtime arguments, with int64 stride arithmetic.
    Supported head shapes use TileLang by default. Unsupported combinations
    raise an error; the PyTorch reference requires ``backend="reference"``.

    Vector K/V gathers require unit feature stride and 16-byte alignment;
    unsupported layouts raise an error. Aligned outer K/V slices retain their
    independent runtime strides. The regular prefill path requires contiguous
    K/V; its WS variant supports aligned outer slices. Q/O head padding stays
    inside the shared-memory compute tile. Native paths bind ``out`` directly;
    dense stores retain TME and sliced stores use LSU. Gate/output overlap and
    input/output overlap raise errors.

    The structured TME prefill and WS specializations retain canonical
    24/2/256 geometry; unsupported optional hints use the ordinary path. SplitKV
    accumulates and merges in FP32; partial output storage can use the input
    dtype. Its fused merge applies
    the gate in that merge; direct paths use an in-place TileLang post-op.
    Native logical and physical paths export LSE without repeating QK or PV,
    and support public MUSA Graph capture/replay after warmup.
    ``warp_specialized`` is an opt-in MUSA prefill experiment that splits QK
    and PV consumer groups and uses a two-stage K/V/score ring; it is not
    enabled by default.
    For the canonical short-context structured prefill window, metadata may
    additionally order complete physical descriptors to improve the following
    sparse gather; that guard is disabled outside the measured window.
    """

    if scale is not None and softmax_scale is not None:
        raise ValueError("scale and softmax_scale are mutually exclusive.")
    if softmax_scale is not None:
        scale = softmax_scale
    args = _qsa_arguments("core", locals())
    if selected_kv_indices is not None:
        if (
            token_indices is not None
            or any(
                tensor is not None
                for tensor in (
                    block_indices,
                    cu_seqlens_q,
                    token_to_req,
                    num_requests,
                    cu_seqlens_k,
                    seq_lens_q,
                    seq_lens_k,
                    query_positions,
                    block_table,
                    page_size,
                    gate,
                )
            )
            or warp_specialized
            or assume_compact_blocks
        ):
            raise ValueError(
                "physical indices cannot be combined with logical metadata or gate."
            )
        if backend != "tilelang":
            raise ValueError("physical indices require backend='tilelang'.")
        return physical_attention(
            query,
            key,
            value,
            selected_kv_indices,
            softmax_scale=scale,
            k_scale=k_scale,
            v_scale=v_scale,
            compute_dtype=compute_dtype,
            out=out,
            lse=lse,
            return_lse=return_lse,
            num_splits=num_splits,
        )
    if token_indices is None:
        raise ValueError("provide token_indices or selected_kv_indices.")
    if (
        lse is not None
        or k_scale != 1.0
        or v_scale != 1.0
        or compute_dtype != torch.bfloat16
    ):
        raise ValueError(
            "lse and compute/quantization options require physical indices."
        )
    return sparse_attention(args)


__all__ = [
    "qsa_indexer_mqa_score",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "qsa_mqa",
    "qsa_sparse_attention",
]
