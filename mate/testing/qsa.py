"""PyTorch QSA correctness reference, independent of kernel dispatch and scheduling."""

import math
from typing import Optional

import torch


def paged_kv_fixture(key, value, page_size):
    """Pack test KV into permuted pages, with a poisoned unused physical page."""
    batch, length, heads, dim = key.shape
    pages_per_request = (length + page_size - 1) // page_size
    table = (
        torch.randperm(batch * pages_per_request, device=key.device)
        .reshape(batch, pages_per_request)
        .to(torch.int32)
        + 1
    )
    cache_shape = (table.numel() + 1, page_size, heads, dim)
    cache_k = key.new_full(cache_shape, torch.nan)
    cache_v = value.new_full(cache_shape, torch.nan)
    for request in range(batch):
        for page in range(pages_per_request):
            start = page * page_size
            stop = min(start + page_size, length)
            cache_k[table[request, page], : stop - start] = key[request, start:stop]
            cache_v[table[request, page], : stop - start] = value[request, start:stop]
    unused = torch.full((batch, 1), -1, device=key.device, dtype=torch.int32)
    return cache_k, cache_v, torch.cat((table, unused), dim=1)


def _gather_paged_kv(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    token_indices: torch.Tensor,
    batch_ids: torch.Tensor,
    lengths: torch.Tensor,
    *,
    block_table: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if key.ndim != 4 or key.shape[2] <= 0:
        raise ValueError("paged key/value must have shape [pages, page_size, Hkv, D].")
    if value is not None and value.shape != key.shape:
        raise ValueError("paged key/value shapes must match.")
    if block_table.shape[1] * page_size <= 0:
        raise ValueError("block_table must expose at least one page.")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_table must be int32 or int64.")
    if block_table.device != key.device:
        raise ValueError("block_table must share the KV device.")
    if bool(torch.any(block_table < -1).item()) or bool(
        torch.any(block_table >= key.shape[0]).item()
    ):
        raise ValueError("block_table contains a physical page outside the cache.")
    nonnegative_lengths = lengths.clamp_min(0)
    needed_pages = (
        nonnegative_lengths > 0
        if page_size > torch.iinfo(nonnegative_lengths.dtype).max
        else nonnegative_lengths // page_size + (nonnegative_lengths % page_size != 0)
    )
    page_numbers = torch.arange(
        block_table.shape[1], device=block_table.device, dtype=torch.int64
    ).unsqueeze(0)
    accessed = page_numbers < needed_pages.unsqueeze(1)
    if bool(torch.any((block_table < 0) & accessed).item()):
        raise ValueError("block_table is missing a physical page for a valid KV token.")
    if key.shape[1] != page_size:
        raise ValueError(
            f"page_size={page_size} must equal the cache's physical page dimension {key.shape[1]}."
        )
    safe = token_indices.clamp_min(0)
    page_id = torch.div(safe, page_size, rounding_mode="floor")
    offset = torch.remainder(safe, page_size)
    page_id = page_id.clamp_max(block_table.shape[1] - 1)
    physical = block_table.to(torch.int64)[batch_ids[:, None], page_id]
    physical = physical.clamp(0, key.shape[0] - 1)
    gathered_k = key[physical, offset]
    gathered_v = value[physical, offset] if value is not None else None
    return gathered_k, gathered_v


def _gather_contiguous_kv(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    token_indices: torch.Tensor,
    batch_ids: torch.Tensor,
    *,
    cu_seqlens_k: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    safe = token_indices.clamp_min(0)
    if key.ndim == 4:
        safe = safe.clamp_max(key.shape[1] - 1)
        return key[batch_ids[:, None], safe], (
            value[batch_ids[:, None], safe] if value is not None else None
        )
    if cu_seqlens_k is not None:
        start = cu_seqlens_k[batch_ids[:, None]].to(torch.int64)
        capacity = cu_seqlens_k[batch_ids[:, None] + 1].to(torch.int64) - start
        # Bound local ids before adding the request offset, including int64 padding.
        safe = torch.minimum(safe, (capacity - 1).clamp_min(0))
        global_index = (start + safe).clamp_max(key.shape[0] - 1)
        return key[global_index], value[global_index] if value is not None else None
    safe = safe.clamp_max(key.shape[0] - 1)
    return key[safe], value[safe] if value is not None else None


def sparse_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    token_indices: torch.Tensor,
    *,
    batch_ids: torch.Tensor,
    query_positions: torch.Tensor,
    row_valid: torch.Tensor,
    seq_lens_k: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    page_size: Optional[int],
    scale: float,
    gate: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Compute FP32 selected-token GQA; ``value=None`` requests only LSE."""

    token_indices = token_indices.to(torch.int64)
    q = query.float()
    num_q_heads, dim = q.shape[-2:]
    if num_q_heads <= 0 or dim <= 0:
        raise ValueError("query must have positive head and feature dimensions.")
    if key.shape[-1] != dim or (value is not None and value.shape[-1] != dim):
        raise ValueError("query/key/value head dimensions must match.")
    num_kv_heads = key.shape[-2] if key.ndim in (3, 4) else None
    if num_kv_heads is None or num_kv_heads <= 0:
        raise ValueError("QSA core requires positive Hkv.")
    if num_q_heads % num_kv_heads:
        raise ValueError("QSA core requires Hq divisible by Hkv.")

    if block_table is not None:
        if page_size is None:
            page_size = int(key.shape[1])
        gathered_k, gathered_v = _gather_paged_kv(
            key,
            value,
            token_indices,
            batch_ids,
            seq_lens_k,
            block_table=block_table,
            page_size=page_size,
        )
    else:
        gathered_k, gathered_v = _gather_contiguous_kv(
            key,
            value,
            token_indices,
            batch_ids,
            cu_seqlens_k=cu_seqlens_k,
        )

    token_valid = (
        (token_indices >= 0)
        & (token_indices < seq_lens_k[batch_ids, None])
        & (token_indices <= query_positions[:, None])
        & row_valid[:, None]
    )
    group_size = num_q_heads // num_kv_heads
    # Keep the KV dimension grouped while scoring.  Expanding [K, Hkv, D]
    # to [K, Hq, D] is needlessly large at the QSA width (2051 tokens).
    q_grouped = q.reshape(q.shape[0], num_kv_heads, group_size, dim)
    # Invalid logical IDs can gather a NaN-poisoned unused physical slot.
    # Mask operands before QK/PV: zero probabilities alone do not suppress NaN V.
    gathered_k = torch.where(token_valid[..., None, None], gathered_k, 0).float()
    scores_grouped = torch.einsum("rhgd,rkhd->rhgk", q_grouped, gathered_k) * float(
        scale
    )
    scores = scores_grouped.reshape(q.shape[0], num_q_heads, -1)
    neg_large = torch.full_like(scores, torch.finfo(scores.dtype).min)
    scores = torch.where(token_valid[:, None, :], scores, neg_large)
    max_score = scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores - max_score) * token_valid[:, None, :].float()
    denominator = weights.sum(dim=-1, keepdim=True)
    lse = torch.where(
        token_valid.any(dim=-1, keepdim=True),
        torch.log(denominator.squeeze(-1)) + max_score.squeeze(-1),
        torch.full_like(max_score.squeeze(-1), -float("inf")),
    )

    if value is None:
        return None, lse

    gathered_v = torch.where(token_valid[..., None, None], gathered_v, 0).float()
    weights_grouped = weights.reshape(
        q.shape[0], num_kv_heads, group_size, weights.shape[-1]
    )
    output_grouped = torch.einsum("rhgk,rkhd->rhgd", weights_grouped, gathered_v)
    output = output_grouped.reshape(q.shape[0], num_q_heads, dim)
    output = output / denominator.clamp_min(1.0e-20)

    if gate is not None:
        if gate.device != query.device:
            raise ValueError("gate must share the query device.")
        if gate.ndim == 4 and gate.shape[-1] == 1:
            gate = gate.squeeze(-1)
        if gate.ndim != 2 or gate.shape != (query.shape[0], num_q_heads):
            raise ValueError("gate must have shape [B, Q, Hq] or [total_q, Hq].")
        output = torch.where(
            row_valid[:, None, None],
            output * torch.sigmoid(gate.float()).unsqueeze(-1),
            0,
        )
    return output.to(dtype=query.dtype), lse


def splitkv_merge_reference(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    gate: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge split partials with base-2 LSE; return natural-log statistics."""
    max_lse = partial_lse.max(dim=0).values
    valid = torch.isfinite(max_lse)
    weights = torch.where(
        valid[None] & torch.isfinite(partial_lse),
        torch.exp2(partial_lse - max_lse[None]),
        torch.zeros_like(partial_lse),
    )
    denom = weights.sum(dim=0)
    merged = (
        torch.where(weights[..., None] > 0, partial_output, 0) * weights[..., None]
    ).sum(dim=0)
    merged = torch.where(
        denom[..., None] > 0,
        merged / denom[..., None].clamp_min(1.0e-20),
        torch.zeros_like(merged),
    )
    if gate is not None:
        if tuple(gate.shape) != tuple(partial_output.shape[1:3]):
            raise ValueError("gate must have shape [rows, Hq].")
        merged = torch.where(
            denom[..., None] > 0, merged * torch.sigmoid(gate.float()).unsqueeze(-1), 0
        )
    lse = None
    if return_lse:
        lse = torch.where(
            valid,
            (max_lse + torch.log2(denom.clamp_min(1.0e-20))) / math.log2(math.e),
            -torch.inf,
        )
    return merged.to(output_dtype), lse


def mqa_dense_score_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale: float,
    accumulator_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return unmasked MQA scores for [..., Q, H, D] and [..., N, D]."""
    dots = torch.einsum(
        "...qhd,...nd->...qnh",
        query.to(accumulator_dtype),
        key.to(accumulator_dtype),
    )
    return dots.relu().sum(dim=-1) * float(scale)


def mqa_score_reference(query, block_keys, batch_ids, visible_counts, *, scale):
    """Return FP32 ReLU-summed MQA scores over request-local compressed keys."""
    keys_for_rows = block_keys[batch_ids]
    dots = torch.einsum("rhd,rnd->rhn", query.float(), keys_for_rows.float())
    scores = torch.relu(dots).sum(dim=1) * float(scale)
    columns = torch.arange(block_keys.shape[1], device=query.device)[None, :]
    return scores.masked_fill(columns >= visible_counts[:, None], -float("inf"))


def mqa_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    multiplier: float,
) -> torch.Tensor:
    logits = torch.einsum("rhd,nd->rnh", q.float(), k.float())
    logits = torch.relu(logits).sum(dim=-1) * float(multiplier)
    columns = torch.arange(k.shape[0], device=q.device, dtype=torch.int64).unsqueeze(0)
    valid = (columns >= starts.unsqueeze(1)) & (columns < ends.unsqueeze(1))
    return logits.masked_fill(~valid, -float("inf"))


def mqa_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    *,
    multiplier: float,
) -> torch.Tensor:
    batch, _, dim = q.shape
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    safe_table = page_table.to(torch.int64).clamp(0, k_cache.shape[0] - 1)
    gathered = k_cache[safe_table].reshape(batch, total, dim)
    logits = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    logits = torch.relu(logits).sum(dim=-1) * float(multiplier)
    positions = torch.arange(total, device=q.device, dtype=torch.int64).unsqueeze(0)
    logits = logits.masked_fill(
        positions >= context_lens.to(torch.int64).unsqueeze(1), -float("inf")
    )
    output = torch.full(
        (batch, max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        output[:, :copy_len] = logits[:, :copy_len]
    return output


def expand_qsa_block_indices(
    block_indices, query_positions, *, output_width=None, sequence_lengths=None
):
    """Construct selected-token fixtures from caller-owned four-token blocks."""
    visible = (query_positions.to(torch.int64) + 1).clamp_min(0)
    if sequence_lengths is not None:
        visible = torch.minimum(visible, sequence_lengths.to(torch.int64))
    blocks = block_indices.to(torch.int64)
    offsets = torch.arange(4, device=blocks.device)
    valid = (blocks >= 0) & (blocks < visible[..., None] // 4)
    safe_blocks = torch.where(valid, blocks, 0)
    tokens = (safe_blocks[..., None] * 4 + offsets).flatten(-2)
    tokens = torch.where(valid.repeat_interleave(4, -1), tokens, -1)
    tail_lengths = visible % 4
    tail_offsets = torch.arange(3, device=blocks.device)
    tail = visible[..., None] // 4 * 4 + tail_offsets
    tail = torch.where(tail_offsets < tail_lengths[..., None], tail, -1)
    tokens = torch.cat((tokens, tail), dim=-1)
    slots = torch.arange(tokens.shape[-1], device=blocks.device)
    order = torch.where(tokens >= 0, slots, slots + tokens.shape[-1]).argsort(-1)
    tokens = tokens.gather(-1, order)
    width = tokens.shape[-1] if output_width is None else output_width
    result = torch.full(
        (*tokens.shape[:-1], width), -1, dtype=torch.int32, device=blocks.device
    )
    copied = min(width, tokens.shape[-1])
    result[..., :copied] = tokens[..., :copied].to(torch.int32)
    return result, tail_lengths.to(torch.int32)


def gather_paged_mqa_keys_reference(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: Optional[int],
    batch_ids: torch.Tensor,
    positions: torch.Tensor,
    row_valid: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> torch.Tensor:
    """Materialize logical compressed keys from a paged single-head cache."""

    if cache.ndim == 4:
        if cache.shape[2] != 1:
            raise ValueError(
                "paged indexer cache must have one KV head, "
                f"got shape {tuple(cache.shape)}."
            )
        cache = cache.squeeze(2)
    if cache.ndim != 3:
        raise ValueError(
            "paged indexer cache must have shape [pages, page_size, D] or "
            f"[pages, page_size, 1, D], got {tuple(cache.shape)}."
        )
    if page_size is None:
        page_size = int(cache.shape[1])
    if page_size <= 0 or page_size > cache.shape[1]:
        raise ValueError(
            f"page_size must be in [1, {cache.shape[1]}], got {page_size}."
        )
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must have shape [B, max_pages], got {tuple(block_table.shape)}."
        )
    if block_table.shape[0] != batch_size:
        raise ValueError("block_table batch must match query requests.")
    if block_table.device != cache.device:
        raise ValueError("block_table and paged indexer cache must share a device.")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_table must be int32 or int64.")
    if bool(torch.any(block_table < -1).item()) or bool(
        torch.any(block_table >= cache.shape[0]).item()
    ):
        raise ValueError("block_table contains a physical page outside the cache.")
    # A page can contain more entries than the logical QSA page size.  The
    # first ``page_size`` entries are the ABI-visible compressed keys.
    cache = cache[:, :page_size]
    visible = (
        positions // block_size + (positions % block_size == block_size - 1)
    ).clamp(0, block_table.shape[1] * page_size)
    visible = torch.where(row_valid, visible, torch.zeros_like(visible))
    required_blocks = torch.zeros(
        block_table.shape[0], dtype=torch.int64, device=cache.device
    )
    required_blocks.scatter_reduce_(
        0,
        batch_ids.to(torch.int64),
        visible.to(torch.int64),
        reduce="amax",
        include_self=True,
    )
    required_pages = (required_blocks + page_size - 1) // page_size
    active_pages = (
        torch.arange(block_table.shape[1], device=cache.device)[None, :]
        < required_pages[:, None]
    )
    if bool(((block_table < 0) & active_pages).any().item()):
        raise ValueError("page_table is missing a page for a valid context token.")
    if cache.shape[0] == 0:
        return cache.new_zeros(
            (block_table.shape[0], block_table.shape[1] * page_size, cache.shape[-1])
        )
    # Only unused pages may alias page zero; active pages were checked above.
    safe_table = block_table.to(torch.int64).clamp(0, cache.shape[0] - 1)
    logical = cache[safe_table].reshape(
        block_table.shape[0], block_table.shape[1] * page_size, cache.shape[-1]
    )
    return logical
