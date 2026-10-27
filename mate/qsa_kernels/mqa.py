"""MQA score dispatch; inputs retain their storage, dtype, and strides."""

from __future__ import annotations

import math

import torch

from ..testing.qsa import (
    gather_paged_mqa_keys_reference,
    mqa_decode_reference,
    mqa_prefill_reference,
    mqa_score_reference,
)
from .tilelang.qsa_mqa import (
    _qsa_mqa_decode_kernel,
    _qsa_mqa_prefill_kernel,
    _qsa_mqa_score_packed_kernel,
    _qsa_mqa_score_short_context_kernel,
    _qsa_mqa_score_split_m_kernel,
)

_SPLIT_M_CTA_THRESHOLD = 256


_WIDE_SPLIT_M_CTA_BUCKETS = (
    (32_768, 512),
    (8_192, 256),
    (4_096, 128),
)


_DECODE_SCORE_GN_BUCKETS = (
    (24_576, 1),  # below 96K tokens
    (40_960, 2),  # below 160K tokens
    (81_920, 4),  # below 320K tokens
    (163_840, 8),  # below 640K tokens
)


def _validate_mqa_vector_layout(tensor):
    """Check the aligned feature rows consumed by vector loads and TME."""

    aligned = (
        tensor.stride(-1) == 1
        and tensor.data_ptr() % 16 == 0
        and all(
            stride > 0 and stride * tensor.element_size() % 16 == 0
            for size, stride in zip(tensor.shape[:-1], tensor.stride()[:-1])
            if size > 1
        )
    )
    if aligned:
        return
    raise ValueError(
        "vector loads require unit feature stride and 16-byte alignment, with positive active outer strides; prepare inputs in the caller."
    )


def _qsa_decode_score_config(
    batch: int, rows: int, num_blocks: int
) -> tuple[int, int, int] | None:
    """Return the measured S5000 Q=1 score tile, or ``None`` otherwise.

    BN32 is the smallest legal MP31 SQMMA M tile. The GN crossover points were
    measured from 8K to 1M tokens, including non-power-of-two lengths, with
    direct executable launches and MUSA Events.
    """

    if batch != 1 or rows != 1 or num_blocks <= 0:
        return None
    block_n = 32
    group_n = 16
    for upper_blocks, candidate_group_n in _DECODE_SCORE_GN_BUCKETS:
        if num_blocks < upper_blocks:
            group_n = candidate_group_n
            break
    return block_n, 8, group_n


def _qsa_split_m_ctas(batch: int, rows: int, num_blocks: int) -> int:
    return batch * math.ceil(rows / 64) * math.ceil(num_blocks / 128)


def _qsa_should_split_m(batch: int, rows: int, num_blocks: int) -> bool:
    """Choose split-M once its BM64 grid reaches the measured crossover.

    On S5000, 128 split CTAs made Q256/K4096 only 2.4% faster and was within
    run-to-run noise. At 256 CTAs, Q256/K8192 and Q512/K4096 improved by 1.73x
    and 1.69x, respectively.
    """

    if rows < 128:
        return False
    return _qsa_split_m_ctas(batch, rows, num_blocks) >= _SPLIT_M_CTA_THRESHOLD


def _qsa_split_m_block_m(batch: int, rows: int, num_blocks: int) -> int:
    """Use BM128 at the measured low and steady-state grid buckets.

    BM128/M8 reduced paired latency by 2.0--3.4% for two 256-CTA workloads.
    Above that special case, S5000 measurements select BM128, BM256, and BM512
    at 4096, 8192, and 32768 BM64-grid CTAs. Each wider tile reuses staged K
    across more rows. Single-batch long prefill allows up to 12.5% inactive
    rows in its final tile; the kernel masks those rows in shared memory.
    """

    ctas = _qsa_split_m_ctas(batch, rows, num_blocks)
    if ctas == _SPLIT_M_CTA_THRESHOLD:
        return 128
    for threshold, block_m in _WIDE_SPLIT_M_CTA_BUCKETS:
        padded_rows = math.ceil(rows / block_m) * block_m
        bounded_padding = (
            batch == 1
            and rows >= 4096
            and num_blocks >= 131072
            and padded_rows * 8 <= rows * 9
        )
        if ctas >= threshold and (rows % block_m == 0 or bounded_padding):
            return block_m
    return 64


def _qsa_split_m_use_crossbb_fence(rows: int, num_blocks: int, block_m: int) -> bool:
    """Enable the measured long-context MTCC schedule.

    On S5000 this helps the long-running BM512 mainloop once there are at least
    2K query rows and 64K compressed-key blocks. Smaller row grids regressed,
    so keep the compiler option behind the measured crossover.
    """

    return block_m == 512 and rows >= 2_048 and num_blocks >= 65_536


def run_qsa_mqa_score(
    query: torch.Tensor,
    block_keys: torch.Tensor,
    visible_counts: torch.Tensor,
    *,
    block_n: int = 128,
    scale: float = 128**-0.5,
    split_m: bool | None = None,
    split_full_prefix: bool | None = None,
    full_prefix_blocks: int | None = None,
) -> torch.Tensor:
    """Run the PackGQA score kernel selected for the query-row bucket.

    Long-prefill dispatch can split a common fully visible prefix from the
    final causal tail when ``full_prefix_blocks`` is supplied by the scheduler
    in compressed-block units. The caller guarantees this lower bound does
    not exceed any row's visible count; no device reduction is read on the host.
    """

    if query.ndim != 4 or query.shape[-2:] != (4, 128):
        raise ValueError("query must have shape [B, Q, 4, 128].")
    if block_keys.ndim != 3 or block_keys.shape[0] != query.shape[0]:
        raise ValueError("block_keys must have shape [B, N, 128].")
    if block_keys.shape[-1] != 128:
        raise ValueError("block_keys must have feature dimension 128.")
    batch = int(query.shape[0])
    rows = int(query.shape[1])
    block_m = 32 if rows >= 128 else 8
    if split_m is None:
        split_m = _qsa_should_split_m(batch, rows, int(block_keys.shape[1]))
    if split_m:
        # The packed dispatcher selects the split-M tile; logical lengths
        # and tail masks remain runtime inputs.
        block_m = 32
    if split_full_prefix is None:
        split_full_prefix = split_m and full_prefix_blocks is not None
    elif not isinstance(split_full_prefix, bool):
        raise TypeError("split_full_prefix must be a bool or None.")
    if full_prefix_blocks is not None:
        if isinstance(full_prefix_blocks, bool) or not isinstance(
            full_prefix_blocks, int
        ):
            raise TypeError("full_prefix_blocks must be an integer or None.")
        if full_prefix_blocks < 0:
            raise ValueError("full_prefix_blocks must be non-negative.")
        split_full_prefix = True
    if split_full_prefix and not split_m:
        raise ValueError("split_full_prefix requires split_m=True.")
    group_n = 2
    decode_config = _qsa_decode_score_config(batch, rows, int(block_keys.shape[1]))
    if decode_config is not None and block_n == 128 and not split_m:
        block_n, block_m, group_n = decode_config
    return run_qsa_mqa_score_packed(
        query,
        block_keys,
        visible_counts,
        block_n=block_n,
        block_m=block_m,
        group_n=group_n,
        scale=scale,
        split_m=split_m,
        split_full_prefix=split_full_prefix,
        full_prefix_blocks=full_prefix_blocks,
    )


def _qsa_mqa_split_m_config(
    rows: int,
    num_blocks: int,
    block_m: int,
    scale: float,
    *,
    key_start: int,
    key_count: int,
    skip_mask: bool,
    bounded_tail: bool = False,
):
    """Select measured tile/compiler profiles without caching logical lengths."""
    rounded_blocks = math.ceil(num_blocks / 128) * 128
    whole_range = key_start == 0 and key_count == num_blocks
    schedule_blocks = rounded_blocks if skip_mask else key_count
    profile = (
        "long"
        if _qsa_split_m_use_crossbb_fence(rows, schedule_blocks, block_m)
        else "default"
    )
    config = dict(
        block_n=128,
        block_m=block_m,
        group_n=1,
        scale=float(scale),
        skip_mask=skip_mask,
        compile_profile=profile,
    )
    if rounded_blocks < 131072:
        if block_m >= 128 and 1024 <= rows <= 2048 and 32768 <= rounded_blocks <= 65536:
            # Keep the measured short-prefill M8 tile in the register budget.
            config.update(block_m=128, compile_profile="long")
        return _qsa_mqa_score_short_context_kernel, config

    if not skip_mask and not whole_range and key_count <= 4096:
        # A narrow masked key range needs more independent query CTAs.
        block_m = 128
        config["block_m"] = block_m

    vector_store = (rows >= 2048 and rounded_blocks >= 131072) or (
        skip_mask and rows % block_m != 0
    )
    overlap = (
        not bounded_tail
        and rows % block_m == 0
        and (vector_store or not skip_mask)
        and ((skip_mask and rows >= 8192) or whole_range)
        and (rows == 512 or (rows >= 1024 and rows % 1024 == 0))
    )
    partition = (
        not bounded_tail and rows % block_m != 0 and block_m >= 512 and not skip_mask
    )
    config.update(
        bound_tail_rows=bounded_tail,
        score_epilogue_overlap=overlap,
        partition_rows=partition,
        mainloop_visible_min=block_m <= 512 and not partition,
    )
    if overlap:
        config["block_m"] = 512
    if skip_mask or overlap:
        config["compile_profile"] = "default"
    elif profile == "default" and not bounded_tail:
        config["compile_profile"] = "combine_fop"
    return _qsa_mqa_score_split_m_kernel, config


def run_qsa_mqa_score_packed(
    query: torch.Tensor,
    block_keys: torch.Tensor,
    visible_counts: torch.Tensor,
    *,
    block_n: int = 128,
    block_m: int = 32,
    group_n: int = 2,
    scale: float = 128**-0.5,
    split_m: bool = False,
    split_full_prefix: bool = False,
    full_prefix_blocks: int | None = None,
) -> torch.Tensor:
    """Score caller-owned tensors with kernel-side query/key tail handling."""

    if query.ndim != 4 or query.shape[-2:] != (4, 128):
        raise ValueError("query must have shape [B, Q, 4, 128].")
    if block_keys.ndim != 3 or block_keys.shape[0] != query.shape[0]:
        raise ValueError("block_keys must have shape [B, N, 128].")
    if block_keys.shape[-1] != 128:
        raise ValueError("block_keys must have feature dimension 128.")
    if visible_counts.shape != query.shape[:2] or visible_counts.dtype != torch.int32:
        raise ValueError("visible_counts must be int32 with shape [B, Q].")
    if visible_counts.device != query.device:
        raise ValueError("visible_counts must be on the query device.")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("TileLang QSA MQA supports BF16 and FP16 query tensors.")
    if block_keys.dtype != query.dtype:
        raise ValueError("query and block_keys must share BF16 or FP16 dtype.")
    if not (query.device.type == "musa") or block_keys.device != query.device:
        raise ValueError("TileLang QSA MQA requires tensors on one MUSA device.")
    _validate_mqa_vector_layout(query)
    _validate_mqa_vector_layout(block_keys)
    if not isinstance(split_full_prefix, bool):
        raise TypeError("split_full_prefix must be a bool.")
    if full_prefix_blocks is not None:
        if isinstance(full_prefix_blocks, bool) or not isinstance(
            full_prefix_blocks, int
        ):
            raise TypeError("full_prefix_blocks must be an integer or None.")
        if full_prefix_blocks < 0:
            raise ValueError("full_prefix_blocks must be non-negative.")
        split_full_prefix = True

    batch, rows = int(query.shape[0]), int(query.shape[1])
    num_blocks = int(block_keys.shape[1])
    if full_prefix_blocks is not None:
        if full_prefix_blocks > num_blocks:
            raise ValueError("full_prefix_blocks must not exceed num_blocks.")
    if rows <= 0:
        return torch.empty(
            (batch, 0, num_blocks), dtype=torch.float32, device=query.device
        )
    if num_blocks <= 0:
        return torch.empty((batch, rows, 0), dtype=torch.float32, device=query.device)
    if split_m:
        if (block_n, block_m, group_n) != (128, 32, 2):
            raise ValueError(
                "split_m dispatch expects the default block_n=128, "
                "block_m=32, group_n=2 contract."
            )
        block_m = _qsa_split_m_block_m(batch, rows, num_blocks)
        group_n = 1
    elif split_full_prefix:
        raise ValueError("split_full_prefix requires split_m=True.")
    if not split_m:
        return _qsa_mqa_score_packed_kernel(
            block_n,
            block_m,
            group_n,
            float(scale),
            use_tme_q=block_m == 8 and batch * math.ceil(rows / block_m) == 1,
            use_tme_k=block_m == 8 and batch * math.ceil(rows / block_m) == 1,
            dtype_name=str(query.dtype).removeprefix("torch."),
        )(query, block_keys, visible_counts)
    full_blocks = 0
    if split_full_prefix and full_prefix_blocks is not None:
        # The caller supplies a prefix visible to every row. Without that
        # static hint, retain masking instead of reading a device minimum.
        full_blocks = (full_prefix_blocks // block_n) * block_n
    ranges = (
        ((0, full_blocks, True), (full_blocks, num_blocks - full_blocks, False))
        if block_n <= full_blocks < num_blocks
        else ((0, num_blocks, False),)
    )
    output = torch.empty(
        (batch, rows, num_blocks), dtype=torch.float32, device=query.device
    )
    for key_start, key_count, skip_mask in ranges:
        if math.ceil(num_blocks / 128) * 128 < 131072:
            factory, config = _qsa_mqa_split_m_config(
                rows,
                num_blocks,
                block_m,
                scale,
                key_start=key_start,
                key_count=key_count,
                skip_mask=skip_mask,
            )
            kernel = factory(
                **config, dtype_name=str(query.dtype).removeprefix("torch.")
            )
            kernel.adapter._get_executable(query.device)(
                query, block_keys, visible_counts, output, key_start, key_count
            )
            continue
        complete_rows = rows // block_m * block_m
        for row_start, row_count, bounded_tail in (
            (0, complete_rows, False),
            (complete_rows, rows - complete_rows, True),
        ):
            if row_count == 0:
                continue
            factory, config = _qsa_mqa_split_m_config(
                rows,
                num_blocks,
                block_m,
                scale,
                key_start=key_start,
                key_count=key_count,
                skip_mask=skip_mask,
                bounded_tail=bounded_tail,
            )
            kernel = factory(
                **config, dtype_name=str(query.dtype).removeprefix("torch.")
            )
            kernel.adapter._get_executable(query.device)(
                query,
                block_keys,
                visible_counts,
                output,
                key_start,
                key_count,
                row_start,
                row_count,
            )
    return output


def run_qsa_mqa_prefill(
    query: torch.Tensor,
    keys: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    scale: float = 128**-0.5,
    block_n: int = 64,
    block_m: int = 32,
) -> torch.Tensor:
    """Run packed-row MQA on MUSA BF16/FP16 tensor views."""

    if query.ndim != 3 or query.shape[1:] != (4, 128):
        raise ValueError("query must have shape [rows, 4, 128].")
    if keys.ndim != 2 or keys.shape[-1] != 128:
        raise ValueError("keys must have shape [num_keys, 128].")
    rows, num_keys = int(query.shape[0]), int(keys.shape[0])
    if row_starts.shape != (rows,) or row_ends.shape != (rows,):
        raise ValueError("row_starts and row_ends must have shape [rows].")
    if row_starts.dtype != torch.int32 or row_ends.dtype != torch.int32:
        raise ValueError("row_starts and row_ends must be int32.")
    if query.dtype not in (torch.bfloat16, torch.float16) or keys.dtype != query.dtype:
        raise TypeError("query and keys must have matching BF16/FP16 dtype.")
    if not (query.device.type == "musa") or any(
        tensor.device != query.device for tensor in (keys, row_starts, row_ends)
    ):
        raise ValueError("MQA prefill tensors must share one MUSA device.")
    _validate_mqa_vector_layout(query)
    _validate_mqa_vector_layout(keys)
    if rows == 0 or num_keys == 0:
        return torch.empty((rows, num_keys), dtype=torch.float32, device=query.device)
    return _qsa_mqa_prefill_kernel(
        block_n,
        block_m,
        float(scale),
        dtype_name=str(query.dtype).removeprefix("torch."),
    )(query, keys, row_starts, row_ends)


def run_qsa_mqa_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    *,
    scale: float = 128**-0.5,
    group_size: int = 64,
) -> torch.Tensor:
    """Run paged-decode MQA logits on BF16/FP16 MUSA tensor views."""

    if query.ndim != 3 or query.shape[1:] != (4, 128):
        raise ValueError("query must have shape [batch, 4, 128].")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, 128):
        raise ValueError("k_cache must have shape [pages, page_size, 1, 128].")
    if page_table.ndim != 2 or page_table.shape[0] != query.shape[0]:
        raise ValueError("page_table must have shape [batch, max_pages].")
    if context_lens.shape != (query.shape[0],):
        raise ValueError("context_lens must have shape [batch].")
    if page_table.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise TypeError("page_table and context_lens must be int32.")
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or k_cache.dtype != query.dtype
    ):
        raise TypeError("query and k_cache must have matching BF16/FP16 dtype.")
    if not (query.device.type == "musa") or any(
        tensor.device != query.device for tensor in (k_cache, page_table, context_lens)
    ):
        raise ValueError("MQA decode tensors must share one MUSA device.")
    page_size = int(k_cache.shape[1])
    if max_model_len < 0:
        raise ValueError("max_model_len must be non-negative.")
    if (
        query.shape[0] == 0
        or max_model_len == 0
        or page_table.shape[1] == 0
        or k_cache.shape[0] == 0
    ):
        return torch.full(
            (query.shape[0], max_model_len),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
    output = torch.empty(
        query.shape[0], max_model_len, dtype=torch.float32, device=query.device
    )
    kernel = _qsa_mqa_decode_kernel(
        page_size,
        group_size,
        float(scale),
        str(query.dtype).removeprefix("torch."),
        use_tme_q=(
            query.stride(-1) == 1
            and query.data_ptr() % 16 == 0
            and all(
                size <= 1 or (stride > 0 and stride % 8 == 0)
                for size, stride in zip(query.shape[:-1], query.stride()[:-1])
            )
        ),
    )
    kernel.adapter._get_executable(query.device)(
        query, k_cache, page_table, context_lens, output
    )
    return output


def _validate_mqa_q(q: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor.")
    if q.ndim == 4:
        if q.shape[1] != 1:
            raise ValueError("MQA decode q rank-4 layout must have Q=1.")
        q = q[:, 0]
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(
            f"QSA MQA q must have shape [rows, heads, dim], got {tuple(q.shape)}."
        )
    return q, int(q.shape[0]), int(q.shape[1]), int(q.shape[2])


def _score_multiplier(dim, scale, score_scale):
    if scale is not None and score_scale is not None:
        raise ValueError("scale and score_scale are mutually exclusive.")
    if score_scale is not None:
        denominator = float(score_scale)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("score_scale must be a positive finite denominator.")
        return 1.0 / denominator
    multiplier = dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("scale must be a positive finite multiplier.")
    return multiplier


def mqa_prefill(inputs):
    q, rows, heads, dim = _validate_mqa_q(inputs.q)
    k = inputs.k
    if not isinstance(k, torch.Tensor):
        raise TypeError("k must be a torch.Tensor.")
    if k.ndim == 3 and k.shape[1] == 1:
        k = k[:, 0]
    if k.ndim != 2 or k.shape[-1] != dim:
        raise ValueError("k must have shape [keys,dim] or [keys,1,dim].")
    if k.device != q.device:
        raise ValueError("q and k must share a device.")
    starts, ends = inputs.row_starts, inputs.row_ends
    native = inputs.backend == "tilelang"
    for name, tensor in (("row_starts", starts), ("row_ends", ends)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device != q.device or tuple(tensor.shape) != (rows,):
            raise ValueError(f"{name} must have shape [rows] on q's device.")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be int32 or int64.")
        if native and tensor.dtype != torch.int32:
            raise TypeError(
                f"{name} must be int32 on the TileLang path; prepare it in the caller."
            )
    multiplier = _score_multiplier(dim, inputs.scale, inputs.score_scale)
    eligible = (
        native
        and (q.device.type == "musa")
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and (heads, dim) == (4, 128)
    )
    if eligible:
        return run_qsa_mqa_prefill(q, k, starts, ends, scale=multiplier)
    if inputs.backend == "tilelang":
        raise ValueError("TileLang MQA prefill requires MUSA BF16/FP16 H4/D128 inputs.")
    return mqa_prefill_reference(q, k, starts, ends, multiplier=multiplier)


def mqa_decode(inputs):
    q, batch, heads, dim = _validate_mqa_q(inputs.q)
    k_cache, page_table, context_lens = (
        inputs.k_cache,
        inputs.page_table,
        inputs.context_lens,
    )
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("k_cache must be a torch.Tensor.")
    if k_cache.ndim == 3:
        k_cache = k_cache.unsqueeze(2)
    if (
        k_cache.ndim != 4
        or tuple(k_cache.shape[2:]) != (1, dim)
        or k_cache.shape[1] <= 0
    ):
        raise ValueError("k_cache must have shape [pages,page_size,1,dim].")
    if k_cache.device != q.device:
        raise ValueError("q and k_cache must share a device.")
    native = inputs.backend == "tilelang"
    for name, tensor in (("page_table", page_table), ("context_lens", context_lens)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device != q.device or tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be int32/int64 on q's device.")
        if native and tensor.dtype != torch.int32:
            raise TypeError(
                f"{name} must be int32 on the TileLang path; prepare it in the caller."
            )
    if page_table.ndim != 2 or page_table.shape[0] != batch:
        raise ValueError("page_table must have shape [batch,max_pages].")
    if tuple(context_lens.shape) != (batch,):
        raise ValueError("context_lens must have shape [batch].")
    page_size = int(k_cache.shape[1])
    if page_table.shape[1] <= 0:
        raise ValueError("page_table must expose at least one logical page.")
    if inputs.max_model_len < 0:
        raise ValueError("max_model_len must be non-negative.")
    multiplier = _score_multiplier(dim, inputs.scale, inputs.score_scale)
    eligible = (
        native
        and (q.device.type == "musa")
        and q.dtype in (torch.bfloat16, torch.float16)
        and k_cache.dtype == q.dtype
        and (heads, dim) == (4, 128)
        and page_size in (8, 16, 32, 64)
    )
    if eligible:
        return run_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, inputs.max_model_len, scale=multiplier
        )
    if inputs.backend == "tilelang":
        raise ValueError(
            "TileLang MQA decode requires MUSA BF16/FP16 H4/D128 inputs and page_size in 8/16/32/64."
        )
    if k_cache.shape[0] == 0:
        return torch.full(
            (batch, inputs.max_model_len),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
    return mqa_decode_reference(
        q,
        k_cache,
        page_table,
        context_lens,
        inputs.max_model_len,
        multiplier=multiplier,
    )


def _visible_block_counts(metadata, num_blocks, block_size):
    positions = metadata.positions
    counts = positions // block_size + (positions % block_size == block_size - 1)
    return counts.clamp(min=0, max=num_blocks)


def mqa_score(inputs):
    inputs._prepare(inputs)
    q, keys, metadata = inputs.q_idx, inputs.block_keys, inputs.metadata
    if inputs.block_size <= 0:
        raise ValueError("block_size must be positive.")
    native = inputs.backend == "tilelang"
    if inputs.block_table is not None:
        if native:
            raise ValueError(
                "paged MQA score uses qsa_mqa_decode; pass backend='reference' for the paged score oracle."
            )
        keys = gather_paged_mqa_keys_reference(
            keys,
            inputs.block_table,
            page_size=inputs.page_size,
            batch_ids=metadata.batch_ids,
            positions=metadata.positions,
            row_valid=(metadata.positions >= 0)
            if metadata.row_valid is None
            else metadata.row_valid,
            batch_size=metadata.batch_size,
            block_size=inputs.block_size,
        )
    if not isinstance(keys, torch.Tensor):
        raise TypeError("block_keys must be a torch.Tensor.")
    if keys.ndim == 2 and metadata.batch_size == 1:
        keys = keys.unsqueeze(0)
    elif keys.ndim == 4 and keys.shape[2] == 1:
        keys = keys.squeeze(2)
    if keys.ndim != 3 or keys.shape[0] != metadata.batch_size:
        raise ValueError("block_keys must have shape [B,N,D] or [B,N,1,D].")
    if (
        q.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or keys.dtype != q.dtype
    ):
        raise TypeError("q_idx and block_keys must share BF16, FP16, or FP32 dtype.")
    if (
        keys.device != q.device
        or keys.shape[-1] != q.shape[-1]
        or min(q.shape[-2:]) <= 0
    ):
        raise ValueError(
            "q_idx and block_keys must share a device and positive feature dimension."
        )
    scale = _score_multiplier(q.shape[-1], inputs.scale, inputs.score_scale)
    visible = _visible_block_counts(metadata, keys.shape[1], inputs.block_size)
    if metadata.row_valid is not None:
        visible = torch.where(metadata.row_valid, visible, 0)
    visible = visible.to(torch.int32)
    eligible = (
        native
        and (q.device.type == "musa")
        and q.ndim == 4
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[-2:] == (4, 128)
        and inputs.block_table is None
        and inputs.block_size == 4
    )
    if eligible:
        return run_qsa_mqa_score(
            q, keys, visible.view(q.shape[0], q.shape[1]), scale=scale
        )
    if inputs.backend == "tilelang":
        raise ValueError(
            "TileLang MQA score requires MUSA BF16/FP16 q=[B,Q,4,128], block_keys=[B,N,128]; use qsa_mqa_prefill for packed rows."
        )
    return metadata.restore(
        mqa_score_reference(
            metadata.flat, keys, metadata.batch_ids, visible, scale=scale
        )
    )
