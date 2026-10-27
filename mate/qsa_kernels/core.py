"""QSA sparse core attention with reference and TileLang-MUSA paths.

The PyTorch path remains the correctness oracle for the full metadata
contract. BF16/FP16 contiguous-cache and direct-paged paths dispatch to
TileLang; official-width paged prefill uses a TME producer/consumer kernel,
while SplitKV partitions token tiles and uses a fused merge kernel. The public
contiguous and one-split paged paths use the measured M=16 dual-GEMM
configuration. For canonical K=2051, the regular direct-gather path starts at
6 structured rows or 7 generic rows. Smaller batches use M16 SplitKV; the
measured 2--5 row contiguous range uses async16 gathers with invalid V lanes
zero-filled before publication. Direct paths apply one in-place TileLang gate;
SplitKV fuses the gate into its merge.
LSE is exported from native softmax statistics without recomputing QK.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Optional

import torch

from ..testing.qsa import sparse_attention_reference, splitkv_merge_reference
from .tilelang.qsa_core import (
    _qsa_core_kernel,
    _qsa_core_paged_prefill_kernel,
    _qsa_core_splitk_merge_kernel,
    _qsa_gate_kernel,
    _qsa_prefill_metadata_kernel,
)

_LOG2E = 1.4426950408889634


def qsa_validate_writable_tensor(tensor):
    """Require independently writable positive-stride slice elements."""
    span = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    ):
        if stride < span:
            raise ValueError("output must have nonoverlapping positive strides")
        span += (size - 1) * stride


def _validate_qsa_output(out, reference):
    """Accept nonoverlapping slices, including feature steps and storage offsets."""

    if out.shape != reference.shape:
        raise ValueError("out must have the same shape as query")
    if out.dtype != reference.dtype or out.device != reference.device:
        raise ValueError("out must match the query dtype and device")
    qsa_validate_writable_tensor(out)


def _qsa_tensors_overlap(left, right):
    """Reject shared byte ranges, allowing disjoint views of one allocation."""
    if left.device != right.device or not left.numel() or not right.numel():
        return False

    def bounds(tensor):
        start = tensor.data_ptr()
        offsets = [
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        ]
        itemsize = tensor.element_size()
        return (
            start + sum(min(offset, 0) for offset in offsets) * itemsize,
            start + (sum(max(offset, 0) for offset in offsets) + 1) * itemsize,
        )

    left_start, left_end = bounds(left)
    right_start, right_end = bounds(right)
    return left_start < right_end and right_start < left_end


def _launch_qsa_kernel(kernel, *inputs, out=None, return_lse=False):
    """Bind caller-owned output directly; aliases require caller preparation."""
    if out is None:
        return kernel(*inputs)
    qsa_validate_writable_tensor(out)
    if any(_qsa_tensors_overlap(out, tensor) for tensor in inputs):
        raise ValueError("out must not overlap QSA inputs.")
    if return_lse:
        lse = torch.empty(out.shape[:-1], dtype=torch.float32, device=out.device)
        kernel.adapter._get_executable(out.device)(*inputs, out, lse)
        return out, lse
    kernel.adapter._get_executable(out.device)(*inputs, out)
    return out


def _qsa_core_layout_eligible(tensor, *, kv=False):
    """Require contiguous D lanes; KV vector loads also need 16-byte alignment."""

    if tensor.ndim not in (3, 4) or (kv and tensor.stride(-1) != 1):
        return False
    if any(stride < 0 for stride in tensor.stride()):
        return False
    if kv:
        vector_elements = 16 // tensor.element_size()
        return tensor.data_ptr() % 16 == 0 and all(
            size <= 1 or stride % vector_elements == 0
            for size, stride in zip(tensor.shape[:-1], tensor.stride()[:-1])
        )
    return True


def _validate_qsa_core_layout(query, key, value):
    if not _qsa_core_layout_eligible(query):
        raise ValueError("QSA query requires nonnegative strides.")
    if not all(_qsa_core_layout_eligible(tensor, kv=True) for tensor in (key, value)):
        raise ValueError(
            "QSA K/V require unit feature stride and 16-byte alignment; "
            "active outer strides must be nonnegative and 16-byte aligned."
        )


def _qsa_core_head_shape(query, key, tile_m):
    """Validate logical heads against the physical D256 GEMM tile."""

    heads, kv_heads = int(query.shape[-2]), int(key.shape[-2])
    if query.shape[-1] != 256 or key.shape[-1] != 256:
        raise ValueError("TileLang QSA core requires head_dim=256.")
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError("QSA core requires positive heads and Hq divisible by Hkv.")
    group = heads // kv_heads
    if tile_m not in (16, 32) or group > tile_m:
        raise ValueError("GQA group must fit the physical M16 or M32 tile.")
    return kv_heads, group


def _qsa_core_narrow_address_eligible(*tensors, output_bytes=0):
    """Bound byte offsets for dense tensors; strided and large views stay wide."""

    limit = 2**31
    return output_bytes < limit and all(
        tensor.is_contiguous() and tensor.numel() * tensor.element_size() < limit
        for tensor in tensors
    )


def _empty_qsa_core_output(query, out, return_lse):
    output = out if out is not None else torch.empty_like(query)
    if return_lse:
        return output, torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )
    return output


def run_qsa_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float = 256**-0.5,
    block_n: int = 64,
    pv_block_h: int = 32,
    annotate_v_layout: bool = True,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    """Run QSA core with unit D stride and independent outer K/V strides."""
    if out is not None:
        _validate_qsa_output(out, query)

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query/key must have shape [B, Q/L, Hq/Hkv, D].")
    tile_m = pv_block_h
    kv_heads, group = _qsa_core_head_shape(query, key, tile_m)
    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch dimensions must match.")
    if value.shape != key.shape:
        raise ValueError("value must have the same shape as key.")
    if token_indices.ndim != 3 or token_indices.shape[:2] != query.shape[:2]:
        raise ValueError("token_indices must have shape [B, Q, K].")
    if token_indices.dtype != torch.int32:
        raise ValueError("token_indices must be int32.")
    if query_positions.shape != query.shape[:2] or query_positions.dtype != torch.int32:
        raise ValueError("query_positions must be int32 with shape [B, Q].")
    if (
        seq_lens.ndim != 1
        or seq_lens.shape[0] != query.shape[0]
        or seq_lens.dtype != torch.int32
    ):
        raise ValueError("seq_lens must be int32 with shape [B].")
    if not (query.device.type == "musa"):
        raise ValueError("TileLang QSA core requires MUSA tensors.")
    if any(
        t.device != query.device
        for t in (key, value, token_indices, query_positions, seq_lens)
    ):
        raise ValueError("all QSA core tensors must share a device.")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("TileLang QSA core supports BF16 and FP16 tensors.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key, and value must share BF16 or FP16 dtype.")
    _validate_qsa_core_layout(query, key, value)
    if key.shape[1] == 0:
        raise ValueError("key/value must expose at least one cache token.")
    if block_n <= 0 or block_n % 8:
        raise ValueError("block_n must be positive and divisible by 8.")

    if query.numel() == 0:
        return _empty_qsa_core_output(query, out, return_lse)

    return _launch_qsa_kernel(
        _qsa_core_kernel(
            block_n=block_n,
            scale=float(scale) * _LOG2E,
            pv_block_h=int(pv_block_h),
            annotate_v_layout=bool(annotate_v_layout),
            kv_heads=kv_heads,
            gqa_group_size=group,
            dtype_name=str(query.dtype).removeprefix("torch."),
            return_lse=return_lse,
        ),
        query,
        key,
        value,
        token_indices,
        query_positions,
        seq_lens,
        out=out,
        return_lse=return_lse,
    )


def _validate_qsa_paged_arguments(
    query,
    key,
    value,
    token_indices,
    token_to_req,
    query_positions,
    seq_lens,
    block_table,
    *,
    tile_m,
    block_n,
    index_name="token_indices",
):
    """Validate paged core metadata before any address arithmetic."""

    if query.ndim != 3 or key.ndim != 4:
        raise ValueError(
            "query/key must have shape [rows, Hq, D]/[pages, page, Hkv, D]."
        )
    kv_heads, group = _qsa_core_head_shape(query, key, tile_m)
    if value.shape != key.shape:
        raise ValueError("value must have the same shape as key.")
    rows = query.shape[0]
    if token_indices.ndim != 2 or token_indices.shape[0] != rows:
        raise ValueError(f"{index_name} must have shape [rows, K].")
    if token_indices.dtype != torch.int32:
        raise ValueError(f"{index_name} must be int32.")
    if token_to_req.shape != (rows,) or token_to_req.dtype != torch.int32:
        raise ValueError("token_to_req must be int32 with shape [rows].")
    if query_positions.shape != (rows,) or query_positions.dtype != torch.int32:
        raise ValueError("query_positions must be int32 with shape [rows].")
    if seq_lens.ndim != 1 or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 vector.")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != seq_lens.shape[0]
        or block_table.dtype != torch.int32
    ):
        raise ValueError("block_table must be int32 with shape [requests, max_pages].")
    if not (query.device.type == "musa"):
        raise ValueError("TileLang QSA core requires MUSA tensors.")
    tensors = (
        key,
        value,
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
    )
    if any(t.device != query.device for t in tensors):
        raise ValueError("all QSA core tensors must share a device.")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("TileLang QSA core supports BF16 and FP16 tensors.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key, and value must share BF16 or FP16 dtype.")
    _validate_qsa_core_layout(query, key, value)
    if block_n <= 0 or block_n % 8:
        raise ValueError("block_n must be positive and divisible by 8.")
    num_tokens = int(token_indices.shape[-1])
    if num_tokens <= 0:
        raise ValueError(f"{index_name} width must be positive.")
    page_size = int(key.shape[1])
    if page_size <= 0 or key.shape[0] <= 0 or block_table.shape[1] <= 0:
        raise ValueError("paged KV and block_table dimensions must be positive.")
    return int(rows), kv_heads, group, num_tokens, page_size


def run_qsa_core_paged(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    scale: float = 256**-0.5,
    block_n: int = 64,
    pv_block_h: int = 32,
    annotate_v_layout: bool = True,
    cache_metadata: bool = True,
    flat_physical_metadata: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the direct-paged kernel with caller-supplied metadata policy."""
    if out is not None:
        _validate_qsa_output(out, query)

    rows, kv_heads, group, num_tokens, page_size = _validate_qsa_paged_arguments(
        query,
        key,
        value,
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        tile_m=pv_block_h,
        block_n=block_n,
    )
    tile_m = pv_block_h

    physical_metadata, page_offsets = _prepare_qsa_paged_metadata(
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        int(key.shape[0]),
        page_size,
        flat_physical_metadata,
    )
    if (
        kv_heads == 1
        and group == 3
        and rows >= 7
        and num_tokens == 2051
        and block_n == 64
        and tile_m == 16
        and cache_metadata
        and flat_physical_metadata
        and key.is_contiguous()
        and value.is_contiguous()
    ):
        # Reuse the local-head TME pipeline without padding the input metadata.
        output = _launch_qsa_kernel(
            _qsa_core_paged_prefill_kernel(
                block_n=block_n,
                scale=float(scale) * _LOG2E,
                predecode_physical_ids=True,
                direct_async_gather=True,
                kv_heads=kv_heads,
                gqa_group_size=group,
                dtype_name=str(query.dtype).removeprefix("torch."),
            ),
            query,
            key.view(-1, kv_heads, 256),
            value.view(-1, kv_heads, 256),
            physical_metadata,
            out=out,
        )
    else:
        output = _launch_qsa_kernel(
            _qsa_core_kernel(
                block_n=block_n,
                page_size=page_size,
                scale=float(scale) * _LOG2E,
                pv_block_h=int(pv_block_h),
                annotate_v_layout=bool(annotate_v_layout),
                flat_physical_metadata=bool(flat_physical_metadata),
                kv_heads=kv_heads,
                gqa_group_size=group,
                dtype_name=str(query.dtype).removeprefix("torch."),
            ),
            query,
            key,
            value,
            token_indices,
            physical_metadata,
            page_offsets,
            token_to_req,
            query_positions,
            seq_lens,
            out=out,
        )
    return output


def _can_qsa_prefetch_next_descriptor(
    *,
    num_tokens: int,
    block_n: int,
    predecode_physical_ids: bool,
    direct_async_gather: bool,
    compact_dynamic_tiles: bool,
) -> bool:
    """Return whether the fixed direct-gather kernel can prefetch descriptors."""

    return bool(
        num_tokens in (2048, 2051)
        and block_n == 64
        and predecode_physical_ids
        and direct_async_gather
        and not compact_dynamic_tiles
    )


def _validate_prefill_inputs(
    query,
    key,
    value,
    indices,
    token_to_req,
    query_positions,
    seq_lens,
    block_table,
    *,
    block_n,
    allow_strided_kv,
    index_name,
):
    """Validate the shared paged prefill ABI without repairing input layouts."""
    if query.ndim != 3 or tuple(query.shape[-2:]) != (24, 256):
        raise ValueError("query must have shape [rows, 24, 256].")
    if key.ndim != 4 or tuple(key.shape[2:]) != (2, 256):
        raise ValueError("key must have shape [pages, page_size, 2, 256].")
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise TypeError("TileLang QSA prefill supports BF16 and FP16 tensors.")
    if block_n != 64:
        raise ValueError("prefill requires block_n=64 for its producer layout.")
    *_, page_size = _validate_qsa_paged_arguments(
        query,
        key,
        value,
        indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        tile_m=16,
        block_n=block_n,
        index_name=index_name,
    )
    if not allow_strided_kv and (not key.is_contiguous() or not value.is_contiguous()):
        raise ValueError("regular QSA prefill requires contiguous K/V tensors.")
    if int(key.shape[0]) * page_size - 1 > torch.iinfo(torch.int32).max:
        raise ValueError("flattened physical-token metadata exceeds int32 range.")
    return page_size


def _prepare_prefill_metadata(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_n: int,
    allow_strided_kv: bool = False,
):
    """Validate the paged ABI and pack valid entries as physical token ids."""

    page_size = _validate_prefill_inputs(
        query,
        key,
        value,
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        block_n=block_n,
        allow_strided_kv=allow_strided_kv,
        index_name="token_indices",
    )
    num_tokens = int(token_indices.shape[-1])
    if num_tokens <= 0:
        raise ValueError("token_indices width must be positive.")
    physical, _ = _prepare_qsa_paged_metadata(
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        int(key.shape[0]),
        page_size,
        True,
    )

    return query, physical, num_tokens


def _prepare_prefill_block_metadata(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_n: int,
    sort_physical: bool = False,
    omit_empty_tail: bool = False,
    compact_tail: bool = False,
    active_tile_marker: bool = False,
    allow_strided_kv: bool = False,
    _full_blocks_proven: bool = False,
):
    """Pack complete QSA blocks and the current tail as one descriptor each.

    The low two bits encode ``valid_tokens - 1``.  QSA logical block starts
    and supported paged-cache physical starts are four-token aligned, leaving
    those bits available without widening the metadata ABI.
    """

    page_size = _validate_prefill_inputs(
        query,
        key,
        value,
        block_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        block_n=block_n,
        allow_strided_kv=allow_strided_kv,
        index_name="block_indices",
    )
    num_blocks = int(block_indices.shape[-1])
    max_blocks = (2051 - 3) // 4
    if num_blocks <= 0 or num_blocks > max_blocks:
        raise ValueError(f"block_indices width must be in [1, {max_blocks}].")
    if sort_physical and num_blocks != 512:
        raise ValueError("physical descriptor sorting requires 512 blocks.")
    if omit_empty_tail and num_blocks != 512:
        raise ValueError("omit_empty_tail requires structured K=2051 metadata")
    if active_tile_marker and (not compact_tail or omit_empty_tail):
        raise ValueError("active_tile_marker requires fused compact dynamic metadata")
    if not isinstance(active_tile_marker, bool):
        raise TypeError("active_tile_marker must be a bool")
    if page_size % 4:
        raise ValueError("structured QSA prefill requires page_size divisible by four.")
    num_tokens = num_blocks * 4 if omit_empty_tail else num_blocks * 4 + 3
    # Tail and optional active-tile marker are algorithm metadata. The core
    # masks its final tile instead of padding descriptors to a tile multiple.
    descriptor_groups = num_blocks + 1 + int(active_tile_marker)
    # Optional mask-eliding schedules are explicit caller contracts. Never
    # inspect device contents to choose a Python launch during Graph capture.
    if omit_empty_tail and not _full_blocks_proven:
        raise ValueError("omit_empty_tail requires _full_blocks_proven=True.")

    descriptors = run_qsa_prefill_metadata_fused(
        query,
        block_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        page_size=page_size,
        num_physical_pages=int(key.shape[0]),
        descriptor_groups=descriptor_groups,
        sort_physical=bool(sort_physical),
        # One descriptor per thread reduces the sort's loop/barrier overhead.
        threads=512 if sort_physical else 256,
        compact_tail=bool(compact_tail),
        active_tile_marker=bool(active_tile_marker),
    )
    if omit_empty_tail:
        descriptors = descriptors[:, :num_blocks]
    return query, descriptors, num_tokens


def run_qsa_core_paged_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    scale: float = 256**-0.5,
    block_n: int = 64,
    predecode_physical_ids: bool = False,
    direct_async_gather: bool = False,
    use_tme_output: bool = True,
    assume_full_blocks: bool = False,
    warp_specialized: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the dedicated paged prefill kernel.

    ``predecode_physical_ids`` is safe for wrapper-validated metadata and is
    enabled by the public paged dispatch. Low-level callers may disable it for
    an apples-to-apples baseline measurement.
    """
    if out is not None:
        _validate_qsa_output(out, query)

    if (
        direct_async_gather
        and not warp_specialized
        and (not key.is_contiguous() or not value.is_contiguous())
    ):
        raise ValueError("direct_async_gather requires contiguous K/V tensors")
    query, physical, _ = _prepare_prefill_metadata(
        query,
        key,
        value,
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        block_n=block_n,
        allow_strided_kv=warp_specialized,
    )
    key_flat = key if warp_specialized else key.view(-1, 2, 256)
    value_flat = value if warp_specialized else value.view(-1, 2, 256)
    robust_size_bytes = key_flat.numel() * key_flat.element_size()
    wide_address = robust_size_bytes >= (1 << 31)
    output = _launch_qsa_kernel(
        _qsa_core_paged_prefill_kernel(
            block_n=block_n,
            scale=float(scale) * _LOG2E,
            structured_microblocks=False,
            predecode_physical_ids=bool(predecode_physical_ids),
            disable_index_type_promotion=not (wide_address or warp_specialized),
            direct_async_gather=bool(direct_async_gather),
            use_tme_output=bool(use_tme_output),
            assume_full_blocks=bool(assume_full_blocks),
            warp_specialized=warp_specialized,
            kv_page_size=int(key.shape[1]) if warp_specialized else 0,
            dtype_name=str(query.dtype).removeprefix("torch."),
        ),
        query,
        key_flat,
        value_flat,
        physical,
        out=out,
    )
    return output


def run_qsa_core_paged_prefill_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    scale: float = 256**-0.5,
    block_n: int = 64,
    wide_address: bool | None = None,
    predecode_physical_ids: bool = False,
    direct_async_gather: bool = False,
    sort_physical: bool = False,
    use_tme_output: bool = True,
    assume_full_blocks: bool = False,
    omit_empty_tail: bool = False,
    compact_tail: bool = False,
    compact_dynamic_tiles: bool = False,
    prefetch_next_descriptor: bool | None = None,
    warp_specialized: bool = False,
    out: torch.Tensor | None = None,
    _full_blocks_proven: bool = False,
) -> torch.Tensor:
    """Run paged prefill directly from QSA's selected four-token blocks.

    The current incomplete block is derived from ``query_positions`` and
    ``seq_lens``.  Unlike the arbitrary-token entry point, this path transfers
    one int32 descriptor per complete block and one descriptor for the tail.
    ``predecode_physical_ids`` hoists invalid-id handling before the K/V
    gather loop; it preserves ``-1`` in the mask metadata.
    ``sort_physical`` is an opt-in short-context optimization: it orders the
    complete descriptors by physical base in the fused metadata kernel and
    leaves the tail slot untouched.
    ``assume_full_blocks`` is an opt-in canonical specialization; callers must
    prove that all 512 complete block descriptors are valid.  The final tail
    tile remains masked.
    ``omit_empty_tail`` requires the caller's ``_full_blocks_proven`` assertion
    of 512 valid blocks and a zero-length tail, and launches a 2048-token core.
    No host content checks are performed for these explicit scheduling hints.
    ``compact_tail`` places the tail descriptor after the valid complete-block
    prefix; ``compact_dynamic_tiles`` uses that layout to stop at the first
    empty tile.

    Q/O use logical heads directly. Query loading zero-fills the shared M16
    tail because TileLang 0.1.12 does not correctly lower an M12 TME load.
    ``prefetch_next_descriptor=None`` selects the measured S5000 optimization
    for at least 64 rows on the fixed direct-gather path. ``False`` retains the
    baseline schedule for controlled A/B measurements.
    ``out`` binds caller-owned nonoverlapping strided storage directly.
    Input/output aliases are rejected. Sliced outputs use LSU stores.
    """
    if out is not None:
        _validate_qsa_output(out, query)

    if (
        direct_async_gather
        and not warp_specialized
        and (not key.is_contiguous() or not value.is_contiguous())
    ):
        raise ValueError("direct_async_gather requires contiguous K/V tensors")
    query, descriptors, num_tokens = _prepare_prefill_block_metadata(
        query,
        key,
        value,
        block_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        block_n=block_n,
        sort_physical=sort_physical,
        omit_empty_tail=omit_empty_tail,
        compact_tail=bool(compact_tail or compact_dynamic_tiles),
        active_tile_marker=bool(compact_dynamic_tiles),
        allow_strided_kv=warp_specialized,
        _full_blocks_proven=_full_blocks_proven,
    )
    key_flat = key if warp_specialized else key.view(-1, 2, 256)
    value_flat = value if warp_specialized else value.view(-1, 2, 256)
    robust_size_bytes = key_flat.numel() * key_flat.element_size()
    requires_wide = robust_size_bytes >= (1 << 31)
    if wide_address is None:
        wide_address = requires_wide
    elif not wide_address and requires_wide:
        raise ValueError(
            "wide_address=False is unsafe for a K/V buffer at or above 2 GiB."
        )
    can_prefetch_descriptor = _can_qsa_prefetch_next_descriptor(
        num_tokens=int(num_tokens),
        block_n=int(block_n),
        predecode_physical_ids=bool(predecode_physical_ids),
        direct_async_gather=bool(direct_async_gather),
        compact_dynamic_tiles=bool(compact_dynamic_tiles),
    )
    if warp_specialized:
        prefetch_next_descriptor = False
    elif prefetch_next_descriptor is None:
        prefetch_next_descriptor = bool(
            query.shape[0] >= 64 and num_tokens == 2051 and can_prefetch_descriptor
        )
    elif prefetch_next_descriptor and not can_prefetch_descriptor:
        raise ValueError(
            "prefetch_next_descriptor requires fixed K=2048/2051, "
            "BLOCK_N=64, and predecoded direct gather"
        )
    kernel = _qsa_core_paged_prefill_kernel(
        block_n=block_n,
        scale=float(scale) * _LOG2E,
        structured_microblocks=True,
        predecode_physical_ids=bool(predecode_physical_ids),
        disable_index_type_promotion=not (wide_address or warp_specialized),
        direct_async_gather=bool(direct_async_gather),
        use_tme_output=bool(use_tme_output),
        assume_full_blocks=bool(assume_full_blocks or omit_empty_tail),
        compact_dynamic_tiles=bool(compact_dynamic_tiles),
        active_tile_marker=bool(compact_dynamic_tiles),
        prefetch_next_descriptor=bool(prefetch_next_descriptor),
        v_gather_stripe_major=bool(
            int(query.shape[0]) >= 8192 and int(robust_size_bytes) >= 1 << 30
        ),
        warp_specialized=warp_specialized,
        kv_page_size=int(key.shape[1]) if warp_specialized else 0,
        dtype_name=str(query.dtype).removeprefix("torch."),
    )
    return _launch_qsa_kernel(kernel, query, key_flat, value_flat, descriptors, out=out)


def _prepare_qsa_paged_metadata(
    token_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_pages: int,
    page_size: int,
    flat_physical_metadata: bool = False,
):
    if num_pages * page_size - 1 > torch.iinfo(torch.int32).max:
        raise ValueError("flattened physical-token metadata exceeds int32 range.")
    physical = torch.empty_like(token_indices)
    kernel = _qsa_prefill_metadata_kernel(
        page_size=page_size, structured_microblocks=False
    )
    kernel.adapter._get_executable(token_indices.device)(
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        physical,
        num_pages,
    )
    if flat_physical_metadata:
        return physical, physical
    return physical // page_size, physical.clamp_min(0) % page_size


def run_qsa_core_paged_splitk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_splits: int,
    scale: float = 256**-0.5,
    block_n: int = 64,
    pv_block_h: int = 32,
    cache_metadata: bool = False,
    flat_physical_metadata: bool = False,
    compact_partial_output: bool = False,
    direct_async_gather: bool = False,
    gate: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    if out is not None:
        _validate_qsa_output(out, query)
    rows, kv_heads, group, width, page_size = _validate_qsa_paged_arguments(
        query,
        key,
        value,
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        tile_m=pv_block_h,
        block_n=block_n,
    )
    strided_kv = not (key.is_contiguous() and value.is_contiguous())
    if direct_async_gather and strided_kv:
        raise ValueError("direct_async_gather requires contiguous K/V tensors.")
    if direct_async_gather and not (cache_metadata and flat_physical_metadata):
        raise ValueError("direct_async_gather requires flat physical cache metadata")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive.")
    physical, offsets = _prepare_qsa_paged_metadata(
        token_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        int(key.shape[0]),
        page_size,
        bool(flat_physical_metadata),
    )
    padded = math.ceil(width / block_n) * block_n
    tiles = padded // block_n
    if num_splits > tiles:
        raise ValueError(f"num_splits={num_splits} exceeds token tiles={tiles}.")
    kernel = _qsa_core_kernel(
        block_n=block_n,
        page_size=page_size,
        num_splits=num_splits,
        scale=float(scale) * _LOG2E,
        pv_block_h=int(pv_block_h),
        flat_physical_metadata=bool(flat_physical_metadata),
        compact_partial_output=bool(compact_partial_output),
        direct_async_gather=bool(direct_async_gather),
        kv_heads=kv_heads,
        gqa_group_size=group,
        dtype_name=str(query.dtype).removeprefix("torch."),
        annotate_v_layout=False,
    )
    partial_output, partial_lse = kernel(
        query,
        key,
        value,
        token_indices,
        physical,
        offsets,
        token_to_req,
        query_positions,
        seq_lens,
    )
    return merge_qsa_core_paged_splitk(
        partial_output,
        partial_lse,
        gate=gate,
        output_dtype=query.dtype,
        out=out,
        return_lse=return_lse,
    )


def merge_qsa_core_paged_splitk(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor:
    """Merge BF16/FP16/FP32 split partials and optionally apply the gate."""

    if return_lse and gate is not None:
        raise ValueError("apply the gate after an LSE-producing merge.")

    if partial_output.ndim != 4 or partial_lse.shape != partial_output.shape[:3]:
        raise ValueError(
            "partials/LSE must have shape [splits, rows, Hq, D]/[splits, rows, Hq]."
        )
    heads = int(partial_output.shape[2])
    result_dtype = output_dtype or (
        torch.float16 if partial_output.dtype == torch.float16 else torch.bfloat16
    )
    if out is not None:
        if tuple(out.shape) != tuple(partial_output.shape[1:]):
            raise ValueError("out must have shape [rows,Hq,D].")
        if out.device != partial_output.device or out.dtype != result_dtype:
            raise ValueError("out must match the merge output dtype and device.")
        qsa_validate_writable_tensor(out)
        if any(
            _qsa_tensors_overlap(out, tensor)
            for tensor in (partial_output, partial_lse, gate)
            if tensor is not None
        ):
            raise ValueError("out must not overlap SplitKV inputs.")

    if min(partial_output.shape) <= 0:
        raise ValueError("SplitKV partial dimensions must be positive.")
    if partial_output.device.type == "musa":
        if partial_output.shape[-1] != 256:
            raise ValueError("TileLang SplitKV merge requires head_dim=256.")
        if partial_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("TileLang SplitKV partials must be BF16, FP16, or FP32.")
        if partial_lse.dtype != torch.float32:
            raise TypeError("TileLang SplitKV LSE must be FP32.")
        if partial_lse.device != partial_output.device:
            raise ValueError("SplitKV partials and LSE must share a device.")
        num_splits, rows = partial_output.shape[:2]
        inputs: tuple[torch.Tensor, ...] = (partial_output, partial_lse)
        gate_dtype_name = None
        if gate is not None:
            if tuple(gate.shape) != (rows, heads):
                raise ValueError("gate must have shape [rows, Hq].")
            if gate.device != partial_output.device:
                raise ValueError("gate and SplitKV partials must share a device.")
            if gate.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise TypeError("gate must be float16, bfloat16, or float32.")
            gate_dtype_name = str(gate.dtype).removeprefix("torch.")
            inputs += (gate,)
        return _launch_qsa_kernel(
            _qsa_core_splitk_merge_kernel(
                num_splits,
                str(partial_output.dtype).removeprefix("torch."),
                heads,
                str(result_dtype).removeprefix("torch."),
                gate_dtype_name=gate_dtype_name,
                return_lse=return_lse,
            ),
            *inputs,
            out=out,
            return_lse=return_lse,
        )

    result, lse = splitkv_merge_reference(
        partial_output,
        partial_lse,
        output_dtype=result_dtype,
        gate=gate,
        return_lse=return_lse,
    )
    if out is not None:
        out[...] = result
        result = out
    return (result, lse) if return_lse else result


def run_qsa_prefill_metadata_fused(
    query: torch.Tensor,
    block_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    num_physical_pages: int,
    descriptor_groups: int,
    sort_physical: bool = False,
    threads: int = 256,
    compact_tail: bool = False,
    active_tile_marker: bool = False,
    active_tile_width: int = 16,
) -> torch.Tensor:
    """Predecode caller-owned block selections into safe physical descriptors."""

    if not (query.device.type == "musa"):
        raise ValueError("query must be on MUSA")
    if query.dtype not in (torch.bfloat16, torch.float16) or tuple(query.shape[1:]) != (
        24,
        256,
    ):
        raise TypeError("query must be BF16 or FP16 [rows, 24, 256]")
    rows = int(query.shape[0])
    if block_indices.dtype != torch.int32 or block_indices.ndim != 2:
        raise TypeError("block_indices must be int32 [rows, num_blocks]")
    if int(block_indices.shape[0]) != rows:
        raise ValueError("block_indices rows must match query")
    for name, tensor, shape in (
        ("token_to_req", token_to_req, (rows,)),
        ("query_positions", query_positions, (rows,)),
    ):
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != shape:
            raise TypeError(f"{name} must be int32 {shape}")
    if seq_lens.dtype != torch.int32 or seq_lens.ndim != 1:
        raise TypeError("seq_lens must be int32 [requests]")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise TypeError("block_table must be int32 [requests, max_pages]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if num_physical_pages <= 0:
        raise ValueError("num_physical_pages must be positive")
    if num_physical_pages * page_size - 1 > torch.iinfo(torch.int32).max:
        raise ValueError("flattened physical-token metadata exceeds int32 range")
    num_blocks = int(block_indices.shape[1])
    if num_blocks <= 0 or block_table.shape[1] <= 0:
        raise ValueError("block selection and page table widths must be positive")
    if descriptor_groups < num_blocks + 1 + int(active_tile_marker):
        raise ValueError("descriptor_groups must reserve tail and marker slots")
    if sort_physical and num_blocks != 512:
        raise ValueError("physical descriptor sorting requires 512 blocks")
    if active_tile_marker and (not compact_tail or active_tile_width <= 0):
        raise ValueError(
            "active_tile_marker requires compact_tail and a positive tile width"
        )
    if int(block_table.shape[0]) != int(seq_lens.shape[0]):
        raise ValueError("block_table requests must match seq_lens")
    tensors = (
        block_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all metadata tensors must share the query device")

    kernel = _qsa_prefill_metadata_kernel(
        page_size=int(page_size),
        sort_physical=bool(sort_physical),
        threads=int(threads),
        compact_tail=bool(compact_tail),
        active_tile_marker=bool(active_tile_marker),
        active_tile_width=int(active_tile_width),
    )
    executable = kernel.adapter._get_executable(query.device)
    descriptors = torch.empty(
        (rows, int(descriptor_groups)), dtype=torch.int32, device=query.device
    )
    executable(
        block_indices,
        token_to_req,
        query_positions,
        seq_lens,
        block_table,
        descriptors,
        int(num_physical_pages),
    )
    return descriptors


def apply_qsa_gate_inplace(
    output: torch.Tensor,
    gate: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    pack_heads: bool | None = None,
) -> torch.Tensor:
    """Apply sigmoid(gate) to flattened [rows, Hq, 256] BF16/FP16 output."""

    if not (output.device.type == "musa") or gate.device != output.device:
        raise ValueError("output and gate must share a MUSA device.")
    if output.dtype not in (torch.bfloat16, torch.float16) or output.ndim != 3:
        raise TypeError("output must be BF16 or FP16 [rows, Hq, 256].")
    rows = int(output.shape[0])
    heads = int(output.shape[1])
    if heads <= 0 or output.shape[2] != 256:
        raise ValueError("output must have positive Hq and head_dim=256.")
    if rows <= 0:
        raise ValueError("rows must be positive.")
    if tuple(gate.shape) != (rows, heads):
        raise ValueError("gate must have shape [rows, Hq].")
    gate_dtype_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(gate.dtype)
    if gate_dtype_name is None:
        raise TypeError("gate must be float16, bfloat16, or float32.")
    qsa_validate_writable_tensor(output)
    if _qsa_tensors_overlap(output, gate):
        raise ValueError("gate must not overlap output.")
    if (
        query_positions.shape != (rows,)
        or query_positions.dtype != torch.int32
        or query_positions.device != output.device
    ):
        raise ValueError("query_positions must be int32 [rows] on output's device.")
    if _qsa_tensors_overlap(output, query_positions):
        raise ValueError("query_positions must not overlap output.")
    if pack_heads is None:
        pack_heads = rows >= 48
    _qsa_gate_kernel(
        gate_dtype_name,
        bool(pack_heads),
        heads,
        str(output.dtype).removeprefix("torch."),
    )(output, gate, query_positions)
    return output


def _validate_kv_layout(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    batch_size: int,
    cu_seqlens_k: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    page_size: Optional[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[int], bool]:
    """Validate KV layout and identify its page/varlen metadata."""

    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("key and value must be torch.Tensor objects.")
    if key.device != device or value.device != device:
        raise ValueError("query, key, and value must share a device.")
    if key.dtype != value.dtype:
        raise TypeError(
            f"key/value dtypes must match, got {key.dtype} and {value.dtype}."
        )
    if key.ndim == 4:
        if value.shape != key.shape:
            raise ValueError("value must have the same shape as key.")
        if key.shape[2] <= 0:
            raise ValueError("key/value must have positive Hkv.")
        if block_table is not None:
            if cu_seqlens_k is not None:
                raise ValueError("block_table and cu_seqlens_k are mutually exclusive.")
            if block_table.ndim != 2 or block_table.shape[0] != batch_size:
                raise ValueError("block_table must have shape [B, max_pages].")
            if block_table.device != device or block_table.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise TypeError(
                    "block_table must be an int32/int64 tensor on the query device."
                )
            if page_size is None:
                page_size = int(key.shape[1])
            if page_size != key.shape[1] or page_size <= 0:
                raise ValueError(
                    f"page_size must equal the paged cache dimension {key.shape[1]}."
                )
            return key, value, None, page_size, True
        if key.shape[0] != batch_size:
            raise ValueError(
                "contiguous key must have shape [B, L, Hkv, D], "
                f"got {tuple(key.shape)}."
            )
        if cu_seqlens_k is not None:
            raise ValueError("cu_seqlens_k is only valid for flattened rank-3 KV.")
        return key, value, None, None, False

    if key.ndim != 3:
        raise ValueError(
            "key/value must be [B, L, Hkv, D], [total_k, Hkv, D], or paged "
            f"[pages, page_size, Hkv, D], got {tuple(key.shape)}."
        )
    if value.shape != key.shape:
        raise ValueError("value must have the same shape as key.")
    if key.shape[1] <= 0:
        raise ValueError("key/value must have positive Hkv.")
    if cu_seqlens_k is not None:
        cu = cu_seqlens_k
        if cu.numel() != batch_size + 1:
            raise ValueError("cu_seqlens_k must contain B + 1 entries.")
        if block_table is not None:
            raise ValueError("block_table and cu_seqlens_k are mutually exclusive.")
        return key, value, cu, None, False
    if block_table is not None:
        if block_table.ndim != 2 or block_table.shape[0] != batch_size:
            raise ValueError("block_table must have shape [B, max_pages].")
        if block_table.device != device or block_table.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "block_table must be an int32/int64 tensor on the query device."
            )
        if page_size is None:
            page_size = int(key.shape[0])
        if page_size <= 0 or page_size > key.shape[0]:
            raise ValueError(
                f"page_size must be in [1, {key.shape[0]}], got {page_size}."
            )
        # Rank-3 cache with a page table is interpreted as [pages, Hkv, D]
        # only for page_size=1.  A real paged cache must be rank-4; reject the
        # ambiguous layout instead of silently using the wrong address.
        raise ValueError(
            "paged QSA core KV must use rank-4 [pages, page_size, Hkv, D] tensors."
        )
    if batch_size != 1:
        raise ValueError(
            "rank-3 contiguous KV without cu_seqlens_k is only valid for B=1."
        )
    return key, value, None, None, False


def _core_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    token_indices: torch.Tensor,
    metadata,
    *,
    out=None,
    **kwargs,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    rows = metadata.flat.shape[0]
    selected = token_indices.shape[-1]
    heads, dim = metadata.flat.shape[-2:]
    kv_heads = int(key.shape[-2])
    bytes_per_row = max(1, selected * (2 * kv_heads * dim * 4 + heads * 4 * 5))
    chunk_rows = max(1, (64 * 1024 * 1024) // bytes_per_row)
    output = (
        None
        if value is None
        else (
            torch.empty_like(metadata.flat)
            if out is None
            else out.view(*metadata.flat.shape)
        )
    )
    lse = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    gate = kwargs.pop("gate")
    if gate is not None:
        if gate.ndim == 4 and gate.shape[-1] == 1:
            gate = gate.squeeze(-1)
        if len(metadata.leading_shape) == 2 and gate.ndim == 3:
            gate = gate.view(rows, gate.shape[-1])
        if gate.device != query.device or gate.shape != (rows, heads):
            raise ValueError(
                "gate must have shape [B, Q, Hq] or [total_q, Hq] on the query device."
            )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk_output, chunk_lse = sparse_attention_reference(
            metadata.flat[start:end],
            key,
            value,
            token_indices[start:end],
            batch_ids=metadata.batch_ids[start:end],
            query_positions=metadata.positions[start:end],
            row_valid=(metadata.positions[start:end] >= 0)
            if metadata.row_valid is None
            else metadata.row_valid[start:end],
            gate=None if gate is None else gate[start:end],
            **kwargs,
        )
        if output is not None:
            output[start:end] = chunk_output
        lse[start:end] = chunk_lse
    return (
        metadata.restore(output) if output is not None else None,
        metadata.restore(lse),
    )


def _tilelang_head_shape_eligible(query: torch.Tensor, key: torch.Tensor) -> bool:
    """Logical GQA groups must fit the M16 compute tile."""

    if query.ndim not in (3, 4) or key.ndim != 4:
        return False
    heads, kv_heads = int(query.shape[-2]), int(key.shape[-2])
    return bool(
        query.shape[-1] == key.shape[-1] == 256
        and heads > 0
        and kv_heads > 0
        and heads % kv_heads == 0
        and heads // kv_heads <= 16
    )


def _tilelang_eligible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    metadata,
    *,
    cu_seqlens_k: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
) -> bool:
    paged = block_table is not None
    return bool(
        query.ndim in ((3, 4) if paged else (4,))
        and len(metadata.leading_shape) in ((1, 2) if paged else (2,))
        and _qsa_core_layout_eligible(query)
        and key.ndim == value.ndim == 4
        and _qsa_core_layout_eligible(key, kv=True)
        and _qsa_core_layout_eligible(value, kv=True)
        and token_indices.ndim in (2, 3)
        and cu_seqlens_k is None
        and (not paged or (block_table.ndim == 2 and block_table.shape[1] > 0))
        and (
            gate is None
            or (
                tuple(gate.shape) == (metadata.flat.shape[0], metadata.flat.shape[-2])
                and gate.dtype in (torch.float16, torch.bfloat16, torch.float32)
            )
        )
        and (query.device.type == "musa")
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype == value.dtype
        and _tilelang_head_shape_eligible(query, key)
    )


def physical_attention(
    q,
    k,
    v,
    selected_kv_indices,
    *,
    softmax_scale=None,
    k_scale=1.0,
    v_scale=1.0,
    compute_dtype=torch.bfloat16,
    out=None,
    lse=None,
    return_lse=False,
    num_splits=None,
):
    """Consume framework-prepared physical indices without host content reads."""
    tensors = (q, k, v, selected_kv_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("q, k, v, and selected_kv_indices must be tensors.")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q and k/v must have shape [rows/tokens, heads, 256].")
    kv_heads, group = _qsa_core_head_shape(q, k, 16)
    if k.shape[0] <= 0:
        raise ValueError("KV storage must contain at least one token.")
    if selected_kv_indices.ndim != 2 or selected_kv_indices.shape[0] != q.shape[0]:
        raise ValueError("selected_kv_indices must have shape [rows, selected_tokens].")
    if selected_kv_indices.shape[1] <= 0 or selected_kv_indices.dtype != torch.int32:
        raise ValueError("selected_kv_indices must be int32 with a positive width.")
    if q.device.type != "musa" or any(t.device != q.device for t in tensors[1:]):
        raise ValueError("all QSA tensors must share a MUSA device.")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError(
            "physical QSA currently supports BF16 q/k/v; FP8 is not enabled."
        )
    if compute_dtype != torch.bfloat16:
        raise ValueError("physical QSA currently requires BF16 compute_dtype.")
    scale = 256**-0.5 if softmax_scale is None else float(softmax_scale)
    if not all(
        math.isfinite(float(s)) and float(s) > 0 for s in (scale, k_scale, v_scale)
    ):
        raise ValueError(
            "softmax_scale, k_scale, and v_scale must be finite and positive."
        )
    if float(k_scale) != 1.0 or float(v_scale) != 1.0:
        raise ValueError("k_scale and v_scale must remain 1 for BF16 K/V.")
    if not isinstance(return_lse, bool):
        raise TypeError("return_lse must be a bool.")
    if num_splits is not None and (type(num_splits) is not int or num_splits <= 0):
        raise ValueError("num_splits must be a positive integer or None.")
    _validate_qsa_core_layout(q, k, v)
    if any(stride < 0 for stride in selected_kv_indices.stride()):
        raise ValueError("selected_kv_indices requires nonnegative strides.")
    for name, output, shape, dtype in (
        ("out", out, q.shape, q.dtype),
        ("lse", lse, q.shape[:-1], torch.float32),
    ):
        if output is None:
            continue
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{name} must be a tensor.")
        if output.shape != shape or output.dtype != dtype or output.device != q.device:
            raise ValueError(f"{name} has an incompatible shape, dtype, or device.")
        qsa_validate_writable_tensor(output)
        if any(_qsa_tensors_overlap(output, tensor) for tensor in tensors):
            raise ValueError(f"{name} must not overlap QSA inputs.")
    if out is not None and lse is not None and _qsa_tensors_overlap(out, lse):
        raise ValueError("out and lse must not overlap.")
    write_lse = return_lse or lse is not None
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if write_lse and lse is None:
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    rows, width = int(q.shape[0]), int(selected_kv_indices.shape[1])
    if rows:
        splits = (
            _choose_qsa_split_count(rows, width) if num_splits is None else num_splits
        )
        if (
            num_splits is None
            and write_lse
            and kv_heads == 1
            and group == 3
            and 7 <= rows <= 32
            and width >= 1024
        ):
            splits = 8
        if (
            not write_lse
            and splits == 1
            and rows >= 7
            and k.is_contiguous()
            and v.is_contiguous()
        ):
            _launch_qsa_kernel(
                _qsa_core_paged_prefill_kernel(
                    block_n=64,
                    scale=scale * _LOG2E,
                    predecode_physical_ids=True,
                    direct_async_gather=True,
                    kv_heads=kv_heads,
                    gqa_group_size=group,
                ),
                q,
                k,
                v,
                selected_kv_indices,
                out=out,
            )
        else:
            # The flat-physical specialization never reads request/page metadata.
            # Reuse zero-copy views for those dead ABI operands; no remapping occurs.
            unused_rows = selected_kv_indices[:, 0]
            kernel = _qsa_core_kernel(
                block_n=64,
                scale=scale * _LOG2E,
                pv_block_h=16,
                annotate_v_layout=False,
                kv_heads=kv_heads,
                gqa_group_size=group,
                page_size=1,
                num_splits=splits if write_lse or splits > 1 else 0,
                flat_physical_metadata=True,
                compact_partial_output=True,
                direct_async_gather=True,
            )
            arguments = (
                q,
                k.unsqueeze(1),
                v.unsqueeze(1),
                selected_kv_indices,
                selected_kv_indices,
                selected_kv_indices,
                unused_rows,
                unused_rows,
                unused_rows,
            )
            if splits == 1 and not write_lse:
                _launch_qsa_kernel(kernel, *arguments, out=out)
            else:
                partial, partial_lse = kernel(*arguments)
            if write_lse:
                merge = _qsa_core_splitk_merge_kernel(
                    splits,
                    "bfloat16",
                    heads=int(q.shape[1]),
                    return_lse=True,
                )
                merge.adapter._get_executable(q.device)(partial, partial_lse, out, lse)
            elif splits > 1:
                merge_qsa_core_paged_splitk(partial, partial_lse, out=out)
    return (out, lse) if return_lse else out


def _choose_qsa_split_count(
    rows: int,
    selected_tokens: int,
    *,
    structured_microblocks: bool = False,
) -> int:
    """Choose the measured conservative SplitKV count for decode-sized work."""

    tiles = max(1, math.ceil(selected_tokens / 64))
    base_programs = rows * 2
    if tiles == 33:
        # S5000 wave-alignment table for QSA's canonical K=2051.  One CTA per
        # 64-token tile wins for the very small row regime; the merge cost is
        # still lower than serializing 3--4 tiles in each split.
        direct_row_threshold = 6 if structured_microblocks else 7
        if rows < direct_row_threshold:
            return 33
        return 1
    if base_programs <= 4:
        target = 16
    elif base_programs <= 64:
        target = 8
    else:
        target = 1
    count = 1
    while count * 2 <= min(target, tiles):
        count *= 2
    return count


def _use_qsa_splitkv_async_gather(
    rows: int,
    selected_tokens: int,
    num_splits: int,
    *,
    canonical_heads: bool,
    contiguous_kv: bool,
    page_size: int,
) -> bool:
    """Use async16 only in the measured small-batch SplitKV range."""

    return bool(
        canonical_heads
        and contiguous_kv
        and 2 <= rows <= 5
        and selected_tokens == 2051
        and num_splits == 33
        and page_size in (16, 64, 128)
    )


def _use_qsa_tme_prefill(
    rows: int,
    selected_tokens: int,
    num_splits: int,
    *,
    structured_microblocks: bool = False,
) -> bool:
    """Use the direct producer/consumer kernel once its wave wins SplitKV."""

    direct_row_threshold = 6 if structured_microblocks else 7
    return bool(
        num_splits == 1 and rows >= direct_row_threshold and selected_tokens == 2051
    )


def _use_qsa_physical_descriptor_sort(
    rows: int,
    block_count: int,
    page_capacity: int,
    *,
    structured_microblocks: bool,
    num_splits: int,
    contiguous_kv: bool,
    logically_ordered_blocks: bool = False,
) -> bool:
    """Enable fused physical-ordering only in its measured ROI.

    The bitonic network is useful when an unordered selected set occupies a
    relatively small cache (the S5000 8K regime).  The ballot Top-K path already
    emits logical order in its prefill window, where another physical sort does
    not repay its metadata cost for one layer. Keep the rule narrow for generic/WS paths.
    """

    return bool(
        structured_microblocks
        and not logically_ordered_blocks
        and num_splits == 1
        and contiguous_kv
        and block_count == 512
        and 256 <= rows <= 1024
        and page_capacity <= 8192
    )


def _apply_tilelang_gate(
    output: torch.Tensor,
    gate: Optional[torch.Tensor],
    metadata,
) -> torch.Tensor:
    if gate is None or output.numel() == 0:
        return output
    if (
        (output.device.type == "musa")
        and output.dtype in (torch.bfloat16, torch.float16)
        and gate.dtype in {torch.bfloat16, torch.float16, torch.float32}
    ):
        apply_qsa_gate_inplace(
            output.view(-1, metadata.flat.shape[-2], 256),
            gate,
            query_positions=metadata.positions,
        )
        return output
    gate_view = gate.view(*metadata.leading_shape, metadata.flat.shape[-2])
    return output * torch.sigmoid(gate_view.float()).to(output.dtype).unsqueeze(-1)


def _validate_sparse_attention(inputs):
    if not isinstance(inputs.query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor.")
    for name in (
        "warp_specialized",
        "assume_compact_blocks",
    ):
        if not isinstance(getattr(inputs, name), bool):
            raise TypeError(f"{name} must be a bool.")
    query, key, value = inputs.query, inputs.key, inputs.value
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("key and value must be torch.Tensor objects.")
    if query.ndim not in (3, 4) or key.ndim not in (3, 4) or value.shape != key.shape:
        raise ValueError("QSA requires rank-3/4 query and matching rank-3/4 key/value.")
    if inputs.backend == "tilelang":
        if (
            not (query.device.type == "musa")
            or key.device != query.device
            or value.device != query.device
        ):
            raise ValueError("TileLang QSA core requires Q/K/V on one MUSA device.")
        if any(
            t.dtype not in (torch.bfloat16, torch.float16) for t in (query, key, value)
        ):
            raise TypeError("TileLang QSA core supports BF16 and FP16 Q/K/V.")
        if any(t.dtype != query.dtype for t in (key, value)):
            raise ValueError("TileLang QSA core requires matching Q/K/V dtype.")
        hq, hkv = query.shape[-2], key.shape[-2]
        if hq <= 0 or hkv <= 0 or hq % hkv or hq // hkv > 16:
            raise ValueError(
                "TileLang QSA requires positive divisible Hq/Hkv with group <= 16."
            )
        if query.shape[-1] != 256 or key.shape[-1] != 256:
            raise ValueError("TileLang QSA core requires head_dim=256.")
        if not _qsa_core_layout_eligible(query):
            raise ValueError("QSA query requires nonnegative strides.")


def sparse_attention(inputs):
    inputs._prepare(inputs)
    inputs.out_flat = None
    if inputs.out is not None:
        _validate_qsa_output(inputs.out, inputs.query)
        if any(
            _qsa_tensors_overlap(inputs.out, tensor)
            for name, tensor in vars(inputs).items()
            if name != "out" and isinstance(tensor, torch.Tensor)
        ):
            raise ValueError("out must not overlap QSA inputs.")
        try:
            inputs.out_flat = inputs.out.view(*inputs.metadata.flat.shape)
        except RuntimeError as error:
            raise ValueError("out must form a zero-copy row view.") from error
    result = _run_sparse_attention(inputs)
    if inputs.out is None:
        return result
    return (inputs.out, result[1]) if inputs.return_lse else inputs.out


def _run_sparse_attention(inputs):
    _validate_sparse_attention(inputs)
    query = inputs.query
    key = inputs.key
    value = inputs.value
    token_indices = inputs.token_indices
    cu_seqlens_k = inputs.cu_seqlens_k
    block_table = inputs.block_table
    page_size = inputs.page_size
    scale = inputs.scale
    gate = inputs.gate_rows
    num_splits = inputs.num_splits
    return_lse = inputs.return_lse
    backend = inputs.backend
    warp_specialized = inputs.warp_specialized
    assume_compact_blocks = inputs.assume_compact_blocks
    logically_ordered_blocks = False
    if warp_specialized and backend == "reference":
        raise ValueError(
            "warp_specialized is only available on the TileLang-MUSA path."
        )
    auto_split = num_splits is None
    if num_splits is None:
        num_splits = 1
    if not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError("num_splits must be a positive integer or None.")
    metadata = inputs.metadata
    indices = inputs.indices
    if inputs.seq_lens_q is not None:
        metadata = replace(
            metadata,
            positions=torch.where(metadata.row_valid, metadata.positions, -1),
        )
    blocks = inputs.blocks
    batch_size = metadata.batch_size

    lengths_k = inputs.lengths_k
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError(f"scale must be a positive finite number, got {scale}.")
    _validate_kv_layout(
        key,
        value,
        batch_size=batch_size,
        cu_seqlens_k=cu_seqlens_k,
        block_table=block_table,
        page_size=page_size,
        device=query.device,
    )
    heads, kv_heads = int(query.shape[-2]), int(key.shape[-2])
    if heads <= 0 or heads % kv_heads:
        raise ValueError("QSA core requires positive Hq divisible by Hkv.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key/value head dimensions must match.")
    canonical_heads = query.shape[-2:] == (24, 256) and key.shape[-2:] == (2, 256)
    if backend == "tilelang":
        _validate_qsa_core_layout(query, key, value)
        if warp_specialized and not canonical_heads:
            raise ValueError("warp_specialized requires Hq=24, Hkv=2, D=256.")
    if key.numel() == 0:
        raise ValueError("key/value must expose at least one cache token.")

    tilelang_eligible = _tilelang_eligible(
        query,
        key,
        value,
        token_indices,
        metadata,
        cu_seqlens_k=cu_seqlens_k,
        block_table=block_table,
        gate=gate,
    )
    paged_eligible = tilelang_eligible and block_table is not None
    structured_prefill = blocks is not None and int(key.shape[1]) % 4 == 0
    if auto_split and backend == "tilelang" and paged_eligible:
        num_splits = _choose_qsa_split_count(
            int(metadata.flat.shape[0]),
            int(indices.shape[-1]),
            structured_microblocks=structured_prefill,
        )
    eligible = tilelang_eligible and block_table is None and num_splits == 1
    if backend == "tilelang" and query.numel() == 0 and (paged_eligible or eligible):
        if paged_eligible and num_splits > math.ceil(indices.shape[-1] / 64):
            raise ValueError("num_splits exceeds the selected-token tile count.")
        return _empty_qsa_core_output(query, inputs.out, return_lse)
    fused_split_gate = None
    if backend == "tilelang" and paged_eligible:
        fused_split_gate = gate if num_splits > 1 and not return_lse else None
        if num_splits == 1 and not return_lse:
            tme_prefill = (
                canonical_heads
                and key.is_contiguous()
                and value.is_contiguous()
                and _use_qsa_tme_prefill(
                    int(metadata.flat.shape[0]),
                    int(indices.shape[-1]),
                    num_splits,
                    structured_microblocks=structured_prefill,
                )
            )
            if warp_specialized and num_splits == 1 and int(indices.shape[-1]) == 2051:
                # The normal TME guard keeps the measured structured rows>=6
                # and generic rows>=7 defaults.
                # An explicit experiment is allowed at smaller row counts so
                # correctness/profiling probes do not silently fall back.
                tme_prefill = True
            # Both generic and structured wrappers specialize the robust
            # descriptor with the actual K/V byte size.  The wrapper selects
            # the static 64-bit form at the 2 GiB boundary, so large paged
            # caches no longer need a silent fallback here.
            if tme_prefill:
                prefill_kwargs: dict[str, Any] = {
                    "scale": float(scale),
                    # Decode invalid physical ids once before the K/V gather.
                    "predecode_physical_ids": True,
                    "warp_specialized": warp_specialized,
                    "out": inputs.out_flat,
                }
                if not warp_specialized:
                    prefill_kwargs["direct_async_gather"] = (
                        key.is_contiguous() and value.is_contiguous()
                    )
                if structured_prefill:
                    if not warp_specialized:
                        # Compact layout is a producer-supplied contract;
                        # arbitrary block lists retain the fully masked path.
                        compact_dynamic_tiles = assume_compact_blocks
                        # Sorting helps locality when all 512 descriptors are
                        # active, but its metadata cost and reordered partial
                        # tiles hurt the compact dynamic case.  Keep the two
                        # specializations mutually exclusive.
                        prefill_kwargs["sort_physical"] = (
                            _use_qsa_physical_descriptor_sort(
                                int(metadata.flat.shape[0]),
                                int(blocks.shape[-1]),
                                int(block_table.shape[1]) * int(page_size),
                                structured_microblocks=structured_prefill,
                                num_splits=num_splits,
                                contiguous_kv=key.is_contiguous()
                                and value.is_contiguous(),
                                logically_ordered_blocks=logically_ordered_blocks,
                            )
                            and not compact_dynamic_tiles
                        )
                        prefill_kwargs["compact_dynamic_tiles"] = compact_dynamic_tiles
                        prefill_kwargs["compact_tail"] = compact_dynamic_tiles
                run_prefill = (
                    run_qsa_core_paged_prefill_blocks
                    if structured_prefill
                    else run_qsa_core_paged_prefill
                )
                out = run_prefill(
                    metadata.flat,
                    key,
                    value,
                    blocks if structured_prefill else indices,
                    metadata.batch_ids,
                    metadata.positions,
                    lengths_k,
                    block_table,
                    **prefill_kwargs,
                )
            else:
                out = run_qsa_core_paged(
                    metadata.flat,
                    key,
                    value,
                    indices,
                    metadata.batch_ids,
                    metadata.positions,
                    lengths_k,
                    block_table,
                    scale=float(scale),
                    # TileLang-MUSA 0.1.12 uses M=16 with an inferred V layout.
                    # Stage page metadata once per token so all feature lanes can
                    # reuse it.  The M=32/swizzled form stays available below.
                    pv_block_h=16,
                    annotate_v_layout=False,
                    cache_metadata=True,
                    flat_physical_metadata=True,
                    out=inputs.out_flat,
                )
        else:
            out = run_qsa_core_paged_splitk(
                metadata.flat,
                key,
                value,
                indices,
                metadata.batch_ids,
                metadata.positions,
                lengths_k,
                block_table,
                num_splits=num_splits,
                scale=float(scale),
                pv_block_h=16,
                cache_metadata=True,
                flat_physical_metadata=True,
                compact_partial_output=True,
                direct_async_gather=_use_qsa_splitkv_async_gather(
                    int(metadata.flat.shape[0]),
                    int(indices.shape[-1]),
                    num_splits,
                    canonical_heads=canonical_heads,
                    contiguous_kv=key.is_contiguous() and value.is_contiguous(),
                    page_size=int(page_size),
                ),
                gate=fused_split_gate,
                out=inputs.out_flat,
                return_lse=return_lse,
            )
        if return_lse:
            out, lse = out
            lse = metadata.restore(lse)
        out = metadata.restore(out)
    elif backend == "tilelang" and eligible:
        positions = metadata.positions.view(query.shape[0], query.shape[1])
        lengths = lengths_k
        out = run_qsa_core(
            query,
            key,
            value,
            indices.view(query.shape[0], query.shape[1], indices.shape[-1]),
            positions,
            lengths,
            scale=float(scale),
            pv_block_h=16,
            annotate_v_layout=False,
            out=inputs.out,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = out
    elif backend == "tilelang":
        raise ValueError(
            "TileLang QSA core requires supported BF16/FP16 contiguous-cache or direct-paged "
            "metadata, D256, and a positive divisible GQA group of at most 16; "
            "this layout or option combination is not enabled."
        )

    else:
        output, lse = _core_reference(
            query,
            key,
            value,
            indices,
            metadata,
            out=inputs.out,
            seq_lens_k=lengths_k,
            cu_seqlens_k=cu_seqlens_k,
            block_table=block_table,
            page_size=page_size,
            scale=float(scale),
            gate=gate,
        )
        return (output, lse) if return_lse else output

    if fused_split_gate is None:
        out = _apply_tilelang_gate(out, gate, metadata)
    if return_lse:
        return out, lse
    return out
