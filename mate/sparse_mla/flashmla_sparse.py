from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from .tilelang.sparse_mla_decode import sparse_mla_decode_fwd
from .tilelang.sparse_mla_prefill import sparse_mla_prefill_fwd


MetadataGetter = Callable[..., Tuple[torch.Tensor, torch.Tensor]]


def flashmla_sparse_prefill(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single FlashMLA sparse-prefill dispatcher for MODEL1 and V3.2."""
    return sparse_mla_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )


def flashmla_sparse_decode(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    metadata_getter: MetadataGetter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single FlashMLA sparse-decode dispatcher for MODEL1 and V3.2."""
    return sparse_mla_decode_fwd(
        q=q,
        k_cache=k_cache,
        indices=indices,
        head_dim_v=head_dim_v,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        metadata_getter=metadata_getter,
    )
