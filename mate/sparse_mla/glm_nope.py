"""GLM NoPE token-major cache contract (distinct from FlashMLA V4.1 pages)."""

from typing import Optional

import torch

from mate.api_logging import mate_api

from .flashmla_checks import normalize_sparse_decode_indices, require_batch_topk_length
from .tilelang.glm_nope_cache import write_glm_nope_cache as write_glm_nope_cache
from .tilelang.sparse_mla_v32_decode_fwd_scheduled import tilelang_flashmla_interface


@mate_api
def sparse_mla_nope_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 Q512 and inline 512 E4M3 + four arbitrary FP32 scales per token.

    Accept compact 528-byte rows and legacy 656-byte padded rows. Candidate
    lengths bound a prefix, including holes; they are not counts of valid IDs.
    The model's softmax scale is required explicitly after weight absorption.
    """
    from mate.flashmla import get_mla_metadata

    assert q.ndim == 4 and q.shape[-1] == 512 and q.dtype == torch.bfloat16
    assert k_cache.ndim == 4 and k_cache.shape[2] == 1
    assert k_cache.shape[-1] in (528, 656)
    assert k_cache.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn)
    assert k_cache.stride(-1) == 1
    assert q.device == k_cache.device == indices.device
    batch, queries, heads, _ = q.shape
    indices = normalize_sparse_decode_indices(indices, batch, queries, 1, "indices")
    topk_length = require_batch_topk_length(topk_length, batch, "topk_length")
    assert indices.shape[-1] % 64 == 0
    assert (tile_scheduler_metadata is None) == (num_splits is None)
    if tile_scheduler_metadata is None:
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=queries * heads,
            num_heads_k=1,
            num_heads_q=heads,
            is_fp8_kvcache=True,
            topk=indices.shape[-1],
            q=q,
            bs=batch,
            topk_length=topk_length,
        )
    # Native FlashInfer's inline cache has a uniform token stride. Slice the
    # payload after flattening so the 656-byte legacy stride remains intact.
    cache = k_cache.view(torch.uint8).view(-1, 1, k_cache.shape[-1])[..., :528]
    return tilelang_flashmla_interface(
        q,
        cache,
        indices,
        tile_scheduler_metadata,
        num_splits,
        sm_scale=softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
