from typing import Optional, Tuple

import torch

import mate._C  # noqa: F401
from mate.api_logging import mate_api


@mate_api
def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Get metadata for MLA decoding.

    Parameters
    ----------
    cache_seqlens: Tensor
        The sequence lengths of the KV cache with shape ``(batch_size)``
    num_q_tokens_per_head_k: int
        Equals to num_q_tokens_per_q_seq * num_heads_q // num_heads_k.
    num_heads_k: int
        The number of k heads.
    num_heads_q: Optional[int]
        The number of q heads. This argument is optional when sparse attention is not enabled
    is_fp8_kvcache: bool
        Whether the k_cache and v_cache are in fp8 format.
    topk: Optional[int]
        If not None, sparse attention will be enabled, and only tokens in the `indices` array passed to `flash_mla_with_kvcache_sm90` will be attended to.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of two tensors:

        * tile_scheduler_metadata, shape ``(num_sm_parts, TileSchedulerMetaDataSize)``
        * num_splits, shape ``(batch_size + 1)``
    """
    return torch.ops.mate.get_mla_decoding_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
    )


@mate_api
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mla forward with kv cache

    Parameters
    ----------
    q : Tensor
        The query tensor with shape ``(batch_size, seq_len_q, num_heads_q, head_dim)``.
    k_cache: Tensor
        The compressed kv cache tensor with shape ``(num_blocks, page_block_size, num_heads_k, head_dim)``.
    block_table: Tensor
        The page table with shape ``(batch_size, max_num_blocks_per_seq)``.
    cache_seqlens: Tensor
        The sequence lengths of the ckv cache wtih shape ``(batch_size)``.
    head_dim_v: int
        Head dimension of v.
    tile_scheduler_metadata: Tensor
        The scheduler metadata with shape ``(num_sm_parts, TileSchedulerMetaDataSize)``, returned by get_mla_metadata.
    num_splits: Tensor
        The num_splits tensor with shape ``(batch_size + 1)``, returned by get_mla_metadata.
    softmax_scale: Optional[float]
        The scale of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
    causal: bool
        Whether to apply causal attention mask.
    is_fp8_kvcache: bool
        Whether the k_cache and v_cache are in fp8 format. For the format of FP8 KV cache, please refer to README.md

        *Not supported now.*
    indices: Optinal[Tensor]
        The token indices tensor with shape ``(batch_size, seq_len_q, topk)``.
        If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to.
        Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.

        *Not supported now.*
    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple of two tensors:

        * out, shape ``(batch_size, seq_len_q, num_heads_q, head_dim_v)``.
        * softmax_lse, shape ``(batch_size, num_heads_q, seq_len_q)``.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        assert not causal, "causal must be `false` if sparse attention is enabled."

    should_run_with_asm = q.shape[-2] == 128
    assert head_dim_v == 512
    if should_run_with_asm:
        return torch.ops.mate.flash_mla_asm(
            q[:, :, :, :head_dim_v],
            q[:, :, :, head_dim_v:],
            k_cache[:, :, :, :head_dim_v],
            k_cache[:, :, :, head_dim_v:],
            cache_seqlens,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale,
            causal,
        )
    else:
        return torch.ops.mate.mla_with_kvcache(
            q[:, :, :, :head_dim_v],
            q[:, :, :, head_dim_v:],
            k_cache[:, :, :, :head_dim_v],
            k_cache[:, :, :, head_dim_v:],
            cache_seqlens,
            None,
            None,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale,
            causal,
        )
