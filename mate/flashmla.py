import functools
from typing import Optional, Tuple

import torch

from mate.api_logging import mate_api
from mate.jit.mla_ops import get_mla_ops_module
from mate.jit.runtime import ffi_to_torch
from .sparse_mla.tilelang.sparse_mla_decode_fwd_scheduled import (
    tilelang_flashmla_interface,
)
from .sparse_mla.tilelang.sparse_mla_fwd_pipelined import (
    tilelang_sparse_mla_prefill_fwd_interface,
)


@functools.cache
def _get_module():
    return get_mla_ops_module()


def _allocate_flashmla_outputs(
    q: torch.Tensor, head_dim_v: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (*q.shape[:-1], head_dim_v),
        dtype=q.dtype,
        device=q.device,
    )
    softmax_lse = torch.empty(
        q.shape[:-1],
        dtype=torch.float32,
        device=q.device,
    )
    return out, softmax_lse


def _prepare_mla_query_input(
    x: torch.Tensor, *, require_seq_dense: bool
) -> torch.Tensor:
    # Keep Python-side materialization aligned with the actual FFI layout contract.
    if x.stride(-1) != 1:
        return x.contiguous()
    if require_seq_dense and x.dim() == 4:
        if x.stride(1) != x.shape[-2] * x.stride(2):
            return x.contiguous()
    return x


@mate_api
def get_mla_metadata(
    cache_seqlens: Optional[torch.Tensor],
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
    q: Optional[torch.Tensor] = None,
    bs: Optional[int] = None,
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
    return ffi_to_torch(
        _get_module().get_function("get_mla_decoding_metadata")(
            cache_seqlens,
            num_q_tokens_per_head_k,
            num_heads_k,
            num_heads_q,
            is_fp8_kvcache,
            topk,
            None,
            None,
            q,
            bs,
        )
    )


@mate_api
def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
            * Not Supported Now *
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.
            * Not Supported Now *
    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert attn_sink is None, "attn_sink Not supported Now"
    assert topk_length is None, "topk_length Not supported Now"
    assert d_v == 512, "sprase prefill only support d_v 512"
    seq_len_q, num_heads_q, head_dim_q = q.shape
    s_kv, h_k, _ = kv.shape
    _, _, topk = indices.shape
    assert head_dim_q == kv.shape[-1]
    assert seq_len_q == indices.shape[0]
    assert h_k == indices.shape[1]
    assert h_k == 1

    tl_out, lse = tilelang_sparse_mla_prefill_fwd_interface(
        q,
        kv,
        indices,
        sm_scale,
    )
    return tl_out, None, lse


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
    indices: Optinal[Tensor]
        The token indices tensor with shape ``(batch_size, seq_len_q, topk)``.
        If not None, sparse attention will be enabled, and only tokens in the `indices` array will be attended to.
        Invalid indices should be set to -1 or numbers >= total_seq_len_kv. For details about how to set up `indices`, please refer to README.md.
    For DeepSeek V3, DeepSeek V3.1, and DeepSeek V3.2:
        head_dim should be 576 while head_dim_v should be 512.
        In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
            - The shape of the tensor `k_cache` is (num_blocks, page_block_size, num_heads_k, head_dim), and num_heads_k must be 1.
            - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
            - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
            - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.
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
        dsa_backend = "tilelang"
        if dsa_backend == "tilelang":
            if is_fp8_kvcache:
                # decode fp8 kv cache
                batch_size, seq_len_q, num_heads_q, head_dim_q = q.shape
                num_blocks, block_size, h_k, dim_bytes = k_cache.shape
                assert dim_bytes == 656
                assert h_k == 1
                assert batch_size == indices.shape[0]
                assert seq_len_q == indices.shape[1]
                assert h_k == indices.shape[2]
                _, _, _, topk = indices.shape
                k_cache = k_cache.view(-1, h_k, dim_bytes)
                assert tile_scheduler_metadata is not None
                assert num_splits is not None
                return tilelang_flashmla_interface(
                    q,
                    k_cache,
                    indices,
                    tile_scheduler_metadata,
                    num_splits,
                    softmax_scale,
                )
            else:
                assert False, "sparse mla decode only support fp8 kvcache"
        else:
            assert False, "Unsupported DSA backend"

    assert head_dim_v == 512

    q_nope = q[:, :, :, :head_dim_v]
    q_pe = q[:, :, :, head_dim_v:]
    should_run_with_asm = q.shape[-2] == 128

    q_nope = _prepare_mla_query_input(q_nope, require_seq_dense=not should_run_with_asm)
    q_pe = _prepare_mla_query_input(q_pe, require_seq_dense=not should_run_with_asm)

    out, softmax_lse = _allocate_flashmla_outputs(q, head_dim_v)

    if should_run_with_asm:
        _get_module().get_function("flash_mla_asm")(
            q_nope,
            q_pe,
            k_cache[:, :, :, :head_dim_v],
            k_cache[:, :, :, head_dim_v:],
            cache_seqlens,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            None,
            None,
        )
    else:
        _get_module().get_function("mla_with_kvcache")(
            q_nope,
            q_pe,
            k_cache[:, :, :, :head_dim_v],
            k_cache[:, :, :, head_dim_v:],
            cache_seqlens,
            None,
            None,
            block_table,
            tile_scheduler_metadata,
            num_splits,
            out,
            softmax_lse,
            softmax_scale,
            causal,
        )
    return out, softmax_lse
