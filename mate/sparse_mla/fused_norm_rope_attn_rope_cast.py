"""FlashMLA fused-module contract with PH1 Q-preparation/O-epilogue fusion."""

import torch

from .tilelang.fused_rope_cast import prepare_q


def _column_major_scales(values):
    *leading, rows, columns = values.shape
    storage = torch.empty(
        (*leading, columns, (rows + 3) // 4 * 4),
        dtype=torch.int32,
        device=values.device,
    )
    result = storage.transpose(-1, -2)[..., :rows, :]
    result.copy_(values)
    return result


def permute_q_b_proj(weight_and_sf, h_q, d_q):
    weight, sf = weight_and_sf
    assert d_q == 512 and weight.shape[0] == h_q * d_q
    assert weight.dtype == torch.float8_e4m3fn and sf.dtype == torch.int32
    assert weight.stride(-1) == 1 and sf.stride(0) == 1
    assert sf.shape[0] == h_q * d_q and weight.shape[1] // (4 * sf.shape[1]) in (
        32,
        128,
    )

    def permute(t):
        return t.reshape(h_q, d_q // 16, 16, -1).permute(1, 0, 2, 3).reshape_as(t)

    return permute(weight).contiguous(), _column_major_scales(permute(sf))


def permute_wv_proj(weight_and_sf, wv_group_size, d_o):
    weight, sf = weight_and_sf
    groups, rows, cols = weight.shape
    assert d_o == 512 and cols == wv_group_size * d_o
    assert weight.dtype == torch.float8_e4m3fn and sf.dtype == torch.int32
    assert sf.shape == (groups, rows, cols // 128) and sf.stride(1) == 1
    result = (
        weight.reshape(groups, rows, wv_group_size, 16, 32)
        .transpose(2, 3)
        .reshape_as(weight)
    )
    shifts = torch.arange(4, device=sf.device, dtype=torch.int32) * 8
    unpacked = ((sf.unsqueeze(-1) >> shifts) & 255).reshape(
        groups, rows, wv_group_size, 16
    )
    unpacked = unpacked.transpose(2, 3).reshape(groups, rows, cols // 128, 4)
    packed = (unpacked << shifts).sum(dim=-1).to(torch.int32)
    return result.contiguous(), _column_major_scales(packed)


def _check(
    q, positions, neox, rope_dim, table, groups, gran, col_major, rounded, packed, d_v
):
    assert q.ndim == 3 and q.shape[-1] == d_v == 512 and q.dtype == torch.bfloat16
    assert q.shape[1] in (64, 128), "FlashMLA fused attention supports H64 or H128"
    assert q.stride(-1) == 1 and q.stride(1) == 512
    assert not neox and rope_dim == 64
    assert (
        positions.dtype == torch.int32
        and positions.shape == (q.shape[0],)
        and positions.is_contiguous()
    )
    assert (
        table.dtype == torch.float32
        and table.ndim == 2
        and table.shape[1] == 64
        and table.is_contiguous()
    )
    assert groups > 0 and q.shape[1] % groups == 0
    assert gran == 32 and col_major and rounded and packed


def _pad_indices(indices):
    padding = (-indices.shape[-1]) % 64
    return (
        torch.nn.functional.pad(indices, (0, padding), value=-1) if padding else indices
    )


def prefill(
    enable_q_norm,
    rms_norm_eps,
    token_positions,
    is_rope_neox_style,
    rope_dim,
    cos_sin_cache,
    n_wv_group,
    num_per_channels,
    use_tma_aligned_col_major_sf,
    round_sf,
    use_packed_ue8m0,
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    attn_sink=None,
    topk_length=None,
):
    """Run the upstream FlashMLA fused prefill contract on MUSA.

    The adapter prepares Q with optional RMS normalization and RoPE, runs sparse
    prefill, then applies the inverse-RoPE, Wv-layout, and FP8 output epilogue.
    """
    from .tilelang.sparse_mla_model1_fwd_pipelined import (
        sparse_mla_fwd_interface_model1,
    )

    _check(
        q,
        token_positions,
        is_rope_neox_style,
        rope_dim,
        cos_sin_cache,
        n_wv_group,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        d_v,
    )
    prepared = prepare_q(q, token_positions, cos_sin_cache, enable_q_norm, rms_norm_eps)
    return sparse_mla_fwd_interface_model1(
        prepared,
        kv,
        _pad_indices(indices),
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        _fused_output=(token_positions, cos_sin_cache, n_wv_group),
    )


def decode(
    enable_q_norm,
    rms_norm_eps,
    token_positions,
    is_rope_neox_style,
    rope_dim,
    cos_sin_cache,
    n_wv_group,
    num_per_channels,
    use_tma_aligned_col_major_sf,
    round_sf,
    use_packed_ue8m0,
    q,
    k_cache,
    indices_in_kvcache,
    sm_scale,
    d_v=512,
    attn_sink=None,
    topk_length=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
):
    """Run the upstream FlashMLA fused paged-decode contract on MUSA.

    The adapter prepares Q, builds scheduler metadata for per-query candidate
    lists, and runs decode with the fused inverse-RoPE and FP8 output epilogue.
    """
    from mate.flashmla import get_mla_metadata
    from .tilelang.sparse_mla_decode import sparse_mla_decode_fwd

    _check(
        q,
        token_positions,
        is_rope_neox_style,
        rope_dim,
        cos_sin_cache,
        n_wv_group,
        num_per_channels,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        d_v,
    )
    prepared = prepare_q(q, token_positions, cos_sin_cache, enable_q_norm, rms_norm_eps)
    # Fused API lengths are per query. Represent queries as single-query
    # requests in the ordinary scheduled ABI, preserving every prefix boundary.
    indices = _pad_indices(indices_in_kvcache)[:, None, :]
    extra_indices = (
        None
        if extra_indices_in_kvcache is None
        else _pad_indices(extra_indices_in_kvcache)[:, None, :]
    )
    metadata, splits = get_mla_metadata(
        None,
        q.shape[1],
        1,
        q.shape[1],
        True,
        topk=indices.shape[-1],
        extra_topk=None if extra_indices is None else extra_indices.shape[-1],
        q=prepared[:, None],
        bs=q.shape[0],
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    quant, sf, lse = sparse_mla_decode_fwd(
        q=prepared[:, None],
        k_cache=k_cache,
        indices=indices,
        head_dim_v=d_v,
        tile_scheduler_metadata=metadata,
        num_splits=splits,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
        metadata_getter=get_mla_metadata,
        _fused_output=(token_positions, cos_sin_cache, n_wv_group),
    )
    return quant, sf, lse.squeeze(-1)
