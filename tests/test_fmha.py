# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
import pdb  # noqa: F401

from einops import repeat, rearrange
from typing import Optional, Union, Tuple  # noqa: F401

from mate import flash_attn_varlen_func, flash_attn_with_kvcache
from mate.jit.attention.fmha import _fmha_fwd_combine as fmha_fwd_combine
from mate.mha_interface import get_scheduler_metadata
from mate.testing.flash_attn import (
    attention_ref,
    attention_accum_ref,  # noqa: F401
    attention_combine_ref,  # noqa: F401
    pad_accum,
    gen_input_tensor,
    gen_seqlen_data,
    mask_unused,
    generate_block_kvcache,
    gen_padding_mask_from_seqlens,
    lse_ref_from_score,
    unpad_input,
    pad_input,
    _combine_cp_partials,
    make_cp_rank_local_paged_kvcache,
)

# from mate.jit.fmha import _fmha_fwd as jit_fmha_fwd  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True, False])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 16) for _ in range(1000)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_metadata(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    page_size: int,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
    # total_seqlen_qo = sum(seqlens_qo)
    cu_seqlens_qo = torch.cumsum(
        torch.tensor([0] + seqlens_qo, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    query_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    query_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, None, device
    )
    query_unpad, indices_query, cu_seqlens_query, max_seqlen_query, *rest = unpad_input(
        query_pad, query_padding_mask
    )

    seqlens_kv = [x[1] for x in seq_lens]
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)
    # total_seqlen_kv = sum(seqlens_kv)
    # cu_seqlens_kv = torch.cumsum(
    #     torch.tensor([0] + seqlens_kv, dtype=torch.int32, device=device), dim=0
    # )
    cache_seqlens = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = generate_block_kvcache(
        max_seqlen_kv,
        page_size,
        batch_size,
        head_kv,
        headdim_qk,
        headdim_vo,
        device,
        dtype,
    )

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    learnable_sink = None

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")
    out_ref, _, score_ref = attention_ref(
        q=query_pad,
        k=k_cache_rep,
        v=v_cache_rep,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    atol, rtol = 1.5e-2, 1e-2

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=cache_seqlens,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=page_size,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
        # batch size of 1000 should never splits
        splits = metadata[:batch_size]
        torch.testing.assert_close(splits, torch.ones_like(splits), atol=0, rtol=0)
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse, out_accum, lse_accum, *rest = flash_attn_with_kvcache(
        q=query_unpad,
        k_cache=k_cache if page_size is None else k_cache_paged,
        v_cache=v_cache if page_size is None else v_cache_paged,
        k=None,
        v=None,
        qv=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k_new=None,
        max_seqlen_q=max_seqlen_qo,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_softmax_lse=True,
    )

    if num_splits >= 0:
        out_accum = torch.utils.dlpack.from_dlpack(out_accum)
        lse_accum = torch.utils.dlpack.from_dlpack(lse_accum)
        out_accum_pad, lse_accum_pad = pad_accum(
            out_accum=out_accum,
            lse_accum=lse_accum,
            seqused_q=None,
            cu_seqlens_q=cu_seqlens_qo,
        )

    out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)
    lse_ref, _ = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=learnable_sink,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


# ==============================================================================
# ============================== flash_attn_varlen_func ========================
# ==============================================================================


@pytest.mark.skip(reason="Skip for CI. Avoid Long JIT")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [False])
@pytest.mark.parametrize("mask", [None])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "seq_lens",
    [
        (333, 444),
        (888, 1328),
        (222, 463),
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (96, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128)])
@pytest.mark.parametrize("backend", ["auto", "mutlass"])
@torch.inference_mode()
def test_varlen_func_bshd(
    batch_size: int,
    seq_lens: tuple[int, int],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
    backend: str,
):
    device = "musa"

    seqlen_qo = seq_lens[0]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    query, _, _ = gen_input_tensor(
        batch_size=batch_size,
        seqlens=seqlen_qo,
        num_head=head_qo,
        headdim=headdim_qk,
        use_cu_seqlens=False,
        datatype=dtype,
        device=device,
        randop=torch.randn,
    )

    seqlen_kv = seq_lens[1]
    head_kv = num_heads[0]
    headdim_vo = headdim[1]
    key, _, _ = gen_input_tensor(
        batch_size=batch_size,
        seqlens=seqlen_kv,
        num_head=head_kv,
        headdim=headdim_qk,
        use_cu_seqlens=False,
        datatype=dtype,
        device=device,
        randop=torch.randn,
    )
    value, _, _ = gen_input_tensor(
        batch_size=batch_size,
        seqlens=seqlen_kv,
        num_head=head_kv,
        headdim=headdim_vo,
        use_cu_seqlens=False,
        datatype=dtype,
        device=device,
        randop=torch.randn,
    )

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    softmax_scale = headdim_qk**-0.5

    metadata = None
    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=seqlen_qo,
            max_seqlen_k=seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=None,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )

    out_ref, _, score_ref = attention_ref(
        q=query,
        k=key,
        v=value,
        query_padding_mask=None,
        key_padding_mask=None,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=None,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    out, lse = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        seqused_q=None,
        seqused_k=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        learnable_sink=None,
        attention_chunk=0,
        softcap=0.0,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=False,
        sm_margin=0,
        return_softmax_lse=True,
        backend=backend,
    )

    _, lse_ref_pad = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        learnable_sink=None,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse_ref_pad, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("cp_world_size", [1, 2, 4])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize(
    "seq_lens",  # (seqlen_q, seqlen_kv) or (seqlen_q, seqlen_kv, seqlen_used_q, seqlen_used_kv)
    [
        [(32, 512), (192, 192), (128, 256)],
        [(32, 512, 32, -1), (192, 192, 64, 128), (128, 256, 64, 128)],
        [(i, i + 1234) for i in range(11)],
        [(55, 666), (256, 463), (111, 1328)],
        [(111, 1328), (55, 666), (222, 463)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (96, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@pytest.mark.parametrize("backend", ["auto", "mutlass"])
@torch.inference_mode()
def test_varlen_func_ragged_qkv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
    cp_world_size: int,
    backend: str,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    # prepare seqlen
    seqlens_qo, seqlens_kv, _, _, seqused_qo, seqused_kv = gen_seqlen_data(
        seq_lens, device
    )

    # prepare q data
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)

    q_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    q_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, seqused_qo, device
    )
    q_unpad, indices_q, cu_seqlens_qo, _, *rest = unpad_input(q_pad, q_padding_mask)

    # prepare kv data
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)

    k_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_qk, dtype=dtype, device=device
    )
    v_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_vo, dtype=dtype, device=device
    )
    kv_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_kv, batch_size, seqused_kv, device
    )
    k_unpad, indices_k, cu_seqlens_kv, _, *rest = unpad_input(k_pad, kv_padding_mask)
    v_unpad, indices_v, cu_seqlens_kv, _, *rest = unpad_input(v_pad, kv_padding_mask)

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    learnable_sink = None

    out_ref, _, score_ref = attention_ref(
        q=q_pad,
        k=k_pad,
        v=v_pad,
        query_padding_mask=q_padding_mask,
        key_padding_mask=kv_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    atol, rtol = 1.5e-2, 1e-2

    if cp_world_size > 1:
        # CP path: split KV interleaved across ranks, combine partial outputs.
        # Use effective seqlens (seqused if present) as the split boundary.
        eff_seqlens_kv = [
            seqused_kv[b].item() if seqused_kv is not None else seqlens_kv[b]
            for b in range(batch_size)
        ]
        cp_tot_seqused_k = torch.tensor(
            eff_seqlens_kv, dtype=torch.int32, device=device
        )

        rank_outs, rank_lses = [], []
        for cp_rank in range(cp_world_size):
            k_segs, v_segs, local_seqlens_k = [], [], []
            for b in range(batch_size):
                sk = eff_seqlens_kv[b]
                k_segs.append(k_pad[b, cp_rank:sk:cp_world_size].contiguous())
                v_segs.append(v_pad[b, cp_rank:sk:cp_world_size].contiguous())
                local_seqlens_k.append(
                    (sk - cp_rank + cp_world_size - 1) // cp_world_size
                )

            cu_seqlens_k_local = torch.tensor(
                [0] + torch.cumsum(torch.tensor(local_seqlens_k), dim=0).tolist(),
                dtype=torch.int32,
                device=device,
            )
            k_local = torch.cat(k_segs, dim=0)
            v_local = torch.cat(v_segs, dim=0)

            if num_splits >= 0:
                metadata = get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_qo,
                    max_seqlen_k=max(local_seqlens_k),
                    num_heads_q=head_qo,
                    num_heads_kv=head_kv,
                    headdim=headdim_qk,
                    seqused_q=seqused_qo,
                    seqused_k=None,
                    headdim_v=headdim_vo,
                    qkv_dtype=dtype,
                    cu_seqlens_q=cu_seqlens_qo,
                    cu_seqlens_k=cu_seqlens_k_local,
                    cu_seqlens_k_new=None,
                    cache_leftpad=None,
                    page_size=None,
                    max_seqlen_k_new=0,
                    causal=is_causal,
                    window_size=window_size,
                    attention_chunk=0,
                    has_softcap=False,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    mp_margin=0,
                )
            else:
                # not split
                metadata = None  # noqa: F841
                num_splits = -1

            out_rank, lse_rank = flash_attn_varlen_func(
                q=q_unpad,
                k=k_local,
                v=v_local,
                cu_seqlens_q=cu_seqlens_qo,
                cu_seqlens_k=cu_seqlens_k_local,
                max_seqlen_q=max_seqlen_qo,
                max_seqlen_k=max(local_seqlens_k),
                seqused_q=seqused_qo,
                seqused_k=None,
                softmax_scale=softmax_scale,
                causal=is_causal,
                qv=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                window_size=window_size,
                learnable_sink=learnable_sink,
                attention_chunk=0,
                softcap=0.0,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                deterministic=False,
                sm_margin=0,
                return_softmax_lse=True,
                backend=backend,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                cp_tot_seqused_k=cp_tot_seqused_k,
            )
            rank_outs.append(out_rank.float())
            rank_lses.append(lse_rank.float())

        # Combine per batch item (Q is ragged via cu_seqlens_qo)
        total_q = q_unpad.shape[0]
        combined_unpad = torch.zeros(
            total_q, head_qo, headdim_vo, dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            s = cu_seqlens_qo[b].item()
            e = cu_seqlens_qo[b + 1].item()
            if s == e:
                continue
            out_b, _ = _combine_cp_partials(
                [rank_outs[r][s:e] for r in range(cp_world_size)],
                [rank_lses[r][:, s:e] for r in range(cp_world_size)],
            )
            combined_unpad[s:e] = out_b

        out_pad = mask_unused(
            pad_input(combined_unpad.to(dtype), indices_q, batch_size, max_seqlen_qo),
            seqused_qo,
        )
        out_ref = mask_unused(out_ref, seqused_qo)
        torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)
        return

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=seqused_qo,
            seqused_k=seqused_kv,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=cu_seqlens_kv,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse = flash_attn_varlen_func(
        q=q_unpad,
        k=k_unpad,
        v=v_unpad,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_qo,
        max_seqlen_k=max_seqlen_kv,
        seqused_q=seqused_qo,
        seqused_k=seqused_kv,
        softmax_scale=softmax_scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=0.0,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=False,
        sm_margin=0,
        return_softmax_lse=True,
        backend=backend,
    )

    # lse_ref_unpad, _ = lse_ref_from_score(
    #     score_ref,
    #     is_causal=is_causal,
    #     cu_seqlens_q=cu_seqlens_qo,
    #     cu_seqlens_k=cu_seqlens_kv,
    #     seqused_q=seqused_qo,
    #     seqused_k=seqused_kv,
    #     learnable_sink=learnable_sink,
    # )

    out_pad = mask_unused(
        pad_input(out, indices_q, batch_size, max_seqlen_qo), seqused_qo
    )
    out_ref = mask_unused(out_ref, seqused_qo)

    atol, rtol = 1.5e-2, 1e-2

    # import pdb
    # pdb.set_trace()

    # torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@pytest.mark.skip(reason="Skip for CI. Avoid Long JIT")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("is_learnable_sink", [False, True])
@pytest.mark.parametrize("mask", [None, "causal", (44, 88)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(111, 1328), (55, 666), (222, 463)],
        [(111, 1328, 66, 888), (55, 666, 55, -1), (222, 463, -1, 256)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_varlen_func_ragged_qkv_advance(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    is_learnable_sink: bool,
    num_splits: int,
    softcap: float,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    # prepare seqlen
    seqlens_qo, seqlens_kv, _, _, seqused_qo, seqused_kv = gen_seqlen_data(
        seq_lens, device
    )

    # prepare q data
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)

    q_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    q_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, seqused_qo, device
    )
    q_unpad, indices_q, cu_seqlens_qo, _, *rest = unpad_input(q_pad, q_padding_mask)

    # prepare kv data
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)

    k_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_qk, dtype=dtype, device=device
    )
    v_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_vo, dtype=dtype, device=device
    )
    kv_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_kv, batch_size, seqused_kv, device
    )
    k_unpad, indices_k, cu_seqlens_kv, _, *rest = unpad_input(k_pad, kv_padding_mask)
    v_unpad, indices_v, cu_seqlens_kv, _, *rest = unpad_input(v_pad, kv_padding_mask)

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    is_local = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore
        is_local = True

    if is_learnable_sink:
        if not is_local:
            pytest.skip("learnable sink is tested for local attention")
        learnable_sink = torch.randn(head_qo, device=device, dtype=dtype) * 10
    else:
        learnable_sink = None

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=seqused_qo,
            seqused_k=seqused_kv,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=cu_seqlens_kv,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=softcap != 0.0,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse = flash_attn_varlen_func(
        q=q_unpad,
        k=k_unpad,
        v=v_unpad,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_qo,
        max_seqlen_k=max_seqlen_kv,
        seqused_q=seqused_qo,
        seqused_k=seqused_kv,
        softmax_scale=softmax_scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=softcap,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=False,
        sm_margin=0,
        return_softmax_lse=True,
        backend="mutlass",
    )

    out_ref, _, score_ref = attention_ref(
        q=q_pad,
        k=k_pad,
        v=v_pad,
        query_padding_mask=q_padding_mask,
        key_padding_mask=kv_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=softcap,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    # import pdb
    # pdb.set_trace()

    # lse_ref_unpad, lse_ref_pad = lse_ref_from_score(
    #     score_ref,
    #     is_causal=is_causal,
    #     cu_seqlens_q=cu_seqlens_qo,
    #     cu_seqlens_k=cu_seqlens_kv,
    #     seqused_q=seqused_qo,
    #     seqused_k=seqused_kv,
    #     learnable_sink=learnable_sink,
    # )

    out_pad = mask_unused(
        pad_input(out, indices_q, batch_size, max_seqlen_qo), seqused_qo
    )
    out_ref = mask_unused(out_ref, seqused_qo)

    atol, rtol = 1.5e-2, 1e-2

    # torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


# ==============================================================================
# ============================= flash_attn_with_kvcache ========================
# ==============================================================================


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("cp_world_size", [1, 2, 4])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True, False])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("page_size", [1, 3, 4, 16, 64, 111])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(32, 512), (192, 192), (128, 256)],
        [(i, i + 1234) for i in range(11)],
        [(55, 666), (222, 463), (111, 1328)],
        [(111, 1328), (55, 666), (222, 463)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (96, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_paged_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    page_size: int,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
    cp_world_size: int,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
    # total_seqlen_qo = sum(seqlens_qo)
    cu_seqlens_qo = torch.cumsum(
        torch.tensor([0] + seqlens_qo, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    query_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    query_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, None, device
    )
    query_unpad, indices_query, cu_seqlens_query, max_seqlen_query, *rest = unpad_input(
        query_pad, query_padding_mask
    )

    seqlens_kv = [x[1] for x in seq_lens]
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)
    # total_seqlen_kv = sum(seqlens_kv)
    # cu_seqlens_kv = torch.cumsum(
    #     torch.tensor([0] + seqlens_kv, dtype=torch.int32, device=device), dim=0
    # )
    cache_seqlens = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = generate_block_kvcache(
        max_seqlen_kv,
        page_size,
        batch_size,
        head_kv,
        headdim_qk,
        headdim_vo,
        device,
        dtype,
    )

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    learnable_sink = None

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")
    out_ref, _, score_ref = attention_ref(
        q=query_pad,
        k=k_cache_rep,
        v=v_cache_rep,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    atol, rtol = 1.5e-2, 1e-2

    if cp_world_size > 1:
        cp_tot_seqused_k = cache_seqlens.clone()

        rank_outs, rank_lses = [], []
        for cp_rank in range(cp_world_size):
            # pre-gather rank-local paged KV
            k_local, v_local, pt_local, local_seqlens = (
                make_cp_rank_local_paged_kvcache(
                    k_cache_paged,
                    v_cache_paged,
                    page_table,
                    cache_seqlens,
                    page_size,
                    cp_rank,
                    cp_world_size,
                    device,
                )
            )

            if num_splits >= 0:
                metadata = get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_qo,
                    max_seqlen_k=max(local_seqlens).item(),
                    num_heads_q=head_qo,
                    num_heads_kv=head_kv,
                    headdim=headdim_qk,
                    seqused_q=None,
                    seqused_k=local_seqlens,
                    headdim_v=headdim_vo,
                    qkv_dtype=dtype,
                    cu_seqlens_q=cu_seqlens_qo,
                    cu_seqlens_k=None,
                    cu_seqlens_k_new=None,
                    cache_leftpad=None,
                    page_size=page_size,
                    max_seqlen_k_new=0,
                    causal=is_causal,
                    window_size=window_size,
                    attention_chunk=0,
                    has_softcap=False,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    mp_margin=0,
                )
            else:
                # not split
                metadata = None  # noqa: F841
                num_splits = -1

            out_rank, lse_rank, *_ = flash_attn_with_kvcache(
                q=query_unpad,
                k_cache=k_local,
                v_cache=v_local,
                cache_seqlens=local_seqlens,  # rank-local seqlen
                page_table=pt_local,
                cu_seqlens_q=cu_seqlens_qo,
                max_seqlen_q=max_seqlen_qo,
                softmax_scale=softmax_scale,
                causal=is_causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=0,
                return_softmax_lse=True,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                cp_tot_seqused_k=cp_tot_seqused_k,
            )
            rank_outs.append(out_rank.float())
            rank_lses.append(lse_rank.float())

        total_q = query_unpad.shape[0]
        combined_unpad = torch.zeros(
            total_q, head_qo, headdim_vo, dtype=torch.float32, device=device
        )
        cu_q = cu_seqlens_qo.tolist()
        for b in range(batch_size):
            s, e = cu_q[b], cu_q[b + 1]
            if s == e:
                continue
            out_b, _ = _combine_cp_partials(
                [rank_outs[r][s:e] for r in range(cp_world_size)],
                [rank_lses[r][:, s:e] for r in range(cp_world_size)],
            )
            combined_unpad[s:e] = out_b

        out_pad = pad_input(
            combined_unpad.to(dtype), indices_query, batch_size, max_seqlen_query
        )
        torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)
        return

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=cache_seqlens,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=page_size,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
        # print(metadata[:4*batch_size].view(4, -1))

        # TODO: too many feat unsupported
        # out_accum_ref, lse_accum_ref = attention_accum_ref(
        #     query=query_pad,
        #     key=k_cache_rep,
        #     value=v_cache_rep,
        #     metadata=metadata,
        #     num_splits=num_splits,
        #     softmax_scale=softmax_scale,
        #     seqused_q=None,
        #     seqused_kv=cache_seqlens,
        #     cu_seqlens_q=cu_seqlens_qo,
        #     cu_seqlens_kv=None,
        # )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse, out_accum, lse_accum, *rest = flash_attn_with_kvcache(
        q=query_unpad,
        k_cache=k_cache if page_size is None else k_cache_paged,
        v_cache=v_cache if page_size is None else v_cache_paged,
        k=None,
        v=None,
        qv=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k_new=None,
        max_seqlen_q=max_seqlen_qo,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_softmax_lse=True,
    )

    if num_splits >= 0:
        out_accum = torch.utils.dlpack.from_dlpack(out_accum)
        lse_accum = torch.utils.dlpack.from_dlpack(lse_accum)
        out_accum_pad, lse_accum_pad = pad_accum(
            out_accum=out_accum,
            lse_accum=lse_accum,
            seqused_q=None,
            cu_seqlens_q=cu_seqlens_qo,
        )

    out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)
    lse_ref, _ = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=learnable_sink,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@pytest.mark.skip(reason="Skip for CI. Avoid Long JIT")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True, False])
@pytest.mark.parametrize("is_learnable_sink", [True, False])
@pytest.mark.parametrize("is_varlen_q", [True, False])
@pytest.mark.parametrize("mask", [None, "causal", (44, 88)])
@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(111, 1328), (55, 666), (222, 463)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_paged_attn_advance(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    page_size: int,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    is_learnable_sink: bool,
    is_varlen_q: bool,
    num_splits: int,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
    # total_seqlen_qo = sum(seqlens_qo)
    cu_seqlens_qo = torch.cumsum(
        torch.tensor([0] + seqlens_qo, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    query_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    query_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, None, device
    )
    query_unpad, indices_query, cu_seqlens_query, max_seqlen_query, *rest = unpad_input(
        query_pad, query_padding_mask
    )

    seqlens_kv = [x[1] for x in seq_lens]
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)
    # total_seqlen_kv = sum(seqlens_kv)
    # cu_seqlens_kv = torch.cumsum(
    #     torch.tensor([0] + seqlens_kv, dtype=torch.int32, device=device), dim=0
    # )
    cache_seqlens = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = generate_block_kvcache(
        max_seqlen_kv,
        page_size,
        batch_size,
        head_kv,
        headdim_qk,
        headdim_vo,
        device,
        dtype,
    )

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    is_local = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore
        is_local = True

    if is_learnable_sink:
        if not is_local:
            pytest.skip("learnable sink is tested for local attention")
        learnable_sink = torch.randn(head_qo, device=device, dtype=dtype) * 10
    else:
        learnable_sink = None

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")
    out_ref, _, score_ref = attention_ref(
        q=query_pad,
        k=k_cache_rep,
        v=v_cache_rep,
        query_padding_mask=query_padding_mask if is_varlen_q else None,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=cache_seqlens,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo if is_varlen_q else None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=page_size,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
        # print(metadata[:4*batch_size].view(4, -1))

        # TODO: too many feat unsupported
        # out_accum_ref, lse_accum_ref = attention_accum_ref(
        #     query=query_pad,
        #     key=k_cache_rep,
        #     value=v_cache_rep,
        #     metadata=metadata,
        #     num_splits=num_splits,
        #     softmax_scale=softmax_scale,
        #     seqused_q=None,
        #     seqused_kv=cache_seqlens,
        #     cu_seqlens_q=cu_seqlens_qo,
        #     cu_seqlens_kv=None,
        # )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse, out_accum, lse_accum, *rest = flash_attn_with_kvcache(
        q=query_unpad if is_varlen_q else query_pad,
        k_cache=k_cache if page_size is None else k_cache_paged,
        v_cache=v_cache if page_size is None else v_cache_paged,
        k=None,
        v=None,
        qv=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_qo if is_varlen_q else None,
        cu_seqlens_k_new=None,
        max_seqlen_q=max_seqlen_qo if is_varlen_q else None,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_softmax_lse=True,
    )

    # if num_splits >= 0:
    #     out_accum = torch.utils.dlpack.from_dlpack(out_accum)
    #     lse_accum = torch.utils.dlpack.from_dlpack(lse_accum)
    #     if is_varlen_q:
    #         out_accum_pad, lse_accum_pad = pad_accum(
    #             out_accum=out_accum,
    #             lse_accum=lse_accum,
    #             seqused_q=None,
    #             cu_seqlens_q=cu_seqlens_qo,
    #         )
    #     else:
    #         out_accum_pad = out_accum
    #         lse_accum_pad = lse_accum

    if is_varlen_q:
        out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)
    else:
        out_pad = out
    lse_ref_unpad, lse_ref_pad = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo if is_varlen_q else None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=learnable_sink,
    )
    lse_ref = lse_ref_unpad if is_varlen_q else lse_ref_pad

    atol, rtol = 1.5e-2, 1e-2

    torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@pytest.mark.skip(reason="Will be verified in fwd PR.")
@pytest.mark.parametrize("seqlen_q", [1, 3])
@pytest.mark.parametrize("headdim_v", [128, 256])
@pytest.mark.parametrize("num_head", [1, 4, 5, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "ragged", "padded"])
@pytest.mark.parametrize("reorder", [False])
@pytest.mark.parametrize("batch_size", [1, 5, 32])
@pytest.mark.parametrize("max_splits", [5])
def test_combine(
    max_splits: int,
    batch_size: int,
    reorder: bool,
    mode: str,
    dtype: torch.dtype,
    num_head: int,
    headdim_v: int,
    seqlen_q: int,
) -> None:
    device = torch.device("musa")

    # torch.manual_seed(666)
    # torch.musa.manual_seed(666)

    assert not reorder, "Reorder is not supported yet."

    # Prepare metadata
    num_splits = torch.randint(
        2, max_splits + 1, (batch_size,), device=device, dtype=torch.int32
    )
    batch_table = torch.arange(batch_size, device=device, dtype=torch.int32)
    batch_table = batch_table[torch.randperm(batch_size)] if reorder else batch_table
    metadata = torch.cat(
        [
            num_splits,
            batch_table,
            torch.empty_like(num_splits),
            torch.empty_like(num_splits),
        ],
        dim=0,
    )

    # Prepare data
    cu_seqlens_q, seqused_q, max_seqlen_q = None, None, None
    if mode in ("padded", "normal"):
        shape_oaccum = (batch_size, num_head, max_splits, seqlen_q, headdim_v)  # type: ignore[assignment]
        shape_lseaccum = (batch_size, num_head, max_splits, seqlen_q)  # type: ignore[assignment]
        shape_out = (batch_size, seqlen_q, num_head, headdim_v)  # type: ignore[assignment]
        shape_lse = (batch_size, num_head, seqlen_q)  # type: ignore[assignment]
    elif mode == "ragged":
        seqlens_q = torch.randint(
            1, seqlen_q + 1, (batch_size,), device=device, dtype=torch.int32
        )
        cu_seqlens_q = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                seqlens_q.cumsum(dim=0),
            ]
        ).to(torch.int32)
        total_q = cu_seqlens_q[-1].item()
        max_seqlen_q = seqlen_q
        shape_oaccum = (1, num_head, max_splits, total_q, headdim_v)  # type: ignore[assignment]
        shape_lseaccum = (1, num_head, max_splits, total_q)  # type: ignore[assignment]
        shape_out = (total_q, num_head, headdim_v)  # type: ignore[assignment]
        shape_lse = (num_head, total_q)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "padded":
        seqused_q = torch.randint(
            1, seqlen_q + 1, (batch_size,), device=device, dtype=torch.int32
        )
        # max_seqlen_q = seqlen_q

    out_accum = torch.randn(shape_oaccum, device=device, dtype=torch.float32)
    lse_accum = torch.randn(shape_lseaccum, device=device, dtype=torch.float32)
    out = torch.empty(shape_out, device=device, dtype=dtype)
    lse = torch.empty(shape_lse, device=device, dtype=torch.float32)

    fmha_fwd_combine(
        out,
        lse,
        out_accum,
        lse_accum,
        64,  # tile_n
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_splits,
        metadata,
    )

    # Mask Seqlen
    if mode == "padded":
        for batch_idx, cur_seqlen_q in enumerate(seqused_q):
            out_accum[batch_idx, ..., cur_seqlen_q:, :] = 0
            lse_accum[batch_idx, ..., cur_seqlen_q:] = float("-inf")

    # Mask Splits
    for batch_idx, cur_splits in enumerate(num_splits):
        cur_splits = cur_splits.item()
        if cur_splits < max_splits:
            if mode == "ragged":
                # pdb.set_trace()
                out_accum[
                    ...,
                    cur_splits:,
                    cu_seqlens_q[batch_idx] : cu_seqlens_q[batch_idx + 1],
                    :,
                ] = 0
                lse_accum[
                    ...,
                    cur_splits:,
                    cu_seqlens_q[batch_idx] : cu_seqlens_q[batch_idx + 1],
                ] = float("-inf")
            else:
                out_accum[batch_idx, :, cur_splits:, :, :] = 0
                lse_accum[batch_idx, :, cur_splits:, :] = float("-inf")

    ref_out, ref_lse = attention_combine_ref(
        out_accum,
        lse_accum,
    )

    if mode == "ragged":
        ref_out.squeeze_(0)
        ref_lse.squeeze_(0)
    elif mode == "padded":
        for batch_idx, cur_seqlen_q in enumerate(seqused_q):
            cur_seqlen_q = cur_seqlen_q.item()
            ref_out[batch_idx, cur_seqlen_q:] = 0
            ref_lse[batch_idx, ..., cur_seqlen_q:] = 0
            out[batch_idx, cur_seqlen_q:] = 0
            lse[batch_idx, ..., cur_seqlen_q:] = 0

    torch.set_printoptions(sci_mode=False)
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "cache_seqlen_list",
    [
        [1],
        [1, 63],
        [63, 64, 65],
        [1, 63, 64, 127],
    ],
)
@pytest.mark.parametrize("append_seqlen", [1, 3, 7])
@torch.inference_mode()
def test_paged_attn_appendkv(cache_seqlen_list: list[int], append_seqlen: int):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"
    dtype = torch.bfloat16
    batch_size = len(cache_seqlen_list)
    seqlen_q = 1
    total_seqlens_kv = [
        cache_seqlen + append_seqlen for cache_seqlen in cache_seqlen_list
    ]
    max_total_seqlen_kv = max(total_seqlens_kv)
    head_qo = 8
    head_kv = 1
    headdim_qk = 128
    headdim_vo = 128
    page_size = 64

    q = torch.randn(
        batch_size, seqlen_q, head_qo, headdim_qk, device=device, dtype=dtype
    )
    k_new = torch.randn(
        batch_size, append_seqlen, head_kv, headdim_qk, device=device, dtype=dtype
    )
    v_new = torch.randn(
        batch_size, append_seqlen, head_kv, headdim_vo, device=device, dtype=dtype
    )
    cache_seqlens = torch.tensor(cache_seqlen_list, dtype=torch.int32, device=device)
    total_seqlens = torch.tensor(total_seqlens_kv, dtype=torch.int32, device=device)

    k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, _ = (
        generate_block_kvcache(
            max_total_seqlen_kv,
            page_size,
            batch_size,
            head_kv,
            headdim_qk,
            headdim_vo,
            device,
            dtype,
        )
    )
    k_cache = k_cache.clone().fill_(0)
    v_cache = v_cache.clone().fill_(0)
    k_cache_paged = k_cache_paged.clone().fill_(0)
    v_cache_paged = v_cache_paged.clone().fill_(0)

    softmax_scale = headdim_qk**-0.5

    k_full = k_cache.clone()
    v_full = v_cache.clone()
    for batch_idx, cache_seqlen in enumerate(cache_seqlen_list):
        logical_slice = slice(cache_seqlen, cache_seqlen + append_seqlen)
        k_full[batch_idx, logical_slice] = k_new[batch_idx]
        v_full[batch_idx, logical_slice] = v_new[batch_idx]
    key_padding_mask = rearrange(
        torch.arange(max_total_seqlen_kv, device=device), "s -> 1 s"
    ) < rearrange(total_seqlens, "b -> b 1")
    out_ref, _, score_ref = attention_ref(
        q=q,
        k=k_full,
        v=v_full,
        query_padding_mask=None,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=True,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(None, None),
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=None,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )
    _, lse_ref = lse_ref_from_score(
        score_ref,
        is_causal=True,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=total_seqlens,
        learnable_sink=None,
    )

    out, lse, *_ = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        k=k_new,
        v=v_new,
        qv=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=page_table,
        cu_seqlens_q=None,
        cu_seqlens_k_new=None,
        max_seqlen_q=None,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(None, None),
        learnable_sink=None,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=-1,
        pack_gqa=True,
        sm_margin=0,
        return_softmax_lse=True,
    )
    torch.musa.synchronize()

    atol, rtol = 1.5e-2, 1e-2
    try:
        for batch_idx, cache_seqlen in enumerate(cache_seqlen_list):
            for append_idx in range(append_seqlen):
                logical_pos = cache_seqlen + append_idx
                page_idx = logical_pos // page_size
                page_offset = logical_pos % page_size
                physical_page = page_table[batch_idx, page_idx]
                torch.testing.assert_close(
                    k_cache_paged[physical_page, page_offset],
                    k_new[batch_idx, append_idx],
                    atol=atol,
                    rtol=rtol,
                )
                torch.testing.assert_close(
                    v_cache_paged[physical_page, page_offset],
                    v_new[batch_idx, append_idx],
                    atol=atol,
                    rtol=rtol,
                )

        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
        assert lse.shape == (batch_size, head_qo, seqlen_q)
    except AssertionError:
        raise
