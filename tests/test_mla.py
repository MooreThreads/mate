# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch_musa  # noqa: F401

import mate
from mate import mla


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    # v = ckv.repeat_interleave(num_heads, dim=1)
    v = ckv.view(batch_size, kv_len, 1, ckv_dim).repeat_interleave(num_heads, dim=2)

    return k, v


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    batch_size = q.shape[0]
    qo_len = q.shape[1]
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[2]
    head_dim_qk = q.shape[3]
    head_dim_vo = v.shape[3]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .to(q)
    )

    # return o_ref, lse_ref * math.log2(math.e)
    return o_ref, lse_ref


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7, 157])
@pytest.mark.parametrize("kv_len", [33, 64, 96, 97, 114, 514, 1024])
@pytest.mark.parametrize("qo_len", [1, 3, 5, 7, 9, 11, 13, 15, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("dtype", [torch.half])
def test_batch_mla_page_attention(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    dtype,
):
    device = torch.device("musa")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        batch_size, qo_len, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size, qo_len, num_heads, head_dim_kpe, dtype=dtype, device=device
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    page_table = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    ).reshape(batch_size, pages_num)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)

    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    o, lse = mla(
        q_nope,
        q_pe,
        ckv,
        kpe,
        page_table=page_table,
        kv_len=kv_lens,
        sm_scale=sm_scale,
        out=None,
        lse=None,
        is_causal=causal,
    )
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(o, o_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("batch_size", [33])
@pytest.mark.parametrize("kv_len", [33, 77, 111])
@pytest.mark.parametrize("qo_len", [1])
@pytest.mark.parametrize("num_heads", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("dtype", [torch.half])
def test_flashmla_api(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    dtype,
):
    device = torch.device("musa")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        qo_len, num_heads, batch_size, head_dim_ckv, dtype=dtype, device=device
    ).permute(2, 0, 1, 3)
    q_pe = torch.randn(
        qo_len, num_heads, batch_size, head_dim_kpe, dtype=dtype, device=device
    ).permute(2, 0, 1, 3)
    pages_num = math.ceil(kv_len / page_size)
    num_layers = 3
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        num_layers,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        num_layers,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    ckv = ckv[:, :, 1, :]
    kpe = kpe[:, :, 1, :]

    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    page_table = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    ).reshape(batch_size, pages_num)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = torch.cat([q_nope, q_pe], dim=-1)

    o_ref, lse_ref = attention_ref(
        batch_size, q.to("cpu"), k.to("cpu"), v.to("cpu"), causal, sm_scale
    )
    o_ref = o_ref.to(device)
    lse_ref = lse_ref.to(device)

    tile_scheduler_metadata, num_splits = mate.flashmla.get_mla_metadata(
        kv_lens, qo_len * num_heads // 1, 1
    )
    o, lse = mate.flashmla.flash_mla_with_kvcache(
        q,
        torch.cat([ckv, kpe], dim=-1).unsqueeze(-2),
        page_table,
        kv_lens,
        head_dim_ckv,
        tile_scheduler_metadata,
        num_splits,
        sm_scale,
        causal,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(o, o_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
