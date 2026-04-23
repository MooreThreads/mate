# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
from mate import flash_attn_varlen_func
from typing import Optional  # noqa: F401


def ref_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    max_query_len: int,
    max_kv_len: int,
    is_causal: bool = True,
    is_varlen: bool = True,
    scale: float = 1.0,
):
    batch = len(cu_query_lens) - 1 if is_varlen else query.shape[0]
    # head_size_qk = key_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    lse_outputs: list[torch.Tensor] = []

    for i in range(batch):
        query_len = (
            cu_query_lens[i + 1] - cu_query_lens[i] if is_varlen else query.shape[1]
        )
        kv_len = (
            cu_kv_lens[i + 1] - cu_kv_lens[i] if is_varlen else value_cache.shape[1]
        )
        if is_varlen:
            q = query[cu_query_lens[i] : cu_query_lens[i + 1]]
            q = q * scale

            k = key_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
            v = value_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
        else:
            q = query[i]
            q = q * scale
            k = key_cache[i]
            v = value_cache[i]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        # raw attention score
        attn = torch.einsum("qhd,khd->hqk", q, k).float()

        # mask
        if is_causal:
            empty_mask = torch.ones(query_len, kv_len, device=query.device)
            mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
            attn.masked_fill_(mask, float("-inf"))

        # LSE (head, query)
        lse = torch.logsumexp(attn, dim=-1)  # shape [H, Q]
        lse = torch.nan_to_num(lse)
        lse_outputs.append(lse)

        # softmax = exp(score - lse)
        attn = torch.exp(attn - lse.unsqueeze(-1)).to(v.dtype)

        # output
        out = torch.einsum("hqk,khd->qhd", attn, v)
        out = torch.nan_to_num(out)
        outputs.append(out)
    if is_varlen:
        return torch.cat(outputs, dim=0), torch.cat(lse_outputs, dim=1)
    else:
        return torch.stack(outputs, dim=0), torch.stack(lse_outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
        ]
    ],
)
@pytest.mark.parametrize("num_heads", [(40, 40)])
@pytest.mark.parametrize("head_size", [(128, 128), (192, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [True])
def test_varlen_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
) -> None:
    torch.set_default_device("musa")
    torch.manual_seed(0)
    torch.musa.manual_seed(0)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    head_size_qk = head_size[0]
    head_size_v = head_size[1]
    scale = head_size_qk**-0.5

    if is_varlen:
        query = torch.randn(
            (sum(query_lens), num_query_heads, head_size_qk),
            dtype=dtype,
            requires_grad=True,
        )
        key_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_qk), dtype=dtype, requires_grad=True
        )
        value_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_v), dtype=dtype, requires_grad=True
        )
        query.retain_grad()
        key_cache.retain_grad()
        value_cache.retain_grad()
    else:
        return

    def clone_like(t):
        c = t.clone().detach().requires_grad_(True)
        return c

    q_fa, k_fa, v_fa = map(clone_like, (query, key_cache, value_cache))
    q_t, k_t, v_t = map(clone_like, (query, key_cache, value_cache))

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    cu_kv_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        causal=is_causal,
    )

    ref_output, _ = ref_attn(
        query=q_t,
        key_cache=k_t,
        value_cache=v_t,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
    )
    grad_out = torch.randn_like(out_fa)

    grad_fa = clone_like(grad_out)
    grad_t = clone_like(grad_out)

    out_fa.backward(gradient=grad_fa, retain_graph=True)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    ref_output.backward(gradient=grad_t, retain_graph=True)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    atol, rtol = 2e-2, 2e-2

    torch.testing.assert_close(out_fa, ref_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(dq_fa, dq_t, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk_fa, dk_t, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv_fa, dv_t, atol=atol, rtol=rtol)
