# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
from mate import flash_attn_varlen_func, flash_attn_with_kvcache
from typing import Optional


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
            q *= scale

            k = key_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
            v = value_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
        else:
            q = query[i]
            q *= scale
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


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens,
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    verlen_q = query_lens is not None
    num_seqs = len(query_lens) if verlen_q else query.shape[0]
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    lse_outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i] if verlen_q else query.shape[1]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] if verlen_q else query[i]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]
        head_num_q = q.shape[1]
        if head_num_q != k.shape[1]:
            k = torch.repeat_interleave(k, head_num_q // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, head_num_q // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=query.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if is_causal:
            attn.masked_fill_(mask, float("-inf"))

        lse = torch.logsumexp(attn, dim=-1)  # shape [H, Q]
        lse_outputs.append(lse)

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len
        if not verlen_q:
            start_idx = 0
    if verlen_q:
        return torch.cat(outputs, dim=0), torch.cat(lse_outputs, dim=1)
    else:
        return torch.stack(outputs, dim=0), torch.stack(lse_outputs, dim=0)


@pytest.mark.parametrize("seq_lens", [[(1024, 1024), (333, 888), (555, 222)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2), (16, 1)])
@pytest.mark.parametrize("head_size", [(128, 128), (192, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize("batch", [8])
@pytest.mark.parametrize("is_varlen", [False, True])
@pytest.mark.parametrize("backend", ["auto"])
@torch.inference_mode()
def test_varlen_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    batch: int,
    is_varlen: bool,
    backend: str,
) -> None:
    if backend == "mubin" and not is_causal:
        pytest.skip("mubin backend does not support non-causal attention")

    torch.set_default_device("musa")

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

    if not is_varlen:
        query = torch.rand(
            batch, max_query_len, num_query_heads, head_size_qk, dtype=dtype
        )
        key_cache = torch.rand(
            batch, max_kv_len, num_kv_heads, head_size_qk, dtype=dtype
        )
        value_cache = torch.rand(
            batch, max_kv_len, num_kv_heads, head_size_v, dtype=dtype
        )
    else:
        query = torch.rand(
            (sum(query_lens), num_query_heads, head_size_qk), dtype=dtype
        )
        key_cache = torch.rand((sum(kv_lens), num_kv_heads, head_size_qk), dtype=dtype)
        value_cache = torch.rand((sum(kv_lens), num_kv_heads, head_size_v), dtype=dtype)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    cu_kv_lens = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    output, lse = flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_query_lens if is_varlen else None,
        cu_kv_lens if is_varlen else None,
        max_query_len if is_varlen else None,
        max_kv_len if is_varlen else None,  # max_seqlen_q/k
        None,  # seqused_q
        None,  # seqused_k
        None,  # softmax_scale
        is_causal,
        None,  # qv
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_attn_probs=True,
        backend=backend,
    )

    ref_output, ref_lse = ref_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
    )

    atol, rtol = 1.5e-2, 1e-2

    if backend == "mutlass":
        lse = torch.nan_to_num(lse)
        output = torch.nan_to_num(output)

    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("use_out", [True])
@pytest.mark.parametrize("seq_lens", [[(2, 1328), (2, 18), (2, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2), (16, 1)])
@pytest.mark.parametrize("head_size", [(128, 128)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
@pytest.mark.parametrize("is_causal", [False, True])
@torch.inference_mode()
def test_varlen_with_paged_kv(
    use_out: bool,
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    is_causal,
) -> None:
    torch.set_default_device("musa")
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size[0] ** -0.5

    query = torch.randn(
        len(query_lens), max_query_len, num_query_heads, head_size[0], dtype=dtype
    )
    key_cache = (
        torch.randn(num_blocks, block_size, num_kv_heads, head_size[1], dtype=dtype)
        / 10
    )
    value_cache = torch.randn_like(key_cache) / 10
    # cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
    #     dim=0, dtype=torch.int32
    # )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output, lse, *rest = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        None,
        None,
        None,
        kv_lens,  # cache_seqlens
        None,  # cache_batch_idx
        None,
        block_tables,  # page_table
        None,  # cu_query_lens
        None,
        max_query_len,  # max_seqlen_q
        None,
        None,
        None,
        None,
        scale,
        is_causal,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=True,
    )

    torch.set_default_device("musa")
    ref_output, ref_lse = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=None,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        is_causal=is_causal,
    )
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)
