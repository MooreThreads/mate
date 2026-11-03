# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
from mate import flash_attn_varlen_func, fmha_backward
from mate.execution_context import (
    is_dry_run_enabled,
    raise_complete_if_dry_run,
)
from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (
    flashattn_varlen_bwd_interface,
)
from typing import NamedTuple, Optional  # noqa: F401
from mate.testing import supported_musa_compute_capability


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
    upcast: bool = True,
    reorder_ops: bool = False,
):
    batch = len(cu_query_lens) - 1 if is_varlen else query.shape[0]
    dtype_og = query.dtype
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

            k = key_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
            v = value_cache[cu_kv_lens[i] : cu_kv_lens[i + 1]]
        else:
            q = query[i]
            k = key_cache[i]
            v = value_cache[i]

        if upcast:
            q, k, v = q.float(), k.float(), v.float()

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        # raw attention score
        if reorder_ops:
            attn = torch.einsum("qhd,khd->hqk", q, k * scale)
        else:
            attn = torch.einsum("qhd,khd->hqk", q * scale, k)
        attn = attn.float()

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
        outputs.append(out.to(dtype=dtype_og))
    if is_varlen:
        return torch.cat(outputs, dim=0), torch.cat(lse_outputs, dim=1)
    else:
        return torch.stack(outputs, dim=0), torch.stack(lse_outputs, dim=0)


def clone_with_grad(t: torch.Tensor) -> torch.Tensor:
    return t.clone().detach().requires_grad_(True)


class FmhaRunResult(NamedTuple):
    out: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


def strided_last_dim_tensor(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    pad: int = 8,
) -> torch.Tensor:
    base = torch.randn((*shape[:-1], shape[-1] + pad), device=device, dtype=dtype)
    tensor = base[..., : shape[-1]].detach()
    if requires_grad:
        tensor.requires_grad_(True)
    assert tensor.stride(-1) == 1
    assert not tensor.is_contiguous()
    return tensor


def copy_to_strided_last_dim(tensor: torch.Tensor, pad: int = 8) -> torch.Tensor:
    base = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] + pad),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    out = base[..., : tensor.shape[-1]]
    out.copy_(tensor)
    assert out.stride(-1) == 1
    assert not out.is_contiguous()
    return out


def assert_fa_error_within_pt(
    actual: torch.Tensor,
    ref: torch.Tensor,
    pt: torch.Tensor,
    name: str,
    *,
    multiplier: float = 2.0,
) -> None:
    actual_err = (actual - ref).abs().max().item()
    pt_err = (pt - ref).abs().max().item()
    atol_floor = 2 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    assert actual_err <= multiplier * pt_err + atol_floor, (
        f"{name} max error {actual_err} exceeds "
        f"{multiplier}x PyTorch low-precision baseline {pt_err} "
        f"+ numerical floor {atol_floor}"
    )


def fa_error_limit(
    ref: torch.Tensor,
    pt: torch.Tensor,
    *,
    multiplier: float = 2.0,
) -> float:
    pt_err = (pt - ref).abs().max().item()
    atol_floor = 2 * (ref + 0.3 - 0.3 - ref).abs().max().item()
    return multiplier * pt_err + atol_floor


def _detach_result(
    out: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> FmhaRunResult:
    return FmhaRunResult(
        out.detach().clone(),
        dq.detach().clone(),
        dk.detach().clone(),
        dv.detach().clone(),
    )


def _run_fmha_case(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    cu_query_lens: Optional[torch.Tensor],
    cu_kv_lens: Optional[torch.Tensor],
    max_query_len: Optional[int],
    max_kv_len: Optional[int],
    is_causal: bool,
    deterministic: bool,
) -> FmhaRunResult:
    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))
    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        causal=is_causal,
        deterministic=deterministic,
    )
    out_fa.backward(gradient=grad_out.detach().clone())
    torch.musa.synchronize()
    return _detach_result(out_fa, q_fa.grad, k_fa.grad, v_fa.grad)


def _assert_bitwise_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str,
    iteration: int,
) -> None:
    if torch.equal(actual, expected):
        return
    max_abs = (actual.float() - expected.float()).abs().max().item()
    raise AssertionError(
        f"{name} changed at FMHA BWD stress iteration {iteration}: "
        f"max_abs_diff={max_abs}"
    )


def _stress_fmha_bwd_case(
    pytestconfig: pytest.Config,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    cu_query_lens: Optional[torch.Tensor],
    cu_kv_lens: Optional[torch.Tensor],
    max_query_len: Optional[int],
    max_kv_len: Optional[int],
    is_causal: bool,
    deterministic: bool,
    ref_result: FmhaRunResult,
    pt_result: FmhaRunResult,
) -> None:
    iterations = pytestconfig.getoption("fmha_bwd_stress_iters")
    if iterations < 0:
        raise pytest.UsageError("--fmha-bwd-stress-iters must be >= 0")
    if iterations <= 0:
        return
    mode = pytestconfig.getoption("fmha_bwd_stress_mode")
    progress_interval = pytestconfig.getoption("fmha_bwd_stress_progress_interval")
    if progress_interval < 0:
        raise pytest.UsageError("--fmha-bwd-stress-progress-interval must be >= 0")
    baseline: Optional[FmhaRunResult] = None

    for iteration in range(1, iterations + 1):
        try:
            current = _run_fmha_case(
                query,
                key_cache,
                value_cache,
                grad_out,
                cu_query_lens=cu_query_lens,
                cu_kv_lens=cu_kv_lens,
                max_query_len=max_query_len,
                max_kv_len=max_kv_len,
                is_causal=is_causal,
                deterministic=deterministic,
            )
        except Exception as exc:
            layout = "varlen" if cu_query_lens is not None else "bshd"
            raise RuntimeError(
                "FMHA BWD stress failed "
                f"at iteration {iteration}/{iterations}, "
                f"mode={mode}, layout={layout}, causal={is_causal}, "
                f"deterministic={deterministic}"
            ) from exc
        if progress_interval and iteration % progress_interval == 0:
            print(
                f"FMHA BWD stress progress: iteration {iteration}/{iterations}, "
                f"mode={mode}, deterministic={deterministic}",
                flush=True,
            )
        if mode == "kernel-only":
            continue

        if not deterministic:
            assert_fa_error_within_pt(
                current.dq,
                ref_result.dq,
                pt_result.dq,
                f"dq stress iter {iteration}",
            )

        if baseline is None:
            baseline = current
            continue

        _assert_bitwise_equal(current.out, baseline.out, "out", iteration)
        _assert_bitwise_equal(current.dk, baseline.dk, "dk", iteration)
        _assert_bitwise_equal(current.dv, baseline.dv, "dv", iteration)
        if deterministic:
            _assert_bitwise_equal(current.dq, baseline.dq, "dq", iteration)


def _dry_run_fmha_forward_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    is_causal: bool,
    deterministic: bool,
) -> None:
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=is_causal,
        deterministic=deterministic,
    )

    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    if q.ndim == 3:
        softmax_lse = torch.empty(
            (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
        )
    else:
        softmax_lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    flashattn_varlen_bwd_interface(
        q,
        k,
        v,
        out,
        torch.empty_like(out),
        softmax_lse,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        is_causal=is_causal,
        smscale=q.shape[-1] ** -0.5,
        dtype=None,
        is_bhsd=False,
        deterministic=deterministic,
    )
    raise_complete_if_dry_run()


@pytest.mark.mate_dry_run
@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [
            (512, 512),
            (511, 511),
            (384, 384),
            (257, 257),
            (128, 128),
            (65, 65),
            (320, 320),
            (192, 192),
        ]
    ],
)
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 6)])
@pytest.mark.parametrize(
    "head_size",
    [(64, 64), (100, 80), (128, 128), (192, 128), (192, 192), (256, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [True])
@pytest.mark.parametrize("deterministic", [True, False])
def test_varlen_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool,
    pytestconfig: pytest.Config,
) -> None:
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    device = torch.device("musa")
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
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        key_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_qk),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        value_cache = torch.randn(
            (sum(kv_lens), num_kv_heads, head_size_v),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query.retain_grad()
        key_cache.retain_grad()
        value_cache.retain_grad()
    else:
        return

    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))

    cu_query_lens = torch.tensor(
        [0] + query_lens, device=device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)

    cu_kv_lens = torch.tensor([0] + kv_lens, device=device, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )

    if is_dry_run_enabled():
        _dry_run_fmha_forward_backward(
            q_fa,
            k_fa,
            v_fa,
            cu_seqlens_q=cu_query_lens,
            cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            is_causal=is_causal,
            deterministic=deterministic,
        )
        return

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        causal=is_causal,
        deterministic=deterministic,
    )

    grad_out = torch.randn_like(out_fa)
    q_t, k_t, v_t = map(clone_with_grad, (query, key_cache, value_cache))
    q_pt, k_pt, v_pt = map(clone_with_grad, (query, key_cache, value_cache))

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
        upcast=True,
    )
    pt_output, _ = ref_attn(
        query=q_pt,
        key_cache=k_pt,
        value_cache=v_pt,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    grad_fa = clone_with_grad(grad_out)
    grad_t = clone_with_grad(grad_out)
    grad_pt = clone_with_grad(grad_out)

    out_fa.backward(gradient=grad_fa, retain_graph=True)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    ref_output.backward(gradient=grad_t, retain_graph=True)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    pt_output.backward(gradient=grad_pt, retain_graph=True)
    dq_pt, dk_pt, dv_pt = q_pt.grad, k_pt.grad, v_pt.grad

    assert_fa_error_within_pt(out_fa, ref_output, pt_output, "out")
    assert_fa_error_within_pt(dk_fa, dk_t, dk_pt, "dk")
    assert_fa_error_within_pt(dv_fa, dv_t, dv_pt, "dv")
    assert_fa_error_within_pt(dq_fa, dq_t, dq_pt, "dq")

    _stress_fmha_bwd_case(
        pytestconfig,
        query,
        key_cache,
        value_cache,
        grad_out,
        cu_query_lens=cu_query_lens,
        cu_kv_lens=cu_kv_lens,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        deterministic=deterministic,
        ref_result=_detach_result(ref_output, dq_t, dk_t, dv_t),
        pt_result=_detach_result(pt_output, dq_pt, dk_pt, dv_pt),
    )


@pytest.mark.mate_dry_run
@supported_musa_compute_capability([31])
@pytest.mark.parametrize("seq_lens", [[(511, 511)], [(63, 63)]])
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 6)])
@pytest.mark.parametrize(
    "head_size",
    [(64, 64), (100, 80), (128, 128), (192, 128), (192, 192), (256, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [False])
@pytest.mark.parametrize("deterministic", [True, False])
def test_bshd_fast_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: tuple[int, int],
    dtype: torch.dtype,
    is_causal: bool,
    is_varlen: bool,
    deterministic: bool,
    pytestconfig: pytest.Config,
) -> None:
    torch.manual_seed(42)
    torch.musa.manual_seed(42)
    device = torch.device("musa")
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

    batch = len(seq_lens)
    assert len(set(query_lens)) == 1
    assert len(set(kv_lens)) == 1
    query = torch.randn(
        (batch, max_query_len, num_query_heads, head_size_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    key_cache = torch.randn(
        (batch, max_kv_len, num_kv_heads, head_size_qk),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    value_cache = torch.randn(
        (batch, max_kv_len, num_kv_heads, head_size_v),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    q_fa, k_fa, v_fa = map(clone_with_grad, (query, key_cache, value_cache))

    if is_dry_run_enabled():
        _dry_run_fmha_forward_backward(
            q_fa,
            k_fa,
            v_fa,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            is_causal=is_causal,
            deterministic=deterministic,
        )
        return

    out_fa = flash_attn_varlen_func(
        q=q_fa,
        k=k_fa,
        v=v_fa,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        causal=is_causal,
        deterministic=deterministic,
    )

    grad_out = torch.randn_like(out_fa)
    q_t, k_t, v_t = map(clone_with_grad, (query, key_cache, value_cache))
    q_pt, k_pt, v_pt = map(clone_with_grad, (query, key_cache, value_cache))

    ref_output, _ = ref_attn(
        query=q_t,
        key_cache=k_t,
        value_cache=v_t,
        cu_query_lens=None,
        cu_kv_lens=None,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=True,
    )
    pt_output, _ = ref_attn(
        query=q_pt,
        key_cache=k_pt,
        value_cache=v_pt,
        cu_query_lens=None,
        cu_kv_lens=None,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        is_causal=is_causal,
        is_varlen=is_varlen,
        scale=scale,
        upcast=False,
        reorder_ops=True,
    )
    grad_fa = clone_with_grad(grad_out)
    grad_t = clone_with_grad(grad_out)
    grad_pt = clone_with_grad(grad_out)

    out_fa.backward(gradient=grad_fa, retain_graph=True)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    ref_output.backward(gradient=grad_t, retain_graph=True)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    pt_output.backward(gradient=grad_pt, retain_graph=True)
    dq_pt, dk_pt, dv_pt = q_pt.grad, k_pt.grad, v_pt.grad

    assert_fa_error_within_pt(out_fa, ref_output, pt_output, "out")
    assert_fa_error_within_pt(dk_fa, dk_t, dk_pt, "dk")
    assert_fa_error_within_pt(dv_fa, dv_t, dv_pt, "dv")
    assert_fa_error_within_pt(dq_fa, dq_t, dq_pt, "dq")

    _stress_fmha_bwd_case(
        pytestconfig,
        query,
        key_cache,
        value_cache,
        grad_out,
        cu_query_lens=None,
        cu_kv_lens=None,
        max_query_len=None,
        max_kv_len=None,
        is_causal=is_causal,
        deterministic=deterministic,
        ref_result=_detach_result(ref_output, dq_t, dk_t, dv_t),
        pt_result=_detach_result(pt_output, dq_pt, dk_pt, dv_pt),
    )


@pytest.mark.parametrize(
    "dims_causal", [(128, False), (128, True), (192, False), (192, True)]
)
@pytest.mark.parametrize(
    "dtype,kv_heads,layout,lengths",
    [
        (dtype, kv_heads, layout, (129, 385))
        for dtype in (torch.float16, torch.bfloat16)
        for kv_heads in (2, 4)
        for layout in ("dense", "singleton", "packed")
    ]
    + [
        (dtype, kv_heads, layout, (2, 5))
        for dtype in (torch.float16, torch.bfloat16)
        for kv_heads in (2, 4)
        for layout in ("dense", "singleton")
    ]
    + [
        (dtype, kv_heads, layout, lengths)
        for lengths in ((129, 65), (65, 129))
        for dtype, kv_heads, layout in (
            (torch.float16, 2, "dense"),
            (torch.bfloat16, 4, "singleton"),
        )
    ],
)
@supported_musa_compute_capability([31])
def test_asm_persistent_backward_routing(
    monkeypatch, dtype, dims_causal, kv_heads, layout, lengths
):
    import mate.jit.mubin.flash_attention_bwd as api

    original = api.get_flash_attention_bwd_mubin_launch_plan
    selected = []

    def plan(*args, **kwargs):
        selected.append(kwargs.get("is_persistence", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "get_flash_attention_bwd_mubin_launch_plan", plan)
    torch.manual_seed(47)
    dim, causal = dims_causal
    sq, sk = lengths
    heads = 4
    batch = (
        1
        if layout == "singleton"
        else max(
            5,
            torch.musa.get_device_properties(0).multi_processor_count
            // (heads * ((sk + 127) // 128))
            + 1,
        )
    )
    q, k, v = [
        torch.randn(shape, device="musa", dtype=dtype).requires_grad_()
        for shape in (
            (batch, sq, heads, dim),
            (batch, sk, kv_heads, dim),
            (batch, sk, kv_heads, 128),
        )
    ]
    kwargs = {}
    if layout != "dense":
        q, k, v = [t.detach().flatten(0, 1).requires_grad_() for t in (q, k, v)]
        kwargs = dict(
            cu_seqlens_q=torch.arange(batch + 1, device="musa", dtype=torch.int32) * sq,
            cu_seqlens_k=torch.arange(batch + 1, device="musa", dtype=torch.int32) * sk,
            max_seqlen_q=sq + 17,
            max_seqlen_k=sk + 19,
        )
    out = flash_attn_varlen_func(
        q, k, v, causal=causal, backend="mubin", bwd_backend="mubin", **kwargs
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    assert selected == [layout != "packed"]
    ref = [t.detach().float().cpu().requires_grad_() for t in (q, k, v)]
    qr, kr, vr = [t.reshape(batch, -1, *t.shape[-2:]) for t in ref]
    ref_out = torch.stack(
        [
            _asm_reference(qr[b], kr[b], vr[b], dim**-0.5, causal)[0]
            for b in range(batch)
        ]
    )
    ref_out.backward(grad.float().cpu().reshape_as(ref_out))
    atol = 0.04 if dtype == torch.bfloat16 else 0.006
    for actual, expected in zip((q, k, v), ref):
        torch.testing.assert_close(
            actual.grad.float().cpu(), expected.grad, atol=atol, rtol=atol
        )


def _asm_reference(q, k, v, scale, causal):
    groups = q.shape[-2] // k.shape[-2]
    k = k.repeat_interleave(groups, dim=-2)
    v = v.repeat_interleave(groups, dim=-2)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if causal:
        mask = (
            torch.arange(q.shape[0])[:, None] + k.shape[0] - q.shape[0]
            < torch.arange(k.shape[0])[None, :]
        )
        scores = scores.masked_fill(mask[None], -float("inf"))
    # Avoid NaN softmax derivatives for lower-right causal rows with no keys.
    empty_rows = torch.isneginf(scores).all(-1, keepdim=True)
    probs = scores.masked_fill(empty_rows, 0).softmax(-1).masked_fill(empty_rows, 0)
    return torch.einsum("hqk,khd->qhd", probs, v), scores.amax((1, 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("head_counts", [(4, 2), (2, 2)])
@pytest.mark.parametrize("long_sequence", [False, True])
@pytest.mark.mate_dry_run
@supported_musa_compute_capability([31])
def test_asm_max_logits_and_backward(
    dtype, dims, causal, packed, head_counts, long_sequence
):
    if torch.musa.get_device_capability() != (3, 1):
        pytest.skip("MP31 ASM")
    torch.manual_seed(42)
    qlens, klens = ([37, 83], [19, 97]) if packed else ([65, 65], [79, 79])
    if long_sequence:
        qlens, klens = ([257, 389], [513, 193]) if packed else ([257, 257], [389, 389])
    dim, vdim = dims
    heads, kv_heads = head_counts
    shapes = (
        (sum(qlens), heads, dim),
        (sum(klens), kv_heads, dim),
        (sum(klens), kv_heads, vdim),
    )
    tensors = [torch.randn(shape, device="musa", dtype=dtype) for shape in shapes]
    if not packed:
        tensors = [t.view(2, -1, *t.shape[-2:]) for t in tensors]
    q, k, v = [t.requires_grad_() for t in tensors]
    scale = dim**-0.5
    kwargs = {}
    if packed:
        kwargs = dict(
            cu_seqlens_q=torch.tensor(
                [0, qlens[0], sum(qlens)], device="musa", dtype=torch.int32
            ),
            cu_seqlens_k=torch.tensor(
                [0, klens[0], sum(klens)], device="musa", dtype=torch.int32
            ),
            max_seqlen_q=max(qlens),
            max_seqlen_k=max(klens),
        )
    out, lse, maximum = flash_attn_varlen_func(
        q,
        k,
        v,
        causal=causal,
        softmax_scale=scale,
        backend="mubin",
        bwd_backend="mubin",
        return_max_logits=True,
        return_softmax_lse=True,
        **kwargs,
    )
    assert maximum.shape == (heads,)
    assert maximum.dtype == torch.float32
    assert not maximum.requires_grad
    grad = torch.randn_like(out)
    out.backward(grad)
    from mate.execution_context import raise_complete_if_dry_run

    raise_complete_if_dry_run()
    reference = [t.detach().float().cpu().requires_grad_() for t in (q, k, v)]
    ref_q, ref_k, ref_v = [t.reshape(-1, *t.shape[-2:]) for t in reference]
    ref_out, ref_max = [], []
    q0, k0 = 0, 0
    for nq, nk in zip(qlens, klens):
        o, row_maximum = _asm_reference(
            ref_q[q0 : q0 + nq], ref_k[k0 : k0 + nk], ref_v[k0 : k0 + nk], scale, causal
        )
        ref_out.append(o)
        ref_max.append(row_maximum)
        q0, k0 = q0 + nq, k0 + nk
    ref_out = torch.cat(ref_out).reshape(*q.shape[:-1], vdim)
    ref_max = torch.stack(ref_max).amax(0)
    torch.testing.assert_close(maximum.cpu(), ref_max.detach(), atol=2e-4, rtol=2e-4)
    atol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        out.float().cpu(), ref_out.detach(), atol=atol, rtol=atol
    )
    ref_out.backward(grad.float().cpu())
    for actual, expected in zip((q, k, v), reference):
        torch.testing.assert_close(
            actual.grad.float().cpu(), expected.grad, atol=atol, rtol=atol
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.mate_dry_run
@supported_musa_compute_capability([31])
def test_asm_backward_with_mutlass_varlen_lse(dtype, dims, causal, kv_heads):
    # Separate batch lengths and upper bounds distinguish compact [H,total_q]
    # addressing from the imported ASM's batch/head pitch based on max_seqlen.
    torch.manual_seed(73)
    qlens, klens = [37, 83, 65], [19, 97, 71]
    dim, vdim = dims
    heads = 4
    q = torch.randn(
        (sum(qlens), heads, dim), device="musa", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        (sum(klens), kv_heads, dim), device="musa", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        (sum(klens), kv_heads, vdim), device="musa", dtype=dtype, requires_grad=True
    )
    cu_q = torch.tensor([0, 37, 120, 185], device="musa", dtype=torch.int32)
    cu_k = torch.tensor([0, 19, 116, 187], device="musa", dtype=torch.int32)
    scale = dim**-0.5
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=113,
        max_seqlen_k=129,
        softmax_scale=scale,
        causal=causal,
        backend="mutlass",
        bwd_backend="mubin",
        return_softmax_lse=True,
    )
    assert lse.shape == (heads, sum(qlens))
    assert lse.dtype == torch.float32
    grad = torch.randn_like(out)
    out.backward(grad)
    from mate.execution_context import raise_complete_if_dry_run

    raise_complete_if_dry_run()
    reference = [t.detach().float().cpu().requires_grad_() for t in (q, k, v)]
    ref_q, ref_k, ref_v = reference
    ref_out, ref_lse = [], []
    q0, k0 = 0, 0
    for nq, nk in zip(qlens, klens):
        qb, kb, vb = ref_q[q0 : q0 + nq], ref_k[k0 : k0 + nk], ref_v[k0 : k0 + nk]
        batch_out, _ = _asm_reference(qb, kb, vb, scale, causal)
        ref_out.append(batch_out)
        scores = (
            torch.einsum(
                "qhd,khd->hqk", qb, kb.repeat_interleave(heads // kv_heads, dim=1)
            )
            * scale
        )
        if causal:
            mask = torch.arange(nq)[:, None] + nk - nq < torch.arange(nk)[None, :]
            scores = scores.masked_fill(mask[None], -float("inf"))
        ref_lse.append(scores.logsumexp(-1))
        q0, k0 = q0 + nq, k0 + nk
    ref_out = torch.cat(ref_out)
    ref_lse = torch.cat(ref_lse, dim=1).detach()
    finite_rows = torch.isfinite(ref_lse)
    atol = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        lse.cpu()[finite_rows], ref_lse[finite_rows], atol=atol, rtol=atol
    )
    torch.testing.assert_close(
        out.float().cpu(), ref_out.detach(), atol=atol, rtol=atol
    )
    ref_out.backward(grad.float().cpu())
    for actual, expected in zip((q, k, v), reference):
        torch.testing.assert_close(
            actual.grad.float().cpu(), expected.grad, atol=atol, rtol=atol
        )


@supported_musa_compute_capability([31])
def test_asm_auto_backend_zero_scale(monkeypatch):
    from mate.flash_attention import backward
    from mate import mha_interface

    launches = []
    original = backward._mubin_backward

    def asm_backward(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)

    def unexpected_mutlass(*args, **kwargs):
        pytest.fail("auto should select an available ASM max-logits variant")

    monkeypatch.setattr(backward, "_mubin_backward", asm_backward)
    monkeypatch.setattr(mha_interface, "_flash_attn_forward", unexpected_mutlass)
    q = torch.randn(
        (1, 9, 4, 64), device="musa", dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        (1, 3, 2, 64), device="musa", dtype=torch.float16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    out, maximum = flash_attn_varlen_func(
        q, k, v, causal=True, softmax_scale=0.0, return_max_logits=True
    )
    out.sum().backward()
    assert launches == [True]
    assert torch.count_nonzero(maximum) == 0
    assert torch.count_nonzero(q.grad) == 0
    assert torch.count_nonzero(k.grad) == 0
    ref_q, ref_k, ref_v = [
        t.detach()[0].float().cpu().requires_grad_() for t in (q, k, v)
    ]
    ref_out, _ = _asm_reference(ref_q, ref_k, ref_v, 0.0, True)
    ref_out.sum().backward()
    torch.testing.assert_close(
        out[0].float().cpu(), ref_out.detach(), atol=3e-3, rtol=3e-3
    )
    torch.testing.assert_close(
        v.grad[0].float().cpu(), ref_v.grad, atol=3e-3, rtol=3e-3
    )


@supported_musa_compute_capability([31])
def test_backward_dispatch_fallback_and_strict(monkeypatch):
    from mate.flash_attention import backward
    from mate.flash_attention.tilelang import flash_attention_varlen_bwd as tilelang

    q = torch.empty((1, 16, 2, 256), device="musa", dtype=torch.float16)
    lse = torch.empty((1, 2, 16), device="musa", dtype=torch.float32)
    args = (q, q, q, q, q, lse, None, None)
    sentinel = object()
    monkeypatch.setattr(
        tilelang, "flashattn_varlen_bwd_interface", lambda *args, **kwargs: sentinel
    )
    assert backward.flashattn_varlen_bwd_interface(*args) is sentinel
    with pytest.raises(ValueError, match="supports D"):
        backward.flashattn_varlen_bwd_interface(*args, backend="mubin")
    supported = q[..., :128].contiguous()
    supported_args = (supported,) * 5 + (lse, None, None)
    monkeypatch.setattr(backward, "_mubin_backward", lambda *args, **kwargs: sentinel)
    assert backward.flashattn_varlen_bwd_interface(*supported_args) is sentinel
    assert (
        backward.flashattn_varlen_bwd_interface(*supported_args, deterministic=True)
        is sentinel
    )
    with pytest.raises(ValueError, match="deterministic"):
        backward.flashattn_varlen_bwd_interface(
            *supported_args, backend="mubin", deterministic=True
        )
    from mate import artifacts

    def unavailable_target(*args, **kwargs):
        raise ValueError("no published target")

    monkeypatch.setattr(artifacts.ArtifactPath, "for_module", unavailable_target)
    assert backward.flashattn_varlen_bwd_interface(*supported_args) is sentinel
    with pytest.raises(ValueError, match="published and pinned"):
        backward.flashattn_varlen_bwd_interface(*supported_args, backend="mubin")


@pytest.mark.parametrize("reason", ["deterministic", "missing_artifacts"])
@supported_musa_compute_capability([31])
def test_auto_backward_tilelang_gradients(monkeypatch, reason):
    from mate import artifacts
    from mate.flash_attention import backward

    torch.manual_seed(123)
    q = torch.randn(
        (1, 65, 4, 64), device="musa", dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        (1, 79, 2, 64), device="musa", dtype=torch.float16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)

    def unexpected_asm(*args, **kwargs):
        pytest.fail("unsupported or unpublished ASM must fall back to TileLang")

    monkeypatch.setattr(backward, "_mubin_backward", unexpected_asm)
    if reason == "missing_artifacts":
        original_path = artifacts.ArtifactPath.for_module

        def get_path(family):
            if family == "flash_attention_bwd":
                raise ValueError("no published target")
            return original_path(family)

        monkeypatch.setattr(artifacts.ArtifactPath, "for_module", get_path)
    out = flash_attn_varlen_func(
        q, k, v, backend="mubin", deterministic=reason == "deterministic"
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref = [t.detach()[0].float().cpu().requires_grad_() for t in (q, k, v)]
    ref_out, _ = _asm_reference(*ref, 64**-0.5, False)
    ref_out.backward(grad[0].float().cpu())
    for actual, expected in zip((q, k, v), ref):
        torch.testing.assert_close(
            actual.grad[0].float().cpu(), expected.grad, atol=3e-3, rtol=3e-3
        )


@pytest.mark.parametrize(
    "feature",
    [
        None,
        "qv",
        "page_table",
        "learnable_sink",
        "attention_chunk",
        "cp_world_size",
        "q_descale",
        "k_descale",
        "v_descale",
    ],
)
@supported_musa_compute_capability([31])
def test_backward_feature_dispatch_contract(monkeypatch, feature):
    from mate import mha_interface
    from mate.flash_attention import backward

    selected = []

    def forward(q, **kwargs):
        lse = torch.zeros((q.shape[0], q.shape[2], q.shape[1]), device=q.device)
        return torch.zeros_like(q), lse

    def bwd(q, k, v, *args, **kwargs):
        selected.append(kwargs["backend"])
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    monkeypatch.setattr(mha_interface, "_flash_attn_forward", forward)
    monkeypatch.setattr(backward, "flashattn_varlen_bwd_interface", bwd)
    q = torch.randn(
        (1, 17, 2, 64), device="musa", dtype=torch.float16, requires_grad=True
    )
    options = {
        "qv": torch.zeros_like(q),
        "page_table": torch.zeros((1, 1), device=q.device, dtype=torch.int32),
        "learnable_sink": torch.zeros(2, device=q.device),
        "attention_chunk": 8,
        "cp_world_size": 2,
        "q_descale": torch.ones(1, device=q.device),
        "k_descale": torch.ones(1, device=q.device),
        "v_descale": torch.ones(1, device=q.device),
    }
    kwargs = {} if feature is None else {feature: options[feature]}
    out = flash_attn_varlen_func(q, q, q, backend="mutlass", **kwargs)
    out.sum().backward()
    assert selected == ["auto" if feature is None else "tilelang"]
    out = flash_attn_varlen_func(
        q, q, q, backend="mutlass", bwd_backend="mubin", **kwargs
    )
    if feature is None:
        out.sum().backward()
        assert selected == ["auto", "mubin"]
    else:
        with pytest.raises(ValueError, match="MUBIN backward does not support"):
            out.sum().backward()
        assert selected == ["tilelang"]
    # Inference does not enter backward or enforce its feature restrictions.
    plain = q.detach()
    flash_attn_varlen_func(
        plain, plain, plain, backend="mutlass", bwd_backend="mubin", **kwargs
    )


@pytest.mark.parametrize("dims", [(8, 8), (120, 120), (192, 128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.mate_dry_run
@supported_musa_compute_capability([31])
def test_asm_empty_packed_sequences(dims, causal):
    # One batch has no queries; another has no keys. Include a normal batch.
    dim, vdim = dims
    q = torch.randn(
        (72, 4, dim), device="musa", dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        (110, 2, dim), device="musa", dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        (110, 2, vdim), device="musa", dtype=torch.float16, requires_grad=True
    )
    cu_q = torch.tensor([0, 0, 7, 72], device="musa", dtype=torch.int32)
    cu_k = torch.tensor([0, 13, 13, 110], device="musa", dtype=torch.int32)
    out, maximum = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=65,
        max_seqlen_k=97,
        causal=causal,
        backend="mubin",
        bwd_backend="mubin",
        return_max_logits=True,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    from mate.execution_context import raise_complete_if_dry_run

    raise_complete_if_dry_run()
    assert torch.count_nonzero(out[:7]) == 0
    assert torch.count_nonzero(q.grad[:7]) == 0
    assert torch.count_nonzero(k.grad[:13]) == 0
    assert torch.count_nonzero(v.grad[:13]) == 0
    ref_q, ref_k, ref_v = [
        t.detach().float().cpu().requires_grad_() for t in (q[7:], k[13:], v[13:])
    ]
    ref_out, ref_max = _asm_reference(ref_q, ref_k, ref_v, dim**-0.5, causal)
    ref_out.backward(grad[7:].float().cpu())
    torch.testing.assert_close(maximum.cpu(), ref_max.detach(), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        out[7:].float().cpu(), ref_out.detach(), atol=3e-3, rtol=3e-3
    )
    for actual, expected in zip(
        (q.grad[7:], k.grad[13:], v.grad[13:]), (ref_q.grad, ref_k.grad, ref_v.grad)
    ):
        torch.testing.assert_close(actual.float().cpu(), expected, atol=3e-3, rtol=3e-3)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "packed,dtype", [(False, torch.float16), (True, torch.bfloat16)]
)
def test_fmha_backward_interface(packed, dtype):
    if is_dry_run_enabled():
        pytest.skip("interface correctness requires real kernel execution")
    torch.manual_seed(51)
    q = torch.randn((1, 11, 2, 128), device="musa", dtype=dtype)
    k = torch.randn((1, 17, 1, 128), device="musa", dtype=dtype)
    v = torch.randn_like(k)
    if packed:
        q, k, v = (t.squeeze(0) for t in (q, k, v))
    cu_q = torch.tensor([0, 11], dtype=torch.int32) if packed else None
    cu_k = torch.tensor([0, 17], dtype=torch.int32) if packed else None
    reference = [t.cpu().float().requires_grad_(True) for t in (q, k, v)]
    out_ref, lse_ref = ref_attn(
        *reference,
        cu_q,
        cu_k,
        11,
        17,
        is_causal=True,
        is_varlen=packed,
        scale=128**-0.5,
    )
    out = out_ref.detach().to(device=q.device, dtype=dtype)
    lse = lse_ref.detach().to(q.device).contiguous()
    dout = torch.randn_like(out)
    buffers = [copy_to_strided_last_dim(torch.empty_like(t)) for t in (q, k, v)]
    if packed:
        buffers[0] = buffers[2] = None
    dq, dk, dv, softmax_d = fmha_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        *buffers,
        cu_seqlens_q=cu_q.to(q.device) if packed else None,
        cu_seqlens_k=cu_k.to(q.device) if packed else None,
        max_seqlen_q=11 if packed else None,
        max_seqlen_k=17 if packed else None,
        is_causal=True,
        sm_margin=7,
        backend="tilelang" if packed else "auto",
    )
    out_ref.backward(dout.cpu().float())
    for actual, expected, buffer in zip((dq, dk, dv), reference, buffers):
        if buffer is not None:
            assert actual is buffer
        torch.testing.assert_close(
            actual.cpu().float(), expected.grad, atol=0.035, rtol=0.035
        )
    expected_delta = (out.float() * dout.float()).sum(-1).transpose(-2, -1).contiguous()
    torch.testing.assert_close(softmax_d, expected_delta, atol=1e-5, rtol=1e-5)
