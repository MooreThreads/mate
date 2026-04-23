import pytest
import torch
import math
from mate.gdn_kernels.tilelang.gdn_prefill import (
    _build_chunk_metadata,
    fused_prepare_compute_w_u_tl,
    _h_recurrence_tl,
    _output_o_tl,
    gdn_prefill,
)
from mate.testing.utils import gen_qkv

_LOG2E = 1.4426950408889634
KERNEL_HEAD_CASES = [
    pytest.param(8, 8, 8, id="gqa-hq8-hk8-hv8"),
    pytest.param(16, 8, 8, id="gqa-hq16-hk8-hv8"),
    pytest.param(16, 16, 16, id="gqa-hq16-hk16-hv16"),
    pytest.param(8, 8, 16, id="gva-hq8-hk8-hv16"),
]


def _to_cpu_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32)


def _assert_close_like_kernel(
    actual: torch.Tensor, expected: torch.Tensor, is_output: bool
):
    if is_output:
        if actual.dtype == torch.bfloat16:
            rtol = 1e-2
            atol = 1e-2
        else:
            rtol = 2e-3
            atol = 1e-3
    else:
        if actual.dtype == torch.bfloat16:
            rtol = 5e-3
            atol = 1e-3
        else:
            rtol = 1e-3
            atol = 1e-4
    actual_f = _to_cpu_fp32(actual)
    expected_f = _to_cpu_fp32(expected)
    torch.testing.assert_close(actual_f, expected_f, rtol=rtol, atol=atol)


def _inverse(p: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Mirror tilelang recurrence in fused_prepare_compute_w_u:
      S_0 = I, P_0 = P
      S_{r+1} = S_r + P_r @ S_r
      P_{r+1} = P_r @ P_r
    After ceil(log2(chunk_size)) rounds this equals I + P + ... + P^(chunk_size-1),
    which is exact for strictly lower-triangular P used in this test.
    """
    length = p.shape[0]
    s = torch.eye(length, device=p.device, dtype=torch.float32)
    p_pow = p.to(torch.float32)
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0
    for _ in range(num_rounds):
        s = s + p_pow @ s
        p_pow = p_pow @ p_pow
    return s


def _compute_w_u_cug_reference(k, v, alpha, beta, cu_seqlens, chunk_size):
    total_tokens, head_k, _ = k.shape
    head_sab = v.shape[1]
    assert head_sab % head_k == 0, "head_sab must be divisible by head_k"
    sab_to_k_group_size = head_sab // head_k

    cu_g = torch.zeros(total_tokens, head_sab, dtype=torch.float32, device=k.device)
    w = torch.zeros(total_tokens, head_sab, k.shape[-1], dtype=k.dtype, device=k.device)
    u = torch.zeros_like(v)
    batch = cu_seqlens.numel() - 1
    for bid in range(batch):
        seq_start = int(cu_seqlens[bid].item())
        seq_end = int(cu_seqlens[bid + 1].item())
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            if chunk_end <= chunk_start:
                continue

            k_chunk = k[chunk_start:chunk_end].to(torch.float32)  # [L, H, DK]
            v_chunk = v[chunk_start:chunk_end].to(torch.float32)  # [L, H, DV]
            alpha_chunk = alpha[chunk_start:chunk_end].to(torch.float32)  # [L, H]
            beta_chunk = beta[chunk_start:chunk_end].to(torch.float32)  # [L, H]

            g_chunk = torch.cumsum(torch.log(alpha_chunk), dim=0)  # [L, H]
            cu_g[chunk_start:chunk_end] = g_chunk

            for hid in range(head_sab):
                k_hid = hid // sab_to_k_group_size
                k_h = k_chunk[:, k_hid, :]  # [L, DK]
                v_h = v_chunk[:, hid, :]  # [L, DV]
                beta_h = beta_chunk[:, hid]  # [L]
                g_h = g_chunk[:, hid]  # [L]

                gram = k_h @ k_h.transpose(0, 1)  # [L, L]
                diff = g_h[:, None] - g_h[None, :]
                p = (
                    -gram
                    * beta_h[:, None]
                    * torch.tril(torch.exp2(diff * _LOG2E), diagonal=-1)
                )

                k_beta = k_h * beta_h[:, None]
                v_beta = v_h * beta_h[:, None]
                s = _inverse(p, chunk_size)
                w_chunk_h = s @ k_beta
                u_chunk_h = s @ v_beta

                w[chunk_start:chunk_end, hid, :] = w_chunk_h.to(w.dtype)
                u[chunk_start:chunk_end, hid, :] = u_chunk_h.to(u.dtype)
    return w, u, cu_g


def _compute_w_u_cug_reference_from_chunks(
    k, v, alpha, beta, chunk_token_starts, chunk_token_lens, chunk_size
):
    total_tokens, head_k, _ = k.shape
    head_sab = v.shape[1]
    assert head_sab % head_k == 0, "head_sab must be divisible by head_k"
    sab_to_k_group_size = head_sab // head_k

    cu_g = torch.zeros(total_tokens, head_sab, dtype=torch.float32, device=k.device)
    w = torch.zeros(total_tokens, head_sab, k.shape[-1], dtype=k.dtype, device=k.device)
    u = torch.zeros_like(v)
    _ = chunk_size

    total_chunks = int(chunk_token_starts.numel())
    for tid in range(total_chunks):
        start = int(chunk_token_starts[tid].item())
        length = int(chunk_token_lens[tid].item())
        end = start + length
        if length <= 0:
            continue

        k_chunk = k[start:end].to(torch.float32)  # [L, H, DK]
        v_chunk = v[start:end].to(torch.float32)  # [L, H, DV]
        alpha_chunk = alpha[start:end].to(torch.float32)  # [L, H]
        beta_chunk = beta[start:end].to(torch.float32)  # [L, H]

        g_chunk = torch.cumsum(torch.log(alpha_chunk), dim=0)  # [L, H]
        cu_g[start:end] = g_chunk

        for hid in range(head_sab):
            k_hid = hid // sab_to_k_group_size
            k_h = k_chunk[:, k_hid, :]  # [L, DK]
            v_h = v_chunk[:, hid, :]  # [L, DV]
            beta_h = beta_chunk[:, hid]  # [L]
            g_h = g_chunk[:, hid]  # [L]

            gram = k_h @ k_h.transpose(0, 1)  # [L, L]
            diff = g_h[:, None] - g_h[None, :]
            p = (
                -gram
                * beta_h[:, None]
                * torch.tril(torch.exp2(diff * _LOG2E), diagonal=-1)
            )

            k_beta = k_h * beta_h[:, None]
            v_beta = v_h * beta_h[:, None]
            s = _inverse(p, chunk_size)
            w_chunk_h = s @ k_beta
            u_chunk_h = s @ v_beta

            w[start:end, hid, :] = w_chunk_h.to(w.dtype)
            u[start:end, hid, :] = u_chunk_h.to(u.dtype)
    return w, u, cu_g


def _recurrent_reference(
    k, g, w, u, cu_seqlens, chunk_offsets, initial_state, output_state, chunk_size
):
    total_tokens, head_k, dim_k = k.shape
    head_sab = g.shape[1]
    assert head_sab % head_k == 0, "head_sab must be divisible by head_k"
    sab_to_k_group_size = head_sab // head_k
    dim_v = u.shape[-1]
    total_chunks = int(chunk_offsets[-1].item())
    batch = cu_seqlens.numel() - 1

    v_new = torch.empty(total_tokens, head_sab, dim_v, dtype=u.dtype, device=u.device)
    S_buf = torch.empty(
        total_chunks,
        head_sab,
        dim_v,
        dim_k,
        dtype=output_state.dtype,
        device=output_state.device,
    )
    for bid in range(batch):
        seq_start = int(cu_seqlens[bid].item())
        seq_end = int(cu_seqlens[bid + 1].item())
        cid = int(chunk_offsets[bid].item())
        num_chunks = (seq_end - seq_start + chunk_size - 1) // chunk_size
        for hid in range(head_sab):
            k_hid = hid // sab_to_k_group_size
            h_cur = initial_state[bid, hid].clone()
            for t in range(num_chunks):
                chunk_token_start = seq_start + t * chunk_size
                chunk_token_end = min(chunk_token_start + chunk_size, seq_end)
                if chunk_token_end <= chunk_token_start:
                    continue

                k_c = k[chunk_token_start:chunk_token_end, k_hid]
                w_c = w[chunk_token_start:chunk_token_end, hid]
                u_c = u[chunk_token_start:chunk_token_end, hid]
                g_c = g[chunk_token_start:chunk_token_end, hid]

                ws = w_c @ h_cur.to(w_c.dtype).T
                v_new_c = (u_c - ws * torch.exp2(g_c * _LOG2E).unsqueeze(-1)).to(
                    k.dtype
                )
                v_new[chunk_token_start:chunk_token_end, hid] = v_new_c

                g_last = g_c[-1]
                k_scaled = (k_c * torch.exp2((g_last - g_c) * _LOG2E).unsqueeze(-1)).to(
                    k.dtype
                )
                h_cur = (
                    h_cur * torch.exp2(g_last * _LOG2E)
                    + v_new_c.transpose(0, 1) @ k_scaled
                )
                S_buf[cid + t, hid] = h_cur
            output_state[bid, hid] = h_cur

    return S_buf, v_new


def _output_reference(
    q, k, g, cu_seqlens, chunk_offsets, s_buf, v_new, chunk_size, scale
):
    total_tokens, head_q, _ = q.shape
    head_k = k.shape[1]
    head_sab = g.shape[1]
    head_o = max(head_q, head_sab)
    assert head_o % head_q == 0, "head_o must be divisible by head_q"
    assert head_o % head_sab == 0, "head_o must be divisible by head_sab"
    assert head_sab % head_k == 0, "head_sab must be divisible by head_k"
    q_group_size = head_o // head_q
    sab_group_size = head_o // head_sab
    sab_to_k_group_size = head_sab // head_k

    dim_v = v_new.shape[-1]
    batch = cu_seqlens.numel() - 1
    o = torch.empty(total_tokens, head_o, dim_v, dtype=q.dtype, device=q.device)

    for bid in range(batch):
        seq_start = int(cu_seqlens[bid].item())
        seq_end = int(cu_seqlens[bid + 1].item())
        cid = int(chunk_offsets[bid].item())
        num_chunks = (seq_end - seq_start + chunk_size - 1) // chunk_size
        for t in range(num_chunks):
            chunk_token_start = seq_start + t * chunk_size
            chunk_token_end = min(chunk_token_start + chunk_size, seq_end)
            if chunk_token_end <= chunk_token_start:
                continue
            for out_hid in range(head_o):
                q_hid = out_hid // q_group_size
                sab_hid = out_hid // sab_group_size
                k_hid = sab_hid // sab_to_k_group_size
                q_c = q[chunk_token_start:chunk_token_end, q_hid]
                k_c = k[chunk_token_start:chunk_token_end, k_hid]
                g_c = g[chunk_token_start:chunk_token_end, sab_hid]
                h_c = s_buf[cid + t, sab_hid]
                v_new_c = v_new[chunk_token_start:chunk_token_end, sab_hid]

                o_chunk = (q_c @ h_c.to(q_c.dtype).T) * torch.exp2(
                    g_c * _LOG2E
                ).unsqueeze(-1)
                attn = q_c @ k_c.transpose(0, 1)
                diff = g_c.unsqueeze(1) - g_c.unsqueeze(0)
                attn = (attn * torch.tril(torch.exp2(diff * _LOG2E))).to(q.dtype)
                o_chunk = scale * (o_chunk + attn @ v_new_c)
                o[chunk_token_start:chunk_token_end, out_hid] = o_chunk.to(q.dtype)
    return o


def _output_reference_from_chunks(
    q,
    k,
    g,
    chunk_token_starts,
    chunk_token_lens,
    s_buf,
    v_new,
    scale,
):
    total_tokens, head_q, _ = q.shape
    head_k = k.shape[1]
    head_sab = g.shape[1]
    head_o = max(head_q, head_sab)
    assert head_o % head_q == 0, "head_o must be divisible by head_q"
    assert head_o % head_sab == 0, "head_o must be divisible by head_sab"
    assert head_sab % head_k == 0, "head_sab must be divisible by head_k"
    q_group_size = head_o // head_q
    sab_group_size = head_o // head_sab
    sab_to_k_group_size = head_sab // head_k

    dim_v = v_new.shape[-1]
    o = torch.empty(total_tokens, head_o, dim_v, dtype=q.dtype, device=q.device)

    total_chunks = int(chunk_token_starts.numel())
    for tid in range(total_chunks):
        chunk_token_start = int(chunk_token_starts[tid].item())
        actual_len = int(chunk_token_lens[tid].item())
        chunk_token_end = chunk_token_start + actual_len
        if actual_len <= 0:
            continue
        for out_hid in range(head_o):
            q_hid = out_hid // q_group_size
            sab_hid = out_hid // sab_group_size
            k_hid = sab_hid // sab_to_k_group_size
            q_c = q[chunk_token_start:chunk_token_end, q_hid]
            k_c = k[chunk_token_start:chunk_token_end, k_hid]
            g_c = g[chunk_token_start:chunk_token_end, sab_hid]
            h_c = s_buf[tid, sab_hid]
            v_new_c = v_new[chunk_token_start:chunk_token_end, sab_hid]

            o_chunk = (
                (q_c @ h_c.to(q_c.dtype).T) * torch.exp2(g_c * _LOG2E).unsqueeze(-1)
            ).to(q.dtype)
            attn = q_c @ k_c.transpose(0, 1)
            diff = g_c.unsqueeze(1) - g_c.unsqueeze(0)
            attn = (attn * torch.tril(torch.exp2(diff * _LOG2E))).to(q.dtype)
            o_chunk = scale * (o_chunk + attn @ v_new_c)
            o[chunk_token_start:chunk_token_end, out_hid] = o_chunk.to(q.dtype)
    return o


def torch_reference_chunk_gated_delta_rule(
    q,
    k,
    v,
    alpha,
    beta,
    cu_seqlens,
    chunk_size,
    initial_state,
    output_state,
    scale=None,
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks_per_batch = (seq_lens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.cat(
        [
            torch.tensor([0], device=cu_seqlens.device, dtype=torch.int32),
            chunks_per_batch.cumsum(0).to(torch.int32),
        ]
    )

    w, u, cu_g = _compute_w_u_cug_reference(k, v, alpha, beta, cu_seqlens, chunk_size)
    S_buf, v_new = _recurrent_reference(
        k,
        cu_g,
        w,
        u,
        cu_seqlens,
        chunk_offsets,
        initial_state,
        output_state,
        chunk_size,
    )
    o = _output_reference(
        q,
        k,
        cu_g,
        cu_seqlens,
        chunk_offsets,
        S_buf,
        v_new,
        chunk_size,
        scale,
    )
    return o, output_state


def _make_test_case(
    total_tokens,
    batch,
    head_q,
    head_k,
    dim_k,
    dim_v,
    head_v=None,
    chunk_size=64,
    dtype=torch.bfloat16,
    seed=42,
):
    device = "musa"
    if head_v is None:
        head_v = head_k

    is_gqa = head_v == head_k and head_q % head_k == 0
    is_gva = head_q == head_k and head_v % head_q == 0
    assert is_gqa or is_gva, (
        "Only GQA/GVA head layouts are supported in test case builder: "
        "GQA(head_v==head_k and head_q%head_k==0) or "
        "GVA(head_q==head_k and head_v%head_q==0)."
    )

    head_sab = max(head_k, head_v)
    head_o = max(head_q, head_v)
    q_group_size = head_o // head_q
    sab_group_size = head_o // head_sab
    group_size = sab_group_size
    case_dtype = "bfloat16"
    if dtype == torch.float16:
        case_dtype = "float16"
    elif dtype == torch.float32:
        case_dtype = "float32"
    else:
        assert dtype == torch.bfloat16, (
            "Only float16, bfloat16, and float32 are supported for dtype."
        )

    torch.manual_seed(seed)
    q, k, v = gen_qkv(
        total_tokens=total_tokens,
        num_q_heads=head_q,
        num_k_heads=head_k,
        num_v_heads=head_v,
        dim_qk=dim_k,
        dim_v=dim_v,
        device=device,
        dtype=dtype,
    )

    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    A_log = torch.rand(head_sab, device=device, dtype=torch.float32)
    dt_bias = torch.rand(head_sab, device=device, dtype=torch.float32)
    a = torch.rand(total_tokens, head_sab, device=device, dtype=dtype)
    g = -A_log.float().exp() * torch.where(
        a + dt_bias <= 20, torch.nn.functional.softplus(a + dt_bias), (a + dt_bias)
    )
    alpha = torch.exp(g)
    beta = torch.sigmoid(
        torch.randn(total_tokens, head_sab, device=device, dtype=torch.float32)
    )

    step = total_tokens // batch
    bounds = [0]
    for i in range(1, batch):
        bounds.append(i * step)
    bounds.append(total_tokens)
    cu_seqlens = torch.tensor(bounds, device=device, dtype=torch.int32)

    initial_state = torch.zeros(
        batch, head_sab, dim_v, dim_k, device=device, dtype=torch.float32
    )
    output_state = torch.randn(
        batch, head_sab, dim_v, dim_k, device=device, dtype=torch.float32
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "alpha": alpha,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "output_state": output_state,
        "chunk_size": chunk_size,
        "group_size": group_size,
        "q_group_size": q_group_size,
        "sab_group_size": sab_group_size,
        "head_o": head_o,
        "head_sab": head_sab,
        "head_q": head_q,
        "head_k": head_k,
        "head_v": head_v,
        "head_kv": head_k,
        "dim_k": dim_k,
        "dim_v": dim_v,
        "total_tokens": total_tokens,
        "device": device,
        "dtype": case_dtype,
    }


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("total_tokens", [128, 2048])
@pytest.mark.parametrize(("head_q", "head_k", "head_v"), KERNEL_HEAD_CASES)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("dim_v", [64, 128])
def test_kernel1_mock_fused_prepare_compute_w_u_matches_reference(
    total_tokens, batch, head_q, head_k, head_v, dim_k, dim_v
):
    case = _make_test_case(
        total_tokens, batch, head_q, head_k, dim_k, dim_v, head_v=head_v
    )

    chunk_offsets, chunk_token_starts, chunk_token_lens, total_chunks = (
        _build_chunk_metadata(case["cu_seqlens"], case["chunk_size"])
    )
    _ = chunk_offsets  # only needed by kernel2 path
    k1 = fused_prepare_compute_w_u_tl(
        total_chunks,
        case["total_tokens"],
        case["head_sab"],
        case["chunk_size"],
        case["dim_k"],
        case["dim_v"],
        head_k=case["head_k"],
        dtype=case["dtype"],
    )(num_stages=1, threads=128, block_DK=case["dim_k"], block_DV=case["dim_v"])

    w, u, cu_g = k1(
        case["k"],
        case["v"],
        case["alpha"],
        case["beta"],
        chunk_token_starts,
        chunk_token_lens,
    )
    w_ref, u_ref, cu_g_ref = _compute_w_u_cug_reference_from_chunks(
        case["k"].cpu(),
        case["v"].cpu(),
        case["alpha"].cpu(),
        case["beta"].cpu(),
        chunk_token_starts.cpu(),
        chunk_token_lens.cpu(),
        case["chunk_size"],
    )
    _assert_close_like_kernel(w, w_ref, is_output=True)
    _assert_close_like_kernel(u, u_ref, is_output=True)
    _assert_close_like_kernel(cu_g, cu_g_ref, is_output=False)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("total_tokens", [128, 2048])
@pytest.mark.parametrize(("head_q", "head_k", "head_v"), KERNEL_HEAD_CASES)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("dim_v", [64, 128])
def test_kernel2_mock_h_recurrence_matches_reference(
    batch, total_tokens, head_q, head_k, head_v, dim_k, dim_v
):
    case = _make_test_case(
        total_tokens, batch, head_q, head_k, dim_k, dim_v, head_v=head_v
    )

    chunk_offsets, chunk_token_starts, chunk_token_lens, total_chunks = (
        _build_chunk_metadata(case["cu_seqlens"], case["chunk_size"])
    )
    _ = chunk_token_starts, chunk_token_lens

    k_ref = case["k"].cpu()
    v_ref = case["v"].cpu()
    alpha_ref = case["alpha"].cpu()
    beta_ref = case["beta"].cpu()
    cu_ref = case["cu_seqlens"].cpu()

    w_ref_in, u_ref_in, cu_g_ref_in = _compute_w_u_cug_reference(
        k_ref, v_ref, alpha_ref, beta_ref, cu_ref, case["chunk_size"]
    )
    w = w_ref_in.to(device=case["device"], dtype=case["k"].dtype)
    u = u_ref_in.to(device=case["device"], dtype=case["v"].dtype)
    cu_g = cu_g_ref_in.to(device=case["device"])
    output_state = case["output_state"]
    k2 = _h_recurrence_tl(
        total_chunks,
        case["total_tokens"],
        case["cu_seqlens"].numel() - 1,
        case["head_sab"],
        case["chunk_size"],
        case["dim_k"],
        case["dim_v"],
        head_k=case["head_k"],
        dtype=case["dtype"],
    )(num_stages=1, threads=128, block_DV=case["dim_v"])

    S_buf, v_new = k2(
        case["k"],
        cu_g,
        w,
        u,
        case["cu_seqlens"],
        chunk_offsets,
        case["initial_state"],
        output_state,
    )
    chunk_offsets_ref = chunk_offsets.cpu()
    init_ref = case["initial_state"].cpu()
    output_state_ref = case["output_state"].cpu()
    S_buf_ref, v_new_ref = _recurrent_reference(
        k_ref,
        cu_g_ref_in,
        w_ref_in,
        u_ref_in,
        cu_ref,
        chunk_offsets_ref,
        init_ref,
        output_state_ref,
        case["chunk_size"],
    )
    # _assert_close_like_kernel(v_new, v_new_ref, is_output=False)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("total_tokens", [128, 2048])
@pytest.mark.parametrize(("head_q", "head_k", "head_v"), KERNEL_HEAD_CASES)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("dim_v", [64, 128])
def test_kernel3_mock_output_o_matches_reference(
    batch, total_tokens, head_q, head_k, head_v, dim_k, dim_v
):
    case = _make_test_case(
        total_tokens, batch, head_q, head_k, dim_k, dim_v, head_v=head_v
    )

    chunk_offsets, chunk_token_starts, chunk_token_lens, total_chunks = (
        _build_chunk_metadata(case["cu_seqlens"], case["chunk_size"])
    )
    k_ref = case["k"].cpu()
    v_ref = case["v"].cpu()
    alpha_ref = case["alpha"].cpu()
    beta_ref = case["beta"].cpu()
    cu_ref = case["cu_seqlens"].cpu()
    chunk_offsets_ref = chunk_offsets.cpu()
    init_ref = case["initial_state"].cpu()
    output_state_ref = case["output_state"].cpu()

    w_ref, u_ref, cu_g_ref = _compute_w_u_cug_reference(
        k_ref, v_ref, alpha_ref, beta_ref, cu_ref, case["chunk_size"]
    )
    S_buf_ref, v_new_ref = _recurrent_reference(
        k_ref,
        cu_g_ref,
        w_ref,
        u_ref,
        cu_ref,
        chunk_offsets_ref,
        init_ref,
        output_state_ref,
        case["chunk_size"],
    )
    cu_g = cu_g_ref.to(device=case["device"])
    S_buf = S_buf_ref.to(device=case["device"])
    v_new = v_new_ref.to(device=case["device"], dtype=case["v"].dtype)

    scale = case["dim_k"] ** -0.5
    k3 = _output_o_tl(
        total_chunks,
        case["total_tokens"],
        case["head_q"],
        case["head_sab"],
        case["group_size"],
        case["chunk_size"],
        scale,
        case["dim_k"],
        case["dim_v"],
        head_o=case["head_o"],
        head_sab=case["head_sab"],
        q_group_size=case["q_group_size"],
        sab_group_size=case["sab_group_size"],
        head_k=case["head_k"],
        dtype=case["dtype"],
    )(num_stages=1, threads=128, block_DV=case["dim_v"])

    o = torch.empty(
        case["total_tokens"],
        case["head_o"],
        case["dim_v"],
        dtype=case["q"].dtype,
        device=case["device"],
    )
    k3(
        case["q"],
        case["k"],
        cu_g,
        chunk_token_starts,
        chunk_token_lens,
        S_buf,
        v_new,
        o,
    )
    q_ref = case["q"].cpu()
    o_ref = _output_reference_from_chunks(
        q_ref,
        k_ref,
        cu_g_ref,
        chunk_token_starts.cpu(),
        chunk_token_lens.cpu(),
        S_buf_ref,
        v_new_ref,
        scale,
    )
    _assert_close_like_kernel(o, o_ref, is_output=True)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("total_tokens", [128, 2048])
@pytest.mark.parametrize(("head_q", "head_k", "head_v"), KERNEL_HEAD_CASES)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("dim_v", [64, 128])
def test_chunk_gated_delta_rule_mock_matches_reference(
    batch, total_tokens, head_q, head_k, head_v, dim_k, dim_v
):
    case = _make_test_case(
        total_tokens, batch, head_q, head_k, dim_k, dim_v, head_v=head_v
    )

    o = torch.empty(
        case["total_tokens"],
        case["head_o"],
        case["dim_v"],
        dtype=case["q"].dtype,
        device=case["device"],
    )
    gdn_prefill(
        output=o,
        output_state=case["output_state"],
        q=case["q"],
        k=case["k"],
        v=case["v"],
        alpha=case["alpha"],
        beta=case["beta"],
        initial_state=case["initial_state"],
        cu_seqlens=case["cu_seqlens"],
        chunk_size=case["chunk_size"],
    )

    o_ref, final_state_ref = torch_reference_chunk_gated_delta_rule(
        q=case["q"].cpu(),
        k=case["k"].cpu(),
        v=case["v"].cpu(),
        alpha=case["alpha"].cpu(),
        beta=case["beta"].cpu(),
        cu_seqlens=case["cu_seqlens"].cpu(),
        chunk_size=case["chunk_size"],
        initial_state=case["initial_state"].cpu(),
        output_state=case["output_state"].cpu(),
    )

    _assert_close_like_kernel(o, o_ref, is_output=True)
