from __future__ import annotations

import os
import random
from functools import wraps

import pytest
import torch
import torch.nn.functional as F

from mate.testing import repeat_check

CHUNK_SIZE = 32
HEAD_SIZE = 128


@pytest.mark.parametrize(
    "lengths", ([0, 64, 65], [64, 0, 65], [64, 65, 0], [8192, 0, 0])
)
@pytest.mark.parametrize("state_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("state_mode", ("dense", "pool", "none"))
@pytest.mark.parametrize("output_final_state", (False, True))
def test_empty_sequences_preserve_state(
    lengths, state_dtype, state_mode, output_final_state
):
    """Empty work skips all recurrence roles and preserves state without rounding."""
    from mate.kda import chunk_kda

    device = _get_runtime_device()
    _manual_seed(73, device)
    live = [i for i, length in enumerate(lengths) if length]
    empty = [i for i, length in enumerate(lengths) if not length]
    cu = torch.tensor(exclusive_cumsum(lengths), dtype=torch.int64, device=device)
    compact_cu = torch.tensor(
        exclusive_cumsum([lengths[i] for i in live]), dtype=torch.int64, device=device
    )
    q, k, v, g = [
        torch.randn(1, sum(lengths), 12, 128, dtype=torch.bfloat16, device=device)
        for _ in range(4)
    ]
    beta = torch.randn(1, sum(lengths), 12, dtype=torch.bfloat16, device=device)
    args = dict(
        A_log=torch.randn(12, device=device),
        dt_bias=torch.randn(12, 128, device=device),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )
    # Empty sequences share a placeholder pool slot; live sequences never alias.
    slots = [2 * live.index(i) + 1 if i in live else 0 for i in range(len(lengths))]
    ids = torch.tensor(slots, dtype=torch.int32, device=device)
    before = (torch.randn(7, 12, 128, 128, device=device) * 0.1).to(state_dtype)
    state = before.clone() if state_mode == "pool" else before[ids].contiguous()
    if state_mode == "none":
        state = None
    final = None
    if state_mode == "none" and output_final_state:
        final = torch.empty(
            len(lengths), 12, 128, 128, dtype=state_dtype, device=device
        )
    output_ref, state_ref = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None if state is None else before[ids[live]].contiguous(),
        final_state=None if final is None else torch.empty_like(final[: len(live)]),
        output_final_state=True,
        cu_seqlens=compact_cu,
        **args,
    )
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        result = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            initial_state_indices=ids if state_mode == "pool" else None,
            final_state=final,
            output_final_state=output_final_state,
            cu_seqlens=cu,
            **args,
        )
    stream.synchronize()
    output, final = result if output_final_state else (result, None)
    torch.testing.assert_close(output, output_ref, atol=0.03, rtol=0.03)
    if state_mode == "pool":
        final = state
        untouched = [i for i in range(7) if i not in slots]
        assert torch.equal(final[untouched], before[untouched])
    if final is not None:
        actual = final[ids] if state_mode == "pool" else final
        torch.testing.assert_close(actual[live], state_ref, atol=0.03, rtol=0.03)
        if state_mode == "none":
            assert torch.count_nonzero(actual[empty]).item() == 0
        else:
            assert torch.equal(actual[empty], before[ids[empty]])


def exclusive_cumsum(seq_lens: list[int]) -> list[int]:
    out = [0]
    for seq_len in seq_lens:
        out.append(out[-1] + seq_len)
    return out


def _get_runtime_device() -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    pytest.skip("MUSA is not available")


def _manual_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if device.type == "musa":
        torch.musa.manual_seed(seed)


def _resolve_seed(seed_offset: int) -> int:
    return int(os.environ.get("SEED", "0")) + seed_offset


def _synchronize(device: torch.device) -> None:
    if device.type == "musa":
        torch.musa.synchronize()


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    value_dtype: torch.dtype,
    state_dtype: torch.dtype,
    is_output: bool,
    min_cosine: float | None = None,
) -> None:
    dtype = value_dtype if is_output else state_dtype
    if dtype == torch.bfloat16:
        rtol = 2e-2
        atol = 2e-2 if is_output else 8e-2
    else:
        rtol = 1.5e-2
        atol = 1.25e-2 if is_output else 6e-2
    actual_f = actual.detach().float().cpu()
    expected_f = expected.detach().float().cpu()
    diff = (actual_f - expected_f).abs()
    threshold = atol + rtol * expected_f.abs()
    bad = int((diff > threshold).sum().item())
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected_f.abs().clamp_min(1.0e-12)).max().item())
    cosine = float(
        F.cosine_similarity(actual_f.reshape(1, -1), expected_f.reshape(1, -1)).item()
    )
    print(
        f"{'output' if is_output else 'state'}: "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} bad={bad}/{diff.numel()} cosine={cosine:.9f}"
    )
    if min_cosine is not None:
        assert cosine >= min_cosine, f"cosine {cosine:.9f} is below {min_cosine:.9f}"
    torch.testing.assert_close(
        actual_f,
        expected_f,
        rtol=rtol,
        atol=atol,
    )


def _exp2_ftz_like(x: torch.Tensor) -> torch.Tensor:
    y = torch.exp2(torch.clamp(x, max=127.0))
    return torch.where(x < -126.0, torch.zeros_like(y), y)


def _full_precision_musa_matmul(fn):
    @wraps(fn)
    def wrapped(q: torch.Tensor, *args, **kwargs):
        if q.device.type != "musa":
            return fn(q, *args, **kwargs)
        allow_tf32 = torch.backends.mudnn.allow_tf32
        torch.backends.mudnn.allow_tf32 = False
        try:
            return fn(q, *args, **kwargs)
        finally:
            torch.backends.mudnn.allow_tf32 = allow_tf32

    return wrapped


def gen_qk(
    total: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    normalize: bool,
) -> torch.Tensor:
    qk = torch.empty(
        (total, num_heads, HEAD_SIZE), device=device, dtype=torch.float32
    ).uniform_(-0.25, 0.25)
    if normalize:
        # MUSA's broadcast division path used by F.normalize currently leaves
        # the singleton norm dimension un-applied.  Multiplication by rsqrt
        # is mathematically equivalent and lowers correctly on MUSA.
        qk = _normalize_last_dim(qk)
    return qk.to(dtype)


def _normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    # Match K1's normalization exactly. The epsilon is observable for the
    # all-zero Q/K vectors used by the single-token residual-layout probe:
    # without it the reference computes 0 * inf, which is NaN on CI devices.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1.0e-6)


def gen_kda_inputs(
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    normalize_qk: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    total = sum(seq_lens)
    q_dim = num_qk_heads * HEAD_SIZE
    k_dim = num_qk_heads * HEAD_SIZE
    v_dim = num_v_heads * HEAD_SIZE
    fused_dim = q_dim + k_dim + v_dim

    mixed_qkv = torch.empty((total, fused_dim), device=device, dtype=dtype)
    q_flat, k_flat, v_flat = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    # Keep the reference tensors on the runtime device.  The blockwise
    # reference below is intentionally a separate implementation, but moving
    # every operand to CPU made long-varlen tests dominated by host execution
    # and GPU/CPU synchronization rather than validating the kernel.
    q_ref = gen_qk(total, num_qk_heads, dtype, device, normalize=normalize_qk)
    k_ref = gen_qk(total, num_qk_heads, dtype, device, normalize=normalize_qk)
    v_ref = torch.empty(
        (total, num_v_heads, HEAD_SIZE), device=device, dtype=dtype
    ).uniform_(-0.25, 0.25)
    g_ref = torch.empty(
        (total, num_v_heads, HEAD_SIZE), device=device, dtype=dtype
    ).uniform_(-1.0, 1.0)
    beta_ref = torch.empty((total, num_v_heads), device=device, dtype=dtype).uniform_(
        -1.0, 1.0
    )

    q_dense = q_ref
    k_dense = k_ref
    v_dense = v_ref
    g = g_ref
    beta = beta_ref

    q_flat.copy_(q_dense.reshape(total, q_dim))
    k_flat.copy_(k_dense.reshape(total, k_dim))
    v_flat.copy_(v_dense.reshape(total, v_dim))

    q = q_flat.view(total, num_qk_heads, HEAD_SIZE)
    k = k_flat.view(total, num_qk_heads, HEAD_SIZE)
    v = v_flat.view(total, num_v_heads, HEAD_SIZE)

    assert q.stride() == (fused_dim, HEAD_SIZE, 1)
    assert k.stride() == (fused_dim, HEAD_SIZE, 1)
    assert v.stride() == (fused_dim, HEAD_SIZE, 1)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    return q, k, v, g, beta, q_ref, k_ref, v_ref, g_ref, beta_ref


def gen_gate_params(
    num_v_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    A_log = torch.empty((num_v_heads,), device=device, dtype=torch.float32).uniform_(
        -1.0, 0.5
    )
    dt_bias = torch.empty(
        (num_v_heads, HEAD_SIZE), device=device, dtype=torch.float32
    ).uniform_(-0.5, 0.5)
    return A_log, dt_bias


def gen_initial_state(
    num_seqs: int,
    num_v_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_initial_state: bool,
) -> torch.Tensor | None:
    if not use_initial_state:
        return None
    return torch.empty(
        (num_seqs, num_v_heads, HEAD_SIZE, HEAD_SIZE), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)


def _to_kernel_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    seq_lens: list[int],
    *,
    cu_seqlens_dtype: torch.dtype = torch.int64,
    varlen: bool,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if varlen:
        cu_seqlens = torch.tensor(
            exclusive_cumsum(seq_lens), device=device, dtype=cu_seqlens_dtype
        )
        return q, k, v, g, beta, cu_seqlens

    assert len(set(seq_lens)) == 1
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return (
        q.reshape(batch_size, num_tokens, q.shape[1], q.shape[2]).contiguous(),
        k.reshape(batch_size, num_tokens, k.shape[1], k.shape[2]).contiguous(),
        v.reshape(batch_size, num_tokens, v.shape[1], v.shape[2]).contiguous(),
        g.reshape(batch_size, num_tokens, g.shape[1], g.shape[2]),
        beta.reshape(batch_size, num_tokens, beta.shape[1]),
        None,
    )


def _reference_output_for_kernel_shape(
    output: torch.Tensor,
    *,
    seq_lens: list[int],
    varlen: bool,
) -> torch.Tensor:
    if varlen:
        return output
    batch_size = len(seq_lens)
    num_tokens = seq_lens[0]
    return output.reshape(batch_size, num_tokens, output.shape[1], output.shape[2])


def _assert_last_dim_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        assert tensor.stride(-1) == 1


def _assert_last_dim_contiguous_views(*tensors: torch.Tensor) -> None:
    _assert_last_dim_contiguous(*tensors)
    for tensor in tensors:
        assert not tensor.is_contiguous()


def _concat_varlen_segments(
    x1: torch.Tensor,
    seq_lens1: list[int],
    x2: torch.Tensor,
    seq_lens2: list[int],
) -> torch.Tensor:
    cu_seqlens1 = exclusive_cumsum(seq_lens1)
    cu_seqlens2 = exclusive_cumsum(seq_lens2)
    pieces = []
    for seq_idx in range(len(seq_lens1)):
        pieces.append(x1[cu_seqlens1[seq_idx] : cu_seqlens1[seq_idx + 1]])
        pieces.append(x2[cu_seqlens2[seq_idx] : cu_seqlens2[seq_idx + 1]])
    return torch.cat(pieces, dim=0)


def _run_fused_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mate.kda import chunk_kda

    result = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state=final_state,
        initial_state_indices=initial_state_indices,
    )
    assert isinstance(result, tuple)
    return result


@torch.inference_mode()
@_full_precision_musa_matmul
def blockwise_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = q.shape[-1]
    Hqk = q.shape[-2]
    H = v.shape[-2]
    assert D == HEAD_SIZE
    assert k.shape[-2] == Hqk
    assert g.shape[-2] == H
    assert beta.shape[-1] == H
    assert H % Hqk == 0
    qk_group = H // Hqk

    if cu_seqlens is None and q.ndim == 4:
        batch_size, num_tokens = q.shape[:2]
        flat_q = q.reshape(-1, Hqk, D)
        flat_k = k.reshape(-1, Hqk, D)
        flat_v = v.reshape(-1, H, D)
        flat_g = g.reshape(-1, H, D)
        flat_beta = beta.reshape(-1, H)
        seq_offsets = [i * num_tokens for i in range(batch_size + 1)]
        out_shape = v.shape
    elif cu_seqlens is None:
        if q.ndim != 3:
            raise AssertionError(
                "Dense reference inputs must be [B, T, H, D] or flattened to [B*T, H, D]."
            )
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_len = flat_q.shape[0] // initial_state.shape[0]
        seq_offsets = [i * seq_len for i in range(initial_state.shape[0] + 1)]
        out_shape = v.shape
    else:
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_offsets = [int(x) for x in cu_seqlens.tolist()]
        out_shape = v.shape

    if use_qk_l2norm_in_kernel:
        qn = _normalize_last_dim(flat_q.float()).to(q.dtype).float()
        kn = _normalize_last_dim(flat_k.float()).to(k.dtype).float()
    else:
        qn = flat_q.float()
        kn = flat_k.float()
    out = torch.empty_like(flat_v)
    state = initial_state.clone()
    log2e = 1.4426950408889634
    qk_head_indices = torch.arange(H, device=q.device, dtype=torch.long) // qk_group

    def round_workspace(x: torch.Tensor) -> torch.Tensor:
        return x.to(q.dtype).float()

    for seq_idx, bos in enumerate(seq_offsets[:-1]):
        eos = seq_offsets[seq_idx + 1]
        s = state[seq_idx].float().transpose(-1, -2)
        for start in range(bos, eos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, eos)
            actual = end - start
            q_blk = torch.zeros(
                (H, CHUNK_SIZE, D), device=q.device, dtype=torch.float32
            )
            k_blk = torch.zeros_like(q_blk)
            v_blk = torch.zeros_like(q_blk)
            g_blk = torch.zeros_like(q_blk)
            beta_blk = torch.full(
                (H, CHUNK_SIZE), -80.0, device=q.device, dtype=torch.float32
            )

            q_blk[:, :actual] = (
                qn[start:end].index_select(1, qk_head_indices).permute(1, 0, 2)
            )
            k_blk[:, :actual] = (
                kn[start:end].index_select(1, qk_head_indices).permute(1, 0, 2)
            )
            v_blk[:, :actual] = flat_v[start:end].permute(1, 0, 2).float()
            g_blk[:, :actual] = flat_g[start:end].permute(1, 0, 2).float()
            beta_blk[:, :actual] = flat_beta[start:end].transpose(0, 1).float()

            gate = (
                lower_bound
                * log2e
                * torch.sigmoid(
                    torch.exp(A_log)[:, None, None] * (g_blk + dt_bias[:, None, :])
                )
            )
            if actual < CHUNK_SIZE:
                gate[:, actual:] = 0.0
            g_cumsum = gate.cumsum(dim=1)
            g_total = g_cumsum[:, -1]
            shift = 0.5 * (g_cumsum[:, 0] + g_total)
            k_decayed = round_workspace(k_blk * _exp2_ftz_like(g_cumsum))
            q_decayed = round_workspace(q_blk * _exp2_ftz_like(g_cumsum) * scale)
            k_decayed_for_tri = round_workspace(
                k_blk * _exp2_ftz_like(g_cumsum - shift[:, None])
            )
            q_decayed_for_p = round_workspace(
                q_blk * _exp2_ftz_like(g_cumsum - shift[:, None]) * scale
            )
            k_inv = round_workspace(k_blk * _exp2_ftz_like(shift[:, None] - g_cumsum))
            beta_sigmoid = torch.sigmoid(beta_blk)
            l_mat = round_workspace(
                torch.bmm(k_decayed_for_tri, k_inv.transpose(1, 2)).tril(-1)
                * beta_sigmoid[:, :, None]
            )

            a = torch.zeros(
                (H, CHUNK_SIZE, CHUNK_SIZE), device=q.device, dtype=torch.float32
            )
            a[:, 0, 0] = beta_sigmoid[:, 0]
            for i in range(1, CHUNK_SIZE):
                a[:, i, i] = beta_sigmoid[:, i]
                a[:, i, :i] = -torch.bmm(l_mat[:, i : i + 1, :i], a[:, :i, :i]).squeeze(
                    1
                )
            a = round_workspace(a)
            p = round_workspace(
                torch.bmm(q_decayed_for_p, k_inv.transpose(1, 2)).tril()
            )
            s_committed = round_workspace(s)
            tmp = round_workspace(v_blk - torch.bmm(k_decayed, s_committed))
            u = round_workspace(torch.bmm(a, tmp))
            out_chunk = round_workspace(
                torch.bmm(q_decayed, s_committed) + torch.bmm(p, u)
            )
            out[start:end] = out_chunk[:, :actual].permute(1, 0, 2)

            k_restored = round_workspace(
                k_blk * _exp2_ftz_like(g_total[:, None] - g_cumsum)
            )
            s = s * _exp2_ftz_like(g_total)[:, None, :] + torch.bmm(
                k_restored.transpose(1, 2), u
            )
        state[seq_idx].copy_(s.transpose(-1, -2).to(initial_state.dtype))

    return out.reshape(out_shape), state


def _test_kda_kernel(
    *,
    dtype_name: str,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    state_dtype_name: str,
    varlen: bool,
    cu_seqlens_dtype: torch.dtype = torch.int64,
    seed: int,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    zero_v: bool = False,
    min_output_cosine: float | None = None,
) -> None:
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    state_dtype = getattr(torch, state_dtype_name)
    scale = HEAD_SIZE**-0.5
    q, k, v, g, beta, q_ref, k_ref, v_ref, g_ref, beta_ref = gen_kda_inputs(
        seq_lens,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=not use_qk_l2norm_in_kernel,
    )
    if zero_v:
        v.zero_()
        v_ref.zero_()
    A_log, dt_bias = gen_gate_params(num_v_heads, device)
    initial_state = gen_initial_state(
        len(seq_lens),
        num_v_heads,
        state_dtype,
        device,
        use_initial_state=use_initial_state,
    )
    if initial_state is None:
        ref_initial_state = torch.zeros(
            (len(seq_lens), num_v_heads, HEAD_SIZE, HEAD_SIZE),
            device=device,
            dtype=state_dtype,
        )
    else:
        ref_initial_state = initial_state.clone()

    q_in, k_in, v_in, g_in, beta_in, cu_seqlens = _to_kernel_inputs(
        q,
        k,
        v,
        g,
        beta,
        seq_lens,
        cu_seqlens_dtype=cu_seqlens_dtype,
        varlen=varlen,
        device=device,
    )
    if varlen:
        _assert_last_dim_contiguous_views(q_in, k_in, v_in)
    else:
        _assert_last_dim_contiguous(q_in, k_in, v_in)

    actual_o, actual_state = _run_fused_kda(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    _synchronize(device)
    if initial_state is not None:
        assert actual_state.data_ptr() != initial_state.data_ptr()
        torch.testing.assert_close(initial_state, ref_initial_state, rtol=0, atol=0)

    expected_o, expected_state = blockwise_kda_reference(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        scale=scale,
        initial_state=ref_initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens if cu_seqlens is not None else None,
    )
    expected_o = _reference_output_for_kernel_shape(
        expected_o, seq_lens=seq_lens, varlen=varlen
    )

    _assert_close(
        actual_o,
        expected_o,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=True,
        min_cosine=min_output_cosine,
    )
    _assert_close(
        actual_state,
        expected_state,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=False,
    )


def _test_chunked_kda(
    *,
    dtype_name: str,
    seq_lens1: list[int],
    seq_lens2: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    state_dtype_name: str,
    seed: int,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
) -> None:
    assert len(seq_lens1) == len(seq_lens2)
    device = _get_runtime_device()
    _manual_seed(seed, device)

    dtype = getattr(torch, dtype_name)
    state_dtype = getattr(torch, state_dtype_name)
    scale = HEAD_SIZE**-0.5
    q1, k1, v1, g1, beta1, q1_ref, k1_ref, v1_ref, g1_ref, beta1_ref = gen_kda_inputs(
        seq_lens1,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=not use_qk_l2norm_in_kernel,
    )
    q2, k2, v2, g2, beta2, q2_ref, k2_ref, v2_ref, g2_ref, beta2_ref = gen_kda_inputs(
        seq_lens2,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=not use_qk_l2norm_in_kernel,
    )
    A_log, dt_bias = gen_gate_params(num_v_heads, device)
    _assert_last_dim_contiguous_views(q1, k1, v1, q2, k2, v2)

    cu_seqlens1 = torch.tensor(
        exclusive_cumsum(seq_lens1), device=device, dtype=torch.int64
    )
    cu_seqlens2 = torch.tensor(
        exclusive_cumsum(seq_lens2), device=device, dtype=torch.int64
    )
    state1_buffer = torch.empty(
        (len(seq_lens1), num_v_heads, HEAD_SIZE, HEAD_SIZE),
        device=device,
        dtype=state_dtype,
    )
    state2_buffer = torch.empty(
        (len(seq_lens2), num_v_heads, HEAD_SIZE, HEAD_SIZE),
        device=device,
        dtype=state_dtype,
    )

    actual_o1, state1 = _run_fused_kda(
        q1,
        k1,
        v1,
        g1,
        beta1,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens1,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state=state1_buffer,
    )
    actual_o2, state2 = _run_fused_kda(
        q2,
        k2,
        v2,
        g2,
        beta2,
        scale=scale,
        initial_state=state1,
        output_final_state=True,
        cu_seqlens=cu_seqlens2,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        final_state=state2_buffer,
    )
    _synchronize(device)

    q = _concat_varlen_segments(q1_ref, seq_lens1, q2_ref, seq_lens2)
    k = _concat_varlen_segments(k1_ref, seq_lens1, k2_ref, seq_lens2)
    v = _concat_varlen_segments(v1_ref, seq_lens1, v2_ref, seq_lens2)
    g = _concat_varlen_segments(g1_ref, seq_lens1, g2_ref, seq_lens2)
    beta = _concat_varlen_segments(beta1_ref, seq_lens1, beta2_ref, seq_lens2)
    seq_lens = [a + b for a, b in zip(seq_lens1, seq_lens2)]
    expected_o, expected_state = blockwise_kda_reference(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=torch.zeros(
            (len(seq_lens), num_v_heads, HEAD_SIZE, HEAD_SIZE),
            device=device,
            dtype=state_dtype,
        ),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=torch.tensor(
            exclusive_cumsum(seq_lens), device=device, dtype=torch.int64
        ),
    )
    actual_o = _concat_varlen_segments(actual_o1, seq_lens1, actual_o2, seq_lens2)

    _assert_close(
        actual_o, expected_o, value_dtype=dtype, state_dtype=state_dtype, is_output=True
    )
    _assert_close(
        state2,
        expected_state,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=False,
    )


DENSE_KDA_CASES = [
    pytest.param([64], 1, 1, False, "bfloat16", -5.0, 0, id="dense-64-h1-no-state"),
    pytest.param([64, 64], 2, 2, True, "bfloat16", -5.0, 1, id="dense-batch2-h2-state"),
    pytest.param([32, 32], 1, 2, True, "float32", -5.0, 2, id="dense-gva-fp32-state"),
]

VARLEN_KDA_CASES = [
    pytest.param([17], 1, 1, False, "bfloat16", -5.0, 3, id="varlen-tail-17"),
    pytest.param([31, 63], 1, 2, True, "bfloat16", -5.0, 4, id="varlen-gva-tails"),
    pytest.param(
        [19, 32, 57], 2, 2, True, "float32", -5.0, 5, id="varlen-multi-fp32-state"
    ),
]

CHUNKED_KDA_CASES = [
    pytest.param([31], [33], 1, 1, "bfloat16", -5.0, 6, id="chunked-single-seq"),
    pytest.param(
        [31, 63], [33, 65], 1, 2, "float32", -5.0, 7, id="chunked-gva-fp32-state"
    ),
]


@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize(
    "use_qk_l2norm_in_kernel",
    [False, True],
    ids=["pre-norm", "kernel-l2norm"],
)
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "use_initial_state",
        "state_dtype_name",
        "lower_bound",
        "seed_offset",
    ),
    DENSE_KDA_CASES,
)
def test_kda_fused_matches_reference(
    dtype_name: str,
    use_qk_l2norm_in_kernel: bool,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    state_dtype_name: str,
    lower_bound: float,
    seed_offset: int,
) -> None:
    _test_kda_kernel(
        dtype_name=dtype_name,
        seq_lens=seq_lens,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        use_initial_state=use_initial_state,
        state_dtype_name=state_dtype_name,
        varlen=False,
        cu_seqlens_dtype=torch.int64,
        seed=_resolve_seed(seed_offset),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize(
    "use_qk_l2norm_in_kernel",
    [False, True],
    ids=["pre-norm", "kernel-l2norm"],
)
@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_qk_heads",
        "num_v_heads",
        "use_initial_state",
        "state_dtype_name",
        "lower_bound",
        "seed_offset",
    ),
    VARLEN_KDA_CASES,
)
def test_kda_fused_varlen_matches_reference(
    dtype_name: str,
    use_qk_l2norm_in_kernel: bool,
    seq_lens: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    use_initial_state: bool,
    state_dtype_name: str,
    lower_bound: float,
    seed_offset: int,
) -> None:
    _test_kda_kernel(
        dtype_name=dtype_name,
        seq_lens=seq_lens,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        use_initial_state=use_initial_state,
        state_dtype_name=state_dtype_name,
        varlen=True,
        cu_seqlens_dtype=torch.int64,
        seed=_resolve_seed(seed_offset),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def test_kda_fused_varlen_int32_cu_seqlens_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[19, 32, 57],
        num_qk_heads=2,
        num_v_heads=2,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=True,
        cu_seqlens_dtype=torch.int32,
        seed=_resolve_seed(105),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.parametrize("output_final_state", [False, True])
def test_kda_fused_varlen_state_pool_indices_matches_reference(
    output_final_state: bool,
) -> None:
    from mate.kda import chunk_kda

    device = _get_runtime_device()
    _manual_seed(_resolve_seed(110), device)
    dtype = torch.bfloat16
    state_dtype = torch.float32
    seq_lens = [31, 47]
    num_qk_heads = 1
    num_v_heads = 2
    scale = HEAD_SIZE**-0.5
    q, k, v, g, beta, q_ref, k_ref, v_ref, g_ref, beta_ref = gen_kda_inputs(
        seq_lens,
        num_qk_heads,
        num_v_heads,
        dtype,
        device,
        normalize_qk=False,
    )
    q_in, k_in, v_in, g_in, beta_in, cu_seqlens = _to_kernel_inputs(
        q,
        k,
        v,
        g,
        beta,
        seq_lens,
        cu_seqlens_dtype=torch.int32,
        varlen=True,
        device=device,
    )
    A_log, dt_bias = gen_gate_params(num_v_heads, device)
    state_pool = gen_initial_state(
        4,
        num_v_heads,
        state_dtype,
        device,
        use_initial_state=True,
    )
    assert state_pool is not None
    state_pool_before = state_pool.clone()
    state_indices = torch.tensor([3, 1], device=device, dtype=torch.int32)
    selected_initial_state = state_pool_before.index_select(
        0, state_indices.to(torch.long)
    )

    expected_o, expected_selected_state = blockwise_kda_reference(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        scale=scale,
        initial_state=selected_initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    expected_pool = state_pool_before.clone()
    expected_pool.index_copy_(0, state_indices.to(torch.long), expected_selected_state)

    result = chunk_kda(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        scale=scale,
        initial_state=state_pool,
        initial_state_indices=state_indices,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )
    if output_final_state:
        assert isinstance(result, tuple)
        actual_o, returned_pool = result
        assert returned_pool.data_ptr() == state_pool.data_ptr()
    else:
        assert isinstance(result, torch.Tensor)
        actual_o = result
    _synchronize(device)

    _assert_close(
        actual_o,
        expected_o,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=True,
    )
    _assert_close(
        state_pool,
        expected_pool,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=False,
    )


@repeat_check(outputs="result", repeat=3, scope=_run_fused_kda)
def test_kda_fused_single_chunk_zero_v_with_initial_state_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[32],
        num_qk_heads=1,
        num_v_heads=1,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=False,
        cu_seqlens_dtype=torch.int64,
        seed=_resolve_seed(107),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
        zero_v=True,
        min_output_cosine=0.999,
    )


def test_kda_fused_varlen_single_sequence_four_chunks_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[128],
        num_qk_heads=2,
        num_v_heads=2,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=True,
        cu_seqlens_dtype=torch.int32,
        seed=_resolve_seed(106),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )


@repeat_check(outputs="result", repeat=3, scope=_run_fused_kda)
def test_kda_fused_varlen_single_sequence_2048_matches_reference() -> None:
    _test_kda_kernel(
        dtype_name="bfloat16",
        seq_lens=[2048],
        num_qk_heads=1,
        num_v_heads=1,
        use_initial_state=True,
        state_dtype_name="float32",
        varlen=True,
        cu_seqlens_dtype=torch.int32,
        seed=_resolve_seed(109),
        lower_bound=-5.0,
        use_qk_l2norm_in_kernel=True,
    )


def test_kda_single_token_state_residual_v_layout() -> None:
    device = _get_runtime_device()
    dtype = torch.bfloat16
    state_dtype = torch.float32
    k_index = 18

    q_ref = torch.zeros((1, 1, HEAD_SIZE), device=device, dtype=dtype)
    k_ref = torch.zeros_like(q_ref)
    k_ref[0, 0, k_index] = 1
    v_ref = torch.zeros_like(q_ref)
    g_ref = torch.zeros_like(q_ref)
    beta_ref = torch.zeros((1, 1), dtype=dtype)
    initial_state = torch.zeros(
        (1, 1, HEAD_SIZE, HEAD_SIZE), device=device, dtype=state_dtype
    )
    # Encoding V in one K column makes any residual V-row permutation directly observable.
    initial_state[0, 0, :, k_index] = torch.arange(HEAD_SIZE, dtype=state_dtype)
    A_log = torch.zeros((1,), device=device, dtype=torch.float32)
    dt_bias = torch.zeros((1, HEAD_SIZE), device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)

    actual_o, actual_state = _run_fused_kda(
        q_ref.to(device),
        k_ref.to(device),
        v_ref.to(device),
        g_ref.to(device),
        beta_ref.to(device),
        scale=1.0,
        initial_state=initial_state.to(device),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=0.0,
        use_qk_l2norm_in_kernel=True,
    )
    _synchronize(device)

    expected_o, expected_state = blockwise_kda_reference(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        scale=1.0,
        initial_state=initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=0.0,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, 1], device=device, dtype=torch.int64),
    )
    _assert_close(
        actual_o,
        expected_o,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=True,
    )
    _assert_close(
        actual_state,
        expected_state,
        value_dtype=dtype,
        state_dtype=state_dtype,
        is_output=False,
    )


@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize(
    "use_qk_l2norm_in_kernel",
    [False, True],
    ids=["pre-norm", "kernel-l2norm"],
)
@pytest.mark.parametrize(
    (
        "seq_lens1",
        "seq_lens2",
        "num_qk_heads",
        "num_v_heads",
        "state_dtype_name",
        "lower_bound",
        "seed_offset",
    ),
    CHUNKED_KDA_CASES,
)
def test_kda_fused_chunked_matches_reference(
    dtype_name: str,
    use_qk_l2norm_in_kernel: bool,
    seq_lens1: list[int],
    seq_lens2: list[int],
    num_qk_heads: int,
    num_v_heads: int,
    state_dtype_name: str,
    lower_bound: float,
    seed_offset: int,
) -> None:
    _test_chunked_kda(
        dtype_name=dtype_name,
        seq_lens1=seq_lens1,
        seq_lens2=seq_lens2,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        state_dtype_name=state_dtype_name,
        seed=_resolve_seed(seed_offset),
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
