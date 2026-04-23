from __future__ import annotations

import os
import random

import pytest
import torch
import torch.nn.functional as F

import mate


# Keep the reference path numerically aligned with gdn_unified_tilelang testing.
if hasattr(torch.backends, "mudnn"):
    torch.backends.mudnn.allow_tf32 = False


@torch.inference_mode
def decode_delta_rule(
    q: torch.Tensor,  # [B, num_q_heads, K]
    k: torch.Tensor,  # [B, num_k_heads, K]
    v: torch.Tensor,  # [B, num_v_heads, V]
    state: torch.Tensor,  # [B, num_heads, K, V]
    A_log: torch.Tensor,  # [num_heads] - log decay parameter
    a: torch.Tensor,  # [B, num_heads] - input-dependent decay
    dt_bias: torch.Tensor,  # [num_heads] - decay bias
    b: torch.Tensor,  # [B, num_heads] - update gate input
    scale_factor: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_l2_norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for single-step decode with GDN formula.

    Strictly follows the Triton kernel logic from fused_sigmoid_gating_recurrent.py:
        1. Compute g = -exp(A_log) * softplus(a + dt_bias)
        2. Compute beta = sigmoid(b)
        3. Apply L2 norm to q and k (if enabled)
        4. h *= exp(g)                    # Apply decay to state
        5. v_new = v - k^T @ h            # Delta rule (h is [K,V], k is [K])
        6. v_new *= beta                  # Apply update gate
        7. h += k @ v_new^T               # Update state (outer product)
        8. o = q^T @ h                    # Compute output (h is [K,V], q is [K])

    Args:
        q: Query [B, num_q_heads, K]
        k: Key [B, num_k_heads, K]
        v: Value [B, num_v_heads, V]
        state: Input state [B, num_heads, K, V], where num_heads = num_v_heads
        A_log: Log decay parameter [num_heads]
        a: Input-dependent decay [B, num_heads]
        dt_bias: Decay bias [num_heads]
        b: Update gate input [B, num_heads]
        scale_factor: Scale factor for q
        softplus_beta: Beta parameter for softplus activation
        softplus_threshold: Threshold for softplus numerical stability
        use_l2_norm: Whether to apply L2 normalization to q and k
        state_dtype: Storage dtype for the hidden state (read in fp32, stored in this dtype)

    Returns:
        output: [B, num_heads, V]
        new_state: [B, num_heads, K, V]
    """
    B = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    K = q.size(2)
    V = v.size(2)

    # State and output are always based on num_v_heads (matches kernel's HV dimension)
    num_heads = num_v_heads

    device = q.device
    dtype = torch.float32

    # Convert to float32 for computation
    A_log = A_log.to(dtype).to(device)
    a = a.to(dtype).to(device)
    dt_bias = dt_bias.to(dtype).to(device)
    b = b.to(dtype).to(device)

    # ============================================
    # Compute gating values (following Triton kernel exactly)
    # ============================================

    # Step 1: Compute g = -exp(A_log) * softplus(a + dt_bias)
    # Triton kernel lines 100-109
    x = a + dt_bias  # [B, num_heads]
    beta_x = softplus_beta * x

    # Apply softplus with numerical stability
    # softplus(x) = (1/beta) * log(1 + exp(beta*x)) if beta*x <= threshold, else x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )

    # Compute g (log-space decay gate)
    # Triton kernel line 109: b_g = -tl.exp(b_A_log) * softplus_x
    g = -torch.exp(A_log) * softplus_x  # [B, num_heads]

    # Step 2: Compute beta = sigmoid(b)
    # Triton kernel line 112: b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    beta = 1.0 / (1.0 + torch.exp(-b))  # [B, num_heads]

    # Expand heads if needed (for GQA/GVA)
    # The reference works at v_heads level
    # For GQA (num_q_heads > num_v_heads): k and q need to be averaged/pooled per v_head
    # For GVA (num_v_heads > num_q_heads): q and k need to be repeated
    if num_k_heads < num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    if num_q_heads < num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    elif num_q_heads > num_v_heads:
        # GQA: multiple q_heads per v_head, reshape and average
        # [B, num_q_heads, K] -> [B, num_v_heads, num_q_heads//num_v_heads, K]
        q = q.reshape(B, num_v_heads, num_q_heads // num_v_heads, K).mean(dim=2)
        if num_k_heads == num_q_heads:
            k = k.reshape(B, num_v_heads, num_k_heads // num_v_heads, K).mean(dim=2)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    state = state.to(dtype)

    # Apply L2 normalization if requested
    if use_l2_norm:
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

    # Apply scale to q
    q = q * scale_factor

    # ============================================
    # Process each batch and head
    # ============================================
    new_state = torch.zeros(B, num_heads, K, V, device=device, dtype=state_dtype)
    output = torch.zeros(B, num_heads, V, device=device, dtype=dtype)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            # Get current vectors
            q_h = q[b_idx, h_idx]  # [K]
            k_h = k[b_idx, h_idx]  # [K]
            v_h = v[b_idx, h_idx]  # [V]
            h_state = (
                state[b_idx, h_idx].clone().to(torch.float32)
            )  # [K, V] read as fp32

            # Get gating values for this batch and head
            g_val = g[b_idx, h_idx]  # scalar
            beta_val = beta[b_idx, h_idx]  # scalar

            # ============================================
            # Recurrent update (following Triton kernel lines 121-134)
            # ============================================

            # Step 1: Apply gating to hidden state: h *= exp(g)
            # Triton kernel line 122: b_h *= tl.exp(b_g)
            h_state = h_state * torch.exp(g_val)

            # Step 2: Delta rule: v -= sum(h * k, dim=0)
            # Triton kernel line 125: b_v -= tl.sum(b_h * b_k[:, None], 0)
            # Triton: b_h is [BK, BV], b_k is [BK]
            # b_k[:, None] makes it [BK, 1]
            # b_h * b_k[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: k^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            v_new = v_h - (k_h @ h_state)

            # Step 3: Apply beta gating: v *= beta
            # Triton kernel line 128: b_v *= b_beta
            v_new = v_new * beta_val

            # Step 4: Update hidden state: h += k[:, None] * v[None, :]
            # Triton kernel line 131: b_h += b_k[:, None] * b_v[None, :]
            # Triton: [BK, BV] += [BK, 1] * [1, BV]
            # This is outer product: k @ v^T
            # [K, V] += [K, 1] @ [1, V]
            h_state = h_state + k_h.unsqueeze(1) @ v_new.unsqueeze(0)

            # Step 5: Compute output: o = sum(h * q, dim=0)
            # Triton kernel line 134: b_o = tl.sum(b_h * b_q[:, None], 0)
            # Triton: b_h is [BK, BV], b_q is [BK]
            # b_q[:, None] makes it [BK, 1]
            # b_h * b_q[:, None] gives [BK, BV] (element-wise per row)
            # tl.sum(..., 0) sums over BK dimension -> [BV]
            #
            # Equivalent to: q^T @ h where h is [K, V]
            # [K] @ [K, V] = [V]
            output[b_idx, h_idx] = q_h @ h_state

            # Store updated state (cast back to state_dtype)
            new_state[b_idx, h_idx] = h_state.to(state_dtype)

    return output, new_state


def _get_default_seed() -> int:
    return int(os.environ.get("SEED", "0"))


def _get_runtime_device(*, allow_skip: bool) -> torch.device:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.device("musa")
    if allow_skip:
        pytest.skip("MUSA is not available")
    raise RuntimeError("MUSA is not available")


def _manual_seed_device(seed: int | None, device_type: str) -> None:
    if seed is None:
        return
    if device_type == "musa":
        torch.musa.manual_seed(seed)


def _synchronize_device(device_type: str) -> None:
    if device_type == "musa":
        torch.musa.synchronize()


def _test_decode_kernel_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int | None = None,
    compare_values: bool = False,
    allow_skip: bool = False,
):
    del alpha

    random.seed(seed)
    torch.random.manual_seed(seed)

    num_sab_heads = num_v_heads
    dtype_torch = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = _get_runtime_device(allow_skip=allow_skip)
    _manual_seed_device(seed, device.type)

    q = torch.randn(
        batch_size, 1, num_q_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_k_heads, head_size, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_v_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    input_state_ref = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=kv_dtype,
        device=device,
    )
    input_state_kernel = input_state_ref.transpose(-2, -1).contiguous()

    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = (
        torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
        * 0.1
    )

    if beta:
        b_tensor = torch.randn(
            batch_size,
            1,
            num_sab_heads,
            dtype=dtype_torch,
            device=device,
        )
    else:
        b_tensor = (
            torch.ones(
                batch_size,
                1,
                num_sab_heads,
                dtype=dtype_torch,
                device=device,
            )
            * 10.0
        )

    our_state = input_state_kernel.clone()
    our_state_ptr = our_state.data_ptr()
    our_o, returned_state = mate.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=our_state,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b_tensor,
        state_layout="VK",
        scale=scale,
        use_qk_l2norm=True,
    )
    _synchronize_device(device.type)
    assert our_state.data_ptr() == our_state_ptr
    assert returned_state.data_ptr() == our_state_ptr
    our_state = returned_state
    our_o = our_o.squeeze(1)

    ref_o, ref_state = decode_delta_rule(
        q.squeeze(1).float(),
        k.squeeze(1).float(),
        v.squeeze(1).float(),
        input_state_ref,
        A_log=A_log.float(),
        a=a.squeeze(1).float(),
        dt_bias=dt_bias.float(),
        b=b_tensor.squeeze(1).float(),
        scale_factor=scale,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_l2_norm=True,
    )

    ref_o = ref_o.to(dtype_torch)
    ref_state = ref_state.transpose(-2, -1).contiguous().to(kv_dtype)

    assert our_o.shape == ref_o.shape
    assert our_state.shape == ref_state.shape

    if compare_values:
        atol_o = 5e-3
        rtol_o = 5e-3
        atol_kv = 5e-3
        rtol_kv = 5e-3

        torch.testing.assert_close(our_o, ref_o, atol=atol_o, rtol=rtol_o)
        torch.testing.assert_close(our_state, ref_state, atol=atol_kv, rtol=rtol_kv)

    print(f"decode kernel test passed (batch={batch_size}, dtype={dtype})")


def _test_decode_kernel_bitwise_stable_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    beta: bool,
    seed: int | None = None,
    repeats: int = 50,
    allow_skip: bool = False,
):
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_sab_heads = num_v_heads
    dtype_torch = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = _get_runtime_device(allow_skip=allow_skip)
    _manual_seed_device(seed, device.type)

    q = torch.randn(
        batch_size, 1, num_q_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.randn(
        batch_size, 1, num_k_heads, head_size, dtype=dtype_torch, device=device
    )
    v = torch.randn(
        batch_size, 1, num_v_heads, head_size, dtype=dtype_torch, device=device
    )
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)

    input_state_ref = torch.randn(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=kv_dtype,
        device=device,
    )
    input_state_kernel = input_state_ref.transpose(-2, -1).contiguous()

    A_log = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(num_sab_heads, dtype=torch.float32, device=device) * 0.1
    a = (
        torch.randn(batch_size, 1, num_sab_heads, dtype=dtype_torch, device=device)
        * 0.1
    )

    if beta:
        b_tensor = torch.randn(
            batch_size,
            1,
            num_sab_heads,
            dtype=dtype_torch,
            device=device,
        )
    else:
        b_tensor = (
            torch.ones(
                batch_size,
                1,
                num_sab_heads,
                dtype=dtype_torch,
                device=device,
            )
            * 10.0
        )

    baseline_o = None
    baseline_state = None

    for repeat_idx in range(repeats):
        state = input_state_kernel.clone()
        state_ptr = state.data_ptr()
        out, returned_state = mate.gated_delta_rule_decode(
            q=q,
            k=k,
            v=v,
            state=state,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b_tensor,
            state_layout="VK",
            scale=scale,
            use_qk_l2norm=True,
        )
        _synchronize_device(device.type)
        assert state.data_ptr() == state_ptr
        assert returned_state.data_ptr() == state_ptr

        out = out.squeeze(1)

        if repeat_idx == 0:
            baseline_o = out.clone()
            baseline_state = returned_state.clone()
            continue

        if not torch.equal(out, baseline_o):
            diff = (out.float() - baseline_o.float()).abs().max().item()
            raise AssertionError(
                f"Output is not bitwise stable at repeat={repeat_idx}, max_abs_diff={diff}"
            )
        if not torch.equal(returned_state, baseline_state):
            diff = (returned_state - baseline_state).abs().max().item()
            raise AssertionError(
                f"State is not bitwise stable at repeat={repeat_idx}, max_abs_diff={diff}"
            )

    print(
        f"decode kernel bitwise stability test passed "
        f"(batch={batch_size}, dtype={dtype}, repeats={repeats})"
    )


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_basic_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    alpha: bool,
    beta: bool,
    seed: int = _get_default_seed(),
) -> None:
    _test_decode_kernel_pretranspose(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        scale=scale,
        alpha=alpha,
        beta=beta,
        seed=seed,
        compare_values=True,
        allow_skip=True,
    )


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("scale", [1.0])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(16, 16, 32)])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_decode_kernel_bitwise_stable_pretranspose(
    dtype: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    scale: float,
    beta: bool,
    seed: int = _get_default_seed(),
) -> None:
    _test_decode_kernel_bitwise_stable_pretranspose(
        dtype=dtype,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        scale=scale,
        beta=beta,
        seed=seed,
        allow_skip=True,
    )


def main() -> None:
    _test_decode_kernel_bitwise_stable_pretranspose(
        dtype="bfloat16",
        batch_size=8,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        beta=True,
        seed=_get_default_seed(),
    )
    _test_decode_kernel_pretranspose(
        dtype="bfloat16",
        batch_size=8,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=32,
        head_size=128,
        scale=1.0,
        alpha=True,
        beta=True,
        seed=_get_default_seed(),
        compare_values=True,
    )


if __name__ == "__main__":
    main()
