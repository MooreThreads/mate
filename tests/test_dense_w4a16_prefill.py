# SPDX-License-Identifier: Apache-2.0
"""Device parity for the large-M dense W4A16 kernel against an fp32 dequant reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

try:
    import torch_musa  # noqa: F401

    _HAVE_MUSA = torch.musa.is_available()
except Exception:  # pragma: no cover - host without the MUSA runtime
    _HAVE_MUSA = False

requires_musa = pytest.mark.skipif(not _HAVE_MUSA, reason="needs a MUSA device")

GROUP = 128
QWEN2_1_5B = [("qkv", 1536, 2048), ("o", 1536, 1536), ("gate_up", 1536, 17920), ("down", 8960, 1536)]
TAIL_SHAPES = [("n_tail", 1536, 320), ("k_odd_group", 4480, 1536)]
MS = (65, 127, 128, 129, 512, 2047, 2048)


def _checkpoint(size_k: int, size_n: int, seed: int, device: str):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 16, (size_n, size_k), generator=gen, dtype=torch.int32)
    # Per-group, per-column distinct scales, so a scale read from the wrong group or column shows.
    scales = (torch.rand((size_n, size_k // GROUP), generator=gen) * 0.02 + 0.001).half()
    nibbles = (codes ^ 0x8).to(torch.uint8).view(size_n, size_k // 2, 2)
    weight_b = (nibbles[..., 0] | (nibbles[..., 1] << 4)).view(torch.int8)  # (N, K/2), even k low
    dequant = scales.float().repeat_interleave(GROUP, dim=1) * (codes - 8).float()  # (N, K)
    return weight_b.to(device), scales.to(device), dequant.to(device)


def _run_prefill(a, weight_b, scales, out_dtype, tile=None, zero_points=None):
    from mate.jit.gemm.dense_w4a16_gemm import (
        get_dense_w4a16_prefill_module,
        prefill_row_sums,
        prepare_w4a16_prefill_operand,
    )

    out = torch.full((a.shape[0], weight_b.shape[0]), float("nan"), dtype=out_dtype, device=a.device)
    name, module = get_dense_w4a16_prefill_module(out_dtype, tile, zero_points is not None)
    operand = prepare_w4a16_prefill_operand(scales, zero_points)
    row_sums = prefill_row_sums(a) if zero_points is not None else operand
    module.get_function(name)(a, weight_b, operand, row_sums, out, 0)
    torch.musa.synchronize()
    return out


def _assert_parity(out, ref):
    assert torch.isfinite(out).all(), "non-finite output (an unwritten element keeps its NaN fill)"
    out_f = out.float()
    assert out_f.abs().max() > 0, "all-zero output"
    assert out_f.std() > 0, "constant output"
    err = (out_f - ref).abs()
    mean_rel = err.mean() / ref.abs().mean()
    max_rel = err.max() / ref.abs().max()
    tol = 2e-3 if out.dtype == torch.float16 else 8e-3
    assert mean_rel < tol and max_rel < 4 * tol, f"mean_rel={mean_rel:.3e} max_rel={max_rel:.3e}"


@requires_musa
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("shape", QWEN2_1_5B + TAIL_SHAPES, ids=lambda s: s[0])
def test_prefill_matches_dequant_reference(shape, m, out_dtype):
    _, size_k, size_n = shape
    weight_b, scales, dequant = _checkpoint(size_k, size_n, seed=size_k * 7 + size_n, device="musa")
    gen = torch.Generator(device="cpu").manual_seed(m)
    a = (torch.randn((m, size_k), generator=gen) * 0.5).half().to("musa")
    ref = a.float() @ dequant.t()
    _assert_parity(_run_prefill(a, weight_b, scales, out_dtype), ref)


@requires_musa
@pytest.mark.parametrize("tile", [(128, 128, 128, 4)], ids=["n128"])
@pytest.mark.parametrize("m", (129, 2048))
def test_prefill_narrow_tile_matches_reference(tile, m):
    weight_b, scales, dequant = _checkpoint(8960, 1536, seed=3, device="musa")
    a = torch.randn((m, 8960), device="musa").half()
    _assert_parity(_run_prefill(a, weight_b, scales, torch.float16, tile), a.float() @ dequant.t())


@requires_musa
def test_prefill_is_deterministic():
    weight_b, scales, _ = _checkpoint(1536, 17920, seed=5, device="musa")
    a = torch.randn((2048, 1536), device="musa").half()
    first = _run_prefill(a, weight_b, scales, torch.float16)
    second = _run_prefill(a, weight_b, scales, torch.float16)
    assert torch.equal(first, second)


@requires_musa
def test_prefill_rows_are_independent():
    """Each output row depends only on its own activation row: a wrong A tile shows up here."""
    weight_b, scales, dequant = _checkpoint(1536, 2048, seed=9, device="musa")
    a = torch.zeros((256, 1536), device="musa").half()
    a[200] = torch.randn(1536, device="musa").half()
    out = _run_prefill(a, weight_b, scales, torch.float16).float()
    assert out[:200].abs().max() == 0 and out[201:].abs().max() == 0
    _assert_parity(out[200:201].half(), (a[200:201].float() @ dequant.t()))


# Measured winner per Qwen2-1.5B cell on a 60-MP S5000: the wide tile unless the narrow one fits in
# the same number of waves.
@pytest.mark.parametrize(
    "m,k,n,tile_n",
    [
        (128, 1536, 2048, 128), (512, 1536, 2048, 256), (2048, 1536, 2048, 256),
        (128, 1536, 1536, 128), (512, 1536, 1536, 128), (2048, 1536, 1536, 256),
        (128, 1536, 17920, 256), (512, 1536, 17920, 256), (2048, 1536, 17920, 256),
        (128, 8960, 1536, 128), (512, 8960, 1536, 128), (2048, 8960, 1536, 256),
    ],
)
def test_prefill_tile_rule_matches_measured_winner(m, k, n, tile_n):
    from mate.jit.gemm.dense_w4a16_gemm import dense_w4a16_prefill_tile_for

    assert dense_w4a16_prefill_tile_for(m, n, k, 60)[1] == tile_n


def _zero_points(size_k: int, size_n: int, seed: int, device: str):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 16, (size_n, size_k // GROUP), generator=gen).half().to(device)


def _awq_dequant(weight_b, scales, zero_points):
    """fp32 ``s * (u - zp)`` from the packed operand, with ``u = q + 8``."""
    lo = (weight_b.view(torch.uint8) & 0xF).int()
    hi = (weight_b.view(torch.uint8) >> 4).int()
    q = torch.stack([lo, hi], dim=-1).reshape(weight_b.shape[0], -1)  # (N, K), even k low
    u = (q ^ 0x8).float()
    s = scales.float().repeat_interleave(GROUP, dim=1)
    zp = zero_points.float().repeat_interleave(GROUP, dim=1)
    return s * (u - zp)


@requires_musa
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("m", (65, 128, 129, 512, 2047))
@pytest.mark.parametrize("shape", QWEN2_1_5B + TAIL_SHAPES, ids=lambda s: s[0])
def test_prefill_zero_points_match_dequant_reference(shape, m, out_dtype):
    _, size_k, size_n = shape
    weight_b, scales, _ = _checkpoint(size_k, size_n, seed=size_k * 7 + size_n, device="musa")
    zp = _zero_points(size_k, size_n, seed=size_n, device="musa")
    gen = torch.Generator(device="cpu").manual_seed(m)
    a = (torch.randn((m, size_k), generator=gen) * 0.5 + 0.25).half().to("musa")  # nonzero row sums
    ref = a.float() @ _awq_dequant(weight_b, scales, zp).t()
    _assert_parity(_run_prefill(a, weight_b, scales, out_dtype, zero_points=zp), ref)


@requires_musa
@pytest.mark.parametrize("tile", [None, (128, 128, 128, 4)], ids=["n256", "n128"])
def test_prefill_zero_point_eight_is_bit_equal_to_symmetric(tile):
    weight_b, scales, _ = _checkpoint(1536, 2048, seed=11, device="musa")
    a = torch.randn((640, 1536), device="musa").half()
    sym = _run_prefill(a, weight_b, scales, torch.float16, tile)
    eight = torch.full_like(scales, 8.0)
    assert torch.equal(_run_prefill(a, weight_b, scales, torch.float16, tile, eight), sym)


@requires_musa
def test_prefill_zero_point_correction_tracks_each_row():
    """zp = 0 adds 8 * s * sum(a) per row: distinct rows must receive distinct corrections."""
    weight_b, scales, _ = _checkpoint(1536, 2048, seed=13, device="musa")
    a = torch.zeros((256, 1536), device="musa").half()
    a[:, :] = torch.arange(256, device="musa").half().unsqueeze(1) / 256
    zero = torch.zeros_like(scales)
    out = _run_prefill(a, weight_b, scales, torch.float16, zero_points=zero).float()
    sym = _run_prefill(a, weight_b, scales, torch.float16).float()
    s_sum = scales.float().sum(dim=1) * GROUP  # sum over k of s for each n
    expected = 8 * a.float()[:, :1] * s_sum.unsqueeze(0)
    torch.testing.assert_close(out - sym, expected, rtol=2e-2, atol=2e-2)


def test_aot_list_carries_every_prefill_variant():
    from mate.jit.gemm.dense_w4a16_gemm import (
        DENSE_W4A16_PREFILL_TILES,
        dense_w4a16_prefill_dispatch_name,
        gen_dense_w4a16_gemm_aot,
    )

    names = {spec.name for spec in gen_dense_w4a16_gemm_aot()}
    for out_dtype in (torch.float16, torch.bfloat16):
        for zero_points in (False, True):
            for tile in DENSE_W4A16_PREFILL_TILES:
                assert dense_w4a16_prefill_dispatch_name(out_dtype, tile, zero_points) in names


def test_prefill_row_sums_matches_reference_layout_and_values():
    """The zero-point row-sum operand must be (K//128, >= M) fp32, group-outer and M-contiguous.

    Round 911 replaced the two-step form (reduce to (M, G), then copy the transpose) with a single
    transposed reduction for ``M % 4 == 0``, because that operand is rebuilt on every prefill call and the
    second launch cost ~5.6 us of a ~55 us op. The layout is load-bearing: the kernel asserts
    ``row_sum_t.stride(1) == 1`` and ``stride(0) % 4 == 0``, so a value-only check would pass on a
    transposed buffer that the kernel then has to reject or silently misread. Both branches are covered,
    including ``M % 4 != 0``, which still takes the padded two-step path.
    """
    from mate.jit.gemm.dense_w4a16_gemm import prefill_row_sums

    device = "musa" if _HAVE_MUSA else "cpu"
    for m in (8, 32, 64, 12, 9, 1, 30):
        for k in (1536, 1792, 8960):
            a = torch.randn(m, k, dtype=torch.float16, device=device) * 0.1
            got = prefill_row_sums(a)
            groups = k // GROUP
            # reference in fp64 on the host, so the assertion is about the reduction and the layout
            ref = a.double().reshape(m, groups, GROUP).sum(dim=-1)  # (M, G)
            assert got.shape[0] == groups, "one row per K128 group"
            assert got.shape[1] >= m, "the operand must hold at least M columns"
            assert got.shape[1] % 4 == 0, "rows must be 16-byte aligned for the TME"
            assert got.stride(1) == 1, "M-contiguous is what the kernel asserts"
            assert got.stride(0) % 4 == 0, "stride(0) % 4 is a hard kernel precondition"
            assert got.dtype == torch.float32
            err = (got[:, :m].double() - ref.t()).abs().max().item()
            scale = ref.abs().max().item() or 1.0
            assert err / scale < 2e-6, (m, k, err / scale)
