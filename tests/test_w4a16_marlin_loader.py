# SPDX-License-Identifier: Apache-2.0
"""The loader-frame entry, act order, and the input handling of ``apply_gptq_marlin_linear``.

Host tests prove that ``prepare_w4a16_from_gptq_frame`` builds exactly the frame that upstream's repack
followed by ``prepare_marlin_w4a16`` builds, and that a frame means what the checkpoint means: a fp32
dequantization of the frame, applied with the frame's act-order gather, reproduces the checkpoint's
own linear. ``apply_gptq_marlin_linear``'s plumbing (leading dimensions, act-order gather, bias,
output dtype) is checked on the host by substituting a reference GEMM for the device op. The device
tests run the real kernel through the same entry points.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
marlin = pytest.importorskip("mate.marlin")
w4a16_jit = pytest.importorskip("mate.jit.gemm.dense_w4a16_gemm")

GROUP = 128
# Per-rank (K, N) of Qwen2-1.5B at TP=1: qkv, o, down, gate_up.
QWEN2_SHAPES = [(1536, 2048), (1536, 1536), (8960, 1536), (1536, 17920)]
SMALL_SHAPES = [(256, 64), (512, 256), (1536, 2048)]


def _musa_available() -> bool:
    try:
        import torch_musa  # noqa: F401
    except Exception:
        pass
    return hasattr(torch, "musa") and torch.musa.is_available()


needs_musa = pytest.mark.skipif(not _musa_available(), reason="needs a MUSA device")


def make_checkpoint(size_k, size_n, zero_points, act_order, seed=0):
    """A random GPTQ-layout checkpoint plus the dequantized weight it stands for.

    Returns ``(words, scales, zp, g_idx, w_true)``: ``words`` are ``(K/8, N)`` int32 with row
    ``8r + i`` in bits ``4i``, ``w_true[k, n] = scales[g_idx[k], n] * (u[k, n] - zp[g_idx[k], n])``
    with ``zp = 8`` when symmetric. With act order, ``g_idx`` assigns every group exactly ``GROUP``
    rows scattered over K, the way GPTQ's ``desc_act`` does.
    """
    rng = np.random.default_rng(seed)
    num_groups = size_k // GROUP
    codes = rng.integers(0, 16, (size_k, size_n)).astype(np.int64)
    words = marlin.layout.pack_rows(codes).astype(np.uint32).view(np.int32)
    scales = (rng.random((num_groups, size_n)) * 0.02 + 0.001).astype(np.float16)
    zp = rng.integers(0, 16, (num_groups, size_n)).astype(np.int32) if zero_points else None
    if act_order:
        order = rng.permutation(size_k)
        g_idx = np.empty(size_k, dtype=np.int32)
        g_idx[order] = np.arange(size_k) // GROUP
    else:
        g_idx = None
    group_of_k = (np.arange(size_k) // GROUP) if g_idx is None else g_idx
    zero = 8 if zp is None else zp[group_of_k]
    w_true = scales[group_of_k].astype(np.float64) * (codes - zero)
    return (
        torch.from_numpy(words),
        torch.from_numpy(scales),
        None if zp is None else torch.from_numpy(zp),
        None if g_idx is None else torch.from_numpy(g_idx),
        torch.from_numpy(w_true),
    )


def via_marlin_route(words, scales, zp, g_idx, size_k, size_n):
    """Upstream's sequence: repack, permute scales, pack zero points, then MATE's Marlin entry."""
    num_groups = size_k // GROUP
    if g_idx is None:
        sort_indices = None
        perm = torch.arange(size_k, dtype=torch.int32)
    else:
        sort_indices = torch.from_numpy(np.argsort(g_idx.numpy(), kind="stable"))
        perm = sort_indices
    repacked = torch.as_tensor(marlin.gptq_marlin_repack(words, perm, size_k, size_n, 4))
    permuted_scales = marlin.marlin_permute_scales(scales, size_k, size_n, GROUP)
    marlin_zp = None if zp is None else marlin.marlin_zero_points(zp, num_groups, size_n, 4)
    return marlin.prepare_marlin_w4a16(
        repacked,
        permuted_scales,
        marlin_zp,
        size_k=size_k,
        size_n=size_n,
        group_size=GROUP,
        g_idx=g_idx,
        g_idx_sort_indices=sort_indices,
    )


def frame_weight(frame, size_k):
    """The ``(K, N)`` fp32 weight a frame encodes, rows in the frame's stored (gathered) order."""
    size_n = frame.weight_b.shape[0]
    raw = frame.weight_b.view(torch.uint8).to(torch.int32)
    nibbles = torch.stack([raw & 0xF, raw >> 4], dim=-1).reshape(size_n, size_k)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).double()
    scale = frame.scale_b.double().repeat_interleave(GROUP, dim=1)
    weight = signed * scale
    if frame.zero_points_b is not None:
        weight = weight + (8 - frame.zero_points_b.double().repeat_interleave(GROUP, dim=1)) * scale
    return weight.t()


def _equal_or_both_none(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


# --------------------------------------------------------------------------------------------------
# Host: the loader entry is the Marlin route without the round trip
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("size_k,size_n", QWEN2_SHAPES)
@pytest.mark.parametrize("zero_points", [False, True])
def test_loader_frame_is_bit_identical_to_the_marlin_route(size_k, size_n, zero_points):
    words, scales, zp, g_idx, _ = make_checkpoint(size_k, size_n, zero_points, False, seed=size_k)
    expected = via_marlin_route(words, scales, zp, g_idx, size_k, size_n)
    actual = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP
    )
    for field in ("weight_b", "scale_b", "scale_b_padded", "zero_points_b", "correction_operand"):
        assert _equal_or_both_none(getattr(actual, field), getattr(expected, field)), field
    assert actual.input_perm is None and expected.input_perm is None
    for field in ("logical_n", "padded_n", "padded_k", "group_size", "has_zero_points"):
        assert getattr(actual, field) == getattr(expected, field), field


@pytest.mark.parametrize("size_k,size_n", [(512, 256), (1536, 2048)])
@pytest.mark.parametrize("zero_points", [False, True])
def test_loader_frame_act_order_is_bit_identical_to_the_marlin_route(size_k, size_n, zero_points):
    words, scales, zp, g_idx, _ = make_checkpoint(size_k, size_n, zero_points, True, seed=7)
    expected = via_marlin_route(words, scales, zp, g_idx, size_k, size_n)
    actual = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=g_idx
    )
    assert expected.input_perm is not None
    for field in ("weight_b", "scale_b", "scale_b_padded", "zero_points_b", "correction_operand",
                  "input_perm"):
        assert _equal_or_both_none(getattr(actual, field), getattr(expected, field)), field


@pytest.mark.parametrize("size_k,size_n", SMALL_SHAPES)
@pytest.mark.parametrize("zero_points", [False, True])
@pytest.mark.parametrize("act_order", [False, True])
def test_frame_reproduces_the_checkpoint_linear(size_k, size_n, zero_points, act_order):
    """The frame's meaning, independent of how it was built: gather, then the frame's own weight."""
    words, scales, zp, g_idx, w_true = make_checkpoint(size_k, size_n, zero_points, act_order, 3)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=g_idx
    )
    x = torch.randn(5, size_k, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    gathered = x if frame.input_perm is None else x.index_select(1, frame.input_perm)
    torch.testing.assert_close(gathered @ frame_weight(frame, size_k), x @ w_true, rtol=1e-9, atol=1e-9)


def test_identity_g_idx_is_not_act_order():
    size_k, size_n = 512, 256
    words, scales, zp, _, _ = make_checkpoint(size_k, size_n, False, False)
    identity = torch.arange(size_k, dtype=torch.int32) // GROUP
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=identity
    )
    assert frame.input_perm is None
    empty = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP,
        g_idx=torch.empty(0, dtype=torch.int32),
    )
    assert empty.input_perm is None


def test_act_order_refusals():
    size_k, size_n = 512, 256
    words, scales, _, g_idx, _ = make_checkpoint(size_k, size_n, False, True, seed=11)
    uneven = g_idx.clone()
    uneven[0] = (uneven[0] + 1) % (size_k // GROUP)  # one group gets GROUP+1 rows, another GROUP-1
    with pytest.raises(ValueError, match="contiguous run"):
        marlin.prepare_w4a16_from_gptq_frame(
            words, scales, None, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=uneven
        )
    # The Marlin entry cannot know which order the repack used, so it insists on being told.
    repacked = torch.as_tensor(
        marlin.gptq_marlin_repack(words, torch.arange(size_k, dtype=torch.int32), size_k, size_n, 4)
    )
    permuted = marlin.marlin_permute_scales(scales, size_k, size_n, GROUP)
    with pytest.raises(ValueError, match="g_idx_sort_indices"):
        marlin.prepare_marlin_w4a16(
            repacked, permuted, None, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=g_idx
        )
    with pytest.raises(ValueError, match="permutation"):
        marlin.prepare_marlin_w4a16(
            repacked, permuted, None, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=g_idx,
            g_idx_sort_indices=torch.zeros(size_k, dtype=torch.int64),
        )


def test_loader_frame_refusals():
    words, scales, _, _, _ = make_checkpoint(512, 256, False, False)
    with pytest.raises(ValueError, match="unpacked"):
        marlin.prepare_w4a16_from_gptq_frame(
            words, scales, torch.zeros((4, 256 // 8), dtype=torch.int32),
            size_k=512, size_n=256, group_size=GROUP,
        )
    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        marlin.prepare_w4a16_from_gptq_frame(
            words, scales, torch.full((4, 256), 16, dtype=torch.int32),
            size_k=512, size_n=256, group_size=GROUP,
        )
    with pytest.raises(ValueError):  # a transposed word tensor has the same element count
        marlin.prepare_w4a16_from_gptq_frame(
            words.t().contiguous(), scales, None, size_k=512, size_n=256, group_size=GROUP
        )
    with pytest.raises(ValueError, match="multiple of 256"):
        marlin.prepare_w4a16_from_gptq_frame(
            torch.zeros((4480 // 8, 1536), dtype=torch.int32),
            torch.zeros((4480 // GROUP, 1536), dtype=torch.float16),
            None, size_k=4480, size_n=1536, group_size=GROUP,
        )


@pytest.mark.parametrize("packed_dim", [0, 1])
def test_unpack_zero_points_both_orientations(packed_dim):
    num_groups, size_n = 12, 2048
    zp = torch.from_numpy(np.random.default_rng(5).integers(0, 16, (num_groups, size_n)).astype(np.int32))
    packed = marlin.quant.pack_cols(zp)  # (G, N/8): the GPTQ checkpoint orientation
    if packed_dim == 0:
        packed = packed.t().contiguous()  # (N/8, G): vLLM's AWQ-to-standard orientation
    assert torch.equal(marlin.unpack_zero_points(packed, num_groups, size_n, packed_dim), zp)
    with pytest.raises(ValueError, match="packed_dim"):
        marlin.unpack_zero_points(packed, num_groups, size_n, 1 - packed_dim)


def test_unpack_gptq_words_matches_the_numpy_layout():
    codes = np.random.default_rng(9).integers(0, 16, (256, 64)).astype(np.int64)
    words = marlin.layout.pack_rows(codes).astype(np.uint32).view(np.int32)
    unpacked = marlin.unpack_gptq_words(torch.from_numpy(words), 256, 64)
    assert np.array_equal(unpacked.numpy(), codes)


# --------------------------------------------------------------------------------------------------
# Host: apply's plumbing, with a reference GEMM standing in for the device op
# --------------------------------------------------------------------------------------------------
def _reference_op(input_a, weight_b, scale_b, out, zero_points=None, g_idx=None, num_mps=None,
                  scale_b_padded=None, correction_operand=None, scale_t=None):
    assert input_a.dim() == 2 and input_a.dtype == torch.float16 and out.dim() == 2
    assert g_idx is None, "apply must gather the activation itself and pass g_idx=None"
    assert scale_t is not None and scale_t.is_contiguous()
    from mate.jit.gemm.dense_w4a16_gemm import prepare_w4a16_prefill_operand

    assert torch.equal(scale_t, prepare_w4a16_prefill_operand(scale_b, zero_points))
    frame = marlin.MarlinW4A16KernelFrame(
        weight_b=weight_b, scale_b=scale_b, zero_points_b=zero_points, scale_b_padded=scale_b,
        correction_operand=None, logical_n=weight_b.shape[0], padded_n=weight_b.shape[0],
        padded_k=input_a.shape[1], group_size=GROUP, has_zero_points=zero_points is not None,
    )
    out.copy_((input_a.double() @ frame_weight(frame, input_a.shape[1])).to(out.dtype))


@pytest.fixture
def reference_op(monkeypatch):
    monkeypatch.setattr(w4a16_jit, "dense_w4a16_gemm_op", _reference_op)


@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("zero_points", [False, True])
def test_apply_flattens_gathers_and_restores(reference_op, act_order, zero_points):
    size_k, size_n = 512, 256
    words, scales, zp, g_idx, w_true = make_checkpoint(size_k, size_n, zero_points, act_order, 21)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=g_idx
    )
    x = (torch.randn(2, 3, size_k, generator=torch.Generator().manual_seed(2)) * 0.5).half()
    bias = torch.randn(size_n, generator=torch.Generator().manual_seed(4)).half()
    out = marlin.apply_gptq_marlin_linear(x, frame, bias=bias)
    assert out.shape == (2, 3, size_n) and out.dtype == torch.float16
    expected = (x.double() @ w_true + bias.double()).half()
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    as_bf16 = marlin.apply_gptq_marlin_linear(x, frame, out_dtype=torch.bfloat16)
    assert as_bf16.dtype == torch.bfloat16 and as_bf16.shape == (2, 3, size_n)


# --------------------------------------------------------------------------------------------------
# Device: the real kernel through the loader entry
# --------------------------------------------------------------------------------------------------
def _relative_mean_error(out, ref):
    return float((out.double() - ref).abs().mean() / ref.abs().mean())


@needs_musa
@pytest.mark.parametrize("size_k,size_n", QWEN2_SHAPES)
@pytest.mark.parametrize("zero_points", [False, True])
@pytest.mark.parametrize("size_m", [1, 8, 33, 128])
def test_device_loader_frame_matches_the_checkpoint(size_k, size_n, zero_points, size_m):
    words, scales, zp, g_idx, w_true = make_checkpoint(size_k, size_n, zero_points, False, size_n)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, device="musa"
    )
    x = (torch.randn(size_m, size_k, generator=torch.Generator().manual_seed(size_m)) * 0.5).half()
    out = marlin.apply_gptq_marlin_linear(x.to("musa"), frame).cpu()
    assert torch.isfinite(out).all()
    assert _relative_mean_error(out, x.double() @ w_true) < 0.01


@needs_musa
@pytest.mark.parametrize("zero_points", [False, True])
def test_device_act_order_and_leading_dims(zero_points):
    size_k, size_n = 1536, 2048
    words, scales, zp, g_idx, w_true = make_checkpoint(size_k, size_n, zero_points, True, seed=31)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, g_idx=g_idx,
        device="musa",
    )
    assert frame.input_perm is not None and frame.input_perm.device.type == "musa"
    x = (torch.randn(2, 5, size_k, generator=torch.Generator().manual_seed(3)) * 0.5).half()
    bias = torch.randn(size_n, generator=torch.Generator().manual_seed(6)).half()
    out = marlin.apply_gptq_marlin_linear(x.to("musa"), frame, bias=bias.to("musa")).cpu()
    assert out.shape == (2, 5, size_n)
    assert _relative_mean_error(out, x.double() @ w_true + bias.double()) < 0.01


@needs_musa
def test_device_bf16_output():
    size_k, size_n = 1536, 1536
    words, scales, zp, _, w_true = make_checkpoint(size_k, size_n, True, False, seed=41)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, device="musa"
    )
    x = (torch.randn(4, size_k, generator=torch.Generator().manual_seed(5)) * 0.5).half()
    out = marlin.apply_gptq_marlin_linear(x.to("musa"), frame, out_dtype=torch.bfloat16).cpu()
    assert out.dtype == torch.bfloat16
    assert _relative_mean_error(out, x.double() @ w_true) < 0.02


@needs_musa
@pytest.mark.parametrize("size_m", [1, 64, 200, 2048])  # GEMV, SQMMA tile, prefill with a tail, prefill
@pytest.mark.parametrize("zero_points", [False, True])
def test_device_graph_replay_matches_eager(size_m, zero_points):
    size_k, size_n = 1536, 2048
    words, scales, zp, _, _ = make_checkpoint(size_k, size_n, zero_points, False, seed=51)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, device="musa"
    )
    x = (torch.randn(size_m, size_k, generator=torch.Generator().manual_seed(7)) * 0.5).half().to("musa")
    eager = marlin.apply_gptq_marlin_linear(x, frame)  # also builds the JIT module outside capture
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = marlin.apply_gptq_marlin_linear(x, frame)
    graph.replay()
    torch.musa.synchronize()
    assert torch.equal(captured, eager)


@needs_musa
@pytest.mark.skipif(
    os.environ.get("TORCHDYNAMO_DISABLE") == "1", reason="dynamo is disabled in this environment"
)
def test_device_fullgraph_compile_matches_eager():
    size_k, size_n = 1536, 2048
    words, scales, zp, _, _ = make_checkpoint(size_k, size_n, True, False, seed=61)
    frame = marlin.prepare_w4a16_from_gptq_frame(
        words, scales, zp, size_k=size_k, size_n=size_n, group_size=GROUP, device="musa"
    )
    bias = torch.randn(size_n, generator=torch.Generator().manual_seed(8)).half().to("musa")

    def forward(x):
        return frame.unpad_output(marlin.apply_gptq_marlin_linear(x, frame, bias=bias))

    compiled = torch.compile(forward, fullgraph=True)
    for size_m in (1, 64, 200):
        x = (torch.randn(size_m, size_k, generator=torch.Generator().manual_seed(size_m))).half()
        x = x.to("musa")
        assert torch.equal(compiled(x), forward(x))


# --------------------------------------------------------------------------------------------------
# Device: the small-M GEMV path the op dispatches to, at the rows it serves
# --------------------------------------------------------------------------------------------------
def _raw_operands(size_k, size_n, zero_points, seed):
    gen = torch.Generator().manual_seed(seed)
    weight_b = torch.randint(-128, 128, (size_n, size_k // 2), generator=gen, dtype=torch.int8)
    scale_b = (torch.rand(size_n, size_k // GROUP, generator=gen) * 0.02 + 0.001).half()
    zp = torch.randint(0, 16, (size_n, size_k // GROUP), generator=gen).half() if zero_points else None
    return weight_b, scale_b, zp


def _raw_reference(a, weight_b, scale_b, zp, size_k):
    raw = weight_b.view(torch.uint8).to(torch.int32)
    nib = torch.stack([raw & 0xF, raw >> 4], dim=-1).reshape(weight_b.shape[0], size_k)
    signed = torch.where(nib >= 8, nib - 16, nib).double()
    scale = scale_b.double().repeat_interleave(GROUP, dim=1)
    w = signed * scale
    if zp is not None:
        w = w + (8 - zp.double().repeat_interleave(GROUP, dim=1)) * scale
    return a.double() @ w.t()


@needs_musa
@pytest.mark.parametrize("size_m", [1, 2, 3, 4])
@pytest.mark.parametrize("size_k,size_n", [(1536, 2048), (1536, 17920), (8960, 1536), (512, 1028)])
@pytest.mark.parametrize("zero_points", [False, True])
def test_device_small_m_matches_the_reference(size_m, size_k, size_n, zero_points):
    weight_b, scale_b, zp = _raw_operands(size_k, size_n, zero_points, seed=size_m * 1000 + size_n)
    a = (torch.randn(size_m, size_k, generator=torch.Generator().manual_seed(size_k)) * 0.5).half()
    out = torch.empty(size_m, size_n, dtype=torch.float16, device="musa")
    w4a16_jit.dense_w4a16_gemm_mutlass(
        a.to("musa"), weight_b.to("musa"), scale_b.to("musa"), out,
        zero_points=None if zp is None else zp.to("musa"),
    )
    ref = _raw_reference(a, weight_b, scale_b, zp, size_k)
    assert torch.isfinite(out).all()
    assert _relative_mean_error(out.cpu(), ref) < 0.01


def test_small_m_dispatch_rules():
    """The GEMV takes the rows it was measured to win and hands everything else to the SQMMA kernel."""
    pick = w4a16_jit.dense_w4a16_gemv_config_for
    assert pick(1, 2048, 1536) is not None and pick(4, 2048, 1536) is not None
    assert pick(5, 2048, 1536) is None and pick(8, 2048, 1536) is None
    assert pick(1, 17920, 1536) is not None and pick(2, 17920, 1536) is None
    assert pick(2, 17920, 1536, zero_points=True) is not None
    assert pick(2, 1536, 8960) is not None and pick(3, 1536, 8960) is None
    assert pick(3, 1536, 8960, zero_points=True) is None  # long K: the SQMMA kernel from 3 rows
    assert pick(1, 1536, 1000) is None  # K must be whole 128-wide scale groups
    config = pick(3, 2048, 1536)
    assert config.m_max == 4 and config.lanes_per_row * config.k_split * config.chunks_per_lane * 32 >= 1536
