# SPDX-License-Identifier: Apache-2.0
"""W4A16 (Marlin-class) dense GEMM correctness tests for MATE.

INSTALL TARGET (MATE repository): ``tests/test_w4a16_marlin.py``
    cp tests-port/tests/test_w4a16_marlin.py  <mate>/tests/test_w4a16_marlin.py
Companion file: ``tests/bench_w4a16_marlin.py`` (from ``tests-port/tests/bench_w4a16_marlin.py``).

Ported from upstream vLLM (pristine pin checkout ``/home/xiaodongye/ws/vllm-musa100015-vllm-verify``,
upstream vLLM at ``2cf0a6915c``) for ticket MUSA-100077:

    upstream                                                -> here
    ------------------------------------------------------------------------------------------------
    tests/kernels/quantization/test_marlin_gemm.py
      compute_max_diff()                                    -> compute_max_diff()
      DENSE_MARLIN_QUANT_TEST_CONFIGS / MNK_FACTORS          -> SHAPES x MS (real checkpoint lattice)
      marlin_generate_valid_test_cases()/is_invalid()        -> the explicit K%256 / group-128 gates
      test_marlin_gemm() (quantize -> op -> w_ref gate)      -> test_w4a16_symmetric_*
                                                               test_w4a16_asymmetric_*
      test_marlin_gemm_with_bias / _subset_input             -> NOT PORTED (no bias arm, no subset
                                                                input in MATE's op; see "Not covered")
    tests/kernels/quantization/test_marlin_tile_padding.py
      ODD_SHAPES / ALIGNED_SHAPES / _is_tile_aligned()       -> ODD_SHAPES / ALIGNED_SHAPES / ...
      marlin_padded_nk() minimality + shape assertions       -> test_w4a16_padding_arithmetic_*
      marlin_pad_qweight()/marlin_pad_scales() zero-fill     -> _pad_*_to_padded()
      test_gptq_marlin_padded_round_trip()                   -> test_w4a16_padded_round_trip_*
    vllm/model_executor/layers/quantization/utils/marlin_utils_test.py
      marlin_quantize()/awq_marlin_quantize() oracle         -> _quantize_symmetric_u4b8() /
                                                                _quantize_awq_u4() (+ w_ref returned)
      get_weight_perm()/marlin_permute_weights()             -> parity-checked against the layout oracle
                                                                (this file does not re-implement it)
      marlin_permute_scales()/marlin_zero_points()           -> _marlin_permute_scales() /
                                                                _marlin_permute_zero_points() ports
    tests/kernels/quantization/utils (+ quant_utils.quantize_weights, the reference quantizer)
      uint4b8 scale = max(|max|/7, |min|/8)                  -> _quantize_symmetric_u4b8()
      uint4    scale = (max-min).clamp(1e-5)/15, zp          -> _quantize_awq_u4()

What is under test (MATE, mid-draft as of 2026-10-13):
    ``mate.gemm.dense_w4a16_gemm`` / ``mate.jit.gemm.dense_w4a16_gemm.dense_w4a16_gemm_mutlass``
    Contract: ``input_a`` (M, K) fp16; ``weight_b`` (N, K // 2) int8, byte [n, k//2] carries K index
    2*k in the low nibble and 2*k+1 in the high nibble, each nibble storing the *biased* code
    ``unsigned_nibble = signed_value + kWeightNibbleBias`` with ``kWeightNibbleBias == 8``;
    ``scale_b`` (N, K // 128) fp16; ``zero_points`` (N, K // 128) optional; K % 256 == 0.
    The kernel dequantizes as ``w = scale * (signed_value + 8 - zero_point)`` with an implicit
    ``zero_point == 0`` when ``zero_points is None``, i.e. ``w = scale * (stored - zp_eff)`` with
    ``zp_eff = 8`` for the symmetric path (``IsZeroPoints == false``, where the zero-point tensor is
    not even TME-loaded: ``TmeTransactionBytesZeroB == 0``).

Hardware-verified premises (MUSA probe, 2026-10-13, on a leased mp_31 S5000). These were measured by
driving the existing ``mp31_masked_moe_gemm_mixed_dtype`` kernel with an fp8 identity over k and unit
scales, so ``out[0, m, n]`` is the value the hardware decodes for output column n at k = m:

    * the int4 B operand's 4-bit field is read as **two's complement**: code ``0`` decodes to 0, ``8``
      to -8 and ``15`` to -1, with **no implicit -8 and no +1024** anywhere in that operand. The
      tempting reading "a stored nibble decodes to exactly ``stored - 8``" is true **only at
      ``stored == 8``** -- the single code a probe naturally picks. It is false for the whole lower
      half, and no checker that only ever used code 8 could tell the difference. That is the trap this
      note exists to close; do not weaken these gates to ``scale * stored``.
    * the presentation MATE's frame requires follows from MATE's own contract, not from the atom's
      incidental behaviour. The packed-weight contract says each nibble "already holds the signed INT4
      value ``unsigned_nibble - nibble_bias``" and the public dequant is
      ``w = scale * (q + nibble_bias - zero_point)`` (``mate/gemm.py:1503-1507``,
      ``mate/jit/gemm/dense_w4a16_gemm.py:28-31,46-49``), while
      ``tests/test_dense_w4a16_gemm.py:42-47`` states that GPTQ uint4b8 "decodes as ``nibble - 8``
      with no further bias" and that a symmetric checkpoint's zero point equals the bias, so its rank-1
      correction vanishes exactly (and the ``sym`` instantiation never loads the zero-point tensor:
      ``TmeTransactionBytesZeroB == 0``). Requiring that declared decode for *every* code forces the
      presented code to be ``u ^ 0x8``, because ``s(u ^ 0x8) == u - 8`` for all 16 codes. Presenting
      upstream's ``uint4b8`` words verbatim instead costs a uniform ``8 * scale`` per element -- which
      no shape-based comparison detects, and which surfaces as exactly ``8 * scale * sum_k(a)`` per
      group in the R3 gate.
    * byte ``j`` of row ``n`` carries k = 2j in the low nibble and k = 2j + 1 in the high nibble, with
      consecutive k: a probe with ``low = j % 16`` read k = 16 from byte 8, which rules out NVIDIA's
      {r, r+1, r+8, r+9} grouping. So ``pack_mate_weight_b``'s assertion and the byte order above are
      measured facts, not documentation.
    * route B (consuming Marlin's ``[K // 16, 2N]`` word-tiled buffer) needs a **real load-time
      shuffle**, not a static address permutation: this atom consumes consecutive k while Marlin's
      buffer is word-tiled, *and* the codes need the ``u ^ 0x8`` presentation above. That corrects the
      earlier "static permutation" note in ``MateW4A16WeightLayout``.

Still unverified, and not implied by any gate here: the smem/atom internals (the register -> C
fragment map of ``MP31_16x16x32_F32F16S4F32``).

Gate table (gate | test | expected observable | owning rule/skill)
    byte parity (CPU)  | test_repack_byte_parity_*        | oracle == MATE repack, maxdiff 0 | kernel-dashboard byte-parity override
    order              | test_mate_weight_b_nibble_order_* | even k low / odd k high         | MUSA probe 2026-10-13 (measured)
    sign-bit frame     | test_wrapper_sign_bit_presentation_* | s(u ^ 0x8) == u - 8, verbatim fails below 8 | mate/gemm.py:1503-1507
    zero points raw    | test_wrapper_zero_points_are_raw_* | symmetric zp stays 8, folded 0 fails | mate/gemm.py:1512-1515
    repack round trip  | test_mate_weight_b_round_trip    | codes survive pack->unpack      | shape-indexing-walkthrough
    R3 negative ctrl   | test_r3_nibble_bias_*           | analytic 8*sum(s) error, non-zero| MUSA-100077 R3 trap
    quantization gate  | test_w4a16_symmetric_*          | relative mean error < 0.04      | upstream test_marlin_gemm
    zero-point gate    | test_w4a16_asymmetric_*         | relative mean error < 0.04      | upstream test_marlin_gemm
    padding gate (R5)  | test_w4a16_padded_round_trip_*  | assert_close rtol/atol 2e-2     | upstream test_marlin_tile_padding
    contract boundary  | test_w4a16_contract_*           | ValueError / NotImplementedError| dense_w4a16_gemm.py validation
    poison inputs      | test_w4a16_poison_*             | no crash, documented value      | write-kernel-test-plan

Scope gaps a reader must know about before promising coverage:

    * **K-side tile padding is unreachable under the group-128 contract**, so ``padded_k == size_k``
      whenever ``size_k % 128 == 0`` (proved in
      ``test_padding_cannot_grow_k_under_the_group_128_contract``; upstream only pads K for shapes the
      group does not divide, and its own test skips those). ``test_w4a16_padded_k_is_not_expressible_yet``
      is the honest encoding of the refusal, not a placeholder for coverage that exists.
    * **The op requires ``K % 256 == 0``**, so a TP-sharded ``down`` projection at TP=2
      (K = 8960 / 2 = **4480**, and 4480 % 256 = 128) is **refused**. Upstream serves that shape by
      padding K to a tile-aligned extent; MATE cannot yet. Not blocking for Qwen2-1.5B-AWQ at TP=1
      (1536 and 8960 are both multiples of 256), but it is a real scope limit for sharded serving, and
      it is the reason the padding gate asserts a ``ValueError`` there.
    * the byte-parity gates need **no** external artifacts: the oracle and the shipping repack are
      MATE's own ``mate.marlin.reference`` and ``mate.marlin.layout``, imported from the package by
      ``_load_mate_marlin`` below, so those gates run everywhere numpy does.

Not covered here (must not be over-read):
    * The Marlin ``[K // 16, 2 * N]`` int32 repacked layout. The draft op consumes MATE's native
      ``(N, K // 2)`` byte layout and documents the Marlin contract as a TODO in
      ``MateW4A16WeightLayout``; the byte-parity gate here pins the *current* contract so the swap is
      a reviewed change rather than a silent one.
    * ``act_order`` / ``g_idx`` permutation. The header asserts ``GidxKPermuteHook::IsIdentityOnly``
      and the host raises on a non-identity ``g_idx``; the ported test asserts that refusal.
    * bf16 activations, fp8/int8 activations, int8/uint8 weights, NVFP4/MXFP4, bias, subset input,
      ``use_atomic_add``/``use_fp32_reduce``. None of these arms exist in MATE's op.
    * Performance. That is ``tests/bench_w4a16_marlin.py``.

Two portability notes: this module imports torch lazily so the numpy-only gates run on a box without
torch (verified on the authoring host, which has numpy 1.26.0 and no torch), and the fp32 (not fp64)
reference is forced by MUSA having no device float64; fp32 accumulation is the documented fallback.

Installability: nothing here depends on the authoring workspace. The oracle and the repack live in the
package itself (``mate.marlin.reference`` / ``mate.marlin.layout``), so the parity gates need no
artifact directory and no environment variable, and they run wherever numpy does. There is no skip
path that could silently weaken a gate into a pass.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------------------------
# torch is optional: the numpy-only section below is a real gate on a box without torch.
# ---------------------------------------------------------------------------------------------
_TORCH_IMPORT_ERROR: Optional[BaseException] = None
try:  # pragma: no cover - the import outcome is environment-dependent
    import torch

    import torch_musa  # noqa: F401  (registers the MUSA backend before mate imports it)
    from mate.testing import supported_musa_compute_capability

    _HAVE_TORCH = True
except BaseException as _exc:  # noqa: BLE001 - report the real reason, do not hide it
    torch = None
    _HAVE_TORCH = False
    _TORCH_IMPORT_ERROR = _exc

    def supported_musa_compute_capability(*capabilities, **kwargs):  # type: ignore[misc]
        """No-op stand-in: without torch every test is skipped anyway (see ``requires_torch``)."""

        def _decorate(func):
            return func

        return _decorate


requires_torch = pytest.mark.skipif(
    not _HAVE_TORCH, reason=f"torch/torch_musa/mate.testing unavailable: {_TORCH_IMPORT_ERROR}"
)

# ---------------------------------------------------------------------------------------------
# The real-checkpoint lattice this ticket must serve (Qwen2-1.5B-AWQ): group size 128, K % 128 == 0.
# MATE's tile K is 256, so every shape here is also K % 256 == 0 - asserted, not assumed.
# ---------------------------------------------------------------------------------------------
GROUP_SIZE = 128
MATE_TILE_K = 256
NIBBLE_BIAS = 8

# K is the reduction dim (input features), N the output dim: for Qwen2-1.5B the projections are
# qkv (1536 -> 2048), o (1536 -> 1536), gate_up (1536 -> 17920) and down (8960 -> 1536).
CHECKPOINT_SHAPES = (
    (1536, 2048),   # qkv
    (1536, 1536),   # o_proj
    (1536, 17920),  # gate_up
    (8960, 1536),   # down_proj
)
# M from decode (1, a few tokens) through prefill. 2048 is the prefill point the ticket measures.
MS = (1, 8, 32, 128, 2048)

# Upstream tests/kernels/quantization/test_marlin_tile_padding.py:49-55, verbatim.
ODD_SHAPES = [
    (200, 288),  # N padded
    (256, 208),  # K padded
    (200, 208),  # both padded
    (4640, 512),  # Nemotron-Super-120B q_proj shard at TP=4
]
ALIGNED_SHAPES = [(64, 128), (128, 64), (256, 256), (4608, 4096)]


def _shape_id(index: int) -> str:
    """pytest id for a (K, N) pair, so a failure names the projection rather than an index."""
    names = ("qkv", "o", "gate_up", "down")
    size_k, size_n = CHECKPOINT_SHAPES[index]
    label = names[index] if index < len(names) else f"shape{index}"
    return f"{label}_k{size_k}_n{size_n}"


@dataclass(frozen=True)
class QuantizedWeight:
    """A quantized weight with the exact dequantized reference the GEMM must reproduce.

    ``stored`` is the biased unsigned code the packed operand carries (0..15). ``zero_point`` is
    ``None`` for the symmetric path, where the effective zero point is the nibble bias itself.
    """

    w_ref: np.ndarray          # (K, N) float32, the dequantized reference - the test's ground truth
    stored: np.ndarray         # (K, N) int, biased unsigned codes 0..15
    scale: np.ndarray          # (num_groups, N) float32
    zero_point: Optional[np.ndarray] = None  # (num_groups, N) int for the asymmetric path


def dequantized_reference(quantized: QuantizedWeight) -> np.ndarray:
    """``w = scale * (stored - zp_eff)`` with ``zp_eff == NIBBLE_BIAS`` on the symmetric path.

    This single formula is the whole contract: the kernel's signed atom computes ``stored - 8`` and
    adds the rank-1 correction ``(8 - zero_point) * scale * sum_k(a)``, so the symmetric path is
    ``scale * (stored - 8)`` and the asymmetric path is ``scale * (stored - zero_point)``.
    """
    if quantized.zero_point is None:
        effective = np.float32(NIBBLE_BIAS)
    else:
        # The zero point is per GROUP, exactly like `scale`, and `stored` is (K, N): it must be
        # repeated along K the same way, or the subtraction fails to broadcast ((2,64) vs (256,64)).
        effective = np.repeat(quantized.zero_point.astype(np.float32), GROUP_SIZE, axis=0)
    # (num_groups, N) -> (K, N) by repeating each group along its K run.
    scale_full = np.repeat(quantized.scale, GROUP_SIZE, axis=0)
    assert scale_full.shape == quantized.stored.shape, (scale_full.shape, quantized.stored.shape)
    return scale_full * (quantized.stored.astype(np.float32) - effective)


def synthetic_weight(
    stored: np.ndarray, scale: np.ndarray, zero_point: Optional[np.ndarray] = None
) -> QuantizedWeight:
    """A weight whose reference comes from the codes alone, with no quantizer in the loop.

    Used by the R3 and padding gates, where the point is the arithmetic of ``scale * (stored - zp)``
    rather than the accuracy of a quantizer.
    """
    probe = QuantizedWeight(
        w_ref=np.zeros(np.asarray(stored).shape, dtype=np.float32),
        stored=np.asarray(stored),
        scale=np.asarray(scale),
        zero_point=None if zero_point is None else np.asarray(zero_point),
    )
    return replace(probe, w_ref=dequantized_reference(probe))


# =============================================================================================
# Section A - numpy only. No torch, no MUSA device. These gates run anywhere.
# =============================================================================================
def _load_mate_marlin():
    """Import ``mate.marlin`` — the reference and the shipped repack are the package's own modules.

    The oracle used to be vendored next to this test, which made a third copy of the index maps and
    meant these gates guarded something other than the code that ships. They now import the package:
    ``LAYOUT_REF`` is ``mate.marlin.reference`` and ``REPACK_MUSA`` is ``mate.marlin.layout``.

    ``import mate.marlin`` is preferred, but MATE's top-level ``mate/__init__.py`` imports torch while
    both modules are numpy-only by design. On a host without torch the import would fail for a reason
    that has nothing to do with this code, so the fallback registers a synthetic ``mate`` parent and
    loads the package by path. Same modules, same gates, no torch needed.
    """
    import sys
    import types

    try:
        import mate.marlin as marlin_module

        return marlin_module
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    package = root / "mate" / "marlin"
    if not (package / "__init__.py").is_file():
        return None
    if "mate" not in sys.modules:
        parent = types.ModuleType("mate")
        parent.__path__ = [str(root / "mate")]
        sys.modules["mate"] = parent
    spec = importlib.util.spec_from_file_location(
        "mate.marlin", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None, f"cannot load {package}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["mate.marlin"] = module
    spec.loader.exec_module(module)
    return module


_MATE_MARLIN = _load_mate_marlin()
_ARTIFACTS = _MATE_MARLIN is not None  # kept as the "reference available" sentinel
LAYOUT_REF = _MATE_MARLIN.reference if _ARTIFACTS else None
REPACK_MUSA = _MATE_MARLIN.layout if _ARTIFACTS else None

needs_artifacts = pytest.mark.skipif(
    not _ARTIFACTS,
    reason="mate.marlin is not importable (the numpy-only reference lives in that package)",
)


def pack_mate_weight_b(stored_kN: np.ndarray) -> np.ndarray:
    """``(K, N)`` biased codes -> MATE's ``weight_b`` ``(N, K // 2)`` int8.

    Documented MATE contract (``MateW4A16WeightLayout``): byte ``[n, k // 2]`` holds K index ``2k``
    in the low nibble and ``2k + 1`` in the high nibble. Note this is a *different physical order*
    from GPTQ's ``gptq_pack`` (which packs 8 consecutive K into one int32); the logical codes are the
    same, which is exactly what the byte-parity gates below pin down.
    """
    codes = np.asarray(stored_kN, dtype=np.int64)
    size_k, size_n = codes.shape
    assert size_k % 2 == 0, f"K must be even, got {size_k}"
    assert codes.min() >= 0 and codes.max() <= 15, "stored codes must be unsigned 4-bit"
    low = codes[0::2, :] & 0xF      # even K -> low nibble
    high = codes[1::2, :] & 0xF     # odd  K -> high nibble
    packed = (low | (high << 4)).astype(np.uint8).T  # (N, K // 2)
    return np.ascontiguousarray(packed.view(np.int8))


def unpack_mate_weight_b(packed_nk: np.ndarray, size_k: int) -> np.ndarray:
    """Inverse of :func:`pack_mate_weight_b`: ``(N, K // 2)`` -> ``(K, N)`` codes."""
    raw = np.asarray(packed_nk).astype(np.uint8)
    size_n = raw.shape[0]
    assert raw.shape[1] * 2 == size_k
    low = (raw & 0xF).astype(np.int64)
    high = ((raw >> 4) & 0xF).astype(np.int64)
    codes = np.empty((size_k, size_n), dtype=np.int64)
    codes[0::2, :] = low.T
    codes[1::2, :] = high.T
    return codes


def gptq_layout_pack(codes_kN: np.ndarray) -> np.ndarray:
    """``(K, N)`` codes -> GPTQ ``[K // 8, N]`` int32 words - the shape upstream's C++ repack takes.

    This is upstream ``quant_utils.gptq_pack``, restricted to int4 and expressed on the already
    quantized codes so the packing step itself is what gets compared.
    """
    if LAYOUT_REF is not None:
        return LAYOUT_REF.pack_rows(np.asarray(codes_kN, dtype=np.uint32))
    codes = np.asarray(codes_kN, dtype=np.uint32)
    size_k, size_n = codes.shape
    assert size_k % 8 == 0
    packed = np.zeros((size_k // 8, size_n), dtype=np.uint32)
    for i in range(8):
        packed |= (codes[i::8, :] & 0xF) << (4 * i)
    return packed


def marlin_padded_nk(size_n: int, size_k: int, group_size: int = -1) -> tuple[int, int]:
    """Port of upstream ``marlin_utils.marlin_padded_nk`` (``marlin_utils.py:221-243``).

    Minimal ``(padded_n, padded_k)`` in the Marlin thread-tile family ``(n % 64 and k % 128)`` or
    ``(n % 128 and k % 64)``, keeping K divisible by the group size, chosen by ``(area, sum)``.
    """
    group = group_size if group_size > 0 else 1
    candidates = (
        (math.ceil(size_n / 64) * 64, math.ceil(size_k / math.lcm(128, group)) * math.lcm(128, group)),
        (math.ceil(size_n / 128) * 128, math.ceil(size_k / math.lcm(64, group)) * math.lcm(64, group)),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def marlin_repacked_nk(qweight_shape: tuple[int, int], num_bits: int = 4) -> tuple[int, int]:
    """Port of upstream ``marlin_utils.marlin_repacked_nk``: recover ``(N, K)`` from ``[K/16, 2N]``."""
    pack_factor = 32 // num_bits
    size_k = qweight_shape[0] * 16
    size_n = qweight_shape[1] * pack_factor // 16
    return size_n, size_k


def is_tile_aligned(size_n: int, size_k: int) -> bool:
    """Upstream ``test_marlin_tile_padding.py:58-61``, verbatim."""
    return (size_n % 64 == 0 and size_k % 128 == 0) or (size_n % 128 == 0 and size_k % 64 == 0)


def _marlin_permute_scales(
    s: np.ndarray, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> np.ndarray:
    """Port of upstream ``marlin_utils.marlin_permute_scales`` + ``get_scale_perms`` (numpy side).

    Grouped quantization takes the 64-wide permutation, channelwise the 8-wide "single" one.
    """
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scale_perm_single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    out = np.asarray(s)
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        out = out.reshape(-1, len(scale_perm))[:, scale_perm]
    else:
        out = out.reshape(-1, len(scale_perm_single))[:, scale_perm_single]
    return out.reshape(-1, size_n)


def _marlin_permute_zero_points(zp: np.ndarray, size_k: int, size_n: int, num_bits: int = 4) -> np.ndarray:
    """Port of the *permutation* half of upstream ``marlin_utils.marlin_zero_points``.

    Upstream then interleaves the column dim and packs to int32; that packing is compared separately
    against the oracle rather than re-implemented here.
    """
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7]) if num_bits == 4 else np.array([0, 2, 1, 3])
    out = np.asarray(zp).reshape(-1, len(scale_perm))[:, scale_perm]
    out = out.reshape(-1, len(interleave))[:, interleave].ravel()
    return out.reshape(-1, size_n)


def _rand_codes(size_k: int, size_n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 16, size=(size_k, size_n), dtype=np.int64)


# ----- A1: MATE's own packing is a bijection --------------------------------------------------
@pytest.mark.parametrize("size_k", [256, 512, 1024])
@pytest.mark.parametrize("size_n", [64, 200, 1536])
def test_mate_weight_b_round_trip(size_k, size_n):
    """Codes must survive pack(->)unpack(->) exactly: the nibble order is the whole contract."""
    codes = _rand_codes(size_k, size_n, seed=size_k + size_n)
    packed = pack_mate_weight_b(codes)
    assert packed.shape == (size_n, size_k // 2), packed.shape
    assert packed.dtype == np.int8
    assert np.array_equal(unpack_mate_weight_b(packed, size_k), codes)


def test_mate_weight_b_nibble_order_is_even_low_odd_high():
    """Pin the documented nibble order with a value that cannot alias: 0x1 and 0x2 in one byte."""
    codes = np.zeros((2, 1), dtype=np.int64)
    codes[0, 0] = 0x1  # even K -> low nibble
    codes[1, 0] = 0x2  # odd  K -> high nibble
    packed = pack_mate_weight_b(codes)
    assert int(np.uint8(packed[0, 0])) == 0x21


# ----- A2: repack byte parity (the load-bearing layout gate) ----------------------------------
@pytest.mark.parametrize(
    "size_k,size_n",
    [(16, 64), (32, 128), (16, 256), (128, 64), (256, 128), (1536, 64)],
)
@needs_artifacts
def test_repack_byte_parity_oracle_vs_repack_musa(size_k, size_n):
    """Two independent implementations of the Marlin GPTQ repack must agree byte for byte.

    ``layout_ref.cuda_repack_gptq`` is the scalar transcription of ``gptq_marlin_repack.cu``;
    ``repack_musa.repack_gptq`` is the per-tile vectorized implementation that ships. A byte of
    disagreement here is a contract violation, not a tolerance question.
    """
    codes = _rand_codes(size_k, size_n, seed=size_k * 7 + size_n)
    qweight = gptq_layout_pack(codes)
    reference = np.asarray(LAYOUT_REF.cuda_repack_gptq(qweight, size_k, size_n), dtype=np.int64)
    shipped = np.asarray(REPACK_MUSA.repack_gptq(qweight, size_k, size_n), dtype=np.int64)
    assert reference.shape == shipped.shape == (size_k // 16, 2 * size_n)
    assert int(np.abs(reference - shipped).max()) == 0

    # And the repacked buffer must still carry the same logical codes as the GPTQ words it came
    # from: this is what makes the Marlin-layout swap a physical-order change only.
    assert np.array_equal(LAYOUT_REF.unpack_rows(qweight, size_k, size_n), codes)


@pytest.mark.parametrize("size_k,size_n", [(16, 64), (32, 128), (128, 64)])
@needs_artifacts
def test_repack_byte_parity_round_trip(size_k, size_n):
    """``unpack_gptq(repack_gptq(x)) == x`` - catches a permutation that is a bijection on the tile
    but not an involution on the packed words."""
    codes = _rand_codes(size_k, size_n, seed=3 * size_k + size_n)
    qweight = gptq_layout_pack(codes)
    repacked = np.asarray(REPACK_MUSA.repack_gptq(qweight, size_k, size_n), dtype=np.int64)
    restored = np.asarray(REPACK_MUSA.unpack_gptq(repacked, size_k, size_n), dtype=np.int64)
    assert np.array_equal(restored, qweight)


@needs_artifacts
def test_repack_repacked_nk_recovers_shape():
    """``marlin_repacked_nk`` must invert the repack's shape change, including padding."""
    for size_n, size_k in ODD_SHAPES + ALIGNED_SHAPES:
        padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size=GROUP_SIZE)
        oracle = LAYOUT_REF.marlin_padded_nk(size_n, size_k, GROUP_SIZE)
        assert (padded_n, padded_k) == tuple(oracle), (size_n, size_k, padded_n, padded_k, oracle)
        assert marlin_repacked_nk((padded_k // 16, 2 * padded_n)) == (padded_n, padded_k)


# ----- A3/A4: scale and zero-point permutation parity -----------------------------------------
@needs_artifacts
def test_scale_permutation_parity():
    """The ported ``marlin_permute_scales`` must equal the verified transcription."""
    size_n, size_k = 64, 512
    num_groups = size_k // GROUP_SIZE
    s = np.arange(num_groups * size_n, dtype=np.float32).reshape(num_groups, size_n)
    mine = _marlin_permute_scales(s, size_k, size_n, GROUP_SIZE)
    theirs = np.asarray(LAYOUT_REF.marlin_permute_scales(s, size_k, size_n, GROUP_SIZE))
    assert mine.shape == theirs.shape
    assert np.array_equal(mine, theirs)


@needs_artifacts
def test_zero_point_permutation_parity():
    """The ported zero-point permutation must equal the oracle's, up to its int32 packing step.

    Upstream permutes, interleaves the column dim, then packs to int32 (``pack_cols``). The oracle's
    ``unpack_cols`` inverts only that packing, so it exposes exactly the array this port produces -
    which is why the comparison uses it instead of ``unpack_marlin_zero_points`` (that one inverts
    the permutation too, and therefore returns the *original* zero points, not the permuted ones).

    Note the scale permutation here is the 64-wide one: upstream never uses the "single" (8-wide)
    permutation for zero points, because zero points are applied on every MMA.
    """
    size_n, size_k = 64, 512
    num_groups = size_k // GROUP_SIZE
    zp = np.random.default_rng(5).integers(0, 16, size=(num_groups, size_n)).astype(np.int64)
    mine = _marlin_permute_zero_points(zp, size_k, size_n, 4)
    packed = np.asarray(LAYOUT_REF.marlin_zero_points(zp, size_k, size_n, 4))
    theirs = np.asarray(LAYOUT_REF.unpack_cols(packed, packed.shape[0], size_n))
    # Sanity: the oracle's own inverse restores the input, so `packed` really is the packing of the
    # array compared above and not something else that happens to have the same shape.
    assert np.array_equal(
        np.asarray(LAYOUT_REF.unpack_marlin_zero_points(packed, size_k, size_n, 4)), zp
    )
    assert mine.shape == theirs.shape == (num_groups, size_n), (mine.shape, theirs.shape)
    assert np.array_equal(mine, theirs)


# ----- A5: padding arithmetic ----------------------------------------------------------------
@pytest.mark.parametrize("shape", ODD_SHAPES + ALIGNED_SHAPES, ids=str)
@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128])
def test_padding_arithmetic_matches_upstream(shape, group_size):
    """Upstream's assertions, restated: monotone, tile-aligned, group-divisible, and *minimal*.

    The minimality sweep is the part that catches a "pad harder" regression: a cheaper candidate
    inside the padded rectangle means the padding is not minimal.
    """
    size_n, size_k = shape
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)
    assert padded_n >= size_n and padded_k >= size_k
    assert is_tile_aligned(padded_n, padded_k)
    if group_size > 0 and size_k % group_size == 0:
        assert padded_k % group_size == 0
    if is_tile_aligned(size_n, size_k) and (group_size <= 0 or size_k % group_size == 0):
        assert (padded_n, padded_k) == (size_n, size_k)
    area = padded_n * padded_k
    for cand_n in range(size_n, padded_n + 1):
        for cand_k in range(size_k, padded_k + 1):
            if not is_tile_aligned(cand_n, cand_k):
                continue
            if group_size > 0 and cand_k % group_size != 0:
                continue
            # Upstream flags area only, so a tie with a smaller sum is allowed (its tie-break is
            # a selection rule, not a minimality assertion).
            if cand_n * cand_k < area:
                pytest.fail(f"({cand_n}, {cand_k}) beats ({padded_n}, {padded_k}) for {shape}")


def test_padding_cannot_grow_k_under_the_group_128_contract():
    """K-side padding is unreachable when ``group_size`` divides K: prove it instead of asserting it.

    If ``size_k % 128 == 0`` then both candidate families round K up to ``size_k`` itself, so
    ``padded_k == size_k``. Upstream's ``(256, 208)`` case only pads K because group 128 does not
    divide 208 and the test skips it (``test_marlin_tile_padding.py:315-316``). The consequence for
    MATE is recorded in the ticket: the draft op requires ``K % 256 == 0``, so TP-sharded shapes
    whose padded K lands on 320/384 (channels-wise or group-64) are not expressible yet.
    """
    for size_k in (256, 512, 1536, 8960):
        for size_n in (200, 256, 1536):
            padded_n, padded_k = marlin_padded_nk(size_n, size_k, GROUP_SIZE)
            assert padded_k == size_k, (size_n, size_k, padded_n, padded_k)
    # The same shapes *with* a group that does not divide K do pad K, which is why the constraint
    # above is about the group contract and not about the padding helper.
    pad_n, pad_k = marlin_padded_nk(256, 208, -1)
    assert (pad_n, pad_k) == (256, 256) and pad_k > 208


# ----- A6: the R3 nibble-bias regression, computed analytically -------------------------------
@pytest.mark.parametrize("size_k", [256, 512])
def test_r3_nibble_bias_error_is_exactly_eight_times_scale(size_k):
    """R3: dropping the fused ``-8`` must be wrong by exactly ``8 * scale`` per group.

    The trap: MATE stores the *biased* code ``stored = signed + 8`` (GPTQ's ``uint4b8``), so a plain
    int4 -> fp16 dequant that reads the nibble as the weight is wrong by ``8 * scale`` on every
    element of the group, not by a rounding-sized amount. With a constant code and constant scales
    the error is analytic, so this gate runs without a device and cannot be satisfied accidentally.

    Numbers for the unit-scale case: ``stored = 0`` (signed -8) and ``scale = 1`` give
    ``w_naive = 0`` where the reference is ``w_ref = -8``, so the per-element error is exactly 8 and
    the output error is ``8 * sum_k(a)`` - a constant per group, which is what makes the failure mode
    recognisable in a diff instead of looking like noise.
    """
    size_n, group_size, nibble_bias = 64, GROUP_SIZE, NIBBLE_BIAS
    num_groups = size_k // group_size
    scale = np.full((num_groups, size_n), 1.0, dtype=np.float32)
    stored = np.zeros((size_k, size_n), dtype=np.int64)  # signed value -8, the symmetric zero

    quantized = synthetic_weight(stored, scale)
    w_ref = quantized.w_ref
    w_naive = np.repeat(scale, group_size, axis=0) * stored.astype(np.float32)

    per_element_error = w_naive - w_ref
    assert np.allclose(per_element_error, nibble_bias * 1.0), "naive dequant is off by exactly 8"

    # Per group, the error a GEMM accumulates is 8 * scale * sum_k(a) - the rank-1 form, which is
    # the term the kernel applies as `(nibble_bias - zero_point) * scale * reduce_k(a)`.
    a = np.random.default_rng(11).normal(size=(4, size_k)).astype(np.float32)
    group_sums = a.reshape(4, num_groups, group_size).sum(axis=2)  # (4, num_groups)
    out_ref = a @ w_ref
    out_naive = a @ w_naive
    analytic = np.einsum("mg,gn->mn", group_sums, (nibble_bias * scale))
    assert np.allclose(out_naive - out_ref, analytic, rtol=1e-5, atol=1e-4)
    # The failure must be visible: a silent-error trap whose error is zero is not a trap.
    assert float(np.abs(analytic).max()) > 0
    # And it must scale with the group's scale, which is what the kernel's correction stream carries.
    assert np.allclose(analytic, nibble_bias * np.einsum("mg,gn->mn", group_sums, scale), rtol=1e-5)


# =============================================================================================
# Section B - MUSA. Quantization oracle, the op, and the gates that need a device.
# =============================================================================================
def _to_numpy(tensor):
    return tensor.detach().to("cpu").numpy()


def _quantize_symmetric_u4b8(weight: "torch.Tensor") -> QuantizedWeight:
    """GPTQ ``uint4b8``: upstream ``quant_utils.quantize_weights`` (unbiased branch) + ``bias = 8``.

    ``scale = max(|max| / 7, |min| / -8)`` per group, ``q = clamp(round(w / scale), -8, 7)``, and
    the stored code is ``q + 8``. The reference is ``q * scale``, computed before the bias is added
    (``quant_utils.py:779`` runs ahead of ``:782``), which is what makes it exact.
    """
    size_k, size_n = weight.shape
    grouped = weight.reshape(-1, GROUP_SIZE, size_n).permute(1, 0, 2).reshape(GROUP_SIZE, -1)
    max_val = grouped.max(dim=0, keepdim=True).values
    min_val = grouped.min(dim=0, keepdim=True).values
    scale = torch.maximum((max_val / 7.0).abs(), (min_val / -8.0).abs())
    # An all-zero group has scale 0 and would divide by zero; upstream guards this with an inf
    # substitution. Setting the scale to 1 for that group reproduces the same exact reference
    # (all-zero weights dequantize to zero whatever the code is) without a NaN.
    zero_group = scale == 0
    if bool(zero_group.any()):
        scale = torch.where(zero_group, torch.ones_like(scale), scale)
    q_signed = torch.clamp(torch.round(grouped / scale), -8, 7)
    w_ref = q_signed * scale

    def ungroup(tensor):
        return tensor.reshape(GROUP_SIZE, -1, size_n).permute(1, 0, 2).reshape(size_k, size_n)

    return QuantizedWeight(
        w_ref=_to_numpy(ungroup(w_ref).float()),
        stored=_to_numpy(ungroup(q_signed + NIBBLE_BIAS)),
        scale=_to_numpy(scale.reshape(-1, size_n).float()),
        zero_point=None,
    )


def _quantize_awq_u4(weight: "torch.Tensor") -> QuantizedWeight:
    """AWQ ``uint4`` asymmetric: upstream ``quant_utils.quantize_weights(zero_points=True)``.

    ``scale = (max - min).clamp(min=1e-5) / 15``, ``zp = round(|min / scale|).clamp(0, 15)``,
    ``q = clamp(round(w / scale) + zp, 0, 15)`` and the reference is ``(q - zp) * scale``. The clamp
    happens *after* adding ``zp`` (``quant_utils.py:769-770``), which matters for outliers.
    """
    size_k, size_n = weight.shape
    grouped = weight.reshape(-1, GROUP_SIZE, size_n).permute(1, 0, 2).reshape(GROUP_SIZE, -1)
    max_val = grouped.max(dim=0, keepdim=True).values
    min_val = grouped.min(dim=0, keepdim=True).values
    scale = (max_val - min_val).clamp(min=1e-5) / 15.0
    zero_point = torch.round(torch.abs(min_val / scale)).clamp(0, 15).int()
    q = torch.clamp(torch.round(grouped / scale).int() + zero_point, 0, 15)
    w_ref = (q - zero_point) * scale

    def ungroup(tensor):
        return tensor.reshape(GROUP_SIZE, -1, size_n).permute(1, 0, 2).reshape(size_k, size_n)

    return QuantizedWeight(
        w_ref=_to_numpy(ungroup(w_ref).float()),
        stored=_to_numpy(ungroup(q)),
        scale=_to_numpy(scale.reshape(-1, size_n).float()),
        zero_point=_to_numpy(zero_point.reshape(-1, size_n)),
    )


def _mate_op():
    """The MATE entry point under test, or skip with the reason it is missing."""
    import mate  # noqa: PLC0415 - imported here so the numpy section stays importable

    gemm_module = getattr(mate, "gemm", None)
    kernel = getattr(gemm_module, "dense_w4a16_gemm", None) if gemm_module is not None else None
    if kernel is None:
        kernel = getattr(mate, "dense_w4a16_gemm", None)
    if kernel is None:
        pytest.skip("mate.gemm.dense_w4a16_gemm is not present in this MATE revision")
    return kernel


def _mate_kernel_module():
    """The validating implementation, for contract tests that must not launch a kernel."""
    return importlib.import_module("mate.jit.gemm.dense_w4a16_gemm")


def _pad_codes_and_scales(quantized: QuantizedWeight, size_n: int, size_k: int, padded_n: int, padded_k: int):
    """Pad to the Marlin extents the way upstream ``marlin_pad_qweight``/``marlin_pad_scales`` do.

    The padded region gets *zero scales*, which is the whole point: for symmetric ``uint4b8`` the
    stored code 0 decodes to ``-8``, not to 0.0, so a zero-padded weight only contributes nothing
    because its scale is zero. Upstream says the same in ``test_gptq_marlin_padded_round_trip``:
    "this exercises the zero-padded-scales cancellation, not just zero weights".
    """
    stored = np.zeros((padded_k, padded_n), dtype=quantized.stored.dtype)
    stored[:size_k, :size_n] = quantized.stored
    num_groups = padded_k // GROUP_SIZE
    scale = np.zeros((num_groups, padded_n), dtype=quantized.scale.dtype)
    scale[: size_k // GROUP_SIZE, :size_n] = quantized.scale
    zero_point = None
    if quantized.zero_point is not None:
        zero_point = np.zeros((num_groups, padded_n), dtype=quantized.zero_point.dtype)
        zero_point[: size_k // GROUP_SIZE, :size_n] = quantized.zero_point
    return synthetic_weight(stored, scale, zero_point)


def _run_op(
    quantized: QuantizedWeight,
    size_m: int,
    size_k: int,
    size_n: int,
    device: str = "musa",
    out_dtype=None,
    seed: int = 0,
    input_a=None,
    explicit_zero_points: bool = True,
):
    """Build MATE's operands, run the op, and return ``(out_fp32, a_fp32)`` for comparison."""
    if out_dtype is None:
        out_dtype = torch.float16
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if input_a is None:
        input_a = torch.randn(size_m, size_k, generator=generator, dtype=torch.float16).to(device)

    # ``pack_mate_weight_b`` is the *verbatim* MATE-frame packer, and the operand frame MATE declares
    # presents each nibble sign-flipped (``mate/marlin/linear.py:95-118``): the declared contract reads a
    # presented nibble ``v`` as ``v - 8``, the atom actually reads two's complement (measured on
    # hardware -- code 0 reads 0 and code 8 reads -8), and upstream's ``uint4b8`` codes therefore need
    # their sign bit flipped before presentation. Feeding them verbatim is a uniform ``+8 * scale``
    # error per element, invisible to every shape-based comparison. :1377 pins this equality for the
    # wrapper path (``frame.weight_b == packer(codes) ^ 0x88``); the op-level path presents the same
    # frame, so it flips the same bit here.
    weight_b = torch.from_numpy(pack_mate_weight_b(quantized.stored)).to(device)
    weight_b = weight_b ^ 0x88
    scale_b = torch.from_numpy(quantized.scale.T.astype(np.float16)).contiguous().to(device)
    zero_points = None
    if quantized.zero_point is not None and explicit_zero_points:
        zero_points = (
            torch.from_numpy(quantized.zero_point.T.astype(np.float16)).contiguous().to(device)
        )
    if quantized.zero_point is not None and not explicit_zero_points:
        # Symmetric call on an asymmetric weight: express the offset with zero_points = NIBBLE_BIAS,
        # which makes the folded correction exactly zero.
        zero_points = torch.full_like(scale_b, float(NIBBLE_BIAS))

    out = torch.empty((size_m, size_n), dtype=out_dtype, device=device)
    _mate_op()(input_a, weight_b, scale_b, out, zero_points=zero_points)
    return out.float(), input_a.float()


def compute_max_diff(output, output_ref):
    """Upstream ``test_marlin_gemm.py:136-139``: relative mean absolute error.

    Upstream's expression divides by the reference magnitude, which is 0/0 = NaN when the
    reference is identically zero -- and that is exactly the all-zero-weights poison case, whose
    whole purpose is to catch a kernel that "returns zeros fast" (see
    ``test_w4a16_poison_weights``). A NaN there would hide the very false positive the case
    exists to find, so the degenerate branch uses the absolute mean error instead: it is a
    stronger check, because it requires the output to be zero, not merely to be relative to zero.
    """
    abs_err = torch.mean(torch.abs(output - output_ref))
    ref_mag = torch.mean(torch.abs(output_ref))
    if float(ref_mag) == 0.0:
        return float(abs_err)
    return float(abs_err / ref_mag)


MAX_DIFF_TOLERANCE = 0.04  # upstream test_marlin_gemm.py:524


# ----- B1: symmetric uint4b8 (the first gate per the ticket) ----------------------------------
@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize("size_m", MS)
@pytest.mark.parametrize("shape_index", range(len(CHECKPOINT_SHAPES)), ids=_shape_id)
def test_w4a16_symmetric_uint4b8_matches_dequant_reference(shape_index, size_m):
    """GPTQ ``uint4b8``: the op must reproduce the dequantized fp32 reference within 0.04.

    The reference is fp32, not upstream's fp16 matmul, because MUSA has no device float64; both the
    code and the scale are identical to the op's inputs, so the residual is fp16 A-rounding plus
    output rounding, far below the quantization error the 0.04 gate is sized for.
    """
    size_k, size_n = CHECKPOINT_SHAPES[shape_index]
    assert size_k % MATE_TILE_K == 0, f"{size_k} is not a multiple of MATE's tile K"
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_symmetric_u4b8(weight)
    out, input_a = _run_op(quantized, size_m, size_k, size_n, seed=size_k + size_m)
    out_ref = input_a @ torch.from_numpy(quantized.w_ref).to(input_a.device)
    max_diff = compute_max_diff(out, out_ref)
    print(f"[w4a16 sym] K={size_k} N={size_n} M={size_m} max_diff={max_diff:.5f}")
    assert max_diff < MAX_DIFF_TOLERANCE, f"max_diff={max_diff} for K={size_k} N={size_n} M={size_m}"


@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize("size_m", [1, 128])
def test_w4a16_symmetric_with_bf16_output(size_m):
    """Out dtype bf16 is a separate instantiation (``_bf16_..._sym``); A stays fp16."""
    size_k, size_n = 1536, 1536
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_symmetric_u4b8(weight)
    out, input_a = _run_op(
        quantized, size_m, size_k, size_n, out_dtype=torch.bfloat16, seed=7
    )
    out_ref = input_a @ torch.from_numpy(quantized.w_ref).to(input_a.device)
    assert compute_max_diff(out, out_ref) < MAX_DIFF_TOLERANCE


@supported_musa_compute_capability([31])
@requires_torch
def test_w4a16_zero_points_equal_to_the_bias_matches_the_symmetric_path():
    """``zero_points = nibble_bias`` folds to a zero correction, so it must equal the sym path.

    This also documents the host's ``else scale_b_padded`` filler: on the symmetric instantiation
    ``IsZeroPoints`` is false and ``TmeTransactionBytesZeroB == 0``, i.e. the tensor is never loaded,
    so the argument is inert rather than meaningful.
    """
    size_m, size_k, size_n = 8, 512, 256
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_symmetric_u4b8(weight)
    out_sym, input_a = _run_op(quantized, size_m, size_k, size_n, seed=13)
    out_zp8, _ = _run_op(quantized, size_m, size_k, size_n, seed=13, explicit_zero_points=False)
    assert torch.allclose(out_sym, out_zp8, rtol=1e-3, atol=1e-3), (
        (out_sym - out_zp8).abs().max().item(),
        (out_sym - out_zp8).abs().mean().item(),
    )


# ----- B2: asymmetric AWQ with real per-group zero points ------------------------------------
@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize("size_m", MS)
@pytest.mark.parametrize("shape_index", range(len(CHECKPOINT_SHAPES)), ids=_shape_id)
def test_w4a16_asymmetric_awq_matches_dequant_reference(shape_index, size_m):
    """AWQ: the per-group zero point must be applied through the rank-1 correction, exactly.

    The reference is ``scale * (stored - zero_point)``, which is what the kernel's
    ``scale * signed + (nibble_bias - zero_point) * scale * sum_k(a)`` reduces to.
    """
    size_k, size_n = CHECKPOINT_SHAPES[shape_index]
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_awq_u4(weight)
    assert quantized.zero_point is not None and quantized.zero_point.size > 0
    out, input_a = _run_op(quantized, size_m, size_k, size_n, seed=size_k * 3 + size_m)
    out_ref = input_a @ torch.from_numpy(quantized.w_ref).to(input_a.device)
    max_diff = compute_max_diff(out, out_ref)
    print(f"[w4a16 awq] K={size_k} N={size_n} M={size_m} max_diff={max_diff:.5f}")
    assert max_diff < MAX_DIFF_TOLERANCE, f"max_diff={max_diff} for K={size_k} N={size_n} M={size_m}"


@supported_musa_compute_capability([31])
@requires_torch
def test_w4a16_zero_points_change_the_result():
    """Negative control for the correction stream: a wrong zero point must *not* pass the gate.

    With ``zero_points = 0`` instead of the real ones, the kernel adds ``8 * scale * sum_k(a)`` and
    the error is a per-group constant - so a test that passes both ways is not testing zero points.
    """
    size_m, size_k, size_n = 8, 256, 64
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_awq_u4(weight)
    out_ok, input_a = _run_op(quantized, size_m, size_k, size_n, seed=5)
    out_ref = input_a @ torch.from_numpy(quantized.w_ref).to(input_a.device)
    assert compute_max_diff(out_ok, out_ref) < MAX_DIFF_TOLERANCE

    wrong = synthetic_weight(
        quantized.stored, quantized.scale, np.zeros_like(quantized.zero_point)
    )
    out_wrong, _ = _run_op(wrong, size_m, size_k, size_n, seed=5)
    assert compute_max_diff(out_wrong, out_ref) > MAX_DIFF_TOLERANCE, (
        "a zeroed zero-point tensor must fail the gate; otherwise the correction stream is inert"
    )


# ----- B3: the R3 regression on the device ---------------------------------------------------
@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize("size_k", [256, 512])
def test_w4a16_r3_device_constant_nibble_unit_scales(size_k):
    """The device half of R3: constant nibble 0 and unit scales must give exactly ``-8 * sum_k(a)``.

    A refactor that drops the fused ``-8`` produces ``0`` here instead, and the diff is a *constant*
    ``8 * sum_k(a)`` per group - the recognisable signature this gate exists to catch.
    """
    size_m, size_n = 4, 64
    num_groups = size_k // GROUP_SIZE
    quantized = synthetic_weight(
        np.zeros((size_k, size_n), dtype=np.int64),
        np.ones((num_groups, size_n), dtype=np.float32),
    )
    out, input_a = _run_op(quantized, size_m, size_k, size_n, seed=17)
    out_ref = input_a @ torch.from_numpy(dequantized_reference(quantized)).to(input_a.device)
    assert torch.allclose(out, out_ref, rtol=2e-2, atol=2e-2), (out - out_ref).abs().max().item()
    # The naive reading (nibble == weight) is wrong by 8 * sum(a) per group and must be rejected.
    naive = input_a @ torch.zeros(size_k, size_n, device=input_a.device)
    assert compute_max_diff(out, naive) > MAX_DIFF_TOLERANCE


# ----- B4: tile padding (risk R5) ------------------------------------------------------------
@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize("size_n,size_k", ODD_SHAPES, ids=str)
def test_w4a16_padded_round_trip_n_padding(size_n, size_k):
    """N-side padding: pad, run, slice - the padded columns must contribute nothing.

    Only shapes whose ``padded_k`` is a multiple of MATE's tile K can run; the rest are the
    documented constraint exercised by :func:`test_w4a16_padded_k_is_not_expressible_yet`.
    """
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, GROUP_SIZE)
    if padded_k != size_k:
        pytest.skip(f"K-side padding ({size_k} -> {padded_k}) is not expressible: see the A5 gate")
    if size_k % GROUP_SIZE or size_k % MATE_TILE_K:
        pytest.skip(f"group {GROUP_SIZE} and tile K {MATE_TILE_K} both have to divide K={size_k}")
    # The reference is rank-local: only the true size_k x size_n weight is meaningful, and the
    # padded columns/rows carry zero scales so they cannot leak into the sliced output.
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_symmetric_u4b8(weight)
    padded = _pad_codes_and_scales(quantized, size_n, size_k, padded_n, padded_k)
    size_m = 8
    out, input_a = _run_op(padded, size_m, padded_k, padded_n, seed=23)
    out = out[:, :size_n]
    out_ref = input_a @ torch.from_numpy(quantized.w_ref).to(input_a.device)
    assert out.shape == (size_m, size_n)
    torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)


@supported_musa_compute_capability([31])
@requires_torch
def test_w4a16_padded_scales_are_what_make_padding_invisible():
    """The padding mechanism itself: zero scales, not zero weights. Stated as an assertion.

    For symmetric ``uint4b8`` the stored code 0 decodes to ``-8``, so a naive zero-padded operand
    with non-zero padded scales would add a large constant instead of nothing.
    """
    size_k, size_n, padded_n = 512, 200, 256
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_symmetric_u4b8(weight)
    padded = _pad_codes_and_scales(quantized, size_n, size_k, padded_n, size_k)
    assert np.all(padded.scale[:, size_n:] == 0), "padded columns must carry zero scales"
    # The padded codes themselves are zero, i.e. signed -8 - showing that the scale is load-bearing.
    assert np.all(dequantized_reference(padded)[:, size_n:] == 0), "zero scale must suppress the pad"
    # Negative control: keep the same zero-padded codes but leak a real scale into the pad. The pad
    # then contributes -8 * scale instead of nothing, which is what proves the zero scale - not the
    # zero code - is what makes tile padding invisible for symmetric int4.
    leaked_scale = padded.scale.copy()
    leaked_scale[:, size_n:] = padded.scale[:, size_n - 1 : size_n]
    leaked = dequantized_reference(synthetic_weight(padded.stored, leaked_scale))
    assert np.any(leaked[:, size_n:] != 0), (
        "a non-zero scale over the zero-padded codes must leak; otherwise this control is decoration"
    )


@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize("size_n,size_k", [(200, 288), (256, 208), (200, 208)], ids=str)
def test_w4a16_padded_k_is_not_expressible_yet(size_n, size_k):
    """Documented gap, asserted instead of skipped: MATE's op refuses a non-multiple-of-256 K.

    These are upstream's K-padded shapes. Under the group-128 contract they cannot be padded to a
    multiple of 256 (an odd shape is skipped upstream too), and with no group contract the padded K
    lands on 320/384. Either way MATE's draft op raises, and the concrete serving casualty is a
    TP-sharded ``down`` projection: at TP=2 the rank-local K is 8960 / 2 = 4480, and 4480 % 256 = 128,
    so the shape is refused even though upstream serves it by padding K to a tile-aligned extent.
    The kernel needs a smaller K tile (or a relaxed tile-K check) before that works; when it lands this
    test is expected to be *replaced* by a real round trip, not deleted.
    """
    module = pytest.importorskip("mate.jit.gemm.dense_w4a16_gemm")
    size_m = 8
    input_a = torch.randn(size_m, size_k, dtype=torch.float16, device="musa")
    weight_b = torch.zeros(size_n, size_k // 2, dtype=torch.int8, device="musa")
    scale_b = torch.ones(size_n, size_k // GROUP_SIZE, dtype=torch.float16, device="musa")
    out = torch.empty(size_m, size_n, dtype=torch.float16, device="musa")
    if size_k % MATE_TILE_K == 0:
        pytest.skip(f"K={size_k} is a multiple of the tile K; the round trip covers it instead")
    with pytest.raises(ValueError, match="multiple of the tile K"):
        module.dense_w4a16_gemm_mutlass(input_a, weight_b, scale_b, out)


# ----- B5: contract boundaries ---------------------------------------------------------------
@requires_torch
def test_w4a16_contract_rejections():
    """Every validation branch in ``dense_w4a16_gemm_mutlass``, asserted where it is reachable.

    These are argument-shape checks that run before any launch, so they are CPU-runnable and do not
    need mp31. Bytes matter more than arithmetic here: a silently accepted wrong shape is the failure
    mode, so each case asserts the specific error text.
    """
    module = pytest.importorskip("mate.jit.gemm.dense_w4a16_gemm")
    size_m, size_k, size_n = 4, 512, 128
    device = "cpu"
    input_a = torch.randn(size_m, size_k, dtype=torch.float16, device=device)
    weight_b = torch.zeros(size_n, size_k // 2, dtype=torch.int8, device=device)
    scale_b = torch.ones(size_n, size_k // GROUP_SIZE, dtype=torch.float16, device=device)
    out = torch.empty(size_m, size_n, dtype=torch.float16, device=device)

    # K must be a multiple of the tile K (boundary grid around 256).
    for bad_k in (128, 384):
        with pytest.raises(ValueError, match="multiple of the tile K"):
            module.dense_w4a16_gemm_mutlass(
                torch.zeros(size_m, bad_k, dtype=torch.float16),
                torch.zeros(size_n, bad_k // 2, dtype=torch.int8),
                torch.ones(size_n, bad_k // GROUP_SIZE, dtype=torch.float16),
                torch.empty(size_m, size_n, dtype=torch.float16),
            )

    # A must be fp16 (bf16 activations are not an arm of this kernel).
    with pytest.raises(ValueError, match="input_a must be fp16"):
        module.dense_w4a16_gemm_mutlass(
            input_a.bfloat16(), weight_b, scale_b, out
        )
    # weight_b must be the int8 packed operand.
    with pytest.raises(ValueError, match="weight_b must be int8"):
        module.dense_w4a16_gemm_mutlass(input_a, weight_b.to(torch.uint8), scale_b, out)
    # K must agree between A and the packed operand.
    with pytest.raises(ValueError, match="K values"):
        module.dense_w4a16_gemm_mutlass(input_a, weight_b[:, :-8], scale_b, out)
    # scale_b is (N, K // 128), not the transposed (K // 128, N).
    with pytest.raises(ValueError, match="scale_b must have shape"):
        module.dense_w4a16_gemm_mutlass(input_a, weight_b, scale_b.T.contiguous(), out)
    # out must be (M, N).
    with pytest.raises(ValueError, match="out must have shape"):
        module.dense_w4a16_gemm_mutlass(input_a, weight_b, scale_b, torch.empty(size_n, size_m, dtype=torch.float16))
    # act-order (g_idx) has to be materialized into the packed weights first.
    with pytest.raises(NotImplementedError, match="act-order"):
        module.dense_w4a16_gemm_mutlass(
            input_a, weight_b, scale_b, out, g_idx=torch.arange(size_k - 1, -1, -1, dtype=torch.int32)
        )
    # The identity g_idx is a no-op: checked on the pure helper, because the full call would launch
    # on a CPU tensor here. The hook is a contract marker, not a permutation.
    identity = torch.arange(size_k, dtype=torch.int32)
    assert module.apply_gidx_permutation(weight_b, identity) is weight_b
    assert module.apply_gidx_permutation(weight_b, None) is weight_b


@requires_torch
def test_w4a16_accuracy_vs_reference_quantizer_is_byte_exact():
    """Byte parity for the host-side packing: numpy packer == torch packer, and codes survive.

    Cheapest possible gate on the contract change that matters: if either implementation drifts, the
    operand the kernel reads is not the operand the reference was built from.
    """
    size_k = 512
    codes = _rand_codes(size_k, 128, seed=99)
    numpy_packed = pack_mate_weight_b(codes)
    torch_codes = torch.from_numpy(codes).to(torch.int16)
    low = torch_codes[0::2, :] & 0xF
    high = torch_codes[1::2, :] & 0xF
    torch_packed = (low | (high << 4)).T.contiguous().to(torch.uint8).view(torch.int8)
    assert np.array_equal(numpy_packed, _to_numpy(torch_packed))
    assert np.array_equal(unpack_mate_weight_b(_to_numpy(torch_packed), size_k), codes)
    # And the same codes in the GPTQ/Marlin layout round trip through the shipping repack.
    if REPACK_MUSA is not None and LAYOUT_REF is not None:
        gptq_words = gptq_layout_pack(codes)
        # Repack needs tile-aligned extents; a padded-N round trip is the padding test's job.
        repacked = np.asarray(REPACK_MUSA.repack_gptq(gptq_words, size_k, 128), dtype=np.int64)
        restored = np.asarray(LAYOUT_REF.unpack_marlin_tiles(repacked, size_k, 128), dtype=np.int64)
        assert np.array_equal(restored, codes)


# ----- B6: poison inputs ---------------------------------------------------------------------
@supported_musa_compute_capability([31])
@requires_torch
@pytest.mark.parametrize(
    "name,mode",
    [
        ("all_zero_weights", "zero"),
        ("constant_nibble", "constant"),
        ("one_outlier_per_row", "outlier"),
        ("alternating_sign", "alternating"),
    ],
)
def test_w4a16_poison_weights(name, mode):
    """Poison weights at the extremes; each case must match its own reference.

    The classic false positive is a kernel that returns zeros fast: the all-zero-*weight* case is the
    one that would hide it, which is why the constant-nibble case (which is analytically non-zero) is
    in the same family.
    """
    size_m, size_k, size_n = 8, 512, 128
    weight = torch.zeros(size_k, size_n, dtype=torch.float32)
    if mode == "zero":
        weight[:] = 0.0
    elif mode == "constant":
        weight[:] = 0.125
    elif mode == "outlier":
        weight[:, :] = 1e-4
        weight[::GROUP_SIZE, ::16] = 12.0
    else:
        weight[0::2, :] = 3.0
        weight[1::2, :] = -3.0
    quantized = _quantize_symmetric_u4b8(weight)
    out, input_a = _run_op(quantized, size_m, size_k, size_n, seed=31)
    out_ref = input_a @ torch.from_numpy(quantized.w_ref).to(input_a.device)
    max_diff = compute_max_diff(out, out_ref)
    print(f"[w4a16 poison:{name}] max_diff={max_diff:.5f}")
    assert torch.isfinite(out).all(), f"{name} produced non-finite output"
    assert max_diff < MAX_DIFF_TOLERANCE, f"{name}: max_diff={max_diff}"


@supported_musa_compute_capability([31])
@requires_torch
def test_w4a16_poison_infinite_activations_stay_non_finite():
    """An Inf activation cannot be silently swallowed: the output must be non-finite somewhere.

    Only the qualitative property is asserted; fp16 Inf times a finite weight is Inf in every
    accumulation order, so this is safe to require without a tolerance.
    """
    size_m, size_k, size_n = 4, 256, 64
    weight = torch.randn(size_k, size_n, dtype=torch.float32) / math.sqrt(size_k)
    quantized = _quantize_symmetric_u4b8(weight)
    input_a = torch.randn(size_m, size_k, dtype=torch.float16, device="musa")
    input_a[0, 0] = float("inf")
    out, _ = _run_op(quantized, size_m, size_k, size_n, input_a=input_a)
    assert not bool(torch.isfinite(out[0]).all())


@supported_musa_compute_capability([31])
@requires_torch
def test_w4a16_m_is_not_a_frozen_choice():
    """One instantiation must serve M = 1 .. 2048 without a shape-dependent recompile.

    The dispatch name is keyed on ``(out_dtype, zero_points)`` only, so a caller cannot freeze an
    M-specific decision at trace time. Asserted on the dispatch name, which is the observable.
    """
    import inspect

    module = pytest.importorskip("mate.jit.gemm.dense_w4a16_gemm")
    # The public entry must not accept an M-shaped argument: if it did, a caller could freeze a
    # shape-specific decision at trace time.
    for fn in (_mate_op(), module.dense_w4a16_gemm_mutlass):
        parameters = set(inspect.signature(fn).parameters)
        assert not ({"m", "size_m", "num_rows", "batch_size"} & parameters), parameters
    # And the dispatch name is keyed on (out dtype, zero-point arm) only.
    sym = module.dense_w4a16_gemm_dispatch_name(torch.float16, False)
    zp = module.dense_w4a16_gemm_dispatch_name(torch.float16, True)
    assert sym != zp and sym == module.dense_w4a16_gemm_dispatch_name(torch.float16, False)
    # The tile is part of the name, but it is NOT frozen to the shipped value: reading it from the
    # module keeps this test meaningful when the tile is deliberately changed (a per-shape tile
    # dispatch is a supported configuration, and hard-coding "32x256x256" made that look like a
    # numerics failure when it was only this assertion -- measured at tile_n=128, round 305).
    tile = "x".join(str(v) for v in module.DENSE_W4A16_GEMM_TILE[:3])
    assert tile in sym, f"{tile!r} not in dispatch name {sym!r}"
    # De-hardcode the k/stage half the same way the line above already de-hardcodes M and N: the
    # shipped "256x2" is the *current* choice, not the contract. Hard-coding it here turned a
    # deliberate tile change into a numerics-looking failure twice (tile_n=128 in round 305, the
    # M=16 atom in round 357); what this assertion must protect is that the name carries the tile.
    tile_k, stages = module.DENSE_W4A16_GEMM_TILE[2], module.DENSE_W4A16_GEMM_TILE[3]
    # The name format is `..._{m}x{n}x{k}_s{stages}_{zp_tag}`: k and stages are separated by `_s`,
    # not by `x`. The shipped assertion `"256x2" in sym` passed only by accidental substring
    # overlap (`..._x256x256_s2_...` contains "256x2"), which is why a deliberate tile change broke
    # it; deriving the separator from the rendered name keeps the assertion meaning what it says.
    assert f"{tile_k}_s{stages}" in sym, sym


# =============================================================================================
# Self-check entry point: runs the numpy-only gates without pytest so the CPU half of this port is
# executable on a box without torch (the authoring host has numpy 1.26.0 and no torch).
# =============================================================================================
def _self_check_without_torch() -> int:
    failures = []
    codes = _rand_codes(512, 128, seed=1)
    packed = pack_mate_weight_b(codes)
    if not np.array_equal(unpack_mate_weight_b(packed, 512), codes):
        failures.append("mate_weight_b round trip")
    if _ARTIFACTS is None:
        # The oracle is in-tree, so this cannot normally happen: report it as a failure rather than a
        # clean skip, because a missing reference means the parity gates silently did not run.
        failures.append("mate.marlin is not importable (byte-parity gates did not run)")
    else:
        print(f"reference: {Path(_MATE_MARLIN.reference.__file__).resolve()}")
        qweight = gptq_layout_pack(codes)
        reference = np.asarray(LAYOUT_REF.cuda_repack_gptq(qweight, 512, 128), dtype=np.int64)
        shipped = np.asarray(REPACK_MUSA.repack_gptq(qweight, 512, 128), dtype=np.int64)
        maxdiff = int(np.abs(reference - shipped).max())
        print(f"repack byte parity maxdiff = {maxdiff}")
        if maxdiff != 0:
            failures.append("repack byte parity")
        if not np.array_equal(LAYOUT_REF.unpack_rows(qweight, 512, 128), codes):
            failures.append("gptq pack round trip")
    for size_n, size_k in ODD_SHAPES + ALIGNED_SHAPES:
        mine = marlin_padded_nk(size_n, size_k, GROUP_SIZE)
        if _ARTIFACTS is not None and mine != tuple(LAYOUT_REF.marlin_padded_nk(size_n, size_k, GROUP_SIZE)):
            failures.append(f"padded_nk {size_n}x{size_k}")
        print(f"padded_nk({size_n}, {size_k}, {GROUP_SIZE}) = {mine}")

    # R3, analytically: constant nibble 0, unit scales.
    size_k, size_n = 512, 64
    num_groups = size_k // GROUP_SIZE
    scale = np.ones((num_groups, size_n), dtype=np.float32)
    stored = np.zeros((size_k, size_n), dtype=np.int64)
    quantized = synthetic_weight(stored, scale)
    w_ref = quantized.w_ref
    w_naive = np.repeat(scale, GROUP_SIZE, axis=0) * stored.astype(np.float32)
    a = np.random.default_rng(3).normal(size=(4, size_k)).astype(np.float32)
    group_sums = a.reshape(4, num_groups, GROUP_SIZE).sum(axis=2)
    delta = (a @ w_naive) - (a @ w_ref)
    analytic = np.einsum("mg,gn->mn", group_sums, NIBBLE_BIAS * scale)
    print(f"R3 naive-vs-biased reference error max = {np.abs(delta).max():.4f}")
    print(f"R3 analytic 8*sum_k(a) per group max   = {np.abs(analytic).max():.4f}")
    if not np.allclose(delta, analytic, rtol=1e-5, atol=1e-4):
        failures.append("R3 analytic identity")
    if float(np.abs(delta).max()) <= 0:
        failures.append("R3 error collapsed to zero")

    if failures:
        print("SELF-CHECK FAILED: " + ", ".join(failures))
        return 1
    print("SELF-CHECK PASSED (numpy-only gates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check_without_torch())


# =============================================================================================
# Section E - the Marlin wrapper's own gates (mate.marlin): upstream frame in, MATE frame out.
# =============================================================================================
WRAPPER_SHAPES = [(256, 64), (384, 256), (512, 128), (1024, 320)]


@pytest.mark.parametrize("size_k,size_n", WRAPPER_SHAPES)
def test_wrapper_shuffle_agrees_with_this_files_mate_packer(size_k, size_n):
    """``marlin_weight_to_musa`` against the independent MATE-frame packer defined in this file.

    One is written from upstream's repacked words, the other from the kernel's documented ``(N, K//2)``
    contract. Byte-for-byte agreement between two implementations written from different frames is the
    point: a bug would have to be the same bug in both.
    """
    codes = np.random.default_rng(size_k + size_n).integers(0, 16, (size_k, size_n)).astype(np.int64)
    repacked = REPACK_MUSA.repack_gptq(REPACK_MUSA.pack_rows(codes), size_k, size_n)
    mine = np.asarray(_MATE_MARLIN.marlin_weight_to_musa(repacked, size_k, size_n))
    assert mine.shape == (size_n, size_k // 2)

    # The wrapper presents the declared operand frame, which flips the sign bit of every nibble, so it
    # must differ from the verbatim MATE-frame packer by exactly 0x8 in both nibbles -- no more, no less.
    verbatim = pack_mate_weight_b(codes).view(np.uint8)
    assert np.array_equal(mine.view(np.uint8), verbatim ^ 0x88)
    assert np.array_equal(unpack_mate_weight_b(mine, size_k) ^ 0x8, codes)
    assert not np.array_equal(mine.view(np.uint8), verbatim), "no XOR would be a silent +8*scale"


@pytest.mark.parametrize("size_k,size_n", WRAPPER_SHAPES)
def test_wrapper_perm_frames_and_global_parity(size_k, size_n):
    """Two ``perm`` frames, and the global lookup the CUDA kernel really performs.

    The *tile* frame here is MATE's own shorthand -- one permutation of the 16 k positions, repeated
    in every k tile -- and the op frame is one entry per k (``g_idx_sort_indices``). They coincide
    exactly when ``K == 16``, which is how a tile-offset bug hides: the oracle here carried
    ``k_tile * TILE_K`` on top of a value that is already a global k, and every checker that used
    ``K == 16`` agreed with it. A non-trivial global shuffle is what makes this gate bite. (Upstream's
    ``get_weight_perm(4)`` is a third frame, 1024 flat ``(k_local, n_local)`` slots; see
    tests/test_marlin_repack_ground_truth.py, which executes that function as the oracle.)
    """
    rng = np.random.default_rng(size_k)
    words = REPACK_MUSA.pack_rows(rng.integers(0, 16, (size_k, size_n)).astype(np.int64))
    local = rng.permutation(16)
    global_perm = np.concatenate([local + t * 16 for t in range(size_k // 16)])

    assert np.array_equal(
        REPACK_MUSA.repack_gptq(words, size_k, size_n, local),
        REPACK_MUSA.repack_gptq(words, size_k, size_n, global_perm),
    ), "the 16-entry test frame must equal the per-tile repetition of the global perm"

    for perm in (global_perm, rng.permutation(size_k)):
        assert np.array_equal(
            REPACK_MUSA.repack_gptq(words, size_k, size_n, perm),
            LAYOUT_REF.cuda_repack_gptq(words, size_k, size_n, perm),
        ), "shipped repack and the CUDA transcription must agree under a global k permutation"

    with pytest.raises(ValueError):
        REPACK_MUSA.repack_gptq(words, size_k, size_n, np.arange(8))


def test_wrapper_padding_agrees_with_the_reference():
    """``compat.marlin_padded_nk`` vs ``reference.marlin_padded_nk``: two transcriptions, one table."""
    grid = [(64, 128), (200, 288), (256, 256), (1536, 1536), (4608, 4096), (4672, 512), (8960, 1536)]
    for size_n, size_k in grid:
        for group in (128, -1):
            assert tuple(_MATE_MARLIN.marlin_padded_nk(size_n, size_k, group)) == tuple(
                LAYOUT_REF.marlin_padded_nk(size_n, size_k, group)
            ), f"marlin_padded_nk({size_n}, {size_k}, {group}) diverged"


def test_wrapper_shape_gate_accepts_the_checkpoint_and_refuses_tp2_down():
    """The four checkpoint shapes are servable; rank-local K=4480 (TP=2 ``down``) is refused, with why."""
    for size_k, size_n in [(1536, 2048), (1536, 1536), (1536, 17920), (8960, 1536)]:
        ok, reason = _MATE_MARLIN.check_musa_marlin_supports_shape(size_k, size_n, 128)
        assert ok, f"K={size_k} N={size_n} should be servable, got: {reason}"
    ok, reason = _MATE_MARLIN.check_musa_marlin_supports_shape(4480, 1536, 128)
    assert not ok and "multiple of 256" in reason and "4480" in reason
    ok, reason = _MATE_MARLIN.check_musa_marlin_supports_shape(1536, 1536, 64)
    assert not ok and "group size" in reason


@pytest.mark.parametrize("group_size", [128, -1])
def test_wrapper_scale_inverse_is_the_inverse(group_size):
    """``marlin_unpermute_scales`` o ``marlin_permute_scales`` is the identity in *both* branches.

    A wrong inverse is still shape-correct, so only a round trip catches it. Both branches are gated
    because upstream picks between two different permutations.
    """
    size_k, size_n = 1024, 256
    rows = 1 if group_size == -1 else size_k // group_size
    plain = np.random.default_rng(group_size + 7).standard_normal((rows, size_n)).astype(np.float32)
    permuted = LAYOUT_REF.marlin_permute_scales(plain, size_k, size_n, group_size)
    back = LAYOUT_REF.marlin_unpermute_scales(permuted, size_k, size_n, group_size)
    assert back.shape == plain.shape
    assert np.array_equal(back, plain), "the host-direction inverse must recover the plain frame"


def test_wrapper_zero_point_inverse_is_the_inverse():
    """``unpack_marlin_zero_points`` o ``marlin_zero_points`` is the identity and the frame is (G, N)."""
    size_k, size_n = 1024, 512
    group = GROUP_SIZE
    zp = np.random.default_rng(11).integers(0, 16, (size_k // group, size_n)).astype(np.int64)
    packed = LAYOUT_REF.marlin_zero_points(zp, size_k, size_n, 4)
    assert packed.shape == (size_k // group, size_n // 8), "zero points pack along N"
    assert np.array_equal(LAYOUT_REF.unpack_marlin_zero_points(packed, size_k, size_n, 4), zp)


def _torch_or_none():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


needs_torch = pytest.mark.skipif(not _torch_or_none(), reason="torch is not installed")


@needs_torch
def test_wrapper_torch_conversions_match_the_reference():
    """The torch-side frame conversions against the numpy oracle, including the (N, G) orientation."""
    import torch

    m = _MATE_MARLIN
    size_k, size_n, group = 512, 256, 128
    rng = np.random.default_rng(3)

    scales = torch.from_numpy(rng.standard_normal((size_k // group, size_n)).astype(np.float32))
    permuted = m.quant.marlin_permute_scales(scales, size_k, size_n, group)
    assert np.array_equal(
        permuted.numpy(), LAYOUT_REF.marlin_permute_scales(scales.numpy(), size_k, size_n, group)
    )
    plain = m.marlin_scales_to_musa(permuted, size_k, size_n, group)
    assert plain.shape == (size_n, size_k // group) and plain.dtype == torch.float16
    assert np.allclose(plain.numpy().T, scales.numpy(), atol=1e-3), "transpose + fp16 cast"

    zp = torch.from_numpy(rng.integers(0, 16, (size_k // group, size_n)).astype(np.int32))
    packed = m.quant.marlin_zero_points(zp, size_k, size_n, 4)
    assert np.array_equal(
        m.quant.unpack_marlin_zero_points(packed, size_k, size_n).numpy(), zp.numpy()
    )
    zp_b = m.marlin_zero_points_to_musa(packed, size_k, size_n, group)
    assert zp_b.shape == (size_n, size_k // group) and zp_b.dtype == torch.float16


@needs_torch
def test_wrapper_prepare_frame_is_kernel_ready():
    """``prepare_marlin_w4a16`` hands back the exact shapes and dtypes the kernel's FFI documents."""
    import torch

    m = _MATE_MARLIN
    size_k, size_n, group = 512, 256, 128
    padded_n, padded_k = m.marlin_padded_nk(size_n, size_k, group)
    codes = np.random.default_rng(13).integers(0, 16, (padded_k, padded_n)).astype(np.int64)

    qweight = torch.from_numpy(
        LAYOUT_REF.cuda_repack_gptq(
            (REPACK_MUSA.pack_rows(codes) if padded_n == size_n else
             pack_mate_weight_b(codes).T.astype(np.int32) * 0 + REPACK_MUSA.pack_rows(codes)),
            padded_k, padded_n,
        ).astype(np.int32)
    )
    scales = torch.from_numpy(
        LAYOUT_REF.marlin_permute_scales(
            np.random.default_rng(17).standard_normal((padded_k // group, padded_n)).astype(np.float32),
            padded_k, padded_n, group,
        )
    )
    zp = torch.from_numpy(
        LAYOUT_REF.marlin_zero_points(
            np.random.default_rng(19).integers(0, 16, (padded_k // group, padded_n)).astype(np.int64),
            padded_k, padded_n, 4,
        ).astype(np.int32)
    )

    frame = m.prepare_marlin_w4a16(qweight, scales, zp, size_k, size_n, group_size=group)
    assert frame.weight_b.shape == (padded_n, padded_k // 2) and frame.weight_b.dtype == torch.int8
    assert frame.scale_b.shape == (padded_n, padded_k // group) and frame.scale_b.dtype == torch.float16
    assert frame.zero_points_b.shape == (padded_n, padded_k // group)
    assert frame.has_zero_points and frame.logical_n == size_n and frame.padded_k == padded_k

    # the weight conversion must also match this file's independent MATE-frame packer, up to the
    # sign-bit presentation the declared operand frame requires
    assert np.array_equal(
        frame.weight_b.numpy().view(np.uint8), pack_mate_weight_b(codes).view(np.uint8) ^ 0x88
    )
    # symmetric path: no zero-point tensor, and the kernel's sym instantiation ignores it
    symmetric = m.prepare_marlin_w4a16(qweight, scales, None, size_k, size_n, group_size=group)
    assert symmetric.zero_points_b is None and not symmetric.has_zero_points

    # refusals: the TP=2 down casualty, and an unmaterialised act order
    with pytest.raises(ValueError, match="multiple of 256"):
        m.prepare_marlin_w4a16(
            torch.zeros((4480 // 16, 2 * 1536), dtype=torch.int32),
            torch.zeros((4480 // group, 1536), dtype=torch.float32),
            None, 4480, 1536, group_size=group,
        )
    with pytest.raises(ValueError, match="act order"):
        m.prepare_marlin_w4a16(
            qweight, scales, None, size_k, size_n, group_size=group,
            g_idx=torch.arange(padded_k - 1, -1, -1, dtype=torch.int32),
        )


def test_wrapper_sign_bit_presentation_is_the_declared_frame():
    """The authority for ``^ 0x8``, made executable, plus the half-range error verbatim would cause.

    MATE's packed-weight contract says a nibble "already holds the signed INT4 value
    ``unsigned_nibble - nibble_bias``" (mate/jit/gemm/dense_w4a16_gemm.py:46-49) and its public
    dequant is ``w = scale * (q + nibble_bias - zero_point)`` (mate/gemm.py:1503-1507), while the
    atom reads two's complement -- so the presented nibble must satisfy ``s(v) = u - 8`` for *every*
    ``u``, and ``v = u ^ 0x8`` is the unique such presentation. ``s(u ^ 0x8) = u - 8`` is an identity,
    which is exactly why verbatim is only accidentally right at one code and wrong everywhere below 8.
    """
    def two_complement(code):
        """The atom's measured read: plain two's complement, no `-8` bias of its own."""
        code = np.asarray(code, dtype=np.int64)
        return np.where(code >= 8, code - 16, code)

    for u in range(16):
        assert two_complement(u ^ 0x8) == u - 8, f"the identity must hold at u={u}"

    low_half = np.arange(8)
    assert not np.all(two_complement(low_half) == low_half - 8), "verbatim must fail below u=8"
    assert two_complement(8) == -8, "the case that made -8 look like 'u - 8'"
    assert two_complement(0) == 0, "the case that falsifies it"


def test_wrapper_zero_points_are_raw_and_a_double_fold_is_visible():
    """The zero-point stream is raw, and a symmetric checkpoint's zero point is ``nibble_bias``.

    MATE's op applies the ``(nibble_bias - zero_point)`` term itself (mate/gemm.py:1512-1515), so a
    stream already folded by ``fold_zero_points`` would be folded twice. The detector here is sharp
    because a symmetric checkpoint must carry 8, and ``8 - 8`` is exactly the zero correction: a
    double fold shows up as a zero-point tensor where 8s belong, and every asymmetric weight moves by
    ``8 * scale`` per element.
    """
    size_k, size_n = 512, 256
    group = GROUP_SIZE
    nibble_bias = NIBBLE_BIAS

    symmetric = np.full((size_k // group, size_n), nibble_bias, dtype=np.int64)
    packed = LAYOUT_REF.marlin_zero_points(symmetric, size_k, size_n, 4)
    back = LAYOUT_REF.unpack_marlin_zero_points(packed, size_k, size_n, 4)
    assert np.array_equal(back, symmetric), "raw values must survive packing unchanged"
    assert np.all(back == nibble_bias), "a symmetric checkpoint's zero point is the nibble bias"

    # what a double fold would look like: the already-folded factor is what must NOT be stored
    folded = (nibble_bias - symmetric) * np.ones_like(symmetric, dtype=np.float32)
    assert np.all(folded == 0.0), "the symmetric correction is exactly zero, not 8"
    assert not np.array_equal(back.astype(np.float32), folded), "stored zeros means a double fold"


def test_wrapper_zero_point_frame_matches_scale_frame():
    """The three operands of one frame must share one n order.

    ``prepare_marlin_w4a16`` returns ``weight_b``, ``scale_b`` and ``zero_points_b`` and the kernel
    consumes them as one frame: row ``n`` of each is output column ``n``. The shape guards cannot see
    a disagreement, so this compares values: given a permuted (upstream-frame) scale tensor and a
    permuted zero-point tensor built from the *same* plain arrays, both inverse functions must return
    exactly those plain arrays, transposed.

    This is the regression test for the defect where ``marlin_zero_points_to_musa`` went through
    ``unpack_marlin_zero_points`` -- which applies ``argsort([0,2,4,6,1,3,5,7])`` within each 8 and a
    per-tile ``scale_perm`` on top of ``unpack_cols`` -- and therefore returned the zero points
    permuted relative to the scales. Measured end to end on MUSA, that pairing gave a 1.034323
    relative error against an fp32 reference while every shape guard passed; with the frames aligned
    it is 0.000315.
    """
    torch = pytest.importorskip("torch")
    if not _ARTIFACTS:
        pytest.skip("mate.marlin is not importable")

    m = _MATE_MARLIN
    size_k, size_n, group = 512, 256, 128
    n_groups = size_k // group

    plain_scales = np.random.default_rng(23).standard_normal((n_groups, size_n)).astype(np.float32)
    plain_zp = np.random.default_rng(29).integers(0, 16, (n_groups, size_n)).astype(np.int64)

    permuted_scales = torch.from_numpy(
        LAYOUT_REF.marlin_permute_scales(plain_scales, size_k, size_n, group).astype(np.float32)
    )
    permuted_zp = torch.from_numpy(
        LAYOUT_REF.marlin_zero_points(plain_zp, size_k, size_n, 4).astype(np.int32)
    )

    scales_b = m.marlin_scales_to_musa(permuted_scales, size_k, size_n, group)
    # upstream's own packing permutes before packing, so it needs the upstream inverse
    zp_b = m.marlin_zero_points_to_musa(
        permuted_zp, size_k, size_n, group, upstream_permuted=True
    )

    expect_scales = torch.from_numpy(np.ascontiguousarray(plain_scales.T)).to(torch.float16)
    expect_zp = torch.from_numpy(np.ascontiguousarray(plain_zp.T)).to(torch.float16)

    assert scales_b.shape == expect_scales.shape == (size_n, n_groups)
    assert zp_b.shape == expect_zp.shape == (size_n, n_groups), (
        "scale_b and zero_points_b must share the (size_n, num_groups) frame"
    )
    assert torch.equal(scales_b, expect_scales), "scale_b is not in the caller's n order"
    assert torch.equal(zp_b, expect_zp), (
        "zero_points_b is in a different n order from scale_b; the kernel would scale column n by "
        "one group's scales and correct it by another group's zero points"
    )


def test_wrapper_zero_point_frame_matches_loader_convention():
    """The serving path's packed zero points are NOT upstream's ``marlin_zero_points`` output.

    vLLM's loader converts an AWQ checkpoint into the plain GPTQ packing, so a packed word runs in the
    caller's order (``n = 8 * word + nibble``) with no interleave and no tile permutation applied
    before packing. ``unpack_cols`` is the exact inverse of that packing; upstream's
    ``marlin_zero_points`` -- which :func:`test_wrapper_zero_point_frame_matches_scale_frame` exercises
    -- permutes first and therefore needs ``unpack_marlin_zero_points`` instead.

    Both tests exist on purpose: the two conventions need different inverses, and a single
    unconditional call cannot serve both. Whichever one ``prepare_marlin_w4a16`` picks has to match
    what the caller's loader actually produced, and this pin records which convention was measured end
    to end on MUSA (0.000315 relative error against an fp32 reference at K=512, N=256, group 128).
    """
    torch = pytest.importorskip("torch")
    if not _ARTIFACTS:
        pytest.skip("mate.marlin is not importable")

    m = _MATE_MARLIN
    size_k, size_n, group = 512, 256, 128
    n_groups = size_k // group

    plain_zp = np.random.default_rng(31).integers(0, 16, (n_groups, size_n)).astype(np.int64)
    shifts = (np.arange(8) * 4).astype(np.int64)
    packed = np.zeros((n_groups, size_n // 8), np.int64)
    for j in range(8):
        packed |= (plain_zp[:, j::8] & 0xF) << shifts[j]

    zp_b = m.marlin_zero_points_to_musa(torch.from_numpy(packed.astype(np.int32)),
                                       size_k, size_n, group)
    expect = torch.from_numpy(np.ascontiguousarray(plain_zp.T)).to(torch.float16)
    assert zp_b.shape == expect.shape == (size_n, n_groups)
    assert torch.equal(zp_b, expect), (
        "the loader's plain packing must invert to the caller's n order; if this fails while the "
        "upstream-convention test passes, prepare_marlin_w4a16 is serving the loader's tensors with "
        "upstream's inverse"
    )
