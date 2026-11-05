# SPDX-License-Identifier: Apache-2.0
"""Upstream Marlin's scale and zero-point layout for 4-bit weights, implemented by MATE.

These mirror ``vllm/model_executor/layers/quantization/utils/marlin_utils.py`` name for name and
argument for argument, so a vllm-musa call site reads exactly as the NVIDIA one does. That is the
point of this package: the *layout* stays upstream's, the implementation is MATE's.

``torch`` is imported lazily, because ``import mate.marlin`` must keep working on a host without
torch — the gates that guard this code are numpy-only and live in ``mate.marlin.reference``.

Divergences from upstream, each with its reason:

* ``num_bits`` must be 4. MATE's kernel is int4 x fp16 only, so the 8-bit branches are refused with
  a reason instead of being approximated. Upstream handles 4 and 8 bits in one code path.
* ``marlin_permute_scales`` keeps upstream's ``is_a_8bit`` argument for signature compatibility, but
  the 8-bit branch is unreachable here for the same reason.
* :func:`unpack_marlin_zero_points` has no upstream counterpart; upstream's tests invert the packing
  by hand. MATE keeps one, because a round trip is the cheapest proof that packing is a bijection.
"""

from __future__ import annotations

import numpy as np

from . import guards

try:  # lazy by design: see the module docstring
    import torch
except ImportError:  # pragma: no cover - exercised on the numpy-only gate host
    torch = None  # type: ignore[assignment]

NUM_BITS_SUPPORTED = 4
PACK_FACTOR_4BIT = 8


def _require_torch():
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError(
            "mate.marlin.quant needs torch; mate.marlin.reference is the numpy-only oracle"
        )
    return torch


def _require_int4(num_bits: int) -> None:
    if num_bits != NUM_BITS_SUPPORTED:
        raise NotImplementedError(
            f"MATE's Marlin wrapper is int4 x fp16 only, so num_bits must be 4; got {num_bits}. "
            "The upstream 8-bit branches are deliberately not implemented rather than guessed at."
        )


def get_scale_perms() -> tuple[list[int], list[int]]:
    """``marlin_utils.py:460-467`` verbatim — the two permutations and nothing else."""
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def pack_cols(q_w, num_bits: int = NUM_BITS_SUPPORTED):
    """``quant_utils.py:886-909``, 4-bit arm: ``(size_k, size_n)`` nibbles -> ``(size_k, size_n/8)``.

    Kept local to ``quant`` rather than reused from :mod:`mate.marlin.layout` because upstream's
    version preserves the input device and returns int32, which the call sites depend on; the
    layout module's packer is the numpy form used by the repack. Both are gated against the
    reference in ``tests/test_w4a16_marlin.py``, so they cannot drift apart silently.
    """
    torch_mod = _require_torch()
    _require_int4(num_bits)
    size_k, size_n = q_w.shape
    assert size_n % PACK_FACTOR_4BIT == 0, f"size_n={size_n} is not a multiple of 8"
    orig_device = q_w.device
    w = q_w.cpu().numpy().astype(np.uint32)
    q_res = np.zeros((size_k, size_n // PACK_FACTOR_4BIT), dtype=np.uint32)
    for i in range(PACK_FACTOR_4BIT):
        q_res |= w[:, i::PACK_FACTOR_4BIT] << (num_bits * i)
    return torch_mod.from_numpy(q_res.astype(np.int32)).to(orig_device).contiguous()


def unpack_cols(packed_q_w, size_k: int, size_n: int, num_bits: int = NUM_BITS_SUPPORTED):
    """``quant_utils.py:912-940``, 4-bit arm: the inverse of :func:`pack_cols`."""
    torch_mod = _require_torch()
    _require_int4(num_bits)
    assert size_n % PACK_FACTOR_4BIT == 0, f"size_n={size_n} is not a multiple of 8"
    assert packed_q_w.shape == (size_k, size_n // PACK_FACTOR_4BIT), (
        f"packed zero points have shape {tuple(packed_q_w.shape)}, expected "
        f"{(size_k, size_n // PACK_FACTOR_4BIT)} for size_k={size_k}, size_n={size_n}"
    )
    orig_device = packed_q_w.device
    packed = packed_q_w.cpu().numpy().astype(np.uint32)
    q_res = np.zeros((size_k, size_n), dtype=np.uint32)
    mask = (1 << num_bits) - 1
    for i in range(PACK_FACTOR_4BIT):
        vals = packed & mask
        packed >>= num_bits
        q_res[:, i::PACK_FACTOR_4BIT] = vals
    return torch_mod.from_numpy(q_res.astype(np.int32)).to(orig_device).contiguous()


def marlin_permute_scales(
    s, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
):
    """``marlin_utils.py:470-480``: which perm applies is decided by the group layout.

    Per-group scales (``0 < group_size < size_k``) take the 8-wide ``scale_perm``; channel-wise
    scales (``group_size == -1`` or ``>= size_k``) take the 32-wide ``scale_perm_single``. Both are
    upstream's, so a checkpoint converted for NVIDIA Marlin lands in the same place on MUSA.
    """
    _require_torch()
    # The 64-wide (or 32-wide) reshape succeeds for any row count whose product is a multiple of
    # the block width, so scales that belong to a different K permute the wrong number of groups
    # and come back with a plausible shape. Check the frame upstream's arithmetic assumes.
    guards.require_scale_frame(s.shape, size_k, size_n, group_size)  # marlin_utils.py:470-480
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def marlin_zero_points(
    zp, size_k: int, size_n: int, num_bits: int, is_a_8bit: bool = False
):
    """``marlin_utils.py:512-533``: the *non-single* perm, then the 4-bit column interleave.

    Upstream's comment gives the reason the two differ: zero points are applied on every MMA, so the
    channel-wise "single" permutation does not describe them.
    """
    _require_torch()
    _require_int4(num_bits)
    # Upstream's pack_cols asserts the row count it is handed (quant_utils.py:892-895); the frame
    # check here carries that assertion forward so a wrong-group-count tensor is refused instead of
    # being packed into a plausible word array.
    guards.require_zero_point_frame(zp.shape, size_n)  # marlin_utils.py:512-533
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])  # marlin_utils.py:522, num_bits == 4 arm
    if not is_a_8bit:
        zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    return pack_cols(zp, num_bits)


def unpack_marlin_zero_points(zp_packed, size_k: int, size_n: int, num_bits: int = 4):
    """Inverse of :func:`marlin_zero_points`; returns ``(size_k, size_n)`` zero points."""
    _require_torch()
    _require_int4(num_bits)
    scale_perm, _ = get_scale_perms()
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])

    # The packed tensor's first axis is the GROUP count, never size_k: marlin_zero_points packs a
    # (num_groups, size_n) array via pack_cols, so the packed shape is (num_groups, size_n // 8).
    # reference.py:401 has always passed packed.shape[0] here; passing size_k was a port-time defect
    # and is exactly the `(4, 32)` vs `(512, 32)` failure (4 == 512 / 128 groups).
    assert zp_packed.shape[1] * PACK_FACTOR_4BIT == size_n, (
        f"packed zero points have {zp_packed.shape[1]} column words; size_n={size_n} needs "
        f"{size_n // PACK_FACTOR_4BIT}"
    )
    zp = unpack_cols(zp_packed, zp_packed.shape[0], size_n, num_bits).reshape((-1, size_n))
    zp = zp.reshape((zp.shape[0], -1, len(interleave)))[:, :, np.argsort(interleave)]
    # scale_perm is the PER-TILE permutation (len(scale_perm) columns), so it must be applied
    # within each tile, exactly as marlin_permute_scales does; indexing the whole size_n axis
    # with it selects only len(scale_perm) columns and silently narrows the result by size_n/len.
    zp = zp.reshape((-1, len(scale_perm)))[:, np.argsort(np.array(scale_perm))]
    zp = zp.reshape((-1, size_n))
    # (num_groups, size_n), NOT (size_k, size_n): this is the inverse of marlin_zero_points,
    # which packs a (num_groups, size_n) array, and the suite's own caller at
    # tests/test_w4a16_marlin.py:1341 compares the result against a (num_groups, size_n) array.
    # size_k stays in the signature to mirror upstream's API and to carry the packing factor.
    assert size_n % PACK_FACTOR_4BIT == 0, f"size_n={size_n} is not a multiple of 8"
    assert zp.shape == (zp_packed.shape[0], size_n), (
        f"zero points unpacked to {tuple(zp.shape)}, expected {(zp_packed.shape[0], size_n)}"
    )
    return zp.contiguous()


def awq_to_marlin_zero_points(
    q_zp_packed, size_k: int, size_n: int, num_bits: int, is_a_8bit: bool = False
):
    """``marlin_utils.py:536-561``: undo AWQ's column packing and interleave, then re-permute.

    Only for zero points still in AWQ's own packing. vLLM converts AWQ checkpoints to the standard
    GPTQ packing at load time, and those zero points go through :func:`marlin_zero_points` (upstream's
    ``marlin.py`` path) or, without the Marlin frame, through
    :func:`mate.marlin.unpack_zero_points`. Either way the zero points still need converting; what
    MUSA skips is only the AWQ-specific repack.
    """
    _require_torch()
    _require_int4(num_bits)
    # Same convention: AWQ's packed zero points are (num_groups, size_n // 8), not (size_k, size_n // 8).
    q_zp = unpack_cols(q_zp_packed, q_zp_packed.shape[0], size_n, num_bits)

    undo_interleave = np.argsort(np.array([0, 2, 4, 6, 1, 3, 5, 7]))  # marlin_utils.py:551
    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()
    return marlin_zero_points(q_zp, size_k, size_n, num_bits, is_a_8bit)
