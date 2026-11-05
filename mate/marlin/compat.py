# SPDX-License-Identifier: Apache-2.0
"""Shape compatibility and tile padding for Marlin on MUSA, in upstream's vocabulary.

Two families of facts live here and they must not be confused:

* the **upstream** contracts, mirrored so a vllm-musa call site makes the same decisions it would on
  NVIDIA — ``marlin_padded_nk`` / ``marlin_repacked_nk`` and the pad/unpad helpers, which are
  ``marlin_utils.py:221-302``;
* MATE's **kernel-side** requirements, which are stricter than upstream's in one direction (K) and
  identical in another (tile-aligned N), exposed as :func:`check_musa_marlin_supports_shape`.

K is the interesting one. Upstream keeps a shape alive by tile-padding K up to ``lcm(64 or 128,
group_size)``; MATE's K256 mainloop tile carries two K128 scale blocks, so the landed op requires
``K % 256 == 0`` and refuses anything else rather than padding it. The practical consequence, worth
knowing before it is discovered the hard way: a TP-sharded ``down`` projection at TP=2 is K=4480,
which is not a multiple of 256, so it is refused. TP=1 for the checkpoint this ticket targets is
unaffected (all four K are multiples of 256). Removing that gap needs a smaller K tile, not a
different argument, which is why it is a shaped check and not a flag.
"""

from __future__ import annotations

import math

from . import guards

try:  # lazy by design, mirroring mate.marlin.quant
    import torch
except ImportError:  # pragma: no cover - exercised on the numpy-only gate host
    torch = None  # type: ignore[assignment]

GPTQ_MARLIN_TILE = 16  # marlin_utils.py:31
MUSA_MARLIN_GROUP_SIZE = 128
MUSA_MARLIN_K_MULTIPLE = 256


def _require_torch():
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError(
            "mate.marlin.compat needs torch for the tensor helpers; marlin_padded_nk, "
            "marlin_repacked_nk and marlin_is_tile_aligned are pure integer arithmetic"
        )
    return torch


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def marlin_padded_nk(size_n: int, size_k: int, group_size: int = -1) -> tuple[int, int]:
    """``marlin_utils.py:221-243``: the minimal ``(padded_n, padded_k)`` for a Marlin tile family.

    Two families qualify ``(n % 64 == 0, k % 128 == 0)`` and ``(n % 128 == 0, k % 64 == 0)``; shapes
    satisfying neither are padded up to the cheaper one, and K stays divisible by ``group_size`` so
    the padded scales keep an integral group count. Ties are allowed and broken by area then by
    ``n + k``, exactly as upstream does — a different tie-break would silently change the packed
    shape, so this is a place where "close enough" is not.
    """
    group = group_size if group_size > 0 else 1
    candidates = (
        (_round_up(size_n, 64), _round_up(size_k, math.lcm(128, group))),
        (_round_up(size_n, 128), _round_up(size_k, math.lcm(64, group))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def marlin_is_tile_aligned(size_n: int, size_k: int) -> bool:
    """True when ``(size_n, size_k)`` already satisfies one Marlin tile family, so no padding is
    needed. Equivalent to ``marlin_padded_nk(n, k) == (n, k)`` without the tie-break question."""
    return (size_n % 64 == 0 and size_k % 128 == 0) or (size_n % 128 == 0 and size_k % 64 == 0)


def marlin_repacked_nk(qweight, num_bits: int = 4) -> tuple[int, int]:
    """``marlin_utils.py:246-252``: recover ``(size_n, size_k)`` from a repacked weight's shape.

    Accepts either the packed tensor (upstream's form) or its shape, because MATE's own gates reason
    about shapes while the call sites hold tensors. ``num_bits`` is int4-only, as elsewhere.
    """
    if num_bits != 4:
        raise NotImplementedError(f"MATE's Marlin wrapper is int4 only; got num_bits={num_bits}")
    shape = qweight.shape if hasattr(qweight, "shape") else tuple(qweight)
    pack_factor = 32 // num_bits
    size_k = shape[0] * GPTQ_MARLIN_TILE
    size_n = shape[1] * pack_factor // GPTQ_MARLIN_TILE
    return size_n, size_k


def marlin_pad_qweight(qweight, size_n: int, size_k: int, padded_n: int, padded_k: int):
    """``marlin_utils.py:255-265``: zero-pad a GPTQ-layout weight before ``gptq_marlin_repack``."""
    torch_mod = _require_torch()
    if (padded_n, padded_k) == (size_n, size_k):
        return qweight
    pack_factor = size_k // qweight.size(0)
    return torch_mod.nn.functional.pad(
        qweight, (0, padded_n - size_n, 0, (padded_k - size_k) // pack_factor)
    )


def marlin_pad_scales(
    scales, size_n: int, size_k: int, padded_n: int, padded_k: int, group_size: int
):
    """``marlin_utils.py:268-282``: zero-pad ``(num_groups, size_n)`` scales before permuting them."""
    torch_mod = _require_torch()
    if (padded_n, padded_k) == (size_n, size_k):
        return scales
    pad_rows = padded_k // group_size - scales.size(0) if group_size > 0 else 0
    assert pad_rows >= 0
    return torch_mod.nn.functional.pad(scales, (0, padded_n - size_n, 0, pad_rows))


def marlin_pad_dim(x, size: int, padded: int):
    """``marlin_utils.py:285-289``: zero-pad an activation's last dim from ``size`` to ``padded``.

    Padding works through zero *scales*, not zero weights: a pad made of zero weights decodes to
    whatever that code means, which for a symmetric checkpoint is not zero, so it would contribute.
    Zero for the extents above is what makes the padded region vanish in both the GPTQ and the AWQ
    arms. (The exact value a code decodes to depends on the presentation; see
    ``mate.marlin.linear.marlin_weight_to_musa`` for that convention and its authority.)
    """
    torch_mod = _require_torch()
    if padded == size:
        return x
    return torch_mod.nn.functional.pad(x, (0, padded - size))


def marlin_unpad_output(output, size_n: int, padded_n: int):
    """``marlin_utils.py:292-302``: strip padded output columns back to the logical N."""
    if padded_n == size_n:
        return output
    return output[..., :size_n].contiguous()


def check_musa_marlin_supports_shape(
    size_k: int, size_n: int, group_size: int = MUSA_MARLIN_GROUP_SIZE
) -> tuple[bool, str | None]:
    """Whether MATE's kernel can serve this shape, with the reason when it cannot.

    Returns ``(ok, reason)`` in the same shape as ``MPLinearKernel.can_implement``, because that is
    where this is called from. The checks are the ones that are *structural* in the landed device
    header and host module, not preferences: group 128 because the K256 tile carries two K128 scale
    blocks, ``K % 256 == 0`` because that tile cannot express a partial group, and the repack's own
    ``K % 16 == 0`` / ``N % 64 == 0`` because a Marlin tile is 16x64.

    This is the single place that owns the port's ``K % 256`` requirement:
    ``mate.marlin.guards.require_musa_k_multiple`` carries the invariant and its reason, and the
    upstream-mirroring repack functions deliberately do not enforce it (upstream's repack has no
    such rule, and the byte-parity gates exercise K values it would refuse).
    """
    if group_size != MUSA_MARLIN_GROUP_SIZE:
        return (
            False,
            f"MATE's K256 mainloop tile holds exactly two K128 weight-scale blocks, so group size is "
            f"structurally {MUSA_MARLIN_GROUP_SIZE}, not {group_size}",
        )
    try:
        guards.require_musa_k_multiple(size_k)
    except ValueError as exc:
        return (False, str(exc))
    try:
        guards.require_repack_tile_alignment(size_k, size_n)
        guards.require_nibble_packing(size_k, size_n, "gptq")
    except ValueError as exc:
        return (False, str(exc))
    return True, None
