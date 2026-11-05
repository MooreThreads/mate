# SPDX-License-Identifier: Apache-2.0
"""Shape guards for the permute/pack axis: the invariants the index derivation implies.

Every frame error on this axis is *silent*. The repack, the scale permutation and the zero-point
permutation are reshapes and gathers: feed one the wrong frame and it returns an array of the right
dtype, often of a plausible shape, holding plausible values in the wrong places. A test that only
ever compares the result against another copy of the same convention cannot see that.

These guards enforce the invariants that upstream's own host code enforces or that its reshape
arithmetic silently assumes, so that a mismatched convention announces itself at the call that broke
it. Each guard cites the upstream line carrying the invariant it enforces -- check the guard against
that line, not against this comment.

Importable with no torch and no numpy: the checks are integer and shape arithmetic only, which is
what lets the numpy-only gate host exercise the ones reachable from ``mate.marlin.layout``.
"""

from __future__ import annotations

# Upstream constants, each with its own source (pinned checkout vLLM 2cf0a6915c; the repack sources
# are byte-identical in the newer workspace checkout, see the ticket report).
MARLIN_TILE_K = 16  # csrc/.../marlin.cuh:30,38  (tile_size = 16, tile_k_size = tile_size)
MARLIN_TILE_N = 64  # csrc/.../marlin.cuh:39     (tile_n_size = tile_k_size * 4)
PACK_FACTOR_4BIT = 8  # 32 / 4 bits; gptq_marlin_repack.cu:20,297
GPTQ_MIN_THREAD_N = 64  # marlin_utils.py:32 (GPTQ_MARLIN_MIN_THREAD_N)
MUSA_MARLIN_GROUP_SIZE = 128
MUSA_MARLIN_K_MULTIPLE = 256


def _shape_tuple(shape) -> tuple[int, ...]:
    if hasattr(shape, "shape"):  # a tensor-like object was passed instead of its shape
        shape = shape.shape
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError as exc:  # pragma: no cover - a scalar was passed
        raise TypeError(f"expected a shape or an array-like, got {shape!r}") from exc


def require_repack_tile_alignment(size_k: int, size_n: int) -> None:
    """The Marlin tile is 16x64, and upstream's host code refuses anything that is not a multiple.

    Upstream invariant: ``gptq_marlin_repack.cu:290-293`` (``size_k % tile_k_size`` and
    ``size_n % tile_n_size``, the same two checks in ``awq_marlin_repack.cu:232-235``). Without
    them the kernel's ``k_tiles``/``n_tiles`` floor-division silently drops the remainder.

    Callers that hit this usually skipped a pad step rather than mis-shaped a buffer: upstream pads
    to a legal extent *before* repacking (``marlin_utils.py:221-243``, ``marlin_padded_nk``, exposed
    here as ``mate.marlin.compat.marlin_padded_nk``).
    """
    if size_k % MARLIN_TILE_K:
        raise ValueError(
            f"upstream requires size_k % {MARLIN_TILE_K} == 0 (gptq_marlin_repack.cu:290); "
            f"got size_k={size_k}"
        )
    if size_n % MARLIN_TILE_N:
        raise ValueError(
            f"upstream requires size_n % {MARLIN_TILE_N} == 0 (gptq_marlin_repack.cu:292); "
            f"got size_n={size_n}"
        )


def require_nibble_packing(size_k: int, size_n: int, layout: str) -> None:
    """The packed word holds 8 nibbles along the axis that layout packs.

    Upstream invariant: GPTQ packs along K (``quant_utils.py:868-871``: ``assert size_k %
    pack_factor == 0``), AWQ packs along N (``quant_utils.py:892-895``). The other extents are
    tile-guarded; this is the one that says *which* axis carries the nibble index, so a caller that
    fed the wrong frame gets the reason rather than a reshape error or a shifted result.
    """
    if layout == "gptq":
        if size_k % PACK_FACTOR_4BIT:
            raise ValueError(
                f"the GPTQ word axis packs {PACK_FACTOR_4BIT} k-nibbles per word "
                f"(quant_utils.py:871); got size_k={size_k}"
            )
    elif layout == "awq":
        if size_n % PACK_FACTOR_4BIT:
            raise ValueError(
                f"the AWQ word axis packs {PACK_FACTOR_4BIT} n-nibbles per word "
                f"(quant_utils.py:895); got size_n={size_n}"
            )
    else:  # pragma: no cover - programming error, not a shape error
        raise ValueError(f"unknown layout {layout!r}; expected 'gptq' or 'awq'")


def require_gptq_word_shape(shape, size_k: int, size_n: int) -> None:
    """A GPTQ packed weight is exactly ``(size_k / 8, size_n)``.

    Upstream invariant: ``gptq_marlin_repack.cu:300-305`` checks both extents separately
    (``size_k / pack_factor == b_q_weight.size(0)`` and ``b_q_weight.size(1) == size_n``). Checking
    them separately is the point: a transpose of an N x K/8 buffer has the same element count, so a
    product-only check (or no check at all) accepts it and permutes the wrong axis.
    """
    require_nibble_packing(size_k, size_n, "gptq")
    expected = (size_k // PACK_FACTOR_4BIT, size_n)
    actual = _shape_tuple(shape)
    if actual != expected:
        raise ValueError(
            f"GPTQ packed weight must be (size_k/{PACK_FACTOR_4BIT}, size_n) = {expected} "
            f"(gptq_marlin_repack.cu:300-305); got {actual}"
        )


def require_awq_word_shape(shape, size_k: int, size_n: int) -> None:
    """An AWQ packed weight is exactly ``(size_k, size_n / 8)``.

    Upstream invariant: ``awq_marlin_repack.cu:242-247`` checks both extents separately.
    """
    require_nibble_packing(size_k, size_n, "awq")
    expected = (size_k, size_n // PACK_FACTOR_4BIT)
    actual = _shape_tuple(shape)
    if actual != expected:
        raise ValueError(
            f"AWQ packed weight must be (size_k, size_n/{PACK_FACTOR_4BIT}) = {expected} "
            f"(awq_marlin_repack.cu:242-247); got {actual}"
        )


def require_repacked_shape(shape, size_k: int, size_n: int) -> None:
    """A Marlin repacked buffer is exactly ``(size_k / 16, size_n * 16 / 8)``.

    Upstream invariant: the output allocation of both repacks,
    ``gptq_marlin_repack.cu:324-326`` and ``awq_marlin_repack.cu:261-263``
    (``{size_k / marlin::tile_size, size_n * marlin::tile_size / pack_factor}``). The kernel writes
    ``out_offset = (k_tile * n_tiles + n_tile) * tile_size`` inside exactly that buffer, so a buffer
    of another shape indexes the wrong tile, not an out-of-range offset.
    """
    require_repack_tile_alignment(size_k, size_n)
    expected = (size_k // MARLIN_TILE_K, size_n * MARLIN_TILE_K // PACK_FACTOR_4BIT)
    actual = _shape_tuple(shape)
    if actual != expected:
        raise ValueError(
            f"Marlin repacked weight must be (size_k/{MARLIN_TILE_K}, "
            f"size_n*{MARLIN_TILE_K}/{PACK_FACTOR_4BIT}) = {expected} "
            f"(gptq_marlin_repack.cu:324-326); got {actual}"
        )


def require_act_order_perm(perm, size_k: int) -> None:
    """Act order is one entry per k of the *whole* weight, whose values index k rows.

    Upstream invariant: ``gptq_marlin_repack.cu:329`` -- an empty ``perm`` *is* the no-act-order
    arm (``has_perm = perm.size(0) != 0``) -- and, when it is not empty, the kernel's shared-memory
    loader reads ``perm[first_k + k_id]`` with ``first_k = k_tile_id * target_tile_k_size`` and
    ``k_id`` in ``[0, target_tile_k_size)`` (``:62, :87-90``) and indexes ``b_q_weight_ptr`` with
    ``src_k / pack_factor`` at that value (``:90-96``). So the array must have ``size_k`` entries and
    each entry must be a source k row, i.e. in ``[0, size_k)``. A shorter array is an
    out-of-bounds read on device and an IndexError here; neither is a *layout* error, which is what
    makes a short array worth refusing explicitly.
    """
    if perm is None:
        return
    length = len(perm)
    if length == 0:  # upstream: empty perm means "no act order"
        return
    if length != size_k:
        raise ValueError(
            f"act-order perm must have one entry per k (size_k={size_k}) because the kernel reads "
            f"perm[k_tile_id * {MARLIN_TILE_K} + k_id] directly (gptq_marlin_repack.cu:62,151); "
            f"got {length} entries"
        )
    try:
        lo, hi = int(min(perm)), int(max(perm))
    except (TypeError, ValueError) as exc:  # pragma: no cover - non-numeric perm
        raise ValueError(f"act-order perm must be integer k indices; got {perm!r}") from exc
    if lo < 0 or hi >= size_k:
        raise ValueError(
            f"act-order perm entries are source k rows, so they must lie in [0, {size_k}) "
            f"(gptq_marlin_repack.cu:90-96); got [{lo}, {hi}]"
        )


def require_scale_frame(shape, size_k: int, size_n: int, group_size: int) -> None:
    """Scales are ``(num_groups, size_n)`` with ``num_groups = size_k / group_size``.

    Upstream invariant: ``marlin_permute_scales`` permutes in ``(num_groups, size_n)`` blocks of 64
    -- grouped scales use ``len(scale_perm) == 64`` and channel-wise ones
    ``len(scale_perm_single) == 32`` (``marlin_utils.py:470-480``, :460-467) -- the N multiple is
    upstream's own op gate (``marlin_utils.py:172-190``: ``output_size_per_partition %
    GPTQ_MARLIN_MIN_THREAD_N != 0`` is refused, i.e. ``size_n % 64``), and the group contract
    ``size_k % group_size == 0`` is checked at ``marlin_utils.py:198-202``
    (``verify_group_size_divides_partition``, defined in ``vllm/distributed/utils.py:67``).

    The row count is the load-bearing part: ``reshape((-1, 64))`` succeeds for *any* row count whose
    product is a multiple of 64, so scales that belong to a different K silently permute the wrong
    number of groups and come back with a plausible shape.
    """
    actual = _shape_tuple(shape)
    if size_n % GPTQ_MIN_THREAD_N:
        raise ValueError(
            f"upstream refuses size_n % {GPTQ_MIN_THREAD_N} != 0 for a Marlin projection "
            f"(marlin_utils.py:172-190: GPTQ_MARLIN_MIN_THREAD_N at :32, checked at :179); "
            f"got size_n={size_n}"
        )
    if group_size > 0:
        if size_k % group_size:
            raise ValueError(
                f"size_k={size_k} is not divisible by group_size={group_size} "
                "(marlin_utils.py:198-202, verify_group_size_divides_partition); the group count is "
                "not an integer"
            )
        expected_rows = size_k // group_size
    else:
        # Channel-wise (group_size == -1 or >= size_k): one row of scales per output column.
        expected_rows = 1
    if actual != (expected_rows, size_n):
        raise ValueError(
            f"scales must be (num_groups, size_n) = {(expected_rows, size_n)} for size_k={size_k}, "
            f"size_n={size_n}, group_size={group_size} (marlin_utils.py:470-480); got {actual}"
        )


def require_zero_point_frame(shape, size_n: int, num_groups: int | None = None) -> None:
    """Zero points are ``(num_groups, size_n)``, permuted then packed along N.

    Upstream invariant: ``marlin_zero_points`` permutes ``(num_groups, size_n)`` with the 64-wide
    ``scale_perm``, interleaves N, then hands the array to ``pack_cols``, which asserts
    ``q_w.shape == (size_k, size_n)`` against the row count it was given
    (``marlin_utils.py:512-533``; ``quant_utils.py:892-895``). So the column extent and the
    8-nibble packing are upstream invariants; the row count is too, but only where the caller knows
    it (``num_groups``), because MATE's signature carries the *first* upstream argument, which
    upstream's own callers set to a number of groups (``marlin_utils_test.py:210-212``).
    """
    actual = _shape_tuple(shape)
    if size_n % GPTQ_MIN_THREAD_N:
        raise ValueError(
            f"zero points are permuted in {GPTQ_MIN_THREAD_N}-wide blocks "
            f"(marlin_utils.py:512-522: scale_perm); got size_n={size_n}"
        )
    if size_n % PACK_FACTOR_4BIT:
        raise ValueError(
            f"zero points pack {PACK_FACTOR_4BIT} n-nibbles per word "
            f"(quant_utils.py:895); got size_n={size_n}"
        )
    if not actual or actual[-1] != size_n:
        raise ValueError(
            f"zero points must have size_n={size_n} as their last extent "
            f"(marlin_utils.py:512-533); got {actual}"
        )
    if num_groups is not None and actual != (num_groups, size_n):
        raise ValueError(
            f"zero points must be (num_groups, size_n) = {(num_groups, size_n)} "
            f"(marlin_utils.py:512-533; quant_utils.py:892-895 asserts the row count); got {actual}"
        )


def require_packed_zero_point_frame(shape, num_groups: int, size_n: int) -> None:
    """Packed zero points are ``(num_groups, size_n / 8)``: ``marlin_zero_points`` ends in ``pack_cols``.

    Upstream invariant: ``marlin_zero_points`` packs along N with ``pack_cols``, whose own output
    extent is ``size_n // pack_factor`` (``marlin_utils.py:512-533``; ``quant_utils.py:892-895``).
    The row count is the group count, never ``size_k`` -- MATE's ``linear`` module documents a
    port-time defect that came from exactly that confusion.
    """
    if size_n % PACK_FACTOR_4BIT:
        raise ValueError(
            f"packed zero points carry {PACK_FACTOR_4BIT} n-nibbles per word "
            f"(quant_utils.py:895); got size_n={size_n}"
        )
    actual = _shape_tuple(shape)
    expected = (num_groups, size_n // PACK_FACTOR_4BIT)
    if actual != expected:
        raise ValueError(
            f"packed zero points must be (num_groups, size_n/{PACK_FACTOR_4BIT}) = {expected} "
            f"(marlin_utils.py:512-533 ends in pack_cols, quant_utils.py:892-895); got {actual}"
        )


def require_musa_k_multiple(size_k: int) -> None:
    """MATE's port requirement, not upstream's: the landed K256 mainloop tile holds two K128 blocks.

    Upstream keeps a shape alive by tile-padding K to ``lcm(64 or 128, group_size)``
    (``marlin_utils.py:221-243``); MATE's kernel expresses no partial group, so it refuses instead
    of padding. This is deliberately *not* enforced inside the upstream-mirroring repack functions:
    upstream's repack has no such invariant, and pushing it down would refuse shapes the repack --
    and the byte-parity gates -- deliberately exercise. The op boundary
    (``compat.check_musa_marlin_supports_shape``, called by ``linear.prepare_marlin_w4a16``) is the
    single place that owns it.
    """
    if size_k % MUSA_MARLIN_K_MULTIPLE:
        raise ValueError(
            f"K must be a multiple of {MUSA_MARLIN_K_MULTIPLE} for MATE's K256 tile, got K={size_k}; "
            "upstream would tile-pad K here, which this kernel does not express"
        )
