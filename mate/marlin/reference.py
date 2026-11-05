# SPDX-License-Identifier: Apache-2.0
"""Reference (numpy-only) implementation of the upstream vLLM Marlin W4A16 layouts.

Target: upstream vLLM pin ``2cf0a6915ce544dc493a0990f2ea38d81601128a`` (v0.28.0-dev),
Marlin sources under ``csrc/libtorch_stable/quantization/marlin/``.

Why numpy only: the gates that guard the shipping implementation live in
``tests/test_w4a16_marlin.py`` and must run wherever numpy exists, including a host with no torch.
Nothing at runtime imports this module; it is evidence, and every routine below is a transcription
of a specific upstream function whose docstring carries its ``file:line``.

Two independent implementations live here on purpose:

* ``ref_*``  -- transcription of the *Python* reference that vLLM's own tests use as the
  oracle (``vllm/model_executor/layers/quantization/utils/marlin_utils_test.py``).
* ``cuda_repack_*`` -- transcription of the *CUDA* repack kernel index maps
  (``csrc/libtorch_stable/quantization/marlin/{gptq,awq}_marlin_repack.cu``).

They were written from the two different sources, so agreement between them (asserted by
``tests/test_w4a16_marlin.py``) is real evidence, not a tautology.
"""

from __future__ import annotations

import numpy as np

from . import guards

# --- constants ---------------------------------------------------------------

GPTQ_MARLIN_TILE = 16  # marlin_utils.py:31
PACK_FACTOR_4BIT = 8  # 32 / num_bits
TILE_K = 16  # csrc .../marlin.cuh:30,38  (tile_size = 16, tile_k_size = tile_size)
TILE_N = 64  # csrc .../marlin.cuh:39     (tile_n_size = tile_k_size * 4)
TILE_WORDS_4BIT = TILE_K * TILE_N // PACK_FACTOR_4BIT  # 128, gptq_marlin_repack.cu:194-196

# marlin_utils_test.py:93-106 (4-bit, non-a8bit branch) -- array-layout order
_LAYOUT_PERM_PACK_IDX = [0, 2, 4, 6, 1, 3, 5, 7]  # gptq_marlin_repack.cu:201
_TC_OFFSETS = (0, 1, 8, 9)  # gptq_marlin_repack.cu:132
_AWQ_UNDO_PACK = [0, 4, 1, 5, 2, 6, 3, 7]  # awq_marlin_repack.cu:107
# auto_awq.py:77 -- the same 8-entry order, independently stated by the AWQ loader
_AWQ_REVERSE_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


# --- input packers (what a checkpoint / loader hands to the repack) -----------


def pack_rows(w: np.ndarray) -> np.ndarray:
    """GPTQ input layout ``[size_k, size_n] -> [size_k/8, size_n]``.

    ``quant_utils.pack_rows`` (quant_utils.py:862-881): row ``k`` lands in word
    ``k // 8`` at nibble ``k % 8`` (LSB first).
    """
    k, n = w.shape
    # Upstream's assert, with its own citation: quant_utils.py:862-864 (size_k % pack_factor).
    guards.require_nibble_packing(k, n, "gptq")
    assert k % PACK_FACTOR_4BIT == 0
    out = np.zeros((k // PACK_FACTOR_4BIT, n), dtype=np.int64)
    for i in range(PACK_FACTOR_4BIT):
        out |= (w[i::PACK_FACTOR_4BIT, :].astype(np.int64) & 0xF) << (4 * i)
    return out


def unpack_rows(packed: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """Inverse of :func:`pack_rows` -- ``[size_k/8, size_n] -> [size_k, size_n]``.

    The frame is checked before the unpack, because ``[size_n, size_k/8]`` holds the same number of
    words: upstream's ``gptq_marlin_repack`` checks the two extents separately
    (``gptq_marlin_repack.cu:300-305``).
    """
    guards.require_gptq_word_shape(packed.shape, size_k, size_n)
    out = np.zeros((size_k, size_n), dtype=np.int64)
    for i in range(PACK_FACTOR_4BIT):
        out[i::PACK_FACTOR_4BIT, :] = (packed >> (4 * i)) & 0xF
    return out


def pack_cols(w: np.ndarray) -> np.ndarray:
    """AWQ input layout ``[size_k, size_n] -> [size_k, size_n/8]``.

    ``quant_utils.pack_cols`` (quant_utils.py:886-908): column ``n`` lands in word
    ``n // 8`` at nibble ``n % 8``.
    """
    k, n = w.shape
    # Upstream's assert, with its own citation: quant_utils.py:892-895 (size_n % pack_factor).
    guards.require_nibble_packing(k, n, "awq")
    assert n % PACK_FACTOR_4BIT == 0
    out = np.zeros((k, n // PACK_FACTOR_4BIT), dtype=np.int64)
    for i in range(PACK_FACTOR_4BIT):
        out |= (w[:, i::PACK_FACTOR_4BIT].astype(np.int64) & 0xF) << (4 * i)
    return out


def unpack_cols(packed: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """Inverse of :func:`pack_cols` -- ``[size_k, size_n/8] -> [size_k, size_n]``.

    Upstream's ``unpack_cols`` asserts exactly this frame
    (``quant_utils.py:912-925``: ``packed_q_w.shape == (size_k, size_n // pack_factor)``).
    """
    guards.require_awq_word_shape(packed.shape, size_k, size_n)
    out = np.zeros((size_k, size_n), dtype=np.int64)
    for i in range(PACK_FACTOR_4BIT):
        out[:, i::PACK_FACTOR_4BIT] = (packed >> (4 * i)) & 0xF
    return out


def awq_pack(w: np.ndarray) -> np.ndarray:
    """AWQ checkpoint packing: interleave the column dim, then pack along it.

    ``quant_utils.awq_pack`` (quant_utils.py:952-...) applies
    ``interleave = [0, 2, 4, 6, 1, 3, 5, 7]`` to the ``size_n`` axis before
    ``pack_cols``; ``awq_marlin_repack.cu:104-112`` undoes exactly that with
    ``undo_pack = {0, 4, 1, 5, 2, 6, 3, 7}``, and ``auto_awq.py:77`` states the same
    order as ``_REVERSE_AWQ_PACK_ORDER``.
    """
    k, n = w.shape
    interleaved = w.reshape(k, -1, len(_LAYOUT_PERM_PACK_IDX))[
        :, :, _LAYOUT_PERM_PACK_IDX
    ].reshape(k, n)
    return pack_cols(interleaved)


# --- the Python reference the upstream tests use as the oracle ---------------


def get_weight_perm(num_bits: int = 4, is_a_8bit: bool = False) -> np.ndarray:
    """``marlin_utils_test.get_weight_perm`` (marlin_utils_test.py:73-125).

    Returns a permutation of one full 16x64 tile flattened in ``(n_tile, k, n)`` order
    (1024 entries for 4-bit, non-a8bit).
    """
    assert num_bits == 4, "only the 4-bit path is transcribed"
    assert not is_a_8bit
    perm_list: list[int] = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return perm


def ref_permute_weights(
    q_w: np.ndarray, size_k: int, size_n: int, perm: np.ndarray, is_a_8bit: bool = False
) -> np.ndarray:
    """``marlin_utils_test.marlin_permute_weights`` (marlin_utils_test.py:33-51)."""
    tile = GPTQ_MARLIN_TILE
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0 and size_n % tile == 0
    if is_a_8bit:
        q_w = q_w.reshape((size_k // (tile * 2), tile * 2, size_n // tile, tile))
    else:
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.transpose((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))
    q_w = q_w.reshape((-1, perm.size))[:, perm].reshape(q_w.shape)
    return q_w


def ref_marlin_weights(
    q_w: np.ndarray, size_k: int, size_n: int, perm: np.ndarray, is_a_8bit: bool = False
) -> np.ndarray:
    """``marlin_utils_test.marlin_weights`` (marlin_utils_test.py:54-70).

    Replicates the tile order and then packs 8 consecutive flat elements per word
    (LSB first) -- this *is* the byte layout the MMA consumes.
    """
    q_w = ref_permute_weights(q_w, size_k, size_n, perm, is_a_8bit)
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // PACK_FACTOR_4BIT), dtype=np.int64)
    for i in range(PACK_FACTOR_4BIT):
        q_packed |= (q_w[:, i::PACK_FACTOR_4BIT].astype(np.int64) & 0xF) << (4 * i)
    return q_packed


# --- independent transcription of the CUDA repack kernels -------------------


def cuda_tile_placement(size_k: int, size_n: int, num_bits: int = 4):
    """Where each logical ``(k, n)`` lands, from the CUDA kernel's thread indexing.

    Yields ``(k_tile, n_tile, word_in_tile, nibble, k_local, n_local, k_off, n_hi)``:
    ``k_off`` is the index into ``tc_offsets`` and ``n_hi`` is the ``+8`` half.

    Sources: ``gptq_marlin_repack.cu:122-137`` (thread decomposition), ``:132``
    (``tc_offsets = {0,1,8,9}``), ``:149-162`` (B-fragment halves), ``:200-209``
    (the ``pack_idx = {0,2,4,6,1,3,5,7}`` write) and ``:196`` (``out_offset``).
    """
    assert size_k % TILE_K == 0 and size_n % TILE_N == 0
    assert num_bits == 4
    for k_tile in range(size_k // TILE_K):
        for n_tile in range(size_n // TILE_N):
            for warp_id in range(4):
                for th_id in range(32):
                    tc_col = th_id // 4  # gptq_marlin_repack.cu:129
                    tc_row = (th_id % 4) * 2  # :130
                    cur_n = warp_id * 16 + tc_col  # :134
                    word = th_id * 4 + warp_id  # :209
                    for nib_idx, src_idx in enumerate(_LAYOUT_PERM_PACK_IDX):
                        n_hi = 1 if src_idx >= 4 else 0  # vals[4..7] are the +8 half
                        k_off = _TC_OFFSETS[src_idx % 4]
                        yield (
                            k_tile,
                            n_tile,
                            word,
                            nib_idx,
                            tc_row + k_off,
                            cur_n + (8 if n_hi else 0),
                            src_idx % 4,
                            n_hi,
                        )


def _global_perm(perm: np.ndarray, size_k: int) -> np.ndarray | None:
    """Expand either accepted perm frame to the global one; see ``layout.resolve_perm``.

    An empty array is upstream's "no act order" (``gptq_marlin_repack.cu:329``:
    ``has_perm = perm.size(0) != 0``), so it resolves to ``None`` here rather than raising.
    """
    perm = np.asarray(perm, dtype=np.int64).reshape(-1)
    if perm.shape[0] == 0:
        return None
    if perm.shape[0] == size_k:
        guards.require_act_order_perm(perm, size_k)
        return perm
    if perm.shape[0] == TILE_K:
        expanded = np.concatenate([perm + t * TILE_K for t in range(size_k // TILE_K)])
        guards.require_act_order_perm(expanded, size_k)
        return expanded
    raise ValueError(f"perm has {perm.shape[0]} entries; expected {size_k} or {TILE_K}")


def cuda_repack_gptq(
    b_q_weight: np.ndarray, size_k: int, size_n: int, perm: np.ndarray | None = None
) -> np.ndarray:
    """Transcription of ``gptq_marlin_repack_kernel`` (4-bit, non-a8bit).

    ``b_q_weight`` is the GPTQ-layout packed weight ``[size_k/8, size_n]``.
    ``perm`` is the act-order / ``g_idx`` sort order, in either accepted frame (see
    ``layout.resolve_perm``): the op frame is ``argsort(g_idx)`` with one entry per k, and MATE's own
    tile frame is one permutation of the 16 k positions repeated in every k tile. (Upstream's
    ``get_weight_perm(4)`` is a *third* frame -- 1024 entries over a tile's flattened
    ``(k_local, n_local)`` slots -- and is what :func:`ref_marlin_weights` uses as the oracle; it is
    not a k permutation and no CUDA entry point accepts it.) Its value is used *directly* as a global
    k, because that is what the kernel does -- ``src_k`` goes straight into ``b_q_weight_ptr``
    (``gptq_marlin_repack.cu:90-97, 146-151``) with no tile offset added. The earlier transcription
    here added ``k_tile * TILE_K`` on top, which is indistinguishable from the correct result at
    K == 16 and wrong for every larger K; the two-frame resolver is what keeps that honest.
    """
    guards.require_gptq_word_shape(b_q_weight.shape, size_k, size_n)  # gptq_marlin_repack.cu:300-305
    guards.require_repack_tile_alignment(size_k, size_n)  # gptq_marlin_repack.cu:290-293
    w = unpack_rows(b_q_weight, size_k, size_n)
    out = np.zeros((size_k // TILE_K, 2 * size_n), dtype=np.int64)
    perm_global = None if perm is None else _global_perm(perm, size_k)
    for k_tile, n_tile, word, nib, k_local, n_local, _, _ in cuda_tile_placement(
        size_k, size_n
    ):
        k_dst = k_tile * TILE_K + k_local
        src_k = k_dst if perm_global is None else int(perm_global[k_dst])
        val = w[src_k, n_tile * TILE_N + n_local]
        out[k_tile, n_tile * TILE_WORDS_4BIT + word] |= int(val) << (4 * nib)
    return out


def cuda_repack_awq(b_q_weight: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """Transcription of ``awq_marlin_repack_kernel`` (4-bit, non-a8bit).

    ``b_q_weight`` is the AWQ-layout packed weight ``[size_k, size_n/8]`` whose nibble
    order is the AWQ interleave. ``awq_marlin_repack.cu:104-112`` undoes that order before
    reading, so the *logical* values and the write pattern are identical to
    :func:`cuda_repack_gptq` (``:142-157`` mirrors ``gptq_marlin_repack.cu:194-209``).
    """
    guards.require_awq_word_shape(b_q_weight.shape, size_k, size_n)  # awq_marlin_repack.cu:242-247
    guards.require_repack_tile_alignment(size_k, size_n)  # awq_marlin_repack.cu:232-235
    w = unpack_cols(b_q_weight, size_k, size_n)
    interleave = np.array(_LAYOUT_PERM_PACK_IDX)
    w = w.reshape(size_k, -1, len(interleave))[:, :, np.argsort(interleave)].reshape(
        size_k, size_n
    )
    out = np.zeros((size_k // TILE_K, 2 * size_n), dtype=np.int64)
    for k_tile, n_tile, word, nib, k_local, n_local, _, _ in cuda_tile_placement(
        size_k, size_n
    ):
        val = w[k_tile * TILE_K + k_local, n_tile * TILE_N + n_local]
        out[k_tile, n_tile * TILE_WORDS_4BIT + word] |= int(val) << (4 * nib)
    return out


def unpack_marlin_tiles(repacked: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """Inverse of :func:`cuda_repack_gptq` (no perm): Marlin words -> ``[size_k, size_n]``.

    Independent of the forward path: it walks the *same* documented placement map and
    writes the nibble back to its logical ``(k, n)``.
    """
    w = np.zeros((size_k, size_n), dtype=np.int64)
    for k_tile, n_tile, word, nib, k_local, n_local, _, _ in cuda_tile_placement(
        size_k, size_n
    ):
        worder = repacked[k_tile, n_tile * TILE_WORDS_4BIT + word]
        w[k_tile * TILE_K + k_local, n_tile * TILE_N + n_local] = (worder >> (4 * nib)) & 0xF
    return w


# --- B-fragment decode (the "fragment algebra" the MUSA port must reproduce) --


def dequant_fragment_halves(word: int) -> tuple[int, int, int, int]:
    """Nibble -> B-fragment register order for one Marlin word (4-bit path).

    ``dequant.h:121-142`` builds the fragment with two LOP3 masks: ``LO = 0x000f000f``
    keeps nibbles 0 and 4, ``HI = 0x00f000f0`` keeps nibbles 1 and 5, and the ``-8``
    symmetric zero point is fused into ``SUB = 0x64086408`` / ``MUL = 0x2c002c00``
    (``+`` ``ADD = 0xd480d480``). The two mask pairs land in the two 32-bit registers of
    the ``m16n8k16`` B fragment ``{b0, b1}`` / ``{b2, b3}``
    (``marlin_template.h:1286-1289`` passes ``b[0]`` and ``b[1]``), i.e. register 0 gets
    ``(nibble0, nibble4)`` and register 1 gets ``(nibble1, nibble5)``.

    Returns ``(reg0.x, reg0.y, reg1.x, reg1.y)`` for the *n* column of the word;
    the caller gets the ``n + 8`` column from ``word >> 8``
    (``marlin_template.h:1235-1241``).
    """
    n0 = word & 0xF
    n1 = (word >> 4) & 0xF
    n4 = (word >> 16) & 0xF
    n5 = (word >> 20) & 0xF
    return (n0, n4, n1, n5)


def fragment_k_map() -> dict[str, tuple[int, int]]:
    """Which ``(k_offset, n_offset)`` each B-fragment element holds for a word.

    ``k`` uses the tensor-core layout (``marlin_template.h:1041`` "Due to tensor-core
    layout for fp16 B matrix": ``k = 2 * (th_id % 4) + {0,1}``, ``+8`` for the second
    register), ``n`` is ``th_id // 4`` for the low nibble group and ``+8`` for the high
    one (``marlin_template.h:1261-1274`` reads ``frag_s`` half as ``x``/``y`` per n).
    """
    return {
        "b0": (0, 0),
        "b1": (1, 0),
        "b2": (8, 0),
        "b3": (9, 0),
        "b0_hi": (0, 8),
        "b1_hi": (1, 8),
        "b2_hi": (8, 8),
        "b3_hi": (9, 8),
    }


# --- scales and zero points --------------------------------------------------


def get_scale_perms() -> tuple[list[int], list[int]]:
    """``marlin_utils.get_scale_perms`` (marlin_utils.py:460-467)."""
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scale_perm_single = [
        2 * i + j for i in range(4) for j in [0, 1, 8, 9, 16, 17, 24, 25]
    ]
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: np.ndarray, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> np.ndarray:
    """``marlin_utils.marlin_permute_scales`` (marlin_utils.py:470-480).

    Grouped quantization (``group_size < size_k`` and ``group_size != -1``) uses the
    8x8-transposed ``scale_perm``; per-channel (``group_size == -1``) uses
    ``scale_perm_single``. ``marlin_template.h:658-659`` states why the two differ: the
    kernel scales a ``half2`` tile column-major for grouped and row-major for column-wise.
    """
    scale_perm, scale_perm_single = get_scale_perms()
    guards.require_scale_frame(s.shape, size_k, size_n, group_size)  # marlin_utils.py:470-480
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n))


def marlin_unpermute_scales(
    s: np.ndarray, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> np.ndarray:
    """Inverse of :func:`marlin_permute_scales` -- the direction the *host* path needs.

    Upstream does not have this function because its kernel consumes the permuted frame directly.
    MATE's kernel takes plain ``(N, G)`` row-major scales, so the conversion is explicit. Both
    branches must invert the matching branch above: getting this backwards is silent, because a
    wrong inverse is still shape-correct.
    """
    scale_perm, scale_perm_single = get_scale_perms()
    guards.require_scale_frame(s.shape, size_k, size_n, group_size)  # inverse of the same frame
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        width, perm = len(scale_perm), scale_perm
    else:
        width, perm = len(scale_perm_single), scale_perm_single
    rows = s.reshape((-1, width))[:, np.argsort(np.array(perm))]
    return rows.reshape((-1, size_n))


def marlin_zero_points(
    zp: np.ndarray, size_k: int, size_n: int, num_bits: int = 4, is_a_8bit: bool = False
) -> np.ndarray:
    """``marlin_utils.marlin_zero_points`` (marlin_utils.py:512-533).

    Always ``scale_perm`` (never the "single" perm: zero points are applied on every
    MMA), then the 4-bit interleave ``[0,2,4,6,1,3,5,7]``, then word packing along N
    (``pack_cols``) -> ``[num_groups, size_n/8]``.
    """
    scale_perm, _ = get_scale_perms()
    guards.require_zero_point_frame(zp.shape, size_n)  # marlin_utils.py:512-533
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]
    interleave = np.array(_LAYOUT_PERM_PACK_IDX)
    if not is_a_8bit:
        zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n))
    return pack_cols(zp)


def unpack_marlin_zero_points(
    packed: np.ndarray, size_k: int, size_n: int, num_bits: int = 4
) -> np.ndarray:
    """Inverse of :func:`marlin_zero_points` (grouped, non-a8bit).

    ``size_k`` is unused here for the same reason it is in :func:`marlin_zero_points`: upstream's
    first argument is the ``size_k`` slot of ``pack_cols``, which its own callers fill with a group
    count (``marlin_utils_test.py:210-212``). The row count is therefore read from the tensor, and
    the tensor's frame is checked instead (``quant_utils.py:892-895``).
    """
    guards.require_packed_zero_point_frame(packed.shape, packed.shape[0], size_n)
    zp = unpack_cols(packed, packed.shape[0], size_n)
    interleave = np.array(_LAYOUT_PERM_PACK_IDX)
    zp = zp.reshape((-1, len(interleave)))[:, np.argsort(interleave)].ravel()
    zp = zp.reshape((-1, size_n))
    scale_perm, _ = get_scale_perms()
    return zp.reshape((-1, len(scale_perm)))[:, np.argsort(scale_perm)].reshape(
        (-1, size_n)
    )


def marlin_padded_nk(size_n: int, size_k: int, group_size: int = -1) -> tuple[int, int]:
    """``marlin_utils.marlin_padded_nk`` (marlin_utils.py:221-243).

    Marlin needs ``(n % 64, k % 128)`` or ``(n % 128, k % 64)``; K stays divisible by
    ``group_size``. The cheaper family (by padded area, then by side sum) wins.
    """

    def round_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    group = group_size if group_size > 0 else 1
    candidates = (
        (round_up(size_n, 64), round_up(size_k, np.lcm(128, group))),
        (round_up(size_n, 128), round_up(size_k, np.lcm(64, group))),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def marlin_repacked_nk(qweight_shape: tuple[int, int], num_bits: int = 4):
    """``marlin_utils.marlin_repacked_nk`` (marlin_utils.py:246-252), on a shape."""
    size_k = qweight_shape[0] * GPTQ_MARLIN_TILE
    size_n = qweight_shape[1] * PACK_FACTOR_4BIT // GPTQ_MARLIN_TILE
    return size_n, size_k
