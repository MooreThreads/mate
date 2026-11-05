# SPDX-License-Identifier: Apache-2.0
"""MATE's repack for the upstream Marlin W4A16 weight layout (the GPTQ one the model uses).

``mate.marlin.reference`` holds a *scalar* transcription of ``gptq_marlin_repack_kernel`` and stays
the oracle. This module is the form MATE ships: the index arithmetic is vectorised per 16x64 tile
and the outer loops are over tiles, which is exactly the grid a device kernel wants — there is no
Python loop over elements. ``tests/test_w4a16_marlin.py`` gates the two against each other byte-for-byte on the map and on
values, so the oracle stays independent evidence rather than a second copy of this code.

Offset equation, 4-bit non-a8bit path (one tile; ``w`` is the unpacked ``[size_k, size_n]`` nibbles)::

    out[k_tile, n_tile * TILE_WORDS + word] |= value << (4 * nib)
    value     = w[k_tile * 16 + k_local, n_tile * 64 + n_local]
    k_local   = 2 * ((word // 4) % 4) + TC_OFFSETS[src % 4]
    n_local   = (word % 4) * 16 + word // 16 + 8 * (src >= 4)
    src       = PACK_IDX[nib]
    PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7),  TC_OFFSETS = (0, 1, 8, 9)
    TILE_WORDS = 128,  word = th_id * 4 + warp_id   (th_id 0..31, warp_id 0..3)

Sources (pinned vLLM 2cf0a6915c, ``csrc/libtorch_stable/quantization/marlin/``):
``gptq_marlin_repack.cu:122-137, 149-162, 196, 200-209``, ``marlin.cuh:30,38-39``. Each
``(word, nib)`` slot is written exactly once, so the OR is order-independent, and the per-tile map is
a permutation of the 1024 ``(k, n)`` slots — asserted, not assumed, by the checker.

That derivation is not taken on trust. ``tests/test_marlin_repack_ground_truth.py`` executes
upstream's *own* ``marlin_weights``/``marlin_permute_weights``/``get_weight_perm`` (extracted from
the checkout, sha256-pinned) plus a from-the-control-flow simulation of the kernel, and requires
bit-identical words from this module; ``mate.marlin.reference`` is a third, independent
transcription. Every entry point checks the frame it is handed through ``mate.marlin.guards``,
because a wrong frame that happens to have the right element count is otherwise silent.

Model path: always this repack. AWQ checkpoints and their zero points are converted to the GPTQ
packed layout at load, so there is no second repack (contract §0 item 7). Values here are unsigned
nibbles; ``uint4b8``'s fused ``-8`` belongs to the dequant, not to the repack. Presenting those codes
to the device kernel is a separate step with its own convention -- see
``mate.marlin.linear.marlin_weight_to_musa``, which must flip the sign bit.
"""

from __future__ import annotations

import numpy as np

from . import guards

TILE_K = 16
TILE_N = 64
PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7)
TC_OFFSETS = (0, 1, 8, 9)
TILE_WORDS = TILE_K * TILE_N // 8  # 128 int32 words per 16x64 int4 tile


def tile_index_arrays():
    """Per-tile ``[TILE_WORDS, 8]`` coordinate arrays: ``(k_local, n_local, word, nib)``.

    Shape-independent by construction: every tile uses the same map, translated by
    ``k_tile * 16`` / ``n_tile * 64``. This is the per-thread/per-word work a kernel would do.
    """
    words = np.arange(TILE_WORDS)
    nib = np.arange(8)
    src = np.asarray(PACK_IDX)[nib]
    k_local = (2 * ((words // 4) % 4))[:, None] + np.asarray(TC_OFFSETS)[src % 4][None, :]
    n_local = ((words % 4) * 16 + words // 16)[:, None] + 8 * (src >= 4)[None, :]
    word = np.broadcast_to(words[:, None], (TILE_WORDS, 8))
    nib = np.broadcast_to(nib[None, :], (TILE_WORDS, 8))
    return k_local, n_local, word, nib


def _require_tiles(size_k: int, size_n: int) -> tuple[int, int]:
    # Upstream's own host-side gate: size_k % 16 and size_n % 64 (gptq_marlin_repack.cu:290-293).
    guards.require_repack_tile_alignment(size_k, size_n)
    # Which axis carries the nibble index: K for GPTQ words, N for AWQ words (quant_utils.py:871,895).
    guards.require_nibble_packing(size_k, size_n, "gptq")
    return size_k // TILE_K, size_n // TILE_N


def resolve_perm(perm, size_k: int) -> np.ndarray | None:
    """Normalise ``perm`` to upstream's *global* k permutation, or ``None`` for the identity.

    Two frames exist under this name and they are not interchangeable:

    * the **op frame** — ``gptq_marlin_repack``'s argument, which the model passes as
      ``layer.g_idx_sort_indices``: one entry per k of the whole GPTQ weight, whose *values* are
      source k rows (``gptq_marlin_repack.cu:62,90-96,151``: ``src_k`` goes straight into
      ``b_q_weight_ptr``). An empty array is upstream's "no act order" (``:329``).
    * the **tile frame** — a permutation of the 16 k positions *within* one 16x64 tile, repeated
      identically in every k tile. Upstream's own *Python* oracle uses a third, wider frame: one
      permutation of a whole tile's 1024 flat ``(k_local, n_local)`` slots
      (``marlin_utils_test.get_weight_perm(4)``, ``marlin_utils_test.py:73-125``, 1024 entries — not
      16, and not a k permutation either: ``marlin_permute_weights`` indexes the *flattened* tile
      with it, ``:33-51``). No upstream CUDA entry point accepts that frame, and no upstream call
      site produces it; it exists here only so the gates can express "the same k order in every
      tile" without inventing a length-``size_k`` array, and
      ``tests/test_w4a16_marlin.py::test_wrapper_perm_frames_and_global_parity`` pins that the
      expansion into the op frame is exactly the per-tile repetition.

    The frames coincide only when ``size_k == TILE_K``, which is exactly why a checker that only
    tried K == 16 could not tell them apart.
    """
    if perm is None:
        return None
    perm = np.asarray(perm, dtype=np.int64).reshape(-1)
    if perm.shape[0] == 0:
        return None
    if perm.shape[0] == size_k:
        guards.require_act_order_perm(perm, size_k)
        return perm
    if perm.shape[0] == TILE_K:
        expanded = np.concatenate(
            [perm + k_tile * TILE_K for k_tile in range(size_k // TILE_K)]
        )
        guards.require_act_order_perm(expanded, size_k)
        return expanded
    raise ValueError(
        f"perm has {perm.shape[0]} entries; expected {size_k} (the op frame: one per k, e.g. "
        f"argsort(g_idx)) or {TILE_K} (the tile frame: one permutation of the 16 k positions "
        f"repeated in every k tile)"
    )


def unpack_rows(qweight: np.ndarray, size_k: int, size_n: int) -> np.ndarray:
    """GPTQ input layout ``[size_k/8, size_n]`` int words -> ``[size_k, size_n]`` nibble values.

    The shape is checked, not reshaped into: ``(size_k/8, size_n)`` is upstream's own gate
    (``gptq_marlin_repack.cu:300-305``), and a transposed ``(size_n, size_k/8)`` buffer has the same
    element count, so an unchecked ``reshape`` would quietly unpack the wrong axis.
    """
    guards.require_gptq_word_shape(qweight.shape, size_k, size_n)
    w = np.asarray(qweight, dtype=np.int64)
    vals = (w[:, None, :] >> (4 * np.arange(8))[None, :, None]) & 0xF  # [k/8, nib, n]
    return vals.reshape(size_k, size_n)


def pack_rows(w: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_rows`: ``[size_k, size_n]`` nibbles -> ``[size_k/8, size_n]``.

    Only the packed axis is constrained: upstream's ``pack_rows`` asserts ``size_k % pack_factor ==
    0`` and nothing about N (``quant_utils.py:752-763``), and a stricter check here would refuse
    shapes that upstream's own packer accepts.
    """
    w = np.asarray(w, dtype=np.int64)
    guards.require_nibble_packing(w.shape[0], w.shape[1], "gptq")
    out = np.zeros((w.shape[0] // 8, w.shape[1]), dtype=np.int64)
    for i in range(8):
        out |= (w[i::8, :] & 0xF) << (4 * i)
    return out


def pack_cols(w: np.ndarray) -> np.ndarray:
    """AWQ input layout ``[size_k, size_n]`` -> ``[size_k, size_n/8]`` (nibbles of word n // 8).

    N is the packed axis here, so it is N that must be a multiple of the 8-nibble word
    (``quant_utils.py:892-895``: AWQ packs along N, unlike GPTQ's K).
    """
    w = np.asarray(w, dtype=np.int64)
    guards.require_nibble_packing(w.shape[0], w.shape[1], "awq")
    out = np.zeros((w.shape[0], w.shape[1] // 8), dtype=np.int64)
    for i in range(8):
        out |= (w[:, i::8] & 0xF) << (4 * i)
    return out


def repack_gptq(
    qweight: np.ndarray, size_k: int, size_n: int, perm: np.ndarray | None = None
) -> np.ndarray:
    """``[size_k/8, size_n]`` GPTQ words -> Marlin ``[size_k/16, 2*size_n]`` int32.

    ``perm`` is the act-order / ``g_idx`` order: destination k-slot ``k`` of a k-tile reads source k
    ``perm[k]`` — a pure k-axis gather inside the tile (``gptq_marlin_repack.cu:61-70, 146-162``).
    """
    kt, nt = _require_tiles(size_k, size_n)
    perm = resolve_perm(perm, size_k)
    w = unpack_rows(qweight, size_k, size_n)
    k_local, n_local, word, nib = tile_index_arrays()
    out = np.zeros((kt, 2 * size_n), dtype=np.int64)
    for k_tile in range(kt):
        k_dst = k_tile * TILE_K + k_local
        # ``perm`` is upstream's *global* k permutation (g_idx_sort_indices, one entry per k, as
        # gptq_marlin_repack.cu:146-151 uses it: src_k indexes the whole GPTQ weight), so it is a
        # direct lookup -- not a per-tile offset. Getting this wrong is invisible at K == TILE_K,
        # which is why the gate below compares against the oracle with a full-K perm at K > 16.
        k_src = k_dst if perm is None else np.asarray(perm, dtype=np.int64)[k_dst]
        for n_tile in range(nt):
            vals = w[k_src, n_tile * TILE_N + n_local]
            np.bitwise_or.at(out[k_tile], n_tile * TILE_WORDS + word, vals << (4 * nib))
    return out


def unpack_gptq(
    repacked: np.ndarray, size_k: int, size_n: int, perm: np.ndarray | None = None
) -> np.ndarray:
    """Inverse of :func:`repack_gptq`: ``[size_k/16, 2*size_n]`` -> ``[size_k/8, size_n]`` words.

    The Marlin buffer's own extent is checked, not reshaped into: ``(size_k/16, 2*size_n)`` is what
    upstream allocates and what its ``out_offset`` arithmetic addresses
    (``gptq_marlin_repack.cu:324-326``). A buffer with the K and N tile axes swapped has a
    different shape and would index another tile entirely.
    """
    kt, nt = _require_tiles(size_k, size_n)
    perm = resolve_perm(perm, size_k)
    guards.require_repacked_shape(repacked.shape, size_k, size_n)
    k_local, n_local, word, nib = tile_index_arrays()
    r = np.asarray(repacked, dtype=np.int64)
    w = np.zeros((size_k, size_n), dtype=np.int64)
    for k_tile in range(kt):
        k_dst = k_tile * TILE_K + k_local
        # inverse of the above: the value written into destination k came from source perm[k]
        k_src = k_dst if perm is None else np.asarray(perm, dtype=np.int64)[k_dst]
        for n_tile in range(nt):
            vals = (r[k_tile, n_tile * TILE_WORDS + word] >> (4 * nib)) & 0xF
            w[k_src, n_tile * TILE_N + n_local] = vals  # exactly-once slots
    return pack_rows(w)


def gptq_marlin_repack(qweight, perm, size_k: int, size_n: int, num_bits: int = 4):
    """Upstream's ``gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)``.

    Same name, same argument order, same result as the NVIDIA op — MATE's implementation instead of a
    CUDA kernel. ``perm`` is upstream's act-order / ``g_idx`` order (``argsort(g_idx)``); pass an
    ``arange`` when there is none, exactly as upstream's call sites do.

    ``num_bits`` must be 4: MATE's kernel is int4 x fp16, so the 8-bit arm is refused rather than
    approximated. Sizes must already be tile-aligned; call
    :func:`mate.marlin.compat.marlin_padded_nk` and the pad helpers first, as upstream does.
    """
    if num_bits != 4:
        raise NotImplementedError(
            f"MATE's Marlin wrapper is int4 x fp16 only, so num_bits must be 4; got {num_bits}"
        )
    # resolve_perm accepts both upstream frames and rejects anything else with a reason
    return repack_gptq(np.asarray(qweight), size_k, size_n, resolve_perm(perm, size_k))


def awq_marlin_repack(qweight, size_k: int, size_n: int, num_bits: int = 4):
    """Upstream's ``awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)``.

    Kept for API compatibility and for the gates, and deliberately implemented from the reference
    transcription rather than duplicated here: **this is not on the serving path.** AWQ checkpoints
    are converted into the GPTQ packed layout at load time, so ``gptq_marlin_repack`` above is the
    only repack the model ever runs. If that ever changes, this function must move to the shipping
    form and be gated the same way.
    """
    if num_bits != 4:
        raise NotImplementedError(
            f"MATE's Marlin wrapper is int4 x fp16 only, so num_bits must be 4; got {num_bits}"
        )
    from . import reference

    # AWQ words are packed along N, so this is (size_k, size_n/8) -- upstream's own gate
    # (awq_marlin_repack.cu:242-247). The AWQ kernel has no perm argument at all (:16-17).
    qweight = np.asarray(qweight)
    guards.require_awq_word_shape(qweight.shape, size_k, size_n)
    guards.require_repack_tile_alignment(size_k, size_n)
    return reference.cuda_repack_awq(qweight, size_k, size_n)
