# SPDX-License-Identifier: Apache-2.0
"""Marlin-frame to MATE-kernel-frame conversion, and the compute entry the consumer calls.

Read this first if you are writing the vllm-musa side. **What the consumer calls:**

    from mate.marlin import prepare_marlin_w4a16, apply_gptq_marlin_linear

    prepared = prepare_marlin_w4a16(
        qweight,        # upstream gptq_marlin_repack output, [padded_k/16, 2*padded_n] int32
        scales,         # upstream marlin_permute_scales output, (padded_k/group, padded_n)
        zero_points,    # upstream marlin_zero_points output, or None for symmetric
        size_k,         # logical K (before tile padding)
        size_n,         # logical N (before tile padding)
        group_size=128, # MATE's kernel is structurally group 128
    )
    out = apply_gptq_marlin_linear(input_a, prepared, bias=bias)

``prepared`` carries the kernel-ready tensors and the extents; it is a plain dataclass with no
behaviour, so the consumer can log it. Load-time work happens inside ``prepare_marlin_w4a16`` and
nowhere else, which is the point: the serving path must not do conversion per token.

The consumer hands over upstream's **output**, not raw GPTQ tensors — by the time vLLM's Marlin path
reaches a kernel it has already run the load-time utilities. So every direction below is an
*inverse* of an upstream layout, and each one says which direction the host path uses:

===========================  ==============================  ==========================
upstream (what arrives)      MATE kernel (what is needed)     function
===========================  ==============================  ==========================
``[K/16, 2N]`` int32 words   ``(N, K//2)`` int8, even k low   :func:`marlin_weight_to_musa`
permuted scales ``(G, N)``   plain ``(N, G)``, fp16           :func:`marlin_scales_to_musa`
packed + permuted zp         plain ``(N, G)``, fp16 or None   :func:`marlin_zero_points_to_musa`
===========================  ==============================  ==========================

Why the weight needs a real conversion and not a stride trick: hardware measurement on the existing
MATE kernel showed its int4 B operand is consumed as **consecutive k**, even k in the low nibble, odd
k in the high nibble — while upstream's words are a 16x64 tile map. On top of that the two sides
disagree about what a nibble *means* (upstream: GPTQ ``uint4b8``; MATE's atom: two's complement, which
the declared frame reconciles by ``u ^ 0x8``); see :func:`marlin_weight_to_musa`. No TME stride
descriptor expresses either difference, so route B pays one load-time pass.

Act order (``g_idx``): supported by *materialisation*, not by a kernel feature. The weight is repacked
with ``perm=g_idx_sort_indices`` (upstream's own convention), which puts the k rows in group order;
the per-group scales need no reordering, because they are indexed by group rather than by k. The
activation has to be gathered with the *same* permutation, so ``prepare_marlin_w4a16`` takes that
permutation, stores it in the frame, and ``apply_gptq_marlin_linear`` gathers the input columns with
it before the GEMM. A ``g_idx`` whose sorted groups are not contiguous runs of ``group_size`` is
refused, because no per-group kernel can serve it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # lazy by design, as in the rest of the package
    import torch
except ImportError:  # pragma: no cover - exercised on the numpy-only gate host
    torch = None  # type: ignore[assignment]

from . import compat
from . import guards
from . import quant

WEIGHT_B_DTYPE_INT8 = "int8"


@dataclass(frozen=True)
class MarlinW4A16KernelFrame:
    """Kernel-ready operands plus the extents the caller needs to interpret the output.

    ``padded_n`` may exceed ``logical_n``: MATE's kernel accepts a tile-padded N (padding works
    through zero *scales*, so the extra columns compute to zero), and the caller slices the output to
    ``logical_n``. ``padded_k`` never exceeds ``size_k`` in practice — under group 128 both upstream
    candidate families round K up to ``size_k`` itself — but the field is here so nobody has to
    assume it.

    ``input_perm`` is the act-order k permutation the weight rows were stored in, or ``None``. When it
    is set, ``apply_gptq_marlin_linear`` gathers the activation columns with it.
    """

    weight_b: "torch.Tensor"  # (padded_n, padded_k//2) int8
    scale_b: "torch.Tensor"  # (padded_n, padded_k//group_size) fp16
    zero_points_b: "torch.Tensor | None"  # (padded_n, padded_k//group_size) fp16, or None
    scale_b_padded: "torch.Tensor"  # scale_b, group dim padded to an even count
    correction_operand: "torch.Tensor | None"  # folded zero-point correction, or None when symmetric
    logical_n: int
    padded_n: int
    padded_k: int
    group_size: int
    has_zero_points: bool
    input_perm: "torch.Tensor | None" = None  # (padded_k,) int64 act-order gather, or None
    scale_t: "torch.Tensor | None" = None  # large-M scale operand, see prepare_w4a16_prefill_operand

    @property
    def padded(self) -> bool:
        return self.padded_n != self.logical_n

    def unpad_output(self, output):
        """Strip the padded N columns; a no-op when nothing was padded."""
        # Deliberately unconditional. Dynamo cannot fold a python-level predicate inside the
        # fullgraph capture the serving path uses: it constant-folds `self.padded()` and then
        # CALLS the folded constant, which fails as "Dynamo does not know how to trace the
        # function False". With logical_n == padded_n the slice below is already the identity, so
        # no condition is needed in the traced path at all.
        return compat.marlin_unpad_output(output, self.logical_n, self.padded_n)


def marlin_weight_to_musa(
    repacked, size_k: int, size_n: int, num_bits: int = 4, biased_codes: bool = True
) -> np.ndarray:
    """Upstream ``[size_k/16, 2*size_n]`` int32 words -> MATE ``(size_n, size_k//2)`` int8.

    Two layouts differ, and so do two *conventions* about what a nibble means; both have to be handled
    here, and only the second one is subtle.

    Layout: consecutive k, even k in the low nibble, odd k in the high nibble, against upstream's
    16x64 tile map. That is index arithmetic.

    Convention, and the reason for ``^ 0x8``. MATE's declared operand frame is *biased*: the
    documented dequant is ``w = scale * (q + nibble_bias - zero_point)`` and the packed-weight contract
    says each nibble "already holds the signed INT4 value ``unsigned_nibble - nibble_bias``"
    (``mate/gemm.py:1503-1507``, ``mate/jit/gemm/dense_w4a16_gemm.py:28-31,46-49``), with
    ``nibble_bias = 8`` and a symmetric checkpoint's zero point equal to it
    (``tests/test_dense_w4a16_gemm.py:42-47``: "GPTQ uint4b8 decodes as ``nibble - 8`` with no further
    bias", and the symmetric correction must vanish). So the atom is expected to read a presented
    nibble ``v`` as ``v - 8``. The atom actually reads two's complement -- measured on hardware:
    code 0 reads 0 and code 8 reads -8, which is why the neighbouring ``-8`` for code 8 was mistaken
    for ``u - 8``: that identity holds only *at* ``u == 8``. Upstream's words carry GPTQ ``uint4b8``
    codes ``u``, whose value is ``u - 8``, so presenting them verbatim is wrong for the whole negative
    half by ``+8 * scale`` per element -- a uniform error that no shape-based comparison catches, and
    one that shows up as exactly ``8 * scale * sum_k(a)`` per group in the R3 sensitivity gate.
    The unique presentation satisfying the declared frame is ``v = u ^ 0x8``, since
    ``s(u ^ 0x8) = u - 8`` for all 16 codes. That identity is asserted as a gate, not trusted.

    ``biased_codes=False`` is for words that are already in MATE's frame (two's-complement patterns
    whose signed read is the dequantized integer), e.g. a MATE-native packer; it is not a tolerance
    knob. Returns numpy because the conversion is pure index arithmetic;
    :func:`prepare_marlin_w4a16` moves it to the device once, at load time.
    """
    if num_bits != 4:
        raise NotImplementedError(f"MATE's Marlin wrapper is int4 only; got num_bits={num_bits}")
    from . import layout

    # The buffer's own extent, before anything is indexed by it: upstream allocates and addresses
    # exactly (size_k/16, 2*size_n) (gptq_marlin_repack.cu:324-326). Swapped tile axes have a
    # different shape, and a buffer unpacked in the wrong frame still produces (size_n, size_k//2).
    guards.require_repacked_shape(repacked.shape, size_k, size_n)
    words = layout.unpack_gptq(repacked, size_k, size_n)  # (size_k//8, size_n) packed words
    codes = layout.unpack_rows(words, size_k, size_n)  # (size_k, size_n) uint4b8 codes
    if biased_codes:
        codes = codes ^ 0x8  # uint4b8 code -> the two's-complement pattern the atom must see

    half = codes.reshape(size_k // 2, 2, size_n)
    packed_kn = (half[:, 0, :] | (half[:, 1, :] << 4)).astype(np.uint8)  # even k low, odd k high
    return np.ascontiguousarray(packed_kn.T).view(np.int8)  # (size_n, size_k//2)


def marlin_scales_to_musa(permuted, size_k: int, size_n: int, group_size: int):
    """Inverse of ``marlin_permute_scales``: permuted ``(G, size_n)`` -> plain ``(size_n, G)``.

    Direction matters and is easy to get backwards: the *host* path receives scales that are already
    permuted, so it needs this inverse. ``quant.marlin_permute_scales`` stays upstream-identical and
    is used by the conversion path (and byte-gated by the ported tests); it is not what the serving
    path calls. The transpose is not cosmetic — upstream's layout is K-major ``(G, size_n)`` while the
    kernel's scale operand is ``(n, k_blocks)``.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("marlin_scales_to_musa needs torch")
    return _unpermute(permuted, size_k, size_n, group_size).t().contiguous().to(torch.float16)


def _unpermute(permuted, size_k: int, size_n: int, group_size: int):
    # Same frame requirement as upstream's forward permutation: (num_groups, size_n) with
    # num_groups = size_k / group_size (marlin_utils.py:470-480, :198-202). Without it, scales
    # belonging to another K are permuted group-for-group and returned with a plausible shape.
    guards.require_scale_frame(permuted.shape, size_k, size_n, group_size)
    scale_perm, scale_perm_single = quant.get_scale_perms()
    if group_size < size_k and group_size != -1:
        width = len(scale_perm)
        inverse = np.argsort(np.array(scale_perm))
    else:
        width = len(scale_perm_single)
        inverse = np.argsort(np.array(scale_perm_single))
    rows = permuted.reshape((-1, width))[:, inverse]
    return rows.reshape((-1, size_n))


def marlin_zero_points_to_musa(
    packed_zp, size_k: int, size_n: int, group_size: int, upstream_permuted: bool = False
):
    """Inverse of ``marlin_zero_points``: packed + permuted zp -> plain ``(size_n, G)`` fp16.

    ``quant.unpack_marlin_zero_points`` undoes both the column packing and the permutation and
    returns ``(G, size_n)`` in the same K-major frame as upstream's scales, which is what the
    kernel's zero-point operand mirrors; this adds the transpose and the fp16 cast the staging
    expects. The values are upstream's **raw** per-group zero points -- the ``zero_point`` term of
    the documented ``w = scale * (q + nibble_bias - zero_point)``, with the same shape and dtype
    contract as ``scale_b`` (``mate/gemm.py:1512-1515``). Pass them raw: the op applies the
    ``(nibble_bias - zero_point)`` term itself, so a stream already folded by
    ``fold_zero_points`` would be folded twice, and a *symmetric* checkpoint's zero point is
    ``nibble_bias`` (8), which folds to exactly zero. Passing 0 where 8 belongs is the double-fold
    error in its most common shape.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("marlin_zero_points_to_musa needs torch")
    # unpack_marlin_zero_points is the inverse of marlin_zero_points, whose size_k is the
    # NUMBER OF GROUPS, not K: the packed operand holds one zero point per group. Passing
    # padded_k here made the result per-K (1536 rows) instead of per-group (12), so the
    # frame's zp_b could never match scale_b's (padded_n, padded_k/group_size).
    n_groups = size_k // group_size
    # The packed operand's frame, checked rather than inferred from its own shape: the row count IS
    # the group count (marlin_utils.py:512-533 ends in pack_cols; quant_utils.py:892-895 asserts the
    # row count). Deriving n_groups from the tensor's first axis would accept a per-K tensor.
    guards.require_packed_zero_point_frame(packed_zp.shape, n_groups, size_n)
    # TWO conventions reach this function and they need different inverses. Both return the same
    # shape and the same multiset of values, so ``require_packed_zero_point_frame`` cannot tell them
    # apart -- the wrong choice silently mis-orders the zero points relative to ``scale_b`` and
    # ``weight_b`` (measured: 1.034323 relative error against an fp32 reference, every guard silent).
    #
    #   upstream_permuted=False  the loader converted the checkpoint to the plain GPTQ packing, so a
    #                            word runs in the caller's n order (``n = 8 * word + nibble``) and
    #                            ``unpack_cols`` is the exact inverse. This is the serving path;
    #                            paired with ``scale_b``/``weight_b`` it measures 0.000315.
    #   upstream_permuted=True   the tensor came from ``marlin_zero_points``, which permutes before
    #                            packing, so its inverse is ``unpack_marlin_zero_points``
    #                            (``argsort([0,2,4,6,1,3,5,7])`` within each 8, then a per-tile
    #                            ``scale_perm``).
    #
    # tests/test_w4a16_marlin.py pins both directions
    # (``..._zero_point_frame_matches_loader_convention`` and ``..._matches_scale_frame``).
    if upstream_permuted:
        plain = quant.unpack_marlin_zero_points(packed_zp, n_groups, size_n)
    else:
        plain = quant.unpack_cols(packed_zp, n_groups, size_n)
    return plain.t().contiguous().to(torch.float16)


def _as_index_array(values) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().cpu().to(torch.int64).numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def resolve_act_order(g_idx, sort_indices, size_k: int, group_size: int) -> np.ndarray | None:
    """Return the k permutation the activation must be gathered with, or ``None`` for no act order.

    ``g_idx[k]`` is the group of input row ``k``. Without act order it is ``k // group_size`` and no
    gather is needed. With act order the weight rows are stored sorted by group, ``sort_indices`` is
    that order, and the kernel's per-group scale indexing is only right if every group becomes one
    contiguous run of exactly ``group_size`` rows -- which is checked rather than assumed.
    ``sort_indices=None`` sorts here (stable); pass the permutation the weight was actually stored
    with whenever one exists.
    """
    if g_idx is None:
        return None
    g = _as_index_array(g_idx)
    if g.shape[0] == 0:  # upstream's "no act order" (gptq_marlin_repack.cu:329)
        return None
    if g.shape[0] != size_k:
        raise ValueError(f"g_idx has {g.shape[0]} entries but the weight has K={size_k}")
    grouped = np.arange(size_k, dtype=np.int64) // group_size
    if np.array_equal(g, grouped):
        return None
    perm = np.argsort(g, kind="stable") if sort_indices is None else _as_index_array(sort_indices)
    if perm.shape[0] != size_k or not np.array_equal(np.sort(perm), np.arange(size_k)):
        raise ValueError(f"the act-order sort indices must be a permutation of range({size_k})")
    if not np.array_equal(g[perm], grouped):
        raise ValueError(
            "act order is only servable when, sorted, every group is one contiguous run of "
            f"{group_size} rows (g_idx[perm] == arange(K) // {group_size}); this g_idx is not"
        )
    return perm


def prepare_marlin_w4a16(
    qweight,
    scales,
    zero_points=None,
    size_k: int | None = None,
    size_n: int | None = None,
    group_size: int = compat.MUSA_MARLIN_GROUP_SIZE,
    g_idx=None,
    device=None,
    g_idx_sort_indices=None,
    zero_points_upstream_permuted: bool = False,
) -> MarlinW4A16KernelFrame:
    """The single load-time entry: upstream Marlin frame in, MATE kernel frame out.

    ``qweight`` and ``scales`` are upstream's already-converted tensors. When ``size_k``/``size_n``
    are omitted they are recovered with ``marlin_repacked_nk``, i.e. the *padded* extents, and
    ``logical_n`` is then the padded N — pass the logical extents when the model padded them, and the
    returned frame's ``unpad_output`` will slice the result.

    ``zero_points`` is upstream's ``marlin_zero_points`` output, or ``None`` for a symmetric
    (``uint4b8``) checkpoint. Decide that from the quantization config, not from whether the layer
    carries a ``qzeros`` tensor: GPTQ loaders register one for symmetric checkpoints too.

    Act order: pass the checkpoint's ``g_idx`` *and* ``g_idx_sort_indices``, the permutation the words
    were repacked with (``gptq_marlin_repack(perm=...)``). The frame keeps that permutation and
    ``apply_gptq_marlin_linear`` gathers the activation with it; recomputing it here instead could pick
    a different order within a group than the repack did.

    Refusals are deliberate and carry reasons, because the consumer must not discover these by
    crashing: a group size other than 128, ``K % 256 != 0`` (a TP-sharded ``down`` at TP=2 is
    rank-local K=4480, the concrete casualty), a repack whose K axis is not tile-aligned, and an act
    order whose groups are not contiguous runs of ``group_size`` once sorted.
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("prepare_marlin_w4a16 needs torch; the numpy gates live in mate.marlin.reference")

    qweight = qweight.contiguous()
    padded_n, padded_k = compat.marlin_repacked_nk(qweight, num_bits=4)
    if size_k is None:
        size_k = padded_k
    if size_n is None:
        size_n = padded_n

    if padded_k < size_k or padded_n < size_n:
        raise ValueError(
            f"the repacked weight is smaller than the logical shape: packed (n={padded_n}, k={padded_k}) "
            f"vs logical (n={size_n}, k={size_k}); pad the weight before repacking, not after"
        )

    ok, reason = compat.check_musa_marlin_supports_shape(size_k, size_n, group_size)
    if not ok:
        raise ValueError(reason)

    # Act order: the words were repacked with g_idx_sort_indices, so the activation is gathered with
    # the same permutation at apply time. Identity act order resolves to no gather at all. A sort
    # recomputed here may order rows within a group differently from the repack, so it is required.
    if g_idx_sort_indices is None and resolve_act_order(g_idx, None, padded_k, group_size) is not None:
        raise ValueError(
            "act order is present: pass g_idx_sort_indices, the permutation the weight words were "
            "repacked with (gptq_marlin_repack(perm=...)), so the activation is gathered in the same "
            "order"
        )
    input_perm = resolve_act_order(g_idx, g_idx_sort_indices, padded_k, group_size)

    weight_b = torch.from_numpy(marlin_weight_to_musa(qweight.cpu().numpy(), padded_k, padded_n))
    scale_b = marlin_scales_to_musa(scales, padded_k, padded_n, group_size)
    zp_b = (
        marlin_zero_points_to_musa(
            zero_points, padded_k, padded_n, group_size, zero_points_upstream_permuted
        )
        if zero_points is not None
        else None
    )
    if device is not None:
        weight_b = weight_b.to(device)
        scale_b = scale_b.to(device)
        zp_b = zp_b.to(device) if zp_b is not None else None
    else:
        weight_b = weight_b.to(scales.device)
        scale_b = scale_b.to(scales.device)
        zp_b = zp_b.to(scales.device) if zp_b is not None else None

    from mate.jit.gemm.dense_w4a16_gemm import (
        prepare_w4a16_call_operands,
        prepare_w4a16_prefill_operand,
    )

    scale_b = scale_b.contiguous()
    scale_b_padded, correction_operand = prepare_w4a16_call_operands(
        scale_b, zp_b.contiguous() if zp_b is not None else None
    )
    return MarlinW4A16KernelFrame(
        weight_b=weight_b.contiguous(),
        scale_b=scale_b,
        scale_b_padded=scale_b_padded,
        correction_operand=correction_operand,
        zero_points_b=zp_b.contiguous() if zp_b is not None else None,
        logical_n=size_n,
        padded_n=padded_n,
        padded_k=padded_k,
        group_size=group_size,
        has_zero_points=zero_points is not None,
        input_perm=(
            torch.as_tensor(input_perm, dtype=torch.int64, device=weight_b.device)
            if input_perm is not None
            else None
        ),
        scale_t=prepare_w4a16_prefill_operand(scale_b, zp_b),
    )



# The jit entry is marked dynamo-invisible where it is defined, so this module only imports it.

def apply_gptq_marlin_linear(
    input_a, prepared: MarlinW4A16KernelFrame, bias=None, num_mps=None, out_dtype=None
):
    """Run one W4A16 linear on a prepared frame. No branch on M: decode and prefill take this path.

    ``input_a`` is ``(..., K)`` fp16; leading dimensions are flattened for the GEMM and restored on the
    result, as upstream's ``apply_gptq_marlin_linear`` does. ``out_dtype`` selects an fp16 or bf16
    result (default: the input dtype). The result is ``(..., padded_n)``; the caller slices it with
    ``prepared.unpad_output``.

    The symmetric case passes ``zero_points=None``, which is inert by design on the kernel's symmetric
    instantiation (its ``sym`` build never stages the tensor).
    """
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError("apply_gptq_marlin_linear needs torch")

    # Imported from the jit module rather than mate.gemm: this module is the one delivered with
    # the marlin wrapper, so a consumer only has to install this subtree. The public wrapper in
    # mate.gemm carries the same signature and delegates here; going straight to the jit entry
    # also avoids depending on a name that an installed base package may not export yet.
    from mate.jit.gemm.dense_w4a16_gemm import dense_w4a16_gemm_op as dense_w4a16_gemm

    leading = input_a.shape[:-1]
    x = input_a.reshape(-1, input_a.shape[-1])
    if prepared.input_perm is not None:
        x = x.index_select(1, prepared.input_perm)
    out = torch.empty(
        (x.shape[0], prepared.padded_n),
        dtype=input_a.dtype if out_dtype is None else out_dtype,
        device=input_a.device,
    )
    dense_w4a16_gemm(
        x,
        prepared.weight_b,
        prepared.scale_b,
        out,
        zero_points=prepared.zero_points_b,
        g_idx=None,
        num_mps=num_mps,
        scale_b_padded=prepared.scale_b_padded,
        correction_operand=prepared.correction_operand,
        scale_t=prepared.scale_t,
    )
    out = out.view(*leading, prepared.padded_n)
    if bias is not None:
        if bias.shape[-1] != prepared.padded_n:
            bias = torch.nn.functional.pad(bias, (0, prepared.padded_n - bias.shape[-1]))
        out = out + bias
    return out
