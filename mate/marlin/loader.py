# SPDX-License-Identifier: Apache-2.0
"""Loader-frame entry: vLLM's GPTQ-layout tensors straight to the MATE kernel frame.

``prepare_marlin_w4a16`` takes upstream's *Marlin* tensors, so a consumer that starts from the
checkpoint has to run ``gptq_marlin_repack`` / ``marlin_permute_scales`` / ``marlin_zero_points``
only for the wrapper to invert all three. This entry takes the tensors vLLM's ``MPLinearKernel``
already holds after ``permute_param_layout_`` -- the same ones upstream's Marlin kernel feeds to
``gptq_marlin_repack`` -- and produces a byte-identical frame without the round trip:

=====================  ===========================================  =================================
argument               layout                                       upstream source of the frame
=====================  ===========================================  =================================
``qweight``            ``(K/8, N)`` int32, packed along K, nibble   ``permute_param_layout_(x,
                       ``i`` of word ``r`` is row ``k = 8r + i``    input_dim=0, output_dim=1,
                                                                    packed_dim=0)``
``scales``             ``(K/group_size, N)``                         ``permute_param_layout_(x,
                                                                    input_dim=0, output_dim=1)``
``zero_points``        ``(K/group_size, N)`` *unpacked* integers,   upstream's ``unpack_cols`` output
                       or ``None`` for a symmetric checkpoint       (see :func:`unpack_zero_points`)
``g_idx``              ``(K,)`` group of each input row, or None    the checkpoint's ``g_idx``
=====================  ===========================================  =================================

The weight codes are the unsigned nibbles ``u`` of both GPTQ ``uint4b8`` and AWQ ``uint4`` (after
vLLM's AWQ-to-standard conversion); the dequantized weight is ``scale * (u - zero_point)`` with
``zero_point = 8`` for a symmetric checkpoint. Everything runs as vectorized torch on the input's
device.
"""

from __future__ import annotations

try:  # lazy by design, as in the rest of the package
    import torch
except ImportError:  # pragma: no cover - exercised on the numpy-only gate host
    torch = None  # type: ignore[assignment]

from . import compat
from . import guards
from .linear import MarlinW4A16KernelFrame, resolve_act_order

PACK_FACTOR_4BIT = guards.PACK_FACTOR_4BIT


def _require_torch(name: str):
    if torch is None:  # pragma: no cover - exercised on the numpy-only gate host
        raise RuntimeError(f"{name} needs torch")


def unpack_gptq_words(qweight, size_k: int, size_n: int):
    """``(K/8, N)`` int32 GPTQ words -> ``(K, N)`` int32 nibbles, row ``k = 8r + i`` from bits ``4i``."""
    _require_torch("unpack_gptq_words")
    guards.require_gptq_word_shape(qweight.shape, size_k, size_n)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    words = qweight.to(torch.int32)
    return ((words.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).reshape(size_k, size_n)


def unpack_zero_points(qzeros, num_groups: int, size_n: int, packed_dim: int):
    """Packed int32 zero points -> ``(num_groups, size_n)`` integers.

    ``packed_dim`` names the axis that holds 8 N-nibbles per word, because both orientations occur:
    GPTQ checkpoints store ``(num_groups, N/8)`` (``packed_dim=1``), while vLLM's AWQ-to-standard
    conversion produces ``(N/8, num_groups)`` (``packed_dim=0``). A square ``num_groups == N/8``
    tensor cannot reveal which one it is, so the caller states it.
    """
    _require_torch("unpack_zero_points")
    if packed_dim not in (0, 1):
        raise ValueError(f"packed_dim must be 0 or 1, got {packed_dim}")
    if size_n % PACK_FACTOR_4BIT:
        raise ValueError(f"zero points pack {PACK_FACTOR_4BIT} N values per word; got N={size_n}")
    expected = (
        (num_groups, size_n // PACK_FACTOR_4BIT)
        if packed_dim == 1
        else (size_n // PACK_FACTOR_4BIT, num_groups)
    )
    if tuple(qzeros.shape) != expected:
        raise ValueError(
            f"packed zero points must be {expected} for num_groups={num_groups}, N={size_n}, "
            f"packed_dim={packed_dim}; got {tuple(qzeros.shape)}"
        )
    words = qzeros.to(torch.int32)
    if packed_dim == 0:
        words = words.t()
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qzeros.device)
    values = (words.unsqueeze(-1) >> shifts) & 0xF  # (num_groups, N/8, 8): n = 8c + i
    return values.reshape(num_groups, size_n).contiguous()


def prepare_w4a16_from_gptq_frame(
    qweight,
    scales,
    zero_points=None,
    *,
    size_k: int,
    size_n: int,
    group_size: int = compat.MUSA_MARLIN_GROUP_SIZE,
    g_idx=None,
    device=None,
) -> MarlinW4A16KernelFrame:
    """Build the kernel frame from the loader's GPTQ-layout tensors (see the module table).

    ``zero_points`` is ``None`` for a symmetric checkpoint. Decide that from the quantization config,
    not from whether the layer carries a ``qzeros`` tensor: GPTQ loaders register one for symmetric
    checkpoints too, and its contents are not this kernel's zero points.

    Act order is applied here: the weight rows are stored sorted by ``g_idx`` (stable) and the frame
    keeps that order, which ``apply_gptq_marlin_linear`` uses to gather the activation. The result is
    the frame :func:`mate.marlin.prepare_marlin_w4a16` builds from the same checkpoint after
    upstream's repack, bit for bit.
    """
    _require_torch("prepare_w4a16_from_gptq_frame")
    ok, reason = compat.check_musa_marlin_supports_shape(size_k, size_n, group_size)
    if not ok:
        raise ValueError(reason)
    guards.require_scale_frame(scales.shape, size_k, size_n, group_size)
    num_groups = size_k // group_size
    if zero_points is not None:
        if tuple(zero_points.shape) != (num_groups, size_n):
            raise ValueError(
                f"zero_points must be the unpacked (num_groups, N) = {(num_groups, size_n)} integers; "
                f"got {tuple(zero_points.shape)}. Unpack a packed tensor with unpack_zero_points."
            )
        if bool((zero_points < 0).any()) or bool((zero_points > 15).any()):
            raise ValueError("zero_points must be 4-bit values in [0, 15]")

    if device is None:
        device = qweight.device
    codes = unpack_gptq_words(qweight.to(device), size_k, size_n)

    perm = resolve_act_order(g_idx, None, size_k, group_size)
    input_perm = None
    if perm is not None:
        input_perm = torch.as_tensor(perm, dtype=torch.int64, device=device)
        codes = codes.index_select(0, input_perm)

    # The atom reads each nibble as two's complement, so u is stored as u ^ 0x8 (the code of u - 8).
    nibbles = (codes ^ 0x8).to(torch.uint8).view(size_k // 2, 2, size_n)
    packed = nibbles[:, 0, :] | (nibbles[:, 1, :] << 4)  # even k low, odd k high
    weight_b = packed.t().contiguous().view(torch.int8)  # (N, K/2)

    scale_b = scales.to(device).t().contiguous().to(torch.float16)
    zp_b = (
        zero_points.to(device).t().contiguous().to(torch.float16)
        if zero_points is not None
        else None
    )

    from mate.jit.gemm.dense_w4a16_gemm import (
        prepare_w4a16_call_operands,
        prepare_w4a16_prefill_operand,
    )

    scale_b_padded, correction_operand = prepare_w4a16_call_operands(scale_b, zp_b)
    return MarlinW4A16KernelFrame(
        weight_b=weight_b,
        scale_b=scale_b,
        scale_b_padded=scale_b_padded,
        correction_operand=correction_operand,
        zero_points_b=zp_b,
        logical_n=size_n,
        padded_n=size_n,
        padded_k=size_k,
        group_size=group_size,
        has_zero_points=zero_points is not None,
        input_perm=input_perm,
        scale_t=prepare_w4a16_prefill_operand(scale_b, zp_b),
    )
