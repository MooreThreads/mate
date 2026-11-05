# SPDX-License-Identifier: Apache-2.0
"""MATE's Marlin wrapper: upstream vLLM's Marlin Python surface, implemented by MATE.

The goal this package serves is that MATE owns a Marlin wrapper vllm-musa consumes *the way upstream
consumes NVIDIA Marlin*. So the names, the argument order and the layouts are upstream's, while the
implementation is MATE's kernel and MATE's repack. Nothing here imports torch at module scope: a host
with numpy and no torch can import the package and run the gates that guard it.

What the vllm-musa consumer calls
---------------------------------
Two names, and nothing else:

* :func:`mate.marlin.prepare_marlin_w4a16` — load time only. Takes upstream's *already converted*
  tensors (``gptq_marlin_repack`` output, ``marlin_permute_scales`` output, ``marlin_zero_points``
  output) plus the logical extents, and returns a :class:`MarlinW4A16KernelFrame` holding the
  kernel-ready operands. It refuses shapes the kernel cannot serve, with the reason attached.
* :func:`mate.marlin.apply_gptq_marlin_linear` — every forward. Takes an activation and the prepared
  frame, calls ``mate.gemm.dense_w4a16_gemm``, and has no branch on M, so nothing is frozen at trace
  time.

A consumer that starts from the checkpoint rather than from upstream's Marlin tensors can build the
same frame with :func:`mate.marlin.prepare_w4a16_from_gptq_frame`, which takes the loader's GPTQ-layout
words, scales and unpacked zero points directly and skips the repack round trip.

Name mapping, and where a divergence is deliberate
--------------------------------------------------
================================  ==================================  ============================
upstream (``marlin_utils.py``)    here                                status
================================  ==================================  ============================
``gptq_marlin_repack``            :func:`gptq_marlin_repack`          mirrored; ``num_bits`` must be 4
``awq_marlin_repack``             :func:`awq_marlin_repack`           mirrored; **not on the serving path**
``get_scale_perms``               :func:`quant.get_scale_perms`       identical
``marlin_permute_scales``         :func:`quant.marlin_permute_scales` identical
``marlin_zero_points``            :func:`quant.marlin_zero_points`    identical
``awq_to_marlin_zero_points``     :func:`quant.awq_to_marlin_zero_points`  identical
``marlin_padded_nk``              :func:`compat.marlin_padded_nk`     identical
``marlin_repacked_nk``            :func:`compat.marlin_repacked_nk`   accepts a tensor or a shape
``marlin_pad_qweight/scales/dim`` :func:`compat.marlin_pad_*`          identical
``marlin_unpad_output``           :func:`compat.marlin_unpad_output`  identical
``apply_gptq_marlin_linear``      :func:`apply_gptq_marlin_linear`    same name and shape of call,
                                                                      no workspace/g_idx/wtype
================================  ==================================  ============================

Deliberate divergences, each with its reason:

* **int4 only.** ``num_bits != 4`` is refused with a message rather than approximated. MATE's kernel
  is int4 x fp16, so the 8-bit branches of upstream's code paths do not exist here.
* **Inverses that upstream does not have.** ``marlin_scales_to_musa``, ``marlin_zero_points_to_musa``
  and ``unpack_marlin_zero_points`` invert upstream layouts. Upstream does not need them because its
  kernel consumes the permuted frame directly; MATE's kernel does not, so the conversion is explicit
  and tested instead of being implied.
* **A real weight conversion.** ``marlin_weight_to_musa`` is not a stride trick: hardware measurement
  on the existing MATE kernel showed the int4 B operand is consumed with consecutive k, even k in the
  low nibble, while upstream's words are a 16x64 tile map -- *and* upstream's ``uint4b8`` codes must
  have their sign bit flipped, because the declared operand frame decodes a nibble as ``v - 8`` while
  the atom reads two's complement. Route B therefore pays one load-time pass. See
  ``mate.marlin.linear.marlin_weight_to_musa`` for the file:line authority and the measurement.
* **Act order by materialisation.** The weight rows are stored in ``g_idx`` sort order (upstream's
  repack convention) and the frame carries that permutation; ``apply_gptq_marlin_linear`` gathers the
  activation columns with it before the GEMM, because the device kernel itself only reads K in order.
* **``marlin_permute_bias``, ``marlin_moe_permute_scales``, workspace helpers, ``get_marlin_input_dtype``
  and the FP8 helpers are not mirrored**: none has a call site on the MUSA W4A16 path, and a wrapper
  that mirrors 40 names it does not implement is worse than one that mirrors the 14 it does.
"""

from __future__ import annotations

from . import compat, layout, loader, quant, reference
from .compat import (
    GPTQ_MARLIN_TILE,
    MUSA_MARLIN_GROUP_SIZE,
    MUSA_MARLIN_K_MULTIPLE,
    check_musa_marlin_supports_shape,
    marlin_is_tile_aligned,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_repacked_nk,
    marlin_unpad_output,
)
from .layout import (
    TILE_K,
    TILE_N,
    TILE_WORDS,
    awq_marlin_repack,
    gptq_marlin_repack,
    repack_gptq,
    unpack_gptq,
)
from .linear import (
    MarlinW4A16KernelFrame,
    apply_gptq_marlin_linear,
    marlin_scales_to_musa,
    marlin_weight_to_musa,
    marlin_zero_points_to_musa,
    prepare_marlin_w4a16,
    resolve_act_order,
)
from .loader import (
    prepare_w4a16_from_gptq_frame,
    unpack_gptq_words,
    unpack_zero_points,
)
from .quant import (
    awq_to_marlin_zero_points,
    get_scale_perms,
    marlin_permute_scales,
    marlin_zero_points,
    unpack_marlin_zero_points,
)

__all__ = [
    # the two names a consumer calls
    "prepare_marlin_w4a16",
    "apply_gptq_marlin_linear",
    "MarlinW4A16KernelFrame",
    # loader-frame entry: the same frame without the Marlin repack round trip
    "prepare_w4a16_from_gptq_frame",
    "unpack_gptq_words",
    "unpack_zero_points",
    "resolve_act_order",
    # upstream-mirrored surface
    "get_scale_perms",
    "marlin_permute_scales",
    "marlin_zero_points",
    "awq_to_marlin_zero_points",
    "gptq_marlin_repack",
    "awq_marlin_repack",
    "marlin_padded_nk",
    "marlin_repacked_nk",
    "marlin_pad_qweight",
    "marlin_pad_scales",
    "marlin_pad_dim",
    "marlin_unpad_output",
    # MATE additions, with the reasons in the mapping table above
    "unpack_marlin_zero_points",
    "marlin_scales_to_musa",
    "marlin_zero_points_to_musa",
    "marlin_weight_to_musa",
    "marlin_is_tile_aligned",
    "check_musa_marlin_supports_shape",
    "repack_gptq",
    "unpack_gptq",
    "TILE_K",
    "TILE_N",
    "TILE_WORDS",
    "GPTQ_MARLIN_TILE",
    "MUSA_MARLIN_GROUP_SIZE",
    "MUSA_MARLIN_K_MULTIPLE",
    # submodules, for callers that want the numpy oracle or the raw layout
    "compat",
    "layout",
    "loader",
    "quant",
    "reference",
]
