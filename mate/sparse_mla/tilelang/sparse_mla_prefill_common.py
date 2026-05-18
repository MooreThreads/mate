# ruff: noqa
"""Shared sparse prefill host/JIT plumbing for MODEL1 and V3.2."""

from __future__ import annotations

from typing import Optional

import torch
import tilelang


SPARSE_PREFILL_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}


SPARSE_PREFILL_COMPILE_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def require_token_lengths(
    lengths: Optional[torch.Tensor],
    seq_len: int,
    fill: int,
    device,
    name: str,
) -> torch.Tensor:
    if lengths is None:
        return torch.full((seq_len,), fill, dtype=torch.int32, device=device)
    assert lengths.dtype == torch.int32, f"{name} must be int32"
    assert lengths.shape == (seq_len,), f"{name} must have shape [S_q]"
    assert lengths.stride(-1) == 1, f"{name} last dimension must be contiguous"
    return lengths.contiguous()


def optional_prefill_attn_sink(
    attn_sink: Optional[torch.Tensor],
    heads: int,
    device,
):
    if attn_sink is None:
        return torch.empty((heads,), dtype=torch.float32, device=device), False
    assert attn_sink.dtype == torch.float32, "attn_sink must be float32"
    assert attn_sink.shape == (heads,), "attn_sink must have shape [H_q]"
    assert attn_sink.stride(-1) == 1, "attn_sink last dimension must be contiguous"
    return attn_sink.contiguous(), True
