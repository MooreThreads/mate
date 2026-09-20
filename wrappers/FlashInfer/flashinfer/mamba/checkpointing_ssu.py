"""Checkpointing selective state update, FlashInfer-compatible on MUSA.

``checkpointing_ssu`` and ``allocate_checkpointing_ssu_scratch`` mirror upstream
FlashInfer's signatures. ``CheckpointingSSURunner`` is the autotuner-facing class
that downstream frameworks detect to decide whether the ReplaySSM checkpointing
path is available, so it is exported **only** when the MATE backend actually
provides it: probing must observe capability, not an importable placeholder that
would raise later. This follows upstream FlashInfer's own optional-surface
pattern for symbols whose backend may be absent.
"""

from __future__ import annotations

from typing import Any

from . import _backend

__all__ = [
    "allocate_checkpointing_ssu_scratch",
    "checkpointing_ssu",
]


def checkpointing_ssu(
    state: Any,
    x_cache: Any,
    B_cache: Any,
    dt_cache: Any,
    ring_start: Any,
    prev_num_accepted_tokens: Any,
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    out: Any,
    D: Any = None,
    z: Any = None,
    dt_bias: Any = None,
    dt_softplus: bool = False,
    state_batch_indices: Any = None,
    pad_slot_id: int = -1,
    state_scale: Any = None,
    rand_seed: Any = None,
    philox_rounds: int = 10,
    d_split: Any = None,
    cu_seqlens: Any = None,
    max_seqlen: Any = None,
    enable_pdl: bool = False,
    cb_scaled: Any = None,
    cumAdt_vec: Any = None,
    cb_old: Any = None,
    precompute_heads_per_cta: int = 0,
    algorithm: str = "auto",
) -> Any:
    """Run the checkpointing selective state update.

    Arguments are forwarded positionally to the MATE implementation in the
    upstream FlashInfer order.
    """
    implementation = _backend.resolve("checkpointing_ssu")
    return implementation(
        state,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        x,
        dt,
        A,
        B,
        C,
        out,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        state_scale,
        rand_seed,
        philox_rounds,
        d_split,
        cu_seqlens,
        max_seqlen,
        enable_pdl,
        cb_scaled,
        cumAdt_vec,
        cb_old,
        precompute_heads_per_cta,
        algorithm,
    )


def allocate_checkpointing_ssu_scratch(
    batch_size: int,
    num_heads: int,
    num_predicted_tokens: int,
    max_window: int,
    dtype: Any,
    device: Any,
) -> Any:
    """Allocate the scratch tensors the checkpointing SSU path expects."""
    implementation = _backend.resolve("allocate_checkpointing_ssu_scratch")
    return implementation(
        batch_size,
        num_heads,
        num_predicted_tokens,
        max_window,
        dtype,
        device,
    )


if _backend.has_symbol("CheckpointingSSURunner"):
    CheckpointingSSURunner = _backend.resolve("CheckpointingSSURunner")
    __all__ = [*__all__, "CheckpointingSSURunner"]
