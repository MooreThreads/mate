"""ReplaySSM state materialization, FlashInfer-compatible on MUSA.

Mirrors upstream FlashInfer's signature, including the keyword-only shape and
dtype description. The first sixteen arguments are positional and describe raw
slot pointers and strides, so their order is part of the contract.
"""

from __future__ import annotations

from typing import Any

from . import _backend

__all__ = ["replayssm_materialize"]


def replayssm_materialize(
    state_ptrs: Any,
    state_slot_strides: Any,
    x_cache_ptrs: Any,
    x_cache_slot_strides: Any,
    B_cache_ptrs: Any,
    B_cache_slot_strides: Any,
    dt_cache_ptrs: Any,
    dt_cache_slot_strides: Any,
    A_ptrs: Any,
    state_scale_ptrs: Any,
    state_scale_slot_strides: Any,
    src_slots: Any,
    dst_slots: Any,
    ring_start: Any,
    replay_prefix_len: int,
    active_request_indices: Any,
    *,
    state_dtype: Any,
    input_dtype: Any,
    matrixA_dtype: Any,
    dim: int,
    dstate: int,
    num_heads: int,
    heads_per_group: int,
    max_window: int,
    ring_buffer_len: int,
    pad_slot_id: int = -1,
    rand_seed: Any = None,
    philox_rounds: int = 0,
    dependency_inputs: Any = None,
    dependency_outputs: Any = None,
) -> Any:
    """Materialize replay state into the destination slots."""
    implementation = _backend.resolve("replayssm_materialize")
    return implementation(
        state_ptrs,
        state_slot_strides,
        x_cache_ptrs,
        x_cache_slot_strides,
        B_cache_ptrs,
        B_cache_slot_strides,
        dt_cache_ptrs,
        dt_cache_slot_strides,
        A_ptrs,
        state_scale_ptrs,
        state_scale_slot_strides,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        active_request_indices,
        state_dtype=state_dtype,
        input_dtype=input_dtype,
        matrixA_dtype=matrixA_dtype,
        dim=dim,
        dstate=dstate,
        num_heads=num_heads,
        heads_per_group=heads_per_group,
        max_window=max_window,
        ring_buffer_len=ring_buffer_len,
        pad_slot_id=pad_slot_id,
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
        dependency_inputs=dependency_inputs,
        dependency_outputs=dependency_outputs,
    )
