"""Selective state update, FlashInfer-compatible on MUSA.

The signature mirrors upstream FlashInfer's ``selective_state_update`` exactly,
including argument order and defaults, because vLLM's SSU dispatch calls it with
keyword arguments and probes it for capability flags. MUSA-specific selection
(native kernel versus portability fallback, reduction splits, Philox mode) is a
MATE implementation concern or an environment switch; it must not appear as a new
parameter on this name.
"""

from __future__ import annotations

from typing import Any

import torch

from . import _backend

__all__ = ["selective_state_update"]

#: vLLM's sentinel for "this state row belongs to no block"
#: (``vllm.v1.attention.backends.utils.NULL_BLOCK_ID``).
NULL_BLOCK_ID = 0

#: Widest vector load MATE's SSU kernels issue, in bytes.
_VECTOR_BYTES = 16


def _vector_aligned(tensor: Any) -> bool:
    """Whether every row of ``tensor`` starts where MATE's vector loads may begin.

    The alignment is 16 bytes in elements, narrowed to the last axis's extent, and
    holds for the storage offset and for the stride of every non-unit outer axis.
    """
    extent = tensor.shape[-1]
    align = min(_VECTOR_BYTES // tensor.element_size(), (extent & -extent) or 1)
    if tensor.storage_offset() % align:
        return False
    return all(
        size == 1 or stride % align == 0
        for size, stride in zip(tensor.shape[:-1], tensor.stride()[:-1])
    )


def _readable(tensor: Any, *, vector_loads: bool = False) -> Any:
    """Return ``tensor`` unchanged when MATE's SSU kernels can read it in place.

    The kernels read x, B and C through their strides, so vLLM's views of the
    projected states pass as they are. A copy is made only when the last axis is
    not dense or, for the vector-loaded B and C, a row start is misaligned.
    """
    if tensor is None or not hasattr(tensor, "stride"):
        return tensor
    if tensor.stride(-1) != 1:
        return tensor.contiguous()
    if vector_loads and not _vector_aligned(tensor):
        # contiguous() keeps a dense tensor whose start is off the grid; a fresh
        # allocation is aligned.
        return tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _flat_slots(indices: Any) -> Any:
    """Return a slot table in the form MATE addresses.

    A ``[sequences, 1]`` table is the one-slot-per-sequence vector and is passed
    flat. A ``[sequences, steps]`` table keeps its shape: MATE reads entry
    ``[b, t]`` through the table's strides, as the Triton kernel does.
    """
    if indices is None or not hasattr(indices, "dim"):
        return indices
    if indices.dim() == 2 and indices.shape[1] == 1:
        return indices.reshape(-1)
    return indices


_FP32_VECTOR_CACHE: dict[tuple, Any] = {}
_FP32_VECTOR_CACHE_LIMIT = 256


def _per_head_fp32(tensor: Any) -> Any:
    """Return MATE's per-head fp32 view of a vLLM per-head vector.

    D and dt_bias are loaded model parameters, so the widened copy is made once
    per tensor and reused. Widening bf16/fp16 to fp32 is exact; doing it on every
    decode step instead put one cast launch per mixer per step on the critical
    path, which is measurable at 52 mixers.
    """
    if tensor is None or not hasattr(tensor, "dtype"):
        # Forwarding tests and non-tensor sentinels pass through untouched; MATE
        # validates whatever it actually receives.
        return tensor
    if tensor.dim() >= 2 and all(stride == 0 for stride in tensor.stride()[1:]):
        tensor = tensor[:, 0]
    if tensor.dtype == torch.float32:
        return tensor
    key = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )
    cached = _FP32_VECTOR_CACHE.get(key)
    if cached is None:
        cached = tensor.float().contiguous()
        if len(_FP32_VECTOR_CACHE) < _FP32_VECTOR_CACHE_LIMIT:
            _FP32_VECTOR_CACHE[key] = cached
    return cached


def _dt_input(tensor: Any) -> Any:
    """Return dt in the form MATE reads without a copy.

    vLLM expands dt across head_dim, so it arrives as a zero-stride view in the
    model dtype. MATE reads one step value per head and widens it inside the
    kernel, so the broadcast's base -- a [batch, heads] view with a dense head
    axis -- is handed over as [batch, heads, 1]. The values are identical by
    construction, and the decode path loses a per-layer cast launch.
    """
    if tensor is None or not hasattr(tensor, "stride"):
        return tensor
    if tensor.dim() == 3 and tensor.stride(-1) == 0:
        return tensor.select(-1, 0).unsqueeze(-1)
    return tensor


def selective_state_update(
    state: Any,
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    z: Any = None,
    dt_bias: Any = None,
    dt_softplus: bool = False,
    state_batch_indices: Any = None,
    pad_slot_id: int = -1,
    state_scale: Any = None,
    out: Any = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Any = None,
    intermediate_state_indices: Any = None,
    intermediate_state_scales: Any = None,
    rand_seed: Any = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: Any = None,
    cu_seqlens: Any = None,
    num_accepted_tokens: Any = None,
    backend: str = "auto",
    *,
    null_block_id: int = NULL_BLOCK_ID,
    is_blackwell: bool = False,
    enable_stochastic_rounding: bool = False,
    cache_philox_rounds: int | None = None,
) -> Any:
    """Update the Mamba SSM state for one decode step.

    The first block of parameters is MATE's own order and is forwarded
    positionally. The keyword-only block after it exists for vLLM's backend
    dispatch, which is the only production caller and passes these four by
    keyword: without them the call raises ``TypeError`` before any kernel runs,
    which is how an otherwise-correct MATE SSU stayed unreachable behind
    ``--mamba-backend flashinfer``.

    ``is_blackwell`` and a ``null_block_id`` other than the sentinel are refused
    rather than ignored: MATE has no Blackwell path, and a non-sentinel null
    block would change which state rows a kernel may skip, so accepting it
    silently would produce wrong states instead of an error.
    """
    if is_blackwell:
        raise NotImplementedError(
            "is_blackwell=True selects a Blackwell-only SSU path; this is the MUSA "
            "surface and MATE implements no such path."
        )
    if null_block_id != NULL_BLOCK_ID:
        raise NotImplementedError(
            f"null_block_id={null_block_id} asks the SSU to treat that block as "
            "carrying no state, which MATE's selective_state_update has no concept "
            "of; only the sentinel is accepted."
        )
    if enable_stochastic_rounding and rand_seed is None:
        raise NotImplementedError(
            "enable_stochastic_rounding=True needs a seed: MATE takes stochastic "
            "rounding from rand_seed, and this call supplied none."
        )
    implementation = _backend.resolve("selective_state_update")
    # vLLM builds its per-head vectors as `self.D[:, None].expand(-1, head_dim)`:
    # a zero-stride 2-D broadcast in the model dtype. MATE's native kernel takes
    # one fp32 value per head, and widening bf16/fp16 to fp32 is exact, so the
    # adapter hands over the broadcast's first column instead of materializing a
    # [heads, head_dim] copy. Refusing bf16 D here made the flag fail closed for a
    # reason the caller could not act on: vLLM has no fp32 D to pass.
    D = _per_head_fp32(D)
    dt_bias = _per_head_fp32(dt_bias)
    x = _readable(x)
    B = _readable(B, vector_loads=True)
    C = _readable(C, vector_loads=True)
    dt = _dt_input(dt)
    if hasattr(dt, "stride") and dt.shape[-1] != 1 and dt.stride(-1) != 1:
        # The per-head [rows, heads, 1] form never advances its last axis; any other
        # dt is read along it and needs it dense.
        dt = dt.contiguous()
    state_batch_indices = _flat_slots(state_batch_indices)
    dst_state_batch_indices = _flat_slots(dst_state_batch_indices)
    return implementation(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        state_scale,
        out,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_philox_rounds if cache_philox_rounds is not None else philox_rounds,
        cache_steps,
        algorithm,
        dst_state_batch_indices,
        cu_seqlens,
        num_accepted_tokens,
        backend,
    )
