"""Variable-length SSD combined forward, FlashInfer-compatible on MUSA.

``ssd_combined_fwd_varlen`` is the MUSA entry point that vLLM's Mamba2/SSD
implementation calls for packed variable-length prefill. Its parameter list is a
frozen contract: the names, order, and arity must not change, and callers may bind
them by position or by name. vLLM 0.28.0 binds them all by name, which is what a
live dump of the routed call showed (``MUSA_SSD_TRACE_ARGS`` in the route patch
records zero positional arguments). ``mamba_chunk_scan_combined_varlen``
is the historical alias and must stay an alias of the same object rather than a
second definition.

Upstream FlashInfer's ``ssd_combined_fwd`` (the dense, CUTLASS/CuTe-backed
entry point) is intentionally not provided here; this wrapper covers the
packed-varlen path the MUSA consumers use. MUSA-specific tuning knobs are not
added as parameters, because the surface has to stay upstream-shaped; they
belong to the MATE implementation or its environment.
"""

from __future__ import annotations

from typing import Any

import torch

from . import _backend

__all__ = [
    "mamba_chunk_scan_combined_varlen",
    "ssd_combined_fwd_varlen",
]


def _contiguous(tensor: Any) -> Any:
    """Give MATE a contiguous tensor, copying only when it is not already one.

    MATE's packed entry point rejects strided inputs, and vLLM hands it views: the
    projected states are slices of one projection buffer, and the chunk metadata can
    be a strided slice of the scheduler's own arrays. Copying unconditionally would
    add an allocation to every prefill step, so the copy is skipped in the common
    contiguous case.
    """
    if tensor is None or not hasattr(tensor, "is_contiguous"):
        return tensor
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


# vLLM hands D and dt_bias in the model dtype; MATE's stages take fp32 per-head
# vectors. Widening bf16/fp16 to fp32 is exact, and the copies are cached because
# these are loaded parameters rather than per-step tensors -- the same treatment
# the SSU entry gives its own copy of D and dt_bias.
_PER_HEAD_FP32_CACHE: dict[tuple, Any] = {}
_SKIP_MATRIX_CACHE: dict[tuple, Any] = {}


def _per_head_fp32(tensor: Any) -> Any:
    if tensor is None or not hasattr(tensor, "dtype"):
        return tensor
    if tensor.dtype == torch.float32:
        return tensor
    key = (id(tensor), tuple(tensor.shape), tensor.dtype, tensor.data_ptr())
    cached = _PER_HEAD_FP32_CACHE.get(key)
    if cached is None:
        cached = tensor.detach().to(torch.float32)
        if not cached.is_contiguous():
            cached = cached.contiguous()
        _PER_HEAD_FP32_CACHE[key] = cached
    return cached


def _skip_matrix(tensor: Any, reference: Any) -> Any:
    """Broadcast vLLM's per-head `D` to the `[heads, dim]` matrix MATE's scan reads.

    vLLM passes one skip value per head and applies it to every element of that
    head; MATE's chunk scan indexes `D[head, dim]`. Filling the second axis with the
    head's own value reproduces vLLM's semantics exactly, and the replay against the
    Triton implementation checks that numerically rather than by assumption.
    """
    if tensor is None or not hasattr(tensor, "dim") or tensor.dim() != 1:
        return tensor
    shape = getattr(reference, "shape", None)
    if not shape:
        return tensor
    head_dim = int(shape[-1])
    key = (id(tensor), tuple(tensor.shape), tensor.dtype, tensor.data_ptr(), head_dim)
    cached = _SKIP_MATRIX_CACHE.get(key)
    if cached is None:
        cached = tensor[:, None].expand(-1, head_dim).contiguous()
        _SKIP_MATRIX_CACHE[key] = cached
    return cached


def ssd_combined_fwd_varlen(
    x: Any,
    dt: Any,
    A: Any,
    B: Any,
    C: Any,
    chunk_size: int,
    cu_seqlens: Any,
    cu_chunk_seqlens: Any,
    last_chunk_indices: Any,
    seq_idx: Any,
    out: Any,
    D: Any = None,
    z: Any = None,
    dt_bias: Any = None,
    initial_states: Any = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_intermediate_states: bool = False,
    state_dtype: Any = None,
    checkpoint_token_indices: Any = None,
    checkpoint_state_slots: Any = None,
    checkpoint_states: Any = None,
) -> Any:
    """Run the combined SSD forward pass for packed variable-length sequences.

    Arguments are forwarded to MATE by name rather than by position, and ``D`` and
    ``dt_bias`` are widened to fp32, because MATE's family orders
    ``dt_softplus, dt_limit, initial_states`` where upstream FlashInfer orders
    ``initial_states, dt_softplus, dt_limit``: a positional forward put ``dt_limit``'s
    tuple where MATE reads ``initial_states``. See the MATE mamba family for the shape
    and dtype contract of each argument.
    """
    D = _per_head_fp32(D)
    dt_bias = _per_head_fp32(dt_bias)
    D = _skip_matrix(D, out)

    implementation = _backend.resolve("ssd_combined_fwd_varlen")
    x = _contiguous(x)
    dt = _contiguous(dt)
    B = _contiguous(B)
    C = _contiguous(C)
    cu_seqlens = _contiguous(cu_seqlens)
    cu_chunk_seqlens = _contiguous(cu_chunk_seqlens)
    last_chunk_indices = _contiguous(last_chunk_indices)
    seq_idx = _contiguous(seq_idx)

    # Forwarded by name, not by position: upstream FlashInfer orders
    # `initial_states, dt_softplus, dt_limit` while MATE's family orders
    # `dt_softplus, dt_limit, initial_states`. A positional forward therefore put
    # dt_limit's tuple where MATE reads initial_states, and the names are identical,
    # so binding by name is both correct today and immune to that reordering.
    return implementation(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        out=out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        return_intermediate_states=return_intermediate_states,
        state_dtype=state_dtype,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
    )


# Historical name kept as a true alias: downstream code asserts identity, not
# just equal behavior.
mamba_chunk_scan_combined_varlen = ssd_combined_fwd_varlen
