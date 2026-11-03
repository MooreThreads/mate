import os

import torch
import tilelang
import tilelang.language as T

from mate.execution_context import skip_kernel_launch_if_dry_run

# Opt-in host-side validation. Enabling it reads ``state_indices`` back to the
# host, so it must not be used inside CUDA graph capture or in production steps.
_DEBUG_STATE_VALIDATION = os.environ.get("MATE_GDN_DEBUG_STATE_VALIDATION") == "1"


def _view_byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the ``[lo, hi)`` byte range the view addresses in its storage."""
    element = tensor.element_size()
    lo = hi = tensor.storage_offset()
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size > 1:
            step = (size - 1) * stride
            if step < 0:
                lo += step
            else:
                hi += step
    return lo * element, (hi + 1) * element


def check_state_views(initial_state: torch.Tensor, final_state: torch.Tensor) -> None:
    """Reject partially overlapping input/output state views.

    Exact in-place use of one view is supported. Views over the same storage are
    rejected only when their byte ranges actually intersect, so distinct pools
    carved out of one allocation keep working.
    """
    if initial_state is final_state:
        return
    if (
        initial_state.data_ptr() == final_state.data_ptr()
        and initial_state.shape == final_state.shape
        and initial_state.stride() == final_state.stride()
    ):
        return
    base = initial_state.data_ptr() - (
        initial_state.storage_offset() * initial_state.element_size()
    )
    other_base = final_state.data_ptr() - (
        final_state.storage_offset() * final_state.element_size()
    )
    if base != other_base:
        return
    lo, hi = _view_byte_span(initial_state)
    other_lo, other_hi = _view_byte_span(final_state)
    if lo < other_hi and other_lo < hi:
        raise ValueError(
            "initial_state and output_state overlap; pass the same view for "
            "in-place use or two disjoint pools."
        )


def _check_index_bounds(
    state_indices: torch.Tensor | None,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
) -> None:
    """Debug-only bound check for non-negative state indices."""
    if not _DEBUG_STATE_VALIDATION or state_indices is None:
        return
    if state_indices.numel() == 0:
        return
    try:
        if torch.musa.is_current_stream_capturing():
            return
    except (AttributeError, RuntimeError):
        pass
    largest = int(state_indices.max())
    for name, pool in (
        ("initial_state", initial_state),
        ("output_state", final_state),
    ):
        if largest >= pool.shape[0]:
            raise ValueError(
                f"state index {largest} is out of bounds for {name} with "
                f"{pool.shape[0]} rows."
            )


def prepare_prefill_states(
    initial_state: torch.Tensor | None,
    output_state: torch.Tensor | None,
    *,
    num_seqs: int,
    state_shape: tuple[int, int, int],
    device: torch.device,
    output_final_state: bool,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if state_indices is not None or has_initial_state is not None:
        for name, tensor, dtypes in (
            ("state_indices", state_indices, (torch.int32, torch.int64)),
            ("has_initial_state", has_initial_state, (torch.bool,)),
        ):
            if tensor is not None:
                if tensor.shape != (num_seqs,):
                    raise ValueError(
                        f"{name} must have shape [{num_seqs}], got {tensor.shape}."
                    )
                if tensor.dtype not in dtypes:
                    raise ValueError(
                        f"{name} must have dtype in {dtypes}, got {tensor.dtype}."
                    )
                if tensor.device != device:
                    raise ValueError(
                        f"{name} must be on {device}, got {tensor.device}."
                    )

    if has_initial_state is not None and initial_state is None:
        raise ValueError("has_initial_state requires initial_state.")
    if state_indices is not None and output_final_state and output_state is None:
        raise ValueError(
            "output_state is required with state_indices and output_final_state=True."
        )

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (num_seqs, *state_shape), dtype=torch.float32, device=device
        )
    elif state_indices is not None or has_initial_state is not None:
        if initial_state.ndim != 4 or initial_state.shape[1:] != state_shape:
            raise ValueError(
                f"Expected initial_state shape [rows, {state_shape}], got {initial_state.shape}."
            )
        if state_indices is None and initial_state.shape[0] != num_seqs:
            raise ValueError(
                f"Expected {num_seqs} initial states, got {initial_state.shape[0]}."
            )
        if initial_state.dtype != torch.float32 or initial_state.device != device:
            raise ValueError(f"initial_state must be float32 on {device}.")
        if num_seqs and initial_state.shape[0] == 0:
            raise ValueError("initial_state pool must not be empty.")

    final_state_shape = (num_seqs, *state_shape)
    if output_state is None:
        final_state = torch.empty(final_state_shape, dtype=torch.float32, device=device)
    else:
        if state_indices is None:
            valid_shape = output_state.shape == final_state_shape
        else:
            valid_shape = (
                output_state.ndim == 4 and output_state.shape[1:] == state_shape
            )
        if not valid_shape:
            expected = (
                final_state_shape
                if state_indices is None
                else ("pool_size", *state_shape)
            )
            raise ValueError(
                f"Expected output_state shape {expected}, got {output_state.shape}."
            )
        if output_state.dtype != torch.float32:
            raise ValueError(
                f"Expected output_state dtype torch.float32, got {output_state.dtype}."
            )
        if output_state.device != device:
            raise ValueError(
                f"Expected output_state device {device}, got {output_state.device}."
            )
        if state_indices is not None and num_seqs and output_state.shape[0] == 0:
            raise ValueError("output_state pool must not be empty.")
        final_state = output_state
    check_state_views(initial_state, final_state)
    _check_index_bounds(state_indices, initial_state, final_state)
    return initial_state, final_state, use_initial_state


@tilelang.jit
def _prefill_state_copy(
    H, V, K, index_dtype, mask_dtype, use_indices, use_mask, use_initial_state
):
    h0_shape = (T.dynamic("h0_rows"), H, V, K)
    ht_shape = (T.dynamic("ht_rows"), H, V, K)
    h0_strides = tuple(T.dynamic(f"h0_stride_{i}", "int64") for i in range(4))
    ht_strides = tuple(T.dynamic(f"ht_stride_{i}", "int64") for i in range(4))
    indices_shape = (T.dynamic("indices_n"),)
    mask_shape = (T.dynamic("mask_n"),)
    indices_strides = (T.dynamic("indices_stride", "int64"),)
    mask_strides = (T.dynamic("mask_stride", "int64"),)

    @T.prim_func
    def kernel(
        h0: T.StridedTensor(h0_shape, h0_strides, "float32"),
        ht: T.StridedTensor(ht_shape, ht_strides, "float32"),
        indices: T.StridedTensor(indices_shape, indices_strides, index_dtype),
        mask: T.StridedTensor(mask_shape, mask_strides, mask_dtype),
        num_seqs: T.int32,
    ):
        with T.Kernel(T.ceildiv(H * V * K, 256), num_seqs, threads=256) as (block, bb):
            slot = T.alloc_var("int64")
            valid = T.alloc_var("bool")
            read_initial = T.alloc_var("bool")
            if use_indices:
                slot = T.cast(indices[bb], "int64")
                valid = slot >= 0
                slot = T.max(slot, 0)
            else:
                slot = T.cast(bb, "int64")
                valid = True
            read_initial = False
            if use_initial_state:
                read_initial = valid
                if use_mask:
                    read_initial = valid and mask[bb]
            for tx in T.Parallel(256):
                i = block * 256 + tx
                if i < H * V * K and valid:
                    h, v, k = i // (V * K), i // K % V, i % K
                    value = T.alloc_var("float32")
                    value = 0.0
                    if read_initial:
                        value = h0[slot, h, v, k]
                    ht[slot, h, v, k] = value

    return kernel


def pass_through_prefill_states(
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    num_seqs: int,
    use_initial_state: bool,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
) -> None:
    if state_indices is None and has_initial_state is None:
        if use_initial_state:
            final_state.copy_(initial_state)
        else:
            final_state.zero_()
        return
    if num_seqs == 0:
        return
    indices = state_indices if state_indices is not None else cu_seqlens
    mask = has_initial_state if has_initial_state is not None else cu_seqlens
    kernel = _prefill_state_copy(
        *final_state.shape[1:],
        index_dtype=indices.dtype,
        mask_dtype=mask.dtype,
        use_indices=state_indices is not None,
        use_mask=has_initial_state is not None,
        use_initial_state=use_initial_state,
    )
    if not skip_kernel_launch_if_dry_run():
        kernel(initial_state, final_state, indices, mask, num_seqs)
