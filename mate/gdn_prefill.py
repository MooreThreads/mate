from typing import Optional, Tuple, Union

import torch

from mate.api_logging import mate_api
from mate.gdn_kernels.tilelang import gdn_prefill as gdn_prefill_tilelang


@mate_api
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule (GDN) attention for prefill.

    This implements the gated delta rule linear attention mechanism for efficient
    training and inference. Supports both GQA (grouped query attention) and GVA
    (grouped value attention) configurations.

    Args:
        q (torch.Tensor):
            Queries of shape ``[total_seq_len, num_q_heads, head_size]``.
            Must be contiguous and on MUSA.
        k (torch.Tensor):
            Keys of shape ``[total_seq_len, num_k_heads, head_size]``.
            Must be contiguous and on MUSA.
        v (torch.Tensor):
            Values of shape ``[total_seq_len, num_v_heads, head_size]``.
            Must be contiguous and on MUSA.
        g (Optional[torch.Tensor]):
            Forget gate (alpha) of shape ``[total_seq_len, num_sab_heads]`` where
            ``num_sab_heads = max(num_q_heads, num_v_heads)``. Must be float32.
            If None, defaults to all ones. Default: ``None``.
        beta (Optional[torch.Tensor]):
            Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
            Must be float32. If None, defaults to all ones. Default: ``None``.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, defaults to ``1 / sqrt(head_size)``. Default: ``None``.
        initial_state (Optional[torch.Tensor]):
            Initial KV state of shape ``[num_seqs, num_sab_heads, head_size, head_size]``.
            Must be float32. If None, starts from zero state. Default: ``None``.
        output_final_state (bool):
            Whether to output the final state. Default: ``False``.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape ``[num_seqs + 1]``, int64.
            Required for variable-length sequences (varlen mode).
        use_qk_l2norm_in_kernel (bool):
            Whether to use QK L2 normalization in kernel. Default: ``False``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[total_seq_len, num_o_heads, head_size]``
            where ``num_o_heads = max(num_q_heads, num_v_heads)``.
            If None, will be allocated automatically. Default: ``None``.
        output_state (Optional[torch.Tensor]):
            Pre-allocated output state tensor of shape
            ``[num_seqs, num_sab_heads, head_size, head_size]``, float32.
            Required if ``output_final_state=True``. Default: ``None``.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If ``output_final_state=False``: Returns output tensor of shape
              ``[total_seq_len, num_o_heads, head_size]``.
            - If ``output_final_state=True``: Returns tuple of (output, final_state) where
              final_state has shape ``[num_seqs, num_sab_heads, head_size, head_size]``.

    Note:
        - Supports:
          - GQA: ``num_q_heads % num_k_heads == 0`` and ``num_v_heads == num_k_heads``
          - GVA: ``num_q_heads == num_k_heads`` and
            ``num_v_heads % num_q_heads == 0``
        - The final state is in k-last layout ``[N, H, V, K]``.
    """
    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"

    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must be 3D tensors: [total_tokens, heads, dim].")
    if q.size(0) != k.size(0) or q.size(0) != v.size(0):
        raise ValueError("q, k, v must have the same total token count.")
    if q.size(2) != k.size(2):
        raise ValueError("q and k must have the same dim_k.")
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    is_gqa = num_v_heads == num_k_heads and num_q_heads % num_k_heads == 0
    is_gva = num_q_heads == num_k_heads and num_v_heads % num_q_heads == 0
    if not (is_gqa or is_gva):
        raise ValueError(
            "Unsupported head configuration. "
            "Supported: GQA (num_q_heads % num_k_heads == 0 and "
            "num_v_heads == num_k_heads) or GVA "
            "(num_q_heads == num_k_heads and "
            "num_v_heads % num_q_heads == 0)."
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    head_k_size = q.size(2)
    head_v_size = v.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    if use_qk_l2norm_in_kernel:
        # note: better to do this via a fused kernel.
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

    # Allocate output if not provided
    if output is None:
        output = torch.empty(
            (total_seq_len, num_o_heads, head_v_size),
            dtype=q.dtype,
            device=q.device,
        )

    # Allocate output_state if needed
    if output_state is None:
        output_state = torch.empty(
            (num_seqs, num_sab_heads, head_v_size, head_k_size),
            dtype=torch.float32,
            device=q.device,
        )

    gdn_prefill_tilelang.gdn_prefill(
        output,
        output_state,
        q,
        k,
        v,
        cu_seqlens,
        initial_state,
        g,
        beta,
        scale,
        chunk_size,
    )

    if output_final_state:
        return output, output_state
    else:
        return output
