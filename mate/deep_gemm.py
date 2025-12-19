import torch
import mate._C  # noqa: F401
from mate.gemm import ragged_moe_gemm_8bit, masked_moe_gemm_8bit
from typing import Tuple, Optional


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_contiguous UE8M0 cast is not supported!")

    ragged_moe_gemm_8bit(a, b, m_indices, d, recipe, alignment_m=128)


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = True,
    enable_overlap: bool = False,
    signal: torch.Tensor = None,
):
    if not disable_ue8m0_cast:
        raise Exception("m_grouped_fp8_gemm_nt_masked UE8M0 cast is not supported!")

    res = masked_moe_gemm_8bit(
        a,
        b,
        masked_m,
        d,
        recipe,
        expected_m,
        enable_overlap=enable_overlap,
        signal=signal,
    )

    return res[2:] if enable_overlap else None


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_kv: int, num_mps: int = 0
) -> torch.Tensor:
    r"""Get metadata for paged MQA logits

    Parameters
    ----------
    context_lens: Tensor
        Context lengths of each query, shape ``(batch_size)``
    block_kv: Tensor
        Block size of kv cache, **must be 64 now**.
    num_mps: int
        Number of MP to execute. 0 means use all MPs of the current device

    Returns
    -------
    Tensor
        Schedule metadata, shape ``(num_mps + 1, 2)``
    """
    return torch.ops.mate.get_paged_mqa_logits_metadata(context_lens, block_kv, num_mps)


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    r"""FP8 Paged MQA logits

    Parameters
    ----------
    q: Tensor
        The FP8 query tensor with shape ``(batch_size, next_n, heads, index_dim)``
    fused_kv_cache: Tensor
        The FP8 kv cache with fp32 scale, shape ``(num_blocks, block_size, 1, index_dim + 4)``
    weights: Tensor
        The FP32 weight tensor for each query, shape ``(batch_size * next_n, heads)``
    context_lens: Tensor
        Context lengths tensor of each query, shape ``(batch_size)``
    block_table: Tensor
        Block table tensor with shape ``(batch_size, max_blocks)``
    schedule_meta: Tensor
        Schedule metadata tensor with shape ``(num_mps + 1, 2)``
    max_context_len: int
        Maximum context length
    clean_logits: bool
        Whether to clean logits

    Returns
    -------
    Tensor
        FP32 logits, shape ``(batch_size * next_n, max_context_len)``
    """
    return torch.ops.mate.fp8_paged_mqa_logits(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
    )
