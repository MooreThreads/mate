import torch
from mate.gemm import ragged_m_moe_gemm_8bit, masked_moe_gemm_8bit

from typing import Tuple, Optional


def group_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    alignment_m: Optional[int] = None,
):
    if out is None:
        m = a.size(0)
        n = b.size(1)
        out_dtype = torch.bfloat16 if out_dtype is None else out_dtype

        out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    return ragged_m_moe_gemm_8bit(
        (a, a_scale), (b, b_scale), m_indices, out, scale_granularity_mnk=scale_granularity_mnk, alignment_m=alignment_m
    )


def batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    if out is None:
        batch_sz = a.size(0)
        max_m = a.size(1)
        n = b.size(1)
        out_dtype = torch.bfloat16 if out_dtype is None else out_dtype

        out = torch.empty((batch_sz, max_m, n), dtype=out_dtype, device=a.device)

    return masked_moe_gemm_8bit(
        (a, a_scale),
        (b, b_scale),
        masked_m,
        out,
        scale_granularity_mnk,
        expected_m,
        enable_overlap=False,
        signal=None,
    )
