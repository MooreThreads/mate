import torch

def get_mk_alignment_for_contiguous_layout() -> int:
    """Return the M-axis alignment requirement for contiguous grouped GEMM.

    Each expert segment in a contiguous-layout grouped GEMM must have its
    token count padded to a multiple of this value before being passed to
    m_grouped_{fp8,bf16}_gemm_nt_contiguous.

    Returns 256, matching the BlockM tile size used by MP31 kernels.
    """
    return 256


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Return a column-major, TMA-aligned view of a scale tensor.

    This function returns a contiguous copy of x without reordering data.

    Callers porting DeepGEMM code should call this on the LHS FP32 scale
    tensor (shape [m, ceil(k/128)]) before packaging it as a (fp8, scale)
    tuple — the call is a no-op here but keeps the call-site portable.
    """
    return x.contiguous()


get_mn_major_tma_aligned_tensor = get_col_major_tma_aligned_tensor
