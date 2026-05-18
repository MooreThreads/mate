import pytest
import torch

import mate.deep_gemm as deep_gemm
from mate.testing.utils import calc_diff
from mate.testing import supported_musa_compute_capability


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("m", [13, 137, 4096, 8192])
@pytest.mark.parametrize("n,k", [(24, 28672), (24, 7680), (24, 7168)])
@pytest.mark.parametrize("num_splits", [None, 16])
def test_hc_prenorm_gemm(m: int, n: int, k: int, num_splits: int | None) -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.mudnn.allow_tf32 = True
    a = torch.randn((m, k), dtype=torch.bfloat16, device="musa")
    b = torch.randn((n, k), dtype=torch.float32, device="musa")

    if num_splits is None:
        d = torch.empty((m, n), dtype=torch.float32, device="musa")
        s = torch.empty((m,), dtype=torch.float32, device="musa")
    else:
        d = torch.empty((num_splits, m, n), dtype=torch.float32, device="musa")
        s = torch.empty((num_splits, m), dtype=torch.float32, device="musa")

    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)

    final_d = d if num_splits is None else d.sum(0)
    final_s = s if num_splits is None else s.sum(0)

    ref_d = a.float() @ b.T
    ref_s = a.float().square().sum(-1)

    diff_d = calc_diff(final_d, ref_d)
    diff_s = calc_diff(final_s, ref_s)
    diff = max(diff_d, diff_s)

    assert diff < 1e-8, (
        f"FAILED m={m}, n={n}, k={k}, num_splits={num_splits}: "
        f"diff_d={diff_d:.2e}, diff_s={diff_s:.2e}"
    )
