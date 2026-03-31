import torch
import torch_musa  # noqa: F401
import pytest

import mate
import mate.gemm


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("m", [128, 2048])
@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("k", [128, 2048])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_bmm_fp16(batch, m, n, k, dtype, strided):
    if strided:
        a = torch.rand((batch, m + 99, k), device="musa", dtype=dtype)[:, :m, :]
        b = torch.rand((batch, n + 111, k), device="musa", dtype=dtype)[:, :n, :]
        d = torch.empty((batch, m + 777, n), device="musa", dtype=dtype)[:, :m, :]
    else:
        a = torch.rand((batch, m, k), device="musa", dtype=dtype)
        b = torch.rand((batch, n, k), device="musa", dtype=dtype)
        d = torch.empty((batch, m, n), device="musa", dtype=dtype)

    ref_d = torch.bmm(a, b.transpose(-2, -1))

    mate.gemm.bmm_fp16(
        a,
        b.transpose(-2, -1),
        dtype,
        d,
    )

    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)
