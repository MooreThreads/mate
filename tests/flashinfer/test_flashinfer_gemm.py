import torch
import torch_musa  # noqa: F401
import pytest

import mate
from mate.testing.utils import (
    group_quantize_fp8,
    group_dequantize_fp8,
    align,
)


def get_group_deepgemm_fp8_nt_groupwise_cases():
    return [
        [111, 222, 333, 444],
        [111, 222, 333, 444, 555, 666],
    ]


@pytest.mark.parametrize("ms_per_group", get_group_deepgemm_fp8_nt_groupwise_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("alignment_m", [128, 256])
def test_group_deepgemm_fp8_nt_groupwise(
    ms_per_group, n, k, a_fp8_type, b_fp8_type, out_dtype, alignment_m
):
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")

    # calc ref
    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        m_indices[r] = i
        ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
        m_base += aligned_ms[i]

    mate.flashinfer.gemm.group_deepgemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        m_indices,
        scale_granularity_mnk,
        d,
        alignment_m=alignment_m,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


def get_batch_deepgemm_fp8_nt_groupwise_cases():
    return [
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@pytest.mark.parametrize("ms_per_group", get_batch_deepgemm_fp8_nt_groupwise_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("expected_m", [None, 8192])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_batch_deepgemm_fp8_nt_groupwise(
    ms_per_group,
    n,
    k,
    expected_m,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
):
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=out_dtype)

    quant_tile_shape_a = (1, 1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (num_expert, max_m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
    )

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
    ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

    mate.flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        masked_m,
        expected_m,
        scale_granularity_mnk,
        d,
    )

    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )
