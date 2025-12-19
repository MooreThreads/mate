import torch
import torch_musa  # noqa: F401
import pytest

import mate
from mate.testing.utils import (
    tensor_quantize_fp8,
    group_quantize_fp8,
    group_dequantize_fp8,
)


@pytest.mark.parametrize("m", [128, 4096])
@pytest.mark.parametrize("n", [128, 4096])
@pytest.mark.parametrize("k", [128, 4096])
@pytest.mark.parametrize(
    "ab_fp8_type",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize(
    "scale_granularity_mnk", [(1, 128, 128), (1, 1, -1), (1, -1, -1)]
)
@pytest.mark.parametrize("scale_major", ["K", "MN"])
@pytest.mark.parametrize("use_graph", [False, True])
def test_groupwise_fp8_scaled_gemm(
    m, n, k, ab_fp8_type, out_dtype, scale_granularity_mnk, scale_major, use_graph
):
    a_fp8_type, b_fp8_type = ab_fp8_type

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    quant_tile_shape_a = (scale_granularity_m, scale_granularity_k)
    quant_tile_shape_b = (scale_granularity_n, scale_granularity_k)
    if scale_major == "K":
        scale_a_shape = (m // scale_granularity_m, k // scale_granularity_k)
        scale_b_shape = (n // scale_granularity_n, k // scale_granularity_k)
    else:
        # MN Major
        scale_a_shape = (k // scale_granularity_k, m // scale_granularity_m)
        scale_b_shape = (k // scale_granularity_k, n // scale_granularity_n)

    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
    )
    if scale_granularity_mnk[1] == -1 and scale_granularity_mnk[2] == -1:
        fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)
    else:
        fp8_b, scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
        )

    if use_graph:
        g = torch.musa.MUSAGraph()
        # capture
        with torch.musa.graph(g):
            mate.gemm.gemm_fp8_nt_groupwise(
                fp8_a,
                fp8_b,
                scale_a,
                scale_b,
                scale_major,
                1,
                scale_granularity_mnk,
                d,
                out_dtype,
                "mudnn",
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        new_fp8_a, new_scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
        )
        if scale_granularity_mnk[1] == -1 and scale_granularity_mnk[2] == -1:
            new_fp8_b, new_scale_b = tensor_quantize_fp8(b, b_fp8_type)
        else:
            new_fp8_b, new_scale_b = group_quantize_fp8(
                b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
            )

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
        ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)
        ref_d = torch.matmul(ref_a, ref_b.t())

        g.replay()

    else:
        ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
        ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)
        ref_d = torch.matmul(ref_a, ref_b.t())

        mate.gemm.gemm_fp8_nt_groupwise(
            fp8_a,
            fp8_b,
            scale_a,
            scale_b,
            scale_major,
            1,
            scale_granularity_mnk,
            d,
            out_dtype,
            "mudnn",
        )

    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("m", [128, 4096])
@pytest.mark.parametrize("n", [128, 4096])
@pytest.mark.parametrize("k", [128, 4096])
@pytest.mark.parametrize(
    "ab_fp8_type",
    [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize(
    "scale_granularity_mnk", [(1, 128, 128), (1, 1, -1), (1, -1, -1)]
)
@pytest.mark.parametrize("scale_major", ["K", "MN"])
def test_groupwise_fp8_scaled_gemm_not_contiguous_out(
    m, n, k, ab_fp8_type, out_dtype, scale_granularity_mnk, scale_major
):
    a_fp8_type, b_fp8_type = ab_fp8_type

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((n, k), device="musa", dtype=torch.float)

    d_factor = 2
    d_shape = (m, n)
    d_stride = (n * d_factor, 1)
    d_storage = sum((s - 1) * st for s, st in zip(d_shape, d_stride)) + 1
    d_storage_tensor = torch.empty(d_storage, dtype=out_dtype, device="musa")
    d = torch.as_strided(d_storage_tensor, size=d_shape, stride=d_stride)

    scale_granularity_m, scale_granularity_n, scale_granularity_k = (
        scale_granularity_mnk
    )
    scale_granularity_m = m if scale_granularity_m == -1 else scale_granularity_m
    scale_granularity_n = n if scale_granularity_n == -1 else scale_granularity_n
    scale_granularity_k = k if scale_granularity_k == -1 else scale_granularity_k

    quant_tile_shape_a = (scale_granularity_m, scale_granularity_k)
    quant_tile_shape_b = (scale_granularity_n, scale_granularity_k)
    if scale_major == "K":
        scale_a_shape = (m // scale_granularity_m, k // scale_granularity_k)
        scale_b_shape = (n // scale_granularity_n, k // scale_granularity_k)
    else:
        # MN Major
        scale_a_shape = (k // scale_granularity_k, m // scale_granularity_m)
        scale_b_shape = (k // scale_granularity_k, n // scale_granularity_n)

    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, a_fp8_type, scale_major
    )
    if scale_granularity_mnk[1] == -1 and scale_granularity_mnk[2] == -1:
        fp8_b, scale_b = tensor_quantize_fp8(b, b_fp8_type)
    else:
        fp8_b, scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, scale_major
        )

    ref_a = group_dequantize_fp8(fp8_a, scale_a, scale_major)
    ref_b = group_dequantize_fp8(fp8_b, scale_b, scale_major)
    ref_d = torch.matmul(ref_a, ref_b.t())

    mate.gemm.gemm_fp8_nt_groupwise(
        fp8_a,
        fp8_b,
        scale_a,
        scale_b,
        scale_major,
        1,
        scale_granularity_mnk,
        d,
        out_dtype,
        "mudnn",
    )

    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)
