import torch
import torch_musa  # noqa: F401
import pytest

import mate
from mate.testing.utils import (
    group_quantize_fp8,
    group_dequantize_fp8,
    align,
    ceil_div,
    check_gemm_sbo_signal,
)


def get_ragged_moe_gemm_16bit_cases():
    return [
        [111],
        [111, 222, 333, 444],
        [4096 for _ in range(8)],
        [8192 for _ in range(4)],
    ]


@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_16bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("psum_layout_param", [(False, 0)])
@pytest.mark.parametrize("use_graph", [False, True])
def test_ragged_moe_gemm_16bit(
    ms_per_group, n, k, data_type, alignment_m, psum_layout_param, use_graph
):
    use_psum_layout, expected_m_for_psum_layout = psum_layout_param

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    d = torch.empty((m, n), device="musa", dtype=data_type)

    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    ref_d = torch.zeros((m, n), device="musa", dtype=data_type)

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            mate.gemm.ragged_m_moe_gemm_16bit(
                a,
                b,
                m_indices,
                d,
                gemm_mode = 'per_token',
                alignment_m=alignment_m,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(a[r], b[i].t())
            m_base += aligned_ms[i]

        g.replay()
    else:
        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(a[r], b[i].t())
            m_base += aligned_ms[i]

        mate.gemm.ragged_m_moe_gemm_16bit(
            a,
            b,
            m_indices,
            d,
            gemm_mode = 'per_token',
            alignment_m=alignment_m,
        )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)


def get_masked_moe_gemm_16bit_cases():
    return [
        [1024 for _ in range(6)],
        [192 for _ in range(32)],
        [50 for _ in range(32)],
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@pytest.mark.parametrize("ms_per_group", get_masked_moe_gemm_16bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("expected_m", [None, 128, 256])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("enable_overlap", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_masked_moe_gemm_16bit(
    ms_per_group,
    n,
    k,
    expected_m,
    data_type,
    enable_overlap,
    use_graph,
):
    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=data_type)

    tile_signal = 64
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

    if use_graph:
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g):
            res = mate.gemm.masked_moe_gemm_16bit(
                a,
                b,
                masked_m,
                d,
                expected_m,
                enable_overlap=enable_overlap,
                signal=signal,
            )
        a.uniform_(0, 1)
        b.uniform_(0, 1)
        ref_d = torch.einsum("bmk,bnk->bmn", a, b).to(data_type)

        g.replay()
    else:
        ref_d = torch.einsum("bmk,bnk->bmn", a, b).to(data_type)

        res = mate.gemm.masked_moe_gemm_16bit(
            a,
            b,
            masked_m,
            d,
            expected_m,
            enable_overlap=enable_overlap,
            signal=signal,
        )

    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )

    if enable_overlap:
        block_m = res[2]
        threshold = res[3]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


def get_ragged_moe_gemm_8bit_cases():
    return [
        [111],
        [111, 222, 333, 444],
        [256],
        [4096 for _ in range(8)],
        [8192 for _ in range(4)],
    ]


@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_8bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("alignment_m", [128, 256])
@pytest.mark.parametrize("use_graph", [False, True])
def test_ragged_moe_gemm_8bit(
    ms_per_group, n, k, a_fp8_type, b_fp8_type, out_dtype, alignment_m, use_graph
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

    if use_graph:
        g = torch.musa.MUSAGraph()
        # capture
        with torch.musa.graph(g):
            mate.gemm.ragged_m_moe_gemm_8bit(
                (fp8_a, scale_a),
                (fp8_b, scale_b),
                m_indices,
                d,
                scale_granularity_mnk=scale_granularity_mnk,
                alignment_m=alignment_m,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        new_fp8_a, new_scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
        new_fp8_b, new_scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
        )

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")

        # calc ref
        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
            m_base += aligned_ms[i]

        g.replay()

    else:
        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")

        # calc ref
        m_base = 0
        for i in range(num_expert):
            r = slice(m_base, m_base + ms_per_group[i])
            m_indices[r] = i
            ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
            m_base += aligned_ms[i]

        mate.gemm.ragged_m_moe_gemm_8bit(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            m_indices,
            d,
            scale_granularity_mnk=scale_granularity_mnk,
            alignment_m=alignment_m,
        )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


def get_masked_moe_gemm_8bit_cases():
    return [
        [1024 for _ in range(6)],
        [192 for _ in range(32)],
        [50 for _ in range(32)],
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@pytest.mark.parametrize("ms_per_group", get_masked_moe_gemm_8bit_cases())
@pytest.mark.parametrize("n", [128, 4096, 7168])
@pytest.mark.parametrize("k", [512, 4096, 7168])
@pytest.mark.parametrize("expected_m", [None, 128, 256])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("enable_overlap", [True])
@pytest.mark.parametrize("use_graph", [False, True])
def test_masked_moe_gemm_8bit(
    ms_per_group,
    n,
    k,
    expected_m,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
    enable_overlap,
    use_graph,
):
    tile_signal = 64
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=out_dtype)
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

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

    if use_graph:
        g = torch.musa.MUSAGraph()
        # capture
        with torch.musa.graph(g):
            res = mate.gemm.masked_moe_gemm_8bit(
                (fp8_a, scale_a),
                (fp8_b, scale_b),
                masked_m,
                d,
                scale_granularity_mnk,
                expected_m,
                enable_overlap=enable_overlap,
                signal=signal,
            )

        a.uniform_(0, 1)
        b.uniform_(0, 1)

        new_fp8_a, new_scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
        new_fp8_b, new_scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
        )

        fp8_a.copy_(new_fp8_a)
        fp8_b.copy_(new_fp8_b)
        scale_a.copy_(new_scale_a)
        scale_b.copy_(new_scale_b)

        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

        g.replay()
    else:
        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

        res = mate.gemm.masked_moe_gemm_8bit(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            masked_m,
            d,
            scale_granularity_mnk,
            expected_m,
            enable_overlap=enable_overlap,
            signal=signal,
        )

    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i, : ms_per_group[i], :],
            ref_d[i, : ms_per_group[i], :],
            rtol=5e-3,
            atol=5e-3,
        )

    if enable_overlap:
        block_m = res[2]
        threshold = res[3]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)

def k_grouped_contig_cases():
    return [
        [128, 128],
        [0, 256, 0, 768, 512, 0, 512, 512, 1024, 384],
        [128],
        [256],
        [111],
        [111, 222, 333, 444, 555, 666],
        [1024, 0, 512, 333, 666, 444],
        [0, 0, 0, 0, 111, 0, 0, 0, 0],
    ]

@pytest.mark.parametrize("ks_per_group",k_grouped_contig_cases())
@pytest.mark.parametrize("m", [2048, 4096, 7168])
@pytest.mark.parametrize("n", [2048, 4096, 7168])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.float])
def test_k_grouped_contig_gemm_8bit(
    ks_per_group,
    m,
    n,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
):
    quant_tile = 128
    scale_granularity_mnk = (1, 1, quant_tile)

    k = sum(ks_per_group)
    num_expert = len(ks_per_group)
    a_ = []
    b_ = []
    scale_a_ = []
    scale_b_ = []
    d = torch.rand((num_expert, m, n), device="musa", dtype=out_dtype)
    d_ref = d.clone()

    ceil_div = lambda m,n: (m + n - 1) // n

    for i in range(num_expert):
        if ks_per_group[i] == 0:
            continue
        a = torch.rand((m, ceil_div(ks_per_group[i], quant_tile) * quant_tile), device="musa", dtype=torch.float)
        b = torch.rand((n, ceil_div(ks_per_group[i], quant_tile) * quant_tile), device="musa", dtype=torch.float)
        quant_tile_shape_a = (1, quant_tile)
        quant_tile_shape_b = (1, quant_tile)
        scale_a_shape = (m, ceil_div(ks_per_group[i], quant_tile))
        scale_b_shape = (n, ceil_div(ks_per_group[i], quant_tile))
        fp8_a, scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, a_fp8_type, "K"
        )
        fp8_b, scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, b_fp8_type, "K"
        )
        dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
        dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
        d_ref[i] += dequant_a[: , : ks_per_group[i]] @ dequant_b[: , : ks_per_group[i]].T
        a_.append(fp8_a[: , : ks_per_group[i]].transpose(1, 0).contiguous())
        b_.append(fp8_b[: , : ks_per_group[i]].transpose(1, 0).contiguous())
        scale_a_.append(scale_a.transpose(1, 0).contiguous())
        scale_b_.append(scale_b.transpose(1, 0).contiguous())
    
    group_k_idx = torch.tensor(ks_per_group, device="musa", dtype=torch.int32)
    
    fp8_a = torch.cat(a_, dim=0).contiguous()
    scale_a = torch.cat(scale_a_, dim=0).contiguous()
    fp8_b = torch.cat(b_, dim=0).contiguous()
    scale_b = torch.cat(scale_b_, dim=0).contiguous()

    res = mate.gemm.ragged_k_moe_gemm_8bit(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        group_k_idx,
        d,
    )

    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i],
            d_ref[i],
            rtol=5e-3,
            atol=5e-3,
        )

@pytest.mark.parametrize("ks_per_group",k_grouped_contig_cases())
@pytest.mark.parametrize("m", [2048, 4096, 7168])
@pytest.mark.parametrize("n", [2048, 4096, 7168])
@pytest.mark.parametrize("a_type", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("b_type", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("out_dtype", [torch.float])
def test_k_grouped_contig_gemm_16bit(
    ks_per_group,
    m,
    n,
    a_type,
    b_type,
    out_dtype,
):
    if a_type != b_type:
        return
    k = sum(ks_per_group)
    num_expert = len(ks_per_group)

    a_fp32 = torch.rand((k, m), device="musa", dtype=torch.float)
    b_fp32 = torch.rand((k, n), device="musa", dtype=torch.float)
    group_k_idx = torch.tensor(ks_per_group, device="musa", dtype=torch.int32)

    d = torch.zeros((num_expert, m, n), device="musa", dtype=out_dtype)
    d_ref = d.clone()

    a_bf16 = a_fp32.clone().to(a_type)
    b_bf16 = b_fp32.clone().to(b_type)

    start_k = 0
    for i in range(num_expert):
        nr_k = ks_per_group[i]
        a_i = a_fp32[start_k: start_k + nr_k,]
        b_i = b_fp32[start_k: start_k + nr_k,]
        start_k += nr_k
        d_ref[i] += a_i.T @ b_i

    res = mate.gemm.ragged_k_moe_gemm_16bit(
        a_bf16,
        b_bf16,
        group_k_idx, 
        d,   
    )
    d = d.to(torch.float)
    for i in range(num_expert):
        torch.testing.assert_close(
            d[i],
            d_ref[i],
            rtol=5e-3,
            atol=5e-3,
        )

@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_8bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.half])
def test_m_contig_gemm_8bit(
    ms_per_group, n, k, a_fp8_type, b_fp8_type, out_dtype,
):
    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)
    ceil_div = lambda m,n: (m + n - 1) // n

    num_expert = len(ms_per_group)
    m = sum(ms_per_group)

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((num_expert, n, k), device="musa", dtype=torch.float)
    m_indices = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)
    ref_d = torch.zeros((m, n), device="musa", dtype=torch.float)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, ceil_div(k , quant_tile))
    scale_b_shape = (num_expert, ceil_div(n , quant_tile), ceil_div(k , quant_tile))
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
        ref_d[r] = torch.matmul(dequant_a[r], dequant_b[i].t())
        m_base += ms_per_group[i]
    mate.gemm.ragged_m_moe_gemm_8bit(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        m_indices,
        d,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="K",
        scale_granularity_mnk=scale_granularity_mnk,
    )

    torch.testing.assert_close(d.to(torch.float), ref_d.to(torch.float), rtol=5e-3, atol=5e-3)

@pytest.mark.parametrize("ms_per_group", get_ragged_moe_gemm_16bit_cases())
@pytest.mark.parametrize("n", [4096, 6144, 7168])
@pytest.mark.parametrize("k", [2048, 3072, 7168])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.half])
def test_m_contig_gemm_16bit(
    ms_per_group, n, k, data_type, 
):

    num_expert = len(ms_per_group)
    m = sum(ms_per_group)

    a = torch.rand((m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, k, n), device="musa", dtype=data_type)
    d = torch.empty((m, n), device="musa", dtype=data_type)

    m_indices = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    ref_d = torch.zeros((m, n), device="musa", dtype=data_type)

    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        ref_d[r] = torch.matmul(a[r], b[i])
        m_base += ms_per_group[i]

    mate.gemm.ragged_m_moe_gemm_16bit(
        a,
        b,
        m_indices,
        d,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode="N",
    )

    torch.testing.assert_close(d.to(data_type), ref_d.to(data_type), rtol=5e-3, atol=5e-3)