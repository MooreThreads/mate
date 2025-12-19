import numpy as np
import torch

from mate.testing.utils import (
    bench_gpu_time,
    group_quantize_fp8,
    make_deepgemm_contig_m_indices,
    make_deepgemm_masked_m,
)
from mate.gemm import ragged_moe_gemm_8bit, masked_moe_gemm_8bit


def bench_group_deepgemm_fp8_nt_groupwise(
    nr_group, m_per_group, n, k, select_tile_m, in_dtype, out_dtype, verbose=True
):
    m, m_indices = make_deepgemm_contig_m_indices(
        nr_group, m_per_group, "random", select_tile_m
    )

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((nr_group, n, k), device="musa", dtype=torch.float)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)

    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, k // quant_tile)
    scale_b_shape = (nr_group, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, in_dtype, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, in_dtype, "K"
    )

    measurements = bench_gpu_time(
        lambda: ragged_moe_gemm_8bit(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            m_indices,
            d,
            scale_granularity_mnk,
            alignment_m=select_tile_m,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        l2_flush=True,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    memory_bandwidth_per_second = (
        sum(
            [
                _.numel() * _.element_size()
                for _ in [fp8_a, fp8_b, scale_a, scale_b, m_indices, d]
            ]
        )
        * 1e-6
        / ms
    )

    if verbose:
        print(
            f"group_deepgemm_fp8_nt_groupwise nr_group={nr_group} m={m} n={n} k={k} select_tile_m={select_tile_m}\n"
            f"in_dtype={in_dtype} out_dtype={out_dtype}: \n"
            f"===> {tflops_per_second:.4f} TFLOPs/s\n"
            f"===> {memory_bandwidth_per_second:.4f} GB/s"
        )
        print()

    return tflops_per_second, memory_bandwidth_per_second


def bench_batch_deepgemm_fp8_nt_groupwise(
    nr_group,
    max_m_per_group,
    n,
    k,
    expected_m_per_group,
    select_tile_m,
    in_dtype,
    out_dtype,
    verbose=True,
):
    valid_m, masked_m = make_deepgemm_masked_m(
        nr_group, expected_m_per_group, max_m_per_group, "fixed"
    )

    a = torch.rand((nr_group, max_m_per_group, k), device="musa", dtype=torch.float)
    b = torch.rand((nr_group, n, k), device="musa", dtype=torch.float)

    d = torch.empty((nr_group, max_m_per_group, n), device="musa", dtype=out_dtype)

    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)
    quant_tile_shape_a = (1, 1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (nr_group, max_m_per_group, k // quant_tile)
    scale_b_shape = (nr_group, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, in_dtype, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, in_dtype, "K"
    )

    measurements = bench_gpu_time(
        lambda: masked_moe_gemm_8bit(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            masked_m,
            d,
            scale_granularity_mnk,
            expected_m_per_group,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        l2_flush=True,
    )
    ms = np.median(measurements)

    tflops_per_second = 2 * valid_m * n * k * 1e-9 / ms
    total_bytes = valid_m * k + nr_group * n * k + valid_m * n * 2
    total_bytes += (
        valid_m * (k // quant_tile) + scale_b.numel() + masked_m.numel()
    ) * 4
    memory_bandwidth_per_second = total_bytes * 1e-6 / ms

    if verbose:
        print(
            f"group_deepgemm_fp8_nt_groupwise nr_group={nr_group} max_m={max_m_per_group} expected_m = {expected_m_per_group} valid_m = {valid_m} n={n} k={k} select_tile_m={select_tile_m}\n"
            f"in_dtype={in_dtype} out_dtype={out_dtype}: \n"
            f"===> {tflops_per_second:.4f} TFLOPs/s\n"
            f"===> {memory_bandwidth_per_second:.4f} GB/s"
        )
        print()

    return tflops_per_second, memory_bandwidth_per_second


def get_group_deepgemm_fp8_nt_groupwise_cases():
    # [nr_group, m_per_group, n, k, select_tile_m]
    # valid select_tile_m: 128(default), 256
    return [
        (4, 8192, 7168, 4096, 128),
        (4, 8192, 7168, 4096, 256),
        (4, 8192, 2048, 7168, 128),
        (4, 8192, 2048, 7168, 256),
        (8, 4096, 7168, 4096, 128),
        (8, 4096, 7168, 4096, 256),
        (8, 4096, 2048, 7168, 128),
        (8, 4096, 2048, 7168, 256),
        (32, 256, 7168, 4096, 128),
        (32, 256, 7168, 4096, 256),
        (32, 256, 2048, 7168, 128),
        (32, 256, 2048, 7168, 256),
        (256, 128, 7168, 512, 128),
        (256, 128, 512, 7168, 128),
    ]


def get_batch_deepgemm_fp8_nt_groupwise_cases():
    # [nr_group, max_m_per_group, expect_m_per_group, n, k, select_tile_m]
    # valid select_tile_m: 0(default), 128, 256
    cases = []
    cases.extend(
        [
            (nr_group, 4096, expect_m_per_group, 4096, 7168, select_tile_m)
            for nr_group in [4, 8, 16]
            for expect_m_per_group in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            for select_tile_m in [128, 256]
        ]
    )

    cases.extend(
        [
            (nr_group, 4096, expect_m_per_group, 7168, 2048, select_tile_m)
            for nr_group in [4, 8, 16]
            for expect_m_per_group in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            for select_tile_m in [128, 256]
        ]
    )

    return cases


if __name__ == "__main__":
    print("=== DeepGEMM Grouped FP8 GEMM Benchmark ===\n")

    for (
        nr_group,
        max_m_per_group,
        expect_m_per_group,
        n,
        k,
        select_tile_m,
    ) in get_batch_deepgemm_fp8_nt_groupwise_cases():
        bench_batch_deepgemm_fp8_nt_groupwise(
            nr_group,
            max_m_per_group,
            n,
            k,
            expect_m_per_group,
            select_tile_m,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )

    for (
        nr_group,
        m,
        n,
        k,
        select_tile_m,
    ) in get_group_deepgemm_fp8_nt_groupwise_cases():
        bench_group_deepgemm_fp8_nt_groupwise(
            nr_group, m, n, k, select_tile_m, torch.float8_e4m3fn, torch.bfloat16
        )
