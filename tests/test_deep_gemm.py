import torch
import mate
import pytest
import random
from mate.testing.utils import (
    align,
    bench_kineto,
    ceil_div,
    group_quantize_fp8,
    group_dequantize_fp8,
    check_gemm_sbo_signal,
)


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), x_scaled, sf


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(ceil_div(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@pytest.mark.parametrize("batch_size, next_n", [(64, 1), (64, 2), (128, 1)])
@pytest.mark.parametrize("avg_kv", [8192, 32768])
def test_paged_mqa_logits(batch_size, next_n, avg_kv):
    torch.manual_seed(0)
    random.seed(0)
    print("Testing FP8 Paged MQA Logits:")
    max_model_len = 111 * 1000
    for heads, index_dim in [(64, 128)]:
        num_blocks, blocksize = max_model_len * 3, 64

        q = torch.randn(
            (batch_size, next_n, heads, index_dim),
            device="musa",
            dtype=torch.bfloat16,
        )
        kv_cache = torch.randn(
            (num_blocks, blocksize, 1, index_dim),
            device="musa",
            dtype=torch.bfloat16,
        )
        weights = torch.randn(
            (batch_size * next_n, heads), device="musa", dtype=torch.float32
        )

        context_lens = (
            torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size,))
            .musa()
            .to(torch.int32)
        )
        context_lens = torch.ones(batch_size).musa().to(torch.int32) * avg_kv
        max_block_len = (
            (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
        )
        block_tables = torch.zeros(
            (batch_size, max_block_len), device="musa", dtype=torch.int32
        )

        counter = 0
        block_idx_pool = list(range(num_blocks))
        random.shuffle(block_idx_pool)
        for i in range(batch_size):
            ctx_len = context_lens[i].item()
            for j in range(ceil_div(ctx_len, blocksize)):
                block_tables[i][j] = block_idx_pool[counter]
                counter += 1

        ref_logits = ref_fp8_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )

        q_fp8 = q.to(torch.float8_e4m3fn)
        kv_cache_fp8, x_scaled, sf = kv_cache_cast_to_fp8(kv_cache)
        schedule_metadata = mate.deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, blocksize
        )
        logits = mate.deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            True,
        )

        positions = (
            torch.arange(max_model_len, device="musa")
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )
        row_indices = torch.arange(batch_size * next_n, device="musa") // next_n
        next_n_offset = torch.arange(batch_size * next_n, device="musa") % next_n
        ref_neginf_mask = ~(
            positions
            <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
        )

        neginf_mask = logits == float("-inf")
        assert torch.equal(neginf_mask, ref_neginf_mask)

        logits = logits.masked_fill(ref_neginf_mask, 0)
        ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
        diff = calc_diff(logits, ref_logits)
        assert diff < 1e-3, f"{diff=}"

        sum_lens = sum(context_lens.to(torch.int64))
        tflops = 2 * sum_lens * next_n * heads * index_dim / 1e12
        input_bytes = (
            count_bytes(q_fp8, weights, context_lens)
            + sum_lens * (index_dim + 4)
            + (sum_lens / blocksize) * 4
        )
        output_bytes = sum_lens * next_n * 4
        t, clean_t = bench_kineto(
            lambda: mate.deep_gemm.fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
                True,
            ),
            ("PagedMqaLogits", "clean_logits"),
        )
        clean_bytes = (
            batch_size * next_n * max_model_len - neginf_mask.sum().item()
        ) * 4 + count_bytes(context_lens)

        print(
            f" > BSZ={batch_size:3}, NextN={next_n:1}, H={heads:2}, D={index_dim:2}, L={avg_kv:6}: "
            f"{tflops / t:4.0f} TFLOPS, {t * 1e6:3.0f} us, "
            f"{(input_bytes + output_bytes) / t / 1e9:4.0f} GB/s | "
            f"clean: {clean_t * 1e6:3.0f} us, {clean_bytes / clean_t / 1e9:4.0f} GB/s"
        )


def get_deepgemm_group_gemm_contig_cases():
    return [
        [4096 for _ in range(8)],
        [8192 for _ in range(4)],
    ]


@pytest.mark.parametrize("ms_per_group", get_deepgemm_group_gemm_contig_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
def test_m_grouped_fp8_gemm_nt_contiguous(
    ms_per_group, n, k, a_fp8_type, b_fp8_type, out_dtype
):
    alignment_m = 128
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

    mate.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        m_indices,
        scale_granularity_mnk,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


def get_m_grouped_fp8_gemm_nt_masked_cases():
    return [
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@pytest.mark.parametrize("ms_per_group", get_m_grouped_fp8_gemm_nt_masked_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("expected_m", [None, 8192])
@pytest.mark.parametrize("a_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("b_fp8_type", [torch.float8_e4m3fn])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_overlap", [True])
def test_m_grouped_fp8_gemm_nt_masked(
    ms_per_group,
    n,
    k,
    expected_m,
    a_fp8_type,
    b_fp8_type,
    out_dtype,
    enable_overlap,
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

    dequant_a = group_dequantize_fp8(fp8_a, scale_a, "K")
    dequant_b = group_dequantize_fp8(fp8_b, scale_b, "K")
    ref_d = torch.einsum("bmk,bnk->bmn", dequant_a, dequant_b).to(torch.float)

    res = mate.deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        masked_m,
        expected_m,
        scale_granularity_mnk,
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
        block_m = res[0]
        threshold = res[1]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


if __name__ == "__main__":
    test_paged_mqa_logits()
