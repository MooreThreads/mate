import torch
import mate
import math
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
    context_lens: torch.Tensor,  # 1D [batch] or 2D [batch, next_n]
    block_tables: torch.Tensor,
    max_model_len: int,
):
    """
    context_lens layout:
      1D [batch]         — shared context length for all next_n positions in a batch row
      2D [batch, next_n] — per-(batch, next_n) context length;
                           context_lens[i, j] is the number of valid KV tokens when
                           computing attention for the j-th draft token of batch i.
                           Must satisfy context_lens[i, j] <= context_lens[i, j+1].
    """
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, _ = kv_cache.size()
    is_2d = context_lens.dim() == 2

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    ctx_lens_cpu = context_lens.tolist()

    for i in range(batch_size):
        if is_2d:
            per_slot_lens = ctx_lens_cpu[i]  # list of length next_n
        else:
            per_slot_lens = [ctx_lens_cpu[i]] * next_n
        max_ctx_len = per_slot_lens[-1]

        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )  # [heads, next_n]

        for block_rk in range(ceil_div(max_ctx_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx = q[i]  # [next_n, heads, dim]
            kx = kv_cache[block_idx]  # [block_size, 1, dim]

            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device=q.device,
            )  # [block_size]

            q_offsets = torch.tensor(
                [per_slot_lens[j] - 1 for j in range(next_n)],
                device=q.device,
                dtype=torch.long,
            )  # [next_n]

            mask = k_offsets[None, :] <= q_offsets[:, None]  # [next_n, block_size]

            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]  # [heads, next_n, block_size]
            s = s.sum(dim=0)  # [next_n, block_size]

            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(mask, s, float("-inf"))

    return logits


@pytest.mark.parametrize("batch_size, next_n", [(64, 1), (64, 4), (64, 2), (128, 1)])
@pytest.mark.parametrize("avg_kv", [8192, 32768])
@pytest.mark.parametrize("is_context_lens_2d", [False, True])
def test_paged_mqa_logits(batch_size, next_n, avg_kv, is_context_lens_2d):
    # 2D context_lens only makes sense when next_n > 1
    if is_context_lens_2d and next_n == 1:
        pytest.skip(
            "2D context_lens with next_n=1 is identical to 1D; skip to avoid redundancy"
        )

    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 111 * 1000
    for heads, index_dim in [(32, 128)]:
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

        base_lens = torch.randint(
            int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size,)
        ).to(torch.int32)
        base_lens = torch.ones(batch_size, dtype=torch.int32) * avg_kv

        if is_context_lens_2d:
            offsets = torch.arange(next_n, dtype=torch.int32)  # [0, 1, ..., next_n-1]
            context_lens_cpu = (
                base_lens.unsqueeze(1) - (next_n - 1) + offsets.unsqueeze(0)
            )  # [batch, next_n]
            context_lens = context_lens_cpu.musa()
            max_context_len_per_row = context_lens[:, -1]
        else:
            context_lens = base_lens.musa()
            max_context_len_per_row = context_lens

        max_block_len = (
            (max_context_len_per_row.max().item() + blocksize - 1)
            // blocksize
            * blocksize
        )
        block_tables = torch.zeros(
            (batch_size, max_block_len), device="musa", dtype=torch.int32
        )

        counter = 0
        block_idx_pool = list(range(num_blocks))
        random.shuffle(block_idx_pool)
        for i in range(batch_size):
            ctx_len = max_context_len_per_row[i].item()
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
        )  # [batch*next_n, max_model_len]

        if is_context_lens_2d:
            per_slot_lens = context_lens.view(-1)  # [batch*next_n]
            ref_neginf_mask = positions >= per_slot_lens.unsqueeze(1)
        else:
            row_indices = torch.arange(batch_size * next_n, device="musa") // next_n
            next_n_offset = torch.arange(batch_size * next_n, device="musa") % next_n
            q_positions = (
                context_lens[row_indices] - next_n + next_n_offset
            ).unsqueeze(1)
            ref_neginf_mask = positions > q_positions

        neginf_mask = logits == float("-inf")
        assert torch.equal(neginf_mask, ref_neginf_mask), (
            "neginf mask mismatch between kernel and reference"
        )

        logits = logits.masked_fill(ref_neginf_mask, 0)
        ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
        diff = calc_diff(logits, ref_logits)
        assert diff < 1e-3, f"{diff=}"

        sum_lens = max_context_len_per_row.to(torch.int64).sum().item()
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

        if t > 0.0 and clean_t > 0.0:
            clean_bytes = (
                batch_size * next_n * max_model_len - neginf_mask.sum().item()
            ) * 4 + count_bytes(context_lens)

            print(
                f" > BSZ={batch_size:3}, NextN={next_n:1}, 2D={is_context_lens_2d}, "
                f"H={heads:2}, D={index_dim:3}, L={avg_kv:6}: "
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
@pytest.mark.parametrize("alignment_m", [128, 256])
def test_m_grouped_fp8_gemm_nt_contiguous(
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

    mate.deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        m_indices,
        scale_granularity_mnk,
        alignment_m=alignment_m,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d.to(torch.float), ref_d, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("ms_per_group", get_deepgemm_group_gemm_contig_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
@pytest.mark.parametrize("alignment_m", [128, 256])
def test_m_grouped_bf16_gemm_nt_contiguous(ms_per_group, n, k, data_type, alignment_m):
    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    m_indices = torch.full((m,), -1, device="musa", dtype=torch.int32)

    d = torch.empty((m, n), device="musa", dtype=data_type)
    ref_d = torch.zeros((m, n), device="musa", dtype=data_type)

    # calc ref
    m_base = 0
    for i in range(num_expert):
        r = slice(m_base, m_base + ms_per_group[i])
        m_indices[r] = i
        ref_d[r] = torch.matmul(a[r], b[i].t())
        m_base += aligned_ms[i]

    mate.deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        a,
        b,
        d,
        m_indices,
        alignment_m=alignment_m,
    )

    d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
    torch.testing.assert_close(d, ref_d, rtol=5e-3, atol=5e-3)


def get_m_grouped_gemm_nt_masked_cases():
    return [
        [256, 256, 256, 256],
        [333, 444, 555, 666],
    ]


@pytest.mark.parametrize("ms_per_group", get_m_grouped_gemm_nt_masked_cases())
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

    res = mate.deep_gemm.fp8_m_grouped_gemm_nt_masked(
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


def kv_cache_cast_to_fp8_for_nonpaged(kv: torch.Tensor, block_size: int = 64):
    assert kv.dim() == 2
    seq_kv, head_dim = kv.shape
    num_blocks = (seq_kv + block_size - 1) // block_size
    padded = num_blocks * block_size

    if padded != seq_kv:
        kv_pad = torch.zeros((padded, head_dim), device=kv.device, dtype=kv.dtype)
        kv_pad[:seq_kv].copy_(kv)
    else:
        kv_pad = kv

    x = kv_pad.view(num_blocks, block_size, 1, head_dim)
    _, x_scaled, sf = kv_cache_cast_to_fp8(x)

    kv_fp8 = x_scaled.view(padded, head_dim)[:seq_kv]
    kv_scale = sf.view(padded)[:seq_kv]
    return kv_fp8, kv_scale


def generate_cp_test_data(seq_len, seq_len_kv, device):
    assert seq_len % 2 == 0
    chunk_size = seq_len // 2

    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.empty(seq_len, dtype=torch.int32, device=device)

    base0 = seq_len_kv // 3
    base1 = (seq_len_kv * 2) // 3

    t = torch.arange(chunk_size, dtype=torch.int32, device=device)
    ke0 = base0 + t
    ke1 = base1 + t

    ke0 = ke0.clamp(min=0, max=seq_len_kv)
    ke1 = ke1.clamp(min=0, max=seq_len_kv)

    ke[:chunk_size] = ke0
    ke[chunk_size:] = ke1
    return ks, ke


def ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    cost_only: bool = False,
    kv_chunk_size: int = 4096,
):
    device = q.device
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_kv = kv.shape[0]

    ks = cu_seqlen_ks.clamp(min=0, max=seq_len_kv)
    ke = cu_seqlen_ke.clamp(min=0, max=seq_len_kv)

    if cost_only:
        count_ones_per_row = (ke - ks).clamp(min=0)
        return count_ones_per_row.sum()

    q_f = q.to(torch.float32)
    k_f = kv.to(torch.float32)
    w_f = weights.to(torch.float32)

    cost = (ke - ks).clamp(min=0).sum()
    logits = torch.zeros(
        (seq_len_q, seq_len_kv),
        device=device,
        dtype=torch.float32,
    )

    for kv_start in range(0, seq_len_kv, kv_chunk_size):
        kv_end = min(seq_len_kv, kv_start + kv_chunk_size)
        cur_kv_len = kv_end - kv_start
        k_block = k_f[kv_start:kv_end, :]
        logits_block = torch.zeros(
            (seq_len_q, cur_kv_len),
            device=device,
            dtype=torch.float32,
        )

        for h in range(num_heads):
            q_h = q_f[:, h, :]
            w_h = w_f[:, h].unsqueeze(-1)
            score_block = q_h @ k_block.t()
            logits_block += torch.relu(score_block) * w_h

        positions_block = torch.arange(
            kv_start,
            kv_end,
            device=device,
        ).unsqueeze(0)

        mask_lo = positions_block >= ks.unsqueeze(1)
        mask_hi = positions_block < ke.unsqueeze(1)
        mask_block = mask_lo & mask_hi

        logits_block = logits_block.masked_fill(
            ~mask_block,
            float("-inf"),
        )

        logits[:, kv_start:kv_end] = logits_block

    return logits, cost


@pytest.mark.parametrize("seq_q", [2048, 4096])
@pytest.mark.parametrize("compressed_logits", [True, False])
@pytest.mark.parametrize(
    "seq_kv",
    [4000, 8192],
)
@pytest.mark.parametrize("disable_cp", [False, True])
def test_mqa_logits(seq_q, seq_kv, compressed_logits, disable_cp):
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("musa")
    num_heads, head_dim = 32, 32
    q = torch.randn(
        (seq_q, num_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        (seq_kv, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (seq_q, num_heads),
        device=device,
        dtype=torch.float32,
    )

    if disable_cp:
        ks = torch.zeros(seq_q, dtype=torch.int32, device=device)
        ke = torch.arange(seq_q, dtype=torch.int32, device=device) + (seq_kv - seq_q)
    else:
        ks, ke = generate_cp_test_data(seq_q, seq_kv, device=device)

    ks = ks.clamp(0, seq_kv)
    ke = ke.clamp(0, seq_kv)
    ke = torch.maximum(ke, ks)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scale = kv_cache_cast_to_fp8_for_nonpaged(kv)
    seq_kv_orig = seq_kv

    if compressed_logits:
        max_seqlen_k = (ke - ks).max().item()
        logits_comp = mate.deep_gemm.fp8_mqa_logits(
            q_fp8,
            (kv_fp8, kv_scale),
            weights,
            ks,
            ke,
            clean_logits=False,
            max_seqlen_k=max_seqlen_k,
        )
        assert logits_comp.shape == (seq_q, max_seqlen_k)

        logits = torch.full(
            (seq_q, seq_kv),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        for i in range(seq_q):
            start_i = int(ks[i].item())
            end_i = int(ke[i].item())
            length_i = max(0, end_i - start_i)
            if length_i > 0:
                logits[i, start_i:end_i] = logits_comp[i, :length_i]
    else:
        logits_pad = mate.deep_gemm.fp8_mqa_logits(
            q_fp8,
            (kv_fp8, kv_scale),
            weights,
            ks,
            ke,
            clean_logits=True,
        )
        logits = logits_pad[:, :seq_kv_orig]

    torch.musa.synchronize()

    do_check = seq_kv < 20000
    if do_check:
        ref_logits, ref_cost = ref_fp8_mqa_logits(
            q=q,
            kv=kv,
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
        )

        ref_neginf_mask = ref_logits == float("-inf")
        neginf_mask = logits == float("-inf")
        assert torch.equal(neginf_mask, ref_neginf_mask)

        logits_masked = logits.masked_fill(neginf_mask, 0)
        ref_logits_masked = ref_logits.masked_fill(ref_neginf_mask, 0)

        diff = calc_diff(logits_masked, ref_logits_masked)
        assert diff < 1e-3, f"{diff=}"

        ref_cost_val = float(ref_cost)
    else:
        ref_cost = ref_fp8_mqa_logits(
            q=q,
            kv=kv,
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            cost_only=True,
        )
        ref_cost_val = float(ref_cost)

    tflops_nominal = 2.0 * ref_cost_val * num_heads * head_dim / 1e12
    arith_bytes = (
        count_bytes(q_fp8, kv_fp8, weights, ks, ke, kv_scale) + ref_cost_val * 4
    )
    clean_bytes = (seq_q * seq_kv - ref_cost_val) * 4 + count_bytes(ks, ke)

    if compressed_logits:

        def run_main():
            mate.deep_gemm.fp8_mqa_logits(
                q_fp8,
                (kv_fp8, kv_scale),
                weights,
                ks,
                ke,
                clean_logits=False,
                max_seqlen_k=max_seqlen_k,
            )

        t = bench_kineto(
            run_main,
            "Fp8NonPagedMqaLogits",
        )

        print(
            f"[fp8_mqa_logits] "
            f"S={seq_q:4d}, SKV={seq_kv:6d}, "
            f"H={num_heads:3d}, D={head_dim:3d}, "
            f"CP={0 if disable_cp else 1}: "
            f"(compressed logits)"
        )
    else:

        def run_all():
            mate.deep_gemm.fp8_mqa_logits(
                q_fp8,
                (kv_fp8, kv_scale),
                weights,
                ks,
                ke,
                clean_logits=True,
            )

        t, clean_t = bench_kineto(
            run_all,
            ("Fp8NonPagedMqaLogits", "mpxx_clean_logits"),
        )

        print(
            f"[fp8_mqa_logits] "
            f"S={seq_q:4d}, SKV={seq_kv:6d}, "
            f"H={num_heads:3d}, D={head_dim:3d}, "
            f"CP={0 if disable_cp else 1}: "
        )


@pytest.mark.parametrize("ms_per_group", get_m_grouped_gemm_nt_masked_cases())
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("expected_m", [None, 8192])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
@pytest.mark.parametrize("enable_overlap", [True])
def test_m_grouped_bf16_gemm_nt_masked(
    ms_per_group,
    n,
    k,
    expected_m,
    data_type,
    enable_overlap,
):
    tile_signal = 64

    max_m = max(ms_per_group)
    expected_m = expected_m if expected_m is not None else max_m

    num_expert = len(ms_per_group)

    a = torch.rand((num_expert, max_m, k), device="musa", dtype=data_type)
    b = torch.rand((num_expert, n, k), device="musa", dtype=data_type)
    masked_m = torch.tensor(ms_per_group, device="musa", dtype=torch.int32)

    d = torch.empty((num_expert, max_m, n), device="musa", dtype=data_type)
    signal = torch.zeros(
        num_expert * ceil_div(max_m, tile_signal), dtype=torch.int32, device=a.device
    )

    ref_d = torch.einsum("bmk,bnk->bmn", a, b)
    res = mate.deep_gemm.bf16_m_grouped_gemm_nt_masked(
        a,
        b,
        d,
        masked_m,
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
        block_m = res[0]
        threshold = res[1]
        check_gemm_sbo_signal(num_expert, max_m, block_m, threshold, signal, masked_m)


if __name__ == "__main__":
    test_paged_mqa_logits()
    test_mqa_logits()
