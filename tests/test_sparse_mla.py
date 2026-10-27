from __future__ import annotations

import pytest
import torch

import mate
from mate.execution_context import raise_complete_if_dry_run
from mate.sparse_mla_interface import (
    get_batch_decode_metadata_mla,
    mla_rope_quantize_fp8,
    sparse_mla_fp8_decode,
)
from mate.testing import supported_musa_compute_capability
from mate.testing.sparse_mla import (
    Fp8DecodeCase,
    RopeCase,
    SparseMlaCase,
    assert_sparse_mla_close,
    make_decode_data,
    make_fp8_decode_data,
    make_fp8_decode_stress_data,
    make_prefill_data,
    make_rope_data,
)


pytestmark = pytest.mark.mate_dry_run


PREFILL_CASES = (
    SparseMlaCase("v32_regular", "v32", 1, 128, 64, 128),
    SparseMlaCase(
        "v32_features",
        "v32",
        62,
        95,
        128,
        128,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
    ),
    SparseMlaCase("v4_regular", "v4", 1, 256, 64, 256),
    SparseMlaCase(
        "v4_features",
        "v4",
        62,
        153,
        128,
        256,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
    ),
    SparseMlaCase("v32_large", "v32", 213, 1592, 128, 384, full=True),
    SparseMlaCase("v4_large", "v4", 213, 1840, 128, 256, full=True),
)


ROPE_CASES = (
    RopeCase("neox_fp32", 5, 3, True, torch.float32),
    RopeCase("interleaved_bf16", 62, 64, False, torch.bfloat16),
    RopeCase("neox_bf16", 19, 16, True, torch.bfloat16),
    RopeCase("large", 213, 128, False, torch.float32, full=True),
)


DECODE_CASES = (
    SparseMlaCase("v32_page2", "v32", 1, 512, 64, 64, batch=4, page_size=2),
    # Sink preload must ignore the padding lanes of a partial head tile.
    SparseMlaCase(
        name="v32_sink_padded_heads",
        variant="v32",
        q_len=3,
        kv_len=512,
        heads=24,
        topk=64,
        batch=4,
        attn_sink=True,
        invalid_rows=True,
    ),
    SparseMlaCase(
        name="v4_sink_padded_heads",
        variant="v4",
        q_len=3,
        kv_len=512,
        heads=24,
        topk=64,
        batch=4,
        attn_sink=True,
        invalid_rows=True,
    ),
    SparseMlaCase(
        "v32_irregular",
        "v32",
        3,
        650,
        128,
        576,
        batch=4,
        page_size=61,
        invalid_rows=True,
    ),
    SparseMlaCase("v4_page69", "v4", 1, 512, 64, 64, batch=4, page_size=69),
    SparseMlaCase(
        "v4_features",
        "v4",
        3,
        650,
        128,
        128,
        batch=4,
        page_size=61,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
        extra_kv_len=512,
        extra_topk=64,
    ),
    SparseMlaCase(
        "v32_large",
        "v32",
        2,
        32768,
        128,
        2048,
        batch=74,
        full=True,
    ),
    SparseMlaCase(
        "v4_large",
        "v4",
        3,
        2046,
        128,
        2048,
        batch=74,
        full=True,
    ),
)


DECODE_CASES += tuple(
    SparseMlaCase(
        f"{variant}_h{heads}_{'mixed' if extra else 'main'}",
        "v41",
        3,
        1952,
        heads,
        2112,
        batch=4,
        page_size=61,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
        extra_kv_len=512 if extra else 0,
        extra_topk=128 if extra else 0,
        extra_variant=variant,
        extra_page_size=69,
    )
    for heads in (64, 128)
    for variant, extra in (("v41", False), ("v41", True), ("v41_fp4", True))
)


GLM_DECODE_CASES = tuple(
    SparseMlaCase(
        f"glm53_h{heads}_k{topk}",
        "glm53",
        3,
        2112,
        heads,
        topk,
        batch=4,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
    )
    for heads in (8, 16, 32, 64)
    for topk in (64, 2112, 2176)
)


FP8_DECODE_CASES = (
    Fp8DecodeCase("q3_page61_4d", 2, 3, 183, 256, 61, 4),
    Fp8DecodeCase("mixed_invalid_page53_3d", 4, 2, 159, 64, 53, 3, True),
    Fp8DecodeCase("topk2112", 2, 1, 2176, 2112, 64, 4, full=True),
)


def _skip_large_case(case, pytestconfig: pytest.Config) -> None:
    if case.full and not pytestconfig.getoption("sparse_mla_full"):
        pytest.skip("enable with --sparse-mla-full")


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", PREFILL_CASES, ids=lambda case: case.name)
def test_sparse_mla_prefill(case: SparseMlaCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_prefill_data(case)

    out, max_logits, lse = mate.flashmla.flash_mla_sparse_fwd(
        q=data.q,
        kv=data.kv,
        indices=data.indices,
        sm_scale=case.head_dim**-0.5,
        d_v=512,
        attn_sink=data.attn_sink,
        topk_length=data.topk_length,
    )
    raise_complete_if_dry_run()

    assert_sparse_mla_close(
        out,
        data.ref_out,
        lse,
        data.ref_lse,
        max_logits=max_logits,
        ref_max_logits=data.ref_max_logits,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", ROPE_CASES, ids=lambda case: case.name)
def test_mla_rope_quantize_fp8(case: RopeCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_rope_data(case)
    q_merged = torch.empty(
        case.nnz, case.heads, 576, dtype=torch.float8_e4m3fn, device="musa"
    )
    k_merged = torch.empty(case.nnz, 576, dtype=torch.float8_e4m3fn, device="musa")

    outputs = mla_rope_quantize_fp8(
        data.q_rope,
        data.k_rope,
        data.q_nope,
        data.k_nope,
        data.cos_sin_cache,
        data.pos_ids,
        is_neox=case.is_neox,
        quant_scale_q=0.625,
        quant_scale_kv=1.75,
        q_rope_out=q_merged[..., 512:],
        k_rope_out=k_merged[..., 512:],
        q_nope_out=q_merged[..., :512],
        k_nope_out=k_merged[..., :512],
    )
    raise_complete_if_dry_run()

    for actual, expected in zip(outputs, data.ref_outputs):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.01
        )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", DECODE_CASES, ids=lambda case: case.name)
def test_sparse_mla_decode(case: SparseMlaCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_decode_data(case)
    if case.variant == "v41":
        # Independent page strides, including eight-byte but not 16-byte
        # aligned pages. The packed page footer is part of the copied blob.
        for name, gap in (("k_cache", 24), ("extra_k_cache", 40)):
            cache = getattr(data, name)
            if cache is not None:
                strides = (cache.shape[1] * cache.shape[-1] + gap, *cache.stride()[1:])
                padded = torch.empty_strided(
                    cache.shape, strides, dtype=cache.dtype, device=cache.device
                )
                padded.copy_(cache)
                setattr(data, name, padded)
    metadata, num_splits = mate.flashmla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=case.q_len * case.heads,
        num_heads_k=1,
        num_heads_q=case.heads,
        is_fp8_kvcache=True,
        topk=case.topk,
        extra_topk=case.extra_topk or None,
        q=data.q,
        bs=case.batch,
        topk_length=data.topk_length,
        extra_topk_length=data.extra_topk_length,
    )

    decode = mate.flashmla.flash_mla_with_kvcache
    if case.variant == "v41":
        from wrappers.FlashMLA import flash_mla

        metadata, num_splits = flash_mla.get_mla_metadata()
        decode = flash_mla.flash_mla_with_kvcache
    out, lse = decode(
        q=data.q,
        k_cache=data.k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=metadata,
        num_splits=num_splits,
        softmax_scale=case.head_dim**-0.5,
        is_fp8_kvcache=True,
        indices=data.indices,
        attn_sink=data.attn_sink,
        extra_k_cache=data.extra_k_cache,
        extra_indices_in_kvcache=data.extra_indices,
        topk_length=data.topk_length,
        extra_topk_length=data.extra_topk_length,
    )
    raise_complete_if_dry_run()

    assert_sparse_mla_close(out, data.ref_out, lse, data.ref_lse)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", FP8_DECODE_CASES, ids=lambda case: case.name)
def test_sparse_mla_fp8_decode(case: Fp8DecodeCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_fp8_decode_data(case)
    metadata = get_batch_decode_metadata_mla(data.query, data.seq_lens, case.topk)
    lse_buffer = torch.empty_like(data.ref_lse)

    out, lse = sparse_mla_fp8_decode(
        query=data.query,
        kv_cache=data.kv_cache,
        workspace_buffer=object(),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=data.indices,
        seq_lens=data.seq_lens,
        max_seq_len=case.kv_len,
        sparse_mla_top_k=case.topk,
        bmm1_scale=0.0625,
        lse=lse_buffer,
        return_lse=True,
        metadata=metadata,
    )
    raise_complete_if_dry_run()

    assert_sparse_mla_close(out, data.ref_out, lse, data.ref_lse)


@supported_musa_compute_capability([31])
def test_sparse_mla_fp8_decode_stress(pytestconfig: pytest.Config):
    iterations = pytestconfig.getoption("sparse_mla_stress_iters")
    if iterations == 0:
        pytest.skip("enable with --sparse-mla-stress-iters=N")
    if iterations < 10000:
        raise pytest.UsageError("--sparse-mla-stress-iters must be 0 or >= 10000")

    data = make_fp8_decode_stress_data()
    out_buffer = torch.empty_like(data.ref_out)
    lse_buffer = torch.empty_like(data.ref_lse)
    metadata = get_batch_decode_metadata_mla(
        data.query, data.seq_lens, data.indices.shape[-1]
    )
    decode_kwargs = {
        "query": data.query,
        "kv_cache": data.kv_cache,
        "workspace_buffer": object(),
        "qk_nope_head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "block_tables": data.indices,
        "seq_lens": data.seq_lens,
        "max_seq_len": data.indices.shape[-1],
        "sparse_mla_top_k": data.indices.shape[-1],
        "out": out_buffer,
        "bmm1_scale": 0.0625,
        "lse": lse_buffer,
        "return_lse": True,
        "metadata": metadata,
    }

    baseline_out, baseline_lse = sparse_mla_fp8_decode(**decode_kwargs)
    raise_complete_if_dry_run()
    torch.musa.synchronize()
    assert_sparse_mla_close(baseline_out, data.ref_out, baseline_lse, data.ref_lse)
    baseline_out = baseline_out.clone()
    baseline_lse = baseline_lse.clone()
    progress_interval = max(1000, iterations // 10)

    for iteration in range(1, iterations + 1):
        try:
            current_out, current_lse = sparse_mla_fp8_decode(**decode_kwargs)
            torch.musa.synchronize()
            if not torch.equal(current_out, baseline_out) or not torch.equal(
                current_lse, baseline_lse
            ):
                raise AssertionError("output or LSE changed")
        except Exception as exc:
            raise RuntimeError(
                "FP8 sparse decode persistent-row stress failed at "
                f"iteration {iteration}/{iterations}"
            ) from exc
        if iteration % progress_interval == 0:
            print(
                f"sparse MLA stress: {iteration}/{iterations} iterations",
                flush=True,
            )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", GLM_DECODE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("row_bytes", [528, 656])
def test_sparse_mla_glm_nope(case, row_bytes):
    from mate.sparse_mla.glm_nope import sparse_mla_nope_decode
    from mate.testing.sparse_mla import (
        dequantize_sparse_mla_cache,
        ref_sparse_mla_attention,
    )

    data = make_decode_data(case)
    cache = data.k_cache
    if row_bytes == 656:
        padded = torch.full(
            (*cache.shape[:-1], 656), 255, dtype=torch.uint8, device=cache.device
        )
        padded[..., :528] = cache
        cache = padded
    # KPool: a valid three-token tail after holes, with a non-padded prefix end.
    if case.topk > 2048:
        data.indices[..., 2048:2051] = torch.tensor(
            [41, 53, 67], dtype=torch.int32, device=cache.device
        )
        data.indices[..., 2051:] = -1
        data.topk_length.fill_(2051)
        data.indices[0].fill_(-1)
    scale = 0.0625
    out, lse = sparse_mla_nope_decode(
        data.q,
        cache,
        data.indices,
        softmax_scale=scale,
        topk_length=data.topk_length,
        attn_sink=data.attn_sink,
    )
    raise_complete_if_dry_run()
    kv = dequantize_sparse_mla_cache(cache, "glm53").reshape(-1, 1, 512)
    ref_out, _, ref_lse = ref_sparse_mla_attention(
        data.q.flatten(0, 1),
        kv,
        data.indices.flatten(0, 1),
        sm_scale=scale,
        topk_length=data.topk_length.repeat_interleave(case.q_len),
        attn_sink=data.attn_sink,
    )
    assert_sparse_mla_close(
        out,
        ref_out.reshape_as(out),
        lse,
        ref_lse.reshape(case.batch, case.q_len, case.heads).transpose(1, 2),
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("topk", [2112, 2176])
def test_sparse_mla_prefill_independent_kpool(heads, topk):
    from mate.testing.sparse_mla import ref_sparse_mla_attention

    case = SparseMlaCase("glm_kpool", "glm53", 8, 8, heads, topk, topk_length=True)
    data = make_prefill_data(case)
    data.indices.fill_(-1)
    data.indices[:, 0, 0] = torch.arange(8, dtype=torch.int32, device=data.q.device)
    data.indices[:, 0, 2050] = (
        torch.arange(8, dtype=torch.int32, device=data.q.device) + 3
    ) % 8
    data.indices[0].fill_(-1)
    data.topk_length.fill_(2051)
    out, max_logits, lse = mate.flashmla.flash_mla_sparse_fwd(
        data.q,
        data.kv,
        data.indices,
        0.0625,
        topk_length=data.topk_length,
    )
    raise_complete_if_dry_run()
    ref_out, ref_max, ref_lse = ref_sparse_mla_attention(
        data.q,
        data.kv,
        data.indices,
        sm_scale=0.0625,
        topk_length=data.topk_length,
    )
    assert_sparse_mla_close(
        out, ref_out, lse, ref_lse, max_logits=max_logits, ref_max_logits=ref_max
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("row_bytes", [528, 656])
def test_glm_cache_writer_and_native_runner(row_bytes):
    from mate.sparse_mla.tilelang.glm_nope_cache import write_glm_nope_cache
    from mate.testing.sparse_mla import (
        quantize_sparse_mla_cache,
        dequantize_sparse_mla_cache,
        ref_sparse_mla_attention,
    )
    from wrappers.FlashInfer.flashinfer.mla import SparseMLASm120Wrapper

    latent = torch.randn((128, 512), dtype=torch.bfloat16, device="musa") * 0.1
    cache = torch.full((2, 64, 1, row_bytes), 255, dtype=torch.uint8, device="musa")
    slots = torch.arange(128, dtype=torch.int64, device="musa").flip(0)
    write_glm_nope_cache(latent, cache, slots)
    q = torch.randn((3, 8, 512), dtype=torch.bfloat16, device="musa") * 0.1
    indices = torch.arange(64, dtype=torch.int32, device="musa").repeat(3, 1)
    indices[:, 7] = -1
    indices[0].fill_(-1)
    out = torch.empty_like(q)
    runner = SparseMLASm120Wrapper(
        3, 8, kv_scale_format="arbitrary_fp32", device="musa"
    )
    lse = runner.run(q, cache.squeeze(2), indices, out, 0.0625, return_lse=True)
    raise_complete_if_dry_run()
    expected = quantize_sparse_mla_cache(latent.flip(0).reshape(2, 64, 1, 512), "glm53")
    torch.testing.assert_close(cache[..., :528], expected, atol=0, rtol=0)
    if row_bytes == 656:
        assert torch.all(cache[..., 528:] == 255)
    kv = dequantize_sparse_mla_cache(cache, "glm53").reshape(-1, 1, 512)
    ref_out, _, ref_lse = ref_sparse_mla_attention(
        q, kv, indices[:, None], sm_scale=0.0625
    )
    assert_sparse_mla_close(out, ref_out, lse, ref_lse)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("tokens,topk", [(0, 64), (3, 0), (3, 1), (3, 65)])
def test_glm_native_runner_candidate_boundaries(tokens, topk):
    from mate.testing.sparse_mla import (
        dequantize_sparse_mla_cache,
        ref_sparse_mla_attention,
    )
    from wrappers.FlashInfer.flashinfer.mla import SparseMLASm120Wrapper

    data = make_decode_data(
        SparseMlaCase("native_boundary", "glm53", 1, 128, 8, 128, batch=3)
    )
    q = data.q[:tokens]
    indices = data.indices[:tokens, 0, 0, :topk]
    out = torch.empty_like(q)
    supplied_lse = torch.empty((3, 16), dtype=torch.float32, device=q.device)
    runner = SparseMLASm120Wrapper(
        3, 8, kv_scale_format="arbitrary_fp32", device=q.device
    )
    lse = runner.run(
        q,
        data.k_cache.squeeze(2),
        indices,
        out,
        0.0625,
        out_lse=supplied_lse,
        return_lse=True,
    )
    raise_complete_if_dry_run()
    assert lse.shape == (tokens, 8)
    if tokens == 0:
        return
    assert lse.data_ptr() == supplied_lse.data_ptr()
    if topk == 0:
        assert torch.all(out == 0) and torch.all(torch.isposinf(lse))
        return
    kv = dequantize_sparse_mla_cache(data.k_cache, "glm53").reshape(-1, 1, 512)
    expected, _, expected_lse = ref_sparse_mla_attention(
        q[:, 0], kv, indices[:, None], sm_scale=0.0625
    )
    assert_sparse_mla_close(out[:, 0], expected, lse, expected_lse)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize(
    "variant", ["prefill", "v4", "v4_extra", "v41", "v41_extra", "v41_fp4"]
)
@pytest.mark.parametrize(
    "tokens,heads,groups,topk,sink",
    [
        (3, 64, 8, 128, False),
        (1, 128, 16, 2112, True),
        (65, 64, 8, 128, True),
        (3, 64, 64, 128, False),
    ],
)
def test_sparse_mla_fused_module(norm, variant, tokens, heads, groups, topk, sink):
    from mate.sparse_mla import fused_norm_rope_attn_rope_cast as fused
    from mate.testing.sparse_mla import (
        dequantize_sparse_mla_cache,
        ref_sparse_mla_attention,
    )

    cache_variant = "v4" if variant in ("v4", "v4_extra") else "v41"
    case = SparseMlaCase(
        "fused",
        cache_variant,
        1,
        max(192, topk),
        heads,
        topk,
        batch=tokens,
        attn_sink=sink,
        topk_length=groups != 64,
        extra_kv_len=128 if variant in ("v4_extra", "v41_extra", "v41_fp4") else 0,
        extra_topk=64 if variant in ("v4_extra", "v41_extra", "v41_fp4") else 0,
        extra_variant="v41_fp4" if variant == "v41_fp4" else cache_variant,
    )
    data = make_decode_data(case)
    canonical = data.q.squeeze(1)
    q = (
        canonical.reshape(tokens, heads, 32, 16)
        .transpose(1, 2)
        .contiguous()
        .reshape(tokens, heads, 512)
    )
    positions = (torch.arange(tokens, dtype=torch.int32, device="musa") * 3 + 2) % 16
    angles = torch.randn((16, 32), device="musa")
    table = torch.cat((angles.cos(), angles.sin()), dim=1)
    args = (norm, 1e-6, positions, False, 64, table, groups, 32, True, True, True)
    kv = dequantize_sparse_mla_cache(data.k_cache, cache_variant).reshape(-1, 1, 512)
    if variant == "prefill":
        quant, sf, maximum, lse = fused.prefill(
            *args,
            q,
            kv,
            data.indices[:, 0],
            0.0625,
            topk_length=data.topk_length,
            attn_sink=data.attn_sink,
        )
    else:
        quant, sf, lse = fused.decode(
            *args,
            q,
            data.k_cache,
            data.indices[:, 0, 0],
            0.0625,
            topk_length=data.topk_length,
            attn_sink=data.attn_sink,
            extra_k_cache=data.extra_k_cache,
            extra_indices_in_kvcache=None
            if data.extra_indices is None
            else data.extra_indices[:, 0, 0],
            extra_topk_length=data.extra_topk_length,
        )
    raise_complete_if_dry_run()
    # Independent stepwise reference in canonical head order.
    x = canonical.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) if norm else 1.0
    cosine, sine = (
        table[positions.long(), :32, None].transpose(1, 2),
        table[positions.long(), 32:, None].transpose(1, 2),
    )
    even, odd = x[..., 448::2].clone(), x[..., 449::2].clone()
    x[..., 448::2] = even * cosine - odd * sine
    x[..., 449::2] = even * sine + odd * cosine
    x = x.to(torch.bfloat16)
    extra_kv = (
        None
        if data.extra_k_cache is None
        else dequantize_sparse_mla_cache(
            data.extra_k_cache, case.extra_variant
        ).reshape(-1, 1, 512)
    )
    reference, ref_maximum, ref_lse = ref_sparse_mla_attention(
        x,
        kv,
        data.indices[:, 0],
        sm_scale=0.0625,
        topk_length=data.topk_length,
        extra_kv=extra_kv,
        extra_indices=None if data.extra_indices is None else data.extra_indices[:, 0],
        extra_topk_length=data.extra_topk_length,
        q_scale=inv if norm else None,
        attn_sink=data.attn_sink,
        out_dtype=torch.float32,
    )
    reference = reference.float()
    even, odd = reference[..., 448::2].clone(), reference[..., 449::2].clone()
    reference[..., 448::2] = even * cosine + odd * sine
    reference[..., 449::2] = odd * cosine - even * sine
    reference = (
        reference.reshape(tokens, groups, heads // groups, 16, 32)
        .transpose(2, 3)
        .reshape(tokens, groups, -1)
    )
    shifts = torch.arange(4, dtype=torch.int32, device="musa") * 8
    exponents = ((sf.unsqueeze(-1) >> shifts) & 255).reshape(tokens, groups, -1)
    scales = (exponents << 23).view(torch.float32)
    dequantized = (
        quant.float().reshape(tokens, groups, -1, 32) * scales[..., None]
    ).reshape_as(reference)
    torch.testing.assert_close(dequantized, reference, rtol=0.15, atol=0.004)
    torch.testing.assert_close(lse, ref_lse, rtol=0.002, atol=0.02)
    if variant == "prefill":
        torch.testing.assert_close(maximum, ref_maximum, rtol=0.002, atol=0.02)
    assert sf.stride(0) == 1 and sf.stride(2) % 4 == 0

    empty = torch.isinf(ref_lse).all(dim=-1)
    if empty.any():
        assert torch.count_nonzero(quant.float()[empty]).item() == 0
        assert torch.all(exponents[empty] == 105).item()  # ceil_pow2(1e-4 / 448)


@supported_musa_compute_capability([31])
def test_flashmla_weight_permutations():
    from mate.sparse_mla.fused_norm_rope_attn_rope_cast import (
        permute_q_b_proj,
        permute_wv_proj,
    )

    weight = torch.randn((8 * 512, 128), device="musa").to(torch.float8_e4m3fn)
    sf = torch.arange(8 * 512, dtype=torch.int32, device="musa").reshape(-1, 1)
    permuted, psf = permute_q_b_proj((weight, sf), 8, 512)
    wv = weight.reshape(2, 128, 4 * 512)
    wsf = torch.full(
        (2, 16, 128), 0x7F7E7D7C, dtype=torch.int32, device="musa"
    ).transpose(1, 2)
    p_wv, p_wsf = permute_wv_proj((wv, wsf), 4, 512)
    raise_complete_if_dry_run()
    torch.testing.assert_close(
        permuted.view(torch.uint8)
        .reshape(32, 8, 16, 128)
        .transpose(0, 1)
        .reshape_as(weight),
        weight.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        psf.reshape(32, 8, 16, 1).transpose(0, 1).reshape_as(sf), sf, rtol=0, atol=0
    )
    torch.testing.assert_close(
        p_wv.view(torch.uint8)
        .reshape(2, 128, 16, 4, 32)
        .transpose(2, 3)
        .reshape_as(wv),
        wv.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    expected = torch.tensor(
        [0x7C7C7C7C, 0x7D7D7D7D, 0x7E7E7E7E, 0x7F7F7F7F] * 4,
        dtype=torch.int32,
        device="musa",
    )
    torch.testing.assert_close(p_wsf, expected.expand_as(p_wsf), rtol=0, atol=0)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("variant", ["glm53", "v41", "v41_fp4"])
def test_sparse_mla_packed_poisoned_mask(variant):
    from mate.sparse_mla.glm_nope import sparse_mla_nope_decode
    from mate.testing.sparse_mla import (
        quantize_sparse_mla_cache,
        dequantize_sparse_mla_cache,
        ref_sparse_mla_attention,
    )

    case = SparseMlaCase(
        "poison", "glm53" if variant == "glm53" else "v41", 1, 128, 64, 128, batch=3
    )
    data = make_decode_data(case)
    data.indices[data.indices == 0] = -1
    data.indices[0].fill_(-1)
    data.indices[1, 0, 0, -1] = 0  # Valid index beyond the prefix must also be masked.
    lengths = torch.tensor([128, 127, 128], dtype=torch.int32, device="musa")
    cache = data.k_cache
    # Poison only unused row zero, preserving footer addressing in the paged format.
    if variant == "glm53":
        cache[0, 0].fill_(255)
    else:
        flat = cache.view(torch.uint8).view(cache.shape[0], -1)
        flat[0, :512] = 255
        flat[0, 64 * 512 : 64 * 512 + 16] = 255
    extra = extra_indices = extra_kv = None
    if variant == "v41_fp4":
        latent = torch.randn((2, 69, 1, 512), device="musa", dtype=torch.bfloat16) * 0.1
        extra = quantize_sparse_mla_cache(latent, "v41_fp4")
        flat = extra.view(torch.uint8).view(2, -1)
        flat[0, :256] = 255
        flat[0, 69 * 256 : 69 * 256 + 32] = 255
        extra_indices = data.indices.clone()
        extra_kv = dequantize_sparse_mla_cache(extra, "v41_fp4").reshape(-1, 1, 512)
    if variant == "glm53":
        out, lse = sparse_mla_nope_decode(
            data.q, cache, data.indices, softmax_scale=0.0625, topk_length=lengths
        )
    else:
        out, lse = mate.flashmla.flash_mla_with_kvcache(
            data.q,
            cache,
            None,
            None,
            512,
            None,
            None,
            softmax_scale=0.0625,
            is_fp8_kvcache=True,
            indices=data.indices,
            topk_length=lengths,
            extra_k_cache=extra,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=lengths if extra is not None else None,
        )
    raise_complete_if_dry_run()
    kv = dequantize_sparse_mla_cache(cache, case.variant).reshape(-1, 1, 512)
    ref_out, _, ref_lse = ref_sparse_mla_attention(
        data.q[:, 0],
        kv,
        data.indices[:, 0],
        sm_scale=0.0625,
        topk_length=lengths,
        extra_kv=extra_kv,
        extra_indices=None if extra_indices is None else extra_indices[:, 0],
        extra_topk_length=lengths if extra is not None else None,
    )
    assert_sparse_mla_close(out[:, 0], ref_out, lse[..., 0], ref_lse)


@supported_musa_compute_capability([31])
def test_v41_known_packed_values():
    """Anchor both page-footer layouts to explicit bytes, independently of quantizers."""
    import math

    q = torch.zeros((1, 1, 64, 512), dtype=torch.bfloat16, device="musa")
    main = torch.full((1, 1, 1, 528), 0x38, dtype=torch.uint8, device="musa")
    main_exponents = torch.tensor(
        [126, 127, 128, 129] * 4, dtype=torch.uint8, device="musa"
    )
    main.view(-1)[512:] = main_exponents
    extra = torch.zeros((1, 2, 1, 288), dtype=torch.uint8, device="musa")
    # Token 1: low/even, high/odd E2M1 codes cover all sixteen representations.
    payload = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 32,
        dtype=torch.uint8,
        device="musa",
    )
    extra.view(-1)[256:512] = payload
    extra_scales = torch.tensor([0.5, 1.0, 2.0, 4.0] * 8, device="musa")
    extra.view(-1)[544:] = extra_scales.to(torch.float8_e4m3fn).view(torch.uint8)
    indices = torch.full((1, 1, 64), -1, dtype=torch.int32, device="musa")
    extra_indices = indices.clone()
    indices[..., 0], extra_indices[..., 0] = 0, 1
    out, lse = mate.flashmla.flash_mla_with_kvcache(
        q,
        main,
        None,
        None,
        512,
        None,
        None,
        softmax_scale=0.0625,
        is_fp8_kvcache=True,
        indices=indices,
        extra_k_cache=extra,
        extra_indices_in_kvcache=extra_indices,
    )
    raise_complete_if_dry_run()
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="musa",
    )
    expected_main = torch.tensor(
        [0.5, 1.0, 2.0, 4.0] * 4, device="musa"
    ).repeat_interleave(32)
    expected_extra = (extra_scales[:, None] * values).reshape(-1)
    expected = ((expected_main + expected_extra) / 2).to(torch.bfloat16)
    torch.testing.assert_close(out, expected.expand_as(out), rtol=0, atol=0)
    torch.testing.assert_close(
        lse, torch.full_like(lse, math.log(2)), rtol=0, atol=1e-5
    )


@supported_musa_compute_capability([31])
def test_sparse_mla_fused_output_quantization_boundaries():
    """Keep the FP32 attention result through cast, including FP8 rounding ties."""
    from mate.sparse_mla import fused_norm_rope_attn_rope_cast as fused
    from mate.testing.sparse_mla import fused_mla_finish_o_reference

    amplitudes = torch.tensor(
        [0.0, 1e-7, 448 * 2.0**-10, 448 * 2.0**-10 * (1 + 2.0**-7)] * 3
        + [448.0, 448.0, 448.0, 448.0],
        device="musa",
    )
    kv = torch.zeros((2, 1, 16, 32), dtype=torch.bfloat16, device="musa")
    kv[:, 0, :, 0] = amplitudes
    # Averaging these BF16 values crosses an FP8 midpoint. Rounding the
    # attention result to BF16 first would pick a different FP8 value.
    kv[0, 0, :, 1] = amplitudes / 448 * 0.265625
    kv[1, 0, :, 1] = amplitudes / 448 * 0.267578125
    kv = kv.reshape(2, 1, 512)
    q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="musa")
    indices = torch.tensor([[[0, 1]]], dtype=torch.int32, device="musa")
    positions = torch.zeros((1,), dtype=torch.int32, device="musa")
    table = torch.cat(
        (torch.ones((1, 32), device="musa"), torch.zeros((1, 32), device="musa")),
        dim=-1,
    )
    quant, sf, _, _ = fused.prefill(
        False,
        1e-6,
        positions,
        False,
        64,
        table,
        8,
        32,
        True,
        True,
        True,
        q,
        kv,
        indices,
        0.0625,
    )
    raise_complete_if_dry_run()
    expected = kv.float().mean(dim=0).unsqueeze(0).expand(1, 64, 512).contiguous()
    expected_quant, expected_sf = fused_mla_finish_o_reference(
        expected, positions, table, 8
    )
    torch.testing.assert_close(
        quant.view(torch.uint8), expected_quant.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)
