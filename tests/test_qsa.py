"""Functional tests for QSA MQA scores and selected-token GQA attention."""

import math

import pytest
import torch

from mate.qsa_interface import (
    qsa_indexer_mqa_score,
    qsa_mqa_decode,
    qsa_mqa_prefill,
    qsa_sparse_attention,
)
from mate.qsa_kernels.core import (
    merge_qsa_core_paged_splitk,
    run_qsa_core,
    run_qsa_core_paged,
    run_qsa_core_paged_splitk,
)
from mate.qsa_kernels.mqa import run_qsa_mqa_score_packed
from mate.testing.qsa import (
    expand_qsa_block_indices,
    mqa_decode_reference,
    mqa_dense_score_reference,
    mqa_prefill_reference,
    paged_kv_fixture,
    sparse_attention_reference,
)
from mate.testing.utils import sliced_tensor
from mate.qsa_kernels import core as qsa_core_impl, mqa as qsa_mqa_impl


@pytest.fixture(autouse=True)
def _seed_inputs():
    torch.manual_seed(20261021)


@pytest.fixture
def native_only(monkeypatch):
    def reject_reference(*args, **kwargs):
        pytest.fail("default QSA execution entered the PyTorch reference")

    for name in ("_core_reference", "splitkv_merge_reference"):
        monkeypatch.setattr(qsa_core_impl, name, reject_reference)
    for name in (
        "mqa_score_reference",
        "mqa_prefill_reference",
        "mqa_decode_reference",
    ):
        monkeypatch.setattr(qsa_mqa_impl, name, reject_reference)


def _attention_reference(q, k, v, indices, options):
    flat = q.reshape(-1, *q.shape[-2:])
    rows = flat.shape[0]
    if rows == 0:
        return torch.empty_like(q), torch.empty(q.shape[:-1], device=q.device)
    positions = options["query_positions"].reshape(rows)
    output, lse = sparse_attention_reference(
        flat,
        k,
        v,
        indices.reshape(rows, indices.shape[-1]),
        batch_ids=options.get(
            "token_to_req", torch.zeros(rows, device=q.device, dtype=torch.int32)
        ),
        query_positions=positions,
        row_valid=positions >= 0,
        seq_lens_k=options["seq_lens_k"],
        cu_seqlens_k=options.get("cu_seqlens_k"),
        block_table=options.get("block_table"),
        page_size=options.get("page_size"),
        scale=options.get("scale", 256**-0.5),
        gate=None
        if options.get("gate") is None
        else options["gate"].reshape(rows, q.shape[-2]),
    )
    return output.reshape_as(q), lse.reshape(q.shape[:-1])


def _attention_inputs(
    device,
    dtype,
    *,
    rows=4,
    heads=24,
    kv_heads=2,
    width=65,
    length=67,
    layout="paged",
    page=16,
):
    """Small public-ABI inputs; each layout describes the same logical cache."""
    batch = 2 if layout == "varlen" else 1
    q = torch.randn(rows, heads, 256, device=device, dtype=dtype) / 4
    k = torch.randn(batch, length, kv_heads, 256, device=device, dtype=dtype) / 4
    v = torch.randn_like(k) / 4
    indices = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
    indices.remainder_(length + 3)
    if width > 3:
        indices[:, 1] = indices[:, 0]
        indices[:, -3:] = torch.tensor([-(2**31), 2**31 - 1, -1], device=device)
    if rows > 1:
        indices[0] = -1
    positions = (torch.arange(rows, device=device, dtype=torch.int32) * 13) % length
    positions[-1] = length - 1
    options = dict(
        query_positions=positions,
        seq_lens_k=torch.full((batch,), length, device=device, dtype=torch.int32),
    )
    if layout == "contiguous":
        q, indices = q[None], indices[None]
        options["query_positions"] = positions[None]
    else:
        options.update(
            token_to_req=torch.arange(rows, device=device, dtype=torch.int32) % batch,
            num_requests=batch,
        )
        if layout == "paged":
            k, v, table = paged_kv_fixture(k, v, page)
            options.update(block_table=table, page_size=page)
        else:
            k, v = k.flatten(0, 1), v.flatten(0, 1)
            options["cu_seqlens_k"] = (
                torch.arange(batch + 1, device=device, dtype=torch.int32) * length
            )
    return q, k, v, indices, options


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "rows,columns", [(1, 1), (3, 17), (17, 65), (65, 257), (129, 513)]
)
@pytest.mark.usefixtures("native_only")
def test_qsa_mqa_prefill_matches_reference(musa, dtype, rows, columns):
    torch.manual_seed(101)
    q = torch.randn(rows, 4, 128, device=musa, dtype=dtype) / 8
    k = torch.randn(columns, 1, 128, device=musa, dtype=dtype) / 8
    starts = torch.arange(rows, device=musa, dtype=torch.int32) % columns
    ends = (starts + 13).clamp_max(columns)
    ends[-1] = columns
    if rows > 1:
        ends[0] = starts[0]
    q, k, starts, ends = [sliced_tensor(t)[0] for t in (q, k, starts, ends)]
    actual = qsa_mqa_prefill(q, k, starts, ends, score_scale=8.0)
    expected = mqa_prefill_reference(q, k[:, 0], starts, ends, multiplier=0.125)
    assert torch.isfinite(actual[-1, -1])
    if rows > 1:
        assert torch.isneginf(actual[0]).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,columns", [(1, 17), (8, 73), (17, 257), (128, 513)])
@pytest.mark.usefixtures("native_only")
def test_qsa_mqa_score_matches_reference(musa, dtype, rows, columns):
    q = torch.randn(2, rows, 4, 128, device=musa, dtype=dtype) / 8
    k = torch.randn(2, columns, 128, device=musa, dtype=dtype) / 8
    context = 4 * columns
    positions = (4 * torch.arange(rows, device=musa, dtype=torch.int32) + 3).repeat(
        2, 1
    )
    positions[:, -1] = context - 1
    positions[0, 0] = -1
    lengths = torch.full((2,), rows, device=musa, dtype=torch.int32)
    if rows > 1:
        positions[1, 0] = context - 2
        lengths[1] = rows - 1
    q = sliced_tensor(q.flatten(0, 1))[0].view_as(q)
    k = sliced_tensor(k)[0]
    positions = sliced_tensor(positions.flatten())[0].view_as(positions)
    actual = qsa_indexer_mqa_score(q, k, query_positions=positions, seq_lens_q=lengths)
    visible = ((positions.long() + 1) // 4).clamp(0, columns)
    valid_rows = (positions >= 0) & (
        torch.arange(rows, device=musa)[None] < lengths[:, None]
    )
    expected = mqa_dense_score_reference(q, k, scale=128**-0.5)
    valid = (torch.arange(columns, device=musa) < visible[..., None]) & valid_rows[
        ..., None
    ]
    expected.masked_fill_(~valid, -torch.inf)
    assert valid.any()
    assert torch.isfinite(actual[0 if rows > 1 else 1, -1]).all()
    assert torch.isneginf(actual[0, 0]).all()
    if rows > 1:
        assert torch.isfinite(actual[1, 0, :-1]).all()
        assert torch.isneginf(actual[1, 0, -1])
        assert torch.isneginf(actual[1, -1]).all()
    assert torch.equal(torch.isfinite(actual), valid)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page", [8, 16, 32, 64])
@pytest.mark.usefixtures("native_only")
def test_qsa_mqa_decode_matches_reference(musa, dtype, page):
    q = torch.randn(3, 4, 128, device=musa, dtype=dtype) / 8
    cache = torch.randn(5, page, 1, 128, device=musa, dtype=dtype) / 8
    cache[0] = torch.nan
    table = torch.tensor(
        [[-1, -1, -1], [3, 1, -1], [4, 2, -1]], device=musa, dtype=torch.int32
    )
    lengths = torch.tensor([0, page + 3, 2 * page], device=musa, dtype=torch.int32)
    q, cache = [sliced_tensor(t, feature_step=True)[0] for t in (q, cache)]
    table, lengths = [sliced_tensor(t, 3)[0] for t in (table, lengths)]
    actual = qsa_mqa_decode(q, cache, table, lengths, 3 * page + 1)
    expected = mqa_decode_reference(
        q, cache, table, lengths, 3 * page + 1, multiplier=128**-0.5
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qsa_mqa_reference_interleaved_requests(musa, dtype):
    q = torch.zeros(4, 4, 128, device=musa, dtype=dtype)
    q[..., 0] = 1
    k = torch.zeros(2, 17, 128, device=musa, dtype=dtype)
    k[0, :, 0], k[1, :, 0] = 1, 2
    request = torch.tensor([1, 0, 1, 0], device=musa, dtype=torch.int32)
    positions = torch.tensor([3, 7, 67, -1], device=musa, dtype=torch.int32)
    actual = qsa_indexer_mqa_score(
        q,
        k,
        token_to_req=request,
        num_requests=2,
        query_positions=positions,
        backend="reference",
    )
    expected = (
        torch.tensor([8, 4, 8, 0], device=musa, dtype=torch.float32)[:, None]
        .expand(4, 17)
        .clone()
        * 128**-0.5
    )
    visible = torch.tensor([1, 2, 17, 0], device=musa)
    expected.masked_fill_(torch.arange(17, device=musa) >= visible[:, None], -torch.inf)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("entry", ["score", "decode"])
def test_qsa_mqa_missing_active_page(musa, entry):
    q = torch.zeros(1, 4, 128, device=musa, dtype=torch.bfloat16)
    cache = torch.zeros(1, 8, 1, 128, device=musa, dtype=q.dtype)
    table = torch.tensor([[-1]], device=musa, dtype=torch.int32)
    if entry == "score":
        with pytest.raises(ValueError, match="missing a (physical )?page"):
            qsa_indexer_mqa_score(
                q[None],
                cache,
                block_table=table,
                page_size=8,
                query_positions=torch.tensor([[3]], device=musa, dtype=torch.int32),
                backend="reference",
            )
    else:
        result = qsa_mqa_decode(
            q,
            cache,
            table,
            torch.ones(1, device=musa, dtype=torch.int32),
            8,
            backend="tilelang",
        )
        assert torch.isneginf(result).all()


@pytest.mark.parametrize(
    "rows,columns,prefix,context",
    [
        (17, 257, 128, None),
        (5, 251, 249, 1003),
        (2, 258, 256, 1029),
        (2, 258, 256, 1030),
        (2, 258, 256, 1031),
        (2, 257, 257, None),
    ],
    ids=["odd", "context_1003", "tail1", "tail2", "tail3", "prefix257"],
)
def test_qsa_mqa_prefix_tail_regression(musa, rows, columns, prefix, context):
    q = torch.randn(1, rows, 4, 128, device=musa, dtype=torch.bfloat16) / 8
    k = torch.randn(1, columns, 128, device=musa, dtype=q.dtype) / 8
    visible = (
        torch.full((1, rows), prefix, device=musa, dtype=torch.int32)
        if context is None
        else (
            torch.arange(
                context - rows + 1, context + 1, device=musa, dtype=torch.int32
            )
            // 4
        )[None]
    )
    actual = run_qsa_mqa_score_packed(
        q,
        k,
        visible,
        block_n=128,
        block_m=32,
        group_n=2,
        split_m=True,
        split_full_prefix=True,
        full_prefix_blocks=prefix,
    )
    expected = mqa_dense_score_reference(q, k, scale=128**-0.5)
    expected.masked_fill_(
        torch.arange(columns, device=musa) >= visible[..., None], -torch.inf
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout,rows,width,heads,kv_heads,page,splits",
    [
        ("contiguous", 1, 1, 24, 2, 16, 1),
        ("contiguous", 3, 32, 12, 1, 16, 1),
        ("contiguous", 7, 65, 6, 1, 16, 1),
        ("paged", 3, 63, 24, 2, 8, 1),
        ("paged", 4, 64, 3, 1, 16, 1),
        ("paged", 1, 65, 3, 1, 32, 2),
        ("paged", 3, 129, 24, 2, 64, 3),
        ("paged", 4, 2049, 24, 2, 128, None),
        ("paged", 8, 2051, 24, 2, 16, 1),
        ("paged", 8, 2055, 3, 1, 64, None),
        ("paged", 1, 4097, 3, 1, 64, 4),
        ("paged", 3, 65, 2, 1, 16, 1),
        ("paged", 3, 65, 1, 1, 16, 1),
        ("paged", 3, 65, 32, 2, 16, 1),
        ("paged", 3, 65, 48, 4, 16, 1),
    ],
)
@pytest.mark.usefixtures("native_only")
def test_qsa_attention_matches_reference(
    musa, dtype, layout, rows, width, heads, kv_heads, page, splits
):
    q, k, v, indices, options = _attention_inputs(
        musa,
        dtype,
        rows=rows,
        heads=heads,
        kv_heads=kv_heads,
        width=width,
        layout=layout,
        page=page,
    )
    options["scale"] = 0.1
    if rows > 1:
        options["gate"] = torch.randn(q.shape[:-1], device=musa)
    expected, _ = _attention_reference(q, k, v, indices, options)
    actual = qsa_sparse_attention(q, k, v, indices, num_splits=splits, **options)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["varlen", "paged"])
def test_qsa_attention_interleaved_requests(musa, dtype, layout):
    q = torch.zeros(4, 24, 256, device=musa, dtype=dtype)
    logical_k = torch.zeros(2, 5, 2, 256, device=musa, dtype=dtype)
    logical_v = (
        (
            torch.arange(1, 6, device=musa, dtype=dtype)[None, :, None, None]
            * torch.tensor([1, 10], device=musa, dtype=dtype)[:, None, None, None]
            * torch.tensor([1, 2], device=musa, dtype=dtype)[None, None, :, None]
        )
        .expand_as(logical_k)
        .clone()
    )
    indices = torch.tensor(
        [[0, 1, -1, 4], [0, -1, 1, 3], [4, 1, 1, 5], [-1, -1, -1, -1]],
        device=musa,
        dtype=torch.int32,
    )
    options = dict(
        token_to_req=torch.tensor([1, 0, 1, 0], device=musa, dtype=torch.int32),
        num_requests=2,
        query_positions=torch.tensor([1, 0, 4, 2], device=musa, dtype=torch.int32),
        seq_lens_k=torch.tensor([3, 5], device=musa, dtype=torch.int32),
    )
    if layout == "paged":
        k, v, table = paged_kv_fixture(logical_k, logical_v, 16)
        options.update(block_table=table, page_size=16)
    else:
        options["backend"] = "reference"
        k = torch.cat((logical_k[0, :3], logical_k[1]))
        v = torch.cat((logical_v[0, :3], logical_v[1]))
        options["cu_seqlens_k"] = torch.tensor(
            [0, 3, 8], device=musa, dtype=torch.int32
        )
    # QK=0: O is the mean of the selected V, and LSE=log(valid selection count).
    expected = (
        torch.tensor([[15, 30], [1, 2], [30, 60], [0, 0]], device=musa, dtype=dtype)
        .repeat_interleave(12, dim=1)[..., None]
        .expand_as(q)
    )
    expected_lse = torch.tensor(
        [math.log(2), 0, math.log(3), -torch.inf], device=musa, dtype=torch.float32
    )[:, None].expand(4, 24)
    actual, lse = qsa_sparse_attention(q, k, v, indices, return_lse=True, **options)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)
    order = torch.tensor([1, 3, 0, 2], device=musa)
    grouped_options = {
        name: value
        for name, value in options.items()
        if name not in ("token_to_req", "num_requests")
    }
    grouped_options.update(
        cu_seqlens_q=torch.tensor([0, 2, 4], device=musa, dtype=torch.int32),
        query_positions=options["query_positions"][order],
    )
    grouped = qsa_sparse_attention(q[order], k, v, indices[order], **grouped_options)
    torch.testing.assert_close(grouped, expected[order], rtol=0.02, atol=0.002)


@pytest.mark.parametrize("ws", [False, True])
@pytest.mark.parametrize(
    "length,dtype",
    [
        (length, torch.float16 if length == 2051 else torch.bfloat16)
        for length in (32, 33, 34, 35, 2048, 2049, 2050, 2051, 8192)
    ],
)
def test_qsa_attention_structured_blocks_and_tail(musa, ws, length, dtype):
    rows = 6
    q = torch.zeros(rows, 24, 256, device=musa, dtype=dtype)
    logical_k = torch.zeros(1, length, 2, 256, device=musa, dtype=q.dtype)
    logical_v = torch.ones_like(logical_k)
    if length != 32:
        q.normal_(std=0.125)
        logical_k.normal_(std=0.125)
        logical_v.normal_(std=0.25)
    k, v, table = paged_kv_fixture(logical_k, logical_v, 16)
    positions = torch.full((rows,), length - 1, device=musa, dtype=torch.int32)
    blocks = torch.full((rows, 512), -1, device=musa, dtype=torch.int32)
    count = min(length // 4, 512)
    blocks[:, :count] = torch.arange(count, device=musa, dtype=torch.int32).flip(0)
    indices, _ = expand_qsa_block_indices(blocks, positions, output_width=2051)
    blocks, indices = [sliced_tensor(t, 3)[0] for t in (blocks, indices)]
    options = dict(
        token_to_req=torch.zeros(rows, device=musa, dtype=torch.int32),
        query_positions=positions,
        seq_lens_k=torch.tensor([length], device=musa, dtype=torch.int32),
        block_table=table,
    )
    expected, _ = _attention_reference(q, k, v, indices, options)
    output = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        block_indices=blocks,
        num_splits=1,
        warp_specialized=ws,
        backend="tilelang",
        **options,
    )
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout,rows,width,heads,kv_heads,splits,ws",
    [
        ("contiguous", 3, 65, 6, 1, 1, False),
        ("paged", 3, 65, 24, 2, 1, False),
        ("paged", 8, 2051, 24, 2, 1, False),
        ("paged", 8, 2051, 3, 1, 1, False),
        ("paged", 1, 2051, 3, 1, 4, False),
        ("paged", 7, 2051, 24, 2, 1, True),
    ],
)
def test_qsa_attention_sliced_inputs_and_output(
    musa, dtype, layout, rows, width, heads, kv_heads, splits, ws
):
    q, k, v, indices, options = _attention_inputs(
        musa,
        dtype,
        layout=layout,
        rows=rows,
        width=width,
        heads=heads,
        kv_heads=kv_heads,
    )
    expected, _ = _attention_reference(q, k, v, indices, options)
    q = sliced_tensor(q, feature_step=True)[0]
    k, v, indices = [
        sliced_tensor(t, step)[0] for t, step in zip((k, v, indices), (2, 3, 3))
    ]
    options = {
        name: sliced_tensor(value, 3)[0] if isinstance(value, torch.Tensor) else value
        for name, value in options.items()
    }
    originals = [t.clone() for t in (q, k, v, indices)]
    out, storage = sliced_tensor(torch.full_like(q, 97), 3, feature_step=True)
    result = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        out=out,
        num_splits=splits,
        warp_specialized=ws,
        backend="tilelang",
        **options,
    )
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.002)
    for tensor, before in zip((q, k, v, indices), originals):
        torch.testing.assert_close(tensor, before, rtol=0, atol=0, equal_nan=True)
    out.fill_(97)
    assert (storage == 97).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qsa_attention_padding_and_causal_mask(musa, dtype):
    q = torch.zeros(2, 3, 6, 256, device=musa, dtype=dtype)
    k = torch.zeros(2, 5, 1, 256, device=musa, dtype=dtype)
    v = torch.ones_like(k)
    v[:, 2:] = torch.nan
    q[:, 2] = torch.nan
    indices = torch.tensor([0, 1, 2, 4, -1], device=musa, dtype=torch.int32).expand(
        2, 3, -1
    )
    output = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        seq_lens_q=torch.tensor([2, 1], device=musa, dtype=torch.int32),
        seq_lens_k=torch.tensor([2, 2], device=musa, dtype=torch.int32),
        backend="tilelang",
        num_splits=1,
    )
    expected = torch.zeros_like(q)
    expected[0, :2] = 1
    expected[1, :1] = 1
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "layout,backend",
    [("contiguous", "reference"), ("contiguous", "tilelang"), ("paged", "tilelang")],
)
def test_qsa_attention_disjoint_output_storage(musa, layout, backend):
    q, k, v, indices, options = _attention_inputs(
        musa, torch.bfloat16, layout=layout, rows=8, width=2051
    )
    expected, _ = _attention_reference(q, k, v, indices, options)
    arena = torch.empty(2 * q.numel(), device=musa, dtype=q.dtype)
    query, out = arena[: q.numel()].view_as(q), arena[q.numel() :].view_as(q)
    query.copy_(q)
    result = qsa_sparse_attention(
        query, k, v, indices, out=out, backend=backend, num_splits=1, **options
    )
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.002)


def test_qsa_attention_wide_cache_address(musa):
    page, pages = 16, 262145
    if torch.musa.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("requires 6 GiB free for two caches spanning 2 GiB each")
    q = torch.randn(8, 3, 256, device=musa, dtype=torch.bfloat16) / 4
    k = torch.empty(pages, page, 1, 256, device=musa, dtype=q.dtype)
    v = torch.empty_like(k)
    k[-1].normal_(std=0.25)
    v[-1].normal_(std=0.25)
    indices = torch.full((8, 2051), -1, device=musa, dtype=torch.int32)
    indices[:, :2] = torch.tensor([14, 15], device=musa, dtype=torch.int32)
    options = dict(
        token_to_req=torch.zeros(8, device=musa, dtype=torch.int32),
        query_positions=torch.full((8,), 15, device=musa, dtype=torch.int32),
        seq_lens_k=torch.tensor([16], device=musa, dtype=torch.int32),
        block_table=torch.tensor([[pages - 1]], device=musa, dtype=torch.int32),
    )
    expected, _ = _attention_reference(
        q,
        k[-1:],
        v[-1:],
        indices,
        dict(options, block_table=torch.zeros_like(options["block_table"])),
    )
    actual = qsa_sparse_attention(
        q, k, v, indices, num_splits=1, backend="tilelang", **options
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize(
    "rows,heads,kv_heads,width,layout,lse_mode,splits",
    [
        (0, 3, 1, 1, "dense", "returned", None),
        (1, 3, 1, 65, "sliced", "none", None),
        (1, 3, 1, 64, "sliced", "none", None),
        (1, 3, 1, 129, "sliced", "none", 1),
        (2, 24, 2, 63, "io_sliced", "none", 1),
        (4, 24, 2, 2055, "sliced", "provided", None),
        (8, 24, 2, 2051, "dense", "none", None),
        (8, 24, 2, 2051, "io_sliced", "none", None),
        (8, 24, 2, 2051, "io_sliced", "returned", None),
        (8, 3, 1, 1, "io_sliced", "none", 1),
        (8, 3, 1, 65, "io_sliced", "none", 1),
        (8, 3, 1, 4097, "io_sliced", "none", 1),
        (32, 3, 1, 2055, "io_sliced", "none", None),
        (32, 3, 1, 2055, "sliced", "provided", None),
        (129, 3, 1, 65, "sliced", "returned", None),
        (2, 3, 1, 71, "dense", "returned", 1),
        (2, 3, 1, 71, "dense", "returned", 3),
    ],
)
def test_qsa_attention_physical_indices(
    musa, rows, heads, kv_heads, width, layout, lse_mode, splits
):
    q = torch.randn(rows, heads, 256, device=musa, dtype=torch.bfloat16) / 4
    k = torch.randn(96, kv_heads, 256, device=musa, dtype=q.dtype) / 4
    v = torch.randn_like(k) / 4
    k[0] = v[0] = torch.nan
    indices = (
        1 + torch.arange(width, device=musa, dtype=torch.int32).repeat(rows, 1) % 95
    )
    if width > 3:
        indices[:, 1] = indices[:, 0]
        indices[:, 2] = -1
        indices[:, -3:] = torch.tensor([-(2**31), 2**31 - 1, -1], device=musa)
    if rows > 1:
        indices[0] = -1
    options = dict(
        query_positions=torch.full((rows,), 95, device=musa, dtype=torch.int32),
        seq_lens_k=torch.tensor([96], device=musa, dtype=torch.int32),
        scale=0.1,
    )
    expected, expected_lse = _attention_reference(q, k, v, indices, options)
    if layout != "dense":
        q, indices = (
            sliced_tensor(q, feature_step=True)[0],
            sliced_tensor(indices, 3)[0],
        )
        if layout == "sliced":
            k, v = sliced_tensor(k)[0], sliced_tensor(v, 3)[0]
    out, storage = sliced_tensor(torch.full_like(q, 97), 3, feature_step=True)
    lse, lse_storage = sliced_tensor(
        torch.full(q.shape[:-1], 97.0, device=musa), feature_step=True
    )
    result = qsa_sparse_attention(
        q,
        k,
        v,
        selected_kv_indices=indices,
        softmax_scale=0.1,
        out=out,
        lse=lse if lse_mode != "none" else None,
        return_lse=lse_mode == "returned",
        num_splits=splits,
    )
    if lse_mode == "returned":
        result, actual_lse = result
        assert actual_lse is lse
    assert result is out
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.002)
    if lse_mode != "none":
        torch.testing.assert_close(lse, expected_lse, rtol=0.002, atol=0.002)
    out.fill_(97)
    lse.fill_(97)
    assert (storage == 97).all() and (lse_storage == 97).all()


def test_qsa_attention_physical_allocated_outputs(musa):
    q = torch.randn(2, 3, 256, device=musa, dtype=torch.bfloat16) / 4
    k = torch.randn(96, 1, 256, device=musa, dtype=q.dtype) / 4
    v = torch.randn_like(k) / 4
    indices = torch.arange(71, device=musa, dtype=torch.int32).expand(2, -1)
    options = dict(
        query_positions=torch.full((2,), 95, device=musa, dtype=torch.int32),
        seq_lens_k=torch.tensor([96], device=musa, dtype=torch.int32),
    )
    actual, lse = qsa_sparse_attention(
        q, k, v, selected_kv_indices=indices, num_splits=3, return_lse=True
    )
    expected, expected_lse = _attention_reference(q, k, v, indices, options)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(lse, expected_lse, rtol=0.002, atol=0.002)


@pytest.mark.parametrize(
    "rows,width,write_lse",
    [(2, 2055, True), (8, 2051, False), (8, 2055, False), (1, 64, False)],
)
def test_qsa_attention_physical_public_graph(musa, rows, width, write_lse):
    q = torch.randn(rows, 3, 256, device=musa, dtype=torch.bfloat16) / 4
    k = torch.randn(96, 1, 256, device=musa, dtype=q.dtype) / 4
    v = torch.randn_like(k) / 4
    v[0] = torch.nan  # Invalid gathers use token zero and must clear it after waiting.
    indices = torch.full((rows, width), -1, device=musa, dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty(rows, 3, device=musa) if write_lse else None
    options = dict(
        query_positions=torch.full((rows,), 95, device=musa, dtype=torch.int32),
        seq_lens_k=torch.tensor([96], device=musa, dtype=torch.int32),
    )
    qsa_sparse_attention(q, k, v, selected_kv_indices=indices, out=out, lse=lse)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        captured = qsa_sparse_attention(
            q, k, v, selected_kv_indices=indices, out=out, lse=lse, return_lse=write_lse
        )
    assert (captured[0] is out and captured[1] is lse) if write_lse else captured is out
    for start in (None, 7, None):
        q.mul_(-0.75)
        indices.fill_(-1)
        if start is not None:
            indices[-1, :4] = torch.arange(
                start, start + 4, device=musa, dtype=torch.int32
            )
        out.fill_(torch.nan)
        if write_lse:
            lse.fill_(torch.nan)
        graph.replay()
        expected, expected_lse = _attention_reference(q, k, v, indices, options)
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.002)
        if write_lse:
            torch.testing.assert_close(lse, expected_lse, rtol=0.002, atol=0.002)


@pytest.mark.parametrize(
    "invalid",
    [
        "fp8",
        "compute",
        "scale",
        "k_scale",
        "v_scale",
        "out",
        "lse",
        "out_lse",
        "broadcast_out",
    ],
)
def test_qsa_attention_physical_rejects_invalid_inputs(musa, invalid):
    q = torch.zeros(1, 3, 256, device=musa, dtype=torch.bfloat16)
    k = torch.zeros(16, 1, 256, device=musa, dtype=q.dtype)
    indices = torch.zeros(1, 65, device=musa, dtype=torch.int32)
    options = {}
    if invalid == "fp8":
        k = k.to(torch.float8_e4m3fn)
    elif invalid == "compute":
        options["compute_dtype"] = torch.float8_e4m3fn
    elif invalid == "scale":
        options["softmax_scale"] = float("nan")
    elif invalid in ("k_scale", "v_scale"):
        options[invalid] = 2.0
    elif invalid == "out":
        options["out"] = q
    elif invalid == "lse":
        options["lse"] = q.view(torch.float32).view(-1)[:3].view(1, 3)
    elif invalid == "out_lse":
        options["out"] = torch.empty_like(q)
        options["lse"] = options["out"].view(torch.float32).view(-1)[:3].view(1, 3)
    else:
        options["out"] = torch.empty(1, 1, 256, device=musa, dtype=q.dtype).expand_as(q)
    with pytest.raises(ValueError):
        qsa_sparse_attention(q, k, k, selected_kv_indices=indices, **options)


@pytest.mark.parametrize(
    "option,message",
    [
        ("both_indices", "logical metadata"),
        ("block_table", "logical metadata"),
        ("both_scales", "mutually exclusive"),
        ("lse", "physical indices"),
        ("reference", "backend"),
        ("missing_indices", "provide token_indices"),
    ],
)
def test_qsa_attention_index_mode_contract(option, message):
    q, k = (
        torch.zeros(1, 3, 256, dtype=torch.bfloat16),
        torch.zeros(16, 1, 256, dtype=torch.bfloat16),
    )
    selected = torch.zeros(1, 65, dtype=torch.int32)
    options = dict(selected_kv_indices=selected)
    if option == "both_indices":
        options["token_indices"] = selected
    elif option == "block_table":
        options["block_table"] = selected
    elif option == "both_scales":
        options.update(scale=0.1, softmax_scale=0.1)
    elif option == "lse":
        options = dict(token_indices=selected, lse=torch.empty(1, 3))
    elif option == "reference":
        options["backend"] = "reference"
    else:
        options.clear()
    with pytest.raises(ValueError, match=message):
        qsa_sparse_attention(q, k, k, **options)


def test_qsa_attention_rejects_kv_feature_stride(musa):
    q, k, v, indices, options = _attention_inputs(musa, torch.bfloat16)
    k = sliced_tensor(k, feature_step=True)[0]
    out = torch.full_like(q, 97)
    with pytest.raises(ValueError, match="unit feature stride and 16-byte alignment"):
        qsa_sparse_attention(q, k, v, indices, out=out, **options)
    assert (out == 97).all()


@pytest.mark.parametrize(
    "name", ["query_positions", "token_to_req", "block_table", "token_indices"]
)
def test_qsa_attention_rejects_int64_metadata(musa, name):
    q, k, v, indices, options = _attention_inputs(musa, torch.bfloat16)
    if name == "token_indices":
        indices = indices.long()
    else:
        options[name] = options[name].long()
    with pytest.raises(TypeError, match="int32"):
        qsa_sparse_attention(q, k, v, indices, backend="tilelang", **options)


@pytest.mark.parametrize("layout", ["contiguous", "paged"])
def test_qsa_attention_int32_position_limit(musa, layout):
    q, k, v, indices, options = _attention_inputs(musa, torch.bfloat16, layout=layout)
    options["query_positions"].fill_(2**31 - 1)
    expected, _ = _attention_reference(q, k, v, indices, options)
    actual = qsa_sparse_attention(
        q, k, v, indices, backend="tilelang", num_splits=1, **options
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize(
    "invalid",
    ["splits", "alias", "broadcast_out"],
)
def test_qsa_attention_rejects_invalid_inputs(musa, invalid):
    q, k, v, indices, options = _attention_inputs(musa, torch.bfloat16, width=1)
    options["backend"] = "tilelang"
    if invalid == "splits":
        options["num_splits"] = 4
    elif invalid == "alias":
        options["out"] = q
    else:
        options["out"] = torch.empty(1, 1, 256, device=musa, dtype=q.dtype).expand_as(q)
    with pytest.raises(ValueError):
        qsa_sparse_attention(q, k, v, indices, **options)


def test_qsa_attention_low_level_capacity_guard(musa):
    q = torch.zeros(1, 1, 24, 256, device=musa, dtype=torch.bfloat16)
    k = torch.zeros(1, 4, 2, 256, device=musa, dtype=q.dtype)
    v = torch.ones_like(k)
    v[:, 3] = 2
    output = run_qsa_core(
        q,
        k,
        v,
        torch.tensor([[[0, 3, 4, 2**31 - 1]]], device=musa, dtype=torch.int32),
        torch.tensor([[100]], device=musa, dtype=torch.int32),
        torch.tensor([5], device=musa, dtype=torch.int32),
    )
    torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)


@pytest.mark.parametrize("page_id", [-1, 2**31 - 1])
@pytest.mark.parametrize("return_lse", [False, True])
def test_qsa_attention_masks_invalid_pages(musa, page_id, return_lse):
    q, k, v, indices, options = _attention_inputs(musa, torch.bfloat16, width=65)
    options["block_table"].fill_(page_id)
    options["seq_lens_k"].fill_(2**31 - 1)
    v.fill_(torch.nan)
    result = qsa_sparse_attention(
        q, k, v, indices, backend="tilelang", return_lse=return_lse, **options
    )
    output = result[0] if return_lse else result
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    if return_lse:
        assert torch.isneginf(result[1]).all()


@pytest.mark.parametrize(
    "entry,rows,columns",
    [
        ("score", 17, 65),
        ("prefill", 17, 65),
        ("decode", 17, 65),
        ("score", 257, 131073),
    ],
)
@pytest.mark.usefixtures("native_only")
def test_qsa_mqa_public_graph(musa, entry, rows, columns):
    q = torch.randn(1, rows, 4, 128, device=musa, dtype=torch.bfloat16) / 8
    k = torch.randn(1, columns, 128, device=musa, dtype=q.dtype) / 8
    positions = torch.full((1, rows), columns * 4 - 1, device=musa, dtype=torch.int32)
    starts, ends = (
        torch.zeros_like(positions[0]),
        torch.full_like(positions[0], columns),
    )
    cache = k[0, :64].view(8, 8, 1, 128)
    table = torch.arange(8, device=musa, dtype=torch.int32)[None]
    context = torch.full((1,), 64, device=musa, dtype=torch.int32)

    def run():
        if entry == "score":
            return qsa_indexer_mqa_score(q, k, query_positions=positions)
        if entry == "prefill":
            return qsa_mqa_prefill(q[0], k[0], starts, ends)
        return qsa_mqa_decode(q[:, 0], cache, table, context, 67)

    run()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        actual = run()
    for visible in (0, 31, 64):
        q.mul_(-0.75)
        positions.fill_(4 * visible - 1)
        ends.fill_(visible)
        context.fill_(visible)
        actual.fill_(torch.nan)
        graph.replay()
        if entry == "score":
            expected = mqa_dense_score_reference(q, k, scale=128**-0.5)
            expected[..., visible:] = -torch.inf
        elif entry == "prefill":
            expected = mqa_prefill_reference(
                q[0], k[0], starts, ends, multiplier=128**-0.5
            )
        else:
            expected = mqa_decode_reference(
                q[:, 0], cache, table, context, 67, multiplier=128**-0.5
            )
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize(
    "layout,rows,structured,heads,kv_heads",
    [
        ("contiguous", 2, False, 24, 2),
        ("paged", 2, False, 24, 2),
        ("paged", 8, False, 24, 2),
        ("paged", 8, True, 24, 2),
        ("paged", 8, "compact", 24, 2),
        ("contiguous", 2, False, 3, 1),
        ("paged", 2, False, 3, 1),
        ("paged", 8, False, 3, 1),
    ],
)
@pytest.mark.usefixtures("native_only")
def test_qsa_attention_logical_public_graph(
    musa, return_lse, layout, rows, structured, heads, kv_heads
):
    q, k, v, indices, options = _attention_inputs(
        musa,
        torch.bfloat16,
        layout=layout,
        rows=rows,
        width=2051,
        heads=heads,
        kv_heads=kv_heads,
    )
    if layout == "paged":
        options["num_requests"] = None
        if rows == 8 and not structured:
            del options["token_to_req"], options["num_requests"]
            options["cu_seqlens_q"] = torch.tensor(
                [0, rows], device=musa, dtype=torch.int32
            )
    blocks = (
        torch.full((rows, 512), -1, device=musa, dtype=torch.int32)
        if structured
        else None
    )
    if structured:
        rank = torch.arange(512, device=musa, dtype=torch.int32)[None]
        blocks.copy_(
            torch.where(rank < (options["query_positions"][:, None] + 1) // 4, rank, -1)
        )
        initial, _ = expand_qsa_block_indices(blocks, options["query_positions"])
        indices.copy_(initial)
    out = sliced_tensor(torch.empty_like(q), feature_step=True)[0]

    def run():
        return qsa_sparse_attention(
            q,
            k,
            v,
            indices,
            block_indices=blocks,
            assume_compact_blocks=structured == "compact",
            out=out,
            return_lse=return_lse,
            **options,
        )

    run()
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        result = run()
    assert (result[0] if return_lse else result) is out
    for visible in (0, 35, 67):
        q.mul_(-0.75)
        options["query_positions"].fill_(visible - 1)
        options["seq_lens_k"].fill_(visible)
        if structured:
            blocks.fill_(-1)
            blocks[:, : visible // 4] = torch.arange(
                visible // 4, device=musa, dtype=torch.int32
            )
            selected, _ = expand_qsa_block_indices(blocks, options["query_positions"])
            indices.copy_(selected)
        out.fill_(torch.nan)
        graph.replay()
        expected, expected_lse = _attention_reference(q, k, v, indices, options)
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.002)
        if return_lse:
            torch.testing.assert_close(result[1], expected_lse, rtol=0.002, atol=0.002)


@pytest.mark.usefixtures("native_only")
@pytest.mark.parametrize("splits", [1, 2])
@pytest.mark.parametrize("requests", [0, 1])
def test_qsa_paged_invalid_request_ids(musa, splits, requests):
    q = torch.zeros(5, 3, 256, device=musa, dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 1, 256, device=musa, dtype=q.dtype)
    v = torch.ones_like(k)
    request_ids = torch.tensor(
        [-(2**31), -1, 0, 1, 2**31 - 1], device=musa, dtype=torch.int32
    )
    indices = torch.zeros(5, 65, device=musa, dtype=torch.int32)
    positions = torch.zeros(5, device=musa, dtype=torch.int32)
    lengths = torch.ones(requests, device=musa, dtype=torch.int32)
    table = torch.zeros(requests, 1, device=musa, dtype=torch.int32)
    runner = run_qsa_core_paged if splits == 1 else run_qsa_core_paged_splitk
    result = runner(
        q,
        k,
        v,
        indices,
        request_ids,
        positions,
        lengths,
        table,
        flat_physical_metadata=False,
        **({"num_splits": splits, "return_lse": True} if splits > 1 else {}),
    )
    output, lse = result if splits > 1 else (result, None)
    expected = torch.zeros_like(q)
    if requests:
        expected[2] = 1
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if lse is not None:
        expected_lse = torch.full_like(lse, -torch.inf)
        if requests:
            expected_lse[2] = math.log(indices.shape[1])
        torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("entry", ["mqa", "contiguous", "paged"])
def test_qsa_cached_query_padding(musa, dtype, entry):
    positions = torch.tensor(
        [[64, 65, 66], [128, 129, 130]], device=musa, dtype=torch.int32
    )
    lengths = torch.tensor([1, 2], device=musa, dtype=torch.int32)
    if entry == "mqa":
        q = torch.full((2, 3, 4, 128), 0.125, device=musa, dtype=dtype)
        k = torch.full((2, 40, 128), 0.125, device=musa, dtype=dtype)
        actual = qsa_indexer_mqa_score(
            q, k, query_positions=positions, seq_lens_q=lengths
        )
        valid = (
            torch.arange(3, device=musa)[None, :, None] < lengths[:, None, None]
        ) & (
            torch.arange(40, device=musa)[None, None, :]
            < (positions[:, :, None] + 1) // 4
        )
        expected = torch.where(valid, 8 * 128**-0.5, -torch.inf)
    else:
        q = torch.zeros(2, 3, 24, 256, device=musa, dtype=dtype)
        k = torch.zeros(2, 131, 2, 256, device=musa, dtype=dtype)
        v = torch.ones_like(k)
        options = {}
        if entry == "paged":
            k, v, options["block_table"] = paged_kv_fixture(k, v, 16)
        actual = qsa_sparse_attention(
            q,
            k,
            v,
            torch.zeros(2, 3, 1, device=musa, dtype=torch.int32),
            query_positions=positions,
            seq_lens_q=lengths,
            seq_lens_k=torch.full((2,), 131, device=musa, dtype=torch.int32),
            backend="tilelang",
            **options,
        )
        expected = (
            (torch.arange(3, device=musa)[None] < lengths[:, None])[:, :, None, None]
            .expand_as(q)
            .to(dtype)
        )
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qsa_cached_varlen_query_padding(musa, dtype):
    q = torch.zeros(6, 3, 256, device=musa, dtype=dtype)
    k = torch.zeros(2, 67, 1, 256, device=musa, dtype=dtype)
    v = torch.ones_like(k)
    v[1] = 2
    k, v, table = paged_kv_fixture(k, v, 16)
    actual, lse = qsa_sparse_attention(
        q,
        k,
        v,
        torch.zeros(6, 1, device=musa, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 6], device=musa, dtype=torch.int32),
        query_positions=torch.arange(64, 70, device=musa, dtype=torch.int32),
        seq_lens_q=torch.tensor([1, 2], device=musa, dtype=torch.int32),
        seq_lens_k=torch.full((2,), 67, device=musa, dtype=torch.int32),
        block_table=table,
        return_lse=True,
        backend="tilelang",
    )
    expected = torch.tensor([1, 0, 2, 2, 0, 0], device=musa, dtype=dtype)
    torch.testing.assert_close(
        actual, expected[:, None, None].expand_as(q), rtol=0, atol=0
    )
    expected_lse = torch.where(expected > 0, 0.0, -torch.inf)
    torch.testing.assert_close(lse, expected_lse[:, None].expand(6, 3), rtol=0, atol=0)


@pytest.mark.parametrize("scale", [256**-0.5, 1.0])
@pytest.mark.parametrize(
    "layout,rows,width,heads,kv_heads,splits,ws",
    [
        ("contiguous", 2, 129, 3, 1, 1, False),
        ("paged", 2, 257, 3, 1, 2, False),
        ("paged", 7, 2051, 24, 2, 1, False),
        ("paged", 7, 2051, 24, 2, 1, True),
    ],
)
def test_qsa_attention_finite_negative_logits(
    musa, layout, rows, width, heads, kv_heads, splits, ws, scale
):
    q, k, v, indices, options = _attention_inputs(
        musa,
        torch.bfloat16,
        layout=layout,
        rows=rows,
        width=width,
        heads=heads,
        kv_heads=kv_heads,
    )
    q.fill_(1e16)
    k.fill_(-1e16)
    v.fill_(1)
    q.view(rows, heads, 256)[1] = torch.nan
    indices.fill_(-1)
    indices.view(rows, width)[0, 64] = 0
    options["query_positions"].fill_(66)
    options["scale"] = scale
    want_lse = rows == 2
    result = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        num_splits=splits,
        warp_specialized=ws,
        return_lse=want_lse,
        backend="tilelang",
        **options,
    )
    actual = result[0] if want_lse else result
    expected = torch.zeros_like(q)
    expected.view(rows, heads, 256)[0] = 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference, _ = _attention_reference(q, k, v, indices, options)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    if want_lse:
        expected_lse = torch.full(q.shape[:-1], -torch.inf, device=musa)
        expected_lse.view(rows, heads)[0] = -256 * scale * float(q.flatten()[0]) ** 2
        torch.testing.assert_close(result[1], expected_lse, rtol=0.002, atol=0)


@pytest.mark.parametrize("explicit_positions", [False, True])
def test_qsa_attention_interleaved_padding_graph(musa, explicit_positions):
    q = torch.zeros(8, 3, 256, device=musa, dtype=torch.bfloat16)
    k = torch.zeros(3, 16, 1, 256, device=musa, dtype=q.dtype)
    v = torch.ones_like(k)
    v[2] = 2
    table = torch.tensor([[1], [2]], device=musa, dtype=torch.int32)
    requests = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], device=musa, dtype=torch.int32)
    lengths = torch.tensor([2, 1], device=musa, dtype=torch.int32)
    options = dict(
        token_to_req=requests,
        num_requests=2,
        seq_lens_q=lengths,
        seq_lens_k=torch.full((2,), 16, device=musa, dtype=torch.int32),
        block_table=table,
        backend="tilelang",
    )
    if explicit_positions:
        options["query_positions"] = torch.full(
            (8,), 13, device=musa, dtype=torch.int32
        )
    indices = torch.zeros(8, 1, device=musa, dtype=torch.int32)
    qsa_sparse_attention(q, k, v, indices, **options)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        actual = qsa_sparse_attention(q, k, v, indices, **options)
    for ids, counts in [
        ([1, 0, 1, 0, 0, 1, 0, 1], [2, 1]),
        ([0, 1, 1, 0, 0, 0, 0, 1], [1, 3]),
        ([0, 1, 1, 0, 0, 0, 0, 1], [0, 0]),
    ]:
        requests.copy_(torch.tensor(ids, device=musa, dtype=torch.int32))
        lengths.copy_(torch.tensor(counts, device=musa, dtype=torch.int32))
        actual.fill_(torch.nan)
        graph.replay()
        expected = torch.tensor(
            [
                req + 1 if ids[:row].count(req) < counts[req] else 0
                for row, req in enumerate(ids)
            ],
            device=musa,
            dtype=q.dtype,
        )
        torch.testing.assert_close(
            actual, expected[:, None, None].expand_as(q), rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qsa_reference_varlen_extreme_indices(dtype):
    q = torch.zeros(4, 3, 256, dtype=dtype)
    k = torch.zeros(3, 1, 256, dtype=dtype)
    v = torch.tensor([2, 6, 10], dtype=dtype)[:, None, None].expand_as(k)
    lo, hi = torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max
    indices = torch.tensor(
        [[0, hi, lo, -1], [hi, lo, -1, hi], [0, 1, hi, lo], [1, hi, -1, lo]]
    )
    out, storage = sliced_tensor(torch.full_like(q, -123))
    before = storage.clone()
    actual, lse = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1, 3], dtype=torch.int32),
        query_positions=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        seq_lens_k=torch.tensor([1, 2], dtype=torch.int32),
        backend="reference",
        out=out,
        return_lse=True,
    )
    assert actual is out
    expected = torch.tensor([2, 0, 8, 10], dtype=dtype)[:, None, None].expand_as(q)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_lse = torch.tensor([0, -torch.inf, math.log(2), 0])[:, None].expand(4, 3)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)
    before.as_strided(out.shape, out.stride(), out.storage_offset()).copy_(expected)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device,dtype,mode",
    [("cpu", torch.float32, "lse")]
    + [
        ("musa", dtype, mode)
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
        for mode in ("plain", "gate", "lse")
    ],
)
def test_qsa_attention_split_merge_empty_and_extreme_lse(request, device, dtype, mode):
    device = (
        request.getfixturevalue("musa") if device == "musa" else torch.device(device)
    )
    if device.type == "musa":
        request.getfixturevalue("native_only")
    partial = torch.full((2, 4, 3, 256), 3.0, device=device, dtype=dtype)
    lse = torch.zeros((2, 4, 3), device=device)
    # Exercise all nonfinite values in separate heads of the same invocation.
    poison = torch.tensor(
        [torch.nan, torch.inf, -torch.inf], device=device, dtype=dtype
    )
    poison = poison[:, None]
    partial[0, :3] = poison
    lse[0, :2] = -torch.inf
    partial[1, 1] = poison
    lse[1, 1] = -torch.inf
    lse[0, 2] = -1e25
    lse[0, 3], lse[1, 3] = -1e25, -torch.inf
    gate = torch.zeros((4, 3), device=device) if mode == "gate" else None
    if gate is not None:
        gate[1] = torch.nan
    output_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
    out = (
        sliced_tensor(
            torch.empty((4, 3, 256), device=device, dtype=output_dtype),
            feature_step=True,
        )[0]
        if mode != "plain"
        else None
    )
    result = merge_qsa_core_paged_splitk(
        partial, lse, gate=gate, return_lse=mode == "lse", out=out
    )
    output = result[0] if mode == "lse" else result
    if out is not None:
        assert output is out
    expected = torch.full_like(output, 1.5 if gate is not None else 3.0)
    expected[1] = 0
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if mode == "lse":
        expected_lse = torch.zeros_like(lse[0])
        expected_lse[1], expected_lse[3] = -torch.inf, -1e25 / math.log2(math.e)
        torch.testing.assert_close(result[1], expected_lse, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "entry,backend",
    [(entry, None) for entry in ("prefill", "score", "decode", "attention")]
    + [("prefill", "auto")],
)
def test_qsa_native_backend_validation(entry, backend):
    q = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    k = torch.zeros(8, 128, dtype=q.dtype)
    counts = torch.ones(1, dtype=torch.int32)
    options = {} if backend is None else {"backend": backend}
    message = "unsupported QSA backend" if backend == "auto" else "TileLang"
    with pytest.raises(ValueError, match=message):
        if entry == "prefill":
            qsa_mqa_prefill(q, k, counts - 1, counts, **options)
        elif entry == "score":
            qsa_indexer_mqa_score(q[None], k[None], **options)
        elif entry == "decode":
            qsa_mqa_decode(
                q, k[None, :, None], counts[:, None] - 1, counts, 8, **options
            )
        else:
            query = torch.zeros(1, 1, 3, 256, dtype=q.dtype)
            key = torch.zeros(1, 8, 1, 256, dtype=q.dtype)
            qsa_sparse_attention(
                query,
                key,
                key,
                torch.zeros(1, 1, 1, dtype=torch.int32),
                **options,
            )


@pytest.mark.usefixtures("native_only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["contiguous", "paged"])
@pytest.mark.parametrize("rows,return_lse", [(0, False), (3, True)])
def test_qsa_attention_flattened_indices(musa, dtype, layout, rows, return_lse):
    q, k, v, indices, options = _attention_inputs(
        musa,
        dtype,
        layout=layout,
        rows=max(1, rows),
        heads=3,
        kv_heads=1,
        width=7,
    )
    q = q[:, :rows] if layout == "contiguous" else q[:rows]
    indices = sliced_tensor(indices.view(-1, 7), 3)[0][:rows]
    options["query_positions"] = (
        options["query_positions"][:, :rows]
        if layout == "contiguous"
        else options["query_positions"][:rows]
    )
    if layout == "paged":
        options["token_to_req"] = options["token_to_req"][:rows]
    options["gate"] = torch.ones(q.shape[:-1], device=musa)
    out, storage = sliced_tensor(torch.empty_like(q), feature_step=True)
    expected, expected_lse = _attention_reference(q, k, v, indices, options)
    result = qsa_sparse_attention(
        q, k, v, indices, out=out, return_lse=return_lse, **options
    )
    actual = result[0] if return_lse else result
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    if return_lse:
        assert result[1].dtype == torch.float32
        torch.testing.assert_close(result[1], expected_lse, rtol=0.002, atol=0.002)
    out.fill_(97)
    assert (storage == 97).all()


@pytest.mark.usefixtures("native_only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("entry", ["prefill", "decode"])
def test_qsa_mqa_empty_storage(musa, dtype, entry):
    q = torch.empty(2, 4, 128, device=musa, dtype=dtype)
    counts = torch.zeros(2, device=musa, dtype=torch.int32)
    if entry == "prefill":
        output = qsa_mqa_prefill(q, q.new_empty(0, 128), counts, counts)
        assert output.shape == (2, 0)
    else:
        output = qsa_mqa_decode(
            q, q.new_empty(0, 16, 1, 128), counts[:, None], counts, 7
        )
        assert output.shape == (2, 7)
        assert torch.isneginf(output).all()
    assert output.dtype == torch.float32


@pytest.mark.parametrize(
    "layout,splits,return_lse",
    [
        ("contiguous", 1, False),
        ("contiguous", 1, True),
        ("paged", 1, False),
        ("paged", 2, False),
        ("paged", 2, True),
    ],
)
@pytest.mark.usefixtures("native_only")
def test_qsa_attention_padded_gate_graph(musa, layout, splits, return_lse):
    q = torch.zeros(1, 3, 3, 256, device=musa, dtype=torch.bfloat16)
    q[:, 2] = torch.nan
    k = torch.zeros(1, 16, 1, 256, device=musa, dtype=q.dtype)
    v = torch.ones_like(k)
    indices = (torch.arange(65, device=musa, dtype=torch.int32) % 2).expand(1, 3, -1)
    gate = sliced_tensor(torch.zeros(1, 3, 3, device=musa, dtype=q.dtype))[0]
    gate[:, 2] = torch.nan
    lengths = torch.tensor([2], device=musa, dtype=torch.int32)
    positions = sliced_tensor(torch.arange(3, device=musa, dtype=torch.int32))[0][None]
    out = sliced_tensor(torch.empty_like(q), feature_step=True)[0]
    options = dict(
        seq_lens_q=lengths,
        seq_lens_k=torch.full_like(lengths, 2),
        query_positions=positions,
        gate=gate,
        out=out,
        num_splits=splits,
        return_lse=return_lse,
    )
    if layout == "paged":
        options["block_table"] = torch.zeros(1, 1, device=musa, dtype=torch.int32)
    qsa_sparse_attention(q, k, v, indices, **options)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        result = qsa_sparse_attention(q, k, v, indices, **options)
    output = result[0] if return_lse else result
    assert output is out
    for valid_rows in (2, 1, 2):
        lengths.fill_(valid_rows)
        gate[:, 1] = 0 if valid_rows == 2 else torch.nan
        out.fill_(torch.nan)
        graph.replay()
        expected = torch.zeros_like(out)
        expected[:, :valid_rows] = 0.5
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if return_lse:
            expected_lse = torch.full_like(result[1], -torch.inf)
            for row, count in enumerate((33, 65)[:valid_rows]):
                expected_lse[:, row] = math.log(count)
            torch.testing.assert_close(result[1], expected_lse, rtol=1e-5, atol=1e-5)
