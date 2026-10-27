from __future__ import annotations

import ast
import inspect

import pytest
import torch

import mate
from mate import msa_interface as msa
from mate.execution_context import is_dry_run_enabled, raise_complete_if_dry_run
from mate.testing import supported_musa_compute_capability
from mate.testing.operators import OpMode
from mate.testing.operators.msa import (
    MsaApi,
    MsaKernelMode,
    MsaKvLayout,
    MsaOperator,
    MsaWorkload,
    UnsupportedMsaWorkload,
)

_HAS_FP8_E4M3 = hasattr(torch, "float8_e4m3fn")
_FWD_DTYPES = [torch.float16, torch.bfloat16]
if _HAS_FP8_E4M3:
    _FWD_DTYPES.append(torch.float8_e4m3fn)


@pytest.mark.parametrize("arch", ["mp31"])
def test_msa_bf16_prefetch_and_dispatch_module_cache(monkeypatch, arch):
    from mate.jit import msa_ops

    builds = []

    class FakeSpec:
        def build_and_load(self):
            module = object()
            builds.append(module)
            return module

    monkeypatch.setattr(msa_ops.jit_env, "resolve_musa_arch_list", lambda: (arch,))
    monkeypatch.setattr(msa_ops, "_MSA_MAXSCORE_MODULES", {})
    monkeypatch.setattr(
        msa_ops, "_gen_msa_maxscore_spec_from_config", lambda config: FakeSpec()
    )
    options = dict(
        is_paged_kv=True, causal=True, head_dim=128, head_ratio=1, is_varlen=False
    )
    for q_len, tile_q, prefetch in [(128, 64, False), (513, 128, True)]:
        config = msa_ops.make_msa_maxscore_config(
            torch.bfloat16, max_seqlen_q=q_len, **options
        )
        assert config["tile_q"] == tile_q
        assert config["enable_k_prefetch"] is prefetch
    modules = [
        msa_ops.get_msa_maxscore_module(torch.bfloat16, max_seqlen_q=q_len, **options)
        for q_len in (513, 1025, 4096, 128)
    ]
    assert modules[0] is modules[1] is modules[2]
    assert modules[3] is not modules[0]
    assert len(builds) == 2


def test_msa_maxscore_prefill_tile_q_dispatch():
    from mate.jit.msa_ops import _maxscore_tile_q

    for max_seqlen_q in (513, 32 * 1024, 64 * 1024):
        assert (
            _maxscore_tile_q(
                1,
                True,
                dtype=torch.bfloat16,
                max_seqlen_q=max_seqlen_q,
            )
            == 128
        )
    for max_seqlen_q in (64, 128, 512):
        assert (
            _maxscore_tile_q(
                1,
                True,
                dtype=torch.bfloat16,
                max_seqlen_q=max_seqlen_q,
            )
            == 64
        )
    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.bfloat16,
            max_seqlen_q=63,
        )
        == 16
    )
    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.float8_e4m3fn if _HAS_FP8_E4M3 else torch.bfloat16,
            max_seqlen_q=128 * 1024,
        )
        == 128
    )
    if _HAS_FP8_E4M3:
        for max_seqlen_q in (1, 16, 17, 64, 128, 512):
            assert (
                _maxscore_tile_q(
                    1,
                    True,
                    dtype=torch.float8_e4m3fn,
                    max_seqlen_q=max_seqlen_q,
                )
                == 16
            )
        assert (
            _maxscore_tile_q(
                1,
                True,
                dtype=torch.float8_e4m3fn,
                max_seqlen_q=513,
            )
            == 128
        )


def test_msa_maxscore_template_config_encodes_page_and_pipeline_options():
    from mate.jit.msa_ops import _msa_maxscore_encode_config, make_msa_maxscore_config

    config = make_msa_maxscore_config(
        torch.float8_e4m3fn if _HAS_FP8_E4M3 else torch.bfloat16,
        is_paged_kv=True,
        causal=True,
        head_dim=128,
        head_ratio=1,
        max_seqlen_q=3584,
        page_table_kind="flat",
        page_size=128,
        is_varlen=False,
    )
    encoded = _msa_maxscore_encode_config(config)
    assert config["page_table_kind"] == "flat"
    assert config["page_size"] == 128
    assert config["q_stages"] == 1
    assert config["mma_tile_q"] in {16, 32}
    assert config["is_varlen"] is False
    assert config["has_metadata"] is False
    assert "_pt_flat_ps_128_" in encoded
    assert f"_ks_{config['k_stages']}_pf_{int(config['enable_k_prefetch'])}_" in encoded
    assert f"_mq_{config['mma_tile_q']}_vl_0" in encoded

    metadata_config = make_msa_maxscore_config(
        torch.bfloat16,
        is_paged_kv=True,
        causal=True,
        head_dim=128,
        head_ratio=1,
        max_seqlen_q=64,
        page_table_kind="batched2d",
        page_size=128,
        is_varlen=True,
    )
    assert metadata_config["has_metadata"] is True
    assert _msa_maxscore_encode_config(metadata_config).endswith("_hm_1")

    with pytest.raises(ValueError, match="page_size == 128"):
        make_msa_maxscore_config(
            torch.bfloat16,
            is_paged_kv=True,
            causal=True,
            head_dim=128,
            head_ratio=1,
            max_seqlen_q=64,
            page_table_kind="batched2d",
            page_size=64,
        )


def test_msa_per_mp_schedule_slot_count_tracks_q_work():
    from mate.jit.msa_ops import _msa_schedule_num_mps

    # A decode-sized Q range should not need a full-device persistent grid,
    # while the long prefill shape still uses all logical MP slots.
    assert (
        _msa_schedule_num_mps(
            hardware_mps=56,
            q_work_count=1,
            max_k_tiles=8,
            parallel_k_tiles=True,
        )
        == 3
    )
    assert (
        _msa_schedule_num_mps(
            hardware_mps=56,
            q_work_count=28,
            max_k_tiles=1024,
            parallel_k_tiles=True,
        )
        == 56
    )


def test_msa_schedule_q_threshold_accounts_for_k_work():
    from mate.jit.msa_ops import (
        _msa_schedule_q_work_threshold,
        _msa_should_use_schedule,
    )

    # The threshold is expressed in logical Q-work tiles, not raw query rows.
    # A long-K work item amortizes the metadata launch sooner than a short-K
    # work item.
    assert _msa_schedule_q_work_threshold(1024) == 8
    assert _msa_schedule_q_work_threshold(32) == 256
    assert not _msa_should_use_schedule(
        q_work_count=1, valid_k_tiles=1024, schedule_requested=True
    )
    assert not _msa_should_use_schedule(
        q_work_count=16, valid_k_tiles=32, schedule_requested=True
    )
    assert not _msa_should_use_schedule(
        q_work_count=256, valid_k_tiles=32, schedule_requested=True, tile_q=16
    )
    assert not _msa_should_use_schedule(
        q_work_count=4, valid_k_tiles=1024, schedule_requested=True
    )
    assert _msa_should_use_schedule(
        q_work_count=8, valid_k_tiles=1024, schedule_requested=True, tile_q=128
    )
    assert not _msa_should_use_schedule(
        q_work_count=1024, valid_k_tiles=32, schedule_requested=False
    )


def test_msa_plan_from_lengths_dense_contract():
    qo_lens = torch.tensor([8, 16], dtype=torch.int64)
    kv_lens = torch.tensor([32, 64], dtype=torch.int64)

    has_mixed, split, batch_size, plan, prefill = msa._msa_plan_from_lengths(
        qo_lens.cpu(),
        kv_lens.cpu(),
        16,
        num_kv_heads=4,
        causal=True,
    )

    assert has_mixed is False
    assert split == 0
    assert batch_size == 2
    assert prefill is None
    assert plan.mode == "dense"
    assert plan.num_qo_heads == 16
    assert plan.num_kv_heads == 4
    assert plan.prefill_plan.is_varlen is True
    assert plan.kv_page_indptr is None
    assert plan.prefill_plan.total_seqlen_k == 96
    assert torch.equal(
        plan.qo_offset, kv_lens.to(torch.int32) - qo_lens.to(torch.int32)
    )
    assert torch.equal(
        plan.prefill_plan.cu_seqlens_q,
        torch.tensor([0, 8, 24], dtype=torch.int32),
    )


def test_msa_sparse_prefill_mode_is_declared():
    qo_lens = torch.tensor([64], dtype=torch.int32)
    kv_lens = torch.tensor([256], dtype=torch.int32)

    _, _, _, plan, _ = msa._msa_plan_from_lengths(
        qo_lens.cpu(),
        kv_lens.cpu(),
        16,
        num_kv_heads=4,
        page_size=128,
        kv_block_num=16,
    )

    assert plan.mode == "sparse_prefill"
    assert plan.prefill_plan.is_varlen is False
    assert plan.prefill_plan.total_seqlen_k == 256
    assert torch.equal(
        plan.kv_page_indptr,
        torch.tensor([0, 2], dtype=torch.int32),
    )


def test_msa_plan_from_lengths_keeps_topk_separate_from_selected_block_count():
    _, _, _, plan, _ = msa._msa_plan_from_lengths(
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([4096], dtype=torch.int32),
        8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
    )

    assert plan.kv_block_num == 16
    assert plan.selected_block_count == 18
    assert plan.force_begin_blocks == 1
    assert plan.force_end_blocks == 1
    assert not plan.force_blocks_count_in_topk


def test_msa_plan_requires_device_runtime_metadata():
    plan_info = msa.msa_plan(
        batch_size=1,
        max_seqlen_q=128,
        max_seqlen_k=4096,
        num_qo_heads=8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        sparse_kernel_mode="prefill",
    )
    _, _, _, plan, extra = plan_info
    assert extra is None
    assert plan.is_static
    assert plan.mode == "sparse_prefill"
    assert plan.prefill_plan.max_seqlen_q == 128
    assert plan.prefill_plan.max_seqlen_k == 4096
    assert torch.count_nonzero(plan.qo_lens) == 0

    q = torch.empty((1, 8, 128), dtype=torch.float16)
    k = torch.empty((32, 1, 128, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indexes = torch.zeros((1, 1, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match="static MSA plans require runtime_metadata"):
        msa.sparse_msa(
            q,
            k,
            v,
            plan_info,
            kv_indices=torch.arange(32, dtype=torch.int32),
            kv_block_indexes=block_indexes,
        )


def test_msa_plan_is_the_only_public_capacity_planner():
    assert "msa_plan" in msa.__all__
    assert "msa_plan_static" not in msa.__all__
    assert "msa_plan" in mate.__all__
    assert "msa_plan_static" not in mate.__all__
    assert not hasattr(msa, "msa_plan_static")
    assert not hasattr(mate, "msa_plan_static")


def test_msa_interface_does_not_use_tensor_item():
    tree = ast.parse(inspect.getsource(msa))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    assert offenders == []


def test_eager_msa_plan_rejects_accelerator_lengths():
    qo_lens = torch.ones((1,), dtype=torch.int32, device="meta")
    kv_lens = torch.ones((1,), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="must be a CPU tensor"):
        msa._msa_plan_from_lengths(qo_lens, kv_lens, 8, num_kv_heads=1)


def test_msa_static_runtime_metadata_reaches_fwd_kernel(monkeypatch):
    calls = []

    def fake_msa_fwd(q, k, v, block_indexes, *args, **kwargs):
        calls.append((q, k, v, block_indexes, args, kwargs))
        return torch.empty_like(q), torch.empty((q.shape[0], q.shape[1]))

    monkeypatch.setattr(msa, "_msa_fwd", fake_msa_fwd)
    plan_info = msa.msa_plan(
        batch_size=1,
        max_seqlen_q=128,
        max_seqlen_k=4096,
        num_qo_heads=8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        sparse_kernel_mode="prefill",
    )
    runtime = msa.MsaRuntimeMetadata(
        qo_lens=torch.tensor([1], dtype=torch.int32),
        kv_lens=torch.tensor([256], dtype=torch.int32),
        qo_offset=torch.tensor([255], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 256], dtype=torch.int32),
        seqused_k=torch.tensor([256], dtype=torch.int32),
        page_table=torch.cat(
            [
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.zeros((1, 30), dtype=torch.int32),
            ],
            dim=1,
        ),
    )
    q = torch.empty((1, 8, 128), dtype=torch.float16)
    k = torch.empty((2, 1, 128, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indexes = torch.zeros((1, 1, 16), dtype=torch.int32)
    lse_buffer = torch.empty((1, 8), dtype=torch.float32)
    out, lse = msa.sparse_msa(
        q,
        k,
        v,
        plan_info,
        kv_block_indexes=block_indexes,
        lse=lse_buffer,
        runtime_metadata=runtime,
    )
    assert out.shape == q.shape
    assert lse is None
    assert len(calls) == 1
    # args: cu_seqlens_q, seqused_k, qo_offset, page_table, ...
    assert calls[0][4][0] is runtime.cu_seqlens_q
    assert calls[0][4][1] is runtime.seqused_k
    assert calls[0][4][2] is runtime.qo_offset
    assert calls[0][4][3] is runtime.page_table
    assert calls[0][5]["kv_page_indptr"] is None
    assert calls[0][5]["lse"] is lse_buffer


def test_msa_static_runtime_metadata_reaches_paged_maxscore(monkeypatch):
    calls = []

    def fake_msa_maxscore(q, k, cu_q, cu_k, qo_offset, **kwargs):
        calls.append((q, k, cu_q, cu_k, qo_offset, kwargs))
        return kwargs["max_score"]

    monkeypatch.setattr(msa, "_msa_maxscore", fake_msa_maxscore)
    plan_info = msa.msa_plan(
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        total_seqlen_k=8192,
        num_qo_heads=1,
        num_kv_heads=1,
        page_size=128,
        output_maxscore=True,
    )
    _, _, _, plan, _ = plan_info
    assert plan.prefill_plan.total_seqlen_k == 8192

    runtime = msa.MsaRuntimeMetadata(
        qo_lens=torch.ones((2,), dtype=torch.int32),
        kv_lens=torch.tensor([256, 384], dtype=torch.int32),
        qo_offset=torch.tensor([255, 383], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 256, 640], dtype=torch.int32),
        seqused_k=torch.tensor([256, 384], dtype=torch.int32),
        page_table=torch.zeros((2, 32), dtype=torch.int32),
        maxscore_schedule=torch.empty((4, 2), dtype=torch.int32),
        maxscore_schedule_ready=True,
    )
    q = torch.empty((2, 1, 128), dtype=torch.float16)
    k = torch.empty((64, 128, 1, 128), dtype=torch.float16)
    max_score = torch.empty((2, 1, 128), dtype=torch.float32)
    out, actual = msa.msa(
        q,
        k,
        k,
        plan_info,
        output_o=False,
        output_maxscore=True,
        max_score=max_score,
        runtime_metadata=runtime,
    )

    assert out is None
    assert actual is max_score
    assert len(calls) == 1
    assert calls[0][2] is runtime.cu_seqlens_q
    assert calls[0][3] is runtime.cu_seqlens_k
    assert calls[0][4] is runtime.qo_offset
    assert calls[0][5]["page_table"] is runtime.page_table
    assert calls[0][5]["kv_page_indptr"] is None
    assert calls[0][5]["schedule_metadata"] is runtime.maxscore_schedule
    assert calls[0][5]["schedule_metadata_ready"] is True


def test_msa_maxscore_reuses_supplied_schedule_metadata(monkeypatch):
    from mate.jit import msa_ops

    metadata_calls = []
    kernel_calls = []

    class FakeModule:
        def get_function(self, name):
            if name.endswith("_metadata"):
                return lambda *args: metadata_calls.append(args)
            return lambda *args: kernel_calls.append(args)

    monkeypatch.setattr(
        msa_ops,
        "make_msa_maxscore_config",
        lambda *args, **kwargs: {
            "tile_q": 128,
            "parallel_k_tiles": False,
        },
    )
    monkeypatch.setattr(msa_ops, "_msa_maxscore_encode_config", lambda config: "fake")
    # This test isolates the ready-workspace ABI from the independent host-side
    # scheduler policy.  Upstream may legitimately bypass metadata for tiny
    # work sets, so force the scheduled branch here and verify build-vs-reuse.
    monkeypatch.setattr(msa_ops, "_msa_should_use_schedule", lambda **kwargs: True)
    monkeypatch.setattr(
        msa_ops, "get_msa_maxscore_module", lambda *args, **kwargs: FakeModule()
    )
    monkeypatch.setattr(msa_ops, "resolve_num_mps", lambda device, limit: 4)

    q = torch.empty((1, 1, 128), dtype=torch.float16)
    k = torch.empty((1, 128, 1, 128), dtype=torch.float16)
    cu_lens = torch.tensor([0, 1], dtype=torch.int32)
    qo_offset = torch.zeros((1,), dtype=torch.int32)
    schedule = torch.empty((4, 2), dtype=torch.int32)

    for ready in (False, True):
        msa_ops._msa_maxscore(
            q,
            k,
            cu_lens,
            cu_lens,
            qo_offset,
            max_seqlen_q=1,
            max_seqlen_k=128,
            causal=True,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            schedule_metadata=schedule,
            schedule_metadata_ready=ready,
            max_score=torch.empty((1, 1, 128), dtype=torch.float32),
        )

    assert len(metadata_calls) == 1
    assert len(kernel_calls) == 2
    assert kernel_calls[0][5] is schedule
    assert kernel_calls[1][5] is schedule


def test_msa_maxscore_rejects_schedule_rows_that_change_k_partitioning(
    monkeypatch,
):
    from mate.jit import msa_ops

    class FakeModule:
        def get_function(self, name):
            raise AssertionError("invalid schedule must fail before kernel launch")

    monkeypatch.setattr(
        msa_ops,
        "make_msa_maxscore_config",
        lambda *args, **kwargs: {
            "tile_q": 64,
            "parallel_k_tiles": True,
        },
    )
    monkeypatch.setattr(msa_ops, "_msa_maxscore_encode_config", lambda config: "fake")
    monkeypatch.setattr(
        msa_ops, "get_msa_maxscore_module", lambda *args, **kwargs: FakeModule()
    )
    monkeypatch.setattr(msa_ops, "resolve_num_mps", lambda device, limit: 56)
    monkeypatch.setattr(msa_ops, "_msa_should_use_schedule", lambda **kwargs: True)

    with pytest.raises(ValueError, match="changes max-score K partitioning"):
        msa_ops._msa_maxscore(
            torch.empty((512, 1, 128), dtype=torch.bfloat16),
            torch.empty((1, 128, 1, 128), dtype=torch.bfloat16),
            torch.tensor([0, 512], dtype=torch.int32),
            torch.tensor([0, 131072], dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            max_seqlen_q=512,
            max_seqlen_k=131072,
            causal=True,
            page_table=torch.zeros((1, 1024), dtype=torch.int32),
            schedule_metadata=torch.empty((1, 2), dtype=torch.int32),
            schedule_metadata_ready=True,
            max_score=torch.empty((512, 1, 1024), dtype=torch.float32),
        )


def test_msa_maxscore_ready_schedule_requires_workspace():
    from mate.jit.msa_ops import _msa_maxscore

    with pytest.raises(ValueError, match="requires a supplied schedule_metadata"):
        _msa_maxscore(
            torch.empty((1, 1, 128), dtype=torch.float16),
            torch.empty((1, 128, 1, 128), dtype=torch.float16),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_k=128,
            causal=True,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            schedule_metadata_ready=True,
            max_score=torch.empty((1, 1, 128), dtype=torch.float32),
        )


def test_msa_maxscore_ready_schedule_requires_exact_contiguous_workspace():
    from mate.jit.msa_ops import _msa_maxscore

    noncontiguous_schedule = torch.empty((4, 4), dtype=torch.int32)[:, ::2]
    assert not noncontiguous_schedule.is_contiguous()
    with pytest.raises(ValueError, match="exact contiguous schedule workspace"):
        _msa_maxscore(
            torch.empty((1, 1, 128), dtype=torch.float16),
            torch.empty((1, 128, 1, 128), dtype=torch.float16),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_k=128,
            causal=True,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            schedule_metadata=noncontiguous_schedule,
            schedule_metadata_ready=True,
            max_score=torch.empty((1, 1, 128), dtype=torch.float32),
        )


def test_sparse_msa_plan_rejects_page_block_size_mismatch():
    with pytest.raises(ValueError, match="page_size == sparse_block_size"):
        msa.sparse_msa_plan(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            sparse_block_size=128,
            kv_block_num=4,
            causal=True,
        )


def test_msa_plan_from_lengths_rejects_page64_paged_maxscore():
    with pytest.raises(ValueError, match="paged MSA maxscore requires page_size"):
        msa._msa_plan_from_lengths(
            torch.tensor([64], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            output_maxscore=True,
            causal=True,
        )


def test_sparse_msa_plan_rejects_unsupported_sparse_block_size():
    with pytest.raises(ValueError, match="sparse_block_size == 128"):
        msa.sparse_msa_plan(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            sparse_block_size=64,
            kv_block_num=4,
            causal=True,
        )


def test_build_page_table_from_flat_kv_indices():
    kv_lens = torch.tensor([256, 384], dtype=torch.int32)
    kv_indices = torch.tensor([3, 5, 7, 11, 13], dtype=torch.int32)

    page_table = msa._build_page_table_from_flat_kv_indices(
        kv_indices,
        kv_lens,
        page_size=128,
    )

    assert torch.equal(
        page_table,
        torch.tensor([[3, 5, 0], [7, 11, 13]], dtype=torch.int32),
    )


@pytest.mark.parametrize("mode,q_len", [("prefill", 4), ("decode", 1)])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (16, 1), (32, 2)])
@pytest.mark.parametrize("dtype", _FWD_DTYPES)
def test_sparse_modes_route_to_unified_msa_fwd(
    monkeypatch,
    mode: str,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
):
    calls = []

    def fake_msa_fwd(q, k, v, block_indexes, *args, **kwargs):
        calls.append((q, k, v, block_indexes, args, kwargs))
        output = kwargs["out"] if kwargs["out"] is not None else torch.empty_like(q)
        return (
            output,
            torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32),
        )

    monkeypatch.setattr(msa, "_msa_fwd", fake_msa_fwd)
    qo_lens = torch.tensor([q_len], dtype=torch.int32)
    kv_lens = torch.tensor([256], dtype=torch.int32)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens.cpu(),
        kv_lens.cpu(),
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=128,
        sparse_block_size=128,
        kv_block_num=16,
        sparse_kernel_mode=mode,
        split_prefill_decode=False,
        causal=False,
    )
    q = torch.empty((q_len, num_q_heads, 128), dtype=dtype)
    k = torch.empty((2, num_kv_heads, 128, 128), dtype=dtype)
    v = torch.empty_like(k)
    block_indexes = torch.full((q_len, num_kv_heads, 16), -1, dtype=torch.int32)
    block_indexes[..., :2] = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.arange(2, dtype=torch.int32)
    mixed_out = (
        torch.empty(q.shape, dtype=torch.bfloat16)
        if _HAS_FP8_E4M3 and dtype == torch.float8_e4m3fn
        else None
    )

    if mode == "prefill":
        out, lse = msa.sparse_msa(
            q,
            k,
            v,
            plan_info,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            out=mixed_out,
            k_scale=1.25,
            v_scale=1.5,
        )
        assert lse is None
    else:
        out = msa.sparse_decode_atten_func(
            q,
            k,
            v,
            plan_info,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            out=mixed_out,
            k_scale=1.25,
            v_scale=1.5,
        )

    assert out.shape == q.shape
    if mixed_out is not None:
        assert out is mixed_out
        assert calls[0][5]["out"] is mixed_out
    assert len(calls) == 1
    assert calls[0][3] is block_indexes
    assert calls[0][5]["k_scale"] == 1.25
    assert calls[0][5]["v_scale"] == 1.5


_MAXSCORE_WORKLOADS = [
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=4,
        num_kv_heads=2,
        q_lengths=(3, 5),
        kv_lengths=(140, 65),
        kv_layout=MsaKvLayout.DENSE,
        dtype=torch.float16,
        causal=True,
        seed=301,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=4,
        num_kv_heads=2,
        q_lengths=(2, 4),
        kv_lengths=(130, 70),
        kv_layout=MsaKvLayout.PAGED,
        page_mode="reverse",
        dtype=torch.float16,
        causal=True,
        preallocated_max_score=True,
        seed=302,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=8,
        num_kv_heads=2,
        q_lengths=(33, 64),
        kv_lengths=(257, 130),
        kv_layout=MsaKvLayout.DENSE,
        dtype=torch.bfloat16,
        causal=True,
        seed=304,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=8,
        num_kv_heads=2,
        q_lengths=(33, 64),
        kv_lengths=(257, 130),
        kv_layout=MsaKvLayout.PAGED,
        page_mode="reverse",
        dtype=torch.bfloat16,
        causal=True,
        seed=304,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=1,
        num_kv_heads=1,
        q_lengths=(128,),
        kv_lengths=(256,),
        kv_layout=MsaKvLayout.PAGED,
        page_mode="reverse",
        dtype=torch.bfloat16,
        causal=True,
        seed=306,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=1,
        num_kv_heads=1,
        q_lengths=(513,),
        kv_lengths=(1537,),
        kv_layout=MsaKvLayout.PAGED,
        page_mode="reverse",
        dtype=torch.bfloat16,
        causal=True,
        preallocated_max_score=True,
        seed=309,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=1,
        num_kv_heads=1,
        q_lengths=(2, 4),
        kv_lengths=(128, 16384),
        kv_layout=MsaKvLayout.DENSE,
        dtype=torch.bfloat16,
        causal=False,
        preallocated_max_score=True,
        seed=307,
    ),
]
if _HAS_FP8_E4M3:
    _MAXSCORE_WORKLOADS.extend(
        [
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=8,
                num_kv_heads=2,
                q_lengths=(33, 64),
                kv_lengths=(257, 130),
                kv_layout=MsaKvLayout.DENSE,
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=303,
            ),
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=8,
                num_kv_heads=2,
                q_lengths=(33, 64),
                kv_lengths=(257, 130),
                kv_layout=MsaKvLayout.PAGED,
                page_mode="reverse",
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=303,
            ),
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=16,
                num_kv_heads=1,
                q_lengths=(1, 1),
                kv_lengths=(385, 130),
                kv_layout=MsaKvLayout.PAGED,
                page_mode="reverse",
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=305,
            ),
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=1,
                num_kv_heads=1,
                q_lengths=(128,),
                kv_lengths=(256,),
                kv_layout=MsaKvLayout.PAGED,
                page_mode="reverse",
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=308,
            ),
            *[
                MsaWorkload(
                    api=MsaApi.MAXSCORE,
                    num_q_heads=2,
                    num_kv_heads=2,
                    q_lengths=(513, 65),
                    kv_lengths=(8193, 1025),
                    kv_layout=layout,
                    page_mode="reverse",
                    dtype=torch.float8_e4m3fn,
                    causal=causal,
                    preallocated_max_score=True,
                    seed=4217,
                )
                for layout in (MsaKvLayout.DENSE, MsaKvLayout.PAGED)
                for causal in (False, True)
            ],
        ]
    )


def _sparse_topk_workloads() -> list[MsaWorkload]:
    workloads = [
        MsaWorkload(
            api=MsaApi.SPARSE_TOPK,
            num_q_heads=17,
            num_kv_heads=1,
            total_q=3,
            max_k_tiles=128,
            topk=topk,
            num_valid_pages=91,
            spike_indices=(19, 57),
            spike_values=(8.0, 6.0),
            invalid_after=91,
            seed=401 + topk,
        )
        for topk in (4, 8, 16)
    ]
    workloads.append(
        MsaWorkload(
            api=MsaApi.SPARSE_TOPK,
            num_q_heads=9,
            num_kv_heads=1,
            total_q=2,
            max_k_tiles=128,
            topk=8,
            num_valid_pages=126,
            dip_indices=(3, 125),
            invalid_after=126,
            force_begin_blocks=4,
            force_end_blocks=3,
            preallocated_output=True,
            seed=402,
        )
    )
    for topk, (max_k_tiles, num_valid_pages) in (
        (4, (128, 126)),
        (8, (128, 126)),
        (16, (128, 126)),
        (4, (512, 500)),
        (8, (512, 500)),
        (16, (512, 500)),
    ):
        workloads.append(
            MsaWorkload(
                api=MsaApi.SPARSE_TOPK,
                num_q_heads=3,
                num_kv_heads=1,
                total_q=2,
                max_k_tiles=max_k_tiles,
                topk=topk,
                num_valid_pages=num_valid_pages,
                dip_indices=(0, num_valid_pages - 1),
                dip_value=-1000.0,
                invalid_after=num_valid_pages,
                force_begin_blocks=1,
                force_end_blocks=1,
                force_blocks_count_in_topk=False,
                seed=403 + max_k_tiles,
            )
        )
    workloads.append(
        MsaWorkload(
            api=MsaApi.SPARSE_TOPK,
            num_q_heads=5,
            num_kv_heads=1,
            total_q=1,
            max_k_tiles=8,
            topk=8,
            num_valid_pages=5,
            score_pattern="zeros",
            seed=404,
        )
    )
    for max_k_tiles, num_valid_pages in ((128, 96), (512, 500)):
        workloads.append(
            MsaWorkload(
                api=MsaApi.SPARSE_TOPK,
                num_q_heads=3,
                num_kv_heads=1,
                total_q=4,
                max_k_tiles=max_k_tiles,
                topk=16,
                num_valid_pages=num_valid_pages,
                force_begin_blocks=1,
                force_end_blocks=1,
                force_blocks_count_in_topk=False,
                query_positions=(0, 129, 2303, 6144),
                seed=405,
            )
        )
    return workloads


_SPARSE_TOPK_WORKLOADS = _sparse_topk_workloads()


_SPARSE_FWD_WORKLOADS = [
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(64,),
        kv_lengths=(128,),
        dtype=torch.float16,
        causal=True,
        pattern="first",
        page_mode="identity",
        seed=501,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(48, 64),
        kv_lengths=(128, 192),
        dtype=torch.float16,
        causal=True,
        pattern="rolling",
        page_mode="reverse",
        seed=502,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=16,
        num_kv_heads=2,
        q_lengths=(64,),
        kv_lengths=(256,),
        dtype=torch.float16,
        causal=True,
        pattern="tail",
        page_mode="identity",
        seed=503,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=16,
        num_kv_heads=2,
        q_lengths=(64,),
        kv_lengths=(256,),
        dtype=torch.float16,
        causal=False,
        pattern="rolling",
        page_mode="reverse",
        seed=504,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(64,),
        kv_lengths=(192,),
        dtype=torch.bfloat16,
        causal=True,
        pattern="rolling",
        page_mode="identity",
        seed=505,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(1,),
        kv_lengths=(128,),
        dtype=torch.float16,
        causal=True,
        pattern="first",
        page_mode="identity",
        seed=506,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=16,
        num_kv_heads=1,
        q_lengths=(4,),
        kv_lengths=(256,),
        dtype=torch.float16,
        causal=True,
        pattern="tail",
        page_mode="reverse",
        seed=507,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=16,
        num_kv_heads=2,
        q_lengths=(2, 4),
        kv_lengths=(192, 256),
        dtype=torch.float16,
        causal=True,
        pattern="rolling",
        page_mode="identity",
        seed=508,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(1, 4),
        kv_lengths=(256, 128),
        dtype=torch.bfloat16,
        causal=True,
        pattern="rolling",
        page_mode="reverse",
        seed=509,
    ),
]


def _run_msa_workload(workload: MsaWorkload) -> None:
    operator = MsaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL

    with operator.use_mode(mode):
        try:
            inputs = operator.generate(workload)
        except UnsupportedMsaWorkload as error:
            pytest.skip(str(error))
        reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
        outputs = operator.call(inputs)
        raise_complete_if_dry_run()
        if reference is not None:
            operator.verify(inputs, outputs, reference)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", _MAXSCORE_WORKLOADS, ids=str)
@torch.inference_mode()
def test_msa_output_maxscore_matches_reference(workload: MsaWorkload) -> None:
    _run_msa_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.skipif(not _HAS_FP8_E4M3, reason="FP8 E4M3 unavailable")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["dense", "batched2d", "flat"])
@pytest.mark.parametrize(
    "q_lens,k_lens,offsets,ratio,dim",
    [
        ([1, 73], [0, 129], [-1, 50], 4, 128),
        ([0, 513], [385, 131], [-10, 18], 1, 128),
        ([513], [385], [-3], 1, 64),
        ([513], [385], [-3], 1, 256),
    ],
)
def test_msa_maxscore_ragged_tails_and_empty_sequences(
    causal, layout, q_lens, k_lens, offsets, ratio, dim
):
    from mate.jit import msa_ops

    generator = torch.Generator().manual_seed(4217)
    dtype = torch.float8_e4m3fn
    heads = 2
    q = (
        torch.randint(
            -2, 3, (sum(q_lens), heads * ratio, dim), generator=generator
        ).float()
        * 0.25
    ).to(dtype)
    page_counts = [(length + 127) // 128 for length in k_lens]
    table = torch.full((len(q_lens), max(page_counts)), -1, dtype=torch.int32)
    mapping = torch.randperm(sum(page_counts), generator=generator).to(torch.int32)
    page_cu = torch.tensor([0] + page_counts, dtype=torch.int32).cumsum(0).int()
    for batch, count in enumerate(page_counts):
        table[batch, :count] = mapping[page_cu[batch] : page_cu[batch + 1]]
    shape = (
        (sum(k_lens), heads, dim)
        if layout == "dense"
        else (sum(page_counts), 128, heads, dim)
    )
    k = torch.randint(-2, 3, shape, generator=generator).to(dtype)
    expected = MsaOperator._dense_msa_output_maxscore_reference(
        q.cpu().float(),
        k.cpu().float() if layout == "dense" else k.cpu().float().permute(0, 2, 1, 3),
        torch.tensor(q_lens),
        torch.tensor(k_lens),
        heads,
        msa_ops._maxscore_k_tiles(max(k_lens), 128),
        causal=causal,
        qo_offset=torch.tensor(offsets),
        page_table=None if layout == "dense" else table,
        page_size=128,
    )
    table_arg = None if layout == "dense" else (mapping if layout == "flat" else table)
    backing = torch.full((expected.numel() + 128,), 1234.5, device="musa")
    out = backing[64:-64].view(expected.shape)
    qd, kd = q.to("musa"), k.to("musa")
    cu_q = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0).int().to("musa")
    cu_k = torch.tensor([0] + k_lens, dtype=torch.int32).cumsum(0).int().to("musa")
    for _ in range(2):
        out.fill_(float("nan"))
        actual = msa_ops._msa_maxscore(
            qd,
            kd,
            cu_q,
            cu_k,
            torch.tensor(offsets, dtype=torch.int32, device="musa"),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            causal=causal,
            page_table=table_arg.to("musa") if table_arg is not None else None,
            kv_page_indptr=page_cu.to("musa") if layout == "flat" else None,
            max_score=out,
        )
        assert actual is out
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert (backing[:64] == 1234.5).all() and (backing[-64:] == 1234.5).all()


@supported_musa_compute_capability([31])
@pytest.mark.skipif(not _HAS_FP8_E4M3, reason="FP8 E4M3 unavailable")
@pytest.mark.parametrize("causal,heads", [(True, 2), (False, 3)])
def test_msa_maxscore_metadata_follows_head_then_q_tile_order(causal, heads):
    from mate.jit import msa_ops

    options = dict(
        is_paged_kv=False,
        causal=causal,
        head_dim=128,
        head_ratio=1,
        max_seqlen_q=513,
        has_metadata=True,
    )
    config = msa_ops.make_msa_maxscore_config(torch.float8_e4m3fn, **options)
    spec = msa_ops.gen_msa_maxscore_spec(torch.float8_e4m3fn, **options)
    module = spec.build_and_load()
    q_lens, k_lens, offsets = [513, 65], [8193, 131073], [33, 400]
    cu_q = torch.tensor([0, 513, 578], dtype=torch.int32, device="musa")
    cu_k = torch.tensor([0, 8193, 139266], dtype=torch.int32, device="musa")
    num_mps = min(32, msa_ops.resolve_num_mps(cu_q.device, None))
    metadata = torch.empty((num_mps, 2), dtype=torch.int32, device="musa")
    module.get_function(f"{spec.name}_metadata")(
        cu_q,
        cu_k,
        torch.tensor(offsets, dtype=torch.int32, device="musa"),
        metadata,
        513,
        131073,
        heads,
    )
    q_tokens = config["tile_q"] // config["head_ratio"]
    q_tiles = (513 + q_tokens - 1) // q_tokens
    costs = []
    for q_len, k_len, offset in zip(q_lens, k_lens, offsets):
        for _ in range(heads):
            for tile in range(q_tiles):
                begin = tile * q_tokens
                count = max(0, min(q_tokens, q_len - begin))
                visible = (
                    min(k_len, max(0, begin + count + offset)) if causal else k_len
                )
                costs.append((visible + 127) // 128 if count else 0)
    ctas = 3 if config["parallel_k_tiles"] else 1
    partitions = max(1, min(1025, (num_mps * ctas + len(costs) // 2) // len(costs)))
    groups = (partitions + ctas - 1) // ctas
    base_rows, extra = divmod(num_mps, groups)
    expected = []
    for group in range(groups):
        rows = base_rows + (group < extra)
        if len(costs) <= base_rows:
            expected.extend(
                [min(row, len(costs)), int(row < len(costs))] for row in range(rows)
            )
            continue
        begin, prefix = 0, 0
        for row in range(rows):
            end = len(costs)
            if row + 1 < rows and sum(costs) > 0:
                target = (sum(costs) * (row + 1) + rows - 1) // rows
                end = begin
                while end < len(costs) and prefix < target:
                    prefix += costs[end]
                    end += 1
            expected.append([begin, end - begin])
            begin = end
    torch.testing.assert_close(
        metadata.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
    )


@supported_musa_compute_capability([31])
@torch.inference_mode()
def test_msa_maxscore_runtime_offset_kv_tail() -> None:
    from mate.jit.msa_ops import _msa_maxscore

    device = torch.device("musa")
    torch.manual_seed(309)
    q = (torch.randn((128, 1, 128), dtype=torch.float16, device=device) * 0.2).to(
        torch.bfloat16
    )
    k = (torch.randn((2, 128, 1, 128), dtype=torch.float16, device=device) * 0.2).to(
        torch.bfloat16
    )
    cu_q = torch.tensor([0, 128], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 130], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([100], dtype=torch.int32, device=device)
    page_table = torch.tensor([[1, 0]], dtype=torch.int32, device=device)
    actual = _msa_maxscore(
        q,
        k,
        cu_q,
        cu_k,
        qo_offset,
        max_seqlen_q=128,
        max_seqlen_k=256,
        causal=True,
        page_table=page_table,
        max_score=torch.full((128, 1, 128), 7.0, dtype=torch.float32, device=device),
    )

    logical_k = torch.cat((k[1, :, 0], k[0, :2, 0]), dim=0).float()
    scores = q[:, 0].float() @ logical_k.transpose(0, 1)
    valid = torch.arange(130, device=device).view(1, -1) <= (
        torch.arange(128, device=device).view(-1, 1) + 100
    )
    scores.masked_fill_(~valid, -torch.inf)
    expected = torch.full_like(actual, -torch.inf)
    expected[:, 0, 0] = scores[:, :128].amax(dim=1)
    expected[:, 0, 1] = scores[:, 128:].amax(dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@supported_musa_compute_capability([31])
@torch.inference_mode()
def test_msa_maxscore_reuses_schedule_across_k_barrier_wraps() -> None:
    from mate.jit import msa_ops

    device = torch.device("musa")
    page_size = 128
    num_pages = 1024
    q_len = 512
    q = torch.ones((q_len, 1, 128), dtype=torch.bfloat16, device=device)
    page_values = (((torch.arange(num_pages) % 32) + 1).float() / 32).to(
        device=device, dtype=torch.bfloat16
    )
    k = page_values[:, None, None, None].expand(-1, page_size, 1, 128).contiguous()
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, num_pages * page_size], dtype=torch.int32, device=device)
    qo_offset = torch.zeros(1, dtype=torch.int32, device=device)
    page_table = torch.arange(
        num_pages - 1, -1, -1, dtype=torch.int32, device=device
    ).unsqueeze(0)
    schedule = torch.empty(
        (msa_ops.resolve_num_mps(device, None), 2), dtype=torch.int32, device=device
    )

    outputs = [
        msa_ops._msa_maxscore(
            q,
            k,
            cu_q,
            cu_k,
            qo_offset,
            max_seqlen_q=q_len,
            max_seqlen_k=num_pages * page_size,
            causal=False,
            page_table=page_table,
            schedule_metadata=schedule,
        )
    ]
    for _ in range(31):
        outputs.append(
            msa_ops._msa_maxscore(
                q,
                k,
                cu_q,
                cu_k,
                qo_offset,
                max_seqlen_q=q_len,
                max_seqlen_k=num_pages * page_size,
                causal=False,
                page_table=page_table,
                schedule_metadata=schedule,
                schedule_metadata_ready=True,
            )
        )
    torch.musa.synchronize()

    expected_pages = page_values.flip(0).float() * 128
    expected = torch.full_like(outputs[0], -torch.inf)
    expected[:, :, :num_pages] = expected_pages.view(1, 1, num_pages)
    for actual in outputs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@supported_musa_compute_capability([31])
@torch.inference_mode()
def test_msa_maxscore_flat_page_indices() -> None:
    from mate.jit.msa_ops import _msa_maxscore

    device = torch.device("musa")
    torch.manual_seed(310)
    q = (torch.randn((2, 1, 128), dtype=torch.float16, device=device) * 0.2).to(
        torch.bfloat16
    )
    k = (torch.randn((3, 128, 1, 128), dtype=torch.float16, device=device) * 0.2).to(
        torch.bfloat16
    )
    cu_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 128, 256], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([127, 127], dtype=torch.int32, device=device)
    page_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    actual = _msa_maxscore(
        q,
        k,
        cu_q,
        cu_k,
        qo_offset,
        max_seqlen_q=1,
        max_seqlen_k=128,
        causal=True,
        page_table=page_indices,
        kv_page_indptr=page_indptr,
    )

    expected = torch.full_like(actual, -torch.inf)
    expected[0, 0, 0] = (q[0, 0].float() @ k[2, :, 0].float().T).amax()
    expected[1, 0, 0] = (q[1, 0].float() @ k[0, :, 0].float().T).amax()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", _SPARSE_TOPK_WORKLOADS, ids=str)
@torch.inference_mode()
def test_msa_sparse_topk_select_matches_reference(workload: MsaWorkload) -> None:
    _run_msa_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "num_pages,topk",
    [
        (64, 4),
        (255, 8),
        (256, 16),
        (257, 16),
        (511, 8),
        (512, 16),
        (513, 4),
        (1024, 16),
        (2048, 8),
        (4096, 16),
    ],
)
@pytest.mark.parametrize("count_in_topk", [False, True])
def test_msa_sparse_topk_fp32_order_ties_and_rescan(num_pages, topk, count_in_topk):
    index = torch.arange(num_pages)
    low_bits = torch.ones(num_pages) + (index % 4).float() * 2**-23
    rescan = torch.full((num_pages,), -10.0)
    ordinary_topk = topk - 2 if count_in_topk else topk
    # Previously selected high scores must not be counted again during rescan.
    rescan[1:ordinary_topk] = 2.0
    end = min(ordinary_topk + 129, num_pages - 1)
    rescan[ordinary_topk:end] = 1.0 + (4 + index[ordinary_topk:end] % 4) * 2**-23
    rescan[end:-1] = 1.0
    tiny = torch.tensor([0.0, -0.0, -(2**-26), 2**-26, -(2**-25), 2**-25])
    patterns = torch.stack(
        [
            torch.zeros(num_pages),
            low_bits,
            rescan,
            tiny[index % len(tiny)],
            torch.full((num_pages,), float("inf")),
        ]
    )
    valid_counts = [0, topk - 1, topk + 1, num_pages]
    scores = patterns[:, None, :].expand(-1, 2, -1).clone()
    scores[:, 1] = -scores[:, 1]
    scores = scores.repeat(len(valid_counts), 1, 1)
    positions = torch.tensor(
        [count * 128 - 1 for count in valid_counts for _ in patterns],
        dtype=torch.int64,
    )
    for row, position in enumerate(positions.tolist()):
        count = max(0, position // 128 + 1)
        scores[row, :, count:] = float("nan")
        if count:
            scores[row, :, [0, count - 1]] = float("nan")
    options = dict(
        topk=topk,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=count_in_topk,
    )
    expected = MsaOperator._sparse_topk_select_reference(
        scores, query_positions=positions, **options
    )
    score_backing = torch.full((scores.numel() + 65,), float("nan"), device="musa")
    device_scores = score_backing[1:-64].view(scores.shape)
    device_scores.copy_(scores)
    backing = torch.full(
        (expected.numel() + 128,), -7319, dtype=torch.int32, device="musa"
    )
    output = backing[64:-64].view(expected.shape)
    for _ in range(2):
        output.fill_(-9321)
        actual = msa.sparse_topk_select(
            device_scores,
            query_positions=positions.to("musa"),
            output=output,
            **options,
        )
        assert actual is output
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert (backing[:64] == -7319).all() and (backing[-64:] == -7319).all()
    torch.testing.assert_close(
        device_scores.cpu(), scores, rtol=0, atol=0, equal_nan=True
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", _SPARSE_FWD_WORKLOADS, ids=str)
@torch.inference_mode()
def test_msa_sparse_forward_matches_reference(workload: MsaWorkload) -> None:
    _run_msa_workload(workload)
