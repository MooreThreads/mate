import argparse
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_musa  # noqa: F401
import torch.distributed as dist

import mate.mega_moe as mega_moe_module
from mate.mega_moe import (
    fp8_fp8_mega_moe,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
)
from mate.testing import supported_musa_compute_capability
from utils import init_dist


def public_replay_worker(
    local_rank: int, num_processes: int, args: argparse.Namespace
) -> None:
    rank, world_size, group = init_dist(local_rank, num_processes)
    replay_tokens = tuple(args.replay_tokens)
    max_active_tokens = max(replay_tokens)
    max_tokens = max(256, max_active_tokens)
    num_experts = args.experts_per_rank * world_size
    experts_per_rank = num_experts // world_size

    sym = get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        max_tokens,
        args.topk,
        args.hidden,
        args.intermediate_hidden,
        activation=getattr(args, "activation", "swiglu"),
        prefill_pool_capacity_ratio=getattr(args, "prefill_pool_capacity_ratio", 1.0),
        prefill_l1_ring=getattr(args, "prefill_l1_ring", False),
    )
    token_value = torch.tensor(
        0.5 + rank * 0.01, dtype=torch.float32, device="musa"
    ).to(torch.float8_e4m3fn)
    sym.x[:max_active_tokens].copy_(token_value.expand(max_active_tokens, args.hidden))
    sym.x_sf[:max_active_tokens].fill_(1.0)
    token_idx = torch.arange(max_active_tokens, dtype=torch.int64, device="musa")
    route_idx = torch.arange(args.topk, dtype=torch.int64, device="musa")
    sym.topk_idx[:max_active_tokens].copy_(
        (
            rank * max_active_tokens * args.topk
            + token_idx[:, None] * args.topk
            + route_idx[None, :]
        )
        % num_experts
    )
    sym.topk_weights[:max_active_tokens].fill_(1.0 / args.topk)

    def prepare_inputs(tokens: int) -> None:
        token_idx = torch.arange(tokens, dtype=torch.int64, device="musa")
        route_idx = torch.arange(args.topk, dtype=torch.int64, device="musa")
        sym.x[:tokens].copy_(token_value.expand(tokens, args.hidden))
        sym.x_sf[:tokens].fill_(1.0)
        sym.topk_idx[:tokens].copy_(
            (
                rank * tokens * args.topk
                + token_idx[:, None] * args.topk
                + route_idx[None, :]
            )
            % num_experts
        )
        sym.topk_weights[:tokens].fill_(1.0 / args.topk)

    logical_l1 = (
        torch.full(
            (experts_per_rank, args.intermediate_hidden * 2, args.hidden),
            0.25,
            dtype=torch.float8_e4m3fn,
            device="musa",
        ),
        torch.full(
            (
                experts_per_rank,
                (args.intermediate_hidden * 2) // 128,
                args.hidden // 128,
            ),
            0.01,
            dtype=torch.float32,
            device="musa",
        ),
    )
    logical_l2 = (
        torch.full(
            (experts_per_rank, args.hidden, args.intermediate_hidden),
            0.125,
            dtype=torch.float8_e4m3fn,
            device="musa",
        ),
        torch.full(
            (
                experts_per_rank,
                args.hidden // 128,
                args.intermediate_hidden // 128,
            ),
            0.02,
            dtype=torch.float32,
            device="musa",
        ),
    )
    l1_weights, l2_weights = transform_weights_for_mega_moe(logical_l1, logical_l2)
    outputs = {
        tokens: torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device="musa")
        for tokens in replay_tokens
    }

    references = {}
    for tokens in replay_tokens:
        prepare_inputs(tokens)
        fp8_fp8_mega_moe(
            outputs[tokens],
            l1_weights,
            l2_weights,
            sym,
            activation=getattr(args, "activation", "swiglu"),
            activation_clamp=getattr(args, "activation_clamp", None),
            activation_beta=getattr(args, "activation_beta", None),
            activation_linear_beta=getattr(args, "activation_linear_beta", None),
        )
        torch.musa.synchronize()
        dist.barrier(group=group)
        references[tokens] = outputs[tokens].clone()
    torch.musa.synchronize()

    for replay_idx in range(args.public_replay_count):
        tokens = replay_tokens[replay_idx % len(replay_tokens)]
        prepare_inputs(tokens)
        fp8_fp8_mega_moe(
            outputs[tokens],
            l1_weights,
            l2_weights,
            sym,
            activation=getattr(args, "activation", "swiglu"),
            activation_clamp=getattr(args, "activation_clamp", None),
            activation_beta=getattr(args, "activation_beta", None),
            activation_linear_beta=getattr(args, "activation_linear_beta", None),
        )
        torch.musa.synchronize()
        dist.barrier(group=group)

    dist.barrier(group=group)
    for tokens in replay_tokens:
        if not torch.equal(outputs[tokens], references[tokens]):
            diff = (outputs[tokens].float() - references[tokens].float()).abs()
            raise AssertionError(
                f"rank={rank} tokens={tokens} changed after alternating public replay "
                f"mismatches={int(torch.count_nonzero(diff).item())} "
                f"max_abs={float(diff.max().item())}"
            )
    if rank == 0:
        print(
            f"UNIFIED_PUBLIC_REPLAY_PASS count={args.public_replay_count} "
            f"tokens={','.join(map(str, replay_tokens))}",
            flush=True,
        )

    sym.destroy()
    dist.destroy_process_group()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def all_gather(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world = dist.get_world_size(group)
    out = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor.contiguous(), group=group)
    return torch.stack(out, dim=0)


def logical_l1_expert(e: int, hidden: int, intermediate: int):
    """Expert-dependent but 128x128 block-constant logical L1 weight/scale."""
    n = torch.arange((intermediate * 2) // 128, dtype=torch.float32, device="musa")
    k = torch.arange(hidden // 128, dtype=torch.float32, device="musa")
    block = (0.02 + (e % 32) * 0.05 + n[:, None] * 0.001 + k[None, :] * 0.0005).to(
        torch.float8_e4m3fn
    )
    scale = 0.5 + (e % 16) * 0.02 + n[:, None] * 0.001 + k[None, :] * 0.0005
    weight = block.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight, scale.contiguous()


def logical_l2_expert(e: int, hidden: int, intermediate: int):
    n = torch.arange(hidden // 128, dtype=torch.float32, device="musa")
    k = torch.arange(intermediate // 128, dtype=torch.float32, device="musa")
    block = (0.02 + (e % 32) * 0.03 + n[:, None] * 0.001 + k[None, :] * 0.0005).to(
        torch.float8_e4m3fn
    )
    scale = 0.5 + (e % 16) * 0.01 + n[:, None] * 0.001 + k[None, :] * 0.0005
    weight = block.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight, scale.contiguous()


def local_logical_weights(
    rank: int, experts_per_rank: int, hidden: int, intermediate: int
):
    l1_weight, l1_scale, l2_weight, l2_scale = (
        torch.empty(
            (experts_per_rank, intermediate * 2, hidden),
            dtype=torch.float8_e4m3fn,
            device="musa",
        ),
        torch.empty(
            (experts_per_rank, (intermediate * 2) // 128, hidden // 128),
            dtype=torch.float32,
            device="musa",
        ),
        torch.empty(
            (experts_per_rank, hidden, intermediate),
            dtype=torch.float8_e4m3fn,
            device="musa",
        ),
        torch.empty(
            (experts_per_rank, hidden // 128, intermediate // 128),
            dtype=torch.float32,
            device="musa",
        ),
    )
    for i in range(experts_per_rank):
        e = rank * experts_per_rank + i
        w1, s1 = logical_l1_expert(e, hidden, intermediate)
        w2, s2 = logical_l2_expert(e, hidden, intermediate)
        l1_weight[i].copy_(w1)
        l1_scale[i].copy_(s1)
        l2_weight[i].copy_(w2)
        l2_scale[i].copy_(s2)
    return (l1_weight, l1_scale), (l2_weight, l2_scale)


def dequant_128(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)


def activate(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
    beta: float | None,
    linear_beta: float | None,
) -> torch.Tensor:
    """Mirror the direct activation and BF16 rounding used by Stage 1."""
    gate = gate.to(torch.bfloat16).float()
    if activation == "situ":
        assert beta is not None
        gate_act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        up_act = up.to(torch.bfloat16).float()
        if linear_beta is not None:
            up_act = linear_beta * torch.tanh(up_act / linear_beta)
        return (up_act * gate_act).to(torch.bfloat16)
    gate_act = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
    return up.to(torch.bfloat16) * gate_act


def torch_reference(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden: int,
    intermediate: int,
    activation: str,
    beta: float | None,
    linear_beta: float | None,
) -> torch.Tensor:
    """Split-prefill stage 1/2/3 reference for one rank's local tokens.

    ``x``/``x_sf``/``topk_idx``/``topk_weights`` come from the all-gather, so the
    token table is global; only the source rank's slice is reduced because that
    is what the kernel writes back through stage 3.
    """
    tokens, _ = x.shape
    topk = topk_idx.shape[1]
    x_block_sum = x * x_sf.repeat_interleave(128, 1)

    # One row per (token, top-k) route.
    x_rows = x_block_sum.repeat_interleave(topk, 0)
    flat_expert = topk_idx.reshape(-1)
    partials = torch.empty((tokens * topk, hidden), dtype=torch.bfloat16, device="musa")

    order = torch.argsort(flat_expert)
    sorted_expert = flat_expert[order]
    unique, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
    start = 0
    for e, count in zip(unique.tolist(), counts.tolist()):
        rows = order[start : start + count]
        start += count

        w1, s1 = logical_l1_expert(e, hidden, intermediate)
        h = x_rows[rows] @ dequant_128(w1, s1).t()

        act = activate(
            h[:, :intermediate], h[:, intermediate:], activation, beta, linear_beta
        )
        act = act.float()
        act_scale = (
            act.view(count, intermediate // 128, 128).abs().amax(dim=2).clamp_min(1e-10)
            / 448.0
        )
        act_fp8 = (
            (act / act_scale.repeat_interleave(128, 1))
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        act_dequant = act_fp8.float() * act_scale.repeat_interleave(128, 1)

        w2, s2 = logical_l2_expert(e, hidden, intermediate)
        partials[rows] = (act_dequant @ dequant_128(w2, s2).t()).to(torch.bfloat16)

    weighted = partials.float().view(tokens, topk, hidden) * topk_weights.unsqueeze(-1)
    return weighted.sum(dim=1).to(torch.bfloat16)


def prefill_reference_worker(
    local_rank: int, world_size: int, args: SimpleNamespace
) -> None:
    dist.init_process_group(
        "mccl",
        init_method=args.master_addr,
        world_size=world_size,
        rank=local_rank,
        device_id=torch.device("musa", local_rank),
    )
    torch.musa.set_device(local_rank)
    group = dist.group.WORLD

    num_experts = args.experts
    experts_per_rank = num_experts // world_size
    tokens = args.tokens
    capacity = (max(256, tokens) + 255) // 256 * 256

    sym = get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        capacity,
        args.topk,
        args.hidden,
        args.intermediate,
        activation=args.activation,
        prefill_l1_ring=args.l1_ring,
    )

    # Random per-token inputs, scales, routes and combine weights.
    torch.manual_seed(local_rank)
    sym.x[:tokens].copy_(
        (0.25 + 0.5 * torch.rand((tokens, args.hidden), device="musa")).to(
            torch.float8_e4m3fn
        )
    )
    sym.x_sf[:tokens].copy_(
        0.5 + torch.rand((tokens, args.hidden // 128), device="musa")
    )
    sym.topk_idx[:tokens].copy_(
        torch.rand((tokens, num_experts), device="musa").topk(args.topk, dim=1).indices
    )
    weights = torch.rand((tokens, args.topk), device="musa")
    sym.topk_weights[:tokens].copy_(weights / weights.sum(dim=1, keepdim=True))

    logical_weights = local_logical_weights(
        local_rank, experts_per_rank, args.hidden, args.intermediate
    )
    l1_weights, l2_weights = transform_weights_for_mega_moe(*logical_weights)

    y = torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device="musa")
    fp8_fp8_mega_moe(
        y,
        l1_weights,
        l2_weights,
        sym,
        activation=args.activation,
        activation_beta=args.activation_beta,
        activation_linear_beta=args.activation_linear_beta,
    )
    torch.musa.synchronize()
    dist.barrier(group=group)

    if not args.no_check:
        # All-gather every rank's tokens/routing so the reference works on the
        # global token table instead of relying on the kernel's dispatch.
        global_x = all_gather(sym.x[:tokens].float(), group)
        global_sf = all_gather(sym.x_sf[:tokens], group)
        global_idx = all_gather(sym.topk_idx[:tokens], group)
        global_weights = all_gather(sym.topk_weights[:tokens], group)
        reference = torch_reference(
            global_x[local_rank],
            global_sf[local_rank],
            global_idx[local_rank],
            global_weights[local_rank],
            args.hidden,
            args.intermediate,
            args.activation,
            args.activation_beta,
            args.activation_linear_beta,
        )
        diff = (y.float() - reference.float()).abs()
        reference_f = reference.float()
        allowed = args.atol + args.rtol * reference_f.abs()
        max_abs = float(diff.max())
        max_rel = float((diff / reference_f.abs().clamp_min(1.0)).max())
        violation = float((diff - allowed).clamp_min(0.0).max())
        bad = int((diff > allowed).sum().item())
        stats = torch.tensor(
            [max_abs, max_rel, violation, float(bad)],
            dtype=torch.float32,
            device="musa",
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=group)
        passed = float(stats[2]) == 0.0
        if local_rank == 0:
            print(
                f"reference vs kernel: max_abs={float(stats[0]):.4f} "
                f"max_rel={float(stats[1]):.4e} "
                f"violation={float(stats[2]):.4f} bad={int(stats[3])} "
                f"(atol={args.atol} rtol={args.rtol}) -> "
                f"{'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        if not passed:
            raise AssertionError("MegaMoE output does not match the torch reference")

    if local_rank == 0:
        print(
            f"MegaMoE prefill OK: ranks={world_size} experts={num_experts} "
            f"topk={args.topk} hidden={args.hidden} "
            f"intermediate={args.intermediate} tokens={tokens} "
            f"l1_ring_tokens={sym.prefill_l1_ring_tokens} "
            f"buffer={sym.num_bytes / 2**30:.2f} GiB/rank y={tuple(y.shape)}",
            flush=True,
        )

    sym.destroy()
    dist.destroy_process_group()


@supported_musa_compute_capability([31])
def test_prefill_chunked_weight_reference_layout() -> None:
    pytest.importorskip("torch_musa")
    experts_per_rank = 2
    hidden = 256
    intermediate_hidden = 256
    expert_idx = torch.arange(experts_per_rank, dtype=torch.float32, device="musa")
    n_block = torch.arange(
        (intermediate_hidden * 2) // 128, dtype=torch.float32, device="musa"
    )
    k_block = torch.arange(hidden // 128, dtype=torch.float32, device="musa")
    logical_weight = (
        0.25
        + expert_idx[:, None, None] * 0.03125
        + n_block[None, :, None] * 0.015625
        + k_block[None, None, :] * 0.0078125
    ).to(torch.float8_e4m3fn)
    logical_weight = logical_weight.repeat_interleave(128, dim=1).repeat_interleave(
        128, dim=2
    )
    logical_scale = (
        0.01
        + expert_idx[:, None, None] * 0.001
        + n_block[None, :, None] * 0.00025
        + k_block[None, None, :] * 0.0001
    ).contiguous()
    physical_weight, physical_scale = transform_weights_for_mega_moe(
        (logical_weight, logical_scale)
    )

    chunks = physical_weight.view(
        experts_per_rank, (intermediate_hidden * 2) // 64, 64, hidden
    )
    gate_weight = chunks[:, 0::2].reshape(experts_per_rank, intermediate_hidden, hidden)
    up_weight = chunks[:, 1::2].reshape(experts_per_rank, intermediate_hidden, hidden)

    def _fp8_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left.view(torch.uint8), right.view(torch.uint8))

    assert _fp8_equal(gate_weight, logical_weight[:, :intermediate_hidden])
    assert _fp8_equal(up_weight, logical_weight[:, intermediate_hidden:])
    half_blocks = logical_scale.size(1) // 2
    gate_scale = logical_scale[:, :half_blocks]
    up_scale = logical_scale[:, half_blocks:]
    assert torch.equal(physical_scale[:, 0::4], gate_scale)
    assert torch.equal(physical_scale[:, 1::4], up_scale)
    assert torch.equal(physical_scale[:, 2::4], gate_scale)
    assert torch.equal(physical_scale[:, 3::4], up_scale)

    l2_n_block = torch.arange(hidden // 128, dtype=torch.float32, device="musa")
    l2_k_block = torch.arange(
        intermediate_hidden // 128, dtype=torch.float32, device="musa"
    )
    l2_weight = (
        0.125
        + expert_idx[:, None, None] * 0.03125
        + l2_n_block[None, :, None] * 0.0078125
        + l2_k_block[None, None, :] * 0.00390625
    ).to(torch.float8_e4m3fn)
    l2_weight = l2_weight.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    l2_scale = (
        (0.02 + expert_idx[:, None, None] * 0.001)
        .expand(
            experts_per_rank,
            hidden // 128,
            intermediate_hidden // 128,
        )
        .contiguous()
    )
    (got_l1_weight, got_l1_scale), (got_l2_weight, got_l2_scale) = (
        transform_weights_for_mega_moe(
            (logical_weight, logical_scale), (l2_weight, l2_scale)
        )
    )
    assert _fp8_equal(got_l1_weight, physical_weight)
    assert torch.equal(got_l1_scale, physical_scale)
    assert _fp8_equal(got_l2_weight, l2_weight)
    assert torch.equal(got_l2_scale, l2_scale)


def test_prefill_pool_capacity_ratio_resolution() -> None:
    ratio, rows = mega_moe_module._resolve_prefill_pool_capacity_rows(
        8, 896, 8192, 16, 0.8
    )
    assert ratio == 0.8
    assert rows == 861_952
    assert mega_moe_module._resolve_prefill_pool_capacity_rows(
        8, 896, 8192, 16, 1.0
    ) == (1.0, 1_077_248)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.01, float("inf"), float("nan")])
def test_prefill_pool_capacity_ratio_rejects_invalid_values(ratio: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        mega_moe_module._resolve_prefill_pool_capacity_rows(8, 896, 8192, 16, ratio)


def test_prefill_memory_reduction_modes_are_mutually_exclusive() -> None:
    validate = mega_moe_module._validate_prefill_memory_options
    validate(1.0, False)
    validate(1.0, True)
    validate(0.8, False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate(0.8, True)


def test_prefill_l1_ring_auto_capacity() -> None:
    resolve = mega_moe_module._resolve_prefill_l1_ring_tokens
    # Kimi K3 8K: 8192 * 16 routes -> 1024 blocks of ring headroom, well above the
    # 256-block single-expert safety floor.
    assert resolve(False, 8, 16, 8192, 1_077_248) == 0
    assert resolve(True, 8, 16, 8192, 1_077_248) == 262_144
    # Low top-k: the throughput term is small, the expert floor takes over.
    assert resolve(True, 8, 1, 8192, 1_077_248) == 65_536
    # Pool smaller than the expert floor: recycling is disabled, not unsafe.
    assert resolve(True, 8, 16, 8192, 8192) == 0
    assert resolve(True, 8, 16, 8192, 4096) == 0


@pytest.mark.parametrize("mode", ["full", "dispatch_only", "compute_only", "invalid"])
@pytest.mark.parametrize("ring_tokens", [0, 512, 4096])
def test_prefill_jit_profile_ring_guard(monkeypatch, mode, ring_tokens) -> None:
    import mate.jit.mega_moe_prefill as jit

    # Even when ring capacity equals the pool, split mode requires opt-out.
    pool_tokens = 4096
    config = jit.PrefillMegaMoEKernelConfig(
        8,
        128,
        256,
        1,
        512,
        1024,
        60,
        num_max_pool_tokens=pool_tokens,
        num_l1_ring_tokens=ring_tokens,
    )

    def after_validation(config):
        raise RuntimeError("profile validation passed")

    monkeypatch.setattr(jit, "_num_owner_ctas", after_validation)
    if mode == "invalid":
        error, message = ValueError, "invalid Stage 1 profile mode"
    elif ring_tokens and mode != "full":
        error, message = ValueError, "prefill_l1_ring=False"
    else:
        error, message = RuntimeError, "profile validation passed"
    with pytest.raises(error, match=message):
        jit.gen_prefill_stage1_spec(config, profile_mode=mode)


@pytest.mark.parametrize(
    "factory,template,header",
    [
        (
            "mate.jit.mega_moe:gen_fp8_fp8_mega_moe_stage1_spec",
            "fp8_fp8_decode_stage1_kern.j2",
            "decode/mp31_decode_stage1.hpp",
        ),
        (
            "mate.jit.mega_moe_prefill:gen_prefill_stage1_spec",
            "fp8_fp8_prefill_stage1_kern.j2",
            "prefill/mp31_prefill_stage1.hpp",
        ),
    ],
)
def test_mega_moe_jit_dependencies_follow_shared_builder(factory, template, header):
    from mate.testing.resolver.jit_spec import JitSpecDependencyAnalyzer

    repository = Path(__file__).resolve().parents[1]
    result = JitSpecDependencyAnalyzer(repository).analyze(factory)
    assert Path("csrc/templates/mega_moe") / template in result.resolved.templates
    assert Path("include/mate/mega_moe") / header in result.resolved.headers
    assert "mate/jit/mega_moe_common.py" in result.python.files
    assert not result.python.unresolved


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    "experts,capacity,topk,hidden,intermediate,ring_blocks,rejected",
    [
        pytest.param(128, 256, 1, 512, 3072, 2, True, id="top1-deadlock"),
        pytest.param(128, 256, 1, 512, 3072, 8, False, id="exact-expert-bound"),
        pytest.param(128, 256, 1, 512, 3072, 0, False, id="no-ring"),
        pytest.param(896, 8192, 16, 3584, 3072, 1024, False, id="k3-8k"),
        pytest.param(256, 8192, 8, 7168, 2048, 512, False, id="deepseek-8k"),
    ],
)
def test_prefill_l1_ring_kernel_compile_guard(
    experts, capacity, topk, hidden, intermediate, ring_blocks, rejected
) -> None:
    """Compile actual kernels; an unsafe configuration must never be launched."""
    pytest.importorskip("torch_musa")
    from mate.jit.mega_moe_prefill import (
        PrefillMegaMoEKernelConfig,
        gen_prefill_stage1_spec,
    )

    config = PrefillMegaMoEKernelConfig(
        8,
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        60,
        num_l1_ring_tokens=ring_blocks * 256,
    )
    spec = gen_prefill_stage1_spec(config)
    if rejected:
        with pytest.raises(RuntimeError, match="Unsafe L1 ring capacity"):
            spec.build()
    else:
        assert spec.build().is_file()


@supported_musa_compute_capability([31])
def test_prefill_l1_ring_shrinks_unified_buffer() -> None:
    pytest.importorskip("torch_musa")
    base = (8, 896, 8192, 16, 3584, 3072, 1_077_248)
    plain, _ = mega_moe_module._get_unified_buffer_layout(*base, 0)
    # 256 blocks is the single-expert safety floor for this shape.
    ringed, _ = mega_moe_module._get_unified_buffer_layout(*base, 65_536)
    assert ringed < plain


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("activation", ["swiglu", "situ"])
def test_public_replay_shared_buffer_decode_prefill(activation: str) -> None:
    """Reuse one symmetric buffer while crossing the 128-token dispatch split."""
    pytest.importorskip("torch_musa")
    requested = int(os.getenv("MATE_MEGA_MOE_PREFILL_TEST_PROCESSES", "8"))
    if requested != 8:
        pytest.skip("shared decode/prefill replay uses the 8-rank production launch")
    if requested > int(torch.musa.device_count()):
        pytest.skip("shared decode/prefill replay requires 8 MUSA devices")
    args = argparse.Namespace(
        experts_per_rank=8,
        topk=4,
        hidden=512,
        intermediate_hidden=512,
        activation=activation,
        activation_clamp=10.0 if activation == "swiglu" else None,
        activation_beta=4.0 if activation == "situ" else None,
        activation_linear_beta=25.0 if activation == "situ" else None,
        public_replay_count=4,
        replay_tokens=[128, 129],
    )
    torch.multiprocessing.spawn(
        public_replay_worker,
        args=(requested, args),
        nprocs=requested,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("tokens", [4096, 8192], ids=["4k", "8k"])
@pytest.mark.parametrize(
    "activation,l1_ring",
    [("swiglu", False), ("situ", True)],
    ids=["swiglu-full-pool", "situ-ring"],
)
def test_mega_moe_prefill_matches_torch_reference(tokens, activation, l1_ring) -> None:
    pytest.importorskip("torch_musa")
    available = int(torch.musa.device_count())
    requested = int(os.getenv("MATE_MEGA_MOE_PREFILL_TEST_PROCESSES", "8"))
    if requested != 8:
        pytest.skip("Kimi K3 MegaMoE prefill accuracy regression requires 8 ranks")
    if requested > available:
        pytest.skip(f"requested {requested} MUSA devices, found {available}")

    port = free_port()
    args = SimpleNamespace(
        tokens=tokens,
        experts=896,
        topk=16,
        hidden=3584,
        intermediate=3072,
        activation=activation,
        activation_beta=4.0 if activation == "situ" else None,
        activation_linear_beta=25.0 if activation == "situ" else None,
        l1_ring=l1_ring,
        no_check=False,
        atol=0.5,
        rtol=1e-2,
        master_addr=f"tcp://127.0.0.1:{port}",
    )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))

    torch.multiprocessing.spawn(
        prefill_reference_worker, args=(requested, args), nprocs=requested
    )


def _replay_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--experts-per-rank", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate-hidden", type=int, default=256)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--activation-beta", type=float)
    parser.add_argument("--activation-linear-beta", type=float)
    parser.add_argument("--prefill-pool-capacity-ratio", type=float, default=1.0)
    parser.add_argument(
        "--prefill-l1-ring",
        action="store_true",
        dest="prefill_l1_ring",
        default=False,
    )
    parser.add_argument("--public-replay-count", type=int, default=2)
    parser.add_argument("--replay-tokens", default="128,129")
    cli_args = parser.parse_args()
    if cli_args.activation == "situ":
        cli_args.activation_clamp = None
        if cli_args.activation_beta is None:
            parser.error("--activation-beta is required for --activation situ")
    if cli_args.public_replay_count <= 0:
        parser.error("--public-replay-count must be positive")
    cli_args.replay_tokens = [
        int(value) for value in cli_args.replay_tokens.split(",") if value.strip()
    ]
    if not cli_args.replay_tokens:
        parser.error("--replay-tokens must contain at least one token count")
    torch.multiprocessing.spawn(
        public_replay_worker,
        args=(cli_args.num_processes, cli_args),
        nprocs=cli_args.num_processes,
    )


if __name__ == "__main__":
    _replay_main()
