"""Benchmark MegaMoE prefill through fp8_fp8_mega_moe at 4K and 8K tokens."""

from __future__ import annotations

import argparse
import os
import statistics
from typing import Callable

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401

from mate.mega_moe import (
    fp8_fp8_mega_moe,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
)


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * quantile)]


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.9),
    }


def _max_rank(value: float, group: dist.ProcessGroup) -> float:
    tensor = torch.tensor(value, dtype=torch.float32, device="musa")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _measure_stage(fn, group: dist.ProcessGroup) -> float:
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    dist.barrier(group=group)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return _max_rank(float(start.elapsed_time(end)), group)


def _profiler_event_time_range_us(event) -> tuple[float, float] | None:
    time_range = getattr(event, "time_range", None)
    if time_range is not None:
        start = getattr(time_range, "start", None)
        if start is not None:
            elapsed = (
                time_range.elapsed_us()
                if callable(getattr(time_range, "elapsed_us", None))
                else getattr(time_range, "elapsed_us", None)
            )
            if elapsed is not None:
                return float(start), float(elapsed)

    start = getattr(event, "start_us", None)
    duration = getattr(event, "duration_us", None)
    if start is not None and duration is not None:
        return float(start), float(duration)
    return None


def _extract_profiler_kernel_events(prof, kernel_name: str) -> list[dict[str, float]]:
    kernel_events = []
    for event in prof.events():
        if kernel_name not in getattr(event, "name", ""):
            continue
        time_range = _profiler_event_time_range_us(event)
        if time_range is None:
            continue
        start_us, duration_us = time_range
        kernel_events.append(
            {
                "start_us": start_us,
                "duration_us": duration_us,
                "end_us": start_us + duration_us,
            }
        )
    return sorted(kernel_events, key=lambda item: item["start_us"])


def _average_max_rank_kernel_duration(
    kernel_events: list[dict[str, float]], group: dist.ProcessGroup
) -> float:
    rank_kernel_events = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(rank_kernel_events, kernel_events, group=group)
    assert all(rank_kernel_events)
    num_iters = min(len(events) for events in rank_kernel_events)
    assert num_iters > 0

    max_rank_duration_us = 0.0
    for iter_idx in range(num_iters):
        max_rank_duration_us += max(
            events[iter_idx]["duration_us"] for events in rank_kernel_events
        )
    return max_rank_duration_us / num_iters / 1e6


def _bench_kineto(
    fn: Callable[[], object],
    kernel_name: str,
    *,
    num_tests: int,
    suppress_kineto_output: bool,
    group: dist.ProcessGroup,
    flush_l2_bytes: int,
) -> float:
    class empty_suppress:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class suppress_stdout_stderr:
        def __enter__(self):
            self.outnull_file = open(os.devnull, "w")
            self.errnull_file = open(os.devnull, "w")
            self.old_stdout_fileno_undup = os.dup(1)
            self.old_stderr_fileno_undup = os.dup(2)
            os.dup2(self.outnull_file.fileno(), 1)
            os.dup2(self.errnull_file.fileno(), 2)
            return self

        def __exit__(self, *_):
            os.dup2(self.old_stdout_fileno_undup, 1)
            os.dup2(self.old_stderr_fileno_undup, 2)
            os.close(self.old_stdout_fileno_undup)
            os.close(self.old_stderr_fileno_undup)
            self.outnull_file.close()
            self.errnull_file.close()

    flush_l2_elems = flush_l2_bytes // 4
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.MUSA], schedule=schedule
        ) as prof:
            for _ in range(2):
                for _ in range(num_tests):
                    if flush_l2_elems > 0:
                        torch.empty(
                            flush_l2_elems, dtype=torch.int, device="musa"
                        ).zero_()
                    dist.barrier(group=group)
                    fn()
                prof.step()

    kernel_events = _extract_profiler_kernel_events(prof, kernel_name)
    if not kernel_events:
        event_names = sorted(
            {
                getattr(event, "name", "")
                for event in prof.events()
                if getattr(event, "name", "")
            }
        )
        raise RuntimeError(
            f"Profiler did not find kernel name: {kernel_name}. "
            f"Available events: {event_names[:64]}"
        )
    return _average_max_rank_kernel_duration(kernel_events, group)


def _make_routes(args: argparse.Namespace, rank: int) -> torch.Tensor:
    torch.manual_seed(rank)
    scores = torch.rand((args.tokens, args.experts), dtype=torch.float32, device="musa")
    return torch.topk(scores, args.topk, dim=-1, sorted=False).indices


def _make_weights(
    args: argparse.Namespace,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    experts_per_rank = args.experts // args.ranks
    logical_l1 = (
        torch.full(
            (experts_per_rank, args.intermediate * 2, args.hidden),
            0.01,
            dtype=torch.float8_e4m3fn,
            device="musa",
        ),
        torch.ones(
            (experts_per_rank, (args.intermediate * 2) // 128, args.hidden // 128),
            dtype=torch.float32,
            device="musa",
        ),
    )
    logical_l2 = (
        torch.full(
            (experts_per_rank, args.hidden, args.intermediate),
            0.01,
            dtype=torch.float8_e4m3fn,
            device="musa",
        ),
        torch.ones(
            (experts_per_rank, args.hidden // 128, args.intermediate // 128),
            dtype=torch.float32,
            device="musa",
        ),
    )
    return transform_weights_for_mega_moe(logical_l1, logical_l2)


def _run(
    args: argparse.Namespace, group: dist.ProcessGroup, rank: int
) -> dict[str, object]:
    routes = _make_routes(args, rank)
    sym = get_symm_buffer_for_mega_moe(
        group,
        args.experts,
        args.capacity,
        args.topk,
        args.hidden,
        args.intermediate,
        activation=args.activation,
        prefill_pool_capacity_ratio=args.prefill_pool_capacity_ratio,
        prefill_l1_ring=args.prefill_l1_ring,
    )
    try:
        sym.x[: args.tokens].fill_(1.0)
        sym.x_sf[: args.tokens].fill_(1.0)
        sym.topk_idx[: args.tokens].copy_(routes)
        sym.topk_weights[: args.tokens].fill_(1.0 / args.topk)
        l1_weights, l2_weights = _make_weights(args)
        output = torch.empty(
            (args.tokens, args.hidden), dtype=torch.bfloat16, device="musa"
        )

        def run_prefill():
            fp8_fp8_mega_moe(
                output,
                l1_weights,
                l2_weights,
                sym,
                activation=args.activation,
                activation_clamp=args.activation_clamp,
                activation_beta=(
                    args.activation_beta if args.activation == "situ" else None
                ),
                activation_linear_beta=(
                    args.activation_linear_beta if args.activation == "situ" else None
                ),
            )

        for _ in range(args.warmup):
            run_prefill()
        torch.musa.synchronize()
        samples = [_measure_stage(run_prefill, group) for _ in range(args.iters)]

        kineto_args = {
            "num_tests": args.kineto_iters,
            "suppress_kineto_output": not args.show_kineto_output,
            "group": group,
            "flush_l2_bytes": args.flush_l2_bytes,
        }
        stage1_seconds = _bench_kineto(
            run_prefill, "mp31_prefill_stage1_impl", **kineto_args
        )
        stage2_seconds = _bench_kineto(
            run_prefill, "mp31_prefill_stage2_impl", **kineto_args
        )
        stage3_seconds = _bench_kineto(
            run_prefill, "mp31_prefill_stage3_impl", **kineto_args
        )

        routed_rows = args.tokens * args.topk
        stage1_flops = 2 * routed_rows * args.hidden * (args.intermediate * 2)
        stage2_flops = 2 * routed_rows * args.intermediate * args.hidden
        stage3_bytes = (
            routed_rows * args.hidden * 2
            + routed_rows * (8 + 4)
            + args.tokens * args.hidden * 2
        )
        total_flops = stage1_flops + stage2_flops
        e2e_result = _summarize(samples)
        e2e_result["tflops"] = total_flops / (e2e_result["p50_ms"] * 1e9)
        stage1_result = {
            "time_us": stage1_seconds * 1e6,
            "tflops": stage1_flops / stage1_seconds / 1e12,
        }
        stage2_result = {
            "time_us": stage2_seconds * 1e6,
            "tflops": stage2_flops / stage2_seconds / 1e12,
        }
        stage3_result = {
            "time_us": stage3_seconds * 1e6,
            "effective_gbps": stage3_bytes / stage3_seconds / 1e9,
        }

        return {
            "tokens": args.tokens,
            "shape": {
                "ranks": args.ranks,
                "experts": args.experts,
                "topk": args.topk,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
            },
            "prefill_pool_capacity_ratio": args.prefill_pool_capacity_ratio,
            "prefill_l1_ring": args.prefill_l1_ring,
            "activation": args.activation,
            "warmup": args.warmup,
            "iters": args.iters,
            "kineto_iters": args.kineto_iters,
            "metrics": {
                "fp8_fp8_mega_moe": e2e_result,
                "prefill_stage1": stage1_result,
                "prefill_stage2": stage2_result,
                "prefill_stage3": stage3_result,
            },
            "formulas": {
                "e2e_tflops": ("(stage1_flops + stage2_flops) / e2e_time"),
                "stage1_flops": ("2 * (tokens * topk) * hidden * (2 * intermediate)"),
                "stage2_flops": ("2 * (tokens * topk) * intermediate * hidden"),
                "stage3_effective_bytes": (
                    "(tokens * topk) * (hidden * sizeof(bf16) + "
                    "sizeof(int64) + sizeof(float)) + "
                    "tokens * hidden * sizeof(bf16)"
                ),
            },
        }
    finally:
        sym.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="4096,8192")
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--experts", type=int, default=896)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--prefill-pool-capacity-ratio", type=float, default=1.0)
    parser.add_argument("--prefill-l1-ring", action="store_true")
    parser.add_argument("--activation", choices=("swiglu", "situ"), default="swiglu")
    parser.add_argument("--activation-clamp", type=float)
    parser.add_argument("--activation-beta", type=float, default=4.0)
    parser.add_argument("--activation-linear-beta", type=float, default=25.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--kineto-iters", type=int, default=5)
    parser.add_argument("--flush-l2-bytes", type=int, default=int(8e9))
    parser.add_argument("--show-kineto-output", action="store_true")
    args = parser.parse_args()
    token_values = [int(value) for value in args.tokens.split(",") if value]
    if not token_values or any(value <= 0 for value in token_values):
        parser.error("--tokens must contain positive integers")
    if max(token_values) > args.capacity or args.capacity % 256:
        parser.error("max tokens must be <= capacity and capacity must be 256-aligned")
    if args.experts % args.ranks:
        parser.error("experts must be divisible by ranks")
    if args.iters <= 0 or args.kineto_iters <= 0:
        parser.error("--iters and --kineto-iters must be positive")
    if args.flush_l2_bytes < 0:
        parser.error("--flush-l2-bytes must be non-negative")
    if int(os.environ.get("WORLD_SIZE", "0")) != args.ranks:
        parser.error(f"launch with torchrun --nproc_per_node={args.ranks}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.musa.set_device(local_rank)
    dist.init_process_group(
        "mccl",
        init_method="env://",
        rank=rank,
        world_size=args.ranks,
        device_id=torch.device("musa", local_rank),
    )
    group = dist.group.WORLD
    try:
        results = []
        for tokens in token_values:
            args.tokens = tokens
            dist.barrier(group=group)
            results.append(_run(args, group, rank))
            torch.musa.empty_cache()
        if rank == 0:
            print(
                {
                    "benchmark": "mega_moe_prefill_fp8_fp8_mega_moe",
                    "timing": {
                        "fp8_fp8_mega_moe": "MUSA events, max across ranks per sample",
                        "prefill_stages": "Kineto average max kernel duration across ranks",
                    },
                    "results": results,
                },
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
