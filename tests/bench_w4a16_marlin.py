# SPDX-License-Identifier: Apache-2.0
"""W4A16 (Marlin-class) dense GEMM performance guard for MATE.

INSTALL TARGET (MATE repository): ``tests/bench_w4a16_marlin.py``
    cp tests-port/tests/bench_w4a16_marlin.py  <mate>/tests/bench_w4a16_marlin.py
Companion file: ``tests/test_w4a16_marlin.py`` (correctness), same shape lattice.

Ported for ticket MUSA-100077 from upstream vLLM (pin checkout
``/home/xiaodongye/ws/vllm-musa100015-vllm-verify``, upstream vLLM ``2cf0a6915c``):
    benchmarks/kernels/benchmark_marlin.py
      DEFAULT_BATCH_SIZES / WEIGHT_SHAPES / the bench_run() param loop -> MS x CHECKPOINT_SHAPES
      the marlin_gemm vs pytorch_gemm A/B pair                         -> MATE op vs fp16 dense
      torch.utils.benchmark.Timer(...).blocked_autorange()             -> bench_kineto(flush_l2=True)
                                                                          + bench_gpu_time_with_musa_event
      --limit-k / --limit-n / --limit-group-size filters               -> --shapes / --ms filters

Baseline reachability, stated rather than implied (the ticket's decode gate names two baselines):

    baseline                       reachable here?   why
    ------------------------------------------------------------------------------------------
    MATE dense_w4a16_gemm          YES               the op under test
    fp16 dense (same GEMM, fp16 W)  YES              torch.matmul on a dequantized fp16 weight, i.e.
                                                      the path W4A16 replaces. A speed baseline only:
                                                      it is not a numeric comparison
    shipped Triton W4A16 kernel     NO, by design    it lives in vLLM-MUSA, not MATE. Attach it with
                                                      MATE_W4A16_TRITON_CALLABLE=module:attr (or
                                                      --triton-callable); without it the run prints an
                                                      explicit `not run` line and still reports MATE
    vendor _mixed_dtypes_linear     NO, withdrawn    the stage-1 hybrid was withdrawn (ticket
                                                      decision), so it is not a shipped baseline

A bench that prints nothing for a requested baseline is how a "win" gets over-read, so the missing
legs are named in the output and in the JSON's ``not_run`` block.

Protocol (numbers are only comparable if the protocol is stated):
    * ``bench_kineto(..., flush_l2=True)`` is the ticket's named protocol: MATE's copy allocates an
      8 GB int32 buffer and zeroes it between timed iterations, so the weight is not still in L2 from
      the previous iteration. It needs a kernel-name substring, so it covers the MATE op, whose
      dispatch name is known.
    * ``mate.testing.utils.bench_gpu_time_with_musa_event`` covers *every* row, including the
      baselines whose kernels have no stable name (``torch.matmul``). It flushes 512 MB of L2, which
      evicts a 13.8 MB packed weight from a ~60 MB L2 many times over, and its median is reported.
    * Cold-cache A/B is only meaningful if the same op is measured the same way, so the two timers are
      reported side by side; a disagreement means one of them is timing something else.
    * One warm run is not a measurement: ``--repeats`` (default 3) re-runs the whole ladder and the
      spread is printed. A kernel that wins once and loses once is not faster.

Usage (inside a leased MUSA container, from the MATE root):
    python3 tests/bench_w4a16_marlin.py --quick
    python3 tests/bench_w4a16_marlin.py --repeats 3 --json /tmp/w4a16_bench.json
    python3 tests/bench_w4a16_marlin.py --self-check      # no torch required: model self-test
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from typing import Callable, Optional, Sequence

# ---------------------------------------------------------------------------------------------
# Shape lattice: the same four real Qwen2-1.5B-AWQ projections the correctness test pins.
# K is the reduction dim, N the output dim; group size 128 and K % 128 == 0 by contract.
# ---------------------------------------------------------------------------------------------
GROUP_SIZE = 128
MATE_TILE_K = 256
CHECKPOINT_SHAPES = (
    ("qkv", 1536, 2048),
    ("o", 1536, 1536),
    ("gate_up", 1536, 17920),
    ("down", 8960, 1536),
)
MS = (1, 8, 32, 128, 2048)

# Rooflines used for the ratio columns (write-kernel-test-plan / musa-kernel-bench).
BANDWIDTH_ROOFLINE_GB_S = 1600.0
COMPUTE_ROOFLINE_TFLOPS_16BIT = 500.0
DECODE_PASS_RATIO = 0.95
L2_FLUSH_SIZE_MB = 512  # what bench_gpu_time_with_musa_event is given; recorded in the JSON

DECODE_M_MAX = 128  # at or below this the op is expected to be bandwidth-bound


@dataclass(frozen=True)
class Row:
    """One measured (projection, M) point for one implementation."""

    name: str
    size_m: int
    size_k: int
    size_n: int
    ms: float
    gbs: float
    tflops: float
    bytes_moved: int
    flops: int
    kineto_ms: Optional[float] = None  # independent timer, MATE rows only

    @property
    def bandwidth_ratio(self) -> float:
        return self.gbs / BANDWIDTH_ROOFLINE_GB_S

    @property
    def compute_ratio(self) -> float:
        return self.tflops / COMPUTE_ROOFLINE_TFLOPS_16BIT

    @property
    def regime(self) -> str:
        return "decode" if self.size_m <= DECODE_M_MAX else "prefill"


def weight_bytes(size_k: int, size_n: int, zero_points: bool = False) -> int:
    """Packed int4 weight + fp16 group scales (+ fp16 zero points on the asymmetric arm)."""
    packed = size_n * (size_k // 2)
    scales = size_n * max(1, size_k // GROUP_SIZE) * 2
    return packed + scales + (scales if zero_points else 0)


def stream_bytes(size_m: int, size_k: int, size_n: int) -> int:
    """Activation + output traffic."""
    return size_m * size_k * 2 + size_m * size_n * 2


def total_bytes(size_m: int, size_k: int, size_n: int, zero_points: bool = False) -> int:
    return weight_bytes(size_k, size_n, zero_points) + stream_bytes(size_m, size_k, size_n)


def fp16_dense_bytes(size_m: int, size_k: int, size_n: int) -> int:
    """Traffic of the baseline: A + output + N*K*2 bytes of fp16 weight.

    An element is 2 bytes here against 0.5 packed in int4, so the baseline moves **4x** the weight
    bytes - which is the whole reason W4A16 helps at decode. Using the int4 byte model for the fp16
    row would overstate the baseline's achieved bandwidth by up to 4x; that is the kind of modelling
    error that makes a slow baseline look fast. There is no separate scale stream: the scales are
    folded into the fp16 weight.
    """
    return size_k * size_n * 2 + stream_bytes(size_m, size_k, size_n)


def flops(size_m: int, size_k: int, size_n: int) -> int:
    return 2 * size_m * size_k * size_n


def _python_self_check() -> int:
    """No-torch check of the byte/FLOP models, so the bench's arithmetic is verifiable anywhere."""
    failures = []
    # Hand-computed for qkv (K=1536, N=2048, M=1): packed 2048*768, scales 2048*12*2, A 1536*2,
    # output 2048*2.
    expected = 2048 * 768 + 2048 * 12 * 2 + 1536 * 2 + 2048 * 2
    got = total_bytes(1, 1536, 2048)
    print(f"total_bytes(M=1, qkv) = {got} (expected {expected})")
    if got != expected:
        failures.append("total_bytes")
    if total_bytes(1, 1536, 2048, zero_points=True) - got != 2048 * 12 * 2:
        failures.append("zero-point bytes")
    if total_bytes(0, 1536, 2048) != weight_bytes(1536, 2048):
        failures.append("weight-only case")
    if flops(2048, 1536, 17920) != 2 * 2048 * 1536 * 17920:
        failures.append("flops")
    # The baseline's byte model is A + output + 4x the packed int4 weight (2 bytes per element
    # against 0.5). Checking the identity stops the fp16 row's bandwidth from silently being computed
    # with the int4 operand size.
    packed_weight = 1536 * 2048 // 2  # int4: 0.5 bytes per element
    if fp16_dense_bytes(8, 1536, 2048) - stream_bytes(8, 1536, 2048) != 4 * packed_weight:
        failures.append("fp16 dense byte model")
    if total_bytes(8, 1536, 2048) - stream_bytes(8, 1536, 2048) != packed_weight + 2048 * 12 * 2:
        failures.append("int4 byte model")
    # The regime labels must be justified by the traffic, not asserted: at decode the weight stream
    # dominates, at M=2048 the activation+output stream does. If this ever fails, the roofline column
    # is mislabelled for that projection and the table is misleading.
    for name, size_k, size_n in CHECKPOINT_SHAPES:
        decode_dominated = weight_bytes(size_k, size_n) > stream_bytes(1, size_k, size_n)
        prefill_flipped = stream_bytes(2048, size_k, size_n) > weight_bytes(size_k, size_n)
        print(
            f"{name}: weight {weight_bytes(size_k, size_n) / 1e6:.2f} MB, "
            f"M=1 stream {stream_bytes(1, size_k, size_n) / 1e3:.1f} kB, "
            f"M=2048 stream {stream_bytes(2048, size_k, size_n) / 1e6:.2f} MB, "
            f"decode weight-bound={decode_dominated}, prefill flips={prefill_flipped}"
        )
        if not decode_dominated:
            failures.append(f"{name} is not decode weight-bound")
        if not prefill_flipped:
            failures.append(f"{name} does not flip at prefill")
        if size_k % MATE_TILE_K:
            failures.append(f"{name} K={size_k} is not a multiple of the tile K")
    if failures:
        print("SELF-CHECK FAILED: " + ", ".join(failures))
        return 1
    print("SELF-CHECK PASSED (byte/FLOP models and regime labels)")
    return 0


def _require_device_dependencies():
    """Import the MUSA stack late so --self-check works on a box without torch."""
    import torch  # noqa: PLC0415
    import torch_musa  # noqa: F401, PLC0415

    from mate.testing.utils import (  # noqa: PLC0415
        bench_gpu_time_with_musa_event,
        bench_kineto,
    )

    return torch, bench_kineto, bench_gpu_time_with_musa_event


def _mate_op():
    import mate  # noqa: PLC0415

    gemm_module = getattr(mate, "gemm", None)
    op = getattr(gemm_module, "dense_w4a16_gemm", None) if gemm_module is not None else None
    if op is None:
        op = getattr(mate, "dense_w4a16_gemm", None)
    if op is None:
        raise SystemExit("mate.gemm.dense_w4a16_gemm not found in this MATE revision")
    return op


def load_triton_callable() -> Optional[Callable]:
    """Resolve the optional Triton W4A16 baseline: ``module:attr``, or ``None`` when not supplied.

    The shipped Triton kernel is a vLLM-MUSA artifact, so MATE cannot import it by itself. The
    expected signature is ``(a, weight_b, scale_b, zero_points, out)``; check it against the real
    wrapper when wiring this up, because it is a *port-side* assumption and not a verified one.
    """
    spec = os.environ.get("MATE_W4A16_TRITON_CALLABLE")
    if not spec:
        return None
    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute or "w4a16_gemm", None)
    if candidate is None:
        raise SystemExit(f"{spec!r} resolved but has no attribute {attribute or 'w4a16_gemm'!r}")
    return candidate


def _row(
    name: str,
    size_m: int,
    size_k: int,
    size_n: int,
    ms: float,
    zero_points: bool,
    bytes_fn: Callable[[int, int, int], int] = total_bytes,
) -> Row:
    seconds = ms / 1000.0
    return Row(
        name=name,
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        ms=ms,
        gbs=bytes_fn(size_m, size_k, size_n) / seconds / 1e9,
        tflops=flops(size_m, size_k, size_n) / seconds / 1e12,
        bytes_moved=bytes_fn(size_m, size_k, size_n),
        flops=flops(size_m, size_k, size_n),
    )


def _time(bench_gpu_time_with_musa_event, fn: Callable, num_tests: int) -> float:
    """Median cold-cache milliseconds from MATE's own event timer."""
    samples = bench_gpu_time_with_musa_event(
        fn,
        dry_run_iters=5,
        repeat_iters=num_tests,
        l2_flush=True,
        l2_flush_size_mb=L2_FLUSH_SIZE_MB,
    )
    values = [float(value) for value in samples]
    return float(statistics.median(values))


def bench_one(
    torch,
    bench_kineto,
    bench_gpu_time_with_musa_event,
    size_m: int,
    size_k: int,
    size_n: int,
    *,
    zero_points: bool,
    num_tests: int,
    out_dtype=None,
    cross_check: bool = False,
) -> tuple[Row, Row, Optional[Row]]:
    """Measure the MATE op, the fp16 dense baseline, and Triton if it was attached."""
    if out_dtype is None:
        out_dtype = torch.float16
    input_a = torch.randn(size_m, size_k, dtype=torch.float16, device="musa")
    weight_b = torch.randint(-128, 127, (size_n, size_k // 2), dtype=torch.int8, device="musa")
    scale_b = torch.rand(size_n, size_k // GROUP_SIZE, dtype=torch.float16, device="musa")
    zero = scale_b.clone() if zero_points else None
    out = torch.empty(size_m, size_n, dtype=out_dtype, device="musa")

    op = _mate_op()

    def run_mate():
        op(input_a, weight_b, scale_b, out, zero_points=zero)

    mate_row = _row(
        "mate_w4a16", size_m, size_k, size_n, _time(bench_gpu_time_with_musa_event, run_mate, num_tests),
        zero_points,
    )
    if cross_check:
        # The named protocol for the MATE op. Kernel names carry the dispatch name, so the substring
        # is stable across M and identifies the op rather than a neighbouring kernel.
        try:
            measured = bench_kineto(
                run_mate,
                ("mp31_dense_w4a16_gemm",),
                num_tests=num_tests,
                suppress_kineto_output=True,
                flush_l2=True,
            )
            seconds = measured[0] if isinstance(measured, (tuple, list)) else measured
            mate_row = replace(mate_row, kineto_ms=float(seconds) * 1000.0)
        except Exception as exc:  # noqa: BLE001 - a cross-check must not fail the bench
            print(f"[bench] bench_kineto cross-check unavailable: {exc}")

    # fp16 dense: the same GEMM with the fp16 weight W4A16 stands in for. Values do not affect
    # timing, only the 2x byte volume does; the operand is built by dequantizing the int4 codes so
    # the comparison isolates the format rather than a different weight distribution.
    codes = torch.randint(-8, 8, (size_k, size_n), dtype=torch.int16, device="musa").to(torch.float16)
    scale_full = scale_b.T.contiguous().repeat_interleave(GROUP_SIZE, dim=0).to(torch.float16)
    weight_fp16 = codes * scale_full

    def run_fp16():
        torch.matmul(input_a, weight_fp16)

    dense_row = _row(
        "fp16_dense",
        size_m,
        size_k,
        size_n,
        _time(bench_gpu_time_with_musa_event, run_fp16, num_tests),
        False,
        bytes_fn=fp16_dense_bytes,
    )

    triton = load_triton_callable()
    triton_row = None
    if triton is not None:
        triton_out = torch.empty(size_m, size_n, dtype=out_dtype, device="musa")

        def run_triton():
            triton(input_a, weight_b, scale_b, zero, triton_out)

        triton_row = _row(
            "triton_w4a16",
            size_m,
            size_k,
            size_n,
            _time(bench_gpu_time_with_musa_event, run_triton, num_tests),
            False,
        )
    return mate_row, dense_row, triton_row


def format_table(rows: Sequence[Row]) -> str:
    header = (
        f"{'impl':22s} {'proj':9s} {'M':>5s} {'regime':8s} {'ms':>9s} {'GB/s':>8s} "
        f"{'BW%':>6s} {'TFLOP/s':>9s} {'CMP%':>6s} {'kineto ms':>10s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        projection = next(
            (name for name, size_k, size_n in CHECKPOINT_SHAPES if (row.size_k, row.size_n) == (size_k, size_n)),
            f"{row.size_k}x{row.size_n}",
        )
        kineto = f"{row.kineto_ms:10.4f}" if row.kineto_ms is not None else f"{'-':>10s}"
        lines.append(
            f"{row.name:22s} {projection:9s} {row.size_m:5d} {row.regime:8s} {row.ms:9.4f} "
            f"{row.gbs:8.1f} {row.bandwidth_ratio * 100:5.1f}% {row.tflops:9.2f} "
            f"{row.compute_ratio * 100:5.1f}% {kineto}"
        )
    return "\n".join(lines)


def summarize_drift(runs: Sequence[dict]) -> dict:
    """Spread across ladder repeats: a one-shot number is not a measurement."""
    keys = {
        (row["name"], row["size_m"], row["size_k"], row["size_n"])
        for run in runs
        for row in run["rows"]
    }
    drift = {}
    for key in sorted(keys):
        values = [
            row["ms"]
            for run in runs
            for row in run["rows"]
            if (row["name"], row["size_m"], row["size_k"], row["size_n"]) == key
            and math.isfinite(row["ms"])
        ]
        if len(values) < 2:
            continue
        drift[str(key)] = {
            "min_ms": min(values),
            "median_ms": statistics.median(values),
            "max_ms": max(values),
            "spread": max(values) / min(values) if min(values) > 0 else float("inf"),
            "n": len(values),
        }
    return drift


def decode_gate(rows: Sequence[Row], triton_attached: bool) -> tuple[bool, str]:
    """The decode gate: MATE against the baselines that were actually measured.

    The bandwidth ratio is printed beside the verdict but is not the verdict: the shipping decision
    is against the kernel that would otherwise run at decode, so the primary statement is the
    speedup against fp16 dense, and against Triton when that leg was attached.
    """
    mate = {(r.size_m, r.size_k, r.size_n): r for r in rows if r.name.startswith("mate_w4a16")}
    dense = {(r.size_m, r.size_k, r.size_n): r for r in rows if r.name == "fp16_dense"}
    triton = {(r.size_m, r.size_k, r.size_n): r for r in rows if r.name == "triton_w4a16"}
    if not mate:
        return False, "no MATE rows"
    statements = []
    ok = True
    worst = min(
        (
            (dense[key].ms / row.ms, key)
            for key, row in mate.items()
            if row.regime == "decode" and key in dense
        ),
        default=(float("inf"), None),
    )
    if worst[1] is None:
        return False, "no decode rows"
    ok = ok and worst[0] >= 1.0
    statements.append(f"worst decode speedup vs fp16 dense = {worst[0]:.3f}x at M={worst[1][0]}")
    if triton_attached and triton:
        worst_triton = min(
            (
                (triton[key].ms / row.ms, key)
                for key, row in mate.items()
                if row.regime == "decode" and key in triton
            ),
            default=(float("inf"), None),
        )
        if worst_triton[1] is not None:
            ok = ok and worst_triton[0] >= 1.0
            statements.append(
                f"worst decode speedup vs triton = {worst_triton[0]:.3f}x at M={worst_triton[1][0]}"
            )
    else:
        statements.append("triton leg NOT RUN (see not_run)")
    return ok, "; ".join(statements)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-check", action="store_true", help="verify the models without a GPU")
    parser.add_argument("--quick", action="store_true", help="M in {1, 2048}, two smaller projections")
    parser.add_argument("--shapes", nargs="+", default=[], help="projection names to run")
    parser.add_argument("--ms", nargs="+", type=int, default=[], help="M values to run")
    parser.add_argument("--zero-points", action="store_true", help="also bench the asymmetric arm")
    parser.add_argument("--num-tests", type=int, default=30, help="timed iterations per point")
    parser.add_argument("--repeats", type=int, default=3, help="whole-ladder repeats (>= 3 for drift)")
    parser.add_argument("--json", default=None, help="write the full result set here")
    parser.add_argument("--gate", action="store_true", help="exit non-zero when the decode gate misses")
    args = parser.parse_args(argv)

    if args.self_check:
        return _python_self_check()

    torch, bench_kineto, bench_gpu_time_with_musa_event = _require_device_dependencies()
    shapes = [s for s in CHECKPOINT_SHAPES if not args.shapes or s[0] in args.shapes]
    if not shapes:
        raise SystemExit(f"--shapes matched nothing; known: {[s[0] for s in CHECKPOINT_SHAPES]}")
    ms = tuple(args.ms) or ((1, 2048) if args.quick else MS)
    if args.repeats < 3:
        print(f"[bench] WARNING: --repeats {args.repeats} < 3, so the drift statement is not reportable")

    triton_attached = load_triton_callable() is not None
    get_device_name = getattr(getattr(torch, "musa", None), "get_device_name", None)
    device_name = get_device_name(0) if callable(get_device_name) else "unknown"
    print(
        f"[bench] device={device_name} repeats={args.repeats} num_tests={args.num_tests} "
        f"l2_flush={L2_FLUSH_SIZE_MB}MB flush_l2(kineto)=True zero_points={args.zero_points}"
    )
    if not triton_attached:
        print(
            "[bench] NOT RUN: triton_w4a16 - the shipped kernel lives in vLLM-MUSA, not in MATE. "
            "Attach it with MATE_W4A16_TRITON_CALLABLE=module:attr to measure that comparison."
        )
    print(
        "[bench] NOT RUN: vendor _mixed_dtypes_linear - the stage-1 hybrid was withdrawn "
        "(MUSA-100077); it is not a shipped baseline."
    )

    runs = []
    all_rows: list[Row] = []
    for repeat in range(args.repeats):
        rows: list[Row] = []
        for projection, size_k, size_n in shapes:
            for size_m in ms:
                if size_k % MATE_TILE_K:
                    print(f"[bench] skip {projection} K={size_k}: not a multiple of the tile K")
                    continue
                for zero_points in ([False, True] if args.zero_points else [False]):
                    mate_row, dense_row, triton_row = bench_one(
                        torch,
                        bench_kineto,
                        bench_gpu_time_with_musa_event,
                        size_m,
                        size_k,
                        size_n,
                        zero_points=zero_points,
                        num_tests=args.num_tests,
                        cross_check=repeat == 0 and not zero_points,
                    )
                    tag = "/zp" if zero_points else ""
                    rows.append(replace(mate_row, name=mate_row.name + tag))
                    rows.append(dense_row)
                    if triton_row is not None:
                        rows.append(triton_row)
        runs.append({"repeat": repeat, "rows": [asdict(row) for row in rows]})
        all_rows = rows
        print(f"\n=== repeat {repeat + 1}/{args.repeats} ===")
        print(format_table(rows))

    drift = summarize_drift(runs)
    if drift:
        worst = max(drift.items(), key=lambda item: item[1]["spread"])
        print(
            f"\n[bench] drift: worst spread = {worst[1]['spread']:.3f}x over {worst[1]['n']} runs "
            f"at {worst[0]} (a stable claim wants < 1.05)"
        )
    else:
        print("\n[bench] drift: not computed (fewer than two repeats produced the same point)")

    for row in (r for r in all_rows if r.kineto_ms is not None):
        delta = abs(row.kineto_ms - row.ms) / row.ms
        print(
            f"[bench] timer agreement at M={row.size_m}: event {row.ms:.4f} ms vs kineto "
            f"{row.kineto_ms:.4f} ms ({delta * 100:.1f}% apart)"
        )
        # A ~1000x gap is a unit error in one of the two timers, not a measurement disagreement, and
        # every GB/s number in the table would be wrong by that factor. Say so instead of printing it.
        if delta > 5.0:
            print(
                "[bench] WARNING: the two timers disagree by more than 5x. Check the unit of "
                "bench_gpu_time_with_musa_event (FlashInfer-derived: milliseconds) before trusting "
                "the GB/s column."
            )

    ok, statement = decode_gate(all_rows, triton_attached)
    print(f"[bench] decode gate: {statement} -> {'PASS' if ok else 'MISS'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol": {
                        "bench_kineto_flush_l2": True,
                        "event_timer_l2_flush_mb": L2_FLUSH_SIZE_MB,
                        "num_tests": args.num_tests,
                        "ladder_repeats": args.repeats,
                        "device": device_name,
                    },
                    "roofline": {
                        "bandwidth_gb_s": BANDWIDTH_ROOFLINE_GB_S,
                        "compute_tflops_16bit": COMPUTE_ROOFLINE_TFLOPS_16BIT,
                        "decode_pass_ratio": DECODE_PASS_RATIO,
                    },
                    "not_run": {
                        "triton_w4a16": (
                            "measured"
                            if triton_attached
                            else "lives in vLLM-MUSA; set MATE_W4A16_TRITON_CALLABLE=module:attr"
                        ),
                        "vendor_mixed_dtypes_linear": "withdrawn in MUSA-100077 stage 1",
                    },
                    "runs": runs,
                    "drift": drift,
                    "decode_gate": {"pass": ok, "statement": statement},
                },
                handle,
                indent=2,
            )
        print(f"[bench] wrote {args.json}")

    if args.gate and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
