"""Measure preallocated MQA launches with MUSA Events; exclude preparation."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch_musa  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument(
        "--mode", choices=("both", "single", "prefix_tail"), default="both"
    )
    args = parser.parse_args()
    if not 0 < args.rows <= args.context or args.repeat < 2:
        parser.error("require 0 < rows <= context and repeat >= 2")
    if not os.environ.get("MUSA_VISIBLE_DEVICES") or torch.musa.device_count() != 1:
        raise RuntimeError(
            "Pass the scheduler's MUSA_VISIBLE_DEVICES into the container"
        )
    source = args.source.resolve()
    # Production QSA sources use package-relative imports.  Import those
    # sources through their package name so the device benchmark matches the
    # real MATE dispatch; retain file loading for standalone candidate files.
    module = None
    for package_root in source.parents:
        relative = source.relative_to(package_root).with_suffix("")
        if (
            relative.parts[0] == "mate"
            and (package_root / "mate" / "__init__.py").is_file()
        ):
            sys.path.insert(0, str(package_root))
            module = importlib.import_module(".".join(relative.parts))
            break
    if module is None:
        spec = importlib.util.spec_from_file_location("qsa_device_source", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load benchmark source {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    dispatch = module
    if not hasattr(module, "_qsa_mqa_split_m_config"):
        dispatch = importlib.import_module(
            module.__package__.rsplit(".", 1)[0] + ".mqa"
        )
    torch.manual_seed(230923)
    rows, blocks = args.rows, math.ceil(args.context / 4)
    bm = dispatch._qsa_split_m_block_m(1, rows, blocks)
    q = torch.randn(1, rows, 4, 128, device="musa", dtype=torch.bfloat16)
    k = torch.randn(1, blocks, 128, device="musa", dtype=torch.bfloat16)
    visible = torch.empty(1, rows, device="musa", dtype=torch.int32)
    visible[0] = (
        torch.arange(args.context - rows, args.context, device="musa") + 1
    ) // 4
    full = int(visible.min().item()) // 128 * 128
    out = torch.empty(1, rows, blocks, device="musa", dtype=torch.float32)

    def kernels(key_start, key_count, skip_mask):
        long_context = math.ceil(blocks / 128) * 128 >= 131072
        complete = rows // bm * bm
        ranges = (
            ((0, complete, False), (complete, rows - complete, True))
            if long_context
            else ((0, rows, False),)
        )
        launches = []
        for row_start, row_count, bounded_tail in ranges:
            if row_count == 0:
                continue
            factory, config = dispatch._qsa_mqa_split_m_config(
                rows,
                blocks,
                bm,
                128**-0.5,
                key_start=key_start,
                key_count=key_count,
                skip_mask=skip_mask,
                bounded_tail=bounded_tail,
            )
            compiled = factory(**config)
            if args.dump_dir is not None:
                args.dump_dir.mkdir(parents=True, exist_ok=True)
                code = compiled.get_kernel_source()
                digest = hashlib.sha256(code.encode()).hexdigest()[:12]
                (args.dump_dir / f"qsa_mqa_{digest}.mu").write_text(code)
            execute = compiled.adapter._get_executable(q.device)
            runtime_args = (q, k, visible, out, key_start, key_count)
            if long_context:
                runtime_args += (row_start, row_count)
            launches.append((execute, runtime_args))
        return launches

    launches = {}
    if args.mode in ("both", "single"):
        launches["single"] = kernels(0, blocks, False)
    if args.mode in ("both", "prefix_tail") and 128 <= full < blocks:
        launches["prefix_tail"] = kernels(0, full, True) + kernels(
            full, blocks - full, False
        )
    if not launches:
        parser.error("prefix_tail requires a nonempty aligned prefix and tail")
    ids = torch.tensor(sorted({0, rows // 2, rows - 1}), device="musa")
    expected = (
        torch.einsum("bqhd,bnd->bqnh", q[:, ids].float(), k[:, :blocks].float())
        .relu()
        .sum(-1)
        * 128**-0.5
    )
    expected.masked_fill_(
        torch.arange(blocks, device="musa")[None, None] >= visible[:, ids, None],
        -torch.inf,
    )
    errors = {}
    for name, fns in launches.items():
        for fn, runtime_args in fns:
            fn(*runtime_args)
        actual = out[:, ids, :blocks]
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        valid = torch.isfinite(expected)
        errors[name] = (actual[valid] - expected[valid]).abs().max().item()
    for _ in range(args.warmup):
        for fns in launches.values():
            for fn, runtime_args in fns:
                fn(*runtime_args)
    torch.musa.synchronize()
    pending = {name: [] for name in launches}
    names = list(launches)
    for iteration in range(args.repeat):
        for name in names if iteration % 2 == 0 else names[::-1]:
            events = []
            for fn, runtime_args in launches[name]:
                start, end = (
                    torch.musa.Event(enable_timing=True),
                    torch.musa.Event(enable_timing=True),
                )
                start.record()
                fn(*runtime_args)
                end.record()
                events.append((start, end))
            pending[name].append(events)
    torch.musa.synchronize()
    samples = {
        name: [sum(a.elapsed_time(b) * 1000 for a, b in events) for events in rounds]
        for name, rounds in pending.items()
    }
    medians = {name: statistics.median(values) for name, values in samples.items()}
    component_medians = {
        name: [
            statistics.median(
                [
                    events[index][0].elapsed_time(events[index][1]) * 1000
                    for events in rounds
                ]
            )
            for index in range(len(launches[name]))
        ]
        for name, rounds in pending.items()
    }
    print(
        "RESULT "
        + json.dumps(
            {
                "rows": rows,
                "context": args.context,
                "mode": args.mode,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "benchmark_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "visible_devices": os.environ["MUSA_VISIBLE_DEVICES"],
                "device": str(torch.musa.get_device_properties(0)),
                "timing": "sum of per-kernel direct-launch Event intervals; no wrapper or allocation",
                "warmup": args.warmup,
                "repeat": args.repeat,
                "samples_us": samples,
                "median_us": medians,
                "component_median_us": component_medians,
                "sampled_max_abs_error": errors,
                "score_gemm_tflops": {
                    name: 2 * rows * blocks * 4 * 128 / us / 1e6
                    for name, us in medians.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
