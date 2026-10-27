"""Measure preallocated QSA prefill/decode with MUSA Events; exclude preparation."""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from mate.qsa_kernels import core as core_dispatch
from mate.qsa_kernels.tilelang import qsa_core as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prefill", "decode"], required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--page-size", type=int, choices=[16, 64], default=64)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--prefill-schedule", choices=["auto", "direct"], default="auto"
    )
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if not os.environ.get("MUSA_VISIBLE_DEVICES") or torch.musa.device_count() != 1:
        parser.error("forward the reserved MUSA_VISIBLE_DEVICES into the container")
    if not (
        2048 <= args.context <= 1048576
        and args.rows > 0
        and 1 <= args.splits <= 33
        and args.repeat > 0
        and args.warmup >= 0
        and args.kv_heads > 0
        and args.heads > 0
        and args.heads % args.kv_heads == 0
        and args.heads // args.kv_heads <= 16
    ):
        parser.error(
            "require context 2048..1M, positive rows/repeat, splits 1..33, and a valid GQA group <=16"
        )
    torch.manual_seed(230924)
    rows, context, page = args.rows, args.context, args.page_size
    heads, kv_heads = args.heads, args.kv_heads
    group = heads // kv_heads
    dtype = getattr(torch, args.dtype)
    pages = math.ceil(context / page)
    q = torch.randn(rows, heads, 256, device="musa", dtype=dtype)
    k = torch.randn(pages, page, kv_heads, 256, device="musa", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.randperm(pages, device="musa")
    rank = torch.arange(512, device="musa")[None]
    complete_blocks = context // 4
    stride = 509
    while math.gcd(stride, complete_blocks) != 1:
        stride += 1
    block = (
        torch.arange(rows, device="musa")[:, None] * 37 + rank * stride
    ) % complete_blocks
    logical = torch.full((rows, 2051), -1, device="musa", dtype=torch.int64)
    logical[:, :2048] = (
        block[:, :, None] * 4 + torch.arange(4, device="musa")
    ).reshape(rows, 2048)
    tail = context % 4
    logical[:, 2048 : 2048 + tail] = torch.arange(
        context - tail, context, device="musa"
    )
    safe = logical.clamp_min(0)
    physical = torch.where(
        logical >= 0, table[safe // page] * page + safe % page, -1
    ).to(torch.int32)
    output = torch.empty_like(q)
    launches = []
    canonical_heads = heads == 24 and kv_heads == 2
    local_head_prefill = (
        args.phase == "prefill"
        and heads == 3
        and kv_heads == 1
        and rows >= 7
        and args.prefill_schedule == "auto"
    )
    kernel_path = (
        "structured_prefill"
        if args.phase == "prefill" and canonical_heads
        else "local_head_prefill"
        if local_head_prefill
        else "direct_paged"
        if args.phase == "prefill"
        else "splitkv"
    )
    if args.phase == "prefill" and canonical_heads:
        descriptors = torch.full((rows, 513), -1, device="musa", dtype=torch.int32)
        descriptors[:, :512] = physical[:, :2048:4] + 3
        if tail:
            descriptors[:, 512] = physical[:, 2048] + tail - 1
        kernel = core._qsa_core_paged_prefill_kernel(
            block_n=64,
            scale=256**-0.5 * math.log2(math.e),
            structured_microblocks=True,
            predecode_physical_ids=True,
            direct_async_gather=True,
            use_tme_output=True,
            prefetch_next_descriptor=True,
            v_gather_stripe_major=True,
            disable_index_type_promotion=k.numel() * k.element_size() < 2**31,
            dtype_name=args.dtype,
        )
        launches.append(
            (
                "core",
                kernel.adapter._get_executable(q.device),
                (q, k.reshape(-1, 2, 256), v.reshape(-1, 2, 256), descriptors, output),
            )
        )
    elif local_head_prefill:
        qp = q
        output = torch.empty_like(qp)
        kernel = core._qsa_core_paged_prefill_kernel(
            block_n=64,
            scale=256**-0.5 * math.log2(math.e),
            predecode_physical_ids=True,
            direct_async_gather=True,
            kv_heads=kv_heads,
            gqa_group_size=group,
            dtype_name=args.dtype,
        )
        launches.append(
            (
                "core",
                kernel.adapter._get_executable(q.device),
                (
                    qp,
                    k.reshape(-1, kv_heads, 256),
                    v.reshape(-1, kv_heads, 256),
                    physical,
                    output,
                ),
            )
        )
    elif args.phase == "prefill":
        qp = q
        req = torch.zeros(rows, device="musa", dtype=torch.int32)
        pos = torch.full_like(req, context - 1)
        lengths = torch.tensor([context], device="musa", dtype=torch.int32)
        output = torch.empty_like(qp)
        kernel = core._qsa_core_kernel(
            block_n=64,
            page_size=page,
            scale=256 ** (-0.5) * math.log2(math.e),
            pv_block_h=16,
            annotate_v_layout=False,
            flat_physical_metadata=True,
            kv_heads=kv_heads,
            gqa_group_size=group,
            dtype_name=args.dtype,
        )
        launches.append(
            (
                "core",
                kernel.adapter._get_executable(q.device),
                (
                    qp,
                    k,
                    v,
                    physical,
                    physical,
                    physical,
                    req,
                    pos,
                    lengths,
                    output,
                ),
            )
        )
    else:
        splits = args.splits
        selected = physical[:, :2051]
        req = torch.zeros(rows, device="musa", dtype=torch.int32)
        pos = torch.full_like(req, context - 1)
        lengths = torch.tensor([context], device="musa", dtype=torch.int32)
        partial = torch.empty(
            splits, rows, heads, 256, device="musa", dtype=torch.float32
        )
        lse = torch.empty(splits, rows, heads, device="musa", dtype=torch.float32)
        kernel = core._qsa_core_kernel(
            block_n=64,
            page_size=page,
            num_splits=splits,
            scale=256 ** (-0.5) * math.log2(math.e),
            pv_block_h=32,
            flat_physical_metadata=True,
            compact_partial_output=False,
            direct_async_gather=True,
            kv_heads=kv_heads,
            gqa_group_size=group,
            narrow_address=core_dispatch._qsa_core_narrow_address_eligible(
                q, k, v, selected, req, pos, lengths, partial, lse
            ),
            dtype_name=args.dtype,
            annotate_v_layout=False,
        )
        launches.append(
            (
                "split",
                kernel.adapter._get_executable(q.device),
                (
                    q,
                    k,
                    v,
                    selected,
                    selected,
                    selected,
                    req,
                    pos,
                    lengths,
                    partial,
                    lse,
                ),
            )
        )
        merge = core._qsa_core_splitk_merge_kernel(
            splits, "float32", heads, output_dtype_name=args.dtype
        )
        launches.append(
            (
                "merge",
                merge.adapter._get_executable(q.device),
                (
                    partial,
                    lse,
                    output,
                ),
            )
        )
    for _, fn, inputs in launches:
        fn(*inputs)
    max_error = 0.0
    raw_fp64_max_error = 0.0
    for row in sorted({0, rows // 2, rows - 1}):
        ids = physical[row, physical[row] >= 0].long()
        keys, values = (
            k.reshape(-1, kv_heads, 256)[ids].cpu().double(),
            v.reshape(-1, kv_heads, 256)[ids].cpu().double(),
        )
        scores = (
            torch.einsum(
                "hgd,nhd->hgn",
                q[row].cpu().double().reshape(kv_heads, group, 256),
                keys,
            )
            * 256**-0.5
        )
        expected = torch.einsum("hgn,nhd->hgd", scores.softmax(-1), values).reshape(
            heads, 256
        )
        actual = output[row].cpu().double()
        rounded_expected = expected.to(dtype).double()
        torch.testing.assert_close(actual, rounded_expected, rtol=0, atol=0.002)
        max_error = max(max_error, (actual - rounded_expected).abs().max().item())
        raw_fp64_max_error = max(
            raw_fp64_max_error, (actual - expected).abs().max().item()
        )
    assert torch.isfinite(output).all().item()
    for _ in range(args.warmup):
        for _, fn, inputs in launches:
            fn(*inputs)
    torch.musa.synchronize()
    events = []
    for _ in range(args.repeat):
        sample = []
        for name, fn, inputs in launches:
            start, end = (
                torch.musa.Event(enable_timing=True),
                torch.musa.Event(enable_timing=True),
            )
            start.record()
            fn(*inputs)
            end.record()
            sample.append((name, start, end))
        events.append(sample)
    torch.musa.synchronize()
    samples = [sum(a.elapsed_time(b) * 1000 for _, a, b in sample) for sample in events]
    median = statistics.median(samples)
    paths = (Path(core.__file__),)
    print(
        "RESULT "
        + json.dumps(
            {
                **vars(args),
                "kernel_path": kernel_path,
                "workload": "steady cached attention; all queries at context-1",
                "visible_devices": os.environ["MUSA_VISIBLE_DEVICES"],
                "device": str(torch.musa.get_device_properties(0)),
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                "timing": "sum of preallocated per-kernel Event intervals",
                "samples_us": samples,
                "median_us": median,
                "component_median_us": {
                    name: statistics.median(
                        [
                            next(
                                a.elapsed_time(b) * 1000
                                for n, a, b in sample
                                if n == name
                            )
                            for sample in events
                        ]
                    )
                    for name, _, _ in launches
                },
                "sampled_max_abs_error": max_error,
                "reference": "FP64 rounded to the output dtype; atol=0.002, rtol=0",
                "raw_fp64_max_abs_error": raw_fp64_max_error,
                "effective_tflops": 4
                * rows
                * heads
                * (2048 + tail)
                * 256
                / median
                / 1e6,
                "logical_kv_gbs": rows
                * (2048 + tail)
                * kv_heads
                * 2
                * 256
                * 2
                / median
                / 1e3,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
