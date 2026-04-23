# Copyright (c) 2025, Ted Zadouri, Tri Dao.

# We recommend locking GPU clocks before running the benchmark to ensure consistent results.
# This can be done using the following commands (1830 MHz is the clock for H100):
# sudo nvidia-smi -i 0 -pm 1
# sudo nvidia-smi -i 0 --lock-gpu-clocks 1830,1830
# See more here: https://github.com/triton-lang/triton/blob/d9f10ebdc5da53f73eb852fde73d8d7d80b679d1/python/triton/testing.py#L487
import time
import json
import torch
import mate

from einops import rearrange
from torch.profiler import profile, ProfilerActivity  # noqa: F401
from itertools import product
from collections import namedtuple
from mate.testing.utils import bench_kineto


# Device Setup
device = "musa"
torch.manual_seed(0)

# Metadata
dtype = torch.bfloat16
seqlen_kv = 4096
seqlen_q = 1
# nheads_q = 128
sm_scale = 1 / ((512 + 64) ** 0.5)
workspace_buffer = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
scheduler_metadata = (workspace_buffer, False)

PerfCfg = namedtuple(
    "PerfCfg",
    [
        "batch_size",
        "seqlen_q",
        "seqlen_kv",
        "nheads_q",
        "causal",
    ],
)

nheads_kv = 1
headdim_rope = 64
headdim_latent = 512
has_qv = headdim_rope == 64 and headdim_latent > 64
page_size = 64

# Perf Configs
batch_sizes = [30, 36]
seqlen_qs = [[2, 2]]  # [min, max] for varlen
seqlen_kvs = [int(s * 1024) for s in [4, 5, 5.5]]
nheads_qs = [64, 128]  # [8, 16, 32, 64, 128]
causals = [True]
perf_configs = product(batch_sizes, seqlen_qs, seqlen_kvs, nheads_qs, causals)
for conf_idx, cfg in enumerate(PerfCfg(*perf_config) for perf_config in perf_configs):
    batch_size = cfg.batch_size
    seqlen_q = cfg.seqlen_q
    seqlen_kv = cfg.seqlen_kv
    nheads_q = cfg.nheads_q
    causal = cfg.causal
    varlen_q = type(seqlen_q) is not int
    cu_seqlens_q = None
    max_seqlen_q = max(seqlen_q) if varlen_q else seqlen_q
    if varlen_q:
        seqlen_q = torch.randint(
            seqlen_q[0], seqlen_q[1] + 1, (batch_size,), device=device
        )
        cu_seqlens_q = (
            torch.nn.functional.pad(torch.cumsum(seqlen_q, dim=0), (1, 0))
            .to(torch.int32)
            .to(device)
        )
        # seqlen_q = seqlen_q.tolist()

    cache_seqlens = torch.tensor(
        [seqlen_kv] * batch_size, device=device, dtype=torch.int
    )
    if varlen_q:
        q_rope = torch.randn(
            sum(seqlen_q).item(),
            nheads_q,
            headdim_rope,
            dtype=dtype,
            device=device,
        )
        q_latent = torch.randn(
            sum(seqlen_q).item(),
            nheads_q,
            headdim_latent,
            dtype=dtype,
            device=device,
        )
    else:
        q_rope = torch.randn(
            batch_size,
            seqlen_q,
            nheads_q,
            headdim_rope,
            dtype=dtype,
            device=device,
        )
        q_latent = torch.randn(
            batch_size,
            seqlen_q,
            nheads_q,
            headdim_latent,
            dtype=dtype,
            device=device,
        )
    try:
        v_cache = torch.randn(
            batch_size,
            seqlen_kv,
            nheads_kv,
            headdim_latent,
            dtype=dtype,
            device=device,
        )
        k_cache = torch.randn(
            batch_size,
            seqlen_kv,
            nheads_kv,
            headdim_rope,
            dtype=dtype,
            device=device,
        )
        if page_size is not None:
            assert seqlen_kv % page_size == 0
            k_cache, v_cache = [
                rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size)
                for x in [k_cache, v_cache]
            ]
            page_table = rearrange(
                torch.arange(
                    batch_size * seqlen_kv // page_size,
                    device=device,
                    dtype=torch.int32,
                ),
                "(b s) -> b s",
                s=seqlen_kv // page_size,
            )
        else:
            page_table = None
        concat_kv = torch.cat([v_cache, k_cache], dim=-1).contiguous()
    except torch.OutOfMemoryError:
        continue

    q = torch.cat([q_latent, q_rope], dim=-1)
    kv = torch.cat([v_cache, k_cache], dim=-1)

    q_seq_per_hk = max_seqlen_q * nheads_q // nheads_kv

    # Use for combine memio calculation
    tile_scheduler_metadata, num_splits = mate.flashmla.get_mla_metadata(
        cache_seqlens, q_seq_per_hk, nheads_kv
    )
    tile_scheduler_metadata[:, 5:] = 0

    time.sleep(1)  # to avoid power throttling

    def fn0():
        return mate.flash_attn_with_kvcache(
            q=q_rope,
            k_cache=kv[..., headdim_latent:],
            v_cache=kv[..., :headdim_latent],
            qv=q_latent,
            cache_seqlens=cache_seqlens,  # cache_seqlens
            page_table=page_table,  # page_table
            cu_seqlens_q=cu_seqlens_q,  # cu_query_lens
            max_seqlen_q=max_seqlen_q,  # max_seqlen_q
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            rotary_interleaved=False,
            scheduler_metadata=scheduler_metadata,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
            return_softmax_lse=True,
        )

    t_splitkv, t_combine = bench_kineto(
        fn0,
        ("device_kernel", "flash_mla_combine"),
        suppress_kineto_output=True,
        num_tests=10,
        trace_path=f"trace_{conf_idx:02d}.json",
    )

    # fn0()
    # t0 = 100
    # t0 = triton.musa_testing.do_bench(fn0, warmup=3, rep=10)
    # with torch.musa.stream(torch.musa.Stream()):
    #     t0 = do_bench_cudagraph(fn0, rep=10)
    # graph = torch.musa.MUSAGraph()
    # with torch.musa.graph(graph):
    #     fn0()

    # def g():
    #     return graph.replay()

    # t0 = do_bench(graph.replay, warmup=1, rep=10)
    size_dtype = torch.finfo(dtype).bits // 8  # 2 for fp16/bf16, 4 for fp32
    size_lse = torch.finfo(torch.float32).bits // 8
    total_seqlen_kv = (
        seqlen_kv * batch_size if cache_seqlens is None else cache_seqlens.sum().item()
    )
    total_seqlen_q = sum(seqlen_q).item() if varlen_q else seqlen_q * batch_size
    # Split-related
    total_num_splits = batch_size + tile_scheduler_metadata.size(0)
    total_combine_splits, total_combine_batchs = 0, 0
    intact_batchs = []
    for batch_idx in range(len(num_splits) - 1):
        splits = num_splits[batch_idx + 1] - num_splits[batch_idx]
        total_combine_splits += splits if splits > 1 else 0
        total_combine_batchs += 1 if splits > 1 else 0
        if splits == 1:
            intact_batchs.append(batch_idx)
    intact_batchs = torch.tensor(intact_batchs, device=device, dtype=torch.int32)
    total_intact_splits = total_num_splits - total_combine_splits
    total_intact_batchs = batch_size - total_combine_batchs
    if varlen_q:
        total_intact_seqlen_q = seqlen_q[intact_batchs].sum().item()
    else:
        total_intact_seqlen_q = seqlen_q * total_intact_batchs
    # mem io for splitkv kernel. Need to handle differently depending on split case.
    mem_io = (
        total_seqlen_kv
        * nheads_kv
        * (headdim_rope + headdim_latent)
        * size_dtype  # Load K
        + q_rope.numel() * size_dtype  # Load q_rope
        + (q_latent.numel() if q_latent is not None else 0)
        * size_dtype  # Load q_latent
        + size_dtype
        * (
            max_seqlen_q * nheads_q * headdim_latent * total_combine_splits  # out_accum
            + total_intact_seqlen_q * nheads_q * headdim_latent  # out
        )  # out / out_accum
        + size_lse
        * (
            max_seqlen_q * nheads_q * total_combine_splits  # lse_accum
            + total_intact_seqlen_q * nheads_q  # lse
        )  # lse / lse_accum
    )  # last term is for the output
    if varlen_q:
        flops = (
            total_seqlen_q
            * nheads_q  # q
            * seqlen_kv
            * (headdim_rope + headdim_latent * (2 if has_qv else 1))  # kv
            * 2  # dtype
        )
    else:
        flops = (
            seqlen_q
            * nheads_q  # q
            * total_seqlen_kv
            * (headdim_rope + headdim_latent * (2 if has_qv else 1))  # kv
            * 2  # dtype
        )
    # Get combine mem io
    mem_io_combine = (
        total_combine_splits
        * (
            max_seqlen_q * nheads_q * headdim_latent * size_dtype  # out_accum
            + max_seqlen_q * nheads_q * size_lse  # lse_accum
        )  # Read from accum buffers
        + total_combine_batchs
        * (
            max_seqlen_q * nheads_q * headdim_latent * size_dtype  # out
            + max_seqlen_q * nheads_q * size_lse  # lse
        )  # Write to output & lse
    )
    result = {
        "config": ", ".join(f"{k}={v}" for k, v in cfg._asdict().items()),
        "perf units  ": "Time (us), Bandwidth (GB/s), Compute (TFLOPS/s)",
        "splitkv-perf": f"{t_splitkv * 1e6:.2f}     {mem_io * 1e-9 / (t_splitkv):.2f}            {flops * 1e-12 / (t_splitkv):.2f}",
        "combine-perf": f"{t_combine * 1e6:05.2f}      {mem_io_combine * 1e-9 / (t_combine):06.2f}            {'N/A'}",
    }
    print(
        f"[PERF {conf_idx:02d} {'VarlenQ' if varlen_q else 'FixedQ'}]"
        + json.dumps(result, indent=2)
    )
