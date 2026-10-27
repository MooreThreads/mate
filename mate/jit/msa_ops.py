from __future__ import annotations

import functools
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, cast

import torch

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from mate.execution_context import skip_kernel_launch_if_dry_run

from ..mate_runtime import resolve_num_mps
from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import TVM_HEADER, maybe_contiguous


CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

CUDA_FLAGS = [
    "-Od3",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]


@functools.cache
def _get_msa_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def get_msa_template(template_name: str):
    template_dir = (jit_env.MATE_TEMPLATE_DIR / "attention" / "msa").as_posix()
    return _get_msa_template_env(template_dir).get_template(template_name)


MSA_MAXSCORE_KERN_TEMPLATE = get_msa_template("maxscore_kern.j2")
MSA_SPARSE_TOPK_SELECT_TEMPLATE = get_msa_template("sparse_topk_select_kern.j2")
MSA_FWD_KERN_TEMPLATE = get_msa_template("fwd_kern.j2")
_FP8_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)

MSA_DTYPE_MUTLASS = {
    torch.float16: "mutlass::half_t",
    torch.bfloat16: "mutlass::bfloat16_t",
}
if _FP8_E4M3_DTYPE is not None:
    MSA_DTYPE_MUTLASS[_FP8_E4M3_DTYPE] = "mutlass::float_e4m3_t"

MSA_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}
if _FP8_E4M3_DTYPE is not None:
    MSA_DTYPE_FFI[_FP8_E4M3_DTYPE] = "dl_float8_e4m3fn"

MSA_DTYPE_NAMES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
if _FP8_E4M3_DTYPE is not None:
    MSA_DTYPE_NAMES[_FP8_E4M3_DTYPE] = "fp8e4m3"

MSA_FWD_DTYPE_CONFIG: dict[torch.dtype, dict[str, str]] = {
    torch.float16: {
        "name": "f16",
        "element": "mutlass::half_t",
        "element_dtype": "dl_float16",
    },
    torch.bfloat16: {
        "name": "bf16",
        "element": "mutlass::bfloat16_t",
        "element_dtype": "dl_bfloat16",
    },
}
if _FP8_E4M3_DTYPE is not None:
    MSA_FWD_DTYPE_CONFIG[_FP8_E4M3_DTYPE] = {
        "name": "fp8e4m3",
        "element": "mutlass::float_e4m3_t",
        "element_dtype": "dl_float8_e4m3fn",
    }


def _maxscore_k_tiles(max_seqlen_k: int, page_size: int = 128) -> int:
    if int(page_size) <= 0:
        raise ValueError(f"maxscore page_size must be positive, got {page_size}")
    kv_tiles = (int(max_seqlen_k) + int(page_size) - 1) // int(page_size)
    return ((kv_tiles + 127) // 128) * 128


@lru_cache(maxsize=256)
def _msa_maxscore_encode(
    dtype_name: str,
    is_paged_kv: bool,
    page_table_kind: str,
    page_size: int,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    tile_q: int,
    parallel_k_tiles: bool,
    q_stages: int,
    k_stages: int,
    enable_k_prefetch: bool,
    mma_tile_q: int,
    is_varlen: bool,
    has_metadata: bool,
) -> str:
    return (
        f"msa_maxscore_dtype_{dtype_name}"
        f"_paged_{int(is_paged_kv)}"
        f"_pt_{page_table_kind}"
        f"_ps_{int(page_size)}"
        f"_causal_{int(causal)}"
        f"_hd_{head_dim}"
        f"_hr_{head_ratio}"
        f"_tq_{tile_q}"
        f"_pk_{int(parallel_k_tiles)}"
        f"_qs_{int(q_stages)}"
        f"_ks_{int(k_stages)}"
        f"_pf_{int(enable_k_prefetch)}"
        f"_mq_{int(mma_tile_q)}"
        f"_vl_{int(is_varlen)}"
        f"_hm_{int(has_metadata)}"
    )


def _msa_maxscore_encode_config(config: Mapping[str, object]) -> str:
    return _msa_maxscore_encode(
        str(config["dtype_name"]),
        bool(config["is_paged_kv"]),
        str(config["page_table_kind"]),
        cast(int, config["page_size"]),
        bool(config["causal"]),
        cast(int, config["head_dim"]),
        cast(int, config["head_ratio"]),
        cast(int, config["tile_q"]),
        bool(config["parallel_k_tiles"]),
        cast(int, config["q_stages"]),
        cast(int, config["k_stages"]),
        bool(config["enable_k_prefetch"]),
        cast(int, config["mma_tile_q"]),
        bool(config["is_varlen"]),
        bool(config["has_metadata"]),
    )


@lru_cache(maxsize=256)
def _maxscore_tile_q(
    head_ratio: int,
    parallel_k_tiles: bool,
    *,
    dtype: torch.dtype | None = None,
    max_seqlen_q: int | None = None,
) -> int:
    # Match the original FP8 MSA TileLang prefill geometry: one 128-row Q
    # tile feeds four consumer warp-squads (512 consumer threads). Keep the
    # small decode geometry because a one-row request cannot amortize that
    # tile.
    if (
        dtype == _FP8_E4M3_DTYPE
        and parallel_k_tiles
        and int(head_ratio) == 1
        and max_seqlen_q is not None
        and int(max_seqlen_q) > 512
    ):
        return 128
    # BF16 prefill takes the same 128-row tile, with K prefetch (see
    # make_msa_maxscore_config). A 64-row BF16 tile carries only 64 FLOP per
    # byte of K, so the kernel streams K at about HBM rate. Each score is the
    # same reduction over the same inputs with either tile, so the scores are
    # bit-identical; MiniMax-M3 prefill on MP31 ran 1.4x (32K-query chunks)
    # to 4.8x (4K queries at a 496K prefix) faster. Up to 512 query tokens
    # (short extends, decode) BF16 keeps its tested 64- and 16-row tiles.
    if (
        dtype == torch.bfloat16
        and parallel_k_tiles
        and int(head_ratio) == 1
        and max_seqlen_q is not None
        and int(max_seqlen_q) > 512
    ):
        return 128
    if (
        dtype == torch.bfloat16
        and parallel_k_tiles
        and int(head_ratio) == 1
        and max_seqlen_q is not None
        and int(max_seqlen_q) >= 64
    ):
        return 64
    if (
        parallel_k_tiles
        and max_seqlen_q is not None
        and (
            (
                dtype == _FP8_E4M3_DTYPE
                and int(head_ratio) == 1
                and int(max_seqlen_q) <= 512
            )
            or dtype == torch.bfloat16
        )
        and 16 % int(head_ratio) == 0
    ):
        return 16
    tile_q = max(64, int(head_ratio))
    while tile_q % int(head_ratio) != 0 or tile_q % 4 != 0:
        tile_q += 4
    return tile_q


@lru_cache(maxsize=256)
def make_msa_maxscore_config(
    dtype: torch.dtype,
    *,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    max_seqlen_q: int,
    page_table_kind: str | None = None,
    page_size: int = 128,
    is_varlen: bool = True,
    has_metadata: bool | None = None,
) -> dict[str, object]:
    if dtype not in MSA_DTYPE_MUTLASS:
        raise TypeError(
            "msa maxscore only supports torch.float16, torch.bfloat16, "
            "and torch.float8_e4m3fn."
        )
    if has_metadata is None:
        has_metadata = bool(is_varlen)
    if int(head_dim) <= 0 or int(head_dim) % 8 != 0:
        raise ValueError(
            f"msa maxscore expects head_dim divisible by 8, got {head_dim}"
        )
    if int(head_ratio) <= 0:
        raise ValueError(f"msa maxscore expects positive head_ratio, got {head_ratio}")
    if page_table_kind is None:
        page_table_kind = "batched2d" if is_paged_kv else "dense"
    page_table_kind = str(page_table_kind)
    if page_table_kind not in {"dense", "batched2d", "flat"}:
        raise ValueError(f"unsupported MSA maxscore page_table_kind: {page_table_kind}")
    if bool(is_paged_kv) != (page_table_kind != "dense"):
        raise ValueError(
            "page_table_kind and is_paged_kv disagree: "
            f"{page_table_kind=} {is_paged_kv=}"
        )
    if int(page_size) <= 0:
        raise ValueError(f"msa maxscore expects positive page_size, got {page_size}")
    # The current SQMMA/output contract is one 128-column K tile per page.
    # Keep the parameter explicit in the specialization key while rejecting
    # unsupported cross-page TME layouts until they are implemented.
    if int(page_size) != 128:
        raise ValueError(
            f"paged MSA maxscore currently requires page_size == 128, got {page_size}"
        )
    # The MSA selector uses one proxy-Q head per KV head.  Split K whenever that
    # contract holds so short-Q and tail waves expose enough independent work.
    # The scheduler chooses the partition count from the actual Q-work capacity.
    parallel_k_tiles = int(head_ratio) == 1
    selected_tile_q = _maxscore_tile_q(
        int(head_ratio),
        bool(parallel_k_tiles),
        dtype=dtype,
        max_seqlen_q=int(max_seqlen_q),
    )
    if (
        selected_tile_q <= 0
        or selected_tile_q % 4 != 0
        or selected_tile_q % int(head_ratio) != 0
    ):
        raise ValueError(
            "msa maxscore tile_q must be a positive multiple of 4 and "
            f"head_ratio, got tile_q={selected_tile_q}, head_ratio={head_ratio}"
        )
    q_stages = 1
    # One K stage is faster for this max-score epilogue: the register rowmax
    # and store already hide the next TME load, while a second stage adds
    # barrier/shared-state traffic. The MUTLASS builder still uses its
    # required dispatch-policy stage count internally.
    k_stages = 1
    # BF16 prefetches K only with the 128-row prefill tile; its 64-row tile
    # keeps the tested no-prefetch configuration.
    enable_k_prefetch = bool(
        selected_tile_q > 16 and (dtype != torch.bfloat16 or selected_tile_q >= 128)
    )
    # Keep the tested generic geometry for production. MmaTileQ remains an
    # explicit option for A/Bs; the current long-tile path uses M32, while
    # the small decode tile uses M16.
    mma_tile_q = 32 if selected_tile_q >= 32 else 16
    return {
        "element": MSA_DTYPE_MUTLASS[dtype],
        "element_dtype": MSA_DTYPE_FFI[dtype],
        "dtype_name": MSA_DTYPE_NAMES[dtype],
        "is_paged_kv": bool(is_paged_kv),
        "page_table_kind": page_table_kind,
        "page_size": int(page_size),
        "causal": bool(causal),
        "head_dim": int(head_dim),
        "head_ratio": int(head_ratio),
        "tile_q": selected_tile_q,
        "parallel_k_tiles": bool(parallel_k_tiles),
        "q_stages": q_stages,
        "k_stages": k_stages,
        "enable_k_prefetch": enable_k_prefetch,
        "mma_tile_q": mma_tile_q,
        "is_varlen": bool(is_varlen),
        "has_metadata": bool(has_metadata),
    }


def _render_msa_maxscore_kernel_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = _msa_maxscore_encode_config(config)
    return MSA_MAXSCORE_KERN_TEMPLATE.render(render_config)


def _msa_k_partitions_for_mps(
    *,
    num_mps: int,
    q_work_count: int,
    max_k_tiles: int,
    parallel_k_tiles: bool,
) -> int:
    ctas_per_mp = 3 if parallel_k_tiles else 1
    target_ctas = max(1, int(num_mps)) * ctas_per_mp
    q_works = max(1, int(q_work_count))
    return min(
        max(1, int(max_k_tiles)),
        max(1, (target_ctas + q_works // 2) // q_works),
    )


def _msa_schedule_num_mps(
    *,
    hardware_mps: int,
    q_work_count: int,
    max_k_tiles: int,
    parallel_k_tiles: bool,
) -> int:
    """Choose the active MP-slot count for the per-MP persistent schedule.

    The metadata kernel uses the same fixed-point calculation.  Keeping the
    slot count close to the amount of Q/K work avoids launching a full-device
    grid for decode-sized inputs while retaining one Q range per active slot.
    """
    ctas_per_mp = 3 if parallel_k_tiles else 1
    mps = max(1, int(hardware_mps))
    q_work_count = max(1, int(q_work_count))
    max_k_tiles = max(1, int(max_k_tiles))
    for _ in range(3):
        partitions = _msa_k_partitions_for_mps(
            num_mps=mps,
            q_work_count=q_work_count,
            max_k_tiles=max_k_tiles,
            parallel_k_tiles=parallel_k_tiles,
        )
        partition_groups = (partitions + ctas_per_mp - 1) // ctas_per_mp
        next_mps = min(mps, max(1, q_work_count * partition_groups))
        if next_mps == mps:
            break
        mps = next_mps
    return mps


# Building the weighted range table is a fixed-cost device launch (and the
# metadata-enabled main kernel has a small prologue of its own).  A Q-count
# threshold by itself is not stable: two Q work items with 1K of KV do not
# amortize that cost, while the same two items with 128K of KV can.  Express
# the policy as an aggregate Q-work/K-tile budget and derive the Q threshold
# from the actual, unpadded K tile count.  The constants are deliberately
# conservative for the decode path where metadata is rebuilt on every call.
_MSA_MAXSCORE_SCHEDULE_MIN_Q_WORK = 2
_MSA_MAXSCORE_SCHEDULE_WORK_BUDGET = 8192


def _msa_schedule_q_work_threshold(valid_k_tiles: int) -> int:
    """Return the minimum logical Q-work count worth scheduling.

    ``q_work_count`` counts ``(batch, Q tile, KV head)`` work items, rather
    than individual query tokens.  The threshold decreases for long KV
    sequences because each Q-work then carries more SQMMA work.  Keeping this
    as a pure helper makes the dispatch rule easy to test and tune without
    changing the kernel ABI.
    """

    k_tiles = max(1, int(valid_k_tiles))
    return max(
        _MSA_MAXSCORE_SCHEDULE_MIN_Q_WORK,
        math.ceil(_MSA_MAXSCORE_SCHEDULE_WORK_BUDGET / k_tiles),
    )


def _msa_should_use_schedule(
    *,
    q_work_count: int,
    valid_k_tiles: int,
    schedule_requested: bool,
    tile_q: int | None = None,
) -> bool:
    """Decide whether to pay for the per-MP metadata schedule.

    The decision is made on the host from launch-capacity metadata; no device
    length readback is introduced.  ``schedule_requested`` retains the old
    contract: a varlen call or an explicit schedule workspace opts into the
    policy, while uniform non-varlen calls continue to use the legacy path.
    """

    if not schedule_requested:
        return False
    # TileQ16 is the guarded small-Q geometry.  Its direct grid is cheaper
    # than entering the weighted metadata path (the latter can hit the slow
    # serial scan once the number of Q works exceeds one partition group).
    # Long-Q geometries retain the aggregate-work threshold below.
    if tile_q is not None and int(tile_q) <= 16:
        return False
    return int(q_work_count) >= _msa_schedule_q_work_threshold(valid_k_tiles)


def gen_msa_maxscore_spec(
    dtype: torch.dtype,
    *,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    max_seqlen_q: int,
    page_table_kind: str | None = None,
    page_size: int = 128,
    is_varlen: bool = True,
    has_metadata: bool | None = None,
) -> JitSpec:
    config = make_msa_maxscore_config(
        dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        max_seqlen_q=max_seqlen_q,
        page_table_kind=page_table_kind,
        page_size=page_size,
        is_varlen=is_varlen,
        has_metadata=has_metadata,
    )
    return _gen_msa_maxscore_spec_from_config(config)


def _gen_msa_maxscore_spec_from_config(config: Mapping[str, object]) -> JitSpec:
    dispatch_name = _msa_maxscore_encode_config(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / f"{dispatch_name}.mu")
    return gen_jit_spec(
        dispatch_name,
        [source_file],
        generated_sources={source_file: _render_msa_maxscore_kernel_source(config)},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


# Loaded maxscore modules, keyed by dispatch name (the kernel configuration).
_MSA_MAXSCORE_MODULES: dict[str, object] = {}


def get_msa_maxscore_module(
    dtype: torch.dtype,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    max_seqlen_q: int,
    page_table_kind: str | None = None,
    page_size: int = 128,
    is_varlen: bool = True,
    has_metadata: bool | None = None,
):
    # max_seqlen_q only selects the tile geometry (_maxscore_tile_q), so cache
    # the loaded module by the kernel configuration it resolves to. Caching by
    # max_seqlen_q itself sent every new query length back through
    # JitSpec.build_and_load: a file lock shared by every rank on the host, a
    # no-op ninja run and a reload, about 0.5 s per new prefill length at 8
    # ranks per node.
    config = make_msa_maxscore_config(
        dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        max_seqlen_q=max_seqlen_q,
        page_table_kind=page_table_kind,
        page_size=page_size,
        is_varlen=is_varlen,
        has_metadata=has_metadata,
    )
    dispatch_name = _msa_maxscore_encode_config(config)
    module = _MSA_MAXSCORE_MODULES.get(dispatch_name)
    if module is None:
        module = _gen_msa_maxscore_spec_from_config(config).build_and_load()
        _MSA_MAXSCORE_MODULES[dispatch_name] = module
    return module


def _msa_maxscore(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    qo_offset: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    page_size: int | None = None,
    is_varlen: bool = True,
    page_table: Optional[torch.Tensor] = None,
    kv_page_indptr: Optional[torch.Tensor] = None,
    schedule_metadata: Optional[torch.Tensor] = None,
    schedule_metadata_ready: bool = False,
    max_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if schedule_metadata_ready and schedule_metadata is None:
        raise ValueError(
            "schedule_metadata_ready requires a supplied schedule_metadata tensor"
        )
    if schedule_metadata_ready and not schedule_metadata.is_contiguous():
        raise ValueError(
            "schedule_metadata_ready requires the exact contiguous schedule workspace"
        )
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    cu_seqlens_k = maybe_contiguous(cu_seqlens_k)
    qo_offset = maybe_contiguous(qo_offset)
    page_table = maybe_contiguous(page_table)
    kv_page_indptr = maybe_contiguous(kv_page_indptr)
    if q.ndim != 3:
        raise ValueError(f"msa maxscore expects packed q with rank 3, got {q.shape}")
    if q.dtype != k.dtype:
        raise TypeError(f"q and k must share dtype, got {q.dtype} and {k.dtype}")
    if q.dtype not in MSA_DTYPE_MUTLASS:
        raise TypeError(f"unsupported MSA maxscore dtype: {q.dtype}")
    is_paged_kv = page_table is not None
    if is_paged_kv:
        if page_size is None:
            page_size = int(k.shape[1])
        if int(page_size) != int(k.shape[1]):
            raise ValueError(
                f"page_size must match paged K layout, got {page_size} and {int(k.shape[1])}"
            )
        if kv_page_indptr is not None:
            if page_table.ndim != 1:
                raise ValueError("flat paged maxscore requires a rank-1 page_table")
            page_table_kind = "flat"
        elif page_table.ndim == 2:
            page_table_kind = "batched2d"
        else:
            raise ValueError(
                "paged maxscore expects a rank-2 page_table or flat page indices "
                "with kv_page_indptr"
            )
    else:
        page_size = 128 if page_size is None else int(page_size)
        page_table_kind = "dense"
    if kv_page_indptr is not None:
        if page_table is None:
            raise ValueError("kv_page_indptr requires flat page indices")
        if page_table.ndim != 1:
            raise ValueError(
                "flat paged maxscore expects rank-1 page indices when "
                "kv_page_indptr is provided"
            )
        if kv_page_indptr.ndim != 1:
            raise ValueError("kv_page_indptr must be rank 1")
    expected_k_rank = 4 if is_paged_kv else 3
    if k.ndim != expected_k_rank:
        raise ValueError(
            f"msa maxscore expected rank-{expected_k_rank} k, got shape {k.shape}"
        )
    total_q, num_qo_heads, head_dim = (
        int(q.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
    )
    num_kv_heads = int(k.shape[2] if is_paged_kv else k.shape[1])
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "num_qo_heads must be divisible by num_kv_heads, got "
            f"{num_qo_heads} and {num_kv_heads}"
        )
    head_ratio = num_qo_heads // num_kv_heads
    if int(k.shape[-1]) != head_dim:
        raise ValueError(f"q/k head_dim mismatch: {head_dim} vs {int(k.shape[-1])}")

    valid_k_tiles = (int(max_seqlen_k) + int(page_size) - 1) // int(page_size)
    max_k_tiles = _maxscore_k_tiles(max_seqlen_k, int(page_size))
    if max_score is None:
        max_score = torch.full(
            (total_q, num_qo_heads, max_k_tiles),
            -torch.inf,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        max_score = maybe_contiguous(max_score)
        expected_shape = (total_q, num_qo_heads, max_k_tiles)
        if tuple(max_score.shape) != expected_shape:
            raise ValueError(
                f"max_score shape must be {expected_shape}, got {tuple(max_score.shape)}"
            )
        if max_score.dtype != torch.float32:
            raise TypeError(f"max_score must be torch.float32, got {max_score.dtype}")
        # Every specialization may skip sequence-local or padded K tiles.
        # Always poison a caller-provided buffer so ragged noncausal batches do
        # not retain values from an earlier invocation.
        max_score.fill_(-torch.inf)

    if total_q == 0 or valid_k_tiles == 0:
        return max_score

    # Small Q work sets have no useful range to balance: the legacy grid
    # scheduler can launch them directly, avoiding the extra metadata CTA and
    # the fixed persistent-grid prologue.  Keep IsVarlen so cu-seqlens remain
    # supported, but decouple it from HasMetadata for this decode fast path.
    q_work_count = 0
    if is_varlen or schedule_metadata is not None:
        config_for_work = make_msa_maxscore_config(
            q.dtype,
            is_paged_kv=is_paged_kv,
            causal=causal,
            head_dim=head_dim,
            head_ratio=head_ratio,
            max_seqlen_q=int(max_seqlen_q),
            page_table_kind=page_table_kind,
            page_size=int(page_size),
            is_varlen=bool(is_varlen),
            has_metadata=False,
        )
        q_tokens_per_tile = cast(int, config_for_work["tile_q"]) // max(1, head_ratio)
        q_tiles = (int(max_seqlen_q) + q_tokens_per_tile - 1) // q_tokens_per_tile
        q_work_count = max(1, (int(cu_seqlens_q.numel()) - 1) * q_tiles * num_kv_heads)
    use_schedule_metadata = _msa_should_use_schedule(
        q_work_count=q_work_count,
        valid_k_tiles=valid_k_tiles,
        schedule_requested=schedule_metadata is not None or bool(is_varlen),
        tile_q=cast(int, config_for_work["tile_q"]) if q_work_count else None,
    )
    config = make_msa_maxscore_config(
        q.dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        max_seqlen_q=int(max_seqlen_q),
        page_table_kind=page_table_kind,
        page_size=int(page_size),
        is_varlen=bool(is_varlen),
        has_metadata=use_schedule_metadata,
    )
    dispatch_name = _msa_maxscore_encode_config(config)
    module = get_msa_maxscore_module(
        q.dtype,
        is_paged_kv,
        bool(causal),
        head_dim,
        head_ratio,
        int(max_seqlen_q),
        page_table_kind,
        int(page_size),
        bool(is_varlen),
        use_schedule_metadata,
    )
    if skip_kernel_launch_if_dry_run():
        return max_score
    if use_schedule_metadata:
        hardware_mps = resolve_num_mps(q.device, None)
        schedule_num_mps = _msa_schedule_num_mps(
            hardware_mps=hardware_mps,
            q_work_count=q_work_count,
            max_k_tiles=valid_k_tiles,
            parallel_k_tiles=bool(config["parallel_k_tiles"]),
        )
        if schedule_metadata is None:
            schedule_metadata = torch.empty(
                (schedule_num_mps, 2), dtype=torch.int32, device=q.device
            )
        else:
            if not schedule_metadata_ready:
                schedule_metadata = maybe_contiguous(schedule_metadata)
            if schedule_metadata.dtype != torch.int32:
                raise TypeError("schedule_metadata must be int32")
            if schedule_metadata.ndim != 2 or schedule_metadata.shape[1] != 2:
                raise ValueError("schedule_metadata must have shape [num_mps, 2]")
            if (
                schedule_metadata.shape[0] <= 0
                or schedule_metadata.shape[0] > hardware_mps
            ):
                raise ValueError(
                    "schedule_metadata MP count is outside the active device range"
                )
            schedule_rows = int(schedule_metadata.shape[0])
            kernel_k_partitions = _msa_k_partitions_for_mps(
                num_mps=hardware_mps,
                q_work_count=q_work_count,
                max_k_tiles=valid_k_tiles,
                parallel_k_tiles=bool(config["parallel_k_tiles"]),
            )
            schedule_k_partitions = _msa_k_partitions_for_mps(
                num_mps=schedule_rows,
                q_work_count=q_work_count,
                max_k_tiles=valid_k_tiles,
                parallel_k_tiles=bool(config["parallel_k_tiles"]),
            )
            if schedule_k_partitions != kernel_k_partitions:
                valid_row_counts = " or ".join(
                    str(rows) for rows in sorted({schedule_num_mps, hardware_mps})
                )
                raise ValueError(
                    "schedule_metadata row count changes max-score K partitioning: "
                    f"{schedule_rows} rows select {schedule_k_partitions} partitions, "
                    f"but the kernel uses {kernel_k_partitions}; use {valid_row_counts} rows"
                )

        if not schedule_metadata_ready:
            module.get_function(f"{dispatch_name}_metadata")(
                cu_seqlens_q,
                cu_seqlens_k,
                qo_offset,
                schedule_metadata,
                int(max_seqlen_q),
                int(max_seqlen_k),
                num_kv_heads,
            )
    kernel = module.get_function(dispatch_name)
    kernel(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        qo_offset,
        schedule_metadata if use_schedule_metadata else None,
        page_table,
        kv_page_indptr,
        max_score,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(causal),
    )
    return max_score


def gen_msa_sparse_topk_select_spec() -> JitSpec:
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / "sparse_topk_select.mu")
    return gen_jit_spec(
        "msa_sparse_topk_select",
        [source_file],
        generated_sources={source_file: MSA_SPARSE_TOPK_SELECT_TEMPLATE.render({})},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def get_msa_sparse_topk_select_module():
    return gen_msa_sparse_topk_select_spec().build_and_load()


def _msa_sparse_topk_select(
    max_score: torch.Tensor,
    output_indices: torch.Tensor,
    *,
    topk: int,
    num_valid_pages: int,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    force_blocks_count_in_topk: bool = True,
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    max_score = maybe_contiguous(max_score)
    output_indices = maybe_contiguous(output_indices)
    kernel = get_msa_sparse_topk_select_module().get_function("sparse_topk_select")
    if skip_kernel_launch_if_dry_run():
        return output_indices
    kernel(
        max_score,
        output_indices,
        int(topk),
        int(num_valid_pages),
        int(force_begin_blocks),
        int(force_end_blocks),
        bool(force_blocks_count_in_topk),
        query_positions,
    )
    return output_indices


@lru_cache(maxsize=256)
def _resolve_fwd_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        if _FP8_E4M3_DTYPE is None:
            raise RuntimeError("torch.float8_e4m3fn is unavailable")
        return _FP8_E4M3_DTYPE
    if dtype not in MSA_FWD_DTYPE_CONFIG:
        raise TypeError(f"unsupported MSA forward dtype: {dtype}")
    return dtype


@lru_cache(maxsize=256)
def _msa_fwd_encode(
    *, dtype: torch.dtype, output_dtype: torch.dtype, causal: bool
) -> str:
    dtype_name = MSA_FWD_DTYPE_CONFIG[dtype]["name"]
    output_name = MSA_FWD_DTYPE_CONFIG[output_dtype]["name"]
    output_suffix = "" if output_dtype == dtype else f"_out_{output_name}"
    return f"msa_fwd_{dtype_name}{output_suffix}_causal_{int(causal)}"


@lru_cache(maxsize=256)
def _resolve_fwd_output_dtype(
    dtype: torch.dtype, output_dtype: Optional[torch.dtype]
) -> torch.dtype:
    if output_dtype is None:
        return dtype
    if output_dtype not in MSA_FWD_DTYPE_CONFIG:
        raise TypeError(f"unsupported MSA forward output dtype: {output_dtype}")
    return output_dtype


@lru_cache(maxsize=256)
def make_msa_fwd_config(
    *,
    causal: bool,
    dtype: Optional[torch.dtype] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> dict[str, object]:
    dtype = _resolve_fwd_dtype(dtype)
    output_dtype = _resolve_fwd_output_dtype(dtype, output_dtype)
    input_config = MSA_FWD_DTYPE_CONFIG[dtype]
    output_config = MSA_FWD_DTYPE_CONFIG[output_dtype]
    return {
        **input_config,
        "output_name": output_config["name"],
        "element_output": output_config["element"],
        "element_output_dtype": output_config["element_dtype"],
        "causal": bool(causal),
        "func_name": _msa_fwd_encode(
            dtype=dtype, output_dtype=output_dtype, causal=causal
        ),
        "head_ratio": 16,
        "head_dim": 128,
        "tile_kv": 128,
        "topk": 16,
    }


def _render_msa_fwd_kernel_source(config: dict[str, object]) -> str:
    return TVM_HEADER + MSA_FWD_KERN_TEMPLATE.render(config)


def gen_msa_fwd_spec(
    *,
    causal: bool,
    dtype: Optional[torch.dtype] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> JitSpec:
    dtype = _resolve_fwd_dtype(dtype)
    output_dtype = _resolve_fwd_output_dtype(dtype, output_dtype)
    config = make_msa_fwd_config(causal=causal, dtype=dtype, output_dtype=output_dtype)
    dispatch_name = _msa_fwd_encode(
        dtype=dtype, output_dtype=output_dtype, causal=causal
    )
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / f"{dispatch_name}.mu")
    return gen_jit_spec(
        dispatch_name,
        [source_file],
        generated_sources={source_file: _render_msa_fwd_kernel_source(config)},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def get_msa_fwd_module(dtype: torch.dtype, output_dtype: torch.dtype, causal: bool):
    return gen_msa_fwd_spec(
        dtype=dtype, output_dtype=output_dtype, causal=causal
    ).build_and_load()


def _require_int32_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    value = maybe_contiguous(value)
    if value.ndim != 1 or value.dtype != torch.int32:
        raise TypeError(f"{name} must be a contiguous rank-1 int32 tensor")
    return value


def _msa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    qo_offset: torch.Tensor,
    page_table: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    kv_page_indptr: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)
    kv_block_indexes = maybe_contiguous(kv_block_indexes)
    page_table = maybe_contiguous(page_table)
    cu_seqlens_q = _require_int32_vector("cu_seqlens_q", cu_seqlens_q)
    seqused_k = _require_int32_vector("seqused_k", seqused_k)
    qo_offset = _require_int32_vector("qo_offset", qo_offset)
    if kv_page_indptr is not None:
        kv_page_indptr = _require_int32_vector("kv_page_indptr", kv_page_indptr)

    if q.dtype not in MSA_FWD_DTYPE_CONFIG:
        raise TypeError("MSA forward supports float8_e4m3fn, float16, and bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must share the same dtype")
    if q.ndim != 3 or int(q.shape[-1]) != 128:
        raise ValueError(f"q must have shape [total_q, Hq, 128], got {tuple(q.shape)}")
    if k.ndim != 4 or v.ndim != 4 or tuple(k.shape) != tuple(v.shape):
        raise ValueError("k and v must share shape [num_pages, 128, Hkv, 128]")
    if int(k.shape[1]) != 128 or int(k.shape[-1]) != 128:
        raise ValueError(f"paged k/v must use page_size=D=128, got {tuple(k.shape)}")
    k_scale = float(k_scale)
    v_scale = float(v_scale)
    if not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError(f"k_scale must be finite and positive, got {k_scale}")
    if not math.isfinite(v_scale) or v_scale <= 0.0:
        raise ValueError(f"v_scale must be finite and positive, got {v_scale}")

    total_q = int(q.shape[0])
    num_q_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[2])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("MSA SQMMA forward requires Hq to be divisible by Hkv")
    if num_q_heads // num_kv_heads not in (8, 16):
        raise ValueError("MSA SQMMA forward requires local Hq/Hkv ratio 8 or 16")
    expected_blocks_shape = (total_q, num_kv_heads, 16)
    if tuple(kv_block_indexes.shape) != expected_blocks_shape:
        raise ValueError(
            f"kv_block_indexes must have shape {expected_blocks_shape}, "
            f"got {tuple(kv_block_indexes.shape)}"
        )
    if kv_block_indexes.dtype != torch.int32:
        raise TypeError("kv_block_indexes must use torch.int32")

    batch_size = int(cu_seqlens_q.numel() - 1)
    if tuple(seqused_k.shape) != (batch_size,):
        raise ValueError("seqused_k must contain one value per batch")
    if tuple(qo_offset.shape) != (batch_size,):
        raise ValueError("qo_offset must contain one value per batch")
    if kv_page_indptr is not None:
        if page_table.ndim != 1 or int(kv_page_indptr.numel()) != batch_size + 1:
            raise ValueError(
                "flat page_table requires kv_page_indptr with batch_size + 1 entries"
            )
    elif page_table.ndim != 2 or int(page_table.shape[0]) != batch_size:
        raise ValueError(
            "page_table must be [batch, max_pages] when kv_page_indptr is absent"
        )
    elif int(page_table.shape[1]) != (int(max_seqlen_k) + 127) // 128:
        raise ValueError(
            "page_table width must equal ceil(max_seqlen_k / 128) for the "
            "MSA forward TME descriptor"
        )
    if page_table.dtype != torch.int32:
        raise TypeError("page_table must use torch.int32")

    if out is None:
        out = torch.empty_like(q)
    elif tuple(out.shape) != tuple(q.shape):
        raise ValueError("out must have the same shape as q")
    output_dtype = _resolve_fwd_output_dtype(q.dtype, out.dtype)
    if lse is None:
        lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=q.device)
    elif tuple(lse.shape) != (total_q, num_q_heads) or lse.dtype != torch.float32:
        raise ValueError("lse must have shape [total_q, Hq] and dtype float32")

    config = make_msa_fwd_config(
        causal=causal, dtype=q.dtype, output_dtype=output_dtype
    )
    dispatch_name = _msa_fwd_encode(
        dtype=q.dtype,
        output_dtype=output_dtype,
        causal=bool(config["causal"]),
    )
    kernel = get_msa_fwd_module(q.dtype, output_dtype, causal).get_function(
        dispatch_name
    )
    if skip_kernel_launch_if_dry_run():
        return out, lse
    kernel(
        q,
        k,
        v,
        kv_block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        kv_page_indptr,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(softmax_scale if softmax_scale is not None else 128**-0.5),
        k_scale,
        v_scale,
        bool(causal),
        out,
        lse,
    )
    return out, lse


def gen_msa_ops_spec() -> JitSpec:
    return gen_msa_fwd_spec(causal=True)


def gen_msa_ops_aot() -> list[JitSpec]:
    dtypes = [torch.float16, torch.bfloat16]
    if _FP8_E4M3_DTYPE is not None:
        dtypes.append(_FP8_E4M3_DTYPE)
    dtype_pairs = [(dtype, dtype) for dtype in dtypes]
    if _FP8_E4M3_DTYPE is not None:
        dtype_pairs.append((_FP8_E4M3_DTYPE, torch.bfloat16))
    specs = [
        gen_msa_fwd_spec(dtype=dtype, output_dtype=output_dtype, causal=causal)
        for dtype, output_dtype in dtype_pairs
        for causal in (False, True)
    ]
    specs.append(gen_msa_sparse_topk_select_spec())
    return specs


@functools.cache
def get_msa_ops_module():
    if _FP8_E4M3_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    return get_msa_fwd_module(_FP8_E4M3_DTYPE, _FP8_E4M3_DTYPE, causal=True)
