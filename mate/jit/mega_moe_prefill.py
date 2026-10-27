from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Mapping

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .mega_moe_common import (
    MEGA_MOE_EXTRA_CFLAGS,
    MEGA_MOE_EXTRA_MUSA_CFLAGS,
    MEGA_MOE_LAYOUT_REVISION,
    _align,
    _extra_include_paths,
    _generated_source_path,
    _render_source,
    _normalize_activation,
)


PREFILL_BLOCK_M = 256
PREFILL_BLOCK_N = 256
PREFILL_BLOCK_K = 128
PREFILL_STAGE1_NUM_STAGES = 2
PREFILL_STAGE1_NUM_THREADS = 5 * 128
PREFILL_STAGE2_NUM_THREADS = 5 * 128
PREFILL_STAGE3_NUM_THREADS = 640


def _build_spec(
    kernel_name: str, template: str, render_config: dict[str, object]
) -> JitSpec:
    # Keep the spec declaration local for static test dependency discovery.
    template_dir = jit_env.MATE_TEMPLATE_DIR / "mega_moe"
    source_file = _generated_source_path(kernel_name)
    source = _render_source(kernel_name, template_dir / template, render_config)
    return gen_jit_spec(
        name=kernel_name,
        sources=[source_file],
        generated_sources={source_file: source},
        extra_cflags=list(MEGA_MOE_EXTRA_CFLAGS),
        extra_cuda_cflags=list(MEGA_MOE_EXTRA_MUSA_CFLAGS),
        extra_include_paths=_extra_include_paths(),
    )


@dataclass(frozen=True)
class PrefillMegaMoEKernelConfig:
    num_ranks: int
    num_experts: int
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_mps: int
    fast_math: bool = True
    num_max_pool_tokens: int = 0
    # Routed L1 activation ring capacity. 0 keeps the worst-case pool.
    num_l1_ring_tokens: int = 0
    cc: int = 31


def _validate_config(config: PrefillMegaMoEKernelConfig) -> None:
    if config.cc != 31:
        raise ValueError(
            f"MegaMoE prefill currently supports MP31 only, got MP{config.cc}"
        )
    if config.num_ranks <= 0 or config.num_ranks > 8:
        raise ValueError("num_ranks must be in [1, 8]")
    if config.num_experts <= 0 or config.num_experts % config.num_ranks != 0:
        raise ValueError("num_experts must be positive and divisible by num_ranks")
    if config.num_experts // config.num_ranks > 1024:
        raise ValueError("prefill route metadata supports at most 1024 local experts")
    if (
        config.num_max_tokens_per_rank <= 0
        or config.num_max_tokens_per_rank % PREFILL_BLOCK_M != 0
        or config.num_max_tokens_per_rank > 65536
    ):
        raise ValueError(
            "num_max_tokens_per_rank must be 256-aligned and no larger than 65536"
        )
    if config.num_topk <= 0 or config.num_topk > 32:
        raise ValueError("production prefill supports num_topk in [1, 32]")
    if config.num_topk > config.num_experts:
        raise ValueError("num_topk must not exceed num_experts")
    if config.hidden <= 0 or config.hidden % PREFILL_BLOCK_N != 0:
        raise ValueError("hidden must be positive and divisible by 256")
    if (
        config.intermediate_hidden <= 0
        or config.intermediate_hidden % PREFILL_BLOCK_N != 0
        or config.intermediate_hidden > 8192
    ):
        raise ValueError(
            "intermediate_hidden must be 256-aligned and no larger than 8192"
        )
    if config.num_mps <= 0:
        raise ValueError("num_mps must be positive")
    worst_case_pool_tokens = _worst_case_num_max_pool_tokens(config)
    if config.num_max_pool_tokens != 0 and (
        config.num_max_pool_tokens <= 0
        or config.num_max_pool_tokens % PREFILL_BLOCK_M != 0
        or config.num_max_pool_tokens > worst_case_pool_tokens
    ):
        raise ValueError(
            "num_max_pool_tokens must be zero or a 256-aligned value no larger "
            "than the worst-case pool"
        )
    if config.num_l1_ring_tokens != 0:
        ring_tokens = config.num_l1_ring_tokens
        pool_tokens = _num_max_pool_tokens(config)
        if ring_tokens % PREFILL_BLOCK_M != 0 or ring_tokens > pool_tokens:
            raise ValueError(
                "num_l1_ring_tokens must be zero or a 256-aligned value no "
                "larger than the prefill pool capacity"
            )
        ring_blocks = ring_tokens // PREFILL_BLOCK_M
        if ring_blocks & (ring_blocks - 1):
            raise ValueError("num_l1_ring_tokens must use a power-of-two block count")


def _normalize_config(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
) -> PrefillMegaMoEKernelConfig:
    if isinstance(config, PrefillMegaMoEKernelConfig):
        result = config
    else:
        result = PrefillMegaMoEKernelConfig(
            num_ranks=int(config["num_ranks"]),
            num_experts=int(config["num_experts"]),
            num_max_tokens_per_rank=int(config["num_max_tokens_per_rank"]),
            num_topk=int(config["num_topk"]),
            hidden=int(config["hidden"]),
            intermediate_hidden=int(config["intermediate_hidden"]),
            num_mps=int(config["num_mps"]),
            fast_math=bool(config.get("fast_math", True)),
            num_max_pool_tokens=int(config.get("num_max_pool_tokens", 0)),
            num_l1_ring_tokens=int(config.get("num_l1_ring_tokens", 0)),
            cc=int(config.get("cc", 31)),
        )
    _validate_config(result)
    return result


def _worst_case_num_max_pool_tokens(config: PrefillMegaMoEKernelConfig) -> int:
    experts_per_rank = config.num_experts // config.num_ranks
    max_routes = (
        config.num_ranks
        * config.num_max_tokens_per_rank
        * min(config.num_topk, experts_per_rank)
    )
    return _align(
        max_routes + experts_per_rank * (PREFILL_BLOCK_M - 1),
        PREFILL_BLOCK_M,
    )


def _num_max_pool_tokens(config: PrefillMegaMoEKernelConfig) -> int:
    if config.num_max_pool_tokens > 0:
        return config.num_max_pool_tokens
    return _worst_case_num_max_pool_tokens(config)


def _num_l1_ring_tokens(config: PrefillMegaMoEKernelConfig) -> int:
    if config.num_l1_ring_tokens > 0:
        return config.num_l1_ring_tokens
    return _num_max_pool_tokens(config)


def _num_owner_ctas(config: PrefillMegaMoEKernelConfig) -> int:
    # Per-MP-count production shapes from the CTA-group sweeps. Every other
    # launch uses 4 cooperating CTAs; the kernel static_assert still requires
    # num_mps % owner_ctas == 0, so each entry below must divide its own
    # part's MP count (56 % 10 and 56 % 12 do not, hence the 56-MP block).
    # 60-MP, worker35091: direct SiTU activation re-sweep. Optimize E2E for
    # the heavier burst4 epilogue with c20 at 4K and c10 at 8K.
    if config.num_mps == 60 and config.num_max_tokens_per_rank == 4096:
        return 20
    if config.num_mps == 60 and config.num_max_tokens_per_rank == 8192:
        return 10
    # 56-MP,  worker32016: mcu_results/cta56_20260924
    # 4096: c14 wins on every shape swept. 8192: c8 is best on the served
    # DeepSeek-V4-Flash shape and on the repo benchmark shape; c7 only wins on
    # the older wide-H shape, and c14 gives nothing at 8192.
    if config.num_mps == 56 and config.num_max_tokens_per_rank == 4096:
        return 14
    if config.num_mps == 56 and config.num_max_tokens_per_rank == 8192:
        return 8
    return 4


def _kernel_name(prefix: str, config: PrefillMegaMoEKernelConfig) -> str:
    return (
        f"mate_jit_mega_moe_prefill_{prefix}"
        f"_r{config.num_ranks}_e{config.num_experts}"
        f"_mt{config.num_max_tokens_per_rank}_topk{config.num_topk}"
        f"_h{config.hidden}_ih{config.intermediate_hidden}"
        f"_pool{_num_max_pool_tokens(config)}"
        f"_l1ring{_num_l1_ring_tokens(config)}"
        f"_mps{config.num_mps}_fm{int(config.fast_math)}"
        f"_production_v1_layout{MEGA_MOE_LAYOUT_REVISION}"
    )


def _validate_stage1_profile_mode(profile_mode: str, num_l1_ring_tokens: int) -> None:
    if profile_mode not in ("full", "dispatch_only", "compute_only"):
        raise ValueError("invalid Stage 1 profile mode")
    if profile_mode != "full" and num_l1_ring_tokens != 0:
        raise ValueError(
            "Stage 1 dispatch_only/compute_only requires L1 ring mode to be disabled. "
            "Create the symmetric buffer with prefill_l1_ring=False "
            "(num_l1_ring_tokens=0 for direct JIT callers)."
        )


def _stage1_kernel_name(
    config: PrefillMegaMoEKernelConfig,
    activation: str,
    profile_mode: str,
    use_situ_linear: bool,
) -> str:
    return _kernel_name(
        f"mp{config.cc}_stage1_act{activation}_lin{int(use_situ_linear)}_{profile_mode}_v30_"
        f"dual_sync_barrier_c{_num_owner_ctas(config)}w3",
        config,
    )


def gen_prefill_stage1_spec(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
    *,
    activation: str = "swiglu",
    profile_mode: str = "full",
    use_situ_linear: bool = False,
) -> JitSpec:
    kernel_config = _normalize_config(config)
    activation = _normalize_activation(activation)
    use_situ_linear = bool(use_situ_linear and activation == "situ")
    _validate_stage1_profile_mode(profile_mode, kernel_config.num_l1_ring_tokens)
    owner_ctas = _num_owner_ctas(kernel_config)
    kernel_name = _stage1_kernel_name(
        kernel_config,
        activation,
        profile_mode,
        use_situ_linear,
    )
    pool_tokens = _num_max_pool_tokens(kernel_config)
    l1_ring_tokens = _num_l1_ring_tokens(kernel_config)
    compute_smem_size = _align(
        PREFILL_STAGE1_NUM_STAGES
        * (PREFILL_BLOCK_M * PREFILL_BLOCK_K + PREFILL_BLOCK_N * PREFILL_BLOCK_K)
        + (PREFILL_BLOCK_N // 128) * PREFILL_BLOCK_M * 4,
        4096,
    )
    dispatch_scratch_size = (
        _align(kernel_config.num_experts * 8, 4096)
        + _align(kernel_config.hidden, 4096) * 3
    )
    render_config = {
        **kernel_config.__dict__,
        "func_name": kernel_name,
        "num_experts_per_rank": kernel_config.num_experts // kernel_config.num_ranks,
        "num_max_pool_tokens": pool_tokens,
        "num_padded_sf_pool_tokens": (pool_tokens // PREFILL_BLOCK_M)
        * _align(PREFILL_BLOCK_M, 128),
        "num_l1_ring_tokens": l1_ring_tokens,
        "num_l1_ring_blocks": l1_ring_tokens // PREFILL_BLOCK_M,
        "num_l1_ring_padded_sf_tokens": l1_ring_tokens,
        "num_threads": PREFILL_STAGE1_NUM_THREADS,
        "smem_size": compute_smem_size + dispatch_scratch_size,
        "use_situ": activation == "situ",
        "use_situ_linear": use_situ_linear,
        "dispatch_only": profile_mode == "dispatch_only",
        "compute_only": profile_mode == "compute_only",
        "num_owner_ctas": owner_ctas,
    }
    return _build_spec(kernel_name, "fp8_fp8_prefill_stage1_kern.j2", render_config)


def gen_prefill_stage2_spec(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
) -> JitSpec:
    kernel_config = _normalize_config(config)
    kernel_name = _kernel_name(
        f"mp{kernel_config.cc}_stage2_remote_store_v3_dual_sync_barrier",
        kernel_config,
    )
    pool_tokens = _num_max_pool_tokens(kernel_config)
    l1_ring_tokens = _num_l1_ring_tokens(kernel_config)
    smem_size = PREFILL_STAGE1_NUM_STAGES * (
        PREFILL_BLOCK_M * PREFILL_BLOCK_K + PREFILL_BLOCK_N * PREFILL_BLOCK_K
    )
    render_config = {
        **kernel_config.__dict__,
        "func_name": kernel_name,
        "num_experts_per_rank": kernel_config.num_experts // kernel_config.num_ranks,
        "num_max_pool_tokens": pool_tokens,
        "num_padded_sf_pool_tokens": (pool_tokens // PREFILL_BLOCK_M)
        * _align(PREFILL_BLOCK_M, 128),
        "num_l1_ring_tokens": l1_ring_tokens,
        "num_threads": PREFILL_STAGE2_NUM_THREADS,
        "smem_size": smem_size,
    }
    return _build_spec(kernel_name, "fp8_fp8_prefill_stage2_kern.j2", render_config)


def gen_prefill_stage3_spec(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
) -> JitSpec:
    kernel_config = _normalize_config(config)
    kernel_name = _kernel_name(f"mp{kernel_config.cc}_stage3_reduce", kernel_config)
    pool_tokens = _num_max_pool_tokens(kernel_config)
    l1_ring_tokens = _num_l1_ring_tokens(kernel_config)
    render_config = {
        **kernel_config.__dict__,
        "func_name": kernel_name,
        "num_max_pool_tokens": pool_tokens,
        "num_padded_sf_pool_tokens": (pool_tokens // PREFILL_BLOCK_M)
        * _align(PREFILL_BLOCK_M, 128),
        "num_l1_ring_tokens": l1_ring_tokens,
        "num_threads": PREFILL_STAGE3_NUM_THREADS,
    }
    return _build_spec(kernel_name, "fp8_fp8_prefill_stage3_kern.j2", render_config)


@functools.cache
def _load_prefill_stage1_module(
    config: PrefillMegaMoEKernelConfig,
    activation: str,
    profile_mode: str,
    use_situ_linear: bool,
):
    return gen_prefill_stage1_spec(
        config,
        activation=activation,
        profile_mode=profile_mode,
        use_situ_linear=use_situ_linear,
    ).build_and_load()


@functools.cache
def _load_prefill_stage2_module(
    config: PrefillMegaMoEKernelConfig,
):
    return gen_prefill_stage2_spec(config).build_and_load()


@functools.cache
def _load_prefill_stage3_module(
    config: PrefillMegaMoEKernelConfig,
):
    return gen_prefill_stage3_spec(config).build_and_load()


def get_prefill_stage1_module(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
    *,
    activation: str = "swiglu",
    profile_mode: str = "full",
    use_situ_linear: bool = False,
):
    kernel_config = _normalize_config(config)
    activation = _normalize_activation(activation)
    use_situ_linear = bool(use_situ_linear and activation == "situ")
    return (
        _stage1_kernel_name(
            kernel_config,
            activation,
            profile_mode,
            use_situ_linear,
        ),
        _load_prefill_stage1_module(
            kernel_config,
            activation,
            profile_mode,
            use_situ_linear,
        ),
    )


def get_prefill_stage2_module(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
):
    kernel_config = _normalize_config(config)
    return (
        _kernel_name(
            f"mp{kernel_config.cc}_stage2_remote_store_v3_dual_sync_barrier",
            kernel_config,
        ),
        _load_prefill_stage2_module(kernel_config),
    )


def get_prefill_stage3_module(
    config: Mapping[str, Any] | PrefillMegaMoEKernelConfig,
):
    kernel_config = _normalize_config(config)
    return (
        _kernel_name(f"mp{kernel_config.cc}_stage3_reduce", kernel_config),
        _load_prefill_stage3_module(kernel_config),
    )


__all__ = [
    "PREFILL_BLOCK_M",
    "PREFILL_BLOCK_N",
    "PREFILL_BLOCK_K",
    "PREFILL_STAGE1_NUM_THREADS",
    "PREFILL_STAGE2_NUM_THREADS",
    "PREFILL_STAGE3_NUM_THREADS",
    "PrefillMegaMoEKernelConfig",
    "gen_prefill_stage1_spec",
    "gen_prefill_stage2_spec",
    "gen_prefill_stage3_spec",
    "get_prefill_stage1_module",
    "get_prefill_stage2_module",
    "get_prefill_stage3_module",
]
