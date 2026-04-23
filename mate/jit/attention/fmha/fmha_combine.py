from functools import lru_cache
import functools
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import dtype_torch2mutlass_map, TVM_HEADER, EXPORT_FUNC
from ...configs import KernelConfigGraph, ParamSpec
from .fmha_utils import (
    FMHA_EXTRA_CUDA_CFLAGS,
    fmha_extra_include_paths,
    get_fmha_template,
)


def _fmha_fwd_combine_encode(config: Mapping[str, object]) -> str:
    name_list = ["fmha_fwd_combine"]
    if config["element"] == "mutlass::half_t":
        name_list.append("f16")
    elif config["element"] == "mutlass::bfloat16_t":
        name_list.append("bf16")
    else:
        raise ValueError(f"Unsupported element type: {config['element']}")

    name_list.append(f"{config['tile_m']}x{config['tile_n']}x{config['max_splits']}")
    if config["has_cu_seqlens_q"]:
        mode = "ragged_q"
        name_list.append(mode)
    elif config["has_seqused_q"]:
        mode = "padded_q"
        name_list.append(mode)
    return "_".join(name_list)


def _render_fmha_fwd_combine_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = str(
        render_config.get("func_name") or _fmha_fwd_combine_encode(render_config)
    )
    return (
        TVM_HEADER
        + get_fmha_template("combine_kern.j2").render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


@lru_cache
def _get_fwd_combine_kernel_config(tile_n: int, num_split: int):
    assert tile_n % 32 == 0, "tile_n must be multiple of 32"
    if tile_n % 128 == 0:
        tile_m = 8
    elif tile_n % 64 == 0:
        tile_m = 16
    else:
        tile_m = 32

    if tile_m >= 16 and num_split <= 16:
        max_splits = 16
    elif num_split <= 32:
        max_splits = 32
    elif num_split <= 64:
        max_splits = 64
    elif num_split <= 128:
        max_splits = 128
    else:
        raise ValueError("num_split exceeds max supported splits 128")

    return tile_m, max_splits


def _fmha_fwd_combine_module(config: Mapping[str, object]):
    dispatch_name = _fmha_fwd_combine_encode(config)
    return dispatch_name, _load_fmha_fwd_combine_module(tuple(sorted(config.items())))


@functools.cache
def _load_fmha_fwd_combine_module(frozen_config: tuple[tuple[str, object], ...]):
    return gen_fmha_fwd_combine_spec(dict(frozen_config)).build_and_load()


specs = []
mode_q = [
    ParamSpec(
        name="mode_q",
        domain=["padded", "ragged", "normal"],
        default="normal",
        export=False,
    ),
    ParamSpec(
        name="has_cu_seqlens_q",
        default=False,
        compute=lambda cfg: cfg["mode_q"] == "ragged",
        depends_on=("mode_q",),
        sweep=False,
    ),
    ParamSpec(
        name="has_seqused_q",
        default=False,
        compute=lambda cfg: cfg["mode_q"] == "padded",
        depends_on=("mode_q",),
        sweep=False,
    ),
]
specs.append(
    ParamSpec(
        name="element",
        domain=[dtype_torch2mutlass_map[x] for x in [torch.bfloat16]],
    )
)
specs_select = [
    ParamSpec(
        name="tile_n",
        domain=[64],
    ),
    ParamSpec(
        name="tile_m",
        domain=[16],
    ),
    ParamSpec(
        name="max_splits",
        domain=[16, 32, 64, 128],
    ),
]
specs.extend(mode_q)
specs.extend(specs_select)
aot_combine_configs = KernelConfigGraph(specs).resolve_and_expand()


def get_fmha_fwd_combine_aot_configs(config_level: int) -> list[dict[str, object]]:
    if config_level != 1:
        return []
    return aot_combine_configs


def gen_fmha_fwd_combine_spec(config: Mapping[str, object]) -> JitSpec:
    dispatch_name = _fmha_fwd_combine_encode(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu")
    return gen_jit_spec(
        name=dispatch_name,
        sources=[source_file],
        generated_sources={source_file: _render_fmha_fwd_combine_source(config)},
        extra_cuda_cflags=list(FMHA_EXTRA_CUDA_CFLAGS),
        extra_include_paths=fmha_extra_include_paths(),
    )


def gen_fmha_fwd_combine_specs(
    configs: Sequence[Mapping[str, object]],
) -> list[JitSpec]:
    return [gen_fmha_fwd_combine_spec(config) for config in configs]


def gen_fmha_fwd_combine_aot(config_level: int = 0) -> list[JitSpec]:
    return gen_fmha_fwd_combine_specs(get_fmha_fwd_combine_aot_configs(config_level))


def _fmha_fwd_combine(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_accum: torch.Tensor,
    lse_accum: torch.Tensor,
    tile_n: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    num_split: int = 0,
    metadata: Optional[torch.Tensor] = None,
) -> None:
    if metadata is None:
        return

    if cu_seqlens_q is None:
        batch_size = out_accum.shape[0]
    else:
        assert max_seqlen_q is not None
        batch_size = cu_seqlens_q.shape[0] - 1

    assert metadata.shape[0] >= batch_size * 4, "metadata buffer is too small"
    (
        num_splits_dynamic,
        batch_table,
        num_m_blocks,
        num_nheads_in_l2,
    ) = (metadata[batch_size * i : batch_size * (i + 1)] for i in range(4))

    tile_m, max_splits = _get_fwd_combine_kernel_config(tile_n, num_split=num_split)
    constexpr_dict = {
        "has_cu_seqlens_q": cu_seqlens_q is not None,
        "has_seqused_q": seqused_q is not None,
        "element": dtype_torch2mutlass_map[out.dtype],
        "tile_m": tile_m,
        "tile_n": tile_n,
        "max_splits": max_splits,
    }

    dispatch_name, mod = _fmha_fwd_combine_module(constexpr_dict)
    fmha_fwd_combine_impl = mod.get_function(dispatch_name)
    fmha_fwd_combine_impl(
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        out,
        lse,
        out_accum,
        lse_accum,
        num_splits_dynamic,
        batch_table,
        num_split,
    )
