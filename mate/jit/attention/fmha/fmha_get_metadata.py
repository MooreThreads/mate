import json
import math
import pathlib
import torch
from typing import Optional, Dict, Any
from functools import lru_cache

import tvm_ffi.cpp

from ... import env as jit_env
from .fmha_utils import fmha_template_env
from .fmha_utils import _get_fwd_kernel_config as _get_metadata_kernel_config
from ...utils import TVM_HEADER, EXPORT_FUNC, update_aot_mod
from ...configs import KernelConfigGraph, ParamSpec, domain_by_case


kern_metadata = fmha_template_env.get_template("metadata_kern.j2")


@lru_cache()
def _fmha_get_metadata_encode(config_tuple: tuple) -> str:
    config = dict(config_tuple)
    name_list = ["fmha_get_metadata"]
    head_ratio, num_warps = config["head_ratio"], config["num_warps"]
    name_list.append(f"{head_ratio}x{num_warps}")
    if config["has_cu_seqlens_q"]:
        mode_q = "ragged_q"
        name_list.append(mode_q)
    elif config["has_seqused_q"]:
        mode_q = "padded_q"
        name_list.append(mode_q)

    if config["has_cu_seqlens_k"]:
        mode_k = "ragged_k"
        name_list.append(mode_k)
    elif config["has_seqused_k"]:
        mode_k = "padded_k"
        name_list.append(mode_k)

    if config["is_causal"]:
        mode_causal = "causal"
        name_list.append(mode_causal)
    if config["is_packgqa"]:
        mode_packgqa = "packgqa"
        name_list.append(mode_packgqa)

    name = "_".join(name_list)

    return name


# module cache: dispatch_name -> compiled module
_FMHA_METADATA_MOD_CACHE: Dict[str, Any] = {}


def config_selector(cfg):
    return cfg["config_level"]


_CONFIG_TABLE: Dict[int, Dict[str, Any]] = {
    1: {
        "mode_q": ["ragged"],
        "mode_k": ["ragged", "padded"],
    },
    2: {
        "mode_q": ["normal", "ragged"],
        "mode_k": ["normal", "ragged", "padded"],
    },
}

base_specs = []
mode_q = [
    ParamSpec(
        name="mode_q",
        domain=domain_by_case(config_selector, _CONFIG_TABLE, "mode_q"),
        default="normal",
        depends_on=("config_level",),
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
mode_k = [
    ParamSpec(
        name="mode_k",
        domain=domain_by_case(config_selector, _CONFIG_TABLE, "mode_k"),
        default="normal",
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="has_cu_seqlens_k",
        default=False,
        compute=lambda cfg: cfg["mode_k"] == "ragged",
        depends_on=("mode_k",),
        sweep=False,
    ),
    ParamSpec(
        name="has_seqused_k",
        default=False,
        compute=lambda cfg: cfg["mode_k"] == "padded",
        depends_on=("mode_k",),
        sweep=False,
    ),
    ParamSpec(  # NOTE:  Change me when supported.
        name="has_cu_seqlens_k_new",
        domain=[False],
        default=False,
    ),
    ParamSpec(  # NOTE:  Change me when supported.
        name="has_leftpad_k",
        domain=[False],
        default=False,
    ),
]
mode_mask = [
    ParamSpec(
        name="mode_mask",
        domain=["none", "causal"],
        default="none",
        export=False,
    ),
    ParamSpec(
        name="is_causal",
        default=False,
        compute=lambda cfg: cfg["mode_mask"] == "causal",
        depends_on=("mode_mask",),
        sweep=False,
    ),
]
score_mode = [
    ParamSpec(  # NOTE:  Change me when supported.
        name="has_softcap",
        domain=[False],
    ),
]
specs_attn = [
    ParamSpec(
        name="is_packgqa",
        domain=[False, True],
    ),
    ParamSpec(
        name="head_ratio",
        domain=[1, 2, 4, 5, 8, 12, 16],
    ),
]
specs_metadata = [
    ParamSpec(
        name="sort",
        domain=[True],
        default=True,
    ),
    ParamSpec(
        name="num_warps",
        domain=list(range(1, 6)),  # batch 1 - 31 * 5
        default=1,
    ),
]
base_specs.extend(mode_q)
base_specs.extend(mode_k)
base_specs.extend(mode_mask)
base_specs.extend(score_mode)
base_specs.extend(specs_attn)
base_specs.extend(specs_metadata)


def _gen_specs_from_config(cfg_level: int):
    config = [
        ParamSpec(
            name="config_level",
            domain=[cfg_level],
            default=1,
            export=False,
        ),
    ]

    specs = []
    specs.extend(base_specs)
    specs.extend(config)

    config_graph = KernelConfigGraph(specs)
    aot_fwd_configs = config_graph.resolve_and_expand()

    return aot_fwd_configs


def _get_aot_configs():
    cfg_level_key = "attention.fmha.metadata".split(".")
    try:
        aot_level_file = pathlib.Path(__file__).parents[2] / "aot-levels.json"
        with open(aot_level_file.resolve(), "r") as f:
            config_level = json.load(f)
        for key in cfg_level_key:
            config_level = config_level[key]
    except FileNotFoundError:
        # Default to 0 if file doesn't exist
        config_level = 0

    if config_level == 0:
        return []

    configs = _gen_specs_from_config(config_level)

    return configs


aot_metadata_configs = _get_aot_configs()
update_aot_mod(
    aot_metadata_configs,
    _fmha_get_metadata_encode,
    jit_env.MATE_AOT_DIR / "fmha_get_metadata_aot.so",
    _FMHA_METADATA_MOD_CACHE,
)


def _fmha_get_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: Optional[int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    causal: bool = False,
    num_splits: int = 1,
    packgqa: Optional[bool] = None,
    mp_margin: int = 0,
) -> torch.Tensor:
    if cu_seqlens_q is not None:
        assert max_seqlen_q is not None

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )

    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )

    assert all(
        t is None or t.is_musa
        for t in (
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )
    ), "inputs must be on MUSA device"

    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )

    qhead_per_kvhead = num_heads_q // num_heads_kv

    # Metadata
    metadata = torch.empty((batch_size * 4), dtype=torch.int32, device="musa")
    (
        num_splits_dynamic,
        batch_table,
        num_m_blocks,
        num_nheads_in_l2,
    ) = (metadata[batch_size * i : batch_size * (i + 1)] for i in range(4))

    (
        tile_m,
        tile_n,
        stages_k,
        stages_v,
        headdim_rounded,
        headdim_v_rounded,
        consumers_qk,
        consumers_pv,
        enable_packgqa,
    ) = _get_metadata_kernel_config(
        max_seqlen_q,
        qhead_per_kvhead,
        headdim,
        headdim_v,
        packgqa,
    )

    packgqa = enable_packgqa if packgqa is None else packgqa

    constexpr_dict = {
        "has_cu_seqlens_q": cu_seqlens_q is not None,
        "has_cu_seqlens_k": cu_seqlens_k is not None,
        "has_seqused_q": seqused_q is not None,
        "has_seqused_k": seqused_k is not None,
        "has_cu_seqlens_k_new": False,
        "has_leftpad_k": False,
        "is_causal": causal,
        "is_packgqa": packgqa,
        "head_ratio": qhead_per_kvhead,
        "sort": True,
        "num_warps": math.ceil(batch_size / 31),
    }

    dispatch_name = _fmha_get_metadata_encode(tuple(constexpr_dict.items()))

    if dispatch_name in _FMHA_METADATA_MOD_CACHE:
        mod = _FMHA_METADATA_MOD_CACHE[dispatch_name]
    else:
        constexpr_dict["func_name"] = dispatch_name
        # print("Compiling metadata kernel JIT...", flush=True)
        mod = tvm_ffi.cpp.load_inline(
            name="fmha_get_metadata",
            cuda_sources=kern_metadata.render(constexpr_dict),
            functions=[dispatch_name],
            extra_include_paths=[
                jit_env.MATE_INCLUDE_DIR.as_posix(),
                jit_env.MATE_CSRC_DIR.as_posix(),
                jit_env.MUTLASS_INCLUDE_DIR.as_posix(),
            ],
            extra_ldflags=["-lmusa"],
            extra_cuda_cflags=[
                "-Od3",
                "-DNDEBUG",
                "-fno-strict-aliasing",
                "-fno-signed-zeros",
                "-mllvm",
                "-mtgpu-load-cluster-mutation=1",
                "-mllvm",
                "--num-dwords-of-load-in-mutation=64",
            ],
        )
        _FMHA_METADATA_MOD_CACHE[dispatch_name] = mod
        # print("Done.")

    fmha_metadata_impl = mod.get_function(dispatch_name)

    fmha_metadata_impl(
        batch_size,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        num_splits_dynamic,
        batch_table,
        num_m_blocks,
        num_nheads_in_l2,
        num_splits,
        tile_m,
        tile_n,
        mp_margin,
    )
    return metadata


def gen_fmha_metadata_aot(dry_run: bool = False) -> int:
    configs = _get_aot_configs()

    musa_instances = []
    musa_set = set()

    for config in configs:
        dispatch_name = _fmha_get_metadata_encode(tuple(config.items()))
        if dispatch_name in musa_set:
            continue
        musa_set.add(dispatch_name)
        config["func_name"] = dispatch_name

        mu_path = jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu"
        with open(mu_path, "w") as f:
            f.write(
                TVM_HEADER + kern_metadata.render(config) + EXPORT_FUNC.render(config)
            )
        musa_instances.append(mu_path)

    if not musa_instances:
        return 0

    if dry_run:
        return len(musa_instances)
    print(f"MATE AOT Building {len(musa_instances)} fmha_get_metadata...")
    tvm_ffi.cpp.build(
        name="fmha_get_metadata_aot",
        cuda_files=musa_instances,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR.as_posix(),
            jit_env.MATE_CSRC_DIR.as_posix(),
            jit_env.MUTLASS_INCLUDE_DIR.as_posix(),
        ],
        extra_cuda_cflags=[
            "-Od3",
            "-DNDEBUG",
            "-fno-strict-aliasing",
            "-fno-signed-zeros",
            "-mllvm",
            "-mtgpu-load-cluster-mutation=1",
            "-mllvm",
            "--num-dwords-of-load-in-mutation=64",
        ],
        extra_ldflags=["-lmusa"],
        build_directory=jit_env.MATE_AOT_DIR.as_posix(),
    )
    print(
        f"MATE AOT fmha_get_metadata Done -> {(jit_env.MATE_AOT_DIR / 'fmha_get_metadata_aot.so').as_posix()}."
    )

    return len(musa_instances)
