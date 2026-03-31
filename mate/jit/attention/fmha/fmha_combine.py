import torch
from typing import Optional, Dict, Any
from functools import lru_cache

import tvm_ffi.cpp

from ... import env as jit_env
from .fmha_utils import fmha_template_env
from ...utils import dtype_torch2mutlass_map, TVM_HEADER, EXPORT_FUNC, update_aot_mod
from ...configs import KernelConfigGraph, ParamSpec


kern_fwd_combine = fmha_template_env.get_template("combine_kern.j2")


@lru_cache()
def _fmha_fwd_combine_encode(config_tuple: tuple) -> str:
    config = dict(config_tuple)

    name_list = ["fmha_fwd_combine"]
    if config["element"] == "mutlass::half_t":
        element = "f16"
    elif config["element"] == "mutlass::bfloat16_t":
        element = "bf16"
    else:
        raise ValueError(f"Unsupported element type: {config['element']}")
    name_list.append(element)

    tile_m, tile_n, max_splits = (
        config["tile_m"],
        config["tile_n"],
        config["max_splits"],
    )
    name_list.append(f"{tile_m}x{tile_n}x{max_splits}")

    if config["has_cu_seqlens_q"]:
        mode = "ragged_q"
        name_list.append(mode)
    elif config["has_seqused_q"]:
        mode = "padded_q"
        name_list.append(mode)

    name = "_".join(name_list)
    return name


@lru_cache
def _get_fwd_combine_kernel_config(tile_n: int, num_split: int):
    assert tile_n % 32 == 0, "tile_n must be multiple of 32"
    if tile_n % 128 == 0:
        tile_m = 8
    elif tile_n % 64 == 0:
        tile_m = 16
    else:
        tile_m = 32

    if tile_m >= 16:
        if num_split <= 16:
            max_splits = 16  # 16
    if num_split <= 32:
        max_splits = 32  # 32
    elif num_split <= 64:
        max_splits = 64  # 64
    elif num_split <= 128:
        max_splits = 128  # 128
    else:
        raise ValueError("num_split exceeds max supported splits 128")
        max_splits = 256  # 256

    return tile_m, max_splits


# module cache: dispatch_name -> compiled module
_FMHA_FWD_COMBINE_MOD_CACHE: Dict[str, Any] = {}

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
update_aot_mod(
    aot_combine_configs,
    _fmha_fwd_combine_encode,
    jit_env.MATE_AOT_DIR / "fmha_fwd_combine_aot.so",
    _FMHA_FWD_COMBINE_MOD_CACHE,
)


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
    # If metadata not provided, No need to combine as no split happens.
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

    dispatch_name = _fmha_fwd_combine_encode(tuple(constexpr_dict.items()))

    if dispatch_name in _FMHA_FWD_COMBINE_MOD_CACHE:
        mod = _FMHA_FWD_COMBINE_MOD_CACHE[dispatch_name]
    else:
        constexpr_dict["func_name"] = dispatch_name
        # print("Compiling combine kernel JIT...", flush=True)
        mod = tvm_ffi.cpp.load_inline(
            name="fmha_fwd_combine",
            cuda_sources=kern_fwd_combine.render(constexpr_dict),
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
        _FMHA_FWD_COMBINE_MOD_CACHE[dispatch_name] = mod
        # print("Done.")

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


def gen_fmha_fwd_combine_aot(dry_run: bool = False) -> int:
    configs = aot_combine_configs

    musa_instances = []
    musa_set = set()

    for config in configs:
        dispatch_name = _fmha_fwd_combine_encode(tuple(config.items()))
        if dispatch_name in musa_set:
            continue
        musa_set.add(dispatch_name)
        config["func_name"] = dispatch_name

        mu_path = jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu"
        with open(mu_path, "w") as f:
            f.write(
                TVM_HEADER
                + kern_fwd_combine.render(config)
                + EXPORT_FUNC.render(config)
            )
        musa_instances.append(mu_path)

    if not musa_instances:
        return 0

    if dry_run:
        return len(musa_instances)

    print(f"MATE AOT Building {len(musa_instances)} fmha_fwd_combine instances...")
    tvm_ffi.cpp.build(
        name="fmha_fwd_combine_aot",
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
        f"MATE AOT fmha_fwd_combine Done -> {(jit_env.MATE_AOT_DIR / 'fmha_fwd_combine_aot.so').as_posix()}."
    )
    return len(musa_instances)
