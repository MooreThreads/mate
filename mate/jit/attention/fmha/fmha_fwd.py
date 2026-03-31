import pathlib
import torch
from typing import Optional, Dict, Any
import math

import tvm_ffi.cpp

import hashlib
import json

from ... import env as jit_env
from .fmha_utils import fmha_template_env, _get_fwd_kernel_config, ceil_div
from .fmha_combine import _fmha_fwd_combine
from ...utils import (
    dtype_torch2mutlass_map,
    maybe_contiguous,
    TVM_HEADER,
    EXPORT_FUNC,
    update_aot_mod,
)
from ...configs import KernelConfigGraph, ParamSpec, domain_by_case


kern_fwd = fmha_template_env.get_template("fwd_kern.j2")


def _fmha_fwd_encode(config_tuple: tuple) -> str:
    config = dict(config_tuple)
    uri = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return f"fmha_fwd_{uri}"


# module cache: dispatch_name -> compiled module
_FMHA_FWD_MOD_CACHE: Dict[str, Any] = {}


def config_selector(cfg):
    return cfg["config_level"]


CONFIG_TABLE: Dict[int, Dict[str, Any]] = {
    1: {
        "element": [dtype_torch2mutlass_map[dtype] for dtype in [torch.bfloat16]],
        "headdim": [(128, 128), (256, 256)],
        "head_ratio": [1, 4, 5, 8, 12, 16],
        "mode_q": ["ragged"],
        "mode_k": ["ragged", "padded"],
        "mode_mask": ["none", "causal"],
        "is_packgqa": [False, True],
        "has_metadata": [False, True],
        "has_cu_seqlens_k_new": [False],
        "has_leftpad_k": [False],
        "has_softcap": [False],
    },
    2: {
        "element": [dtype_torch2mutlass_map[dtype] for dtype in [torch.bfloat16]],
        "headdim": [(128, 128), (192, 128), (256, 256)],
        "head_ratio": [1, 4, 5, 8, 12, 16],
        "mode_q": ["ragged"],
        "mode_k": ["ragged", "padded"],
        "mode_mask": ["none", "causal", "local"],
        "is_packgqa": [False, True],
        "has_metadata": [False, True],
        "has_cu_seqlens_k_new": [False],
        "has_leftpad_k": [False],
        "has_softcap": [False, True],
    },
}

base_specs = []
mode_q = [
    ParamSpec(
        name="mode_q",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "mode_q"),
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
        domain=domain_by_case(config_selector, CONFIG_TABLE, "mode_k"),
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
    ParamSpec(
        name="paged_kv",
        domain=[True, False],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["has_seqused_k"]),
        depends_on=("has_seqused_k",),
    ),
    ParamSpec(
        name="force_lsu_kv",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["paged_kv"]),
        depends_on=("paged_kv",),
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
        domain=domain_by_case(config_selector, CONFIG_TABLE, "mode_mask"),
        default="none",
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="is_causal",
        default=False,
        compute=lambda cfg: cfg["mode_mask"] == "causal",
        depends_on=("mode_mask",),
        sweep=False,
    ),
    ParamSpec(
        name="is_local",
        default=False,
        compute=lambda cfg: cfg["mode_mask"] == "local",
        depends_on=("mode_mask",),
        sweep=False,
    ),
    ParamSpec(
        name="has_learnable_sink",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["is_local"]),
        depends_on=("is_local",),
    ),
]
score_mode = [
    ParamSpec(
        name="has_softcap",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "has_softcap"),
        depends_on=("config_level",),
    ),
]
specs_attn = [
    ParamSpec(
        name="is_packgqa",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "is_packgqa"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="element",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "element"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="head_ratio",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "head_ratio"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="has_metadata",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "has_metadata"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="enable_cp",
        domain=[False],
        default=False,
    ),
]


def headdim_selector(cfg):
    hd = cfg["headdim"]
    if hd == (128, 128):
        return "128_128"
    elif hd == (192, 128):
        return "192_128"
    elif hd == (256, 256):
        return "256_256"
    # elif hd == (384, 384):
    #     return "384_384"
    else:
        raise ValueError(f"Unsupported headdim {hd}")


HEADDIM_TABLE: Dict[str, Dict[str, Any]] = {
    "128_128": {
        "tile_mn": [(32, 64), (64, 64), (128, 64), (192, 64), (256, 64)],
        "stages_kv": [(2, 2)],
    },
    "192_128": {
        "tile_mn": [(32, 64), (64, 64), (128, 64), (192, 64), (256, 64)],
        "stages_kv": [(1, 1)],
    },
    "256_256": {
        "tile_mn": [(32, 64), (192, 64)],
        "stages_kv": [(1, 1), (2, 2)],
    },
    "384_384": {
        "tile_mn": [(64, 64)],
        "stages_kv": [(1, 1)],
    },
}
specs_sel = [
    ParamSpec(
        name="headdim",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "headdim"),
        default=(128, 128),
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="head_dim_qk",
        compute=lambda cfg: cfg["headdim"][0],
        depends_on=("headdim",),
        sweep=False,
    ),
    ParamSpec(
        name="head_dim_vo",
        compute=lambda cfg: cfg["headdim"][1],
        depends_on=("headdim",),
        sweep=False,
    ),
    ParamSpec(
        name="is_even_headdim",
        compute=lambda cfg: cfg["headdim"][0] == cfg["headdim"][1],
        depends_on=("headdim",),
        sweep=False,
    ),
    ParamSpec(
        name="tile_mn",
        domain=domain_by_case(headdim_selector, HEADDIM_TABLE, "tile_mn"),
        default=(0, 0),
        depends_on=("headdim",),
        export=False,
    ),
    ParamSpec(
        name="tile_m",
        compute=lambda cfg: cfg["tile_mn"][0],
        depends_on=("tile_mn",),
        sweep=False,
    ),
    ParamSpec(
        name="tile_n",
        compute=lambda cfg: cfg["tile_mn"][1],
        depends_on=("tile_mn",),
        sweep=False,
    ),
    ParamSpec(
        name="stages_kv",
        domain=domain_by_case(headdim_selector, HEADDIM_TABLE, "stages_kv"),
        default=(0, 0),
        depends_on=("headdim",),
        export=False,
    ),
    ParamSpec(
        name="stages_k",
        compute=lambda cfg: cfg["stages_kv"][0],
        depends_on=("stages_kv",),
        sweep=False,
    ),
    ParamSpec(
        name="stages_v",
        compute=lambda cfg: cfg["stages_kv"][1],
        depends_on=("stages_kv",),
        sweep=False,
    ),
    ParamSpec(
        name="consumers_qk",
        compute=lambda cfg: ceil_div(cfg["tile_m"], 64),
        depends_on=("tile_m",),
        sweep=False,
    ),
    ParamSpec(
        name="consumers_pv",
        compute=lambda cfg: cfg["consumers_qk"],
        depends_on=("consumers_qk",),
        sweep=False,
    ),
]


base_specs.extend(mode_q)
base_specs.extend(mode_k)
base_specs.extend(mode_mask)
base_specs.extend(score_mode)
base_specs.extend(specs_attn)
base_specs.extend(specs_sel)


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
    cfg_level_key = "attention.fmha.fwd".split(".")
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


aot_fwd_configs = _get_aot_configs()
update_aot_mod(
    aot_fwd_configs,
    _fmha_fwd_encode,
    jit_env.MATE_AOT_DIR / "fmha_fwd_aot.so",
    _FMHA_FWD_MOD_CACHE,
)


def _fmha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    q_v: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    is_rotary_interleaved: bool = False,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    mp_margin: int = 0,
    return_lse: bool = False,
    lse: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    # CP params — default to non-CP (world_size=1)
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
):
    assert k_new is None, "k_new parameter is not supported yet"
    assert kv_batch_idx is None, "kv_batch_idx parameter is not supported yet"
    assert leftpad_k is None, "leftpad_k parameter is not supported yet"
    assert rotary_cos is None, "rotary_cos parameter is not supported yet"
    assert q_v is None, "qv parameter is not supported yet"
    assert attention_chunk == 0, "attention_chunk parameter is not supported yet"

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]

    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
        max_seqlen_q = q.shape[1]
    else:
        assert max_seqlen_q is not None
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q
        total_q = q.shape[0]

    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        seqlen_k = k.shape[-3]
        num_pages, page_size = 0, 1

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
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

    assert q.dtype in [torch.float16, torch.bfloat16], (
        "inputs must be float16 or bfloat16"
    )
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )

    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

        # FIXME: bug learnable_sink + split
        num_splits = -1
        scheduler_metadata = None

    assert all(
        t is None or t.is_musa
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on MUSA device"

    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qhead_per_kvhead = num_head // num_head_kv

    local = False
    if window_size_left is not None and window_size_right is not None:
        if window_size_left >= 0 and window_size_right >= 0:
            local = True
    if is_causal:
        local = False
    if not local:
        window_size_left, window_size_right = -1, -1

    # CP sanity checks
    assert cp_world_size > 0, (
        "cp_world_size must be positive, required by downstream unified code path. Use 1 if CP is not enabled."
    )
    assert cp_world_size != 1 or cp_rank == 0, (
        "When context parallelism is disabled, cp_rank must be zero"
    )
    assert cp_world_size == 1 or cp_tot_seqused_k is not None, (
        "cp_tot_seqused_k must be provided when context parallelism is enabled."
    )
    assert not (local and cp_world_size > 1), (
        "Local attention (sliding window) is not currently supported with context parallelism (cp_world_size > 1)."
    )

    out_torch_dtype = q.dtype
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=q.device,
        )
    else:
        assert out.shape == (*q_batch_seqlen_shape, num_head, head_dim_v)
        assert out.dtype == out_torch_dtype
        assert out.device == q.device

    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    else:
        assert lse.shape == lse_shape
        assert lse.dtype == torch.float32
        assert lse.device == q.device

    # Metadata
    # If not provided, fallback to SingleTileScheduler.
    if scheduler_metadata is not None:
        assert scheduler_metadata.shape[0] >= batch_size * 4, (
            "metadata buffer is too small"
        )
        (
            num_splits_dynamic,
            batch_table,
            num_m_blocks,
            num_nheads_in_l2,
        ) = (
            scheduler_metadata[batch_size * i : batch_size * (i + 1)] for i in range(4)
        )
    else:
        num_splits_dynamic, batch_table, num_m_blocks, num_nheads_in_l2 = (
            None for _ in range(4)
        )

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
    ) = _get_fwd_kernel_config(
        max_seqlen_q,
        qhead_per_kvhead,
        head_dim,
        head_dim_v,
        pack_gqa,
    )

    constexpr_dict = {
        "has_cu_seqlens_q": cu_seqlens_q is not None,
        "has_cu_seqlens_k": cu_seqlens_k is not None,
        "has_seqused_q": seqused_q is not None,
        "has_seqused_k": seqused_k is not None,
        "has_cu_seqlens_k_new": False,
        "has_leftpad_k": False,
        "paged_kv": page_table is not None,
        "has_softcap": softcap != 0.0,
        "has_learnable_sink": learnable_sink is not None,
        "is_local": local,
        "is_causal": is_causal,
        "is_packgqa": enable_packgqa,
        "head_ratio": qhead_per_kvhead,
        "element": dtype_torch2mutlass_map[q.dtype],
        "force_lsu_kv": page_table is not None and page_size != 64,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "stages_k": stages_k,
        "stages_v": stages_v,
        "head_dim_qk": headdim_rounded,
        "head_dim_vo": headdim_v_rounded,
        "consumers_qk": consumers_qk,
        "consumers_pv": consumers_pv,
        "is_even_headdim": headdim_v_rounded == head_dim_v,
        "has_metadata": scheduler_metadata is not None,
        "enable_cp": cp_world_size > 1,
    }
    # print(f"tile_m: {tile_m}, tile_n: {tile_n}")

    uri = hashlib.sha256(
        json.dumps(constexpr_dict, sort_keys=True).encode()
    ).hexdigest()

    dispatch_name = f"fmha_fwd_{uri}"

    if dispatch_name in _FMHA_FWD_MOD_CACHE:
        mod = _FMHA_FWD_MOD_CACHE[dispatch_name]
    else:
        constexpr_dict["func_name"] = dispatch_name
        # print("Compiling forward kernel JIT...", flush=True)
        mod = tvm_ffi.cpp.load_inline(
            name="fmha_fwd",
            cuda_sources=kern_fwd.render(constexpr_dict),
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
        _FMHA_FWD_MOD_CACHE[dispatch_name] = mod
        # print("Done.")

    fmha_fwd_impl = mod.get_function(dispatch_name)

    accums, num_splits = fmha_fwd_impl(
        q,
        k,
        v,
        k_new,
        v_new,
        q_v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        softcap,
        is_rotary_interleaved,
        mp_margin,
        num_splits,
        num_splits_dynamic,
        batch_table,
        num_m_blocks,
        learnable_sink,
        out,
        lse,
        # CP params — always passed, kernel ignores them when cp_world_size==1
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
    )
    o_accum, lse_accum = accums

    _fmha_fwd_combine(
        out,
        lse,
        o_accum,
        lse_accum,
        tile_n,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        num_split=num_splits,
        metadata=scheduler_metadata,
    )

    return (out, lse, o_accum, lse_accum) if return_lse else (out, o_accum)


def gen_fmha_fwd_aot(dry_run: bool = False) -> int:
    configs = _get_aot_configs()

    musa_instances = []
    musa_set = set()

    for config in configs:
        dispatch_name = _fmha_fwd_encode(tuple(config.items()))
        if dispatch_name in musa_set:
            continue
        musa_set.add(dispatch_name)
        config["func_name"] = dispatch_name

        mu_path = jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu"
        with open(mu_path, "w") as f:
            f.write(TVM_HEADER + kern_fwd.render(config) + EXPORT_FUNC.render(config))

        musa_instances.append(mu_path)

    if not musa_instances:
        return 0

    if dry_run:
        return len(musa_instances)

    print(f"MATE AOT Building {len(musa_instances)} fmha_fwd...", flush=True)
    tvm_ffi.cpp.build(
        name="fmha_fwd_aot",
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
        f"MATE AOT fmha_fwd Done -> {(jit_env.MATE_AOT_DIR / 'fmha_fwd_aot.so').as_posix()}."
    )
    return len(musa_instances)
