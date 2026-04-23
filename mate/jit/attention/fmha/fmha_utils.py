from functools import lru_cache
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ... import env as jit_env


@lru_cache
def _get_fmha_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


FMHA_EXTRA_CUDA_CFLAGS = [
    "-Od3",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]


def get_fmha_template(template_name: str):
    template_dir = (jit_env.MATE_TEMPLATE_DIR / "attention" / "fmha").as_posix()
    return _get_fmha_template_env(template_dir).get_template(template_name)


def fmha_extra_include_paths():
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def ceil_div(x, y):
    return (x + y - 1) // y


@lru_cache
def _roundup_headdim(headdim: int, headdim_v: int):
    # round up headdim
    if headdim <= 64:
        headdim = 64
    elif headdim <= 128:
        headdim = 128
    elif headdim <= 192:
        headdim = 192
    elif headdim <= 256:
        headdim = 256
    elif headdim <= 384:
        headdim = 384
    else:
        headdim = 512

    if headdim_v <= 64:
        headdim_v = 64
    elif headdim_v <= 128:
        headdim_v = 128
    elif headdim_v <= 192:
        headdim_v = 192
    elif headdim_v <= 256:
        headdim_v = 256
    elif headdim_v <= 384:
        headdim_v = 384
    else:
        headdim_v = 512

    return (headdim, headdim_v)


@lru_cache
def _check_enable_packgqa(m: int, head_ratio: int):
    m_hr = m * head_ratio

    if m_hr <= 32 or m_hr <= 64 or m_hr <= 128 or m_hr <= 256:
        return True

    return False


@lru_cache
def _get_tile_m(m: int, head_ratio: int, enable_packgqa: bool = None):
    enable_packgqa = (
        _check_enable_packgqa(m, head_ratio)
        if enable_packgqa is None
        else enable_packgqa
    )

    assert m is not None
    assert head_ratio is not None

    m = m * head_ratio if enable_packgqa else m
    if m <= 32:
        tile_m = 32
    elif m <= 64:
        tile_m = 64
    elif m <= 128:
        tile_m = 128
    elif m <= 192:
        tile_m = 192
    else:
        tile_m = 256

    return tile_m, enable_packgqa


@lru_cache
def _get_fwd_kernel_config(
    m: int,
    head_ratio: int,
    headdim: int,
    headdim_v: int,
    enable_packgqa: bool = None,
):
    headdim, headdim_v = _roundup_headdim(headdim, headdim_v)

    candidate_tile_m, enable_packgqa = _get_tile_m(m, head_ratio, enable_packgqa)

    decode_mode = enable_packgqa
    if headdim == 64 and headdim_v == 64:
        tile_m = candidate_tile_m
        tile_n = 64
        stages_k = 2
        stages_v = 2
    elif headdim == 128 and headdim_v == 128:
        tile_m = candidate_tile_m
        tile_n = 64
        stages_k = 2
        stages_v = 2
    elif headdim == 192 and headdim_v == 128:
        tile_m = candidate_tile_m
        tile_n = 64
        stages_k = 1
        stages_v = 1
    elif headdim == 192 and headdim_v == 192:
        tile_m = candidate_tile_m
        tile_n = 64
        stages_k = 1
        stages_v = 1
    elif headdim == 256 and headdim_v == 256:
        tile_m = 192 if not decode_mode else 32
        tile_n = 64
        stages_k = 1 if not decode_mode else 2
        stages_v = 1 if not decode_mode else 2
    elif headdim == 384 and headdim_v == 384:
        tile_m = 64
        tile_n = 64
        stages_k = 1
        stages_v = 1
    elif headdim == 512 and headdim_v == 512:
        tile_m = 32
        tile_n = 64
        stages_k = 1
        stages_v = 1
    else:
        assert False, f"Add config for headdim {headdim}-{headdim_v}"

    consumers_qk = ceil_div(tile_m, 64)
    consumers_pv = consumers_qk

    return (
        tile_m,
        tile_n,
        stages_k,
        stages_v,
        headdim,
        headdim_v,
        consumers_qk,
        consumers_pv,
        enable_packgqa,
    )
