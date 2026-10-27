from functools import lru_cache

import torch

from ... import env as jit_env
from ....utils import ceil_div

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_ELEMENT_NAME_SUFFIX = {
    "mutlass::float_e4m3_t": "fp8_e4m3",
    "mutlass::float_e5m2_t": "fp8_e5m2",
    "mutlass::bfloat16_t": "bf16",
    "mutlass::half_t": "fp16",
}

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


def fmha_extra_include_paths():
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
    ]


def _resolve_mask(
    seqlen_q,
    seqlen_k,
    is_causal,
    window_size_left,
    window_size_right,
    attention_chunk=0,
):
    if window_size_left is None or window_size_left >= seqlen_k - 1:
        window_size_left = -1
    if window_size_right is None or window_size_right >= seqlen_q - 1:
        window_size_right = -1

    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0 and attention_chunk == 0
    is_local = (
        window_size_left >= 0 or window_size_right >= 0 or attention_chunk >= 1
    ) and not is_causal

    # chunk
    if window_size_left < 0:
        window_size_left = seqlen_k - 1
    if window_size_right < 0:
        window_size_right = seqlen_q - 1
    if attention_chunk > 0:
        window_size_left = min(window_size_left, attention_chunk - 1)
        window_size_right = min(window_size_right, attention_chunk - 1)

    return is_causal, is_local, window_size_left, window_size_right


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
def _check_enable_packgqa(m: int):
    return m <= 256


@lru_cache
def _get_tile_m(m: int, head_ratio: int, enable_packgqa: bool = None):
    enable_packgqa = (
        _check_enable_packgqa(m) if enable_packgqa is None else enable_packgqa
    )

    assert m is not None
    assert head_ratio is not None

    m = m * head_ratio if enable_packgqa else m
    if m <= 16:
        tile_m = 16
    elif m <= 32:
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


def _get_qv_stages_by_smem(
    tile_m: int, tile_n: int, headdim: int, headdim_v: int, element_size: int
):
    smem_limit = 192 * 1024

    def estimate(stages: int):
        smem_q = tile_m * headdim
        smem_qv = tile_m * headdim_v
        smem_k = tile_n * headdim * stages
        smem_v_for_qv = tile_n * headdim_v * stages
        smem_p = tile_m * tile_n
        smem_v_for_pv = tile_n * headdim_v * stages
        return (
            smem_q + smem_qv + smem_k + smem_v_for_qv + smem_p + smem_v_for_pv
        ) * element_size

    return (2, 2) if estimate(2) <= smem_limit else (1, 1)


def round_multiple(x, m):
    return (x + m - 1) // m * m


@lru_cache
def _get_fwd_kernel_config(
    m: int,
    head_ratio: int,
    headdim: int,
    headdim_v: int,
    element_size: int,
    enable_packgqa: bool = None,
    has_qv: bool = False,
    is_fp8: bool = False,
    is_high_regpressure: bool = False,
):
    packgqa_alignment = 128 // (element_size * 8)
    if headdim % packgqa_alignment != 0 or headdim_v % packgqa_alignment != 0:
        enable_packgqa = False
    headdim, headdim_v = _roundup_headdim(headdim, headdim_v)

    candidate_tile_m, enable_packgqa = _get_tile_m(m, head_ratio, enable_packgqa)
    if headdim % 128 != 0:
        candidate_tile_m = max(candidate_tile_m, 64 if is_fp8 else 32)
    if is_fp8:
        candidate_tile_m = max(candidate_tile_m, 32)
    tile_head_dim = 0

    if is_fp8 and has_qv:
        tile_m = min(candidate_tile_m, 32)
        tile_n = 128
        stages_k, stages_v = _get_qv_stages_by_smem(
            tile_m, tile_n, headdim, headdim_v, element_size
        )
    elif not is_fp8:
        tile_n = 64
        if has_qv:
            match (headdim, headdim_v):
                case (64, 64):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 4, 4
                case (64, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 1, 1
                case (64, 256):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 128), 1, 1
                case (64, 512):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (4, 4, 64) if tile_m > 32 else (1, 1, 0)
                    )
                case (128, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 192), 1, 1
                case (192, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 128), 1, 1
                case _:
                    assert False, f"Add QV config for headdim {headdim}-{headdim_v}"
        else:
            match (headdim, headdim_v):
                case (64, 64):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 2, 2
                case (64, 256):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 3, 3
                case (64, 512):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (4, 4, 64) if tile_m > 32 else (1, 1, 0)
                    )
                case (128, 128):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 3, 3
                case (192, 128) | (192, 192):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 256), 1, 1
                case (256, 256):
                    tile_m, stages_k, stages_v = min(candidate_tile_m, 192), 1, 1
                case (384, 384):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (4, 4, 128) if tile_m > 32 else (1, 1, 0)
                    )
                case (512, 512):
                    tile_m = min(candidate_tile_m, 128)
                    stages_k, stages_v, tile_head_dim = (
                        (3, 3, 128) if tile_m > 32 else (1, 1, 0)
                    )
                case _:
                    assert False, f"Add config for headdim {headdim}-{headdim_v}"

    if is_fp8 and not has_qv:
        match (headdim, headdim_v, is_high_regpressure):
            case (
                (64, 64, _) | (128, 128, False) | (192, 128, False) | (192, 192, False)
            ):
                tile_m = min(candidate_tile_m, 256)
                tile_n = 128
                stages_k = 2
                stages_v = 2
            case (256, 256, False):
                tile_m = min(candidate_tile_m, 192)
                tile_n = 128
                stages_k = 1
                stages_v = 1
            case (
                (128, 128, True)
                | (192, 128, True)
                | (192, 192, True)
                | (256, 256, True)
                | (384, 384, _)
            ):
                tile_m = min(candidate_tile_m, 128)
                tile_n = 128
                stages_k = 1
                stages_v = 1
            case (512, 512, _):
                tile_m = min(candidate_tile_m, 32)
                tile_n = 128
                stages_k = 1
                stages_v = 1
            case _:
                assert False, f"Add FP8 config for headdim {headdim}-{headdim_v}"

    multi_chunk = tile_head_dim > 0
    consumers_qk = consumers_pv = ceil_div(tile_m, 32 if multi_chunk else 64)

    return (
        tile_m,
        tile_n,
        stages_k,
        stages_v,
        headdim,
        headdim_v,
        consumers_qk,
        consumers_pv,
        tile_head_dim,
        enable_packgqa,
    )
