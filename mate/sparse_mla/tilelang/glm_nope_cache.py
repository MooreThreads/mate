"""Write GLM's inline FP8 latent and arbitrary FP32 group-128 scales."""

import torch
import tilelang
from tilelang import language as T

from mate.execution_context import skip_kernel_launch_if_dry_run


@tilelang.jit(out_idx=[])
def _write_cache(row_bytes: int):
    tokens, capacity = T.dynamic("tokens"), T.dynamic("capacity")

    @T.prim_func
    def glm_nope_cache_write(
        latent: T.Tensor[(tokens, 512), T.bfloat16],
        slots: T.Tensor[(tokens,), T.int64],
        data: T.Tensor[(capacity, row_bytes), T.float8_e4m3],
        scales: T.Tensor[(capacity, row_bytes // 4), T.float32],
    ):
        with T.Kernel(tokens, threads=128) as token:
            x = T.alloc_fragment((4, 128), "float32")
            magnitude = T.alloc_fragment((4, 128), "float32")
            sf = T.alloc_fragment((4,), "float32")
            if slots[token] >= 0 and slots[token] < capacity:
                for g, c in T.Parallel(4, 128):
                    x[g, c] = T.Cast("float32", latent[token, g * 128 + c])
                    magnitude[g, c] = T.abs(x[g, c])
                T.reduce_max(magnitude, sf, dim=1)
                for g in T.Parallel(4):
                    sf[g] = T.max(sf[g] / 448.0, 1e-4)
                    scales[slots[token], 128 + g] = sf[g]
                for g, c in T.Parallel(4, 128):
                    data[slots[token], g * 128 + c] = T.Cast(
                        "float8_e4m3", x[g, c] / sf[g]
                    )

    return glm_nope_cache_write


def write_glm_nope_cache(latent, cache, slots):
    assert (
        latent.dtype == torch.bfloat16 and latent.ndim == 2 and latent.shape[1] == 512
    )
    assert latent.is_contiguous()
    assert cache.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn)
    assert cache.ndim == 4 and cache.shape[2] == 1 and cache.shape[-1] in (528, 656)
    assert cache.is_contiguous(), "cache writer requires uniform contiguous token rows"
    assert (
        slots.dtype == torch.int64
        and slots.shape == (latent.shape[0],)
        and slots.is_contiguous()
    )
    assert latent.device == cache.device == slots.device
    if latent.shape[0] == 0:
        return
    rows = cache.view(torch.uint8).view(-1, cache.shape[-1])
    kernel = _write_cache(cache.shape[-1])
    if not skip_kernel_launch_if_dry_run():
        kernel(latent, slots, rows.view(torch.float8_e4m3fn), rows.view(torch.float32))
