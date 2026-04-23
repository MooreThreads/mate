from __future__ import annotations

import functools
import re

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


MCC_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

CXX_FLAGS = [
    "-Wno-switch-bool",
]

_SAGE_ATTENTION_KERNEL_PATTERN = re.compile(
    r"(?:"
    r"e4m3tce_flash_atten_quant_mode_\d+_512_256x128x128(?:_kvcache)?(?:_causal_persistence)?"
    r"|"
    r"e4m3tce_flash_atten_qk_int8_quant_mode_\d+_512_256x128x128(?:_kvcache)?(?:_causal_persistence)?"
    r")\.cpp"
)

_SAGE_ATTENTION_KERNEL_OBJECT_PATTERN = re.compile(
    r"(?:"
    r"e4m3tce_flash_atten_quant_mode_\d+_512_256x128x128(?:_kvcache)?(?:_causal_persistence)?"
    r"|"
    r"e4m3tce_flash_atten_qk_int8_quant_mode_\d+_512_256x128x128(?:_kvcache)?(?:_causal_persistence)?"
    r")\.o"
)


def _get_sage_attention_kernel_sources() -> list:
    flash_atten_dir = jit_env.MATE_CSRC_DIR / "mubin/mp31/flash_atten"
    return sorted(
        path
        for path in flash_atten_dir.iterdir()
        if path.is_file() and _SAGE_ATTENTION_KERNEL_PATTERN.fullmatch(path.name)
    )


def _get_sage_attention_prebuilt_objects() -> list:
    cached_dir = jit_env.MATE_JIT_DIR / "sage_attention"
    if not cached_dir.exists():
        return []
    return sorted(
        path
        for path in cached_dir.iterdir()
        if path.is_file() and _SAGE_ATTENTION_KERNEL_OBJECT_PATTERN.fullmatch(path.name)
    )


def gen_sage_attention_spec() -> JitSpec:
    kernel_sources = _get_sage_attention_kernel_sources()
    extra_ldflags = ["-lmusa"]
    if not kernel_sources:
        extra_ldflags.extend(
            str(path) for path in _get_sage_attention_prebuilt_objects()
        )
    return gen_jit_spec(
        "sage_attention",
        [jit_env.MATE_CSRC_DIR / "sage_attention_asm.mu", *kernel_sources],
        extra_cflags=CXX_FLAGS,
        extra_cuda_cflags=MCC_FLAGS + CXX_FLAGS,
        extra_ldflags=extra_ldflags,
        extra_include_paths=[
            jit_env.MATE_INCLUDE_DIR,
            jit_env.MATE_CSRC_DIR,
            jit_env.MUTLASS_INCLUDE_DIR,
        ],
    )


def gen_sage_attention_aot() -> list[JitSpec]:
    return [gen_sage_attention_spec()]


@functools.cache
def get_sage_attention_module():
    return gen_sage_attention_spec().build_and_load()
