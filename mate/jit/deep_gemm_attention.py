from __future__ import annotations

import functools

from jinja2 import Environment, FileSystemLoader

from . import env as jit_env
from .cpp_ext import get_mudnn_ldflags
from .core import JitSpec, gen_jit_spec
from .gemm import gemm_utils
from .utils import EXPORT_FUNC, TVM_HEADER

_ALIGN = 32

CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]
CUDA_FLAGS = [
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
INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]


def gen_deep_gemm_attention_spec() -> JitSpec:
    return gen_jit_spec(
        "deep_gemm_attention_ops",
        [
            jit_env.MATE_CSRC_DIR / "deepgemm_attention.mu",
        ],
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_ldflags=list(get_mudnn_ldflags()),
        extra_include_paths=list(INCLUDE_PATHS),
    )


_AOT_METADATA_BATCH_SIZES = [32, 64, 128]


def gen_deep_gemm_attention_aot() -> list[JitSpec]:
    specs = [gen_deep_gemm_attention_spec()]
    seen: set[int] = set()
    for bs in _AOT_METADATA_BATCH_SIZES:
        aligned = _round_up(bs, _ALIGN)
        if aligned not in seen:
            seen.add(aligned)
            specs.append(_gen_metadata_spec(aligned))
    return specs


@functools.cache
def get_deep_gemm_attention_module():
    return gen_deep_gemm_attention_spec().build_and_load()


_template_env = Environment(
    loader=FileSystemLoader((gemm_utils.gemm_template_dir / "deep_gemm").as_posix()),
    keep_trailing_newline=True,
)


def _round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _render_metadata_source(aligned_batch_size: int) -> str:
    template = _template_env.get_template("mqa_logits_metadata_kern.j2")
    render_config = {
        "aligned_batch_size": aligned_batch_size,
        "func_name": "get_paged_mqa_logits_metadata",
    }
    return (
        TVM_HEADER + template.render(render_config) + EXPORT_FUNC.render(render_config)
    )


def _gen_metadata_spec(aligned_batch_size: int) -> JitSpec:
    source_path = (
        jit_env.MATE_GEN_SRC_DIR
        / "deep_gemm"
        / f"mqa_logits_metadata_b{aligned_batch_size}.mu"
    )
    return gen_jit_spec(
        f"mqa_logits_metadata_b{aligned_batch_size}",
        [source_path],
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_ldflags=list(get_mudnn_ldflags()),
        extra_include_paths=list(INCLUDE_PATHS),
        generated_sources={source_path: _render_metadata_source(aligned_batch_size)},
    )


@functools.cache
def _load_metadata_module(aligned_batch_size: int):
    return _gen_metadata_spec(aligned_batch_size).build_and_load()


def get_metadata_module(batch_size: int):
    aligned = _round_up(batch_size, _ALIGN)
    return _load_metadata_module(aligned)
