"""Shared source generation and layout ABI for MegaMoE JIT modules."""

import functools
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from . import env as jit_env
from .gemm.deep_gemm.deep_gemm_utils import DEEP_GEMM_CUDA_FLAGS
from .utils import EXPORT_FUNC, TVM_HEADER

# Header-only layout changes must invalidate runtime utilities and every stage.
MEGA_MOE_LAYOUT_REVISION = 3
MEGA_MOE_EXTRA_CFLAGS = ["-O3", "-Wno-switch-bool"]
MEGA_MOE_EXTRA_MUSA_CFLAGS = list(DEEP_GEMM_CUDA_FLAGS) + [
    "-fno-signed-zeros",
    "-DMARCH_TYPE=310",
]


@functools.cache
def _template_env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir.as_posix()),
        keep_trailing_newline=True,
    )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _normalize_activation(activation: str) -> str:
    activation = activation.lower()
    if activation not in ("swiglu", "situ"):
        raise ValueError("activation must be 'swiglu' or 'situ'")
    return activation


def _extra_include_paths() -> list[Path]:
    return [
        jit_env.MATE_INCLUDE_DIR,
        jit_env.MATE_CSRC_DIR,
        jit_env.MUTLASS_INCLUDE_DIR,
        jit_env.MUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include",
        jit_env.MUTLASS_INCLUDE_DIR.parent / "experimental" / "fmha",
    ]


def _generated_source_path(kernel_name: str) -> Path:
    return jit_env.MATE_GEN_SRC_DIR / "mega_moe" / f"{kernel_name}.mu"


def _render_source(
    kernel_name: str, template_path: Path, render_config: dict[str, object]
) -> str:
    render_config = {**render_config, "func_name": kernel_name}
    return (
        TVM_HEADER
        + _template_env(template_path.parent)
        .get_template(template_path.name)
        .render(render_config)
        + EXPORT_FUNC.render(render_config)
    )
