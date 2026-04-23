import importlib

from . import env as env
from .core import JitSpec as JitSpec
from .core import JitSpecRegistry as JitSpecRegistry
from .core import JitSpecStatus as JitSpecStatus
from .core import MissingJITCacheError as MissingJITCacheError
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import copy_built_kernels as copy_built_kernels
from .core import default_aot_path as default_aot_path
from .core import gen_jit_spec as gen_jit_spec
from .core import jit_spec_registry as jit_spec_registry
from .core import temporary_max_jobs as temporary_max_jobs
from .deep_gemm_attention import (
    gen_deep_gemm_attention_aot as gen_deep_gemm_attention_aot,
)
from .deep_gemm_attention import (
    gen_deep_gemm_attention_spec as gen_deep_gemm_attention_spec,
)
from .deep_gemm_attention import (
    get_deep_gemm_attention_module as get_deep_gemm_attention_module,
)
from .flash_attention_ops import (
    gen_flash_attention_ops_aot as gen_flash_attention_ops_aot,
)
from .flash_attention_ops import (
    gen_flash_attention_ops_spec as gen_flash_attention_ops_spec,
)
from .flash_attention_ops import (
    get_flash_attention_ops_module as get_flash_attention_ops_module,
)
from .gemm_ops import gen_gemm_ops_aot as gen_gemm_ops_aot
from .gemm_ops import gen_gemm_ops_spec as gen_gemm_ops_spec
from .gemm_ops import get_gemm_ops_module as get_gemm_ops_module
from .mla_ops import gen_mla_ops_aot as gen_mla_ops_aot
from .mla_ops import gen_mla_ops_spec as gen_mla_ops_spec
from .mla_ops import get_mla_ops_module as get_mla_ops_module
from .moe_fused_gate import gen_moe_fused_gate_aot as gen_moe_fused_gate_aot
from .moe_fused_gate import gen_moe_fused_gate_spec as gen_moe_fused_gate_spec
from .moe_fused_gate import get_moe_fused_gate_module as get_moe_fused_gate_module
from .sage_attention import gen_sage_attention_aot as gen_sage_attention_aot
from .sage_attention import gen_sage_attention_spec as gen_sage_attention_spec
from .sage_attention import get_sage_attention_module as get_sage_attention_module

__all__ = [
    "env",
    "attention",
    "JitSpec",
    "JitSpecRegistry",
    "JitSpecStatus",
    "MissingJITCacheError",
    "build_jit_specs",
    "clear_cache_dir",
    "copy_built_kernels",
    "default_aot_path",
    "gen_deep_gemm_attention_aot",
    "gen_deep_gemm_attention_spec",
    "get_deep_gemm_attention_module",
    "gen_flash_attention_ops_aot",
    "gen_flash_attention_ops_spec",
    "get_flash_attention_ops_module",
    "gen_gemm_ops_aot",
    "gen_gemm_ops_spec",
    "get_gemm_ops_module",
    "gen_mla_ops_aot",
    "gen_mla_ops_spec",
    "get_mla_ops_module",
    "gen_jit_spec",
    "jit_spec_registry",
    "gen_moe_fused_gate_aot",
    "gen_moe_fused_gate_spec",
    "get_moe_fused_gate_module",
    "gen_sage_attention_aot",
    "gen_sage_attention_spec",
    "get_sage_attention_module",
    "prewarm_modules",
    "temporary_max_jobs",
]


def __getattr__(name):
    if name == "attention":
        return importlib.import_module(".attention", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def prewarm_modules(
    *,
    fmha_metadata_configs=None,
    fmha_combine_configs=None,
    fmha_fwd_configs=None,
    include_flash_attention: bool = False,
    include_gemm: bool = False,
    include_deep_gemm_attention: bool = False,
    include_mla: bool = False,
    include_moe_fused_gate: bool = False,
    include_sage_attention: bool = False,
    jobs=None,
    verbose: bool = False,
    skip_prebuilt: bool = True,
):
    specs = []
    from .attention.fmha import (
        gen_fmha_fwd_combine_specs,
        gen_fmha_fwd_specs,
        gen_fmha_metadata_specs,
    )

    if fmha_metadata_configs:
        specs.extend(gen_fmha_metadata_specs(fmha_metadata_configs))
    if fmha_combine_configs:
        specs.extend(gen_fmha_fwd_combine_specs(fmha_combine_configs))
    if fmha_fwd_configs:
        specs.extend(gen_fmha_fwd_specs(fmha_fwd_configs))
    if include_flash_attention:
        from .flash_attention_ops import gen_flash_attention_ops_spec

        specs.append(gen_flash_attention_ops_spec())
    if include_gemm:
        from .gemm_ops import gen_gemm_ops_spec

        specs.append(gen_gemm_ops_spec())
    if include_deep_gemm_attention:
        from .deep_gemm_attention import gen_deep_gemm_attention_spec

        specs.append(gen_deep_gemm_attention_spec())
    if include_mla:
        from .mla_ops import gen_mla_ops_spec

        specs.append(gen_mla_ops_spec())
    if include_moe_fused_gate:
        from .moe_fused_gate import gen_moe_fused_gate_spec

        specs.append(gen_moe_fused_gate_spec())
    if include_sage_attention:
        from .sage_attention import gen_sage_attention_spec

        specs.append(gen_sage_attention_spec())

    build_jit_specs(
        specs,
        jobs=jobs,
        verbose=verbose,
        skip_prebuilt=skip_prebuilt,
    )
    return specs
