from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import decode as decode
from . import gemm as gemm
from . import mamba as mamba
from . import norm as norm
from . import rope as rope
from .gemm import bmm_bf16 as bmm_bf16
from .gemm import bmm_fp8 as bmm_fp8
from .norm import fused_add_rmsnorm as fused_add_rmsnorm
from .norm import (
    fused_add_rmsnorm_fp8_block_quant as fused_add_rmsnorm_fp8_block_quant,
)
from .norm import fused_add_rmsnorm_quant as fused_add_rmsnorm_quant
from .norm import (
    fused_dit_gate_residual_layernorm_gamma_beta as fused_dit_gate_residual_layernorm_gamma_beta,
)
from .norm import (
    fused_dit_gate_residual_layernorm_scale_shift as fused_dit_gate_residual_layernorm_scale_shift,
)
from .norm import (
    fused_dit_residual_layernorm_scale_shift as fused_dit_residual_layernorm_scale_shift,
)
from .norm import fused_qk_rmsnorm_rope as fused_qk_rmsnorm_rope
from .norm import fused_rmsnorm_silu as fused_rmsnorm_silu
from .norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
from .norm import gemma_rmsnorm as gemma_rmsnorm
from .norm import layernorm as layernorm
from .norm import layernorm_quant as layernorm_quant
from .norm import rmsnorm as rmsnorm
from .norm import rmsnorm_quant as rmsnorm_quant

try:
    from ._build_meta import __git_version__ as __git_version__
except Exception:
    __git_version__ = "unknown"


def _load_version() -> str:
    try:
        return version("flashinfer-python")
    except PackageNotFoundError:
        pass

    try:
        from ._build_meta import __version__ as build_version

        return build_version
    except Exception:
        pass

    try:
        return (
            (Path(__file__).resolve().parents[3] / "version.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


__version__ = _load_version()

__all__ = [
    "__git_version__",
    "__version__",
    "bmm_bf16",
    "bmm_fp8",
    "decode",
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_fp8_block_quant",
    "fused_add_rmsnorm_quant",
    "fused_dit_gate_residual_layernorm_gamma_beta",
    "fused_dit_gate_residual_layernorm_scale_shift",
    "fused_dit_residual_layernorm_scale_shift",
    "fused_qk_rmsnorm_rope",
    "fused_rmsnorm_silu",
    "gemm",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "layernorm",
    "layernorm_quant",
    "norm",
    "rope",
    "rmsnorm",
    "rmsnorm_quant",
]
