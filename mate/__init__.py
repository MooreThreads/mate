# Try to get version from installed package, fall back to build meta
try:
    from importlib.metadata import version

    __version__ = version("mate")
except Exception:
    try:
        from ._build_meta import __version__
    except Exception:
        __version__ = "unknown"

from . import jit as jit
import mate.deep_gemm  # noqa: F401
import mate.flashinfer  # noqa: F401
import mate.flashmla  # noqa: F401
from mate.api_logging import mate_api
from mate.flashmla import flash_mla_with_kvcache, get_mla_metadata
from mate.mha_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from mate.mla_interface import mla
from mate.moe_fused_gate import moe_fused_gate

__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "get_scheduler_metadata",
    "mla",
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "moe_fused_gate",
    "mate_api",
    "__version__",
]
