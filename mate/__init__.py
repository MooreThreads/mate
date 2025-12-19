from importlib.metadata import version

import mate.deep_gemm  # noqa: F401
import mate.flashinfer  # noqa: F401
import mate.flashmla  # noqa: F401
from mate.flashmla import flash_mla_with_kvcache, get_mla_metadata
from mate.mha_interface import flash_attn_varlen_func, flash_attn_with_kvcache
from mate.mla_interface import mla

__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "mla",
    "get_mla_metadata",
    "flash_mla_with_kvcache",
]

__version__ = version("mate")
