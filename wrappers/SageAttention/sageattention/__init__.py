"""
SageAttention compatibility package for MATE.
"""

from importlib.metadata import PackageNotFoundError, version

from sageattention.interface import sageattn, sageattn_qk_int8_pv_fp8_cuda_sm90


try:
    __version__ = version("sageattention")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = [
    "__version__",
    "sageattn",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
]
