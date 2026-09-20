"""Mamba/SSD APIs, FlashInfer-compatible on MUSA.

The exported surface is the subset of ``flashinfer.mamba`` that MUSA consumers
call, with upstream FlashInfer signatures. Two upstream groups are deliberately
absent because they are not part of the MUSA path this wrapper covers:

``cake_selective_state_update`` / ``CakeSSDCombined``
    CUTLASS/Cake-backed CUDA implementations.
``ssd_combined_fwd`` / ``SSDCombined``
    The dense SSD entry point; MUSA consumers use ``ssd_combined_fwd_varlen``.

Every callable here resolves its implementation from the MATE mamba family at
call time; see :mod:`flashinfer.mamba._backend`.
"""

from .checkpointing_ssu import (
    allocate_checkpointing_ssu_scratch as allocate_checkpointing_ssu_scratch,
)
from .checkpointing_ssu import checkpointing_ssu as checkpointing_ssu
from .replayssm_materialize import replayssm_materialize as replayssm_materialize
from .selective_state_update import selective_state_update as selective_state_update
from .ssd_combined import (
    mamba_chunk_scan_combined_varlen as mamba_chunk_scan_combined_varlen,
)
from .ssd_combined import ssd_combined_fwd_varlen as ssd_combined_fwd_varlen

__all__ = [
    "allocate_checkpointing_ssu_scratch",
    "checkpointing_ssu",
    "mamba_chunk_scan_combined_varlen",
    "replayssm_materialize",
    "selective_state_update",
    "ssd_combined_fwd_varlen",
]

# Capability-shaped symbol: present only when the MATE backend provides it, so
# that downstream probing observes capability rather than an importable
# placeholder that would raise later. Same pattern upstream FlashInfer uses for
# symbols whose backend may be absent.
try:
    from .checkpointing_ssu import CheckpointingSSURunner as CheckpointingSSURunner
except ImportError:
    pass
else:
    __all__ = [*__all__, "CheckpointingSSURunner"]
