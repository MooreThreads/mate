from dataclasses import dataclass

import torch

from ..common import Arch


@dataclass(frozen=True)
class FlashAttentionBwdMubinId:
    arch: Arch
    dtype: torch.dtype
    is_causal: bool
    headdim_qk: int
    is_gqa: bool
    # False: original packed ABI; True: reserved non-varlen persistent ABI.
    is_persistence: bool = False
