"""Architecture dispatch for TileLang KDA decode kernels.

This release line carries the MP31 (PH1/QY2) implementation only.
"""

import torch

__all__ = ["run_gated_delta_rule_decode_vk_fp32"]


def _get_musa_compute_capability(tensor: torch.Tensor) -> int:
    if not tensor.is_musa or not hasattr(torch, "musa"):
        raise RuntimeError("KDA decode requires a MUSA tensor.")

    device = tensor.device.index
    if device is None:
        device = torch.musa.current_device()
    major, minor = torch.musa.get_device_capability(device)
    return int(major) * 10 + int(minor)


def run_gated_delta_rule_decode_vk_fp32(*args, **kwargs):
    if args:
        q = args[0]
    else:
        try:
            q = kwargs["q"]
        except KeyError as exc:
            raise TypeError("missing required argument: q") from exc

    musa_cc = _get_musa_compute_capability(q)
    if musa_cc != 31:
        raise RuntimeError(
            "KDA decode on this branch supports MUSA compute capability 3.1 "
            f"only, got {musa_cc // 10}.{musa_cc % 10}."
        )

    from .mp31.kda_decode import run_gated_delta_rule_decode_vk_fp32 as impl

    return impl(*args, **kwargs)
