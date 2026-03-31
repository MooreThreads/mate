import torch
from typing import Optional

_num_sms: Optional[int] = None


def get_num_sms() -> int:
    """Return the effective SM count for kernel dispatch.

    Returns the value set by set_num_sms() if called,
    otherwise returns the physical SM count of the current device.
    """
    if _num_sms is not None:
        return _num_sms
    return torch.musa.get_device_properties(
        torch.musa.current_device()
    ).multi_processor_count


def set_num_sms(num: Optional[int]) -> None:
    """Limit the maximum SM count available to kernel dispatch.

    Pass None to reset to device default.

    NOTE: Currently a stub. To take effect requires plumbing through
    to the C++ kernel launch layer. Tracked as a follow-up task.
    """
    global _num_sms
    assert num is None or (isinstance(num, int) and num > 0), (
        f"num_sms must be a positive int or None, got {num}"
    )
    _num_sms = num


_tc_util: float = 1.0


def get_tc_util() -> float:
    """Return the tensor-core utilization ratio used by heuristics.

    mate kernels do not consume it internally.
    """
    return _tc_util


def set_tc_util(ratio: float) -> None:
    """Set the tensor-core utilization ratio used by heuristics.

    get_tc_util(). Has no effect on kernel dispatch on MUSA.
    """
    global _tc_util
    assert 0.0 < ratio <= 1.0, f"tc_util ratio must be in (0, 1], got {ratio}"
    _tc_util = ratio
