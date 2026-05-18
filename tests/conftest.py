from __future__ import annotations

from functools import cache
import warnings

import os
import hashlib
import pytest
import torch
from pathlib import Path

from mate.execution_context import MateDryRunComplete
from mate.testing.arch import MUSA_ARCH_CHECKER_ATTR, MUSA_ARCH_REQUIREMENT_ATTR


def _format_cc(cc: int) -> str:
    return f"MP{cc}"


def _format_arch_requirement(requirement: dict) -> str:
    if requirement["mode"] == "allowlist":
        ccs = ", ".join(_format_cc(cc) for cc in sorted(requirement["ccs"]))
        return f"one of [{ccs}]"
    if requirement["mode"] == "ge":
        return f">= {_format_cc(requirement['cc'])}"
    return "an unsupported MUSA architecture requirement"


@cache
def _current_musa_compute_capability() -> tuple[int | None, str]:
    try:
        import torch
        import torch_musa  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, f"unable to import torch/torch_musa: {exc}"

    if not hasattr(torch, "musa"):
        return None, "torch.musa is not available"

    try:
        if not torch.musa.is_available():
            return None, "MUSA is not available"
        if torch.musa.device_count() <= 0:
            return None, "no MUSA devices are available"
        device = torch.musa.current_device()
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, f"unable to initialize MUSA: {exc}"

    try:
        major, minor = torch.musa.get_device_capability(device)
    except Exception as capability_exc:
        try:
            properties = torch.musa.get_device_properties(device)
            major, minor = properties.major, properties.minor
        except Exception as properties_exc:  # pragma: no cover - environment dependent
            return (
                None,
                "unable to determine MUSA compute capability: "
                f"{capability_exc}; fallback failed: {properties_exc}",
            )

    try:
        cc = int(major) * 10 + int(minor)
    except (TypeError, ValueError) as exc:
        return None, f"invalid MUSA compute capability ({major!r}, {minor!r}): {exc}"

    return cc, f"current MUSA device {device} has {_format_cc(cc)}"


def _musa_arch_skip_reason(
    requirement: dict, current_cc: int | None, detail: str
) -> str:
    required = _format_arch_requirement(requirement)
    if current_cc is None:
        return f"requires MUSA compute capability {required}; {detail}"
    return (
        f"requires MUSA compute capability {required}; "
        f"detected {_format_cc(current_cc)} ({detail})"
    )


def _is_dry_run_complete(excinfo) -> bool:
    return excinfo is not None and excinfo.errisinstance(MateDryRunComplete)


def _is_musa_oom(excinfo) -> bool:
    if excinfo is None:
        return False

    musa_oom_types = tuple(
        exc
        for exc in (
            getattr(torch, "MusaOutOfMemory", None),
            getattr(getattr(torch, "musa", None), "OutOfMemoryError", None),
        )
        if exc is not None
    )
    if bool(musa_oom_types) and excinfo.errisinstance(musa_oom_types):
        return True

    # TVM-FFI/DLPack allocation failures can surface as a plain MemoryError.
    if excinfo.errisinstance(MemoryError):
        message = str(excinfo.value).lower()
        return "musa" in message and "out of memory" in message

    return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    current_cc, detail = _current_musa_compute_capability()

    for item in items:
        test_func = getattr(item, "obj", None)
        requirement = getattr(test_func, MUSA_ARCH_REQUIREMENT_ATTR, None)
        if requirement is None:
            continue

        checker = getattr(test_func, MUSA_ARCH_CHECKER_ATTR)
        if current_cc is None or not checker(current_cc):
            item.add_marker(
                pytest.mark.skip(
                    reason=_musa_arch_skip_reason(requirement, current_cc, detail)
                )
            )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """
    Treat selected runtime exceptions as non-failures for the test call phase.
    """
    report = yield

    if call.when == "call" and _is_dry_run_complete(call.excinfo):
        report.outcome = "passed"
        report.longrepr = None
    elif call.when == "call" and _is_musa_oom(call.excinfo):
        warnings.warn(
            # f"MUSA out of memory; skipping test: {call.excinfo.value}",
            "MUSA out of memory; skipping test",
            stacklevel=1,
        )
        report.outcome = "skipped"
        report.longrepr = ("", 0, "Skipped: MUSA out of memory")

    return report


def _get_shard_config():
    shard_total = int(os.environ.get("MATE_PYTEST_SHARD_TOTAL", "1"))
    shard_index = int(os.environ.get("MATE_PYTEST_SHARD_INDEX", "0"))
    mode = os.environ.get("MATE_PYTEST_SHARD_MODE", "file")

    if shard_total < 1:
        raise pytest.UsageError(
            f"MATE_PYTEST_SHARD_TOTAL must be >= 1, got {shard_total}"
        )
    if shard_index < 0 or shard_index >= shard_total:
        raise pytest.UsageError(
            f"PYTEST_SHARD_INDEX must be in the range [0, {shard_total}), got {shard_index}"
        )
    if mode not in ("file", "item"):
        raise pytest.UsageError(
            f"MATE_PYTEST_SHARD_MODE must be either 'file' or 'item', got {mode}"
        )

    return shard_total, shard_index, mode


def _stable_shard_id(key: Path, total: int) -> int:
    if "test_fmha.py" in key.name:
        return total - 1  # FA3 tests always at the end in file mode

    if total == 1:
        return 0

    digest = hashlib.blake2b(key.as_posix().encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (total - 1)


def pytest_ignore_collect(collection_path, config):
    total, index, mode = _get_shard_config()

    # NOTE: Only support file now.
    if total == 1 or mode != "file":
        return None

    path = Path(collection_path)
    if not path.name.startswith("test_") or path.suffix != ".py":
        return None

    try:
        key = path.relative_to(config.rootpath)
    except ValueError:
        key = path

    return _stable_shard_id(key, total) != index
