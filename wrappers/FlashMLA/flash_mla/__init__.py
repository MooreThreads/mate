from pathlib import Path
import re

from flash_mla.flash_mla_interface import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)


def _load_version() -> str:
    try:
        from importlib.metadata import version

        return version("flash_mla")
    except Exception:
        try:
            pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
            pyproject_text = pyproject_path.read_text()
            match = re.search(
                r"^\[project\]\s.*?^version = [\"']([^\"']+)[\"']",
                pyproject_text,
                re.MULTILINE | re.DOTALL,
            )
            if match is not None:
                return match.group(1)
            return "unknown"
        except Exception:
            return "unknown"


__version__ = _load_version()

__all__ = [
    "FlashMLASchedMeta",
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_mla_sparse_fwd",
]
