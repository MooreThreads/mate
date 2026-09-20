"""Resolution of the MATE implementation behind the FlashInfer MUSA mamba API.

The mamba family is the one part of this wrapper whose MATE backend may not be
installed yet, so it resolves lazily at call time instead of binding at import
time the way ``flashinfer.norm`` and ``flashinfer.gemm`` do. Two consequences
that downstream code depends on:

* ``import flashinfer.mamba`` succeeds on any MATE build, so import probes and
  ``from flashinfer.mamba... import ...`` statements keep working.
* A symbol that has no MATE backend raises an actionable error when *called*
  instead of failing silently or returning a wrong result. Capability probes
  must use :func:`has_symbol` rather than importing a name and assuming that
  calling it will work.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Single point of change if the MATE mamba family is renamed or relocated.
BACKEND_PACKAGE = "mate.mamba"

_CACHE: dict[str, Any] = {}


_ABSENT_NAMES = frozenset({BACKEND_PACKAGE, BACKEND_PACKAGE.split(".")[0]})


def _backend_module() -> Any | None:
    if "module" not in _CACHE:
        try:
            _CACHE["module"] = importlib.import_module(BACKEND_PACKAGE)
        except ModuleNotFoundError as exc:
            # Only "the backend (or MATE itself) is not installed" counts as
            # absence. A dependency missing *inside* MATE is a broken install
            # and must surface as its own error instead of masquerading as an
            # unimplemented API.
            if exc.name not in _ABSENT_NAMES:
                raise
            _CACHE["module"] = None
    return _CACHE["module"]


def has_symbol(name: str) -> bool:
    """Return whether the installed MATE build provides ``name``."""
    module = _backend_module()
    return module is not None and hasattr(module, name)


def resolve(name: str) -> Any:
    """Return the MATE implementation of the FlashInfer ``name``.

    Raises:
        NotImplementedError: when the installed MATE build does not provide the
            symbol. The message names both sides so the gap is unambiguous.
    """
    module = _backend_module()
    if module is None:
        raise NotImplementedError(
            f"flashinfer.mamba.{name} requires {BACKEND_PACKAGE}, which is not "
            f"importable in this environment. Install a MATE build that ships "
            f"the mamba/SSD family, or use the MATE default mamba backend."
        )
    try:
        return _CACHE.setdefault(name, getattr(module, name))
    except AttributeError:
        raise NotImplementedError(
            f"{BACKEND_PACKAGE} does not provide {name}, which flashinfer.mamba "
            f"declares. This MATE build predates that symbol; see the MATE "
            f"FlashInfer wrapper README for the supported surface."
        ) from None
