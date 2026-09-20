"""Compiled-kernel reuse for the SSD stage launchers.

A launcher asks its TileLang factory for the compiled kernel on every call. Each such
call rebuilds the specialization key from the arguments on the host, and any wrapper
installed around the JIT runs again, before the kernel is launched. The factories take
plain hashable arguments, so the compiled kernel is looked up once per key and reused.
"""

from __future__ import annotations

from typing import Any, Callable


class _Reuse:
    """Factory stand-in that returns the kernel compiled for the same arguments."""

    __slots__ = ("_factory", "_kernels")

    def __init__(self, factory: Callable[..., Any]) -> None:
        self._factory = factory
        self._kernels: dict = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = self._kernels[key] = self._factory(*args, **kwargs)
        return kernel

    def __getattr__(self, name: str) -> Any:
        return getattr(self._factory, name)


def reuse_compiled(module: Any, factory_name: str) -> None:
    """Route ``module.<factory_name>`` through a per-key compiled-kernel cache."""
    factory = getattr(module, factory_name)
    if not isinstance(factory, _Reuse):
        setattr(module, factory_name, _Reuse(factory))
