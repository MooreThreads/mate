"""Operation registration API for incremental dependency resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from .python_ast import PythonSymbolRef


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PythonAstDependency:
    roots: tuple[PythonSymbolRef, ...] = ()
    modules: tuple[str, ...] = ()
    extra_files: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class JitDependency:
    spec_refs: tuple[PythonSymbolRef, ...]


@dataclass(frozen=True, slots=True)
class FileDependency:
    files: tuple[Path, ...]


DependencySpec = PythonAstDependency | JitDependency | FileDependency


@dataclass(frozen=True, slots=True)
class KernelSpec:
    name: str
    backend: str
    dependencies: tuple[DependencySpec, ...]


@dataclass(frozen=True, slots=True)
class OperationSpec:
    name: str
    dependencies: tuple[FileDependency, ...]
    kernels: tuple[KernelSpec, ...]
    tests: tuple[Path, ...]
    interfaces: tuple[str, ...] = ()


class OperationCatalog:
    """Registry of operations and their top-level dependency entry points."""

    def __init__(
        self,
        operations: Sequence[OperationSpec] = (),
        *,
        definition_files: Sequence[str | Path] = (),
    ) -> None:
        self._operations: dict[str, OperationSpec] = {}
        self.definition_files = _unique(
            tuple(
                _python_file(value, field="catalog definition")
                for value in definition_files
            ),
            field="definition_files",
        )
        for operation in operations:
            self._add(operation)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operations))

    def register(
        self,
        name: str,
        *,
        interfaces: Sequence[str | Path],
        tests: Sequence[str | Path],
    ) -> OperationSpec:
        """Register one operation from only its public interfaces and test files."""
        normalized_name = _operation_name(name)
        normalized_interfaces = _unique(
            tuple(_interface_module(value) for value in interfaces),
            field="interfaces",
        )
        normalized_tests = _unique(
            tuple(_python_file(value, field="test") for value in tests),
            field="tests",
        )
        if not normalized_interfaces:
            raise ValueError(f"operator {normalized_name!r} must have interfaces")
        if not normalized_tests:
            raise ValueError(f"operator {normalized_name!r} must have tests")

        operation = OperationSpec(
            name=normalized_name,
            dependencies=(FileDependency(self.definition_files),)
            if self.definition_files
            else (),
            kernels=(
                KernelSpec(
                    "interface-ast",
                    "python",
                    (PythonAstDependency(modules=normalized_interfaces),),
                ),
            ),
            tests=normalized_tests,
            interfaces=normalized_interfaces,
        )
        self._add(operation)
        return operation

    def get(self, name: str) -> OperationSpec:
        try:
            return self._operations[name]
        except KeyError as exc:
            known = ", ".join(self.names)
            raise ValueError(
                f"unknown operator {name!r}; registered operators: {known or '(none)'}"
            ) from exc

    def _add(self, operation: OperationSpec) -> None:
        if operation.name in self._operations:
            raise ValueError(f"operator {operation.name!r} is already registered")
        self._operations[operation.name] = operation


def _operation_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("operator name must be a non-empty dotted name")
    if not all(part.isidentifier() for part in value.split(".")):
        raise ValueError(f"invalid operator name: {value!r}")
    return value


def _interface_module(value: str | Path) -> str:
    raw = value.as_posix() if isinstance(value, Path) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError("interface must be a non-empty module or Python file")
    if raw.endswith(".py") or "/" in raw or "\\" in raw:
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError(f"invalid interface Python file: {raw!r}")
        parts = path.with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
    else:
        parts = tuple(raw.split("."))
    if not parts or not all(part.isidentifier() for part in parts):
        raise ValueError(f"invalid interface module: {raw!r}")
    return ".".join(parts)


def _python_file(value: str | Path, *, field: str) -> Path:
    path = Path(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".py"
    ):
        raise ValueError(f"invalid {field} Python file: {value!r}")
    return path


def _unique(values: tuple[T, ...], *, field: str) -> tuple[T, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return values
