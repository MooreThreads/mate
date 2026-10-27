"""Resolve repository dependencies for registered MATE operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .catalog import (
    JitDependency,
    OperationCatalog,
    PythonAstDependency,
)
from .jit_spec import (
    JitSpecAnalysisError,
    JitSpecDependencyAnalyzer,
    JitSpecDependencyProvider,
)
from .python_ast import (
    ProviderBoundary,
    PythonDslDependencyAnalyzer,
    PythonSymbolRef,
    PythonUnresolvedReference,
)


@dataclass(frozen=True, slots=True)
class DependencyOwner:
    operation: str
    kernel: str | None
    backend: str | None
    provider: str


@dataclass(frozen=True, slots=True)
class DependencyRecord:
    path: str
    owner: DependencyOwner
    via: str


@dataclass(frozen=True, slots=True)
class ResolverIssue:
    owner: DependencyOwner
    source: str
    line: int | None
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class OperationResolutionResult:
    root: str
    tests: tuple[str, ...]
    records: tuple[DependencyRecord, ...]
    boundaries: tuple[ProviderBoundary, ...]
    issues: tuple[ResolverIssue, ...]

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(sorted({record.path for record in self.records}))

    def as_dict(self) -> dict[str, object]:
        owners_by_file: dict[str, list[dict[str, object]]] = {}
        for path in self.files:
            owners_by_file[path] = [
                {**asdict(record.owner), "via": record.via}
                for record in self.records
                if record.path == path
            ]
        return {
            "root": self.root,
            "tests": list(self.tests),
            "files": owners_by_file,
            "boundaries": [asdict(item) for item in self.boundaries],
            "issues": [asdict(item) for item in self.issues],
        }


class OperationDependencyResolver:
    """Compose provider results for one operation without runtime selection."""

    def __init__(
        self,
        repository: Path,
        catalog: OperationCatalog,
        *,
        provider_support_files: Mapping[str, tuple[Path, ...]] | None = None,
    ) -> None:
        self.repository = repository.resolve()
        self.catalog = catalog
        self.provider_support_files = dict(provider_support_files or {})
        self.jit_provider = JitSpecDependencyProvider(self.repository)

    def resolve(self, root: str) -> OperationResolutionResult:
        operation = self.catalog.get(root)
        records: set[DependencyRecord] = set()
        boundaries: set[ProviderBoundary] = set()
        issues: set[ResolverIssue] = set()
        jit_provider = self.jit_provider

        operation_owner = DependencyOwner(operation.name, None, None, "operation")
        self._add_files(
            records,
            _repository_configuration_files(self.repository),
            operation_owner,
            "repository-configuration",
        )
        for operation_dependency in operation.dependencies:
            self._add_files(
                records,
                operation_dependency.files,
                operation_owner,
                "operation-file",
            )

        for kernel in operation.kernels:
            for kernel_dependency in kernel.dependencies:
                if isinstance(kernel_dependency, PythonAstDependency):
                    owner = DependencyOwner(
                        operation.name, kernel.name, kernel.backend, "python-ast"
                    )
                    result = PythonDslDependencyAnalyzer(
                        self.repository,
                        reference_boundary=jit_provider.provider_for,
                    ).analyze(
                        kernel_dependency.roots,
                        modules=kernel_dependency.modules,
                    )
                    self._add_files(records, result.files, owner, "symbol-closure")
                    self._add_files(
                        records,
                        kernel_dependency.extra_files,
                        owner,
                        "explicit-python-file",
                    )
                    boundaries.update(result.boundaries)
                    issues.update(self._python_issues(owner, result.unresolved))
                    for boundary in result.boundaries:
                        if boundary.provider != "jit":
                            continue
                        jit_owner = DependencyOwner(
                            operation.name,
                            f"discovered:{boundary.target}",
                            "mate-jit",
                            "jit",
                        )
                        self._resolve_jit_reference(
                            PythonSymbolRef.parse(boundary.target),
                            owner=jit_owner,
                            records=records,
                            issues=issues,
                            provider=jit_provider,
                        )
                elif isinstance(kernel_dependency, JitDependency):
                    owner = DependencyOwner(
                        operation.name, kernel.name, kernel.backend, "jit"
                    )
                    for spec_ref in kernel_dependency.spec_refs:
                        self._resolve_jit_reference(
                            spec_ref,
                            owner=owner,
                            records=records,
                            issues=issues,
                        )
                else:
                    owner = DependencyOwner(
                        operation.name, kernel.name, kernel.backend, "file"
                    )
                    self._add_files(
                        records,
                        kernel_dependency.files,
                        owner,
                        "kernel-file",
                    )

            providers = {
                record.owner.provider
                for record in records
                if record.owner.kernel == kernel.name
            }
            for provider in providers:
                support_owner = DependencyOwner(
                    operation.name, kernel.name, kernel.backend, provider
                )
                self._add_files(
                    records,
                    self.provider_support_files.get(provider, ()),
                    support_owner,
                    "provider-support",
                )

        test_ast_owner = DependencyOwner(operation.name, None, None, "test-ast")
        for test in operation.tests:
            self._resolve_test_dependencies(
                test,
                owner=test_ast_owner,
                records=records,
                boundaries=boundaries,
                issues=issues,
                interfaces=operation.interfaces,
                jit_provider=jit_provider,
            )
        test_owner = DependencyOwner(operation.name, None, None, "test-target")
        self._add_files(records, operation.tests, test_owner, "test-target")
        return OperationResolutionResult(
            root=operation.name,
            tests=tuple(path.as_posix() for path in operation.tests),
            records=tuple(
                sorted(
                    records,
                    key=lambda record: (
                        record.path,
                        record.owner.operation,
                        record.owner.kernel or "",
                        record.owner.provider,
                        record.via,
                    ),
                )
            ),
            boundaries=tuple(
                sorted(
                    boundaries,
                    key=lambda item: (
                        item.target,
                        item.source,
                        item.line or 0,
                        item.provider,
                    ),
                )
            ),
            issues=tuple(
                sorted(
                    issues,
                    key=lambda item: (
                        item.owner.kernel or "",
                        item.source,
                        item.line or 0,
                        item.kind,
                        item.detail,
                    ),
                )
            ),
        )

    def _resolve_jit_reference(
        self,
        spec_ref: PythonSymbolRef,
        *,
        owner: DependencyOwner,
        records: set[DependencyRecord],
        issues: set[ResolverIssue],
        provider: JitSpecDependencyProvider | None = None,
    ) -> None:
        try:
            analysis = (
                provider.resolve(spec_ref)
                if provider is not None
                else JitSpecDependencyAnalyzer(self.repository).analyze(spec_ref)
            )
        except JitSpecAnalysisError as exc:
            issues.add(
                ResolverIssue(
                    owner,
                    str(spec_ref),
                    None,
                    "jit-spec",
                    str(exc),
                )
            )
            return
        result = analysis.resolved
        self._add_files(records, analysis.python.files, owner, "jit-python")
        self._add_files(records, result.direct_sources, owner, "jit-source")
        self._add_files(records, result.templates, owner, "jit-template")
        self._add_files(records, result.generator_files, owner, "jit-generator")
        self._add_files(records, result.headers, owner, "jit-header")
        self._add_files(records, result.configuration_files, owner, "jit-configuration")
        issues.update(self._python_issues(owner, analysis.python.unresolved))
        for unresolved_include in result.unresolved_includes:
            issues.add(
                ResolverIssue(
                    owner,
                    unresolved_include.source.as_posix(),
                    unresolved_include.line,
                    "jit-include",
                    f"{unresolved_include.reason}: {unresolved_include.target}",
                )
            )
        for unresolved_template in result.unresolved_templates:
            issues.add(
                ResolverIssue(
                    owner,
                    unresolved_template.source.as_posix(),
                    unresolved_template.line,
                    "jit-template",
                    f"{unresolved_template.reason}: {unresolved_template.target}",
                )
            )

    def _resolve_test_dependencies(
        self,
        test: Path,
        *,
        owner: DependencyOwner,
        records: set[DependencyRecord],
        boundaries: set[ProviderBoundary],
        issues: set[ResolverIssue],
        interfaces: Sequence[str],
        jit_provider: JitSpecDependencyProvider,
    ) -> None:
        test_path = self.repository / test
        conftests = _applicable_conftests(test_path, self.repository)
        modules = tuple(
            _module_name(path, self.repository) for path in (test_path, *conftests)
        )
        search_paths = tuple(
            dict.fromkeys(path.parent for path in (test_path, *conftests))
        )

        def test_boundary(reference: PythonSymbolRef) -> str | None:
            if reference.module in interfaces:
                return "operator-interface"
            return jit_provider.provider_for(reference)

        result = PythonDslDependencyAnalyzer(
            self.repository,
            reference_boundary=test_boundary,
            module_search_paths=search_paths,
        ).analyze(modules=modules)
        self._add_files(records, result.files, owner, "test-symbol-closure")
        boundaries.update(result.boundaries)
        issues.update(self._python_issues(owner, result.unresolved))
        for boundary in result.boundaries:
            if boundary.provider != "jit":
                continue
            jit_owner = DependencyOwner(
                owner.operation,
                f"discovered:{boundary.target}",
                "mate-jit",
                "jit",
            )
            self._resolve_jit_reference(
                PythonSymbolRef.parse(boundary.target),
                owner=jit_owner,
                records=records,
                issues=issues,
                provider=jit_provider,
            )

    def _add_files(
        self,
        records: set[DependencyRecord],
        paths: Iterable[Path | str],
        owner: DependencyOwner,
        via: str,
    ) -> None:
        for value in paths:
            path = Path(value)
            absolute = path if path.is_absolute() else self.repository / path
            try:
                relative = absolute.resolve().relative_to(self.repository).as_posix()
            except ValueError:
                continue
            records.add(DependencyRecord(relative, owner, via))

    @staticmethod
    def _python_issues(
        owner: DependencyOwner,
        unresolved: Sequence[PythonUnresolvedReference],
    ) -> set[ResolverIssue]:
        return {
            ResolverIssue(owner, item.source, item.line, item.kind, item.detail)
            for item in unresolved
        }


def _applicable_conftests(test: Path, repository: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    current = test.parent.resolve()
    repository = repository.resolve()
    while True:
        conftest = current / "conftest.py"
        if conftest.is_file():
            paths.append(conftest)
        if current == repository:
            break
        if repository not in current.parents:
            break
        current = current.parent
    return tuple(paths)


def _repository_configuration_files(repository: Path) -> tuple[Path, ...]:
    path = repository / "pyproject.toml"
    return (Path("pyproject.toml"),) if path.is_file() else ()


def _module_name(path: Path, repository: Path) -> str:
    try:
        relative = path.resolve().relative_to(repository.resolve()).with_suffix("")
    except ValueError as exc:
        raise ValueError(f"test dependency escapes repository: {path}") from exc
    if not relative.parts or not all(part.isidentifier() for part in relative.parts):
        raise ValueError(
            f"test dependency is not an importable Python path: {relative}"
        )
    return ".".join(relative.parts)


def render_text(result: OperationResolutionResult, *, show_files: bool) -> str:
    owners = {record.owner for record in result.records}
    backend_owners = sorted(
        (owner for owner in owners if owner.kernel is not None),
        key=lambda owner: (owner.backend or "", owner.kernel or "", owner.provider),
    )
    lines = [
        "MATE operator dependency resolution",
        f"Operator: {result.root}",
        "Runtime selection: disabled; all operator backends resolved",
        f"Summary: {len(result.files)} files, {len(backend_owners)} backend/provider owners, "
        f"{len(result.boundaries)} provider boundaries, {len(result.issues)} issues",
        "",
        "Backends:",
    ]
    for owner in backend_owners:
        count = len({record.path for record in result.records if record.owner == owner})
        lines.append(
            f"  {owner.kernel} [{owner.backend} / {owner.provider}]: {count} files"
        )
    lines.extend(("", "Tests:", *(f"  {test}" for test in result.tests)))

    if show_files:
        lines.extend(("", "Dependency files:"))
        for path in result.files:
            path_records = [record for record in result.records if record.path == path]
            owner_labels = ", ".join(
                sorted(
                    {
                        f"{record.owner.kernel or 'operator'}:{record.owner.provider}"
                        for record in path_records
                    }
                )
            )
            lines.append(f"  {path}")
            lines.append(f"    <- {owner_labels}")

    if result.boundaries:
        lines.extend(("", "Provider boundaries (AST did not cross):"))
        for boundary in result.boundaries:
            lines.append(
                f"  {boundary.source}:{boundary.line or '-'} -> "
                f"{boundary.target} [{boundary.provider}]"
            )
    if result.issues:
        lines.extend(("", "Issues:"))
        for issue in result.issues:
            location = f"{issue.source}:{issue.line}" if issue.line else issue.source
            lines.append(f"  {location} [{issue.kind}] {issue.detail}")
    return "\n".join(lines)
