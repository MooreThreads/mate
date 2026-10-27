"""Validate an operator catalog against a repository checkout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import OperationCatalog
from .dependencies import OperationDependencyResolver


@dataclass(frozen=True, slots=True)
class CatalogValidationIssue:
    """One actionable catalog or dependency-graph validation failure."""

    kind: str
    detail: str
    operation: str | None = None
    source: str | None = None
    line: int | None = None


@dataclass(frozen=True, slots=True)
class CatalogValidationResult:
    """Repository-wide validation result for one operator catalog."""

    operators: tuple[str, ...]
    registered_tests: tuple[str, ...]
    discovered_tests: tuple[str, ...]
    dependency_files: tuple[str, ...]
    issues: tuple[CatalogValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "operators": list(self.operators),
            "registered_tests": list(self.registered_tests),
            "discovered_tests": list(self.discovered_tests),
            "dependency_file_count": len(self.dependency_files),
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_catalog(
    repository: Path,
    catalog: OperationCatalog,
) -> CatalogValidationResult:
    """Validate catalog files, pytest coverage, and every dependency closure."""
    repository = repository.resolve()
    issues: set[CatalogValidationIssue] = set()
    dependency_files: set[str] = set()

    if not catalog.names:
        issues.add(
            CatalogValidationIssue(
                "empty-catalog",
                "the operator catalog must register at least one operator",
            )
        )

    for path in catalog.definition_files:
        if not (repository / path).is_file():
            issues.add(
                CatalogValidationIssue(
                    "missing-catalog-definition",
                    "catalog definition file does not exist",
                    source=path.as_posix(),
                )
            )

    registered_tests = {
        test.as_posix() for name in catalog.names for test in catalog.get(name).tests
    }
    for name in catalog.names:
        operation = catalog.get(name)
        for interface in operation.interfaces:
            module_path = repository.joinpath(*interface.split("."))
            if (
                not module_path.is_dir()
                and not module_path.with_suffix(".py").is_file()
            ):
                issues.add(
                    CatalogValidationIssue(
                        "missing-interface",
                        "registered interface module does not exist",
                        operation=name,
                        source=interface,
                    )
                )
        for test in operation.tests:
            test_name = test.as_posix()
            if test.parts[0] != "tests" or not test.name.startswith("test_"):
                issues.add(
                    CatalogValidationIssue(
                        "invalid-test-target",
                        "registered tests must match tests/**/test_*.py",
                        operation=name,
                        source=test_name,
                    )
                )
            if not (repository / test).is_file():
                issues.add(
                    CatalogValidationIssue(
                        "missing-test-target",
                        "registered test file does not exist",
                        operation=name,
                        source=test_name,
                    )
                )

    tests_root = repository / "tests"
    discovered_tests = (
        {
            path.relative_to(repository).as_posix()
            for path in tests_root.rglob("test_*.py")
            if path.is_file()
        }
        if tests_root.is_dir()
        else set()
    )
    for unregistered_test in discovered_tests - registered_tests:
        issues.add(
            CatalogValidationIssue(
                "unregistered-test",
                "pytest file is not owned by any registered operator",
                source=unregistered_test,
            )
        )

    resolver = OperationDependencyResolver(repository, catalog)
    for name in catalog.names:
        result = resolver.resolve(name)
        dependency_files.update(result.files)
        for issue in result.issues:
            issues.add(
                CatalogValidationIssue(
                    f"dependency-{issue.kind}",
                    issue.detail,
                    operation=name,
                    source=issue.source,
                    line=issue.line,
                )
            )

    return CatalogValidationResult(
        operators=catalog.names,
        registered_tests=tuple(sorted(registered_tests)),
        discovered_tests=tuple(sorted(discovered_tests)),
        dependency_files=tuple(sorted(dependency_files)),
        issues=tuple(sorted(issues, key=_issue_sort_key)),
    )


def _issue_sort_key(issue: CatalogValidationIssue) -> tuple[object, ...]:
    return (
        issue.kind,
        issue.operation or "",
        issue.source or "",
        issue.line or 0,
        issue.detail,
    )


def render_text(result: CatalogValidationResult) -> str:
    """Render a concise pre-commit-friendly validation report."""
    lines = [
        "MATE test catalog validation",
        f"Status: {'valid' if result.valid else 'invalid'}",
        f"Summary: {len(result.operators)} operators, "
        f"{len(result.registered_tests)} registered tests, "
        f"{len(result.dependency_files)} dependency files, "
        f"{len(result.issues)} issues",
    ]
    if result.issues:
        lines.extend(("", "Issues:"))
        for issue in result.issues:
            owner = f"{issue.operation}: " if issue.operation else ""
            location = issue.source or "catalog"
            if issue.line is not None:
                location = f"{location}:{issue.line}"
            lines.append(f"  {owner}{location} [{issue.kind}] {issue.detail}")
    return "\n".join(lines)
