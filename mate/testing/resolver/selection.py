"""Select tests by intersecting a Git diff with operation dependencies."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from .catalog import OperationCatalog
from .dependencies import (
    DependencyRecord,
    OperationDependencyResolver,
    OperationResolutionResult,
)
from .git import ChangedPath, GitRepository
from .policy import SelectionPolicy


@dataclass(frozen=True, slots=True)
class SelectionMatch:
    changed_path: str
    test: str
    root: str
    record: DependencyRecord
    snapshots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IncrementalSelectionResult:
    mode: Literal["skip", "selected", "full"]
    reason: str
    target_ref: str
    target_commit: str | None
    merge_base: str | None
    head_ref: str
    head_commit: str | None
    change_source: str
    changes: tuple[ChangedPath, ...]
    roots: tuple[str, ...]
    ownership_roots: tuple[str, ...]
    selected_tests: tuple[str, ...]
    not_selected_tests: tuple[str, ...]
    matches: tuple[SelectionMatch, ...]
    ownership_matches: tuple[SelectionMatch, ...]
    ignored_paths: tuple[str, ...]
    unowned_paths: tuple[str, ...]
    forced_full_paths: tuple[str, ...]
    resolution_issues: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "target": {"ref": self.target_ref, "commit": self.target_commit},
            "merge_base": self.merge_base,
            "head": {"ref": self.head_ref, "commit": self.head_commit},
            "change_source": self.change_source,
            "changes": [asdict(change) for change in self.changes],
            "roots": list(self.roots),
            "ownership_roots": list(self.ownership_roots),
            "selected_tests": list(self.selected_tests),
            "not_selected_tests": list(self.not_selected_tests),
            "matches": [_match_as_dict(match) for match in self.matches],
            "ownership_matches": [
                _match_as_dict(match) for match in self.ownership_matches
            ],
            "ignored_paths": list(self.ignored_paths),
            "unowned_paths": list(self.unowned_paths),
            "forced_full_paths": list(self.forced_full_paths),
            "resolution_issues": list(self.resolution_issues),
        }


class IncrementalTestSelector:
    def __init__(
        self,
        repository: Path,
        catalog: OperationCatalog,
        *,
        policy: SelectionPolicy | None = None,
    ) -> None:
        self.repository = repository.resolve()
        self.catalog = catalog
        self.policy = policy or SelectionPolicy()
        self.git = GitRepository(self.repository)

    def select(
        self,
        roots: Sequence[str],
        *,
        target: str,
        head: str,
        changed_files: Sequence[str] = (),
        force_full_reason: str | None = None,
    ) -> IncrementalSelectionResult:
        requested_roots = tuple(dict.fromkeys(roots))
        for root in requested_roots:
            self.catalog.get(root)
        ownership_roots = self.catalog.names
        all_tests = _full_test_suite(self.repository, self.catalog)
        if force_full_reason is not None:
            if not force_full_reason:
                raise ValueError("forced-full reason must not be empty")
            return IncrementalSelectionResult(
                mode="full",
                reason=force_full_reason,
                target_ref=target,
                target_commit=None,
                merge_base=None,
                head_ref=head,
                head_commit=None,
                change_source="forced full; Git diff not evaluated",
                changes=(),
                roots=ownership_roots,
                ownership_roots=ownership_roots,
                selected_tests=all_tests,
                not_selected_tests=(),
                matches=(),
                ownership_matches=(),
                ignored_paths=(),
                unowned_paths=(),
                forced_full_paths=(),
                resolution_issues=(),
            )

        target_commit = self.git.resolve_commit(target)
        head_commit = self.git.resolve_commit(head)
        merge_base = self.git.merge_base(target_commit, head_commit)
        if changed_files:
            changes = tuple(
                ChangedPath("M", _normalize_changed_path(path))
                for path in changed_files
            )
            worktree_changes: tuple[ChangedPath, ...] = ()
            change_source = "manual --changed-file"
        else:
            committed_changes = self.git.changed_paths(merge_base, head_commit)
            current_head_commit = self.git.resolve_commit("HEAD")
            worktree_changes = (
                self.git.working_tree_changed_paths()
                if head_commit == current_head_commit
                else ()
            )
            changes = _merge_changed_paths(committed_changes, worktree_changes)
            change_source = (
                "git diff + working tree" if worktree_changes else "git diff"
            )

        resolutions: list[tuple[str, OperationResolutionResult]] = []
        with ExitStack() as stack:
            target_snapshot = stack.enter_context(self.git.snapshot(target_commit))
            head_snapshot = (
                target_snapshot
                if head_commit == target_commit
                else stack.enter_context(self.git.snapshot(head_commit))
            )
            snapshots = [("target", target_snapshot), ("head", head_snapshot)]
            if worktree_changes:
                snapshots.append(("worktree", None))
            resolvers: dict[Path, OperationDependencyResolver] = {}
            resolution_cache: dict[tuple[Path, str], OperationResolutionResult] = {}
            for root in ownership_roots:
                for label, snapshot in snapshots:
                    snapshot_root = (
                        self.repository if snapshot is None else snapshot.root
                    )
                    resolver = resolvers.get(snapshot_root)
                    if resolver is None:
                        resolver = OperationDependencyResolver(
                            snapshot_root, self.catalog
                        )
                        resolvers[snapshot_root] = resolver
                    key = (snapshot_root, root)
                    result = resolution_cache.get(key)
                    if result is None:
                        result = resolver.resolve(root)
                        resolution_cache[key] = result
                    resolutions.append((label, result))

        ownership_matches = _match_changes(changes, resolutions)
        requested_root_set = set(requested_roots)
        matches = tuple(
            match for match in ownership_matches if match.root in requested_root_set
        )
        owned_paths = {match.changed_path for match in ownership_matches}
        catalog_definition_paths = {
            path.as_posix() for path in self.catalog.definition_files
        }
        catalog_changed_paths: set[str] = set()
        ignored_paths: set[str] = set()
        unowned_paths: set[str] = set()
        forced_full_paths: set[str] = set()
        for changed_path in _changed_dependency_paths(changes):
            if changed_path in catalog_definition_paths:
                catalog_changed_paths.add(changed_path)
                forced_full_paths.add(changed_path)
            elif self.policy.forces_full(changed_path):
                forced_full_paths.add(changed_path)
            elif changed_path in owned_paths:
                continue
            elif self.policy.ignores(changed_path):
                ignored_paths.add(changed_path)
            else:
                unowned_paths.add(changed_path)

        issues = tuple(
            sorted(
                {
                    f"{snapshot}:{result.root}:{issue.kind}:{issue.source}:"
                    f"{issue.line or '-'}:{issue.detail}"
                    for snapshot, result in resolutions
                    if result.root in requested_root_set
                    for issue in result.issues
                }
            )
        )
        if forced_full_paths or unowned_paths:
            return IncrementalSelectionResult(
                mode="full",
                reason=(
                    "catalog_changed"
                    if catalog_changed_paths
                    else "policy_full"
                    if forced_full_paths
                    else "unowned_paths"
                ),
                target_ref=target,
                target_commit=target_commit,
                merge_base=merge_base,
                head_ref=head,
                head_commit=head_commit,
                change_source=change_source,
                changes=changes,
                roots=ownership_roots,
                ownership_roots=ownership_roots,
                selected_tests=all_tests,
                not_selected_tests=(),
                matches=matches,
                ownership_matches=ownership_matches,
                ignored_paths=tuple(sorted(ignored_paths)),
                unowned_paths=tuple(sorted(unowned_paths)),
                forced_full_paths=tuple(sorted(forced_full_paths)),
                resolution_issues=issues,
            )

        selected_tests = {match.test for match in matches}
        if issues and changes:
            selected_tests.update(
                test
                for _, result in resolutions
                if result.root in requested_root_set and result.issues
                for test in result.tests
            )
        selected_test_paths = tuple(sorted(selected_tests))
        return IncrementalSelectionResult(
            mode="selected" if selected_tests else "skip",
            reason=(
                "resolution_issues"
                if issues and changes
                else "dependency_match"
                if selected_tests
                else "no_changes"
                if not changes
                else "filtered_operators"
                if ownership_matches
                else "ignored_paths"
                if ignored_paths
                else "no_dependency_match"
            ),
            target_ref=target,
            target_commit=target_commit,
            merge_base=merge_base,
            head_ref=head,
            head_commit=head_commit,
            change_source=change_source,
            changes=changes,
            roots=requested_roots,
            ownership_roots=ownership_roots,
            selected_tests=selected_test_paths,
            not_selected_tests=tuple(
                test for test in all_tests if test not in selected_tests
            ),
            matches=matches,
            ownership_matches=ownership_matches,
            ignored_paths=tuple(sorted(ignored_paths)),
            unowned_paths=(),
            forced_full_paths=(),
            resolution_issues=issues,
        )


def _match_as_dict(match: SelectionMatch) -> dict[str, object]:
    return {
        "changed_path": match.changed_path,
        "test": match.test,
        "root": match.root,
        "snapshots": list(match.snapshots),
        "dependency": asdict(match.record),
    }


def _full_test_suite(repository: Path, catalog: OperationCatalog) -> tuple[str, ...]:
    tests = {
        test.as_posix() for name in catalog.names for test in catalog.get(name).tests
    }
    tests_root = repository / "tests"
    if tests_root.is_dir():
        tests.update(
            path.relative_to(repository).as_posix()
            for path in tests_root.rglob("test_*.py")
            if path.is_file()
        )
    return tuple(sorted(tests))


def _match_changes(
    changes: Sequence[ChangedPath],
    resolutions: Sequence[tuple[str, OperationResolutionResult]],
) -> tuple[SelectionMatch, ...]:
    snapshots_by_match: dict[tuple[str, str, str, DependencyRecord], set[str]] = {}
    for change in changes:
        for changed_path in change.dependency_paths:
            for snapshot, result in resolutions:
                for record in result.records:
                    if record.path != changed_path:
                        continue
                    for test in result.tests:
                        key = (changed_path, test, result.root, record)
                        snapshots_by_match.setdefault(key, set()).add(snapshot)
    return tuple(
        SelectionMatch(path, test, root, record, tuple(sorted(snapshots)))
        for (path, test, root, record), snapshots in sorted(
            snapshots_by_match.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3].owner.kernel or "",
                item[0][3].owner.provider,
                item[0][3].via,
            ),
        )
    )


def _normalize_changed_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"changed path must be repository-relative: {value!r}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("changed path must not be empty")
    return normalized


def _changed_dependency_paths(changes: Sequence[ChangedPath]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(path for change in changes for path in change.dependency_paths)
    )


def _merge_changed_paths(
    *change_sets: Sequence[ChangedPath],
) -> tuple[ChangedPath, ...]:
    changes_by_identity: dict[tuple[str, str | None], ChangedPath] = {}
    for changes in change_sets:
        for change in changes:
            identity = (change.path, change.old_path)
            changes_by_identity[identity] = change
    return tuple(changes_by_identity.values())


def render_text(
    result: IncrementalSelectionResult,
    *,
    color: bool = False,
) -> str:
    lines = [
        "MATE incremental test selection",
        f"Outcome: {result.mode} ({result.reason})",
        f"Target: {_render_revision(result.target_ref, result.target_commit)}",
        f"Merge base: {_render_commit(result.merge_base)}",
        f"Head: {_render_revision(result.head_ref, result.head_commit)}",
        f"Changes: {len(result.changes)} paths from {result.change_source}",
        "Dependency snapshots: " + _dependency_snapshots(result),
        f"Operators: {', '.join(result.roots)}",
        f"Ownership catalog: {', '.join(result.ownership_roots)}",
        "",
        "Changed paths:",
    ]
    if result.changes:
        for change in result.changes:
            rename = f"{change.old_path} -> " if change.old_path else ""
            lines.append(f"  {change.status:<4} {rename}{change.path}")
    else:
        lines.append("  (none)")

    if result.forced_full_paths:
        lines.extend(("", "Paths requiring full selection:"))
        lines.extend(f"  {path}" for path in result.forced_full_paths)
    if result.unowned_paths:
        lines.extend(("", "Unowned paths (full fallback):"))
        lines.extend(f"  {path}" for path in result.unowned_paths)
    if result.ignored_paths:
        lines.extend(("", "Ignored paths:"))
        lines.extend(f"  {path}" for path in result.ignored_paths)

    total_tests = len(result.selected_tests) + len(result.not_selected_tests)
    selected_heading = f"Selected tests [{len(result.selected_tests)} / {total_tests}]:"
    lines.extend(("", _colorize(selected_heading, "32", enabled=color)))
    if result.selected_tests:
        lines.extend(
            _colorize(f"  {test}", "32", enabled=color)
            for test in result.selected_tests
        )
    else:
        lines.append(_colorize("  (none)", "32", enabled=color))

    not_selected_heading = (
        f"Not selected tests [{len(result.not_selected_tests)} / {total_tests}]:"
    )
    lines.extend(("", _colorize(not_selected_heading, "33", enabled=color)))
    if result.not_selected_tests:
        lines.extend(
            _colorize(f"  {test}", "33", enabled=color)
            for test in result.not_selected_tests
        )
    else:
        lines.append(_colorize("  (none)", "33", enabled=color))

    if result.matches:
        lines.extend(("", "Why selected:"))
        for match in result.matches:
            owner = match.record.owner
            label = owner.kernel or "operator"
            snapshots = "+".join(match.snapshots)
            lines.append(f"  {match.test}")
            lines.append(f"    <- changed {match.changed_path}")
            lines.append(
                f"    <- {match.root} / {label} / {owner.provider} "
                f"({match.record.via}; {snapshots})"
            )

    filtered_matches = tuple(
        match for match in result.ownership_matches if match.root not in result.roots
    )
    if filtered_matches:
        lines.extend(("", "Owned outside operator filter:"))
        for path, root in sorted(
            {(match.changed_path, match.root) for match in filtered_matches}
        ):
            lines.append(f"  {path} <- {root}")

    if result.resolution_issues:
        lines.extend(
            (
                "",
                "Resolution issues (fail closed when changes exist):",
                *(f"  {issue}" for issue in result.resolution_issues),
            )
        )
    return "\n".join(lines)


def _colorize(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _render_revision(ref: str, commit: str | None) -> str:
    return f"{ref} ({_render_commit(commit)})"


def _render_commit(commit: str | None) -> str:
    return commit[:12] if commit is not None else "not evaluated"


def _dependency_snapshots(result: IncrementalSelectionResult) -> str:
    if result.change_source.startswith("forced full"):
        return "not evaluated (forced full)"
    if result.change_source == "git diff + working tree":
        return "target + head + worktree"
    return "target + head"
