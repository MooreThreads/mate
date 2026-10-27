"""Read-only Git snapshots and changed-path discovery for test selection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Iterator


class GitSnapshotError(ValueError):
    """Raised when a repository revision or archive cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ChangedPath:
    status: str
    path: str
    old_path: str | None = None

    @property
    def dependency_paths(self) -> tuple[str, ...]:
        """Return both sides of a rename/copy for dependency matching."""
        return (self.old_path, self.path) if self.old_path is not None else (self.path,)


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    revision: str
    commit: str
    root: Path

    def read_text(self, path: str | Path) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def exists(self, path: str | Path) -> bool:
        return (self.root / path).exists()

    def list_files(self, prefix: str | Path = ".") -> tuple[str, ...]:
        base = self.root / prefix
        if not base.exists():
            return ()
        return tuple(
            sorted(
                path.relative_to(self.root).as_posix()
                for path in base.rglob("*")
                if path.is_file()
            )
        )


class GitRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not (self.root / ".git").exists():
            raise GitSnapshotError(f"not a Git repository: {self.root}")

    def resolve_commit(self, revision: str) -> str:
        return self._run_text("rev-parse", "--verify", f"{revision}^{{commit}}")

    def merge_base(self, target: str, head: str) -> str:
        return self._run_text("merge-base", target, head)

    def changed_paths(self, base: str, head: str) -> tuple[ChangedPath, ...]:
        output = self._run_bytes(
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--ignore-submodules=none",
            base,
            head,
        )
        return _parse_name_status(output)

    def working_tree_changed_paths(self) -> tuple[ChangedPath, ...]:
        """Return staged, unstaged, and untracked changes in the checkout."""
        output = self._run_bytes(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        )
        return _parse_porcelain_status(output)

    @contextmanager
    def snapshot(self, revision: str) -> Iterator[RepositorySnapshot]:
        commit = self.resolve_commit(revision)
        archive = self._run_bytes("archive", "--format=tar", commit)
        with tempfile.TemporaryDirectory(prefix="mate-resolver-snapshot-") as value:
            root = Path(value)
            _extract_git_archive(archive, root)
            yield RepositorySnapshot(revision, commit, root)

    def _run_text(self, *arguments: str) -> str:
        return self._run_bytes(*arguments).decode("utf-8").strip()

    def _run_bytes(self, *arguments: str) -> bytes:
        try:
            process = subprocess.run(
                ("git", *arguments),
                cwd=self.root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                exc.stderr.decode("utf-8", errors="replace").strip()
                if isinstance(exc, subprocess.CalledProcessError)
                else str(exc)
            )
            command = "git " + " ".join(arguments)
            raise GitSnapshotError(f"{command} failed: {detail}") from exc
        return process.stdout


def _parse_name_status(output: bytes) -> tuple[ChangedPath, ...]:
    fields = output.decode("utf-8", errors="surrogateescape").split("\0")
    if fields and not fields[-1]:
        fields.pop()
    changes: list[ChangedPath] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        if index + path_count > len(fields):
            raise GitSnapshotError("malformed git diff --name-status output")
        if path_count == 2:
            old_path, path = fields[index : index + 2]
            changes.append(ChangedPath(status, path, old_path))
        else:
            changes.append(ChangedPath(status, fields[index]))
        index += path_count
    return tuple(changes)


def _parse_porcelain_status(output: bytes) -> tuple[ChangedPath, ...]:
    fields = output.decode("utf-8", errors="surrogateescape").split("\0")
    if fields and not fields[-1]:
        fields.pop()
    changes: list[ChangedPath] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if len(entry) < 4 or entry[2] != " ":
            raise GitSnapshotError("malformed git status --porcelain output")
        status, path = entry[:2], entry[3:]
        if "R" in status or "C" in status:
            if index >= len(fields):
                raise GitSnapshotError("malformed git status --porcelain output")
            old_path = fields[index]
            index += 1
            changes.append(ChangedPath(status, path, old_path))
        else:
            changes.append(ChangedPath(status, path))
    return tuple(changes)


def _extract_git_archive(content: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=BytesIO(content), mode="r:") as archive:
        for member in archive.getmembers():
            _validate_archive_member(member, destination)
        archive.extractall(destination, filter="fully_trusted")


def _validate_archive_member(member: tarfile.TarInfo, destination: Path) -> None:
    root = destination.resolve()
    member_path = (root / member.name).resolve()
    if not _is_relative_to(member_path, root):
        raise GitSnapshotError(f"archive path escapes snapshot: {member.name}")
    if member.issym():
        target = (member_path.parent / member.linkname).resolve()
        if not _is_relative_to(target, root):
            raise GitSnapshotError(f"archive symlink escapes snapshot: {member.name}")
    elif member.islnk():
        target = (root / member.linkname).resolve()
        if not _is_relative_to(target, root):
            raise GitSnapshotError(f"archive hardlink escapes snapshot: {member.name}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
