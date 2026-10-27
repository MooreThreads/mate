"""Repository-level policy for incremental test selection."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Classify repository paths independently from operator ownership."""

    ignored_paths: tuple[str, ...] = ()
    forced_full_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ignored_paths",
            _normalize_patterns(self.ignored_paths, field="ignored_paths"),
        )
        object.__setattr__(
            self,
            "forced_full_paths",
            _normalize_patterns(
                self.forced_full_paths,
                field="forced_full_paths",
            ),
        )

    def ignores(self, path: str) -> bool:
        return _matches(path, self.ignored_paths)

    def forces_full(self, path: str) -> bool:
        return _matches(path, self.forced_full_paths)


def _normalize_patterns(patterns: Sequence[str], *, field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{field} entries must be non-empty strings")
        candidate = pattern.replace("\\", "/")
        path = Path(candidate)
        if path.is_absolute() or ".." in path.parts or candidate in {"", "."}:
            raise ValueError(
                f"{field} entries must be repository-relative globs: {pattern!r}"
            )
        normalized.append(candidate)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(normalized)


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
