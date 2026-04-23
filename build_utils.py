"""Build utilities for version management and git operations."""

import os
import subprocess
from pathlib import Path
from typing import Optional


def get_musa_version_suffix() -> str:
    """Get MUSA version suffix for wheel name.

    Returns:
        MUSA version string like "mu436" for MUSA 4.3.6 or "mu-unknown".
    """
    musa_path = os.environ.get("MUSA_PATH", "/usr/local/musa")
    musa_header = os.path.join(musa_path, "include", "musa.h")

    if not os.path.exists(musa_header):
        return "mu-unknown"

    try:
        with open(musa_header, "r") as f:
            for line in f:
                if "MUSA_VERSION" in line and "define" in line:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "MUSA_VERSION":
                        try:
                            version_int = int(parts[2])
                            # Convert version integer (e.g., 50100 for 5.1.0) to suffix
                            major = version_int // 10000
                            minor = (version_int % 10000) // 100
                            patch = version_int % 100
                            return f"mu{major}{minor}{patch}"
                        except ValueError:
                            continue
    except Exception:
        pass
    return "mu-unknown"


def get_git_version(cwd: Optional[Path] = None) -> str:
    """Get git commit hash (full).

    Args:
        cwd: Working directory for git command. If None, uses current directory.

    Returns:
        Git commit hash or "unknown" if git is not available.
    """
    try:
        git_version = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return git_version
    except Exception:
        return "unknown"


def get_git_short_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Get short git commit hash (7 characters).

    Args:
        cwd: Working directory for git command. If None, uses current directory.

    Returns:
        Short commit hash (e.g., "g40c8139") or None if git is not available.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return f"g{commit}"
    except Exception:
        return None


def build_version_string(
    base_version: str,
    cwd: Optional[Path] = None,
    dev_suffix: str = "",
    local_version: Optional[str] = None,
) -> tuple[str, str]:
    """Build full version string with MUSA info (git version separate).

    Args:
        base_version: Base version from version.txt (e.g., "0.1.2")
        cwd: Working directory for git commands
        dev_suffix: Optional dev release suffix (e.g., "1" for .dev1)
        local_version: Optional additional local version string

    Returns:
        Tuple of (full_version, git_version)
        - full_version: e.g., "0.1.2+mu436" (no git commit in package name)
        - git_version: full git commit hash for CLI display
    """
    version = base_version

    # Add dev suffix if specified
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version (full hash for metadata/CLI display)
    git_version = get_git_version(cwd=cwd)

    # Build local version parts.
    # Note: git commit is NOT included in package name. The wheel local suffix
    # still carries the MUSA SDK version plus any explicit local suffix.
    local_parts = []

    # Add MUSA version
    musa_suffix = get_musa_version_suffix()
    local_parts.append(musa_suffix)

    # Append additional local version suffix if available
    if local_version:
        local_parts.append(local_version)

    # Combine all parts
    if local_parts:
        local_version_str = "".join(local_parts)
        version = f"{version}+{local_version_str}"

    return version, git_version
