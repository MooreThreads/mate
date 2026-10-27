"""Command-line interface for a repository-provided operator catalog."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from .catalog import OperationCatalog
from .dependencies import OperationDependencyResolver
from .dependencies import render_text as render_dependencies
from .git import GitSnapshotError
from .policy import SelectionPolicy
from .selection import IncrementalSelectionResult, IncrementalTestSelector
from .selection import render_text as render_selection
from .validation import render_text as render_validation
from .validation import validate_catalog


_CI_ENVIRONMENT_VARIABLES = (
    "CI",
    "GITLAB_CI",
    "JENKINS_HOME",
    "JENKINS_SERVER_COOKIE",
    "JENKINS_URL",
)


@dataclass(frozen=True, slots=True)
class TestRunRequest:
    """Selection and command helper passed to a repository-defined test runner."""

    repository: Path
    selection: IncrementalSelectionResult
    verbose: bool = False

    def run(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str | None] | None = None,
    ) -> int:
        """Run one command, optionally overriding or removing environment values."""
        if isinstance(command, str) or not command:
            raise ValueError(
                "test runner command must be a non-empty argument sequence"
            )
        argv = [str(argument) for argument in command]
        if self.verbose:
            print(f"+ {shlex.join(argv)}", flush=True)
        process_environment = os.environ.copy()
        for name, value in (environment or {}).items():
            if value is None:
                process_environment.pop(name, None)
            else:
                process_environment[name] = value
        completed = subprocess.run(
            argv,
            cwd=self.repository,
            env=process_environment,
            check=False,
        )
        return completed.returncode


TestRunner = Callable[[TestRunRequest], int]


def find_repository(start: Path) -> Path:
    """Find the nearest Git-backed Python project containing ``start``."""
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("could not find repository root; pass --repo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve operator dependencies and select incremental tests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate the complete catalog and dependency graph"
    )
    validate.add_argument("--repo", type=Path, help="repository root")
    validate.add_argument("--json", action="store_true", help="emit JSON")

    resolve = subparsers.add_parser(
        "resolve", help="show the dependency closure for one operator"
    )
    resolve.add_argument("operator", help="registered operator name")
    resolve.add_argument("--repo", type=Path, help="repository root")
    resolve.add_argument(
        "--json", action="store_true", help="emit JSON with dependency provenance"
    )
    resolve.add_argument(
        "--files", action="store_true", help="show every dependency file"
    )
    resolve.add_argument(
        "--strict", action="store_true", help="fail on unresolved dependencies"
    )

    select = subparsers.add_parser(
        "select", help="select tests affected by changes from a target revision"
    )
    _add_selection_arguments(select)
    select.add_argument("--json", action="store_true", help="emit JSON")

    run = subparsers.add_parser(
        "run", help="select affected tests and invoke the configured test runner"
    )
    _add_selection_arguments(run)
    run.add_argument(
        "--verbose",
        action="store_true",
        help="show each command invoked by the configured test runner",
    )
    return parser


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "operators",
        nargs="*",
        help="registered operator names; defaults to the complete catalog",
    )
    parser.add_argument("--repo", type=Path, help="repository root")
    parser.add_argument("--target", default="origin/master", help="target branch/ref")
    parser.add_argument("--head", default="HEAD", help="head revision")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="override Git diff with one changed path; may be repeated",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="bypass dependency selection and select the complete catalog",
    )
    parser.add_argument(
        "--reason",
        help="machine-readable reason recorded for a forced full selection",
    )


def main(
    catalog: OperationCatalog,
    argv: Sequence[str] | None = None,
    *,
    runner: TestRunner | None = None,
    policy: SelectionPolicy | None = None,
) -> int:
    """Run the resolver CLI with an explicitly supplied repository catalog."""
    args = build_parser().parse_args(argv)
    try:
        repository = args.repo.resolve() if args.repo else find_repository(Path.cwd())
        if args.command == "validate":
            validation = validate_catalog(repository, catalog)
            print(
                json.dumps(validation.as_dict(), indent=2, sort_keys=True)
                if args.json
                else render_validation(validation)
            )
            return 0 if validation.valid else 1

        if args.command == "resolve":
            resolution = OperationDependencyResolver(repository, catalog).resolve(
                args.operator
            )
            print(
                json.dumps(resolution.as_dict(), indent=2, sort_keys=True)
                if args.json
                else render_dependencies(resolution, show_files=args.files)
            )
            return 1 if args.strict and resolution.issues else 0

        operators = tuple(args.operators) or catalog.names
        selection = IncrementalTestSelector(
            repository,
            catalog,
            policy=policy,
        ).select(
            operators,
            target=args.target,
            head=args.head,
            changed_files=args.changed_file,
            force_full_reason=_force_full_reason(args),
        )
        if args.command == "select":
            print(
                json.dumps(selection.as_dict(), indent=2, sort_keys=True)
                if args.json
                else render_selection(selection, color=_supports_color(sys.stdout))
            )
            return 0

        print(render_selection(selection, color=_supports_color(sys.stdout)))
        if not selection.selected_tests:
            print("No tests selected; the test runner was not invoked.")
            return 0
        if runner is None:
            print(
                "error: no test runner is configured; use select or pass runner= to "
                "mate.testing.resolver.cli.main",
                file=sys.stderr,
            )
            return 2
        return runner(TestRunRequest(repository, selection, args.verbose))
    except (GitSnapshotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _force_full_reason(args: argparse.Namespace) -> str | None:
    ci_mode = os.environ.get("MATE_CI_MODE", "").strip().lower()
    forced_by_environment = _environment_truthy("MATE_CI_FORCE_FULL")
    if not args.full and not forced_by_environment and ci_mode not in {"full", "daily"}:
        if args.reason:
            raise ValueError(
                "--reason requires --full or a forced-full environment mode"
            )
        return None
    if args.reason:
        return args.reason
    if ci_mode == "daily":
        return "daily"
    if ci_mode == "full":
        return "ci_mode_full"
    return "forced_full"


def _environment_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _supports_color(stream: object) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    if any(name in os.environ for name in _CI_ENVIRONMENT_VARIABLES):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())
