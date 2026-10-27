"""Build-input metadata and dependency resolution for MATE JIT specs."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
import fnmatch
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

_INCLUDE_DIRECTIVE = re.compile(r"^\s*#\s*include\s*(?P<target>.+?)\s*$")
_LITERAL_INCLUDE = re.compile(r'^(?:"(?P<quoted>[^"]+)"|<(?P<angled>[^>]+)>)')


@dataclass(frozen=True)
class JitSourceTemplate:
    """A generated source template rendered by JitSpec during materialization."""

    template: Path
    context: Mapping[str, object]
    prefix: str = ""
    suffix: str = ""

    def render(self) -> str:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        environment = Environment(
            loader=FileSystemLoader(self.template.parent.as_posix()),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        context = dict(self.context)
        prefix = environment.from_string(self.prefix).render(context)
        content = environment.get_template(self.template.name).render(context)
        suffix = environment.from_string(self.suffix).render(context)
        return prefix + content + suffix


@dataclass(frozen=True, slots=True)
class JitDependencySpec:
    """Repository inputs needed to generate and compile a JIT kernel family."""

    sources: tuple[Path, ...] = ()
    templates: tuple[Path, ...] = ()
    generator_files: tuple[Path, ...] = ()
    include_dirs: tuple[Path, ...] = ()
    header_seeds: tuple[Path, ...] = ()
    opaque_roots: tuple[Path, ...] = ()
    opaque_include_patterns: tuple[str, ...] = ()
    external_include_patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UnresolvedInclude:
    source: Path
    line: int | None
    target: str
    reason: str


@dataclass(frozen=True, slots=True)
class UnresolvedTemplate:
    source: Path
    line: int | None
    target: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResolvedJitDependencies:
    direct_sources: frozenset[Path]
    templates: frozenset[Path]
    generator_files: frozenset[Path]
    headers: frozenset[Path]
    configuration_files: frozenset[Path]
    generated_outputs: frozenset[Path]
    unresolved_includes: tuple[UnresolvedInclude, ...]
    unresolved_templates: tuple[UnresolvedTemplate, ...]

    @property
    def watched_files(self) -> frozenset[Path]:
        """Return repository files whose changes can affect this JIT family."""
        return (
            self.direct_sources
            | self.templates
            | self.generator_files
            | self.headers
            | self.configuration_files
        )


def resolve_jit_dependencies(
    spec: JitDependencySpec,
    *,
    repository: Path,
    generated_sources: Mapping[Path, str] | None = None,
) -> ResolvedJitDependencies:
    """Resolve repository-local inputs without generating or compiling anything."""
    repository = repository.resolve()
    generated_sources = generated_sources or {}
    generated_paths = {_absolute(path, repository) for path in generated_sources}
    gitlink_roots = _gitlink_roots(repository)

    source_inputs = tuple(
        _absolute(path, repository)
        for path in spec.sources
        if _absolute(path, repository) not in generated_paths
    )
    template_inputs = tuple(_absolute(path, repository) for path in spec.templates)
    generator_inputs = tuple(
        _absolute(path, repository) for path in spec.generator_files
    )
    header_inputs = tuple(_absolute(path, repository) for path in spec.header_seeds)
    declared_inputs = (
        *source_inputs,
        *template_inputs,
        *generator_inputs,
        *header_inputs,
    )

    unresolved: set[UnresolvedInclude] = set()
    unresolved_templates: set[UnresolvedTemplate] = set()
    direct_sources = _existing_repository_files(
        source_inputs,
        repository=repository,
        kind="source",
        unresolved=unresolved,
        gitlink_roots=gitlink_roots,
    )
    templates = _existing_repository_files(
        template_inputs,
        repository=repository,
        kind="template",
        unresolved=unresolved,
        gitlink_roots=gitlink_roots,
    )
    generator_files = _existing_repository_files(
        generator_inputs,
        repository=repository,
        kind="generator",
        unresolved=unresolved,
        gitlink_roots=gitlink_roots,
    )
    header_seeds = _existing_repository_files(
        header_inputs,
        repository=repository,
        kind="header seed",
        unresolved=unresolved,
        gitlink_roots=gitlink_roots,
    )
    include_dirs = tuple(_absolute(path, repository) for path in spec.include_dirs)
    declared_opaque_roots = tuple(
        _absolute(path, repository) for path in spec.opaque_roots
    )
    opaque_roots = tuple(dict.fromkeys((*declared_opaque_roots, *gitlink_roots)))

    templates.update(
        _resolve_template_fragments(
            templates,
            repository=repository,
            gitlink_roots=gitlink_roots,
            unresolved=unresolved_templates,
        )
    )

    headers: set[Path] = set(header_seeds)
    active_gitlink_roots = {
        root
        for root in gitlink_roots
        if any(
            _is_relative_to(path, root) for path in (*declared_inputs, *include_dirs)
        )
    }
    projected_gitlinks = {
        relative
        for root in active_gitlink_roots
        if (relative := _repository_relative(root, repository)) is not None
    }
    for opaque_root in (*declared_opaque_roots, *active_gitlink_roots):
        relative_opaque_root = _repository_relative(opaque_root, repository)
        if relative_opaque_root is not None:
            headers.add(relative_opaque_root)
    queue: list[tuple[Path, str]] = []
    for relative in sorted(direct_sources | templates | header_seeds):
        if relative in projected_gitlinks:
            continue
        path = repository / relative
        try:
            queue.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError) as exc:
            unresolved.add(
                UnresolvedInclude(relative, None, "", f"cannot read dependency: {exc}")
            )

    generated_outputs: set[Path] = set()
    for output, content in generated_sources.items():
        absolute_output = _absolute(output, repository)
        relative_output = _repository_relative(absolute_output, repository)
        if relative_output is not None:
            generated_outputs.add(relative_output)
        queue.append((absolute_output, content))

    scanned: set[Path] = set()
    while queue:
        source_path, content = queue.pop()
        if source_path in scanned:
            continue
        scanned.add(source_path)
        source_relative = _repository_relative(source_path, repository) or source_path
        for line_number, target, quoted in _extract_includes(content):
            if target is None:
                unresolved.add(
                    UnresolvedInclude(
                        source_relative,
                        line_number,
                        quoted,
                        "non-literal include",
                    )
                )
                continue
            resolved = _resolve_include(
                target,
                quoted=quoted == "quoted",
                source_dir=source_path.parent,
                include_dirs=include_dirs,
            )
            if resolved is None:
                if _matches_any(target, spec.opaque_include_patterns):
                    continue
                is_external = _matches_any(target, spec.external_include_patterns)
                if not is_external and _looks_repository_local(
                    target,
                    include_dirs,
                    repository=repository,
                ):
                    unresolved.add(
                        UnresolvedInclude(
                            source_relative,
                            line_number,
                            target,
                            "repository include not found",
                        )
                    )
                continue
            opaque_root = _containing_root(resolved, opaque_roots)
            if opaque_root is not None:
                relative_opaque_root = _repository_relative(opaque_root, repository)
                if relative_opaque_root is not None:
                    headers.add(relative_opaque_root)
                if opaque_root in gitlink_roots:
                    active_gitlink_roots.add(opaque_root)
                continue
            relative = _repository_relative(resolved, repository)
            if relative is None or relative in headers:
                continue
            headers.add(relative)
            try:
                queue.append((resolved, resolved.read_text(encoding="utf-8")))
            except (OSError, UnicodeError) as exc:
                unresolved.add(
                    UnresolvedInclude(
                        relative,
                        None,
                        "",
                        f"cannot read header: {exc}",
                    )
                )

    configuration_files = (
        frozenset({Path(".gitmodules")}) if active_gitlink_roots else frozenset()
    )
    return ResolvedJitDependencies(
        direct_sources=frozenset(direct_sources),
        templates=frozenset(templates),
        generator_files=frozenset(generator_files),
        headers=frozenset(headers),
        configuration_files=configuration_files,
        generated_outputs=frozenset(generated_outputs),
        unresolved_includes=tuple(
            sorted(
                unresolved,
                key=lambda item: (
                    item.source.as_posix(),
                    item.line or 0,
                    item.target,
                    item.reason,
                ),
            )
        ),
        unresolved_templates=tuple(
            sorted(
                unresolved_templates,
                key=lambda item: (
                    item.source.as_posix(),
                    item.line or 0,
                    item.target,
                    item.reason,
                ),
            )
        ),
    )


def _resolve_template_fragments(
    roots: Iterable[Path],
    *,
    repository: Path,
    gitlink_roots: Sequence[Path],
    unresolved: set[UnresolvedTemplate],
) -> set[Path]:
    """Resolve literal Jinja references with the renderer's loader semantics."""
    try:
        from jinja2 import (
            Environment,
            FileSystemLoader,
            TemplateNotFound,
            TemplateSyntaxError,
            meta,
        )
    except ImportError:
        unresolved.update(
            UnresolvedTemplate(
                source,
                None,
                "jinja2",
                "template analysis dependency is not installed",
            )
            for source in roots
        )
        return set()

    fragments: set[Path] = set()
    for relative_root in sorted(roots):
        absolute_root = repository / relative_root
        if _containing_root(absolute_root, gitlink_roots) is not None:
            continue
        if not absolute_root.is_file():
            continue

        loader_root = absolute_root.parent
        environment = Environment(loader=FileSystemLoader(loader_root.as_posix()))
        queue: list[tuple[str, Path]] = [(absolute_root.name, relative_root)]
        scanned: set[str] = set()
        while queue:
            template_name, source_relative = queue.pop()
            if template_name in scanned:
                continue
            scanned.add(template_name)
            try:
                source, filename, _ = environment.loader.get_source(
                    environment, template_name
                )
                parsed = environment.parse(
                    source,
                    name=template_name,
                    filename=filename,
                )
            except TemplateNotFound:
                unresolved.add(
                    UnresolvedTemplate(
                        source_relative,
                        None,
                        template_name,
                        "template not found",
                    )
                )
                continue
            except TemplateSyntaxError as exc:
                unresolved.add(
                    UnresolvedTemplate(
                        source_relative,
                        exc.lineno,
                        template_name,
                        f"template syntax error: {exc.message}",
                    )
                )
                continue
            except (OSError, UnicodeError) as exc:
                unresolved.add(
                    UnresolvedTemplate(
                        source_relative,
                        None,
                        template_name,
                        f"cannot read template: {exc}",
                    )
                )
                continue

            for referenced_name in meta.find_referenced_templates(parsed):
                if referenced_name is None:
                    unresolved.add(
                        UnresolvedTemplate(
                            source_relative,
                            None,
                            "<dynamic>",
                            "dynamic template reference",
                        )
                    )
                    continue
                try:
                    _, referenced_filename, _ = environment.loader.get_source(
                        environment, referenced_name
                    )
                except TemplateNotFound:
                    unresolved.add(
                        UnresolvedTemplate(
                            source_relative,
                            None,
                            referenced_name,
                            "template not found",
                        )
                    )
                    continue
                except (OSError, UnicodeError) as exc:
                    unresolved.add(
                        UnresolvedTemplate(
                            source_relative,
                            None,
                            referenced_name,
                            f"cannot read template: {exc}",
                        )
                    )
                    continue

                referenced_path = Path(referenced_filename).resolve()
                if not _is_relative_to(referenced_path, loader_root):
                    unresolved.add(
                        UnresolvedTemplate(
                            source_relative,
                            None,
                            referenced_name,
                            "template resolves outside loader root",
                        )
                    )
                    continue
                referenced_relative = _repository_relative(referenced_path, repository)
                if referenced_relative is None:
                    unresolved.add(
                        UnresolvedTemplate(
                            source_relative,
                            None,
                            referenced_name,
                            "template resolves outside repository",
                        )
                    )
                    continue
                fragments.add(referenced_relative)
                queue.append((referenced_name, referenced_relative))
    return fragments


def _existing_repository_files(
    paths: Iterable[Path],
    *,
    repository: Path,
    kind: str,
    unresolved: set[UnresolvedInclude],
    gitlink_roots: Sequence[Path] = (),
) -> set[Path]:
    files: set[Path] = set()
    for path in paths:
        absolute = _absolute(path, repository)
        gitlink_root = _containing_root(absolute, gitlink_roots)
        if gitlink_root is not None:
            relative_root = _repository_relative(gitlink_root, repository)
            if relative_root is not None:
                files.add(relative_root)
            continue
        relative = _repository_relative(absolute, repository)
        if relative is None:
            continue
        if not absolute.is_file():
            unresolved.add(
                UnresolvedInclude(relative, None, "", f"missing {kind} file")
            )
            continue
        files.add(relative)
    return files


def _absolute(path: Path, repository: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _repository_relative(path: Path, repository: Path) -> Path | None:
    try:
        return path.resolve().relative_to(repository)
    except ValueError:
        return None


def _extract_includes(content: str):
    for line_number, line in enumerate(content.splitlines(), start=1):
        directive = _INCLUDE_DIRECTIVE.match(line)
        if directive is None:
            continue
        raw_target = directive.group("target")
        literal = _LITERAL_INCLUDE.match(raw_target)
        if literal is None:
            yield line_number, None, raw_target
            continue
        quoted = literal.group("quoted")
        if quoted is not None:
            yield line_number, quoted, "quoted"
        else:
            yield line_number, literal.group("angled"), "angled"


def _resolve_include(
    target: str,
    *,
    quoted: bool,
    source_dir: Path,
    include_dirs: Sequence[Path],
) -> Path | None:
    search_dirs = (source_dir, *include_dirs) if quoted else include_dirs
    for directory in search_dirs:
        candidate = (directory / target).resolve()
        if candidate.is_file():
            return candidate
    return None


def _containing_root(path: Path, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return root
    return None


def _gitlink_roots(repository: Path) -> tuple[Path, ...]:
    config_path = repository / ".gitmodules"
    if not config_path.is_file():
        return ()
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_path, encoding="utf-8")
    except (OSError, configparser.Error):
        return ()
    roots: list[Path] = []
    for section in config.sections():
        value = config.get(section, "path", fallback="").strip()
        if not value:
            continue
        root = (repository / value).resolve()
        if _is_relative_to(root, repository):
            roots.append(root)
    return tuple(dict.fromkeys(roots))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _matches_any(target: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(target, pattern) for pattern in patterns)


def _looks_repository_local(
    target: str,
    include_dirs: Sequence[Path],
    *,
    repository: Path,
) -> bool:
    parts = Path(target).parts
    return bool(parts) and any(
        _is_relative_to(directory, repository) and (directory / parts[0]).exists()
        for directory in include_dirs
    )
