"""Static symbol-level dependency analysis for Python kernel DSLs."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


class PythonAnalysisError(ValueError):
    """Raised when a Python DSL dependency root is invalid."""


@dataclass(frozen=True, order=True, slots=True)
class PythonSymbolRef:
    module: str
    symbol: str

    @classmethod
    def parse(cls, value: str) -> PythonSymbolRef:
        module, separator, symbol = value.partition(":")
        if not separator or not module or not symbol:
            raise PythonAnalysisError(
                f"Python dependency root must be module:symbol, got {value!r}"
            )
        return cls(module, symbol)

    def __str__(self) -> str:
        return f"{self.module}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class PythonDependencyEdge:
    source: str | None
    target: str
    line: int | None
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class ProviderBoundary:
    source: str
    line: int | None
    target: str
    provider: str


@dataclass(frozen=True, slots=True)
class PythonUnresolvedReference:
    source: str
    line: int | None
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class PythonDependencyResult:
    roots: tuple[PythonSymbolRef, ...]
    files: tuple[str, ...]
    edges: tuple[PythonDependencyEdge, ...]
    boundaries: tuple[ProviderBoundary, ...]
    external_modules: tuple[str, ...]
    unresolved: tuple[PythonUnresolvedReference, ...]


@dataclass(frozen=True, slots=True)
class _Binding:
    kind: str
    node: ast.AST
    module: str | None = None
    symbol: str | None = None
    optional: bool = False


@dataclass(slots=True)
class _ModuleInfo:
    module: str
    path: Path
    source: str
    lines: tuple[str, ...]
    tree: ast.Module
    bindings: dict[str, tuple[_Binding, ...]]
    star_imports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingRef:
    ref: PythonSymbolRef
    source: str | None
    line: int | None
    kind: str
    detail: str
    optional: bool = False


class PythonDslDependencyAnalyzer:
    """Resolve repository-local dependencies without importing analyzed modules."""

    def __init__(
        self,
        repository: Path,
        *,
        provider_boundaries: Mapping[str, str] | None = None,
        reference_boundary: Callable[[PythonSymbolRef], str | None] | None = None,
        module_search_paths: Sequence[Path] = (),
    ) -> None:
        self.repository = repository.resolve()
        self.provider_boundaries = dict(provider_boundaries or {})
        self.reference_boundary = reference_boundary
        search_paths = [self.repository]
        for path in module_search_paths:
            resolved = (
                path.resolve()
                if path.is_absolute()
                else (self.repository / path).resolve()
            )
            try:
                resolved.relative_to(self.repository)
            except ValueError as exc:
                raise PythonAnalysisError(
                    f"module search path escapes repository: {path}"
                ) from exc
            if resolved not in search_paths:
                search_paths.append(resolved)
        self.module_search_paths = tuple(search_paths)
        self._module_paths: dict[str, Path | None] = {}
        self._modules: dict[str, _ModuleInfo] = {}
        self._queue: deque[_PendingRef] = deque()
        self._visited: set[tuple[PythonSymbolRef, bool]] = set()
        self._files: set[str] = set()
        self._edges: set[PythonDependencyEdge] = set()
        self._boundaries: set[ProviderBoundary] = set()
        self._external_modules: set[str] = set()
        self._unresolved: set[PythonUnresolvedReference] = set()

    def analyze(
        self,
        roots: Sequence[PythonSymbolRef | str] = (),
        *,
        modules: Sequence[str] = (),
    ) -> PythonDependencyResult:
        self._queue.clear()
        self._visited.clear()
        self._files.clear()
        self._edges.clear()
        self._boundaries.clear()
        self._external_modules.clear()
        self._unresolved.clear()
        normalized_roots = tuple(
            root if isinstance(root, PythonSymbolRef) else PythonSymbolRef.parse(root)
            for root in roots
        )
        if not normalized_roots and not modules:
            raise PythonAnalysisError(
                "at least one Python symbol or module root is required"
            )

        for root in normalized_roots:
            self._queue.append(_PendingRef(root, None, None, "root", str(root)))
        expanded_roots = list(normalized_roots)
        for module in modules:
            pending = _PendingRef(
                PythonSymbolRef(module, "<module>"),
                None,
                None,
                "module-root",
                module,
            )
            expanded_roots.extend(self._expand_module_root(module, pending))

        while self._queue:
            pending = self._queue.popleft()
            visit_key = (pending.ref, pending.optional)
            if visit_key in self._visited:
                continue
            self._visited.add(visit_key)
            self._analyze_symbol(pending)

        return PythonDependencyResult(
            roots=tuple(expanded_roots),
            files=tuple(sorted(self._files)),
            edges=tuple(
                sorted(
                    self._edges,
                    key=lambda edge: (
                        edge.target,
                        edge.source or "",
                        edge.line or 0,
                        edge.kind,
                        edge.detail,
                    ),
                )
            ),
            boundaries=tuple(
                sorted(
                    self._boundaries,
                    key=lambda item: (
                        item.target,
                        item.source,
                        item.line or 0,
                        item.provider,
                    ),
                )
            ),
            external_modules=tuple(sorted(self._external_modules)),
            unresolved=tuple(
                sorted(
                    self._unresolved,
                    key=lambda item: (
                        item.source,
                        item.line or 0,
                        item.kind,
                        item.detail,
                    ),
                )
            ),
        )

    def _expand_module_root(
        self, module: str, pending: _PendingRef
    ) -> tuple[PythonSymbolRef, ...]:
        path = self._module_path(module)
        if path is None:
            self._record_missing_module(pending)
            return ()
        self._add_module_files(path, pending)
        info = self._module_info(module, path)
        if info is None:
            return ()
        roots = tuple(
            PythonSymbolRef(module, symbol) for symbol in sorted(info.bindings)
        )
        for root in roots:
            self._queue.append(
                _PendingRef(
                    root,
                    str(pending.ref),
                    pending.line,
                    "module-binding",
                    root.symbol,
                    pending.optional,
                )
            )
        return roots

    def _analyze_symbol(self, pending: _PendingRef) -> None:
        boundary = self._provider_for_ref(pending.ref)
        if boundary is not None:
            self._boundaries.add(
                ProviderBoundary(
                    pending.source or str(pending.ref),
                    pending.line,
                    str(pending.ref),
                    boundary,
                )
            )
            return

        path = self._module_path(pending.ref.module)
        if path is None:
            if self._namespace_directory(pending.ref.module) is not None:
                candidate = f"{pending.ref.module}.{pending.ref.symbol}"
                if self._module_exists(candidate):
                    self._add_module_only(
                        candidate,
                        source=str(pending.ref),
                        line=pending.line,
                        kind="namespace-submodule",
                        detail=pending.detail,
                        optional=pending.optional,
                    )
                    return
            self._record_missing_module(pending)
            return
        self._add_module_files(path, pending)
        info = self._module_info(pending.ref.module, path)
        if info is None:
            return

        bindings = info.bindings.get(pending.ref.symbol, ())
        if not bindings and path.name == "__init__.py":
            lazy_target = self._static_lazy_export(info, pending.ref.symbol)
            if lazy_target is not None:
                self._enqueue(
                    PythonSymbolRef(lazy_target, pending.ref.symbol),
                    source=str(pending.ref),
                    line=pending.line,
                    kind="package-lazy-export",
                    detail=pending.detail,
                    optional=pending.optional,
                )
                return
            candidate = f"{pending.ref.module}.{pending.ref.symbol}"
            candidate_path = self._module_path(candidate)
            if candidate_path is not None:
                self._add_module_only(
                    candidate,
                    source=str(pending.ref),
                    line=pending.line,
                    kind="package-submodule",
                    detail=pending.detail,
                )
                return
        if not bindings:
            kind = "star-import" if info.star_imports else "missing-symbol"
            self._unresolved.add(
                PythonUnresolvedReference(
                    str(pending.ref),
                    pending.line,
                    kind,
                    (
                        f"cannot resolve symbol; star imports: {info.star_imports}"
                        if info.star_imports
                        else "symbol has no static module-level binding"
                    ),
                )
            )
            return

        for binding in bindings:
            if binding.kind == "from-import":
                assert binding.module is not None and binding.symbol is not None
                self._enqueue(
                    PythonSymbolRef(binding.module, binding.symbol),
                    source=str(pending.ref),
                    line=getattr(binding.node, "lineno", None),
                    kind="imported-symbol",
                    detail=self._statement(info, binding.node),
                    optional=binding.optional,
                )
            elif binding.kind == "import":
                assert binding.module is not None
                self._add_module_only(
                    binding.module,
                    source=str(pending.ref),
                    line=getattr(binding.node, "lineno", None),
                    kind="imported-module",
                    detail=self._statement(info, binding.node),
                    optional=binding.optional,
                )
            else:
                self._analyze_node(info, pending.ref, binding.node)

    def _analyze_node(
        self, info: _ModuleInfo, ref: PythonSymbolRef, node: ast.AST
    ) -> None:
        if ref.symbol == "pytest_plugins":
            self._follow_pytest_plugins(info, ref, node)
        local_imports = self._local_import_bindings(info, ref, node)
        local_names = self._local_names(node) | set(local_imports)

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._check_dynamic_import(info, ref, child)
            if isinstance(child, ast.Attribute):
                self._follow_attribute(info, ref, child, local_imports=local_imports)
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            if child.id in local_imports:
                binding = local_imports[child.id]
                if binding.kind == "from-import":
                    assert binding.module is not None and binding.symbol is not None
                    self._enqueue(
                        PythonSymbolRef(binding.module, binding.symbol),
                        source=str(ref),
                        line=child.lineno,
                        kind="function-import",
                        detail=self._statement(info, binding.node),
                        optional=binding.optional,
                    )
                continue
            if child.id in local_names or child.id not in info.bindings:
                continue
            self._enqueue(
                PythonSymbolRef(info.module, child.id),
                source=str(ref),
                line=child.lineno,
                kind="global-reference",
                detail=child.id,
            )

    def _follow_pytest_plugins(
        self, info: _ModuleInfo, ref: PythonSymbolRef, node: ast.AST
    ) -> None:
        value = _assignment_value(node)
        try:
            declared = ast.literal_eval(value) if value is not None else None
        except (ValueError, TypeError):
            declared = None
        if isinstance(declared, str):
            plugins = (declared,)
        elif isinstance(declared, (tuple, list)) and all(
            isinstance(item, str) for item in declared
        ):
            plugins = tuple(declared)
        else:
            self._unresolved.add(
                PythonUnresolvedReference(
                    str(ref),
                    getattr(node, "lineno", None),
                    "dynamic-pytest-plugins",
                    self._statement(info, node),
                )
            )
            return
        for plugin in plugins:
            pending = _PendingRef(
                PythonSymbolRef(plugin, "<module>"),
                str(ref),
                getattr(node, "lineno", None),
                "pytest-plugin",
                plugin,
            )
            self._expand_module_root(plugin, pending)

    def _local_import_bindings(
        self, info: _ModuleInfo, ref: PythonSymbolRef, node: ast.AST
    ) -> dict[str, _Binding]:
        bindings: dict[str, _Binding] = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                optional = _inside_try(node, child)
                for alias in child.names:
                    local_name = alias.asname or alias.name.split(".", 1)[0]
                    target_module = alias.name if alias.asname else local_name
                    binding = _Binding(
                        "import", child, target_module, optional=optional
                    )
                    bindings[local_name] = binding
                    self._add_module_only(
                        target_module,
                        source=str(ref),
                        line=child.lineno,
                        kind="function-import",
                        detail=self._statement(info, child),
                        optional=optional,
                    )
            elif isinstance(child, ast.ImportFrom):
                optional = _inside_try(node, child)
                base = self._resolve_import_from(info, child)
                if base is None:
                    self._unresolved.add(
                        PythonUnresolvedReference(
                            str(ref),
                            child.lineno,
                            "relative-import",
                            self._statement(info, child),
                        )
                    )
                    continue
                for alias in child.names:
                    if alias.name == "*":
                        self._unresolved.add(
                            PythonUnresolvedReference(
                                str(ref),
                                child.lineno,
                                "star-import",
                                self._statement(info, child),
                            )
                        )
                        continue
                    local_name = alias.asname or alias.name
                    binding = _Binding("from-import", child, base, alias.name, optional)
                    bindings[local_name] = binding
                    self._enqueue(
                        PythonSymbolRef(base, alias.name),
                        source=str(ref),
                        line=child.lineno,
                        kind="function-import",
                        detail=self._statement(info, child),
                        optional=optional,
                    )
        return bindings

    def _follow_attribute(
        self,
        info: _ModuleInfo,
        ref: PythonSymbolRef,
        node: ast.Attribute,
        *,
        local_imports: Mapping[str, _Binding],
    ) -> None:
        flattened = _flatten_attribute(node)
        if flattened is None:
            return
        root_name, attributes = flattened
        bindings = (
            (local_imports[root_name],)
            if root_name in local_imports
            else info.bindings.get(root_name, ())
        )
        for binding in bindings:
            if binding.kind == "import" and binding.module is not None:
                self._follow_module_attribute(
                    binding.module,
                    attributes,
                    source=str(ref),
                    line=node.lineno,
                    detail=self._statement(info, node),
                    optional=binding.optional,
                )
            elif binding.kind == "from-import" and binding.module and binding.symbol:
                candidate_module = f"{binding.module}.{binding.symbol}"
                if self._module_exists(candidate_module):
                    self._follow_module_attribute(
                        candidate_module,
                        attributes,
                        source=str(ref),
                        line=node.lineno,
                        detail=self._statement(info, node),
                        optional=binding.optional,
                    )
                else:
                    self._enqueue(
                        PythonSymbolRef(binding.module, binding.symbol),
                        source=str(ref),
                        line=node.lineno,
                        kind="imported-attribute",
                        detail=self._statement(info, node),
                        optional=binding.optional,
                    )

    def _follow_module_attribute(
        self,
        imported_module: str,
        attributes: Sequence[str],
        *,
        source: str,
        line: int,
        detail: str,
        optional: bool = False,
    ) -> None:
        if (
            not self._module_exists(imported_module)
            and self._provider_for(imported_module) is None
        ):
            self._external_modules.add(imported_module)
            return
        parts = imported_module.split(".") + list(attributes)
        for index in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:index])
            if not self._module_exists(module) and self._provider_for(module) is None:
                continue
            remaining = parts[index:]
            if not remaining:
                self._add_module_only(
                    module,
                    source=source,
                    line=line,
                    kind="module-attribute",
                    detail=detail,
                    optional=optional,
                )
            else:
                self._enqueue(
                    PythonSymbolRef(module, remaining[0]),
                    source=source,
                    line=line,
                    kind="module-attribute",
                    detail=detail,
                    optional=optional,
                )
            return

    def _enqueue(
        self,
        ref: PythonSymbolRef,
        *,
        source: str,
        line: int | None,
        kind: str,
        detail: str,
        optional: bool = False,
    ) -> None:
        self._queue.append(_PendingRef(ref, source, line, kind, detail, optional))

    def _add_module_only(
        self,
        module: str,
        *,
        source: str,
        line: int | None,
        kind: str,
        detail: str,
        optional: bool = False,
    ) -> None:
        provider = self._provider_for(module)
        if provider is not None:
            self._boundaries.add(ProviderBoundary(source, line, module, provider))
            return
        path = self._module_path(module)
        if path is None:
            if self._namespace_directory(module) is not None:
                self._add_package_init_files(module, source, line, detail)
                return
            self._record_missing_module(
                _PendingRef(
                    PythonSymbolRef(module, "<module>"),
                    source,
                    line,
                    kind,
                    detail,
                    optional,
                )
            )
            return
        self._add_module_files(
            path,
            _PendingRef(
                PythonSymbolRef(module, "<module>"),
                source,
                line,
                kind,
                detail,
                optional,
            ),
        )

    def _add_module_files(self, path: Path, pending: _PendingRef) -> None:
        relative = path.relative_to(self.repository).as_posix()
        self._files.add(relative)
        self._edges.add(
            PythonDependencyEdge(
                pending.source,
                relative,
                pending.line,
                pending.kind,
                pending.detail,
            )
        )
        module = self._module_name_for_path(path)
        self._add_package_init_files(
            module, pending.source, pending.line, pending.detail
        )

    def _add_package_init_files(
        self,
        module: str,
        source: str | None,
        line: int | None,
        detail: str,
    ) -> None:
        parts = module.split(".")
        for index in range(1, len(parts)):
            package_path = self._module_path(".".join(parts[:index]))
            if package_path is None or package_path.name != "__init__.py":
                continue
            package_relative = package_path.relative_to(self.repository).as_posix()
            self._files.add(package_relative)
            self._edges.add(
                PythonDependencyEdge(
                    source,
                    package_relative,
                    line,
                    "package-init",
                    detail,
                )
            )

    def _module_info(self, module: str, path: Path) -> _ModuleInfo | None:
        if module in self._modules:
            return self._modules[module]
        relative = path.relative_to(self.repository).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            self._unresolved.add(
                PythonUnresolvedReference(
                    module,
                    getattr(exc, "lineno", None),
                    "parse-error",
                    str(exc),
                )
            )
            return None

        bindings: dict[str, list[_Binding]] = defaultdict(list)
        star_imports: list[str] = []
        info = _ModuleInfo(
            module, path, source, tuple(source.splitlines()), tree, {}, ()
        )
        for statement in _module_level_statements(tree.body):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bindings[statement.name].append(_Binding("definition", statement))
            elif isinstance(statement, ast.ClassDef):
                bindings[statement.name].append(_Binding("class", statement))
            elif isinstance(statement, ast.Import):
                optional = _inside_try(tree, statement)
                for alias in statement.names:
                    local_name = alias.asname or alias.name.split(".", 1)[0]
                    target_module = alias.name if alias.asname else local_name
                    bindings[local_name].append(
                        _Binding("import", statement, target_module, optional=optional)
                    )
            elif isinstance(statement, ast.ImportFrom):
                optional = _inside_try(tree, statement)
                base = self._resolve_import_from(info, statement)
                if base is None:
                    continue
                for alias in statement.names:
                    if alias.name == "*":
                        star_imports.append(base)
                        continue
                    local_name = alias.asname or alias.name
                    bindings[local_name].append(
                        _Binding("from-import", statement, base, alias.name, optional)
                    )
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                for name in _assigned_names(statement):
                    bindings[name].append(_Binding("assignment", statement))
        info.bindings = {name: tuple(values) for name, values in bindings.items()}
        info.star_imports = tuple(star_imports)
        self._modules[module] = info
        return info

    def _module_path(self, module: str) -> Path | None:
        if module in self._module_paths:
            return self._module_paths[module]
        relative = Path(*module.split("."))
        result = None
        for search_path in self.module_search_paths:
            module_file = (search_path / relative).with_suffix(".py")
            package_file = search_path / relative / "__init__.py"
            if module_file.is_file():
                result = module_file.resolve()
                break
            if package_file.is_file():
                result = package_file.resolve()
                break
        self._module_paths[module] = result
        return result

    def _module_exists(self, module: str) -> bool:
        return (
            self._module_path(module) is not None
            or self._namespace_directory(module) is not None
        )

    def _namespace_directory(self, module: str) -> Path | None:
        relative = Path(*module.split("."))
        for search_path in self.module_search_paths:
            candidate = search_path / relative
            if candidate.is_dir():
                return candidate.resolve()
        return None

    def _static_lazy_export(self, info: _ModuleInfo, symbol: str) -> str | None:
        package = info.module
        for bindings in info.bindings.values():
            for binding in bindings:
                if binding.kind != "assignment":
                    continue
                value = _assignment_value(binding.node)
                if not isinstance(value, ast.Dict):
                    continue
                for key, target in zip(value.keys, value.values):
                    if not (
                        isinstance(key, ast.Constant)
                        and key.value == symbol
                        and isinstance(target, ast.Constant)
                        and isinstance(target.value, str)
                    ):
                        continue
                    module = target.value
                    if module.startswith("."):
                        module = package + module
                    elif "." not in module:
                        module = f"{package}.{module}"
                    if self._module_path(module) is not None:
                        return module
        return None

    def _provider_for(self, module: str) -> str | None:
        matches = [
            (prefix, provider)
            for prefix, provider in self.provider_boundaries.items()
            if module == prefix or module.startswith(f"{prefix}.")
        ]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else None

    def _provider_for_ref(self, ref: PythonSymbolRef) -> str | None:
        if self.reference_boundary is not None:
            provider = self.reference_boundary(ref)
            if provider is not None:
                return provider
        return self._provider_for(ref.module)

    def _record_missing_module(self, pending: _PendingRef) -> None:
        if self._looks_repository_local(pending.ref.module):
            for candidate in self._missing_module_candidates(pending.ref.module):
                self._files.add(candidate)
                self._edges.add(
                    PythonDependencyEdge(
                        pending.source,
                        candidate,
                        pending.line,
                        "missing-module-candidate",
                        pending.detail,
                    )
                )
        else:
            self._external_modules.add(pending.ref.module)

    def _missing_module_candidates(self, module: str) -> tuple[str, ...]:
        relative = Path(*module.split("."))
        candidates: list[str] = []
        for search_path in self.module_search_paths:
            for path in (
                (search_path / relative).with_suffix(".py"),
                search_path / relative / "__init__.py",
            ):
                try:
                    candidate = path.relative_to(self.repository).as_posix()
                except ValueError:
                    continue
                if candidate not in candidates:
                    candidates.append(candidate)
        return tuple(candidates)

    def _looks_repository_local(self, module: str) -> bool:
        top_level = module.split(".", 1)[0]
        return any(
            (search_path / top_level).exists()
            or (search_path / f"{top_level}.py").exists()
            for search_path in self.module_search_paths
        )

    def _check_dynamic_import(
        self, info: _ModuleInfo, ref: PythonSymbolRef, node: ast.Call
    ) -> None:
        # getattr selects an attribute on an existing object; it does not load a file.
        # Any import mechanism that constructs that object is tracked separately.
        function = node.func
        dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id == "importlib"
        )
        if dynamic_import and node.args and isinstance(node.args[0], ast.Constant):
            target = node.args[0].value
            if isinstance(target, str):
                self._add_module_only(
                    target,
                    source=str(ref),
                    line=node.lineno,
                    kind="literal-dynamic-import",
                    detail=self._statement(info, node),
                )
                return
        if dynamic_import:
            self._unresolved.add(
                PythonUnresolvedReference(
                    str(ref),
                    node.lineno,
                    "dynamic-import",
                    self._statement(info, node),
                )
            )

    @staticmethod
    def _resolve_import_from(info: _ModuleInfo, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        package = (
            info.module
            if info.path.name == "__init__.py"
            else info.module.rpartition(".")[0]
        )
        if not package:
            return None
        parts = package.split(".")
        ascend = node.level - 1
        if ascend >= len(parts):
            return None
        base = parts[: len(parts) - ascend]
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(base)

    def _module_name_for_path(self, path: Path) -> str:
        parts = list(path.relative_to(self.repository).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def _statement(info: _ModuleInfo, node: ast.AST) -> str:
        line_number = getattr(node, "lineno", None)
        if line_number is None or line_number > len(info.lines):
            return type(node).__name__
        return info.lines[line_number - 1].strip() or type(node).__name__

    @staticmethod
    def _local_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            names.update(arg.arg for arg in arguments.posonlyargs)
            names.update(arg.arg for arg in arguments.args)
            names.update(arg.arg for arg in arguments.kwonlyargs)
            if arguments.vararg:
                names.add(arguments.vararg.arg)
            if arguments.kwarg:
                names.add(arguments.kwarg.arg)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(
                child.ctx, (ast.Store, ast.Del)
            ):
                names.add(child.id)
        return names


def _module_level_statements(statements: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    for statement in statements:
        yield statement
        nested: list[list[ast.stmt]] = []
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            nested.extend((statement.body, statement.orelse))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            nested.append(statement.body)
        elif isinstance(statement, ast.Try):
            nested.extend((statement.body, statement.orelse, statement.finalbody))
            nested.extend(handler.body for handler in statement.handlers)
        elif isinstance(statement, ast.Match):
            nested.extend(case.body for case in statement.cases)
        for body in nested:
            yield from _module_level_statements(body)


def _inside_try(root: ast.AST, target: ast.AST) -> bool:
    def contains(node: ast.AST) -> bool:
        return node is target or any(
            contains(child) for child in ast.iter_child_nodes(node)
        )

    return any(isinstance(node, ast.Try) and contains(node) for node in ast.walk(root))


def _assigned_names(node: ast.AST) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    elif isinstance(node, ast.NamedExpr):
        targets.append(node.target)
    return {
        child.id
        for target in targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name)
    }


def _assignment_value(node: ast.AST) -> ast.expr | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    if isinstance(node, ast.NamedExpr):
        return node.value
    return None


def _flatten_attribute(node: ast.Attribute) -> tuple[str, tuple[str, ...]] | None:
    attributes: list[str] = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        attributes.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    attributes.reverse()
    return value.id, tuple(attributes)
