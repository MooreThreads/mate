"""Static discovery of a JITSpec dependency declaration from a factory."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Mapping

from mate._jit_dependencies import (
    JitDependencySpec,
    ResolvedJitDependencies,
    resolve_jit_dependencies,
)
from .python_ast import (
    PythonDependencyResult,
    PythonDslDependencyAnalyzer,
    PythonSymbolRef,
)


class JitSpecAnalysisError(ValueError):
    """Raised when a JITSpec factory has no static dependency declaration."""


class JitSpecProbeState(str, Enum):
    """Classification of a Python symbol at the JIT provider boundary."""

    NOT_JIT = "not-jit"
    RESOLVED_JIT = "resolved-jit"
    UNRESOLVED_JIT = "unresolved-jit"


@dataclass(frozen=True, slots=True)
class JitSpecDependencyResult:
    reference: PythonSymbolRef
    declaration: JitDependencySpec
    python: PythonDependencyResult
    resolved: ResolvedJitDependencies


@dataclass(frozen=True, slots=True)
class JitSpecProbeResult:
    state: JitSpecProbeState
    result: JitSpecDependencyResult | None = None
    error: JitSpecAnalysisError | None = None

    @property
    def claimed(self) -> bool:
        return self.state is not JitSpecProbeState.NOT_JIT


class JitSpecDependencyAnalyzer:
    """Resolve JIT dependencies without importing or calling the spec factory."""

    def __init__(self, repository: Path) -> None:
        self.repository = repository.resolve()

    def recognizes(self, reference: PythonSymbolRef | str) -> bool:
        """Return whether a symbol reaches ``gen_jit_spec`` without evaluating it."""
        ref = (
            reference
            if isinstance(reference, PythonSymbolRef)
            else PythonSymbolRef.parse(reference)
        )
        module_path = self.repository.joinpath(*ref.module.split(".")).with_suffix(
            ".py"
        )
        try:
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=module_path.as_posix())
        except (OSError, UnicodeError, SyntaxError, ValueError):
            return False
        factory = _top_level_function(tree, ref.symbol)
        if factory is None:
            return False
        try:
            self._find_gen_jit_spec_call(tree, factory, ref)
        except JitSpecAnalysisError:
            return False
        return True

    def analyze(self, reference: PythonSymbolRef | str) -> JitSpecDependencyResult:
        ref = (
            reference
            if isinstance(reference, PythonSymbolRef)
            else PythonSymbolRef.parse(reference)
        )
        module_path = self.repository.joinpath(*ref.module.split(".")).with_suffix(
            ".py"
        )
        try:
            relative_module = module_path.resolve().relative_to(self.repository)
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_module.as_posix())
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            raise JitSpecAnalysisError(
                f"cannot parse JITSpec reference {ref}: {exc}"
            ) from exc

        factory = next(
            (
                statement
                for statement in tree.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name == ref.symbol
            ),
            None,
        )
        if factory is None:
            raise JitSpecAnalysisError(f"JITSpec factory not found: {ref}")
        spec_call, spec_factory = self._find_gen_jit_spec_call(tree, factory, ref)
        assignments = {
            **_assignments(tree.body),
            **_assignments(spec_factory.body),
        }
        path_evaluator = _StaticPathEvaluator(self.repository, ref)
        declaration = self._parse_actual_inputs(
            spec_call,
            assignments=assignments,
            ref=ref,
            module=ref.module,
            tree=tree,
            spec_factory=spec_factory,
            path_evaluator=path_evaluator,
        )
        declaration = replace(
            declaration,
            generator_files=tuple(
                dict.fromkeys((*declaration.generator_files, relative_module))
            ),
        )
        python = PythonDslDependencyAnalyzer(self.repository).analyze((ref,))
        resolved = resolve_jit_dependencies(declaration, repository=self.repository)
        return JitSpecDependencyResult(ref, declaration, python, resolved)

    @staticmethod
    def _find_gen_jit_spec_call(
        tree: ast.Module,
        factory: ast.FunctionDef | ast.AsyncFunctionDef,
        ref: PythonSymbolRef,
    ) -> tuple[ast.Call, ast.FunctionDef | ast.AsyncFunctionDef]:
        functions = {
            statement.name: statement
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        queue = [factory]
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current.name in visited:
                continue
            visited.add(current.name)
            for node in ast.walk(current):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name == "gen_jit_spec":
                    return node, current
                if name in functions and name not in visited:
                    queue.append(functions[name])
        raise JitSpecAnalysisError(f"{ref}: no reachable gen_jit_spec call found")

    def _parse_actual_inputs(
        self,
        call: ast.Call,
        *,
        assignments: dict[str, ast.expr],
        ref: PythonSymbolRef,
        module: str,
        tree: ast.Module,
        spec_factory: ast.FunctionDef | ast.AsyncFunctionDef,
        path_evaluator: _StaticPathEvaluator,
    ) -> JitDependencySpec:
        sources_node = _keyword_value(call, "sources")
        if sources_node is None and len(call.args) >= 2:
            sources_node = call.args[1]
        if sources_node is None:
            raise JitSpecAnalysisError(f"{ref}: gen_jit_spec has no sources")

        generated_node = _keyword_value(call, "generated_sources")
        generated_keys = (
            {ast.dump(key) for key in generated_node.keys if key is not None}
            if isinstance(generated_node, ast.Dict)
            else set()
        )
        source_resolver = _StaticSourceSequenceResolver(
            self.repository,
            ref,
            path_evaluator,
        )
        source_items = source_resolver.items(
            sources_node,
            module=module,
            tree=tree,
            function=spec_factory,
            assignments=assignments,
        )
        sources = tuple(
            dict.fromkeys(
                path_evaluator.resolve(
                    candidate.node,
                    module=candidate.module,
                    tree=candidate.tree,
                    assignments=candidate.assignments,
                )
                for item in source_items
                if ast.dump(item.node) not in generated_keys
                for candidate in source_resolver.path_items(item)
            )
        )

        include_node = _keyword_value(call, "extra_include_paths")
        include_dirs = (
            self._path_sequence(
                include_node,
                assignments,
                ref,
                module=module,
                tree=tree,
                path_evaluator=path_evaluator,
            )
            if include_node is not None
            else ()
        )
        templates = self._generated_template_paths(
            generated_node,
            assignments=assignments,
            ref=ref,
            module=module,
            tree=tree,
            path_evaluator=path_evaluator,
        ) or self._discover_template_paths(tree, ref)
        return JitDependencySpec(
            sources=sources,
            templates=templates,
            include_dirs=include_dirs,
        )

    def _generated_template_paths(
        self,
        generated_sources: ast.expr | None,
        *,
        assignments: dict[str, ast.expr],
        ref: PythonSymbolRef,
        module: str,
        tree: ast.Module,
        path_evaluator: _StaticPathEvaluator,
    ) -> tuple[Path, ...]:
        if isinstance(generated_sources, ast.Name):
            generated_sources = assignments.get(generated_sources.id)
        if not isinstance(generated_sources, ast.Dict):
            return ()
        templates: list[Path] = []
        for value in generated_sources.values:
            if (
                not isinstance(value, ast.Call)
                or _call_name(value.func) != "JitSourceTemplate"
            ):
                continue
            template = _keyword_value(value, "template")
            if template is None and value.args:
                template = value.args[0]
            if template is None:
                raise JitSpecAnalysisError(
                    f"{ref}: JitSourceTemplate has no template path"
                )
            templates.append(
                path_evaluator.resolve(
                    template,
                    module=module,
                    tree=tree,
                    assignments=assignments,
                )
            )
        return tuple(dict.fromkeys(templates))

    def _path_sequence(
        self,
        node: ast.expr,
        assignments: dict[str, ast.expr],
        ref: PythonSymbolRef,
        *,
        module: str,
        tree: ast.Module,
        path_evaluator: _StaticPathEvaluator,
    ) -> tuple[Path, ...]:
        if isinstance(node, ast.Call) and _call_name(node.func) in {"list", "tuple"}:
            if len(node.args) != 1:
                raise JitSpecAnalysisError(f"{ref}: invalid path sequence")
            node = node.args[0]
        elif isinstance(node, ast.Call):
            node, assignments, module, tree = self._path_factory_return(
                node,
                module=module,
                tree=tree,
                ref=ref,
            )
        return tuple(
            path_evaluator.resolve(
                item,
                module=module,
                tree=tree,
                assignments=assignments,
            )
            for item in self._sequence_nodes(node, assignments, ref)
        )

    @staticmethod
    def _sequence_nodes(
        node: ast.expr,
        assignments: dict[str, ast.expr],
        ref: PythonSymbolRef,
    ) -> tuple[ast.expr, ...]:
        if isinstance(node, ast.Name):
            variable_name = node.id
            try:
                node = assignments[variable_name]
            except KeyError as exc:
                raise JitSpecAnalysisError(
                    f"{ref}: path variable {variable_name!r} not found"
                ) from exc
        if not isinstance(node, (ast.List, ast.Tuple)):
            raise JitSpecAnalysisError(f"{ref}: paths must be a static sequence")
        return tuple(node.elts)

    def _path_factory_return(
        self,
        call: ast.Call,
        *,
        module: str,
        tree: ast.Module,
        ref: PythonSymbolRef,
    ) -> tuple[ast.expr, dict[str, ast.expr], str, ast.Module]:
        name = _call_name(call.func)
        if name is None or call.args or call.keywords:
            raise JitSpecAnalysisError(
                f"{ref}: include path helper must be a zero-argument function"
            )
        target_module = module
        target_name = name
        function = _top_level_function(tree, target_name)
        if function is None:
            imported = _imported_symbol(tree, name)
            if imported is None:
                raise JitSpecAnalysisError(
                    f"{ref}: include path helper {name!r} not found"
                )
            target_module = _resolve_import_module(module, imported[0])
            target_name = imported[1]
            tree = self._read_module_tree(target_module, ref)
            function = _top_level_function(tree, target_name)
        if function is None:
            raise JitSpecAnalysisError(
                f"{ref}: include path helper {target_module}:{target_name} not found"
            )
        returned = next(
            (
                statement.value
                for statement in function.body
                if isinstance(statement, ast.Return) and statement.value is not None
            ),
            None,
        )
        if returned is None:
            raise JitSpecAnalysisError(
                f"{ref}: include path helper {target_module}:{target_name} has no static return"
            )
        return returned, _assignments(tree.body), target_module, tree

    def _read_module_tree(self, module: str, ref: PythonSymbolRef) -> ast.Module:
        relative = Path(*module.split(".")).with_suffix(".py")
        path = self.repository / relative
        try:
            return ast.parse(
                path.read_text(encoding="utf-8"), filename=relative.as_posix()
            )
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise JitSpecAnalysisError(
                f"{ref}: cannot parse helper module {module}: {exc}"
            ) from exc

    def _discover_template_paths(
        self, tree: ast.Module, ref: PythonSymbolRef
    ) -> tuple[Path, ...]:
        templates: set[Path] = set()
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for call in calls:
            names = {
                argument.value
                for argument in call.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and Path(argument.value).suffix in {".j2", ".jinja", ".jinja2"}
            }
            if not names:
                continue
            search_dirs = self._template_search_dirs(
                call,
                module=ref.module,
                tree=tree,
                ref=ref,
            )
            for name in names:
                candidates = [
                    directory / name
                    for directory in search_dirs
                    if (directory / name).is_file()
                ]
                if not candidates:
                    candidates = [
                        path
                        for path in self.repository.rglob(name)
                        if path.is_file() and ".git" not in path.parts
                    ]
                candidates = list(dict.fromkeys(candidates))
                if len(candidates) != 1:
                    raise JitSpecAnalysisError(
                        f"{ref}: template {name!r} resolved to {len(candidates)} files"
                    )
                templates.add(candidates[0].relative_to(self.repository))
        return tuple(sorted(templates))

    def _template_search_dirs(
        self,
        call: ast.Call,
        *,
        module: str,
        tree: ast.Module,
        ref: PythonSymbolRef,
    ) -> tuple[Path, ...]:
        name = _call_name(call.func)
        if name is None:
            return ()
        function = _top_level_function(tree, name)
        if function is None:
            imported = _imported_symbol(tree, name)
            if imported is None:
                return ()
            module = _resolve_import_module(module, imported[0])
            tree = self._read_module_tree(module, ref)
            function = _top_level_function(tree, imported[1])
        if function is None:
            return ()
        directories: set[Path] = set()
        for value in _assignments(function.body).values():
            try:
                relative = _StaticPathEvaluator(self.repository, ref).resolve(
                    value,
                    module=module,
                    tree=tree,
                    assignments=_assignments(function.body),
                )
            except JitSpecAnalysisError:
                continue
            absolute = self.repository / relative
            if absolute.is_dir():
                directories.add(absolute)
        return tuple(sorted(directories))


@dataclass(frozen=True, slots=True)
class _StaticSourceItem:
    node: ast.expr
    module: str
    tree: ast.Module
    assignments: Mapping[str, ast.expr]


class _StaticSourceSequenceResolver:
    """Enumerate every statically visible member of a JIT source sequence."""

    def __init__(
        self,
        repository: Path,
        ref: PythonSymbolRef,
        path_evaluator: _StaticPathEvaluator,
    ) -> None:
        self.repository = repository
        self.ref = ref
        self.path_evaluator = path_evaluator

    def items(
        self,
        node: ast.expr,
        *,
        module: str,
        tree: ast.Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        assignments: Mapping[str, ast.expr],
    ) -> tuple[_StaticSourceItem, ...]:
        items = self._items(
            node,
            module=module,
            tree=tree,
            function=function,
            assignments=assignments,
            stack=frozenset(),
        )
        deduplicated: dict[tuple[str, str], _StaticSourceItem] = {}
        for item in items:
            deduplicated[(item.module, ast.dump(item.node))] = item
        return tuple(deduplicated.values())

    def path_items(self, item: _StaticSourceItem) -> tuple[_StaticSourceItem, ...]:
        """Expand direct source-path dispatch helpers into all return branches."""
        return self._path_items(
            item.node,
            module=item.module,
            tree=item.tree,
            function=_containing_top_level_function(item.tree, item.node),
            assignments=item.assignments,
            stack=frozenset(),
        )

    def _path_items(
        self,
        node: ast.expr,
        *,
        module: str,
        tree: ast.Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef | None,
        assignments: Mapping[str, ast.expr],
        stack: frozenset[tuple[str, str]],
    ) -> tuple[_StaticSourceItem, ...]:
        if isinstance(node, ast.IfExp):
            return self._path_items(
                node.body,
                module=module,
                tree=tree,
                function=function,
                assignments=assignments,
                stack=stack,
            ) + self._path_items(
                node.orelse,
                module=module,
                tree=tree,
                function=function,
                assignments=assignments,
                stack=stack,
            )
        if isinstance(node, ast.Name) and function is not None:
            values = _function_assignment_values(function, node.id)
            if values:
                return tuple(
                    item
                    for value in values
                    for item in self._path_items(
                        value,
                        module=module,
                        tree=tree,
                        function=function,
                        assignments=assignments,
                        stack=stack,
                    )
                )
        if isinstance(node, ast.Call):
            target = self._called_function(node, module=module, tree=tree)
            if target is not None:
                target_module, target_tree, target_function = target
                key = (target_module, target_function.name)
                if key in stack:
                    raise JitSpecAnalysisError(
                        f"{self.ref}: recursive static source-path helper "
                        f"{target_module}:{target_function.name}"
                    )
                local_assignments = {
                    **_assignments(target_tree.body),
                    **_assignments(target_function.body),
                    **self._call_bindings(target_function, node),
                }
                returns = _function_return_values(target_function)
                if not returns:
                    raise JitSpecAnalysisError(
                        f"{self.ref}: static source-path helper "
                        f"{target_module}:{target_function.name} has no return"
                    )
                return tuple(
                    item
                    for returned in returns
                    for item in self._path_items(
                        returned,
                        module=target_module,
                        tree=target_tree,
                        function=target_function,
                        assignments=local_assignments,
                        stack=stack | {key},
                    )
                )
        return (_StaticSourceItem(node, module, tree, assignments),)

    def _items(
        self,
        node: ast.expr,
        *,
        module: str,
        tree: ast.Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        assignments: Mapping[str, ast.expr],
        stack: frozenset[tuple[str, str, str]],
    ) -> tuple[_StaticSourceItem, ...]:
        if isinstance(node, ast.Name):
            key = (module, function.name, node.id)
            if key in stack:
                raise JitSpecAnalysisError(
                    f"{self.ref}: recursive static source sequence {node.id!r}"
                )
            next_stack = stack | {key}
            sequence_values = list(_function_assignment_values(function, node.id))
            if (
                not sequence_values
                and (assigned := assignments.get(node.id)) is not None
            ):
                sequence_values.append(assigned)
            sequence_items: list[_StaticSourceItem] = []
            for value in sequence_values:
                sequence_items.extend(
                    self._items(
                        value,
                        module=module,
                        tree=tree,
                        function=function,
                        assignments=assignments,
                        stack=next_stack,
                    )
                )
            for mutation, value in _function_sequence_mutations(function, node.id):
                if mutation == "append":
                    sequence_items.append(
                        _StaticSourceItem(value, module, tree, assignments)
                    )
                else:
                    sequence_items.extend(
                        self._items(
                            value,
                            module=module,
                            tree=tree,
                            function=function,
                            assignments=assignments,
                            stack=next_stack,
                        )
                    )
            if not sequence_items:
                raise JitSpecAnalysisError(
                    f"{self.ref}: source sequence {node.id!r} has no static values"
                )
            return tuple(sequence_items)

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            literal_items: list[_StaticSourceItem] = []
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    literal_items.extend(
                        self._items(
                            element.value,
                            module=module,
                            tree=tree,
                            function=function,
                            assignments=assignments,
                            stack=stack,
                        )
                    )
                else:
                    literal_items.append(
                        _StaticSourceItem(element, module, tree, assignments)
                    )
            return tuple(literal_items)

        if isinstance(node, ast.IfExp):
            return self._items(
                node.body,
                module=module,
                tree=tree,
                function=function,
                assignments=assignments,
                stack=stack,
            ) + self._items(
                node.orelse,
                module=module,
                tree=tree,
                function=function,
                assignments=assignments,
                stack=stack,
            )

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._items(
                node.left,
                module=module,
                tree=tree,
                function=function,
                assignments=assignments,
                stack=stack,
            ) + self._items(
                node.right,
                module=module,
                tree=tree,
                function=function,
                assignments=assignments,
                stack=stack,
            )

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"list", "tuple", "set"} and len(node.args) == 1:
                return self._items(
                    node.args[0],
                    module=module,
                    tree=tree,
                    function=function,
                    assignments=assignments,
                    stack=stack,
                )
            target = self._called_function(node, module=module, tree=tree)
            if target is not None:
                target_module, target_tree, target_function = target
                key = (target_module, target_function.name, "<return>")
                if key in stack:
                    raise JitSpecAnalysisError(
                        f"{self.ref}: recursive static source helper "
                        f"{target_module}:{target_function.name}"
                    )
                local_assignments = {
                    **_assignments(target_tree.body),
                    **_assignments(target_function.body),
                    **self._call_bindings(target_function, node),
                }
                returns = _function_return_values(target_function)
                if not returns:
                    raise JitSpecAnalysisError(
                        f"{self.ref}: static source helper "
                        f"{target_module}:{target_function.name} has no return"
                    )
                helper_items: list[_StaticSourceItem] = []
                for returned in returns:
                    helper_items.extend(
                        self._items(
                            returned,
                            module=target_module,
                            tree=target_tree,
                            function=target_function,
                            assignments=local_assignments,
                            stack=stack | {key},
                        )
                    )
                return tuple(helper_items)

        if isinstance(node, ast.Subscript):
            collection = node.value
            if isinstance(collection, ast.Name):
                collection = assignments.get(collection.id, collection)
            collection_values: tuple[ast.expr, ...] = ()
            if isinstance(collection, ast.Dict):
                collection_values = tuple(collection.values)
            elif isinstance(collection, (ast.List, ast.Tuple)):
                collection_values = tuple(collection.elts)
            if collection_values:
                lookup_items: list[_StaticSourceItem] = []
                for value in collection_values:
                    lookup_items.extend(
                        self._items(
                            value,
                            module=module,
                            tree=tree,
                            function=function,
                            assignments=assignments,
                            stack=stack,
                        )
                    )
                return tuple(lookup_items)

        raise JitSpecAnalysisError(
            f"{self.ref}: sources must be a statically enumerable sequence "
            f"at line {getattr(node, 'lineno', '?')}"
        )

    def _called_function(
        self,
        call: ast.Call,
        *,
        module: str,
        tree: ast.Module,
    ) -> (
        tuple[
            str,
            ast.Module,
            ast.FunctionDef | ast.AsyncFunctionDef,
        ]
        | None
    ):
        if isinstance(call.func, ast.Name):
            function = _top_level_function(tree, call.func.id)
            if function is not None:
                return module, tree, function
            imported = _imported_symbol(tree, call.func.id)
            if imported is not None:
                target_module = _resolve_import_module(module, imported[0])
                target_tree = self.path_evaluator._module_tree(target_module)
                function = _top_level_function(target_tree, imported[1])
                if function is not None:
                    return target_module, target_tree, function
        elif isinstance(call.func, ast.Attribute):
            target_module = _imported_module_for_expression(
                call.func.value,
                current_module=module,
                tree=tree,
                repository=self.repository,
                assignments=_assignments(tree.body),
            )
            if target_module is not None:
                target_tree = self.path_evaluator._module_tree(target_module)
                function = _top_level_function(target_tree, call.func.attr)
                if function is not None:
                    return target_module, target_tree, function
        return None

    def _call_bindings(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call,
    ) -> dict[str, ast.expr]:
        if function.args.vararg or function.args.kwarg:
            raise JitSpecAnalysisError(
                f"{self.ref}: variadic static source helper {function.name}"
            )
        positional = (*function.args.posonlyargs, *function.args.args)
        parameters = {parameter.arg for parameter in positional}
        parameters.update(parameter.arg for parameter in function.args.kwonlyargs)
        if len(call.args) > len(positional):
            raise JitSpecAnalysisError(
                f"{self.ref}: too many arguments for static source helper "
                f"{function.name}"
            )
        bindings = {
            parameter.arg: argument
            for parameter, argument in zip(positional, call.args)
        }
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg not in parameters:
                raise JitSpecAnalysisError(
                    f"{self.ref}: unsupported call to static source helper "
                    f"{function.name}"
                )
            bindings[keyword.arg] = keyword.value
        defaults = positional[len(positional) - len(function.args.defaults) :]
        for parameter, default in zip(defaults, function.args.defaults):
            bindings.setdefault(parameter.arg, default)
        for parameter, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults
        ):
            if default is not None:
                bindings.setdefault(parameter.arg, default)
        return bindings


class _StaticPathEvaluator:
    """Evaluate a restricted set of Path expressions without importing modules."""

    def __init__(self, repository: Path, ref: PythonSymbolRef) -> None:
        self.repository = repository.resolve()
        self.ref = ref
        self._trees: dict[str, ast.Module] = {}

    def resolve(
        self,
        node: ast.expr,
        *,
        module: str,
        tree: ast.Module,
        assignments: Mapping[str, ast.expr],
    ) -> Path:
        values = self.resolve_all(
            node,
            module=module,
            tree=tree,
            assignments=assignments,
        )
        return min(values, key=self._candidate_rank)

    def resolve_all(
        self,
        node: ast.expr,
        *,
        module: str,
        tree: ast.Module,
        assignments: Mapping[str, ast.expr],
    ) -> tuple[Path, ...]:
        """Resolve every statically enumerable value of a path expression."""
        values = self._evaluate_all(
            node,
            module=module,
            tree=tree,
            assignments=assignments,
            stack=frozenset(),
        )
        paths: list[Path] = []
        for value in values:
            path = Path(value)
            if path.is_absolute():
                try:
                    path = path.resolve().relative_to(self.repository)
                except ValueError:
                    path = path.resolve()
            paths.append(path)
        return tuple(dict.fromkeys(paths))

    def _evaluate_all(
        self,
        node: ast.expr,
        *,
        module: str,
        tree: ast.Module,
        assignments: Mapping[str, ast.expr],
        stack: frozenset[tuple[str, str]],
    ) -> tuple[Path | str, ...]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        if isinstance(node, ast.IfExp):
            return self._evaluate_all(
                node.body,
                module=module,
                tree=tree,
                assignments=assignments,
                stack=stack,
            ) + self._evaluate_all(
                node.orelse,
                module=module,
                tree=tree,
                assignments=assignments,
                stack=stack,
            )
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return (self._module_path(module),)
            value = assignments.get(node.id)
            if value is not None:
                return self._evaluate_all(
                    value,
                    module=module,
                    tree=tree,
                    assignments=assignments,
                    stack=stack,
                )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left_values = self._evaluate_all(
                node.left,
                module=module,
                tree=tree,
                assignments=assignments,
                stack=stack,
            )
            right_values = self._evaluate_all(
                node.right,
                module=module,
                tree=tree,
                assignments=assignments,
                stack=stack,
            )
            return tuple(
                Path(left) / right for left in left_values for right in right_values
            )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            return tuple(
                Path(value).parents[node.slice.value]
                for value in self._evaluate_all(
                    node.value.value,
                    module=module,
                    tree=tree,
                    assignments=assignments,
                    stack=stack,
                )
            )
        if isinstance(node, ast.Attribute):
            if node.attr == "parent":
                return tuple(
                    Path(value).parent
                    for value in self._evaluate_all(
                        node.value,
                        module=module,
                        tree=tree,
                        assignments=assignments,
                        stack=stack,
                    )
                )
            imported_module = _imported_module_for_expression(
                node.value,
                current_module=module,
                tree=tree,
                repository=self.repository,
                assignments=assignments,
            )
            if imported_module is not None:
                imported_tree = self._module_tree(imported_module)
                imported_assignments = _assignments(imported_tree.body)
                value = imported_assignments.get(node.attr)
                if value is not None:
                    return self._evaluate_all(
                        value,
                        module=imported_module,
                        tree=imported_tree,
                        assignments=imported_assignments,
                        stack=stack,
                    )
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"as_posix", "resolve"}
                and not node.args
                and not node.keywords
            ):
                return self._evaluate_all(
                    node.func.value,
                    module=module,
                    tree=tree,
                    assignments=assignments,
                    stack=stack,
                )
            if _call_name(node.func) == "Path" and len(node.args) == 1:
                return tuple(
                    Path(value)
                    for value in self._evaluate_all(
                        node.args[0],
                        module=module,
                        tree=tree,
                        assignments=assignments,
                        stack=stack,
                    )
                )
            function_name = _call_name(node.func)
            function = (
                _top_level_function(tree, function_name)
                if function_name is not None
                else None
            )
            if function is not None:
                return self._evaluate_function_all(
                    function,
                    node,
                    module=module,
                    tree=tree,
                    assignments=assignments,
                    stack=stack,
                )
        raise JitSpecAnalysisError(
            f"{self.ref}: unsupported static path expression at line "
            f"{getattr(node, 'lineno', '?')}"
        )

    def _evaluate_function_all(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call,
        *,
        module: str,
        tree: ast.Module,
        assignments: Mapping[str, ast.expr],
        stack: frozenset[tuple[str, str]],
    ) -> tuple[Path | str, ...]:
        key = (module, function.name)
        if key in stack:
            raise JitSpecAnalysisError(
                f"{self.ref}: recursive static path helper {module}:{function.name}"
            )
        parameters = (*function.args.posonlyargs, *function.args.args)
        if len(call.args) > len(parameters) or call.keywords:
            raise JitSpecAnalysisError(
                f"{self.ref}: unsupported call to static path helper {function.name}"
            )
        local_assignments = dict(assignments)
        local_assignments.update(_assignments(function.body))
        local_assignments.update(
            {
                parameter.arg: argument
                for parameter, argument in zip(parameters, call.args)
            }
        )
        returns = _function_return_values(function)
        if not returns:
            raise JitSpecAnalysisError(
                f"{self.ref}: static path helper {module}:{function.name} has no return"
            )
        values = tuple(
            value
            for returned in returns
            for value in self._evaluate_all(
                returned,
                module=module,
                tree=tree,
                assignments=local_assignments,
                stack=stack | {key},
            )
        )
        return tuple(dict.fromkeys(values))

    def _candidate_rank(self, value: Path | str) -> tuple[int, int]:
        path = Path(value)
        absolute = path if path.is_absolute() else self.repository / path
        try:
            absolute.resolve().relative_to(self.repository)
            local = True
        except ValueError:
            local = False
        return (
            0 if local and absolute.exists() else 1 if local else 2,
            len(path.parts),
        )

    def _module_tree(self, module: str) -> ast.Module:
        if module not in self._trees:
            path = self._module_path(module)
            try:
                self._trees[module] = ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=path.relative_to(self.repository).as_posix(),
                )
            except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
                raise JitSpecAnalysisError(
                    f"{self.ref}: cannot parse path module {module}: {exc}"
                ) from exc
        return self._trees[module]

    def _module_path(self, module: str) -> Path:
        module_path = self.repository.joinpath(*module.split(".")).with_suffix(".py")
        package_path = self.repository.joinpath(*module.split("."), "__init__.py")
        if module_path.is_file():
            return module_path.resolve()
        if package_path.is_file():
            return package_path.resolve()
        raise JitSpecAnalysisError(f"{self.ref}: local module not found: {module}")


class JitSpecDependencyProvider:
    """Classify Python references and retain unresolved JIT boundaries."""

    name = "jit"

    def __init__(self, repository: Path) -> None:
        self.analyzer = JitSpecDependencyAnalyzer(repository)
        self._probes: dict[PythonSymbolRef, JitSpecProbeResult] = {}

    def provider_for(self, reference: PythonSymbolRef) -> str | None:
        return self.name if self.probe(reference).claimed else None

    def probe(self, reference: PythonSymbolRef) -> JitSpecProbeResult:
        if reference not in self._probes:
            if not self.analyzer.recognizes(reference):
                self._probes[reference] = JitSpecProbeResult(JitSpecProbeState.NOT_JIT)
                return self._probes[reference]
            try:
                result = self.analyzer.analyze(reference)
            except JitSpecAnalysisError as exc:
                self._probes[reference] = JitSpecProbeResult(
                    JitSpecProbeState.UNRESOLVED_JIT,
                    error=exc,
                )
            else:
                self._probes[reference] = JitSpecProbeResult(
                    JitSpecProbeState.RESOLVED_JIT,
                    result=result,
                )
        return self._probes[reference]

    def resolve(self, reference: PythonSymbolRef) -> JitSpecDependencyResult:
        probe = self.probe(reference)
        if probe.error is not None:
            raise probe.error
        if probe.result is None:
            raise JitSpecAnalysisError(
                f"reference does not expose static JIT dependencies: {reference}"
            )
        return probe.result

    @property
    def resolved_references(self) -> Mapping[PythonSymbolRef, JitSpecDependencyResult]:
        return {
            reference: probe.result
            for reference, probe in self._probes.items()
            if probe.result is not None
        }


class _FunctionBodyVisitor(ast.NodeVisitor):
    """Visit one function body while excluding nested definitions."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


class _FunctionAssignmentVisitor(_FunctionBodyVisitor):
    def __init__(self, name: str) -> None:
        self.name = name
        self.values: list[ast.expr] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(
            isinstance(target, ast.Name) and target.id == self.name
            for target in node.targets
        ):
            self.values.append(node.value)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and node.target.id == self.name
            and node.value is not None
        ):
            self.values.append(node.value)
        if node.value is not None:
            self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id == self.name:
            self.values.append(node.value)
        self.generic_visit(node.value)


class _FunctionSequenceMutationVisitor(_FunctionBodyVisitor):
    def __init__(self, name: str) -> None:
        self.name = name
        self.values: list[tuple[str, ast.expr]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.name
            and node.func.attr in {"append", "extend"}
            and len(node.args) == 1
            and not node.keywords
        ):
            self.values.append((node.func.attr, node.args[0]))
        self.generic_visit(node)


class _FunctionReturnVisitor(_FunctionBodyVisitor):
    def __init__(self) -> None:
        self.values: list[ast.expr] = []

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.values.append(node.value)


def _visit_function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    visitor: _FunctionBodyVisitor,
) -> None:
    for statement in function.body:
        visitor.visit(statement)


def _function_assignment_values(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> tuple[ast.expr, ...]:
    visitor = _FunctionAssignmentVisitor(name)
    _visit_function_body(function, visitor)
    return tuple(visitor.values)


def _function_sequence_mutations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> tuple[tuple[str, ast.expr], ...]:
    visitor = _FunctionSequenceMutationVisitor(name)
    _visit_function_body(function, visitor)
    return tuple(visitor.values)


def _function_return_values(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.expr, ...]:
    visitor = _FunctionReturnVisitor()
    _visit_function_body(function, visitor)
    return tuple(visitor.values)


def _assignments(statements: list[ast.stmt]) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for statement in statements:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            assignments[statement.target.id] = statement.value
    return assignments


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _top_level_function(
    tree: ast.Module, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == name
        ),
        None,
    )


def _containing_top_level_function(
    tree: ast.Module, node: ast.AST
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(candidate is node for candidate in ast.walk(statement))
        ),
        None,
    )


def _imported_symbol(
    tree: ast.Module, local_name: str
) -> tuple[ast.ImportFrom, str] | None:
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        for alias in statement.names:
            if (alias.asname or alias.name) == local_name:
                return statement, alias.name
    return None


def _imported_module_for_expression(
    node: ast.expr,
    *,
    current_module: str,
    tree: ast.Module,
    repository: Path,
    assignments: Mapping[str, ast.expr],
) -> str | None:
    parts = _attribute_parts(node)
    if not parts:
        return None
    local_name, *attributes = parts
    assigned = assignments.get(local_name)
    if assigned is not None:
        assigned_parts = _attribute_parts(assigned)
        if assigned_parts:
            assigned_module = _imported_module_for_expression(
                assigned,
                current_module=current_module,
                tree=tree,
                repository=repository,
                assignments=assignments,
            )
            if assigned_module is not None:
                return _longest_local_module(
                    (*assigned_module.split("."), *attributes), repository
                )
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                if bound_name != local_name:
                    continue
                base = alias.name if alias.asname else bound_name
                return _longest_local_module(
                    (*base.split("."), *attributes), repository
                )
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if (alias.asname or alias.name) != local_name:
                    continue
                base = _resolve_import_module(current_module, statement)
                return _longest_local_module(
                    (*base.split("."), alias.name, *attributes), repository
                )
    return None


def _attribute_parts(node: ast.expr) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ()
    parts.append(node.id)
    return tuple(reversed(parts))


def _longest_local_module(parts: tuple[str, ...], repository: Path) -> str | None:
    for length in range(len(parts), 0, -1):
        module = ".".join(parts[:length])
        relative = repository.joinpath(*parts[:length])
        if (
            relative.with_suffix(".py").is_file()
            or (relative / "__init__.py").is_file()
        ):
            return module
    return None


def _resolve_import_module(current_module: str, statement: ast.ImportFrom) -> str:
    if statement.level == 0:
        return statement.module or ""
    package = current_module.split(".")[:-1]
    keep = len(package) - (statement.level - 1)
    base = package[:keep]
    if statement.module:
        base.extend(statement.module.split("."))
    return ".".join(base)


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
