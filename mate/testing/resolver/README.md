# Incremental Test Resolver

The incremental test resolver selects MATE operator tests from repository changes. It builds a static dependency closure for each registered operator, intersects that closure with a Git diff, and returns the complete test set owned by every affected operator.

The resolver is exposed as a repository CLI and is used by MATE CI and pre-commit. Static commands do not import or execute the operator modules that they analyze.

## Design Principles

### Operator-level selection

An operator is the unit of ownership and execution. A catalog entry may contain several implementations or backends, but the resolver does not select tests at backend granularity. If any dependency of an operator changes, all tests registered for that operator are selected.

### Minimal catalog declarations

The catalog records only:

1. A canonical operator name.
2. One or more top-level public interface modules.
3. One or more pytest files owned by the operator.

Backend names, JIT factories, templates, headers, DSL implementation files, test helpers, and `conftest.py` files are derived by the resolver. They must not be duplicated in the catalog.

```python
TEST_CATALOG.register(
    "mate.ops.attention.fa",
    interfaces=("mate/mha_interface.py",),
    tests=(
        "tests/test_fmha.py",
        "tests/test_fmha_bwd.py",
    ),
)
```

Interface values may be repository-relative Python paths or dotted module names. Test values are repository-relative Python paths.

### Static analysis only

Dependency discovery must not import an analyzed backend, compile a kernel, render build outputs to disk, or load an artifact. This constraint makes target-branch analysis possible and prevents dependency resolution from requiring an accelerator runtime.

The analyzer resolves statically identifiable lazy exports and literal imports. An unknown dynamic import is reported as an unresolved dependency and causes selection to fail closed when changes are present.

### Provider-owned dependency discovery

Python DSL and JIT implementations use different dependency providers:

- The Python AST provider follows reachable repository-local symbols, imports, namespace packages, and static lazy exports.
- The JIT provider obtains the declared build inputs of a `JITSpec`, follows source and template includes, and records source generators and opaque gitlinks.
- The test AST provider follows dependencies imported by pytest files, including local test utilities, `mate.testing` modules, statically declared `pytest_plugins`, and applicable `conftest.py` files. JIT boundaries reached only from tests are expanded by the same JIT provider used for interface dependencies.

The catalog does not reproduce information already owned by these providers.

### File-level dynamic dispatch

Dynamic attribute selection, such as `getattr(dispatcher, method_name)`, does not by itself identify another file. If the receiver and its implementations are already represented by files in the dependency closure, the entire implementation file is sufficient. Method-level dispatch resolution is intentionally not required.

Dynamic behavior remains fail-closed when it may import or select an unknown module or file. Literal dynamic imports are followed statically; imports whose module name cannot be determined are unresolved.

### Conservative Git comparison

Selection resolves dependencies independently in the target and head snapshots. When the requested head is the current checkout, it also resolves the live worktree if staged, unstaged, or untracked changes exist. The union covers both removed and newly introduced dependencies.

Git rename and copy records retain both paths. Direct JIT inputs and includes located in a submodule are represented by the parent-repository gitlink rather than by individual files inside the submodule.

## Dependency Model

For an operator `op`, the effective dependency set is:

```text
Dependencies(op) =
    RepositoryConfiguration
  + CatalogDefinition
  + InterfaceAstClosure(op.interfaces)
  + DiscoveredJitDependencies
  + TestAstClosure(op.tests)
  + ApplicableConftestChain(op.tests)
```

`pyproject.toml` is a repository-wide test environment dependency and therefore belongs to every operator closure.

`.gitmodules` belongs only to JIT closures that use a configured submodule. A change to the submodule commit is matched through the gitlink path; a change to `.gitmodules` reruns the operators whose JIT dependencies use that configuration.

## User Guide

### Prerequisites

- Run commands from the repository root, or pass `--repo <path>`.
- Use Python 3.10 or newer and make the target Git ref available locally. The default target is `origin/master` and the default head is `HEAD`.
- `validate`, `resolve`, and `select` perform static analysis and do not require an accelerator, import analyzed backends, compile kernels, or render templates.
- `run` requires whatever runtime dependencies are used by the repository-defined runner. The current MATE runner invokes pytest.

Refresh the default target before selecting tests when the remote ref may be stale:

```bash
git fetch origin master
```

### Command summary

```text
python tests/main.py validate [options]
python tests/main.py resolve [options] <operator>
python tests/main.py select [options] [operator ...]
python tests/main.py run     [options] [operator ...]
```

`validate` checks the complete catalog. `resolve` inspects one registered operator. `select` computes and reports test files without executing them. `run` computes the same selection, always reports it, and then passes it to the runner configured by `tests/main.py`.

### Validate the catalog

```bash
python tests/main.py validate
python tests/main.py validate --json
```

Validation checks every catalog definition and registered test file, requires
every `tests/**/test_*.py` file to belong to at least one operator, and resolves
the complete static dependency graph for every operator. It returns exit code
`1` for any catalog or dependency issue. The command performs no imports of
analyzed backends and does not execute pytest; it is the pre-commit maintenance
gate for the dependency graph.

### Inspect one operator

```bash
python tests/main.py resolve mate.ops.attention.fa
python tests/main.py resolve mate.ops.attention.fa --files
python tests/main.py resolve mate.ops.attention.fa --strict --json
```

Options:

| Option | Meaning |
| --- | --- |
| `--files` | Print every dependency file and its provider owner. |
| `--json` | Emit structured files, owners, boundaries, tests, and issues. |
| `--strict` | Return exit code `1` when resolution reports any issue. |
| `--repo PATH` | Analyze another repository checkout. |

Use `--strict --files` when registering or reviewing an operator. A successful result must include the public interface, DSL implementation files, discovered JIT sources/templates/headers, test helpers, plugins, and applicable `conftest.py` files.

### Select tests from Git changes

Evaluate the complete catalog against `origin/master` and the current checkout:

```bash
python tests/main.py select
```

Filter the final test selection to specific operators during local diagnosis:

```bash
python tests/main.py select \
  mate.ops.attention.fa \
  mate.ops.attention.dsa \
  --target origin/master \
  --head HEAD
```

The filter does not shrink dependency analysis. The selector still resolves the
complete catalog to classify ownership. For example, a GEMM-owned change remains
owned by GEMM when the command filters for FA; it is reported as owned outside
the filter and does not trigger the unowned-path full fallback.

For `merge-base(target, head)..head`, the selector reads committed changes. If `head` resolves to the current checkout's `HEAD`, it also includes staged, unstaged, and untracked paths and resolves a separate live-worktree dependency graph.

`--changed-file` replaces all Git and worktree change discovery. Repeat it to run a deterministic simulation:

```bash
python tests/main.py select \
  mate.ops.attention.fa \
  --changed-file mate/mha_interface.py \
  --changed-file include/mate/attention/fmha/fmha_fwd_kernel.hpp
```

Do not combine `--changed-file` with the expectation that current working-tree modifications will also be included.

### Consume machine-readable output

```bash
python tests/main.py select --json > selection.json
jq -r '.selected_tests[]' selection.json
```

Important fields:

| Field | Meaning |
| --- | --- |
| `mode` | `skip`, `selected`, or `full`. |
| `reason` | Why that mode was selected. |
| `selected_tests` | Deduplicated repository-relative test files. |
| `not_selected_tests` | Complete current test suite minus `selected_tests`; reported for inspection and never passed to the runner. |
| `changes` | Git status and new/old paths for changes and renames. |
| `matches` | Changed path to operator/provider dependency provenance. |
| `ownership_matches` | All catalog ownership matches, including operators excluded by the command-line filter. |
| `ignored_paths` | Ownerless changed paths accepted by `SelectionPolicy`. |
| `unowned_paths` | Ownerless changed paths that caused a full fallback. |
| `resolution_issues` | Operator-owned static-analysis failures. |

Current normal-selection reasons include `no_changes`, `ignored_paths`,
`filtered_operators`, `dependency_match`, and `resolution_issues`. Ownerless
paths return `full / unowned_paths`; a path matching `forced_full_paths` returns
`full / policy_full`. A changed catalog definition file returns
`full / catalog_changed`.

The text report shows both sets with their share of the complete current test
suite, for example `Selected tests [10 / 17]` and
`Not selected tests [7 / 17]`. Selected entries are green and unselected
entries are yellow in an interactive terminal. Redirected output, `NO_COLOR`,
`TERM=dumb`, CI, and Jenkins environments always use plain text so persisted
logs never contain raw ANSI escape sequences. JSON output never uses color.

### Force a complete test selection

The following forms bypass Git discovery and all dependency resolution and
ignore any operator filter. Full selection recursively discovers every
`tests/**/test_*.py` file in the current checkout and unions those files with
the catalog's registered tests:

```bash
python tests/main.py select --full --reason manual_validation
MATE_CI_FORCE_FULL=1 python tests/main.py select
MATE_CI_MODE=full python tests/main.py select
MATE_CI_MODE=daily python tests/main.py select --json
```

The default reasons are `forced_full`, `ci_mode_full`, and `daily`. Use `--reason <value>` to record a more specific reason. In forced-full JSON output, target/head commits and merge base are `null` because Git was deliberately not evaluated.

### Run selected tests

```bash
python tests/main.py run mate.ops.attention.fa --target origin/master
python tests/main.py run mate.ops.attention.fa --target origin/master --verbose
MATE_CI_MODE=daily python tests/main.py run --verbose
```

`run` always prints the complete selection before execution. `--verbose` additionally prints every command issued by the runner. If no tests are selected, the runner is not called. If selected tests exist but the entry point did not configure a runner, the CLI exits with code `2`.

The MATE runner first passes every selected test to parallel pytest dry-run.
Dry-run support is an explicit pytest capability marker. A fully adapted test
file opts in at module scope:

```python
pytestmark = pytest.mark.mate_dry_run
```

During preflight, unmarked tests are skipped before their test bodies execute.
The terminal summary prints a complete warning section grouped by unmarked test
file. A marked test must terminate through `MateDryRunComplete`; returning
normally is a preflight failure, while an explicit pytest skip remains a skip.
A successful preflight is followed by the correctness run over the same file
list, where the marker has no effect. A failed preflight stops immediately.
Both stages disable legacy pytest collection sharding, and the correctness
stage removes `MATE_DRY_RUN` from its environment.

The resolver CLI does not prescribe pytest flags, preflight behavior, sharding, or environment variables. The MATE policy is implemented by `run_selected_tests()` in `tests/main.py`. Replace that function, or pass another runner to `resolver_main()`, to control execution without changing the catalog or dependency graph.

The runner receives a `TestRunRequest`. `request.selection.selected_tests` is its stable input. `request.run()` executes one argument array without `shell=True`, returns that command's exit code, and may be called repeatedly:

```python
def run_selected_tests(request: TestRunRequest) -> int:
    tests = request.selection.selected_tests
    environment = {
        "MATE_PYTEST_SHARD_TOTAL": "1",
        "MATE_PYTEST_SHARD_INDEX": "0",
    }
    status = request.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "--dry-run",
            "-n",
            "auto",
            "-v",
            *tests,
            "--max-worker-restart",
            "0",
        ),
        environment=environment,
    )
    if status:
        return status
    return request.run(
        (sys.executable, "-m", "pytest", "-v", *tests),
        environment={**environment, "MATE_DRY_RUN": None},
    )
```

An environment value of `None` removes that variable for the command. A runner may also group files, invoke different programs, write reports, or stop after the first failed stage.

### Register an operator

Add exactly one catalog registration to `tests/main.py`:

```python
TEST_CATALOG.register(
    "mate.ops.norm",
    interfaces=("mate/norm/__init__.py",),
    tests=("tests/test_norm.py",),
)
```

Only declare the canonical operator name, public interface roots, and owned tests. Do not add backend names, JIT factories, templates, native sources, headers, DSL implementation roots, test helpers, or source globs.

Validate a new registration with one dependency inspection and two controlled selections:

```bash
python tests/main.py validate
python tests/main.py resolve <operator> --strict --files
python tests/main.py select <operator> --changed-file <known-dependency>
python tests/main.py select <operator> --changed-file <unrelated-file>
```

The known dependency must select every test registered for the operator. An
ownerless unrelated path deliberately returns the complete catalog with
`full / unowned_paths`; a path owned only by another operator is reported as
`filtered_operators` and does not select the requested operator. Every top-level
operator test under `tests/test_*.py` must belong to at least one catalog entry;
the catalog completeness test enforces this rule.

### Exit codes and troubleshooting

| Exit code | Meaning |
| --- | --- |
| `0` | Resolution/selection succeeded, a skip was valid locally, or the runner succeeded. |
| `1` | `validate` or `resolve --strict` found one or more issues. |
| `2` | Invalid arguments, unknown operator, Git/snapshot failure, or missing runner. |
| Other | `run` returns the repository runner's command status unchanged. |

Common checks:

- `unknown operator`: inspect registrations in `tests/main.py` and use the canonical dotted name.
- `git rev-parse ... failed`: fetch the target ref or pass an existing `--target`.
- `No tests selected`: inspect `Changed paths`, `mode`, and `reason`; use `--json` to inspect matches.
- Resolution issues: run `resolve <operator> --strict --files` and fix the reported Python, JIT, or include boundary.
- Slow selection: ownership always uses the complete catalog. Snapshot/result
  reuse is implemented; lower-level Python parse and symbol-closure caches remain
  planned. An operator filter does not reduce ownership analysis.

### CI and pre-commit integration

Pre-commit validates the Catalog without running operator tests:

```bash
python tests/main.py validate
```

Both MATE CI environments select and execute tests in one process:

```bash
python tests/main.py run --target origin/master --head HEAD --verbose
```

There is no persisted test plan or hand-written source-to-test map. Git or
snapshot failures produce a nonzero job status; they cannot become a successful
empty selection.

## Scope

The current catalog covers the primary MATE package and its top-level operator tests.

Wrapper projects under `wrappers/` are deliberately out of scope. They have independent package roots, build metadata, import behavior, and test environments. `tests/main.py` supplies an explicit `wrappers/**` ignore rule through `SelectionPolicy`; wrapper paths are not given artificial operator ownership.

Benchmarks are also not pytest targets. They may share the same future operator identity, but benchmark selection and execution policy remains separate from correctness-test selection.

## Template Policy

The declared template file is a build dependency. The resolver uses Jinja's own
AST and meta API to recursively track literal `extends`, `include`, `import`, and
`from ... import` fragments with the renderer's loader root. Missing, dynamic,
or syntactically invalid references become operator-owned issues. Rendered output
remains outside the dependency model: the resolver does not render templates or
inspect includes that exist only after rendering.

## Failure Policy

Unresolved dependencies are attached to the affected operator. If repository changes exist, an affected unresolved operator fails closed by selecting all tests registered for that operator. An unresolved dependency in one operator must not cause unrelated operators to run.

This policy favors missed-test prevention while preserving operator-level isolation.

## Correctness Status

JIT three-state recognition, static source-branch union, repository-local native
include closure, Jinja fragment tracking, and repository-level unowned/ignore
selection policy are implemented.

### JIT three-state detection (implemented)

The three states classify a Python symbol during static dependency discovery. They are not JIT runtime, compilation, or cache states.

| Probe result | Example | Required provider behavior |
| --- | --- | --- |
| `not-jit` | An ordinary helper with no reachable `gen_jit_spec()` call. | Do not claim the symbol; allow normal Python AST traversal. |
| `resolved-jit` | A factory whose sources, templates, and include directories are statically resolved. | Claim the boundary and return the complete JIT dependency closure. |
| `unresolved-jit` | A reachable `gen_jit_spec()` exists, but one of its build inputs cannot be evaluated statically. | Claim the boundary, preserve the analysis error, and fail closed for the owning operator. |

Recognition and dependency evaluation are separate. Finding a reachable
`gen_jit_spec()` permanently assigns the symbol boundary to the JIT provider.
If dependency evaluation then fails, the provider retains the original error
and reports `unresolved-jit`; it never falls back to ordinary Python traversal.

For example, a backend may contain:

```python
def get_module(config):
    sources = choose_sources(config)
    return gen_jit_spec("demo", sources=sources).build_and_load()
```

The source analyzer conservatively unions every statically enumerable candidate:
literal list/tuple/set members, starred and concatenated sequences, conditional
expressions, assignments in every control-flow branch, `append`/`extend` calls,
static collection lookups, and every return branch of local or statically
imported helpers. Runtime predicates are deliberately not evaluated. If any
possible branch cannot be enumerated, the whole factory becomes
`unresolved-jit` rather than silently dropping that branch.

The current repository has no `choose_sources` helper. Its only conditionally
mutated source list is the MUBIN launcher's `flash_mla` append; both the original
sources and the appended `attention_combine.mu` now belong to every operator
that reaches that factory. Generated output paths remain excluded: the tracked
inputs are their generator Python files and source templates.

### Repository-local native includes (implemented)

`JitSpec.extra_include_dirs` is the build declaration and the resolver's include
search space. The scanner starts from direct source files, source templates, and
header seeds. Quoted includes search the source directory and then the declared
include roots; angled includes search the declared roots. Every resolved file
inside the repository is recorded and scanned recursively. A file inside a
submodule is represented by the parent-repository gitlink.

External libraries are intentionally outside this dependency model. A header
resolved outside the repository is ignored. A missing literal include that
cannot be shown to belong to an existing namespace under a repository-local
include root is also treated as external and ignored; repository-external
include roots never produce local-missing issues. Missing targets in an existing
repository namespace and non-literal directives such as `#include HEADER_MACRO`
remain unresolved issues because they may hide a project dependency.

Include directories are search roots, not recursive dependency globs. This keeps
the graph finer than assigning all of `include/` or `csrc/` to every JIT
operator. The target and head snapshots are both resolved, so a deleted header
is owned through the target graph and a newly added reachable header is owned
through the head graph. No catalog-level include list or toolchain allowlist is
required.

### Jinja template fragments (implemented)

Jinja itself should parse template-to-template dependencies. Its official meta API operates on the AST returned by `Environment.parse()`, and `meta.find_referenced_templates()` returns literal `extends`, `include`, `import`, and `from ... import` targets. It yields `None` for a dynamic template expression. See the [Jinja Meta API](https://jinja.palletsprojects.com/en/stable/api/#the-meta-api).

The resolver implements a read-only recursive closure:

```python
from jinja2 import Environment, meta

tree = Environment().parse(source, name=template_name)
references = tuple(meta.find_referenced_templates(tree))
```

For each literal reference, resolve the name using the same `FileSystemLoader` root as `JitSourceTemplate.render()`, add the file, parse it, and continue. Keep a visited set for cycles and duplicate references. Reject paths escaping the loader root. A missing literal or Jinja syntax error becomes an operator-owned issue; a `None` reference becomes a dynamic-template issue. No template is rendered and no Python context is evaluated.

This is different from inspecting rendered C/C++ output. A fragment such as `{% include "common.j2" %}` is a source dependency and should be tracked. A rendered line such as `#include "{{ selected_header }}"` remains outside the dependency model. The current repository's 21 Jinja template files do not presently contain template-to-template directives, so this change is future-facing rather than a fix for an existing closure.

### Unowned changes and selection policy (implemented)

An unowned path is a changed path that matches no dependency record after evaluating the complete catalog. It returns `full / unowned_paths`; an intentional ignore is reported separately.

`SelectionPolicy` is supplied independently from the catalog:

```python
TEST_SELECTION_POLICY = SelectionPolicy(
    ignored_paths=("wrappers/**",),
    forced_full_paths=(
        ".ciConfig.yaml",
        "mate/testing/resolver/**",
        "tests/main.py",
    ),
)

resolver_main(
    TEST_CATALOG,
    policy=TEST_SELECTION_POLICY,
    runner=run_selected_tests,
)
```

The current MATE entry point declares only `wrappers/**` as ignored; the
`forced_full_paths` example illustrates the available API, not the current MATE
configuration. Catalog definition files are an intrinsic full-selection boundary
and do not need to be repeated in `forced_full_paths`. The policy never maps
sources to operators or tests. Precedence is:

1. An explicit or Daily forced-full request returns `full` immediately.
2. A changed catalog definition file returns `full`.
3. A changed path matching `forced_full_paths` returns `full`.
4. A path with dependency owners contributes those operators, even if it also matches an ignore pattern.
5. An ownerless path matching `ignored_paths` is ignored.
6. Any remaining ownerless path returns `full`.
7. Return `selected` when at least one operator is selected and every other path is ignored; return `skip` only for no changes or an all-ignored change.

Consequently, a wrapper-only change becomes `skip`; wrapper plus owned operator
code becomes `selected`; wrapper plus unknown code becomes `full`. The selector
always evaluates the complete catalog for ownership. An operator filter is
applied only to the final test set, so ownership by a filtered-out operator is
reported as `filtered_operators`, not misclassified as unowned. Delete paths are
checked in the target graph; add paths in the head/worktree graph; both old and
new sides of rename/copy records are classified.
