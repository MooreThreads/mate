# 增量测试解析器

增量测试解析器根据仓库变更选择 MATE 算子测试。它为每个已注册算子构建静态依赖闭包，与 Git diff 求交集，并返回所有受影响算子的完整测试集合。

resolver 以仓库 CLI 的形式提供，并由 MATE CI 和 pre-commit 使用。静态命令不会导入或执行被分析的算子模块。

## 设计原则

### 以算子为选择粒度

算子是依赖归属和测试执行的基本单位。一个算子可以包含多个 backend，但 resolver 不进行 backend 级测试筛选。算子的任意依赖发生变化时，执行该算子登记的全部测试。

### Catalog 保持最小声明

Catalog 只登记三项信息：

1. 规范的算子名称。
2. 一个或多个顶层公开接口模块。
3. 一个或多个归属于该算子的 pytest 文件。

backend、JIT factory、模板、头文件、DSL 实现、测试辅助文件和 `conftest.py` 均由 resolver 推导，不应在 Catalog 中重复登记。

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

接口可以使用仓库相对 Python 路径或 dotted module name；测试必须使用仓库相对 Python 路径。

### 仅进行静态分析

依赖发现不得导入 backend、编译 kernel、向磁盘生成构建产物或加载 artifact。这样才能分析 target branch，并避免 resolver 依赖加速器运行环境。

解析器支持静态可确定的 lazy export 和 literal import。无法确定模块名的动态 import 会形成 unresolved dependency，并在存在变更时对所属算子 fail-closed。

### 依赖由 Provider 负责

- Python AST provider 跟踪可达的本地符号、import、namespace package 和静态 lazy export。
- JIT provider 从 `JITSpec` 获取构建输入，继续解析 source/template include、生成器文件和 gitlink。
- Test AST provider 解析 pytest 文件依赖，包括本地测试工具、`mate.testing`、静态 `pytest_plugins` 和适用的 `conftest.py`。仅由测试侧到达的 JIT boundary 也交给同一个 JIT provider 展开。

Catalog 不复制 provider 已经拥有的信息。

### 动态方法分派按文件处理

`getattr(dispatcher, method_name)` 本身不会指向新的文件。只要 receiver 及其实现文件已经进入依赖闭包，收集整个实现文件即可，不要求方法级精确分析。

如果动态行为可能导入或跳转到未知模块或文件，则继续 fail-closed。literal dynamic import 可以静态跟踪；模块名不可确定时必须报告 unresolved dependency。

### 保守处理 Git 变更

resolver 分别在 target 和 head snapshot 中构建依赖图。head 为当前 checkout 时，还会合并 staged、unstaged 和 untracked 变更，并为 dirty worktree 单独构图。因此删除旧依赖和增加新依赖都能被覆盖。

rename 和 copy 同时保留新旧路径。位于 submodule 内的 JIT 直接输入和递归 include 不展开内部文件，在父仓库中统一表示为 gitlink。

## 依赖模型

```text
Dependencies(op) =
    RepositoryConfiguration
  + CatalogDefinition
  + InterfaceAstClosure(op.interfaces)
  + DiscoveredJitDependencies
  + TestAstClosure(op.tests)
  + ApplicableConftestChain(op.tests)
```

`pyproject.toml` 是全仓库测试环境依赖，因此属于所有算子的依赖闭包。

`.gitmodules` 只属于实际使用对应 submodule 的 JIT 闭包。submodule commit 变化通过 gitlink 路径匹配；`.gitmodules` 变化只重跑依赖其配置的算子。

## 使用手册

### 使用前提

- 在仓库根目录执行命令；也可以通过 `--repo <path>` 指定其他 checkout。
- 使用 Python 3.10 或更高版本，并确保 target Git ref 在本地可用。默认 target 是 `origin/master`，默认 head 是 `HEAD`。
- `validate`、`resolve` 和 `select` 只做静态分析，不要求加速卡，不导入被分析 backend，不编译 kernel，也不渲染模板。
- `run` 需要安装仓库自定义 runner 所需的运行时依赖；当前 MATE 基础 runner 会调用 pytest。

如果本地远端引用可能过期，先更新默认 target：

```bash
git fetch origin master
```

### 命令总览

```text
python tests/main.py validate [options]
python tests/main.py resolve [options] <operator>
python tests/main.py select [options] [operator ...]
python tests/main.py run     [options] [operator ...]
```

`validate` 校验完整 Catalog；`resolve` 查看一个已注册算子的依赖闭包；`select` 只计算并输出测试文件，不执行测试；`run` 先执行相同的选择并完整输出结果，再交给 `tests/main.py` 配置的 runner。

### 校验 Catalog

```bash
python tests/main.py validate
python tests/main.py validate --json
```

校验会检查所有 Catalog definition 和已注册测试文件，要求每个
`tests/**/test_*.py` 至少归属于一个 operator，并解析每个 operator 的完整静态依赖
图。任一 Catalog 或依赖 issue 都会返回退出码 `1`。该命令不导入被分析 backend，
也不执行 pytest；它是 dependency graph 的 pre-commit 维护门禁。

### 查看单个算子的依赖

```bash
python tests/main.py resolve mate.ops.attention.fa
python tests/main.py resolve mate.ops.attention.fa --files
python tests/main.py resolve mate.ops.attention.fa --strict --json
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--files` | 输出每个依赖文件及其 provider owner。 |
| `--json` | 输出结构化的 files、owners、boundaries、tests 和 issues。 |
| `--strict` | 存在任何解析 issue 时返回退出码 `1`。 |
| `--repo PATH` | 分析另一个仓库 checkout。 |

新增或 review 算子时建议使用 `--strict --files`。成功结果应包含公开接口、DSL 实现、自动发现的 JIT source/template/header、测试 helper、plugin 和适用的 `conftest.py`。

### 根据 Git 变更选择测试

使用 `origin/master` 对比当前 checkout，并分析完整 Catalog：

```bash
python tests/main.py select
```

本地排查时可以只筛选指定算子的最终测试：

```bash
python tests/main.py select \
  mate.ops.attention.fa \
  mate.ops.attention.dsa \
  --target origin/master \
  --head HEAD
```

该参数不会缩小 ownership 分析范围。selector 始终解析完整 Catalog；例如只筛选
FA 时，GEMM 依赖变化仍归属于 GEMM，会显示为 filter 外的 owner，而不会误触发
unowned full fallback。

selector 对 `merge-base(target, head)..head` 读取已提交变更。如果 `head` 解析为当前 checkout 的 `HEAD`，还会加入 staged、unstaged 和 untracked 路径，并额外解析 live worktree 依赖图。

`--changed-file` 会完全替代 Git 和 worktree 变更发现。该参数可以重复，用于确定性模拟：

```bash
python tests/main.py select \
  mate.ops.attention.fa \
  --changed-file mate/mha_interface.py \
  --changed-file include/mate/attention/fmha/fmha_fwd_kernel.hpp
```

使用 `--changed-file` 时，不能再预期当前 working tree 的其他修改会被自动加入。

### 使用机器可读输出

```bash
python tests/main.py select --json > selection.json
jq -r '.selected_tests[]' selection.json
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `mode` | `skip`、`selected` 或 `full`。 |
| `reason` | 产生该模式的原因。 |
| `selected_tests` | 去重后的仓库相对测试文件。 |
| `not_selected_tests` | 当前完整测试集合减去 `selected_tests`；仅用于人工检查，不会传给 runner。 |
| `changes` | Git status，以及 rename 的新旧路径。 |
| `matches` | changed path 到 operator/provider dependency 的完整来源。 |
| `ownership_matches` | 完整 Catalog 的所有 owner 命中，包括被命令行 filter 排除的算子。 |
| `ignored_paths` | 被 `SelectionPolicy` 明确接受的无 owner 路径。 |
| `unowned_paths` | 触发 full fallback 的无 owner 路径。 |
| `resolution_issues` | 归属于具体 operator 的静态解析失败。 |

普通增量选择的 `reason` 包括 `no_changes`、`ignored_paths`、
`filtered_operators`、`dependency_match` 和 `resolution_issues`。无 owner 路径
返回 `full / unowned_paths`；命中 `forced_full_paths` 返回
`full / policy_full`；Catalog definition file 发生变化时返回
`full / catalog_changed`。

文本报告会同时显示两组文件及其在当前完整测试集合中的占比，例如
`Selected tests [10 / 17]` 和 `Not selected tests [7 / 17]`。交互式终端中，
已选择项显示为绿色，未选择项显示为黄色。输出重定向、`NO_COLOR`、
`TERM=dumb`、CI 和 Jenkins 环境始终输出纯文本，持久化日志不会包含裸 ANSI
转义序列；JSON 输出也永不着色。

### 强制选择全量测试

以下方式会跳过 Git 发现和全部依赖解析，并忽略 operator 过滤条件。Full selection
会递归发现当前 checkout 中全部 `tests/**/test_*.py`，再与 Catalog 已注册测试取并集：

```bash
python tests/main.py select --full --reason manual_validation
MATE_CI_FORCE_FULL=1 python tests/main.py select
MATE_CI_MODE=full python tests/main.py select
MATE_CI_MODE=daily python tests/main.py select --json
```

默认 reason 分别为 `forced_full`、`ci_mode_full` 和 `daily`。可使用 `--reason <value>` 记录更具体的原因。强制全量的 JSON 中 target/head commit 和 merge base 为 `null`，表示 Git 被明确跳过，而不是解析失败。

### 执行选中的测试

```bash
python tests/main.py run mate.ops.attention.fa --target origin/master
python tests/main.py run mate.ops.attention.fa --target origin/master --verbose
MATE_CI_MODE=daily python tests/main.py run --verbose
```

`run` 总是先输出完整 selection。`--verbose` 还会显示 runner 启动的每一条命令。没有测试被选中时不会调用 runner；已经选中测试但入口没有配置 runner 时，CLI 返回退出码 `2`。

MATE runner 会先将所有选中测试交给并行 pytest dry-run。dry-run 支持通过显式的
pytest capability marker 声明。完整适配的测试文件应在模块级登记：

```python
pytestmark = pytest.mark.mate_dry_run
```

preflight 会在执行测试体之前跳过未标记测试，并在 terminal summary 中按文件完整列出
warning。已标记测试必须通过 `MateDryRunComplete` 结束；正常返回会使 preflight 失败，
显式的 pytest skip 则保持为 skip。preflight 成功后，对同一文件列表执行正式
correctness test，此时 marker 不产生作用；preflight 失败则立即停止。两个阶段都会禁用
旧 pytest collection sharding，正式测试阶段还会从环境中移除 `MATE_DRY_RUN`。

resolver CLI 不规定 pytest 参数、preflight、shard 或环境变量策略。MATE 策略由 `tests/main.py` 中的 `run_selected_tests()` 实现。可以替换这个函数，或向 `resolver_main()` 传入另一个 runner，而无需修改 Catalog 或依赖图。

runner 接收 `TestRunRequest`；`request.selection.selected_tests` 是稳定输入。`request.run()` 使用参数数组执行一条命令，不使用 `shell=True`，返回该命令的退出码，并且可以重复调用：

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

环境变量值为 `None` 表示执行该命令时删除该变量。runner 也可以对测试分组、调用不同程序、写报告，或在任一阶段失败后停止。

### 注册新算子

在 `tests/main.py` 中增加一项 Catalog 注册：

```python
TEST_CATALOG.register(
    "mate.ops.norm",
    interfaces=("mate/norm/__init__.py",),
    tests=("tests/test_norm.py",),
)
```

只声明 canonical operator name、公开 interface roots 和所属 tests。不要登记 backend、JIT factory、template、native source、header、DSL implementation root、测试 helper 或 source glob。

通过一次依赖检查和两次受控选择验证新注册项：

```bash
python tests/main.py validate
python tests/main.py resolve <operator> --strict --files
python tests/main.py select <operator> --changed-file <known-dependency>
python tests/main.py select <operator> --changed-file <unrelated-file>
```

已知依赖必须选择该算子登记的全部测试。无 owner 的无关路径会有意返回完整
Catalog，即 `full / unowned_paths`；只归属于另一个算子的路径会报告
`filtered_operators`，不会选择当前请求的算子。每个 `tests/test_*.py` 顶层算子测试
都必须至少属于一个 Catalog 项，Catalog completeness test 会检查这一约束。

### 退出码与排障

| 退出码 | 含义 |
| --- | --- |
| `0` | 解析/选择成功、本地 skip 有效，或 runner 执行成功。 |
| `1` | `validate` 或 `resolve --strict` 发现一个或多个 issue。 |
| `2` | 参数错误、未知 operator、Git/snapshot 失败，或未配置 runner。 |
| 其他 | `run` 原样返回仓库 runner 的命令退出码。 |

常见排查方式：

- `unknown operator`：检查 `tests/main.py` 的注册项，并使用 canonical dotted name。
- `git rev-parse ... failed`：fetch 对应 target，或传入已存在的 `--target`。
- `No tests selected`：查看 `Changed paths`、`mode` 和 `reason`；使用 `--json` 检查 matches。
- 存在 resolution issue：执行 `resolve <operator> --strict --files`，定位 Python、JIT 或 include boundary。
- 选择较慢：ownership 始终使用完整 Catalog。snapshot/result 复用已经实现；底层
  Python parse 与 symbol-closure cache 仍在 TODO。指定 operator 不会减少 ownership
  分析量。

### CI 与 pre-commit 接入

pre-commit 只校验 Catalog，不执行算子测试：

```bash
python tests/main.py validate
```

两个 MATE CI 环境均在一个进程内完成选择和执行：

```bash
python tests/main.py run --target origin/master --head HEAD --verbose
```

新流程没有持久化 test plan，也没有手工 source-to-test map。Git 或 snapshot 失败会
返回非零 job 状态，不能退化成成功的空选择。

## 范围

当前 Catalog 只覆盖 MATE 主包及其顶层算子测试。

`wrappers/` 下的独立项目明确不在当前范围内。它们拥有独立的 package root、
构建元数据、import 行为和测试环境。`tests/main.py` 通过 `SelectionPolicy` 注入
显式 `wrappers/**` ignore 规则，不为 wrapper 伪造 operator ownership。

benchmark 也不是 pytest target。未来可以与测试共享 operator identity，但 benchmark 的选择和执行策略应保持独立。

## 模板策略

声明的模板文件本身属于构建依赖。resolver 使用 Jinja 自身的 AST 和 meta API，
按 renderer 的 loader root 递归跟踪 literal `extends`、`include`、`import` 和
`from ... import` fragment。缺失、动态或语法错误的引用形成 operator-owned
issue。render 结果仍不属于依赖模型：resolver 不执行模板渲染，也不解析仅在
render 结果中出现的 include。

## 失败策略

unresolved dependency 只归属于对应算子。存在代码变更时，该算子 fail-closed，执行其登记的全部测试；一个算子的 unresolved dependency 不得导致无关算子被执行。

该策略优先避免漏测，同时维持算子级隔离。

## 正确性状态

JIT 三态识别、静态 source 分支并集、仓库内 native include 闭包、Jinja fragment
以及仓库级 unowned/ignore selection policy 均已实现。

### JIT 三态识别（已实现）

这里的三态用于静态依赖发现阶段对 Python symbol 分类，不是 JIT 的运行、编译或缓存状态。

| 探测结果 | 例子 | Provider 应采取的行为 |
| --- | --- | --- |
| `not-jit` | 普通 helper，静态可达路径中没有 `gen_jit_spec()`。 | 不接管该 symbol，继续普通 Python AST 遍历。 |
| `resolved-jit` | factory 的 source、template 和 include directory 均可静态解析。 | 接管 boundary，并返回完整 JIT 依赖闭包。 |
| `unresolved-jit` | 已发现可达的 `gen_jit_spec()`，但某个构建输入无法静态求值。 | 仍由 JIT provider 接管，保留解析错误，并对所属 operator fail-closed。 |

识别与依赖求值相互分离。一旦发现静态可达的 `gen_jit_spec()`，该 symbol boundary
就始终归 JIT provider；后续依赖求值失败时，provider 保留原始错误并返回
`unresolved-jit`，不会降级成普通 Python AST traversal。

例如，backend 可以包含：

```python
def get_module(config):
    sources = choose_sources(config)
    return gen_jit_spec("demo", sources=sources).build_and_load()
```

source analyzer 对所有静态可枚举候选取并集，包括 list/tuple/set 成员、starred
与拼接序列、条件表达式、各控制流分支内的赋值、`append`/`extend`、静态 collection
lookup，以及本地或静态 import helper 的全部 return 分支。运行时 predicate 不会被
求值。只要任一可能分支无法枚举，整个 factory 就变为 `unresolved-jit`，不会静默
丢弃该分支。

当前仓库没有 `choose_sources` helper。唯一按条件修改 source list 的代码是 MUBIN
launcher 的 `flash_mla` append；原始 sources 与追加的 `attention_combine.mu` 现在都
归属所有可达该 factory 的 operator。生成产物路径本身仍不属于依赖；实际跟踪的是
generator Python 文件和 source template。

### 仓库内 native include（已实现）

`JitSpec.extra_include_dirs` 是构建声明，也是 resolver 的 include 搜索空间。scanner
从 direct source、source template 和 header seed 开始。quoted include 依次搜索当前
source 目录和声明的 include roots；angled include 只搜索声明的 roots。仓库内命中的
每个文件都会记录并递归扫描；submodule 内文件统一表示为父仓库 gitlink。

外部库明确不属于本依赖模型。解析到仓库外的 header 会被忽略；无法证明属于某个
仓库内既有 namespace 的 missing literal include 也按 external 忽略；仓库外 include
root 不会产生 local-missing issue。若 missing target 位于仓库内已存在的 namespace，
或者出现 `#include HEADER_MACRO` 这种非 literal directive，则仍报告 unresolved，
因为它们可能隐藏项目自身依赖。

include directory 仅作为搜索根，不会作为递归 dependency glob。这比把整个
`include/` 或 `csrc/` 归给每个 JIT operator 更精确。resolver 同时分析 target 与 head
snapshot：删除的 header 由 target graph 保留 owner，新增且实际可达的 header 由 head
graph 建立 owner。因此无需在 Catalog 中登记 include 清单，也无需维护 toolchain
allowlist。

### Jinja template fragments（已实现）

可以，并且应直接使用 Jinja 解析模板之间的依赖。Jinja 官方 meta API 接收 `Environment.parse()` 产生的 AST；`meta.find_referenced_templates()` 会返回 literal `extends`、`include`、`import` 和 `from ... import` 目标，动态模板表达式则返回 `None`。参见 [Jinja Meta API](https://jinja.palletsprojects.com/en/stable/api/#the-meta-api)。

resolver 已实现只读递归闭包：

```python
from jinja2 import Environment, meta

tree = Environment().parse(source, name=template_name)
references = tuple(meta.find_referenced_templates(tree))
```

对每个 literal reference，使用与 `JitSourceTemplate.render()` 相同的 `FileSystemLoader` root 解析文件名，加入依赖集合，继续解析 fragment。使用 visited set 处理循环和重复引用，并拒绝逃出 loader root 的路径。缺失 literal 或 Jinja syntax error 形成 operator-owned issue；`None` 形成 dynamic-template issue。整个过程不 render 模板，也不求值 Python context。

这与分析渲染后的 C/C++ 内容不同：`{% include "common.j2" %}` 中的 `common.j2` 是模板源文件，必须跟踪；`#include "{{ selected_header }}"` 是渲染结果的一部分，仍不属于依赖模型。目前仓库中的 21 个 Jinja 模板没有使用模板间引用，因此该能力主要用于未来扩展，不是在修复当前已有闭包。

### 未归属变更与 Selection Policy（已实现）

unowned path 是指在完整 Catalog 构图后，没有匹配任何 dependency record 的 changed path。它返回 `full / unowned_paths`；显式 ignore 会单独报告。

`SelectionPolicy` 独立于 Catalog 注入：

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

当前 MATE 入口只声明 `wrappers/**` ignore；上面的 `forced_full_paths` 用于说明已提供
的 API，不是当前 MATE 配置。Catalog definition file 是内建的 full selection
边界，无需在 `forced_full_paths` 中重复声明。policy 绝不建立 source 到 operator/test
的映射。优先级如下：

1. 显式 full 或 Daily 请求立即返回 `full`。
2. Catalog definition file 发生变化时返回 `full`。
3. 任一变更命中 `forced_full_paths` 时返回 `full`。
4. 路径存在 dependency owner 时选择对应 operator；即使它同时匹配 ignore pattern，也不能被 ignore 覆盖。
5. 没有 owner 但命中 `ignored_paths` 时忽略该路径。
6. 其余没有 owner 的路径一律返回 `full`。
7. 至少选择一个 operator，且其他路径均被明确忽略时返回 `selected`；只有无变更或全部路径均被忽略时才能返回 `skip`。

因此：只有 wrapper 变化时为 `skip`；wrapper 与已归属 operator 代码同时变化时为
`selected`；wrapper 与未知代码同时变化时为 `full`。selector 始终使用完整 Catalog
判定 ownership；operator filter 只作用于最终测试集。被 filter 排除的 owner 会得到
`filtered_operators`，不会被误判为 unowned。delete 使用 target graph，add 使用
head/worktree graph；rename/copy 的 old/new 两侧都会分类。
