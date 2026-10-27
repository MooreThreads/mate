#!/usr/bin/env python3
"""MATE test catalog and incremental-test resolver entry point."""

from __future__ import annotations

from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, Sequence


# Put repo into sys.path
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from mate.testing.resolver import (  # noqa: E402
    OperationCatalog,
    SelectionPolicy,
    TestRunRequest,
)
from mate.testing.resolver.cli import main as resolver_main  # noqa: E402


TEST_CATALOG = OperationCatalog(
    definition_files=(Path(__file__).resolve().relative_to(REPOSITORY),),
)

TEST_SELECTION_POLICY = SelectionPolicy(
    ignored_paths=("wrappers/**",),
)

TEST_CATALOG.register(
    "mate.ops.attention.dsa",
    interfaces=(
        "mate/flashmla.py",
        "mate/sparse_mla_interface.py",
    ),
    tests=("tests/test_sparse_mla.py",),
)

TEST_CATALOG.register(
    "mate.ops.attention.fa",
    interfaces=("mate/mha_interface.py",),
    tests=(
        "tests/test_fmha.py",
        "tests/test_fmha_bwd.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.attention.mla",
    interfaces=("mate/mha_interface.py", "mate/flashmla.py"),
    tests=("tests/test_mla.py",),
)

# NOTE: We probably can remove test_msa_interface.py
TEST_CATALOG.register(
    "mate.ops.attention.msa",
    interfaces=("mate/msa_interface.py",),
    tests=(
        "tests/test_msa_fwd.py",
        "tests/test_msa_interface.py",
    ),
)

TEST_CATALOG.register(
    "mate.qsa_kernels",
    interfaces=("mate/qsa_interface.py",),
    tests=("tests/test_qsa.py",),
)

TEST_CATALOG.register(
    "mate.ops.attention.sage",
    interfaces=("mate/sage_attention_interface.py",),
    tests=("tests/test_sage_attention_interface.py",),
)

TEST_CATALOG.register(
    "mate.ops.attention.kda",
    interfaces=("mate/kda.py",),
    tests=(
        "tests/test_kda.py",
        "tests/test_kda_decode.py",
        "tests/test_kda_chunk_states.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.attention.gdn",
    interfaces=(
        "mate/gdn_decode.py",
        "mate/gdn_prefill.py",
    ),
    tests=(
        "tests/test_gdn_decode.py",
        "tests/test_gdn_mtp.py",
        "tests/test_gdn_prefill.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.gemm.gemm",
    interfaces=("mate/gemm.py",),
    tests=("tests/test_gemm.py",),
)

TEST_CATALOG.register(
    "mate.ops.mamba",
    interfaces=("mate/mamba.py",),
    tests=("tests/test_mamba.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.deep_gemm",
    interfaces=("mate/deep_gemm.py",),
    tests=("tests/test_deep_gemm.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.router.hash_topk",
    interfaces=("mate/hash_topk.py",),
    tests=("tests/test_hash_topk.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.gating.moe_fused_gate",
    interfaces=("mate/moe_fused_gate.py",),
    tests=("tests/test_fused_moe_gate.py",),
)

TEST_CATALOG.register(
    "mate.ops.moe.mega_moe",
    interfaces=(
        "mate/mega_moe.py",
        "mate/jit/mega_moe.py",
        "mate/jit/mega_moe_prefill.py",
    ),
    tests=(
        "tests/test_mega_moe_decode.py",
        "tests/test_mega_moe_prefill.py",
    ),
)

TEST_CATALOG.register(
    "mate.ops.norm",
    interfaces=("mate/norm/__init__.py",),
    tests=("tests/test_norm.py",),
)

TEST_CATALOG.register(
    "mate.ops.hyper_connect",
    interfaces=("mate/hyperconnection.py",),
    tests=("tests/test_hyperconnection.py",),
)


def _mudmp_files(directory: Path) -> dict[Path, tuple[int, int, int]]:
    files = {}
    for path in directory.glob("*.mudmp"):
        try:
            info = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            files[path] = (info.st_ino, info.st_mtime_ns, info.st_size)
    return files


def _run_with_mudmp_output(
    request: TestRunRequest,
    command: Sequence[str],
    *,
    environment: Mapping[str, str | None],
) -> int:
    before = _mudmp_files(request.repository)
    status = request.run(command, environment=environment)
    if status:
        after = _mudmp_files(request.repository)
        for path in sorted(after):
            if after[path] == before.get(path):
                continue
            print(f"\n=== mudmp: {path.name} ===", flush=True)
            subprocess.run(("cat", "--", str(path)), check=False)
            print(f"\n=== end mudmp: {path.name} ===", flush=True)
    return status


def run_selected_tests(request: TestRunRequest) -> int:
    """Preflight marked selected tests, then run all selections for correctness."""
    tests = request.selection.selected_tests
    base_environment = {
        "MATE_PYTEST_SHARD_TOTAL": "1",
        "MATE_PYTEST_SHARD_INDEX": "0",
    }
    status = _run_with_mudmp_output(
        request,
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
        environment=base_environment,
    )
    if status:
        return status

    return _run_with_mudmp_output(
        request,
        (
            sys.executable,
            "-m",
            "pytest",
            "-v",
            *tests,
        ),
        environment={**base_environment, "MATE_DRY_RUN": None},
    )


if __name__ == "__main__":
    raise SystemExit(
        resolver_main(
            TEST_CATALOG,
            runner=run_selected_tests,
            policy=TEST_SELECTION_POLICY,
        )
    )
