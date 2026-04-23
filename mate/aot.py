import argparse
from contextlib import contextmanager
from pathlib import Path

import tvm_ffi.cpp  # noqa: F401

from mate.jit import build_jit_specs, copy_built_kernels, env as jit_env
from mate.jit.attention.fmha import gen_fmha_aot
from mate.jit.deep_gemm_attention import gen_deep_gemm_attention_aot
from mate.jit.gemm_ops import gen_gemm_ops_aot
from mate.jit.mla_ops import gen_mla_ops_aot
from mate.jit.moe_fused_gate import gen_moe_fused_gate_aot
from mate.jit.sage_attention import gen_sage_attention_aot


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true"}:
        return True
    if lowered in {"0", "false"}:
        return False
    raise argparse.ArgumentTypeError("Boolean flag must be one of: true/false/1/0")


def validate_aot_level(value: str) -> int:
    ivalue = int(value)
    if ivalue not in [0, 1, 2]:
        raise argparse.ArgumentTypeError("AOT level must be one of: 0, 1, 2")
    return ivalue


@contextmanager
def override_jit_env(
    *,
    include_dir: Path,
    csrc_dir: Path,
    mutlass_include_dir: Path,
    gen_src_dir: Path,
    jit_dir: Path,
    aot_dir: Path,
):
    previous = {
        "MATE_INCLUDE_DIR": jit_env.MATE_INCLUDE_DIR,
        "MATE_CSRC_DIR": jit_env.MATE_CSRC_DIR,
        "MATE_TEMPLATE_DIR": jit_env.MATE_TEMPLATE_DIR,
        "MUTLASS_INCLUDE_DIR": jit_env.MUTLASS_INCLUDE_DIR,
        "MATE_GEN_SRC_DIR": jit_env.MATE_GEN_SRC_DIR,
        "MATE_JIT_DIR": jit_env.MATE_JIT_DIR,
        "MATE_AOT_DIR": jit_env.MATE_AOT_DIR,
    }
    jit_env.MATE_INCLUDE_DIR = include_dir
    jit_env.MATE_CSRC_DIR = csrc_dir
    jit_env.MATE_TEMPLATE_DIR = csrc_dir / "templates"
    jit_env.MUTLASS_INCLUDE_DIR = mutlass_include_dir
    jit_env.MATE_GEN_SRC_DIR = gen_src_dir
    jit_env.MATE_JIT_DIR = jit_dir
    jit_env.MATE_AOT_DIR = aot_dir
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(jit_env, name, value)


def compile_and_package_aot(
    output_dir: Path,
    build_dir: Path,
    project_root: Path,
    *,
    attention_aot_level: int = 1,
    add_gemm: bool = True,
    add_moe: bool = True,
    jobs: int | None = None,
    dry_run: bool = False,
):
    with override_jit_env(
        include_dir=project_root / "include",
        csrc_dir=project_root / "csrc",
        mutlass_include_dir=project_root / "3rdparty" / "mutlass" / "include",
        gen_src_dir=build_dir / "generated",
        jit_dir=build_dir / "cached_ops",
        aot_dir=output_dir,
    ):
        jit_env.MATE_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)
        jit_env.MATE_JIT_DIR.mkdir(parents=True, exist_ok=True)

        specs = []
        if attention_aot_level > 0:
            specs.extend(gen_fmha_aot(attention_aot_level))
            specs.extend(gen_mla_ops_aot())
            specs.extend(gen_sage_attention_aot())
        if add_gemm:
            specs.extend(gen_gemm_ops_aot())
            specs.extend(gen_deep_gemm_attention_aot())
        if add_moe:
            specs.extend(gen_moe_fused_gate_aot())

        if dry_run:
            return specs

        build_jit_specs(specs, jobs=jobs, verbose=True, skip_prebuilt=False)
        copy_built_kernels(specs, output_dir)
        return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build JIT-managed MATE modules"
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory")
    parser.add_argument("--build-dir", type=Path, help="Build directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the spec list without compiling",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Maximum parallel ninja jobs",
    )
    parser.add_argument(
        "--attention-aot-level",
        type=validate_aot_level,
        default=1,
        help="Attention AOT level for the attention family "
        "(FMHA, flash attention ops, and MLA). "
        "Use 0 to disable attention AOT entirely.",
    )
    parser.add_argument(
        "--add-gemm",
        type=parse_bool,
        default=True,
        help="Whether to include the GEMM family in the AOT build "
        "(gemm ops and deep_gemm attention).",
    )
    parser.add_argument(
        "--add-moe",
        type=parse_bool,
        default=True,
        help="Whether to include the MOE family in the AOT build.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.out_dir or (project_root / "aot-ops")
    build_dir = args.build_dir or (project_root / "build" / "aot")

    specs = compile_and_package_aot(
        output_dir=output_dir,
        build_dir=build_dir,
        project_root=project_root,
        attention_aot_level=args.attention_aot_level,
        add_gemm=args.add_gemm,
        add_moe=args.add_moe,
        jobs=args.jobs,
        dry_run=args.dry_run,
    )
    print(f"Resolved {len(specs)} JIT specs")
    if not args.dry_run:
        print(f"AOT kernels saved to: {output_dir}")


if __name__ == "__main__":
    main()
