import tvm_ffi.cpp  # noqa: F401

from jit.attention.fmha import gen_fmha_aot
from jit import env as jit_env

from jinja2 import Template
from pathlib import Path
import argparse
from typing import Optional

from itertools import product  # noqa: F401

import json  # noqa: F401
import hashlib  # noqa: F401

TVM_HEADER = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

"""

EXPORT_FUNC = Template("""
\n
TVM_FFI_DLL_EXPORT_TYPED_FUNC({{func_name}}, {{func_name}});
""")


def compile_and_package_aot(
    args: argparse.Namespace,
    output_dir: Path,
    build_dir: Path,
    project_root: Path,
) -> None:
    # Override JIT ENV
    jit_env.MATE_INCLUDE_DIR = project_root / "include"
    jit_env.MUTLASS_INCLUDE_DIR = project_root / "3rdparty" / "mutlass" / "include"
    jit_env.MATE_GEN_SRC_DIR = build_dir
    jit_env.MATE_AOT_DIR = output_dir
    jit_env.MATE_AOT_DIR.mkdir(parents=True, exist_ok=True)
    jit_env.MATE_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)

    num_fmha_kernels = gen_fmha_aot(dry_run=args.dry_run)
    print(f"Total FMHA AOT instances: {num_fmha_kernels}")


def setup_aot_levels(args) -> None:
    aot_config_path = Path(__file__).parent / "jit" / "aot-levels.json"
    configs = {
        "attention": {
            "flash_mla": {key: 1 for key in ["metadata", "fwd", "combine"]},
            "fmha": {key: 1 for key in ["metadata", "fwd", "combine"]},
        }
    }
    configs["attention"]["fmha"]["metadata"] = args.fmha_metadata_aot_level
    configs["attention"]["fmha"]["fwd"] = args.fmha_fwd_aot_level
    configs["attention"]["fmha"]["combine"] = args.fmha_combine_aot_level
    with open(aot_config_path.resolve(), "w") as f:
        json.dump(configs, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build all modules"
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory")
    parser.add_argument("--build-dir", type=Path, help="Build directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count the number of AOT instances without building",
    )

    def validate_aot_level(value):
        ivalue = int(value)
        if ivalue not in [0, 1, 2]:
            raise argparse.ArgumentTypeError("AOT level must be one of: 0, 1, 2")
        return ivalue

    parser.add_argument(
        "--fmha-aot-level",
        type=validate_aot_level,
        help="FMHA AOT level for all kernels (metadata, fwd, combine). "
        "If set, overrides individual kernel levels.",
        default=None,
    )
    parser.add_argument(
        "--fmha-metadata-aot-level",
        type=validate_aot_level,
        help="FMHA metadata AOT level",
        default=0,
    )
    parser.add_argument(
        "--fmha-fwd-aot-level",
        type=validate_aot_level,
        help="FMHA FWD AOT level",
        default=0,
    )
    parser.add_argument(
        "--fmha-combine-aot-level",
        type=validate_aot_level,
        help="FMHA combine AOT level",
        default=0,
    )

    args = parser.parse_args()

    # If --fmha-aot-level is set, use it for all three sub-levels
    if args.fmha_aot_level is not None:
        args.fmha_metadata_aot_level = args.fmha_aot_level
        args.fmha_fwd_aot_level = args.fmha_aot_level
        args.fmha_combine_aot_level = args.fmha_aot_level

    project_root = Path(__file__).resolve().parents[1]

    out_dir: Optional[Path] = None

    # Override with command line arguments
    if args.out_dir:
        out_dir = Path(args.out_dir)
    if args.build_dir:
        build_dir = Path(args.build_dir)

    setup_aot_levels(args)
    compile_and_package_aot(args, out_dir, build_dir, project_root)


if __name__ == "__main__":
    main()
