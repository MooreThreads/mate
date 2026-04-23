from __future__ import annotations

import functools
import os
import subprocess
import sys
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import List, Optional, Sequence

import tvm_ffi.cpp.extension as tvm_ffi_ext
from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path, find_libtvm_ffi

from . import env as jit_env


def _escape_ninja_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", "$:")


@functools.cache
def _musa_header_version(musa_home: Path) -> int:
    musa_header = musa_home / "include" / "musa.h"
    if not musa_header.exists():
        return 0
    for line in musa_header.read_text().splitlines():
        if "MUSA_VERSION" not in line or "define" not in line:
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "MUSA_VERSION":
            try:
                return int(parts[2])
            except ValueError:
                return 0
    return 0


@functools.cache
def get_mudnn_ldflags() -> tuple[str, ...]:
    musa_home = Path(tvm_ffi_ext._find_musa_home())
    mudnn_lib = "mudnncxx" if _musa_header_version(musa_home) >= 50100 else "mudnn"
    return (f"-l{mudnn_lib}",)


@functools.cache
def get_mudnn_include_dirs() -> tuple[str, ...]:
    musa_home = Path(tvm_ffi_ext._find_musa_home())
    if _musa_header_version(musa_home) < 50100:
        return ()
    mudnncxx_include_dir = musa_home / "include" / "mudnncxx"
    if not mudnncxx_include_dir.exists():
        return ()
    return (str(mudnncxx_include_dir.resolve()),)


def _resolve_include_paths(
    extra_include_dirs: Optional[Sequence[Path]],
) -> List[str]:
    include_paths = [
        find_include_path(),
        find_dlpack_include_path(),
    ]
    include_paths.extend(get_mudnn_include_dirs())
    if extra_include_dirs is not None:
        include_paths.extend(str(path.resolve()) for path in extra_include_dirs)
    return list(dict.fromkeys(include_paths))


def _flatten_flags(*flag_groups: object) -> List[str]:
    flattened: List[str] = []

    def _append(value: object) -> None:
        if value is None:
            return
        if isinstance(value, str):
            flattened.append(value)
            return
        if isinstance(value, SequenceABC):
            for item in value:
                _append(item)
            return
        flattened.append(str(value))

    for group in flag_groups:
        _append(group)
    return flattened


def generate_ninja_build_for_op(
    name: str,
    sources: Sequence[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
) -> str:
    if tvm_ffi_ext.IS_WINDOWS:
        raise RuntimeError("MATE JIT ninja build is only supported on Unix platforms.")

    tvm_ffi_lib = Path(find_libtvm_ffi())
    tvm_ffi_lib_dir = tvm_ffi_lib.parent
    musa_home = Path(tvm_ffi_ext._find_musa_home())

    cflags = _flatten_flags(["-std=c++17", "-fPIC", "-O2"], extra_cflags)
    cuda_cflags = _flatten_flags(
        ["-fPIC", "-std=c++17", "-O2"],
        jit_env.MATE_MUSA_TARGET_FLAGS,
        extra_cuda_cflags,
    )
    ldflags = _flatten_flags(
        [
            "-shared",
            f"-L{tvm_ffi_lib_dir}",
            "-ltvm_ffi",
            f"-L{musa_home / 'lib'}",
            "-lmusa",
            "-lmusart",
        ],
        extra_ldflags,
    )

    for include_path in _resolve_include_paths(extra_include_dirs):
        escaped = include_path.replace(":", "$:")
        cflags.append(f"-I{escaped}")
        cuda_cflags.append(f"-I{escaped}")

    lines = [
        "ninja_required_version = 1.3",
        f"cxx = {os.environ.get('CXX', 'c++')}",
        f"mcc = {(musa_home / 'bin' / 'mcc').as_posix()}",
        f"cflags = {' '.join(cflags)}",
        f"cuda_cflags = {' '.join(cuda_cflags)}",
        f"ldflags = {' '.join(ldflags)}",
        "",
        "rule compile",
        "  depfile = $out.d",
        "  deps = gcc",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out",
        "",
        "rule compile_cuda",
        "  depfile = $out.d",
        "  deps = gcc",
        "  command = $mcc -MD -MF $out.d $cuda_cflags -c $in -o $out",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]

    output_dir = jit_env.MATE_JIT_DIR / name

    objects = []
    for source in sources:
        is_cuda = source.suffix in {".mu", ".cu"}
        object_suffix = ".cuda.o" if is_cuda else ".o"
        rule = "compile_cuda" if is_cuda else "compile"
        object_path = (output_dir / source.with_suffix(object_suffix).name).resolve()
        objects.append(_escape_ninja_path(object_path))
        lines.append(
            f"build {objects[-1]}: {rule} {_escape_ninja_path(source.resolve())}"
        )

    output_so = _escape_ninja_path((output_dir / f"{name}.so").resolve())
    lines.extend(
        [
            "",
            f"build {output_so}: link {' '.join(objects)}",
            f"default {output_so}",
            "",
        ]
    )

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers()
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = "Ninja build failed."
        if exc.output:
            message += " Ninja output:\n" + exc.output
        raise RuntimeError(message) from exc
