import glob
import os
from typing import List

from setuptools import Extension, find_packages, setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

def get_musa_version(musa_path: str) -> int:
    """Get MUSA SDK version from musa.h header file.
    
    Returns:
        Version as integer (e.g., 50100 for 5.1.0), or 0 if not found.
    """
    musa_header = os.path.join(musa_path, "include", "musa.h")
    if not os.path.exists(musa_header):
        return 0
    
    try:
        with open(musa_header, "r") as f:
            for line in f:
                if "MUSA_VERSION" in line and "define" in line:
                    # Parse: #define MUSA_VERSION 50100
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == "MUSA_VERSION":
                        try:
                            return int(parts[2])
                        except ValueError:
                            continue
    except Exception:
        pass
    return 0


mate_sources = []
mate_libraries = []
mate_link_path = []
ext_modules: List[Extension] = []

# setup musa path
musa_path_default = "/usr/local/musa"
musa_path = os.environ.get("MUSA_PATH", musa_path_default)

if not os.path.exists(musa_path):
    raise EnvironmentError(f"MUSA_PATH {musa_path} does not exist!")

musa_lib_path = os.path.join(musa_path, "lib")
if not os.path.exists(musa_lib_path):
    raise EnvironmentError(f"MUSA lib path {musa_lib_path} does not exist!")

# setup mudnn
mate_link_path.extend([musa_lib_path])

# Check MUSA SDK version and set mudnn libraries accordingly
musa_version = get_musa_version(musa_path)
print(f"Detected MUSA SDK version: {musa_version}")

if musa_version >= 50100:
    # MUSA SDK >= 5.1.0: use new mudnncxx library
    print("Using muDNN C++ library (mudnncxx) for MUSA SDK >= 5.1.0")
    mate_libraries.extend(["mudnncxx"])
else:
    # MUSA SDK < 5.1.0: use legacy mudnn library
    print("Using legacy muDNN library for MUSA SDK < 5.1.0")
    mate_libraries.extend(["mudnn"])

# flags
mcc_flags = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-std=c++17",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

cxx_flags = [
    "-O3",
    "-Wno-switch-bool",
    "-DPy_LIMITED_API=0x03090000",
]

# common source
mate_sources.extend(["csrc/torch_bindings.cpp"])

# gemm source
mate_sources.extend(
    [
        "csrc/batch_gemm_fp8.mu",
        "csrc/gemm_fp8_groupwise.mu",
        "csrc/moe_gemm_asm.mu",
        "csrc/deepgemm_attention.mu",
    ]
)
mp31_gemm_mubin_src_dir = os.path.join("csrc", "mubin", "mp31", "gemm")
mp31_gemm_mubin = glob.glob(os.path.join(mp31_gemm_mubin_src_dir, "*.cpp"))
mate_sources.extend(mp31_gemm_mubin)

# FA mubin source
mate_sources.extend(
    [
        "csrc/flash_atten_asm.mu",
        "csrc/flash_atten_bwd.mu",
    ]
)
# flash atten mubin source
mp31_fa_mubin_src_dir = os.path.join("csrc", "mubin", "mp31", "flash_atten")
mp31_fa_mubin = glob.glob(os.path.join(mp31_fa_mubin_src_dir, "*.cpp"))
mate_sources.extend(mp31_fa_mubin)

# mla
mate_sources.extend(
    [
        "csrc/attention_scheduler.mu",
        "csrc/attention_combine.mu",
        "csrc/flash_mla_asm.mu",
        "csrc/mla_pybind.mu",
        "csrc/moe_fused_gate.mu",
    ]
)
# flash mla mubin source
mp31_mla_mubin_src_dir = os.path.join("csrc", "mubin", "mp31", "flash_mla")
mp31_mla_mubin = glob.glob(os.path.join(mp31_mla_mubin_src_dir, "*.cpp"))
mate_sources.extend(mp31_mla_mubin)

include_path = [
    os.path.abspath("3rdparty/mutlass/include"),
    os.path.abspath("3rdparty/mutlass/tools/util/include"),
    os.path.abspath("3rdparty/mutlass/experimental/fmha"),
    os.path.abspath("include/"),
]

if musa_version >= 50100:
    include_path.append(os.path.join(musa_path, "include", "mudnncxx"))

ext_modules.append(
    MUSAExtension(
        name="mate._C",
        sources=mate_sources,
        include_dirs=include_path,
        library_dirs=mate_link_path,
        libraries=mate_libraries,
        extra_compile_args={
            "cxx": cxx_flags,
            "mcc": mcc_flags,
        },
    )
)


def remove_march_native(flags):
    """Remove -march=native from compiler flags to avoid architecture mismatch."""
    return [f for f in flags if f != "-march=native" and not f.startswith("-march=native=")]


def clean_env_flags():
    """Clean -march=native from environment variables that may affect compilation."""
    env_vars = ["CFLAGS", "CXXFLAGS", "CPPFLAGS", "MCCFLAGS"]
    for var in env_vars:
        if var in os.environ:
            flags = os.environ[var].split()
            cleaned = remove_march_native(flags)
            os.environ[var] = " ".join(cleaned)


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # Clean -march=native from environment variables before building
        clean_env_flags()
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            max_num_jobs_cores = max(1, os.cpu_count() or 1)
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        # Remove -march=native from compiler flags
        for ext in self.extensions:
            if hasattr(ext, 'extra_compile_args'):
                if 'cxx' in ext.extra_compile_args:
                    ext.extra_compile_args['cxx'] = remove_march_native(ext.extra_compile_args['cxx'])
                if 'mcc' in ext.extra_compile_args:
                    ext.extra_compile_args['mcc'] = remove_march_native(ext.extra_compile_args['mcc'])
        super().build_extensions()


cmdclass = {}

cmdclass["build_ext"] = NinjaBuildExtension

setup(
    name="mate",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
