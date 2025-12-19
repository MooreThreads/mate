import glob
import os
import sys
from typing import List

from setuptools import Extension, find_packages, setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

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
mate_libraries.extend(["mudnn"])

# flags
mcc_flags = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-std=c++17",
    "-fno-strict-aliasing",
    "-ffast-math",
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
        "csrc/moe_gemm_8bit.mu",
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
        "csrc/mla_pybind.mu",
        "csrc/fmha_pybind.mu",
    ]
)

include_path = [
    os.path.abspath("3rdparty/mutlass/include"),
    os.path.abspath("3rdparty/mutlass/tools/util/include"),
    os.path.abspath("3rdparty/mutlass/experimental/fmha"),
    os.path.abspath("include/"),
]

# fmha
instantiations_dir = os.path.join("csrc", "instantiations")
# instantiations_dir = os.path.abspath("csrc/instantiations")
if not os.path.exists(instantiations_dir):
    current_dir = os.getcwd()
    try:
        os.chdir(os.path.join(current_dir, "csrc"))
        os.system(f"{sys.executable} -m generate_kernels")
    finally:
        os.chdir(current_dir)

instantiation_files = glob.glob(os.path.join(instantiations_dir, "*.mu"))
instantiation_files = [f.replace(os.sep, "/") for f in instantiation_files]

mate_sources.extend(instantiation_files)
# include_path.append(os.path.abspath("csrc"))

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


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            max_num_jobs_cores = max(1, os.cpu_count() or 1)
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)
        super().__init__(*args, **kwargs)


cmdclass = {}

cmdclass["build_ext"] = NinjaBuildExtension

setup(
    name="mate",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
