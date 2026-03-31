import torch
from jinja2 import Template
from . import env as jit_env

import tvm_ffi


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


dtype_torch2mutlass_map = {
    torch.float16: "mutlass::half_t",
    torch.bfloat16: "mutlass::bfloat16_t",
}


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


@jit_env.check_force_jit()
@jit_env.check_no_aot_build()
def update_aot_mod(configs, encoding_fn, mod_path, mod_cache):
    mod = tvm_ffi.load_module(mod_path)
    for config in configs:
        dispatch_name = encoding_fn(tuple(config.items()))
        mod_cache[dispatch_name] = mod
