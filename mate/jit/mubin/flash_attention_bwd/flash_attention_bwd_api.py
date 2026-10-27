import functools

from mate.artifacts import ensure_mubin_module_artifacts

from ..common import get_asm_dtype_from_torch_dtype
from ..launcher import get_mubin_launch_function, render_mubin_launcher
from .dispatch import FlashAttentionBwdMubinDispatcher


@functools.cache
def _get_flash_attention_bwd_mubin_runtime():
    directory = ensure_mubin_module_artifacts("flash_attention_bwd")
    return FlashAttentionBwdMubinDispatcher(directory / "kernel_map.json")


@functools.cache
def get_flash_attention_bwd_mubin_launch_plan(
    dtype, is_causal, headdim_qk, is_gqa, *, is_persistence=False
):
    dispatcher = _get_flash_attention_bwd_mubin_runtime()
    asm_id = dispatcher.get_asm_id(
        dtype, is_causal, headdim_qk, is_gqa, is_persistence=is_persistence
    )
    entry = dispatcher.resolve_kernel_entry(asm_id)
    path = dispatcher.resolve_kernel_path(asm_id)
    launch = get_mubin_launch_function(
        "flash_attention_bwd",
        entry.kernel_name,
        render_mubin_launcher(
            "flash_attention_bwd",
            func_name=entry.kernel_name,
            spec=asm_id,
            asm_dtype=get_asm_dtype_from_torch_dtype(dtype),
        ),
    )
    return str(path), launch
