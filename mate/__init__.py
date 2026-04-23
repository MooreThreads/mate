from importlib import import_module


def _load_version() -> str:
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("mate")
    except Exception:
        try:
            from ._build_meta import __version__ as build_version

            return build_version
        except Exception:
            return "unknown"


__version__ = _load_version()

_LAZY_SUBMODULES = {
    "aot",
    "api_logging",
    "deep_gemm",
    "flashinfer",
    "flashmla",
    "gdn_decode",
    "gdn_prefill",
    "gemm",
    "jit",
    "mha_interface",
    "sage_attention_interface",
    "testing",
    "version",
}

_LAZY_ATTR_MODULES = {
    "flash_attn_varlen_func": "mha_interface",
    "flash_attn_with_kvcache": "mha_interface",
    "get_scheduler_metadata": "mha_interface",
    "get_mla_metadata": "flashmla",
    "flash_mla_with_kvcache": "flashmla",
    "gated_delta_rule_decode": "gdn_decode",
    "gdn_prefill": "gdn_prefill",
    "mate_api": "api_logging",
    "moe_fused_gate": "moe_fused_gate",
    "sage_attn_quantized": "sage_attention_interface",
    "sage_attn_quantized_with_kvcache": "sage_attention_interface",
}

__all__ = [
    "aot",
    "api_logging",
    "deep_gemm",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "flash_mla_with_kvcache",
    "flashinfer",
    "flashmla",
    "gated_delta_rule_decode",
    "gdn_decode",
    "gdn_prefill",
    "gemm",
    "get_mla_metadata",
    "get_scheduler_metadata",
    "jit",
    "mate_api",
    "mha_interface",
    "moe_fused_gate",
    "sage_attn_quantized",
    "sage_attn_quantized_with_kvcache",
    "sage_attention_interface",
    "testing",
    "version",
    "__version__",
]


def _load_relative_module(name):
    module = import_module(f".{name}", __name__)
    return module


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = _load_relative_module(name)
        globals()[name] = module
        return module
    if name in _LAZY_ATTR_MODULES:
        module = _load_relative_module(_LAZY_ATTR_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
