"""TileLang MUSA QSA kernel stages and shared compiler configuration."""


def _pass_configs(tilelang, *, enable_tme=False, disable_tma_lower=None):
    if tilelang is None:
        raise ModuleNotFoundError(
            "QSA kernels require a working TileLang-MUSA installation.", name="tilelang"
        )
    configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    }
    if disable_tma_lower is not None:
        configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = disable_tma_lower
    if enable_tme:
        configs[tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH] = True
    return configs
