from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("flash_attn_3")
except PackageNotFoundError:
    __version__ = "unknown"
