import os
import pathlib
import functools

from packaging.version import Version
# from version import __version__ as mate_version

MATE_VERSION = Version("0.1.0")


def match_jit_aot(jit, aot_build_path):
    if os.environ.get("MATE_AOT_BUILD", "0") == "1":
        return aot_build_path
    else:
        return jit


MATE_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("MATE_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

MATE_CACHE_DIR: pathlib.Path = MATE_BASE_DIR / ".cache" / "mate"

_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]

MATE_WORKSPACE_DIR: pathlib.Path = MATE_CACHE_DIR / MATE_VERSION.base_version
MATE_GEN_SRC_DIR: pathlib.Path = MATE_WORKSPACE_DIR / "generated"
MATE_DATA: pathlib.Path = _package_root / "data"
MATE_CSRC_DIR: pathlib.Path = match_jit_aot(
    MATE_DATA / "csrc", _package_root.parent / "csrc"
)
MATE_TEMPLATE_DIR: pathlib.Path = MATE_CSRC_DIR / "templates"
MATE_INCLUDE_DIR: pathlib.Path = MATE_DATA / "include"
MUTLASS_INCLUDE_DIR: pathlib.Path = MATE_DATA / "mutlass" / "include"
MATE_AOT_DIR: pathlib.Path = MATE_DATA / "aot"


def check_force_jit():
    """
    Decorator to check if JIT should be enforced.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get("MATE_FORCE_JIT", "0") == "1":
                return None
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def check_no_aot_build():
    """
    Decorator to check if AOT build should be enforced.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get("MATE_AOT_BUILD", "0") == "0":
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    return decorator
