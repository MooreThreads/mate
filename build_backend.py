import os
import shutil
from pathlib import Path

from setuptools import build_meta as orig
from build_utils import build_version_string

_root = Path(__file__).parent.resolve()

_data_dir = _root / "mate" / "data"
_aot_package_dir = _data_dir / "aot"


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = _root / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            base_version = f.read().strip()
    else:
        base_version = "0.0.0"

    # Get optional dev suffix from environment
    dev_suffix = os.environ.get("MATE_DEV_RELEASE_SUFFIX", "")

    # Get optional local version from environment
    local_version = os.environ.get("MATE_LOCAL_VERSION")

    # Build full version string using build_utils
    version, git_version = build_version_string(
        base_version=base_version,
        cwd=_root,
        dev_suffix=dev_suffix,
        local_version=local_version,
    )

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "mate"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for mate package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def _prepare_build():
    pass


def _create_data_dir(use_symlinks=True):
    _data_dir.mkdir(parents=True, exist_ok=True)

    def ln(source: str, target: str) -> None:
        src = _root / source
        dst = _data_dir / target
        if dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if use_symlinks:
            dst.symlink_to(src, target_is_directory=True)
        else:
            # For wheel/sdist, copy actual files instead of symlinks
            if src.exists():
                shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)

    ln("3rdparty/mutlass", "mutlass")
    ln("csrc", "csrc")
    ln("include", "include")
    ln("aot-lib", "aot")


def _prepare_for_wheel():
    # For wheel, copy actual files instead of symlinks so they are included in the wheel
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)
    _prepare_build()

    # # Copy license files from licenses/ to root to avoid nested path in wheel
    # licenses_dir = _root / "licenses"
    # if licenses_dir.exists():
    #     for license_file in licenses_dir.glob("*.txt"):
    #         shutil.copy2(
    #             license_file,
    #             _root / f"LICENSE.{license_file.stem.removeprefix('LICENSE.')}.txt",
    #         )


def _prepare_for_editable():
    # For editable install, use symlinks so changes are reflected immediately
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=True)
    _prepare_build()


def _prepare_for_sdist():
    # For sdist, copy actual files instead of symlinks so they are included in the tarball
    if _data_dir.exists():
        shutil.rmtree(_data_dir)
    _create_data_dir(use_symlinks=False)
    _prepare_build()


def get_requires_for_build_wheel(config_settings=None):
    _prepare_for_wheel()
    return []


def get_requires_for_build_sdist(config_settings=None):
    _prepare_for_sdist()
    return []


def get_requires_for_build_editable(config_settings=None):
    _prepare_for_editable()
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare_for_wheel()
    return orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare_for_editable()
    return orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_editable()
    return orig.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _prepare_for_sdist()
    return orig.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare_for_wheel()
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)
