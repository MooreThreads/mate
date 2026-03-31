#!/usr/bin/env python3
"""
MATE CLI - Command Line Interface for MATE (MUSA AI Tensor Engine)

Usage:
    mate --help
    mate show-config
    mate replay --dir mate_dumps/

Requirements:
    pip install click>=8.0.0
"""

import os
import sys
import json
from pathlib import Path

import click


# Version - try multiple sources
def _load_build_meta():
    """Load version info from _build_meta.py directly to avoid importing mate package."""
    import importlib.util

    build_meta_path = Path(__file__).parent / "_build_meta.py"
    if build_meta_path.exists():
        spec = importlib.util.spec_from_file_location("_build_meta", build_meta_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "__version__", "unknown"), getattr(
                module, "__git_version__", "unknown"
            )
    return "unknown", "unknown"


__version__, __git_version__ = _load_build_meta()

# Fallback to installed package if direct load fails
if __version__ == "unknown":
    try:
        from importlib.metadata import version

        __version__ = version("mate")
    except Exception:
        pass


def _get_replay_functions():
    """Load replay functions from api_logging module.

    Returns:
        (replay_from_dump, replay_sequence) on success, or (None, error_message) on failure.
    """
    try:
        from mate.api_logging import replay_from_dump, replay_sequence

        return replay_from_dump, replay_sequence
    except Exception as e:
        return None, str(e)


# Environment variables documentation
ENV_VARS = {
    "MATE_LOGLEVEL": "Logging level (0, 1, 3, 5, 10)",
    "MATE_LOGDEST": "Log destination (stdout, stderr, or file path)",
    "MATE_DUMP_DIR": "Directory for Level 10 tensor dumps",
    "MATE_DUMP_MAX_SIZE_GB": "Max total dump size in GB",
    "MATE_DUMP_MAX_COUNT": "Max number of dumps",
    "MATE_DUMP_SAFETENSORS": "Use safetensors format (0/1)",
    "MATE_DUMP_INCLUDE": "Include patterns for dumping",
    "MATE_DUMP_EXCLUDE": "Exclude patterns for dumping",
    "TVM_FFI_MUSA_ARCH_LIST": "MUSA architecture list for JIT compilation",
    "MATE_AOT_BUILD": "AOT build mode flag",
}


def get_system_info():
    """Gather system information."""
    info = {
        "mate_version": __version__,
        "git_version": __git_version__,
        "python_version": sys.version.split()[0],
    }

    # PyTorch info
    try:
        import torch

        info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "Not installed"

    # MUSA info
    try:
        import torch_musa

        info["torch_musa_version"] = torch_musa.__version__
        info["musa_available"] = torch_musa.is_available()
        if torch_musa.is_available():
            info["musa_device_count"] = torch_musa.device_count()
            info["musa_devices"] = [
                torch_musa.get_device_name(i) for i in range(torch_musa.device_count())
            ]
    except ImportError:
        info["torch_musa_version"] = "Not installed"
        info["musa_available"] = False

    return info


def print_header(text: str, fg: str = "yellow"):
    """Print a section header."""
    click.secho(f"\n=== {text} ===", fg=fg, bold=True)


def print_kv(
    key: str, value: str, key_color: str = "magenta", value_color: str = "cyan"
):
    """Print a key-value pair."""
    click.secho(f"{key}:", fg=key_color, nl=False)
    click.secho(f" {value}", fg=value_color)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="mate")
@click.pass_context
def cli(ctx):
    """MATE CLI - MUSA AI Tensor Engine Command Line Interface

    Examples:
        mate show-config              # Display configuration
        mate replay --dir dumps/      # Replay API calls
        mate env                      # Show environment variables
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("show-config")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def show_config(output_json: bool):
    """Display MATE configuration and system information."""
    info = get_system_info()

    if output_json:
        click.echo(json.dumps(info, indent=2))
        return

    # Version Info
    print_header("Version Information")
    print_kv("MATE version", info["mate_version"])
    print_kv("Git version", info.get("git_version", "unknown")[:16])
    print_kv("Python version", info["python_version"])
    print_kv("PyTorch version", info.get("torch_version", "N/A"))

    # MUSA Info
    print_header("MUSA Information")
    print_kv("torch_musa version", info.get("torch_musa_version", "N/A"))

    if info.get("musa_available"):
        click.secho("MUSA available: ", fg="magenta", nl=False)
        click.secho("Yes", fg="green")
        print_kv("Device count", str(info["musa_device_count"]))

        for i, name in enumerate(info.get("musa_devices", [])):
            click.secho(f"  Device {i}:", fg="white", nl=False)
            click.secho(f" {name}", fg="cyan")
    else:
        click.secho("MUSA available: ", fg="magenta", nl=False)
        click.secho("No", fg="red")

    # AOT Info
    print_header("AOT Libraries")
    try:
        import mate

        mate_path = Path(mate.__file__).parent
        aot_dir = mate_path / "data" / "aot"

        if aot_dir.exists():
            aot_files = list(aot_dir.glob("*.so"))
            print_kv("AOT directory", str(aot_dir))
            print_kv("AOT libraries", str(len(aot_files)))
            for f in aot_files[:5]:
                click.secho(f"  - {f.name}", fg="white")
            if len(aot_files) > 5:
                click.secho(f"  ... and {len(aot_files) - 5} more", fg="white")
        else:
            click.secho("AOT directory not found", fg="yellow")
    except Exception as e:
        click.secho(f"Error checking AOT: {e}", fg="red")

    # Installed packages
    print_header("Installed Packages")
    try:
        import importlib.metadata as metadata

        packages = ["mate", "torch", "torch_musa"]
        for pkg in packages:
            try:
                version = metadata.version(pkg)
                print_kv(pkg, version)
            except metadata.PackageNotFoundError:
                print_kv(pkg, "Not installed", value_color="red")
    except Exception as e:
        click.secho(f"Error checking packages: {e}", fg="yellow")


@cli.command("env")
def env_cmd():
    """Display MATE environment variables and their current values."""
    print_header("Environment Variables")

    max_key_len = max(len(k) for k in ENV_VARS.keys())

    for var, description in ENV_VARS.items():
        value = os.environ.get(var, "<not set>")
        display_value = value[:50] + "..." if len(value) > 50 else value
        value_color = "yellow" if value != "<not set>" else "red"

        click.secho(f"{var:<{max_key_len}} ", fg="cyan", nl=False)
        click.secho(f"{display_value:<53} ", fg=value_color, nl=False)
        click.secho(description, fg="white")


@cli.command("replay")
@click.option(
    "--dir",
    "dump_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing dump files",
)
@click.option(
    "--device", default="musa", help="Target device for replay (musa, cpu, musa:N)"
)
@click.option(
    "--run/--no-run", default=True, help="Actually execute functions (default: run)"
)
@click.option(
    "--compare/--no-compare", default=True, help="Compare outputs with expected values"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def replay_cmd(dump_dir: str, device: str, run: bool, compare: bool, verbose: bool):
    """Replay API calls from Level 10 dump directory.

    Examples:
        mate replay --dir mate_dumps/20260310_xxx_call0001
        mate replay --dir mate_dumps/ --verbose
        mate replay --dir dumps/ --device cpu --no-run
    """
    replay_from_dump, replay_sequence = _get_replay_functions()

    if replay_from_dump is None:
        error_msg = replay_sequence  # second return value holds the error string
        click.secho("❌ Failed to load MATE API logging module:", fg="red")
        click.secho(f"   {error_msg}", fg="yellow")
        click.secho(
            "   Ensure MATE is correctly installed: pip install mate", fg="yellow"
        )
        sys.exit(1)

    dump_path = Path(dump_dir)
    is_single_dump = (dump_path / "metadata.jsonl").exists()

    if is_single_dump:
        click.secho(f"Replaying single dump: {dump_dir}", fg="cyan")

        try:
            result = replay_from_dump(
                str(dump_path), device=device, run=run, compare_outputs=compare
            )

            if verbose:
                click.secho("\nDump Information:", fg="yellow")
                metadata = result.get("metadata", {})
                print_kv("Function", metadata.get("function_name", "N/A"))
                print_kv("Module", metadata.get("module", "N/A"))
                print_kv("Status", metadata.get("execution_status", "N/A"))

                click.secho("\nArguments:", fg="yellow")
                for i, arg in enumerate(result.get("args", [])):
                    if hasattr(arg, "shape"):
                        click.secho(
                            f"  arg[{i}]: {list(arg.shape)} {arg.dtype}", fg="white"
                        )
                    else:
                        click.secho(f"  arg[{i}]: {arg}", fg="white")

            if "error" in result:
                click.secho(f"❌ Error: {result['error']}", fg="red")
                sys.exit(1)

            if run:
                if "execution_error" in result:
                    click.secho(
                        f"❌ Execution failed: {result['execution_error']}", fg="red"
                    )
                    sys.exit(1)

                if compare:
                    if result.get("comparison_match"):
                        click.secho("✅ Replay passed - outputs match!", fg="green")
                    else:
                        click.secho(
                            "⚠️  Replay finished but outputs don't match", fg="yellow"
                        )
                        sys.exit(1)
                else:
                    click.secho("✅ Replay completed", fg="green")
            else:
                click.secho(
                    "✅ Dump loaded successfully (execution skipped)", fg="green"
                )

        except Exception as e:
            click.secho(f"❌ Replay failed: {e}", fg="red")
            if verbose:
                import traceback

                click.echo(traceback.format_exc())
            sys.exit(1)

    else:
        click.secho(f"Replaying sequence from: {dump_dir}", fg="cyan")

        try:
            results = replay_sequence(str(dump_path), device=device)

            if not results:
                click.secho("No dumps found in directory", fg="yellow")
                return

            click.secho(f"\nFound {len(results)} dumps:", fg="yellow")

            passed = 0
            failed = 0

            for i, res in enumerate(results):
                func_name = res.get("metadata", {}).get("function_name", "unknown")

                if "error" in res:
                    click.secho(f"  [{i + 1}] {func_name}: ❌ Error", fg="red")
                    failed += 1
                elif res.get("comparison_match"):
                    click.secho(f"  [{i + 1}] {func_name}: ✅ Passed", fg="green")
                    passed += 1
                else:
                    click.secho(f"  [{i + 1}] {func_name}: ⚠️  Mismatch", fg="yellow")
                    failed += 1

            click.secho(f"\nSummary: {passed} passed, {failed} failed", fg="white")

            if failed > 0:
                sys.exit(1)

        except Exception as e:
            click.secho(f"❌ Replay failed: {e}", fg="red")
            sys.exit(1)


@cli.command("list-dumps")
@click.argument("dump_root", required=False, default="mate_dumps")
@click.option("--details", is_flag=True, help="Show detailed information")
def list_dumps(dump_root: str, details: bool):
    """List all Level 10 dumps in the specified directory."""
    root_path = Path(dump_root)

    if not root_path.exists():
        click.secho(f"Directory not found: {dump_root}", fg="red")
        return

    dumps = []
    for item in root_path.iterdir():
        if item.is_dir() and (item / "metadata.jsonl").exists():
            dumps.append(item)

    if not dumps:
        click.secho(f"No dumps found in {dump_root}", fg="yellow")
        return

    dumps.sort()
    click.secho(f"Found {len(dumps)} dumps in {dump_root}:", fg="cyan")

    for dump_path in dumps:
        try:
            with open(dump_path / "metadata.jsonl") as f:
                lines = f.readlines()
                if lines:
                    metadata = json.loads(lines[-1])
                    func_name = metadata.get("function_name", "unknown")
                    status = metadata.get("execution_status", "unknown")
                    timestamp = metadata.get("timestamp", "unknown")
                    status_color = "green" if status == "completed" else "red"

                    if details:
                        click.echo()
                        click.secho(f"  {dump_path.name}", fg="cyan", bold=True)
                        print_kv("    Function", func_name, key_color="white")
                        print_kv(
                            "    Status",
                            status,
                            key_color="white",
                            value_color=status_color,
                        )
                        print_kv("    Timestamp", timestamp, key_color="white")

                        tensor_info = metadata.get("tensor_info", {})
                        input_keys = tensor_info.get("input_tensor_keys", [])
                        output_keys = tensor_info.get("output_tensor_keys", [])
                        print_kv("    Inputs", str(len(input_keys)), key_color="white")
                        print_kv(
                            "    Outputs", str(len(output_keys)), key_color="white"
                        )
                    else:
                        status_str = click.style(status, fg=status_color)
                        click.echo(f"  {dump_path.name}: {func_name} [{status_str}]")

        except Exception as e:
            click.secho(f"  {dump_path.name}: Error reading metadata - {e}", fg="red")


@cli.command("check")
def check_cmd():
    """Check MATE installation and environment."""
    print_header("MATE Installation Check")

    errors = []
    warnings = []

    if sys.version_info < (3, 9):
        errors.append("Python 3.9+ required")
    else:
        click.secho("✓ Python version", fg="green")

    try:
        import torch  # noqa: F401

        click.secho("✓ PyTorch installed", fg="green")

        try:
            import torch_musa  # noqa: F401

            if torch_musa.is_available():
                click.secho("✓ MUSA available", fg="green")
                click.secho(f"  Devices: {torch_musa.device_count()}", fg="cyan")
            else:
                warnings.append("MUSA not available (CPU mode only)")
        except ImportError:
            errors.append("torch_musa not installed")
    except ImportError:
        errors.append("PyTorch not installed")

    try:
        import mate

        click.secho("✓ MATE installed", fg="green")
        click.secho(f"  Version: {mate.__version__}", fg="cyan")

        mate_path = Path(mate.__file__).parent
        aot_dir = mate_path / "data" / "aot"
        if aot_dir.exists() and list(aot_dir.glob("*.so")):
            click.secho("✓ AOT libraries found", fg="green")
        else:
            warnings.append("AOT libraries not found (JIT mode only)")

    except ImportError:
        errors.append("MATE not installed")

    print_header("Summary")
    if errors:
        click.secho(f"❌ {len(errors)} errors:", fg="red")
        for e in errors:
            click.secho(f"  - {e}", fg="red")

    if warnings:
        click.secho(f"⚠️  {len(warnings)} warnings:", fg="yellow")
        for w in warnings:
            click.secho(f"  - {w}", fg="yellow")

    if not errors and not warnings:
        click.secho("✅ All checks passed!", fg="green")
    elif not errors:
        click.secho("\n⚠️  Installation OK with warnings", fg="yellow")
    else:
        click.secho("\n❌ Installation has errors", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
