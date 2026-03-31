# MATE CLI

MATE CLI provides command-line utilities for installation checks, configuration inspection, environment diagnostics, and Level 10 dump replay.

All examples below use `mate`, but `python -m mate` is equivalent after MATE is installed correctly.

## Installation

Install the base package:

```bash
pip install mate
mate --help
python -m mate --help
```

If you plan to load or replay dumps written in `safetensors` format, install the CLI extra or install `safetensors` directly.

From a package index:

```bash
pip install "mate[cli]"
```

If you only have a local wheel, use the standard direct-reference syntax instead of `./wheel.whl[cli]`:

```bash
python -m pip install "mate[cli] @ file:///abs/path/mate-0.1.3+mu436torch2.7-cp310-cp310-linux_x86_64.whl"
```

Notes:

- The `name[extra] @ file:///...` form is the recommended pip syntax for extras with a local wheel
- If your environment cannot access a package index, the extra dependency must also be available locally. For `mate[cli]`, that means `safetensors`
- Installing only the MATE wheel without resolving `safetensors` is still valid, but replaying `*.safetensors` dumps will fail until `safetensors` is installed

## Quick Reference

| Command | Purpose |
| --- | --- |
| `mate show-config` | Show MATE, Python, PyTorch, MUSA, and AOT information |
| `mate env` | Show relevant environment variables and their current values |
| `mate check` | Validate a usable MATE runtime environment |
| `mate replay --dir PATH` | Replay one Level 10 dump or a directory of dumps |
| `mate list-dumps [PATH]` | List dump directories under a root directory |

Useful global options:

- `mate --help`
- `mate --version`

## Commands

### `show-config`

Show runtime and installation information:

```bash
mate show-config
mate show-config --json
```

The command reports:

- MATE version and git version
- Python version
- PyTorch version
- `torch_musa` version and whether MUSA is available
- MUSA device count and device names
- AOT library directory and a short sample of detected `.so` files
- Installed package versions for `mate`, `torch`, and `torch_musa`

Use `--json` when the output will be consumed by scripts.

### `env`

Print MATE-related environment variables:

```bash
mate env
```

This command is a read-only view of the current process environment. It is useful for checking whether shell exports are in place before launching a workload.

### `check`

Validate whether the current environment is usable for MATE:

```bash
mate check
```

The command checks:

- Python version (`>= 3.9`)
- PyTorch importability
- `torch_musa` importability
- MUSA device availability
- MATE importability
- Presence of AOT libraries

Behavior:

- Hard errors cause exit code `1`
- Warnings such as "MUSA not available" or "AOT libraries not found" do not fail the command

### `replay`

Replay API calls captured from Level 10 logging:

```bash
mate replay --dir mate_dumps/20260310_xxxx_call0001
mate replay --dir mate_dumps/
mate replay --dir dumps/ --device cpu
mate replay --dir dumps/ --no-run
mate replay --dir dumps/ --no-compare
mate replay --dir dumps/ --verbose
```

Supported options:

| Option | Meaning |
| --- | --- |
| `--dir PATH` | Dump directory or dump-root directory to replay |
| `--device {musa,cpu,musa:N}` | Target device for loading tensors and running replay |
| `--run/--no-run` | Execute the resolved function or only reconstruct arguments |
| `--compare/--no-compare` | Compare execution results with dumped outputs |
| `-v, --verbose` | Print metadata and argument summaries for a single dump |

Replay mode depends on the path passed to `--dir`:

- If `PATH/metadata.jsonl` exists, MATE treats `PATH` as a single dump directory.
- Otherwise, MATE scans the immediate subdirectories under `PATH` and replays each one that contains `metadata.jsonl`.

Single-dump replay:

- Honors `--run`, `--compare`, and `--verbose`
- Can be used for argument reconstruction only with `--no-run`
- Prints a detailed summary for one function call

Directory replay:

- Replays each immediate child dump in sorted order
- Preserves stateful objects across calls by tracking constructor-created objects internally
- Summarizes pass/fail status for the full sequence
- Currently always executes with output comparison for batch replay; `--no-run`, `--no-compare`, and `--verbose` do not change batch behavior

Comparison behavior:

- Output comparison only checks tensor outputs
- A single tensor result is compared as `result`
- Tuple or list results are compared item by item for tensor elements only
- Comparison uses `torch.allclose(..., rtol=1e-3, atol=1e-3)`
- If the dump has inputs but no saved outputs, comparison is treated as a mismatch

Practical notes:

- `--device cpu` is useful for validating dump loading and argument reconstruction, but execution only succeeds if the target API supports CPU tensors
- Replaying `safetensors` dumps requires `safetensors` to be installed
- Dumps produced with `MATE_DUMP_SAFETENSORS=1` lose original stride and non-contiguous layout information because tensors are saved as contiguous tensors

### `list-dumps`

List Level 10 dump directories:

```bash
mate list-dumps
mate list-dumps mate_dumps/
mate list-dumps mate_dumps/ --details
```

Behavior:

- Default root directory is `mate_dumps`
- Only immediate child directories are scanned
- A directory is considered a dump if it contains `metadata.jsonl`
- The command reads the last record in each `metadata.jsonl` file and displays its status

With `--details`, the output includes:

- Function name
- Execution status
- Timestamp
- Number of input tensors
- Number of output tensors

## Level 10 Dump Layout

When `MATE_LOGLEVEL=10`, MATE writes one subdirectory per dumped API call under `MATE_DUMP_DIR`:

```text
mate_dumps/
├── session.jsonl
├── 20260310_153045_123_pid43210_flash_attn_with_kvcache_call0001/
│   ├── metadata.jsonl
│   ├── inputs.pt
│   └── outputs.pt
└── 20260310_153045_456_pid43210_gemm_fp8_nt_groupwise_call0001/
    ├── metadata.jsonl
    ├── inputs.safetensors
    └── outputs.safetensors
```

Notes:

- Directory names include timestamp, process id, function name, and call sequence
- `metadata.jsonl` is append-only JSONL; the last line is the latest state of the dump
- `session.jsonl` aggregates the same metadata records at the dump-root level
- If a process crashes after inputs are written but before outputs are saved, the dump directory may contain only `metadata.jsonl` plus input files
- Typical `execution_status` values are `inputs_saved` and `completed`

## Environment Variables

The logging and dumping configuration is read when `mate.api_logging` is imported. Set these variables before launching the Python process you want to observe.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_LOGLEVEL` | `0` | Logging level: `0`, `1`, `3`, `5`, `10` |
| `MATE_LOGDEST` | `stdout` | Log destination: `stdout`, `stderr`, or a file path |
| `MATE_DUMP_DIR` | `mate_dumps` | Root directory for Level 10 dumps |
| `MATE_DUMP_MAX_SIZE_GB` | `20` | Maximum total dump size in GB per process |
| `MATE_DUMP_MAX_COUNT` | `1000` | Maximum number of dumped calls per process |
| `MATE_DUMP_SAFETENSORS` | `0` | Save dumps as `safetensors` instead of `torch.save` |
| `MATE_DUMP_INCLUDE` | empty | Comma-separated `fnmatch` patterns to include |
| `MATE_DUMP_EXCLUDE` | empty | Comma-separated `fnmatch` patterns to exclude |
| `TVM_FFI_MUSA_ARCH_LIST` | unset | MUSA architecture list used by JIT-related workflows |
| `MATE_AOT_BUILD` | unset | AOT build mode flag |

Log-level meaning:

- `0`: disabled
- `1`: function names only
- `3`: function names plus structured inputs and outputs
- `5`: level 3 plus tensor statistics
- `10`: level 5 plus on-disk tensor dumping for replay

`MATE_LOGDEST` also supports `%i` in file paths, which is replaced with the current process id.

## Usage Examples

### Verify an installation

```bash
mate check
mate show-config
mate env
```

### Capture a single failing API call

```bash
export MATE_LOGLEVEL=10
export MATE_DUMP_DIR=./debug_dumps
python your_script.py

mate list-dumps ./debug_dumps
mate replay --dir ./debug_dumps/20260310_xxxx_call0001 --verbose
```

### Limit dumps to selected APIs

```bash
export MATE_LOGLEVEL=10
export MATE_DUMP_DIR=./filtered_dumps
export MATE_DUMP_INCLUDE="*gemm*,*attention*"
export MATE_DUMP_EXCLUDE="*benchmark*"
python your_script.py
```

### Replay a full session

```bash
mate replay --dir ./debug_dumps/
```

This mode is useful when later calls depend on objects created by earlier calls.

### Write logs to a per-process file

```bash
export MATE_LOGLEVEL=5
export MATE_LOGDEST=/tmp/mate_%i.log
python your_script.py
```

## Exit Codes

Command exit behavior is not identical across all subcommands:

- `mate show-config`: returns `0` on successful execution
- `mate env`: returns `0` on successful execution
- `mate check`: returns `1` only when hard errors are found
- `mate replay`: returns `1` on replay failure, execution failure, mismatch, or invalid replay setup
- `mate list-dumps`: informational command; missing directories or empty results are currently reported in output and do not force a non-zero exit code

## Troubleshooting

- `Failed to load MATE API logging module`: ensure the installed MATE package is complete and importable in the current Python environment
- `safetensors not installed`: install `safetensors`, or reinstall from a package index with `pip install "mate[cli]"`, or reinstall from a local wheel with `python -m pip install "mate[cli] @ file:///abs/path/your-mate.whl"`
- `No dumps found`: `mate list-dumps` and batch replay only scan immediate child directories, not nested trees recursively
- `compare_outputs=True but no output file found`: the dump is incomplete, often because the original process crashed after saving inputs
- `AOT libraries not found`: MATE may still work in JIT mode, but startup behavior can differ from an AOT-enabled installation
- Replay mismatches do not always mean argument reconstruction failed; they can also reflect runtime differences, device differences, or numerical drift

## Security Note

Level 10 logging writes full API inputs and outputs to disk. Do not enable it for sensitive workloads unless the dump directory is adequately protected.
