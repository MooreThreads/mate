# Environment Variables

MATE reads configuration from the current process environment. Set variables
before launching the Python process or CLI command that should observe them.

Use `mate env` to print the MATE-related variables visible to the current
shell.

## API Logging and Dumps

The logging and dumping configuration is read when `mate.api_logging` is
imported.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_LOGLEVEL` | `0` | API logging level: `0`, `1`, `3`, `5`, or `10` |
| `MATE_LOGDEST` | `stdout` | Log destination: `stdout`, `stderr`, or a file path |
| `MATE_DUMP_DIR` | `mate_dumps` | Root directory for Level 10 tensor dumps |
| `MATE_DUMP_MAX_SIZE_GB` | `20` | Maximum total dump size in GB per process |
| `MATE_DUMP_MAX_COUNT` | `1000` | Maximum number of dumped calls per process |
| `MATE_DUMP_SAFETENSORS` | `0` | Save dumps as `safetensors` instead of `torch.save` |
| `MATE_DUMP_INCLUDE` | empty | Comma-separated `fnmatch` patterns to include for dumping |
| `MATE_DUMP_EXCLUDE` | empty | Comma-separated `fnmatch` patterns to exclude from dumping |

Log-level meanings:

- `0`: disabled
- `1`: function names only
- `3`: function names plus structured inputs and outputs, including tensor VA ranges
- `5`: level 3 plus tensor statistics
- `10`: level 5 plus on-disk tensor dumping for replay

`MATE_LOGDEST` supports `%i` in file paths. MATE replaces `%i` with the current
process id.

Dumps produced with `MATE_DUMP_SAFETENSORS=1` lose original stride and
non-contiguous layout information because tensors are saved as contiguous
tensors. Use the default `torch.save` format when replay must preserve strides.

Level 10 logging writes full API inputs and outputs to disk. Do not enable it
for sensitive workloads unless the dump directory is adequately protected.

## JIT, AOT, and Cache

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_MUSA_ARCH_LIST` | auto-detect visible devices | MUSA architecture list used by JIT/AOT workflows; accepts space-separated `major.minor` values such as `3.1` or `3.1 4.0` |
| `MATE_WORKSPACE_BASE` | home directory | Base directory for the MATE cache workspace |
| `MATE_DISABLE_JIT` | `0` | Disable runtime JIT and require matching AOT modules |
| `MATE_JIT_VERBOSE` | `0` | Show verbose ninja output for runtime JIT builds |

Runtime loading prefers a matching AOT library when one exists.
`MATE_DISABLE_JIT=1` switches to AOT-only behavior and raises an error if no
matching AOT module exists. MATE does not provide an environment variable to
bypass AOT and force runtime JIT when a matching AOT module is present.

Set `MATE_MUSA_ARCH_LIST` explicitly for offline diagnostics when no MUSA device
is visible:

```bash
MATE_MUSA_ARCH_LIST=3.1 mate module-status
```

## Compiler and Build Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_EXTRA_CFLAGS` | empty | Extra host compiler flags for JIT builds |
| `MATE_EXTRA_MUSAFLAGS` | empty | Extra `mcc` flags for JIT builds |
| `MATE_EXTRA_LDFLAGS` | empty | Extra linker flags for JIT builds |
| `MATE_MCC` | auto-detected | Override the `mcc` compiler path used by JIT builds |

MATE JIT builds also honor common build-tool variables such as `CXX` for the
host C++ compiler and `MAX_JOBS` for ninja parallelism.

## Diagnostic and Test-Only Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATE_DRY_RUN` | `0` | Internal test/diagnostic mode used by MATE tests; not intended as a normal user runtime setting |
| `MATE_PYTEST_SHARD_TOTAL` | `1` | Number of shards to dispatch all the tests to |
| `MATE_PYTEST_SHARD_INDEX` | `0` | Index of current shard |
| `MATE_PYTEST_SHARD_MODE` | `file` | Shard dispatch mode. Currently, only `file` mode is supported |

When `MATE_PYTEST_SHARD_TOTAL > 1`, `test_fmha.py` tests will dominate the last shard.
