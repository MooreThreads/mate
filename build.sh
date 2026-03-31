#!/bin/bash


# Initialize default value
fmha_aot_level=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fmha-fwd-aot-level)
            fmha_aot_level="$2"
            # Validate the value is 0, 1, or 2
            if [[ "$fmha_aot_level" != "0" && "$fmha_aot_level" != "1" && "$fmha_aot_level" != "2" ]]; then
                echo "Error: --fmha-fwd-aot-level must be 0, 1, or 2"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Error: Unknown option $1"
            exit 1
            ;;
    esac
done

set -xe
rm -rf aot-lib mate/data/aot
TVM_FFI_MUSA_ARCH_LIST=3.1 MATE_AOT_BUILD=1 python mate/aot.py --out-dir aot-lib --build-dir ./build/aot --fmha-aot-level $fmha_aot_level
TVM_FFI_MUSA_ARCH_LIST=3.1 MATE_AOT_BUILD=1 python -m build --wheel --no-isolation
