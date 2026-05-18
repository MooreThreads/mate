# !/bin/bash
set -ex
DRYRUN_FILES="test_sparse_mla.py test_fmha.py"

export MATE_DRY_RUN=1

python -m pytest -n auto -vv $DRYRUN_FILES --max-worker-restart 0

unset MATE_DRY_RUN

python -m pytest -vv .
