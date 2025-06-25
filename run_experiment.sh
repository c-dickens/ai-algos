#!/usr/bin/env bash
# Usage: ./run_experiment.sh [CORESET_TYPE] [MODEL_SIZE] [LOG_DIR] [EXTRA_ARGS...]
#
# CORESET_TYPE: Type of coreset to use (uniform, sensitivity, all).
#               Use "all" to run each method sequentially.
# MODEL_SIZE:   GPT-2 model size (124M, 355M, 774M, 1558M). Default: 124M.
# LOG_DIR:      Directory to store CSV log files. Default: logs.
#
# This script runs three epochs for each selected method.
# Any additional arguments are forwarded to train_finetune.py.

set -euo pipefail

CORESET=${1:-all}
MODEL=${2:-124M}
LOGDIR=${3:-logs}
shift 3 || true
mkdir -p "$LOGDIR"

if [[ "$CORESET" == "all" ]]; then
    methods=(uniform sensitivity all)
else
    methods=("$CORESET")
fi

for m in "${methods[@]}"; do
    python3 train_finetune.py \
        --coreset-type "$m" \
        --model-size "$MODEL" \
        --epochs 3 \
        --log-file "$LOGDIR/${m}_${MODEL}.csv" \
        "$@"
done

