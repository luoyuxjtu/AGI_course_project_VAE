#!/usr/bin/env bash
# Run all 4 experiments in parallel on 4 GPUs, then compare.
#
# Prerequisites:
#   - 4 CUDA GPUs available (indices 0–3).  Adjust CUDA_VISIBLE_DEVICES
#     assignments below if your GPU indices differ.
#   - Run from the project root:  bash scripts/run_all.sh
#
# Each experiment is pinned to one GPU via CUDA_VISIBLE_DEVICES and its
# stdout+stderr are tee'd to outputs/<exp>/train.log so you can tail them
# while training is in progress:
#
#   tail -f outputs/baseline/train.log
#
# If training is interrupted, re-running this script will automatically
# resume each experiment from its last saved checkpoint.

set -uo pipefail

total_start=$(date +%s)

mkdir -p outputs/baseline outputs/beta_0.5 outputs/beta_4 outputs/lite

echo "================================================================"
echo "  Starting 4 experiments in parallel (one GPU each)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/baseline.yaml \
    >> outputs/baseline/train.log 2>&1 &
PID_BASELINE=$!

CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/beta_0.5.yaml \
    >> outputs/beta_0.5/train.log 2>&1 &
PID_BETA05=$!

CUDA_VISIBLE_DEVICES=2 python -m src.train --config configs/beta_4.yaml \
    >> outputs/beta_4/train.log   2>&1 &
PID_BETA4=$!

CUDA_VISIBLE_DEVICES=3 python -m src.train --config configs/lite.yaml \
    >> outputs/lite/train.log     2>&1 &
PID_LITE=$!

echo "  [GPU 0] baseline  (β=1.0, ch=32)  PID=$PID_BASELINE  → outputs/baseline/train.log"
echo "  [GPU 1] beta_0.5  (β=0.5, ch=32)  PID=$PID_BETA05   → outputs/beta_0.5/train.log"
echo "  [GPU 2] beta_4    (β=4.0, ch=32)  PID=$PID_BETA4    → outputs/beta_4/train.log"
echo "  [GPU 3] lite      (β=1.0, ch=16)  PID=$PID_LITE     → outputs/lite/train.log"
echo ""
echo "Waiting for all experiments to complete …"
echo "(run  tail -f outputs/<exp>/train.log  in another terminal to monitor)"
echo ""

FAILED=0
NAMES=(baseline beta_0.5 beta_4 lite)
PIDS=($PID_BASELINE $PID_BETA05 $PID_BETA4 $PID_LITE)

for i in 0 1 2 3; do
    if wait "${PIDS[$i]}"; then
        echo "  [GPU $i] ${NAMES[$i]}  — done"
    else
        echo "  [GPU $i] ${NAMES[$i]}  — FAILED (exit $?; see outputs/${NAMES[$i]}/train.log)" >&2
        FAILED=1
    fi
done

total_elapsed=$(( $(date +%s) - total_start ))
echo ""
echo "All experiments finished in ${total_elapsed}s."

if [ $FAILED -ne 0 ]; then
    echo "ERROR: one or more experiments failed." >&2
    exit 1
fi

echo ""
echo "================================================================"
echo "  Comparison analysis"
echo "================================================================"
python -m src.compare

total_elapsed=$(( $(date +%s) - total_start ))
echo ""
echo "All done in ${total_elapsed}s."
echo "Results are in outputs/:"
echo "  outputs/baseline/      outputs/beta_0.5/"
echo "  outputs/beta_4/        outputs/lite/"
echo "  outputs/comparison/    (beta_comparison.png + summary.md)"
