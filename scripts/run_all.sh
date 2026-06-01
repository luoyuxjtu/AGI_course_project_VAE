#!/usr/bin/env bash
# Run all three GAN inpainting experiments in parallel (one GPU each),
# then run the comparison analysis once all experiments finish.
#
# Run from the project root:
#   bash scripts/run_all.sh
#
# GPU assignment — edit the three lines below to match your server:
GPU_BASELINE=0
GPU_RECON_ONLY=1
GPU_LITE=2
#
# Each experiment is fully independent (separate outputs/, separate config),
# so running them concurrently on different GPUs is safe.
#
# All experiment output goes to log files only — no progress bars in this
# terminal.  Monitor in another terminal with:
#   tail -f outputs/baseline/train.log
#   tail -f outputs/recon_only/train.log
#   tail -f outputs/lite/train.log
#
# Interrupted experiments resume automatically from their last checkpoint
# when this script is re-run.

set -uo pipefail

total_start=$(date +%s)

mkdir -p outputs/baseline outputs/recon_only outputs/lite

echo "================================================================"
echo "  GAN inpainting — launching 3 experiments in parallel"
echo "  GPU assignments: baseline=$GPU_BASELINE  recon_only=$GPU_RECON_ONLY  lite=$GPU_LITE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

# Launch each experiment — stdout+stderr go to the log file only.
# Using plain redirection (not tee) so:
#   1. No output reaches this terminal — no interleaved progress bars.
#   2. $! captures the python PID directly; wait gives python's exit code.
#      (With `cmd | tee file &`, $! would be tee's PID and a python crash
#       would go undetected because tee always exits 0.)
CUDA_VISIBLE_DEVICES=$GPU_BASELINE \
    python -m src.train --config configs/baseline.yaml \
    > outputs/baseline/train.log 2>&1 &
PID_BASELINE=$!

CUDA_VISIBLE_DEVICES=$GPU_RECON_ONLY \
    python -m src.train --config configs/recon_only.yaml \
    > outputs/recon_only/train.log 2>&1 &
PID_RECON_ONLY=$!

CUDA_VISIBLE_DEVICES=$GPU_LITE \
    python -m src.train --config configs/lite.yaml \
    > outputs/lite/train.log 2>&1 &
PID_LITE=$!

echo "  [GPU $GPU_BASELINE]   baseline    PID=$PID_BASELINE   → outputs/baseline/train.log"
echo "  [GPU $GPU_RECON_ONLY]   recon_only  PID=$PID_RECON_ONLY → outputs/recon_only/train.log"
echo "  [GPU $GPU_LITE]   lite        PID=$PID_LITE      → outputs/lite/train.log"
echo ""
echo "Waiting for all experiments to finish ..."
echo ""

# Wait for each experiment and report pass/fail individually.
FAILED=0

declare -A NAMES=([0]="baseline" [1]="recon_only" [2]="lite")
declare -A PIDS=([0]=$PID_BASELINE [1]=$PID_RECON_ONLY [2]=$PID_LITE)
declare -A GPUS=([0]=$GPU_BASELINE [1]=$GPU_RECON_ONLY [2]=$GPU_LITE)

for i in 0 1 2; do
    exp="${NAMES[$i]}"
    gpu="${GPUS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  ✓  [GPU $gpu] $exp — done"
    else
        echo "  ✗  [GPU $gpu] $exp — FAILED  (see outputs/$exp/train.log)" >&2
        FAILED=1
    fi
done

if [ $FAILED -ne 0 ]; then
    echo ""
    echo "ERROR: one or more experiments failed; skipping comparison." >&2
    exit 1
fi

echo ""
echo "================================================================"
echo "  Comparison analysis"
echo "================================================================"
python -m src.compare

total_elapsed=$(( $(date +%s) - total_start ))
echo ""
echo "================================================================"
echo "  All done in ${total_elapsed}s"
echo "  Outputs:"
echo "    outputs/baseline/      outputs/recon_only/   outputs/lite/"
echo "    outputs/comparison/    (gan_vs_recon.png + summary.md)"
echo "================================================================"
