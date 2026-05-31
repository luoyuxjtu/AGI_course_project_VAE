#!/usr/bin/env bash
# Run all three GAN inpainting experiments sequentially, then compare.
#
# Run from the project root:
#   bash scripts/run_all.sh
#
# Each `src.train` call automatically invokes `src.evaluate` after
# training, so one command covers:
#
#   train baseline   → eval baseline   (inpainting.png, loss_curve.png, eval_metrics.json)
#   train recon_only → eval recon_only
#   train lite       → eval lite
#   python -m src.compare              (gan_vs_recon.png, summary.md)
#
# Stdout + stderr for each experiment are tee'd to outputs/<exp>/train.log
# so you can monitor progress in another terminal:
#
#   tail -f outputs/baseline/train.log
#
# Interrupted runs resume automatically from the last saved checkpoint.

set -uo pipefail

total_start=$(date +%s)

mkdir -p outputs/baseline outputs/recon_only outputs/lite

echo "================================================================"
echo "  GAN inpainting — running 3 experiments sequentially"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

run_exp() {
    local config="$1"
    local exp="$2"
    local log="outputs/${exp}/train.log"

    echo "----------------------------------------------------------------"
    echo "  Starting: $exp  (config: $config)"
    echo "  Log: $log"
    echo "  $(date '+%H:%M:%S')"
    echo "----------------------------------------------------------------"

    if python -m src.train --config "$config" 2>&1 | tee "$log"; then
        echo "  ✓ $exp complete"
    else
        echo "  ✗ $exp FAILED — see $log" >&2
        exit 1
    fi
    echo ""
}

run_exp configs/baseline.yaml   baseline
run_exp configs/recon_only.yaml recon_only
run_exp configs/lite.yaml       lite

echo "================================================================"
echo "  Comparison analysis"
echo "================================================================"
python -m src.compare

total_elapsed=$(( $(date +%s) - total_start ))
echo ""
echo "================================================================"
echo "  All done in ${total_elapsed}s"
echo "  Outputs:"
echo "    outputs/baseline/     outputs/recon_only/   outputs/lite/"
echo "    outputs/comparison/   (gan_vs_recon.png + summary.md)"
echo "================================================================"
