#!/usr/bin/env bash
# Run all 4 experiments (train + auto-eval) sequentially, then compare.
#
# Run from the project root:
#   bash scripts/run_all.sh
#
# Each `python -m src.train` call also runs evaluation automatically,
# so no separate evaluate step is needed.
set -e

total_start=$(date +%s)

run_exp() {
    local config="$1"
    local label="$2"
    local t0
    t0=$(date +%s)
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "  config: $config"
    echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    python -m src.train --config "$config"
    local elapsed=$(( $(date +%s) - t0 ))
    echo "  finished in ${elapsed}s"
}

run_exp configs/baseline.yaml  "Experiment 1/4 — baseline  (β=1.0, base_channels=32)"
run_exp configs/beta_0.5.yaml  "Experiment 2/4 — beta_0.5  (β=0.5, base_channels=32)"
run_exp configs/beta_4.yaml    "Experiment 3/4 — beta_4    (β=4.0, base_channels=32)"
run_exp configs/lite.yaml      "Experiment 4/4 — lite      (β=1.0, base_channels=16)"

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
