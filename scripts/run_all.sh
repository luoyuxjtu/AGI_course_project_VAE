#!/usr/bin/env bash
# Run all 4 experiments (train + eval) sequentially, then comparison analysis.
set -e

echo "=== Experiment 1/4: baseline (beta=1) ==="
python -m src.train --config configs/baseline.yaml

echo "=== Experiment 2/4: beta_0.5 ==="
python -m src.train --config configs/beta_0.5.yaml

echo "=== Experiment 3/4: beta_4 ==="
python -m src.train --config configs/beta_4.yaml

echo "=== Experiment 4/4: lite (base_channels=16) ==="
python -m src.train --config configs/lite.yaml

echo "=== Comparison analysis ==="
python -m src.compare

echo "All done. Results are in outputs/."
