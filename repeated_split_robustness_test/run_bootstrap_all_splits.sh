#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOTSTRAP_DIR="$ROOT_DIR/bootstrap"
SPLITS=(split_01_seed_101 split_02_seed_202 split_03_seed_303 split_04_seed_404 split_05_seed_505)

cd "$BOOTSTRAP_DIR"
for split in "${SPLITS[@]}"; do
  echo "=== Running paired bootstrap for $split ==="
  python run_paired_bootstrap_comparison.py --split-name "$split"
done
