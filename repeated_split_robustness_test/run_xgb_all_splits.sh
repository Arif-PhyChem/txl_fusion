#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
XGB_DIR="$ROOT_DIR/xgb"
SPLITS=(split_01_seed_101 split_02_seed_202 split_03_seed_303 split_04_seed_404 split_05_seed_505)

cd "$XGB_DIR"
for split in "${SPLITS[@]}"; do
  echo "=== Training XGB for $split ==="
  python run_xgb_split.py --split-name "$split"
done
