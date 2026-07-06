#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TXL_DIR="$ROOT_DIR/txl"
SPLITS=(split_01_seed_101 split_02_seed_202 split_03_seed_303 split_04_seed_404 split_05_seed_505)

cd "$TXL_DIR"
for split in "${SPLITS[@]}"; do
  echo "=== Training TXL for $split ==="
  python run_txl_split.py --split-name "$split"
done
