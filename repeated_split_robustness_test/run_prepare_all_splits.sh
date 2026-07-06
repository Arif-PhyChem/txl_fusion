#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCIBERT_DIR="$ROOT_DIR/scibert_finetuning"
SPLITS=(split_01_seed_101 split_02_seed_202 split_03_seed_303 split_04_seed_404 split_05_seed_505)

cd "$SCIBERT_DIR"
for split in "${SPLITS[@]}"; do
  echo "=== Preparing textual features for $split ==="
  python prepare_textual_features_for_split.py --split-name "$split"
done
