#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <split-name>"
  echo "Example: $0 split_01_seed_101"
  exit 1
fi

SPLIT_NAME="$1"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT_DIR/scibert_finetuning"
echo "=== [$SPLIT_NAME] Preparing textual features ==="
python prepare_textual_features_for_split.py --split-name "$SPLIT_NAME"

echo "=== [$SPLIT_NAME] Fine-tuning SciBERT ==="
python finetune_scibert_for_split.py --split-name "$SPLIT_NAME"

cd "$ROOT_DIR/xgb"
echo "=== [$SPLIT_NAME] Training XGB ==="
python run_xgb_split.py --split-name "$SPLIT_NAME"

cd "$ROOT_DIR/txl"
echo "=== [$SPLIT_NAME] Training TXL ==="
python run_txl_split.py --split-name "$SPLIT_NAME"

cd "$ROOT_DIR/bootstrap"
echo "=== [$SPLIT_NAME] Running paired bootstrap ==="
python run_paired_bootstrap_comparison.py --split-name "$SPLIT_NAME"
