#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPLITS=(split_01_seed_101 split_02_seed_202 split_03_seed_303 split_04_seed_404 split_05_seed_505)

for split in "${SPLITS[@]}"; do
  echo "############################################################"
  echo "### Running full repeated-split pipeline for $split"
  echo "############################################################"
  "$ROOT_DIR/run_full_pipeline_one_split.sh" "$split"
done
