# Repeated Split Robustness Test

This folder contains five alternative stratified subtraining/validation splits built from
`label_noise_analysis/clean_training_data.json` while preserving the same split sizes used in the
current study:

- subtraining: 23,644
- validation: 5,912
- fixed held-out test: 7,397 (`label_noise_analysis/clean_test_data.json`)

The split random states are: [101, 202, 303, 404, 505].

Layout:
- `split_definitions/`: canonical split manifests plus copied subtraining/validation JSON rows
- `xgb/`: per-split XGB configs and a runner script
- `txl/`: per-split TXL configs and a runner script
- `scibert_finetuning/`: per-split SciBERT data folders and helper scripts for preparing/fine-tuning split-specific semantic data
- `bootstrap/`: paired bootstrap scripts/configs for comparing split-matched XGB and TXL predictions

These experiments do not modify the original production split in `label_noise_analysis/shared_split_manifest.json`.


Launch helpers:
- `./run_prepare_all_splits.sh`: build split-specific SciBERT narratives for all 5 splits
- `./run_finetune_all_splits.sh`: fine-tune SciBERT for all 5 splits
- `./run_xgb_all_splits.sh`: train standalone XGB for all 5 splits
- `./run_txl_all_splits.sh`: train the legacy TXL setup for all 5 splits
- `./run_txl_model_pca50_all_splits.sh`: train the standalone PCA50 TXL model for all 5 splits
- `./run_bootstrap_all_splits.sh`: run paired bootstrap for the legacy TXL vs XGB comparison
- `./run_bootstrap_txl_model_pca50_all_splits.sh`: run paired bootstrap for the PCA50 TXL vs XGB comparison
- `./run_full_pipeline_one_split.sh split_01_seed_101`: run the full pipeline for one split
- `./run_full_pipeline_all_splits.sh`: run the full pipeline for all 5 splits

Recommended order if you want to control GPU/queue usage manually:
1. `./run_prepare_all_splits.sh`
2. `./run_finetune_all_splits.sh`
3. `./run_xgb_all_splits.sh`
4. `./run_txl_all_splits.sh`
5. `./run_bootstrap_all_splits.sh`


See also: `HOW_TO_RUN.md` for a step-by-step run guide with exact commands.
