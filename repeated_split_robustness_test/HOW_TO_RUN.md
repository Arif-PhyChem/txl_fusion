# How To Run Repeated-Split Robustness Experiments

This guide explains how to run the 5 repeated-split experiments created under:

`/path/to/3_classes_classification/weighted_version/repeated_split_robustness_test`

## Available splits

The five alternative shared-size splits are:

- `split_01_seed_101`
- `split_02_seed_202`
- `split_03_seed_303`
- `split_04_seed_404`
- `split_05_seed_505`

Each split keeps the same sizes as the current study:

- subtraining: `23644`
- validation: `5912`
- held-out test: fixed cleaned test set (`7397` records)

## Folder layout

- `split_definitions/`: split manifests plus copied subtraining/validation JSON rows
- `scibert_finetuning/`: split-specific SciBERT narrative preparation and fine-tuning
- `xgb/`: standalone XGB training for each split
- `txl/`: TXL Fusion training for each split
- `bootstrap/`: paired bootstrap comparison between XGB and TXL for each split

## Recommended execution order

Run the stages in this order:

1. Prepare SciBERT text inputs
2. Fine-tune SciBERT
3. Train standalone XGB
4. Train TXL Fusion
5. Run paired bootstrap comparison

This is safer than launching the whole pipeline at once, especially if you want to watch GPU usage or restart a failed stage.

## Run everything from the repeated-split folder

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test
```

## Option A: run one stage across all 5 splits

### 1. Prepare split-specific SciBERT narratives

```bash
./run_prepare_all_splits.sh
```

### 2. Fine-tune SciBERT for all splits

```bash
./run_finetune_all_splits.sh
```

### 3. Train standalone XGB for all splits

```bash
./run_xgb_all_splits.sh
```

### 4. Train TXL Fusion for all splits

```bash
./run_txl_all_splits.sh
```

### 5. Run paired bootstrap for all splits

```bash
./run_bootstrap_all_splits.sh
```

## Option B: run the full pipeline for one split

Example:

```bash
./run_full_pipeline_one_split.sh split_01_seed_101
```

This will run, in order:

- SciBERT narrative preparation
- SciBERT fine-tuning
- standalone XGB training
- TXL training
- paired bootstrap comparison

## Option C: run the full pipeline for all splits

```bash
./run_full_pipeline_all_splits.sh
```

Use this only if you are comfortable letting the entire workflow run without manual checks between stages.

## Manual commands by stage

If you prefer to run things manually:

### SciBERT preparation

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test/scibert_finetuning
python prepare_textual_features_for_split.py --split-name split_01_seed_101
```

### SciBERT fine-tuning

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test/scibert_finetuning
python finetune_scibert_for_split.py --split-name split_01_seed_101
```

### Standalone XGB

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test/xgb
python run_xgb_split.py --split-name split_01_seed_101
```

### TXL Fusion

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test/txl
python run_txl_split.py --split-name split_01_seed_101
```

### Paired bootstrap comparison

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test/bootstrap
python run_paired_bootstrap_comparison.py --split-name split_01_seed_101
```

## Where outputs are written

### SciBERT

For each split:

`scibert_finetuning/<split-name>/`

Important outputs include:

- `finetune_dataset_scibert_improved.json`
- `scibert-finetuned-weighted-improved-input/`
- `scibert-finetuned-weighted-improved-input/best_model/`

### XGB

For each split:

`xgb/<split-name>/output/`

### TXL

For each split:

`txl/<split-name>/output/`

### Bootstrap

For each split:

`bootstrap/<split-name>/`

## Important notes

- These repeated-split experiments do **not** overwrite your original production shared split.
- The held-out test set remains fixed across all 5 repeated splits.
- Each split changes only the subtraining/validation partition of the cleaned 29,556-record training pool.
- TXL for a given split expects the SciBERT fine-tuning for that same split to exist first.

## Practical recommendation

A good workflow is:

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test
./run_prepare_all_splits.sh
./run_finetune_all_splits.sh
./run_xgb_all_splits.sh
./run_txl_all_splits.sh
./run_bootstrap_all_splits.sh
```

That gives you checkpoints and results in a controlled stage-by-stage manner.
