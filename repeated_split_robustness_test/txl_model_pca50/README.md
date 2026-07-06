# TXL PCA50 repeated-split runs

This folder contains repeated-split robustness runs for the standalone PCA50 TXL model.

Each split uses:
- the split-specific `split_manifest.json` in this folder
- the matching split-specific fine-tuned SciBERT checkpoint under `scibert_finetuning/<split>/.../best_model`
- the fixed cleaned held-out test set used everywhere else

Run one split with:

```bash
cd /path/to/3_classes_classification/weighted_version/repeated_split_robustness_test/txl_model_pca50
python run_txl_model_pca50_split.py --split-name split_01_seed_101
```
