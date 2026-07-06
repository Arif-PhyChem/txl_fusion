# llm_pca_50

Main weighted-version home for the semantic-only PCA-50 baseline.

This script:
- loads the fine-tuned SciBERT checkpoint
- extracts mean-pooled embeddings from the structured text
- fits a PCA basis on the shared subtraining embeddings
- retains 50 semantic PCA components
- trains an XGBoost head on the shared subtraining split
- evaluates on both the shared validation split and the fixed held-out test set
- pins the run to the third physical GPU by default via `CUDA_VISIBLE_DEVICES=2`

Run it with:

```bash
python /path/to/3_classes_classification/weighted_version/llm_pca_50/train_llm_pca50_model.py
```
