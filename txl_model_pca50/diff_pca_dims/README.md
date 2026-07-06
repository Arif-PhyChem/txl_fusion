# PCA dimensionality tests for the final PCA50 TXL Fusion model

This folder mirrors the original PCA-dimensionality analysis under `weighted_version/txl_model/diff_pca_dims`, but targets the final PCA50 TXL Fusion architecture.

The final TXL model is the gated + hierarchical + calibrated PCA50 architecture from `weighted_version/txl_model_pca50/config.json`. The sweep changes only `pca_components`; all other training settings are copied from the final PCA50 config.

## 1. Explained variance only

This reproduces the PCA explained-variance analysis for fine-tuned SciBERT embeddings. PCA is fit only on the shared subtraining split and validation embeddings are transformed only.

```bash
cd /path/to/3_classes_classification/weighted_version/txl_model_pca50/diff_pca_dims/pca_explained_variance_test
python report_pca_explained_variance.py
```

Outputs:
- `pca_explained_variance_train_fit.csv`
- `pca_explained_variance_train_fit.json`
- `pca_train_fit_full.pkl`
- cached train/validation embeddings under `cache/`

## 2. PCA-dimensionality performance sweep

This trains the final TXL architecture at multiple SciBERT PCA dimensions: 0, 1, 2, 3, 5, 10, 20, 50, 100, and 768.

```bash
cd /path/to/3_classes_classification/weighted_version/txl_model_pca50/diff_pca_dims/pca_f1_sensitivity_test
python train_pca_dim_sensitivity.py
```

Each setting is saved under `models/pca_XXX/output/`. Existing completed runs are skipped unless `--overwrite` is passed.

Important: this sweep calls the same shared TXL fusion pipeline as the production PCA50 model, so it recomputes SciBERT embeddings for each PCA setting and can take substantial time.

## 3. Review figure

After both analyses finish:

```bash
cd /path/to/3_classes_classification/weighted_version/txl_model_pca50/diff_pca_dims/pca_f1_sensitivity_test
python generate_pca_review_figures.py
```

This writes:
- `/path/to/3_classes_classification/paper_review/pca_variance_f1_sensitivity_subfigures_pca50_txl.pdf`
- `/path/to/3_classes_classification/paper_review/pca_variance_f1_sensitivity_subfigures_pca50_txl.png`

The figure highlights 50 PCA components, not 100.
