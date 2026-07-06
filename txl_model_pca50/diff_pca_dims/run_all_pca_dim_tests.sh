#!/usr/bin/env bash
set -euo pipefail

BASE=/path/to/3_classes_classification/weighted_version/txl_model_pca50/diff_pca_dims

cd "$BASE/pca_explained_variance_test" || exit 1
python report_pca_explained_variance.py

cd "$BASE/pca_f1_sensitivity_test" || exit 1
python train_pca_dim_sensitivity.py "$@"
python generate_pca_review_figures.py
