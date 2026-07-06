# TXL Fusion: code release

This repository contains the code accompanying the manuscript **TXL Fusion: A Hybrid Machine Learning Framework Integrating Chemical Heuristics and Large Language Models for Topological Materials Discovery**.

The preprint is available here https://arxiv.org/abs/2511.04068


This repository provides the TXL Fusion implementation used for semantic--numerical fusion in the manuscript. Trained checkpoints, generated predictions, cached embeddings, and database-derived data artifacts are not redistributed.

TXL Fusion combines three sources of evidence for three-class topological-material classification:

- composition-based heuristic/topogivity scores,
- numerical and chemistry descriptors trained with XGBoost,
- semantic embeddings from a fine-tuned SciBERT encoder compressed to 50 PCA components and fused with numerical descriptors.

## License

The code is released under the MIT License. See `LICENSE`.

The MIT License applies only to this software. It does not grant permission to redistribute the underlying Topological Materials Database records, third-party model checkpoints, or any derived datasets that are not included here.

## Why the data are not included

The cleaned training and held-out records were derived from the Topological Materials Database and related topological-material resources. Because those source data are subject to their own terms and copyright/licensing restrictions, they are not redistributed here. To reproduce the experiments, obtain the underlying materials records from the original database providers and follow their license/usage terms.

Recommended source to consult:

- Topological Materials Database / Topological Quantum Chemistry resources from the original authors and hosting institutions.

After obtaining the source records, convert them into the JSON schema expected by the scripts, or adapt the loader functions to your local schema.

## What is included

The release keeps the scripts needed to reproduce the final workflow and manuscript analyses:

- `gm_three_class_baseline/`: direct three-class composition/topogivity heuristic baseline.
- `xgb/`: standalone numerical-descriptor XGBoost training and SHAP utilities.
- `heuristic_xgb/`: XGBoost with numerical descriptors plus heuristic/topogivity features.
- `llm_pca_50/`: Heuristic + LLM/ branch training script.
- `txl_model_pca50/`: final TXL Fusion training pipeline.
- `txl_model_pca50/diff_pca_dims/`: PCA explained-variance and PCA-dimension sensitivity scripts used for manuscript analysis.
- `inference/txl_model_pca50/`: held-out inference script for the final TXL model, requiring locally trained artifacts.
- `inference/external_discovery_space/txl_model_pca50/`: external-screening inference script for the final TXL model.
- `class_imbalance_sensitivity_test/txl_model_pca50/`: TXL class-weighting sensitivity experiment.
- `repeated_split_robustness_test/txl_model_pca50/`:  TXL repeated-split robustness workflow.
- `repeated_split_robustness_test/xgb/`: XGB repeated-split baseline used for comparison.
- `repeated_split_robustness_test/bootstrap_txl_model_pca50/`: paired-bootstrap comparison against XGB.
- `calibration_reliability/`: reliability-diagram and ECE plotting scripts.
- `element_count_performance/` and `element_count_distribution/`: chemical-complexity analysis scripts.
- `semantic_vs_numerical_descriptor/`: UMAP/silhouette comparison of semantic and numerical descriptors.
- `ablation_bar_figure/`: ablation figure generator.

## What is not included

The following artifacts are excluded by design for two reasons. First, the cleaned training/test records are derived from the Topological Materials Database and related resources, so they are not redistributed here because users should obtain those data directly from the original providers under the applicable copyright, license, and citation terms. Second, trained checkpoints, cached embeddings, prediction dumps, and generated figures can be very large and are reproducible from the scripts once the user has obtained the source data and trained local models.

Excluded artifacts include:

- cleaned training/test JSON records derived from the Topological Materials Database,
- SciBERT fine-tuned checkpoints,
- XGBoost and TXL trained model files,
- PCA/scaler pickle files,
- PyTorch model states,
- cached embeddings and feature matrices,
- prediction dumps and generated result tables,
- generated manuscript figures,
- exploratory development folders not required for the published workflow.

The `.gitignore` in this release is configured to keep these artifacts out of version control if you regenerate them locally.

## Expected data layout

Most scripts were developed in a research workspace and use path constants. In this open-source copy, local absolute paths have been replaced with:

```text
/path/to/3_classes_classification
```

Before running experiments, either update these placeholders in the scripts/configs or adapt the scripts to read from an environment variable such as `TXL_PROJECT_ROOT`.

A convenient local layout is:

```text
3_classes_classification/
  label_noise_analysis/
    clean_training_data.json
    clean_test_data.json
    shared_split_manifest.json
  uncased_scibert_improved_input/
    scibert-finetuned-weighted-improved-input/
      checkpoint-*/
  weighted_version/
    ... this code ...
```

The key expected files are:

- `clean_training_data.json`: cleaned training pool used for subtraining/validation.
- `clean_test_data.json`: cleaned held-out test set.
- `shared_split_manifest.json`: material IDs or indices defining subtraining and validation splits.
- a fine-tuned SciBERT checkpoint directory, if running semantic/TXL models.

## Environment

Python 3.9 or newer is recommended. The core dependencies are listed in `requirements.txt`.

Minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

GPU acceleration is recommended for SciBERT embedding extraction and fine-tuning, but the XGBoost-only baselines can run on CPU.

## Main workflows

### 1. Heuristic baseline

```bash
cd gm_three_class_baseline
python build_weighted_clean_topogivities.py
python evaluate_gm_three_class.py
```

### 2. Standalone numerical XGBoost

```bash
cd xgb
python weighted_xgboost_shared_split.py
```

### 3. Heuristic-enhanced XGBoost

```bash
cd heuristic_xgb
python weighted_xgboost_shared_split.py
```

### 4. Heuristic + LLM branch

```bash
cd llm_pca_50
python train_llm_pca50_model.py
```

### 5. Final TXL Fusion model

```bash
cd txl_model_pca50
python train_txl_model_pca50.py
```

The final TXL pipeline fits PCA on subtraining embeddings only, standardizes descriptor blocks using subtraining statistics, trains the gated semantic-numerical fusion module, trains hierarchical XGBoost heads, and applies validation-selected routing/calibration before held-out evaluation.

### 6. PCA dimensionality sensitivity

```bash
cd txl_model_pca50/diff_pca_dims
bash run_all_pca_dim_tests.sh
```

### 7. Repeated-split robustness

```bash
cd repeated_split_robustness_test
bash run_full_pipeline_all_splits.sh
```

### 8. Paired bootstrap comparison

```bash
cd repeated_split_robustness_test
bash run_bootstrap_txl_model_pca50_all_splits.sh
```

### 9. Class-imbalance sensitivity

```bash
cd class_imbalance_sensitivity_test
python run_xgb_class_imbalance_sensitivity.py
cd txl_model_pca50
bash run_txl_model_pca50_class_imbalance.sh
```

### 10. Calibration and reliability figures

```bash
cd calibration_reliability
python plot_conf_bins.py
```

## Reproducing manuscript figures

Figure-generation scripts are included, but most require regenerated local metrics and predictions. Run the relevant training/evaluation scripts first, then run the figure scripts, for example:

```bash
python ablation_bar_figure/generate_ablation_bar_figure.py
python element_count_performance/evaluate_element_count_performance.py
python element_count_performance/plot_element_count_performance.py
python semantic_vs_numerical_descriptor/generate_topology_comparison.py
python plot_features_importance_pca50.py
python plot_shap_beeswarm_pca50.py
```

## Notes on reproducibility

- The public release is intended to reproduce the pipeline, not to redistribute proprietary or copyrighted source data.
- Exact numerical reproduction requires the same cleaned dataset, split manifest, and SciBERT checkpoint used in the manuscript.
- Some scripts still reflect the original research workflow and may need path edits for a new machine.
- Generated outputs should be treated as local artifacts and should not be committed unless they are small, non-restricted summaries.

## Citation

If you use this code, please cite the associated manuscript and the original Topological Materials Database / Topological Quantum Chemistry data sources from which you obtained the materials records.
