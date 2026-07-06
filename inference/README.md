# Inference entry points

This release keeps inference scripts for the manuscript baselines and the final PCA50 TXL model. Trained model artifacts are not included; these scripts require locally trained checkpoints, PCA/scaler files, and XGBoost/TXL outputs generated from your own licensed copy of the data.

Included subfolders:

- `heuristic_only/`: held-out inference for the weighted three-class g(M) baseline.
- `xgb/`: held-out inference for standalone numerical XGB.
- `heuristic_xgb/`: held-out inference for heuristic-enhanced XGB.
- `llm_pca_50/`: held-out inference for the semantic PCA50 branch.
- `txl_model_pca50/`: held-out inference for the final PCA50 TXL Fusion model.
- `external_discovery_space/`: external-screening wrappers for XGB and the final PCA50 TXL model.

Plain SciBERT classifier inference, Topological Materials Database overlap utilities, PCA100, full-PCA, legacy TXL, and trained-model folders are intentionally not included in this GitHub release.
