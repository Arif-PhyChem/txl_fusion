# TXL PCA50 class-imbalance sensitivity

This experiment re-runs the full PCA50 TXL architecture under several class-weight settings:
- `baseline`: no class weighting
- `balanced`: inverse-frequency weighting
- `balanced_ti_1p5`: balanced weighting with 1.5x extra topological weight
- `balanced_ti_2p0`: balanced weighting with 2.0x extra topological weight
- `balanced_ti_3p0`: balanced weighting with 3.0x extra topological weight

Run with:

```bash
cd /path/to/3_classes_classification/weighted_version/class_imbalance_sensitivity_test/txl_model_pca50
python run_txl_model_pca50_class_imbalance.py
```
