# TXL PCA50 calibration reliability

This folder stores reliability outputs for the PCA50 TXL model.

Generate them with:

```bash
cd /path/to/3_classes_classification/weighted_version/calibration_reliability
python plot_conf_bins.py --model txl_pca50
```

The script reads predictions from:

`weighted_version/txl_model_pca50/output/heldout_test_predictions_verbose.json`
