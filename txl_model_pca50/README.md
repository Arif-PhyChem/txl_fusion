# txl_model_pca50

Standalone TXL model folder built from variant 29.

It uses the same weighted XGB configuration and shared split protocol as the production TXL/XGB baselines, but keeps only 50 semantic PCA components.

Run it with:

```bash
python /path/to/3_classes_classification/weighted_version/txl_model_pca50/train_txl_model_pca50.py
```
