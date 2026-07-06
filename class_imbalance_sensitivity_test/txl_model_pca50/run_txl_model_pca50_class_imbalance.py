import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/path/to/3_classes_classification')
PIPELINE = ROOT / 'weighted_version' / 'txl_fusion_variants' / 'common_fusion_pipeline.py'
BASE = Path(__file__).resolve().parent
HELD_OUT = BASE.parent / 'held_out' / 'txl_model_pca50'

CONFIGS = [
    ('baseline', 'Original PCA50 TXL setup without class weighting.'),
    ('balanced', 'Balanced sample weights using inverse-frequency class weights.'),
    ('balanced_ti_1p5', 'Balanced sample weights with additional 1.5x topological weight.'),
    ('balanced_ti_2p0', 'Balanced sample weights with additional 2.0x topological weight.'),
    ('balanced_ti_3p0', 'Balanced sample weights with additional 3.0x topological weight.'),
]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    module = load_module(PIPELINE, 'txl_pca50_imbalance_pipeline')
    val_rows = []
    test_rows = []
    for name, desc in CONFIGS:
        cfg_path = BASE / name / 'config.json'
        module.run_variant(cfg_path)
        out_dir = cfg_path.parent / 'output'
        val_metrics = json.loads((out_dir / 'validation_metrics.json').read_text())
        test_metrics = json.loads((out_dir / 'heldout_test_metrics.json').read_text())
        val_rows.append({
            'config': name,
            'description': desc,
            'val_macro_f1': val_metrics['macro_f1'],
            'val_weighted_f1': val_metrics['weighted_f1'],
            'val_accuracy': val_metrics['accuracy'],
            'val_ti_precision': val_metrics['topological']['precision'],
            'val_ti_recall': val_metrics['topological']['recall'],
            'val_ti_f1': val_metrics['topological']['f1-score'],
        })
        test_rows.append({
            'config': name,
            'description': desc,
            'test_macro_f1': test_metrics['macro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_ti_precision': test_metrics['topological']['precision'],
            'test_ti_recall': test_metrics['topological']['recall'],
            'test_ti_f1': test_metrics['topological']['f1-score'],
        })

    pd.DataFrame(val_rows).to_csv(BASE / 'txl_model_pca50_class_imbalance_summary.csv', index=False)
    HELD_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_rows).to_csv(HELD_OUT / 'txl_model_pca50_held_out_summary.csv', index=False)
    print(pd.DataFrame(val_rows).to_string(index=False))
    print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == '__main__':
    main()
