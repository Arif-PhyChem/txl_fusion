import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/path/to/3_classes_classification')
BASE = Path(__file__).resolve().parent
XGB_SCRIPT = ROOT / 'weighted_version' / 'xgb' / 'weighted_xgboost_shared_split.py'
LABEL_ORDER = ['semimetal', 'topological', 'trivial']
CONFIGS = [
    {'name': 'baseline', 'class_weights': None, 'description': 'Original XGB setup without class weighting.'},
    {'name': 'balanced', 'class_weights': 'balanced', 'description': 'Balanced sample weights using inverse-frequency class weights.'},
    {'name': 'balanced_ti_1p5', 'class_weights': 'balanced', 'ti_multiplier': 1.5, 'description': 'Balanced sample weights with additional 1.5x weight on the TI/topological class.'},
    {'name': 'balanced_ti_2p0', 'class_weights': 'balanced', 'ti_multiplier': 2.0, 'description': 'Balanced sample weights with additional 2.0x weight on the TI/topological class.'},
    {'name': 'balanced_ti_3p0', 'class_weights': 'balanced', 'ti_multiplier': 3.0, 'description': 'Balanced sample weights with additional 3.0x weight on the TI/topological class.'},
]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_model(device):
    return xgb.XGBClassifier(
        device=device,
        tree_method='hist',
        max_depth=4,
        learning_rate=0.01,
        n_estimators=1000,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=2.0,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=42,
    )


def compute_class_weights(y_encoded):
    counts = Counter(map(int, y_encoded))
    n_samples = len(y_encoded)
    n_classes = len(counts)
    return {int(cls): n_samples / (n_classes * count) for cls, count in counts.items()}


def build_sample_weights(y_train, mode=None, ti_multiplier=1.0):
    if mode is None:
        return None, None
    class_weights = compute_class_weights(y_train)
    ti_class = LABEL_ORDER.index('topological')
    class_weights[ti_class] *= ti_multiplier
    sample_weights = np.array([class_weights[int(y)] for y in y_train], dtype=float)
    return sample_weights, class_weights


def train_one(mod, config, X, y, train_idx, val_idx, X_test, y_test):
    sample_weights, class_weights = build_sample_weights(
        y[train_idx],
        mode=config.get('class_weights'),
        ti_multiplier=config.get('ti_multiplier', 1.0),
    )
    model = None
    train_device = None
    last_error = None
    for device in ['cuda', 'cpu']:
        candidate = build_model(device)
        try:
            kwargs = {'eval_set': [(X[val_idx], y[val_idx])], 'verbose': False}
            if sample_weights is not None:
                kwargs['sample_weight'] = sample_weights
            candidate.fit(X[train_idx], y[train_idx], **kwargs)
            model = candidate
            train_device = device
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Training failed for {config['name']}: {last_error}")
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    pred = model.predict(X_test)
    metrics = mod.evaluate(y_test, pred)
    return {
        'config': config['name'],
        'evaluated_test_entries': int(len(y_test)),
        'skipped_test_entries': 0,
        'test_macro_f1': metrics['macro_f1'],
        'test_weighted_f1': metrics['weighted_f1'],
        'test_accuracy': metrics['accuracy'],
        'test_ti_precision': metrics['topological']['precision'],
        'test_ti_recall': metrics['topological']['recall'],
        'test_ti_f1': metrics['topological']['f1-score'],
    }


def main():
    mod = load_module(XGB_SCRIPT, 'weighted_xgb_for_imbalance_test')
    train_rows = mod.load_json(mod.OUT / 'xgb_data_weighted_shared_split.json')
    feature_names = mod.load_json(mod.OUT / 'xgb_feature_names.json')['xgb_feature_names']
    train_df = pd.DataFrame(train_rows)
    X = train_df[feature_names].to_numpy(dtype=np.float32)
    label_to_code = {label: idx for idx, label in enumerate(['semimetal', 'topological', 'trivial'])}
    y = np.asarray([label_to_code[label] for label in train_df['label'].tolist()], dtype=int)

    clean_training = mod.load_json(mod.TRAINING_DATA_PATH)
    clean_test = mod.load_json(ROOT / 'label_noise_analysis' / 'clean_test_data.json')
    train_idx, val_idx = mod.load_shared_split(clean_training)
    subtrain_entries = [clean_training[i] for i in train_idx]
    sg_class_counts, sg_total_counts = mod.build_space_group_priors(subtrain_entries)
    test_rows = mod.build_feature_rows(clean_test, sg_class_counts, sg_total_counts)
    test_df = pd.DataFrame(test_rows)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = np.asarray([label_to_code[label] for label in test_df['label'].tolist()], dtype=int)

    summaries = [train_one(mod, cfg, X, y, train_idx, val_idx, X_test, y_test) for cfg in CONFIGS]
    pd.DataFrame(summaries).to_csv(BASE / 'xgb_held_out_summary.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
