import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from common import EXP_ROOT, TRAINING_DATA_PATH, TEST_DATA_PATH, dump_json, load_json, validate_manifest_against_training_data

ROOT = Path('/path/to/3_classes_classification')
WEIGHTED_XGB_SCRIPT = ROOT / 'weighted_version' / 'xgb' / 'weighted_xgboost_shared_split.py'
LABEL_ORDER = ['semimetal', 'topological', 'trivial']


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_split_dir(split_name: str):
    return EXP_ROOT / 'xgb' / split_name


def build_model(seed, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-name', required=True)
    args = parser.parse_args()

    split_dir = resolve_split_dir(args.split_name)
    output_dir = split_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    mod = load_module(WEIGHTED_XGB_SCRIPT, f'weighted_xgb_{args.split_name}')
    train_data = load_json(TRAINING_DATA_PATH)
    test_data = load_json(TEST_DATA_PATH)
    manifest = load_json(split_dir / 'split_manifest.json')
    validate_manifest_against_training_data(manifest, train_data)
    train_idx = np.asarray(manifest['train_indices'], dtype=int)
    val_idx = np.asarray(manifest['validation_indices'], dtype=int)

    subtrain_entries = [train_data[i] for i in train_idx]
    sg_class_counts, sg_total_counts = mod.build_space_group_priors(subtrain_entries)

    train_rows = mod.build_feature_rows(train_data, sg_class_counts, sg_total_counts)
    test_rows = mod.build_feature_rows(test_data, sg_class_counts, sg_total_counts)
    dump_json(output_dir / 'training_features.json', train_rows)
    dump_json(output_dir / 'heldout_features.json', test_rows)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    feature_names = [c for c in train_df.columns if c != 'label']
    x_all = train_df[feature_names]
    x_val = x_all.iloc[val_idx]
    x_subtrain = x_all.iloc[train_idx]
    x_test = test_df[feature_names]

    le = LabelEncoder()
    le.fit(['topological', 'semimetal', 'trivial'])
    y_all = le.transform(train_df['label'])
    y_subtrain = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test_labels = test_df['label'].tolist()
    label_to_code = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    y_test = np.asarray([label_to_code[label] for label in y_test_labels], dtype=int)

    weights = mod.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    model = None
    last_error = None
    train_device = None
    for device in ['cuda', 'cpu']:
        candidate = build_model(42, device)
        try:
            candidate.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
            model = candidate
            train_device = device
            break
        except Exception as exc:
            last_error = str(exc)
    if model is None:
        raise RuntimeError(f'XGB training failed: {last_error}')

    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    model.get_booster().save_model(str(output_dir / 'xgb_model.json'))

    y_val_pred = model.predict(x_val.to_numpy())
    y_val_prob = model.predict_proba(x_val.to_numpy())
    y_test_pred = model.predict(x_test.to_numpy())
    y_test_prob = model.predict_proba(x_test.to_numpy())

    val_metrics = mod.evaluate(y_val, y_val_pred)
    test_metrics = mod.evaluate(y_test, y_test_pred)
    dump_json(output_dir / 'validation_metrics.json', val_metrics)
    dump_json(output_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(output_dir / 'class_weights.json', {str(k): v for k, v in weights.items()})
    dump_json(output_dir / 'run_metadata.json', {
        'split_name': args.split_name,
        'train_device': train_device,
        'split_manifest': str(split_dir / 'split_manifest.json'),
        'training_data': str(TRAINING_DATA_PATH),
        'test_data': str(TEST_DATA_PATH),
        'feature_names': feature_names,
    })

    dump_json(output_dir / 'validation_predictions.json', {
        'y_true': [int(v) for v in y_val],
        'y_pred': [int(v) for v in y_val_pred],
        'y_prob': [[float(x) for x in row] for row in y_val_prob],
        'label_order': LABEL_ORDER,
    })
    dump_json(output_dir / 'heldout_test_predictions.json', {
        'y_true': [int(v) for v in y_test],
        'y_pred': [int(v) for v in y_test_pred],
        'y_prob': [[float(x) for x in row] for row in y_test_prob],
        'label_order': LABEL_ORDER,
    })

    dump_json(output_dir / 'validation_predictions_verbose.json', [
        {
            'compound': mod.normalize_formula(train_data[i]['compoundName']),
            'space_group': int(round(train_data[i].get('symmetryGroupNumber', 0))),
            'true_label': str(train_df.iloc[i]['label']),
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(y_val_prob[row_idx][j]) for j, label in enumerate(LABEL_ORDER)},
        }
        for row_idx, (i, pred) in enumerate(zip(val_idx, y_val_pred))
    ])
    dump_json(output_dir / 'heldout_test_predictions_verbose.json', [
        {
            'compound': mod.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': y_test_labels[row_idx],
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(y_test_prob[row_idx][j]) for j, label in enumerate(LABEL_ORDER)},
        }
        for row_idx, (entry, pred) in enumerate(zip(test_data, y_test_pred))
    ])

    print('Validation macro-F1:', val_metrics['macro_f1'])
    print('Held-out macro-F1:', test_metrics['macro_f1'])


if __name__ == '__main__':
    main()
