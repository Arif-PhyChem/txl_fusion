import importlib.util
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
XGB_SCRIPT = ROOT / 'weighted_version' / 'xgb' / 'weighted_xgboost_shared_split.py'
XGB_MODEL_PATH = ROOT / 'weighted_version' / 'xgb' / 'xgb_model.json'
FEATURE_NAMES_PATH = ROOT / 'weighted_version' / 'xgb' / 'xgb_feature_names.json'
TEST_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_test_data.json'
LABEL_ORDER = ['semimetal', 'topological', 'trivial']


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main():
    mod = load_module(XGB_SCRIPT, 'weighted_xgb_module_for_inference')
    train_data = mod.load_json(mod.TRAINING_DATA_PATH)
    test_data = mod.load_json(TEST_DATA_PATH)
    train_idx, _ = mod.load_shared_split(train_data)
    subtrain_entries = [train_data[i] for i in train_idx]
    sg_class_counts, sg_total_counts = mod.build_space_group_priors(subtrain_entries)

    feature_rows = mod.build_feature_rows(test_data, sg_class_counts, sg_total_counts)
    dump_json(OUT / 'heldout_numeric_features.json', feature_rows)
    df = pd.DataFrame(feature_rows)
    y_true_labels = df['label'].tolist()
    x_test = df.drop(columns='label')

    with FEATURE_NAMES_PATH.open() as f:
        feature_names = json.load(f)['xgb_feature_names']
    x_test = x_test[feature_names]

    model = xgb.XGBClassifier()
    model.load_model(str(XGB_MODEL_PATH))
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')

    y_pred = model.predict(x_test.to_numpy())
    y_prob = model.predict_proba(x_test.to_numpy())
    label_to_code = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    y_true = [label_to_code[label] for label in y_true_labels]

    metrics = mod.evaluate(y_true, y_pred)
    dump_json(OUT / 'heldout_test_metrics.json', metrics)
    predictions = []
    for entry, pred, prob, true_label in zip(test_data, y_pred, y_prob, y_true_labels):
        predictions.append({
            'compound': mod.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': true_label,
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(prob[i]) for i, label in enumerate(LABEL_ORDER)},
        })
    dump_json(OUT / 'heldout_test_predictions.json', predictions)

    print('Held-out accuracy:', metrics['accuracy'])
    print('Held-out macro-F1:', metrics['macro_f1'])
    print('Held-out weighted-F1:', metrics['weighted_f1'])
    print()
    print('Per-class metrics:')
    for cls in ['semimetal', 'topological', 'trivial']:
        vals = metrics[cls]
        print(
            f"{cls}: precision={vals['precision']:.4f} recall={vals['recall']:.4f} "
            f"f1-score={vals['f1-score']:.4f} support={int(vals['support'])}"
        )


if __name__ == '__main__':
    main()
