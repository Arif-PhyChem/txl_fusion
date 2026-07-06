import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import input_gen_4_xgb

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
DATA_PATH = Path('/path/to/external_material_records/different.json')
MODEL_PATH = ROOT / 'weighted_version' / 'xgb' / 'xgb_model.json'
FEATURE_NAMES_PATH = ROOT / 'weighted_version' / 'xgb' / 'xgb_feature_names.json'
LABEL_ORDER = ['semimetal', 'topological', 'trivial']
LABEL_CODE_MAP = {0: 'semimetal', 1: 'topological', 2: 'trivial'}


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def get_formula(entry):
    return entry.get('reduced_formula') or entry.get('compoundName') or entry.get('formula')


def get_space_group(entry):
    value = entry.get('space_group', entry.get('symmetryGroupNumber', entry.get('sg', 0)))
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def canonical_label(raw) -> str:
    if isinstance(raw, int):
        return LABEL_CODE_MAP.get(raw, 'trivial')
    if isinstance(raw, float) and raw.is_integer():
        return LABEL_CODE_MAP.get(int(raw), 'trivial')
    label = str(raw).strip().lower()
    if label.isdigit():
        return LABEL_CODE_MAP.get(int(label), 'trivial')
    return {'sm': 'semimetal', 'tsm': 'semimetal', 'ti': 'topological'}.get(label, label)


def evaluate(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_ORDER, output_dict=True, zero_division=0)
    report['accuracy'] = float(accuracy_score(y_true, y_pred))
    report['macro_f1'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0))
    report['weighted_f1'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0))
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return report


def main():
    with DATA_PATH.open() as f:
        entries = json.load(f)
    with FEATURE_NAMES_PATH.open() as f:
        feature_names = json.load(f)['xgb_feature_names']

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')

    label_to_code = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    rows = []
    y_true = []
    y_pred = []
    for entry in entries:
        feats = input_gen_4_xgb.input_gen(entry)
        x = pd.DataFrame([feats], columns=feature_names)
        prob = model.predict_proba(x.to_numpy())[0]
        pred = int(model.predict(x.to_numpy())[0])
        true_label = canonical_label(entry.get('label', entry.get('topologicalClassificationShortDescription', 'trivial')))
        y_true.append(label_to_code[true_label])
        y_pred.append(pred)
        rows.append({
            'formula': get_formula(entry),
            'space_group': get_space_group(entry),
            'true_label': true_label,
            'predicted_label': LABEL_ORDER[pred],
            'probabilities': {label: float(prob[i]) for i, label in enumerate(LABEL_ORDER)},
        })

    metrics = evaluate(y_true, y_pred)
    dump_json(OUT / 'predictions.json', rows)
    dump_json(OUT / 'metrics.json', metrics)
    print('Discovery-space-2 XGB accuracy:', metrics['accuracy'])
    print('Discovery-space-2 XGB macro-F1:', metrics['macro_f1'])
    print('Discovery-space-2 XGB weighted-F1:', metrics['weighted_f1'])
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
