import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import input_gen_4_txl
import text_4_inference

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
DATA_PATH = Path('/path/to/external_material_records/different.json')
MODEL_DIR = ROOT / 'weighted_version' / 'txl_model_pca50' / 'output'
PIPELINE = ROOT / 'weighted_version' / 'txl_fusion_variants' / 'common_fusion_pipeline.py'
CHECKPOINT = ROOT / 'uncased_scibert_improved_input' / 'scibert-finetuned-weighted-improved-input' / 'checkpoint-2958'
LABEL_ORDER = ['semimetal', 'topological', 'trivial']
LABEL_CODE_MAP = {0: 'semimetal', 1: 'topological', 2: 'trivial'}
MAX_LENGTH = 512


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def canonical_label(raw):
    if isinstance(raw, int):
        return LABEL_CODE_MAP.get(raw, 'trivial')
    if isinstance(raw, float) and raw.is_integer():
        return LABEL_CODE_MAP.get(int(raw), 'trivial')
    label = str(raw).strip().lower()
    if label.isdigit():
        return LABEL_CODE_MAP.get(int(label), 'trivial')
    return {'sm': 'semimetal', 'tsm': 'semimetal', 'ti': 'topological'}.get(label, label)


def get_formula(entry):
    return entry.get('reduced_formula') or entry.get('compoundName') or entry.get('formula')


def get_space_group(entry):
    value = entry.get('space_group', entry.get('symmetryGroupNumber', entry.get('sg', 0)))
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def evaluate(y_true, y_pred):
    txl = load_module(PIPELINE, 'weighted_pca50_txl_pipeline_for_ds2')
    return txl.evaluate(y_true, y_pred)


def main():
    txl = load_module(PIPELINE, 'weighted_pca50_txl_pipeline_for_ds2_main')

    with DATA_PATH.open() as f:
        entries = json.load(f)
    feature_info = json.loads((MODEL_DIR / 'feature_names.json').read_text())
    pca = joblib.load(MODEL_DIR / 'pca_model.pkl')
    llm_scaler = joblib.load(MODEL_DIR / 'llm_scaler.pkl')
    num_scaler = joblib.load(MODEL_DIR / 'numeric_scaler.pkl')
    routing = json.loads((MODEL_DIR / 'routing_calibration.json').read_text())
    config = json.loads((MODEL_DIR / 'config_used.json').read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    full_model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT))
    bert_model = full_model.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()

    gated = txl.GatedFusionNet(
        llm_dim=len(feature_info['llm_features']),
        num_dim=len(feature_info['numeric_features']),
        hidden_dim=int(config.get('hidden_dim', 128)),
        dropout=float(config.get('dropout', 0.2)),
        n_classes=3,
    ).to(device)
    gated.load_state_dict(torch.load(MODEL_DIR / 'gated_fusion_state.pt', map_location=device))
    gated.eval()

    stage1 = xgb.XGBClassifier()
    stage1.load_model(str(MODEL_DIR / 'hierarchical_stage1_model.json'))
    stage1.get_booster().set_param({'device': 'cpu'})
    stage1.set_params(device='cpu')

    stage2 = xgb.XGBClassifier()
    stage2.load_model(str(MODEL_DIR / 'hierarchical_stage2_model.json'))
    stage2.get_booster().set_param({'device': 'cpu'})
    stage2.set_params(device='cpu')

    rows = []
    y_true = []
    y_pred = []
    label_to_code = {label: idx for idx, label in enumerate(LABEL_ORDER)}

    for entry in entries:
        text_payload = text_4_inference.prep_text(entry)
        inputs = tokenizer(text_payload['text'], return_tensors='pt', truncation=True, padding=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)
        emb_reduced = pca.transform(embedding)
        llm_arr = llm_scaler.transform(np.asarray(emb_reduced, dtype=np.float32)).astype(np.float32)

        numeric_row = input_gen_4_txl.input_gen(entry)
        num_df = pd.DataFrame([numeric_row])
        num_df = num_df.reindex(columns=feature_info['numeric_features']).fillna(0.0)
        num_arr = num_scaler.transform(num_df.to_numpy(dtype=np.float32)).astype(np.float32)

        gated_proba = txl._predict_gated_proba(gated, llm_arr, num_arr, device)

        x_full = pd.concat([
            pd.DataFrame(emb_reduced, columns=feature_info['llm_features']).reset_index(drop=True),
            num_df.reset_index(drop=True),
        ], axis=1)
        x_full = x_full[feature_info['hierarchical_features']]
        stage1_proba = stage1.predict_proba(x_full)
        stage2_proba = stage2.predict_proba(x_full)
        hier_proba = np.zeros((len(x_full), 3), dtype=np.float64)
        hier_proba[:, 2] = stage1_proba[:, 0]
        hier_proba[:, 0] = stage1_proba[:, 1] * stage2_proba[:, 0]
        hier_proba[:, 1] = stage1_proba[:, 1] * stage2_proba[:, 1]
        hier_proba = np.clip(hier_proba, 1e-8, 1.0)
        hier_proba /= hier_proba.sum(axis=1, keepdims=True)

        blend_weight = float(routing['hierarchy_blend_weight'])
        final_proba = np.clip(gated_proba, 1e-8, 1.0) * np.power(np.clip(hier_proba, 1e-8, 1.0), blend_weight)
        final_proba /= final_proba.sum(axis=1, keepdims=True)
        thresholds = routing['validation_search_result']['thresholds']
        pred = int(txl._predict_with_thresholds(final_proba, thresholds)[0])

        true_label = canonical_label(entry.get('label', entry.get('topologicalClassificationShortDescription', 'trivial')))
        y_true.append(label_to_code[true_label])
        y_pred.append(pred)
        rows.append({
            'formula': get_formula(entry),
            'space_group': get_space_group(entry),
            'true_label': true_label,
            'predicted_label': LABEL_ORDER[pred],
            'probabilities': {label: float(final_proba[0, i]) for i, label in enumerate(LABEL_ORDER)},
        })

    metrics = evaluate(y_true, y_pred)
    dump_json(OUT / 'predictions.json', rows)
    dump_json(OUT / 'metrics.json', metrics)
    for label in LABEL_ORDER:
        dump_json(OUT / f'predictions_{label}.json', [r for r in rows if r['predicted_label'] == label])

    print('Discovery-space-2 PCA50 TXL accuracy:', metrics['accuracy'])
    print('Discovery-space-2 PCA50 TXL macro-F1:', metrics['macro_f1'])
    print('Discovery-space-2 PCA50 TXL weighted-F1:', metrics['weighted_f1'])
    print()
    for cls in LABEL_ORDER:
        vals = metrics[cls]
        print(f"{cls}: precision={vals['precision']:.4f} recall={vals['recall']:.4f} f1-score={vals['f1-score']:.4f} support={int(vals['support'])}")


if __name__ == '__main__':
    main()
