import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
PIPELINE = ROOT / 'weighted_version' / 'txl_fusion_variants' / 'common_fusion_pipeline.py'
MODEL_DIR = ROOT / 'weighted_version' / 'txl_model_pca50' / 'output'


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
    txl = load_module(PIPELINE, 'txl_model_pca50_inference_pipeline')

    clean_training_data = txl.heuristic_xgb.load_json(txl.TRAINING_DATA_PATH)
    clean_test_data = txl.heuristic_xgb.load_json(txl.TEST_DATA_PATH)
    train_idx, _ = txl.heuristic_xgb.load_shared_split(clean_training_data)

    _, _, x_test_num, y_test_labels = txl.build_numeric_feature_frames(
        clean_training_data,
        clean_test_data,
        train_idx,
    )
    test_texts = txl.get_texts(clean_test_data)

    tokenizer = AutoTokenizer.from_pretrained(str(txl.SCIBERT_CHECKPOINT))
    full_model = AutoModelForSequenceClassification.from_pretrained(str(txl.SCIBERT_CHECKPOINT))
    bert_model = full_model.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()
    test_embeddings = txl.get_scibert_embeddings(test_texts, tokenizer, bert_model, device)

    pca = joblib.load(MODEL_DIR / 'pca_model.pkl')
    llm_scaler = joblib.load(MODEL_DIR / 'llm_scaler.pkl')
    num_scaler = joblib.load(MODEL_DIR / 'numeric_scaler.pkl')
    feature_info = json.loads((MODEL_DIR / 'feature_names.json').read_text())
    routing = json.loads((MODEL_DIR / 'routing_calibration.json').read_text())
    config = json.loads((MODEL_DIR / 'config_used.json').read_text())

    test_emb_reduced = pca.transform(test_embeddings)
    llm_cols = feature_info['llm_features']
    num_cols = feature_info['numeric_features']

    llm_test = llm_scaler.transform(np.asarray(test_emb_reduced, dtype=np.float32)).astype(np.float32)
    num_test = x_test_num[num_cols].to_numpy(dtype=np.float32)
    num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)
    num_test = num_scaler.transform(num_test).astype(np.float32)

    gated = txl.GatedFusionNet(
        llm_dim=len(llm_cols),
        num_dim=len(num_cols),
        hidden_dim=int(config.get('hidden_dim', 128)),
        dropout=float(config.get('dropout', 0.2)),
        n_classes=3,
    ).to(device)
    gated.load_state_dict(torch.load(MODEL_DIR / 'gated_fusion_state.pt', map_location=device))
    gated.eval()
    gated_test_proba = txl._predict_gated_proba(gated, llm_test, num_test, device)

    x_test_full = pd.concat([
        pd.DataFrame(test_emb_reduced, columns=llm_cols).reset_index(drop=True),
        x_test_num[num_cols].reset_index(drop=True),
    ], axis=1)
    x_test_full = x_test_full[feature_info['hierarchical_features']]

    stage1 = xgb.XGBClassifier()
    stage1.load_model(str(MODEL_DIR / 'hierarchical_stage1_model.json'))
    stage1.get_booster().set_param({'device': 'cpu'})
    stage1.set_params(device='cpu')

    stage2 = xgb.XGBClassifier()
    stage2.load_model(str(MODEL_DIR / 'hierarchical_stage2_model.json'))
    stage2.get_booster().set_param({'device': 'cpu'})
    stage2.set_params(device='cpu')

    stage1_proba = stage1.predict_proba(x_test_full)
    stage2_proba = stage2.predict_proba(x_test_full)
    hier_test_proba = np.zeros((len(x_test_full), 3), dtype=np.float64)
    hier_test_proba[:, 2] = stage1_proba[:, 0]
    hier_test_proba[:, 0] = stage1_proba[:, 1] * stage2_proba[:, 0]
    hier_test_proba[:, 1] = stage1_proba[:, 1] * stage2_proba[:, 1]
    hier_test_proba = np.clip(hier_test_proba, 1e-8, 1.0)
    hier_test_proba /= hier_test_proba.sum(axis=1, keepdims=True)

    blend_weight = float(routing['hierarchy_blend_weight'])
    test_proba = np.clip(gated_test_proba, 1e-8, 1.0) * np.power(np.clip(hier_test_proba, 1e-8, 1.0), blend_weight)
    test_proba /= test_proba.sum(axis=1, keepdims=True)

    thresholds = routing['validation_search_result']['thresholds']
    y_pred = txl._predict_with_thresholds(test_proba, thresholds)
    label_to_code = {label: idx for idx, label in enumerate(txl.LABEL_ORDER)}
    y_true = np.asarray([label_to_code[label] for label in y_test_labels.tolist()], dtype=int)

    metrics = txl.evaluate(y_true, y_pred)
    dump_json(OUT / 'heldout_test_metrics.json', metrics)
    predictions = []
    for row_idx, (entry, pred, prob, true_label) in enumerate(zip(clean_test_data, y_pred, test_proba, y_test_labels.tolist())):
        predictions.append({
            'compound': txl.heuristic_xgb.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': true_label,
            'predicted_label': txl.LABEL_ORDER[int(pred)],
            'probabilities': {label: float(prob[i]) for i, label in enumerate(txl.LABEL_ORDER)},
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
