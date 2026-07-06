import importlib.util
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # force the third physical GPU

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
MAIN_DIR = ROOT / 'weighted_version' / 'llm_pca_50'
TXL_TRAIN_SCRIPT = ROOT / 'weighted_version' / 'txl_model' / 'train_txl_model.py'
MODEL_PATH = MAIN_DIR / 'llm_pca_50_xgb_model.json'
PCA_MODEL_PATH = ROOT / 'weighted_version' / 'llm_pca_50' / 'pca_model.pkl'
CHECKPOINT = ROOT / 'uncased_scibert_improved_input' / 'scibert-finetuned-weighted-improved-input' / 'checkpoint-2958'


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def canonical_label(raw: str) -> str:
    label = raw.strip().lower()
    return {'sm': 'semimetal', 'tsm': 'semimetal', 'ti': 'topological'}.get(label, label)


def main():
    txl = load_module(TXL_TRAIN_SCRIPT, 'weighted_txl_train_module_for_llm_pca_50_heldout')
    clean_test_data = txl.heuristic_xgb.load_json(txl.TEST_DATA_PATH)

    test_texts = txl.get_texts(clean_test_data)

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    full_model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT))
    bert_model = full_model.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()
    test_embeddings = txl.get_scibert_embeddings(test_texts, tokenizer, bert_model, device)

    pca = joblib.load(PCA_MODEL_PATH)
    test_emb_reduced = pca.transform(test_embeddings)
    feature_names = [f'Bert_{i}' for i in range(test_emb_reduced.shape[1])]
    x_test = pd.DataFrame(test_emb_reduced, columns=feature_names)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    canonical = [canonical_label(entry['topologicalClassificationShortDescription']) for entry in clean_test_data]
    label_to_code = {label: idx for idx, label in enumerate(txl.LABEL_ORDER)}
    y_true = [label_to_code[label] for label in canonical]

    metrics = txl.evaluate(y_true, y_pred)
    dump_json(OUT / 'heldout_test_metrics.json', metrics)

    predictions = []
    for entry, pred, prob, true_label in zip(clean_test_data, y_pred, y_prob, canonical):
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


if __name__ == '__main__':
    main()
