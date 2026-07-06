import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # force the third physical GPU

import joblib
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
TXL_TRAIN_SCRIPT = ROOT / 'weighted_version' / 'txl_model' / 'train_txl_model.py'
CHECKPOINT = ROOT / 'uncased_scibert_improved_input' / 'scibert-finetuned-weighted-improved-input' / 'checkpoint-2958'
N_PCA_COMPONENTS = 50


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


def build_model(seed):
    return xgb.XGBClassifier(
        device='cuda',
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
        random_state=seed,
    )


def fit_with_fallback(x_subtrain, y_subtrain, x_val, y_val, sample_weights, seed):
    model = build_model(seed)
    try:
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    except Exception:
        print('Falling back to CPU...')
        model = build_model(seed)
        model.set_params(device='cpu')
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    return model


def main():
    txl = load_module(TXL_TRAIN_SCRIPT, 'weighted_txl_train_module_for_llm_pca100_main')
    clean_training_data = txl.heuristic_xgb.load_json(txl.TRAINING_DATA_PATH)
    clean_test_data = txl.heuristic_xgb.load_json(txl.TEST_DATA_PATH)
    train_idx, val_idx = txl.heuristic_xgb.load_shared_split(clean_training_data)

    train_texts = txl.get_texts(clean_training_data)
    test_texts = txl.get_texts(clean_test_data)

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    full_model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT))
    bert_model = full_model.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()

    train_embeddings = txl.get_scibert_embeddings(train_texts, tokenizer, bert_model, device)
    test_embeddings = txl.get_scibert_embeddings(test_texts, tokenizer, bert_model, device)

    n_pca_components = min(N_PCA_COMPONENTS, train_embeddings.shape[0], train_embeddings.shape[1])
    pca = PCA(n_components=n_pca_components, svd_solver='full')
    train_emb_reduced = pca.fit_transform(train_embeddings)
    test_emb_reduced = pca.transform(test_embeddings)
    joblib.dump(pca, OUT / 'pca_model.pkl')

    feature_names = [f'Bert_{i}' for i in range(train_emb_reduced.shape[1])]
    x_train = pd.DataFrame(train_emb_reduced, columns=feature_names)
    x_test = pd.DataFrame(test_emb_reduced, columns=feature_names)

    le = txl.LabelEncoder()
    le.fit(['topological', 'semimetal', 'trivial'])

    y_train_labels = pd.Series([canonical_label(entry['topologicalClassificationShortDescription']) for entry in clean_training_data])
    y_test_labels = pd.Series([canonical_label(entry['topologicalClassificationShortDescription']) for entry in clean_test_data])
    y_train_all = le.transform(np.asarray(y_train_labels))
    y_test = le.transform(np.asarray(y_test_labels))

    x_subtrain = x_train.iloc[train_idx]
    x_val = x_train.iloc[val_idx]
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    weights = txl.heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    model = fit_with_fallback(x_subtrain, y_subtrain, x_val, y_val, sample_weights, txl.SEED)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_prob = model.predict_proba(x_val)
    test_prob = model.predict_proba(x_test)

    val_metrics = txl.evaluate(y_val, val_pred)
    test_metrics = txl.evaluate(y_test, test_pred)

    dump_json(OUT / 'validation_metrics.json', val_metrics)
    dump_json(OUT / 'heldout_test_metrics.json', test_metrics)
    dump_json(OUT / 'class_weights.json', {str(k): v for k, v in weights.items()})
    dump_json(OUT / 'feature_names.json', {'llm_pca_50_feature_names': feature_names})
    dump_json(OUT / 'metadata.json', {
        'description': 'Semantic-only PCA-50 baseline using a PCA basis fit on the shared subtraining split and an XGBoost head trained on the reduced semantic features.',
        'pca_model': str(OUT / 'pca_model.pkl'),
        'checkpoint': str(CHECKPOINT),
        'shared_split_loader': 'txl.heuristic_xgb.load_shared_split(clean_training_data)',
        'shared_split_manifest': str(txl.SPLIT_MANIFEST_PATH),
        'n_pca_components': int(train_emb_reduced.shape[1]),
        'pca_requested_components': N_PCA_COMPONENTS,
        'pca_explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()),
        'n_subtraining_records': int(len(train_idx)),
        'n_validation_records': int(len(val_idx)),
        'n_heldout_test_records': int(len(clean_test_data)),
        'train_index_count': int(len(train_idx)),
        'validation_index_count': int(len(val_idx)),
    })

    validation_predictions = []
    for i, pred, prob in zip(val_idx, val_pred, val_prob):
        entry = clean_training_data[i]
        validation_predictions.append({
            'compound': txl.heuristic_xgb.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': str(y_train_labels.iloc[i]),
            'predicted_label': str(le.inverse_transform([int(pred)])[0]),
            'probabilities': {label: float(prob[j]) for j, label in enumerate(txl.LABEL_ORDER)},
        })

    heldout_predictions = []
    for entry, true_label, pred, prob in zip(clean_test_data, y_test_labels.tolist(), test_pred, test_prob):
        heldout_predictions.append({
            'compound': txl.heuristic_xgb.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': str(true_label),
            'predicted_label': str(le.inverse_transform([int(pred)])[0]),
            'probabilities': {label: float(prob[j]) for j, label in enumerate(txl.LABEL_ORDER)},
        })

    dump_json(OUT / 'validation_predictions.json', validation_predictions)
    dump_json(OUT / 'heldout_test_predictions.json', heldout_predictions)

    model.get_booster().save_model(str(OUT / 'llm_pca_50_xgb_model.json'))

    print('PCA requested components:', N_PCA_COMPONENTS)
    print('PCA components:', train_emb_reduced.shape[1])
    print('PCA explained variance ratio sum:', float(pca.explained_variance_ratio_.sum()))
    print('Validation accuracy:', val_metrics['accuracy'])
    print('Validation macro-F1:', val_metrics['macro_f1'])
    print('Validation weighted-F1:', val_metrics['weighted_f1'])
    print('Held-out accuracy:', test_metrics['accuracy'])
    print('Held-out macro-F1:', test_metrics['macro_f1'])
    print('Held-out weighted-F1:', test_metrics['weighted_f1'])


if __name__ == '__main__':
    main()
