import importlib.util
import itertools
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path('/path/to/3_classes_classification')
TRAINING_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_training_data.json'
TEST_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_test_data.json'
HEURISTIC_XGB_SCRIPT = ROOT / 'weighted_version' / 'heuristic_xgb' / 'weighted_xgboost_shared_split.py'
TEXT_HELPER_SCRIPT = ROOT / 'uncased_scibert_improved_input' / 'text_4_inference_improved.py'
SCIBERT_CHECKPOINT = ROOT / 'uncased_scibert_improved_input' / 'scibert-finetuned-weighted-improved-input' / 'checkpoint-2958'
TXL_SHAP_PATH = ROOT / 'weighted_version' / 'txl_model' / 'txl_shap_mean_abs.json'
MAX_LENGTH = 512
SEED = 42
LABEL_ORDER = ['semimetal', 'topological', 'trivial']

CURATED_NUMERIC_FEATURES = [
    'Trivial_g', 'SM_g',
    'Is_trivial_g_positive', 'Is_sm_g_positive', 'Are_both_g_negative',
    'Is_trivial_positive_sm_negative', 'Is_sm_positive_trivial_negative',
    'SG', 'Trivial_SG_prob', 'SM_SG_prob', 'TI_SG_prob', 'SG_prior_support',
    'Total_electrons', 'Is_total_electrons_even?',
    'Mean_p_valence_electrons', 'Mean_d_valence_electrons', 'Mean_f_valence_electrons',
    'Is_d_val_electrons_present?', 'Is_f_val_electrons_present?',
    'Bonding_is_mostly_covalent', 'Bonding_is_moderately_ionic',
    'Transition_metal', 'Lanthanide', 'Metalloid', 'Nonmetal',
]

TOP_NUMERIC_FEATURES = [
    'Is_total_electrons_even?', 'TI_SG_prob', 'SG', 'SM_SG_prob',
    'Mean_p_valence_electrons', 'Mean_d_valence_electrons',
    'Trivial_g', 'SM_g', 'Transition_metal', 'Lanthanide',
]

ROUTER_FEATURES = [
    'SG', 'Trivial_SG_prob', 'SM_SG_prob', 'TI_SG_prob', 'SG_prior_support',
    'Total_electrons', 'Is_total_electrons_even?', 'Mean_d_valence_electrons',
    'Mean_f_valence_electrons', 'Is_d_val_electrons_present?', 'Is_f_val_electrons_present?',
    'Bonding_is_mostly_covalent', 'Bonding_is_moderately_ionic',
    'Transition_metal', 'Lanthanide', 'Actinide', 'Metalloid', 'Nonmetal',
]


NUMERIC_FEATURE_FAMILIES = {
    'sg': ['SG', 'Trivial_SG_prob', 'SM_SG_prob', 'TI_SG_prob', 'SG_prior_support'],
    'topogivity': [
        'Trivial_g', 'SM_g',
        'Is_trivial_g_positive', 'Is_sm_g_positive', 'Are_both_g_negative',
        'Is_trivial_positive_sm_negative', 'Is_sm_positive_trivial_negative',
    ],
    'electron': [
        'Total_electrons', 'Is_total_electrons_even?',
        'Mean_p_valence_electrons', 'Mean_d_valence_electrons', 'Mean_f_valence_electrons',
    ],
    'orbital': [
        'Is_d_val_electrons_present?', 'Is_f_val_electrons_present?',
        'Is_d_and_f_val_electrons_present?',
    ],
    'bonding': [
        'Bonding_is_mostly_covalent', 'Bonding_is_moderately_ionic',
        'Bonding_is_very_covalent', 'Bonding_is_highly_ionic',
    ],
    'category': [
        'Transition_metal', 'Lanthanide', 'Actinide', 'Metalloid', 'Nonmetal',
        'Metal', 'Alkali_metal', 'Alkaline_earth_metal', 'Halogen', 'Noble_gas',
    ],
}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


heuristic_xgb = load_module(HEURISTIC_XGB_SCRIPT, 'weighted_heuristic_xgb_module_variant')
text_helper = load_module(TEXT_HELPER_SCRIPT, 'txl_text_helper_module_variant')


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def evaluate(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_ORDER, output_dict=True, zero_division=0,
    )
    report['accuracy'] = float(accuracy_score(y_true, y_pred))
    report['macro_f1'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0))
    report['weighted_f1'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0))
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return report


def _resolve_multiclass_class_weights(y, variant):
    mode = str(variant.get('class_weight_mode', 'balanced')).lower()
    if mode in {'none', 'unweighted'}:
        return None
    if mode != 'balanced':
        raise ValueError(f'Unsupported class_weight_mode: {mode}')
    weights = {int(k): float(v) for k, v in heuristic_xgb.class_weights(y).items()}
    ti_multiplier = float(variant.get('ti_weight_multiplier', 1.0))
    ti_label = LABEL_ORDER.index('topological')
    if ti_label in weights:
        weights[ti_label] *= ti_multiplier
    return weights


def _resolve_binary_class_weights(y, variant, multiplier_key=None):
    mode = str(variant.get('class_weight_mode', 'balanced')).lower()
    if mode in {'none', 'unweighted'}:
        return None
    if mode != 'balanced':
        raise ValueError(f'Unsupported class_weight_mode: {mode}')
    weights = {int(k): float(v) for k, v in heuristic_xgb.class_weights(y).items()}
    multiplier = float(variant.get(multiplier_key, 1.0)) if multiplier_key else 1.0
    if 1 in weights:
        weights[1] *= multiplier
    return weights


def _weights_to_sample_weights(y, weights):
    if weights is None:
        return None
    return np.asarray([weights[int(label)] for label in y], dtype=float)


def _dump_prediction_payload(path, y_true, y_pred, y_prob):
    dump_json(path, {
        'y_true': [int(v) for v in y_true],
        'y_pred': [int(v) for v in y_pred],
        'y_prob': [[float(x) for x in row] for row in y_prob],
        'label_order': LABEL_ORDER,
    })


def build_model(device, seed=SEED, max_depth=4, learning_rate=0.01, n_estimators=1000):
    return xgb.XGBClassifier(
        device=device,
        tree_method='hist',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
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


def get_texts(entries):
    texts = []
    for idx, entry in enumerate(entries):
        result = text_helper.prep_text(entry)
        if result is None or 'text' not in result:
            raise ValueError(f'Could not build text for entry index {idx}: {entry.get("compoundName")}')
        texts.append(result['text'])
    return texts


def get_scibert_embeddings(texts, tokenizer, model, device):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    return np.asarray(embeddings, dtype=np.float32)


def get_scibert_classifier_probabilities(texts, tokenizer, model, device):
    ordered_probs = []
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_to_position = {label: idx for idx, label in enumerate(LABEL_ORDER)}

    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()

        arranged = np.zeros(len(LABEL_ORDER), dtype=np.float32)
        for idx, prob in enumerate(probs):
            label = id2label[idx]
            arranged[label_to_position[label]] = float(prob)
        ordered_probs.append(arranged)

    return np.asarray(ordered_probs, dtype=np.float32)


def build_numeric_feature_frames(clean_training_data, clean_test_data, train_idx):
    trivial_topogivities = heuristic_xgb.load_json(heuristic_xgb.CLEAN_TOPOGIVITY_DIR / 'trivial_topogivities.json')
    sm_topogivities = heuristic_xgb.load_json(heuristic_xgb.CLEAN_TOPOGIVITY_DIR / 'sm_topogivities.json')
    subtrain_entries = [clean_training_data[i] for i in train_idx]
    sg_class_counts, sg_total_counts = heuristic_xgb.build_space_group_priors(subtrain_entries)
    train_feature_rows = heuristic_xgb.build_feature_rows(clean_training_data, sg_class_counts, sg_total_counts, trivial_topogivities, sm_topogivities)
    test_feature_rows = heuristic_xgb.build_feature_rows(clean_test_data, sg_class_counts, sg_total_counts, trivial_topogivities, sm_topogivities)
    train_df = pd.DataFrame(train_feature_rows)
    test_df = pd.DataFrame(test_feature_rows)
    return train_df.drop(columns='label'), train_df['label'], test_df.drop(columns='label'), test_df['label']


def prepare_base_data(out_dir: Path, pca_components: int = 100, split_manifest_path: Optional[Path] = None, scibert_checkpoint: Optional[Path] = None):
    clean_training_data = heuristic_xgb.load_json(TRAINING_DATA_PATH)
    clean_test_data = heuristic_xgb.load_json(TEST_DATA_PATH)
    if split_manifest_path is None:
        train_idx, val_idx = heuristic_xgb.load_shared_split(clean_training_data)
    else:
        manifest = json.loads(Path(split_manifest_path).read_text())
        train_idx = np.asarray(manifest['train_indices'])
        val_idx = np.asarray(manifest['validation_indices'])
    x_train_num, y_train_labels, x_test_num, y_test_labels = build_numeric_feature_frames(clean_training_data, clean_test_data, train_idx)

    if hasattr(text_helper, 'build_space_group_priors'):
        subtrain_entries = [clean_training_data[i] for i in train_idx]
        text_helper.sg_class_counts, text_helper.sg_total_counts = text_helper.build_space_group_priors(subtrain_entries)

    train_texts = get_texts(clean_training_data)
    test_texts = get_texts(clean_test_data)
    checkpoint = Path(scibert_checkpoint) if scibert_checkpoint is not None else SCIBERT_CHECKPOINT
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    full_model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    bert_model = full_model.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_model.to(device)
    full_model.eval()
    bert_model.eval()
    train_embeddings = get_scibert_embeddings(train_texts, tokenizer, bert_model, device)
    test_embeddings = get_scibert_embeddings(test_texts, tokenizer, bert_model, device)
    train_text_proba = get_scibert_classifier_probabilities(train_texts, tokenizer, full_model, device)
    test_text_proba = get_scibert_classifier_probabilities(test_texts, tokenizer, full_model, device)
    pca_components = min(pca_components, train_embeddings.shape[1], len(train_idx))
    pca = PCA(n_components=pca_components, random_state=SEED)
    pca.fit(train_embeddings[train_idx])
    train_emb_reduced = pca.transform(train_embeddings)
    test_emb_reduced = pca.transform(test_embeddings)
    joblib.dump(pca, out_dir / 'pca_model.pkl')
    scibert_feature_names = [f'Bert_{i}' for i in range(train_emb_reduced.shape[1])]
    train_emb_df = pd.DataFrame(train_emb_reduced, columns=scibert_feature_names)
    test_emb_df = pd.DataFrame(test_emb_reduced, columns=scibert_feature_names)
    le = LabelEncoder()
    le.fit(['topological', 'semimetal', 'trivial'])
    y_train_all = le.transform(np.asarray(y_train_labels))
    y_test = le.transform(np.asarray(y_test_labels))
    return {
        'clean_training_data': clean_training_data,
        'clean_test_data': clean_test_data,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'x_train_num': x_train_num,
        'x_test_num': x_test_num,
        'train_emb_df': train_emb_df,
        'test_emb_df': test_emb_df,
        'train_text_proba': train_text_proba,
        'test_text_proba': test_text_proba,
        'y_train_labels': y_train_labels,
        'y_test_labels': y_test_labels,
        'y_train_all': y_train_all,
        'y_test': y_test,
        'label_encoder': le,
    }


def select_pca_complementary_numeric_features(base, variant, out_dir: Path):
    x_train_num = base['x_train_num'].copy()
    train_idx = base['train_idx']
    y_subtrain = base['y_train_all'][train_idx]

    candidate_cols = [c for c in x_train_num.columns if c not in {'label'}]
    top_k_pcs = min(int(variant.get('selector_top_k_pcs', 20)), base['train_emb_df'].shape[1])
    alpha = float(variant.get('selector_corr_penalty', 2.0))
    top_n = int(variant.get('selector_top_n', 12))

    emb_sub = base['train_emb_df'].iloc[train_idx, :top_k_pcs].to_numpy(dtype=float)
    num_sub = x_train_num.iloc[train_idx][candidate_cols].copy()

    # Replace constant/invalid columns before scoring.
    valid_cols = []
    for col in candidate_cols:
        vals = num_sub[col].to_numpy(dtype=float)
        if np.all(np.isfinite(vals)) and np.nanstd(vals) > 1e-12:
            valid_cols.append(col)

    num_sub = num_sub[valid_cols].fillna(0.0)
    X = num_sub.to_numpy(dtype=float)
    f_scores, _ = f_classif(X, y_subtrain)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for j, col in enumerate(valid_cols):
        feature_vals = X[:, j]
        corrs = []
        for pc_idx in range(emb_sub.shape[1]):
            pc_vals = emb_sub[:, pc_idx]
            if np.std(feature_vals) < 1e-12 or np.std(pc_vals) < 1e-12:
                corrs.append(0.0)
            else:
                c = np.corrcoef(feature_vals, pc_vals)[0, 1]
                corrs.append(0.0 if np.isnan(c) else abs(float(c)))
        max_abs_corr = max(corrs) if corrs else 0.0
        mean_abs_corr = float(np.mean(corrs)) if corrs else 0.0
        complementarity_score = float(f_scores[j] / (1.0 + alpha * max_abs_corr))
        rows.append({
            'feature': col,
            'f_score': float(f_scores[j]),
            'max_abs_corr_to_top_llm_pcs': max_abs_corr,
            'mean_abs_corr_to_top_llm_pcs': mean_abs_corr,
            'complementarity_score': complementarity_score,
        })

    rows.sort(key=lambda r: r['complementarity_score'], reverse=True)
    selected = [row['feature'] for row in rows[:top_n]]
    dump_json(out_dir / 'pca_complementary_numeric_ranking.json', rows)
    dump_json(out_dir / 'pca_complementary_numeric_selected.json', {'selected_features': selected})
    return selected


def select_numeric_pca_ranked_features(base, variant, out_dir: Path):
    x_train_num = base['x_train_num'].copy()
    train_idx = base['train_idx']

    candidate_cols = [c for c in x_train_num.columns if c not in {'label'}]
    sub = x_train_num.iloc[train_idx][candidate_cols].copy()

    valid_cols = []
    for col in candidate_cols:
        vals = sub[col].to_numpy(dtype=float)
        if np.all(np.isfinite(vals)) and np.nanstd(vals) > 1e-12:
            valid_cols.append(col)

    sub = sub[valid_cols].fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.to_numpy(dtype=float))

    max_components = min(X.shape[0], X.shape[1])
    requested = variant.get('numeric_pca_components', 20)
    if isinstance(requested, float) and 0 < requested < 1:
        pca = PCA(n_components=requested, random_state=SEED)
    else:
        n_comp = min(int(requested), max_components)
        pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(X)

    components = pca.components_
    explained = pca.explained_variance_ratio_
    feature_scores = np.sum(np.abs(components) * explained[:, None], axis=0)

    rows = []
    for col, score in zip(valid_cols, feature_scores):
        rows.append({
            'feature': col,
            'pca_contribution_score': float(score),
        })

    rows.sort(key=lambda r: r['pca_contribution_score'], reverse=True)
    top_n = int(variant.get('numeric_pca_top_n', 15))
    selected = [row['feature'] for row in rows[:top_n]]

    dump_json(out_dir / 'numeric_pca_ranked_features.json', rows)
    dump_json(out_dir / 'numeric_pca_selected_features.json', {'selected_features': selected})
    joblib.dump(scaler, out_dir / 'numeric_only_pca_scaler.pkl')
    joblib.dump(pca, out_dir / 'numeric_only_pca_model.pkl')
    return selected




def select_pca_shap_agreement_numeric_features(base, variant, out_dir: Path):
    x_train_num = base['x_train_num'].copy()
    train_idx = base['train_idx']
    candidate_cols = [c for c in x_train_num.columns if c not in {'label'}]
    sub = x_train_num.iloc[train_idx][candidate_cols].copy()

    valid_cols = []
    for col in candidate_cols:
        vals = sub[col].to_numpy(dtype=float)
        if np.all(np.isfinite(vals)) and np.nanstd(vals) > 1e-12:
            valid_cols.append(col)

    sub = sub[valid_cols].fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.to_numpy(dtype=float))
    max_components = min(X.shape[0], X.shape[1])
    requested = variant.get('numeric_pca_components', 20)
    if isinstance(requested, float) and 0 < requested < 1:
        pca = PCA(n_components=requested, random_state=SEED)
    else:
        n_comp = min(int(requested), max_components)
        pca = PCA(n_components=n_comp, random_state=SEED)
    pca.fit(X)
    components = pca.components_
    explained = pca.explained_variance_ratio_
    pca_scores = np.sum(np.abs(components) * explained[:, None], axis=0)

    pca_rows = []
    for col, score in zip(valid_cols, pca_scores):
        pca_rows.append({'feature': col, 'pca_contribution_score': float(score)})
    pca_rows.sort(key=lambda r: r['pca_contribution_score'], reverse=True)

    shap_path = Path(variant.get('shap_importance_path', TXL_SHAP_PATH))
    if not shap_path.exists():
        raise FileNotFoundError(f'SHAP importance file not found: {shap_path}')
    shap_data = json.loads(shap_path.read_text())
    shap_rows = []
    for row in shap_data:
        feat = row.get('feature')
        if feat in valid_cols and not str(feat).startswith('Bert_'):
            shap_rows.append({'feature': feat, 'mean_abs_shap': float(row.get('mean_abs_shap', 0.0))})
    shap_rows.sort(key=lambda r: r['mean_abs_shap'], reverse=True)

    pca_top_n = int(variant.get('selector_pca_top_n', 18))
    shap_top_n = int(variant.get('selector_shap_top_n', 18))
    pca_top = [row['feature'] for row in pca_rows[:pca_top_n]]
    shap_top = [row['feature'] for row in shap_rows[:shap_top_n]]
    overlap = [feat for feat in pca_top if feat in shap_top]

    if not overlap:
        combined = {}
        for rank, feat in enumerate(pca_top, start=1):
            combined.setdefault(feat, 0.0)
            combined[feat] += 1.0 / rank
        for rank, feat in enumerate(shap_top, start=1):
            combined.setdefault(feat, 0.0)
            combined[feat] += 1.0 / rank
        overlap = [feat for feat, _ in sorted(combined.items(), key=lambda kv: kv[1], reverse=True)]

    target_n = int(variant.get('selector_target_n', min(len(overlap), 10)))
    selected = overlap[:max(1, target_n)]

    dump_json(out_dir / 'pca_numeric_ranking.json', pca_rows)
    dump_json(out_dir / 'shap_numeric_ranking.json', shap_rows)
    dump_json(out_dir / 'pca_shap_agreement_selected.json', {
        'selected_features': selected,
        'pca_top_n': pca_top_n,
        'shap_top_n': shap_top_n,
        'overlap_features': overlap,
        'shap_source': str(shap_path),
    })
    joblib.dump(scaler, out_dir / 'agreement_numeric_scaler.pkl')
    joblib.dump(pca, out_dir / 'agreement_numeric_pca.pkl')
    return selected

def make_feature_frames(base, variant):
    x_train_num = base['x_train_num'].copy()
    x_test_num = base['x_test_num'].copy()
    train_emb_df = base['train_emb_df'].copy()
    test_emb_df = base['test_emb_df'].copy()
    kind = variant.get('feature_strategy', variant.get('numeric_feature_strategy', 'concat_curated'))
    llm_scale = variant.get('llm_scale', 1.0)
    num_scale = variant.get('numeric_scale', 1.0)
    if llm_scale != 1.0:
        train_emb_df = train_emb_df * llm_scale
        test_emb_df = test_emb_df * llm_scale
    if kind == 'concat_full':
        selected_train_num = x_train_num
        selected_test_num = x_test_num
    elif kind == 'concat_curated':
        cols = [c for c in CURATED_NUMERIC_FEATURES if c in x_train_num.columns]
        selected_train_num = x_train_num[cols]
        selected_test_num = x_test_num[cols]
    elif kind == 'concat_top_numeric':
        cols = [c for c in TOP_NUMERIC_FEATURES if c in x_train_num.columns]
        selected_train_num = x_train_num[cols]
        selected_test_num = x_test_num[cols]
    elif kind == 'llm_only':
        selected_train_num = pd.DataFrame(index=x_train_num.index)
        selected_test_num = pd.DataFrame(index=x_test_num.index)
    elif kind == 'concat_pca_complementary':
        cols = select_pca_complementary_numeric_features(base, variant, variant['out_dir'])
        selected_train_num = x_train_num[cols]
        selected_test_num = x_test_num[cols]
    elif kind == 'concat_numeric_pca_ranked':
        cols = select_numeric_pca_ranked_features(base, variant, variant['out_dir'])
        selected_train_num = x_train_num[cols]
        selected_test_num = x_test_num[cols]
    elif kind == 'concat_pca_shap_agreement':
        cols = select_pca_shap_agreement_numeric_features(base, variant, variant['out_dir'])
        selected_train_num = x_train_num[cols]
        selected_test_num = x_test_num[cols]
    else:
        raise ValueError(f'Unknown feature strategy: {kind}')
    if num_scale != 1.0 and selected_train_num.shape[1] > 0:
        selected_train_num = selected_train_num * num_scale
        selected_test_num = selected_test_num * num_scale
    x_train_full = pd.concat([train_emb_df.reset_index(drop=True), selected_train_num.reset_index(drop=True)], axis=1)
    x_test_full = pd.concat([test_emb_df.reset_index(drop=True), selected_test_num.reset_index(drop=True)], axis=1)
    return x_train_full, x_test_full


def fit_simple_variant(base, variant, out_dir: Path):
    x_train_full, x_test_full = make_feature_frames(base, variant)
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']
    x_subtrain = x_train_full.iloc[train_idx]
    x_val = x_train_full.iloc[val_idx]
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)
    model = build_model('cuda', max_depth=variant.get('max_depth', 4), learning_rate=variant.get('learning_rate', 0.01), n_estimators=variant.get('n_estimators', 1000))
    try:
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    except Exception:
        model = build_model('cpu', max_depth=variant.get('max_depth', 4), learning_rate=variant.get('learning_rate', 0.01), n_estimators=variant.get('n_estimators', 1000))
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test_full)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)
    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    _dump_prediction_payload(out_dir / 'validation_predictions.json', y_val, val_pred, val_proba)
    _dump_prediction_payload(out_dir / 'heldout_test_predictions.json', y_test, test_pred, test_proba)
    dump_json(out_dir / 'validation_predictions_verbose.json', [
        {
            'compound': heuristic_xgb.normalize_formula(base['clean_training_data'][i]['compoundName']),
            'space_group': int(round(base['clean_training_data'][i].get('symmetryGroupNumber', 0))),
            'true_label': str(base['y_train_labels'].iloc[i] if hasattr(base['y_train_labels'], 'iloc') else base['y_train_labels'][i]),
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(val_proba[row_idx][j]) for j, label in enumerate(LABEL_ORDER)},
        }
        for row_idx, (i, pred) in enumerate(zip(val_idx, val_pred))
    ])
    dump_json(out_dir / 'heldout_test_predictions_verbose.json', [
        {
            'compound': heuristic_xgb.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': str(base['y_test_labels'].iloc[row_idx] if hasattr(base['y_test_labels'], 'iloc') else base['y_test_labels'][row_idx]),
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(test_proba[row_idx][j]) for j, label in enumerate(LABEL_ORDER)},
        }
        for row_idx, (entry, pred) in enumerate(zip(base['clean_test_data'], test_pred))
    ])
    dump_json(out_dir / 'feature_names.json', {'feature_names': x_train_full.columns.tolist()})
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    model.get_booster().save_model(str(out_dir / 'model.json'))
    return val_metrics, test_metrics


def fit_stacking_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']
    llm_variant = {'feature_strategy': 'llm_only', 'max_depth': variant.get('base_max_depth', 4), 'learning_rate': variant.get('base_learning_rate', 0.01), 'n_estimators': variant.get('base_n_estimators', 600)}
    x_train_llm, x_test_llm = make_feature_frames(base, llm_variant)
    x_subtrain_llm = x_train_llm.iloc[train_idx]
    x_val_llm = x_train_llm.iloc[val_idx]
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)
    llm_model = build_model('cuda', max_depth=llm_variant['max_depth'], learning_rate=llm_variant['learning_rate'], n_estimators=llm_variant['n_estimators'])
    try:
        llm_model.fit(x_subtrain_llm, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val_llm, y_val)], verbose=False)
    except Exception:
        llm_model = build_model('cpu', max_depth=llm_variant['max_depth'], learning_rate=llm_variant['learning_rate'], n_estimators=llm_variant['n_estimators'])
        llm_model.fit(x_subtrain_llm, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val_llm, y_val)], verbose=False)
    llm_model.get_booster().set_param({'device': 'cpu'})
    llm_model.set_params(device='cpu')
    num_variant = {'feature_strategy': 'concat_curated', 'llm_scale': 0.0, 'numeric_scale': 1.0, 'max_depth': variant.get('base_max_depth', 4), 'learning_rate': variant.get('base_learning_rate', 0.01), 'n_estimators': variant.get('base_n_estimators', 600)}
    x_train_num, x_test_num = make_feature_frames(base, num_variant)
    zero_cols = [c for c in x_train_num.columns if c.startswith('Bert_')]
    x_train_num = x_train_num.drop(columns=zero_cols)
    x_test_num = x_test_num.drop(columns=zero_cols)
    x_subtrain_num = x_train_num.iloc[train_idx]
    x_val_num = x_train_num.iloc[val_idx]
    num_model = build_model('cuda', max_depth=num_variant['max_depth'], learning_rate=num_variant['learning_rate'], n_estimators=num_variant['n_estimators'])
    try:
        num_model.fit(x_subtrain_num, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val_num, y_val)], verbose=False)
    except Exception:
        num_model = build_model('cpu', max_depth=num_variant['max_depth'], learning_rate=num_variant['learning_rate'], n_estimators=num_variant['n_estimators'])
        num_model.fit(x_subtrain_num, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val_num, y_val)], verbose=False)
    num_model.get_booster().set_param({'device': 'cpu'})
    num_model.set_params(device='cpu')
    llm_train_proba = llm_model.predict_proba(x_train_llm)
    llm_test_proba = llm_model.predict_proba(x_test_llm)
    num_train_proba = num_model.predict_proba(x_train_num)
    num_test_proba = num_model.predict_proba(x_test_num)
    curated_cols = [c for c in CURATED_NUMERIC_FEATURES if c in base['x_train_num'].columns]
    meta_train = pd.DataFrame(np.hstack([llm_train_proba, num_train_proba]), columns=[f'llm_p_{i}' for i in range(3)] + [f'num_p_{i}' for i in range(3)])
    meta_test = pd.DataFrame(np.hstack([llm_test_proba, num_test_proba]), columns=[f'llm_p_{i}' for i in range(3)] + [f'num_p_{i}' for i in range(3)])
    meta_train = pd.concat([meta_train, base['x_train_num'][curated_cols].reset_index(drop=True)], axis=1)
    meta_test = pd.concat([meta_test, base['x_test_num'][curated_cols].reset_index(drop=True)], axis=1)
    scaler = StandardScaler()
    meta_train.iloc[:, :] = scaler.fit_transform(meta_train)
    meta_test.iloc[:, :] = scaler.transform(meta_test)
    joblib.dump(scaler, out_dir / 'meta_scaler.pkl')
    x_subtrain_meta = meta_train.iloc[train_idx]
    x_val_meta = meta_train.iloc[val_idx]
    meta_model = build_model('cuda', max_depth=variant.get('meta_max_depth', 3), learning_rate=variant.get('meta_learning_rate', 0.03), n_estimators=variant.get('meta_n_estimators', 400))
    try:
        meta_model.fit(x_subtrain_meta, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val_meta, y_val)], verbose=False)
    except Exception:
        meta_model = build_model('cpu', max_depth=variant.get('meta_max_depth', 3), learning_rate=variant.get('meta_learning_rate', 0.03), n_estimators=variant.get('meta_n_estimators', 400))
        meta_model.fit(x_subtrain_meta, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val_meta, y_val)], verbose=False)
    meta_model.get_booster().set_param({'device': 'cpu'})
    meta_model.set_params(device='cpu')
    val_pred = meta_model.predict(x_val_meta)
    test_pred = meta_model.predict(meta_test)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)
    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {'feature_names': meta_train.columns.tolist()})
    meta_model.get_booster().save_model(str(out_dir / 'model.json'))
    return val_metrics, test_metrics



class GatedFusionNet(nn.Module):
    def __init__(self, llm_dim: int, num_dim: int, hidden_dim: int = 128, dropout: float = 0.2, n_classes: int = 3):
        super().__init__()
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        gate_hidden = max(hidden_dim // 2, 16)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, n_classes),
        )

    def forward(self, llm_x, num_x):
        llm_h = self.llm_proj(llm_x)
        num_h = self.num_proj(num_x)
        gate = self.gate_net(torch.cat([llm_h, num_h], dim=1))
        fused = gate * llm_h + (1.0 - gate) * num_h
        logits = self.classifier(fused)
        return logits, gate


def _predict_gated(model, llm_arr, num_arr, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(llm_arr), batch_size):
            llm_batch = torch.tensor(llm_arr[start:start+batch_size], dtype=torch.float32, device=device)
            num_batch = torch.tensor(num_arr[start:start+batch_size], dtype=torch.float32, device=device)
            logits, _ = model(llm_batch, num_batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def _predict_gated_proba(model, llm_arr, num_arr, device, batch_size=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(llm_arr), batch_size):
            llm_batch = torch.tensor(llm_arr[start:start+batch_size], dtype=torch.float32, device=device)
            num_batch = torch.tensor(num_arr[start:start+batch_size], dtype=torch.float32, device=device)
            logits, _ = model(llm_batch, num_batch)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs)


def fit_gated_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    feature_mode = variant.get('numeric_feature_strategy', 'concat_curated')
    if feature_mode == 'concat_top_numeric':
        num_cols = [c for c in TOP_NUMERIC_FEATURES if c in base['x_train_num'].columns]
    else:
        num_cols = [c for c in CURATED_NUMERIC_FEATURES if c in base['x_train_num'].columns]

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    num_train = base['x_train_num'][num_cols].to_numpy(dtype=np.float32)
    num_test = base['x_test_num'][num_cols].to_numpy(dtype=np.float32)
    num_train = np.nan_to_num(num_train, nan=0.0, posinf=0.0, neginf=0.0)
    num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)

    llm_scaler = StandardScaler()
    num_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    num_scaler.fit(num_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    num_train = num_scaler.transform(num_train).astype(np.float32)
    num_test = num_scaler.transform(num_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = _resolve_multiclass_class_weights(y_subtrain, variant)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GatedFusionNet(
        llm_dim=llm_train.shape[1],
        num_dim=num_train.shape[1],
        hidden_dim=variant.get('hidden_dim', 128),
        dropout=variant.get('dropout', 0.2),
        n_classes=3,
    ).to(device)

    if weights is None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        class_weight_tensor = torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.get('learning_rate', 1e-3), weight_decay=variant.get('weight_decay', 1e-4))

    train_ds = TensorDataset(
        torch.tensor(llm_train[train_idx], dtype=torch.float32),
        torch.tensor(num_train[train_idx], dtype=torch.float32),
        torch.tensor(y_subtrain, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=variant.get('batch_size', 256), shuffle=True)

    best_state = None
    best_macro_f1 = -1.0
    patience = variant.get('patience', 10)
    epochs_no_improve = 0
    n_epochs = variant.get('epochs', 80)

    for _epoch in range(n_epochs):
        model.train()
        for llm_batch, num_batch, y_batch in train_loader:
            llm_batch = llm_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(llm_batch, num_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_pred = _predict_gated(model, llm_train[val_idx], num_train[val_idx], device)
        val_metrics = evaluate(y_val, val_pred)
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    val_pred = _predict_gated(model, llm_train[val_idx], num_train[val_idx], device)
    test_pred = _predict_gated(model, llm_test, num_test, device)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'llm_features': list(base['train_emb_df'].columns),
        'numeric_features': num_cols,
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    torch.save(best_state, out_dir / 'gated_fusion_state.pt')
    return val_metrics, test_metrics



def fit_gated_hierarchical_calibrated_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    feature_mode = variant.get('numeric_feature_strategy', 'concat_curated')
    if feature_mode == 'concat_top_numeric':
        num_cols = [c for c in TOP_NUMERIC_FEATURES if c in base['x_train_num'].columns]
    else:
        num_cols = [c for c in CURATED_NUMERIC_FEATURES if c in base['x_train_num'].columns]

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    num_train = base['x_train_num'][num_cols].to_numpy(dtype=np.float32)
    num_test = base['x_test_num'][num_cols].to_numpy(dtype=np.float32)
    num_train = np.nan_to_num(num_train, nan=0.0, posinf=0.0, neginf=0.0)
    num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)

    llm_scaler = StandardScaler()
    num_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    num_scaler.fit(num_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    num_train = num_scaler.transform(num_train).astype(np.float32)
    num_test = num_scaler.transform(num_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = _resolve_multiclass_class_weights(y_subtrain, variant)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gated = GatedFusionNet(
        llm_dim=llm_train.shape[1],
        num_dim=num_train.shape[1],
        hidden_dim=variant.get('hidden_dim', 128),
        dropout=variant.get('dropout', 0.2),
        n_classes=3,
    ).to(device)

    if weights is None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        class_weight_tensor = torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    optimizer = torch.optim.AdamW(gated.parameters(), lr=variant.get('learning_rate', 1e-3), weight_decay=variant.get('weight_decay', 1e-4))

    train_ds = TensorDataset(
        torch.tensor(llm_train[train_idx], dtype=torch.float32),
        torch.tensor(num_train[train_idx], dtype=torch.float32),
        torch.tensor(y_subtrain, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=variant.get('batch_size', 256), shuffle=True)

    best_state = None
    best_macro_f1 = -1.0
    patience = variant.get('patience', 10)
    epochs_no_improve = 0
    n_epochs = variant.get('epochs', 80)

    for _epoch in range(n_epochs):
        gated.train()
        for llm_batch, num_batch, y_batch in train_loader:
            llm_batch = llm_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = gated(llm_batch, num_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_pred = _predict_gated(gated, llm_train[val_idx], num_train[val_idx], device)
        val_metrics = evaluate(y_val, val_pred)
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_state = {k: v.detach().cpu().clone() for k, v in gated.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in gated.state_dict().items()}
    gated.load_state_dict(best_state)

    x_train_full, x_test_full = make_feature_frames(base, variant)
    x_subtrain = x_train_full.iloc[train_idx]
    x_val = x_train_full.iloc[val_idx]
    x_test = x_test_full

    y_subtrain_stage1 = (y_subtrain != 2).astype(int)
    y_val_stage1 = (y_val != 2).astype(int)
    y_test_stage1 = (y_test != 2).astype(int)

    weights_stage1 = _resolve_binary_class_weights(y_subtrain_stage1, variant, 'stage1_positive_multiplier')
    sample_weights_stage1 = _weights_to_sample_weights(y_subtrain_stage1, weights_stage1)
    stage1 = build_binary_model('cuda', max_depth=variant.get('stage1_max_depth', 4), learning_rate=variant.get('stage1_learning_rate', 0.01), n_estimators=variant.get('stage1_n_estimators', 800))
    try:
        fit_kwargs = {'eval_set': [(x_val, y_val_stage1)], 'verbose': False}
        if sample_weights_stage1 is not None:
            fit_kwargs['sample_weight'] = sample_weights_stage1
        stage1.fit(x_subtrain, y_subtrain_stage1, **fit_kwargs)
    except Exception:
        stage1 = build_binary_model('cpu', max_depth=variant.get('stage1_max_depth', 4), learning_rate=variant.get('stage1_learning_rate', 0.01), n_estimators=variant.get('stage1_n_estimators', 800))
        fit_kwargs = {'eval_set': [(x_val, y_val_stage1)], 'verbose': False}
        if sample_weights_stage1 is not None:
            fit_kwargs['sample_weight'] = sample_weights_stage1
        stage1.fit(x_subtrain, y_subtrain_stage1, **fit_kwargs)
    stage1.get_booster().set_param({'device': 'cpu'})
    stage1.set_params(device='cpu')

    nontrivial_sub_mask = y_subtrain != 2
    nontrivial_val_mask = y_val != 2
    x_subtrain_stage2 = x_subtrain.iloc[nontrivial_sub_mask]
    y_subtrain_stage2 = y_subtrain[nontrivial_sub_mask]
    x_val_stage2 = x_val.iloc[nontrivial_val_mask]
    y_val_stage2 = y_val[nontrivial_val_mask]
    y_subtrain_stage2_bin = (y_subtrain_stage2 == 1).astype(int)
    y_val_stage2_bin = (y_val_stage2 == 1).astype(int)

    if 'stage2_topological_multiplier' not in variant and 'ti_weight_multiplier' in variant:
        variant['stage2_topological_multiplier'] = variant['ti_weight_multiplier']
    weights_stage2 = _resolve_binary_class_weights(y_subtrain_stage2_bin, variant, 'stage2_topological_multiplier')
    sample_weights_stage2 = _weights_to_sample_weights(y_subtrain_stage2_bin, weights_stage2)
    stage2 = build_binary_model('cuda', max_depth=variant.get('stage2_max_depth', 4), learning_rate=variant.get('stage2_learning_rate', 0.01), n_estimators=variant.get('stage2_n_estimators', 800))
    try:
        fit_kwargs = {'eval_set': [(x_val_stage2, y_val_stage2_bin)], 'verbose': False}
        if sample_weights_stage2 is not None:
            fit_kwargs['sample_weight'] = sample_weights_stage2
        stage2.fit(x_subtrain_stage2, y_subtrain_stage2_bin, **fit_kwargs)
    except Exception:
        stage2 = build_binary_model('cpu', max_depth=variant.get('stage2_max_depth', 4), learning_rate=variant.get('stage2_learning_rate', 0.01), n_estimators=variant.get('stage2_n_estimators', 800))
        fit_kwargs = {'eval_set': [(x_val_stage2, y_val_stage2_bin)], 'verbose': False}
        if sample_weights_stage2 is not None:
            fit_kwargs['sample_weight'] = sample_weights_stage2
        stage2.fit(x_subtrain_stage2, y_subtrain_stage2_bin, **fit_kwargs)
    stage2.get_booster().set_param({'device': 'cpu'})
    stage2.set_params(device='cpu')

    hierarchy_blend_weight = float(variant.get('hierarchy_blend_weight', 0.75))
    hierarchy_blend_grid = variant.get('hierarchy_blend_grid')
    if hierarchy_blend_grid is None:
        hierarchy_blend_grid = [hierarchy_blend_weight]
    hierarchy_blend_grid = [float(v) for v in hierarchy_blend_grid]
    threshold_objective = variant.get('threshold_objective', 'macro_f1')
    ti_objective_weight = float(variant.get('ti_objective_weight', 0.0))
    accuracy_objective_weight = float(variant.get('accuracy_objective_weight', 0.0))

    def hierarchy_proba(stage1_model, stage2_model, x_df):
        stage1_proba = stage1_model.predict_proba(x_df)
        stage2_proba = stage2_model.predict_proba(x_df)
        out = np.zeros((len(x_df), 3), dtype=np.float64)
        out[:, 2] = stage1_proba[:, 0]
        out[:, 0] = stage1_proba[:, 1] * stage2_proba[:, 0]
        out[:, 1] = stage1_proba[:, 1] * stage2_proba[:, 1]
        out = np.clip(out, 1e-8, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def blend_proba(gated_proba, hier_proba, blend_weight):
        blended = np.clip(gated_proba, 1e-8, 1.0) * np.power(np.clip(hier_proba, 1e-8, 1.0), blend_weight)
        blended /= blended.sum(axis=1, keepdims=True)
        return blended

    gated_val_proba = _predict_gated_proba(gated, llm_train[val_idx], num_train[val_idx], device)
    gated_test_proba = _predict_gated_proba(gated, llm_test, num_test, device)
    hier_val_proba = hierarchy_proba(stage1, stage2, x_val)
    hier_test_proba = hierarchy_proba(stage1, stage2, x_test)

    best = None
    best_val_proba = None
    for blend_weight in hierarchy_blend_grid:
        candidate_val_proba = blend_proba(gated_val_proba, hier_val_proba, blend_weight)
        candidate = _search_thresholds(
            y_val,
            candidate_val_proba,
            metric=threshold_objective,
            ti_weight=ti_objective_weight,
            accuracy_weight=accuracy_objective_weight,
        )
        candidate['hierarchy_blend_weight'] = float(blend_weight)
        if best is None or candidate['score'] > best['score'] or (
            np.isclose(candidate['score'], best['score']) and candidate['metrics']['accuracy'] > best['metrics']['accuracy']
        ):
            best = candidate
            best_val_proba = candidate_val_proba

    assert best is not None
    selected_hierarchy_blend_weight = float(best['hierarchy_blend_weight'])
    val_proba = best_val_proba
    test_proba = blend_proba(gated_test_proba, hier_test_proba, selected_hierarchy_blend_weight)
    thresholds = best['thresholds']
    val_pred = _predict_with_thresholds(val_proba, thresholds)
    test_pred = _predict_with_thresholds(test_proba, thresholds)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    _dump_prediction_payload(out_dir / 'validation_predictions.json', y_val, val_pred, val_proba)
    _dump_prediction_payload(out_dir / 'heldout_test_predictions.json', y_test, test_pred, test_proba)
    dump_json(out_dir / 'validation_predictions_verbose.json', [
        {
            'compound': heuristic_xgb.normalize_formula(base['clean_training_data'][i]['compoundName']),
            'space_group': int(round(base['clean_training_data'][i].get('symmetryGroupNumber', 0))),
            'true_label': str(base['y_train_labels'].iloc[i] if hasattr(base['y_train_labels'], 'iloc') else base['y_train_labels'][i]),
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(val_proba[row_idx][j]) for j, label in enumerate(LABEL_ORDER)},
        }
        for row_idx, (i, pred) in enumerate(zip(val_idx, val_pred))
    ])
    dump_json(out_dir / 'heldout_test_predictions_verbose.json', [
        {
            'compound': heuristic_xgb.normalize_formula(entry['compoundName']),
            'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
            'true_label': str(base['y_test_labels'].iloc[row_idx] if hasattr(base['y_test_labels'], 'iloc') else base['y_test_labels'][row_idx]),
            'predicted_label': LABEL_ORDER[int(pred)],
            'probabilities': {label: float(test_proba[row_idx][j]) for j, label in enumerate(LABEL_ORDER)},
        }
        for row_idx, (entry, pred) in enumerate(zip(base['clean_test_data'], test_pred))
    ])
    dump_json(out_dir / 'feature_names.json', {
        'llm_features': list(base['train_emb_df'].columns),
        'numeric_features': num_cols,
        'hierarchical_features': list(x_train_full.columns),
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'hierarchical_class_weights_stage1.json', {str(k): v for k, v in (weights_stage1 or {}).items()})
    dump_json(out_dir / 'hierarchical_class_weights_stage2.json', {str(k): v for k, v in (weights_stage2 or {}).items()})
    dump_json(out_dir / 'routing_calibration.json', {
        'hierarchy_blend_weight': selected_hierarchy_blend_weight,
        'hierarchy_blend_grid': hierarchy_blend_grid,
        'threshold_objective': threshold_objective,
        'ti_objective_weight': ti_objective_weight,
        'accuracy_objective_weight': accuracy_objective_weight,
        'validation_search_result': best,
    })
    torch.save(best_state, out_dir / 'gated_fusion_state.pt')
    stage1.get_booster().save_model(str(out_dir / 'hierarchical_stage1_model.json'))
    stage2.get_booster().save_model(str(out_dir / 'hierarchical_stage2_model.json'))
    return val_metrics, test_metrics


def fit_text_anchored_gated_hierarchical_variant(base, variant, out_dir: Path):
    # First train the existing gated + hierarchical branch, then add a text-anchor expert
    # on top of the saved models and recalibrate the final decision rule.
    fit_gated_hierarchical_calibrated_variant(base, variant, out_dir)

    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    feature_mode = variant.get('numeric_feature_strategy', 'concat_curated')
    if feature_mode == 'concat_top_numeric':
        num_cols = [c for c in TOP_NUMERIC_FEATURES if c in base['x_train_num'].columns]
    else:
        num_cols = [c for c in CURATED_NUMERIC_FEATURES if c in base['x_train_num'].columns]

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    num_train = base['x_train_num'][num_cols].to_numpy(dtype=np.float32)
    num_test = base['x_test_num'][num_cols].to_numpy(dtype=np.float32)
    num_train = np.nan_to_num(num_train, nan=0.0, posinf=0.0, neginf=0.0)
    num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)

    llm_scaler = StandardScaler()
    num_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    num_scaler.fit(num_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    num_train = num_scaler.transform(num_train).astype(np.float32)
    num_test = num_scaler.transform(num_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    text_dir = out_dir / 'outputs' / 'text_anchor'
    text_dir.mkdir(parents=True, exist_ok=True)
    text_train_proba = np.asarray(base['train_text_proba'], dtype=np.float32)
    text_val_proba = text_train_proba[val_idx]
    text_test_proba = np.asarray(base['test_text_proba'], dtype=np.float32)
    dump_json(text_dir / 'metrics_val.json', evaluate(y_val, np.argmax(text_val_proba, axis=1)))
    dump_json(text_dir / 'metrics_test.json', evaluate(y_test, np.argmax(text_test_proba, axis=1)))
    _save_probs_npy(text_dir / 'subtrain_probs.npy', text_train_proba[train_idx])
    _save_probs_npy(text_dir / 'val_probs.npy', text_val_proba)
    _save_probs_npy(text_dir / 'test_probs.npy', text_test_proba)
    _save_probs_npy(text_dir / 'subtrain_pred.npy', np.argmax(text_train_proba, axis=1)[train_idx])
    _save_probs_npy(text_dir / 'val_pred.npy', np.argmax(text_val_proba, axis=1))
    _save_probs_npy(text_dir / 'test_pred.npy', np.argmax(text_test_proba, axis=1))
    _uncertainty_frame(text_train_proba[train_idx], 'subtrain').to_csv(text_dir / 'subtrain_uncertainty.csv', index=False)
    _uncertainty_frame(text_val_proba, 'validation').to_csv(text_dir / 'val_uncertainty.csv', index=False)
    _uncertainty_frame(text_test_proba, 'heldout_test').to_csv(text_dir / 'test_uncertainty.csv', index=False)
    dump_json(text_dir / 'text_anchor_source.json', {
        'type': 'direct_scibert_classifier_probabilities',
        'checkpoint': str(SCIBERT_CHECKPOINT),
        'label_order': LABEL_ORDER,
        'note': 'These probabilities come directly from the fine-tuned SciBERT sequence-classification head, matching standalone LLM inference semantics rather than an auxiliary XGBoost-on-embeddings model.'
    })

    base_routing_path = out_dir / 'routing_calibration.json'
    if not base_routing_path.exists():
        raise FileNotFoundError(f'Missing base routing calibration file: {base_routing_path}')
    base_routing = json.loads(base_routing_path.read_text())
    base_hierarchy_blend_weight = float(base_routing.get('hierarchy_blend_weight', 0.75))
    base_thresholds = base_routing.get('validation_search_result', {}).get('thresholds', [0.5, 0.5, 0.5])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gated = GatedFusionNet(
        llm_dim=llm_train.shape[1],
        num_dim=num_train.shape[1],
        hidden_dim=variant.get('hidden_dim', 128),
        dropout=variant.get('dropout', 0.2),
        n_classes=3,
    ).to(device)
    gated.load_state_dict(torch.load(out_dir / 'gated_fusion_state.pt', map_location=device))
    gated.eval()

    stage1 = build_binary_model(
        'cpu',
        max_depth=variant.get('stage1_max_depth', 4),
        learning_rate=variant.get('stage1_learning_rate', 0.01),
        n_estimators=variant.get('stage1_n_estimators', 800),
    )
    stage1.load_model(str(out_dir / 'hierarchical_stage1_model.json'))
    stage2 = build_binary_model(
        'cpu',
        max_depth=variant.get('stage2_max_depth', 4),
        learning_rate=variant.get('stage2_learning_rate', 0.01),
        n_estimators=variant.get('stage2_n_estimators', 800),
    )
    stage2.load_model(str(out_dir / 'hierarchical_stage2_model.json'))

    def hierarchy_proba(stage1_model, stage2_model, x_df):
        stage1_proba = stage1_model.predict_proba(x_df)
        stage2_proba = stage2_model.predict_proba(x_df)
        out = np.zeros((len(x_df), 3), dtype=np.float64)
        out[:, 2] = stage1_proba[:, 0]
        out[:, 0] = stage1_proba[:, 1] * stage2_proba[:, 0]
        out[:, 1] = stage1_proba[:, 1] * stage2_proba[:, 1]
        out = np.clip(out, 1e-8, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def blend_proba(base_proba, correction_proba, alpha):
        base = np.clip(np.asarray(base_proba, dtype=float), 1e-8, 1.0)
        corr = np.clip(np.asarray(correction_proba, dtype=float), 1e-8, 1.0)
        blended = np.power(base, 1.0 - alpha) * np.power(corr, alpha)
        blended /= blended.sum(axis=1, keepdims=True)
        return blended

    gated_val_proba = _predict_gated_proba(gated, llm_train[val_idx], num_train[val_idx], device)
    gated_test_proba = _predict_gated_proba(gated, llm_test, num_test, device)
    hier_val_proba = hierarchy_proba(stage1, stage2, make_feature_frames(base, variant)[0].iloc[val_idx])
    hier_test_proba = hierarchy_proba(stage1, stage2, make_feature_frames(base, variant)[1])

    base_val_proba = blend_proba(gated_val_proba, hier_val_proba, base_hierarchy_blend_weight)
    base_test_proba = blend_proba(gated_test_proba, hier_test_proba, base_hierarchy_blend_weight)

    threshold_objective = variant.get('threshold_objective', 'macro_f1')
    ti_objective_weight = float(variant.get('ti_objective_weight', 0.0))
    accuracy_objective_weight = float(variant.get('accuracy_objective_weight', 0.0))
    text_anchor_grid = [float(v) for v in variant.get('text_anchor_grid', [0.0, 0.15, 0.3, 0.5, 0.75])]

    base_val_metrics = evaluate(y_val, _predict_with_thresholds(base_val_proba, base_thresholds))
    base_test_metrics = evaluate(y_test, _predict_with_thresholds(base_test_proba, base_thresholds))
    dump_json(out_dir / 'base_hierarchical_validation_metrics.json', base_val_metrics)
    dump_json(out_dir / 'base_hierarchical_heldout_test_metrics.json', base_test_metrics)

    best = None
    best_val_proba = None
    for text_anchor_weight in text_anchor_grid:
        candidate_val_proba = blend_proba(text_val_proba, base_val_proba, text_anchor_weight)
        candidate = _search_thresholds(
            y_val,
            candidate_val_proba,
            metric=threshold_objective,
            ti_weight=ti_objective_weight,
            accuracy_weight=accuracy_objective_weight,
        )
        candidate['text_anchor_weight'] = float(text_anchor_weight)
        if best is None or candidate['score'] > best['score'] or (
            np.isclose(candidate['score'], best['score']) and candidate['metrics']['accuracy'] > best['metrics']['accuracy']
        ):
            best = candidate
            best_val_proba = candidate_val_proba

    assert best is not None
    selected_text_anchor_weight = float(best['text_anchor_weight'])
    final_val_proba = best_val_proba
    final_test_proba = blend_proba(text_test_proba, base_test_proba, selected_text_anchor_weight)
    thresholds = best['thresholds']
    val_pred = _predict_with_thresholds(final_val_proba, thresholds)
    test_pred = _predict_with_thresholds(final_test_proba, thresholds)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'llm_features': list(base['train_emb_df'].columns),
        'numeric_features': num_cols,
        'hierarchical_features': list(make_feature_frames(base, variant)[0].columns),
        'text_anchor_features': [f'text_prob_{label}' for label in LABEL_ORDER],
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'hierarchical_class_weights_stage1.json', {str(k): v for k, v in heuristic_xgb.class_weights((y_subtrain != 2).astype(int)).items()})
    dump_json(out_dir / 'hierarchical_class_weights_stage2.json', {str(k): v for k, v in heuristic_xgb.class_weights((y_subtrain[y_subtrain != 2] == 1).astype(int)).items()})
    dump_json(out_dir / 'routing_calibration.json', {
        'hierarchy_blend_weight': base_hierarchy_blend_weight,
        'text_anchor_weight': selected_text_anchor_weight,
        'text_anchor_grid': text_anchor_grid,
        'threshold_objective': threshold_objective,
        'ti_objective_weight': ti_objective_weight,
        'accuracy_objective_weight': accuracy_objective_weight,
        'base_validation_search_result': base_routing.get('validation_search_result', {}),
        'text_anchor_validation_search_result': best,
    })
    return val_metrics, test_metrics


def build_binary_model(device, seed=SEED, max_depth=4, learning_rate=0.01, n_estimators=1000):
    return xgb.XGBClassifier(
        device=device,
        tree_method='hist',
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=2.0,
        early_stopping_rounds=20,
        random_state=seed,
    )


def _predict_with_thresholds(proba, thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    preds = []
    for row in np.asarray(proba, dtype=float):
        eligible = np.where(row >= thresholds)[0]
        if eligible.size:
            pred = int(eligible[np.argmax(row[eligible])])
        else:
            pred = int(np.argmax(row))
        preds.append(pred)
    return np.asarray(preds, dtype=int)


def _minimum_class_f1(metrics):
    return float(min(metrics[label]['f1-score'] for label in LABEL_ORDER))


def _objective_score(metrics, metric='macro_f1', ti_weight=0.0, accuracy_weight=0.0, min_class_weight=0.0):
    if metric == 'macro_f1_plus_ti_f1':
        return float(
            metrics['macro_f1']
            + ti_weight * metrics['topological']['f1-score']
            + min_class_weight * _minimum_class_f1(metrics)
            + accuracy_weight * metrics['accuracy']
        )
    if metric == 'macro_f1_plus_ti_recall':
        return float(
            metrics['macro_f1']
            + ti_weight * metrics['topological']['recall']
            + min_class_weight * _minimum_class_f1(metrics)
            + accuracy_weight * metrics['accuracy']
        )
    if metric == 'macro_f1_plus_min_class_f1':
        return float(metrics['macro_f1'] + min_class_weight * _minimum_class_f1(metrics) + accuracy_weight * metrics['accuracy'])
    if metric == 'ti_f1':
        return float(metrics['topological']['f1-score'] + min_class_weight * _minimum_class_f1(metrics) + accuracy_weight * metrics['accuracy'])
    if metric == 'ti_recall':
        return float(metrics['topological']['recall'] + min_class_weight * _minimum_class_f1(metrics) + accuracy_weight * metrics['accuracy'])
    return float(metrics[metric])


def _search_thresholds(y_true, proba, metric='macro_f1', ti_weight=0.0, accuracy_weight=0.0, min_class_weight=0.0):
    grid = np.round(np.linspace(0.30, 0.80, 11), 2)
    best = {'thresholds': [0.5, 0.5, 0.5], 'metrics': None, 'score': -1.0}
    for t0, t1, t2 in itertools.product(grid, repeat=3):
        preds = _predict_with_thresholds(proba, [t0, t1, t2])
        metrics = evaluate(y_true, preds)
        score = _objective_score(
            metrics,
            metric=metric,
            ti_weight=ti_weight,
            accuracy_weight=accuracy_weight,
            min_class_weight=min_class_weight,
        )
        if (score > best['score']) or (
            np.isclose(score, best['score']) and metrics['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)
        ):
            best = {'thresholds': [float(t0), float(t1), float(t2)], 'metrics': metrics, 'score': float(score)}
    return best




def _proba_to_logits(proba):
    return np.log(np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0))


def _softmax_rows(logits):
    arr = np.asarray(logits, dtype=float)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)


def _predict_with_class_biases(proba, biases):
    logits = _proba_to_logits(proba) + np.asarray(biases, dtype=float)[None, :]
    return np.argmax(logits, axis=1)


def _search_class_biases(y_true, proba, metric='macro_f1', ti_weight=0.0, accuracy_weight=0.0, min_class_weight=0.0, bias_grid=None):
    grid = np.asarray(bias_grid if bias_grid is not None else [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], dtype=float)
    best = {'biases': [0.0, 0.0, 0.0], 'metrics': None, 'score': -1.0}
    for b0, b1, b2 in itertools.product(grid, repeat=3):
        preds = _predict_with_class_biases(proba, [b0, b1, b2])
        metrics = evaluate(y_true, preds)
        score = _objective_score(
            metrics,
            metric=metric,
            ti_weight=ti_weight,
            accuracy_weight=accuracy_weight,
            min_class_weight=min_class_weight,
        )
        if (score > best['score']) or (
            np.isclose(score, best['score']) and metrics['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)
        ):
            best = {'biases': [float(b0), float(b1), float(b2)], 'metrics': metrics, 'score': float(score)}
    return best


def _resolve_numeric_feature_columns(base, feature_mode):
    x_train_num = base['x_train_num']
    if feature_mode == 'concat_top_numeric':
        return [c for c in TOP_NUMERIC_FEATURES if c in x_train_num.columns]
    if feature_mode == 'concat_full':
        return [c for c in x_train_num.columns if c != 'label']
    return [c for c in CURATED_NUMERIC_FEATURES if c in x_train_num.columns]


def _kl_divergence_rows(p, q):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-8, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1, keepdims=True)


def _build_reliability_features(text_proba, numeric_proba):
    text_proba = np.clip(np.asarray(text_proba, dtype=float), 1e-8, 1.0)
    numeric_proba = np.clip(np.asarray(numeric_proba, dtype=float), 1e-8, 1.0)
    text_summary = _probability_summary_features(text_proba)
    numeric_summary = _probability_summary_features(numeric_proba)
    abs_diff = np.abs(text_proba - numeric_proba)
    text_pred = np.argmax(text_proba, axis=1)
    numeric_pred = np.argmax(numeric_proba, axis=1)
    disagree = (text_pred != numeric_pred).astype(float)[:, None]
    text_margin = text_summary[:, -1:]
    numeric_margin = numeric_summary[:, -1:]
    confidence_gap = np.abs(text_summary[:, 3:4] - numeric_summary[:, 3:4])
    margin_gap = np.abs(text_margin - numeric_margin)
    kl_tn = _kl_divergence_rows(text_proba, numeric_proba)
    kl_nt = _kl_divergence_rows(numeric_proba, text_proba)
    return np.hstack([
        text_summary,
        numeric_summary,
        abs_diff,
        confidence_gap,
        margin_gap,
        kl_tn,
        kl_nt,
        disagree,
    ])


def _blend_text_anchor_probabilities(text_proba, correction_proba, gate_scores, lambda_strength):
    text_logits = _proba_to_logits(text_proba)
    correction_logits = _proba_to_logits(correction_proba)
    gate = np.clip(np.asarray(gate_scores, dtype=float).reshape(-1, 1), 0.0, 1.0)
    blended_logits = text_logits + float(lambda_strength) * gate * (correction_logits - text_logits)
    return _softmax_rows(blended_logits)


def _build_rule_based_gate_scores(text_proba, numeric_proba, *, high_margin=0.55, medium_margin=0.20, disagree_gate=0.85, uncertain_gate=0.65, mid_gate=0.20, same_high_conf_gate=0.0):
    text_proba = np.clip(np.asarray(text_proba, dtype=float), 1e-8, 1.0)
    numeric_proba = np.clip(np.asarray(numeric_proba, dtype=float), 1e-8, 1.0)
    text_summary = _probability_summary_features(text_proba)
    text_margin = text_summary[:, -1]
    text_pred = np.argmax(text_proba, axis=1)
    numeric_pred = np.argmax(numeric_proba, axis=1)
    disagree = text_pred != numeric_pred
    gate = np.full(text_margin.shape[0], mid_gate, dtype=float)
    gate[text_margin <= medium_margin] = uncertain_gate
    gate[(text_margin > high_margin) & (~disagree)] = same_high_conf_gate
    gate[disagree & (text_margin <= high_margin)] = np.maximum(gate[disagree & (text_margin <= high_margin)], disagree_gate)
    gate[disagree & (text_margin > high_margin)] = np.maximum(gate[disagree & (text_margin > high_margin)], 0.35)
    return np.clip(gate, 0.0, 1.0)


def _passes_llm_floor(candidate_metrics, llm_metrics, variant):
    if not bool(variant.get('enforce_llm_floor', False)):
        return True
    semimetal_drop = float(variant.get('semimetal_allowed_drop', 0.002))
    trivial_drop = float(variant.get('trivial_allowed_drop', 0.002))
    ti_floor_delta = float(variant.get('ti_floor_delta', 0.0))
    macro_floor_delta = float(variant.get('macro_f1_floor_delta', 0.0))
    accuracy_floor_delta = float(variant.get('accuracy_floor_delta', -1.0))
    if candidate_metrics['macro_f1'] < llm_metrics['macro_f1'] + macro_floor_delta:
        return False
    if candidate_metrics['topological']['f1-score'] < llm_metrics['topological']['f1-score'] + ti_floor_delta:
        return False
    if candidate_metrics['semimetal']['f1-score'] < llm_metrics['semimetal']['f1-score'] - semimetal_drop:
        return False
    if candidate_metrics['trivial']['f1-score'] < llm_metrics['trivial']['f1-score'] - trivial_drop:
        return False
    if candidate_metrics['accuracy'] < llm_metrics['accuracy'] + accuracy_floor_delta:
        return False
    return True


def _fit_binary_gate_with_fallback(x_subtrain, y_subtrain, x_val, y_val, sample_weights, *, seed=SEED, max_depth=3, learning_rate=0.03, n_estimators=400):
    model = build_binary_model('cuda', seed=seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
    try:
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    except Exception:
        model = build_binary_model('cpu', seed=seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    return model


def _fit_text_anchor_residual_core(
    base,
    variant,
    out_dir: Path,
    *,
    numeric_block_train,
    numeric_block_test,
    numeric_feature_names,
    numeric_train_proba=None,
    numeric_val_proba=None,
    numeric_test_proba=None,
    summary_name='text_anchor_residual.json',
):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    numeric_block_train = np.nan_to_num(np.asarray(numeric_block_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    numeric_block_test = np.nan_to_num(np.asarray(numeric_block_test, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    llm_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')

    num_scaler = StandardScaler()
    if numeric_block_train.shape[1] > 0:
        num_scaler.fit(numeric_block_train[train_idx])
        numeric_block_train = num_scaler.transform(numeric_block_train).astype(np.float32)
        numeric_block_test = num_scaler.transform(numeric_block_test).astype(np.float32)
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    text_model = _fit_xgb_with_fallback(
        llm_train[train_idx],
        y_subtrain,
        llm_train[val_idx],
        y_val,
        sample_weights,
        max_depth=variant.get('text_max_depth', 4),
        learning_rate=variant.get('text_learning_rate', 0.01),
        n_estimators=variant.get('text_n_estimators', 1000),
    )
    text_train_proba = text_model.predict_proba(llm_train)
    text_val_proba = text_model.predict_proba(llm_train[val_idx])
    text_test_proba = text_model.predict_proba(llm_test)

    numeric_model = None
    if numeric_train_proba is None or numeric_val_proba is None or numeric_test_proba is None:
        if numeric_block_train.shape[1] == 0:
            raise ValueError('Numeric block is empty, but no numeric probability override was provided.')
        numeric_model = _fit_xgb_with_fallback(
            numeric_block_train[train_idx],
            y_subtrain,
            numeric_block_train[val_idx],
            y_val,
            sample_weights,
            max_depth=variant.get('numeric_max_depth', 4),
            learning_rate=variant.get('numeric_learning_rate', 0.01),
            n_estimators=variant.get('numeric_n_estimators', 1000),
        )
        numeric_train_proba = numeric_model.predict_proba(numeric_block_train)
        numeric_val_proba = numeric_model.predict_proba(numeric_block_train[val_idx])
        numeric_test_proba = numeric_model.predict_proba(numeric_block_test)
    else:
        numeric_train_proba = np.asarray(numeric_train_proba, dtype=float)
        numeric_val_proba = np.asarray(numeric_val_proba, dtype=float)
        numeric_test_proba = np.asarray(numeric_test_proba, dtype=float)

    reliability_train = _build_reliability_features(text_train_proba, numeric_train_proba)
    reliability_val = _build_reliability_features(text_val_proba, numeric_val_proba)
    reliability_test = _build_reliability_features(text_test_proba, numeric_test_proba)

    if numeric_block_train.shape[1] > 0:
        meta_train = np.hstack([reliability_train, numeric_block_train])
        meta_val = np.hstack([reliability_val, numeric_block_train[val_idx]])
        meta_test = np.hstack([reliability_test, numeric_block_test])
    else:
        meta_train = reliability_train
        meta_val = reliability_val
        meta_test = reliability_test

    uncertainty_boost = float(variant.get('uncertainty_boost', 0.5))
    text_confidence = np.max(np.clip(text_train_proba[train_idx], 1e-8, 1.0), axis=1)
    correction_sample_weights = sample_weights * (1.0 + uncertainty_boost * (1.0 - text_confidence))

    correction_model = _fit_xgb_with_fallback(
        meta_train[train_idx],
        y_subtrain,
        meta_val,
        y_val,
        correction_sample_weights,
        max_depth=variant.get('correction_max_depth', 4),
        learning_rate=variant.get('correction_learning_rate', 0.01),
        n_estimators=variant.get('correction_n_estimators', 1000),
    )

    correction_train_proba = correction_model.predict_proba(meta_train)
    correction_val_proba = correction_model.predict_proba(meta_val)
    correction_test_proba = correction_model.predict_proba(meta_test)

    text_pred_sub = np.argmax(text_train_proba[train_idx], axis=1)
    correction_pred_sub = np.argmax(correction_train_proba[train_idx], axis=1)
    text_correct = (text_pred_sub == y_subtrain)
    correction_correct = (correction_pred_sub == y_subtrain)
    gate_targets = ((correction_correct) & (~text_correct)).astype(int)
    gate_sample_weights = sample_weights * (1.0 + 0.5 * (text_pred_sub != correction_pred_sub).astype(float))

    gate_mode = variant.get('gate_mode', 'learned')
    gate_info = {'mode': 'constant', 'constant_gate': 0.0}
    gate_model = None
    if gate_mode == 'rule_based':
        high_margin = float(variant.get('gate_high_margin', 0.55))
        medium_margin = float(variant.get('gate_medium_margin', 0.20))
        disagree_gate = float(variant.get('gate_disagree_value', 0.85))
        uncertain_gate = float(variant.get('gate_uncertain_value', 0.65))
        mid_gate = float(variant.get('gate_mid_value', 0.20))
        same_high_conf_gate = float(variant.get('gate_same_high_conf_value', 0.0))
        gate_train_scores = _build_rule_based_gate_scores(
            text_train_proba,
            numeric_train_proba,
            high_margin=high_margin,
            medium_margin=medium_margin,
            disagree_gate=disagree_gate,
            uncertain_gate=uncertain_gate,
            mid_gate=mid_gate,
            same_high_conf_gate=same_high_conf_gate,
        )
        gate_val_scores = _build_rule_based_gate_scores(
            text_val_proba,
            numeric_val_proba,
            high_margin=high_margin,
            medium_margin=medium_margin,
            disagree_gate=disagree_gate,
            uncertain_gate=uncertain_gate,
            mid_gate=mid_gate,
            same_high_conf_gate=same_high_conf_gate,
        )
        gate_test_scores = _build_rule_based_gate_scores(
            text_test_proba,
            numeric_test_proba,
            high_margin=high_margin,
            medium_margin=medium_margin,
            disagree_gate=disagree_gate,
            uncertain_gate=uncertain_gate,
            mid_gate=mid_gate,
            same_high_conf_gate=same_high_conf_gate,
        )
        gate_info = {
            'mode': 'rule_based',
            'high_margin': high_margin,
            'medium_margin': medium_margin,
            'disagree_gate': disagree_gate,
            'uncertain_gate': uncertain_gate,
            'mid_gate': mid_gate,
            'same_high_conf_gate': same_high_conf_gate,
        }
    elif np.unique(gate_targets).size >= 2:
        gate_model = _fit_binary_gate_with_fallback(
            reliability_train[train_idx],
            gate_targets,
            reliability_val,
            ((np.argmax(correction_val_proba, axis=1) == y_val) & (np.argmax(text_val_proba, axis=1) != y_val)).astype(int),
            gate_sample_weights,
            max_depth=variant.get('gate_max_depth', 3),
            learning_rate=variant.get('gate_learning_rate', 0.03),
            n_estimators=variant.get('gate_n_estimators', 400),
        )
        gate_train_scores = gate_model.predict_proba(reliability_train)[:, 1]
        gate_val_scores = gate_model.predict_proba(reliability_val)[:, 1]
        gate_test_scores = gate_model.predict_proba(reliability_test)[:, 1]
        gate_info = {'mode': 'binary_xgb'}
    else:
        gate_const = float(gate_targets.mean()) if gate_targets.size else 0.0
        gate_train_scores = np.full(text_train_proba.shape[0], gate_const, dtype=float)
        gate_val_scores = np.full(text_val_proba.shape[0], gate_const, dtype=float)
        gate_test_scores = np.full(text_test_proba.shape[0], gate_const, dtype=float)
        gate_info = {'mode': 'constant', 'constant_gate': gate_const}

    lambda_grid = [float(x) for x in variant.get('lambda_grid', [0.0, 0.25, 0.5, 0.75, 1.0])]
    calibration_objective = variant.get('calibration_objective', 'macro_f1_plus_min_class_f1')
    ti_weight = float(variant.get('ti_objective_weight', 0.0))
    accuracy_weight = float(variant.get('accuracy_objective_weight', 0.05))
    min_class_weight = float(variant.get('min_class_objective_weight', 0.1))
    bias_grid = variant.get('bias_grid', [-0.30, -0.15, 0.0, 0.15, 0.30])

    text_val_metrics = evaluate(y_val, np.argmax(text_val_proba, axis=1))
    text_test_metrics = evaluate(y_test, np.argmax(text_test_proba, axis=1))

    best = {
        'lambda': 0.0,
        'biases': [0.0, 0.0, 0.0],
        'metrics': None,
        'score': -1.0,
        'acceptable': False,
    }
    fallback_best = {
        'lambda': 0.0,
        'biases': [0.0, 0.0, 0.0],
        'metrics': None,
        'score': -1.0,
        'acceptable': False,
    }
    best_val_proba = None
    fallback_val_proba = None
    for lam in lambda_grid:
        val_proba = _blend_text_anchor_probabilities(text_val_proba, correction_val_proba, gate_val_scores, lam)
        bias_result = _search_class_biases(
            y_val,
            val_proba,
            metric=calibration_objective,
            ti_weight=ti_weight,
            accuracy_weight=accuracy_weight,
            min_class_weight=min_class_weight,
            bias_grid=bias_grid,
        )
        score = bias_result['score']
        metrics = bias_result['metrics']
        acceptable = _passes_llm_floor(metrics, text_val_metrics, variant)
        if (score > fallback_best['score']) or (
            np.isclose(score, fallback_best['score']) and metrics['accuracy'] > (fallback_best['metrics']['accuracy'] if fallback_best['metrics'] else -1)
        ):
            fallback_best = {
                'lambda': float(lam),
                'biases': [float(x) for x in bias_result['biases']],
                'metrics': metrics,
                'score': float(score),
                'acceptable': acceptable,
            }
            fallback_val_proba = val_proba
        if acceptable and ((score > best['score']) or (
            np.isclose(score, best['score']) and metrics['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)
        )):
            best = {
                'lambda': float(lam),
                'biases': [float(x) for x in bias_result['biases']],
                'metrics': metrics,
                'score': float(score),
                'acceptable': acceptable,
            }
            best_val_proba = val_proba

    selected = best if best['metrics'] is not None else fallback_best
    selected_val_proba = best_val_proba if best_val_proba is not None else fallback_val_proba
    final_lambda = selected['lambda']
    val_proba = selected_val_proba if selected_val_proba is not None else _blend_text_anchor_probabilities(text_val_proba, correction_val_proba, gate_val_scores, final_lambda)
    test_proba = _blend_text_anchor_probabilities(text_test_proba, correction_test_proba, gate_test_scores, final_lambda)
    biases = selected['biases']
    val_pred = _predict_with_class_biases(val_proba, biases)
    test_pred = _predict_with_class_biases(test_proba, biases)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'llm_features': list(base['train_emb_df'].columns),
        'numeric_features': list(numeric_feature_names),
        'reliability_feature_count': int(reliability_train.shape[1]),
        'meta_feature_count': int(meta_train.shape[1]),
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / summary_name, {
        'lambda_grid': lambda_grid,
        'selected_lambda': final_lambda,
        'selected_biases': biases,
        'calibration_objective': calibration_objective,
        'ti_objective_weight': ti_weight,
        'min_class_objective_weight': min_class_weight,
        'accuracy_objective_weight': accuracy_weight,
        'validation_search_result': selected,
        'fallback_validation_search_result': fallback_best,
        'llm_validation_metrics': text_val_metrics,
        'llm_test_metrics': text_test_metrics,
        'gate_info': gate_info,
        'uncertainty_boost': uncertainty_boost,
    })
    text_model.get_booster().save_model(str(out_dir / 'text_expert_model.json'))
    if numeric_model is not None:
        numeric_model.get_booster().save_model(str(out_dir / 'numeric_expert_model.json'))
    correction_model.get_booster().save_model(str(out_dir / 'residual_correction_model.json'))
    if gate_model is not None:
        gate_model.get_booster().save_model(str(out_dir / 'reliability_gate_model.json'))
    return {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'val_proba': val_proba,
        'test_proba': test_proba,
        'text_val_proba': text_val_proba,
        'text_test_proba': text_test_proba,
        'text_val_metrics': text_val_metrics,
        'text_test_metrics': text_test_metrics,
        'numeric_val_proba': numeric_val_proba,
        'numeric_test_proba': numeric_test_proba,
        'selected_lambda': final_lambda,
        'selected_biases': biases,
    }


def fit_balanced_residual_reliability_variant(base, variant, out_dir: Path):
    num_cols = _resolve_numeric_feature_columns(base, variant.get('numeric_feature_strategy', 'concat_curated'))
    result = _fit_text_anchor_residual_core(
        base,
        variant,
        out_dir,
        numeric_block_train=base['x_train_num'][num_cols].to_numpy(dtype=np.float32),
        numeric_block_test=base['x_test_num'][num_cols].to_numpy(dtype=np.float32),
        numeric_feature_names=num_cols,
        summary_name='balanced_residual_reliability.json',
    )
    return result['val_metrics'], result['test_metrics']


def fit_st_specialist_residual_variant(base, variant, out_dir: Path):
    num_cols = _resolve_numeric_feature_columns(base, variant.get('numeric_feature_strategy', 'concat_curated'))
    result = _fit_text_anchor_residual_core(
        base,
        variant,
        out_dir,
        numeric_block_train=base['x_train_num'][num_cols].to_numpy(dtype=np.float32),
        numeric_block_test=base['x_test_num'][num_cols].to_numpy(dtype=np.float32),
        numeric_feature_names=num_cols,
        summary_name='st_specialist_residual_main.json',
    )

    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    specialist_features = pd.concat([
        base['train_emb_df'].reset_index(drop=True),
        base['x_train_num'][num_cols].reset_index(drop=True),
    ], axis=1)
    specialist_test = pd.concat([
        base['test_emb_df'].reset_index(drop=True),
        base['x_test_num'][num_cols].reset_index(drop=True),
    ], axis=1)
    specialist_features = specialist_features.fillna(0.0)
    specialist_test = specialist_test.fillna(0.0)

    nontrivial_sub_mask = y_subtrain != 2
    nontrivial_val_mask = y_val != 2
    y_subtrain_bin = (y_subtrain[nontrivial_sub_mask] == 1).astype(int)
    y_val_bin = (y_val[nontrivial_val_mask] == 1).astype(int)
    x_subtrain = specialist_features.iloc[train_idx].iloc[nontrivial_sub_mask]
    x_val = specialist_features.iloc[val_idx].iloc[nontrivial_val_mask]

    weights_stage = heuristic_xgb.class_weights(y_subtrain_bin)
    sample_weights_stage = np.asarray([weights_stage[int(label)] for label in y_subtrain_bin], dtype=float)
    specialist = _fit_binary_gate_with_fallback(
        x_subtrain,
        y_subtrain_bin,
        x_val,
        y_val_bin,
        sample_weights_stage,
        max_depth=variant.get('specialist_max_depth', 4),
        learning_rate=variant.get('specialist_learning_rate', 0.01),
        n_estimators=variant.get('specialist_n_estimators', 800),
    )

    specialist_val_ti = specialist.predict_proba(specialist_features.iloc[val_idx])[:, 1]
    specialist_test_ti = specialist.predict_proba(specialist_test)[:, 1]
    specialist_val_subset_pred = (specialist.predict_proba(x_val)[:, 1] >= 0.5).astype(int)
    specialist_subset_metrics = {
        'balanced_accuracy': float(0.5 * (((specialist_val_subset_pred[y_val_bin == 0] == 0).mean() if np.any(y_val_bin == 0) else 0.0) + ((specialist_val_subset_pred[y_val_bin == 1] == 1).mean() if np.any(y_val_bin == 1) else 0.0))),
        'semimetal_f1': float(f1_score(y_val_bin, specialist_val_subset_pred, pos_label=0)),
        'topological_f1': float(f1_score(y_val_bin, specialist_val_subset_pred, pos_label=1)),
    }

    main_val_proba = result['val_proba']
    main_test_proba = result['test_proba']
    nontrivial_val = 1.0 - main_val_proba[:, 2]
    nontrivial_test = 1.0 - main_test_proba[:, 2]

    hier_val = np.zeros_like(main_val_proba)
    hier_test = np.zeros_like(main_test_proba)
    hier_val[:, 2] = main_val_proba[:, 2]
    hier_test[:, 2] = main_test_proba[:, 2]
    hier_val[:, 1] = nontrivial_val * specialist_val_ti
    hier_test[:, 1] = nontrivial_test * specialist_test_ti
    hier_val[:, 0] = nontrivial_val * (1.0 - specialist_val_ti)
    hier_test[:, 0] = nontrivial_test * (1.0 - specialist_test_ti)

    blend_grid = [float(x) for x in variant.get('specialist_blend_grid', [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])]
    calibration_objective = variant.get('calibration_objective', 'macro_f1_plus_min_class_f1')
    ti_weight = float(variant.get('ti_objective_weight', 0.0))
    accuracy_weight = float(variant.get('accuracy_objective_weight', 0.05))
    min_class_weight = float(variant.get('min_class_objective_weight', 0.1))
    bias_grid = variant.get('bias_grid', [-0.30, -0.15, 0.0, 0.15, 0.30])

    llm_val_metrics = result.get('text_val_metrics')
    best = {'eta': 0.0, 'biases': [0.0, 0.0, 0.0], 'metrics': None, 'score': -1.0, 'acceptable': False}
    fallback_best = {'eta': 0.0, 'biases': [0.0, 0.0, 0.0], 'metrics': None, 'score': -1.0, 'acceptable': False}
    best_val_proba = None
    fallback_val_proba = None
    for eta in blend_grid:
        val_proba = (1.0 - eta) * main_val_proba + eta * hier_val
        bias_result = _search_class_biases(
            y_val,
            val_proba,
            metric=calibration_objective,
            ti_weight=ti_weight,
            accuracy_weight=accuracy_weight,
            min_class_weight=min_class_weight,
            bias_grid=bias_grid,
        )
        acceptable = _passes_llm_floor(bias_result['metrics'], llm_val_metrics, variant)
        if (bias_result['score'] > fallback_best['score']) or (
            np.isclose(bias_result['score'], fallback_best['score']) and bias_result['metrics']['accuracy'] > (fallback_best['metrics']['accuracy'] if fallback_best['metrics'] else -1)
        ):
            fallback_best = {'eta': float(eta), 'biases': [float(x) for x in bias_result['biases']], 'metrics': bias_result['metrics'], 'score': float(bias_result['score']), 'acceptable': acceptable}
            fallback_val_proba = val_proba
        if acceptable and ((bias_result['score'] > best['score']) or (
            np.isclose(bias_result['score'], best['score']) and bias_result['metrics']['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)
        )):
            best = {'eta': float(eta), 'biases': [float(x) for x in bias_result['biases']], 'metrics': bias_result['metrics'], 'score': float(bias_result['score']), 'acceptable': acceptable}
            best_val_proba = val_proba

    selected = best if best['metrics'] is not None else fallback_best
    final_eta = selected['eta']
    final_val_proba = best_val_proba if best_val_proba is not None else fallback_val_proba if fallback_val_proba is not None else (1.0 - final_eta) * main_val_proba + final_eta * hier_val
    final_test_proba = (1.0 - final_eta) * main_test_proba + final_eta * hier_test
    biases = selected['biases']
    val_pred = _predict_with_class_biases(final_val_proba, biases)
    test_pred = _predict_with_class_biases(final_test_proba, biases)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'st_specialist_summary.json', {
        'main_residual_selected_lambda': result['selected_lambda'],
        'main_residual_biases': result['selected_biases'],
        'specialist_subset_metrics': specialist_subset_metrics,
        'specialist_blend_grid': blend_grid,
        'selected_specialist_eta': final_eta,
        'selected_biases': biases,
        'validation_search_result': selected,
        'fallback_validation_search_result': fallback_best,
        'llm_validation_metrics': llm_val_metrics,
    })
    specialist.get_booster().save_model(str(out_dir / 'st_specialist_model.json'))
    return val_metrics, test_metrics


def fit_family_probability_residual_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    x_train_num = base['x_train_num'].fillna(0.0)
    x_test_num = base['x_test_num'].fillna(0.0)
    sample_weights = np.asarray([heuristic_xgb.class_weights(y_subtrain)[int(label)] for label in y_subtrain], dtype=float)

    family_rows = []
    family_train_probas = {}
    family_val_probas = {}
    family_test_probas = {}
    family_cols = {}

    for family_name, raw_cols in NUMERIC_FEATURE_FAMILIES.items():
        cols = [c for c in raw_cols if c in x_train_num.columns]
        if not cols:
            continue
        family_cols[family_name] = cols
        model = _fit_xgb_with_fallback(
            x_train_num.iloc[train_idx][cols],
            y_subtrain,
            x_train_num.iloc[val_idx][cols],
            y_val,
            sample_weights,
            max_depth=variant.get('family_max_depth', 4),
            learning_rate=variant.get('family_learning_rate', 0.01),
            n_estimators=variant.get('family_n_estimators', 800),
        )
        train_proba = model.predict_proba(x_train_num[cols])
        val_proba = model.predict_proba(x_train_num.iloc[val_idx][cols])
        test_proba = model.predict_proba(x_test_num[cols])
        preds = np.argmax(val_proba, axis=1)
        metrics = evaluate(y_val, preds)
        score = float(metrics['macro_f1'])
        family_rows.append({
            'family': family_name,
            'feature_count': len(cols),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1']),
            'accuracy': float(metrics['accuracy']),
            'topological_f1': float(metrics['topological']['f1-score']),
            'score': score,
        })
        family_train_probas[family_name] = train_proba
        family_val_probas[family_name] = val_proba
        family_test_probas[family_name] = test_proba
        model.get_booster().save_model(str(out_dir / f'family_{family_name}_model.json'))

    family_rows.sort(key=lambda row: row['score'], reverse=True)
    top_k = int(variant.get('selected_family_top_k', 3))
    selected_rows = family_rows[:max(1, min(top_k, len(family_rows)))]
    temperature = float(variant.get('family_weight_temperature', 4.0))
    raw_scores = np.asarray([row['score'] for row in selected_rows], dtype=float)
    shifted = raw_scores - raw_scores.max()
    family_weights = np.exp(temperature * shifted)
    family_weights = family_weights / np.clip(family_weights.sum(), 1e-8, None)
    selected_families = [row['family'] for row in selected_rows]
    weight_map = {fam: float(w) for fam, w in zip(selected_families, family_weights)}

    numeric_train_proba = sum(weight_map[fam] * family_train_probas[fam] for fam in selected_families)
    numeric_val_proba = sum(weight_map[fam] * family_val_probas[fam] for fam in selected_families)
    numeric_test_proba = sum(weight_map[fam] * family_test_probas[fam] for fam in selected_families)
    selected_cols = []
    for fam in selected_families:
        for col in family_cols[fam]:
            if col not in selected_cols:
                selected_cols.append(col)

    result = _fit_text_anchor_residual_core(
        base,
        variant,
        out_dir,
        numeric_block_train=x_train_num[selected_cols].to_numpy(dtype=np.float32),
        numeric_block_test=x_test_num[selected_cols].to_numpy(dtype=np.float32),
        numeric_feature_names=selected_cols,
        numeric_train_proba=numeric_train_proba,
        numeric_val_proba=numeric_val_proba,
        numeric_test_proba=numeric_test_proba,
        summary_name='family_probability_residual.json',
    )
    dump_json(out_dir / 'family_probability_summary.json', {
        'family_rows': family_rows,
        'selected_families': selected_families,
        'family_weights': weight_map,
        'selected_feature_count': len(selected_cols),
        'selected_features': selected_cols,
    })
    return result['val_metrics'], result['test_metrics']



def _save_probs_npy(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float32))


def _uncertainty_frame(proba, split_name):
    proba = np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0)
    summary = _probability_summary_features(proba)
    top1 = np.argmax(proba, axis=1)
    top2 = np.argsort(proba, axis=1)[:, -2]
    return pd.DataFrame({
        'split': split_name,
        'pred_label': top1,
        'second_label': top2,
        'confidence': summary[:, 3],
        'entropy': summary[:, 4],
        'margin': summary[:, 5],
        'prob_semimetal': proba[:, 0],
        'prob_topological': proba[:, 1],
        'prob_trivial': proba[:, 2],
    })


def _disagreement_frame(p_llm, p_phys, split_name):
    p_llm = np.clip(np.asarray(p_llm, dtype=float), 1e-8, 1.0)
    p_phys = np.clip(np.asarray(p_phys, dtype=float), 1e-8, 1.0)
    llm_summary = _probability_summary_features(p_llm)
    phys_summary = _probability_summary_features(p_phys)
    llm_pred = np.argmax(p_llm, axis=1)
    phys_pred = np.argmax(p_phys, axis=1)
    df = pd.DataFrame({
        'split': split_name,
        'llm_pred': llm_pred,
        'phys_pred': phys_pred,
        'llm_entropy': llm_summary[:, 4],
        'phys_entropy': phys_summary[:, 4],
        'llm_margin': llm_summary[:, 5],
        'phys_margin': phys_summary[:, 5],
        'llm_phys_agree': (llm_pred == phys_pred).astype(int),
        'kl_llm_phys': _kl_divergence_rows(p_llm, p_phys).ravel(),
        'kl_phys_llm': _kl_divergence_rows(p_phys, p_llm).ravel(),
        'abs_diff_semimetal': np.abs(p_llm[:, 0] - p_phys[:, 0]),
        'abs_diff_topological': np.abs(p_llm[:, 1] - p_phys[:, 1]),
        'abs_diff_trivial': np.abs(p_llm[:, 2] - p_phys[:, 2]),
    })
    return df


def _class_help_hurt_summary(y_true, pred_ref, pred_new):
    y_true = np.asarray(y_true, dtype=int)
    pred_ref = np.asarray(pred_ref, dtype=int)
    pred_new = np.asarray(pred_new, dtype=int)
    rows = []
    for idx, label in enumerate(LABEL_ORDER):
        mask = y_true == idx
        ref_correct = pred_ref[mask] == y_true[mask]
        new_correct = pred_new[mask] == y_true[mask]
        helped = int(np.sum((~ref_correct) & new_correct))
        hurt = int(np.sum(ref_correct & (~new_correct)))
        rows.append({
            'class': label,
            'helped': helped,
            'hurt': hurt,
            'net': helped - hurt,
            'support': int(mask.sum()),
        })
    return rows


def _llm_vs_candidate_table(y_true, pred_ref, pred_new):
    y_true = np.asarray(y_true, dtype=int)
    pred_ref = np.asarray(pred_ref, dtype=int)
    pred_new = np.asarray(pred_new, dtype=int)
    ref_correct = pred_ref == y_true
    new_correct = pred_new == y_true
    return {
        'llm_correct_candidate_correct': int(np.sum(ref_correct & new_correct)),
        'llm_wrong_candidate_correct': int(np.sum((~ref_correct) & new_correct)),
        'llm_correct_candidate_wrong': int(np.sum(ref_correct & (~new_correct))),
        'llm_wrong_candidate_wrong': int(np.sum((~ref_correct) & (~new_correct))),
        'net_gain': int(np.sum((~ref_correct) & new_correct) - np.sum(ref_correct & (~new_correct))),
    }


def _confidence_group_help_hurt(y_true, pred_ref, pred_new, llm_margin):
    y_true = np.asarray(y_true, dtype=int)
    pred_ref = np.asarray(pred_ref, dtype=int)
    pred_new = np.asarray(pred_new, dtype=int)
    llm_margin = np.asarray(llm_margin, dtype=float)
    groups = {
        'high': llm_margin > 0.60,
        'medium': (llm_margin > 0.30) & (llm_margin <= 0.60),
        'low': llm_margin <= 0.30,
    }
    rows = []
    ref_correct = pred_ref == y_true
    new_correct = pred_new == y_true
    for name, mask in groups.items():
        helped = int(np.sum((~ref_correct[mask]) & new_correct[mask]))
        hurt = int(np.sum(ref_correct[mask] & (~new_correct[mask])))
        rows.append({'group': name, 'helped': helped, 'hurt': hurt, 'net': helped - hurt, 'support': int(mask.sum())})
    return rows


def _error_transition_summary(y_true, pred):
    labels = LABEL_ORDER
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    rows = []
    for src in range(len(labels)):
        for dst in range(len(labels)):
            if src == dst:
                continue
            count = int(np.sum((y_true == src) & (pred == dst)))
            rows.append({'true_class': labels[src], 'pred_class': labels[dst], 'count': count})
    return rows


def _family_block_columns(x_train_num, family_keys):
    cols = []
    for key in family_keys:
        for col in NUMERIC_FEATURE_FAMILIES.get(key, []):
            if col in x_train_num.columns and col not in cols:
                cols.append(col)
    return cols


def fit_llm_anchored_residual_specialist_workflow_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    variant19_metrics_path = Path(variant.get('variant19_validation_metrics_path', ROOT / 'weighted_version' / 'txl_fusion_variants' / 'variants' / '19_ti_aware_gated_hierarchical_fusion' / 'output' / 'validation_metrics.json'))
    variant19_val_metrics = json.loads(variant19_metrics_path.read_text()) if variant19_metrics_path.exists() else None

    splits_dir = out_dir / 'splits'
    dump_json(splits_dir / 'subtrain_ids.json', [int(i) for i in train_idx])
    dump_json(splits_dir / 'val_ids.json', [int(i) for i in val_idx])
    dump_json(splits_dir / 'test_ids.json', [int(i) for i in range(len(y_test))])

    llm_dir = out_dir / 'outputs' / 'llm_only'
    physical_expert_dir = out_dir / 'outputs' / 'physical_experts'
    physical_evidence_dir = out_dir / 'outputs' / 'physical_evidence'
    disagreement_dir = out_dir / 'outputs' / 'disagreement'
    residual_dir = out_dir / 'outputs' / 'residual'
    gates_dir = out_dir / 'outputs' / 'gates'
    residual_search_dir = out_dir / 'outputs' / 'residual_search'
    specialist_dir = out_dir / 'outputs' / 'st_specialist'
    final_blend_dir = out_dir / 'outputs' / 'final_blend'
    diagnostics_dir = out_dir / 'outputs' / 'diagnostics'
    features_dir = out_dir / 'features'

    for required_dir in [
        splits_dir,
        llm_dir,
        physical_expert_dir,
        physical_evidence_dir,
        disagreement_dir,
        residual_dir,
        gates_dir,
        residual_search_dir,
        specialist_dir,
        final_blend_dir,
        diagnostics_dir,
        features_dir,
    ]:
        required_dir.mkdir(parents=True, exist_ok=True)

    dump_json(features_dir / 'feature_families.json', NUMERIC_FEATURE_FAMILIES)

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    llm_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    llm_train_scaled = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test_scaled = llm_scaler.transform(llm_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')

    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    llm_anchor_mode = variant.get('llm_anchor_mode', 'xgb_embedding_expert')
    llm_model = None
    if llm_anchor_mode == 'direct_scibert_probs':
        llm_train_proba = np.asarray(base['train_text_proba'], dtype=np.float32)
        llm_val_proba = llm_train_proba[val_idx]
        llm_test_proba = np.asarray(base['test_text_proba'], dtype=np.float32)
        dump_json(llm_dir / 'llm_anchor_source.json', {
            'mode': 'direct_scibert_probs',
            'checkpoint': str(SCIBERT_CHECKPOINT),
            'label_order': LABEL_ORDER,
            'note': 'Uses the direct fine-tuned SciBERT classifier probabilities as the semantic anchor, rather than an auxiliary XGBoost expert trained on PCA embeddings.'
        })
    else:
        llm_model = _fit_xgb_with_fallback(
            llm_train_scaled[train_idx], y_subtrain, llm_train_scaled[val_idx], y_val, sample_weights,
            max_depth=variant.get('text_max_depth', 4),
            learning_rate=variant.get('text_learning_rate', 0.01),
            n_estimators=variant.get('text_n_estimators', 1000),
        )
        llm_train_proba = llm_model.predict_proba(llm_train_scaled)
        llm_val_proba = llm_model.predict_proba(llm_train_scaled[val_idx])
        llm_test_proba = llm_model.predict_proba(llm_test_scaled)
        dump_json(llm_dir / 'llm_anchor_source.json', {
            'mode': 'xgb_embedding_expert',
            'embedding_source': 'SciBERT PCA branch',
            'note': 'Uses an auxiliary XGBoost expert trained on PCA-compressed SciBERT embeddings as the semantic anchor.'
        })
    llm_train_pred = np.argmax(llm_train_proba, axis=1)
    llm_val_pred = np.argmax(llm_val_proba, axis=1)
    llm_test_pred = np.argmax(llm_test_proba, axis=1)
    llm_val_metrics = evaluate(y_val, llm_val_pred)
    llm_test_metrics = evaluate(y_test, llm_test_pred)
    dump_json(llm_dir / 'metrics_val.json', llm_val_metrics)
    dump_json(llm_dir / 'metrics_test.json', llm_test_metrics)
    _save_probs_npy(llm_dir / 'subtrain_probs.npy', llm_train_proba[train_idx])
    _save_probs_npy(llm_dir / 'val_probs.npy', llm_val_proba)
    _save_probs_npy(llm_dir / 'test_probs.npy', llm_test_proba)
    _save_probs_npy(llm_dir / 'subtrain_pred.npy', llm_train_pred[train_idx])
    _save_probs_npy(llm_dir / 'val_pred.npy', llm_val_pred)
    _save_probs_npy(llm_dir / 'test_pred.npy', llm_test_pred)
    _uncertainty_frame(llm_train_proba[train_idx], 'subtrain').to_csv(llm_dir / 'subtrain_uncertainty.csv', index=False)
    _uncertainty_frame(llm_val_proba, 'validation').to_csv(llm_dir / 'val_uncertainty.csv', index=False)
    _uncertainty_frame(llm_test_proba, 'heldout_test').to_csv(llm_dir / 'test_uncertainty.csv', index=False)
    if llm_model is not None:
        llm_model.get_booster().save_model(str(llm_dir / 'llm_model.json'))

    x_train_num = base['x_train_num'].fillna(0.0)
    x_test_num = base['x_test_num'].fillna(0.0)
    family_rows = []
    family_train_proba_map = {}
    family_val_proba_map = {}
    family_test_proba_map = {}
    family_metrics_val = {}
    family_metrics_test = {}
    for family_name, raw_cols in NUMERIC_FEATURE_FAMILIES.items():
        cols = [c for c in raw_cols if c in x_train_num.columns]
        if not cols:
            continue
        model = _fit_xgb_with_fallback(
            x_train_num.iloc[train_idx][cols], y_subtrain,
            x_train_num.iloc[val_idx][cols], y_val,
            sample_weights,
            max_depth=variant.get('family_max_depth', 2),
            learning_rate=variant.get('family_learning_rate', 0.03),
            n_estimators=variant.get('family_n_estimators', 300),
        )
        train_proba = model.predict_proba(x_train_num[cols])
        val_proba = model.predict_proba(x_train_num.iloc[val_idx][cols])
        test_proba = model.predict_proba(x_test_num[cols])
        val_pred = np.argmax(val_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)
        m_val = evaluate(y_val, val_pred)
        m_test = evaluate(y_test, test_pred)
        family_rows.append({'family': family_name, 'features': cols, 'val_macro_f1': float(m_val['macro_f1']), 'val_topological_f1': float(m_val['topological']['f1-score']), 'test_macro_f1': float(m_test['macro_f1']), 'test_topological_f1': float(m_test['topological']['f1-score'])})
        family_metrics_val[family_name] = m_val
        family_metrics_test[family_name] = m_test
        family_train_proba_map[family_name] = train_proba
        family_val_proba_map[family_name] = val_proba
        family_test_proba_map[family_name] = test_proba
        _save_probs_npy(physical_expert_dir / f'{family_name}_subtrain_probs.npy', train_proba[train_idx])
        _save_probs_npy(physical_expert_dir / f'{family_name}_val_probs.npy', val_proba)
        _save_probs_npy(physical_expert_dir / f'{family_name}_test_probs.npy', test_proba)
        model.get_booster().save_model(str(physical_expert_dir / f'{family_name}_model.json'))
    dump_json(physical_expert_dir / 'metrics_each_expert_val.json', family_metrics_val)
    dump_json(physical_expert_dir / 'metrics_each_expert_test.json', family_metrics_test)
    dump_json(physical_expert_dir / 'feature_family_summary.json', family_rows)

    family_names = list(family_train_proba_map.keys())
    if not family_names:
        raise ValueError('No physical expert families could be built.')
    family_weight_mode = variant.get('family_weight_mode', 'uniform')
    if family_weight_mode == 'uniform':
        family_weights = {name: 1.0 / len(family_names) for name in family_names}
    else:
        val_scores = np.asarray([family_metrics_val[name]['macro_f1'] for name in family_names], dtype=float)
        shifted = val_scores - val_scores.max()
        temp = float(variant.get('family_weight_temperature', 4.0))
        w = np.exp(temp * shifted)
        w = w / np.clip(w.sum(), 1e-8, None)
        family_weights = {name: float(val) for name, val in zip(family_names, w)}

    E_train = np.hstack([family_train_proba_map[name] for name in family_names])
    E_val = np.hstack([family_val_proba_map[name] for name in family_names])
    E_test = np.hstack([family_test_proba_map[name] for name in family_names])
    p_phys_train = sum(family_weights[name] * family_train_proba_map[name] for name in family_names)
    p_phys_val = sum(family_weights[name] * family_val_proba_map[name] for name in family_names)
    p_phys_test = sum(family_weights[name] * family_test_proba_map[name] for name in family_names)
    _save_probs_npy(physical_evidence_dir / 'subtrain_Ephys.npy', E_train[train_idx])
    _save_probs_npy(physical_evidence_dir / 'val_Ephys.npy', E_val)
    _save_probs_npy(physical_evidence_dir / 'test_Ephys.npy', E_test)
    _save_probs_npy(physical_evidence_dir / 'subtrain_pphys.npy', p_phys_train[train_idx])
    _save_probs_npy(physical_evidence_dir / 'val_pphys.npy', p_phys_val)
    _save_probs_npy(physical_evidence_dir / 'test_pphys.npy', p_phys_test)
    dump_json(physical_evidence_dir / 'family_weights.json', family_weights)

    disagree_train_df = _disagreement_frame(llm_train_proba[train_idx], p_phys_train[train_idx], 'subtrain')
    disagree_val_df = _disagreement_frame(llm_val_proba, p_phys_val, 'validation')
    disagree_test_df = _disagreement_frame(llm_test_proba, p_phys_test, 'heldout_test')
    disagree_train_df.to_csv(disagreement_dir / 'subtrain_disagreement_features.csv', index=False)
    disagree_val_df.to_csv(disagreement_dir / 'val_disagreement_features.csv', index=False)
    disagree_test_df.to_csv(disagreement_dir / 'test_disagreement_features.csv', index=False)

    num_small_cols = _family_block_columns(x_train_num, ['heuristic', 'sg', 'electron'])
    num_medium_cols = _family_block_columns(x_train_num, ['heuristic', 'sg', 'electron', 'orbital', 'bonding'])
    raw_num_cols = num_medium_cols if variant.get('raw_numeric_block', 'medium') == 'medium' else num_small_cols
    raw_num_train = x_train_num[raw_num_cols].to_numpy(dtype=np.float32) if raw_num_cols else np.zeros((len(x_train_num), 0), dtype=np.float32)
    raw_num_test = x_test_num[raw_num_cols].to_numpy(dtype=np.float32) if raw_num_cols else np.zeros((len(x_test_num), 0), dtype=np.float32)
    if raw_num_cols:
        raw_num_scaler = StandardScaler()
        raw_num_scaler.fit(raw_num_train[train_idx])
        raw_num_train = raw_num_scaler.transform(raw_num_train).astype(np.float32)
        raw_num_test = raw_num_scaler.transform(raw_num_test).astype(np.float32)
        joblib.dump(raw_num_scaler, out_dir / 'raw_numeric_scaler.pkl')

    llm_train_summary = _probability_summary_features(llm_train_proba)
    llm_val_summary = _probability_summary_features(llm_val_proba)
    llm_test_summary = _probability_summary_features(llm_test_proba)
    phys_train_summary = _probability_summary_features(p_phys_train)
    phys_val_summary = _probability_summary_features(p_phys_val)
    phys_test_summary = _probability_summary_features(p_phys_test)

    X_res_train = np.hstack([
        llm_train_proba,
        E_train,
        p_phys_train,
        llm_train_summary[:, 4:6],
        phys_train_summary[:, 4:6],
        _kl_divergence_rows(llm_train_proba, p_phys_train),
        (np.argmax(llm_train_proba, axis=1) != np.argmax(p_phys_train, axis=1)).astype(float)[:, None],
        np.abs(llm_train_proba - p_phys_train),
        raw_num_train,
    ])
    X_res_val = np.hstack([
        llm_val_proba,
        E_val,
        p_phys_val,
        llm_val_summary[:, 4:6],
        phys_val_summary[:, 4:6],
        _kl_divergence_rows(llm_val_proba, p_phys_val),
        (np.argmax(llm_val_proba, axis=1) != np.argmax(p_phys_val, axis=1)).astype(float)[:, None],
        np.abs(llm_val_proba - p_phys_val),
        raw_num_train[val_idx],
    ])
    X_res_test = np.hstack([
        llm_test_proba,
        E_test,
        p_phys_test,
        llm_test_summary[:, 4:6],
        phys_test_summary[:, 4:6],
        _kl_divergence_rows(llm_test_proba, p_phys_test),
        (np.argmax(llm_test_proba, axis=1) != np.argmax(p_phys_test, axis=1)).astype(float)[:, None],
        np.abs(llm_test_proba - p_phys_test),
        raw_num_test,
    ])

    uncertainty_boost = float(variant.get('uncertainty_boost', 0.5))
    llm_conf_sub = np.max(llm_train_proba[train_idx], axis=1)
    correction_sample_weights = sample_weights * (1.0 + uncertainty_boost * (1.0 - llm_conf_sub))
    residual_model = _fit_xgb_with_fallback(
        X_res_train[train_idx], y_subtrain,
        X_res_val, y_val,
        correction_sample_weights,
        max_depth=variant.get('correction_max_depth', 2),
        learning_rate=variant.get('correction_learning_rate', 0.03),
        n_estimators=variant.get('correction_n_estimators', 300),
    )
    p_res_train = residual_model.predict_proba(X_res_train)
    p_res_val = residual_model.predict_proba(X_res_val)
    p_res_test = residual_model.predict_proba(X_res_test)
    r_train = _proba_to_logits(p_res_train) - _proba_to_logits(llm_train_proba)
    r_val = _proba_to_logits(p_res_val) - _proba_to_logits(llm_val_proba)
    r_test = _proba_to_logits(p_res_test) - _proba_to_logits(llm_test_proba)
    _save_probs_npy(residual_dir / 'val_residual_probs.npy', p_res_val)
    _save_probs_npy(residual_dir / 'test_residual_probs.npy', p_res_test)
    _save_probs_npy(residual_dir / 'val_residual_logits.npy', r_val)
    _save_probs_npy(residual_dir / 'test_residual_logits.npy', r_test)
    dump_json(residual_dir / 'metrics_residual_val.json', evaluate(y_val, np.argmax(p_res_val, axis=1)))
    dump_json(residual_dir / 'metrics_residual_test.json', evaluate(y_test, np.argmax(p_res_test, axis=1)))
    residual_model.get_booster().save_model(str(residual_dir / 'residual_model.json'))

    reliability_train = _build_reliability_features(llm_train_proba, p_phys_train)
    reliability_val = _build_reliability_features(llm_val_proba, p_phys_val)
    reliability_test = _build_reliability_features(llm_test_proba, p_phys_test)
    rule_gate_train = _build_rule_based_gate_scores(
        llm_train_proba, p_phys_train,
        high_margin=float(variant.get('gate_high_margin', 0.60)),
        medium_margin=float(variant.get('gate_medium_margin', 0.30)),
        disagree_gate=float(variant.get('gate_disagree_value', 0.30)),
        uncertain_gate=float(variant.get('gate_uncertain_value', 0.40)),
        mid_gate=float(variant.get('gate_mid_value', 0.20)),
        same_high_conf_gate=float(variant.get('gate_same_high_conf_value', 0.0)),
    )
    rule_gate_val = _build_rule_based_gate_scores(
        llm_val_proba, p_phys_val,
        high_margin=float(variant.get('gate_high_margin', 0.60)),
        medium_margin=float(variant.get('gate_medium_margin', 0.30)),
        disagree_gate=float(variant.get('gate_disagree_value', 0.30)),
        uncertain_gate=float(variant.get('gate_uncertain_value', 0.40)),
        mid_gate=float(variant.get('gate_mid_value', 0.20)),
        same_high_conf_gate=float(variant.get('gate_same_high_conf_value', 0.0)),
    )
    rule_gate_test = _build_rule_based_gate_scores(
        llm_test_proba, p_phys_test,
        high_margin=float(variant.get('gate_high_margin', 0.60)),
        medium_margin=float(variant.get('gate_medium_margin', 0.30)),
        disagree_gate=float(variant.get('gate_disagree_value', 0.30)),
        uncertain_gate=float(variant.get('gate_uncertain_value', 0.40)),
        mid_gate=float(variant.get('gate_mid_value', 0.20)),
        same_high_conf_gate=float(variant.get('gate_same_high_conf_value', 0.0)),
    )
    _save_probs_npy(gates_dir / 'val_gate_rule.npy', rule_gate_val)
    _save_probs_npy(gates_dir / 'test_gate_rule.npy', rule_gate_test)

    residual_train_pred = np.argmax(p_res_train[train_idx], axis=1)
    gate_target = ((residual_train_pred == y_subtrain) & (llm_train_pred[train_idx] != y_subtrain)).astype(int)
    gate_weights = sample_weights * (1.0 + 0.5 * (llm_train_pred[train_idx] != np.argmax(p_phys_train[train_idx], axis=1)).astype(float))
    learned_gate_model = None
    learned_gate_val = None
    learned_gate_test = None
    if np.unique(gate_target).size >= 2:
        learned_gate_model = _fit_binary_gate_with_fallback(
            reliability_train[train_idx], gate_target,
            reliability_val, ((np.argmax(p_res_val, axis=1) == y_val) & (llm_val_pred != y_val)).astype(int),
            gate_weights,
            max_depth=variant.get('learned_gate_max_depth', 2),
            learning_rate=variant.get('learned_gate_learning_rate', 0.05),
            n_estimators=variant.get('learned_gate_n_estimators', 200),
        )
        learned_gate_val = learned_gate_model.predict_proba(reliability_val)[:, 1]
        learned_gate_test = learned_gate_model.predict_proba(reliability_test)[:, 1]
        _save_probs_npy(gates_dir / 'val_gate_learned.npy', learned_gate_val)
        _save_probs_npy(gates_dir / 'test_gate_learned.npy', learned_gate_test)
        learned_gate_model.get_booster().save_model(str(gates_dir / 'learned_gate_model.json'))

    def compose_main(p_llm, residual_logits, gate_scores, lam, class_bias):
        z = _proba_to_logits(p_llm) + float(lam) * np.asarray(gate_scores)[:, None] * residual_logits + np.asarray(class_bias)[None, :]
        return _softmax_rows(z)

    def search_main(gate_val, gate_test, gate_name):
        lambda_grid = [float(x) for x in variant.get('lambda_grid', [0.05, 0.10, 0.20, 0.30, 0.50])]
        bias_grid = [float(x) for x in variant.get('bias_grid', [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])]
        rows = []
        eligible = []
        for lam in lambda_grid:
            val_main_uncal = compose_main(llm_val_proba, r_val, gate_val, lam, [0.0, 0.0, 0.0])
            bias_best = None
            bias_best_proba = None
            for b1 in bias_grid:
                for b2 in bias_grid:
                    biases = [0.0, b1, b2]
                    val_main = compose_main(llm_val_proba, r_val, gate_val, lam, biases)
                    pred = np.argmax(val_main, axis=1)
                    m = evaluate(y_val, pred)
                    macro = float(m['macro_f1'])
                    minf = _minimum_class_f1(m)
                    acc = float(m['accuracy'])
                    score = macro + 0.1 * minf + 0.05 * acc
                    cond = (
                        m['topological']['f1-score'] >= llm_val_metrics['topological']['f1-score'] and
                        m['semimetal']['f1-score'] >= llm_val_metrics['semimetal']['f1-score'] - 0.002 and
                        m['trivial']['f1-score'] >= llm_val_metrics['trivial']['f1-score'] - 0.002 and
                        m['macro_f1'] >= llm_val_metrics['macro_f1'] and
                        (variant19_val_metrics is None or (m['macro_f1'] >= variant19_val_metrics['macro_f1'] and m['accuracy'] >= variant19_val_metrics['accuracy']))
                    )
                    row = {'gate': gate_name, 'lambda': lam, 'bias_semimetal': 0.0, 'bias_topological': b1, 'bias_trivial': b2, 'score': score, 'eligible': cond, 'macro_f1': macro, 'accuracy': acc, 'topological_f1': float(m['topological']['f1-score']), 'semimetal_f1': float(m['semimetal']['f1-score']), 'trivial_f1': float(m['trivial']['f1-score'])}
                    rows.append(row)
                    if bias_best is None or score > bias_best['score']:
                        bias_best = row
                        bias_best_proba = val_main
                    if cond:
                        eligible.append((row, bias_best_proba))
        grid_df = pd.DataFrame(rows)
        grid_df.to_csv(residual_search_dir / f'grid_results_val_{gate_name}.csv', index=False)
        if eligible:
            best_row, best_val_proba = max(eligible, key=lambda item: (item[0]['score'], item[0]['accuracy']))
        else:
            best_idx = grid_df['score'].astype(float).idxmax()
            best_row = rows[int(best_idx)]
            best_val_proba = compose_main(llm_val_proba, r_val, gate_val, best_row['lambda'], [best_row['bias_semimetal'], best_row['bias_topological'], best_row['bias_trivial']])
        test_main = compose_main(llm_test_proba, r_test, gate_test, best_row['lambda'], [best_row['bias_semimetal'], best_row['bias_topological'], best_row['bias_trivial']])
        return best_row, best_val_proba, test_main

    ablation_results = {}
    main_variants = {}
    no_gate_val = np.ones(len(y_val), dtype=float)
    no_gate_test = np.ones(len(y_test), dtype=float)
    for gate_name, gv, gt in [('no_gate', no_gate_val, no_gate_test), ('rule_gate', rule_gate_val, rule_gate_test)]:
        best_row, val_main, test_main = search_main(gv, gt, gate_name)
        val_pred = np.argmax(val_main, axis=1)
        test_pred = np.argmax(test_main, axis=1)
        val_metrics = evaluate(y_val, val_pred)
        test_metrics = evaluate(y_test, test_pred)
        main_variants[gate_name] = {'best_row': best_row, 'val_proba': val_main, 'test_proba': test_main, 'val_pred': val_pred, 'test_pred': test_pred, 'val_metrics': val_metrics, 'test_metrics': test_metrics}
        ablation_results[f'residual_{gate_name}'] = {'validation': val_metrics, 'heldout_test': test_metrics}
    if learned_gate_val is not None:
        best_row, val_main, test_main = search_main(learned_gate_val, learned_gate_test, 'learned_gate')
        val_pred = np.argmax(val_main, axis=1)
        test_pred = np.argmax(test_main, axis=1)
        val_metrics = evaluate(y_val, val_pred)
        test_metrics = evaluate(y_test, test_pred)
        main_variants['learned_gate'] = {'best_row': best_row, 'val_proba': val_main, 'test_proba': test_main, 'val_pred': val_pred, 'test_pred': test_pred, 'val_metrics': val_metrics, 'test_metrics': test_metrics}
        ablation_results['residual_learned_gate'] = {'validation': val_metrics, 'heldout_test': test_metrics}

    rng = np.random.default_rng(SEED)
    shuffled_idx_train = np.arange(len(train_idx))
    rng.shuffle(shuffled_idx_train)
    shuffled_idx_val = np.arange(len(y_val))
    rng.shuffle(shuffled_idx_val)
    shuffled_idx_test = np.arange(len(y_test))
    rng.shuffle(shuffled_idx_test)
    E_train_shuf = E_train.copy()
    E_val_shuf = E_val.copy()
    E_test_shuf = E_test.copy()
    p_phys_train_shuf = p_phys_train.copy()
    p_phys_val_shuf = p_phys_val.copy()
    p_phys_test_shuf = p_phys_test.copy()
    E_train_shuf[train_idx] = E_train_shuf[train_idx][shuffled_idx_train]
    E_val_shuf = E_val_shuf[shuffled_idx_val]
    E_test_shuf = E_test_shuf[shuffled_idx_test]
    p_phys_train_shuf[train_idx] = p_phys_train_shuf[train_idx][shuffled_idx_train]
    p_phys_val_shuf = p_phys_val_shuf[shuffled_idx_val]
    p_phys_test_shuf = p_phys_test_shuf[shuffled_idx_test]
    X_res_train_shuf = np.hstack([llm_train_proba, E_train_shuf, p_phys_train_shuf, llm_train_summary[:, 4:6], _probability_summary_features(p_phys_train_shuf)[:, 4:6], _kl_divergence_rows(llm_train_proba, p_phys_train_shuf), (np.argmax(llm_train_proba, axis=1) != np.argmax(p_phys_train_shuf, axis=1)).astype(float)[:, None], np.abs(llm_train_proba - p_phys_train_shuf), raw_num_train])
    X_res_val_shuf = np.hstack([llm_val_proba, E_val_shuf, p_phys_val_shuf, llm_val_summary[:, 4:6], _probability_summary_features(p_phys_val_shuf)[:, 4:6], _kl_divergence_rows(llm_val_proba, p_phys_val_shuf), (np.argmax(llm_val_proba, axis=1) != np.argmax(p_phys_val_shuf, axis=1)).astype(float)[:, None], np.abs(llm_val_proba - p_phys_val_shuf), raw_num_train[val_idx]])
    X_res_test_shuf = np.hstack([llm_test_proba, E_test_shuf, p_phys_test_shuf, llm_test_summary[:, 4:6], _probability_summary_features(p_phys_test_shuf)[:, 4:6], _kl_divergence_rows(llm_test_proba, p_phys_test_shuf), (np.argmax(llm_test_proba, axis=1) != np.argmax(p_phys_test_shuf, axis=1)).astype(float)[:, None], np.abs(llm_test_proba - p_phys_test_shuf), raw_num_test])
    shuffled_model = _fit_xgb_with_fallback(
        X_res_train_shuf[train_idx], y_subtrain, X_res_val_shuf, y_val, correction_sample_weights,
        max_depth=variant.get('correction_max_depth', 2), learning_rate=variant.get('correction_learning_rate', 0.03), n_estimators=variant.get('correction_n_estimators', 300),
    )
    p_res_val_shuf = shuffled_model.predict_proba(X_res_val_shuf)
    p_res_test_shuf = shuffled_model.predict_proba(X_res_test_shuf)
    r_val_shuf = _proba_to_logits(p_res_val_shuf) - _proba_to_logits(llm_val_proba)
    r_test_shuf = _proba_to_logits(p_res_test_shuf) - _proba_to_logits(llm_test_proba)
    best_row_shuf, val_main_shuf, test_main_shuf = search_main(rule_gate_val, rule_gate_test, 'shuffled_physics')
    # overwrite with shuffled residual logits in main composition
    val_main_shuf = compose_main(llm_val_proba, r_val_shuf, rule_gate_val, best_row_shuf['lambda'], [best_row_shuf['bias_semimetal'], best_row_shuf['bias_topological'], best_row_shuf['bias_trivial']])
    test_main_shuf = compose_main(llm_test_proba, r_test_shuf, rule_gate_test, best_row_shuf['lambda'], [best_row_shuf['bias_semimetal'], best_row_shuf['bias_topological'], best_row_shuf['bias_trivial']])
    ablation_results['residual_shuffled_physics'] = {'validation': evaluate(y_val, np.argmax(val_main_shuf, axis=1)), 'heldout_test': evaluate(y_test, np.argmax(test_main_shuf, axis=1))}
    shuffled_model.get_booster().save_model(str(residual_dir / 'shuffled_physics_residual_model.json'))

    # Specialist variants
    nontrivial_sub_mask = y_subtrain != 2
    nontrivial_val_mask = y_val != 2
    nontrivial_test_mask = y_test != 2
    X_st_full_train = np.hstack([llm_train_scaled, llm_train_proba, E_train, p_phys_train, raw_num_train])
    X_st_full_test = np.hstack([llm_test_scaled, llm_test_proba, E_test, p_phys_test, raw_num_test])
    X_st_text_train = np.hstack([llm_train_scaled, llm_train_proba])
    X_st_text_test = np.hstack([llm_test_scaled, llm_test_proba])
    X_st_phys_train = np.hstack([E_train, p_phys_train, raw_num_train])
    X_st_phys_test = np.hstack([E_test, p_phys_test, raw_num_test])
    specialist_sets = {
        'text_only': (X_st_text_train, X_st_text_test),
        'physical_only': (X_st_phys_train, X_st_phys_test),
        'residual_fusion': (X_st_full_train, X_st_full_test),
    }
    specialist_results = {}
    for name, (Xtr, Xte) in specialist_sets.items():
        x_sub = Xtr[train_idx][nontrivial_sub_mask]
        x_val = Xtr[val_idx][nontrivial_val_mask]
        y_sub_bin = (y_subtrain[nontrivial_sub_mask] == 1).astype(int)
        y_val_bin = (y_val[nontrivial_val_mask] == 1).astype(int)
        y_test_bin = (y_test[nontrivial_test_mask] == 1).astype(int)
        w_stage = heuristic_xgb.class_weights(y_sub_bin)
        sw_stage = np.asarray([w_stage[int(v)] for v in y_sub_bin], dtype=float)
        spec_model = _fit_binary_gate_with_fallback(
            x_sub, y_sub_bin, x_val, y_val_bin, sw_stage,
            max_depth=variant.get('specialist_max_depth', 4), learning_rate=variant.get('specialist_learning_rate', 0.01), n_estimators=variant.get('specialist_n_estimators', 800),
        )
        val_probs = spec_model.predict_proba(Xtr[val_idx])[:, 1]
        test_probs = spec_model.predict_proba(Xte)[:, 1]
        subset_val_pred = (spec_model.predict_proba(x_val)[:, 1] >= 0.5).astype(int)
        subset_test_pred = (spec_model.predict_proba(Xte[nontrivial_test_mask])[:, 1] >= 0.5).astype(int)
        subset_val_metrics = {
            'topological_f1': float(f1_score(y_val_bin, subset_val_pred, pos_label=1)),
            'semimetal_f1': float(f1_score(y_val_bin, subset_val_pred, pos_label=0)),
        }
        subset_test_metrics = {
            'topological_f1': float(f1_score(y_test_bin, subset_test_pred, pos_label=1)),
            'semimetal_f1': float(f1_score(y_test_bin, subset_test_pred, pos_label=0)),
        }
        specialist_results[name] = {'model': spec_model, 'val_probs': val_probs, 'test_probs': test_probs, 'subset_val_metrics': subset_val_metrics, 'subset_test_metrics': subset_test_metrics}
        _save_probs_npy(specialist_dir / f'val_st_probs_{name}.npy', val_probs)
        _save_probs_npy(specialist_dir / f'test_st_probs_{name}.npy', test_probs)
        spec_model.get_booster().save_model(str(specialist_dir / f'st_model_{name}.json'))
    dump_json(specialist_dir / 'metrics_st_subset_val.json', {k: v['subset_val_metrics'] for k, v in specialist_results.items()})
    dump_json(specialist_dir / 'metrics_st_subset_test.json', {k: v['subset_test_metrics'] for k, v in specialist_results.items()})

    llm_subset_val_ti_f1 = specialist_results['text_only']['subset_val_metrics']['topological_f1']
    base_main = main_variants['rule_gate'] if 'rule_gate' in main_variants else next(iter(main_variants.values()))
    chosen_specialist = None
    for name in ['residual_fusion', 'text_only', 'physical_only']:
        sres = specialist_results[name]
        if sres['subset_val_metrics']['topological_f1'] > llm_subset_val_ti_f1 and sres['subset_val_metrics']['semimetal_f1'] >= specialist_results['text_only']['subset_val_metrics']['semimetal_f1'] - 0.002:
            chosen_specialist = name
            break
    if chosen_specialist is None:
        chosen_specialist = 'text_only'

    p_main_val = base_main['val_proba']
    p_main_test = base_main['test_proba']
    p_ti_given_nontrivial_val = specialist_results[chosen_specialist]['val_probs']
    p_ti_given_nontrivial_test = specialist_results[chosen_specialist]['test_probs']
    p_nontrivial_val = p_main_val[:, 0] + p_main_val[:, 1]
    p_nontrivial_test = p_main_test[:, 0] + p_main_test[:, 1]
    p_special_val = np.zeros_like(p_main_val)
    p_special_test = np.zeros_like(p_main_test)
    p_special_val[:, 1] = p_nontrivial_val * p_ti_given_nontrivial_val
    p_special_test[:, 1] = p_nontrivial_test * p_ti_given_nontrivial_test
    p_special_val[:, 0] = p_nontrivial_val * (1.0 - p_ti_given_nontrivial_val)
    p_special_test[:, 0] = p_nontrivial_test * (1.0 - p_ti_given_nontrivial_test)
    p_special_val[:, 2] = p_main_val[:, 2]
    p_special_test[:, 2] = p_main_test[:, 2]

    eta_grid = [float(x) for x in variant.get('eta_grid', [0.0, 0.1, 0.2, 0.3, 0.4])]
    blend_rows = []
    best_blend = None
    best_blend_val = None
    for eta in eta_grid:
        p_blend_val = (1.0 - eta) * p_main_val + eta * p_special_val
        pred = np.argmax(p_blend_val, axis=1)
        m = evaluate(y_val, pred)
        score = float(m['macro_f1'] + 0.1 * _minimum_class_f1(m) + 0.05 * m['accuracy'])
        cond = (
            m['topological']['f1-score'] >= llm_val_metrics['topological']['f1-score'] and
            m['semimetal']['f1-score'] >= llm_val_metrics['semimetal']['f1-score'] - 0.002 and
            m['trivial']['f1-score'] >= llm_val_metrics['trivial']['f1-score'] - 0.002 and
            m['macro_f1'] >= llm_val_metrics['macro_f1']
        )
        row = {'eta': eta, 'score': score, 'eligible': cond, 'macro_f1': float(m['macro_f1']), 'accuracy': float(m['accuracy']), 'topological_f1': float(m['topological']['f1-score']), 'semimetal_f1': float(m['semimetal']['f1-score']), 'trivial_f1': float(m['trivial']['f1-score'])}
        blend_rows.append(row)
        if cond and (best_blend is None or score > best_blend['score']):
            best_blend = row
            best_blend_val = p_blend_val
    if best_blend is None:
        best_blend = max(blend_rows, key=lambda r: r['score'])
        best_blend_val = (1.0 - best_blend['eta']) * p_main_val + best_blend['eta'] * p_special_val
    p_blend_test = (1.0 - best_blend['eta']) * p_main_test + best_blend['eta'] * p_special_test
    final_val_pred = np.argmax(best_blend_val, axis=1)
    final_test_pred = np.argmax(p_blend_test, axis=1)
    final_val_metrics = evaluate(y_val, final_val_pred)
    final_test_metrics = evaluate(y_test, final_test_pred)
    pd.DataFrame(blend_rows).to_csv(final_blend_dir / 'blend_grid_val.csv', index=False)
    dump_json(final_blend_dir / 'best_blend_config.json', {'chosen_specialist': chosen_specialist, 'best_blend': best_blend})
    dump_json(final_blend_dir / 'best_blend_test_metrics.json', final_test_metrics)

    ablation_results['llm_only'] = {'validation': llm_val_metrics, 'heldout_test': llm_test_metrics}
    if variant19_val_metrics is not None:
        ablation_results['variant19_validation_reference'] = {'validation': variant19_val_metrics}
    ablation_results['st_specialist_blend'] = {'validation': final_val_metrics, 'heldout_test': final_test_metrics}
    dump_json(out_dir / 'outputs' / 'ablations.json', ablation_results)

    dump_json(diagnostics_dir / 'llm_vs_candidate_global.json', _llm_vs_candidate_table(y_test, llm_test_pred, final_test_pred))
    dump_json(diagnostics_dir / 'llm_vs_candidate_by_class.json', _class_help_hurt_summary(y_test, llm_test_pred, final_test_pred))
    dump_json(diagnostics_dir / 'confidence_group_help_hurt.json', _confidence_group_help_hurt(y_test, llm_test_pred, final_test_pred, llm_test_summary[:, 5]))
    dump_json(diagnostics_dir / 'error_transition_summary.json', _error_transition_summary(y_test, final_test_pred))
    dump_json(diagnostics_dir / 'feature_family_contribution.json', family_rows)

    dump_json(out_dir / 'validation_metrics.json', final_val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', final_test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'family_names': family_names,
        'family_weights': family_weights,
        'raw_numeric_block': raw_num_cols,
        'num_small_features': num_small_cols,
        'num_medium_features': num_medium_cols,
    })
    dump_json(out_dir / 'workflow_summary.json', {
        'llm_validation_metrics': llm_val_metrics,
        'llm_test_metrics': llm_test_metrics,
        'variant19_validation_metrics': variant19_val_metrics,
        'chosen_rule_gate_best': main_variants.get('rule_gate', {}).get('best_row'),
        'chosen_specialist': chosen_specialist,
        'best_blend': best_blend,
    })
    return final_val_metrics, final_test_metrics

def fit_threshold_tuned_variant(base, variant, out_dir: Path):
    x_train_full, x_test_full = make_feature_frames(base, variant)
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']
    x_subtrain = x_train_full.iloc[train_idx]
    x_val = x_train_full.iloc[val_idx]
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)
    model = build_model('cuda', max_depth=variant.get('max_depth', 4), learning_rate=variant.get('learning_rate', 0.01), n_estimators=variant.get('n_estimators', 1000))
    try:
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    except Exception:
        model = build_model('cpu', max_depth=variant.get('max_depth', 4), learning_rate=variant.get('learning_rate', 0.01), n_estimators=variant.get('n_estimators', 1000))
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')

    val_proba = model.predict_proba(x_val)
    threshold_objective = variant.get('threshold_objective', 'macro_f1')
    best = _search_thresholds(y_val, val_proba, metric=threshold_objective)
    thresholds = best['thresholds']

    val_pred = _predict_with_thresholds(val_proba, thresholds)
    test_pred = _predict_with_thresholds(model.predict_proba(x_test_full), thresholds)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {'feature_names': x_train_full.columns.tolist()})
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'thresholds.json', {
        'thresholds': thresholds,
        'objective': threshold_objective,
        'validation_search_result': best,
    })
    model.get_booster().save_model(str(out_dir / 'model.json'))
    return val_metrics, test_metrics




class NumericAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class RouterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def _predict_xgb_logits(model, x_df):
    proba = np.clip(model.predict_proba(x_df), 1e-8, 1.0)
    return np.log(proba)


def _select_router_features(base, variant, out_dir: Path):
    x_train_num = base['x_train_num']
    cols = [c for c in ROUTER_FEATURES if c in x_train_num.columns]
    dump_json(out_dir / 'router_feature_names.json', {'router_features': cols})
    return cols


def fit_autoencoder_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    x_train_num = base['x_train_num'].copy()
    x_test_num = base['x_test_num'].copy()
    num_cols = [c for c in x_train_num.columns if c not in {'label'}]

    train_num = x_train_num[num_cols].to_numpy(dtype=np.float32)
    test_num = x_test_num[num_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(train_num[train_idx])
    train_num = scaler.transform(train_num).astype(np.float32)
    test_num = scaler.transform(test_num).astype(np.float32)
    joblib.dump(scaler, out_dir / 'numeric_scaler.pkl')

    latent_dim = int(variant.get('latent_dim', 16))
    hidden_dim = int(variant.get('ae_hidden_dim', 64))
    dropout = float(variant.get('ae_dropout', 0.1))
    lr = float(variant.get('ae_learning_rate', 1e-3))
    weight_decay = float(variant.get('ae_weight_decay', 1e-5))
    batch_size = int(variant.get('ae_batch_size', 256))
    epochs = int(variant.get('ae_epochs', 80))
    patience = int(variant.get('ae_patience', 10))
    noise_std = float(variant.get('ae_noise_std', 0.02))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = NumericAutoencoder(input_dim=train_num.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_tensor = torch.tensor(train_num[train_idx], dtype=torch.float32)
    val_tensor = torch.tensor(train_num[val_idx], dtype=torch.float32)
    train_ds = TensorDataset(train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float('inf')
    no_improve = 0
    rng = np.random.default_rng(SEED)

    for _epoch in range(epochs):
        ae.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            noisy = batch + torch.tensor(rng.normal(0.0, noise_std, size=batch.shape), dtype=torch.float32, device=device)
            optimizer.zero_grad()
            recon, _ = ae(noisy)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
        ae.eval()
        with torch.no_grad():
            recon, _ = ae(val_tensor.to(device))
            val_loss = float(loss_fn(recon, val_tensor.to(device)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
    ae.load_state_dict(best_state)
    ae.eval()
    with torch.no_grad():
        train_latent = ae.encoder(torch.tensor(train_num, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
        test_latent = ae.encoder(torch.tensor(test_num, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)

    latent_cols = [f'NumAE_{i}' for i in range(train_latent.shape[1])]
    train_emb_df = base['train_emb_df'].copy().reset_index(drop=True)
    test_emb_df = base['test_emb_df'].copy().reset_index(drop=True)
    latent_train_df = pd.DataFrame(train_latent, columns=latent_cols)
    latent_test_df = pd.DataFrame(test_latent, columns=latent_cols)
    x_train_full = pd.concat([train_emb_df, latent_train_df], axis=1)
    x_test_full = pd.concat([test_emb_df, latent_test_df], axis=1)

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)
    model = build_model('cuda', max_depth=variant.get('max_depth', 4), learning_rate=variant.get('learning_rate', 0.01), n_estimators=variant.get('n_estimators', 1000))
    x_subtrain = x_train_full.iloc[train_idx]
    x_val = x_train_full.iloc[val_idx]
    try:
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    except Exception:
        model = build_model('cpu', max_depth=variant.get('max_depth', 4), learning_rate=variant.get('learning_rate', 0.01), n_estimators=variant.get('n_estimators', 1000))
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test_full)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {'feature_names': x_train_full.columns.tolist()})
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'autoencoder_config.json', {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'noise_std': noise_std,
        'numeric_columns': num_cols,
        'val_reconstruction_loss': best_val,
    })
    torch.save(best_state, out_dir / 'numeric_autoencoder_state.pt')
    model.get_booster().save_model(str(out_dir / 'model.json'))
    return val_metrics, test_metrics


def fit_moe_routed_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    llm_variant = {'feature_strategy': 'llm_only', 'max_depth': variant.get('expert_max_depth', 4), 'learning_rate': variant.get('expert_learning_rate', 0.01), 'n_estimators': variant.get('expert_n_estimators', 800)}
    numeric_strategy = variant.get('numeric_expert_feature_strategy', 'concat_curated')
    num_variant = {'feature_strategy': numeric_strategy, 'llm_scale': 0.0, 'numeric_scale': 1.0, 'max_depth': variant.get('expert_max_depth', 4), 'learning_rate': variant.get('expert_learning_rate', 0.01), 'n_estimators': variant.get('expert_n_estimators', 800)}

    x_llm_train, x_llm_test = make_feature_frames(base, llm_variant)
    x_num_train, x_num_test = make_feature_frames(base, num_variant)
    zero_cols = [c for c in x_num_train.columns if c.startswith('Bert_')]
    if zero_cols:
        x_num_train = x_num_train.drop(columns=zero_cols)
        x_num_test = x_num_test.drop(columns=zero_cols)

    x_llm_sub = x_llm_train.iloc[train_idx]
    x_llm_val = x_llm_train.iloc[val_idx]
    x_num_sub = x_num_train.iloc[train_idx]
    x_num_val = x_num_train.iloc[val_idx]
    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    llm_expert = build_model('cuda', max_depth=llm_variant['max_depth'], learning_rate=llm_variant['learning_rate'], n_estimators=llm_variant['n_estimators'])
    try:
        llm_expert.fit(x_llm_sub, y_subtrain, sample_weight=sample_weights, eval_set=[(x_llm_val, y_val)], verbose=False)
    except Exception:
        llm_expert = build_model('cpu', max_depth=llm_variant['max_depth'], learning_rate=llm_variant['learning_rate'], n_estimators=llm_variant['n_estimators'])
        llm_expert.fit(x_llm_sub, y_subtrain, sample_weight=sample_weights, eval_set=[(x_llm_val, y_val)], verbose=False)
    llm_expert.get_booster().set_param({'device': 'cpu'})
    llm_expert.set_params(device='cpu')

    num_expert = build_model('cuda', max_depth=num_variant['max_depth'], learning_rate=num_variant['learning_rate'], n_estimators=num_variant['n_estimators'])
    try:
        num_expert.fit(x_num_sub, y_subtrain, sample_weight=sample_weights, eval_set=[(x_num_val, y_val)], verbose=False)
    except Exception:
        num_expert = build_model('cpu', max_depth=num_variant['max_depth'], learning_rate=num_variant['learning_rate'], n_estimators=num_variant['n_estimators'])
        num_expert.fit(x_num_sub, y_subtrain, sample_weight=sample_weights, eval_set=[(x_num_val, y_val)], verbose=False)
    num_expert.get_booster().set_param({'device': 'cpu'})
    num_expert.set_params(device='cpu')

    llm_train_logits = _predict_xgb_logits(llm_expert, x_llm_train)
    llm_test_logits = _predict_xgb_logits(llm_expert, x_llm_test)
    num_train_logits = _predict_xgb_logits(num_expert, x_num_train)
    num_test_logits = _predict_xgb_logits(num_expert, x_num_test)

    router_cols = _select_router_features(base, variant, out_dir)
    router_scaler = StandardScaler()
    router_train = base['x_train_num'][router_cols].to_numpy(dtype=np.float32)
    router_test = base['x_test_num'][router_cols].to_numpy(dtype=np.float32)
    router_scaler.fit(router_train[train_idx])
    router_train = router_scaler.transform(router_train).astype(np.float32)
    router_test = router_scaler.transform(router_test).astype(np.float32)
    joblib.dump(router_scaler, out_dir / 'router_scaler.pkl')

    router_train_tensor = torch.tensor(router_train, dtype=torch.float32)
    router_test_tensor = torch.tensor(router_test, dtype=torch.float32)
    llm_train_tensor = torch.tensor(llm_train_logits, dtype=torch.float32)
    num_train_tensor = torch.tensor(num_train_logits, dtype=torch.float32)
    llm_test_tensor = torch.tensor(llm_test_logits, dtype=torch.float32)
    num_test_tensor = torch.tensor(num_test_logits, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_all, dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    router = RouterNet(input_dim=router_train.shape[1], hidden_dim=variant.get('router_hidden_dim', 32), dropout=variant.get('router_dropout', 0.1)).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=variant.get('router_learning_rate', 1e-3), weight_decay=variant.get('router_weight_decay', 1e-4))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([weights[i] for i in range(3)], dtype=torch.float32, device=device))

    train_ds = TensorDataset(
        router_train_tensor[train_idx],
        llm_train_tensor[train_idx],
        num_train_tensor[train_idx],
        y_train_tensor[train_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=variant.get('router_batch_size', 256), shuffle=True)

    best_state = None
    best_macro_f1 = -1.0
    patience = variant.get('router_patience', 10)
    epochs_no_improve = 0
    n_epochs = variant.get('router_epochs', 80)

    def predict_fused(route_arr, llm_arr, num_arr):
        router.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(route_arr), 512):
                route_batch = torch.tensor(route_arr[start:start+512], dtype=torch.float32, device=device)
                llm_batch = torch.tensor(llm_arr[start:start+512], dtype=torch.float32, device=device)
                num_batch = torch.tensor(num_arr[start:start+512], dtype=torch.float32, device=device)
                gate = router(route_batch)
                fused = gate[:, 0:1] * llm_batch + gate[:, 1:2] * num_batch
                preds.append(torch.argmax(fused, dim=1).cpu().numpy())
        return np.concatenate(preds)

    for _epoch in range(n_epochs):
        router.train()
        for route_batch, llm_batch, num_batch, y_batch in train_loader:
            route_batch = route_batch.to(device)
            llm_batch = llm_batch.to(device)
            num_batch = num_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            gate = router(route_batch)
            fused = gate[:, 0:1] * llm_batch + gate[:, 1:2] * num_batch
            loss = loss_fn(fused, y_batch)
            loss.backward()
            optimizer.step()
        val_pred = predict_fused(router_train[val_idx], llm_train_logits[val_idx], num_train_logits[val_idx])
        val_metrics = evaluate(y_val, val_pred)
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
    router.load_state_dict(best_state)

    val_pred = predict_fused(router_train[val_idx], llm_train_logits[val_idx], num_train_logits[val_idx])
    test_pred = predict_fused(router_test, llm_test_logits, num_test_logits)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'router_features': router_cols,
        'llm_expert_features': list(x_llm_train.columns),
        'numeric_expert_features': list(x_num_train.columns),
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'moe_config.json', {
        'numeric_expert_feature_strategy': numeric_strategy,
        'router_hidden_dim': variant.get('router_hidden_dim', 32),
        'router_dropout': variant.get('router_dropout', 0.1),
        'router_learning_rate': variant.get('router_learning_rate', 1e-3),
        'router_weight_decay': variant.get('router_weight_decay', 1e-4),
        'router_batch_size': variant.get('router_batch_size', 256),
        'router_epochs': n_epochs,
        'router_patience': patience,
    })
    llm_expert.get_booster().save_model(str(out_dir / 'moe_llm_expert.json'))
    num_expert.get_booster().save_model(str(out_dir / 'moe_numeric_expert.json'))
    torch.save(best_state, out_dir / 'moe_router_state.pt')
    return val_metrics, test_metrics

def fit_hierarchical_variant(base, variant, out_dir: Path):
    x_train_full, x_test_full = make_feature_frames(base, variant)
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]

    stage1_threshold = float(variant.get('stage1_threshold', 0.50))
    stage2_threshold = float(variant.get('stage2_threshold', 0.50))
    tune_thresholds = bool(variant.get('tune_thresholds', True))

    y_subtrain_stage1 = (y_subtrain != 2).astype(int)  # 0=trivial, 1=non-trivial
    y_val_stage1 = (y_val != 2).astype(int)
    y_test_stage1 = (y_test != 2).astype(int)

    x_subtrain = x_train_full.iloc[train_idx]
    x_val = x_train_full.iloc[val_idx]

    weights_stage1 = _resolve_binary_class_weights(y_subtrain_stage1, variant, 'stage1_positive_multiplier')
    sample_weights_stage1 = _weights_to_sample_weights(y_subtrain_stage1, weights_stage1)
    stage1 = build_binary_model('cuda', max_depth=variant.get('stage1_max_depth', 4), learning_rate=variant.get('stage1_learning_rate', 0.01), n_estimators=variant.get('stage1_n_estimators', 800))
    try:
        stage1.fit(x_subtrain, y_subtrain_stage1, sample_weight=sample_weights_stage1, eval_set=[(x_val, y_val_stage1)], verbose=False)
    except Exception:
        stage1 = build_binary_model('cpu', max_depth=variant.get('stage1_max_depth', 4), learning_rate=variant.get('stage1_learning_rate', 0.01), n_estimators=variant.get('stage1_n_estimators', 800))
        fit_kwargs = {'eval_set': [(x_val, y_val_stage1)], 'verbose': False}
        if sample_weights_stage1 is not None:
            fit_kwargs['sample_weight'] = sample_weights_stage1
        stage1.fit(x_subtrain, y_subtrain_stage1, **fit_kwargs)
    stage1.get_booster().set_param({'device': 'cpu'})
    stage1.set_params(device='cpu')

    nontrivial_sub_mask = y_subtrain != 2
    nontrivial_val_mask = y_val != 2
    x_subtrain_stage2 = x_subtrain.iloc[nontrivial_sub_mask]
    y_subtrain_stage2 = y_subtrain[nontrivial_sub_mask]
    x_val_stage2 = x_val.iloc[nontrivial_val_mask]
    y_val_stage2 = y_val[nontrivial_val_mask]

    # Map semimetal -> 0, topological -> 1 for stage 2.
    y_subtrain_stage2_bin = (y_subtrain_stage2 == 1).astype(int)
    y_val_stage2_bin = (y_val_stage2 == 1).astype(int)

    if 'stage2_topological_multiplier' not in variant and 'ti_weight_multiplier' in variant:
        variant['stage2_topological_multiplier'] = variant['ti_weight_multiplier']
    weights_stage2 = _resolve_binary_class_weights(y_subtrain_stage2_bin, variant, 'stage2_topological_multiplier')
    sample_weights_stage2 = _weights_to_sample_weights(y_subtrain_stage2_bin, weights_stage2)
    stage2 = build_binary_model('cuda', max_depth=variant.get('stage2_max_depth', 4), learning_rate=variant.get('stage2_learning_rate', 0.01), n_estimators=variant.get('stage2_n_estimators', 800))
    try:
        stage2.fit(x_subtrain_stage2, y_subtrain_stage2_bin, sample_weight=sample_weights_stage2, eval_set=[(x_val_stage2, y_val_stage2_bin)], verbose=False)
    except Exception:
        stage2 = build_binary_model('cpu', max_depth=variant.get('stage2_max_depth', 4), learning_rate=variant.get('stage2_learning_rate', 0.01), n_estimators=variant.get('stage2_n_estimators', 800))
        fit_kwargs = {'eval_set': [(x_val_stage2, y_val_stage2_bin)], 'verbose': False}
        if sample_weights_stage2 is not None:
            fit_kwargs['sample_weight'] = sample_weights_stage2
        stage2.fit(x_subtrain_stage2, y_subtrain_stage2_bin, **fit_kwargs)
    stage2.get_booster().set_param({'device': 'cpu'})
    stage2.set_params(device='cpu')

    def predict_hierarchical(x_df, threshold1, threshold2):
        stage1_proba = stage1.predict_proba(x_df)
        nontrivial_prob = stage1_proba[:, 1]
        stage2_proba = stage2.predict_proba(x_df)
        preds = []
        for idx in range(len(x_df)):
            if nontrivial_prob[idx] < threshold1:
                preds.append(2)
            else:
                preds.append(1 if stage2_proba[idx, 1] >= threshold2 else 0)
        return np.asarray(preds, dtype=int)

    if tune_thresholds:
        grid = np.round(np.linspace(0.30, 0.80, 11), 2)
        best = {'thresholds': [stage1_threshold, stage2_threshold], 'metrics': None, 'score': -1.0}
        for t1, t2 in itertools.product(grid, repeat=2):
            preds = predict_hierarchical(x_val, float(t1), float(t2))
            metrics = evaluate(y_val, preds)
            score = metrics['macro_f1']
            if (score > best['score']) or (np.isclose(score, best['score']) and metrics['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)):
                best = {'thresholds': [float(t1), float(t2)], 'metrics': metrics, 'score': float(score)}
        stage1_threshold, stage2_threshold = best['thresholds']
    else:
        best = {'thresholds': [stage1_threshold, stage2_threshold], 'metrics': None, 'score': None}

    val_pred = predict_hierarchical(x_val, stage1_threshold, stage2_threshold)
    test_pred = predict_hierarchical(x_test_full, stage1_threshold, stage2_threshold)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {'feature_names': x_train_full.columns.tolist()})
    dump_json(out_dir / 'class_weights_stage1.json', {str(k): v for k, v in weights_stage1.items()})
    dump_json(out_dir / 'class_weights_stage2.json', {str(k): v for k, v in weights_stage2.items()})
    dump_json(out_dir / 'hierarchical_thresholds.json', {
        'stage1_threshold_nontrivial': stage1_threshold,
        'stage2_threshold_topological': stage2_threshold,
        'threshold_search_result': best,
    })
    stage1.get_booster().save_model(str(out_dir / 'stage1_model.json'))
    stage2.get_booster().save_model(str(out_dir / 'stage2_model.json'))
    return val_metrics, test_metrics




def _fit_xgb_with_fallback(x_subtrain, y_subtrain, x_val, y_val, sample_weights, *, seed=SEED, max_depth=4, learning_rate=0.01, n_estimators=1000):
    model = build_model('cuda', seed=seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
    try:
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    except Exception:
        model = build_model('cpu', seed=seed, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
        model.fit(x_subtrain, y_subtrain, sample_weight=sample_weights, eval_set=[(x_val, y_val)], verbose=False)
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    return model


def _probability_summary_features(proba):
    arr = np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0)
    top2 = np.sort(arr, axis=1)[:, -2:]
    confidence = np.max(arr, axis=1, keepdims=True)
    entropy = -np.sum(arr * np.log(arr), axis=1, keepdims=True)
    margin = (top2[:, 1] - top2[:, 0])[:, None]
    return np.hstack([arr, confidence, entropy, margin])


def _blend_probabilities(base_proba, correction_proba, alpha):
    base = np.clip(np.asarray(base_proba, dtype=float), 1e-8, 1.0)
    corr = np.clip(np.asarray(correction_proba, dtype=float), 1e-8, 1.0)
    blended = np.power(base, 1.0 - alpha) * np.power(corr, alpha)
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


def fit_residual_correction_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    num_cols = [c for c in CURATED_NUMERIC_FEATURES if c in base['x_train_num'].columns]
    num_train = base['x_train_num'][num_cols].to_numpy(dtype=np.float32)
    num_test = base['x_test_num'][num_cols].to_numpy(dtype=np.float32)

    llm_scaler = StandardScaler()
    num_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    num_scaler.fit(num_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    num_train = num_scaler.transform(num_train).astype(np.float32)
    num_test = num_scaler.transform(num_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    text_expert = _fit_xgb_with_fallback(
        llm_train[train_idx],
        y_subtrain,
        llm_train[val_idx],
        y_val,
        sample_weights,
        max_depth=variant.get('text_max_depth', 4),
        learning_rate=variant.get('text_learning_rate', 0.01),
        n_estimators=variant.get('text_n_estimators', 1000),
    )

    text_train_proba = text_expert.predict_proba(llm_train)
    text_val_proba = text_expert.predict_proba(llm_train[val_idx])
    text_test_proba = text_expert.predict_proba(llm_test)

    text_train_feats = _probability_summary_features(text_train_proba)
    text_val_feats = _probability_summary_features(text_val_proba)
    text_test_feats = _probability_summary_features(text_test_proba)

    meta_train = np.hstack([text_train_feats, num_train])
    meta_val = np.hstack([text_val_feats, num_train[val_idx]])
    meta_test = np.hstack([text_test_feats, num_test])

    uncertainty_boost = float(variant.get('uncertainty_boost', 0.5))
    train_confidence = np.max(np.clip(text_train_proba[train_idx], 1e-8, 1.0), axis=1)
    meta_sample_weights = sample_weights * (1.0 + uncertainty_boost * (1.0 - train_confidence))

    residual_model = _fit_xgb_with_fallback(
        meta_train[train_idx],
        y_subtrain,
        meta_val,
        y_val,
        meta_sample_weights,
        max_depth=variant.get('meta_max_depth', 4),
        learning_rate=variant.get('meta_learning_rate', 0.01),
        n_estimators=variant.get('meta_n_estimators', 1000),
    )

    blend_grid = variant.get('blend_grid', [0.0, 0.25, 0.5, 0.75, 1.0])
    best = {'alpha': 0.0, 'metrics': None, 'score': -1.0}
    best_val_proba = None
    for alpha in blend_grid:
        blended = _blend_probabilities(text_val_proba, residual_model.predict_proba(meta_val), float(alpha))
        preds = np.argmax(blended, axis=1)
        metrics = evaluate(y_val, preds)
        score = metrics['macro_f1']
        if (score > best['score']) or (np.isclose(score, best['score']) and metrics['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)):
            best = {'alpha': float(alpha), 'metrics': metrics, 'score': float(score)}
            best_val_proba = blended

    final_alpha = best['alpha']
    val_proba = best_val_proba if best_val_proba is not None else _blend_probabilities(text_val_proba, residual_model.predict_proba(meta_val), final_alpha)
    test_proba = _blend_probabilities(text_test_proba, residual_model.predict_proba(meta_test), final_alpha)
    threshold_objective = variant.get('threshold_objective', 'macro_f1')
    thresholds = _search_thresholds(y_val, val_proba, metric=threshold_objective)['thresholds']
    val_pred = _predict_with_thresholds(val_proba, thresholds)
    test_pred = _predict_with_thresholds(test_proba, thresholds)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'text_features': [f'text_summary_{i}' for i in range(text_train_feats.shape[1])],
        'numeric_features': num_cols,
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'residual_fusion.json', {
        'blend_grid': [float(x) for x in blend_grid],
        'selected_alpha': final_alpha,
        'threshold_objective': threshold_objective,
        'validation_search_result': best,
        'uncertainty_boost': uncertainty_boost,
    })
    text_expert.get_booster().save_model(str(out_dir / 'text_expert_model.json'))
    residual_model.get_booster().save_model(str(out_dir / 'residual_correction_model.json'))
    return val_metrics, test_metrics


def fit_supervised_numeric_bottleneck_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    num_cols = [c for c in base['x_train_num'].columns if c != 'label']
    num_train = base['x_train_num'][num_cols].to_numpy(dtype=np.float32)
    num_test = base['x_test_num'][num_cols].to_numpy(dtype=np.float32)

    llm_scaler = StandardScaler()
    num_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    num_scaler.fit(num_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    num_train = num_scaler.transform(num_train).astype(np.float32)
    num_test = num_scaler.transform(num_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    num_train = np.nan_to_num(num_train, nan=0.0, posinf=0.0, neginf=0.0)
    num_test = np.nan_to_num(num_test, nan=0.0, posinf=0.0, neginf=0.0)

    lda_components = int(variant.get('lda_components', 2))
    lda = LinearDiscriminantAnalysis(n_components=min(lda_components, len(np.unique(y_subtrain)) - 1))
    lda.fit(num_train[train_idx], y_subtrain)
    num_train_b = lda.transform(num_train)
    num_test_b = lda.transform(num_test)
    joblib.dump(lda, out_dir / 'numeric_lda_model.pkl')

    x_train_full = np.hstack([llm_train, num_train_b])
    x_test_full = np.hstack([llm_test, num_test_b])
    x_subtrain = x_train_full[train_idx]
    x_val = x_train_full[val_idx]

    model = _fit_xgb_with_fallback(
        x_subtrain,
        y_subtrain,
        x_val,
        y_val,
        sample_weights,
        max_depth=variant.get('max_depth', 4),
        learning_rate=variant.get('learning_rate', 0.01),
        n_estimators=variant.get('n_estimators', 1000),
    )

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test_full)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'llm_features': [f'Bert_{i}' for i in range(llm_train.shape[1])],
        'numeric_bottleneck_features': [f'Numeric_LDA_{i}' for i in range(num_train_b.shape[1])],
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'supervised_bottleneck.json', {
        'lda_components': int(num_train_b.shape[1]),
        'numeric_columns': num_cols,
    })
    model.get_booster().save_model(str(out_dir / 'model.json'))
    return val_metrics, test_metrics


def fit_cross_feature_interaction_variant(base, variant, out_dir: Path):
    x_train_full, x_test_full = make_feature_frames(base, {'feature_strategy': variant.get('feature_strategy', 'concat_curated'), 'out_dir': out_dir})
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    text_cols = [c for c in x_train_full.columns if c.startswith('Bert_')]
    numeric_cols = [c for c in x_train_full.columns if not c.startswith('Bert_')]
    top_text_n = int(variant.get('interaction_top_text_dims', 8))
    interaction_numeric_cols = [c for c in variant.get('interaction_numeric_features', [
        'Trivial_g', 'SM_g', 'TI_SG_prob', 'Is_total_electrons_even?',
        'Mean_d_valence_electrons', 'Mean_f_valence_electrons', 'Bonding_is_moderately_ionic'
    ]) if c in x_train_full.columns]
    text_sel = text_cols[:max(1, min(top_text_n, len(text_cols)))]

    def build_interactions(frame):
        base_num = frame[numeric_cols].copy()
        inter = []
        names = []
        for tcol in text_sel:
            tvals = frame[tcol].to_numpy(dtype=np.float32)
            for ncol in interaction_numeric_cols:
                inter.append((tvals * frame[ncol].to_numpy(dtype=np.float32))[:, None])
                names.append(f'{tcol}__x__{ncol}')
        if inter:
            inter_mat = np.hstack(inter)
            inter_df = pd.DataFrame(inter_mat, columns=names, index=frame.index)
            return pd.concat([frame.reset_index(drop=True), inter_df.reset_index(drop=True)], axis=1)
        return frame

    x_train_int = build_interactions(x_train_full)
    x_test_int = build_interactions(x_test_full)

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    x_subtrain = x_train_int.iloc[train_idx]
    x_val = x_train_int.iloc[val_idx]
    model = _fit_xgb_with_fallback(
        x_subtrain,
        y_subtrain,
        x_val,
        y_val,
        sample_weights,
        max_depth=variant.get('max_depth', 4),
        learning_rate=variant.get('learning_rate', 0.01),
        n_estimators=variant.get('n_estimators', 1000),
    )

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test_int)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {'feature_names': x_train_int.columns.tolist()})
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'interaction_config.json', {
        'interaction_top_text_dims': top_text_n,
        'interaction_numeric_features': interaction_numeric_cols,
        'base_feature_strategy': variant.get('feature_strategy', 'concat_curated'),
    })
    model.get_booster().save_model(str(out_dir / 'model.json'))
    return val_metrics, test_metrics


def fit_text_probability_residual_variant(base, variant, out_dir: Path):
    train_idx = base['train_idx']
    val_idx = base['val_idx']
    y_train_all = base['y_train_all']
    y_test = base['y_test']

    llm_train = base['train_emb_df'].to_numpy(dtype=np.float32)
    llm_test = base['test_emb_df'].to_numpy(dtype=np.float32)
    num_cols = [c for c in CURATED_NUMERIC_FEATURES if c in base['x_train_num'].columns]
    num_train = base['x_train_num'][num_cols].to_numpy(dtype=np.float32)
    num_test = base['x_test_num'][num_cols].to_numpy(dtype=np.float32)

    llm_scaler = StandardScaler()
    num_scaler = StandardScaler()
    llm_scaler.fit(llm_train[train_idx])
    num_scaler.fit(num_train[train_idx])
    llm_train = llm_scaler.transform(llm_train).astype(np.float32)
    llm_test = llm_scaler.transform(llm_test).astype(np.float32)
    num_train = num_scaler.transform(num_train).astype(np.float32)
    num_test = num_scaler.transform(num_test).astype(np.float32)
    joblib.dump(llm_scaler, out_dir / 'llm_scaler.pkl')
    joblib.dump(num_scaler, out_dir / 'numeric_scaler.pkl')

    y_subtrain = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    weights = heuristic_xgb.class_weights(y_subtrain)
    sample_weights = np.asarray([weights[int(label)] for label in y_subtrain], dtype=float)

    text_model = _fit_xgb_with_fallback(
        llm_train[train_idx],
        y_subtrain,
        llm_train[val_idx],
        y_val,
        sample_weights,
        max_depth=variant.get('text_max_depth', 4),
        learning_rate=variant.get('text_learning_rate', 0.01),
        n_estimators=variant.get('text_n_estimators', 1000),
    )
    numeric_model = _fit_xgb_with_fallback(
        num_train[train_idx],
        y_subtrain,
        num_train[val_idx],
        y_val,
        sample_weights,
        max_depth=variant.get('numeric_max_depth', 4),
        learning_rate=variant.get('numeric_learning_rate', 0.01),
        n_estimators=variant.get('numeric_n_estimators', 1000),
    )

    text_train_proba = text_model.predict_proba(llm_train)
    text_val_proba = text_model.predict_proba(llm_train[val_idx])
    text_test_proba = text_model.predict_proba(llm_test)
    num_train_proba = numeric_model.predict_proba(num_train)
    num_val_proba = numeric_model.predict_proba(num_train[val_idx])
    num_test_proba = numeric_model.predict_proba(num_test)

    blend_grid = variant.get('blend_grid', [0.0, 0.25, 0.5, 0.75, 1.0])
    best = {'alpha': 0.0, 'metrics': None, 'score': -1.0}
    best_val_proba = None
    for alpha in blend_grid:
        blended = _blend_probabilities(text_val_proba, num_val_proba, float(alpha))
        preds = np.argmax(blended, axis=1)
        metrics = evaluate(y_val, preds)
        score = metrics['macro_f1']
        if (score > best['score']) or (np.isclose(score, best['score']) and metrics['accuracy'] > (best['metrics']['accuracy'] if best['metrics'] else -1)):
            best = {'alpha': float(alpha), 'metrics': metrics, 'score': float(score)}
            best_val_proba = blended

    final_alpha = best['alpha']
    val_proba = best_val_proba if best_val_proba is not None else _blend_probabilities(text_val_proba, num_val_proba, final_alpha)
    test_proba = _blend_probabilities(text_test_proba, num_test_proba, final_alpha)
    threshold_objective = variant.get('threshold_objective', 'macro_f1')
    thresholds = _search_thresholds(y_val, val_proba, metric=threshold_objective)['thresholds']
    val_pred = _predict_with_thresholds(val_proba, thresholds)
    test_pred = _predict_with_thresholds(test_proba, thresholds)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    dump_json(out_dir / 'validation_metrics.json', val_metrics)
    dump_json(out_dir / 'heldout_test_metrics.json', test_metrics)
    dump_json(out_dir / 'feature_names.json', {
        'text_probability_features': ['text_prob_semimetal', 'text_prob_topological', 'text_prob_trivial'],
        'numeric_probability_features': ['num_prob_semimetal', 'num_prob_topological', 'num_prob_trivial'],
        'numeric_features': num_cols,
    })
    dump_json(out_dir / 'class_weights.json', {str(k): v for k, v in (weights or {}).items()})
    dump_json(out_dir / 'text_probability_residual.json', {
        'blend_grid': [float(x) for x in blend_grid],
        'selected_alpha': final_alpha,
        'threshold_objective': threshold_objective,
        'validation_search_result': best,
    })
    text_model.get_booster().save_model(str(out_dir / 'text_expert_model.json'))
    numeric_model.get_booster().save_model(str(out_dir / 'numeric_correction_model.json'))
    return val_metrics, test_metrics


def run_variant(config_path: Path):
    config = json.loads(config_path.read_text())
    out_dir = config_path.parent / 'output'
    config['out_dir'] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / 'config_used.json', {k: str(v) if isinstance(v, Path) else v for k, v in config.items()})
    config['out_dir'] = out_dir
    split_manifest_path = config.get('split_manifest_path')
    scibert_checkpoint = config.get('scibert_checkpoint')
    base = prepare_base_data(
        out_dir,
        pca_components=config.get('pca_components', 100),
        split_manifest_path=Path(split_manifest_path) if split_manifest_path else None,
        scibert_checkpoint=Path(scibert_checkpoint) if scibert_checkpoint else None,
    )
    if config['variant_type'] == 'stacking':
        val_metrics, test_metrics = fit_stacking_variant(base, config, out_dir)
    elif config['variant_type'] == 'gated_fusion':
        val_metrics, test_metrics = fit_gated_variant(base, config, out_dir)
    elif config['variant_type'] == 'threshold_tuned':
        val_metrics, test_metrics = fit_threshold_tuned_variant(base, config, out_dir)
    elif config['variant_type'] == 'hierarchical_classifier':
        val_metrics, test_metrics = fit_hierarchical_variant(base, config, out_dir)
    elif config['variant_type'] == 'autoencoder_fusion':
        val_metrics, test_metrics = fit_autoencoder_variant(base, config, out_dir)
    elif config['variant_type'] == 'moe_routed_fusion':
        val_metrics, test_metrics = fit_moe_routed_variant(base, config, out_dir)
    elif config['variant_type'] == 'gated_hierarchical_calibrated_fusion':
        val_metrics, test_metrics = fit_gated_hierarchical_calibrated_variant(base, config, out_dir)
    elif config['variant_type'] == 'text_anchored_gated_hierarchical_calibrated_fusion':
        val_metrics, test_metrics = fit_text_anchored_gated_hierarchical_variant(base, config, out_dir)
    elif config['variant_type'] == 'text_anchored_ti_aware_gated_hierarchical_fusion':
        val_metrics, test_metrics = fit_text_anchored_gated_hierarchical_variant(base, config, out_dir)
    elif config['variant_type'] == 'residual_correction_fusion':
        val_metrics, test_metrics = fit_residual_correction_variant(base, config, out_dir)
    elif config['variant_type'] == 'no_pca_supervised_numeric_bottleneck':
        val_metrics, test_metrics = fit_supervised_numeric_bottleneck_variant(base, config, out_dir)
    elif config['variant_type'] == 'cross_feature_interaction_fusion':
        val_metrics, test_metrics = fit_cross_feature_interaction_variant(base, config, out_dir)
    elif config['variant_type'] == 'text_probability_residual_fusion':
        val_metrics, test_metrics = fit_text_probability_residual_variant(base, config, out_dir)
    elif config['variant_type'] == 'balanced_residual_reliability_fusion':
        val_metrics, test_metrics = fit_balanced_residual_reliability_variant(base, config, out_dir)
    elif config['variant_type'] == 'st_specialist_residual_fusion':
        val_metrics, test_metrics = fit_st_specialist_residual_variant(base, config, out_dir)
    elif config['variant_type'] == 'family_probability_residual_fusion':
        val_metrics, test_metrics = fit_family_probability_residual_variant(base, config, out_dir)
    elif config['variant_type'] == 'conservative_selective_residual_fusion':
        val_metrics, test_metrics = fit_st_specialist_residual_variant(base, config, out_dir)
    elif config['variant_type'] == 'llm_anchored_residual_specialist_workflow':
        val_metrics, test_metrics = fit_llm_anchored_residual_specialist_workflow_variant(base, config, out_dir)
    elif config['variant_type'] == 'direct_llm_residual_txl_fusion':
        val_metrics, test_metrics = fit_llm_anchored_residual_specialist_workflow_variant(base, config, out_dir)
    else:
        val_metrics, test_metrics = fit_simple_variant(base, config, out_dir)
    print('Validation accuracy:', val_metrics['accuracy'])
    print('Validation macro-F1:', val_metrics['macro_f1'])
    print('Held-out accuracy:', test_metrics['accuracy'])
    print('Held-out macro-F1:', test_metrics['macro_f1'])
