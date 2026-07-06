import json
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import joblib

ROOT = Path('/path/to/3_classes_classification/weighted_version')
BASE_ROOT = Path('/path/to/3_classes_classification')
TXL_PCA50_DIR = ROOT / 'txl_model_pca50'
TXL_PCA50_OUT = TXL_PCA50_DIR / 'txl_shap_mean_abs.json'
OUT = ROOT / 'features_shap_weighted.pdf'
VALIDATION_SAMPLE_CAP = 1500
COMMON_PIPELINE = ROOT / 'txl_fusion_variants' / 'common_fusion_pipeline.py'
CACHE_DIR = TXL_PCA50_DIR / 'diff_pca_dims' / 'pca_explained_variance_test' / 'cache'
SHARED_SPLIT_PATH = BASE_ROOT / 'label_noise_analysis' / 'shared_split_manifest.json'


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pipeline = load_module(COMMON_PIPELINE, 'txl_fusion_pipeline_for_shap_pca50')


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def pretty_feature_name(name):
    if name == 'Trivial_g':
        return 'Trivial g(M)'
    if name == 'SM_g':
        return 'TSM g(M)'
    if name == 'Is_trivial_g_positive':
        return 'Is trivial g(M) positive?'
    if name == 'Is_sm_g_positive':
        return 'Is TSM g(M) positive?'
    if name == 'Are_both_g_negative':
        return 'Are both g(M) negative?'
    if name == 'Is_trivial_positive_sm_negative':
        return 'Is trivial g(M)+ / TSM g(M)-?'
    if name == 'Is_sm_positive_trivial_negative':
        return 'Is TSM g(M)+ / trivial g(M)-?'
    if name.startswith('Bert_'):
        return name.replace('Bert_', 'PCA-')
    return name.replace('_', ' ')


def load_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    return model


def compute_mean_abs_shap(model, x_frame):
    if len(x_frame) > VALIDATION_SAMPLE_CAP:
        x_frame = x_frame.iloc[:VALIDATION_SAMPLE_CAP].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_frame)

    if isinstance(shap_values, list):
        arr = np.stack([np.abs(v) for v in shap_values], axis=0)
        mean_abs = arr.mean(axis=(0, 1))
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            mean_abs = np.abs(arr).mean(axis=0)
        elif arr.ndim == 3:
            if arr.shape[1] == x_frame.shape[1]:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            elif arr.shape[2] == x_frame.shape[1]:
                mean_abs = np.abs(arr).mean(axis=(0, 1))
            else:
                raise ValueError(f'Unexpected SHAP value shape: {arr.shape}')
        else:
            raise ValueError(f'Unexpected SHAP value shape: {arr.shape}')

    return pd.DataFrame({
        'feature': x_frame.columns.tolist(),
        'mean_abs_shap': mean_abs.tolist(),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def plot_shap_comparison(shap_frames, output_path, top_n=15):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(len(shap_frames), 1, figsize=(7.4, 6.4))
    if len(shap_frames) == 1:
        axes = [axes]

    max_val = 0.0
    processed = []
    for title, df in shap_frames:
        top = df.head(top_n).copy()
        top['feature_pretty'] = [pretty_feature_name(v) for v in top['feature']]
        max_val = max(max_val, float(top['mean_abs_shap'].max()) if not top.empty else 0.0)
        processed.append((title, top))

    panel_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for idx, (ax, (title, top)) in enumerate(zip(axes, processed)):
        sns.barplot(x='mean_abs_shap', y='feature_pretty', data=top, palette='magma', ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Mean |SHAP value|' if idx == len(axes) - 1 else '')
        ax.set_ylabel('')
        ax.set_xlim(0, max_val * 1.08 if max_val > 0 else 1)
        ax.tick_params(axis='y', labelsize=9)
        ax.text(-0.10, 1.03, panel_labels[idx], transform=ax.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Combined SHAP plot saved to {output_path}')


def load_weighted_xgb_validation_frame():
    rows = load_json(ROOT / 'xgb' / 'xgb_data_weighted_shared_split.json')
    feature_names = load_json(ROOT / 'xgb' / 'xgb_feature_names.json')['xgb_feature_names']
    manifest = load_json(SHARED_SPLIT_PATH)
    val_idx = manifest['validation_indices']
    df = pd.DataFrame(rows)
    x_val = df.iloc[val_idx][feature_names].reset_index(drop=True)
    return x_val


def load_pca50_txl_validation_frame():
    manifest = load_json(SHARED_SPLIT_PATH)
    train_idx = np.asarray(manifest['train_indices'])
    val_idx = np.asarray(manifest['validation_indices'])

    clean_training_data = pipeline.heuristic_xgb.load_json(pipeline.TRAINING_DATA_PATH)
    clean_test_data = pipeline.heuristic_xgb.load_json(pipeline.TEST_DATA_PATH)
    x_train_num, _, _, _ = pipeline.build_numeric_feature_frames(clean_training_data, clean_test_data, train_idx)

    pca = joblib.load(TXL_PCA50_DIR / 'output' / 'pca_model.pkl')
    val_embeddings = np.load(CACHE_DIR / 'val_embeddings.npy')
    val_emb_reduced = pca.transform(val_embeddings)
    val_emb_df = pd.DataFrame(val_emb_reduced, columns=[f'Bert_{i}' for i in range(val_emb_reduced.shape[1])])

    curated_cols = [c for c in pipeline.CURATED_NUMERIC_FEATURES if c in x_train_num.columns]
    x_val = pd.concat(
        [
            val_emb_df.reset_index(drop=True),
            x_train_num.iloc[val_idx][curated_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    return x_val


def main():
    xgb_model = load_model(ROOT / 'xgb' / 'xgb_model.json')
    txl_stage1_model = load_model(TXL_PCA50_DIR / 'output' / 'hierarchical_stage1_model.json')

    xgb_x_val = load_weighted_xgb_validation_frame()
    txl_x_val = load_pca50_txl_validation_frame()

    xgb_shap = compute_mean_abs_shap(xgb_model, xgb_x_val)
    txl_shap = compute_mean_abs_shap(txl_stage1_model, txl_x_val)

    xgb_shap.to_json(ROOT / 'xgb' / 'xgb_shap_mean_abs.json', orient='records', indent=2)
    txl_shap.to_json(TXL_PCA50_OUT, orient='records', indent=2)

    plot_shap_comparison(
        [
            ('Baseline XGB SHAP', xgb_shap),
            ('TXL Fusion SHAP', txl_shap),
        ],
        OUT,
        top_n=15,
    )


if __name__ == '__main__':
    main()
