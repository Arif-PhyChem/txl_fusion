import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

ROOT = Path('/path/to/3_classes_classification/weighted_version')
SHARED_SPLIT_PATH = Path('/path/to/3_classes_classification/label_noise_analysis/shared_split_manifest.json')
VALIDATION_SAMPLE_CAP = 1500


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_feature_importance(file_path):
    data = load_json(file_path)
    features = [item['feature'] for item in data]
    importances = [item['importance'] for item in data]
    return features, importances


def pretty_feature_name(name):
    if name.startswith('Bert_'):
        return name.replace('Bert_', 'PCA-')
    return name.replace('_', ' ')


def plot_weighted_importance(panels, output_path, top_n=15):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.4, 6.4))
    if len(panels) == 1:
        axes = [axes]

    max_importance = 0.0
    processed = []
    for title, file_path in panels:
        features, importances = load_feature_importance(file_path)
        features = [pretty_feature_name(f) for f in features[:top_n]]
        importances = importances[:top_n]
        max_importance = max(max_importance, max(importances) if importances else 0.0)
        processed.append((title, features, importances))

    panel_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for idx, (ax, (title, features, importances)) in enumerate(zip(axes, processed)):
        sns.barplot(x=importances, y=features, palette='viridis', ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Importance Score' if idx == len(axes) - 1 else '')
        ax.set_ylabel('')
        ax.set_xlim(0, max_importance * 1.08 if max_importance > 0 else 1)
        ax.tick_params(axis='y', labelsize=9)
        ax.text(-0.10, 1.03, panel_labels[idx], transform=ax.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Combined importance plot saved to {output_path}')


def load_validation_indices():
    manifest = load_json(SHARED_SPLIT_PATH)
    return manifest['validation_indices']


def load_weighted_xgb_validation_frame():
    rows = load_json(ROOT / 'xgb' / 'xgb_data_weighted_shared_split.json')
    feature_names = load_json(ROOT / 'xgb' / 'xgb_feature_names.json')['xgb_feature_names']
    val_idx = load_validation_indices()
    df = pd.DataFrame(rows)
    x_val = df.iloc[val_idx][feature_names].reset_index(drop=True)
    return x_val, feature_names


def load_weighted_txl_validation_frame():
    data = np.load(ROOT / 'txl_model' / 'data.npz', allow_pickle=True)
    feature_names = load_json(ROOT / 'txl_model' / 'txl_feature_names.json')['txl_feature_names']
    val_idx = load_validation_indices()
    x_val = pd.DataFrame(data['features'][val_idx], columns=feature_names).reset_index(drop=True)
    return x_val, feature_names


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
            # robust to either (samples, features, classes) or (samples, classes, features)
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


def main():
    importance_panels = [
        ('Baseline XGB', ROOT / 'xgb' / 'xgb_features_data.json'),
        ('TXL Fusion', ROOT / 'txl_model' / 'txl_features_data.json'),
    ]
    plot_weighted_importance(importance_panels, ROOT / 'features_importance_weighted.pdf', top_n=15)

    xgb_model = load_model(ROOT / 'xgb' / 'xgb_model.json')
    txl_model = load_model(ROOT / 'txl_model' / 'txl_model.json')
    xgb_x_val, _ = load_weighted_xgb_validation_frame()
    txl_x_val, _ = load_weighted_txl_validation_frame()

    xgb_shap = compute_mean_abs_shap(xgb_model, xgb_x_val)
    txl_shap = compute_mean_abs_shap(txl_model, txl_x_val)

    xgb_shap.to_json(ROOT / 'xgb' / 'xgb_shap_mean_abs.json', orient='records', indent=2)
    txl_shap.to_json(ROOT / 'txl_model' / 'txl_shap_mean_abs.json', orient='records', indent=2)

    plot_shap_comparison(
        [
            ('Baseline XGB SHAP', xgb_shap),
            ('TXL Fusion SHAP', txl_shap),
        ],
        ROOT / 'features_shap_weighted.pdf',
        top_n=15,
    )


if __name__ == '__main__':
    main()
