import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

ROOT = Path('/path/to/3_classes_classification/weighted_version')
XGB_DIR = ROOT / 'xgb'
OUT_PDF = XGB_DIR / 'xgb_class_support_shap.pdf'
OUT_JSON = XGB_DIR / 'xgb_class_support_shap_mean.json'
TOP_POS = 10
CLASS_ORDER = [(2, 'Trivial'), (0, 'TSM'), (1, 'TI')]


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def pretty_feature_name(name):
    return name.replace('_', ' ')


def load_model(model_path):
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    model.get_booster().set_param({'device': 'cpu'})
    model.set_params(device='cpu')
    return model


def build_validation_frame():
    rows = load_json(XGB_DIR / 'xgb_data_weighted_shared_split.json')
    feature_names = load_json(XGB_DIR / 'xgb_feature_names.json')['xgb_feature_names']
    manifest = load_json(Path('/path/to/3_classes_classification/label_noise_analysis/shared_split_manifest.json'))
    val_idx = manifest['validation_indices']
    df = pd.DataFrame(rows)
    x_val = df.iloc[val_idx][feature_names].reset_index(drop=True)
    pred_rows = load_json(XGB_DIR / 'validation_predictions.json')
    true_labels = [row['true_label'] for row in pred_rows]
    return x_val, true_labels


def main():
    sns.set_style('whitegrid')
    x_val, true_labels = build_validation_frame()
    model = load_model(XGB_DIR / 'xgb_model.json')
    shap_values = shap.TreeExplainer(model).shap_values(x_val)

    payload = {}
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 12.8))
    cmap = colormaps['magma_r']

    label_to_name = {0: 'semimetal', 1: 'topological', 2: 'trivial'}

    for ax, (class_idx, class_name), panel_label in zip(axes, CLASS_ORDER, 'ABC'):
        mask = np.array([lbl == label_to_name[class_idx] for lbl in true_labels])
        mean_signed = shap_values[mask, :, class_idx].mean(axis=0)
        class_df = pd.DataFrame({
            'feature': x_val.columns,
            'mean_shap': mean_signed,
        })
        top_pos = class_df[class_df['mean_shap'] > 0].nlargest(TOP_POS, 'mean_shap')
        class_df = top_pos.sort_values('mean_shap').copy()
        class_df['feature_pretty'] = [pretty_feature_name(v) for v in class_df['feature']]

        payload[class_name.lower()] = [
            {'feature': r.feature, 'mean_shap': float(r.mean_shap)}
            for r in class_df.sort_values('mean_shap').itertuples()
        ]

        norm = colors.Normalize(vmin=class_df['mean_shap'].min(), vmax=class_df['mean_shap'].max())
        bar_colors = [cmap(norm(v)) for v in class_df['mean_shap']]
        ax.barh(
            class_df['feature_pretty'],
            class_df['mean_shap'],
            color=bar_colors,
            edgecolor='none',
        )
        ax.set_title(class_name, fontsize=17)
        ax.set_xlabel('Mean SHAP value', fontsize=14)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=13)
        ax.text(-0.16, 1.03, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold')

    fig.subplots_adjust(left=0.36, right=0.98, top=0.96, bottom=0.06, hspace=0.48)
    plt.savefig(OUT_PDF, format='pdf', bbox_inches='tight')
    plt.close(fig)

    with open(OUT_JSON, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'Saved class-support SHAP figure to {OUT_PDF}')
    print(f'Saved class-support SHAP data to {OUT_JSON}')


if __name__ == '__main__':
    main()
