import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

ROOT = Path('/path/to/3_classes_classification/weighted_version')
HELPER = ROOT / 'plot_features_importance_pca50.py'
XGB_OUT = ROOT / 'xgb' / 'xgb_shap_beeswarm.pdf'
TXL_OUT = ROOT / 'txl_model_pca50' / 'txl_shap_beeswarm_pca50.pdf'
XGB_PNG = ROOT / 'xgb' / 'xgb_shap_beeswarm.png'
TXL_PNG = ROOT / 'txl_model_pca50' / 'txl_shap_beeswarm_pca50.png'
COMBINED_OUT = ROOT / 'features_shap_weighted.pdf'
SAMPLE_CAP = 1200
XGB_MAX_DISPLAY = 10
TXL_MAX_DISPLAY = 10


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


helpers = load_module(HELPER, 'plot_features_importance_pca50_helpers')


def pretty_names(columns):
    return [helpers.pretty_feature_name(col) for col in columns]


def build_explanation(values, frame):
    return shap.Explanation(
        values=values,
        data=frame.to_numpy(),
        feature_names=pretty_names(frame.columns),
    )



def trim_explanation(explanation, max_display):
    mean_abs = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    trimmed_values = explanation.values[:, order]
    trimmed_data = explanation.data[:, order] if explanation.data is not None else None
    trimmed_names = [explanation.feature_names[i] for i in order]
    return shap.Explanation(
        values=trimmed_values,
        data=trimmed_data,
        feature_names=trimmed_names,
    )


def save_beeswarm(explanation, title, pdf_path, png_path, max_display):
    explanation = trim_explanation(explanation, max_display)
    plt.figure(figsize=(7.2, 5.6))
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        plot_size=None,
        show=False,
    )
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.tick_params(axis='y', length=0)
    plt.tight_layout()
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=360, bbox_inches='tight')
    plt.close()
    print(f'Saved beeswarm to {pdf_path}')
    print(f'Saved beeswarm to {png_path}')


def plot_summary_axis(ax, title, shap_df, panel_label, top_n, palette):
    top = shap_df.head(top_n).copy()
    top['feature_pretty'] = [helpers.pretty_feature_name(v) for v in top['feature']]
    sns.barplot(x='mean_abs_shap', y='feature_pretty', data=top, palette=palette, ax=ax)
    ax.set_title(title, fontsize=21, pad=14)
    ax.set_xlabel('Mean |SHAP value|', fontsize=19)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=17)
    ax.tick_params(axis='x', labelsize=15)
    ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes, fontsize=23, fontweight='bold')


def save_combined_panel(xgb_expl, txl_expl, xgb_summary, txl_summary, output_path):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17.4, 13.2),
        gridspec_kw={'width_ratios': [1.05, 1.25], 'wspace': 0.003, 'hspace': 0.24},
    )

    shared_palette = 'magma'
    plot_summary_axis(axes[0, 0], 'XGB mean |SHAP|', xgb_summary, 'A', top_n=XGB_MAX_DISPLAY, palette=shared_palette)
    plot_summary_axis(axes[1, 0], 'TXL mean |SHAP|', txl_summary, 'C', top_n=TXL_MAX_DISPLAY, palette=shared_palette)

    for ax, label, title, expl in [
        (axes[0, 1], 'B', 'XGB SHAP beeswarm', xgb_expl),
        (axes[1, 1], 'D', 'TXL SHAP beeswarm', txl_expl),
    ]:
        trimmed = trim_explanation(expl, XGB_MAX_DISPLAY if label == 'B' else TXL_MAX_DISPLAY)
        shap.plots.beeswarm(
            trimmed,
            max_display=XGB_MAX_DISPLAY if label == 'B' else TXL_MAX_DISPLAY,
            plot_size=None,
            show=False,
            ax=ax,
            color_bar=False,
        )
        ax.tick_params(axis='y', labelsize=17)
        ax.set_yticklabels([])
        ax.set_ylabel('')
        ax.set_xlabel('SHAP value (impact on model output)', fontsize=19)
        ax.tick_params(axis='x', labelsize=15)
        ax.text(-0.03, 1.02, label, transform=ax.transAxes, fontsize=23, fontweight='bold')
        ax.set_title(title, fontsize=21, pad=14)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.97, bottom=0.04)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined SHAP summary + beeswarm panel to {output_path}')


def main():
    xgb_model = helpers.load_model(ROOT / 'xgb' / 'xgb_model.json')
    xgb_x = helpers.load_weighted_xgb_validation_frame()
    if len(xgb_x) > SAMPLE_CAP:
        xgb_x = xgb_x.iloc[:SAMPLE_CAP].copy()
    xgb_probs = xgb_model.predict_proba(xgb_x)
    xgb_pred = xgb_probs.argmax(axis=1)
    xgb_shap_raw = shap.TreeExplainer(xgb_model).shap_values(xgb_x)
    xgb_selected = xgb_shap_raw[np.arange(len(xgb_x)), :, xgb_pred]
    xgb_summary = helpers.compute_mean_abs_shap(xgb_model, xgb_x)
    xgb_expl = build_explanation(xgb_selected, xgb_x)
    save_beeswarm(xgb_expl, 'Standalone XGB SHAP beeswarm', XGB_OUT, XGB_PNG, XGB_MAX_DISPLAY)

    txl_model = helpers.load_model(ROOT / 'txl_model_pca50' / 'output' / 'hierarchical_stage1_model.json')
    txl_x = helpers.load_pca50_txl_validation_frame()
    if len(txl_x) > SAMPLE_CAP:
        txl_x = txl_x.iloc[:SAMPLE_CAP].copy()
    txl_shap = shap.TreeExplainer(txl_model).shap_values(txl_x)
    txl_summary = helpers.compute_mean_abs_shap(txl_model, txl_x)
    txl_expl = build_explanation(txl_shap, txl_x)
    save_beeswarm(txl_expl, 'TXL Fusion SHAP beeswarm', TXL_OUT, TXL_PNG, TXL_MAX_DISPLAY)

    save_combined_panel(xgb_expl, txl_expl, xgb_summary, txl_summary, COMBINED_OUT)


if __name__ == '__main__':
    main()
