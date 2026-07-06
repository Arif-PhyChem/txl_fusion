import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
PAPER_REVIEW = Path('/path/to/3_classes_classification/paper_review')
VARIANCE_CSV = BASE.parent / 'pca_explained_variance_test' / 'pca_explained_variance_train_fit.csv'
METRICS_ROOT = BASE / 'models'
HIGHLIGHT_COMPONENTS = 50


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))



def read_f1_rows_from_metrics():
    rows = []
    for model_dir in sorted(METRICS_ROOT.glob('pca_*')):
        try:
            n_components = int(model_dir.name.split('_', 1)[1])
        except (IndexError, ValueError):
            continue
        metrics_path = model_dir / 'output' / 'validation_metrics.json'
        if not metrics_path.exists():
            continue
        with open(metrics_path, encoding='utf-8') as f:
            metrics = json.load(f)
        rows.append({
            'n_components': n_components,
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics.get('macro_f1', metrics['macro avg']['f1-score'])),
            'weighted_f1': float(metrics.get('weighted_f1', metrics['weighted avg']['f1-score'])),
        })
    return sorted(rows, key=lambda row: row['n_components'])

def setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.5,
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.5,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'axes.linewidth': 0.9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def add_bar_labels(ax, bars, fmt, y_offset, rotation=90, fontsize=7.0):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            fmt.format(height),
            ha='center',
            va='bottom',
            rotation=rotation,
            fontsize=fontsize,
            clip_on=True,
        )


def plot_combined():
    variance_rows = read_csv(VARIANCE_CSV)
    f1_rows = read_f1_rows_from_metrics()

    dims = np.array([int(r['n_components']) for r in variance_rows])
    variance = np.array([float(r['train_cumulative_explained_variance']) for r in variance_rows]) * 100

    f1_dims = np.array([int(r['n_components']) for r in f1_rows])
    macro = np.array([float(r['macro_f1']) for r in f1_rows])
    weighted = np.array([float(r['weighted_f1']) for r in f1_rows])
    accuracy = np.array([float(r['accuracy']) for r in f1_rows])

    x_var = np.arange(len(dims))
    x_perf = np.arange(len(f1_dims))
    fig, (ax_var, ax_perf) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        gridspec_kw={'height_ratios': [1.0, 1.15], 'hspace': 0.42},
    )

    colors = ['#d8ecf7' if dim != HIGHLIGHT_COMPONENTS else '#d62728' for dim in dims]
    edges = ['#1f77b4' if dim != HIGHLIGHT_COMPONENTS else '#9b1c1c' for dim in dims]
    bars = ax_var.bar(x_var, variance, color=colors, edgecolor=edges, linewidth=1.0, width=0.68)
    ax_var.axhline(95, color='#555555', linestyle=':', linewidth=1.0)
    ax_var.text(0.01, 1.04, 'A', transform=ax_var.transAxes, fontsize=11, fontweight='bold')
    ax_var.set_title('Variance retained by PCA-compressed SciBERT embeddings')
    ax_var.set_ylabel('Explained variance (%)')
    ax_var.set_xticks(x_var)
    ax_var.set_xticklabels([str(d) for d in dims])
    ax_var.set_xlabel('Number of PCA components')
    ax_var.set_ylim(35, 105.5)
    ax_var.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.65)
    add_bar_labels(ax_var, bars, '{:.1f}', 0.65, rotation=0, fontsize=7.5)
    if HIGHLIGHT_COMPONENTS in dims:
        chosen_var_idx = np.where(dims == HIGHLIGHT_COMPONENTS)[0][0]
        text_idx = max(0, chosen_var_idx - 2)
        ax_var.annotate(
            'selected setting',
            xy=(chosen_var_idx, variance[chosen_var_idx]),
            xytext=(text_idx, 100.5),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#d62728'),
            ha='center',
            va='bottom',
            fontsize=8.0,
            color='#9b1c1c',
        )
    for tick, dim in zip(ax_var.get_xticklabels(), dims):
        if dim == HIGHLIGHT_COMPONENTS:
            tick.set_color('#9b1c1c')
            tick.set_fontweight('bold')

    width = 0.25
    bars_macro = ax_perf.bar(x_perf - width, macro, width, color='#9ecae1', edgecolor='#1f77b4', label='Macro-F1')
    bars_weighted = ax_perf.bar(x_perf, weighted, width, color='#a1d99b', edgecolor='#2ca02c', label='Weighted-F1')
    bars_acc = ax_perf.bar(x_perf + width, accuracy, width, color='#c7b9e8', edgecolor='#9467bd', label='Accuracy')
    ax_perf.text(0.01, 1.04, 'B', transform=ax_perf.transAxes, fontsize=11, fontweight='bold')
    ax_perf.set_title('TXL Fusion sensitivity to SciBERT embedding dimension')
    ax_perf.set_ylabel('Validation score')
    ax_perf.set_xticks(x_perf)
    ax_perf.set_xticklabels([str(d) for d in f1_dims])
    ax_perf.set_xlabel('Number of SciBERT PCA components')
    ymin = max(0.0, min(macro.min(), weighted.min(), accuracy.min()) - 0.04)
    ymax = min(1.0, max(macro.max(), weighted.max(), accuracy.max()) + 0.04)
    ax_perf.set_ylim(ymin, ymax)
    ax_perf.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.65)
    ax_perf.legend(frameon=False, ncol=3, loc='upper left')
    add_bar_labels(ax_perf, bars_macro, '{:.3f}', 0.0025, rotation=90, fontsize=6.6)
    add_bar_labels(ax_perf, bars_weighted, '{:.3f}', 0.0025, rotation=90, fontsize=6.6)
    add_bar_labels(ax_perf, bars_acc, '{:.3f}', 0.0025, rotation=90, fontsize=6.6)
    for tick, dim in zip(ax_perf.get_xticklabels(), f1_dims):
        if dim == HIGHLIGHT_COMPONENTS:
            tick.set_color('#9b1c1c')
            tick.set_fontweight('bold')
    fig.tight_layout()
    pdf = PAPER_REVIEW / 'pca_variance_f1_sensitivity_subfigures.pdf'
    png = PAPER_REVIEW / 'pca_variance_f1_sensitivity_subfigures.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(pdf)
    print(png)


def main():
    setup_style()
    plot_combined()


if __name__ == '__main__':
    main()
