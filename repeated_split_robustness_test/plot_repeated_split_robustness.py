import json
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path('/path/to/3_classes_classification')
BASE = ROOT / 'weighted_version' / 'repeated_split_robustness_test'
PAPER = ROOT / 'paper_review'
SPLITS = [
    'split_01_seed_101',
    'split_02_seed_202',
    'split_03_seed_303',
    'split_04_seed_404',
    'split_05_seed_505',
]
MODELS = ['xgb', 'txl']
MODEL_LABELS = {'xgb': 'XGB', 'txl': 'TXL Fusion'}
MODEL_COLORS = {'xgb': '#4C78A8', 'txl': '#E45756'}
METRICS = [
    ('accuracy', 'Accuracy'),
    ('macro_f1', 'Macro-F1'),
    ('topological.f1-score', 'TI F1'),
    ('topological.recall', 'TI Recall'),
    ('topological.precision', 'TI Precision'),
    ('semimetal.f1-score', 'TSM F1'),
    ('trivial.f1-score', 'Trivial F1'),
]
OUTPUT_PDF = BASE / 'repeated_split_robustness.pdf'
OUTPUT_PNG = BASE / 'repeated_split_robustness.png'
PAPER_PDF = PAPER / 'repeated_split_robustness.pdf'
PAPER_PNG = PAPER / 'repeated_split_robustness.png'


def metric_value(payload, key):
    if '.' not in key:
        return float(payload[key])
    first, second = key.split('.', 1)
    return float(payload[first][second])


def aggregate(split_name):
    raw = {model: {k: [] for k, _ in METRICS} for model in MODELS}
    for model in MODELS:
        for split in SPLITS:
            model_dir = 'txl_model_pca50' if model == 'txl' else model
            path = BASE / model_dir / split / 'output' / f'{split_name}_metrics.json'
            with path.open() as f:
                payload = json.load(f)
            for key, _label in METRICS:
                raw[model][key].append(metric_value(payload, key))

    summary = {model: {} for model in MODELS}
    deltas = {}
    for key, _label in METRICS:
        xgb_vals = raw['xgb'][key]
        txl_vals = raw['txl'][key]
        deltas[key] = [t - x for x, t in zip(xgb_vals, txl_vals)]
        for model in MODELS:
            vals = raw[model][key]
            summary[model][key] = {
                'mean': sum(vals) / len(vals),
                'std': stats.stdev(vals),
            }
    return summary, deltas


def draw_panel(ax, title, summary, deltas, panel_label):
    y = np.arange(len(METRICS))
    xmin, xmax = 0.50, 0.94

    for idx, (key, label) in enumerate(METRICS):
        x_xgb = summary['xgb'][key]['mean']
        x_txl = summary['txl'][key]['mean']
        std_xgb = summary['xgb'][key]['std']
        std_txl = summary['txl'][key]['std']
        yy = y[idx]

        ax.plot([x_xgb, x_txl], [yy, yy], color='#A9A9A9', linewidth=2.0, zorder=1)
        ax.errorbar(
            x_xgb,
            yy,
            xerr=std_xgb,
            fmt='o',
            color=MODEL_COLORS['xgb'],
            ecolor=MODEL_COLORS['xgb'],
            elinewidth=1.6,
            capsize=3,
            markersize=7.5,
            markeredgecolor='white',
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.errorbar(
            x_txl,
            yy,
            xerr=std_txl,
            fmt='o',
            color=MODEL_COLORS['txl'],
            ecolor=MODEL_COLORS['txl'],
            elinewidth=1.6,
            capsize=3,
            markersize=7.5,
            markeredgecolor='white',
            markeredgewidth=0.8,
            zorder=4,
        )

        mean_delta = sum(deltas[key]) / len(deltas[key])
        delta_std = stats.stdev(deltas[key])
        delta_text = f'+{mean_delta:.3f} +/- {delta_std:.3f}' if mean_delta >= 0 else f'{mean_delta:.3f} +/- {delta_std:.3f}'
        text_x = min(max(x_xgb, x_txl) + 0.018, xmax - 0.006)
        ax.text(
            text_x,
            yy,
            delta_text,
            va='center',
            ha='left',
            fontsize=9.1,
            color='#333333',
            fontweight='bold',
        )

    ax.set_yticks(y)
    ax.set_yticklabels([label for _key, label in METRICS], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.0, 1.055, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=13.5, fontweight='bold', pad=11)
    ax.tick_params(axis='x', labelsize=10.5)


def main():
    validation, validation_deltas = aggregate('validation')
    heldout, heldout_deltas = aggregate('heldout_test')

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 10.4), sharex=True)
    draw_panel(axes[0], 'Validation splits', validation, validation_deltas, 'A')
    draw_panel(axes[1], 'Held-out test', heldout, heldout_deltas, 'B')
    axes[1].set_xlabel('Score', fontsize=12, fontweight='bold')

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=MODEL_COLORS['xgb'], markeredgecolor='white', markersize=8.5, label='XGB mean +/- SD'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=MODEL_COLORS['txl'], markeredgecolor='white', markersize=8.5, label='TXL Fusion mean +/- SD'),
        Line2D([0], [0], color='#A9A9A9', linewidth=2.0, label='Paired metric shift'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(0.52, 0.995))
    fig.text(0.80, 0.962, 'labels: TXL-XGB mean +/- SD', ha='center', va='center', fontsize=9.4, color='#333333')
    fig.tight_layout(rect=[0.05, 0.04, 0.995, 0.945], h_pad=2.1)

    for path in [OUTPUT_PDF, OUTPUT_PNG, PAPER_PDF, PAPER_PNG]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUTPUT_PDF}')
    print(f'Wrote {PAPER_PDF}')


if __name__ == '__main__':
    main()
