import json
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
SPLIT_LABELS = {
    'split_01_seed_101': 'Split 1',
    'split_02_seed_202': 'Split 2',
    'split_03_seed_303': 'Split 3',
    'split_04_seed_404': 'Split 4',
    'split_05_seed_505': 'Split 5',
    'mean': 'Mean $\pm$ SD',
}
MODELS = ['xgb', 'txl']
MODEL_LABELS = {'xgb': 'XGB', 'txl': 'TXL Fusion'}
MODEL_COLORS = {'xgb': '#4C78A8', 'txl': '#E45756'}
MEAN_METRICS = [
    ('accuracy', 'Accuracy'),
    ('macro_f1', 'Macro-F1'),
    ('topological.f1-score', 'TI F1'),
    ('topological.recall', 'TI Recall'),
    ('topological.precision', 'TI Precision'),
    ('semimetal.f1-score', 'TSM F1'),
    ('trivial.f1-score', 'Trivial F1'),
]
SPLITWISE_METRICS = [
    ('accuracy', 'Accuracy'),
    ('macro_f1', 'Macro-F1'),
    ('topological.precision', 'TI Prec.'),
    ('topological.recall', 'TI Recall'),
    ('topological.f1-score', 'TI F1'),
]
MEAN_OUTPUT_PDF = BASE / 'repeated_split_mean_bars.pdf'
MEAN_OUTPUT_PNG = BASE / 'repeated_split_mean_bars.png'
MEAN_PAPER_PDF = PAPER / 'repeated_split_mean_bars.pdf'
MEAN_PAPER_PNG = PAPER / 'repeated_split_mean_bars.png'
SPLIT_OUTPUT_PDF = BASE / 'repeated_split_splitwise_bars.pdf'
SPLIT_OUTPUT_PNG = BASE / 'repeated_split_splitwise_bars.png'
SPLIT_PAPER_PDF = PAPER / 'repeated_split_splitwise_bars.pdf'
SPLIT_PAPER_PNG = PAPER / 'repeated_split_splitwise_bars.png'


def metric_value(payload, key):
    if '.' not in key:
        return float(payload[key])
    first, second = key.split('.', 1)
    return float(payload[first][second])


def load_metrics(split_name, eval_name):
    data = {model: {} for model in MODELS}
    for model in MODELS:
        model_dir = 'txl_model_pca50' if model == 'txl' else model
        path = BASE / model_dir / split_name / 'output' / f'{eval_name}_metrics.json'
        with path.open() as f:
            payload = json.load(f)
        data[model] = payload
    return data


def aggregate(eval_name):
    summary = {model: {} for model in MODELS}
    for metric_key, _label in MEAN_METRICS:
        for model in MODELS:
            vals = []
            for split in SPLITS:
                payload = load_metrics(split, eval_name)[model]
                vals.append(metric_value(payload, metric_key))
            summary[model][metric_key] = {
                'mean': sum(vals) / len(vals),
                'std': stats.stdev(vals),
            }
    return summary


def make_mean_figure():
    summaries = {
        'validation': aggregate('validation'),
        'heldout_test': aggregate('heldout_test'),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 9.6), sharey=True)
    width = 0.34
    x = np.arange(len(MEAN_METRICS))
    panel_labels = ['A', 'B']
    titles = {'validation': 'Validation mean $\pm$ SD across 5 random splits', 'heldout_test': 'Held-out test mean $\pm$ SD across 5 random splits'}

    for ax, eval_name, panel_label in zip(axes, ['validation', 'heldout_test'], panel_labels):
        for j, model in enumerate(MODELS):
            means = [summaries[eval_name][model][key]['mean'] for key, _ in MEAN_METRICS]
            stds = [summaries[eval_name][model][key]['std'] for key, _ in MEAN_METRICS]
            xpos = x + (j - 0.5) * width
            bars = ax.bar(
                xpos,
                means,
                width=width,
                color=MODEL_COLORS[model],
                edgecolor='white',
                linewidth=0.7,
                yerr=stds,
                capsize=3,
                error_kw={'elinewidth': 1.2, 'ecolor': '#333333'},
                zorder=3,
                label=MODEL_LABELS[model],
            )
            for bar, mean, std in zip(bars, means, stds):
                label_y = min(mean + std + 0.012, 1.015)
                ax.text(bar.get_x() + bar.get_width() / 2, label_y, f'{mean:.3f}$\pm${std:.3f}', ha='center', va='bottom', fontsize=7.2, color='#333333', rotation=90)
        ax.set_title(titles[eval_name], fontsize=13.2, fontweight='bold', pad=10)
        ax.text(0.0, 1.04, panel_label, transform=ax.transAxes, fontsize=15.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label for _key, label in MEAN_METRICS], rotation=28, ha='right', fontsize=10)
        ax.set_ylim(0.48, 1.03)
        ax.set_ylabel('Score', fontsize=11.8, fontweight='bold')
        ax.grid(axis='y', linestyle='--', linewidth=0.55, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0.03, 0.03, 0.995, 0.955], h_pad=2.0)
    return fig


def splitwise_metric_values(split_name):
    out = {model: {} for model in MODELS}
    for model in MODELS:
        payload = load_metrics(split_name, 'heldout_test')[model]
        for key, _label in SPLITWISE_METRICS:
            out[model][key] = metric_value(payload, key)
    return out


def splitwise_mean_values():
    out = {model: {} for model in MODELS}
    for model in MODELS:
        for key, _label in SPLITWISE_METRICS:
            vals = []
            for split in SPLITS:
                payload = load_metrics(split, 'heldout_test')[model]
                vals.append(metric_value(payload, key))
            out[model][key] = {'mean': sum(vals) / len(vals), 'std': stats.stdev(vals)}
    return out


def make_splitwise_figure():
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 13.4), sharey=True)
    axes = axes.flatten()
    panel_keys = SPLITS + ['mean']
    width = 0.34
    x = np.arange(len(SPLITWISE_METRICS))
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    mean_values = splitwise_mean_values()

    for ax, panel_key, panel_label in zip(axes, panel_keys, panel_labels):
        if panel_key == 'mean':
            payload = mean_values
            use_error = True
        else:
            payload = splitwise_metric_values(panel_key)
            use_error = False
        for j, model in enumerate(MODELS):
            xpos = x + (j - 0.5) * width
            if use_error:
                means = [payload[model][key]['mean'] for key, _ in SPLITWISE_METRICS]
                stds = [payload[model][key]['std'] for key, _ in SPLITWISE_METRICS]
                bars = ax.bar(
                    xpos,
                    means,
                    width=width,
                    color=MODEL_COLORS[model],
                    edgecolor='white',
                    linewidth=0.7,
                    yerr=stds,
                    capsize=3,
                    error_kw={'elinewidth': 1.2, 'ecolor': '#333333'},
                    zorder=3,
                    label=MODEL_LABELS[model],
                )
                value_list = means
            else:
                vals = [payload[model][key] for key, _ in SPLITWISE_METRICS]
                bars = ax.bar(
                    xpos,
                    vals,
                    width=width,
                    color=MODEL_COLORS[model],
                    edgecolor='white',
                    linewidth=0.7,
                    zorder=3,
                    label=MODEL_LABELS[model],
                )
                value_list = vals
            for bar, val in zip(bars, value_list):
                ax.text(bar.get_x() + bar.get_width() / 2, min(val + 0.012, 0.972), f'{val:.3f}', ha='center', va='bottom', fontsize=7.8, color='#333333', rotation=90)

        ax.set_title(SPLIT_LABELS[panel_key], fontsize=12.4, fontweight='bold', pad=9)
        ax.text(0.0, 1.04, panel_label, transform=ax.transAxes, fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label for _key, label in SPLITWISE_METRICS], rotation=22, ha='right', fontsize=9.5)
        ax.set_ylim(0.48, 0.98)
        ax.grid(axis='y', linestyle='--', linewidth=0.55, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if panel_key != 'mean':
            delta_ti_f1 = payload['txl']['topological.f1-score'] - payload['xgb']['topological.f1-score']
            ax.text(
                0.03,
                0.93,
                f'TI F1 gain: {delta_ti_f1 * 100:+.1f}%',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8.8,
                color='#333333',
                fontweight='bold',
            )
        else:
            delta_ti_f1 = payload['txl']['topological.f1-score']['mean'] - payload['xgb']['topological.f1-score']['mean']
            ax.text(
                0.03,
                0.93,
                f'Mean TI F1 gain: {delta_ti_f1 * 100:+.1f}%',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8.8,
                color='#333333',
                fontweight='bold',
            )

    axes[0].set_ylabel('Score', fontsize=11.5, fontweight='bold')
    axes[2].set_ylabel('Score', fontsize=11.5, fontweight='bold')
    axes[4].set_ylabel('Score', fontsize=11.5, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0.03, 0.03, 0.995, 0.955], h_pad=2.1, w_pad=1.1)
    return fig


def save_figure(fig, paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    mean_fig = make_mean_figure()
    save_figure(mean_fig, [MEAN_OUTPUT_PDF, MEAN_OUTPUT_PNG, MEAN_PAPER_PDF, MEAN_PAPER_PNG])
    split_fig = make_splitwise_figure()
    save_figure(split_fig, [SPLIT_OUTPUT_PDF, SPLIT_OUTPUT_PNG, SPLIT_PAPER_PDF, SPLIT_PAPER_PNG])
    print(f'Wrote {MEAN_OUTPUT_PDF}')
    print(f'Wrote {MEAN_PAPER_PDF}')
    print(f'Wrote {SPLIT_OUTPUT_PDF}')
    print(f'Wrote {SPLIT_PAPER_PDF}')


if __name__ == '__main__':
    main()
