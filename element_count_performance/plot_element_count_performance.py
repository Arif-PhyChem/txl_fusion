import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path('/path/to/3_classes_classification')
ELEMENT_DIR = ROOT / 'weighted_version' / 'element_count_performance'
PAPER_DIR = ROOT / 'paper_review'

REPORT_PATHS = {
    'XGB': ELEMENT_DIR / 'xgb_element_count_report.json',
    'TXL Fusion': ELEMENT_DIR / 'txl_element_count_report.json',
}

OUTPUT_PDF = ELEMENT_DIR / 'element_count_performance_bars.pdf'
OUTPUT_PNG = ELEMENT_DIR / 'element_count_performance_bars.png'
PAPER_OUTPUT_PDF = PAPER_DIR / 'element_count_performance_bars.pdf'
PAPER_OUTPUT_PNG = PAPER_DIR / 'element_count_performance_bars.png'

CLASS_ORDER = ['trivial', 'semimetal', 'topological']
CLASS_LABELS = {
    'trivial': 'Trivial',
    'semimetal': 'TSM',
    'topological': 'TI',
}
METRICS = [
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1-score', 'F1'),
]
MODEL_COLORS = {
    'XGB': '#4C78A8',
    'TXL Fusion': '#E45756',
}
PANEL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']


def load_reports():
    data = {}
    for model_name, path in REPORT_PATHS.items():
        with path.open() as f:
            payload = json.load(f)
        data[model_name] = payload['reports']
    return data


def annotate_pair_values(ax, x, y_xgb, y_txl, y_offset):
    label = f'{y_xgb:.2f}/{y_txl:.2f}'
    y = min(max(y_xgb, y_txl) + y_offset, 1.045)
    ax.text(
        x,
        y,
        label,
        ha='center',
        va='bottom',
        rotation=32,
        rotation_mode='anchor',
        fontsize=7.0,
        color='#333333',
        fontweight='bold',
    )


def make_plot(reports):
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 15.2), sharey=True)
    axes = axes.flatten()

    metric_spacing = 0.88
    class_gap = 0.70
    bar_width = 0.28
    group_positions = []
    x = 0.0
    for _cls in CLASS_ORDER:
        class_positions = []
        for _metric in METRICS:
            class_positions.append(x)
            x += metric_spacing
        group_positions.append(class_positions)
        x += class_gap

    all_positions = [pos for class_positions in group_positions for pos in class_positions]
    xtick_labels = [metric_label for _cls in CLASS_ORDER for _, metric_label in METRICS]

    for idx, n_elements in enumerate(range(1, 7)):
        ax = axes[idx]
        report_xgb = reports['XGB'][str(n_elements)]
        total_n = report_xgb['n_samples']
        support_lookup = {
            cls: int(round(report_xgb['classification_report'][cls]['support']))
            for cls in CLASS_ORDER
        }

        # Light class blocks make the repeated Precision/Recall/F1 groups easier to scan.
        for class_idx, cls in enumerate(CLASS_ORDER):
            positions = group_positions[class_idx]
            left = positions[0] - 0.55
            right = positions[-1] + 0.55
            if class_idx % 2 == 0:
                ax.axvspan(left, right, color='#F4F6F8', zorder=0)
            center = np.mean(positions)
            ax.text(
                center,
                -0.18,
                f"{CLASS_LABELS[cls]}\n(n={support_lookup[cls]})",
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=10.6,
                fontweight='bold',
            )

        for class_idx, cls in enumerate(CLASS_ORDER):
            for metric_idx, (metric_key, _metric_label) in enumerate(METRICS):
                xpos = group_positions[class_idx][metric_idx]
                xgb_val = reports['XGB'][str(n_elements)]['classification_report'][cls][metric_key]
                txl_val = reports['TXL Fusion'][str(n_elements)]['classification_report'][cls][metric_key]
                ax.bar(
                    xpos - bar_width / 2,
                    xgb_val,
                    width=bar_width,
                    color=MODEL_COLORS['XGB'],
                    edgecolor='white',
                    linewidth=0.7,
                    zorder=3,
                )
                ax.bar(
                    xpos + bar_width / 2,
                    txl_val,
                    width=bar_width,
                    color=MODEL_COLORS['TXL Fusion'],
                    edgecolor='white',
                    linewidth=0.7,
                    zorder=3,
                )
                y_offset = 0.036 + 0.014 * ((class_idx + metric_idx) % 2)
                annotate_pair_values(ax, xpos, xgb_val, txl_val, y_offset)

        for separator_idx in range(1, len(CLASS_ORDER)):
            prev_last = group_positions[separator_idx - 1][-1]
            next_first = group_positions[separator_idx][0]
            ax.axvline((prev_last + next_first) / 2, color='#B8B8B8', linewidth=0.7, linestyle=':', zorder=1)

        ax.set_xticks(all_positions)
        ax.set_xticklabels(xtick_labels, rotation=35, ha='right', fontsize=10.4)
        ax.tick_params(axis='x', pad=1)
        ax.tick_params(axis='y', labelsize=10.2)
        ax.set_ylim(0.0, 1.08)
        ax.set_xlim(min(all_positions) - 0.65, max(all_positions) + 0.65)
        ax.grid(axis='y', linestyle='-', linewidth=0.45, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')
        ax.text(0.0, 1.06, PANEL_LABELS[idx], transform=ax.transAxes, fontsize=16.5, fontweight='bold', va='bottom')
        ax.set_title(f'{n_elements}-element compounds (N={total_n:,})', fontsize=13.8, pad=14, fontweight='bold')

        if idx % 2 == 0:
            ax.set_ylabel('Score', fontsize=12.6, fontweight='bold')

    handles = [
        Patch(facecolor=MODEL_COLORS['XGB'], edgecolor='none', label='Numerical XGB'),
        Patch(facecolor=MODEL_COLORS['TXL Fusion'], edgecolor='none', label='TXL Fusion'),
        Patch(facecolor='white', edgecolor='none', label='Pair labels above bars show XGB/TXL values'),
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=11.8,
        handlelength=1.6,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.955], h_pad=2.8, w_pad=1.6)
    return fig


def main():
    reports = load_reports()
    fig = make_plot(reports)
    for path in [OUTPUT_PDF, OUTPUT_PNG, PAPER_OUTPUT_PDF, PAPER_OUTPUT_PNG]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUTPUT_PDF}')
    print(f'Wrote {PAPER_OUTPUT_PDF}')


if __name__ == '__main__':
    main()
