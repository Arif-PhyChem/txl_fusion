
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/path/to/3_classes_classification')
NUM_BINS = 10
HERE = Path(__file__).resolve().parent
MODEL_CONFIG = {
    'txl': {
        'predictions_json': ROOT / 'weighted_version' / 'inference' / 'txl_model_pca50' / 'heldout_test_predictions.json',
        'output_pdf': HERE / 'conf_bins_txl.pdf',
        'output_json': HERE / 'calibration_metrics_txl.json',
        'paper_pdf': ROOT / 'paper_review' / 'conf_bins_txl.pdf',
        'paper_json': ROOT / 'paper_review' / 'calibration_metrics_txl.json',
        'title': 'Top-label reliability diagram for TXL Fusion on the held-out test set',
        'print_name': 'TXL Fusion',
    },
    'xgb': {
        'predictions_json': ROOT / 'weighted_version' / 'inference' / 'xgb' / 'heldout_test_predictions.json',
        'output_pdf': HERE / 'conf_bins_xgb.pdf',
        'output_json': HERE / 'calibration_metrics_xgb.json',
        'paper_pdf': ROOT / 'paper_review' / 'conf_bins_xgb.pdf',
        'paper_json': ROOT / 'paper_review' / 'calibration_metrics_xgb.json',
        'title': 'Top-label reliability diagram for standalone XGB on the held-out test set',
        'print_name': 'standalone XGB',
    },
    'txl_pca50': {
        'predictions_json': ROOT / 'weighted_version' / 'txl_model_pca50' / 'output' / 'heldout_test_predictions_verbose.json',
        'output_pdf': HERE / 'txl_model_pca50' / 'conf_bins_txl_model_pca50.pdf',
        'output_json': HERE / 'txl_model_pca50' / 'calibration_metrics_txl_model_pca50.json',
        'paper_pdf': ROOT / 'paper_review' / 'conf_bins_txl_model_pca50.pdf',
        'paper_json': ROOT / 'paper_review' / 'calibration_metrics_txl_model_pca50.json',
        'title': 'Top-label reliability diagram for TXL Fusion on the held-out test set',
        'print_name': 'TXL Fusion (PCA50)',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Plot top-label reliability diagrams from held-out prediction JSON files.')
    parser.add_argument('--model', choices=sorted(MODEL_CONFIG), default='txl')
    return parser.parse_args()


def load_predictions(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def compute_top_label_calibration(predictions, num_bins):
    confidences = np.array([pred['probabilities'][pred['predicted_label']] for pred in predictions], dtype=float)
    correctness = np.array([int(pred['predicted_label'] == pred['true_label']) for pred in predictions], dtype=float)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(confidences, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    records = []
    ece = 0.0
    total = len(predictions)

    for idx in range(num_bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        if count == 0:
            avg_conf = 0.0
            accuracy = 0.0
        else:
            avg_conf = float(confidences[mask].mean())
            accuracy = float(correctness[mask].mean())
            ece += (count / total) * abs(accuracy - avg_conf)

        records.append({
            'bin_index': idx,
            'bin_left': float(bins[idx]),
            'bin_right': float(bins[idx + 1]),
            'bin_center': float((bins[idx] + bins[idx + 1]) / 2.0),
            'count': count,
            'mean_confidence': avg_conf,
            'empirical_accuracy': accuracy,
        })

    return {
        'n_predictions': total,
        'num_bins': num_bins,
        'ece': float(ece),
        'overall_accuracy': float(correctness.mean()),
        'overall_mean_confidence': float(confidences.mean()),
        'bin_stats': records,
    }


def plot_reliability(metrics, output_pdf: Path, title: str):
    records = metrics['bin_stats']
    centers = np.array([r['bin_center'] for r in records], dtype=float)
    counts = np.array([r['count'] for r in records], dtype=int)
    accuracies = np.array([r['empirical_accuracy'] for r in records], dtype=float)
    mean_conf = np.array([r['mean_confidence'] for r in records], dtype=float)

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={'height_ratios': [3.0, 1.25]}
    )

    width = 0.085
    ax_top.plot([0, 1], [0, 1], linestyle='--', linewidth=1.2, color='gray', label='Perfect calibration')
    bars = ax_top.bar(centers, accuracies, width=width, color='#9ecae1', edgecolor='#1f77b4', alpha=0.9, label='Empirical accuracy')
    ax_top.scatter(centers, mean_conf, marker='o', s=42, facecolors='white', edgecolors='#d62728', linewidths=1.5, zorder=3, label='Mean confidence')

    for center, count, acc, conf in zip(centers, counts, accuracies, mean_conf):
        if count <= 0:
            continue
        acc_y = min(acc + 0.025, 1.015)
        ax_top.text(
            center,
            acc_y,
            f'{acc:.3f}',
            ha='center',
            va='bottom',
            fontsize=8.5,
            color='#08519c',
            fontweight='bold',
            zorder=5,
        )

        if acc >= conf or abs(acc - conf) < 0.055:
            conf_y = max(conf - 0.055, 0.035)
            conf_va = 'top'
        else:
            conf_y = min(conf + 0.04, 1.015)
            conf_va = 'bottom'
        ax_top.text(
            center,
            conf_y,
            f'{conf:.3f}',
            ha='center',
            va=conf_va,
            fontsize=8.2,
            color='#b2182b',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.72),
            zorder=6,
        )

    ax_top.text(-0.1, 0.9, 'A', transform=ax_top.transAxes, fontsize=12, fontweight='bold', ha='left', va='bottom')
    ax_bottom.text(-0.1, 0.9, 'B', transform=ax_bottom.transAxes, fontsize=12, fontweight='bold', ha='left', va='bottom')

    ax_top.text(
        0.03,
        0.97,
        f"ECE = {metrics['ece']:.3f}",
        transform=ax_top.transAxes,
        ha='left',
        va='top',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bbbbbb')
    )
    ax_top.set_ylabel('Empirical accuracy')
    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.set_title(title)
    ax_top.grid(axis='y', linestyle=':', alpha=0.55)
    ax_top.legend(loc='lower right', frameon=True)

    count_bars = ax_bottom.bar(centers, counts, width=width, color='#d9d9d9', edgecolor='#7f7f7f')
    max_count = max(counts) if len(counts) else 0
    for bar, count in zip(count_bars, counts):
        if count > 0:
            y = bar.get_height() + max(12, 0.015 * max_count)
            ax_bottom.text(bar.get_x() + bar.get_width() / 2, y, str(count), ha='center', va='bottom', fontsize=9, color='#444444')

    ax_bottom.set_xlabel('Predicted confidence (binned)')
    ax_bottom.set_ylabel('Count')
    ax_bottom.set_xlim(0.0, 1.0)
    ax_bottom.set_ylim(0, max_count * 1.18 if max_count else 1)
    ticks = np.linspace(0.05, 0.95, 10)
    ax_bottom.set_xticks(ticks)
    ax_bottom.set_xticklabels([f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)], rotation=35, ha='right')
    ax_bottom.grid(axis='y', linestyle=':', alpha=0.55)

    fig.tight_layout()
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    config = MODEL_CONFIG[args.model]

    predictions = load_predictions(config['predictions_json'])
    metrics = compute_top_label_calibration(predictions, NUM_BINS)
    config['output_json'].write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    plot_reliability(metrics, config['output_pdf'], config['title'])
    config['paper_pdf'].write_bytes(config['output_pdf'].read_bytes())
    config['paper_json'].write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f"Saved reliability diagram to {config['output_pdf']}")
    print(f"Saved calibration metrics to {config['output_json']}")
    print(f"Model = {config['print_name']}")
    print(f"ECE = {metrics['ece']:.6f}")
    print(f"Overall accuracy = {metrics['overall_accuracy']:.6f}")
    print(f"Overall mean confidence = {metrics['overall_mean_confidence']:.6f}")


if __name__ == '__main__':
    main()
