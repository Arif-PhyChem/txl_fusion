import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Composition

ROOT = Path('/path/to/3_classes_classification')
CLEAN_TRAIN = ROOT / 'label_noise_analysis' / 'clean_training_data.json'
CLEAN_TEST = ROOT / 'label_noise_analysis' / 'clean_test_data.json'
OUT_DIR = ROOT / 'weighted_version' / 'element_count_distribution'
PAPER_REVIEW = ROOT / 'paper_review'

CLASS_ORDER = ['Trivial', 'TSM', 'TI']
COLORS = {
    'Trivial': '#1f77b4',
    'TSM': '#ff7f0e',
    'TI': '#2ca02c',
}
MAX_ELEMENTS = 6


def load_json(path):
    with path.open() as f:
        return json.load(f)


def normalize_label(label):
    value = str(label).strip().lower()
    if value in {'ti', 'topological'}:
        return 'TI'
    if value in {'sm', 'tsm', 'semimetal', 'metal'}:
        return 'TSM'
    if value in {'trivial', 'lcebr'}:
        return 'Trivial'
    raise ValueError(f'Unrecognized label: {label!r}')


def normalize_formula(formula):
    amounts = Composition(formula).get_el_amt_dict()
    normalized = defaultdict(float)
    for element, amount in amounts.items():
        normalized['H' if element == 'D' else element] += amount
    return Composition(dict(normalized))


def count_elements_by_class(records):
    counts = {name: Counter() for name in CLASS_ORDER}
    skipped = []

    for idx, entry in enumerate(records):
        try:
            label = normalize_label(entry['topologicalClassificationShortDescription'])
            comp = normalize_formula(entry['compoundName'])
            n_elements = len(comp.get_el_amt_dict())
            counts[label][n_elements] += 1
        except Exception as exc:
            skipped.append({'index': idx, 'entry': entry, 'error': str(exc)})

    return counts, skipped


def write_summary(counts, skipped):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {'total_records_used': 0, 'skipped_records': len(skipped), 'classes': {}}

    for label in CLASS_ORDER:
        total = sum(counts[label].values())
        summary['total_records_used'] += total
        summary['classes'][label] = {'total': total, 'element_counts': {}}
        for n in range(1, MAX_ELEMENTS + 1):
            count = counts[label].get(n, 0)
            percentage = 100.0 * count / total if total else 0.0
            summary['classes'][label]['element_counts'][str(n)] = {
                'count': count,
                'percentage': percentage,
            }
            rows.append({
                'class': label,
                'n_elements': n,
                'count': count,
                'class_total': total,
                'percentage': percentage,
            })

    (OUT_DIR / 'element_count_distribution_summary.json').write_text(
        json.dumps(summary, indent=2)
    )
    with (OUT_DIR / 'element_count_distribution_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'n_elements', 'count', 'class_total', 'percentage'])
        writer.writeheader()
        writer.writerows(rows)
    if skipped:
        (OUT_DIR / 'element_count_distribution_skipped.json').write_text(
            json.dumps(skipped, indent=2)
        )

    return summary


def plot_distribution(counts):
    bins = np.arange(1, MAX_ELEMENTS + 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(
        'Distribution of Materials by Number of Constituent Elements',
        fontsize=14,
        fontweight='bold',
        y=0.96,
    )

    for ax, label in zip(axes, CLASS_ORDER):
        total = sum(counts[label].values())
        percentages = [counts[label].get(i, 0) / total * 100 for i in bins]
        ax.bar(
            bins,
            percentages,
            width=0.7,
            color=COLORS[label],
            edgecolor='black',
            linewidth=0.6,
        )
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_ylim(0, max(percentages) * 1.3 if percentages else 1)
        ax.text(
            0.02,
            0.93,
            f'{label}\n$n = {total:,}$',
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'),
        )
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Number of Constituent Elements', fontsize=12)
    axes[-1].set_xticks(bins)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    weighted_pdf = OUT_DIR / 'element_count_distribution_all_classes.pdf'
    paper_pdf = PAPER_REVIEW / 'element_count_distribution_all_classes.pdf'
    fig.savefig(weighted_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(paper_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return weighted_pdf, paper_pdf


def main():
    records = load_json(CLEAN_TRAIN) + load_json(CLEAN_TEST)
    counts, skipped = count_elements_by_class(records)
    summary = write_summary(counts, skipped)
    weighted_pdf, paper_pdf = plot_distribution(counts)

    print(f'Loaded cleaned train+test records: {len(records):,}')
    print(f'Used records: {summary["total_records_used"]:,}; skipped: {summary["skipped_records"]:,}')
    for label in CLASS_ORDER:
        total = summary['classes'][label]['total']
        dist = summary['classes'][label]['element_counts']
        pieces = ', '.join(
            f'{n}: {values["count"]} ({values["percentage"]:.1f}%)'
            for n, values in dist.items()
        )
        print(f'{label}: n={total:,}; {pieces}')
    print(f'Wrote {weighted_pdf}')
    print(f'Wrote {paper_pdf}')


if __name__ == '__main__':
    main()
