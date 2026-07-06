import csv
import json
from collections import defaultdict
from pathlib import Path

from pymatgen.core import Composition
from sklearn.metrics import classification_report

ROOT = Path('/path/to/3_classes_classification')
OUT_DIR = ROOT / 'weighted_version' / 'element_count_performance'
TEST_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_test_data.json'
PREDICTION_FILES = {
    'xgb': ROOT / 'weighted_version' / 'inference' / 'xgb' / 'heldout_test_predictions.json',
    'txl': ROOT / 'weighted_version' / 'inference' / 'txl_model_pca50' / 'heldout_test_predictions.json',
}

LABEL_ORDER = ['trivial', 'semimetal', 'topological']
DISPLAY_LABEL = {
    'trivial': 'Trivial',
    'semimetal': 'TSM',
    'topological': 'TI',
}
MODEL_TITLE = {
    'xgb': 'numerical descriptor-based XGB model',
    'txl': 'TXL Fusion model',
}
TABLE_LABEL = {
    'xgb': 'tab:xgb-elemental-report',
    'txl': 'tab:txl-elemental-report',
}
MAX_ELEMENTS = 6


def normalize_formula(formula):
    amounts = Composition(formula).get_el_amt_dict()
    normalized = defaultdict(float)
    for element, amount in amounts.items():
        normalized['H' if element == 'D' else element] += amount
    return Composition(dict(normalized))


def element_count(formula):
    return len(normalize_formula(formula).get_el_amt_dict())


def load_predictions(path):
    with path.open() as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise TypeError(f'{path} should contain a list of prediction rows')
    return rows


def grouped_reports(rows, test_data=None):
    groups = {n: {'true': [], 'pred': [], 'rows': []} for n in range(1, MAX_ELEMENTS + 1)}
    skipped = []
    for idx, row in enumerate(rows):
        try:
            compound = row.get('compound')
            if not compound and test_data is not None and idx < len(test_data):
                compound = test_data[idx].get('compoundName')
            n = element_count(compound)
            if n < 1 or n > MAX_ELEMENTS:
                skipped.append({'index': idx, 'reason': f'element_count={n}', 'row': row})
                continue
            true_label = str(row['true_label']).strip().lower()
            pred_label = str(row['predicted_label']).strip().lower()
            groups[n]['true'].append(true_label)
            groups[n]['pred'].append(pred_label)
            groups[n]['rows'].append(row)
        except Exception as exc:
            skipped.append({'index': idx, 'reason': str(exc), 'row': row})

    reports = {}
    for n, payload in groups.items():
        y_true = payload['true']
        y_pred = payload['pred']
        reports[str(n)] = {
            'n_samples': len(y_true),
            'classification_report': classification_report(
                y_true,
                y_pred,
                labels=LABEL_ORDER,
                target_names=LABEL_ORDER,
                output_dict=True,
                zero_division=0,
            ),
        }
    return reports, skipped


def flatten_for_csv(model_name, reports):
    rows = []
    for n in range(1, MAX_ELEMENTS + 1):
        report = reports[str(n)]['classification_report']
        for label in LABEL_ORDER:
            metrics = report[label]
            rows.append({
                'model': model_name,
                'n_elements': n,
                'group_n': reports[str(n)]['n_samples'],
                'class': DISPLAY_LABEL[label],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': metrics['support'],
            })
    return rows


def fmt(value):
    value = float(value)
    if abs(value - 1.0) < 5e-5:
        return '1.0000'
    if abs(value) < 5e-5:
        return '0.0000'
    return f'{value:.2f}'


def make_latex_table(model_name, reports):
    caption = (
        f'Classification performance of the {MODEL_TITLE[model_name]} for compounds grouped by the number of '
        'constituent elements (1--6) on the cleaned held-out test set. Precision, recall, F1-score, and support '
        '(number of samples) are reported for the trivial, TSM, and TI classes.'
    )
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{{TABLE_LABEL[model_name]}}}')
    lines.append(r'\begin{threeparttable}')
    lines.append(r'\begin{tabular}{@{}l l S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=4.1]@{}}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Elements} & \textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} & \textbf{Support} \\')
    lines.append(r'\midrule')
    for n in range(1, MAX_ELEMENTS + 1):
        total = reports[str(n)]['n_samples']
        lines.append(rf'\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{n}-Element Compounds (n = {total:,})}}}} \\')
        lines.append(r'\cmidrule(l){2-6}')
        report = reports[str(n)]['classification_report']
        for label in LABEL_ORDER:
            m = report[label]
            lines.append(
                rf'& {DISPLAY_LABEL[label]} & {fmt(m["precision"])} & {fmt(m["recall"])} & {fmt(m["f1-score"])} & {float(m["support"]):.1f} \\'
            )
        if n != MAX_ELEMENTS:
            lines.append('')
            lines.append(r'\addlinespace')
    lines.append('')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{threeparttable}')
    lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    csv_rows = []
    test_data = json.loads(TEST_DATA_PATH.read_text())
    for model_name, pred_path in PREDICTION_FILES.items():
        rows = load_predictions(pred_path)
        reports, skipped = grouped_reports(rows, test_data=test_data)
        all_summary[model_name] = {
            'prediction_file': str(pred_path),
            'n_predictions': len(rows),
            'n_skipped': len(skipped),
            'reports': reports,
        }
        (OUT_DIR / f'{model_name}_element_count_report.json').write_text(json.dumps(all_summary[model_name], indent=2))
        (OUT_DIR / f'{model_name}_element_count_table.tex').write_text(make_latex_table(model_name, reports))
        if skipped:
            (OUT_DIR / f'{model_name}_element_count_skipped.json').write_text(json.dumps(skipped, indent=2))
        csv_rows.extend(flatten_for_csv(model_name, reports))

    (OUT_DIR / 'element_count_performance_summary.json').write_text(json.dumps(all_summary, indent=2))
    with (OUT_DIR / 'element_count_performance_summary.csv').open('w', newline='') as f:
        fieldnames = ['model', 'n_elements', 'group_n', 'class', 'precision', 'recall', 'f1_score', 'support']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    for model_name in PREDICTION_FILES:
        summary = all_summary[model_name]
        group_sizes = ', '.join(f'{n}: {summary["reports"][str(n)]["n_samples"]}' for n in range(1, MAX_ELEMENTS + 1))
        print(f'{model_name}: predictions={summary["n_predictions"]:,}, skipped={summary["n_skipped"]}, groups=({group_sizes})')
    print(f'Wrote outputs to {OUT_DIR}')


if __name__ == '__main__':
    main()
