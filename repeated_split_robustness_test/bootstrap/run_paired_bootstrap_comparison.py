import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys

import numpy as np
import pandas as pd

from common import EXP_ROOT, dump_json, load_json

LABELS = np.array([0, 1, 2], dtype=int)
SEMIMETAL_LABEL = 0
TI_LABEL = 1
TRIVIAL_LABEL = 2
N_BOOT = 3000
RNG_SEED = 20260615


def resolve_split_dirs(split_name: str):
    return (
        EXP_ROOT / 'xgb' / split_name / 'output',
        EXP_ROOT / 'txl' / split_name / 'output',
        EXP_ROOT / 'bootstrap' / split_name,
    )


def load_prediction_arrays(path: Path):
    data = load_json(path)
    if isinstance(data, dict) and 'y_true' in data and 'y_pred' in data:
        return np.asarray(data['y_true'], dtype=int), np.asarray(data['y_pred'], dtype=int)
    if isinstance(data, list):
        label_to_code = {'semimetal': 0, 'topological': 1, 'trivial': 2}
        y_true = np.asarray([label_to_code[row['true_label']] for row in data], dtype=int)
        y_pred = np.asarray([label_to_code[row['predicted_label']] for row in data], dtype=int)
        return y_true, y_pred
    raise ValueError(f'Unsupported prediction file format: {path}')


def class_precision_from_preds(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    denom = tp + fp
    return 0.0 if denom == 0 else float(tp / denom)


def class_recall_from_preds(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    denom = tp + fn
    return 0.0 if denom == 0 else float(tp / denom)


def class_f1_from_preds(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / denom)


def macro_f1_from_preds(y_true, y_pred):
    return float(np.mean([class_f1_from_preds(y_true, y_pred, label) for label in LABELS]))


def metric_delta(y_true, pred_xgb, pred_txl):
    deltas = {}
    class_map = [
        ('tsm', SEMIMETAL_LABEL),
        ('ti', TI_LABEL),
        ('trivial', TRIVIAL_LABEL),
    ]
    for prefix, label in class_map:
        deltas[f'delta_{prefix}_precision'] = class_precision_from_preds(y_true, pred_txl, label) - class_precision_from_preds(y_true, pred_xgb, label)
        deltas[f'delta_{prefix}_recall'] = class_recall_from_preds(y_true, pred_txl, label) - class_recall_from_preds(y_true, pred_xgb, label)
        deltas[f'delta_{prefix}_f1'] = class_f1_from_preds(y_true, pred_txl, label) - class_f1_from_preds(y_true, pred_xgb, label)
    deltas['delta_macro_f1'] = macro_f1_from_preds(y_true, pred_txl) - macro_f1_from_preds(y_true, pred_xgb)
    return deltas


def run_bootstrap(y_true_xgb, pred_xgb, y_true_txl, pred_txl):
    if not np.array_equal(y_true_xgb, y_true_txl):
        raise ValueError('Mismatched y_true arrays between XGB and TXL predictions')
    y_true = y_true_xgb
    observed = metric_delta(y_true, pred_xgb, pred_txl)
    n = len(y_true)
    rng = np.random.default_rng(RNG_SEED)
    metrics = [
        'delta_trivial_precision', 'delta_tsm_precision', 'delta_ti_precision',
        'delta_trivial_recall', 'delta_tsm_recall', 'delta_ti_recall',
        'delta_trivial_f1', 'delta_tsm_f1', 'delta_ti_f1', 'delta_macro_f1',
    ]
    boot = np.empty((N_BOOT, len(metrics)), dtype=float)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals = metric_delta(y_true[idx], pred_xgb[idx], pred_txl[idx])
        for j, metric in enumerate(metrics):
            boot[b, j] = vals[metric]
    boot_df = pd.DataFrame(boot, columns=metrics)
    summary_rows = []
    for metric in boot_df.columns:
        summary_rows.append({
            'metric': metric,
            'observed_mean_delta': float(observed[metric]),
            'bootstrap_mean_delta': float(boot_df[metric].mean()),
            'ci_lower_95': float(boot_df[metric].quantile(0.025)),
            'ci_upper_95': float(boot_df[metric].quantile(0.975)),
            'prob_delta_gt_0': float((boot_df[metric] > 0).mean()),
        })
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-name', required=True)
    args = parser.parse_args()
    xgb_dir, txl_dir, out_dir = resolve_split_dirs(args.split_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for split_name, pred_file in [('validation', 'validation_predictions.json'), ('heldout_test', 'heldout_test_predictions.json')]:
        y_true_xgb, pred_xgb = load_prediction_arrays(xgb_dir / pred_file)
        y_true_txl, pred_txl = load_prediction_arrays(txl_dir / pred_file)
        summary_df = run_bootstrap(y_true_xgb, pred_xgb, y_true_txl, pred_txl)
        summary_df.to_csv(out_dir / f'{split_name}_paired_bootstrap_summary.csv', index=False)
        dump_json(out_dir / f'{split_name}_paired_bootstrap_summary.json', summary_df.to_dict(orient='records'))
        outputs[split_name] = summary_df.to_dict(orient='records')
        print(split_name)
        print(summary_df.to_string(index=False))
    dump_json(out_dir / 'run_metadata.json', {
        'split_name': args.split_name,
        'rng_seed': RNG_SEED,
        'n_bootstrap': N_BOOT,
        'xgb_dir': str(xgb_dir),
        'txl_dir': str(txl_dir),
    })


if __name__ == '__main__':
    main()
