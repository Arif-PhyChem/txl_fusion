#!/usr/bin/env python3
"""Sweep SciBERT PCA dimensions for the final PCA50 TXL Fusion architecture."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/path/to/3_classes_classification')
PIPELINE_RUNNER = ROOT / 'weighted_version' / 'txl_fusion_variants' / 'run_variant.py'
FINAL_CONFIG = ROOT / 'weighted_version' / 'txl_model_pca50' / 'config.json'
DEFAULT_COMPONENTS = [0, 1, 2, 3, 5, 10, 20, 50, 100, 768]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train final TXL PCA50 architecture while sweeping SciBERT PCA component count.'
    )
    parser.add_argument('--base-config', default=str(FINAL_CONFIG))
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--summary-csv', default='pca_dim_f1_summary.csv')
    parser.add_argument('--summary-json', default='pca_dim_f1_summary.json')
    parser.add_argument('--components', type=int, nargs='+', default=DEFAULT_COMPONENTS)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--stop-on-failure', action='store_true')
    return parser.parse_args()


def resolve(base_dir, value):
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_metrics(model_dir):
    metrics_path = model_dir / 'output' / 'validation_metrics.json'
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding='utf-8') as f:
        metrics = json.load(f)
    config_path = model_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    feature_path = model_dir / 'output' / 'feature_names.json'
    n_features = None
    if feature_path.exists():
        features = json.loads(feature_path.read_text())
        names = features.get('feature_names') or features.get('all_features') or []
        n_features = len(names)
    report = metrics.get('classification_report', metrics)
    macro_f1 = metrics.get('macro_f1', report.get('macro avg', {}).get('f1-score'))
    weighted_f1 = metrics.get('weighted_f1', report.get('weighted avg', {}).get('f1-score'))
    return {
        'n_components': int(config.get('reported_pca_components', config.get('pca_components', -1))),
        'accuracy': float(metrics['accuracy']),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'semimetal_f1': float(report['semimetal']['f1-score']),
        'topological_f1': float(report['topological']['f1-score']),
        'trivial_f1': float(report['trivial']['f1-score']),
        'n_features': n_features,
        'model_path': str(model_dir / 'output'),
        'pca_model_path': str(model_dir / 'output' / 'pca_model.pkl'),
    }


def write_summary(rows, summary_csv, summary_json):
    rows = [r for r in rows if r is not None]
    rows = sorted(rows, key=lambda r: r['n_components'])
    summary_json.write_text(json.dumps(rows, indent=2))
    fieldnames = [
        'n_components', 'accuracy', 'macro_f1', 'weighted_f1',
        'semimetal_f1', 'topological_f1', 'trivial_f1',
        'n_features', 'model_path', 'pca_model_path',
    ]
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    base_config_path = resolve(base_dir, args.base_config)
    models_dir = resolve(base_dir, args.models_dir)
    summary_csv = resolve(base_dir, args.summary_csv)
    summary_json = resolve(base_dir, args.summary_json)
    models_dir.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(base_config_path.read_text())
    rows = []
    failures = []

    for n_components in args.components:
        if n_components < 0 or n_components > 768:
            raise ValueError(f'Invalid n_components={n_components}; expected 0..768.')
        run_dir = models_dir / f'pca_{n_components:03d}'
        config_path = run_dir / 'config.json'
        metrics_path = run_dir / 'output' / 'validation_metrics.json'

        if metrics_path.exists() and not args.overwrite:
            print(f'[SKIP] PCA={n_components}: existing metrics found at {metrics_path}', flush=True)
            rows.append(load_metrics(run_dir))
            write_summary(rows, summary_csv, summary_json)
            continue

        if args.overwrite and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config = dict(base_config)
        config['description'] = f"PCA sensitivity run for final PCA50 TXL architecture with {n_components} SciBERT PCA components."
        if n_components == 0:
            # The shared embedding pipeline cannot fit PCA(0). Use a numerical/heuristic-only
            # hierarchical baseline so PCA=0 keeps the same meaning as in the original sweep.
            config['variant_type'] = 'hierarchical_classifier'
            config['feature_strategy'] = 'concat_curated'
            config['numeric_feature_strategy'] = 'concat_curated'
            config['pca_components'] = 1
            config['reported_pca_components'] = 0
            config['llm_scale'] = 0.0
            config['description'] += ' This PCA=0 run uses no semantic features in the classifier.'
        else:
            config['pca_components'] = int(n_components)
        config_path.write_text(json.dumps(config, indent=2))

        cmd = [args.python, '-u', str(PIPELINE_RUNNER), '--config', str(config_path)]
        cmd_text = ' '.join(cmd)
        print(f'\n[RUN] PCA={n_components}: {cmd_text}', flush=True)
        result = subprocess.run(cmd, cwd=str(run_dir))
        if result.returncode != 0:
            failures.append({'n_components': n_components, 'returncode': result.returncode})
            print(f'[FAILED] PCA={n_components} return code {result.returncode}', flush=True)
            if args.stop_on_failure:
                break
            continue

        row = load_metrics(run_dir)
        rows.append(row)
        write_summary(rows, summary_csv, summary_json)
        print(
            f"[DONE] PCA={n_components}: macro-F1={row['macro_f1']:.4f}, "
            f"weighted-F1={row['weighted_f1']:.4f}, accuracy={row['accuracy']:.4f}",
            flush=True,
        )

    write_summary(rows, summary_csv, summary_json)
    if failures:
        (base_dir / 'failed_runs.json').write_text(json.dumps(failures, indent=2))
        print(f'Finished with {len(failures)} failed run(s). See failed_runs.json.', flush=True)
        if args.stop_on_failure:
            return 1
    print(f'\nWrote summary CSV: {summary_csv}')
    print(f'Wrote summary JSON: {summary_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
