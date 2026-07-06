import importlib.util
import json
from pathlib import Path

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
GM_EVAL_SCRIPT = ROOT / 'weighted_version' / 'gm_three_class_baseline' / 'evaluate_gm_three_class.py'


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main():
    gm = load_module(GM_EVAL_SCRIPT, 'weighted_gm_eval_module')
    tau_maps = {name: gm.load_json(path) for name, path in gm.TOP_MAP_FILES.items()}
    tau_metadata = gm.load_json(gm.CLEAN_TOPO_DIR / 'metadata.json')
    heldout_entries = gm.load_json(gm.TEST_PATH)

    rows, skipped, metrics = gm.evaluate_split(heldout_entries, tau_maps, 'heldout_test')
    summary = {
        'source': {
            'tau_dir': str(gm.CLEAN_TOPO_DIR),
            'heldout_test_data': str(gm.TEST_PATH),
        },
        'tau_metadata': tau_metadata,
        'heldout_test_metrics': metrics,
    }

    dump_json(OUT / 'heldout_test_predictions.json', rows)
    dump_json(OUT / 'heldout_test_skipped.json', skipped)
    dump_json(OUT / 'heldout_test_metrics.json', metrics)
    dump_json(OUT / 'heldout_test_summary.json', summary)

    print('Held-out accuracy:', metrics['accuracy'])
    print('Held-out macro-F1:', metrics['macro_f1'])
    print('Held-out weighted-F1:', metrics['weighted_f1'])
    print()
    print('Per-class metrics:')
    for cls in ['trivial', 'semimetal', 'topological']:
        vals = metrics['per_class'][cls]
        print(
            f"{cls}: precision={vals['precision']:.4f} recall={vals['recall']:.4f} "
            f"f1={vals['f1']:.4f} support={vals['support']}"
        )

    print()
    print('Classification report:')
    report = metrics['classification_report']
    for cls in ['trivial', 'semimetal', 'topological']:
        vals = report[cls]
        print(
            f"{cls}: precision={vals['precision']:.4f} recall={vals['recall']:.4f} "
            f"f1-score={vals['f1-score']:.4f} support={int(vals['support'])}"
        )


if __name__ == '__main__':
    main()
