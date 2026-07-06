import json
from collections import defaultdict
from pathlib import Path

from pymatgen.core import Composition
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

BASE = Path(__file__).resolve().parent
ROOT = Path("/path/to/3_classes_classification")
CLEAN_TOPO_DIR = ROOT / "weighted_version" / "gm_three_class_baseline" / "weighted_topogivity_artifacts"
TRAIN_PATH = ROOT / "label_noise_analysis" / "clean_training_data.json"
TEST_PATH = ROOT / "label_noise_analysis" / "clean_test_data.json"
SPLIT_MANIFEST_PATH = ROOT / "label_noise_analysis" / "shared_split_manifest.json"

CLASS_MAP = {
    "trivial": "trivial",
    "sm": "semimetal",
    "tsm": "semimetal",
    "ti": "topological",
    "semimetal": "semimetal",
    "topological": "topological",
}
CLASSES = ["trivial", "semimetal", "topological"]
TOP_MAP_FILES = {
    "trivial": CLEAN_TOPO_DIR / "trivial_topogivities.json",
    "semimetal": CLEAN_TOPO_DIR / "sm_topogivities.json",
    "topological": CLEAN_TOPO_DIR / "ti_topogivities.json",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2))


def normalize_label(raw: str):
    if raw is None:
        return None
    return CLASS_MAP.get(raw.strip().lower())


def normalize_element_symbol(symbol: str) -> str:
    return "H" if symbol == "D" else symbol


def normalize_formula(formula: str) -> str:
    comp = Composition(formula)
    normalized = defaultdict(float)
    for el, amt in comp.get_el_amt_dict().items():
        normalized[normalize_element_symbol(el)] += amt
    return Composition(normalized).formula.replace(" ", "")


def record_key(entry):
    return {
        "compound": normalize_formula(entry["compoundName"]),
        "space_group": int(round(entry.get("symmetryGroupNumber", 0))),
        "label": normalize_label(entry.get("topologicalClassificationShortDescription")),
    }


def compute_g_score(formula: str, tau_map: dict[str, float]) -> float:
    comp = Composition(normalize_formula(formula))
    el_amt = comp.get_el_amt_dict()
    total_atoms = comp.num_atoms
    score = 0.0
    for el, amt in el_amt.items():
        score += (amt / total_atoms) * float(tau_map.get(normalize_element_symbol(el), 0.0))
    return float(score)


def evaluate_split(entries, tau_maps, split_name):
    rows = []
    y_true = []
    y_pred = []
    skipped = []

    for idx, entry in enumerate(entries):
        true_label = normalize_label(entry.get("topologicalClassificationShortDescription"))
        formula = entry.get("compoundName")
        if true_label is None or not formula:
            skipped.append({"index": idx, "formula": formula, "reason": "missing_label_or_formula"})
            continue

        try:
            normalized_formula = normalize_formula(formula)
            scores = {name: compute_g_score(normalized_formula, tau_map) for name, tau_map in tau_maps.items()}
        except Exception as exc:
            skipped.append({"index": idx, "formula": formula, "reason": str(exc)})
            continue

        pred_label = max(scores, key=scores.get)
        row = {
            "compoundName": formula,
            "normalized_formula": normalized_formula,
            "true_label": true_label,
            "predicted_label": pred_label,
            "scores": scores,
            "symmetryGroupNumber": entry.get("symmetryGroupNumber"),
            "similarICSD": entry.get("similarICSD", []),
        }
        rows.append(row)
        y_true.append(true_label)
        y_pred.append(pred_label)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, output_dict=True, digits=4, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)

    metrics = {
        "split": split_name,
        "n_evaluated": len(y_true),
        "n_skipped": len(skipped),
        "classes": CLASSES,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)),
        "per_class": {
            cls: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for cls, p, r, f, s in zip(CLASSES, precision, recall, f1, support)
        },
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": cm.tolist(),
        },
        "classification_report": report,
    }
    return rows, skipped, metrics


def main():
    tau_maps = {name: load_json(path) for name, path in TOP_MAP_FILES.items()}
    tau_metadata = load_json(CLEAN_TOPO_DIR / "metadata.json")
    train_data = load_json(TRAIN_PATH)
    test_data = load_json(TEST_PATH)
    split_manifest = load_json(SPLIT_MANIFEST_PATH)

    current_keys = [record_key(entry) for entry in train_data]
    if current_keys != split_manifest.get("record_keys", current_keys):
        raise ValueError("clean_training_data.json no longer matches shared_split_manifest.json")

    val_idx = split_manifest["validation_indices"]
    train_idx = split_manifest["train_indices"]
    validation_entries = [train_data[i] for i in val_idx]

    val_rows, val_skipped, val_metrics = evaluate_split(validation_entries, tau_maps, "validation")
    test_rows, test_skipped, test_metrics = evaluate_split(test_data, tau_maps, "heldout_test")

    dump_json(BASE / "validation_predictions.json", val_rows)
    dump_json(BASE / "validation_skipped.json", val_skipped)
    dump_json(BASE / "validation_metrics.json", val_metrics)
    dump_json(BASE / "heldout_test_predictions.json", test_rows)
    dump_json(BASE / "heldout_test_skipped.json", test_skipped)
    dump_json(BASE / "heldout_test_metrics.json", test_metrics)
    dump_json(BASE / "gm_three_class_predictions.json", {"validation": val_rows, "heldout_test": test_rows})
    dump_json(BASE / "gm_three_class_skipped.json", {"validation": val_skipped, "heldout_test": test_skipped})

    summary = {
        "source": {
            "tau_dir": str(CLEAN_TOPO_DIR),
            "training_data": str(TRAIN_PATH),
            "heldout_test_data": str(TEST_PATH),
            "split_manifest": str(SPLIT_MANIFEST_PATH),
            "n_subtraining_records": len(train_idx),
            "n_validation_records": len(val_idx),
            "n_heldout_test_records": len(test_data),
        },
        "tau_metadata": tau_metadata,
        "validation_metrics": val_metrics,
        "heldout_test_metrics": test_metrics,
    }
    dump_json(BASE / "gm_three_class_metrics.json", summary)

    summary_lines = [
        "split,accuracy,macro_f1,weighted_f1,trivial_f1,semimetal_f1,topological_f1,trivial_precision,trivial_recall,semimetal_precision,semimetal_recall,topological_precision,topological_recall,n_evaluated,n_skipped",
    ]
    for metrics in [val_metrics, test_metrics]:
        summary_lines.append(
            f"{metrics['split']}"
            f",{metrics['accuracy']:.6f}"
            f",{metrics['macro_f1']:.6f}"
            f",{metrics['weighted_f1']:.6f}"
            f",{metrics['per_class']['trivial']['f1']:.6f}"
            f",{metrics['per_class']['semimetal']['f1']:.6f}"
            f",{metrics['per_class']['topological']['f1']:.6f}"
            f",{metrics['per_class']['trivial']['precision']:.6f}"
            f",{metrics['per_class']['trivial']['recall']:.6f}"
            f",{metrics['per_class']['semimetal']['precision']:.6f}"
            f",{metrics['per_class']['semimetal']['recall']:.6f}"
            f",{metrics['per_class']['topological']['precision']:.6f}"
            f",{metrics['per_class']['topological']['recall']:.6f}"
            f",{metrics['n_evaluated']}"
            f",{metrics['n_skipped']}"
        )
    (BASE / "gm_three_class_summary.csv").write_text("\n".join(summary_lines) + "\n")

    print("Using weighted clean topogivities and shared split:")
    print(json.dumps(summary["source"], indent=2))
    print()
    for metrics in [val_metrics, test_metrics]:
        print(f"[{metrics['split']}] Accuracy: {metrics['accuracy']:.4f}")
        print(f"[{metrics['split']}] Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"[{metrics['split']}] Weighted-F1: {metrics['weighted_f1']:.4f}")
        for cls in CLASSES:
            vals = metrics['per_class'][cls]
            print(f"[{metrics['split']}] {cls}: P={vals['precision']:.4f} R={vals['recall']:.4f} F1={vals['f1']:.4f} support={vals['support']}")
        print()


if __name__ == "__main__":
    main()
