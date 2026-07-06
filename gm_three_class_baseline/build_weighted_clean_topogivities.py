import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.core import Composition
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

ROOT = Path('/path/to/3_classes_classification')
LABEL_DIR = ROOT / 'label_noise_analysis'
TRAIN_PATH = LABEL_DIR / 'clean_training_data.json'
TEST_PATH = LABEL_DIR / 'clean_test_data.json'
SPLIT_PATH = LABEL_DIR / 'shared_split_manifest.json'
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / 'weighted_topogivity_artifacts'
EXCLUDED_ELEMENT = 'O'
CLASS_LABELS = ['trivial', 'semimetal', 'topological']
GAMMA_GRID = [
    1.28e-8,
    4.05e-8,
    1.28e-7,
    4.05e-7,
    1.28e-6,
    4.05e-6,
    1.28e-5,
    4.05e-5,
    1.28e-4,
]


def canonical_label(label):
    label = label.strip().lower()
    return {'ti': 'topological', 'sm': 'semimetal', 'tsm': 'semimetal'}.get(label, label)


def normalize_element_symbol(symbol):
    return 'H' if symbol == 'D' else symbol


def normalize_formula(formula):
    comp = Composition(formula)
    normalized = defaultdict(float)
    for el, amt in comp.get_el_amt_dict().items():
        normalized[normalize_element_symbol(el)] += amt
    return Composition(normalized).formula.replace(' ', '')


def record_key_from_entry(entry):
    return {
        'compound': normalize_formula(entry['compoundName']),
        'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
        'label': canonical_label(entry['topologicalClassificationShortDescription']),
    }


def build_element_axis(training_data):
    element_set = set()
    for entry in training_data:
        try:
            comp = Composition(normalize_formula(entry['compoundName']))
        except Exception:
            continue
        element_set.update(el.symbol for el in comp.elements)
    all_elements = sorted(element_set, key=lambda x: Composition(x).elements[0].Z)
    tilde_elements = [el for el in all_elements if el != EXCLUDED_ELEMENT]
    return all_elements, tilde_elements


def get_tilde_fractions(formula, tilde_elements):
    comp = Composition(normalize_formula(formula))
    total = comp.num_atoms
    fractions = comp.get_el_amt_dict()
    return [fractions.get(el, 0.0) / total for el in tilde_elements]


def featurize(entries, tilde_elements):
    X, y, kept_indices = [], [], []
    for idx, entry in enumerate(entries):
        try:
            x_vec = get_tilde_fractions(entry['compoundName'], tilde_elements)
        except Exception as exc:
            print(f"[WARN] Skipping {entry.get('compoundName')} at index {idx}: {exc}")
            continue
        label = canonical_label(entry['topologicalClassificationShortDescription'])
        if label not in CLASS_LABELS:
            print(f"[WARN] Skipping unknown label {label!r} at index {idx}")
            continue
        X.append(x_vec)
        y.append(label)
        kept_indices.append(idx)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=object), kept_indices


def train_tau_maps(X_train, y_train, all_elements, tilde_elements, gamma):
    c_value = 1 / (len(X_train) * gamma)
    tau_maps = {}
    model_info = {}

    for positive_label in CLASS_LABELS:
        binary_y = np.where(y_train == positive_label, 1, -1)
        svm = LinearSVC(
            C=c_value,
            class_weight='balanced',
            dual=False,
            max_iter=100000,
            random_state=42,
        )
        svm.fit(X_train, binary_y)

        w = svm.coef_[0]
        b = svm.intercept_[0]
        tau = {}
        for el in all_elements:
            if el == EXCLUDED_ELEMENT:
                tau[el] = float(b)
            else:
                tau[el] = float(w[tilde_elements.index(el)] + b)
        tau_maps[positive_label] = tau
        model_info[positive_label] = {'intercept': float(b), 'n_iter': int(svm.n_iter_)}

    return tau_maps, {'gamma': gamma, 'C': c_value, 'models': model_info}


def score_formula(formula, tau_map):
    comp = Composition(normalize_formula(formula))
    fractions = comp.get_el_amt_dict()
    total = comp.num_atoms
    score = 0.0
    for el, amt in fractions.items():
        score += (amt / total) * tau_map.get(normalize_element_symbol(el), 0.0)
    return score


def predict_entries(entries, tau_maps):
    y_true, y_pred, scores = [], [], []
    for entry in entries:
        label = canonical_label(entry['topologicalClassificationShortDescription'])
        if label not in CLASS_LABELS:
            continue
        try:
            row_scores = {cls: score_formula(entry['compoundName'], tau_maps[cls]) for cls in CLASS_LABELS}
        except Exception as exc:
            print(f"[WARN] Could not score {entry.get('compoundName')}: {exc}")
            continue
        pred = max(CLASS_LABELS, key=lambda cls: row_scores[cls])
        y_true.append(label)
        y_pred.append(pred)
        scores.append(row_scores)
    return np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object), scores


def evaluate_entries(entries, tau_maps):
    y_true, y_pred, _ = predict_entries(entries, tau_maps)
    return {
        'n': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average='weighted', zero_division=0)),
        'classification_report': classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        'labels': CLASS_LABELS,
    }


def validate_manifest(clean_training_data, manifest):
    record_keys = [record_key_from_entry(entry) for entry in clean_training_data]
    if record_keys != manifest['record_keys']:
        raise ValueError('Clean training data no longer matches shared_split_manifest.json')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_training_data = json.loads(TRAIN_PATH.read_text())
    clean_test_data = json.loads(TEST_PATH.read_text())
    manifest = json.loads(SPLIT_PATH.read_text())
    validate_manifest(clean_training_data, manifest)

    subtrain_entries = [clean_training_data[i] for i in manifest['train_indices']]
    validation_entries = [clean_training_data[i] for i in manifest['validation_indices']]

    all_elements, tilde_elements = build_element_axis(subtrain_entries)
    X_subtrain, y_subtrain, _ = featurize(subtrain_entries, tilde_elements)

    sweep = []
    best = None
    for gamma in GAMMA_GRID:
        tau_maps, model_info = train_tau_maps(X_subtrain, y_subtrain, all_elements, tilde_elements, gamma)
        val_metrics = evaluate_entries(validation_entries, tau_maps)
        row = {
            'gamma': gamma,
            'C': model_info['C'],
            'validation_accuracy': val_metrics['accuracy'],
            'validation_macro_f1': val_metrics['macro_f1'],
            'validation_weighted_f1': val_metrics['weighted_f1'],
            'validation_ti_f1': val_metrics['classification_report']['topological']['f1-score'],
        }
        sweep.append(row)
        print(
            f"gamma={gamma:.2e} C={model_info['C']:.4g} "
            f"val_macro_f1={row['validation_macro_f1']:.4f} "
            f"val_ti_f1={row['validation_ti_f1']:.4f}"
        )
        key = (row['validation_macro_f1'], row['validation_ti_f1'], row['validation_accuracy'])
        if best is None or key > best['key']:
            best = {'key': key, 'gamma': gamma, 'tau_maps': tau_maps, 'model_info': model_info, 'validation_metrics': val_metrics}

    selected_tau_maps = best['tau_maps']
    test_metrics = evaluate_entries(clean_test_data, selected_tau_maps)

    outputs = {
        'trivial': 'trivial_topogivities.json',
        'semimetal': 'sm_topogivities.json',
        'topological': 'ti_topogivities.json',
    }
    for label, filename in outputs.items():
        out_path = OUT_DIR / filename
        with out_path.open('w') as f:
            json.dump(selected_tau_maps[label], f, indent=2)
        print(f'Wrote {out_path} ({len(selected_tau_maps[label])} elements)')

    metadata = {
        'source_training_data': str(TRAIN_PATH),
        'source_test_data': str(TEST_PATH),
        'shared_split_manifest': str(SPLIT_PATH),
        'n_clean_training_records': len(clean_training_data),
        'n_subtraining_records': len(subtrain_entries),
        'n_validation_records': len(validation_entries),
        'n_held_out_test_records': len(clean_test_data),
        'excluded_element': EXCLUDED_ELEMENT,
        'class_weight': 'balanced',
        'gamma_grid': GAMMA_GRID,
        'selection_metric': 'validation_macro_f1, tie-broken by validation_ti_f1 then validation_accuracy',
        'selected_gamma': best['gamma'],
        'selected_C': best['model_info']['C'],
        'elements': all_elements,
        'tilde_elements': tilde_elements,
    }
    (OUT_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (OUT_DIR / 'validation_sweep.json').write_text(json.dumps(sweep, indent=2))
    (OUT_DIR / 'validation_metrics.json').write_text(json.dumps(best['validation_metrics'], indent=2))
    (OUT_DIR / 'heldout_test_metrics.json').write_text(json.dumps(test_metrics, indent=2))

    print(f"Selected gamma={best['gamma']:.2e}, C={best['model_info']['C']:.4g}")
    print(f"Validation macro-F1={best['validation_metrics']['macro_f1']:.4f}, TI F1={best['validation_metrics']['classification_report']['topological']['f1-score']:.4f}")
    print(f"Held-out macro-F1={test_metrics['macro_f1']:.4f}, TI F1={test_metrics['classification_report']['topological']['f1-score']:.4f}")
    print(f'Wrote metrics and metadata to {OUT_DIR}')


if __name__ == '__main__':
    main()
