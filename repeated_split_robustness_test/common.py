import json
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

ROOT = Path('/path/to/3_classes_classification')
EXP_ROOT = ROOT / 'weighted_version' / 'repeated_split_robustness_test'
TRAINING_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_training_data.json'
TEST_DATA_PATH = ROOT / 'label_noise_analysis' / 'clean_test_data.json'
LABEL_ORDER = ['semimetal', 'topological', 'trivial']
LABEL_FIT_ORDER = ['topological', 'semimetal', 'trivial']


def load_json(path):
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def normalize_label(raw):
    label = raw.strip().lower()
    return {'sm': 'semimetal', 'tsm': 'semimetal', 'ti': 'topological'}.get(label, label)


def normalize_element_symbol(symbol):
    return 'H' if symbol == 'D' else symbol


def normalize_formula(formula):
    comp = Composition(formula)
    normalized = {}
    for el, amt in comp.get_el_amt_dict().items():
        key = normalize_element_symbol(el)
        normalized[key] = normalized.get(key, 0.0) + amt
    return Composition(normalized).formula.replace(' ', '')


def record_key(entry):
    return {
        'compound': normalize_formula(entry['compoundName']),
        'space_group': int(round(entry.get('symmetryGroupNumber', 0))),
        'label': normalize_label(entry['topologicalClassificationShortDescription']),
    }


def validate_manifest_against_training_data(manifest, rows):
    expected_keys = [record_key(row) for row in rows]
    if expected_keys != manifest['record_keys']:
        raise ValueError('Split manifest record_keys do not match clean_training_data.json')


def load_split_manifest(split_dir):
    split_dir = Path(split_dir)
    return load_json(split_dir / 'split_manifest.json')


def balanced_class_weights(y):
    labels, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(labels)
    return {int(label): float(n / (k * count)) for label, count in zip(labels, counts)}


def summarize_class_counts(rows):
    return dict(Counter(normalize_label(r['topologicalClassificationShortDescription']) for r in rows))
