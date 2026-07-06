import importlib.util
from pathlib import Path

ROOT = Path('/path/to/3_classes_classification')
XGB_SCRIPT = ROOT / 'weighted_version' / 'xgb' / 'weighted_xgboost_shared_split.py'
LABEL_CODE_MAP = {0: 'semimetal', 1: 'topological', 2: 'trivial'}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


xgb_mod = load_module(XGB_SCRIPT, 'weighted_ds2_xgb_module')
train_data = xgb_mod.load_json(xgb_mod.TRAINING_DATA_PATH)
train_idx, _ = xgb_mod.load_shared_split(train_data)
subtrain_entries = [train_data[i] for i in train_idx]
sg_class_counts, sg_total_counts = xgb_mod.build_space_group_priors(subtrain_entries)


def get_formula(entry):
    return entry.get('reduced_formula') or entry.get('compoundName') or entry.get('formula')


def get_space_group(entry):
    return entry.get('space_group', entry.get('symmetryGroupNumber', entry.get('sg', 0)))


def canonical_label(raw):
    if isinstance(raw, int):
        return LABEL_CODE_MAP.get(raw, 'trivial')
    if isinstance(raw, float) and raw.is_integer():
        return LABEL_CODE_MAP.get(int(raw), 'trivial')
    label = str(raw).strip().lower()
    if label.isdigit():
        return LABEL_CODE_MAP.get(int(label), 'trivial')
    return {'sm': 'semimetal', 'tsm': 'semimetal', 'ti': 'topological'}.get(label, label)


def adapt_entry(entry):
    return {
        'compoundName': get_formula(entry),
        'symmetryGroupNumber': get_space_group(entry),
        'topologicalClassificationShortDescription': canonical_label(entry.get('label', entry.get('topologicalClassificationShortDescription', 'trivial'))),
    }


def input_gen(entry):
    row = xgb_mod.build_feature_rows([adapt_entry(entry)], sg_class_counts, sg_total_counts)[0]
    row.pop('label', None)
    return row
