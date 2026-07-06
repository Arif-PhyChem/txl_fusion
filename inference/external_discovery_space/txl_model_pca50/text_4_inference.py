import importlib.util
from pathlib import Path

ROOT = Path('/path/to/3_classes_classification')
TEXT_HELPER_SCRIPT = ROOT / 'uncased_scibert_improved_input' / 'text_4_inference_improved.py'
LABEL_CODE_MAP = {0: 'semimetal', 1: 'topological', 2: 'trivial'}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


helper = load_module(TEXT_HELPER_SCRIPT, 'weighted_ds2_txl_text_helper')


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


def prep_text(entry):
    return helper.prep_text(adapt_entry(entry))
