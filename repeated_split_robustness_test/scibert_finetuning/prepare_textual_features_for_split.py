import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys

from common import EXP_ROOT, TRAINING_DATA_PATH, dump_json, load_json, validate_manifest_against_training_data

ROOT = Path('/path/to/3_classes_classification')
TEXT_HELPER_SCRIPT = ROOT / 'uncased_scibert_improved_input' / 'text_4_inference_improved.py'


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_split_dir(split_name: str):
    return EXP_ROOT / 'scibert_finetuning' / split_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-name', required=True)
    args = parser.parse_args()

    split_dir = resolve_split_dir(args.split_name)
    text_helper = load_module(TEXT_HELPER_SCRIPT, f'text_helper_{args.split_name}')
    train_data = load_json(TRAINING_DATA_PATH)
    manifest = load_json(split_dir / 'split_manifest.json')
    validate_manifest_against_training_data(manifest, train_data)

    subtrain_entries = [train_data[i] for i in manifest['train_indices']]
    text_helper.sg_class_counts, text_helper.sg_total_counts = text_helper.build_space_group_priors(subtrain_entries)

    rows = []
    for entry in train_data:
        sample = text_helper.describe_compound(entry)
        if sample is None:
            raise RuntimeError(f'Could not serialize entry {entry.get("compoundName")}')
        rows.append(sample)

    source_keys = [text_helper.record_key_from_entry(entry) for entry in train_data]
    serialized_keys = [
        {'compound': row['compound'], 'space_group': int(row['space_group']), 'label': row['output']}
        for row in rows
    ]
    if source_keys != manifest['record_keys'] or serialized_keys != manifest['record_keys']:
        raise ValueError('Split-specific SciBERT narratives do not match split manifest order/content')

    dump_json(split_dir / 'finetune_dataset_scibert_improved.json', rows)
    print(split_dir / 'finetune_dataset_scibert_improved.json')


if __name__ == '__main__':
    main()
