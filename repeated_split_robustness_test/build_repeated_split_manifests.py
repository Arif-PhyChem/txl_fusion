import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from common import TRAINING_DATA_PATH, normalize_label, record_key

EXP = Path(__file__).resolve().parent
SEEDS = [101, 202, 303, 404, 505]
SPLIT_NAMES = [f'split_{i:02d}_seed_{seed}' for i, seed in enumerate(SEEDS, start=1)]


def main():
    rows = json.loads(Path(TRAINING_DATA_PATH).read_text())
    labels = np.asarray([normalize_label(r['topologicalClassificationShortDescription']) for r in rows])
    indices = np.arange(len(rows))
    record_keys = [record_key(r) for r in rows]
    for name, seed in zip(SPLIT_NAMES, SEEDS):
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
        split_def = EXP / 'split_definitions' / name
        split_def.mkdir(parents=True, exist_ok=True)
        manifest = {
            'source_training_data': str(TRAINING_DATA_PATH),
            'random_state': seed,
            'train_indices': train_idx.tolist(),
            'validation_indices': val_idx.tolist(),
            'record_keys': record_keys,
        }
        metadata = {
            'split_name': name,
            'random_state': seed,
            'n_total': len(rows),
            'n_subtraining': int(len(train_idx)),
            'n_validation': int(len(val_idx)),
            'subtraining_class_counts': dict(Counter(labels[train_idx].tolist())),
            'validation_class_counts': dict(Counter(labels[val_idx].tolist())),
        }
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        (split_def / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
        (split_def / 'subtraining_data.json').write_text(json.dumps(train_rows, indent=2))
        (split_def / 'validation_data.json').write_text(json.dumps(val_rows, indent=2))
        (split_def / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        print(name, 'done')


if __name__ == '__main__':
    main()
