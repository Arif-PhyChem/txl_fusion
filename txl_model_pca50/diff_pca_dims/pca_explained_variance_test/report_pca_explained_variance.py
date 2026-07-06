import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path('/path/to/3_classes_classification')
DEFAULT_MODEL_DIR = ROOT / 'uncased_scibert_improved_input' / 'scibert-finetuned-weighted-improved-input' / 'checkpoint-2958'
SHARED_SPLIT_PATH = ROOT / 'label_noise_analysis' / 'shared_split_manifest.json'
DEFAULT_INPUT_JSON = ROOT / 'weighted_version' / 'txl_model' / 'diff_pca_dims' / 'scibert_finetune_data.json'
DEFAULT_COMPONENTS = [1, 2, 3, 5, 10, 20, 50, 100, 768]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fit PCA on shared-subtraining SciBERT embeddings and report cumulative explained variance.'
    )
    parser.add_argument('--input-json', default=str(DEFAULT_INPUT_JSON))
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--shared-split', default=str(SHARED_SPLIT_PATH))
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--components', type=int, nargs='+', default=DEFAULT_COMPONENTS)
    parser.add_argument('--cache-dir', default='cache')
    parser.add_argument('--output-csv', default='pca_explained_variance_train_fit.csv')
    parser.add_argument('--output-json', default='pca_explained_variance_train_fit.json')
    parser.add_argument('--pca-output', default='pca_train_fit_full.pkl')
    return parser.parse_args()


def resolve(base_dir, value):
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_text_rows(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_shared_split(path):
    with open(path, encoding='utf-8') as f:
        manifest = json.load(f)
    return np.array(manifest['train_indices']), np.array(manifest['validation_indices'])


def mean_pool_embeddings(texts, tokenizer, model, device, max_length, batch_size):
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        print(f'Embedded {min(start + batch_size, len(texts))}/{len(texts)} texts', flush=True)
    return np.vstack(embeddings)


def get_or_create_embeddings(all_texts, train_idx, val_idx, model_dir, shared_split, max_length, batch_size, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / 'train_embeddings.npy'
    val_cache = cache_dir / 'val_embeddings.npy'
    metadata_cache = cache_dir / 'cache_metadata.json'
    expected_metadata = {
        'model_dir': str(model_dir),
        'shared_split': str(shared_split),
        'n_samples': len(all_texts),
        'n_train': int(len(train_idx)),
        'n_validation': int(len(val_idx)),
        'max_length': int(max_length),
    }
    if train_cache.exists() and val_cache.exists() and metadata_cache.exists():
        cached_metadata = json.loads(metadata_cache.read_text())
        if cached_metadata == expected_metadata:
            print('Loading cached embeddings.')
            return np.load(train_cache), np.load(val_cache)
        print('Cached embeddings do not match current configuration; regenerating cache.')

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    full_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    bert_model = full_model.bert
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model.to(device)
    bert_model.eval()
    print(f'Generating SciBERT embeddings on {device}.')
    all_embeddings = mean_pool_embeddings(all_texts, tokenizer, bert_model, device, max_length, batch_size)
    train_embeddings = all_embeddings[train_idx]
    val_embeddings = all_embeddings[val_idx]
    np.save(train_cache, train_embeddings)
    np.save(val_cache, val_embeddings)
    metadata_cache.write_text(json.dumps(expected_metadata, indent=2))
    return train_embeddings, val_embeddings


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_json = resolve(base_dir, args.input_json)
    model_dir = resolve(base_dir, args.model_dir)
    shared_split = resolve(base_dir, args.shared_split)
    cache_dir = resolve(base_dir, args.cache_dir)
    output_csv = resolve(base_dir, args.output_csv)
    output_json = resolve(base_dir, args.output_json)
    pca_output = resolve(base_dir, args.pca_output)

    rows = load_text_rows(input_json)
    all_texts = [entry['text'] for entry in rows]
    train_idx, val_idx = load_shared_split(shared_split)
    train_embeddings, val_embeddings = get_or_create_embeddings(
        all_texts, train_idx, val_idx, model_dir, shared_split, args.max_length, args.batch_size, cache_dir
    )

    max_components = max(args.components)
    if max_components > min(train_embeddings.shape):
        raise ValueError(f'Requested {max_components} components, but train embeddings shape is {train_embeddings.shape}.')

    print(f'Fitting PCA(n_components={max_components}) on shared-subtraining embeddings only.')
    pca = PCA(n_components=max_components, random_state=args.seed)
    train_projected = pca.fit_transform(train_embeddings)
    val_projected = pca.transform(val_embeddings)
    joblib.dump(pca, pca_output)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    summary_rows = []
    for n_components in args.components:
        summary_rows.append({
            'n_components': int(n_components),
            'train_cumulative_explained_variance': float(cumulative[n_components - 1]),
            'train_projected_shape': list(train_projected[:, :n_components].shape),
            'val_projected_shape': list(val_projected[:, :n_components].shape),
        })

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        'input_json': str(input_json),
        'model_dir': str(model_dir),
        'shared_split': str(shared_split),
        'seed': int(args.seed),
        'n_samples': int(len(all_texts)),
        'n_train': int(len(train_idx)),
        'n_validation': int(len(val_idx)),
        'embedding_dim': int(train_embeddings.shape[1]),
        'fit_policy': 'PCA fit on shared-subtraining embeddings only; validation embeddings transformed only.',
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'requested_component_summary': summary_rows,
    }
    output_json.write_text(json.dumps(report, indent=2))

    print('\nPCA cumulative explained variance:')
    for row in summary_rows:
        print(f"{row['n_components']:>4} PCs: {row['train_cumulative_explained_variance']:.6f}")
    print(f'\nWrote {output_csv}')
    print(f'Wrote {output_json}')
    print(f'Wrote {pca_output}')


if __name__ == '__main__':
    main()
