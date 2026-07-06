
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import joblib

PY39_LOCAL = '<PYTHON_SITE_PACKAGES>'
if PY39_LOCAL not in sys.path:
    sys.path.append(PY39_LOCAL)

import umap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

ROOT = Path('/path/to/3_classes_classification')
OUT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'uncased_scibert_improved_input' / 'scibert-finetuned-weighted-improved-input' / 'checkpoint-2958'
PCA_MODEL_DIR = ROOT / 'weighted_version' / 'llm_pca_50'
PCA_MODEL_PATH = PCA_MODEL_DIR / 'pca_model.pkl'
XGB_DATA_PATH = ROOT / 'weighted_version' / 'heuristic_xgb' / 'xgb_data_weighted_shared_split.json'
LLM_DATA_PATH = ROOT / 'uncased_scibert_improved_input' / 'finetune_dataset_scibert_improved.json'
EMBEDDING_CACHE_PATH = OUT / 'scibert_embeddings_weighted_improved.npy'
EMBEDDING_META_PATH = OUT / 'scibert_embeddings_weighted_improved_metadata.json'
PDF_OUTPUT_PATH = OUT / 'topology_comparison_weighted.pdf'
PAPER_OUTPUT_PATH = ROOT / 'paper_review' / 'topology_comparison.pdf'
SCORES_OUTPUT_PATH = OUT / 'topology_comparison_scores.json'
UMAP_CACHE_PATH = OUT / 'topology_comparison_umap_cache.npz'
UMAP_META_PATH = OUT / 'topology_comparison_umap_cache_metadata.json'
RUN_LOCK_PATH = OUT / 'topology_comparison_run_lock.json'
MAX_LENGTH = 512
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_ORDER = ['trivial', 'semimetal', 'topological']
LABEL_MAP = {'trivial': 'Trivial', 'semimetal': 'TSM', 'topological': 'TI'}
PALETTE = {'Trivial': '#2ca02c', 'TSM': '#1f77b4', 'TI': '#d62728'}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def set_determinism():
    os.environ.setdefault('PYTHONHASHSEED', '42')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def embedding_cache_metadata(text_list):
    return {
        'model_dir': str(MODEL_DIR),
        'pca_model_path': str(PCA_MODEL_PATH),
        'pca_model_sha256': file_sha256(PCA_MODEL_PATH),
        'llm_data_path': str(LLM_DATA_PATH),
        'llm_data_sha256': file_sha256(LLM_DATA_PATH),
        'n_texts': len(text_list),
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
    }


def load_data():
    df_xgb = pd.DataFrame(load_json(XGB_DATA_PATH)).fillna(0)
    df_llm = pd.DataFrame(load_json(LLM_DATA_PATH))
    if len(df_xgb) != len(df_llm):
        raise ValueError(f'Length mismatch: xgb={len(df_xgb)} llm={len(df_llm)}')
    xgb_labels = df_xgb['label'].astype(str).str.lower().tolist()
    llm_labels = df_llm['output'].astype(str).str.lower().tolist()
    if xgb_labels != llm_labels:
        mismatch = next((i for i, (a, b) in enumerate(zip(xgb_labels, llm_labels)) if a != b), None)
        raise ValueError(f'Label alignment mismatch at row {mismatch}: {xgb_labels[mismatch]} vs {llm_labels[mismatch]}')
    df_xgb['plot_label'] = [LABEL_MAP[l] for l in xgb_labels]
    df_llm['plot_label'] = [LABEL_MAP[l] for l in llm_labels]
    return df_xgb, df_llm


def cache_is_valid(text_list):
    if not EMBEDDING_CACHE_PATH.exists() or not EMBEDDING_META_PATH.exists():
        return False
    try:
        meta = load_json(EMBEDDING_META_PATH)
    except Exception:
        return False
    return meta == embedding_cache_metadata(text_list)


def run_lock_metadata(text_list, n_rows, numeric_shape, embedding_shape):
    return {
        'embedding': embedding_cache_metadata(text_list),
        'umap': umap_cache_metadata(n_rows, numeric_shape, embedding_shape),
        'scores_path': str(SCORES_OUTPUT_PATH),
        'scores_sha256': file_sha256(SCORES_OUTPUT_PATH) if SCORES_OUTPUT_PATH.exists() else None,
        'script_sha256': file_sha256(Path(__file__)),
    }


def run_lock_is_valid(text_list, n_rows, numeric_shape, embedding_shape):
    if not RUN_LOCK_PATH.exists() or not SCORES_OUTPUT_PATH.exists() or not UMAP_CACHE_PATH.exists():
        return False
    try:
        meta = load_json(RUN_LOCK_PATH)
    except Exception:
        return False
    return meta == run_lock_metadata(text_list, n_rows, numeric_shape, embedding_shape)


def load_locked_run():
    with np.load(UMAP_CACHE_PATH) as cache:
        umap_results = {key: cache[key] for key in cache.files if key.startswith(('xgb_', 'llm_'))}
        scores = json.loads(str(cache['scores_json']))
    return umap_results, scores


def get_or_load_embeddings(text_list):
    if cache_is_valid(text_list):
        print(f'Loading cached embeddings from {EMBEDDING_CACHE_PATH}...')
        return np.load(EMBEDDING_CACHE_PATH)

    print('Generating embeddings using weighted improved SciBERT (cache miss)...')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModel.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(text_list), BATCH_SIZE), desc='Encoding'):
        batch_texts = text_list[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_vecs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(batch_vecs)

    embeddings = np.vstack(all_embeddings)
    np.save(EMBEDDING_CACHE_PATH, embeddings)
    dump_json(EMBEDDING_META_PATH, embedding_cache_metadata(text_list))
    return embeddings



def umap_cache_metadata(n_rows, numeric_shape, embedding_shape):
    return {
        'model_dir': str(MODEL_DIR),
        'xgb_data_path': str(XGB_DATA_PATH),
        'xgb_data_sha256': file_sha256(XGB_DATA_PATH),
        'llm_data_path': str(LLM_DATA_PATH),
        'llm_data_sha256': file_sha256(LLM_DATA_PATH),
        'pca_model_path': str(PCA_MODEL_PATH),
        'pca_model_sha256': file_sha256(PCA_MODEL_PATH),
        'embedding_cache_path': str(EMBEDDING_CACHE_PATH),
        'embedding_cache_sha256': file_sha256(EMBEDDING_CACHE_PATH),
        'n_rows': n_rows,
        'numeric_shape': list(numeric_shape),
        'embedding_shape': list(embedding_shape),
        'metrics': ['euclidean', 'cosine'],
        'n_neighbors': 15,
        'min_dist': 0.1,
        'random_state': 42,
    }


def umap_cache_is_valid(n_rows, numeric_shape, embedding_shape):
    if not UMAP_CACHE_PATH.exists() or not UMAP_META_PATH.exists():
        return False
    try:
        meta = load_json(UMAP_META_PATH)
    except Exception:
        return False
    return meta == umap_cache_metadata(n_rows, numeric_shape, embedding_shape)


def get_or_run_umap(xgb_scaled, llm_embeddings, labels):
    metrics = ['euclidean', 'cosine']
    if umap_cache_is_valid(len(labels), xgb_scaled.shape, llm_embeddings.shape):
        print(f'Loading cached UMAP projections from {UMAP_CACHE_PATH}...')
        with np.load(UMAP_CACHE_PATH) as cache:
            umap_results = {key: cache[key] for key in cache.files if key.startswith(('xgb_', 'llm_'))}
            scores = json.loads(str(cache['scores_json']))
        return umap_results, scores

    umap_results = {}
    scores = {}
    print('Running UMAP projections for weighted 2x2 grid...')
    for m in metrics:
        reducer_xgb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=m, random_state=42)
        reducer_llm = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=m, random_state=42)

        umap_xgb = reducer_xgb.fit_transform(xgb_scaled)
        umap_results[f'xgb_{m}'] = umap_xgb
        scores[f'xgb_{m}'] = float(silhouette_score(umap_xgb, labels))

        umap_llm = reducer_llm.fit_transform(llm_embeddings)
        umap_results[f'llm_{m}'] = umap_llm
        scores[f'llm_{m}'] = float(silhouette_score(umap_llm, labels))

    np.savez_compressed(UMAP_CACHE_PATH, **umap_results, scores_json=json.dumps(scores))
    dump_json(UMAP_META_PATH, umap_cache_metadata(len(labels), xgb_scaled.shape, llm_embeddings.shape))
    return umap_results, scores


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        va='top',
        ha='left',
    )


def rasterize_scatter_points(ax):
    for collection in ax.collections:
        collection.set_rasterized(True)


def main():
    parser = argparse.ArgumentParser(description='Generate the topology comparison figure.')
    parser.add_argument('--force-recompute', action='store_true', help='Recompute embeddings and UMAP even if cached artifacts exist.')
    args = parser.parse_args()

    set_determinism()
    sns.set_theme(style='whitegrid', context='paper')
    df_xgb, df_llm = load_data()

    xgb_numeric = df_xgb.select_dtypes(include=[np.number])
    xgb_scaled = StandardScaler().fit_transform(xgb_numeric)
    text_list = df_llm['text'].tolist()
    llm_embeddings = get_or_load_embeddings(text_list)
    pca = joblib.load(PCA_MODEL_PATH)
    llm_pca_embeddings = pca.transform(llm_embeddings)

    locked_run_valid = (
        not args.force_recompute
        and run_lock_is_valid(text_list, len(df_xgb), xgb_scaled.shape, llm_pca_embeddings.shape)
    )
    if locked_run_valid:
        print(f'Loading locked comparison artifacts from {RUN_LOCK_PATH}...')
        umap_results, scores = load_locked_run()
    else:
        umap_results, scores = get_or_run_umap(xgb_scaled, llm_pca_embeddings, df_xgb['plot_label'].tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    scatter_kws = dict(s=18, alpha=0.62, edgecolor='none', rasterized=True)

    sns.scatterplot(x=umap_results['xgb_euclidean'][:, 0], y=umap_results['xgb_euclidean'][:, 1],
                    hue=df_xgb['plot_label'], palette=PALETTE, ax=axes[0, 0], **scatter_kws)
    axes[0, 0].set_title(f"Heuristic+XGB descriptor (Euclidean)\nSilhouette Score: {scores['xgb_euclidean']:.4f}", fontweight='bold')

    sns.scatterplot(x=umap_results['llm_euclidean'][:, 0], y=umap_results['llm_euclidean'][:, 1],
                    hue=df_llm['plot_label'], palette=PALETTE, ax=axes[0, 1], **scatter_kws)
    axes[0, 1].set_title(f"SciBERT semantic descriptor (Euclidean)\nSilhouette Score: {scores['llm_euclidean']:.4f}", fontweight='bold')

    sns.scatterplot(x=umap_results['xgb_cosine'][:, 0], y=umap_results['xgb_cosine'][:, 1],
                    hue=df_xgb['plot_label'], palette=PALETTE, ax=axes[1, 0], **scatter_kws)
    axes[1, 0].set_title(f"Heuristic+XGB descriptor (Cosine)\nSilhouette Score: {scores['xgb_cosine']:.4f}", fontweight='bold')

    sns.scatterplot(x=umap_results['llm_cosine'][:, 0], y=umap_results['llm_cosine'][:, 1],
                    hue=df_llm['plot_label'], palette=PALETTE, ax=axes[1, 1], **scatter_kws)
    axes[1, 1].set_title(f"SciBERT semantic descriptor (Cosine)\nSilhouette Score: {scores['llm_cosine']:.4f}", fontweight='bold')

    for label, ax in zip(['A', 'B', 'C', 'D'], axes.flat):
        add_panel_label(ax, label)
        rasterize_scatter_points(ax)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.legend(title='Phase', loc='best', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(PDF_OUTPUT_PATH, format='pdf', dpi=160, bbox_inches='tight')
    fig.savefig(PAPER_OUTPUT_PATH, format='pdf', dpi=160, bbox_inches='tight')
    dump_json(SCORES_OUTPUT_PATH, scores)
    dump_json(RUN_LOCK_PATH, run_lock_metadata(text_list, len(df_xgb), xgb_scaled.shape, llm_pca_embeddings.shape))
    print(f'Saved weighted comparison PDF to: {PDF_OUTPUT_PATH}')
    print(f'Copied paper-review PDF to: {PAPER_OUTPUT_PATH}')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
