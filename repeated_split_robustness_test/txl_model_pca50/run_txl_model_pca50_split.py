import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import EXP_ROOT

ROOT = Path('/path/to/3_classes_classification')
PIPELINE = ROOT / 'weighted_version' / 'txl_fusion_variants' / 'common_fusion_pipeline.py'


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-name', required=True)
    args = parser.parse_args()

    split_dir = EXP_ROOT / 'txl_model_pca50' / args.split_name
    config_path = split_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing config: {config_path}')
    checkpoint_dir = ROOT / 'weighted_version' / 'repeated_split_robustness_test' / 'scibert_finetuning' / args.split_name / 'scibert-finetuned-weighted-improved-input' / 'best_model'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'Missing split-specific SciBERT checkpoint: {checkpoint_dir}')
    module = load_module(PIPELINE, f'txl_pca50_pipeline_{args.split_name}')
    module.run_variant(config_path)


if __name__ == '__main__':
    main()
