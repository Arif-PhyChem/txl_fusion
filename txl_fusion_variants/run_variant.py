import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
config_path = Path(args.config).resolve()
config = json.loads(config_path.read_text())
visible_devices = config.get('cuda_visible_devices')
if visible_devices is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_devices)

from common_fusion_pipeline import run_variant

run_variant(config_path)
