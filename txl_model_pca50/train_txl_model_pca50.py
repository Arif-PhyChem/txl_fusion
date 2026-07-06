#!/usr/bin/env python3
"""Standalone TXL model trainer built from variant 29."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/path/to/3_classes_classification/weighted_version/txl_fusion_variants')
PIPELINE = ROOT / 'common_fusion_pipeline.py'
CONFIG = Path(__file__).resolve().parent / 'config.json'


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location('txl_fusion_pipeline', PIPELINE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.run_variant(CONFIG)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
