"""Explicit runtime configuration; model paths resolve within the AI module."""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_config(path: str | Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    path = Path(path or os.environ.get('CMU_AI_CONFIG', root / 'configs/rebuild.json'))
    config = json.loads(path.read_text())
    config['ai_root'] = str(root)
    for key in ('grounding_dino_checkpoint', 'sam21_checkpoint', 'qwen_checkpoint', 'da3_checkpoint', 'calibration', 'output_root'):
        if key in config:
            value = Path(os.environ.get('CMU_AI_' + key.upper(), config[key]))
            config[key] = str(value if value.is_absolute() else root / value)
    config['question_time_budget_seconds'] = float(os.environ.get(
        'CHALLENGE_QUESTION_TIME_BUDGET_SECONDS', config['question_time_budget_seconds']))
    return config
