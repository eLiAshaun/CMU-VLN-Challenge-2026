"""Merge the robot profile over the selected research model configuration."""
import json
import os
from pathlib import Path
from rebuild.config import load_config as load_rebuild_config


def load_config(path=None):
    root = Path(__file__).resolve().parents[1]
    config = load_rebuild_config(root / 'configs/rebuild.json')
    profile = Path(path or os.environ.get('GO2_AI_CONFIG', root / 'configs/go2_d435i.json'))
    config.update(json.loads(profile.read_text()))
    for key in ('output_root', 'qwen_checkpoint', 'sam21_checkpoint',
                'grounding_dino_checkpoint', 'da3_checkpoint'):
        value = Path(os.environ.get('CMU_AI_' + key.upper(), config[key]))
        config[key] = str(value if value.is_absolute() else root / value)
    if config['world_frame'] != 'map':
        raise ValueError('The shared object/query core currently requires a map frame')
    return config
