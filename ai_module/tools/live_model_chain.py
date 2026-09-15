#!/usr/bin/env python3
"""Thin production CLI for one CMU-VLN acquisition.

All substantive work lives under ``orchestration`` and ``integrations``.  This
file deliberately exposes no legacy helpers and imports no placeholder stage
runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from orchestration.pipeline import run_episode_chain


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one lean CMU-VLN acquisition")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()

    request_path = args.request.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    config_path = AI_ROOT / "configs" / "model_assets.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_episode_chain(request, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
