#!/usr/bin/env python3
"""Exercise Commit 1 tracing against recorded observations, without models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from integrations.execution.scene_memory import update_scene_memory


def _read(path: Path, default: object) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def _run_once(session: Path) -> tuple[list[dict[str, Any]], list[str]]:
    traces = []
    recorded = sorted(
        value for value in (session / "live_robot").iterdir()
        if value.is_dir()
        and (value / "summary.json").is_file()
    )
    acquisitions = [
        value for value in recorded
        if (value / "05_lidar_geometry" / "observations.json").is_file()
    ]
    with tempfile.TemporaryDirectory(prefix="phase3a_commit1_trace_") as temporary:
        store = Path(temporary) / "scene_memory.json"
        for acquisition in acquisitions:
            geometry = _read(acquisition / "05_lidar_geometry" / "observations.json", {})
            state = _read(acquisition / "state_estimation.json", {})
            summary = _read(acquisition / "summary.json", {})
            view_count = int(
                summary.get("stages", {}).get("perception", {}).get("view_count", 0)
            )
            snapshot = update_scene_memory(
                store,
                geometry.get("observations", ()),
                acquisition_id=acquisition.name,
                viewpoint_position_map=state.get("position_xyz"),
                panorama_view_count=view_count,
            )
            traces.append(snapshot["last_acquisition_trace"])
    return traces, [value.name for value in recorded if value not in acquisitions]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    output = args.output.resolve()
    first, skipped = _run_once(session)
    second, second_skipped = _run_once(session)
    deterministic = first == second and skipped == second_skipped
    if not deterministic:
        raise RuntimeError("commit1_trace_replay_nondeterministic")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in first),
        encoding="utf-8",
    )
    report = {
        "schema_version": "phase3a_commit1_trace_replay_v1",
        "source_session": str(session),
        "recorded_acquisition_count": len(first) + len(skipped),
        "memory_transaction_count": len(first),
        "skipped_no_geometry_acquisition_ids": skipped,
        "deterministic_two_run_match": deterministic,
        "ledger_size_after_replay": (
            first[-1]["observation_ledger"]["ledger_size"] if first else 0
        ),
        "output": str(output),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
