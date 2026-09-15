#!/usr/bin/env python3
"""Re-lift every recorded N1 acquisition through the Commit 2 cutover."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from integrations.execution.scene_memory import materialize_query_view
from integrations.semantics.task_compiler import compile_task
from orchestration.lidar_geometry import lift_detections_to_observations


def _read(path: Path, default: object) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return default


def _localize(value: object, run: Path, container_prefix: Path) -> str:
    path = Path(str(value or ""))
    if not str(value or ""):
        return ""
    try:
        return str(run / path.relative_to(container_prefix))
    except ValueError:
        return str(path)


def _signature(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_to_entity": view["observation_to_entity"],
        "entities": [
            {
                "id": value["object_id"],
                "key": value["deterministic_entity_key"],
                "class": value["class_label"],
                "lifecycle": value.get("entity_lifecycle"),
                "status": value.get("status"),
                "evidence_count": len(value.get("evidence", ())),
            }
            for value in view["objects"]
        ],
        "cannot_links": view["identity_constraints"]["cannot_link"],
        "orphans": view["duplicate_risk_orphans"],
    }


def run(session: Path, output: Path, calibration: Path) -> dict[str, Any]:
    question = "How many pillows are on the bed?"
    task_ir = compile_task(question)
    acquisitions = sorted(
        value for value in (session / "live_robot").iterdir()
        if value.is_dir() and (value / "summary.json").is_file()
    )
    ledger: list[dict[str, Any]] = []
    rows = []
    skipped = []
    a6_fragment_fused = False
    with tempfile.TemporaryDirectory(prefix="phase3a_cutover_n1_") as temporary:
        temporary_root = Path(temporary)
        for acquisition_index, run_dir in enumerate(acquisitions):
            detections = _read(run_dir / "04_verified_detections.json", [])
            manifest = _read(run_dir / "01_perspective_views" / "manifest.json", {})
            if not detections or not manifest.get("views"):
                skipped.append({"acquisition_id": run_dir.name, "reason": "no_recorded_geometry_input"})
                continue
            container_prefix = Path("/home/docker/ai_module/runs/live_robot") / run_dir.name
            views = [dict(value) for value in manifest["views"]]
            for view in views:
                for key in ("image_path", "map_x_path", "map_y_path"):
                    view[key] = _localize(view.get(key), run_dir, container_prefix)
            localized_detections = [dict(value) for value in detections]
            for detection in localized_detections:
                for key in ("mask_path", "representative_crop", "relation_roi_image_path"):
                    if detection.get(key):
                        detection[key] = _localize(detection[key], run_dir, container_prefix)
            state = _read(run_dir / "state_estimation.json", {})
            result = lift_detections_to_observations(
                detections=localized_detections,
                views=views,
                competition_geometry={
                    "sensor_scan_path": str(run_dir / "sensor_scan.npy"),
                    "registered_scan_path": str(run_dir / "registered_scan.npy"),
                    "state_estimation_path": str(run_dir / "state_estimation.json"),
                },
                station_id=f"station:{run_dir.name}",
                acquisition_id=run_dir.name,
                output_dir=temporary_root / run_dir.name,
                calibration_path=calibration,
                timestamp_unix=state.get("stamp_seconds"),
            )
            acquisition_objects = list(result.get("observations", ()))
            ledger.extend(acquisition_objects)
            view = materialize_query_view(task_ir, {"records": ledger})
            if run_dir.name == "20260822T223845_284666Z":
                a6_fragment_fused = any(
                    {
                        "yolo_det_10000092",
                        "qwen_det_000008",
                        "yolo_det_10000086",
                        "qwen_det_000003",
                    }.issubset(set(value.get("source_detection_ids", ())))
                    for value in acquisition_objects
                )
            rows.append({
                "acquisition_index": acquisition_index,
                "acquisition_id": run_dir.name,
                "raw_proposal_count": len(detections),
                "acquisition_object_count": len(acquisition_objects),
                "query_observation_ledger_size": view["observation_ledger_size"],
                "entity_count": len(view["objects"]),
                "bed_entity_count": sum(value["class_label"] == "bed" for value in view["objects"]),
                "pillow_entity_count": sum(value["class_label"] == "pillow" for value in view["objects"]),
                "supported_pillow_count": sum(
                    value["class_label"] == "pillow" and value.get("status") == "confirmed"
                    for value in view["objects"]
                ),
                "provisional_singleton_ids": view["provisional_singleton_ids"],
                "duplicate_risk_orphan_ids": [value["entity_id"] for value in view["duplicate_risk_orphans"]],
                "cannot_link_count": len(view["identity_constraints"]["cannot_link"]),
                "identity_change_reasons": [
                    "new_space_singleton" if view["provisional_singleton_ids"] else "",
                    "duplicate_risk_orphan" if view["duplicate_risk_orphans"] else "",
                    "independent_observation_support" if view["supported_entity_ids"] else "",
                ],
            })
        final_view = materialize_query_view(task_ir, {"records": ledger})
        reversed_view = materialize_query_view(task_ir, {"records": list(reversed(ledger))})

    later_unmatched_definite = [
        {
            "entity_id": value["object_id"],
            "key": value["deterministic_entity_key"],
            "evidence_count": len(value.get("evidence", ())),
        }
        for value in final_view["objects"]
        if value["class_label"] == "pillow"
        and value.get("status") == "confirmed"
        and len(value.get("evidence", ())) < 2
        and value.get("entity_lifecycle") != "SUPPORTED_ENTITY"
    ]
    report = {
        "schema_version": "phase3a_cutover_recorded_replay_v1",
        "source_session": str(session),
        "question": question,
        "model_calls": 0,
        "recorded_acquisition_count": len(acquisitions),
        "relifted_acquisition_count": len(rows),
        "skipped_acquisitions": skipped,
        "rows": rows,
        "checks": {
            "a6_escaped_fragment_fused": a6_fragment_fused,
            "same_view_distinctness_materialized": bool(final_view["identity_constraints"]["cannot_link"]),
            "later_unmatched_not_automatically_definite": not later_unmatched_definite,
            "qwen_unavailable_identity_noop": "covered_by_I6",
            "relation_recompute_identity_noop": "covered_by_I7",
            "order_independent_rematerialization": _signature(final_view) == _signature(reversed_view),
        },
        "later_unmatched_definite_violations": later_unmatched_definite,
        "status": "PASS",
    }
    if not all(value is True or isinstance(value, str) for value in report["checks"].values()):
        report["status"] = "FAIL"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"], "rows": len(rows)}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.session.resolve(), args.output.resolve(), args.calibration.resolve())
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
