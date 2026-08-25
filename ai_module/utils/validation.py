"""Validation utilities extracted from live_model_chain.

Functions for request validation, JSON writing, and competition geometry gating.
"""

from __future__ import annotations

import json
from pathlib import Path


_FORBIDDEN_REQUEST_FIELDS = frozenset({
    "answer",
    "expected_answer",
    "ground_truth",
    "answer_key",
    "correct_answer",
    "reference_answer",
    "numerical_answer",
    "predicted_answer",
    "model_answer",
    "best_supported_hypothesis",
})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_episode_request(request: dict, output: Path) -> dict:
    """Reject evaluator truth fields and cross-episode artifact references."""
    forbidden: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized in _FORBIDDEN_REQUEST_FIELDS:
                    forbidden.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(request, "")
    if forbidden:
        raise ValueError("forbidden_answer_input:" + ",".join(sorted(forbidden)))

    episode_id = str(request.get("episode_id", "")).strip()
    if not episode_id:
        raise ValueError("episode_id_missing")
    artifact_root = output.parent.resolve()
    checked_paths = 0
    for keyframe in request.get("reconstruction_keyframes", ()):
        if str(keyframe.get("episode_id", "")) != episode_id:
            raise ValueError("cross_episode_keyframe_rejected")
        for field in (
            "panorama_path",
            "state_estimation_path",
            "sensor_scan_path",
            "registered_scan_path",
            "summary_path",
            "semantic_observations_path",
        ):
            raw = keyframe.get(field)
            if not raw:
                continue
            resolved = Path(str(raw)).resolve()
            if resolved != artifact_root and artifact_root not in resolved.parents:
                raise ValueError(f"keyframe_path_outside_episode_artifacts:{field}")
            checked_paths += 1
    scene_memory_path = Path(str(request.get("scene_memory_path", ""))).resolve()
    if scene_memory_path != artifact_root / episode_id / "episode_scene_memory.json":
        raise ValueError("scene_memory_not_owned_by_episode")
    return {
        "status": "accepted",
        "question_source": "/challenge_question",
        "sensor_sources": list(
            request.get("competition_geometry", {}).get("allowed_competition_inputs", ())
        ),
        "forbidden_answer_fields_present": False,
        "historical_run_discovery": False,
        "episode_id": episode_id,
        "episode_keyframe_count": len(request.get("reconstruction_keyframes", ())),
        "episode_artifact_paths_checked": checked_paths,
        "scene_memory_scope": "current_episode_only",
        "numerical_hypothesis_input_present": False,
    }


def competition_geometry_gate(payload: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []
    blocking_reasons = []
    maximum_age = float(config["max_age_seconds"])
    maximum_offset = float(config["max_camera_lidar_offset_seconds"])
    checks = (
        ("sensor_scan", "sensor_scan", config["sensor_scan_required"]),
        ("registered_scan", "registered_scan", config["registered_scan_required"]),
        ("state_estimation", "state_estimation", config["state_estimation_required"]),
    )
    for label, prefix, required in checks:
        if not required:
            continue
        path = payload.get(f"{prefix}_path")
        age = payload.get(f"{prefix}_age_seconds")
        if not path or not Path(path).is_file():
            reason = f"{label}_artifact_missing"
            reasons.append(reason)
            blocking_reasons.append(reason)
        if age is None or float(age) > maximum_age:
            reasons.append(f"{label}_stale")
    if int(payload.get("sensor_scan_point_count", 0)) <= 0:
        reasons.append("sensor_scan_empty")
        blocking_reasons.append("sensor_scan_empty")
    if int(payload.get("registered_scan_point_count", 0)) <= 0:
        reasons.append("registered_scan_empty")
    offset = payload.get("camera_sensor_scan_offset_seconds")
    if offset is None or float(offset) > maximum_offset:
        reasons.append("camera_sensor_scan_out_of_sync")
    if payload.get("sensor_scan_frame") != "sensor_at_scan":
        reasons.append("sensor_scan_frame_invalid")
        blocking_reasons.append("sensor_scan_frame_invalid")
    if (
        int(payload.get("registered_scan_point_count", 0)) > 0
        and payload.get("registered_scan_frame") != "map"
    ):
        reasons.append("registered_scan_frame_invalid")
        blocking_reasons.append("registered_scan_frame_invalid")
    if payload.get("state_estimation_frame") != "map":
        reasons.append("state_estimation_frame_invalid")
        blocking_reasons.append("state_estimation_frame_invalid")
    if payload.get("state_estimation_child_frame") != "sensor":
        reasons.append("state_estimation_child_frame_invalid")
        blocking_reasons.append("state_estimation_child_frame_invalid")
    # Sensor age and camera/LiDAR offset remain visible diagnostics.  The lift
    # consumes timestamped source scans and state for each observation, so a
    # fixed wall-clock threshold must not suppress otherwise valid geometry.
    return not blocking_reasons, reasons
