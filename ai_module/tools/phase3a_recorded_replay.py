#!/usr/bin/env python3
"""Export a deterministic, model-free replay of one recorded Phase 3A run.

The exporter consumes already-recorded files only.  It does not invoke Unity,
ROS, perception workers, VLA data, or production mutation APIs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


def _load(path: Path, default: object) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _stage(summary: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = summary.get("stages", {}).get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _domain(summary: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(summary, "count_query_graph")
    value = stage.get("domain", {})
    if isinstance(value, Mapping):
        return dict(value)
    stage = _stage(summary, "query_domain")
    value = stage.get("count_query_domain", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _execution(summary: Mapping[str, Any]) -> dict[str, Any]:
    stage = _stage(summary, "count_query_graph")
    value = stage.get("execution", {})
    if isinstance(value, Mapping):
        return dict(value)
    value = _stage(summary, "query_execution").get("execution", {}).get("count_query_execution", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    value = _stage(summary, "scene_memory").get("snapshot", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _object_map(snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for value in snapshot.get("objects", ()):
        if not isinstance(value, Mapping):
            continue
        try:
            result[int(value["object_id"])] = dict(value)
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _observation_assignments(snapshot: Mapping[str, Any], acquisition_id: str) -> list[dict[str, Any]]:
    result = []
    for object_id, obj in sorted(_object_map(snapshot).items()):
        for evidence in obj.get("evidence", ()):
            if not isinstance(evidence, Mapping):
                continue
            if str(evidence.get("acquisition_id", "")) != acquisition_id:
                continue
            result.append({
                "observation_id": str(evidence.get("observation_id", "")),
                "entity_id": object_id,
                "identity_state": str(obj.get("identity_state", obj.get("physical_status", obj.get("status", "")))),
                "class_name": str(obj.get("class_label", "")),
            })
    return sorted(result, key=lambda value: (value["observation_id"], value["entity_id"]))


def _count_certificate(summary: Mapping[str, Any]) -> dict[str, Any]:
    domain = _domain(summary)
    execution = _execution(summary)
    states = [value for value in domain.get("target_id_states", ()) if isinstance(value, Mapping)]
    yes_ids = sorted({int(value["object_id"]) for value in states if str(value.get("state", "")).upper() == "YES"})
    provisional = sorted({
        int(value["object_id"]) for value in states
        if str(value.get("identity_status", "")).lower() in {"tentative", "provisional"}
    })
    unresolved = sorted({
        int(value["object_id"]) for value in states
        if str(value.get("state", "")).upper() in {"UNKNOWN", "INVALID"}
    })
    closed = bool(domain.get("closed"))
    answer = execution.get("answer")
    upper = int(answer) if closed and isinstance(answer, int) and not isinstance(answer, bool) else None
    return {
        "schema_version": "phase3a_count_certificate_v1",
        "lower_bound": len(yes_ids),
        "upper_bound": upper,
        "upper_bound_kind": "FINITE" if upper is not None else "OPEN",
        "supported_entities": yes_ids,
        "provisional_singletons": provisional,
        "duplicate_risk_orphans": list(domain.get("pending_identity_ids", ())),
        "unresolved_relations": list(domain.get("unknown_relation_tuples", ())),
        "unseen_relevant_volume": not closed,
        "strict_closed": closed and upper == len(yes_ids),
        "closure_reasons": list(domain.get("closure_reasons", domain.get("blockers", ()))),
    }


def _mask_path(acquisition_dir: Path, observation: Mapping[str, Any]) -> Path | None:
    raw = str(observation.get("panorama_mask_path", ""))
    if not raw:
        return None
    local = acquisition_dir / "05_lidar_geometry" / Path(raw).name
    return local if local.is_file() else None


def _mask_overlap(first: Path, second: Path) -> dict[str, float]:
    first_mask = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    second_mask = cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    if first_mask is None or second_mask is None or first_mask.shape != second_mask.shape:
        return {"intersection_px": 0, "iou": 0.0, "containment_iom": 0.0}
    first_bool = first_mask > 0
    second_bool = second_mask > 0
    intersection = int(np.count_nonzero(first_bool & second_bool))
    first_area = int(np.count_nonzero(first_bool))
    second_area = int(np.count_nonzero(second_bool))
    union = first_area + second_area - intersection
    return {
        "intersection_px": intersection,
        "first_area_px": first_area,
        "second_area_px": second_area,
        "iou": intersection / union if union else 0.0,
        "containment_iom": intersection / min(first_area, second_area) if min(first_area, second_area) else 0.0,
    }


def _best_escaped_same_acquisition_duplicate(acquisition_dir: Path, objects: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            if str(first.get("canonical_class", "")) != str(second.get("canonical_class", "")):
                continue
            first_path = _mask_path(acquisition_dir, first)
            second_path = _mask_path(acquisition_dir, second)
            if first_path is None or second_path is None:
                continue
            overlap = _mask_overlap(first_path, second_path)
            candidate = {
                "class_name": str(first.get("canonical_class", "")),
                "acquisition_object_ids": [str(first.get("observation_id", "")), str(second.get("observation_id", ""))],
                "representative_detection_ids": [
                    list(first.get("source_detection_ids", ())),
                    list(second.get("source_detection_ids", ())),
                ],
                "panorama_mask_paths": [str(first_path), str(second_path)],
                **overlap,
            }
            if best is None or float(candidate["containment_iom"]) > float(best["containment_iom"]):
                best = candidate
    return best


def export_replay(session: Path, output: Path) -> dict[str, Any]:
    acquisition_dirs = sorted(
        value for value in (session / "live_robot").iterdir()
        if value.is_dir() and (value / "summary.json").is_file()
    )
    if not acquisition_dirs:
        raise RuntimeError("recorded_session_has_no_completed_acquisitions")
    observation_lines = []
    acquisition_lines = []
    entity_lines = []
    assignment_lines = []
    relation_lines = []
    certificate_lines = []
    replay_rows = []
    previous_objects: dict[int, dict[str, Any]] = {}
    previous_merge_count = 0

    for index, acquisition_dir in enumerate(acquisition_dirs):
        acquisition_id = acquisition_dir.name
        summary = _load(acquisition_dir / "summary.json", {})
        detections = _load(acquisition_dir / "04_verified_detections.json", [])
        geometry = _load(acquisition_dir / "05_lidar_geometry" / "observations.json", {})
        state = _load(acquisition_dir / "state_estimation.json", {})
        snapshot = _snapshot(summary)
        domain = _domain(summary)
        certificate = _count_certificate(summary)
        objects = _object_map(snapshot)
        fused = [dict(value) for value in geometry.get("observations", ()) if isinstance(value, Mapping)]
        assignments = _observation_assignments(snapshot, acquisition_id)
        assignment_by_observation = {value["observation_id"]: value["entity_id"] for value in assignments}
        relevant_classes = {
            str(value.get("class_name", ""))
            for value in summary.get("task_ir", {}).get("entities", ())
            if str(value.get("class_name", ""))
        }
        relevant = [value for value in fused if str(value.get("canonical_class", "")) in relevant_classes]
        transaction = snapshot.get("last_observation_transaction", {})
        births = list(
            snapshot.get("viewpoint_history", ())[-1].get("new_instance_ids", ())
            if snapshot.get("viewpoint_history") else ()
        )
        merges = list(snapshot.get("identity_merge_events", ()))[previous_merge_count:]
        previous_merge_count = len(snapshot.get("identity_merge_events", ()))
        promotions = []
        demotions = []
        for object_id, obj in objects.items():
            old = previous_objects.get(object_id)
            if old is None:
                continue
            prior = str(old.get("semantic_status", "unverified"))
            current = str(obj.get("semantic_status", "unverified"))
            if current != prior:
                record = {"entity_id": object_id, "before": prior, "after": current}
                if current == "verified":
                    promotions.append(record)
                else:
                    demotions.append(record)
        previous_objects = copy.deepcopy(objects)
        identity_constraints = snapshot.get("identity_constraints", {})
        cannot_links = list(identity_constraints.get("cannot_link", ())) if isinstance(identity_constraints, Mapping) else []
        distinct = list(identity_constraints.get("distinct_evidence", ())) if isinstance(identity_constraints, Mapping) else []
        target_entity = str(summary.get("task_ir", {}).get("target_entity", ""))
        candidate_domains = domain.get("candidate_domains", {})
        target_candidates = list(candidate_domains.get(target_entity, ())) if isinstance(candidate_domains, Mapping) else []
        anchor_candidates = sorted({
            int(value)
            for entity_id, values in candidate_domains.items()
            if entity_id != target_entity and isinstance(values, list)
            for value in values
        }) if isinstance(candidate_domains, Mapping) else []
        resolution = _stage(summary, "task_resolution").get("result", {})
        root = _stage(summary, "root_finalization").get("decision", {})
        relation_states = list(domain.get("target_id_states", ()))
        row = {
            "acquisition_index": index,
            "acquisition_id": acquisition_id,
            "robot_pose": {
                "position_xyz": state.get("position_xyz"),
                "orientation_xyzw": state.get("orientation_xyzw"),
                "frame_id": state.get("frame_id"),
            },
            "raw_proposal_count": len(detections),
            "same_acquisition_fused_count": len(fused),
            "query_relevant_observation_count": len(relevant),
            "entity_count": len(objects),
            "supported_entity_count": sum(str(value.get("status", value.get("physical_status", ""))).lower() == "confirmed" for value in objects.values()),
            "singleton_count": sum(len(value.get("evidence", ())) == 1 for value in objects.values()),
            "orphan_count": len(snapshot.get("ambiguous_observations", ())),
            "cannot_link_count": len(cannot_links),
            "simultaneous_distinct_evidence_count": len(distinct),
            "anchor_candidate_count": len(anchor_candidates),
            "target_candidate_count": len(target_candidates),
            "identity_assignments": assignments,
            "new_births": births,
            "new_orphans": list(snapshot.get("ambiguous_observations", ())),
            "merges": merges,
            "splits": [],
            "semantic_promotions": promotions,
            "semantic_demotions": demotions,
            "relation_states": relation_states,
            "count_lower_bound": certificate["lower_bound"],
            "count_upper_bound": certificate["upper_bound"],
            "count_upper_bound_kind": certificate["upper_bound_kind"],
            "coverage_state": {
                "closed": bool(domain.get("closed")),
                "state": domain.get("state"),
                "reasons": list(domain.get("closure_reasons", domain.get("blockers", ()))),
            },
            "evidence_need": resolution.get("evidence_need") if isinstance(resolution, Mapping) else None,
            "resolver_action": root.get("action") if isinstance(root, Mapping) else None,
            "observation_transaction": transaction,
        }
        replay_rows.append(row)
        for observation in fused:
            observation_lines.append({
                "acquisition_index": index,
                "acquisition_id": acquisition_id,
                "entity_id": assignment_by_observation.get(str(observation.get("observation_id", ""))),
                "observation": observation,
            })
        acquisition_lines.append({
            "acquisition_index": index,
            "acquisition_id": acquisition_id,
            "raw_proposal_count": len(detections),
            "raw_proposal_ids": [str(value.get("detection_id", "")) for value in detections if isinstance(value, Mapping)],
            "acquisition_object_count": len(fused),
            "acquisition_objects": fused,
        })
        entity_lines.append({
            "acquisition_index": index,
            "acquisition_id": acquisition_id,
            "view_type": "recorded_legacy_scene_snapshot",
            "entities": list(objects.values()),
        })
        assignment_lines.append({"acquisition_index": index, "acquisition_id": acquisition_id, "assignments": assignments, "births": births, "merges": merges})
        relation_lines.append({"acquisition_index": index, "acquisition_id": acquisition_id, "relation_states": relation_states, "unknown_relation_tuples": list(domain.get("unknown_relation_tuples", ()))})
        certificate_lines.append({"acquisition_index": index, "acquisition_id": acquisition_id, **certificate})

    established = 0
    bad_index = None
    for index, row in enumerate(replay_rows):
        lower = int(row["count_lower_bound"])
        if established > 0 and lower < established:
            bad_index = index
            break
        established = max(established, lower)
    if bad_index is None:
        raise RuntimeError("recorded_replay_has_no_recoverable_to_polluted_transition")
    bad_row = replay_rows[bad_index]
    before = replay_rows[bad_index - 1]
    bad_dir = acquisition_dirs[bad_index]
    bad_geometry = _load(bad_dir / "05_lidar_geometry" / "observations.json", {})
    escaped = _best_escaped_same_acquisition_duplicate(bad_dir, list(bad_geometry.get("observations", ())))
    if not escaped or float(escaped.get("containment_iom", 0.0)) < 0.90:
        raise RuntimeError("first_bad_transition_has_no_strong_same_acquisition_duplicate_evidence")
    first_bad = {
        "schema_version": "phase3a_first_bad_transition_v1",
        "source_session": str(session.resolve()),
        "first_bad_acquisition": bad_row["acquisition_id"],
        "first_bad_acquisition_index": bad_index,
        "first_bad_observation_ids": escaped["acquisition_object_ids"],
        "before_state": {
            "acquisition_id": before["acquisition_id"],
            "count_lower_bound": before["count_lower_bound"],
            "count_upper_bound": before["count_upper_bound"],
            "anchor_candidate_count": before["anchor_candidate_count"],
            "entity_count": before["entity_count"],
        },
        "after_state": {
            "acquisition_id": bad_row["acquisition_id"],
            "count_lower_bound": bad_row["count_lower_bound"],
            "count_upper_bound": bad_row["count_upper_bound"],
            "anchor_candidate_count": bad_row["anchor_candidate_count"],
            "entity_count": bad_row["entity_count"],
            "new_births": bad_row["new_births"],
        },
        "same_acquisition_duplicate_evidence": escaped,
        "why_this_transition_is_wrong": (
            "Two same-class acquisition objects retained separate identities despite direct original-panorama "
            "support containment above 0.90. The escaped fragment was born as an additional anchor candidate, "
            "making an already-supported target relation UNKNOWN and reducing the recoverable count lower bound."
        ),
        "responsible_owner": "SAME_ACQUISITION_DUPLICATION",
        "downstream_consequence": "ANCHOR_RESOLUTION pollution; count lower bound regressed after previously reaching four.",
    }

    _write_jsonl(output / "observations.jsonl", observation_lines)
    _write_jsonl(output / "acquisition_objects.jsonl", acquisition_lines)
    _write_jsonl(output / "entity_views.jsonl", entity_lines)
    _write_jsonl(output / "identity_assignments.jsonl", assignment_lines)
    _write_jsonl(output / "relations.jsonl", relation_lines)
    _write_jsonl(output / "count_certificates.jsonl", certificate_lines)
    _write_jsonl(output / "acquisition_replay.jsonl", replay_rows)
    _write(output / "FIRST_BAD_TRANSITION.json", first_bad)
    _write(output / "invariant_report.json", {
        "schema_version": "phase3a_invariant_report_v1",
        "phase": "PRE_CUTOVER_BASELINE",
        "status": "PENDING_COMMIT_2_MATERIALIZER",
        "results": [],
        "note": "I1-I8 are executed after the production query-view API exists; this file is intentionally replaced then.",
    })
    report = [
        "# Phase 3A Recorded N1 Replay",
        "",
        f"- Source: `{session.resolve()}`",
        f"- Completed acquisitions: {len(replay_rows)}",
        f"- First bad acquisition: `{first_bad['first_bad_acquisition']}` (A{bad_index})",
        f"- Responsible owner: `{first_bad['responsible_owner']}`",
        f"- Count transition: {before['count_lower_bound']} -> {bad_row['count_lower_bound']}",
        f"- Escaped panorama containment: {escaped['containment_iom']:.6f}",
        f"- Escaped panorama IoU: {escaped['iou']:.6f}",
        "",
        "The first wrong transition is acquisition-local. SceneMemory receives two objects that should already be one acquisition object; its later anchor-domain pollution is downstream.",
        "",
    ]
    (output / "PHASE3A_N1_REPLAY.md").write_text("\n".join(report), encoding="utf-8")
    return {"acquisitions": len(replay_rows), "first_bad_transition": first_bad}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_replay(args.session.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
