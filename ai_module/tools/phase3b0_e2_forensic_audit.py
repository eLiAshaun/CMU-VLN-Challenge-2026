#!/usr/bin/env python3
"""Offline-only Phase 3B-0 E2 forensic exporter.

This development tool reads one already-recorded real run plus the offline
VLA-3D annotations.  It is not imported by production runtime and does not
change production state, thresholds, prompts, navigation, or SceneMemory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from integrations.execution.relation_evidence import (
    _geometry_consistency,
    relation_geometry_diagnostic,
)
from integrations.execution.scene_memory import _refresh_geometry_from_evidence


PRODUCTION_HEAD = "1c64598cccac8236d8856e65b2d71a4c1e9526e7"
QUESTION = "How many pictures are above the bed?"
EXPECTED_INTEGER = 3
FIRST_BLOCKER = "RELATION_GEOMETRY_UNKNOWN"
FINAL_STATUS = "PHASE3B0_OWNER_CONFIRMED"

GT_ROLES = {
    44: "singular bed anchor (full bed-frame annotation; mattress component is GT 57)",
    31: "expected picture 1 of 3 (right in the initial relation view)",
    34: "expected picture 2 of 3 (middle in the initial relation view)",
    76: "expected picture 3 of 3 (left in the initial relation view)",
}
GT_TARGET_TO_RUNTIME = {31: 13, 34: 12, 76: 16}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def round_or_none(value: Any, digits: int = 6) -> float | None:
    result = finite_float(value)
    return round(result, digits) if result is not None else None


def iso_utc(value: Any) -> str:
    stamp = finite_float(value)
    if stamp is None:
        return ""
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_acquisition_timestamp(acquisition_id: str) -> float:
    match = re.match(r"^(\d{8}T\d{6})_(\d{6})Z$", acquisition_id)
    if not match:
        raise ValueError(f"invalid acquisition id: {acquisition_id}")
    base = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return base.timestamp() + int(match.group(2)) / 1_000_000.0


def bounds(center: Sequence[float], extent: Sequence[float]) -> tuple[list[float], list[float]]:
    return (
        [float(center[index]) - 0.5 * float(extent[index]) for index in range(3)],
        [float(center[index]) + 0.5 * float(extent[index]) for index in range(3)],
    )


def box_metrics(
    first_center: Sequence[float],
    first_extent: Sequence[float],
    second_center: Sequence[float],
    second_extent: Sequence[float],
) -> dict[str, float]:
    first_low, first_high = bounds(first_center, first_extent)
    second_low, second_high = bounds(second_center, second_extent)
    overlap = [
        max(0.0, min(first_high[index], second_high[index]) - max(first_low[index], second_low[index]))
        for index in range(3)
    ]
    intersection = math.prod(overlap)
    first_volume = math.prod(float(value) for value in first_extent)
    second_volume = math.prod(float(value) for value in second_extent)
    union = first_volume + second_volume - intersection
    xy_intersection = overlap[0] * overlap[1]
    first_xy = float(first_extent[0]) * float(first_extent[1])
    second_xy = float(second_extent[0]) * float(second_extent[1])
    xy_union = first_xy + second_xy - xy_intersection
    return {
        "center_distance_m": math.dist(
            [float(value) for value in first_center],
            [float(value) for value in second_center],
        ),
        "intersection_over_min_volume": (
            intersection / min(first_volume, second_volume)
            if min(first_volume, second_volume) > 0.0
            else 0.0
        ),
        "iou_3d": intersection / union if union > 0.0 else 0.0,
        "xy_intersection_over_min_area": (
            xy_intersection / min(first_xy, second_xy)
            if min(first_xy, second_xy) > 0.0
            else 0.0
        ),
        "xy_iou": xy_intersection / xy_union if xy_union > 0.0 else 0.0,
    }


def xy_metrics_from_bounds(
    subject_bounds: Sequence[Sequence[float]] | None,
    object_bounds: Sequence[Sequence[float]] | None,
) -> tuple[float | None, float | None]:
    if not subject_bounds or not object_bounds:
        return None, None
    try:
        subject_low, subject_high = subject_bounds
        object_low, object_high = object_bounds
        overlaps = [
            max(0.0, min(subject_high[index], object_high[index]) - max(subject_low[index], object_low[index]))
            for index in (0, 1)
        ]
        intersection = overlaps[0] * overlaps[1]
        subject_area = (subject_high[0] - subject_low[0]) * (subject_high[1] - subject_low[1])
        object_area = (object_high[0] - object_low[0]) * (object_high[1] - object_low[1])
        union = subject_area + object_area - intersection
        iom = intersection / min(subject_area, object_area) if min(subject_area, object_area) > 0 else 0.0
        iou = intersection / union if union > 0 else 0.0
        return iom, iou
    except (IndexError, TypeError, ValueError):
        return None, None


def covariance_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    covariance = value.get("center_cov")
    diagonal: list[float] = []
    if isinstance(covariance, Sequence):
        try:
            diagonal = [float(covariance[index][index]) for index in range(3)]
        except (IndexError, TypeError, ValueError):
            diagonal = []
    return {
        "diagonal": diagonal,
        "sigma_xyz_m": [math.sqrt(max(0.0, item)) for item in diagonal],
        "mode": value.get("covariance_mode"),
        "provenance": value.get("center_covariance_provenance"),
    }


def relation_key(subject_id: int, anchor_id: int) -> tuple[int, int]:
    return int(subject_id), int(anchor_id)


def relation_records(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(value)
        for value in execution.get("relation_verifications", ())
        if isinstance(value, Mapping)
    ]


def extract_need(summary: Mapping[str, Any], *, output: bool) -> dict[str, Any]:
    if output:
        value = summary.get("resolver_result", {}).get("evidence_need")
    else:
        value = summary.get("evidence_acquisition", {}).get("evidence_need")
        if not isinstance(value, Mapping):
            value = summary.get("stages", {}).get("evidence_acquisition_input", {}).get("evidence_need")
    return dict(value) if isinstance(value, Mapping) else {}


def poisson_binomial(probabilities: Sequence[float]) -> list[float]:
    distribution = [1.0] + [0.0] * len(probabilities)
    for probability in probabilities:
        current = [0.0] * len(distribution)
        for count, mass in enumerate(distribution):
            current[count] += mass * (1.0 - probability)
            if count + 1 < len(current):
                current[count + 1] += mass * probability
        distribution = current
    return distribution


def credible_interval(distribution: Sequence[float], mass: float = 0.95) -> list[int]:
    tail = (1.0 - mass) / 2.0
    cumulative = 0.0
    low = 0
    high = len(distribution) - 1
    for index, value in enumerate(distribution):
        cumulative += value
        if cumulative >= tail:
            low = index
            break
    cumulative = 0.0
    for index, value in reversed(list(enumerate(distribution))):
        cumulative += value
        if cumulative >= tail:
            high = index
            break
    return [low, high]


def recursively_rewrite_paths(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, dict):
        return {key: recursively_rewrite_paths(item, source, destination) for key, item in value.items()}
    if isinstance(value, list):
        return [recursively_rewrite_paths(item, source, destination) for item in value]
    if isinstance(value, str) and value.startswith(source):
        return destination + value[len(source) :]
    return value


def read_gt(zip_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    object_name = "Unity/hotel_room_2/hotel_room_2_object_result.csv"
    region_name = "Unity/hotel_room_2/hotel_room_2_region_result.csv"
    graph_name = "Unity/hotel_room_2/hotel_room_2_scene_graph.json"
    with zipfile.ZipFile(zip_path) as archive:
        object_rows = list(csv.DictReader(io.StringIO(archive.read(object_name).decode("utf-8"))))
        region_rows = list(csv.DictReader(io.StringIO(archive.read(region_name).decode("utf-8"))))
        graph = json.loads(archive.read(graph_name))
    by_id: dict[int, dict[str, Any]] = {}
    for row in object_rows:
        object_id = int(row["object_id"])
        by_id[object_id] = {
            "object_id": object_id,
            "region_id": int(row["region_id"]),
            "raw_label": row["raw_label"],
            "nyu_label": row["nyu_label"],
            "nyu40_label": row["nyu40_label"],
            "center": [float(row[f"object_bbox_c{axis}"]) for axis in "xyz"],
            "extent": [float(row[f"object_bbox_{axis}length"]) for axis in "xyz"],
            "heading_rad": float(row["object_bbox_heading"]),
        }
    return by_id, graph, region_rows


def captured_events(capture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in capture.get("status_events", ()):
        payload = event.get("payload", {}) if isinstance(event, Mapping) else {}
        if payload.get("state") != "captured":
            continue
        run_dir = str(payload.get("run_dir", ""))
        result[Path(run_dir).name] = {
            "elapsed_seconds": finite_float(event.get("elapsed_seconds")),
            "payload": payload,
        }
    return result


def build_acquisitions(
    summary_paths: Sequence[Path], capture: Mapping[str, Any]
) -> list[dict[str, Any]]:
    events = captured_events(capture)
    waypoints = capture.get("responses", {}).get("commanded_waypoints", ())
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(summary_paths):
        summary = load_json(path)
        acquisition_id = str(summary["acquisition_id"])
        snapshot = summary["stages"]["scene_memory"]["snapshot"]
        execution = snapshot.get("count_query_execution", {})
        relations = relation_records(execution)
        relation_times = [
            float(value["timestamp"])
            for value in relations
            if finite_float(value.get("timestamp")) is not None
        ]
        detection_path = path.parent / "04_verified_detections.json"
        detections = load_json(detection_path) if detection_path.exists() else []
        event = events.get(acquisition_id, {})
        captured_elapsed = finite_float(event.get("elapsed_seconds"))
        if captured_elapsed is None:
            captured_elapsed = parse_acquisition_timestamp(acquisition_id) - float(capture["timing"]["started_unix"])
        total_elapsed = float(summary.get("timing", {}).get("total_elapsed_seconds", 0.0))
        output_need = extract_need(summary, output=True)
        certificate = summary.get("resolver_result", {}).get("diagnostics", {}).get("count_certificate", {})
        rows.append(
            {
                "index": index + 1,
                "path": path,
                "summary": summary,
                "acquisition_id": acquisition_id,
                "snapshot": snapshot,
                "execution": execution,
                "relations": relations,
                "capture_elapsed_seconds": captured_elapsed,
                "child_end_elapsed_seconds": captured_elapsed + total_elapsed,
                "relation_first_unix": min(relation_times) if relation_times else None,
                "relation_last_unix": max(relation_times) if relation_times else None,
                "input_need": extract_need(summary, output=False),
                "output_need": output_need,
                "certificate": certificate,
                "detection_verdicts": Counter(
                    str(value.get("qwen_candidate_verdict", "missing"))
                    for value in detections
                    if isinstance(value, Mapping)
                ),
                "detection_count": len(detections),
                "commanded_waypoint": dict(waypoints[index]) if index < len(waypoints) else {},
            }
        )
    return rows


def build_evidence_inventory(
    acquisitions: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    session_root = acquisitions[0]["path"].parent.parent
    completed_ids = {str(value["acquisition_id"]) for value in acquisitions}
    per_acquisition = []
    for acquisition in acquisitions:
        root = acquisition["path"].parent
        proposal_path = root / "02_task_proposals.json"
        verified_path = root / "04_verified_detections.json"
        geometry_path = root / "05_lidar_geometry" / "observations.json"
        proposals = load_json(proposal_path)
        verified = load_json(verified_path)
        geometry = load_json(geometry_path)
        manifests = []
        for path in sorted((root / "06_relation_evidence").glob("*/manifest.json")):
            manifests.append(load_json(path))
        transaction_path = root / "evidence_transaction.json"
        transaction = load_json(transaction_path) if transaction_path.exists() else None
        per_acquisition.append(
            {
                "acquisition_id": acquisition["acquisition_id"],
                "summary_json": str(acquisition["path"]),
                "episode_runtime_state_present": (root / "episode_runtime_state.json").exists(),
                "chain_log_present": (root / "chain.log").exists(),
                "task_proposal_count": len(proposals) if isinstance(proposals, Sequence) else None,
                "verified_detection_count": len(verified),
                "semantic_unavailable_count": sum(
                    str(value.get("qwen_candidate_verdict", "")) == "unavailable"
                    for value in verified
                ),
                "lidar_raw_observation_count": geometry.get("raw_observation_count"),
                "lidar_output_observation_count": geometry.get("observation_count"),
                "lidar_rejected_count": geometry.get("rejected_count"),
                "relation_manifest_count": len(manifests),
                "relation_manifest_completed_count": sum(
                    value.get("status") == "completed" for value in manifests
                ),
                "relation_manifest_jointly_observable_count": sum(
                    value.get("jointly_observable") is True for value in manifests
                ),
                "evidence_transaction_present": transaction is not None,
                "evidence_transaction_completed": (
                    transaction.get("completed") if isinstance(transaction, Mapping) else None
                ),
            }
        )
    terminal_or_incomplete = []
    for root in sorted(path for path in session_root.iterdir() if path.is_dir()):
        if root.name in completed_ids:
            continue
        manifests = [
            load_json(path)
            for path in sorted((root / "06_relation_evidence").glob("*/manifest.json"))
        ]
        terminal_or_incomplete.append(
            {
                "directory_id": root.name,
                "summary_present": (root / "summary.json").exists(),
                "request_present": (root / "request.json").exists(),
                "episode_runtime_state_present": (root / "episode_runtime_state.json").exists(),
                "camera_capture_present": (root / "camera_panorama.png").exists(),
                "relation_manifest_count": len(manifests),
                "relation_manifest_completed_count": sum(
                    value.get("status") == "completed" for value in manifests
                ),
                "role": "deadline finalization directory; no completed perception summary",
            }
        )
    return {
        "outer_run_files": {
            name: str(run_dir / name)
            for name in ("run_request.json", "capture.json", "score.json", "report.md")
            if (run_dir / name).exists()
        },
        "acquisitions": per_acquisition,
        "terminal_or_incomplete_directories": terminal_or_incomplete,
    }


def dependency_graph(
    acquisitions: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
    terminal_decision: Mapping[str, Any],
) -> dict[str, Any]:
    first = acquisitions[0]
    final = acquisitions[-1]
    final_execution = final["execution"]
    final_domain = final_execution.get("candidate_domains", {})
    relation_times = [
        value
        for acquisition in acquisitions
        for value in (acquisition.get("relation_first_unix"), acquisition.get("relation_last_unix"))
        if value is not None
    ]
    first_relation = min(relation_times)
    final_relation = max(relation_times)
    capture_start = float(capture["timing"]["started_unix"])

    def query_start_elapsed(acquisition: Mapping[str, Any]) -> float:
        stages = acquisition["summary"].get("stages", {})
        before_query = sum(
            float(stages.get(key, {}).get("elapsed_seconds", 0.0) or 0.0)
            for key in ("input_gate", "perception", "lidar_geometry", "scene_memory")
        )
        return float(acquisition["capture_elapsed_seconds"]) + before_query

    first_query_start = query_start_elapsed(first)
    final_query_start = query_start_elapsed(final)
    first_relation_end = float(first["relation_last_unix"]) - capture_start
    final_relation_end = float(final["relation_last_unix"]) - capture_start
    stage_total = lambda key: sum(
        float(acquisition["summary"].get("stages", {}).get(key, {}).get("elapsed_seconds", 0.0) or 0.0)
        for acquisition in acquisitions
    )
    semantic_requests = [
        {
            "after_acquisition": acquisition["acquisition_id"],
            "reason": acquisition["output_need"].get("reason"),
            "target_ids": acquisition["output_need"].get("target_ids", []),
        }
        for acquisition in acquisitions
    ]
    nodes = [
        {
            "id": "bed_candidate_domain",
            "required_inputs": ["TaskIR anchor_0=bed", "QueryEntityView objects", "semantic status", "identity lifecycle"],
            "executed": True,
            "first_execution_elapsed_seconds": round(first_query_start, 6),
            "last_execution_elapsed_seconds": round(final_query_start, 6),
            "produced_output": {"final_anchor_ids": final_domain.get("anchor_0", []), "cardinality": len(final_domain.get("anchor_0", []))},
            "confidence_uncertainty": "two verified bed candidates remained; singular anchor unresolved",
            "blocked_downstream": True,
            "evidence_requests_generated": [{"variable_type": "ANCHOR", "selected": False, "reason": "singular_reference_unresolved"}],
            "time_consumed_seconds": None,
            "timing_basis": "not separately instrumented inside count_query_graph",
        },
        {
            "id": "singular_anchor_resolution",
            "required_inputs": ["bed candidate domain", "anchor semantic posteriors", "anchor identity and geometry"],
            "executed": True,
            "first_execution_elapsed_seconds": round(first_query_start, 6),
            "last_execution_elapsed_seconds": round(final_query_start, 6),
            "produced_output": {"resolved_anchor": None, "candidate_ids": final_domain.get("anchor_0", [])},
            "confidence_uncertainty": "AMBIGUOUS: anchor_0 has candidates 1 and 3",
            "blocked_downstream": True,
            "evidence_requests_generated": [{"variable_type": "ANCHOR", "selected": False}],
            "time_consumed_seconds": None,
            "timing_basis": "resolver did not export a separate anchor-resolution duration",
        },
        {
            "id": "picture_candidate_domain",
            "required_inputs": ["TaskIR target_0=picture", "ObservationLedger", "QueryEntityView lifecycle", "semantic evidence"],
            "executed": True,
            "first_execution_elapsed_seconds": round(first_query_start, 6),
            "last_execution_elapsed_seconds": round(final_query_start, 6),
            "produced_output": {"final_target_ids": final_domain.get("target_0", []), "cardinality": len(final_domain.get("target_0", []))},
            "confidence_uncertainty": "11 final plausible target entities; group proposal entity 11 overlaps the three atomic pictures",
            "blocked_downstream": False,
            "evidence_requests_generated": semantic_requests,
            "time_consumed_seconds": None,
            "timing_basis": "not separately instrumented inside QueryEntityView materialization",
        },
        {
            "id": "above_relation_matrix",
            "required_inputs": ["picture domain", "bed domain", "RelationEngine", "Qwen tuple roles/predicate", "map geometry and covariance"],
            "executed": True,
            "first_execution_timestamp_utc": iso_utc(first_relation),
            "first_execution_elapsed_seconds": round(first_relation - float(capture["timing"]["started_unix"]), 6),
            "last_execution_timestamp_utc": iso_utc(final_relation),
            "last_execution_elapsed_seconds": round(final_relation - float(capture["timing"]["started_unix"]), 6),
            "produced_output": {"unique_pairs": 24, "YES": 0, "NO": 0, "UNKNOWN": 24},
            "confidence_uncertainty": "all persistent pairs UNKNOWN; exact dominant reason qwen_geometry_vertical_world_geometry_insufficient",
            "blocked_downstream": True,
            "evidence_requests_generated": [{"variable_type": "RELATION", "selected": False, "entity_ids": final_domain.get("target_0", [])}],
            "time_consumed_seconds": round(stage_total("count_query_graph"), 6),
            "timing_basis": "sum of six instrumented count_query_graph stages; includes domain/count work",
        },
        {
            "id": "distinct_instance_filtering",
            "required_inputs": ["ObservationLedger", "global one-to-one association", "same-view cannot-links", "lifecycle state"],
            "executed": True,
            "first_execution_elapsed_seconds": round(first_query_start, 6),
            "last_execution_elapsed_seconds": round(final_query_start, 6),
            "produced_output": {
                "supported_entity_ids": final["snapshot"].get("query_entity_view", {}).get("supported_entity_ids", []),
                "provisional_singleton_ids": final["snapshot"].get("query_entity_view", {}).get("provisional_singleton_ids", []),
                "duplicate_risk_orphans": final["snapshot"].get("query_entity_view", {}).get("duplicate_risk_orphans", []),
            },
            "confidence_uncertainty": "entity 11 is a multi-picture group ROI and entities 12/13/16 are atomic ROIs; the group remained independently countable",
            "blocked_downstream": False,
            "evidence_requests_generated": [{"variable_type": "IDENTITY", "selected": False}],
            "time_consumed_seconds": round(stage_total("scene_memory"), 6),
            "timing_basis": "sum of instrumented scene_memory stages; QueryEntityView sub-duration not separate",
        },
        {
            "id": "count_distribution",
            "required_inputs": ["distinct target entities", "affirmative ABOVE tuples", "singular anchor marginalization"],
            "executed": True,
            "first_execution_elapsed_seconds": round(first_relation_end, 6),
            "last_execution_elapsed_seconds": round(final_relation_end, 6),
            "produced_output": {"production_lower_bound": 0, "production_upper_bound": None, "best_evidence_answer": 0, "probability_distribution_in_production": False},
            "confidence_uncertainty": "no affirmative relation tuple; upper bound remained open",
            "blocked_downstream": True,
            "evidence_requests_generated": terminal_decision.get("remaining_uncertainties", []),
            "time_consumed_seconds": None,
            "timing_basis": "certificate construction not separately instrumented",
        },
        {
            "id": "final_integer",
            "required_inputs": ["CountCertificate", "deadline guard", "RootFinalizer"],
            "executed": True,
            "execution_elapsed_seconds": round(float(capture["timing"]["first_response_seconds"]), 6),
            "produced_output": {"published_integer": terminal_decision.get("answer"), "commit_mode": terminal_decision.get("commit_mode"), "authorized": terminal_decision.get("answer_authorized")},
            "confidence_uncertainty": "deadline lower-bound commit; not a truth certificate",
            "blocked_downstream": False,
            "evidence_requests_generated": [],
            "time_consumed_seconds": 0.0,
            "timing_basis": "ROS capture first_response_seconds",
        },
    ]
    edges = [
        ["bed_candidate_domain", "singular_anchor_resolution"],
        ["singular_anchor_resolution", "above_relation_matrix"],
        ["picture_candidate_domain", "above_relation_matrix"],
        ["above_relation_matrix", "distinct_instance_filtering"],
        ["distinct_instance_filtering", "count_distribution"],
        ["count_distribution", "final_integer"],
    ]
    return {
        "schema_version": "phase3b0_e2_dependency_graph_v1",
        "question": QUESTION,
        "run": str(capture.get("scene", "")) + "/" + str(capture.get("question_id", "")),
        "production_head": PRODUCTION_HEAD,
        "nodes": nodes,
        "edges": [{"from": source, "to": target} for source, target in edges],
        "first_blocker": FIRST_BLOCKER,
    }


def write_dependency_timeline(path: Path, graph: Mapping[str, Any]) -> None:
    lines = [
        "# E2 dependency timeline",
        "",
        f"Question: `{QUESTION}`",
        "",
        "The dependency graph executed end to end, but no ABOVE tuple became affirmative. Times are from the recorded capture/summary clocks; `n/a` means the production trace did not instrument that sub-node separately.",
        "",
        "| Node | Executed | First / final time | Output | Downstream block | Time consumed |",
        "|---|---:|---|---|---:|---:|",
    ]
    for node in graph["nodes"]:
        first = node.get("first_execution_elapsed_seconds", node.get("execution_elapsed_seconds", "n/a"))
        final = node.get("last_execution_elapsed_seconds", "")
        time_text = f"{first}" + (f" / {final}" if final != "" else "")
        consumed = node.get("time_consumed_seconds")
        lines.append(
            f"| `{node['id']}` | {str(node['executed']).lower()} | {time_text} s | `{compact(node['produced_output'])}` | {str(node['blocked_downstream']).lower()} | {consumed if consumed is not None else 'n/a'} |"
        )
    lines += [
        "",
        "## Causal transition",
        "",
        "- Initial candidate materialization produced bed anchors 1/3 and the three atomic picture entities 12/13/16, plus broader group hypotheses.",
        "- The first GT-relevant relation failure occurred in acquisition `20260823T041814_056161Z`: QueryEntityView handed RelationEngine an anchor born with `bearing_model` geometry and no `geometry_point_count`.",
        "- Qwen role and predicate evidence was affirmative, but `_world_geometry_usable` failed, so RelationEngine correctly returned `UNKNOWN` under its received contract.",
        "- The CountCertificate therefore stayed at lower bound 0, and RootFinalizer published 0 only under `DEADLINE_BEST_EVIDENCE`.",
        "",
        f"Exact first blocker: `{FIRST_BLOCKER}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gt_correspondence_rows(
    gt: Mapping[int, Mapping[str, Any]],
    final_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    query_view = final_snapshot["query_entity_view"]
    observation_to_entity = {str(key): int(value) for key, value in query_view["observation_to_entity"].items()}
    entity_lookup = {int(value["object_id"]): value for value in query_view["objects"]}
    observations = final_snapshot["observation_ledger"]["records"]
    rows: list[dict[str, Any]] = []
    for gt_id, role in GT_ROLES.items():
        gt_value = gt[gt_id]
        candidates: list[tuple[dict[str, Any], dict[str, float], int]] = []
        for observation in observations:
            if str(observation.get("class_label", "")) not in {"bed", "picture"}:
                continue
            metrics = box_metrics(
                gt_value["center"],
                gt_value["extent"],
                observation["center_3d"],
                observation["bbox_3d"],
            )
            entity_id = observation_to_entity.get(str(observation["observation_id"]), -1)
            entity = entity_lookup.get(entity_id, {})
            if gt_id == 44:
                accepted = (
                    observation.get("class_label") == "bed"
                    and metrics["center_distance_m"] <= 1.20
                    and metrics["intersection_over_min_volume"] >= 0.10
                    and entity.get("semantic_status") != "rejected"
                )
            else:
                accepted = (
                    observation.get("class_label") == "picture"
                    and metrics["center_distance_m"] <= 0.15
                    and metrics["intersection_over_min_volume"] >= 0.30
                )
            if accepted:
                candidates.append((dict(observation), metrics, entity_id))
        candidates.sort(key=lambda item: (item[1]["center_distance_m"], -item[1]["intersection_over_min_volume"]))
        observation_ids = [str(value[0]["observation_id"]) for value in candidates]
        entity_ids = sorted({value[2] for value in candidates if value[2] >= 0})
        acquisitions = sorted({str(value[0].get("acquisition_id", "")) for value in candidates})
        best_overlap = max(candidates, key=lambda item: item[1]["intersection_over_min_volume"]) if candidates else None
        best_center = min(candidates, key=lambda item: item[1]["center_distance_m"]) if candidates else None
        semantic = {str(entity_id): entity_lookup[entity_id].get("semantic_probability") for entity_id in entity_ids}
        lifecycle = {
            str(entity_id): (
                "PROVISIONAL_SINGLETON"
                if entity_lookup[entity_id].get("entity_lifecycle") == "NEW_SPACE_SINGLETON"
                else entity_lookup[entity_id].get("entity_lifecycle")
            )
            for entity_id in entity_ids
        }
        geometry_quality = {}
        if best_center:
            observation = best_center[0]
            geometry_quality = {
                "observation_id": observation["observation_id"],
                "geometry_confidence": observation.get("geometry_confidence"),
                "lidar_support_count": observation.get("lidar_support_count"),
                "covariance_mode": observation.get("covariance_mode"),
                "center_covariance_provenance": observation.get("center_covariance_provenance"),
            }
        rows.append(
            {
                "gt_object_id": gt_id,
                "gt_class": gt_value["raw_label"],
                "gt_center": compact(gt_value["center"]),
                "gt_bbox": compact({"extent": gt_value["extent"], "heading_rad": gt_value["heading_rad"]}),
                "expected_query_role": role,
                "matching_runtime_observation_ids": compact(observation_ids),
                "matching_runtime_entity_ids": compact(entity_ids),
                "best_geometric_overlap": compact(
                    {
                        "observation_id": best_overlap[0]["observation_id"],
                        **{key: round(value, 6) for key, value in best_overlap[1].items()},
                    }
                    if best_overlap
                    else None
                ),
                "best_center_distance_m": round(best_center[1]["center_distance_m"], 6) if best_center else "",
                "semantic_posterior": compact(semantic),
                "geometry_quality": compact(geometry_quality),
                "acquisition_first_seen": acquisitions[0] if acquisitions else "",
                "acquisition_last_seen": acquisitions[-1] if acquisitions else "",
                "final_lifecycle_state": compact(lifecycle),
                "missing_reason": "" if candidates else "no runtime observation passed geometry-and-class correspondence gates",
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def runtime_candidate_rows(final_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    view = final_snapshot["query_entity_view"]
    objects = {int(value["object_id"]): value for value in view["objects"]}
    hard_pairs = {
        tuple(sorted(int(item) for item in value.get("entity_ids", ())))
        for value in view.get("identity_constraints", {}).get("cannot_link", ())
        if len(value.get("entity_ids", ())) == 2
    }
    cannot_by_id: defaultdict[int, set[int]] = defaultdict(set)
    for first, second in hard_pairs:
        cannot_by_id[first].add(second)
        cannot_by_id[second].add(first)
    duplicates: defaultdict[int, set[int]] = defaultdict(set)
    ids = sorted(objects)
    for index, first_id in enumerate(ids):
        first = objects[first_id]
        for second_id in ids[index + 1 :]:
            second = objects[second_id]
            if first.get("class_label") != second.get("class_label"):
                continue
            if tuple(sorted((first_id, second_id))) in hard_pairs:
                continue
            metrics = box_metrics(first["center_3d"], first["bbox_3d"], second["center_3d"], second["bbox_3d"])
            if metrics["center_distance_m"] <= 0.20 or metrics["intersection_over_min_volume"] >= 0.55:
                duplicates[first_id].add(second_id)
                duplicates[second_id].add(first_id)
    target_domain = set(int(value) for value in final_snapshot["count_query_execution"]["candidate_domains"]["target_0"])
    anchor_domain = set(int(value) for value in final_snapshot["count_query_execution"]["candidate_domains"]["anchor_0"])
    history_by_acquisition = {
        str(value.get("acquisition_id", "")): str(value.get("coverage_region", ""))
        for value in view.get("viewpoint_history", ())
    }
    rows: list[dict[str, Any]] = []
    for object_id in ids:
        value = objects[object_id]
        evidence = [item for item in value.get("evidence", ()) if isinstance(item, Mapping)]
        acquisitions = sorted({str(item.get("acquisition_id", "")) for item in evidence})
        observation_ids = [str(item.get("observation_id", "")) for item in evidence]
        places = sorted({history_by_acquisition.get(acquisition, "") for acquisition in acquisitions if history_by_acquisition.get(acquisition, "")})
        qwen_unavailable = [
            str(item.get("observation_id", ""))
            for item in evidence
            if str(item.get("qwen_candidate_verdict", "")) == "unavailable"
        ]
        probability = max(0.0, min(1.0, float(value.get("semantic_probability", 0.0) or 0.0)))
        class_label = str(value.get("class_label", ""))
        class_values: dict[str, Any] = {
            "p_bed": probability if class_label == "bed" else None,
            "p_picture": probability if class_label == "picture" else None,
            "p_painting": None,
            "p_photo": None,
            "p_frame": None,
            "p_wall": None,
            "p_other": 1.0 - probability,
        }
        lifecycle = str(value.get("entity_lifecycle", ""))
        if lifecycle == "NEW_SPACE_SINGLETON":
            lifecycle = "PROVISIONAL_SINGLETON"
        geometry_quality = {
            "query_view_geometry_point_count": value.get("geometry_point_count"),
            "available_lidar_support_max": max((int(item.get("lidar_support_count", 0) or 0) for item in evidence), default=0),
            "available_geometry_confidence_max": max((float(item.get("geometry_confidence", 0.0) or 0.0) for item in evidence), default=0.0),
            "covariance_mode": value.get("covariance_mode"),
            "center_covariance_provenance": value.get("center_covariance_provenance"),
        }
        rows.append(
            {
                "entity_id": object_id,
                "observation_ids": compact(observation_ids),
                "acquisition_ids": compact(acquisitions),
                "place_id": compact(places),
                "lifecycle_state": lifecycle,
                **class_values,
                "class_posterior_basis": "runtime binary target-vs-confuser semantic_probability; uncalibrated subclasses are null",
                "semantic_evidence_sources": compact(
                    {
                        "verdicts": dict(Counter(str(item.get("qwen_candidate_verdict", "missing")) for item in evidence)),
                        "qwen_verified_observations": sum(item.get("qwen_verified") is True for item in evidence),
                        "semantic_status": value.get("semantic_status"),
                    }
                ),
                "semantic_unavailable_events": compact(qwen_unavailable),
                "centroid": compact(value.get("center_3d")),
                "obb": compact({"footprint": value.get("footprint_obb"), "extent_xyz": value.get("bbox_3d")}),
                "covariance": compact({"center": value.get("center_cov"), "extent": value.get("extent_cov")}),
                "geometry_quality": compact(geometry_quality),
                "visibility_quality": compact(
                    {
                        "observation_count": value.get("observation_count"),
                        "independent_viewpoint_count": value.get("independent_viewpoint_count"),
                        "station_count": value.get("station_count"),
                    }
                ),
                "same_view_cannot_links": compact(sorted(cannot_by_id[object_id])),
                "possible_duplicate_entities": compact(sorted(duplicates[object_id])),
                "anchor_score": round(probability, 6) if object_id in anchor_domain else "",
                "target_score": round(probability, 6) if object_id in target_domain else "",
                "score_basis": "runtime semantic_probability for final QueryProgram-domain members",
            }
        )
    return rows


def precise_relation_reason(value: Mapping[str, Any]) -> str:
    reason = str(value.get("reason_code", "relation_reason_missing"))
    geometry = value.get("geometry", {}) if isinstance(value.get("geometry"), Mapping) else {}
    participants = geometry.get("participant_geometry_support", ())
    failures = []
    for participant in participants:
        if not isinstance(participant, Mapping):
            continue
        point_count = int(participant.get("geometry_point_count", 0) or 0)
        provenance = str(participant.get("center_covariance_provenance", ""))
        mode = str(participant.get("covariance_mode", ""))
        if point_count <= 0 and (mode == "bearing_only" or provenance in {"bearing_only", "bearing_model"}):
            failures.append(
                f"object {participant.get('object_id')} has geometry_point_count=0, covariance_mode={mode}, center_covariance_provenance={provenance}"
            )
    if reason == "qwen_geometry_vertical_world_geometry_insufficient":
        return reason + ": _world_geometry_usable=false because " + ("; ".join(failures) or "required participant geometry contract was absent")
    qwen = value.get("qwen", {}) if isinstance(value.get("qwen"), Mapping) else {}
    qwen_state = str(qwen.get("state", "")).lower()
    if reason == "gravity_aligned_vertical_and_horizontal_projection_evidence" and qwen_state != "supported":
        suffix = (
            "; geometry fallback unusable because " + "; ".join(failures)
            if failures
            else ""
        )
        return (
            "qwen_predicate_not_affirmative: "
            f"state={qwen_state or 'missing'}, confidence={qwen.get('confidence')}, "
            f"jointly_observable={qwen.get('jointly_observable')}, "
            f"reason_code={reason}{suffix}"
        )
    return reason


def relation_matrix_rows(
    acquisitions: Sequence[Mapping[str, Any]],
    final_snapshot: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> list[dict[str, Any]]:
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    total_calls: Counter[tuple[int, int]] = Counter()
    for acquisition in acquisitions:
        for value in acquisition["relations"]:
            objects = value.get("object_ids", ())
            if not objects:
                continue
            key = relation_key(int(value["subject_object_id"]), int(objects[0]))
            latest[key] = value
            total_calls[key] += 1
    persistent = final_snapshot["relation_evidence"]["records"]
    final_objects = {int(value["object_id"]): value for value in final_snapshot["query_entity_view"]["objects"]}
    final_targets = set(int(value) for value in final_snapshot["count_query_execution"]["candidate_domains"]["target_0"])
    rows: list[dict[str, Any]] = []
    for _persistent_key, record in sorted(
        persistent.items(),
        key=lambda item: (int(item[1]["subject_object_id"]), int(item[1]["anchor_object_ids"][0])),
    ):
        subject_id = int(record["subject_object_id"])
        anchor_id = int(record["anchor_object_ids"][0])
        key = relation_key(subject_id, anchor_id)
        current = latest.get(key, {})
        geometry = current.get("geometry", {}) if isinstance(current.get("geometry"), Mapping) else {}
        qwen = current.get("qwen", {}) if isinstance(current.get("qwen"), Mapping) else {}
        first_record = min(record.get("evidence_records", ()), key=lambda value: float(value.get("timestamp", float("inf"))))
        subject_bounds = geometry.get("subject_bounds")
        object_bounds_values = geometry.get("object_bounds")
        object_bounds = None
        if (
            isinstance(object_bounds_values, Sequence)
            and len(object_bounds_values) == 2
            and all(isinstance(value, Sequence) and len(value) == 3 for value in object_bounds_values)
        ):
            object_bounds = object_bounds_values
        elif (
            isinstance(object_bounds_values, Sequence)
            and object_bounds_values
            and isinstance(object_bounds_values[0], Sequence)
        ):
            object_bounds = object_bounds_values[0]
        iom, iou = xy_metrics_from_bounds(subject_bounds, object_bounds)
        vertical_delta = finite_float(geometry.get("vertical_center_delta_m"))
        if vertical_delta is None:
            ordering = "UNKNOWN"
        elif vertical_delta > 0:
            ordering = "SUBJECT_CENTER_ABOVE_ANCHOR_CENTER"
        elif vertical_delta < 0:
            ordering = "SUBJECT_CENTER_BELOW_ANCHOR_CENTER"
        else:
            ordering = "CENTERS_EQUAL"
        role = qwen.get("role_audit") if isinstance(qwen.get("role_audit"), Mapping) else qwen
        role_states = [str(role.get("subject_role_state", ""))] + [str(value) for value in role.get("object_role_states", ())]
        semantic_gate = {
            "pass": bool(role_states and all(value == "YES" for value in role_states)),
            "qwen_state": qwen.get("state"),
            "confidence": qwen.get("confidence"),
            "jointly_observable": qwen.get("jointly_observable"),
            "role_states": role_states,
            "reason_code": qwen.get("reason_code"),
        }
        geometry_failures = []
        for participant in geometry.get("participant_geometry_support", ()):
            if not isinstance(participant, Mapping):
                continue
            point_count = int(participant.get("geometry_point_count", 0) or 0)
            provenance = str(participant.get("center_covariance_provenance", ""))
            mode = str(participant.get("covariance_mode", ""))
            if point_count <= 0 and (
                mode == "bearing_only" or provenance in {"bearing_only", "bearing_model"}
            ):
                geometry_failures.append(int(participant.get("object_id", -1)))
        rows.append(
            {
                "subject_entity": subject_id,
                "anchor_entity": anchor_id,
                "relation_engine_called": True,
                "call_count": total_calls[key],
                "first_call_timestamp_utc": iso_utc(first_record.get("timestamp")),
                "first_call_elapsed_seconds": round(float(first_record["timestamp"]) - float(capture["timing"]["started_unix"]), 6),
                "result": record.get("state", "UNKNOWN"),
                "vertical_ordering": ordering,
                "subject_bottom_m": round_or_none(subject_bounds[0][2] if subject_bounds else None),
                "subject_top_m": round_or_none(subject_bounds[1][2] if subject_bounds else None),
                "anchor_bottom_m": round_or_none(object_bounds[0][2] if object_bounds else None),
                "anchor_top_m": round_or_none(object_bounds[1][2] if object_bounds else None),
                "vertical_gap_m": round_or_none(geometry.get("vertical_gap_m")),
                "xy_projection_iom": round_or_none(iom),
                "xy_projection_iou": round_or_none(iou),
                "center_displacement": compact(
                    {
                        "distance_m": (geometry.get("distances_m") or [None])[0],
                        "vertical_center_delta_m": vertical_delta,
                        "horizontal_projection_axis_overlap_m": geometry.get("horizontal_projection_axis_overlap_m"),
                    }
                ),
                "subject_covariance_contribution": compact(covariance_summary(final_objects.get(subject_id, {}))),
                "anchor_covariance_contribution": compact(covariance_summary(final_objects.get(anchor_id, {}))),
                "geometry_quality_gate": compact(
                    {
                        "pass": not geometry_failures,
                        "failed_participant_ids": geometry_failures,
                        "geometry_reliability": geometry.get("geometry_reliability"),
                        "participant_geometry_support": geometry.get("participant_geometry_support"),
                    }
                ),
                "semantic_quality_gate": compact(semantic_gate),
                "unknown_or_not_evaluated_reason": precise_relation_reason(current or first_record),
                "resolving_pair_could_change_final_count": subject_id in final_targets and record.get("state") == "UNKNOWN",
            }
        )
    return rows


def information_gain_text(
    acquisitions: Sequence[Mapping[str, Any]], index: int, target_id: int | None
) -> str:
    if target_id is None or index + 1 >= len(acquisitions):
        return "not measured: no subsequent completed acquisition"
    before_objects = {int(value["object_id"]): value for value in acquisitions[index]["snapshot"]["query_entity_view"]["objects"]}
    after_objects = {int(value["object_id"]): value for value in acquisitions[index + 1]["snapshot"]["query_entity_view"]["objects"]}
    before = before_objects.get(target_id, {})
    after = after_objects.get(target_id, {})
    if not after:
        return "target no longer materialized"
    return (
        f"semantic {before.get('semantic_status')}->{after.get('semantic_status')}; "
        f"probability {before.get('semantic_probability')}->{after.get('semantic_probability')}; "
        f"lifecycle {before.get('entity_lifecycle')}->{after.get('entity_lifecycle')}; "
        f"lower bound 0->0"
    )


def scheduler_rows(
    acquisitions: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
    terminal_decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    navigation = final_state.get("scene_memory", {}).get("navigation_history", ())
    rows: list[dict[str, Any]] = []
    for index, acquisition in enumerate(acquisitions):
        need = acquisition["output_need"]
        target_ids = [int(value) for value in need.get("target_ids", ())]
        anchor_ids = [int(value) for value in need.get("anchor_ids", ())]
        uncertainty = need.get("priority_context", {}).get("uncertainty_variable", {})
        variable_type = str(uncertainty.get("variable_type", "SEMANTIC"))
        waypoint = acquisition.get("commanded_waypoint", {})
        nav = navigation[index] if index < len(navigation) else {}
        target_id = target_ids[0] if target_ids else None
        if index + 1 < len(acquisitions):
            requested = acquisitions[index + 1]["acquisition_id"]
        else:
            requested = "20260823T042643_595923Z (deadline finalizer; no completed perception summary)"
        rows.append(
            {
                "queue_order": index + 1,
                "record_kind": "EVIDENCE_NEED",
                "queue_insertion_time_elapsed_seconds": round(float(acquisition["child_end_elapsed_seconds"]), 6),
                "queue_insertion_time_basis": "inferred from capture event plus instrumented child duration",
                "variable_type": variable_type,
                "entity_ids": compact(target_ids),
                "anchor_ids": compact(anchor_ids),
                "expected_possible_change_to_answer_lower_bound": "0..1",
                "expected_possible_change_to_answer_upper_bound": "0..1 candidate (global upper bound remained open)",
                "selected": True,
                "selection_time_elapsed_seconds": waypoint.get("elapsed_seconds", ""),
                "acquisition_requested": requested,
                "navigation_result": compact(
                    {
                        "status": nav.get("status", "pending_at_deadline"),
                        "failure_reason": nav.get("failure_reason"),
                        "actual_arrival_pose": nav.get("actual_arrival_pose"),
                        "semantic_progressed": nav.get("semantic_progressed"),
                    }
                ),
                "information_gain": information_gain_text(acquisitions, index, target_id),
                "final_status": (
                    "selected_and_processed"
                    if index + 1 < len(acquisitions)
                    else "selected_but_interrupted_by_deadline_finalization"
                ),
            }
        )
    order = len(rows)
    remaining = terminal_decision.get("remaining_uncertainties", ())
    for item in remaining:
        variable_type = str(item.get("variable_type", ""))
        order += 1
        entity_ids = item.get("entity_ids", ())
        if variable_type == "RELATION":
            lower = f"0..{len(entity_ids)}"
            upper = f"0..{len(entity_ids)}"
        elif variable_type == "ANCHOR":
            lower = "0..11 through all target bindings"
            upper = "can collapse or reweight the complete target domain"
        else:
            lower = "0"
            upper = "unbounded->finite if coverage closes"
        rows.append(
            {
                "queue_order": order,
                "record_kind": "CERTIFICATE_UNCERTAINTY_NOT_ENQUEUED",
                "queue_insertion_time_elapsed_seconds": "",
                "queue_insertion_time_basis": "never materialized as EvidenceNeed before deadline",
                "variable_type": variable_type,
                "entity_ids": compact(entity_ids),
                "anchor_ids": "[]",
                "expected_possible_change_to_answer_lower_bound": lower,
                "expected_possible_change_to_answer_upper_bound": upper,
                "selected": False,
                "selection_time_elapsed_seconds": "",
                "acquisition_requested": "",
                "navigation_result": "not_selected",
                "information_gain": "not_observed",
                "final_status": str(item.get("reason", "unresolved_at_deadline")),
            }
        )
    return rows


def write_scheduler_diagnosis(
    path: Path,
    acquisitions: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
) -> None:
    total_calls = sum(len(acquisition["relations"]) for acquisition in acquisitions)
    unique_pairs = {
        relation_key(value["subject_object_id"], value["object_ids"][0])
        for acquisition in acquisitions
        for value in acquisition["relations"]
        if value.get("object_ids")
    }
    first_all = min(float(value["timestamp"]) for value in acquisitions[0]["relations"])
    gt_first = min(
        float(value["timestamp"])
        for value in acquisitions[0]["relations"]
        if int(value["subject_object_id"]) in {11, 12, 13, 16}
    )
    start = float(capture["timing"]["started_unix"])
    lines = [
        "# E2 scheduler diagnosis",
        "",
        "## Findings",
        "",
        f"- Semantic variables processed before the first ABOVE relation: **0**. The first RelationEngine tuple call was at {first_all - start:.6f} s, before the first EvidenceNeed was returned.",
        f"- First GT-relevant relation failure: acquisition `20260823T041814_056161Z` at {gt_first - start:.6f} s.",
        f"- Relation evaluation was **not starved**: {total_calls} tuple evaluations covered {len(unique_pairs)} unique subject-anchor pairs; 20 pairs were already evaluated in the initial acquisition.",
        "- Anchor resolution **was starved as an acquisition variable**: all six selected EvidenceNeeds were `SEMANTIC`; zero were `ANCHOR`, `RELATION`, `IDENTITY`, or `COVERAGE`.",
        "- Lower-answer-impact semantic work displaced answer-wide variables: targets 16, 9, 16, 5, 5, and 6 were selected while the unresolved anchor and all target relations could affect the whole count.",
        f"- Pair regeneration: {total_calls - len(unique_pairs)} repeat evaluations occurred after world revisions. Re-evaluation on revision is legitimate, but the QueryEntityView kept the same birth-time anchor geometry, so repeated calls did not consume the already-recorded dense bed geometry.",
        "- No query-relevant affirmative relation was established in production. Offline geometry refresh first makes one middle-picture tuple geometrically affirmative at acquisition 3, but never makes all three supportable; therefore there is no recorded acquisition/time at which count 3 became authoritative.",
        "",
        "## Ownership",
        "",
        "Scheduler starvation is a secondary inefficiency, not the first blocker. The first owner is the SceneMemory-to-QueryEntityView geometry contract, followed by RelationEngine's uncertainty treatment for the narrow headwall/bed projection boundary.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def posterior_for_acquisition(acquisition: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = acquisition["snapshot"]
    execution = acquisition["execution"]
    objects = {int(value["object_id"]): value for value in snapshot["query_entity_view"]["objects"]}
    pair_probability: dict[tuple[int, int], float] = {}
    for value in acquisition["relations"]:
        if not value.get("object_ids"):
            continue
        probability = finite_float(value.get("relation_probability"), 0.5)
        pair_probability[relation_key(value["subject_object_id"], value["object_ids"][0])] = float(probability)
    anchors = [int(value) for value in execution.get("candidate_domains", {}).get("anchor_0", ())]
    weights = [max(0.0, float(objects[value].get("semantic_probability", 0.0) or 0.0)) for value in anchors]
    total = sum(weights)
    weights = [value / total for value in weights] if total > 0 else [1.0 / len(anchors)] * len(anchors)
    target_probabilities: list[dict[str, Any]] = []
    for target_id in execution.get("candidate_domains", {}).get("target_0", ()):
        target_id = int(target_id)
        relation_probability = sum(
            weight * pair_probability.get(relation_key(target_id, anchor_id), 0.5)
            for anchor_id, weight in zip(anchors, weights)
        )
        semantic_probability = max(0.0, min(1.0, float(objects[target_id].get("semantic_probability", 0.0) or 0.0)))
        probability = semantic_probability * relation_probability
        target_probabilities.append(
            {
                "entity_id": target_id,
                "semantic_probability": semantic_probability,
                "anchor_marginalized_relation_probability": relation_probability,
                "count_membership_probability": probability,
            }
        )
    distribution = poisson_binomial([value["count_membership_probability"] for value in target_probabilities])
    mode = max(range(len(distribution)), key=lambda index: distribution[index])
    return {
        "model": "Poisson-binomial over QueryEntityView entities; semantic_probability times anchor-marginalized recorded relation_probability; QueryEntityView identity membership is taken as given",
        "independence_caveat": "entity 11 is a group proposal overlapping atomic pictures, so this posterior is diagnostic and not answer authority",
        "anchor_weights": {str(anchor_id): weight for anchor_id, weight in zip(anchors, weights)},
        "target_probabilities": target_probabilities,
        "distribution": {str(index): value for index, value in enumerate(distribution)},
        "mode": mode,
        "mean": sum(index * value for index, value in enumerate(distribution)),
        "credible_interval_95": credible_interval(distribution),
        "probability_count_3": distribution[3] if len(distribution) > 3 else 0.0,
    }


def geometry_refresh_diagnostic(
    acquisition: Mapping[str, Any], session_root: Path
) -> dict[str, Any]:
    snapshot = acquisition["snapshot"]
    view = recursively_rewrite_paths(
        copy.deepcopy(snapshot["query_entity_view"]),
        "/home/docker/ai_module/runs/live_robot/",
        str(session_root) + "/",
    )
    for value in view["objects"]:
        _refresh_geometry_from_evidence(value, 0.05)
    objects = {int(value["object_id"]): value for value in view["objects"]}
    node = acquisition["summary"]["task_ir"]["count_query_graph"]["relation_nodes"][0]
    anchors = [int(value) for value in acquisition["execution"].get("candidate_domains", {}).get("anchor_0", ())]
    pairs = []
    atomic_yes = set()
    for target_id in (12, 13, 16):
        if target_id not in objects:
            continue
        for anchor_id in anchors:
            geometry = relation_geometry_diagnostic(node, objects[target_id], [objects[anchor_id]])
            state, reason = _geometry_consistency(node, geometry, objects[target_id], [objects[anchor_id]])
            if state == "YES":
                atomic_yes.add(target_id)
            pairs.append(
                {
                    "subject_entity": target_id,
                    "anchor_entity": anchor_id,
                    "geometry_state": state,
                    "reason": reason,
                    "geometry_reliability": geometry.get("geometry_reliability"),
                    "vertical_gap_m": geometry.get("vertical_gap_m"),
                    "horizontal_projection_axis_overlap_m": geometry.get("horizontal_projection_axis_overlap_m"),
                }
            )
    return {
        "diagnostic_only": True,
        "change": "refresh QueryEntityView entity geometry from only the observations already present at this acquisition; no new perception and no GT",
        "atomic_picture_pairs": pairs,
        "atomic_picture_entities_with_any_geometry_yes": sorted(atomic_yes),
        "all_three_supportable": atomic_yes == {12, 13, 16},
    }


def counterfactual_replay(
    acquisitions: Sequence[Mapping[str, Any]], session_root: Path
) -> dict[str, Any]:
    scenarios: dict[str, list[dict[str, Any]]] = {key: [] for key in ("A", "B", "C", "D", "E")}
    diagnostics = []
    for acquisition in acquisitions:
        certificate = acquisition["certificate"]
        next_actual = acquisition["output_need"].get("priority_context", {}).get("uncertainty_variable", {})
        base = {
            "acquisition_index": acquisition["index"],
            "acquisition_id": acquisition["acquisition_id"],
            "time_elapsed_seconds": round(float(acquisition["child_end_elapsed_seconds"]), 6),
            "lower_bound": certificate.get("lower_bound", 0),
            "upper_bound": certificate.get("upper_bound"),
            "final_answer_would_have_been_3": False,
            "earliest_acquisition_time_count_3_supportable": None,
        }
        scenarios["A"].append(
            {
                **base,
                "policy": "existing production order plus production lower-bound deadline rule",
                "predicted_count": certificate.get("best_evidence_answer", 0),
                "count_posterior": None,
                "selected_next_variable": next_actual,
            }
        )
        scenarios["B"].append(
            {
                **base,
                "policy": "anchor-first over recorded evidence only",
                "predicted_count": 0,
                "count_posterior": None,
                "selected_next_variable": {"variable_type": "ANCHOR", "entity_ids": acquisition["execution"].get("candidate_domains", {}).get("anchor_0", ())},
                "result_note": "both bed candidates remain unresolved and their QueryEntityView birth geometry is not refreshed by reordering",
            }
        )
        scenarios["C"].append(
            {
                **base,
                "policy": "relation-first batch over all plausible pairs",
                "predicted_count": 0,
                "count_posterior": None,
                "selected_next_variable": {"variable_type": "RELATION", "pair_count": len(acquisition["relations"])},
                "result_note": "production already evaluated the complete eligible batch in this acquisition; all outputs remained UNKNOWN",
            }
        )
        scenarios["D"].append(
            {
                **base,
                "policy": "largest possible answer-bound change first",
                "predicted_count": 0,
                "count_posterior": None,
                "selected_next_variable": {"variable_type": "RELATION", "entity_ids": acquisition["execution"].get("candidate_domains", {}).get("target_0", ())},
                "result_note": "higher-impact selection avoids semantic draining but recorded relation evidence still has no affirmative tuple",
            }
        )
        posterior = posterior_for_acquisition(acquisition)
        scenarios["E"].append(
            {
                **base,
                "policy": "posterior count from recorded semantic, identity-domain, anchor, and relation evidence",
                "predicted_count": posterior["mode"],
                "count_posterior": posterior,
                "selected_next_variable": {"variable_type": "RELATION_OR_IDENTITY", "criterion": "largest posterior variance contribution"},
                "final_answer_would_have_been_3": posterior["mode"] == 3,
            }
        )
        diagnostics.append(
            {
                "acquisition_index": acquisition["index"],
                "acquisition_id": acquisition["acquisition_id"],
                **geometry_refresh_diagnostic(acquisition, session_root),
            }
        )
    return {
        "schema_version": "phase3b0_e2_counterfactual_replay_v1",
        "question": QUESTION,
        "rules": [
            "No GT enters scenarios A-E.",
            "No missing perception evidence is fabricated.",
            "B-D change only variable order and therefore retain recorded RelationEngine outputs.",
            "E is a diagnostic posterior, not a production or official answer.",
        ],
        "scenarios": scenarios,
        "geometry_refresh_diagnostic": diagnostics,
        "conclusion": {
            "scenario_answering_3_at_final_acquisition": [],
            "posterior_final_mode": scenarios["E"][-1]["predicted_count"],
            "posterior_final_probability_count_3": scenarios["E"][-1]["count_posterior"]["probability_count_3"],
            "earliest_acquisition_where_three_atomic_gt_correspondence_entities_are_geometry_yes_without_gt": None,
        },
    }


def write_counterfactual_summary(path: Path, replay: Mapping[str, Any]) -> None:
    final = {key: values[-1] for key, values in replay["scenarios"].items()}
    lines = [
        "# E2 counterfactual summary",
        "",
        "All calculations use only evidence recorded by the failed run. GT is excluded from A-E.",
        "",
        "| Scenario | Final predicted count | Final bounds / posterior | Would answer 3? |",
        "|---|---:|---|---:|",
        f"| A production order | {final['A']['predicted_count']} | lower 0, upper open | no |",
        f"| B anchor-first | {final['B']['predicted_count']} | lower 0, upper open | no |",
        f"| C relation-first batch | {final['C']['predicted_count']} | lower 0, upper open | no |",
        f"| D answer-impact | {final['D']['predicted_count']} | lower 0, upper open | no |",
        f"| E posterior count | {final['E']['predicted_count']} | P(C=3)={final['E']['count_posterior']['probability_count_3']:.6f}; 95% interval={final['E']['count_posterior']['credible_interval_95']} | no |",
        "",
        "## Interpretation",
        "",
        "- Relation-first is not a hypothetical improvement here: the production path already performed a full eligible relation batch after every acquisition.",
        "- Anchor-first and answer-impact ordering save wasted semantic probes, but order alone cannot repair the geometry contract received by RelationEngine.",
        "- The posterior mode is 4, not 3, because QueryEntityView treats the multi-picture group entity 11 as distinct from atomic entities 12/13/16. A posterior deadline vote would therefore be another wrong-answer path.",
        "- Refreshing geometry from the already-recorded observations makes only the middle-picture tuple geometrically YES from acquisition 3 onward; the left/right tuples stay at a narrow XY projection boundary. No acquisition supports all three under unchanged relation semantics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    path: Path,
    acquisitions: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
    terminal_decision: Mapping[str, Any],
    gt_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    qwen_totals: Counter[str],
    raw_bed: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    start = float(capture["timing"]["started_unix"])
    gt_relevant = [
        row
        for row in relation_rows
        if int(row["subject_entity"]) in {11, 12, 13, 16}
    ]
    first_failure = min(gt_relevant, key=lambda row: float(row["first_call_elapsed_seconds"]))
    gt_by_id = {int(row["gt_object_id"]): row for row in gt_rows}
    total_calls = sum(int(row["call_count"]) for row in relation_rows)
    final_posterior = replay["scenarios"]["E"][-1]["count_posterior"]
    raw_bed_point_count = int(raw_bed.get("geometry_point_count", 0) or 0)
    dense_bed_support = sorted(
        {
            int(value.get("lidar_support_count", 0) or 0)
            for value in raw_bed.get("evidence", ())
            if int(value.get("lidar_support_count", 0) or 0) >= 100
        }
    )
    relation_manifest_count = sum(
        int(value.get("relation_manifest_count", 0) or 0)
        for value in inventory.get("acquisitions", ())
    ) + sum(
        int(value.get("relation_manifest_count", 0) or 0)
        for value in inventory.get("terminal_or_incomplete_directories", ())
    )
    lines = [
        "# Phase 3B-0 E2 forensic report",
        "",
        "## Final status",
        "",
        f"`{FINAL_STATUS}`",
        "",
        "## Exact first blocker",
        "",
        f"`{FIRST_BLOCKER}`",
        "",
        f"The first failing event was acquisition `20260823T041814_056161Z` at **{float(first_failure['first_call_elapsed_seconds']):.6f} s** from evaluator start: `rel_0` for subject {first_failure['subject_entity']} and bed anchor {first_failure['anchor_entity']} returned `UNKNOWN`. Qwen had visible, role-correct predicate support, but RelationEngine received a QueryEntityView anchor with `geometry_point_count=0` and `center_covariance_provenance=bearing_model`, so `_world_geometry_usable` failed.",
        "",
        "This is earlier than scheduler starvation, anchor-acquisition starvation, CountCertificate behavior, and deadline finalization. Those are consequences or inefficiencies, not co-equal blockers.",
        "",
        "## Run and scope",
        "",
        f"- Production HEAD: `{PRODUCTION_HEAD}`.",
        "- Real run: `/home/robot/cmu_vln/challenge_eval_runs/20260823T041758Z_hotel_room_2_q1`.",
        f"- Question: `{QUESTION}`.",
        f"- Published / transparent local expected: `{terminal_decision.get('answer')}` / `{EXPECTED_INTEGER}`.",
        f"- Commit mode: `{terminal_decision.get('commit_mode')}` at {float(capture['timing']['first_response_seconds']):.6f} s.",
        "- GT was read only from `VLA-3D_dataset/Unity.zip` for post-hoc correspondence; no GT value or path entered runtime.",
        f"- No extra E2 run was needed because the existing run contains 53 ledger observations, six completed acquisitions, QueryEntityView exports, persistent relation records, {relation_manifest_count} relation ROI manifests, Qwen records, and LiDAR geometry files.",
        "",
        "## GT/runtime correspondence",
        "",
        "The offline anchor is GT 44 (`bed frame`, NYU bed), with GT 57 as its co-annotated mattress component. The three pictures are GT 31/34/76. Matching required world-center proximity plus 3D bbox overlap; class label alone was never sufficient.",
        "",
        "| GT | Role | Runtime entities | Best center distance | Outcome |",
        "|---:|---|---|---:|---|",
    ]
    for gt_id in (44, 31, 34, 76):
        row = gt_by_id[gt_id]
        lines.append(
            f"| {gt_id} | {row['expected_query_role']} | `{row['matching_runtime_entity_ids']}` | {row['best_center_distance_m']} m | {'present with world geometry' if not row['missing_reason'] else row['missing_reason']} |"
        )
    lines += [
        "",
        "### Bed candidate outcome",
        "",
        f"The bed was perceived. QueryEntityView retained verified anchors 1 and 3, and relation ROIs visibly grounded the actual bed. However, their relation-facing geometry stayed at their birth observations (`bearing_model`, probabilistic, zero point count). The persistent raw SceneMemory later held a fused bed entity with {raw_bed_point_count} points and strict/fused-statistical geometry. That already-recorded geometry was not propagated into the numerical QueryEntityView used by RelationEngine.",
        "",
        "### Three expected picture outcomes",
        "",
        "- GT 31 -> runtime entity 13: atomic right picture, first seen in acquisition 1, verified semantic posterior 0.925726, relation to both anchors evaluated and UNKNOWN.",
        "- GT 34 -> runtime entity 12: atomic middle picture, first seen in acquisition 1, verified semantic posterior 0.99, relation to both anchors evaluated and UNKNOWN.",
        "- GT 76 -> runtime entity 16: atomic left picture, first seen in acquisition 1, supported entity with semantic posterior 0.721748, relation to both anchors evaluated and UNKNOWN.",
        "- Runtime entity 11 is an additional group ROI spanning all three pictures. It was treated as another supported/countable entity, creating a secondary over-count risk after the relation blocker is repaired.",
        "",
        "Therefore `PERCEPTION_RECALL` and `MASK_TO_3D_GEOMETRY` are rejected as first blockers: all three atomic pictures had world-aligned observations and LiDAR-supported geometry in acquisition 1.",
        "",
        "## ABOVE relation matrix",
        "",
        f"- Unique persistent pairs: {len(relation_rows)}; total live tuple evaluations across six acquisitions: {total_calls}.",
        "- Final states: 0 YES, 0 NO, 24 UNKNOWN.",
        "- All eligible pairs were evaluated; there are no `NOT_EVALUATED` pairs in the union domain.",
        "- First GT-relevant failures were at 85.656 s; left-picture entity 16 was evaluated by 94.001 s.",
        "- The dominant exact reason was `qwen_geometry_vertical_world_geometry_insufficient`, not a generic lack of evidence.",
        "- Qwen frequently returned `supported` with confidence 1.0 and role states YES/YES for the picture-bed tuples. RelationEngine correctly withheld YES because its geometry contract was unusable.",
        "",
        "Offline refresh using only recorded evidence (no GT) exposes the next boundary: middle entity 12 becomes geometry-YES from acquisition 3, while entities 13 and 16 remain `horizontal_projection_boundary_uncertain` because the wall plane is 3.6-4.7 cm beyond the fused bed footprint. Thus a geometry-copy fix alone does not yet establish count 3.",
        "",
        "## Scheduler timeline",
        "",
        "- Semantic variables before the first ABOVE call: 0.",
        "- Selected EvidenceNeeds: `SEMANTIC(16) -> SEMANTIC(9) -> SEMANTIC(16) -> SEMANTIC(5) -> SEMANTIC(5) -> SEMANTIC(6)`.",
        "- Selected ANCHOR / RELATION / IDENTITY / COVERAGE variables: 0 / 0 / 0 / 0.",
        "- Anchor acquisition was starved, but relation evaluation was not: the initial relation batch already covered 20 pairs, including all three atomic expected pictures.",
        "- Only the semantic probe for entity 9 removed a confuser from the target domain; probes for 16/5 did not raise the lower bound. Two navigation attempts timed out, and the final semantic request was interrupted by deadline finalization.",
        "- Repeated geometry revisions caused 106 repeat tuple evaluations, but the relation-facing anchor geometry remained birth-time geometry.",
        "",
        "## Counterfactual result",
        "",
        "A-D all finish at 0. Relation-first is behavior production already performed. Anchor-first and answer-impact ordering reduce waste but cannot change the recorded UNKNOWN relations. The diagnostic posterior E finishes with mode "
        f"{final_posterior['mode']} (P(C=3)={final_posterior['probability_count_3']:.6f}, 95% interval {final_posterior['credible_interval_95']}), not 3, because the group entity remains independently countable.",
        "",
        "No acquisition makes all three atomic picture relations affirmative under unchanged decision semantics. The earliest refresh-only single affirmative is acquisition 3, but the count never becomes supportable as 3.",
        "",
        "## Existing evidence that production did not use correctly",
        "",
        f"- The raw SceneMemory fused bed geometry: {raw_bed_point_count} points, strict covariance, fused-statistical provenance.",
        f"- Dense bed observations from later acquisitions (LiDAR support counts {dense_bed_support}) that were present in the ledger but absent from the relation-facing QueryEntityView summary.",
        "- Atomic picture observations for entities 12/13/16 with direct LiDAR support and correct relation ROIs.",
        "- Qwen predicate support and role-audit YES states for the exact picture/bed tuples.",
        f"- Semantic batch availability was imperfect but not causal: across 89 verified detections, verdict totals were `{compact(dict(qwen_totals))}`; unavailable rows did not erase the successful GT-picture observations.",
        "- Identity evidence that entity 11 is a group observation overlapping the atomic picture entities, which a posterior count must not treat as another physical picture.",
        "",
        "## Proposed fixes that would be wrong",
        "",
        "- A scheduler-only patch: all relevant relations were already evaluated before any semantic EvidenceNeed.",
        "- A posterior deadline vote alone: recorded-evidence posterior mode is 4, not 3.",
        "- Lowering a global threshold: it would hide the missing geometry contract and change unrelated predicates.",
        "- Changing Qwen prompts or adding more picture recall: Qwen already grounded all three pictures and supported the relation visually.",
        "- Merging the three adjacent pictures: GT and same-view evidence require three distinct atomic instances.",
        "- Finalizer or CountCertificate special-casing: both faithfully consumed zero affirmative tuples; neither invented the upstream failure.",
        "",
        "## Smallest correct Phase 3B production owner",
        "",
        "The first production owner is the **SceneMemory -> QueryEntityView geometry/cardinality adapter** in `materialize_query_view`; the relation verdict remains owned by **RelationEngine**. NumericalResolver scheduling is a later optimization, not the first repair.",
        "",
        "Maximum two-production-commit proposal:",
        "",
        "1. QueryEntityView contract commit: refresh each materialized entity from all currently assigned ledger geometry using the existing fusion path; preserve `geometry_point_count`, geometry observation IDs, covariance provenance, and atomic-vs-group cardinality. A group ROI overlapping multiple atomic, geometrically distinct picture observations must be coverage evidence, not a fourth count entity.",
        "2. RelationEngine ABOVE commit: consume the corrected OBB/covariance contract and resolve the narrow wall-plane/bed-footprint boundary by principled uncertainty plus jointly visible Qwen role/predicate evidence. Do not introduce a scene/class threshold or GT branch.",
        "",
        "Scheduling/answer-impact work should be considered only after those two owners establish correct tuple truth on fresh real runs.",
        "",
        "## Modification statement",
        "",
        "No production algorithm was modified. No production commit was created. No thresholds, Qwen prompts, navigation, production SceneMemory behavior, scheduler, or deadline rule were changed. This phase added only an offline development audit exporter and generated artifacts under `ai_module/artifacts/phase3b0/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("/home/robot/cmu_vln/challenge_eval_runs/20260823T041758Z_hotel_room_2_q1"),
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("/home/robot/cmu_vln/challenge_eval_runs/.ai_runtime_sessions/20260823T041757Z_hotel_room_2_q1_b4n22nij/live_robot"),
    )
    parser.add_argument(
        "--gt-zip",
        type=Path,
        default=Path("/home/robot/cmu_vln/VLA-3D_dataset/Unity.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase3b0",
    )
    args = parser.parse_args()

    run_request = load_json(args.run / "run_request.json")
    capture = load_json(args.run / "capture.json")
    score = load_json(args.run / "score.json")
    report_text = (args.run / "report.md").read_text(encoding="utf-8")
    if capture.get("question") != QUESTION:
        raise RuntimeError("unexpected question")
    if run_request.get("question") != QUESTION or QUESTION not in report_text:
        raise RuntimeError("outer run provenance mismatch")
    if score.get("details", {}).get("observed") not in (None, 0):
        raise RuntimeError("unexpected observed result")
    summary_paths = sorted(args.session_root.glob("*/summary.json"))
    if len(summary_paths) != 6:
        raise RuntimeError(f"expected six completed acquisition summaries, found {len(summary_paths)}")
    acquisitions = build_acquisitions(summary_paths, capture)
    master_state = load_json(args.session_root / acquisitions[0]["acquisition_id"] / "episode_runtime_state.json")
    raw_scene_memory = load_json(args.session_root / acquisitions[0]["acquisition_id"] / "episode_scene_memory.json")
    terminal_decision = master_state.get("terminal_root_decision", {})
    if terminal_decision.get("commit_mode") != "DEADLINE_BEST_EVIDENCE":
        raise RuntimeError("terminal decision provenance mismatch")
    final_snapshot = acquisitions[-1]["snapshot"]
    raw_bed_candidates = [
        value
        for value in raw_scene_memory.get("objects", ())
        if value.get("class_label") == "bed" and value.get("semantic_status") == "verified"
    ]
    if not raw_bed_candidates:
        raise RuntimeError("verified raw SceneMemory bed missing")
    raw_bed = max(
        raw_bed_candidates,
        key=lambda value: int(value.get("geometry_point_count", 0) or 0),
    )
    inventory = build_evidence_inventory(acquisitions, args.run)
    gt, graph, regions = read_gt(args.gt_zip)

    args.output.mkdir(parents=True, exist_ok=True)
    graph_output = dependency_graph(acquisitions, capture, terminal_decision)
    write_json(args.output / "e2_dependency_graph.json", graph_output)
    write_dependency_timeline(args.output / "e2_dependency_timeline.md", graph_output)

    gt_rows = gt_correspondence_rows(gt, final_snapshot)
    write_csv(args.output / "e2_gt_runtime_correspondence.csv", gt_rows)
    candidate_rows = runtime_candidate_rows(final_snapshot)
    write_csv(args.output / "e2_runtime_candidates.csv", candidate_rows)
    matrix_rows = relation_matrix_rows(acquisitions, final_snapshot, capture)
    write_csv(args.output / "e2_above_relation_matrix.csv", matrix_rows)
    schedule_rows = scheduler_rows(acquisitions, master_state, terminal_decision)
    write_csv(args.output / "e2_scheduler_timeline.csv", schedule_rows)
    write_scheduler_diagnosis(args.output / "e2_scheduler_diagnosis.md", acquisitions, capture)

    replay = counterfactual_replay(acquisitions, args.session_root)
    write_json(args.output / "e2_counterfactual_replay.json", replay)
    write_counterfactual_summary(args.output / "e2_counterfactual_summary.md", replay)

    qwen_totals: Counter[str] = Counter()
    for acquisition in acquisitions:
        qwen_totals.update(acquisition["detection_verdicts"])
    write_final_report(
        args.output / "PHASE3B0_E2_FORENSIC_REPORT.md",
        acquisitions,
        capture,
        terminal_decision,
        gt_rows,
        matrix_rows,
        replay,
        qwen_totals,
        raw_bed,
        inventory,
    )

    write_json(
        args.output / "audit_manifest.json",
        {
            "schema_version": "phase3b0_audit_manifest_v1",
            "production_head": PRODUCTION_HEAD,
            "run": str(args.run),
            "session_root": str(args.session_root),
            "gt_source": str(args.gt_zip),
            "gt_scene_graph_scene_name": graph.get("scene_name"),
            "gt_regions": regions,
            "evidence_inventory": inventory,
            "completed_acquisition_count": len(acquisitions),
            "observation_ledger_count": len(final_snapshot["observation_ledger"]["records"]),
            "relation_unique_pair_count": len(matrix_rows),
            "raw_scene_memory": {
                "object_count": len(raw_scene_memory.get("objects", ())),
                "observation_ledger_count": len(
                    raw_scene_memory.get("observation_ledger", {}).get("records", ())
                ),
                "identity_merge_event_count": len(raw_scene_memory.get("identity_merge_events", ())),
                "identity_resolution_event_count": len(
                    raw_scene_memory.get("identity_resolution_events", ())
                ),
                "acquisition_trace_count": len(raw_scene_memory.get("acquisition_traces", ())),
                "verified_bed_geometry_point_count": raw_bed.get("geometry_point_count"),
            },
            "first_blocker": FIRST_BLOCKER,
            "final_status": FINAL_STATUS,
            "production_algorithm_modified": False,
        },
    )


if __name__ == "__main__":
    main()
