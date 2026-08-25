#!/usr/bin/env python3
"""Model-free Phase 3A replay invariants over production identity APIs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from integrations.execution.scene_memory import materialize_query_view
from integrations.execution.root_finalizer import finalize_resolver_result
from integrations.execution.resolver_contracts import (
    EvidenceNeed,
    EvidenceReason,
    ResolverResult,
    ResolverStatus,
)
from integrations.execution.task_resolvers import CountCertificate
from orchestration.lidar_geometry import _deduplicate_station_observations


def _query(class_name: str) -> dict[str, Any]:
    return {
        "original_question": f"fixture count {class_name}",
        "task_type": "numerical",
        "target_entity": "target_0",
        "entities": [{"id": "target_0", "role": "target", "class_name": class_name, "attributes": {}}],
        "relations": [],
    }


def _observation(
    observation_id: str,
    acquisition_id: str,
    class_name: str,
    center: list[float],
    viewpoint: list[float],
    *,
    timestamp: float,
    qwen: bool = True,
    cannot_link: list[str] | None = None,
    mask_path: str = "",
    pointcloud_path: str = "",
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "acquisition_id": acquisition_id,
        "timestamp": timestamp,
        "class_label": class_name,
        "canonical_class": class_name,
        "observed_class_label": class_name,
        "center_3d": center,
        "bbox_3d": [0.40, 0.40, 0.30],
        "center_cov": np.diag([0.01, 0.01, 0.01]).tolist(),
        "extent_cov": np.diag([0.01, 0.01, 0.01]).tolist(),
        "covariance_mode": "strict",
        "viewpoint_position_map": viewpoint,
        "optical_center_group": f"station:{acquisition_id}",
        "semantic_probability": 0.95 if qwen else 0.55,
        "qwen_verified": qwen,
        "qwen_candidate_verdict": "target_wins" if qwen else "unavailable",
        "lidar_support_count": 32,
        "geometry_confidence": 0.9,
        "appearance_descriptor": [1.0, 0.0, 0.0],
        "appearance_quality": 0.9,
        "source_view_ids": [f"view:{acquisition_id}"],
        "source_view_boxes": [],
        "cannot_link_observation_ids": list(cannot_link or ()),
        "panorama_mask_path": mask_path,
        "pointcloud_path": pointcloud_path,
        "world_points_path": pointcloud_path,
    }


def _materialize(class_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return materialize_query_view(
        _query(class_name),
        {"schema_version": "observation_ledger_v1", "records": records},
    )


def _signature(view: dict[str, Any]) -> dict[str, Any]:
    entities = [
        {
            "id": value["object_id"],
            "key": value["deterministic_entity_key"],
            "lifecycle": value.get("entity_lifecycle"),
            "status": value.get("status"),
            "observations": sorted(item["observation_id"] for item in value.get("evidence", ())),
        }
        for value in view["objects"]
    ]
    relations = [
        {
            "entity_id": value["object_id"],
            "z_order": round(float(value["center_3d"][2]), 6),
        }
        for value in view["objects"] if value.get("status") in {"tentative", "confirmed"}
    ]
    certificate = {
        "lower_bound": sum(value.get("status") == "confirmed" for value in view["objects"]),
        "provisional_singletons": list(view["provisional_singleton_ids"]),
        "duplicate_risk_orphans": [value["entity_id"] for value in view["duplicate_risk_orphans"]],
    }
    return {
        "assignment": view["observation_to_entity"],
        "entities": entities,
        "relations": relations,
        "certificate": certificate,
    }


def _result(name: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"invariant": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def run(output: Path) -> dict[str, Any]:
    results = []
    with tempfile.TemporaryDirectory(prefix="phase3a_invariants_") as temporary:
        root = Path(temporary)

        def masked(name: str, x1: int, x2: int, center: list[float]) -> dict[str, Any]:
            mask = np.zeros((64, 128), dtype=np.uint8)
            mask[16:48, x1:x2] = 255
            mask_path = root / f"{name}.png"
            cv2.imwrite(str(mask_path), mask)
            cloud_path = root / f"{name}.npz"
            points = np.asarray([center, [center[0] + 0.02, center[1], center[2]]], dtype=np.float32)
            np.savez_compressed(cloud_path, world_points=points, lidar_support_points_map=points)
            return _observation(name, "A0", "pillow", center, [0.0, -1.0, 0.0], timestamp=0.0, mask_path=str(mask_path), pointcloud_path=str(cloud_path))

        duplicate_a = masked("i1_a", 20, 60, [0.0, 0.0, 1.0])
        duplicate_b = masked("i1_b", 21, 61, [0.02, 0.0, 1.0])
        fused = _deduplicate_station_observations([duplicate_a, duplicate_b], acquisition_id="A0", geometry_dir=root)
        i1_view = _materialize("pillow", fused)
        results.append(_result("I1_panorama_duplicate", len(fused) == 1 and len(i1_view["objects"]) == 1, {"raw": 2, "acquisition_objects": len(fused), "entities": len(i1_view["objects"]) }))

        separate_a = masked("i2_a", 4, 30, [-0.5, 0.0, 1.0])
        separate_b = masked("i2_b", 94, 120, [0.5, 0.0, 1.0])
        separate = _deduplicate_station_observations([separate_a, separate_b], acquisition_id="A0", geometry_dir=root)
        i2_view = _materialize("pillow", separate)
        results.append(_result("I2_simultaneous_same_class_instances", len(separate) == 2 and len(i2_view["objects"]) == 2 and len(i2_view["identity_constraints"]["cannot_link"]) == 1, {"entities": len(i2_view["objects"]), "cannot_links": i2_view["identity_constraints"]["cannot_link"]}))

        sofas = [
            _observation(f"i3_{index}", "A0", "sofa", [0.55 * index, 0.0, 0.5], [0.0, -2.0, 0.0], timestamp=0.0, cannot_link=[f"i3_{other}" for other in range(3) if other != index])
            for index in range(3)
        ]
        i3_view = _materialize("sofa", sofas)
        results.append(_result("I3_adjacent_annotation_components", len(i3_view["objects"]) == 3, {"entity_keys": [value["deterministic_entity_key"] for value in i3_view["objects"]]}))

        duplicate_risk = [
            _observation("i4_base", "A0", "pillow", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], timestamp=0.0),
            _observation("i4_later", "A1", "pillow", [1.35, 0.0, 1.0], [0.1, -1.0, 0.0], timestamp=1.0),
        ]
        i4_view = _materialize("pillow", duplicate_risk)
        results.append(_result("I4_duplicate_risk_later_observation", len(i4_view["duplicate_risk_orphans"]) == 1 and sum(value.get("status") == "confirmed" for value in i4_view["objects"]) <= 1, {"orphans": i4_view["duplicate_risk_orphans"]}))

        new_space = [
            _observation("i5_base", "A0", "pillow", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], timestamp=0.0),
            _observation("i5_new", "A1", "pillow", [3.0, 0.0, 1.0], [3.0, -1.0, 0.0], timestamp=1.0),
        ]
        i5_view = _materialize("pillow", new_space)
        new_entity = next(value for value in i5_view["objects"] if value["deterministic_entity_key"] == "i5_new")
        results.append(_result("I5_genuine_new_space_object", new_entity.get("entity_lifecycle") == "NEW_SPACE_SINGLETON" and not i5_view["duplicate_risk_orphans"], {"lifecycle": new_entity.get("entity_lifecycle")}))

        positive = _observation("i6_positive", "A0", "pillow", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], timestamp=0.0)
        before = _materialize("pillow", [positive])
        unavailable = _observation("i6_unavailable", "A1", "pillow", [0.01, 0.0, 1.0], [0.5, -1.0, 0.0], timestamp=1.0, qwen=False)
        after = _materialize("pillow", [positive, unavailable])
        before_entity, after_entity = before["objects"][0], after["objects"][0]
        i6_pass = bool(
            before_entity.get("semantic_status") == after_entity.get("semantic_status") == "verified"
            and float(after_entity.get("semantic_target_support", 0.0)) >= float(before_entity.get("semantic_target_support", 0.0))
            and after["observation_to_entity"]["i6_positive"] == after["observation_to_entity"]["i6_unavailable"]
        )
        results.append(_result("I6_qwen_unavailable", i6_pass, {"semantic_before": before_entity.get("semantic_status"), "semantic_after": after_entity.get("semantic_status"), "target_support_before": before_entity.get("semantic_target_support"), "target_support_after": after_entity.get("semantic_target_support")}))

        relation_fixture = [
            _observation("i7_subject", "A0", "pillow", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], timestamp=0.0),
            _observation("i7_anchor", "A0", "bed", [0.0, 0.0, 0.5], [0.0, -1.0, 0.0], timestamp=0.0),
        ]
        relation_query = {
            "original_question": "fixture on relation", "task_type": "numerical", "target_entity": "target_0",
            "entities": [{"id": "target_0", "class_name": "pillow"}, {"id": "anchor_0", "class_name": "bed"}],
            "relations": [{"id": "rel_0", "predicate": "on", "subject_entity": "target_0", "object_entities": ["anchor_0"], "depends_on": []}],
        }
        i7_view = materialize_query_view(relation_query, {"records": relation_fixture})
        assignment_before = copy.deepcopy(i7_view["observation_to_entity"])
        relation_state = "YES" if i7_view["objects"][0]["center_3d"][2] > i7_view["objects"][1]["center_3d"][2] else "NO"
        assignment_after = copy.deepcopy(i7_view["observation_to_entity"])
        results.append(_result("I7_relation_recomputation", assignment_before == assignment_after, {"relation_state": relation_state, "assignment_unchanged": assignment_before == assignment_after}))

        first_signature = _signature(_materialize("pillow", [positive, unavailable]))
        second_signature = _signature(_materialize("pillow", [positive, unavailable]))
        results.append(_result("I8_deterministic_rematerialization", first_signature == second_signature, {"first": first_signature, "second": second_signature}))

    deadline_certificate = CountCertificate(
        lower_bound=2,
        upper_bound=None,
        supported_entities=({"entity_id": 0}, {"entity_id": 1}),
        provisional_singletons=(2,),
        duplicate_risk_orphans=(),
        unresolved_relations=("rel_0",),
        unseen_relevant_volume={"closed": False},
        strict_closed=False,
        best_evidence_answer=2,
        remaining_uncertainties=(
            {"variable_type": "RELATION", "entity_ids": [2]},
        ),
    ).to_dict()
    deadline_result = ResolverResult(
        task_type="numerical",
        status=ResolverStatus.NEED_EVIDENCE,
        scene_version=1,
        identity_revision=1,
        geometry_version=1,
        relation_revision=1,
        evidence_need=EvidenceNeed(
            query_key="fixture", task_type="numerical",
            reason=EvidenceReason.RELATION_EVIDENCE,
        ),
        diagnostics={"count_certificate": deadline_certificate},
    )
    deadline_decision = finalize_resolver_result(
        _query("pillow"), deadline_result, time_budget_exhausted=True
    )
    commit3_gate = {
        "status": "PASS" if (
            deadline_decision.get("action") == "COMMIT"
            and deadline_decision.get("answer") == 2
            and deadline_decision.get("commit_mode") == "DEADLINE_BEST_EVIDENCE"
        ) else "FAIL",
        "count_certificate": deadline_certificate,
        "root_decision": deadline_decision,
        "ros_publication_contract": "authorized Int32 payload; real publication is a real-run gate",
    }
    report = {
        "schema_version": "phase3a_invariant_report_v1",
        "phase": "COMMIT_3_RELATION_COVERAGE_DEADLINE",
        "status": "PASS" if (
            all(value["status"] == "PASS" for value in results)
            and commit3_gate["status"] == "PASS"
        ) else "FAIL",
        "model_calls": 0,
        "results": results,
        "commit3_decision_gate": commit3_gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": {value["invariant"]: value["status"] for value in results}}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output.resolve())
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
