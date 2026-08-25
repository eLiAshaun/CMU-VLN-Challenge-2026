#!/usr/bin/env python3
"""Replay one recorded numerical episode through current production semantics.

This tool uses only the episode's ObservationLedger, recorded relation images,
and recorded Qwen metadata. It never opens VLA-3D annotations and is diagnostic,
not a substitute for a fresh Unity/ROS run.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from integrations.execution.count_query_executor import (
    execute_count_query_graph,
)
from integrations.execution.relation_evidence import (
    _attach_shared_image_layout,
    _geometry_consistency,
    relation_geometry_diagnostic,
)
from integrations.execution.scene_memory import materialize_query_view


DEFAULT_RUN = Path(
    "/home/robot/cmu_vln/challenge_eval_runs/"
    "20260823T041758Z_hotel_room_2_q1"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_paths(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, dict):
        return {
            key: rewrite_paths(item, source, destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_paths(item, source, destination) for item in value]
    if isinstance(value, str) and value.startswith(source):
        return destination + value[len(source):]
    return value


def runtime_root(run_dir: Path, first_acquisition_id: str) -> Path:
    sessions = run_dir.parent / ".ai_runtime_sessions"
    matches = sorted(sessions.glob(f"*/live_robot/{first_acquisition_id}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"recorded_runtime_root_not_unique:{first_acquisition_id}:{len(matches)}"
        )
    return matches[0].parent


def recorded_acquisition_ids(capture: Mapping[str, Any]) -> list[str]:
    return [
        Path(str(event.get("payload", {}).get("run_dir", ""))).name
        for event in capture.get("status_events", ())
        if isinstance(event, Mapping)
        and event.get("payload", {}).get("state") == "captured"
        and str(event.get("payload", {}).get("run_dir", ""))
    ]


class RecordedRelationReplay:
    def __init__(
        self,
        acquisition_root: Path,
        recorded_tuples: Sequence[Mapping[str, Any]],
    ) -> None:
        self.acquisition_root = acquisition_root
        self.qwen_by_pair = {
            (
                int(value.get("subject_object_id", -1)),
                tuple(int(item) for item in value.get("object_ids", ())),
            ): dict(value.get("qwen", {}))
            for value in recorded_tuples
            if isinstance(value, Mapping)
        }

    def __call__(
        self,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        subject_id = int(subject["object_id"])
        object_ids = tuple(int(value["object_id"]) for value in objects)
        qwen = self.qwen_by_pair.get((subject_id, object_ids), {})
        geometry = relation_geometry_diagnostic(node, subject, objects)
        suffix = "_".join(str(value) for value in object_ids)
        manifest_path = (
            self.acquisition_root
            / "06_relation_evidence"
            / f"{node['id']}_s{subject_id}_o{suffix}"
            / "manifest.json"
        )
        evidence = (
            load_json(manifest_path)
            if manifest_path.is_file()
            else {
                "jointly_observable": False,
                "reason": "recorded_relation_manifest_missing",
                "evidence_ids": [],
            }
        )
        _attach_shared_image_layout(geometry, node, evidence, objects)
        state, reason = _geometry_consistency(
            node, geometry, subject, objects, qwen
        )
        probability = 0.99 if state == "YES" else 0.01 if state == "NO" else 0.5
        return {
            "state": state,
            "relation_probability": probability,
            "reason_code": reason,
            "evidence_ids": list(evidence.get("evidence_ids", ())),
            "source_observation_ids": list(evidence.get("evidence_ids", ())),
            "geometry": geometry,
            "qwen": qwen,
            "evidence_source": "recorded_evidence_current_production_replay",
        }

    def verify_many(
        self,
        items: Sequence[
            tuple[
                Mapping[str, Any],
                Mapping[str, Any],
                Sequence[Mapping[str, Any]],
            ]
        ],
    ) -> list[dict[str, Any]]:
        return [self(node, subject, objects) for node, subject, objects in items]


def replay(run_dir: Path) -> dict[str, Any]:
    capture = load_json(run_dir / "capture.json")
    acquisition_ids = recorded_acquisition_ids(capture)
    if not acquisition_ids:
        raise RuntimeError("recorded_acquisition_ids_missing")
    live_root = runtime_root(run_dir, acquisition_ids[0])
    completed = [
        live_root / acquisition_id / "summary.json"
        for acquisition_id in acquisition_ids
        if (live_root / acquisition_id / "summary.json").is_file()
    ]
    if not completed:
        raise RuntimeError("recorded_summary_missing")
    final_summary_path = completed[-1]
    final_summary = load_json(final_summary_path)
    recorded_snapshot = final_summary["stages"]["scene_memory"]["snapshot"]
    task_ir = final_summary["task_ir"]
    ledger = rewrite_paths(
        copy.deepcopy(recorded_snapshot["observation_ledger"]),
        "/home/docker/ai_module/runs/live_robot",
        str(live_root),
    )
    current_snapshot = rewrite_paths(
        copy.deepcopy(recorded_snapshot),
        "/home/docker/ai_module/runs/live_robot",
        str(live_root),
    )
    query_view = materialize_query_view(
        task_ir,
        ledger,
        current_snapshot=current_snapshot,
    )
    snapshot = copy.deepcopy(recorded_snapshot)
    for key in (
        "scene_version",
        "identity_revision",
        "geometry_version",
        "objects",
        "object_id_aliases",
        "identity_clusters",
        "identity_constraints",
        "identity_ambiguity_groups",
        "ambiguous_observations",
        "cardinality_summary",
        "viewpoint_history",
        "observation_ledger_size",
    ):
        snapshot[key] = copy.deepcopy(query_view[key])
    snapshot["query_entity_view"] = copy.deepcopy(query_view)
    snapshot["scene_memory_authority"] = "ObservationLedger+QueryProgram"
    snapshot["numerical_count_domain_closed"] = False

    recorded_tuples = [
        tuple_result
        for node in recorded_snapshot.get("count_query_execution", {}).get(
            "node_results", ()
        )
        if isinstance(node, Mapping)
        for tuple_result in node.get("tuple_results", ())
        if isinstance(tuple_result, Mapping)
    ]
    execution = execute_count_query_graph(
        task_ir["count_query_graph"],
        snapshot,
        RecordedRelationReplay(final_summary_path.parent, recorded_tuples),
    )
    objects = {
        int(value["object_id"]): value for value in query_view["objects"]
    }
    graph = task_ir["count_query_graph"]
    target_entity = str(graph["target_entity"])
    singular_entities = {
        str(value["entity_id"])
        for value in graph.get("variables", ())
        if str(value.get("quantifier", "")) == "SINGULAR"
    }
    anchor_ids = sorted({
        int(value)
        for entity_id in singular_entities
        for value in execution.get("candidate_domains", {}).get(entity_id, ())
    })
    target_rows = [
        {
            "object_id": int(value["object_id"]),
            "cardinality_role": str(
                objects[int(value["object_id"])].get("cardinality_role", "")
            ),
            "relation_state": str(value.get("relation_state", "UNKNOWN")),
            "identity_status": str(value.get("identity_status", "")),
            "semantic_status": str(value.get("semantic_status", "")),
            "countable": value.get("countable") is True,
        }
        for value in execution.get("target_id_states", ())
    ]
    aggregate_ids = list(
        query_view["cardinality_summary"].get(
            "aggregate_coverage_entity_ids", ()
        )
    )
    partial_ids = list(
        query_view["cardinality_summary"].get(
            "partial_fragment_entity_ids", ()
        )
    )
    target_domain_ids = set(
        execution.get("candidate_domains", {}).get(target_entity, ())
    )
    return {
        "schema_version": "phase3b_recorded_replay_v2",
        "diagnostic_only": True,
        "ground_truth_accessed": False,
        "run_dir": str(run_dir),
        "final_recorded_summary": str(final_summary_path),
        "question": str(task_ir.get("original_question", "")),
        "geometry_contract": {
            str(object_id): {
                "geometry_observation_ids": list(
                    objects[object_id].get("geometry_observation_ids", ())
                ),
                "geometry_point_count": int(
                    objects[object_id].get("geometry_point_count", 0) or 0
                ),
                "covariance_provenance": str(
                    objects[object_id].get(
                        "center_covariance_provenance", ""
                    )
                ),
                "geometry_revision": int(
                    objects[object_id].get("geometry_revision", 0) or 0
                ),
            }
            for object_id in anchor_ids
            if object_id in objects
        },
        "cardinality_summary": copy.deepcopy(
            query_view["cardinality_summary"]
        ),
        "target_entity": target_entity,
        "target_rows": target_rows,
        "aggregate_entities_excluded": aggregate_ids,
        "partial_fragment_entities_excluded": partial_ids,
        "count_execution": {
            "cardinality_lower_bound": execution.get(
                "cardinality_lower_bound"
            ),
            "cardinality_upper_bound": execution.get(
                "cardinality_upper_bound"
            ),
            "answer": execution.get("answer"),
            "counted_target_ids": list(
                execution.get("counted_target_ids", ())
            ),
            "unknown_target_ids": list(
                execution.get("unknown_target_ids", ())
            ),
        },
        "checks": {
            "anchor_geometry_nonzero": bool(
                anchor_ids
                and all(
                    int(objects[value].get("geometry_point_count", 0) or 0) > 0
                    for value in anchor_ids if value in objects
                )
            ),
            "anchor_geometry_sources_present": bool(
                anchor_ids
                and all(
                    objects[value].get("geometry_observation_ids")
                    for value in anchor_ids if value in objects
                )
            ),
            "relation_geometry_contract_complete": bool(
                anchor_ids
                and all(
                    all(
                        key in objects[value]
                        for key in (
                            "center_3d",
                            "bbox_3d",
                            "center_cov",
                            "extent_cov",
                            "footprint_obb",
                            "geometry_observation_ids",
                            "geometry_point_count",
                            "center_covariance_provenance",
                            "geometry_revision",
                        )
                    )
                    for value in anchor_ids if value in objects
                )
            ),
            "aggregate_excluded_from_count": bool(
                aggregate_ids
                and not target_domain_ids.intersection(aggregate_ids)
                and not set(execution.get("counted_target_ids", ())).intersection(
                    aggregate_ids
                )
            ),
            "partial_fragment_excluded_from_independent_domains": bool(
                partial_ids
                and not set(anchor_ids).intersection(partial_ids)
                and not target_domain_ids.intersection(partial_ids)
                and not set(execution.get("counted_target_ids", ())).intersection(
                    partial_ids
                )
            ),
            "affirmative_atomic_relations": sorted(
                int(value["object_id"])
                for value in target_rows
                if value["cardinality_role"] == "ATOMIC"
                and value["relation_state"] == "YES"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir.resolve())
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
