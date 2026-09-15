#!/usr/bin/env python3
"""Recorded diagnostic for the single Instruction receding-horizon path."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from integrations.execution.relation_engine import RelationEngine
from integrations.execution.resolver_contracts import ResolverStatus
from integrations.execution.task_resolvers import resolver_for
from integrations.execution.trajectory_geometry import compile_directive_geometry
from integrations.execution.trajectory_monitor import apply_actual_pose
from navigation.waypoint_planning import select_semantic_waypoints


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recorded_summary_invalid")
    return value


def _resolve(payload: Mapping[str, Any]):
    snapshot = copy.deepcopy(payload["stages"]["scene_memory"]["snapshot"])
    execution = payload["stages"]["query_execution"]["execution"]
    engine = RelationEngine(
        None,
        snapshot.get("persistent_relation_summary", {}),
        snapshot.get("object_id_aliases", {}),
        identity_version=int(snapshot.get("identity_revision", 0)),
        geometry_version=int(snapshot.get("geometry_version", 0)),
    )
    result = resolver_for("instruction_following", engine).resolve(
        payload["task_ir"],
        snapshot,
        execution_steps=execution.get("execution_steps"),
        current_step_index=int(execution.get("current_step_index", 0)),
        navigation_history=snapshot.get("navigation_history", ()),
        trajectory_monitor={},
    )
    assert result.status is ResolverStatus.NEED_EXECUTION
    evidence = result.diagnostics["task_evidence"]
    directives = [
        dict(value) for value in result.execution_need.directives
    ]
    assert directives and evidence.get("probe_object") is None
    return result, evidence, directives


def _binding(directive: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": int(directive["order"]),
        "predicate": str(directive.get("action", "")),
        "target_object_id": directive.get("bound_target_object_id"),
        "anchor_object_ids": list(
            directive.get("bound_anchor_object_ids", ())
        ),
        "binding_revision": copy.deepcopy(
            directive.get("binding_revision")
        ),
        "lookahead_only": bool(directive.get("lookahead_only", False)),
    }


def _between_trajectory(
    directive: Mapping[str, Any], execution_steps: list[dict[str, Any]]
) -> dict[str, Any]:
    compiled = compile_directive_geometry(
        directive, clearance_m=0.75, acceptance_radius_m=0.30
    )
    region = compiled["trajectory_region"]
    anchor_ids = [
        int(value["object_id"])
        for value in directive.get("anchor_objects", ())
    ]
    assert len(anchor_ids) == 2 and len(set(anchor_ids)) == 2
    assert region.get("semantic_object_id") is None
    route = select_semantic_waypoints(
        compiled,
        (0.0, 0.0, 0.0),
        None,
        None,
        None,
        None,
    )
    assert len(route) == 3
    region = dict(region)
    region["ingress_xy"] = list(route[0][:2])
    region["corridor_center_xy"] = list(route[1][:2])
    region["egress_xy"] = list(route[2][:2])
    state = {
        "episode_id": "recorded-instruction-replay",
        "execution_steps": copy.deepcopy(execution_steps),
        "current_step_index": int(directive["order"]),
        "actual_trajectory": [],
        "trajectory_monitor": {
            "episode_id": "recorded-instruction-replay",
            "route_active": True,
            "activation_stamp_seconds": 0.0,
            "constraints": [region],
            "satisfied_orders": [],
            "forbidden_violation": None,
            "completed": False,
        },
    }
    step_index = int(directive["order"])
    state["execution_steps"][step_index]["status"] = "READY"
    ingress = route[0]
    egress = route[2]
    dx = float(egress[0]) - float(ingress[0])
    dy = float(egress[1]) - float(ingress[1])
    start = [float(ingress[0]) - 0.15 * dx, float(ingress[1]) - 0.15 * dy, 0.0]
    updates = []
    for index, waypoint in enumerate((start, *route), start=1):
        update = apply_actual_pose(
            state,
            [float(waypoint[0]), float(waypoint[1]), float(waypoint[2])],
            float(index) * 0.5,
            audit_sample_recorded=False,
            linear_speed_mps=0.5,
            sample_frame_id="map",
            sample_episode_id="recorded-instruction-replay",
        )
        updates.append({
            "progressed": bool(update.progressed),
            "current_step_index": int(update.current_step_index),
        })
    assert state["execution_steps"][step_index]["status"] == "SATISFIED"
    assert state["current_step_index"] == step_index + 1
    return {
        "anchor_object_ids": anchor_ids,
        "semantic_object_id": region.get("semantic_object_id"),
        "waypoint_roles": ["ENTRY", "CENTER", "EXIT"],
        "waypoints": [list(value) for value in route],
        "actual_trajectory_updates": updates,
        "satisfaction_source": state["execution_steps"][step_index].get(
            "satisfaction_source"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i1-initial", required=True, type=Path)
    parser.add_argument("--i1-terminal", required=True, type=Path)
    parser.add_argument("--i2-initial", required=True, type=Path)
    parser.add_argument("--i2-between", required=True, type=Path)
    parser.add_argument("--i2-terminal", required=True, type=Path)
    args = parser.parse_args()

    i1_initial = _payload(args.i1_initial)
    i1_terminal = _payload(args.i1_terminal)
    i2_initial = _payload(args.i2_initial)
    i2_between = _payload(args.i2_between)
    i2_terminal = _payload(args.i2_terminal)

    assert [
        str(value["action"])
        for value in i1_initial["task_ir"]["ordered_trajectory_constraints"]
    ] == ["pass_near", "go_to"]
    assert [
        str(value["action"])
        for value in i2_initial["task_ir"]["ordered_trajectory_constraints"]
    ] == ["go_near", "pass_between", "stop_at"]

    _r, i1_first_evidence, i1_first = _resolve(i1_initial)
    _r, i1_last_evidence, i1_last = _resolve(i1_terminal)
    _r, i2_first_evidence, i2_first = _resolve(i2_initial)
    _r, i2_middle_evidence, i2_middle = _resolve(i2_between)
    _r, i2_last_evidence, i2_last = _resolve(i2_terminal)

    assert len(i1_first) == 2 and i1_first[1]["lookahead_only"] is True
    assert i1_first[1]["bound_target_object_id"] == i1_last[0]["bound_target_object_id"]
    assert i1_first[1]["bound_anchor_object_ids"] == i1_last[0]["bound_anchor_object_ids"]
    assert len(i2_first) == 2 and i2_first[1]["lookahead_only"] is True
    assert i2_first[1]["bound_anchor_object_ids"] == i2_middle[0]["bound_anchor_object_ids"]
    assert i2_middle[1]["bound_target_object_id"] == i2_last[0]["bound_target_object_id"]
    assert i2_middle[1]["bound_anchor_object_ids"] == i2_last[0]["bound_anchor_object_ids"]

    between = _between_trajectory(
        i2_middle[0],
        i2_middle_evidence["execution_steps"],
    )
    output = {
        "schema_version": "instruction_v8_recorded_replay_v1",
        "status": "PASS",
        "completion_authority": "actual_state_estimation_trajectory",
        "explicit_verification_probe_count": 0,
        "i1": {
            "ordered_bindings": [
                _binding(i1_first[0]), _binding(i1_last[0])
            ],
            "stable_terminal_entity": (
                i1_first[1]["bound_target_object_id"]
                == i1_last[0]["bound_target_object_id"]
            ),
            "step_statuses": [
                value["status"]
                for value in i1_last_evidence["execution_steps"]
            ],
        },
        "i2": {
            "ordered_bindings": [
                _binding(i2_first[0]),
                _binding(i2_middle[0]),
                _binding(i2_last[0]),
            ],
            "prefix_preserved": [
                value["status"]
                for value in i2_last_evidence["execution_steps"][:2]
            ] == ["SATISFIED", "SATISFIED"],
            "between": between,
        },
        "task_relevant_active_windows": {
            "i1_initial": i1_first_evidence.get("lookahead_preparation"),
            "i2_initial": i2_first_evidence.get("lookahead_preparation"),
            "i2_between": i2_middle_evidence.get("lookahead_preparation"),
        },
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
