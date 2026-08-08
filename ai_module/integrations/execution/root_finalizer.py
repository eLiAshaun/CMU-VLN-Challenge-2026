"""Sole authority translating resolved task execution into publish intent."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _valid_waypoints(values: object, *, expected_count: int | None = None) -> bool:
    if not isinstance(values, list) or not values:
        return False
    if expected_count is not None and len(values) != expected_count:
        return False
    return all(
        isinstance(item, list)
        and len(item) == 3
        and all(math.isfinite(float(component)) for component in item)
        for item in values
    )


def finalize_execution(
    task_ir: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    route_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_type = str(task_ir["task_type"])
    failed = list(dict.fromkeys(str(value) for value in execution.get("failed_constraints", ())))
    evidence = tuple(dict.fromkeys(str(value) for value in execution.get("evidence_ids", ())))
    decision = {
        "schema_version": "root_decision_v1",
        "task_id": str(task_ir["original_question"]),
        "task_type": task_type,
        "scene_version": int(execution["scene_version"]),
        "action": "SAFE_REJECT",
        "answer": None,
        "selected_object": None,
        "waypoint_sequence": [],
        "evidence_ids": list(evidence),
        "failed_constraints": failed,
        "reason": "task_execution_unresolved",
    }
    if not execution.get("resolved"):
        if (
            route_plan is not None
            and route_plan.get("status") == "completed"
            and route_plan.get("probe") is True
            and _valid_waypoints(route_plan.get("waypoints"), expected_count=1)
            and evidence
        ):
            decision.update(
                action="PROBE",
                waypoint_sequence=list(route_plan.get("waypoints", ())),
                reason="independent_viewpoint_probe_required",
            )
            return decision
        if not decision["failed_constraints"]:
            decision["failed_constraints"] = ["root_not_strictly_supported"]
        return decision
    if not evidence:
        decision.update(
            action="SYSTEM_FAILURE",
            reason="root_envelope_invalid",
            failed_constraints=["evidence_envelope_empty"],
        )
        return decision
    if task_type == "numerical":
        answer = execution.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, int) or answer < 0:
            decision.update(action="SYSTEM_FAILURE", reason="root_envelope_invalid", failed_constraints=["numerical_answer_invalid"])
            return decision
        reason = (
            "semantic_relation_count_bringup"
            if execution.get("resolution_mode") == "semantic_relation_count_bringup"
            else "count_candidate_set_closed"
        )
        decision.update(action="COMMIT", answer=answer, reason=reason, failed_constraints=[])
        return decision
    if task_type == "object_reference":
        selected = execution.get("selected_object")
        if not isinstance(selected, Mapping):
            decision.update(action="SYSTEM_FAILURE", reason="root_envelope_invalid", failed_constraints=["selected_object_missing"])
            return decision
        center = selected.get("center_3d")
        bbox = selected.get("bbox_3d")
        if (
            not isinstance(center, list)
            or not isinstance(bbox, list)
            or len(center) != 3
            or len(bbox) != 3
            or not all(math.isfinite(float(value)) for value in (*center, *bbox))
            or any(float(value) <= 0.0 for value in bbox)
        ):
            decision.update(action="SYSTEM_FAILURE", reason="root_envelope_invalid", failed_constraints=["selected_object_geometry_invalid"])
            return decision
        decision.update(action="COMMIT", answer=int(selected["object_id"]), selected_object=dict(selected), reason="task_ir_object_unique", failed_constraints=[])
        return decision
    if task_type == "instruction_following":
        if route_plan is None or route_plan.get("status") != "completed":
            reason = "route_plan_missing" if route_plan is None else str(route_plan.get("reason", "route_plan_blocked"))
            decision["failed_constraints"] = [*decision["failed_constraints"], reason]
            return decision
        waypoints = list(route_plan.get("waypoints", ()))
        if not _valid_waypoints(waypoints, expected_count=len(execution.get("route_objects", ()))):
            decision.update(action="SYSTEM_FAILURE", reason="root_envelope_invalid", failed_constraints=["navigation_goal_resolution_incomplete"])
            return decision
        decision.update(action="COMMIT", waypoint_sequence=waypoints, reason="task_ir_ordered_sequence_resolved", failed_constraints=[])
        return decision
    decision.update(action="SYSTEM_FAILURE", reason="root_envelope_invalid", failed_constraints=["unsupported_task_type"])
    return decision
