"""Absolute authority for every terminal or dispatch root decision."""

from __future__ import annotations

import math
from typing import Any, Mapping

from integrations.execution.resolver_contracts import ResolverResult, ResolverStatus


ROOT_DECISION_SCHEMA = "root_decision_v2"


def _valid_waypoints(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(waypoint, list)
            and len(waypoint) == 3
            and all(math.isfinite(float(component)) for component in waypoint)
            for waypoint in value
        )
    )


def _base(task_ir: Mapping[str, Any], result: ResolverResult) -> dict[str, Any]:
    return {
        "schema_version": ROOT_DECISION_SCHEMA,
        "task_id": str(task_ir.get("original_question", "")),
        "task_type": result.task_type,
        "scene_version": int(result.scene_version),
        "identity_revision": int(result.identity_revision),
        "geometry_version": int(result.geometry_version),
        "relation_revision": int(result.relation_revision),
        "action": "SAFE_REJECT",
        "answer": None,
        "selected_object": None,
        "answer_authorized": False,
        "evidence_ids": list(result.provenance),
        "failed_constraints": [],
        "reason": "resolver_result_not_finalizable",
        "resolver_status": result.status.value,
    }


def _system_failure(
    task_ir: Mapping[str, Any], result: ResolverResult, reason: str
) -> dict[str, Any]:
    decision = _base(task_ir, result)
    decision.update(
        action="SYSTEM_FAILURE",
        reason=str(reason),
        failed_constraints=[str(reason)],
    )
    return decision


def finalize_resolver_result(
    task_ir: Mapping[str, Any],
    result: ResolverResult | Mapping[str, Any],
    *,
    waypoint_selection: Mapping[str, Any] | None = None,
    time_budget_exhausted: bool = False,
) -> dict[str, Any]:
    """Translate the current ResolverResult into the only root decision."""
    if not isinstance(result, ResolverResult):
        result = ResolverResult.from_dict(result)
    decision = _base(task_ir, result)
    if result.status is ResolverStatus.SYSTEM_FAILURE:
        reason = str(result.diagnostics.get("reason", "resolver_system_failure"))
        return _system_failure(task_ir, result, reason)

    if time_budget_exhausted and result.status is not ResolverStatus.FINALIZABLE:
        if result.task_type == "numerical":
            certificate = result.diagnostics.get("count_certificate", {})
            answer = (
                certificate.get("best_evidence_answer")
                if isinstance(certificate, Mapping) else None
            )
            if isinstance(answer, int) and not isinstance(answer, bool) and answer >= 0:
                decision.update(
                    action="COMMIT",
                    answer=int(answer),
                    answer_authorized=True,
                    reason="deadline_best_evidence",
                    commit_mode="DEADLINE_BEST_EVIDENCE",
                    count_certificate=dict(certificate),
                    remaining_uncertainties=list(
                        certificate.get("remaining_uncertainties", ())
                    ),
                )
                return decision
        decision.update(
            action="SAFE_REJECT",
            reason="TIME_BUDGET_EXHAUSTED",
            failed_constraints=["time_budget_exhausted"],
        )
        return decision

    if result.status is ResolverStatus.FINALIZABLE:
        payload = dict(result.final_payload or {})
        if result.task_type == "numerical":
            answer = payload.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, int) or answer < 0:
                return _system_failure(task_ir, result, "numerical_final_payload_invalid")
            decision.update(
                action="COMMIT",
                answer=int(answer),
                answer_authorized=True,
                reason="count_query_graph_finalized",
                commit_mode=str(payload.get("commit_mode", "STRICT_COMMIT")),
                count_certificate=dict(payload.get("count_certificate", {})),
            )
            return decision
        if result.task_type == "object_reference":
            selected = payload.get("selected_object")
            if not isinstance(selected, Mapping):
                return _system_failure(task_ir, result, "selected_object_missing")
            center = selected.get("center_3d")
            extent = selected.get("bbox_3d")
            try:
                geometry = [float(value) for value in (*center, *extent)]
            except (TypeError, ValueError):
                geometry = []
            if (
                len(geometry) != 6
                or not all(math.isfinite(value) for value in geometry)
                or any(value <= 0.0 for value in geometry[3:])
            ):
                return _system_failure(
                    task_ir, result, "selected_object_geometry_invalid"
                )
            if (
                payload.get("relation_required") is True
                and str(payload.get("relation_state", "UNKNOWN")).upper() != "YES"
            ):
                return _system_failure(
                    task_ir, result, "selected_object_relation_unverified"
                )
            decision.update(
                action="COMMIT",
                answer=int(selected["object_id"]),
                selected_object=dict(selected),
                answer_authorized=True,
                reason="unique_canonical_object_finalized",
            )
            return decision
        if result.task_type == "instruction_following":
            if payload.get("actual_trajectory_complete") is not True:
                return _system_failure(
                    task_ir, result, "actual_trajectory_completion_missing"
                )
            decision.update(
                action="COMMIT",
                answer_authorized=True,
                episode_complete=True,
                reason="ordered_constraints_actual_trajectory_finalized",
            )
            return decision
        return _system_failure(task_ir, result, "unsupported_task_type")

    selection = dict(waypoint_selection or {})
    waypoints = selection.get("waypoints")
    if not _valid_waypoints(waypoints):
        decision.update(
            action="SAFE_REJECT",
            reason=(
                "evidence_observation_unreachable"
                if result.status is ResolverStatus.NEED_EVIDENCE
                else "execution_intent_unreachable"
            ),
            failed_constraints=["waypoint_selection_missing"],
        )
        return decision

    if result.status is ResolverStatus.NEED_EVIDENCE:
        observation_intent = selection.get("observation_intent")
        if not isinstance(observation_intent, Mapping):
            return _system_failure(task_ir, result, "observation_intent_missing")
        decision.update(
            action="PROBE",
            reason="evidence_acquisition_required",
            intent="ACQUIRE_EVIDENCE",
            waypoint_sequence=[list(value) for value in waypoints],
            waypoint_context=dict(selection.get("waypoint_context", {})),
            waypoint_selection=selection,
            observation_intent=dict(observation_intent),
        )
        return decision

    if result.status is ResolverStatus.NEED_EXECUTION:
        execution_intent = selection.get("execution_intent")
        if not isinstance(execution_intent, Mapping):
            return _system_failure(task_ir, result, "execution_intent_missing")
        decision.update(
            action="PROBE",
            reason="ordered_constraint_execution_required",
            intent="EXECUTE_ORDERED_CONSTRAINT",
            waypoint_sequence=[list(value) for value in waypoints],
            waypoint_context=dict(selection.get("waypoint_context", {})),
            waypoint_selection=selection,
            execution_intent=dict(execution_intent),
        )
        return decision

    return _system_failure(task_ir, result, "resolver_status_invalid")


def finalize_system_failure(
    task_ir: Mapping[str, Any],
    *,
    reason: str,
    scene_version: int = 0,
    identity_revision: int = 0,
    geometry_version: int = 0,
    relation_revision: int = 0,
) -> dict[str, Any]:
    result = ResolverResult(
        task_type=str(task_ir.get("task_type", "unknown")),
        status=ResolverStatus.SYSTEM_FAILURE,
        scene_version=max(0, int(scene_version)),
        identity_revision=max(0, int(identity_revision)),
        geometry_version=max(0, int(geometry_version)),
        relation_revision=max(0, int(relation_revision)),
        diagnostics={"reason": str(reason)},
    )
    return finalize_resolver_result(task_ir, result)
