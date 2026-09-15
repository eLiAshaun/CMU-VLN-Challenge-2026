"""Required entity coverage tracking for task execution."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _matches_class(obj: Mapping[str, Any], class_name: str) -> bool:
    """Check if object matches a required class."""
    label = str(obj.get("class_label", "")).strip().lower()
    target = str(class_name).strip().lower()
    if label == target:
        return True
    if "lamp" in target or target.endswith(" lamp"):
        return label == "lamp" or label.endswith(" lamp")
    return False


def _valid_geometry(obj: Mapping[str, Any]) -> bool:
    """Check if object has valid geometry."""
    center = obj.get("center_3d")
    bbox = obj.get("bbox_3d")
    if not (isinstance(center, Sequence) and len(center) == 3 and
            isinstance(bbox, Sequence) and len(bbox) == 3):
        return False
    try:
        values = [float(v) for v in (*center, *bbox)]
        return all(math.isfinite(v) for v in values) and all(v > 0.0 for v in values[3:])
    except (TypeError, ValueError):
        return False


def compute_required_entity_coverage(
    snapshot: Mapping[str, Any],
    task_ir: Mapping[str, Any],
    *,
    current_step_index: int = 0,
    entity_bindings: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """
    Compute required entity coverage for the current task.

    Returns coverage state for all entities required by the task,
    distinguishing between entities needed now vs. future steps.
    """
    task_type = str(task_ir.get("task_type", ""))
    required_classes = set(
        str(c).strip().lower()
        for c in task_ir.get("required_classes", ())
        if str(c).strip()
    )

    entities = {
        str(e["id"]): e
        for e in task_ir.get("entities", ())
        if isinstance(e, Mapping) and str(e.get("id", ""))
    }

    relations = [
        r for r in task_ir.get("relations", ())
        if isinstance(r, Mapping)
    ]

    # Determine entity roles
    role_by_class: dict[str, set[str]] = {}
    for entity_id, entity in entities.items():
        class_name = str(entity.get("class_name", "")).strip().lower()
        if not class_name:
            continue
        role = str(entity.get("role", "")).strip().lower()
        if class_name not in role_by_class:
            role_by_class[class_name] = set()
        if role:
            role_by_class[class_name].add(role)

        # Check if entity is selector anchor
        for relation in relations:
            predicate = str(relation.get("predicate", "")).strip().lower()
            if predicate in {"closest", "farthest", "furthest"}:
                if entity_id in relation.get("object_entities", []):
                    role_by_class[class_name].add("selector_anchor")
                if entity_id == str(relation.get("subject_entity", "")):
                    role_by_class[class_name].add("selector_target")

    # For instruction_following, determine step-specific entities
    step_entities: set[str] = set()
    step_entity_ids: set[str] = set()
    future_entities: set[str] = set()
    if task_type == "instruction_following":
        constraints = list(task_ir.get("ordered_trajectory_constraints", []))
        for i, constraint in enumerate(constraints):
            if not isinstance(constraint, Mapping):
                continue
            target_entity_id = str(constraint.get("target_entity", ""))
            target_entity = entities.get(target_entity_id)
            if target_entity:
                if i == current_step_index:
                    step_entity_ids.add(target_entity_id)
                class_name = str(target_entity.get("class_name", "")).strip().lower()
                if i == current_step_index:
                    step_entities.add(class_name)
                elif i > current_step_index:
                    future_entities.add(class_name)

            # Add relation entities for this step
            for rel_id in constraint.get("relation_ids", []):
                relation = next((r for r in relations if str(r.get("id")) == str(rel_id)), None)
                if relation:
                    for anchor_id in relation.get("object_entities", []):
                        anchor_entity = entities.get(str(anchor_id))
                        if anchor_entity:
                            if i == current_step_index:
                                step_entity_ids.add(str(anchor_id))
                            anchor_class = str(anchor_entity.get("class_name", "")).strip().lower()
                            if i == current_step_index:
                                step_entities.add(anchor_class)
                            elif i > current_step_index:
                                future_entities.add(anchor_class)

    # Get objects from snapshot
    objects = list(snapshot.get("objects", []))

    # Compute coverage for each required class
    coverage_by_class: dict[str, dict[str, Any]] = {}
    for class_name in required_classes:
        matching_objects = [obj for obj in objects if _matches_class(obj, class_name)]

        valid_geometry_objects = [obj for obj in matching_objects if _valid_geometry(obj)]
        rejected_objects = [
            obj for obj in matching_objects
            if str(obj.get("semantic_status", "")) == "rejected"
        ]

        # Coverage is an observation-availability fact. Track confirmation and
        # candidate multiplicity remain continuous query evidence; neither is
        # a reason to hide an otherwise usable entity from the scheduler.
        if not matching_objects:
            state = "MISSING"
        elif not valid_geometry_objects:
            state = "MISSING"
        else:
            state = "SUFFICIENT"

        # Determine role for this class
        roles = role_by_class.get(class_name, set())
        if class_name in step_entities:
            roles.add("current_step")
        if class_name in future_entities:
            roles.add("future_step")

        coverage_by_class[class_name] = {
            "class_name": class_name,
            "state": state,
            "roles": sorted(roles),
            "observation_count": len(matching_objects),
            "valid_geometry_count": len(valid_geometry_objects),
            "rejected_count": len(rejected_objects),
            "object_ids": [int(obj["object_id"]) for obj in valid_geometry_objects],
        }

    # Keep class coverage for broad acquisition planning, but also expose the
    # current entity binding state.  A class-level candidate must not silently
    # stand in for a previously bound physical entity that disappeared or was
    # rejected.
    bindings = entity_bindings if isinstance(entity_bindings, Mapping) else {}
    aliases = snapshot.get("object_id_aliases", {})

    def canonical_object_id(raw: object) -> int | None:
        try:
            current = int(raw)
        except (TypeError, ValueError):
            return None
        visited: set[int] = set()
        while current not in visited:
            visited.add(current)
            next_value = (
                aliases.get(str(current), aliases.get(current))
                if isinstance(aliases, Mapping)
                else None
            )
            if next_value is None:
                break
            try:
                current = int(next_value)
            except (TypeError, ValueError):
                break
        return current

    coverage_by_entity: dict[str, dict[str, Any]] = {}
    for entity_id, entity in entities.items():
        class_name = str(entity.get("class_name", "")).strip().lower()
        class_coverage = coverage_by_class.get(class_name, {})
        class_state = str(class_coverage.get("state", "MISSING")).upper()
        object_ids = [int(value) for value in class_coverage.get("object_ids", ())]
        bound_value = bindings.get(entity_id)
        bound_object_id: int | None = None
        if bound_value is not None:
            bound_object_id = canonical_object_id(bound_value)
        if bound_value is not None and bound_object_id not in object_ids:
            entity_state = "MISSING"
            entity_object_ids: list[int] = []
        else:
            entity_state = class_state
            entity_object_ids = (
                [bound_object_id]
                if bound_object_id is not None
                else object_ids
            )
        roles = set(class_coverage.get("roles", ()))
        role = str(entity.get("role", "")).strip().lower()
        if role:
            roles.add(role)
        coverage_by_entity[entity_id] = {
            "entity_id": entity_id,
            "class_name": class_name,
            "state": entity_state,
            "roles": sorted(roles),
            "bound_object_id": bound_object_id,
            "object_ids": entity_object_ids,
        }

    # Determine missing and unresolved entities
    missing_classes = [
        cn for cn, cov in coverage_by_class.items()
        if cov["state"] == "MISSING"
    ]

    unresolved_classes = [
        cn for cn, cov in coverage_by_class.items()
        if cov["state"] in {"MISSING", "TENTATIVE", "CONFLICTED"}
    ]

    # Check if decision-only is allowed
    decision_only_allowed = True
    decision_only_blocked_reasons: list[str] = []

    # Block decision-only if current step entities are missing
    for class_name in step_entities:
        cov = coverage_by_class.get(class_name, {})
        if cov.get("state") in {"MISSING", "CONFLICTED"}:
            decision_only_allowed = False
            decision_only_blocked_reasons.append(f"current_step_entity_missing:{class_name}")
    for entity_id in step_entity_ids:
        entity_coverage = coverage_by_entity.get(entity_id, {})
        if str(entity_coverage.get("state", "MISSING")) in {"MISSING", "CONFLICTED"}:
            decision_only_allowed = False
            decision_only_blocked_reasons.append(
                f"current_step_entity_binding_missing:{entity_id}"
            )

    # Block decision-only if selector anchors are missing (even for future steps)
    for class_name, cov in coverage_by_class.items():
        if "selector_anchor" in cov.get("roles", []):
            if cov.get("state") == "MISSING":
                decision_only_allowed = False
                decision_only_blocked_reasons.append(f"selector_anchor_missing:{class_name}")
    for entity_id, cov in coverage_by_entity.items():
        if "selector_anchor" in cov.get("roles", ()) and cov.get("state") in {
            "MISSING",
            "CONFLICTED",
        }:
            decision_only_allowed = False
            decision_only_blocked_reasons.append(
                f"selector_anchor_entity_missing:{entity_id}"
            )

    return {
        "schema_version": "required_entity_coverage_v1",
        "task_type": task_type,
        "current_step_index": current_step_index,
        "required_classes": sorted(required_classes),
        "coverage_by_class": coverage_by_class,
        "coverage_by_entity": coverage_by_entity,
        "missing_classes": missing_classes,
        "missing_entities": sorted(
            entity_id
            for entity_id, value in coverage_by_entity.items()
            if value.get("state") == "MISSING"
        ),
        "conflicted_entities": sorted(
            entity_id
            for entity_id, value in coverage_by_entity.items()
            if value.get("state") == "CONFLICTED"
        ),
        "unresolved_classes": unresolved_classes,
        "decision_only_allowed": decision_only_allowed,
        "decision_only_blocked_reasons": decision_only_blocked_reasons,
    }
