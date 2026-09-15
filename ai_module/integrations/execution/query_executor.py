"""Canonical count, unique-object, relation, and trajectory-reference execution."""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Callable, Mapping, Sequence

from integrations.execution.query_domain import (
    relation_identity_blockers,
    relation_identity_ready,
)
from integrations.execution.instruction_executor import (
    RecedingHorizonInstructionExecutor,
    StepStatus,
)
from integrations.semantics.relation_registry import OperatorKind, relation_spec


def numerical_relation_plan(task_ir: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive the numerical target's relation roles without changing TaskIR.

    TaskIR stores directed predicates, but a counted entity can be either the
    predicate subject (``pictures above bed``) or one of its objects
    (``chairs with pillows on them`` -> ``on(pillow, chair)``).  This derived
    plan keeps that distinction explicit for execution while leaving the
    compiler schema and every non-numerical consumer untouched.
    """
    if str(task_ir.get("task_type", "")) != "numerical":
        return None
    target_id = str(task_ir.get("target_entity", ""))
    entities = {
        str(item.get("id", "")): item for item in task_ir.get("entities", ())
    }
    if not target_id or target_id not in entities:
        return None
    bindings = []
    for relation in task_ir.get("relations", ()):
        subject_id = str(relation.get("subject_entity", ""))
        object_ids = [str(value) for value in relation.get("object_entities", ())]
        if subject_id == target_id:
            target_role = "subject"
            target_object_indexes: list[int] = []
        else:
            target_object_indexes = [
                index for index, entity_id in enumerate(object_ids)
                if entity_id == target_id
            ]
            if not target_object_indexes:
                continue
            target_role = "object"
        bindings.append({
            "relation_id": str(relation.get("id", "")),
            "predicate": str(relation.get("predicate", "")).lower(),
            "target_role": target_role,
            "target_object_indexes": target_object_indexes,
            "subject_entity": subject_id,
            "object_entities": object_ids,
            "depends_on": [str(value) for value in relation.get("depends_on", ())],
        })
    return {
        "schema_version": "numerical_relation_plan_v1",
        "target_entity": target_id,
        "relations": bindings,
    }


def _distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Horizontal distance used by closest/farthest and near relations."""
    return math.hypot(
        float(first["center_3d"][0]) - float(second["center_3d"][0]),
        float(first["center_3d"][1]) - float(second["center_3d"][1]),
    )


def _semantic_evidence_count(obj: Mapping[str, Any]) -> int:
    """Count useful semantic evidence without making it a hard gate."""
    return sum(
        bool(evidence.get("qwen_verified"))
        or isinstance(evidence.get("bearing_observation"), Mapping)
        for evidence in obj.get("evidence", ())
    )


def _relation_bound_evidence_count(obj: Mapping[str, Any]) -> int:
    """Count observations generated inside a dependency-bound relation ROI."""
    return sum(
        bool(str(evidence.get("relation_roi_image_path", "")).strip())
        and bool(str(evidence.get("relation_roi_predicate", "")).strip())
        for evidence in obj.get("evidence", ())
    )


def _valid_center_3d(obj: Mapping[str, Any]) -> bool:
    center = obj.get("center_3d")
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return False
    try:
        return all(math.isfinite(float(value)) for value in center)
    except (TypeError, ValueError):
        return False


def _entity_class_names(entity: Mapping[str, Any]) -> set[str]:
    names = {
        str(entity.get("class_name", "")).strip().lower(),
        *(str(value).strip().lower() for value in entity.get("aliases", ())),
    }
    names.discard("")
    return names


def _task_entity_class_names(
    task_ir: Mapping[str, Any], entity: Mapping[str, Any]
) -> set[str]:
    """Resolve the class vocabulary already declared by the TaskIR plan."""
    names = _entity_class_names(entity)
    plan = task_ir.get("grounding_plan", {})
    if not isinstance(plan, Mapping):
        return names
    entity_id = str(entity.get("id", ""))
    for planned in plan.get("entities", ()):
        if not isinstance(planned, Mapping) or str(
            planned.get("entity_id", "")
        ) != entity_id:
            continue
        names.update(
            str(value).strip().lower()
            for value in planned.get("detector_aliases", ())
            if str(value).strip()
        )
        break
    class_name = str(entity.get("class_name", "")).strip()
    detector_classes = plan.get("detector_classes", {})
    if isinstance(detector_classes, Mapping):
        names.update(
            str(value).strip().lower()
            for value in detector_classes.get(class_name, ())
            if str(value).strip()
        )
    return names


def _matches_class_names(
    obj: Mapping[str, Any],
    class_names: Sequence[str] | set[str],
) -> bool:
    label = str(obj.get("class_label", "")).strip().lower()
    names = {str(value).strip().lower() for value in class_names if str(value).strip()}
    if label in names:
        return True
    # Detector labels often retain a compound modifier while TaskIR names the
    # semantic head (for example ``computer monitor`` -> ``monitor``).  This
    # is a generic lexical match, not a scene or question-specific alias.
    if any(label.endswith(f" {name}") for name in names):
        return True
    # SceneMemory keeps detector-specific lamp labels, while TaskIR normally
    # names the semantic class simply as ``lamp``.  These are one class family,
    # not unrelated alternative candidates.
    return bool(
        any(value == "lamp" or value.endswith(" lamp") for value in names)
        and (label == "lamp" or label.endswith(" lamp"))
    )


def _attribute_matches(
    entity: Mapping[str, Any],
    obj: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] | set[str],
) -> bool:
    attributes = entity.get("attributes", {})
    if not isinstance(attributes, Mapping):
        return True
    color = str(attributes.get("color", "")).strip().lower()
    if color:
        if color == "grey":
            color = "gray"
        labels = {
            str(value).strip().lower()
            for value in obj.get("color_labels", ())
            if str(value).strip()
        }
        if not labels or color not in labels:
            return False
    size = str(attributes.get("size", "")).strip().lower()
    if size:
        try:
            volume = math.prod(float(value) for value in obj["bbox_3d"])
            class_volumes = sorted(
                math.prod(float(value) for value in value["bbox_3d"])
                for value in objects
                if _matches_class_names(value, class_names)
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not class_volumes:
            return False
        median = class_volumes[len(class_volumes) // 2]
        if size == "small" and volume > median:
            return False
        if size == "large" and volume < median:
            return False
    return True


def _instruction_relation_discovery_state(
    snapshot: Mapping[str, Any],
    *,
    target_class_names: Sequence[str] | set[str],
    anchor_class_names: Sequence[str] | set[str],
    focus_object_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Summarize open-world discovery for one instruction relation.

    A persistent YES tuple proves that the observed subject and anchor satisfy
    the relation.  It does not prove that the tuple is the only eligible
    subject in the scene.  This state therefore keeps terminal instruction
    selection open while independent views still discover relevant objects,
    unresolved identities, or traversable frontier regions.  Once a relation
    tuple exists, the frontier and saturation calculation is scoped to the
    tuple's observable neighborhood; an unrelated open-room frontier must not
    keep a local relation query open forever.

    The result is an acquisition objective, not an answer gate: a currently
    best tuple remains available as a route hypothesis, while the frontier and
    saturation evidence decide whether the relation may become terminal.
    """
    target_names = {
        str(value).strip().lower()
        for value in target_class_names
        if str(value).strip()
    }
    anchor_names = {
        str(value).strip().lower()
        for value in anchor_class_names
        if str(value).strip()
    }
    relevant_names = target_names | anchor_names
    object_by_id = {
        int(value["object_id"]): value
        for value in snapshot.get("objects", ())
        if isinstance(value, Mapping)
        and str(value.get("object_id", "")).lstrip("-").isdigit()
    }
    aliases = snapshot.get("object_id_aliases", {})

    def canonical_object_id(raw_id: object) -> int | None:
        try:
            current = int(raw_id)
        except (TypeError, ValueError):
            return None
        visited: set[int] = set()
        while current not in visited:
            visited.add(current)
            raw_next = (
                aliases.get(str(current), aliases.get(current))
                if isinstance(aliases, Mapping)
                else None
            )
            if raw_next is None:
                break
            try:
                next_id = int(raw_next)
            except (TypeError, ValueError):
                break
            if next_id == current:
                break
            current = next_id
        return current if current in object_by_id else None

    focus_ids = {
        canonical_id
        for raw_id in focus_object_ids
        for canonical_id in (canonical_object_id(raw_id),)
        if canonical_id is not None
    }

    def object_xy(value: Mapping[str, Any]) -> tuple[float, float] | None:
        center = value.get("center_3d")
        if not (
            isinstance(center, Sequence)
            and not isinstance(center, (str, bytes, bytearray))
            and len(center) >= 2
        ):
            return None
        try:
            x, y = float(center[0]), float(center[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return x, y

    focus_xy: list[tuple[float, float]] = []
    focus_radius_m = 0.0
    for object_id in sorted(focus_ids):
        value = object_by_id.get(object_id)
        if value is None:
            continue
        point = object_xy(value)
        if point is not None:
            focus_xy.append(point)
        extent = value.get("bbox_3d", ())
        if isinstance(extent, Sequence) and len(extent) >= 2:
            try:
                focus_radius_m = max(
                    focus_radius_m,
                    math.hypot(float(extent[0]), float(extent[1])) + 1.0,
                )
            except (TypeError, ValueError):
                pass
        covariance = value.get("center_cov", ())
        if (
            isinstance(covariance, Sequence)
            and len(covariance) >= 2
            and isinstance(covariance[0], Sequence)
            and isinstance(covariance[1], Sequence)
            and len(covariance[0]) >= 2
            and len(covariance[1]) >= 2
        ):
            try:
                sigma = math.sqrt(
                    max(0.0, float(covariance[0][0]))
                    + max(0.0, float(covariance[1][1]))
                )
                focus_radius_m = max(focus_radius_m, 2.0 * sigma + 1.0)
            except (TypeError, ValueError):
                pass
    if focus_xy:
        # Keep the neighborhood derived from the observed participants rather
        # than from the entire traversable map.  The margin represents the
        # nearby occluded area in which another instance can change this
        # relation result.
        focus_radius_m = max(2.0, focus_radius_m)

    def point_in_focus(point: Sequence[object]) -> bool:
        if not focus_xy:
            return True
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            return False
        return min(
            math.hypot(x - focus_x, y - focus_y)
            for focus_x, focus_y in focus_xy
        ) <= focus_radius_m

    def object_in_focus(object_id: int) -> bool:
        if not focus_xy or object_id in focus_ids:
            return True
        point = object_xy(object_by_id.get(object_id, {}))
        return point is not None and point_in_focus(point)

    target_ids = sorted(
        object_id
        for object_id, value in object_by_id.items()
        if _matches_class_names(value, target_names)
        and value.get("status", "confirmed") in {"tentative", "confirmed"}
    )
    anchor_ids = sorted(
        object_id
        for object_id, value in object_by_id.items()
        if _matches_class_names(value, anchor_names)
        and value.get("status", "confirmed") in {"tentative", "confirmed"}
    )
    relevant_ids = set(target_ids) | set(anchor_ids)

    def relevant_new_ids(record: Mapping[str, Any]) -> set[int]:
        result: set[int] = set()
        by_class = record.get("new_instance_ids_by_class")
        if isinstance(by_class, Mapping):
            for raw_class, raw_ids in by_class.items():
                class_name = str(raw_class).strip().lower()
                if class_name not in relevant_names:
                    continue
                if isinstance(raw_ids, Sequence) and not isinstance(
                    raw_ids, (str, bytes, bytearray)
                ):
                    result.update(
                        canonical_id
                        for value in raw_ids
                        for canonical_id in (canonical_object_id(value),)
                        if canonical_id is not None
                    )
        if result:
            return result
        raw_ids = record.get("new_instance_ids", ())
        if not isinstance(raw_ids, Sequence) or isinstance(
            raw_ids, (str, bytes, bytearray)
        ):
            return result
        for raw_id in raw_ids:
            if not str(raw_id).lstrip("-").isdigit():
                continue
            object_id = canonical_object_id(raw_id)
            value = object_by_id.get(object_id)
            if value is not None and _matches_class_names(value, relevant_names):
                result.add(object_id)
        return result

    valid_records: list[dict[str, Any]] = []
    for raw in snapshot.get("viewpoint_history", ()):
        if not isinstance(raw, Mapping):
            continue
        position = raw.get("viewpoint_position_map")
        valid_position = (
            isinstance(position, Sequence)
            and not isinstance(position, (str, bytes, bytearray))
            and len(position) >= 2
        )
        if raw.get("valid_for_count_closure") is not True or not valid_position:
            continue
        record = dict(raw)
        record["_query_related"] = point_in_focus(position)
        record["_relevant_new_ids"] = sorted(
            object_id
            for object_id in relevant_new_ids(record)
            if object_in_focus(object_id)
        )
        record["_independent"] = raw.get("independent_viewpoint") is True
        valid_records.append(record)

    query_related_records = [
        value for value in valid_records
        if value.get("_query_related") is True
    ]
    independent_records = [
        value for value in query_related_records
        if value.get("_independent") is True
    ]
    covered_regions: list[str] = []
    for record in independent_records:
        region = str(record.get("coverage_region", "")).strip()
        if not region:
            position = record.get("viewpoint_position_map")
            if isinstance(position, Sequence) and len(position) >= 2:
                try:
                    region = f"xy:{math.floor(float(position[0]))}:{math.floor(float(position[1]))}"
                except (TypeError, ValueError):
                    region = ""
        if region:
            covered_regions.append(region)
    covered_regions = list(dict.fromkeys(covered_regions))

    recent_new_counts = [
        len(value.get("_relevant_new_ids", ()))
        for value in query_related_records[-8:]
    ]
    consecutive_no_new = 0
    for record in reversed(query_related_records[-8:]):
        if record.get("_independent") is not True:
            continue
        if record.get("_relevant_new_ids"):
            break
        consecutive_no_new += 1

    ambiguous_ids: set[int] = set()
    for raw_group in snapshot.get(
        "identity_ambiguity_groups",
        snapshot.get("identity_ambiguous_groups", ()),
    ):
        if not isinstance(raw_group, Mapping):
            continue
        for raw_id in raw_group.get("canonical_object_ids", ()):
            canonical_id = canonical_object_id(raw_id)
            if canonical_id is not None:
                ambiguous_ids.add(canonical_id)
    identity_scope = focus_ids or relevant_ids
    pending_identity_ids = sorted(
        object_id
        for object_id in identity_scope
        if object_id in object_by_id
        if (
            object_by_id[object_id].get("status") != "confirmed"
            or object_by_id[object_id].get("identity_association_hypotheses")
            or object_id in ambiguous_ids
        )
    )
    identity_ambiguity_pending_ids = sorted(
        object_id for object_id in identity_scope if object_id in ambiguous_ids
    )

    frontier_regions: list[dict[str, Any]] = []
    for raw_region in snapshot.get("observation_frontier_regions", ()):
        if not isinstance(raw_region, Mapping):
            continue
        points = [
            [float(point[0]), float(point[1])]
            for point in raw_region.get("points_xy", ())
            if isinstance(point, Sequence)
            and not isinstance(point, (str, bytes, bytearray))
            and len(point) >= 2
            and all(math.isfinite(float(value)) for value in point[:2])
        ]
        if not points:
            continue
        points = [point for point in points if point_in_focus(point)]
        if not points:
            continue
        region = dict(raw_region)
        region["points_xy"] = points
        region["representative_xy"] = [
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        ]
        frontier_regions.append(region)

    closure_reasons: list[str] = []
    if len(independent_records) < 2:
        closure_reasons.append("independent_viewpoint_coverage_pending")
    if len(covered_regions) < 2:
        closure_reasons.append("query_region_coverage_pending")
    if consecutive_no_new < 2:
        closure_reasons.append("relevant_discovery_not_saturated")
    if identity_ambiguity_pending_ids:
        closure_reasons.append("relation_identity_ambiguity_pending")
    return {
        "target_class_names": sorted(target_names),
        "anchor_class_names": sorted(anchor_names),
        "candidate_object_ids": target_ids,
        "anchor_object_ids": anchor_ids,
        "focus_object_ids": sorted(focus_ids),
        "focus_radius_m": focus_radius_m if focus_xy else None,
        "confirmed_ids": sorted(
            object_id
            for object_id in identity_scope
            if object_id in object_by_id
            and object_by_id[object_id].get("status") == "confirmed"
        ),
        "identity_ambiguous_groups": [
            dict(group)
            for group in snapshot.get(
                "identity_ambiguity_groups",
                snapshot.get("identity_ambiguous_groups", ()),
            )
            if isinstance(group, Mapping)
            and any(
                canonical_object_id(raw_id) in identity_scope
                for raw_id in group.get("canonical_object_ids", ())
            )
        ],
        "valid_view_count": len(valid_records),
        "query_related_view_count": len(query_related_records),
        "independent_viewpoint_count": len(independent_records),
        "covered_regions": covered_regions,
        "recent_new_relevant_counts": recent_new_counts,
        "consecutive_no_new_relevant_views": consecutive_no_new,
        "pending_identity_ids": pending_identity_ids,
        "identity_ambiguity_pending_ids": identity_ambiguity_pending_ids,
        "frontier_regions": frontier_regions,
        "discovery_saturated": not closure_reasons,
        "closure_reasons": list(dict.fromkeys(closure_reasons)),
        "coverage_score": min(
            1.0,
            len(independent_records) / 2.0,
            len(covered_regions) / 2.0,
            consecutive_no_new / 2.0,
        ),
    }


def _probe_navigation_target(obj: Mapping[str, Any]) -> list[float] | None:
    """Resolve a semantic observation request before waypoint selection."""
    explicit = obj.get("navigation_target_xy")
    if (
        isinstance(explicit, Sequence)
        and not isinstance(explicit, (str, bytes))
        and len(explicit) >= 2
    ):
        try:
            target = [float(explicit[0]), float(explicit[1])]
        except (TypeError, ValueError):
            target = None
        if target is not None and all(math.isfinite(value) for value in target):
            # CountQuery targeted acquisition computes this point from the
            # exact tuple participants.  It must survive later probe-object
            # normalization; falling back to the selected anchor here breaks
            # the required common-observable viewpoint.
            return target
    objective = obj.get("observation_objective", {})
    if not isinstance(objective, Mapping):
        objective = {}
    changing = objective.get("unexplored_regions_that_can_change_result", ())
    if isinstance(changing, Sequence) and changing:
        origin = objective.get("trajectory_continuation_origin_xy")
        direction = objective.get("trajectory_continuation_direction_xy")
        for region in changing:
            representative = region.get("representative_xy")
            if not (
                isinstance(representative, Sequence)
                and len(representative) >= 2
            ):
                continue
            if (
                isinstance(origin, Sequence)
                and len(origin) >= 2
                and isinstance(direction, Sequence)
                and len(direction) >= 2
            ):
                signed_progress = sum(
                    (float(representative[index]) - float(origin[index]))
                    * float(direction[index])
                    for index in (0, 1)
                )
                if signed_progress < 0.0:
                    continue
            return [float(representative[0]), float(representative[1])]
    hypotheses = objective.get("candidate_hypotheses", ())
    if isinstance(hypotheses, Sequence):
        centers = [
            value.get("center_xy") for value in hypotheses
            if isinstance(value, Mapping)
            and isinstance(value.get("center_xy"), Sequence)
            and len(value["center_xy"]) >= 2
        ]
        if centers:
            return [
                sum(float(value[0]) for value in centers) / len(centers),
                sum(float(value[1]) for value in centers) / len(centers),
            ]
    origin = objective.get("trajectory_continuation_origin_xy")
    direction = objective.get("trajectory_continuation_direction_xy")
    if (
        isinstance(origin, Sequence)
        and len(origin) >= 2
        and isinstance(direction, Sequence)
        and len(direction) >= 2
    ):
        dx, dy = float(direction[0]), float(direction[1])
        if math.hypot(dx, dy) > 0.0:
            return [float(origin[0]) + dx, float(origin[1]) + dy]
    # An object center is semantic evidence, not an observation viewpoint.
    # Returning it here made PROBE_EVIDENCE physically chase the current
    # provisional winner.  The waypoint layer will construct a standoff from
    # the object/tuple geometry when no explicit evidence viewpoint exists.
    return None


def _candidate_sources(obj: Mapping[str, Any]) -> list[str]:
    sources = ["semantic_class", "scene_memory"]
    if _relation_bound_evidence_count(obj) > 0:
        sources.append("relation_roi")
    if _valid_center_3d(obj):
        sources.append("metric_3d")
    return sources


def _relation_candidate_priority(
    obj: Mapping[str, Any],
) -> tuple[float, float, int, float, int, int, int]:
    """Order relation work by question evidence, then persistence.

    A confirmed but unrelated same-class instance must not outrank the
    candidate that the relation-local grounding actually selected.  Viewpoint
    count remains useful after relation relevance has been established: a
    strong one-view candidate is probed and then associated across views.
    """
    qwen_probabilities: list[float] = []
    for evidence in obj.get("evidence", ()):
        for raw in evidence.get("qwen_verification_probabilities", ()):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                qwen_probabilities.append(value)
    return (
        max(qwen_probabilities, default=-1.0),
        (
            sum(qwen_probabilities) / len(qwen_probabilities)
            if qwen_probabilities else -1.0
        ),
        int(obj.get("independent_viewpoint_count", 0)),
        float(obj.get("semantic_probability", 0.0)),
        _relation_bound_evidence_count(obj),
        len(obj.get("evidence", ())),
        -int(obj["object_id"]),
    )


def _finite_xy(value: object) -> tuple[float, float] | None:
    """Extract a finite map-frame XY pair from a pose-like value."""
    if isinstance(value, Mapping):
        value = value.get("position_xyz", value.get("position", value.get("xy")))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return xy if all(math.isfinite(component) for component in xy) else None


def _latest_navigation_xy(
    navigation_history: Sequence[Mapping[str, Any]] | None,
) -> tuple[float, float] | None:
    """Return the best current-pose evidence available to query execution.

    This is deliberately derived from the accepted navigation transaction
    history, not from a commanded waypoint.  A failed attempt can still carry
    the robot's latest state-estimation pose, which is useful for selecting a
    different hypothesis on the next acquisition.
    """
    for attempt in reversed(navigation_history or ()):
        if not isinstance(attempt, Mapping):
            continue
        for key in ("actual_arrival_pose", "actual_pose", "start_pose"):
            xy = _finite_xy(attempt.get(key))
            if xy is not None:
                return xy
    return None


def _navigation_attempt_object_ids(
    attempt: Mapping[str, Any],
    *keys: str,
) -> set[int]:
    """Collect canonical object references carried by one dispatched action."""
    contexts = [attempt]
    waypoint_context = attempt.get("waypoint_context")
    if isinstance(waypoint_context, Mapping):
        contexts.append(waypoint_context)
    object_ids: set[int] = set()
    for context in contexts:
        for key in keys:
            raw_values = context.get(key, ())
            if not isinstance(raw_values, Sequence) or isinstance(
                raw_values, (str, bytes, bytearray)
            ):
                raw_values = (raw_values,)
            for raw in raw_values:
                try:
                    object_ids.add(int(raw))
                except (TypeError, ValueError):
                    continue
    return object_ids


def _relation_action_repeat_counts(
    navigation_history: Sequence[Mapping[str, Any]] | None,
    *,
    step_index: int,
    relation_id: str,
    subject_id: int,
    anchor_ids: Sequence[int],
) -> tuple[int, int]:
    """Count completed provisional routes and relation-focused probes.

    A provisional terminal route may produce one arrival re-evaluation.  Once
    that exact subject/anchor hypothesis has physically arrived without
    semantic completion, dispatching the same semantic route again is not a
    new action candidate; the remaining useful action is an evidence probe.
    This is transaction deduplication, not a confidence/deadline gate.
    """
    route_count = 0
    probe_count = 0
    required_anchor_ids = {int(value) for value in anchor_ids}
    pair_ids = {int(subject_id), *required_anchor_ids}
    for raw in navigation_history or ():
        if not isinstance(raw, Mapping):
            continue
        context = raw.get("waypoint_context")
        context = context if isinstance(context, Mapping) else {}
        raw_relation_id = str(
            context.get(
                "selector_relation_id",
                raw.get("selector_relation_id", ""),
            )
        )
        purpose = str(raw.get("purpose", "")).lower()
        attempted_ids = _navigation_attempt_object_ids(
            raw,
            "target_object_ids",
            "anchor_object_ids",
            "probe_object_ids",
            "probe_anchor_object_ids",
            "required_visible_object_ids",
        )
        if purpose == "evidence" and pair_ids.intersection(attempted_ids):
            probe_count += 1
            continue
        try:
            same_step = int(raw.get("step_index", -1)) == int(step_index)
        except (TypeError, ValueError):
            same_step = False
        if purpose != "instruction" or not same_step:
            continue
        if raw_relation_id and raw_relation_id != relation_id:
            continue
        target_ids = _navigation_attempt_object_ids(
            raw, "target_object_ids", "semantic_object_id"
        )
        attempted_anchor_ids = _navigation_attempt_object_ids(
            raw, "anchor_object_ids", "probe_anchor_object_ids"
        )
        physically_arrived = bool(
            raw.get("actual_arrival_pose") is not None
            and str(raw.get("status", "")).lower()
            not in {"dispatched", "failed", "cancelled"}
        )
        if (
            physically_arrived
            and context.get("selector_provisional") is True
            and int(subject_id) in target_ids
            and required_anchor_ids.issubset(attempted_anchor_ids)
            and raw.get("semantic_progressed") is not True
        ):
            route_count += 1
    return route_count, probe_count


def _candidate_qwen_strength(obj: Mapping[str, Any]) -> float:
    values: list[float] = []
    for evidence in obj.get("evidence", ()):
        for raw in evidence.get("qwen_verification_probabilities", ()):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                values.append(value)
    if values:
        return max(values)
    try:
        value = float(obj.get("semantic_probability", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _candidate_geometry_strength(obj: Mapping[str, Any]) -> float:
    """Summarize metric support continuously; never remove a hypothesis."""
    try:
        geometry = max(
            0.0,
            min(1.0, float(obj.get("geometry_confidence", 0.0) or 0.0)),
        )
    except (TypeError, ValueError):
        geometry = 0.0
    try:
        point_count = max(
            0.0,
            float(
                obj.get(
                    "geometry_point_count",
                    obj.get("lidar_support_count", 0),
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        point_count = 0.0
    point_support = min(1.0, math.log1p(point_count) / math.log(41.0))
    provenance = str(
        obj.get("center_covariance_provenance", "")
    ).strip().lower()
    if provenance in {
        "pointcloud_statistical",
        "multi_view_statistical",
        "fused_statistical",
        "statistical",
    }:
        provenance_support = 1.0
    elif provenance in {"bearing_model", "bearing_or_sparse_observation"}:
        provenance_support = 0.12
    elif provenance in {"extent_derived_fallback", "sparse_observation"}:
        provenance_support = 0.55
    else:
        provenance_support = 0.35 if _valid_center_3d(obj) else 0.0
    # A valid metric center is useful even before dense support arrives.  The
    # three terms only change ranking; they do not create a membership gate.
    return max(
        0.0,
        min(
            1.0,
            0.45 * geometry
            + 0.30 * point_support
            + 0.25 * provenance_support,
        ),
    )


def _candidate_failed_attempts(
    obj: Mapping[str, Any],
    navigation_history: Sequence[Mapping[str, Any]] | None,
    *,
    step_index: int,
) -> int:
    """Count failed physical transactions for one semantic hypothesis."""
    try:
        object_id = int(obj.get("object_id", -1))
    except (TypeError, ValueError):
        return 0
    count = 0
    for attempt in navigation_history or ():
        if not isinstance(attempt, Mapping):
            continue
        if str(attempt.get("status", "")) != "failed":
            continue
        try:
            if int(attempt.get("step_index", -1)) != int(step_index):
                continue
        except (TypeError, ValueError):
            continue
        ids: set[int] = set()
        for raw in (
            attempt.get("semantic_object_id"),
            *(attempt.get("target_object_ids", ()) or ()),
        ):
            try:
                ids.add(int(raw))
            except (TypeError, ValueError):
                continue
        context = attempt.get("waypoint_context")
        if isinstance(context, Mapping):
            for raw in (
                context.get("semantic_object_id"),
                *(context.get("target_object_ids", ()) or ()),
            ):
                try:
                    ids.add(int(raw))
                except (TypeError, ValueError):
                    continue
        if object_id in ids:
            count += 1
    return count


def _instruction_candidate_priority(
    obj: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    current_xy: tuple[float, float] | None,
    navigation_history: Sequence[Mapping[str, Any]] | None,
    step_index: int,
) -> tuple[tuple[float, float, float, float, float, float, int], dict[str, Any]]:
    """Rank open-world instruction hypotheses with route-aware evidence.

    Ordinary ordered instructions often have several same-class objects.  A
    pure ``max(Qwen confidence)`` policy can choose a well-observed object
    behind the intended route and then repeatedly fail at the same physical
    target.  This score keeps all candidates alive and combines semantic
    evidence, world-geometry quality, current-pose travel fit, and a soft
    penalty for a failed physical transaction.  No term rejects a candidate.
    """
    semantic = (
        0.68 * _candidate_qwen_strength(obj)
        + 0.32 * max(
            0.0,
            min(1.0, float(obj.get("semantic_probability", 0.0) or 0.0)),
        )
    )
    geometry = _candidate_geometry_strength(obj)
    identity = (
        0.55 * float(obj.get("status") == "confirmed")
        + 0.25 * min(
            1.0,
            max(0.0, float(obj.get("independent_viewpoint_count", 0) or 0))
            / 2.0,
        )
        + 0.20 * float(
            str(obj.get("identity_state", "")).upper()
            in {"CONFIRMED", "TRACKED", "STABLE"}
        )
    )
    center = _finite_xy(obj.get("center_3d"))
    distances: list[float] = []
    if current_xy is not None:
        for candidate in candidates:
            candidate_xy = _finite_xy(candidate.get("center_3d"))
            if candidate_xy is None:
                continue
            distance = math.dist(current_xy, candidate_xy)
            if math.isfinite(distance):
                distances.append(distance)
    distance = (
        math.dist(current_xy, center)
        if current_xy is not None and center is not None
        else None
    )
    if distance is None or not math.isfinite(distance) or not distances:
        route_fit = 0.5
        distance_for_score = None
    else:
        scale = max(1.5, sum(distances) / len(distances))
        route_fit = 0.5 + 0.5 * math.exp(-distance / scale)
        distance_for_score = distance
    failed_attempts = _candidate_failed_attempts(
        obj,
        navigation_history,
        step_index=step_index,
    )
    # Softly move a previously infeasible hypothesis down the queue.  A new
    # observation can still bring it back by improving its other evidence.
    failed_attempt_penalty = min(0.36, 0.14 * float(failed_attempts))
    score = (
        # For an ordered physical instruction, route compatibility is the
        # strongest *soft* disambiguator.  Semantic and metric evidence still
        # keep weak/bearing-only tracks below a well-grounded hypothesis.
        0.26 * semantic
        + 0.13 * geometry
        + 0.03 * identity
        + 0.58 * route_fit
        - failed_attempt_penalty
    )
    try:
        object_id = int(obj.get("object_id", -1))
    except (TypeError, ValueError):
        object_id = -1
    details = {
        "object_id": object_id,
        "score": float(score),
        "semantic_strength": float(semantic),
        "geometry_strength": float(geometry),
        "identity_strength": float(identity),
        "route_fit": float(route_fit),
        "distance_from_current_pose_m": (
            float(distance_for_score) if distance_for_score is not None else None
        ),
        "failed_navigation_attempts": int(failed_attempts),
        "failed_attempt_penalty": float(failed_attempt_penalty),
    }
    return (
        (
            float(score),
            float(route_fit),
            float(semantic),
            float(geometry),
            float(identity),
            -float(distance_for_score)
            if distance_for_score is not None
            else float("-inf"),
            -object_id,
        ),
        details,
    )


def _probe_priority(obj: Mapping[str, Any]) -> tuple[int, float, int, bool, int]:
    """Prefer the anchor grounded by the question, then existing evidence."""
    return (
        _semantic_evidence_count(obj),
        float(obj.get("semantic_probability", 0.0)),
        len(obj.get("evidence", ())),
        obj.get("status") == "confirmed",
        -int(obj["object_id"]),
    )


def _bounds(obj: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    center = [
        float(value)
        for value in obj["center_3d"]
    ]
    size = [
        float(value)
        for value in obj["bbox_3d"]
    ]
    half = [0.5 * value for value in size]
    return ([center[i] - half[i] for i in range(3)], [center[i] + half[i] for i in range(3)])


def _horizontal_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    a0, a1 = _bounds(first)
    b0, b1 = _bounds(second)
    return a0[0] <= b1[0] and b0[0] <= a1[0] and a0[1] <= b1[1] and b0[1] <= a1[1]


def _geometric_relation_feature(predicate: str, subject: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]]) -> bool:
    predicate = predicate.lower()
    if predicate == "near" and len(anchors) == 1:
        return True
    if predicate in {"above", "below", "on", "in"} and len(anchors) == 1:
        subject_low, subject_high = _bounds(subject)
        anchor_low, anchor_high = _bounds(anchors[0])
        if predicate == "above":
            return subject["center_3d"][2] > anchors[0]["center_3d"][2]
        if predicate == "below":
            return subject["center_3d"][2] < anchors[0]["center_3d"][2]
        if predicate == "on":
            return (
                _horizontal_overlap(subject, anchors[0])
                and subject["center_3d"][2] > anchors[0]["center_3d"][2]
            )
        return all(
            subject_low[i] >= anchor_low[i]
            and subject_high[i] <= anchor_high[i]
            for i in range(3)
        )
    if predicate == "between" and len(anchors) == 2:
        point = subject["center_3d"][:2]
        first = anchors[0]["center_3d"][:2]
        second = anchors[1]["center_3d"][:2]
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 0.0:
            return False
        t = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / length_squared
        projected = (first[0] + t * dx, first[1] + t * dy)
        perpendicular = math.hypot(point[0] - projected[0], point[1] - projected[1])
        corridor = 0.5 * max(
            math.hypot(*anchors[0]["bbox_3d"][:2]),
            math.hypot(*anchors[1]["bbox_3d"][:2]),
        )
        return 0.0 <= t <= 1.0 and perpendicular <= corridor
    return False


def _clamped_probability(value: object, default: float = 0.0) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = float(default)
    if not math.isfinite(probability):
        probability = float(default)
    return max(0.0, min(1.0, probability))


def _numerical_class_probability(obj: Mapping[str, Any]) -> float:
    """Return the strongest continuous class hypothesis for one object.

    Same-station Qwen and YOLO boxes are alternate proposals for the same
    physical instance. Their verifier scores are therefore alternatives, not
    independent objects and not votes weighted by mask size.
    """
    probabilities = [
        _clamped_probability(obj.get("semantic_probability", 0.0))
    ]
    for evidence in obj.get("evidence", ()):
        if not isinstance(evidence, Mapping):
            continue
        for value in dict(
            evidence.get("proposal_verification_by_key", {})
        ).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                probabilities.append(_clamped_probability(value))
        for value in evidence.get("qwen_verification_probabilities", ()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                probabilities.append(_clamped_probability(value))
    return max(probabilities, default=0.0)


def _numerical_object_probability(obj: Mapping[str, Any]) -> float:
    """Return the current graph-membership probability for one object.

    Once a relation resolver has attached a membership probability it already
    contains the class and relation terms accumulated so far.  Re-reading raw
    proposal scores here would incorrectly erase that relation evidence.
    """
    if "numerical_membership_probability" in obj:
        return _clamped_probability(obj["numerical_membership_probability"])
    return _numerical_class_probability(obj)


def _view_box_records(obj: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: dict[tuple[object, ...], dict[str, Any]] = {}
    for evidence in obj.get("evidence", ()):
        if not isinstance(evidence, Mapping):
            continue
        for raw in evidence.get("source_view_boxes", ()):
            if not isinstance(raw, Mapping):
                continue
            try:
                view_id = str(raw.get("view_id", ""))
                box = tuple(float(value) for value in raw.get("bbox_xyxy", ()))
                width = int(raw.get("image_width", 0))
                height = int(raw.get("image_height", 0))
            except (TypeError, ValueError):
                continue
            if not view_id or len(box) != 4 or width <= 0 or height <= 0:
                continue
            key = (view_id, *(round(value, 3) for value in box), width, height)
            records[key] = {
                "view_id": view_id,
                "bbox_xyxy": box,
                "image_width": width,
                "image_height": height,
            }
    return list(records.values())


def _image_on_probability(
    subject: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> float:
    best = 0.0
    subject_boxes = _view_box_records(subject)
    anchor_boxes = _view_box_records(anchor)
    for first in subject_boxes:
        for second in anchor_boxes:
            if first["view_id"] != second["view_id"]:
                continue
            width = min(first["image_width"], second["image_width"])
            height = min(first["image_height"], second["image_height"])
            if width <= 0 or height <= 0:
                continue
            sx1, sy1, sx2, sy2 = first["bbox_xyxy"]
            ax1, ay1, ax2, ay2 = second["bbox_xyxy"]
            subject_width = max(1.0, sx2 - sx1)
            subject_height = max(1.0, sy2 - sy1)
            anchor_width = max(1.0, ax2 - ax1)
            anchor_height = max(1.0, ay2 - ay1)
            horizontal_gap = max(ax1 - sx2, sx1 - ax2, 0.0)
            contact_gap = abs(sy2 - ay1)
            horizontal_scale = max(
                2.0, 0.35 * subject_width, 0.03 * anchor_width
            )
            contact_scale = max(
                2.0, 0.35 * subject_height, 0.04 * anchor_height
            )
            subject_center_y = 0.5 * (sy1 + sy2)
            anchor_center_y = 0.5 * (ay1 + ay2)
            order_gap = max(0.0, subject_center_y - anchor_center_y)
            score = (
                math.exp(-0.5 * (horizontal_gap / horizontal_scale) ** 2)
                * math.exp(-0.5 * (contact_gap / contact_scale) ** 2)
                * math.exp(-0.5 * (order_gap / contact_scale) ** 2)
            )
            best = max(best, score)
    return _clamped_probability(best)


def _geometry_variance(obj: Mapping[str, Any], axis: int) -> float:
    variance = 0.0
    for key, scale in (("center_cov", 1.0), ("extent_cov", 0.25)):
        covariance = obj.get(key)
        try:
            value = float(covariance[axis][axis])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(value) and value > 0.0:
            variance += scale * value
    return variance


def _horizontal_distance_interval(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    """Return horizontal distance, uncertainty, and conservative bounds."""
    distance = _distance(first, second)
    variance = sum(
        _geometry_variance(value, axis)
        for value in (first, second)
        for axis in (0, 1)
    )
    uncertainty = math.sqrt(variance) if variance > 0.0 else 0.25
    return (
        float(distance),
        float(uncertainty),
        max(0.0, float(distance) - 2.0 * float(uncertainty)),
        float(distance) + 2.0 * float(uncertainty),
    )


def _metric_vertical_relation_probability(
    predicate: str,
    subject: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> float:
    subject_low, subject_high = _bounds(subject)
    anchor_low, anchor_high = _bounds(anchor)
    gap_x = max(anchor_low[0] - subject_high[0], subject_low[0] - anchor_high[0], 0.0)
    gap_y = max(anchor_low[1] - subject_high[1], subject_low[1] - anchor_high[1], 0.0)
    horizontal_scale = max(
        0.05,
        math.sqrt(
            _geometry_variance(subject, 0)
            + _geometry_variance(subject, 1)
            + _geometry_variance(anchor, 0)
            + _geometry_variance(anchor, 1)
        ),
    )
    horizontal = math.exp(
        -0.5 * (math.hypot(gap_x, gap_y) / horizontal_scale) ** 2
    )
    vertical_scale = max(
        0.08,
        math.sqrt(
            _geometry_variance(subject, 2)
            + _geometry_variance(anchor, 2)
        ),
    )
    signed_center_delta = (
        float(subject["center_3d"][2]) - float(anchor["center_3d"][2])
    )
    if predicate == "above":
        ordering = 0.5 * (1.0 + math.tanh(signed_center_delta / vertical_scale))
        return _clamped_probability(horizontal * ordering)
    if predicate == "below":
        ordering = 0.5 * (1.0 - math.tanh(signed_center_delta / vertical_scale))
        return _clamped_probability(horizontal * ordering)
    contact_gap = abs(subject_low[2] - anchor_high[2])
    contact = math.exp(-0.5 * (contact_gap / vertical_scale) ** 2)
    ordering = 0.5 * (1.0 + math.tanh(signed_center_delta / vertical_scale))
    return _clamped_probability(horizontal * contact * ordering)


def _numerical_predicate_probability(
    predicate: str,
    subject: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
) -> float:
    normalized = str(predicate).lower()
    if normalized in {"above", "below", "on"} and len(anchors) == 1:
        metric = _metric_vertical_relation_probability(
            normalized, subject, anchors[0]
        )
        if normalized == "on":
            return max(metric, _image_on_probability(subject, anchors[0]))
        return metric
    if normalized == "near" and len(anchors) == 1:
        distance = _distance(subject, anchors[0])
        subject_bbox = subject.get("bbox_3d") or (0.0, 0.0)
        anchor_bbox = anchors[0].get("bbox_3d") or (0.0, 0.0)
        scale = max(
            0.50,
            0.5 * math.hypot(*subject_bbox[:2])
            + 0.5 * math.hypot(*anchor_bbox[:2]),
        )
        return _clamped_probability(math.exp(-0.5 * (distance / scale) ** 2))
    return float(_geometric_relation_feature(normalized, subject, anchors))


def _numerical_cardinality_posterior(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probabilities = [
        _numerical_object_probability(candidate) for candidate in candidates
    ]
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - probability)
            updated[count + 1] += mass * probability
        distribution = updated
    expected = sum(probabilities)
    peak = max(distribution, default=1.0)
    modes = [
        count for count, mass in enumerate(distribution)
        if abs(mass - peak) <= 1e-12
    ]
    mode = min(modes, key=lambda count: (abs(count - expected), count))
    return {
        "role": "candidate_diagnostic_only",
        "mode": int(mode),
        "expected_count": float(expected),
        "distribution": [float(value) for value in distribution],
        "candidate_membership_probabilities": {
            str(candidate.get("object_id", index)): float(probability)
            for index, (candidate, probability) in enumerate(
                zip(candidates, probabilities)
            )
        },
    }


def _numerical_coverage_estimator(
    snapshot: Mapping[str, Any],
    graph_execution: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate residual unseen-target risk for acquisition selection only."""
    closure = snapshot.get("count_query_domain", {})
    if not isinstance(closure, Mapping):
        closure = {}
    history = [
        value for value in snapshot.get("viewpoint_history", ())
        if isinstance(value, Mapping)
        and value.get("valid_for_count_closure") is not False
    ]
    independent = sum(
        bool(value.get("independent_viewpoint")) for value in history
    )
    regions = len({
        str(value.get("coverage_region", ""))
        for value in history
        if str(value.get("coverage_region", "")).strip()
    })
    # CountClosureState already projects discovery onto this query's target
    # entity.  Reusing its values is important: the coverage estimator is
    # allowed to control acquisition, but it must not mistake an unrelated
    # class discovered at the same station for a new target instance.
    recent = [
        max(0, int(value))
        for value in closure.get("recent_new_instance_counts", ())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ][-6:]
    if not recent:
        target_ids = set()
        candidate_domains = (graph_execution or {}).get("candidate_domains", {})
        target_entity = str((graph_execution or {}).get("target_entity", ""))
        if isinstance(candidate_domains, Mapping):
            try:
                target_ids = {
                    int(value)
                    for value in candidate_domains.get(target_entity, ())
                }
            except (TypeError, ValueError):
                target_ids = set()
        for value in history[-6:]:
            new_ids = value.get("new_instance_ids", ())
            if not isinstance(new_ids, Sequence) or isinstance(
                new_ids, (str, bytes, bytearray)
            ):
                recent.append(0)
                continue
            recent.append(sum(
                1 for item in new_ids
                if str(item).lstrip("-").isdigit() and int(item) in target_ids
            ))
    no_new_streak = 0
    if isinstance(closure.get("consecutive_no_new_views"), (int, float)):
        no_new_streak = max(0, int(closure.get("consecutive_no_new_views", 0)))
    else:
        for value, new_count in zip(reversed(history[-6:]), reversed(recent)):
            if value.get("independent_viewpoint") is not True:
                continue
            if new_count != 0:
                break
            no_new_streak += 1
    unknown_target_count = len(
        (graph_execution or {}).get("unknown_target_ids", ())
    )
    unknown_relation_count = len(
        closure.get("unknown_relation_ids", ())
    )
    ambiguous_count = len(closure.get("identity_ambiguous_groups", ()))
    unconfirmed_count = len(closure.get("unconfirmed_candidate_ids", ()))
    valid_view_count = len(history)
    discovery_rate = sum(recent) / max(1, len(recent))
    saturation = min(1.0, no_new_streak / 2.0)
    viewpoint_coverage = min(1.0, independent / 2.0)
    region_coverage = min(1.0, regions / 2.0)
    residual_risk = (
        0.48 * (1.0 - saturation)
        + 0.22 * (1.0 - viewpoint_coverage)
        + 0.15 * (1.0 - region_coverage)
        + 0.07 * min(1.0, unknown_target_count / 2.0)
        + 0.05 * min(1.0, ambiguous_count / 2.0)
        + 0.03 * min(1.0, unconfirmed_count / 2.0)
    )
    residual_risk = max(0.0, min(1.0, residual_risk))
    return {
        "role": "acquisition_control_only",
        "p_more_unseen_target": float(residual_risk),
        "discovery_rate": float(discovery_rate),
        "recent_new_instance_counts": recent,
        "consecutive_no_new_views": int(no_new_streak),
        "independent_viewpoint_count": int(independent),
        "covered_region_count": int(regions),
        "valid_view_count": int(valid_view_count),
        "unknown_target_count": int(unknown_target_count),
        "unknown_relation_count": int(unknown_relation_count),
        "identity_ambiguity_count": int(ambiguous_count),
        "unconfirmed_candidate_count": int(unconfirmed_count),
        "candidate_count": len(candidates),
        "coverage_score": float(closure.get("coverage_score", 0.0) or 0.0),
    }


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _numerical_count_readiness(
    snapshot: Mapping[str, Any],
    failures: Sequence[str],
) -> dict[str, Any]:
    """Read the sole numerical answer authority: the Count Query Graph."""
    def ready(
        answer: object,
        *,
        source: str,
        evidence_ids: Sequence[object] = (),
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        normalized = _nonnegative_integer(answer)
        if normalized is None:
            return None
        return {
            "ready": True,
            "state": "READY",
            "answer": normalized,
            "source": str(source),
            "evidence_ids": list(dict.fromkeys(
                str(value) for value in evidence_ids if str(value).strip()
            )),
            "reasons": [],
            "diagnostics": [],
            "detail": dict(detail or {}),
        }

    graph_execution = snapshot.get("count_query_execution")
    if isinstance(graph_execution, Mapping) and (
        graph_execution.get("complete") is True
        or graph_execution.get("cardinality_complete") is True
    ):
        graph_ready = ready(
            graph_execution.get("answer"),
            source=str(
                graph_execution.get(
                    "cardinality_resolution_mode", "count_query_graph"
                )
            ),
            evidence_ids=graph_execution.get("evidence_ids", ()),
            detail={
                "kind": "count_query_execution",
                "scene_version": graph_execution.get("scene_version"),
            },
        )
        if graph_ready is not None:
            return graph_ready

    reasons = ["count_answer_evidence_not_ready"]
    if isinstance(graph_execution, Mapping):
        reasons.extend(
            str(value)
            for value in graph_execution.get("failure_reasons", ())
            if str(value).strip()
        )
        if not (
            graph_execution.get("complete") is True
            or graph_execution.get("cardinality_complete") is True
        ):
            reasons.append("count_query_execution_incomplete")
    diagnostics = []
    if snapshot.get("scene_memory_open", True) is True:
        diagnostics.append("scene_memory_open_is_diagnostic_only")
    if snapshot.get("relation_selector_domain_open") is True:
        diagnostics.append("relation_selector_domain_open_is_not_count_authority")
    if failures:
        reasons.extend(str(value) for value in failures)
    return {
        "ready": False,
        "state": "OPEN_PENDING_COUNT_AUTHORITY",
        "answer": None,
        "source": None,
        "evidence_ids": [],
        "reasons": list(dict.fromkeys(reasons)),
        "diagnostics": list(dict.fromkeys(diagnostics)),
        "detail": {"kind": "count_query_graph_incomplete"},
    }


class _Resolver:
    def __init__(self, task_ir: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        self.task_ir = task_ir
        self.snapshot = snapshot
        self.entities = {str(item["id"]): item for item in task_ir["entities"]}
        self.relations = list(task_ir["relations"])
        self.objects = [
            item for item in snapshot["objects"]
            if item.get("status") in {"tentative", "confirmed"}
            and str(item.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
            and item.get("center_3d")
        ]
        self.cache: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
        self.stack: set[str] = set()

    def resolve(self, entity_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        if entity_id in self.cache:
            return self.cache[entity_id]
        if entity_id in self.stack:
            return [], [f"relation_dependency_cycle:{entity_id}"]
        entity = self.entities.get(entity_id)
        if entity is None:
            return [], [f"entity_missing:{entity_id}"]
        self.stack.add(entity_id)
        class_names = _task_entity_class_names(self.task_ir, entity)
        candidates = [
            item for item in self.objects
            if _matches_class_names(item, class_names)
            and _attribute_matches(entity, item, self.objects, class_names)
        ]
        failures: list[str] = []
        for relation in self.relations:
            if relation["subject_entity"] != entity_id:
                continue
            anchor_sets = []
            for anchor_id in relation["object_entities"]:
                anchors, anchor_failures = self.resolve(str(anchor_id))
                failures.extend(anchor_failures)
                if not anchors:
                    failures.append(f"relation_anchor_missing:{relation['id']}:{anchor_id}")
                    anchor_sets.append([])
                    continue
                # Pass all anchors so the predicate check can try each
                # one instead of locking onto the highest-confidence pick.
                anchor_sets.append(sorted(
                    anchors,
                    key=_probe_priority,
                    reverse=True,
                ))
            predicate = str(relation["predicate"]).lower()
            selector_anchor_missing = any(not values for values in anchor_sets)
            if failures or selector_anchor_missing:
                candidates = []
                break
            if predicate in {"closest", "farthest"}:
                anchors_flat = [value for values in anchor_sets for value in values]
                if not anchors_flat or not candidates:
                    failures.append(f"ranking_relation_unresolved:{relation['id']}")
                    candidates = []
                    break
                selector = (
                    self.snapshot.get("query_domain", {})
                    .get("selector_closures", {})
                    .get(str(relation.get("id", "")), {})
                )
                selected_object_id = selector.get("selected_object_id")
                selected_anchor_hint = selector.get("selected_anchor_id")
                selected_anchor_id = selected_anchor_hint
                selected = next(
                    (
                        item for item in candidates
                        if selected_object_id is not None
                        and int(item.get("object_id", -1))
                        == int(selected_object_id)
                    ),
                    None,
                )
                selected_anchor = next(
                    (
                        item for item in anchors_flat
                        if selected_anchor_id is not None
                        and int(item.get("object_id", -1))
                        == int(selected_anchor_id)
                    ),
                    None,
                )
                if selected is None or selected_anchor is None:
                    pair_ranked = [
                        (
                            _distance(item, anchor),
                            int(item["object_id"]),
                            int(anchor["object_id"]),
                            item,
                            anchor,
                        )
                        for item in candidates
                        for anchor in anchors_flat
                    ]
                    selected_pair = (
                        max(
                            pair_ranked,
                            key=lambda value: (value[0], -value[1], -value[2]),
                        )
                        if predicate == "farthest"
                        else min(
                            pair_ranked,
                            key=lambda value: (value[0], value[1], value[2]),
                        )
                    )
                    selected = selected_pair[3]
                    selected_anchor = selected_pair[4]
                candidates = [selected]
            else:
                # Try all anchor combinations: small-object 3D is noisy so
                # the highest-confidence anchor is not always the spatially
                # correct one.
                filtered: list[dict[str, Any]] = []
                for item in candidates:
                    for combo in itertools.product(*anchor_sets):
                        if _geometric_relation_feature(predicate, item, list(combo)):
                            filtered.append(item)
                            break
                seen_ids = set()
                deduped = []
                for obj in filtered:
                    oid = int(obj.get("object_id", -1))
                    if oid not in seen_ids:
                        seen_ids.add(oid)
                        deduped.append(obj)
                candidates = deduped
        self.stack.remove(entity_id)
        result = (candidates, list(dict.fromkeys(failures)))
        self.cache[entity_id] = result
        return result


RelationTupleVerifier = Callable[
    [Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


class _ObjectReferenceRelationResolver:
    """Resolve nested object references in dependency order without products."""

    def __init__(
        self,
        task_ir: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        verifier: RelationTupleVerifier | None,
        entity_bindings: Mapping[str, int] | None = None,
        navigation_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.task_ir = task_ir
        self.snapshot = snapshot
        self.entities = {
            str(item["id"]): item for item in task_ir.get("entities", ())
        }
        self.relations = list(task_ir.get("relations", ()))
        self.objects = [
            item for item in snapshot.get("objects", ())
            if item.get("status") in {"tentative", "confirmed"}
            and str(item.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
            and item.get("center_3d")
        ]
        self.verifier = verifier
        self.entity_bindings = {
            str(entity_id): int(object_id)
            for entity_id, object_id in (entity_bindings or {}).items()
        }
        self.cache: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
        self.stack: set[str] = set()
        self.verifications: list[dict[str, Any]] = []
        self.candidate_states: dict[tuple[str, int], str] = {}
        self.direct_candidate_states: dict[tuple[str, int], str] = {}
        self.relation_attempts: list[dict[str, Any]] = []
        self.relation_anchor_bindings: list[dict[str, Any]] = []
        self.candidate_domains: dict[str, list[dict[str, Any]]] = {}
        self.selector_resolutions: list[dict[str, Any]] = []
        self.selector_anchor_sources: dict[str, str] = {}
        self.navigation_history = [
            dict(value)
            for value in (navigation_history or ())
            if isinstance(value, Mapping)
        ]

    @staticmethod
    def _probe_anchor_key(
        value: Mapping[str, Any],
    ) -> tuple[int, ...] | None:
        raw = value.get("probe_anchor_object_ids")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        try:
            ids = tuple(sorted({int(item) for item in raw}))
        except (TypeError, ValueError):
            return None
        return ids if ids else None

    def _failed_probe_anchor_keys(self) -> set[tuple[int, ...]]:
        return {
            key
            for attempt in self.navigation_history
            if str(attempt.get("purpose", "")) == "evidence"
            and str(attempt.get("status", "")) == "failed"
            for key in (self._probe_anchor_key(attempt),)
            if key is not None
        }

    def _probe_locus_visit_count(self, object_id: object) -> int:
        """Count completed information-gathering visits to one semantic locus."""
        try:
            expected_id = int(object_id)
        except (TypeError, ValueError):
            return 0
        count = 0
        for attempt in self.navigation_history:
            if str(attempt.get("purpose", "")) != "evidence":
                continue
            if str(attempt.get("status", "")) not in {"arrived", "failed"}:
                continue
            raw_id = attempt.get("semantic_object_id")
            if raw_id is None and isinstance(
                attempt.get("waypoint_context"), Mapping
            ):
                raw_id = attempt["waypoint_context"].get(
                    "semantic_object_id"
                )
            try:
                if int(raw_id) == expected_id:
                    count += 1
            except (TypeError, ValueError):
                continue
        return count

    def _subject_relations(self, entity_id: str) -> list[Mapping[str, Any]]:
        return [
            relation for relation in self.relations
            if str(relation.get("subject_entity", "")) == entity_id
        ]

    def _class_candidates(self, entity_id: str) -> list[dict[str, Any]]:
        entity = self.entities[entity_id]
        names = _task_entity_class_names(self.task_ir, entity)
        candidates = [
            dict(item) for item in self.objects
            if _matches_class_names(item, names)
            and _attribute_matches(entity, item, self.objects, names)
        ]
        bound_object_id = self.entity_bindings.get(entity_id)
        if bound_object_id is not None:
            bound_candidates = [
                value for value in candidates
                if int(value["object_id"]) == bound_object_id
            ]
            if bound_candidates:
                candidates = bound_candidates
        relations = self._subject_relations(entity_id)
        specs = [
            relation_spec(str(relation.get("predicate", "")))
            for relation in relations
        ]
        # ROI origin is provenance and scheduling evidence, never membership.
        # Selectors require a metric center but range over the full persistent
        # class domain. Relation filters also start from the full class domain
        # and let exact-tuple verification decide membership.
        if any(spec.operator_kind is OperatorKind.SET_SELECTOR for spec in specs):
            candidates = [value for value in candidates if _valid_center_3d(value)]
        if any(
            spec.operator_kind is OperatorKind.RELATION_FILTER for spec in specs
        ):
            candidates.sort(key=_relation_candidate_priority, reverse=True)
        else:
            candidates.sort(
                key=lambda value: (
                    int(value.get("independent_viewpoint_count", 0)),
                    float(value.get("semantic_probability", 0.0)),
                    len(value.get("evidence", ())),
                    -int(value["object_id"]),
                ),
                reverse=True,
            )
        self.candidate_domains[entity_id] = [
            {
                "object_id": int(value["object_id"]),
                "sources": _candidate_sources(value),
                "rankable": _valid_center_3d(value),
            }
            for value in candidates
        ]
        return candidates

    @staticmethod
    def _evidence_view_tokens(obj: Mapping[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for evidence in obj.get("evidence", ()):
            for key in (
                "source_view_ids",
                "source_proposal_binding_keys",
                "relation_roi_anchor_binding_keys",
            ):
                tokens.update(
                    str(value) for value in evidence.get(key, ()) if str(value)
                )
            for key in (
                "acquisition_id",
                "representative_view_id",
                "relation_roi_view_id",
            ):
                value = str(evidence.get(key, "")).strip()
                if value:
                    tokens.add(value)
        return tokens

    def _bind_one_anchor(
        self,
        candidate: Mapping[str, Any] | None,
        anchors: Sequence[Mapping[str, Any]],
        predicate: str = "",
    ) -> dict[str, Any] | None:
        """Bind one physical anchor without enumerating anchor products."""
        if len(anchors) == 1:
            return dict(anchors[0])
        if not anchors:
            return None
        candidate_tokens = (
            self._evidence_view_tokens(candidate) if candidate is not None else set()
        )
        scored = []
        for anchor in anchors:
            shared = len(candidate_tokens & self._evidence_view_tokens(anchor))
            explicit = _relation_bound_evidence_count(anchor)
            scored.append((shared, explicit, dict(anchor)))
        scored.sort(
            key=lambda value: (
                value[0], value[1], _probe_priority(value[2])
            ),
            reverse=True,
        )
        best = scored[0]
        if best[0] <= 0 and best[1] <= 0:
            return None
        if len(scored) > 1 and best[:2] == scored[1][:2]:
            return None
        return best[2]

    def _selector_anchor(
        self,
        relation_id: str,
        anchor_entity_id: str,
        anchors: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        predicate: str,
    ) -> dict[str, Any] | None:
        explicitly_bound_id = self.entity_bindings.get(anchor_entity_id)
        if explicitly_bound_id is not None:
            if explicitly_bound_id < 0:
                self.selector_anchor_sources[relation_id] = (
                    "qwen_reference_entity_not_visible"
                )
                return None
            explicitly_bound = next((
                dict(value) for value in anchors
                if int(value["object_id"]) == explicitly_bound_id
            ), None)
            if explicitly_bound is not None:
                self.selector_anchor_sources[relation_id] = (
                    "qwen_reference_entity_binding"
                )
                return explicitly_bound
        provenance_bound = []
        for candidate in candidates:
            bound = self._bind_from_candidate_provenance(
                candidate, [anchors], predicate
            )
            if bound:
                provenance_bound.append(dict(bound[0]))
        provenance_ids = {
            int(value["object_id"]) for value in provenance_bound
        }
        if len(provenance_ids) == 1:
            selected_id = next(iter(provenance_ids))
            self.selector_anchor_sources[relation_id] = (
                "candidate_provenance_binding"
            )
            return next(
                dict(value) for value in anchors
                if int(value["object_id"]) == selected_id
            )
        evidence_bound = self._bind_one_anchor(None, anchors)
        if evidence_bound is not None:
            self.selector_anchor_sources[relation_id] = (
                "persistent_evidence_binding"
            )
            return evidence_bound
        if anchors:
            # A missing joint selector winner is uncertainty, not permission to
            # preselect the highest-quality raw anchor.  The open selector
            # policy will provide a joint hypothesis or an evidence probe.
            self.selector_anchor_sources[relation_id] = (
                "joint_selector_winner_missing"
            )
            return None
        return None

    def _bind_linear_anchors(
        self,
        candidate: Mapping[str, Any],
        anchor_sets: Sequence[Sequence[Mapping[str, Any]]],
        predicate: str = "",
    ) -> list[dict[str, Any]] | None:
        """Choose at most one evidence-bound object for every parameter role."""
        bound = [
            self._bind_one_anchor(candidate, anchors, predicate)
            for anchors in anchor_sets
        ]
        if any(value is None for value in bound):
            if str(predicate).lower() == "between" and len(anchor_sets) == 2:
                return self._geometric_bind_between(candidate, anchor_sets)
            return self._geometric_bind_single(
                candidate, anchor_sets, predicate
            )
        return [dict(value) for value in bound if value is not None]

    def _geometric_bind_single(
        self,
        candidate: Mapping[str, Any],
        anchor_sets: Sequence[Sequence[Mapping[str, Any]]],
        predicate: str,
    ) -> list[dict[str, Any]] | None:
        """Bind metric-relation anchors from map-frame geometry alone.

        Provenance binding requires shared observation view tokens and misses
        pairs co-observed at one station in different perspective tiles.  For
        metric predicates the reconstructed map frame is the natural binding
        domain; the tuple still goes through exact-ID Qwen verification.
        """
        normalized = str(predicate).lower()
        if (
            normalized not in {"near", "on", "above", "below", "in"}
            or not _valid_center_3d(candidate)
            or len(anchor_sets) != 1
        ):
            return None
        scored: list[tuple[float, dict[str, Any]]] = []
        for anchor in anchor_sets[0]:
            if not _valid_center_3d(anchor):
                continue
            if not _geometric_relation_feature(normalized, candidate, [anchor]):
                continue
            scored.append((float(_distance(candidate, anchor)), dict(anchor)))
        if not scored:
            return None
        scored.sort(
            key=lambda value: (value[0], _probe_priority(value[1]))
        )
        best = scored[0]
        if len(scored) > 1 and math.isclose(
            best[0], scored[1][0], abs_tol=0.05
        ):
            return None
        return [best[1]]

    def _geometric_bind_between(
        self,
        candidate: Mapping[str, Any],
        anchor_sets: Sequence[Sequence[Mapping[str, Any]]],
    ) -> list[dict[str, Any]] | None:
        """Bind the strongest corridor-plausible BETWEEN anchor pair."""
        pairs = self._geometric_between_pairs(candidate, anchor_sets, limit=1)
        if not pairs:
            return None
        return list(pairs[0][2])

    def _geometric_between_pairs(
        self,
        candidate: Mapping[str, Any],
        anchor_sets: Sequence[Sequence[Mapping[str, Any]]],
        *,
        limit: int,
    ) -> list[tuple[float, float, tuple[dict[str, Any], dict[str, Any]]]]:
        """Enumerate corridor-plausible BETWEEN anchor pairs.

        Enumerates first-role x second-role candidates (bounded by the class
        domains) and returns the best ``limit`` pairs whose segment corridor
        contains the subject.  Every returned tuple still goes through
        exact-ID Qwen verification.
        """
        if not _valid_center_3d(candidate):
            return []
        scored: list[tuple[Any, ...]] = []
        for first in anchor_sets[0]:
            if not _valid_center_3d(first):
                continue
            for second in anchor_sets[1]:
                if not _valid_center_3d(second):
                    continue
                if not _geometric_relation_feature(
                    "between", candidate, [first, second]
                ):
                    continue
                point = candidate["center_3d"][:2]
                first_center = first["center_3d"][:2]
                second_center = second["center_3d"][:2]
                dx, dy = (
                    second_center[0] - first_center[0],
                    second_center[1] - first_center[1],
                )
                length_squared = dx * dx + dy * dy
                if length_squared <= 0.0:
                    continue
                position = (
                    (point[0] - first_center[0]) * dx
                    + (point[1] - first_center[1]) * dy
                ) / length_squared
                projected = (
                    first_center[0] + position * dx,
                    first_center[1] + position * dy,
                )
                perpendicular = math.hypot(
                    point[0] - projected[0], point[1] - projected[1]
                )
                centrality = min(position, 1.0 - position)
                scored.append((
                    centrality,
                    -perpendicular,
                    _probe_priority(first),
                    _probe_priority(second),
                    dict(first),
                    dict(second),
                ))
        scored.sort(key=lambda value: value[:4], reverse=True)
        return [
            (float(value[0]), float(value[1]), (value[4], value[5]))
            for value in scored[: max(1, int(limit))]
        ]

    def _node(self, relation: Mapping[str, Any]) -> dict[str, Any]:
        predicate = str(relation["predicate"]).lower()
        spec = relation_spec(predicate)
        argument_ids = [
            str(relation["subject_entity"]),
            *(str(value) for value in relation.get("object_entities", ())),
        ]
        grounding_by_entity = {
            str(item.get("entity_id", "")): item
            for item in self.task_ir.get("grounding_plan", {}).get(
                "entities", ()
            )
        }
        descriptions = []
        for entity_id in argument_ids:
            entity = self.entities[entity_id]
            grounding = grounding_by_entity.get(entity_id, {})
            descriptions.append(str(
                grounding.get("visual_definition", entity["class_name"])
            ))
        return {
            "id": str(relation["id"]),
            "predicate": predicate,
            "parameter_roles": list(spec.parameter_roles),
            "directed": spec.directed,
            "symmetric": spec.symmetric,
            "evidence_policy": spec.evidence_policy,
            "geometry_policy": spec.geometry_policy,
            "negative_evidence_policy": spec.negative_evidence_policy,
            "qwen_instruction": spec.qwen_instruction,
            "argument_descriptions": descriptions,
        }

    @staticmethod
    def _bind_from_candidate_provenance(
        candidate: Mapping[str, Any],
        anchor_sets: Sequence[Sequence[Mapping[str, Any]]],
        predicate: str = "",
    ) -> list[dict[str, Any]] | None:
        """Recover the exact role proposals used to generate this candidate."""
        def anchor_for_key(
            anchors: Sequence[Mapping[str, Any]], binding_key: str
        ) -> dict[str, Any] | None:
            for anchor in anchors:
                for anchor_evidence in anchor.get("evidence", ()):
                    if binding_key not in {
                        str(value)
                        for value in anchor_evidence.get(
                            "source_proposal_binding_keys", ()
                        )
                    }:
                        continue
                    # Grounding and independent box verification can disagree.
                    # Preserve that disagreement as evidence for the later
                    # role-aware relation verifier; a scalar zero must not erase
                    # the exact proposal binding or force an unrelated anchor.
                    bound = dict(anchor)
                    # The persistent object can contain observations from many
                    # stations and proposal roles.  Once this exact binding key
                    # matches, downstream relation crops and metric geometry
                    # must consume that same observation, not whichever evidence
                    # happened to be appended last to the persistent identity.
                    bound["evidence"] = [dict(anchor_evidence)]
                    return bound
            return None

        bound_options: list[list[dict[str, Any]]] = []
        for evidence in reversed(list(candidate.get("evidence", ()))):
            binding_keys = [
                str(value)
                for value in evidence.get(
                    "relation_roi_anchor_binding_keys", ()
                )
            ]
            if len(anchor_sets) == 1 and binding_keys:
                match = next((
                    anchor_for_key(anchor_sets[0], binding_key)
                    for binding_key in binding_keys
                    if anchor_for_key(anchor_sets[0], binding_key) is not None
                ), None)
                if match is not None:
                    bound_options.append([dict(match)])
                continue
            if len(binding_keys) != len(anchor_sets):
                continue
            bound: list[dict[str, Any]] = []
            for binding_key, anchors in zip(binding_keys, anchor_sets):
                match = anchor_for_key(anchors, binding_key)
                if match is None:
                    bound = []
                    break
                bound.append(dict(match))
            if len(bound) == len(anchor_sets):
                bound_options.append(bound)
        if not bound_options:
            return None
        if str(predicate).lower() != "between" or len(anchor_sets) != 2:
            return bound_options[0]

        # A persistent candidate may carry relation crops from adjacent views.
        # Keep each crop's anchor pair intact, but choose the already-observed
        # pair whose map-frame segment best agrees with the candidate's latest
        # measured LiDAR geometry.  This is selection over provenance pairs,
        # not a candidate-by-anchor Cartesian expansion.
        subject_low, subject_high = _bounds(candidate)
        subject_center = [
            0.5 * (subject_low[index] + subject_high[index])
            for index in (0, 1)
        ]
        subject_half = [
            0.5 * (subject_high[index] - subject_low[index])
            for index in (0, 1)
        ]

        def provenance_pair_score(
            anchors: Sequence[Mapping[str, Any]],
        ) -> tuple[float, float, float]:
            centers: list[list[float]] = []
            half_sizes: list[list[float]] = []
            for anchor in anchors:
                low, high = _bounds(anchor)
                centers.append([
                    0.5 * (low[index] + high[index]) for index in (0, 1)
                ])
                half_sizes.append([
                    0.5 * (high[index] - low[index]) for index in (0, 1)
                ])
            first, second = centers
            dx, dy = second[0] - first[0], second[1] - first[1]
            length_squared = dx * dx + dy * dy
            if length_squared <= 0.0:
                return (float("inf"), float("inf"), float("inf"))
            length = math.sqrt(length_squared)
            position = (
                (subject_center[0] - first[0]) * dx
                + (subject_center[1] - first[1]) * dy
            ) / length_squared
            half_projection = (
                abs(dx) * subject_half[0] + abs(dy) * subject_half[1]
            ) / length_squared
            lower = position - half_projection
            upper = position + half_projection
            outside = max(0.0, -upper, lower - 1.0)
            perpendicular = abs(
                dy * (subject_center[0] - first[0])
                - dx * (subject_center[1] - first[1])
            ) / length
            normal = [-dy / length, dx / length]
            subject_radius = (
                abs(normal[0]) * subject_half[0]
                + abs(normal[1]) * subject_half[1]
            )
            anchor_radius = max(
                abs(normal[0]) * half[0] + abs(normal[1]) * half[1]
                for half in half_sizes
            )
            corridor = subject_radius + anchor_radius
            normalized_perpendicular = (
                perpendicular / corridor
                if corridor > 0.0 else float("inf")
            )
            endpoint_distance = min(
                math.dist(subject_center, first),
                math.dist(subject_center, second),
            )
            same_support = 1.0 if math.isclose(
                endpoint_distance, 0.0, rel_tol=0.0, abs_tol=1e-6
            ) else 0.0
            return (same_support, outside, normalized_perpendicular)

        return min(bound_options, key=provenance_pair_score)

    def resolve(self, entity_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        if entity_id in self.cache:
            return self.cache[entity_id]
        if entity_id in self.stack:
            return [], [f"relation_dependency_cycle:{entity_id}"]
        if entity_id not in self.entities:
            return [], [f"entity_missing:{entity_id}"]
        self.stack.add(entity_id)
        candidates = self._class_candidates(entity_id)
        for candidate in candidates:
            self.candidate_states.setdefault(
                (entity_id, int(candidate["object_id"])), "YES"
            )
            self.direct_candidate_states.setdefault(
                (entity_id, int(candidate["object_id"])), "YES"
            )
        failures: list[str] = []
        subject_relations = self._subject_relations(entity_id)
        for relation_index, relation in enumerate(subject_relations):
            relation_id = str(relation["id"])
            predicate = str(relation.get("predicate", "")).lower()
            spec = relation_spec(predicate)
            anchor_sets: list[list[dict[str, Any]]] = []
            for raw_anchor_id in relation.get("object_entities", ()):
                anchor_id = str(raw_anchor_id)
                anchors, anchor_failures = self.resolve(anchor_id)
                failures.extend(anchor_failures)
                if not anchors:
                    failures.append(
                        f"relation_anchor_missing:{relation_id}:{anchor_id}"
                    )
                anchor_sets.append(anchors)
            selector_anchor_missing = any(not values for values in anchor_sets)
            if selector_anchor_missing:
                candidates = []
                break
            if spec.operator_kind is OperatorKind.SET_SELECTOR:
                closure = dict(
                    self.snapshot.get("query_domain", {})
                    .get("selector_closures", {})
                    .get(relation_id, {})
                )
                selector_state = str(
                    closure.get("selector_state", "")
                )
                selector_evidence_complete = bool(
                    closure.get("selector_evidence_complete") is True
                    and selector_state == "STABLE_GEOMETRIC_WINNER"
                )
                selector_provisional = not selector_evidence_complete
                candidate_ids = list(
                    closure.get(
                        "candidate_object_ids",
                        closure.get("persistent_candidate_object_ids", ()),
                    )
                )
                anchor_ids = list(closure.get("anchor_object_ids", ()))
                selected_object_id = closure.get("selected_object_id")
                selection_available = bool(
                    candidate_ids
                    and anchor_ids
                    and selected_object_id is not None
                    and closure.get("selected_anchor_id") is not None
                )
                closure_closed = closure.get("candidate_domain_closed") is True
                self.selector_resolutions.append({
                    "relation_id": relation_id,
                    "predicate": predicate,
                    "operator_kind": spec.operator_kind.value,
                    "candidate_scope": spec.candidate_scope,
                    "candidate_object_ids": candidate_ids,
                    "anchor_object_ids": anchor_ids,
                    "selected_object_id": (
                        int(selected_object_id)
                        if selected_object_id is not None
                        else None
                    ),
                    "selected_anchor_id": (
                        int(closure["selected_anchor_id"])
                        if str(closure.get("selected_anchor_id", "")).lstrip(
                            "-"
                        ).isdigit()
                        else None
                    ),
                    "distances_m": dict(closure.get("candidate_distances_m", {})),
                    "distance_records": list(
                        closure.get("distance_records", ())
                    ),
                    "candidate_domain_closed": closure_closed,
                    "candidate_domain_state": str(
                        closure.get("candidate_domain_state", "")
                    ),
                    "selector_state": selector_state,
                    "selector_provisional": selector_provisional,
                    "selector_evidence_complete": selector_evidence_complete,
                    "current_winner_stable": bool(
                        closure.get("current_winner_stable", False)
                    ),
                    "current_winner_stability_margin_m": closure.get(
                        "current_winner_stability_margin_m"
                    ),
                    "selection_mode": str(closure.get("selection_mode", "")),
                    "selection_recomputed_from_scene_version": closure.get(
                        "selection_recomputed_from_scene_version"
                    ),
                    "anchor_sensitive": bool(closure.get("anchor_sensitive", False)),
                    "stable_winner_across_anchor_hypotheses": bool(
                        closure.get("stable_winner_across_anchor_hypotheses", False)
                    ),
                    "anchor_hypotheses": list(closure.get("anchor_hypotheses", ())),
                    "selector_hypotheses": list(
                        closure.get("selector_hypotheses", ())
                    ),
                    "selector_evidence_features": dict(
                        closure.get("selector_evidence_features", {})
                    ),
                    "covariance_mode": str(closure.get("covariance_mode", "")),
                    "distance_metric": str(
                        closure.get("distance_metric", "horizontal_xy")
                    ),
                    "closure_reasons": list(
                        closure.get("closure_reasons", ())
                    ),
                    "current_best_distance_m": closure.get(
                        "current_best_distance_m"
                    ),
                    "unexplored_regions_that_can_change_result": list(
                        closure.get(
                            "unexplored_regions_that_can_change_result", ()
                        )
                    ),
                })
                if not selection_available:
                    failures.append(f"selector_current_winner_missing:{relation_id}")
                    candidates = []
                    break
                closed_candidate_ids = {
                    int(value)
                    for value in closure.get(
                        "candidate_object_ids",
                        closure.get("persistent_candidate_object_ids", ()),
                    )
                }
                closed_anchor_ids = {
                    int(value) for value in closure.get("anchor_object_ids", ())
                }
                selected_anchor_id = closure.get("selected_anchor_id")
                if selected_anchor_id is not None:
                    try:
                        closed_anchor_ids = {int(selected_anchor_id)}
                    except (TypeError, ValueError):
                        pass
                candidates = [
                    value for value in candidates
                    if int(value["object_id"]) in closed_candidate_ids
                ]
                anchor_sets = [
                    [
                        value for value in values
                        if int(value["object_id"]) in closed_anchor_ids
                    ]
                    for values in anchor_sets
                ]
                anchor_entity_id = (
                    str(relation.get("object_entities", ())[0])
                    if len(relation.get("object_entities", ())) == 1 else ""
                )
                anchor = (
                    self._selector_anchor(
                        relation_id,
                        anchor_entity_id,
                        anchor_sets[0],
                        candidates,
                        predicate,
                    )
                    if len(anchor_sets) == 1 else None
                )
                rankable = [value for value in candidates if _valid_center_3d(value)]
                if anchor is None:
                    failures.append(f"selector_anchor_ambiguous:{relation_id}")
                    candidates = []
                    break
                if not rankable:
                    failures.append(f"selector_domain_empty:{relation_id}")
                    candidates = []
                    break
                ranked = [
                    (_distance(value, anchor), int(value["object_id"]), value)
                    for value in rankable
                ]
                selected_id = closure.get("selected_object_id")
                selected = next(
                    (value for value in ranked if selected_id is not None and value[1] == int(selected_id)),
                    None,
                )
                if selected is None:
                    selected = (
                        max(ranked, key=lambda value: (value[0], -value[1]))
                        if predicate == "farthest"
                        else min(ranked, key=lambda value: (value[0], value[1]))
                    )
                self.relation_anchor_bindings.append({
                    "entity_id": entity_id,
                    "relation": dict(relation),
                    "bound_anchors": [dict(anchor)],
                    "selector_hypotheses": list(
                        self.selector_resolutions[-1].get(
                            "selector_hypotheses", ()
                        )
                    ),
                })
                self.selector_resolutions[-1].update({
                    "anchor_binding_source": (
                        "open_scene_memory_geometric_recompute"
                        if not closure_closed
                        else "query_domain_closed"
                    ),
                    "selected_object_id": selected[1],
                    "selected_anchor_id": int(anchor["object_id"]),
                    "selector_provisional": selector_provisional,
                    "selector_evidence_complete": selector_evidence_complete,
                    "distances_m": {
                        str(object_id): distance
                        for distance, object_id, _value in ranked
                    },
                })
                candidates = [selected[2]]
                selector_evaluator = getattr(
                    self.verifier, "evaluate_selector", None
                )
                if not callable(selector_evaluator):
                    raise RuntimeError("relation_engine_selector_api_missing")
                selector_verdict = dict(selector_evaluator(
                    relation,
                    selected[2],
                    anchor,
                    domain_complete=selector_evidence_complete,
                    distance_m=selected[0],
                ))
                relation_state = str(
                    selector_verdict.get("state", "UNKNOWN")
                ).upper()
                self.candidate_states[(entity_id, selected[1])] = relation_state
                self.direct_candidate_states[(entity_id, selected[1])] = relation_state
                self.verifications.append({
                    "relation_id": relation_id,
                    "predicate": predicate,
                    "subject_object_id": selected[1],
                    "object_ids": [int(anchor["object_id"])],
                    "state": relation_state,
                    "path_state": relation_state,
                    "reason_code": str(selector_verdict.get("reason_code", "")),
                    "evidence_ids": [],
                    "geometry": dict(selector_verdict.get("geometry", {})),
                    "qwen": dict(selector_verdict.get("qwen", {})),
                    "verdict_authority": "RelationEngine",
                })
                continue
            self.relation_anchor_bindings.append({
                "entity_id": entity_id,
                "relation": dict(relation),
                "bound_anchors": [
                    dict(max(values, key=_probe_priority))
                    for values in anchor_sets
                ],
            })
            if not candidates:
                failures.append(
                    f"relation_bound_candidate_missing:{relation_id}:{entity_id}"
                )
                break

            non_refuted: list[dict[str, Any]] = []
            states: list[str] = []
            node = self._node(relation)
            # One candidate may be verified against several anchor bindings.
            # Provenance and metric-geometry binding are both imperfect, and
            # BETWEEN needs corridor enumeration across the anchor class
            # domains; verify every plausible exact-ID tuple and aggregate.
            bound_options_by_candidate: dict[
                int, list[list[dict[str, Any]]]
            ] = {}
            for candidate in candidates:
                candidate_id = int(candidate["object_id"])
                options: list[list[dict[str, Any]]] = []
                provenance = self._bind_from_candidate_provenance(
                    candidate, anchor_sets, predicate
                )
                if (
                    provenance is not None
                    and str(predicate).lower() == "between"
                    and not _geometric_relation_feature(
                        "between", candidate, provenance
                    )
                ):
                    # Provenance pairs inherit the question grounding ROI and
                    # can bind anchors the subject merely touches.  BETWEEN
                    # still needs map-frame corridor interiority.
                    provenance = None
                if provenance is not None:
                    options.append(provenance)
                if (
                    str(predicate).lower() == "between"
                    and len(anchor_sets) == 2
                ):
                    for _centrality, _perp, pair in self._geometric_between_pairs(
                        candidate, anchor_sets, limit=4
                    ):
                        options.append([dict(pair[0]), dict(pair[1])])
                else:
                    linear = self._bind_linear_anchors(
                        candidate, anchor_sets, predicate
                    )
                    if linear is not None:
                        options.append(linear)
                seen: set[tuple[int, ...]] = set()
                unique: list[list[dict[str, Any]]] = []
                for option in options:
                    key = tuple(
                        int(value["object_id"]) for value in option
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    unique.append(option)
                bound_options_by_candidate[candidate_id] = unique
            if str(relation.get("predicate", "")).lower() == "between":
                def between_schedule_priority(
                    candidate: Mapping[str, Any],
                ) -> tuple[Any, ...]:
                    options = bound_options_by_candidate.get(
                        int(candidate["object_id"]), ()
                    )
                    if not options:
                        return (False, False, float("-inf"))
                    bound = options[0]
                    candidate_low, candidate_high = _bounds(candidate)
                    point = [
                        0.5 * (candidate_low[index] + candidate_high[index])
                        for index in (0, 1)
                    ]
                    anchor_centers = []
                    for anchor in bound:
                        anchor_low, anchor_high = _bounds(anchor)
                        anchor_centers.append([
                            0.5 * (anchor_low[index] + anchor_high[index])
                            for index in (0, 1)
                        ])
                    first, second = anchor_centers
                    dx, dy = second[0] - first[0], second[1] - first[1]
                    length_squared = dx * dx + dy * dy
                    if length_squared <= 0.0:
                        return (False, False, float("-inf"))
                    position = (
                        (point[0] - first[0]) * dx
                        + (point[1] - first[1]) * dy
                    ) / length_squared
                    half_x = 0.5 * (candidate_high[0] - candidate_low[0])
                    half_y = 0.5 * (candidate_high[1] - candidate_low[1])
                    half_projection = (
                        abs(dx) * half_x + abs(dy) * half_y
                    ) / length_squared
                    intersects_segment = (
                        position + half_projection >= 0.0
                        and position - half_projection <= 1.0
                    )
                    overlaps_anchor = any(
                        _horizontal_overlap(candidate, anchor)
                        for anchor in bound
                    )
                    return (
                        intersects_segment,
                        not overlaps_anchor,
                        min(position, 1.0 - position),
                        *_relation_candidate_priority(candidate),
                    )

                # Schedule the physically interior exact tuple first.  This is
                # task-aware ordering only: every attempted tuple still needs
                # its object-ID-grounded Qwen relation verdict.
                candidates.sort(
                    key=between_schedule_priority,
                    reverse=True,
                )
            wave3_verdicts: dict[tuple[int, int], dict[str, Any]] = {}
            verify_many = getattr(self.verifier, "verify_many", None)
            if callable(verify_many):
                batch_keys: list[tuple[int, int]] = []
                batch_items = []
                for candidate in candidates:
                    candidate_id = int(candidate["object_id"])
                    for option_index, bound in enumerate(
                        bound_options_by_candidate.get(candidate_id, ())
                    ):
                        batch_keys.append((candidate_id, option_index))
                        batch_items.append((node, candidate, bound))
                if batch_items:
                    try:
                        wave3_verdicts = dict(zip(
                            batch_keys,
                            verify_many(batch_items),
                        ))
                    except Exception:
                        wave3_verdicts = {}
            for candidate in candidates:
                candidate_id = int(candidate["object_id"])
                candidate_key = (entity_id, candidate_id)
                options = bound_options_by_candidate.get(candidate_id, ())
                candidate_attempts: list[
                    tuple[list[dict[str, Any]], dict[str, Any]]
                ] = []
                if self.verifier is None:
                    bound_anchors = list(options[0]) if options else []
                    verdict = {
                        "state": "UNKNOWN",
                        "reason_code": "object_relation_verifier_unavailable",
                        "evidence_ids": [],
                        "geometry": {},
                        "qwen": {},
                    }
                    candidate_attempts = [(bound_anchors, verdict)]
                else:
                    for option_index, bound in enumerate(options):
                        verdict = wave3_verdicts.get(
                            (candidate_id, option_index)
                        )
                        if verdict is None and bound:
                            verdict = dict(
                                self.verifier(node, candidate, bound)
                            )
                        if verdict is not None:
                            candidate_attempts.append(
                                ([dict(value) for value in bound], verdict)
                            )
                if not candidate_attempts:
                    candidate_attempts = [([], {
                        "state": "UNKNOWN",
                        "reason_code": "relation_role_binding_provenance_missing",
                        "evidence_ids": [],
                        "geometry": {},
                        "qwen": {},
                    })]
                candidate_attempts = [
                    (bound_anchors, dict(raw_verdict))
                    for bound_anchors, raw_verdict in candidate_attempts
                ]
                # Aggregate every exact-ID tuple attempt for this candidate.
                # Alternative anchor identities that disagree are unresolved;
                # a positive tuple must not erase a negative tuple for the same
                # candidate until the binding is disambiguated.
                attempt_states = [
                    str(attempt[1].get("state", "UNKNOWN")).upper()
                    for attempt in candidate_attempts
                ]
                attempt_states = [
                    value if value in {"YES", "NO", "UNKNOWN", "INVALID"} else "UNKNOWN"
                    for value in attempt_states
                ]
                if "INVALID" in attempt_states:
                    state = "INVALID"
                elif "YES" in attempt_states and "NO" in attempt_states:
                    state = "UNKNOWN"
                elif "YES" in attempt_states:
                    state = "YES"
                elif "NO" in attempt_states:
                    state = "NO"
                else:
                    state = "UNKNOWN"
                previous_direct_state = self.direct_candidate_states.get(
                    candidate_key, "YES"
                )
                direct_state = (
                    "INVALID"
                    if state == "INVALID"
                    else "NO"
                    if state == "NO" or previous_direct_state == "NO"
                    else "UNKNOWN"
                    if state == "UNKNOWN" or previous_direct_state == "UNKNOWN"
                    else "YES"
                )
                self.direct_candidate_states[candidate_key] = direct_state
                prior_state = self.candidate_states.get(
                    candidate_key, "YES"
                )
                best_attempt = max(
                    candidate_attempts,
                    key=lambda value: {
                        "INVALID": 3, "YES": 2, "UNKNOWN": 1, "NO": 0
                    }[str(value[1].get("state", "UNKNOWN")).upper()],
                )
                anchor_path_states = [
                    self.candidate_states.get(
                        (str(anchor_id), int(anchor["object_id"])), "YES"
                    )
                    for anchor_id, anchor in zip(
                        relation.get("object_entities", ()),
                        best_attempt[0],
                    )
                ]
                if len(best_attempt[0]) != len(anchor_sets):
                    anchor_path_states.append("UNKNOWN")
                path_state = (
                    "INVALID"
                    if (
                        state == "INVALID"
                        or prior_state == "INVALID"
                        or "INVALID" in anchor_path_states
                    )
                    else "NO"
                    if (
                        state == "NO"
                        or prior_state == "NO"
                        or "NO" in anchor_path_states
                    )
                    else "UNKNOWN"
                    if state == "UNKNOWN"
                    or prior_state == "UNKNOWN"
                    or "UNKNOWN" in anchor_path_states
                    else "YES"
                )
                states.append(path_state)
                for bound_anchors, verdict in candidate_attempts:
                    attempt_state = str(
                        verdict.get("state", "UNKNOWN")
                    ).upper()
                    if attempt_state not in {"YES", "NO", "UNKNOWN", "INVALID"}:
                        attempt_state = "UNKNOWN"
                    self.relation_attempts.append({
                        "entity_id": entity_id,
                        "relation": dict(relation),
                        "candidate": dict(candidate),
                        "bound_anchors": list(bound_anchors),
                        "state": attempt_state,
                        "path_state": path_state,
                        "verdict": dict(verdict),
                    })
                self.verifications.append({
                    "relation_id": relation_id,
                    "predicate": str(relation["predicate"]),
                    "subject_object_id": candidate_id,
                    "object_ids": [
                        int(value["object_id"])
                        for value in best_attempt[0]
                    ],
                    "state": state,
                    "path_state": path_state,
                    "relation_probability": best_attempt[1].get(
                        "relation_probability"
                    ),
                    "reason_code": str(
                        best_attempt[1].get(
                            "reason_code", "relation_state_missing"
                        )
                    ),
                    "evidence_ids": list(
                        best_attempt[1].get("evidence_ids", ())
                    ),
                    "station_id": str(
                        best_attempt[1].get("station_id", "")
                    ),
                    "timestamp": best_attempt[1].get("timestamp"),
                    "source_observation_ids": list(
                        best_attempt[1].get(
                            "source_observation_ids",
                            best_attempt[1].get("evidence_ids", ()),
                        )
                    ),
                    "identity_version": best_attempt[1].get(
                        "identity_version"
                    ),
                    "geometry_version": best_attempt[1].get(
                        "geometry_version"
                    ),
                    "geometry": dict(
                        best_attempt[1].get("geometry", {})
                    ),
                    "qwen": dict(best_attempt[1].get("qwen", {})),
                })
                if path_state != "NO":
                    self.candidate_states[candidate_key] = path_state
                    non_refuted.append(candidate)
                # The final object reference is singular, but intermediate
                # anchors must remain available until the consumer binds its
                # own provenance.  Stopping at the first upstream YES can lock
                # the query onto an unrelated relation instance elsewhere.
            candidates = non_refuted
            if not candidates:
                failures.append(
                    f"relation_{'unknown' if 'UNKNOWN' in states else 'refuted'}:"
                    f"{relation_id}:{entity_id}"
                )
                break
        self.stack.remove(entity_id)
        result = (candidates, list(dict.fromkeys(failures)))
        self.cache[entity_id] = result
        return result

    def relation_search_probe(self, entity_id: str) -> dict[str, Any] | None:
        """Return one semantic search locus from one already-bound tuple.

        This is linear over verified tuples and never enumerates anchor
        products.  BETWEEN searches from the finite segment defined by the
        two role-bound anchors when the current view has no directly supported
        target instance.  An incomplete selector domain searches near the
        bound selector anchor so the missing class instances can close.
        """
        selector_bindings = [
            value for value in self.relation_anchor_bindings
            if value.get("entity_id") == entity_id
            and str(value.get("relation", {}).get("predicate", "")).lower()
            in {"closest", "farthest"}
            and value.get("bound_anchors")
        ]
        if selector_bindings:
            binding = selector_bindings[0]
            relation_id = str(binding.get("relation", {}).get("id", ""))
            resolution = next(
                (
                    value for value in self.selector_resolutions
                    if str(value.get("relation_id", "")) == relation_id
                ),
                {},
            )
            ranked_probe_hypotheses = [
                value
                for value in resolution.get("selector_hypotheses", ())
                if isinstance(value, Mapping)
            ]
            focused_hypotheses: list[dict[str, Any]] = []
            focused_keys: set[tuple[int, int]] = set()

            def add_focused(raw: object) -> None:
                if not isinstance(raw, Mapping):
                    return
                subject_id = raw.get("subject_object_id")
                anchor_id = raw.get("anchor_object_id")
                if not (
                    str(subject_id).lstrip("-").isdigit()
                    and str(anchor_id).lstrip("-").isdigit()
                ):
                    return
                key = (int(subject_id), int(anchor_id))
                if key in focused_keys or len(focused_hypotheses) >= 2:
                    return
                focused_keys.add(key)
                focused_hypotheses.append(dict(raw))

            for raw_hypothesis in ranked_probe_hypotheses:
                add_focused(raw_hypothesis)
                if len(focused_hypotheses) >= 2:
                    break

            objects_by_id = {
                int(value["object_id"]): value
                for value in self.objects
                if str(value.get("object_id", "")).lstrip("-").isdigit()
            }
            candidate_hypotheses: list[dict[str, Any]] = []
            anchor_ids: list[int] = []
            for hypothesis in focused_hypotheses:
                candidate_id = int(hypothesis["subject_object_id"])
                anchor_id = int(hypothesis["anchor_object_id"])
                candidate = objects_by_id.get(candidate_id)
                anchor = objects_by_id.get(anchor_id)
                if candidate is None or anchor is None:
                    continue
                candidate_hypotheses.append({
                    "object_id": candidate_id,
                    "center_xy": [
                        float(candidate["center_3d"][0]),
                        float(candidate["center_3d"][1]),
                    ],
                    "anchor_object_id": anchor_id,
                    "selector_score": hypothesis.get("selector_score"),
                    "ranking_competitor_rank": hypothesis.get(
                        "ranking_competitor_rank"
                    ),
                })
                anchor_ids.append(anchor_id)
            anchor_ids = list(dict.fromkeys(anchor_ids))
            probe_source_candidate_id = (
                int(focused_hypotheses[0]["subject_object_id"])
                if focused_hypotheses else None
            )
            locus_value = objects_by_id.get(probe_source_candidate_id)
            if locus_value is None:
                locus_value = binding["bound_anchors"][0]
            locus = dict(locus_value)
            locus.update({
                "probe_relation": str(
                    binding.get("relation", {}).get("predicate", "")
                ).lower(),
                "probe_relation_ids": [relation_id] if relation_id else [],
                "probe_anchor_object_ids": anchor_ids,
                "probe_source_candidate_id": probe_source_candidate_id,
                "probe_candidate_object_ids": [
                    int(value["object_id"]) for value in candidate_hypotheses
                ],
                "probe_candidate_hypotheses": candidate_hypotheses,
                "probe_selector_hypotheses": focused_hypotheses,
                "probe_target_class": str(
                    self.entities.get(entity_id, {}).get(
                        "class_name", "relation_target"
                    )
                ),
            })
            return locus
        dependency_depth = {entity_id: 0}
        dependency_queue = [entity_id]
        while dependency_queue:
            current_entity_id = dependency_queue.pop(0)
            for relation in self._subject_relations(current_entity_id):
                for raw_dependency_id in relation.get("object_entities", ()):
                    dependency_id = str(raw_dependency_id)
                    if dependency_id in dependency_depth:
                        continue
                    dependency_depth[dependency_id] = (
                        dependency_depth[current_entity_id] + 1
                    )
                    dependency_queue.append(dependency_id)

        unresolved_dependency_attempts = [
            value for value in self.relation_attempts
            if str(value["entity_id"]) in dependency_depth
            # Re-observe only a tuple whose evidence is genuinely incomplete.
            # A map-frame/Qwen refuted tuple is not a useful search locus: in
            # an instruction trajectory it can drag observation back to an object
            # that has already been proved not to satisfy the reference.
            and value.get("path_state") == "UNKNOWN"
            and value.get("bound_anchors")
        ]
        non_between_attempts = [
            value for value in unresolved_dependency_attempts
            if str(value["relation"].get("predicate", "")).lower()
            != "between"
        ]
        if non_between_attempts:
            def evidence_views(obj: Mapping[str, Any]) -> set[str]:
                return {
                    str(view_id)
                    for evidence in obj.get("evidence", ())
                    for view_id in (
                        evidence.get("representative_view_id"),
                        *evidence.get("source_view_ids", ()),
                    )
                    if str(view_id).strip()
                }

            def jointly_visible_view_count(value: Mapping[str, Any]) -> int:
                common = evidence_views(value.get("candidate", {}))
                for anchor_value in value.get("bound_anchors", ()):
                    common.intersection_update(evidence_views(anchor_value))
                return len(common)

            def semantic_support(obj: Mapping[str, Any]) -> float:
                return float(
                    obj.get("semantic_probability", 0.0)
                ) * (1.0 + 0.15 * _semantic_evidence_count(obj))

            def anchor_search_information(
                anchor_value: Mapping[str, Any],
                attempt_entity_id: str,
            ) -> tuple[float, float, int]:
                # A relation anchor defines the local neighborhood in which all
                # target-class hypotheses can be tested.  Its information gain
                # therefore grows smoothly with the unresolved target domain.
                target_names = _entity_class_names(
                    self.entities.get(attempt_entity_id, {})
                )
                target_domain_size = sum(
                    _matches_class_names(value, target_names)
                    for value in self.objects
                )
                role_gain = 0.35 * math.log1p(target_domain_size)
                visits = self._probe_locus_visit_count(
                    anchor_value.get("object_id")
                )
                information = (
                    semantic_support(anchor_value) + role_gain
                ) / (1.0 + 2.0 * visits)
                return float(information), float(role_gain), int(visits)

            def attempt_search_rank(
                value: Mapping[str, Any],
            ) -> tuple[float, float, int]:
                attempt_entity_id = str(value["entity_id"])
                anchor_information = max(
                    anchor_search_information(
                        anchor_value, attempt_entity_id
                    )[0]
                    for anchor_value in value.get("bound_anchors", ())
                )
                candidate_support = semantic_support(
                    value.get("candidate", {})
                )
                search_utility = (
                    anchor_information
                    + 0.15 * candidate_support
                    + 0.10 * jointly_visible_view_count(value)
                )
                return (
                    float(dependency_depth.get(attempt_entity_id, 0)),
                    float(search_utility),
                    -int(value.get("candidate", {}).get("object_id", -1)),
                )

            attempt = max(
                non_between_attempts,
                key=attempt_search_rank,
            )
            attempt_entity_id = str(attempt["entity_id"])
            anchor = max(
                attempt["bound_anchors"],
                key=lambda value: anchor_search_information(
                    value, attempt_entity_id
                )[0],
            )
            # Close the deepest unresolved dependency first. In
            # ``plant near [book on cabinet]`` that means re-observing the
            # book/cabinet tuple before chasing an outer plant hypothesis.
            # For a referential relation, the bound anchor defines the local
            # search neighborhood while the subject is only one hypothesis.
            # Repeated visits continuously discount that anchor's expected
            # information, allowing another binding to take over without an
            # attempt budget or exhausted alternative branch.
            candidate = attempt["candidate"]
            candidate_visits = self._probe_locus_visit_count(
                candidate.get("object_id")
            )
            candidate_information = semantic_support(candidate) / (
                1.0 + 2.0 * candidate_visits
            )
            (
                anchor_information,
                anchor_role_gain,
                anchor_visits,
            ) = anchor_search_information(anchor, attempt_entity_id)
            probe_locus = anchor
            probe = dict(probe_locus)
            probe.update({
                "probe_relation": str(
                    attempt["relation"].get("predicate", "")
                ).lower(),
                "probe_anchor_object_ids": [
                    int(value["object_id"])
                    for value in attempt["bound_anchors"]
                ],
                "probe_source_candidate_id": int(
                    candidate["object_id"]
                ),
                "probe_relation_subject_xy": [
                    float(candidate["center_3d"][0]),
                    float(candidate["center_3d"][1]),
                ],
                "probe_locus_object_id": int(probe_locus["object_id"]),
                "probe_locus_source": "relation_anchor_neighborhood_search",
                "probe_candidate_information_score": candidate_information,
                "probe_anchor_information_score": anchor_information,
                "probe_anchor_role_gain": anchor_role_gain,
                "probe_candidate_repeat_count": candidate_visits,
                "probe_anchor_repeat_count": anchor_visits,
                "probe_target_class": str(
                    self.entities.get(attempt_entity_id, {}).get(
                        "class_name", "relation_target"
                    )
                ),
                "probe_dependency_entity_id": attempt_entity_id,
                "probe_requested_by_entity_id": entity_id,
            })
            probe["evidence"] = [
                *attempt["candidate"].get("evidence", ()),
                *anchor.get("evidence", ()),
            ]
            return probe

        # The final target may be absent because an intermediate dependency
        # (for example a referenced object located on an already-seen anchor)
        # has not yet been observed.  Continue the query at the closest bound
        # dependency locus instead of dropping directly to an unresolved
        # answer.  This follows entity dependencies linearly and never forms
        # candidate x anchor products.
        has_unresolved_between = any(
            str(value.get("relation", {}).get("predicate", "")).lower()
            == "between"
            for value in unresolved_dependency_attempts
        )
        dependency_queue = [] if has_unresolved_between else [entity_id]
        visited_dependencies = {entity_id}
        while dependency_queue:
            current_entity_id = dependency_queue.pop(0)
            for relation in self._subject_relations(current_entity_id):
                for raw_dependency_id in relation.get("object_entities", ()):
                    dependency_id = str(raw_dependency_id)
                    if dependency_id in visited_dependencies:
                        continue
                    visited_dependencies.add(dependency_id)
                    dependency_queue.append(dependency_id)
                    dependency_bindings = [
                        value for value in self.relation_anchor_bindings
                        if str(value.get("entity_id", "")) == dependency_id
                        and value.get("bound_anchors")
                    ]
                    if not dependency_bindings:
                        continue
                    binding = dependency_bindings[0]
                    anchors = list(binding["bound_anchors"])
                    anchor = max(anchors, key=_probe_priority)
                    probe = dict(anchor)
                    probe.update({
                        "probe_relation": str(
                            binding.get("relation", {}).get("predicate", "")
                        ).lower(),
                        "probe_anchor_object_ids": [
                            int(value["object_id"]) for value in anchors
                        ],
                        "probe_source_candidate_id": None,
                        "probe_target_class": str(
                            self.entities.get(dependency_id, {}).get(
                                "class_name", "relation_dependency"
                            )
                        ),
                        "probe_dependency_entity_id": dependency_id,
                        "probe_requested_by_entity_id": entity_id,
                    })
                    probe["evidence"] = [
                        evidence
                        for value in anchors
                        for evidence in value.get("evidence", ())
                    ]
                    return probe

        between_entity_ids = {
            str(value["entity_id"])
            for value in unresolved_dependency_attempts
            if str(value["relation"].get("predicate", "")).lower()
            == "between"
        }
        between_entity_id = (
            max(
                between_entity_ids,
                key=lambda value: dependency_depth.get(value, 0),
            )
            if between_entity_ids else entity_id
        )
        attempts = [
            value for value in self.relation_attempts
            if str(value["entity_id"]) == between_entity_id
            and str(value["relation"].get("predicate", "")).lower() == "between"
            and len(value["bound_anchors"]) == 2
        ]
        failed_probe_keys = self._failed_probe_anchor_keys()
        if failed_probe_keys:
            available_attempts = [
                value for value in attempts
                if tuple(sorted(
                    int(anchor["object_id"])
                    for anchor in value["bound_anchors"]
                )) not in failed_probe_keys
            ]
            attempts = available_attempts
        def rank(value: Mapping[str, Any]) -> tuple[float, ...]:
            qwen = value.get("verdict", {}).get("qwen", {})
            geometry = value.get("verdict", {}).get("geometry", {})
            raw_interval = geometry.get("segment_position_interval", ())
            if (
                isinstance(raw_interval, Sequence)
                and not isinstance(raw_interval, (str, bytes, bytearray))
                and len(raw_interval) == 2
            ):
                try:
                    lower, upper = (float(item) for item in raw_interval)
                    segment_gap = max(0.0, -upper, lower - 1.0)
                except (TypeError, ValueError):
                    segment_gap = float("inf")
            else:
                segment_gap = float("inf")
            try:
                perpendicular = float(
                    geometry.get("perpendicular_distance_m", float("inf"))
                )
                corridor = float(
                    geometry.get("corridor_half_width_m", 0.0)
                )
                lateral_gap = (
                    perpendicular / corridor
                    if corridor > 0.0 else float("inf")
                )
            except (TypeError, ValueError):
                lateral_gap = float("inf")
            return (
                int(str(qwen.get("state", "")).lower() == "supported"),
                -(segment_gap + lateral_gap),
                float(qwen.get("confidence", 0.0) or 0.0),
                len(value.get("candidate", {}).get("evidence", ())),
                -int(value.get("candidate", {}).get("object_id", -1)),
            )

        supported_attempts = [
            value for value in attempts
            if str(
                value.get("verdict", {}).get("qwen", {}).get("state", "")
            ).lower() == "supported"
        ]
        if supported_attempts:
            attempt = max(supported_attempts, key=rank)
            candidate = dict(attempt["candidate"])
            first, second = attempt["bound_anchors"]
        else:
            bindings = [
                {"bound_anchors": value["bound_anchors"]}
                for value in attempts
            ]
            if not bindings:
                bindings = [
                    value for value in self.relation_anchor_bindings
                    if str(value["entity_id"]) == between_entity_id
                    and str(value["relation"].get("predicate", "")).lower()
                    == "between"
                    and len(value["bound_anchors"]) == 2
                ]
            if failed_probe_keys:
                bindings = [
                    value for value in bindings
                    if tuple(sorted(
                        int(anchor["object_id"])
                        for anchor in value["bound_anchors"]
                    )) not in failed_probe_keys
                ]
            if not bindings:
                # No tuple bound yet: the anchor classes may exist while the
                # target instance was never grounded.  Probe the corridor
                # midpoint of the closest anchor pair, so the next station
                # can observe the region where the target must lie.
                between_relation = next((
                    value
                    for value in self._subject_relations(between_entity_id)
                    if str(value.get("predicate", "")).lower() == "between"
                ), {})
                anchor_entity_ids = between_relation.get(
                    "object_entities", ()
                )
                anchor_domains = []
                for anchor_id in anchor_entity_ids:
                    anchor = self.entities.get(str(anchor_id), {})
                    names = _entity_class_names(anchor)
                    domain = [
                        value for value in self.objects
                        if _matches_class_names(value, names)
                        and _valid_center_3d(value)
                    ]
                    anchor_domains.append(domain)
                if len(anchor_domains) == 2 and all(anchor_domains):
                    pairs = []
                    for first in anchor_domains[0]:
                        for second in anchor_domains[1]:
                            segment = _distance(first, second)
                            midpoint = [
                                0.5 * (
                                    float(first["center_3d"][index])
                                    + float(second["center_3d"][index])
                                )
                                for index in (0, 1)
                            ]
                            pairs.append((
                                segment,
                                _probe_priority(first),
                                _probe_priority(second),
                                first,
                                second,
                                midpoint,
                            ))
                    # Prefer anchors the question grounding explicitly
                    # qualified, then the closest pair: the referred anchors
                    # usually sit near each other.
                    pairs.sort(
                        key=lambda value: (
                            -(
                                value[1][0] + value[2][0]
                            ),
                            value[0],
                        )
                    )
                    first, second = pairs[0][3], pairs[0][4]
                    bindings = [{
                        "bound_anchors": [dict(first), dict(second)]
                    }]
                else:
                    return None
            entity = self.entities.get(between_entity_id, {})
            target_class_names = _entity_class_names(entity)
            known = [
                value for value in self.objects
                if _matches_class_names(value, target_class_names)
                and _valid_center_3d(value)
            ]
            if known:
                # Prefer the corridor whose midpoint lies closest to a known
                # candidate: that station re-observes the candidate whose
                # tuple still needs a second independent viewpoint.
                def midpoint_distance(value: Mapping[str, Any]) -> float:
                    first_anchor, second_anchor = value["bound_anchors"]
                    midpoint = [
                        0.5 * (
                            float(first_anchor["center_3d"][index])
                            + float(second_anchor["center_3d"][index])
                        )
                        for index in (0, 1)
                    ]
                    return min(
                        math.hypot(
                            midpoint[0] - float(item["center_3d"][0]),
                            midpoint[1] - float(item["center_3d"][1]),
                        )
                        for item in known
                    )

                binding = min(bindings, key=midpoint_distance)
                first, second = binding["bound_anchors"]
            else:
                first, second = bindings[0]["bound_anchors"]
            candidate = {
                "class_label": str(entity.get("class_name", "relation_target")),
                "target_kind": "relation_search_midpoint",
                "evidence": [],
            }
        center = [
            0.5 * (
                float(first["center_3d"][index])
                + float(second["center_3d"][index])
            )
            for index in range(3)
        ]
        raw_candidate_id = candidate.get("object_id")
        try:
            source_candidate_id = (
                int(raw_candidate_id)
                if raw_candidate_id is not None and int(raw_candidate_id) >= 0
                else None
            )
        except (TypeError, ValueError):
            source_candidate_id = None
        candidate.update({
            "center_3d": center,
            "probe_relation": "between",
            "probe_anchor_object_ids": [
                int(first["object_id"]), int(second["object_id"])
            ],
            "probe_source_candidate_id": source_candidate_id,
        })
        candidate["evidence"] = [
            *candidate.get("evidence", ()),
            *first.get("evidence", ()),
            *second.get("evidence", ()),
        ]
        return candidate


def _dedupe_objects(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for value in values:
        object_id = int(value.get("object_id", -1))
        if object_id in seen:
            continue
        seen.add(object_id)
        result.append(dict(value))
    return result


class _NumericalRelationResolver:
    """Project directed relation constraints onto the entity being counted.

    This resolver is deliberately numerical-only.  The established resolver
    remains authoritative for object-reference and instruction-following tasks.
    """

    def __init__(self, task_ir: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        self.task_ir = task_ir
        self.entities = {str(item["id"]): item for item in task_ir["entities"]}
        self.relations = {
            str(item["id"]): item for item in task_ir.get("relations", ())
        }
        self.objects = [
            item for item in snapshot["objects"]
            if item.get("status") in {"tentative", "confirmed"}
            and str(item.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
        ]
        self.cache: dict[
            tuple[str, tuple[str, ...], tuple[str, ...]],
            tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]],
        ] = {}
        self.stack: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()

    @staticmethod
    def _relation_involves(relation: Mapping[str, Any], entity_id: str) -> bool:
        return (
            str(relation.get("subject_entity", "")) == entity_id
            or entity_id in {
                str(value) for value in relation.get("object_entities", ())
            }
        )

    def _dependencies_for(
        self,
        relation: Mapping[str, Any],
        entity_id: str,
    ) -> tuple[str, ...]:
        explicit = [
            dependency_id
            for dependency_id in (
                str(value) for value in relation.get("depends_on", ())
            )
            if dependency_id in self.relations
            and self._relation_involves(self.relations[dependency_id], entity_id)
        ]
        # DeepSeek normally emits ``depends_on`` for nested qualifiers, but the
        # relation graph itself is sufficient to recover an omitted dependency:
        # in "chairs near the table with a vase on it", ``on(vase, table)``
        # qualifies the table even though table is the predicate object.
        current_id = str(relation.get("id", ""))
        implicit_inverse = [
            relation_id for relation_id, candidate in self.relations.items()
            if relation_id != current_id
            and entity_id in {
                str(value) for value in candidate.get("object_entities", ())
            }
        ]
        return tuple(dict.fromkeys([*explicit, *implicit_inverse]))

    def _class_candidates(
        self,
        entity_id: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        entity = self.entities.get(entity_id)
        if entity is None:
            return [], [f"entity_missing:{entity_id}"]
        class_names = _task_entity_class_names(self.task_ir, entity)
        candidates = []
        for item in self.objects:
            if not _matches_class_names(item, class_names):
                continue
            if not _attribute_matches(entity, item, self.objects, class_names):
                continue
            candidate = dict(item)
            candidate["numerical_membership_probability"] = (
                _numerical_object_probability(candidate)
            )
            candidates.append(candidate)
        return candidates, []

    def _subject_relation(
        self,
        candidates: list[dict[str, Any]],
        relation: Mapping[str, Any],
        excluded_relation_ids: frozenset[str],
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        relation_id = str(relation["id"])
        anchor_sets = []
        failures = []
        for raw_anchor_id in relation.get("object_entities", ()):
            anchor_id = str(raw_anchor_id)
            anchors, anchor_failures, _nested_support = self.resolve(
                anchor_id,
                required_relation_ids=self._dependencies_for(relation, anchor_id),
                excluded_relation_ids=excluded_relation_ids | {relation_id},
            )
            failures.extend(anchor_failures)
            if not anchors:
                failures.append(f"relation_anchor_missing:{relation_id}:{anchor_id}")
            anchor_sets.append(anchors)
        if failures or any(not values for values in anchor_sets):
            return [], list(dict.fromkeys(failures)), []

        predicate = str(relation.get("predicate", "")).lower()
        if predicate in {"closest", "farthest"}:
            if len(anchor_sets) != 1 or not candidates:
                return [], [f"ranking_relation_unresolved:{relation_id}"], []
            ranked = []
            for candidate in candidates:
                for anchor in anchor_sets[0]:
                    distance, uncertainty, lower, upper = (
                        _horizontal_distance_interval(candidate, anchor)
                    )
                    ranked.append((
                        distance,
                        uncertainty,
                        lower,
                        upper,
                        int(candidate["object_id"]),
                        int(anchor["object_id"]),
                        candidate,
                        anchor,
                    ))
            selected = (
                max(
                    ranked,
                    key=lambda value: (value[2], value[0], -value[4], -value[5]),
                )
                if predicate == "farthest"
                else min(
                    ranked,
                    key=lambda value: (value[3], value[0], value[4], value[5]),
                )
            )
            enriched = dict(selected[6])
            competitor_bounds = [
                value for value in ranked if value is not selected
            ]
            if predicate == "farthest":
                competitor_bound = max(
                    (value[3] for value in competitor_bounds),
                    default=None,
                )
                stable = bool(
                    competitor_bound is None or selected[2] > competitor_bound
                )
            else:
                competitor_bound = min(
                    (value[2] for value in competitor_bounds),
                    default=None,
                )
                stable = bool(
                    competitor_bound is None or selected[3] < competitor_bound
                )
            enriched.update({
                "ranking_distance_m": selected[0],
                "ranking_distance_uncertainty_m": selected[1],
                "ranking_lower_bound_m": selected[2],
                "ranking_upper_bound_m": selected[3],
                "ranking_anchor_object_id": selected[5],
                "ranking_metric": "horizontal_xy",
                "ranking_winner_stable": stable,
            })
            enriched["numerical_relation_probability"] = (
                _numerical_object_probability(selected[7])
            )
            enriched["numerical_membership_probability"] = (
                _numerical_object_probability(selected[6])
                * _numerical_object_probability(selected[7])
            )
            return [enriched], [], [selected[7]]

        filtered = []
        support = []
        for candidate in candidates:
            best_joint_probability = 0.0
            best_combo: Sequence[Mapping[str, Any]] = ()
            for combo in itertools.product(*anchor_sets):
                relation_probability = _numerical_predicate_probability(
                    predicate, candidate, list(combo)
                )
                anchor_probability = math.prod(
                    _numerical_object_probability(anchor) for anchor in combo
                )
                joint_probability = relation_probability * anchor_probability
                if joint_probability > best_joint_probability:
                    best_joint_probability = joint_probability
                    best_combo = combo
            membership_probability = (
                _numerical_object_probability(candidate)
                * best_joint_probability
            )
            if membership_probability <= 0.0:
                continue
            enriched = dict(candidate)
            enriched["numerical_relation_probability"] = float(
                best_joint_probability
            )
            enriched["numerical_membership_probability"] = float(
                membership_probability
            )
            filtered.append(enriched)
            support.extend(best_combo)
        return _dedupe_objects(filtered), [], _dedupe_objects(support)

    def _object_relation(
        self,
        candidates: list[dict[str, Any]],
        entity_id: str,
        relation: Mapping[str, Any],
        excluded_relation_ids: frozenset[str],
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        relation_id = str(relation["id"])
        subject_id = str(relation.get("subject_entity", ""))
        subjects, subject_failures, nested_support = self.resolve(
            subject_id,
            required_relation_ids=self._dependencies_for(relation, subject_id),
            excluded_relation_ids=excluded_relation_ids | {relation_id},
        )
        if subject_failures:
            return [], subject_failures, []
        if not subjects:
            return [], [f"relation_subject_missing:{relation_id}:{subject_id}"], []

        predicate = str(relation.get("predicate", "")).lower()
        filtered = []
        support = list(nested_support)
        failures = []
        for candidate in candidates:
            anchor_sets = []
            candidate_failures = []
            for raw_anchor_id in relation.get("object_entities", ()):
                anchor_id = str(raw_anchor_id)
                if anchor_id == entity_id:
                    anchors = [candidate]
                    anchor_failures = []
                else:
                    anchors, anchor_failures, other_support = self.resolve(
                        anchor_id,
                        required_relation_ids=self._dependencies_for(
                            relation, anchor_id
                        ),
                        excluded_relation_ids=excluded_relation_ids | {relation_id},
                    )
                    support.extend(other_support)
                candidate_failures.extend(anchor_failures)
                if not anchors:
                    candidate_failures.append(
                        f"relation_anchor_missing:{relation_id}:{anchor_id}"
                    )
                anchor_sets.append(anchors)
            if candidate_failures or any(not values for values in anchor_sets):
                failures.extend(candidate_failures)
                continue

            matched_subject = None
            matched_combo = None
            best_joint_probability = 0.0
            if predicate in {"closest", "farthest"}:
                if len(anchor_sets) != 1:
                    failures.append(f"ranking_relation_unresolved:{relation_id}")
                    continue
                ranked = []
                for subject in subjects:
                    for anchor in anchor_sets[0]:
                        distance, uncertainty, lower, upper = (
                            _horizontal_distance_interval(subject, anchor)
                        )
                        ranked.append((
                            distance,
                            uncertainty,
                            lower,
                            upper,
                            int(subject["object_id"]),
                            int(anchor["object_id"]),
                            subject,
                            anchor,
                        ))
                selected = (
                    max(
                        ranked,
                        key=lambda value: (
                            value[2], value[0], -value[4], -value[5]
                        ),
                    )
                    if predicate == "farthest"
                    else min(
                        ranked,
                        key=lambda value: (
                            value[3], value[0], value[4], value[5]
                        ),
                    )
                )
                matched_subject = selected[6]
                matched_combo = (selected[7],)
                # ``matched_combo[0]`` is the candidate being evaluated and
                # is multiplied exactly once when the result is enriched.
                best_joint_probability = _numerical_object_probability(
                    matched_subject
                )
            else:
                for subject in subjects:
                    for combo in itertools.product(*anchor_sets):
                        relation_probability = _numerical_predicate_probability(
                            predicate, subject, list(combo)
                        )
                        other_anchor_probability = math.prod(
                            _numerical_object_probability(anchor)
                            for anchor in combo
                            if int(anchor.get("object_id", -1))
                            != int(candidate.get("object_id", -2))
                        )
                        joint_probability = (
                            relation_probability
                            * _numerical_object_probability(subject)
                            * other_anchor_probability
                        )
                        if joint_probability > best_joint_probability:
                            best_joint_probability = joint_probability
                            matched_subject = subject
                            matched_combo = combo
            if matched_subject is not None:
                enriched = dict(candidate)
                enriched["numerical_relation_probability"] = float(
                    best_joint_probability
                )
                enriched["numerical_membership_probability"] = float(
                    _numerical_object_probability(candidate)
                    * best_joint_probability
                )
                filtered.append(enriched)
                support.append(matched_subject)
                support.extend(matched_combo or ())
        return (
            _dedupe_objects(filtered),
            list(dict.fromkeys(failures)),
            _dedupe_objects(support),
        )

    def resolve(
        self,
        entity_id: str,
        *,
        required_relation_ids: Sequence[str] = (),
        excluded_relation_ids: frozenset[str] = frozenset(),
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        required = tuple(dict.fromkeys(str(value) for value in required_relation_ids))
        excluded = tuple(sorted(str(value) for value in excluded_relation_ids))
        key = (str(entity_id), required, excluded)
        if key in self.cache:
            return self.cache[key]
        if key in self.stack:
            return [], [f"relation_dependency_cycle:{entity_id}"], []
        self.stack.add(key)
        candidates, failures = self._class_candidates(str(entity_id))
        support: list[dict[str, Any]] = []
        relation_ids = [
            relation_id for relation_id, relation in self.relations.items()
            if relation_id not in excluded_relation_ids
            and str(relation.get("subject_entity", "")) == str(entity_id)
        ]
        for relation_id in required:
            if relation_id in excluded_relation_ids or relation_id in relation_ids:
                continue
            relation = self.relations.get(relation_id)
            if relation is None:
                failures.append(f"relation_missing:{relation_id}")
                continue
            if self._relation_involves(relation, str(entity_id)):
                relation_ids.append(relation_id)

        for relation_id in relation_ids:
            relation = self.relations[relation_id]
            if str(relation.get("subject_entity", "")) == str(entity_id):
                candidates, relation_failures, relation_support = (
                    self._subject_relation(
                        candidates,
                        relation,
                        frozenset(excluded_relation_ids),
                    )
                )
            else:
                candidates, relation_failures, relation_support = (
                    self._object_relation(
                        candidates,
                        str(entity_id),
                        relation,
                        frozenset(excluded_relation_ids),
                    )
                )
            failures.extend(relation_failures)
            support.extend(relation_support)
            if failures:
                candidates = []
                break

        self.stack.remove(key)
        result = (
            _dedupe_objects(candidates),
            list(dict.fromkeys(failures)),
            _dedupe_objects(support),
        )
        self.cache[key] = result
        return result

    def resolve_target(
        self,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        plan = numerical_relation_plan(self.task_ir)
        if plan is None:
            return [], ["numerical_relation_plan_missing"], []
        inverse_relation_ids = [
            item["relation_id"] for item in plan["relations"]
            if item["target_role"] == "object"
        ]
        return self.resolve(
            str(plan["target_entity"]),
            required_relation_ids=inverse_relation_ids,
        )


def _numerical_relation_probe_object(
    resolver: _Resolver,
    plan: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if plan is None:
        return None
    for binding in plan.get("relations", ()):
        if binding.get("target_role") == "subject":
            related_ids = list(binding.get("object_entities", ()))
        else:
            related_ids = [str(binding.get("subject_entity", ""))]
        for entity_id in related_ids:
            candidates, _failures = resolver.resolve(str(entity_id))
            if candidates:
                return max(candidates, key=_probe_priority)
    return None


def _numerical_probe_object_from_snapshot(
    task_ir: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Select one current numerical PROBE object without relation products."""
    targeted = snapshot.get("count_query_targeted_acquisition")
    if isinstance(targeted, Mapping) and targeted.get("mode"):
        if targeted.get("navigation_required") is False:
            return None
        objects_by_id = {
            int(value["object_id"]): dict(value)
            for value in snapshot.get("objects", ())
            if isinstance(value, Mapping)
            and str(value.get("object_id", "")).lstrip("-").isdigit()
            and str(value.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
        }
        anchor_ids = [
            int(value) for value in targeted.get("anchor_object_ids", ())
            if str(value).lstrip("-").isdigit()
        ]
        target_ids = [
            int(value) for value in targeted.get("target_object_ids", ())
            if str(value).lstrip("-").isdigit()
        ]
        ordered_ids = list(dict.fromkeys(
            ([*anchor_ids, *target_ids] if targeted.get("anchor_first") else [*target_ids, *anchor_ids])
        ))
        selected = next(
            (objects_by_id[object_id] for object_id in ordered_ids if object_id in objects_by_id),
            None,
        )
        if selected is not None or targeted.get("navigation_target_xy") is not None:
            probe = dict(selected or {})
            probe.setdefault("class_label", "relation_anchor")
            probe["probe_relation"] = str(
                targeted.get("reason", "count_query_targeted_acquisition")
            )
            probe["probe_anchor_object_ids"] = anchor_ids
            probe["probe_object_ids"] = ordered_ids
            subject_candidate_id = next(
                (
                    object_id
                    for object_id in target_ids
                    if object_id in objects_by_id and object_id not in anchor_ids
                ),
                next(
                    (
                        object_id
                        for object_id in ordered_ids
                        if object_id in objects_by_id and object_id not in anchor_ids
                    ),
                    None,
                ),
            )
            if subject_candidate_id is not None:
                # The targeted plan may order anchors first for acquisition,
                # but the joint relation probe still needs an explicit
                # subject/anchor binding.  Without it the downstream probe
                # builder can reuse the anchor as both participants and
                # silently degrade to an anchor-only viewpoint.
                probe["probe_source_candidate_id"] = int(subject_candidate_id)
            probe["required_visible_object_ids"] = [
                int(value) for value in targeted.get("required_visible_object_ids", ())
                if str(value).lstrip("-").isdigit()
            ]
            probe["probe_relation_ids"] = [
                str(value) for value in targeted.get("relation_ids", ())
                if str(value).strip()
            ]
            probe["joint_visibility_required"] = bool(
                targeted.get("joint_visibility_required")
            )
            probe["requires_new_station"] = bool(
                targeted.get("requires_new_station", False)
            )
            if isinstance(targeted.get("navigation_target_xy"), Sequence):
                probe["navigation_target_xy"] = [
                    float(targeted["navigation_target_xy"][0]),
                    float(targeted["navigation_target_xy"][1]),
                ]
            probe["observation_objective"] = {
                "mode": str(targeted.get("mode", "")),
                "anchor_first": bool(targeted.get("anchor_first")),
                "required_visible_object_ids": list(
                    probe["required_visible_object_ids"]
                ),
                "requires_new_station": bool(
                    probe.get("requires_new_station", False)
                ),
            }
            query_domain = snapshot.get("count_query_domain", {})
            query_key = str(
                targeted.get("query_key")
                or (
                    query_domain.get("query_key", "")
                    if isinstance(query_domain, Mapping)
                    else ""
                )
            )
            probe["query_key"] = query_key
            probe["unknown_binding"] = copy.deepcopy(
                targeted.get("unknown_binding", {})
            )
            return probe
    entities = {
        str(value["id"]): dict(value) for value in task_ir.get("entities", ())
    }
    related_ids: list[str] = []
    for binding in plan.get("relations", ()) if plan is not None else ():
        if binding.get("target_role") == "subject":
            related_ids.extend(
                str(value) for value in binding.get("object_entities", ())
            )
        else:
            related_ids.append(str(binding.get("subject_entity", "")))
    related_ids.append(str(task_ir.get("target_entity", "")))
    objects = [
        dict(value)
        for value in snapshot.get("objects", ())
        if value.get("status") in {"tentative", "confirmed"}
        and str(value.get("cardinality_role", "")) not in {
            "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
        }
    ]
    for entity_id in dict.fromkeys(value for value in related_ids if value):
        entity = entities.get(entity_id)
        if entity is None:
            continue
        names = _entity_class_names(entity)
        candidates = [
            value for value in objects
            if _matches_class_names(value, names)
        ]
        if candidates:
            return max(candidates, key=_probe_priority)
    return None


def normalize_execution_steps(
    task_ir: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile TaskIR once into the canonical receding-horizon StepSpecs."""
    return RecedingHorizonInstructionExecutor.compile(task_ir)


def evaluate_required_relation_tuples(
    task_ir: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    relation_verifier: RelationTupleVerifier | None,
    *,
    entity_bindings: Mapping[str, int] | None = None,
    execution_steps: Sequence[Mapping[str, Any]] | None = None,
    current_step_index: int = 0,
    navigation_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate the current revision's required tuples before task solving.

    The returned records are an in-memory transaction product.  Callers may
    merge them with relation history for the same frozen scene/identity/
    geometry revision, then invoke the task resolver exactly once without a
    live verifier.  Persisting the merged history is a later side effect.
    """
    task_type = str(task_ir.get("task_type", ""))
    if relation_verifier is None or task_type not in {
        "object_reference",
        "instruction_following",
    }:
        return {
            "relation_verifications": [],
            "candidate_domains": {},
            "selector_resolutions": [],
            "evaluated_entity_ids": [],
        }
    target_entity_ids: list[str] = []
    if task_type == "object_reference":
        target_entity_ids = [str(task_ir.get("target_entity", ""))]
    else:
        steps = [
            dict(value)
            for value in (
                execution_steps
                if execution_steps is not None
                else normalize_execution_steps(task_ir)
            )
            if isinstance(value, Mapping)
        ]
        if 0 <= int(current_step_index) < len(steps):
            step = steps[int(current_step_index)]
            relation_ids = {
                str(value) for value in step.get("relation_ids", ())
            }
            if relation_ids:
                target_entity_ids = [str(step.get("target_entity", ""))]
    target_entity_ids = list(dict.fromkeys(
        value for value in target_entity_ids if value
    ))
    if not target_entity_ids:
        return {
            "relation_verifications": [],
            "candidate_domains": {},
            "selector_resolutions": [],
            "evaluated_entity_ids": [],
        }
    resolver = _ObjectReferenceRelationResolver(
        task_ir,
        snapshot,
        relation_verifier,
        entity_bindings,
        navigation_history,
    )
    failures: list[str] = []
    for entity_id in target_entity_ids:
        _candidates, entity_failures = resolver.resolve(entity_id)
        failures.extend(entity_failures)
    return {
        "relation_verifications": [
            dict(value) for value in resolver.verifications
            if isinstance(value, Mapping)
        ],
        "candidate_domains": copy.deepcopy(resolver.candidate_domains),
        "selector_resolutions": copy.deepcopy(resolver.selector_resolutions),
        "evaluated_entity_ids": target_entity_ids,
        "failures": list(dict.fromkeys(failures)),
    }


def derive_task_resolution_evidence(
    task_ir: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    object_relation_verifier: RelationTupleVerifier | None = None,
    entity_bindings: Mapping[str, int] | None = None,
    execution_steps: Sequence[Mapping[str, Any]] | None = None,
    current_step_index: int = 0,
    navigation_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive task-local evidence without authorizing actions or answers.

    The three public resolvers own FINALIZABLE/NEED_EVIDENCE/NEED_EXECUTION.
    This function retains established candidate, count and ordered-constraint
    calculations only; it is not a control-plane or finalization API.
    """
    resolver = _Resolver(task_ir, snapshot)
    target_id = str(task_ir["target_entity"])
    task_type = str(task_ir["task_type"])
    relation_plan = numerical_relation_plan(task_ir) if task_type == "numerical" else None
    relation_support: list[dict[str, Any]] = []
    object_reference_resolver: _ObjectReferenceRelationResolver | None = None
    if task_type == "numerical":
        numerical_resolver = _NumericalRelationResolver(task_ir, snapshot)
        candidates, failures, relation_support = numerical_resolver.resolve_target()
    elif task_type in {"object_reference", "instruction_following"}:
        object_reference_resolver = _ObjectReferenceRelationResolver(
            task_ir,
            snapshot,
            object_relation_verifier,
            entity_bindings,
            navigation_history,
        )
        candidates, failures = object_reference_resolver.resolve(target_id)
    else:
        candidates, failures = [], ["unsupported_task_type"]
    scene_memory_open = snapshot.get("scene_memory_open", True) is not False
    relation_selector_domain_open = bool(
        snapshot.get("relation_selector_domain_open")
        or snapshot.get("query_domain", {}).get("selector_closures", {})
    )
    numerical_count_domain_closed = bool(
        snapshot.get("numerical_count_domain_closed")
    )
    evidence_ids = sorted({
        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
        for candidate in [*candidates, *relation_support]
        for evidence in candidate.get("evidence", [])
    })
    result: dict[str, Any] = {
        "schema_version": "task_execution_v1",
        "task_type": task_type,
        "scene_version": int(snapshot["scene_version"]),
        "selected_object": None,
        "trajectory_directives": [],
        "instruction_candidate_selection": [],
        "probe_object": None,
        "evidence_ids": evidence_ids,
        "failed_constraints": list(dict.fromkeys(failures)),
        "geometry_reconstruction_ids": list(snapshot.get("geometry_reconstruction_ids", ())),
        "uncertainty": {
            "scene_memory_open": scene_memory_open,
            "relation_selector_domain_open": relation_selector_domain_open,
            "numerical_count_domain_closed": numerical_count_domain_closed,
            "numerical_relation_plan": relation_plan,
        },
        "entity_bindings": {
            str(entity_id): int(object_id)
            for entity_id, object_id in (entity_bindings or {}).items()
        },
        "selector_resolutions": [],
    }
    object_reference_relations = (
        [
            relation
            for relation in task_ir.get("relations", ())
            if str(relation.get("subject_entity", "")) == target_id
        ]
        if task_type == "object_reference"
        else []
    )
    relation_bound_candidates = (
        object_reference_resolver._class_candidates(target_id)
        if object_reference_relations
        and object_reference_resolver is not None
        else list(candidates)
    )
    relation_non_refuted_candidates = list(candidates)
    # Only the fused tuple state owns answer eligibility.  A role-grounded
    # Qwen support with conflicting or incomplete map geometry is UNKNOWN:
    # it is valuable probe evidence, but it cannot become an object answer
    # until a later viewpoint resolves the disagreement.
    relation_states = {
        int(item["object_id"]): (
            object_reference_resolver.candidate_states.get(
                (target_id, int(item["object_id"])), "UNKNOWN"
            )
            if object_reference_resolver is not None else "UNKNOWN"
        )
        for item in relation_non_refuted_candidates
    }
    relation_verified_candidates = [
        item for item in relation_non_refuted_candidates
        if relation_states[int(item["object_id"])] == "YES"
    ]
    # Non-refuted UNKNOWN tuples remain useful probe hypotheses, but only an
    # exact fused YES tuple may authorize an object-reference answer.
    relation_eligible_candidates = list(relation_verified_candidates)
    if task_type == "object_reference":
        result["relation_required"] = bool(object_reference_relations)
        result["relation_candidate_object_ids"] = [
            int(item["object_id"]) for item in relation_bound_candidates
        ]
        result["relation_verified_object_ids"] = [
            int(item["object_id"]) for item in relation_verified_candidates
        ]
        result["relation_eligible_object_ids"] = [
            int(item["object_id"]) for item in relation_eligible_candidates
        ]
        result["relation_candidate_states"] = {
            str(item["object_id"]): relation_states[int(item["object_id"])]
            for item in relation_non_refuted_candidates
        }
        result["relation_direct_candidate_states"] = {
            str(item["object_id"]): (
                object_reference_resolver.direct_candidate_states.get(
                    (target_id, int(item["object_id"])), "UNKNOWN"
                )
                if object_reference_resolver is not None else "YES"
            )
            for item in relation_non_refuted_candidates
        }
        result["relation_verifications"] = list(
            object_reference_resolver.verifications
            if object_reference_resolver is not None else ()
        )
        result["candidate_domains"] = dict(
            object_reference_resolver.candidate_domains
            if object_reference_resolver is not None else {}
        )
        result["selector_resolutions"] = list(
            object_reference_resolver.selector_resolutions
            if object_reference_resolver is not None else ()
        )
        result["candidate_domains"].setdefault(
            target_id, [dict(value) for value in candidates]
        )
    target_entity = resolver.entities[target_id]
    if (
        not target_entity.get("attributes")
        and not (task_type == "object_reference" and object_reference_relations)
    ):
        target_names = _entity_class_names(target_entity)
        probe_candidates = [
            item for item in snapshot["objects"]
            if _matches_class_names(item, target_names)
            and item.get("status") in {"tentative", "confirmed"}
            and str(item.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
        ]
        if probe_candidates:
            result["probe_object"] = sorted(
                probe_candidates,
                key=lambda item: (-float(item.get("semantic_probability", 0.0)), int(item["object_id"])),
            )[0]
            if not result["evidence_ids"]:
                result["evidence_ids"] = sorted({
                    f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                    for evidence in result["probe_object"].get("evidence", [])
                })
    if task_type == "numerical":
        target_relations = list(
            relation_plan.get("relations", ()) if relation_plan is not None else ()
        )
        cardinality_candidates = list(candidates)
        cardinality = _numerical_cardinality_posterior(
            cardinality_candidates
        )
        result["uncertainty"]["cardinality_posterior"] = cardinality
        coverage_estimator = _numerical_coverage_estimator(
            snapshot,
            snapshot.get("count_query_execution")
            if isinstance(snapshot.get("count_query_execution"), Mapping)
            else None,
            cardinality_candidates,
        )
        result["uncertainty"]["count_coverage_estimator"] = coverage_estimator
        count_readiness = _numerical_count_readiness(
            snapshot,
            failures,
        )
        graph_execution = snapshot.get("count_query_execution")
        if isinstance(graph_execution, Mapping):
            result["count_query_execution"] = dict(graph_execution)
            result["relation_verifications"] = [
                dict(value)
                for value in graph_execution.get("relation_verifications", ())
                if isinstance(value, Mapping)
            ]
            result["uncertainty"]["count_query_execution"] = dict(
                graph_execution
            )
        result["uncertainty"]["numerical_count_readiness"] = count_readiness
        result["uncertainty"]["numerical_count_domain_open"] = not bool(
            count_readiness["ready"]
        )
        result["numerical_count_answer_ready"] = bool(
            count_readiness["ready"]
        )
        result["numerical_count_domain_state"] = str(
            count_readiness["state"]
        )
        if isinstance(graph_execution, Mapping):
            result["failed_constraints"] = list(dict.fromkeys([
                *result.get("failed_constraints", ()),
                *(
                    str(value)
                    for value in graph_execution.get("failure_reasons", ())
                ),
            ]))
        if count_readiness["evidence_ids"]:
            result["evidence_ids"] = sorted(set(
                result["evidence_ids"]
                + list(count_readiness["evidence_ids"])
            ))
        targeted_probe = (
            _numerical_probe_object_from_snapshot(task_ir, snapshot, relation_plan)
            if snapshot.get("count_query_targeted_acquisition")
            else None
        )
        if targeted_probe is not None and not count_readiness["ready"]:
            result["probe_object"] = targeted_probe
            result["failed_constraints"] = ["count_query_acquisition_pending"]
            return result
        if count_readiness["ready"]:
            result["resolution_mode"] = str(
                count_readiness["source"] or "count_query_graph"
            )
        elif not failures:
            resolution_mode = (
                "semantic_relation_graph"
                if target_relations else "class_instance_count"
            )
            # Preserve the current count as a diagnostic only.  The reason is
            # explicitly count-specific; SceneMemory and instruction selector
            # openness are not reported as a numerical closure failure.
            relation_probe = (
                _numerical_relation_probe_object(
                    resolver, relation_plan
                )
                if target_relations else result.get("probe_object")
            )
            if relation_probe is not None:
                result["probe_object"] = relation_probe
                result["evidence_ids"] = sorted({
                    f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                    for evidence in relation_probe.get("evidence", ())
                }) or list(result.get("evidence_ids", ()))
            result.update(
                provisional_count=int(cardinality["mode"]),
                resolution_mode=resolution_mode,
            )
            result["failed_constraints"] = list(dict.fromkeys([
                *result.get("failed_constraints", ()),
                *count_readiness["reasons"],
            ]))
        return result
    if task_type == "object_reference":
        eligible_candidates = relation_eligible_candidates
        # Exact object-ID relation evidence owns the answer.  Scene index and
        # repeated-view closure remain diagnostics; neither may override a
        # role-grounded Qwen YES with an unconditional acquisition gate.
        if len(eligible_candidates) == 1:
            selected = eligible_candidates[0]
            result.update(
                selected_object=selected,
                resolution_mode=(
                    "qwen_relation_yes"
                    if int(selected["object_id"])
                    in {
                        int(value)
                        for value in result["relation_verified_object_ids"]
                    }
                    else "qwen_relation_non_refuted"
                ),
            )
        elif len(eligible_candidates) > 1:
            result["failed_constraints"].append(
                "object_reference_canonical_identity_not_unique"
            )
        elif object_reference_relations:
            if object_reference_resolver is not None:
                result["probe_object"] = (
                    object_reference_resolver.relation_search_probe(target_id)
                )
                if result["probe_object"] is not None:
                    result["evidence_ids"] = sorted({
                        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                        for evidence in result["probe_object"].get("evidence", ())
                        if evidence.get("acquisition_id")
                        and evidence.get("observation_id")
                    })
            result["failed_constraints"] = list(dict.fromkeys([
                *result["failed_constraints"],
                (
                    "object_reference_relation_search_required"
                    if result.get("probe_object") is not None
                    else "object_reference_relation_unverified"
                ),
            ]))
        elif not result.get("probe_object"):
            result["failed_constraints"].append("object_reference_not_unique")
        return result
    if task_type != "instruction_following":
        result["failed_constraints"].append("unsupported_task_type")
        return result

    normalized_steps = RecedingHorizonInstructionExecutor.normalize_steps(
        task_ir, execution_steps
    )
    active_window = RecedingHorizonInstructionExecutor.window(
        len(normalized_steps), current_step_index
    )
    result.update({
        "execution_steps": normalized_steps,
        "current_step_index": int(current_step_index),
        "current_step_status": (
            "SATISFIED"
            if current_step_index == len(normalized_steps)
            else str(normalized_steps[current_step_index]["status"])
        ),
        "current_step_binding": None,
        "current_step_ready": False,
        "unresolved_slots": [
            str(step["target_slot"])
            for step in normalized_steps
            if step.get("bound_object_id") is None
        ],
        "remaining_required_classes": [],
        "execution_steps_exhausted": current_step_index == len(normalized_steps),
    })
    if result["execution_steps_exhausted"]:
        result["failed_constraints"] = []
        return result

    entities = {
        str(value["id"]): value for value in task_ir.get("entities", ())
    }
    object_by_id = {
        int(value["object_id"]): value for value in snapshot.get("objects", ())
    }
    relation_by_id = {
        str(value.get("id", "")): value
        for value in task_ir.get("relations", ())
    }
    resolver_entity_bindings = dict(entity_bindings or {})
    instruction_resolver = _ObjectReferenceRelationResolver(
        task_ir,
        snapshot,
        object_relation_verifier,
        resolver_entity_bindings,
        navigation_history,
    )
    def candidate_priority(item: Mapping[str, Any]) -> tuple[Any, ...]:
        verification = [
            float(value)
            for evidence in item.get("evidence", ())
            for value in evidence.get("qwen_verification_probabilities", ())
            if isinstance(value, (int, float))
        ]
        return (
            max(verification, default=0.0),
            float(item.get("semantic_probability", 0.0)),
            item.get("status") == "confirmed",
            len(item.get("evidence", ())),
            -int(item["object_id"]),
        )

    trajectory_directives: list[dict[str, Any]] = []
    instruction_candidate_selection: list[dict[str, Any]] = []
    current_navigation_xy = _latest_navigation_xy(navigation_history)
    ranking_domains: list[dict[str, Any]] = []
    unresolved_entity_id = ""
    singular_observation_objective: dict[str, Any] | None = None
    singular_probe_candidate: dict[str, Any] | None = None
    open_relation_probe: dict[str, Any] | None = None
    lookahead_directives: list[dict[str, Any]] = []
    result["verification_policy"] = {
        "maximum_explicit_probes_per_step": 1,
        "explicit_probe_selected": False,
        "selection_reason": "motion_toward_current_step_is_perception",
        "semantic_probe_cutoff_elapsed_seconds": 420.0,
    }

    def exact_relation_anchors(
        entity_id: str, selected_id: int
    ) -> list[dict[str, Any]]:
        attempts = []
        state_rank = {"YES": 2, "UNKNOWN": 1, "NO": 0, "INVALID": -1}
        for attempt in instruction_resolver.relation_attempts:
            candidate = attempt.get("candidate", {})
            if (
                str(attempt.get("entity_id", "")) != entity_id
                or not isinstance(candidate, Mapping)
                or int(candidate.get("object_id", -1)) != selected_id
            ):
                continue
            anchors = [
                dict(value)
                for value in attempt.get("bound_anchors", ())
                if isinstance(value, Mapping)
                and RecedingHorizonInstructionExecutor.eligible(value)
            ]
            if not anchors:
                continue
            attempts.append((
                state_rank.get(str(attempt.get("path_state", "UNKNOWN")), -1),
                min(
                    RecedingHorizonInstructionExecutor.object_rank(value)
                    for value in anchors
                ),
                tuple(-int(value["object_id"]) for value in anchors),
                anchors,
            ))
        return max(attempts)[3] if attempts else []

    def bind_one_step(
        raw_step: Mapping[str, Any], *, lookahead_only: bool
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
        step = dict(raw_step)
        step_index = int(step["step_index"])
        entity_id = str(step["target_entity"])
        step_relations = [
            relation_by_id[value]
            for value in step.get("relation_ids", ())
            if value in relation_by_id
        ]
        action = str(step.get("action", ""))
        failures: list[str] = []

        if (
            not lookahead_only
            and str(step.get("status")) == StepStatus.EXECUTING.value
        ):
            if not RecedingHorizonInstructionExecutor.execution_binding_valid(
                step, snapshot
            ):
                invalidated = RecedingHorizonInstructionExecutor.invalidate(
                    step, "frozen_binding_no_longer_eligible"
                )
                result["active_action_invalidated"] = {
                    "step_index": step_index,
                    "target_entity": entity_id,
                    "prior_bound_object_id": step.get("bound_object_id"),
                    "prior_bound_anchor_object_ids": list(
                        step.get("bound_anchor_object_ids", ())
                    ),
                    "reason": "frozen_binding_no_longer_eligible",
                    "transition": ["CANCEL", "INVALIDATED", "RECOMPUTE"],
                }
                rebound, directive, rebound_failures = bind_one_step(
                    invalidated, lookahead_only=False
                )
                if rebound is not None:
                    rebound["recomputed_after_invalidation"] = True
                return rebound, directive, list(dict.fromkeys([
                    f"active_action_invalidated:{step_index}",
                    *rebound_failures,
                ]))
            target = (
                object_by_id.get(int(step["bound_object_id"]))
                if step.get("bound_object_id") is not None
                else None
            )
            anchors = [
                object_by_id[int(value)]
                for value in step.get("bound_anchor_object_ids", ())
                if value is not None and int(value) in object_by_id
            ]
            bound = RecedingHorizonInstructionExecutor.bind(
                step,
                snapshot,
                target=target,
                anchors=anchors,
                between_anchor_only=bool(
                    step.get("between_anchor_only_binding", False)
                ),
            )
            bound["status"] = StepStatus.EXECUTING.value
        elif action == "pass_between":
            between = next((
                value for value in step_relations
                if str(value.get("predicate", "")).lower() == "between"
            ), None)
            anchor_domains: list[list[dict[str, Any]]] = []
            for anchor_entity in (
                between.get("object_entities", ())
                if isinstance(between, Mapping) else ()
            ):
                candidates, anchor_failures = instruction_resolver.resolve(
                    str(anchor_entity)
                )
                failures.extend(anchor_failures)
                anchor_domains.append([
                    dict(value) for value in candidates
                    if RecedingHorizonInstructionExecutor.eligible(value)
                ])
            pair = (
                RecedingHorizonInstructionExecutor.distinct_anchor_pair(
                    anchor_domains[0], anchor_domains[1]
                )
                if len(anchor_domains) == 2 else []
            )
            if len(pair) != 2:
                return None, None, list(dict.fromkeys([
                    *failures,
                    f"between_two_distinct_atomic_anchors_unavailable:{step_index}",
                ]))
            bound = RecedingHorizonInstructionExecutor.bind(
                step,
                snapshot,
                target=None,
                anchors=pair,
                between_anchor_only=True,
                lookahead_only=lookahead_only,
            )
            target = pair[0]
            anchors = pair
        else:
            ranking = next((
                value for value in step_relations
                if str(value.get("predicate", "")).lower()
                in {"closest", "farthest"}
            ), None)
            anchors: list[dict[str, Any]] = []
            challenger = None
            if ranking is not None:
                anchor_entity = next(
                    iter(ranking.get("object_entities", ())), ""
                )
                anchor_candidates, anchor_failures = instruction_resolver.resolve(
                    str(anchor_entity)
                )
                failures.extend(anchor_failures)
                anchor = RecedingHorizonInstructionExecutor.best_atomic(
                    anchor_candidates
                )
                target_candidates = instruction_resolver._class_candidates(
                    entity_id
                )
                target, challenger = (
                    RecedingHorizonInstructionExecutor.distance_winner(
                        target_candidates,
                        anchor,
                        farthest=(
                            str(ranking.get("predicate", "")).lower()
                            == "farthest"
                        ),
                    )
                    if anchor is not None else (None, None)
                )
                if anchor is not None:
                    anchors = [anchor]
                if target is not None:
                    step["selector_binding_authorized"] = True
                    step["selector_provisional"] = False
                    step["selector_state"] = "COMMITTED_CURRENT_DISTANCE_WINNER"
                    step["selector_relation_id"] = str(ranking.get("id", ""))
                    step["selector_evidence_complete"] = True
                    ranking_domains.append({
                        "relation_id": str(ranking.get("id", "")),
                        "predicate": str(ranking.get("predicate", "")),
                        "selected_object_id": int(target["object_id"]),
                        "anchor_object_ids": [int(anchor["object_id"])],
                        "challenger_object_id": (
                            int(challenger["object_id"])
                            if challenger is not None else None
                        ),
                        "ranking_policy": (
                            "fused_horizontal_distance_then_semantic_evidence_lexicographic"
                        ),
                        "explicit_verification_probe": False,
                    })
            else:
                candidates, step_failures = instruction_resolver.resolve(entity_id)
                failures.extend(step_failures)
                target = RecedingHorizonInstructionExecutor.best_atomic(candidates)
                if target is not None and step_relations:
                    anchors = exact_relation_anchors(
                        entity_id, int(target["object_id"])
                    )
            if target is None:
                return None, None, list(dict.fromkeys([
                    *failures,
                    f"active_step_atomic_binding_unavailable:{step_index}",
                ]))
            bound = RecedingHorizonInstructionExecutor.bind(
                step,
                snapshot,
                target=target,
                anchors=anchors,
                lookahead_only=lookahead_only,
            )

        directive = {
            "order": step_index,
            "action": action,
            "terminal": bool(step["is_terminal"]),
            "trajectory_constraint": str(step["trajectory_constraint"]),
            "trajectory_region_kind": str(step["trajectory_region_kind"]),
            "forbidden": bool(step.get("forbidden", False)),
            "object": dict(target) if isinstance(target, Mapping) else {},
            "candidate_objects": [dict(target)] if isinstance(target, Mapping) else [],
            "anchor_objects": [dict(value) for value in anchors],
            "selector_provisional": False,
            "selector_state": str(bound.get("selector_state", "")),
            "selector_relation_id": str(bound.get("selector_relation_id", "")),
            "selector_evidence_complete": bool(
                bound.get("selector_evidence_complete", True)
            ),
            "between_anchor_only_binding": bool(
                bound.get("between_anchor_only_binding", False)
            ),
            "binding_revision": copy.deepcopy(bound.get("binding_revision")),
            "bound_target_object_id": bound.get("bound_object_id"),
            "bound_anchor_object_ids": list(
                bound.get("bound_anchor_object_ids", ())
            ),
            "lookahead_only": bool(lookahead_only),
        }
        return bound, directive, list(dict.fromkeys(failures))

    active_indexes = [active_window.current_index]
    if active_window.lookahead_index is not None:
        active_indexes.append(active_window.lookahead_index)
    for step_index in active_indexes:
        step = normalized_steps[step_index]
        lookahead_only = step_index != active_window.current_index
        bound, directive, step_failures = bind_one_step(
            step, lookahead_only=lookahead_only
        )
        if lookahead_only:
            result.setdefault("lookahead_preparation", []).append({
                "step_index": step_index,
                "ready": directive is not None,
                "failures": step_failures,
                "binding_revision": (
                    copy.deepcopy(bound.get("binding_revision"))
                    if isinstance(bound, Mapping) else None
                ),
                "target_object_id": (
                    bound.get("bound_object_id")
                    if isinstance(bound, Mapping) else None
                ),
                "anchor_object_ids": (
                    list(bound.get("bound_anchor_object_ids", ()))
                    if isinstance(bound, Mapping) else []
                ),
            })
            if directive is not None:
                lookahead_directives.append(directive)
            continue
        result["failed_constraints"].extend(step_failures)
        if bound is None or directive is None:
            normalized_steps[step_index] = (
                bound
                if isinstance(bound, Mapping)
                else RecedingHorizonInstructionExecutor.block(
                    step,
                    step_failures[0]
                    if step_failures else "active_step_binding_unavailable",
                )
            )
            unresolved_entity_id = str(step["target_entity"])
            break
        normalized_steps[step_index] = bound
        trajectory_directives.append(directive)
        instruction_candidate_selection.append({
            "step_index": step_index,
            "selection_mode": "stable_task_local_lexicographic",
            "selected_object_id": bound.get("bound_object_id"),
            "selected_anchor_object_ids": list(
                bound.get("bound_anchor_object_ids", ())
            ),
            "binding_revision": copy.deepcopy(bound.get("binding_revision")),
        })

    # Disabled legacy active-step planner: the receding-horizon owner above
    # is the only production Instruction binding/execution path.
    for step in ():
        if str(step.get("status")) == "SATISFIED":
            continue
        step_index = int(step["step_index"])
        if step_index < current_step_index:
            continue
        entity_id = str(step["target_entity"])
        step_relations = [
            relation_by_id[value]
            for value in step.get("relation_ids", ())
            if value in relation_by_id
        ]
        bound_object_id = step.get("bound_object_id")
        if bound_object_id is not None and not step_relations:
            candidates_for_step = [
                value for value in snapshot.get("objects", ())
                if int(value.get("object_id", -1)) == int(bound_object_id)
                and str(value.get("cardinality_role", "")) not in {
                    "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
                }
            ]
            step_failures = [] if candidates_for_step else [
                f"bound_object_missing:{bound_object_id}"
            ]
        else:
            objects, step_failures = instruction_resolver.resolve(entity_id)
            ranking = next((
                relation for relation in step_relations
                if str(relation.get("predicate", "")).lower()
                in {"closest", "farthest"}
            ), None)
            if ranking is not None:
                resolution = next((
                    value for value in instruction_resolver.selector_resolutions
                    if str(value.get("relation_id", ""))
                    == str(ranking.get("id", ""))
                ), None)
                probe_required = bool(
                    isinstance(resolution, Mapping)
                    and resolution.get("selector_provisional") is True
                )
                if probe_required:
                    probe = instruction_resolver.relation_search_probe(entity_id)
                    if isinstance(probe, Mapping):
                        open_relation_probe = dict(probe)
                        open_relation_probe.update({
                            "selector_provisional": True,
                            "selector_state": str(
                                resolution.get(
                                    "selector_state",
                                    "PROVISIONAL_GEOMETRIC_WINNER",
                                )
                            ),
                            "selector_relation_id": str(
                                ranking.get("id", "")
                            ),
                            "selector_evidence_complete": False,
                            "probe_intent": "PROBE_EVIDENCE",
                            "probe_reason": "selector_evidence_incomplete",
                            "selector_hypotheses": list(
                                resolution.get("selector_hypotheses", ())
                            ),
                        })
                    candidates_for_step = []
                    step_failures.append(
                        f"selector_probe_evidence_required:{ranking.get('id', '')}"
                    )
                elif (
                    resolution is None
                    or not resolution.get("candidate_object_ids")
                    or resolution.get("selected_object_id") is None
                    or not objects
                ):
                    candidates_for_step = []
                    step_failures.append(
                        f"ranking_relation_unresolved:{ranking.get('id', '')}"
                    )
                else:
                    candidates_for_step = [objects[0]]
                    ranking_domains.append(dict(resolution))
                    step["selector_binding_authorized"] = True
                    step["selector_provisional"] = bool(
                        resolution.get("selector_provisional", True)
                    )
                    step["selector_state"] = str(
                        resolution.get("selector_state", "UNKNOWN")
                    )
                    step["selector_relation_id"] = str(
                        resolution.get("relation_id", ranking.get("id", ""))
                    )
                    step["selector_evidence_complete"] = bool(
                        resolution.get("selector_evidence_complete", False)
                    )
            elif step_relations:
                # Relation verifiers annotate the current hypotheses; UNKNOWN
                # is a score/state, not a candidate-domain gate.
                candidates_for_step = list(objects)
            else:
                candidates_for_step = list(objects)

        result["failed_constraints"].extend(step_failures)
        if not candidates_for_step:
            unresolved_entity_id = entity_id
            result["failed_constraints"].append(
                f"trajectory_constraint_missing:{step_index}"
            )
            break

        # Every valid track is a route hypothesis.  Verification confidence,
        # station count, and physical-status labels remain evidence fields;
        # they do not gate the current semantic candidate set.
        confirmed_candidates = list(candidates_for_step)
        unsettled_semantic_candidates: list[dict[str, Any]] = []
        resolver_bound_id = resolver_entity_bindings.get(entity_id)
        resolver_binding_applied = bool(
            resolver_bound_id is not None
            and int(resolver_bound_id) >= 0
            and any(
                int(value["object_id"]) == int(resolver_bound_id)
                for value in candidates_for_step
            )
        )
        resolver_binding_applied = bool(
            resolver_binding_applied
            or step.get("selector_binding_authorized") is True
        )
        candidates_for_step = confirmed_candidates
        if step_relations:
            # Relation-filter and selector semantics have already produced
            # their exact-ID domain. Keep their relation evidence ordering
            # authoritative; route-aware ranking is for ordinary ordered
            # referents whose ambiguity is otherwise resolved by confidence.
            selected = max(candidates_for_step, key=candidate_priority)
        else:
            ranked_candidates = [
                (
                    *_instruction_candidate_priority(
                        candidate,
                        candidates_for_step,
                        current_xy=current_navigation_xy,
                        navigation_history=navigation_history,
                        step_index=step_index,
                    ),
                    candidate,
                )
                for candidate in candidates_for_step
            ]
            ranked_candidates.sort(key=lambda value: value[0], reverse=True)
            selected = ranked_candidates[0][2]
            instruction_candidate_selection.append({
                "step_index": int(step_index),
                "entity_id": entity_id,
                "selection_mode": "route_aware_open_world_hypothesis",
                "current_pose_xy": (
                    list(current_navigation_xy)
                    if current_navigation_xy is not None else None
                ),
                "selected_object_id": int(selected["object_id"]),
                "candidates": [
                    dict(details)
                    for _priority, details, _candidate in ranked_candidates
                ],
            })
        step_relation_ids = {
            str(value) for value in step.get("relation_ids", ())
        }
        selector_resolution = next(
            (
                value
                for value in instruction_resolver.selector_resolutions
                if str(value.get("relation_id", "")) in step_relation_ids
            ),
            None,
        )
        anchor_ids = list(
            selector_resolution.get("anchor_object_ids", ())
            if isinstance(selector_resolution, Mapping)
            else ()
        )
        if not anchor_ids:
            anchor_ids = [
                int(value["object_id"])
                for binding in instruction_resolver.relation_anchor_bindings
                if str(binding.get("entity_id", "")) == entity_id
                for value in binding.get("bound_anchors", ())
            ]
        if anchor_ids:
            step["bound_anchor_object_ids"] = list(dict.fromkeys(
                int(value) for value in anchor_ids
            ))
        if (
            bound_object_id is not None
            or resolver_binding_applied
            or len(candidates_for_step) == 1
        ):
            step["bound_object_id"] = int(selected["object_id"])
        step["status"] = (
            str(step["status"])
            if str(step["status"]) in {"EXECUTING", "SATISFIED"}
            else "BOUND"
        )
        support_object = None
        if isinstance(selected.get("support_parent_id"), (int, str)):
            try:
                support_object = object_by_id.get(int(selected["support_parent_id"]))
            except (TypeError, ValueError):
                support_object = None
        directive_anchors = [
            dict(object_by_id[object_id])
            for object_id in step.get("bound_anchor_object_ids", ())
            if object_id in object_by_id
        ]
        if support_object is not None and not any(
            int(value.get("object_id", -1)) == int(support_object.get("object_id", -2))
            for value in directive_anchors
        ):
            directive_anchors.append(dict(support_object))
        trajectory_directive = {
            "order": step_index,
            "action": str(step["action"]),
            "terminal": bool(step["is_terminal"]),
            "trajectory_constraint": str(step["trajectory_constraint"]),
            "trajectory_region_kind": str(step["trajectory_region_kind"]),
            "forbidden": bool(step.get("forbidden", False)),
            "object": selected,
            "candidate_objects": [dict(value) for value in candidates_for_step],
            "anchor_objects": directive_anchors,
            "selector_provisional": bool(
                step.get("selector_provisional", False)
            ),
            "selector_state": str(step.get("selector_state", "")),
            "selector_relation_id": str(
                step.get("selector_relation_id", "")
            ),
            "selector_evidence_complete": bool(
                step.get("selector_evidence_complete", False)
            ),
        }
        trajectory_directives.append(trajectory_directive)

    if unresolved_entity_id:
        active = normalized_steps[current_step_index]
        result.update({
            "execution_steps": normalized_steps,
            "current_step_status": str(active.get("status", StepStatus.BLOCKED.value)),
            "current_step_ready": False,
            "current_step_binding": None,
            "trajectory_directives": [],
            "instruction_candidate_selection": instruction_candidate_selection,
            "relation_verifications": list(instruction_resolver.verifications),
            "candidate_domains": dict(instruction_resolver.candidate_domains),
            "ranking_domains": ranking_domains,
            "selector_resolutions": list(instruction_resolver.selector_resolutions),
            "probe_object": None,
            "explicit_verification_probe": None,
            "unresolved_slots": [str(active["target_slot"])],
            "remaining_required_classes": list(dict.fromkeys(
                str(entities[value]["class_name"])
                for step in normalized_steps[current_step_index:]
                for value in (
                    step["target_entity"], *step.get("anchor_entities", ())
                )
                if value in entities
            )),
            "active_constraint_order": int(current_step_index),
            "remaining_constraint_orders": list(
                range(current_step_index, len(normalized_steps))
            ),
        })
        result["failed_constraints"] = list(dict.fromkeys(
            str(value) for value in result["failed_constraints"]
        ))
        return result

    trajectory_directives.extend(lookahead_directives)

    if not unresolved_entity_id:
        for constraint in (
            task_ir.get("trajectory_ir") or {}
        ).get("forbidden_path_regions", ()):
            entity_id = str(constraint["target_entity"])
            objects, constraint_failures = instruction_resolver.resolve(entity_id)
            confirmed = list(objects)
            result["failed_constraints"].extend(constraint_failures)
            if not confirmed:
                unresolved_entity_id = entity_id
                result["failed_constraints"].append(
                    f"forbidden_trajectory_referent_unresolved:"
                    f"{constraint.get('order')}:0"
                )
                break
            forbidden_relation_ids = {
                str(value) for value in constraint.get("relation_ids", ())
            }
            forbidden_anchor_ids = [
                int(value["object_id"])
                for binding in instruction_resolver.relation_anchor_bindings
                if (
                    str(binding.get("entity_id", "")) == entity_id
                    and str((binding.get("relation") or {}).get("id", ""))
                    in forbidden_relation_ids
                )
                for value in binding.get("bound_anchors", ())
            ]
            forbidden_object = max(confirmed, key=candidate_priority)
            forbidden_step = {
                "order": int(constraint["order"]),
                "action": str(constraint["action"]),
                "terminal": False,
                "trajectory_constraint": "AVOID_REGION",
                "trajectory_region_kind": str(constraint["region_kind"]),
                "forbidden": True,
                "object": forbidden_object,
                "candidate_objects": [dict(value) for value in confirmed],
                "anchor_objects": [
                    dict(object_by_id[object_id])
                    for object_id in dict.fromkeys(forbidden_anchor_ids)
                    if object_id in object_by_id
                ],
            }
            trajectory_directives.append(forbidden_step)

    if unresolved_entity_id:
        result["current_step_status"] = "UNRESOLVED"
        result["current_step_ready"] = False
        result["current_step_binding"] = None
        unresolved_selector = next((
            value for value in instruction_resolver.selector_resolutions
            if value.get("candidate_domain_closed") is False
        ), None)
        result["probe_object"] = (
            dict(open_relation_probe)
            if isinstance(open_relation_probe, Mapping)
            else None
        )
        if unresolved_selector is None:
            if result["probe_object"] is None:
                result["probe_object"] = singular_probe_candidate
            if result["probe_object"] is None:
                result["probe_object"] = (
                    instruction_resolver.relation_search_probe(
                        unresolved_entity_id
                    )
                )
        if result["probe_object"] is None:
            result["probe_object"] = {
                "class_label": "observe",
                "target_kind": "observation_frontier",
                "probe_relation": (
                    "selector_bound_frontier_observation"
                    if unresolved_selector is not None
                    else "frontier_observation"
                ),
                "observation_objective": dict(unresolved_selector or {}),
            }
        if unresolved_selector is None and singular_observation_objective:
            result["probe_object"]["observation_objective"] = dict(
                singular_observation_objective
            )
        bound_prefix = []
        object_by_id = {
            int(value["object_id"]): value
            for value in snapshot.get("objects", ())
            if str(value.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
        }
        for prior_step in normalized_steps[:current_step_index]:
            object_id = prior_step.get("bound_object_id")
            candidate = (
                object_by_id.get(int(object_id))
                if object_id is not None else None
            )
            if candidate is not None and _valid_center_3d(candidate):
                bound_prefix.append(candidate)
        continuation_objective: dict[str, Any] = {}
        if len(bound_prefix) >= 2:
            first = bound_prefix[-2]["center_3d"]
            second = bound_prefix[-1]["center_3d"]
            direction = [
                float(second[index]) - float(first[index])
                for index in (0, 1)
            ]
            if math.hypot(*direction) > 0.0:
                continuation_objective = {
                    "trajectory_continuation_origin_xy": [
                        float(second[0]), float(second[1])
                    ],
                    "trajectory_continuation_direction_xy": direction,
                    "trajectory_continuation_orders": [
                        int(value["step_index"])
                        for value in normalized_steps[
                            max(0, current_step_index - 2):current_step_index
                        ]
                    ],
                }
        if not continuation_objective:
            actual_arrivals: list[list[float]] = []
            dominant_actual_segment: tuple[
                float, list[float], list[float], int
            ] | None = None
            for attempt in navigation_history or ():
                pose = attempt.get("actual_arrival_pose")
                if not (
                    isinstance(pose, Sequence)
                    and not isinstance(pose, (str, bytes))
                    and len(pose) >= 2
                ):
                    continue
                xy = [float(pose[0]), float(pose[1])]
                if not all(math.isfinite(value) for value in xy):
                    continue
                if not actual_arrivals or xy != actual_arrivals[-1]:
                    actual_arrivals.append(xy)
                start_pose = attempt.get("start_pose")
                try:
                    attempt_step_index = int(attempt.get("step_index", -1))
                except (TypeError, ValueError):
                    attempt_step_index = -1
                if not (
                    attempt_step_index < current_step_index
                    and isinstance(start_pose, Sequence)
                    and not isinstance(start_pose, (str, bytes))
                    and len(start_pose) >= 2
                ):
                    continue
                start_xy = [float(start_pose[0]), float(start_pose[1])]
                if not all(math.isfinite(value) for value in start_xy):
                    continue
                displacement = math.dist(start_xy, xy)
                if (
                    dominant_actual_segment is None
                    or displacement > dominant_actual_segment[0]
                ):
                    dominant_actual_segment = (
                        displacement, start_xy, xy, attempt_step_index
                    )
            if dominant_actual_segment is not None:
                _, first, _, source_step_index = dominant_actual_segment
                second = actual_arrivals[-1]
                direction = [
                    dominant_actual_segment[2][index] - first[index]
                    for index in (0, 1)
                ]
                continuation_objective = {
                    "trajectory_continuation_origin_xy": second,
                    "trajectory_continuation_direction_xy": direction,
                    "trajectory_continuation_source": (
                        "dominant_actual_completed_prefix_segment"
                    ),
                    "trajectory_continuation_source_step_index": (
                        source_step_index
                    ),
                }
            elif len(actual_arrivals) >= 2:
                first, second = actual_arrivals[-2:]
                direction = [
                    second[index] - first[index] for index in (0, 1)
                ]
                if math.hypot(*direction) > 0.0:
                    continuation_objective = {
                        "trajectory_continuation_origin_xy": second,
                        "trajectory_continuation_direction_xy": direction,
                        "trajectory_continuation_source": (
                            "actual_navigation_arrival_trajectory"
                        ),
                    }
        if continuation_objective:
            result["probe_object"]["observation_objective"] = {
                **continuation_objective,
                **dict(
                    result["probe_object"].get(
                        "observation_objective", {}
                    )
                ),
            }
        active_step = normalized_steps[current_step_index]
        semantic_probe_object = singular_probe_candidate
        if semantic_probe_object is None and isinstance(
            result.get("probe_object"), Mapping
        ):
            candidate_probe = result["probe_object"]
            if str(candidate_probe.get("class_label", "")).lower() != "observe":
                semantic_probe_object = dict(candidate_probe)
        if semantic_probe_object is not None:
            semantic_hint = {
                "order": int(active_step["step_index"]),
                "action": str(active_step.get("action", "")),
                "terminal": bool(active_step.get("is_terminal", False)),
                "forbidden": bool(active_step.get("forbidden", False)),
                "trajectory_constraint": str(
                    active_step.get("trajectory_constraint", "ENTER_REGION")
                ),
                "trajectory_region_kind": str(
                    active_step.get(
                        "trajectory_region_kind",
                        "STOP_REGION"
                        if active_step.get("is_terminal")
                        else "NEAR_REGION",
                    )
                ),
                "object": dict(semantic_probe_object),
                "candidate_objects": [dict(semantic_probe_object)],
                "anchor_objects": [
                    dict(object_by_id[object_id])
                    for object_id in active_step.get(
                        "bound_anchor_object_ids", ()
                    )
                    if object_id in object_by_id
                ],
            }
            known_hints = [semantic_hint]
            known_hints.extend(
                dict(value) for value in trajectory_directives
                if isinstance(value, Mapping)
            )
            result["probe_object"]["known_trajectory_directives"] = known_hints
            result["probe_object"]["probe_step_index"] = int(
                active_step["step_index"]
            )
            result["probe_object"]["probe_semantic_action"] = str(
                active_step.get("action", "")
            )
        probe_center = result["probe_object"].get("center_3d")
        if (
            continuation_objective
            and str(result["probe_object"].get("class_label", "")).lower()
            != "observe"
            and isinstance(probe_center, Sequence)
            and not isinstance(probe_center, (str, bytes))
            and len(probe_center) >= 2
        ):
            origin = continuation_objective["trajectory_continuation_origin_xy"]
            direction = continuation_objective[
                "trajectory_continuation_direction_xy"
            ]
            signed_progress = sum(
                (float(probe_center[index]) - float(origin[index]))
                * float(direction[index])
                for index in (0, 1)
            )
            if signed_progress < 0.0:
                prior_probe = dict(result["probe_object"])
                # The resolver ranks evidence incompleteness before trajectory
                # direction.  For instruction following, do not discard a
                # different unresolved tuple that lies beyond the completed
                # ordered prefix merely because an older backward candidate
                # was ranked first.  Re-observe the best forward subject and
                # its exact anchors; only fall back to a generic forward
                # frontier when no such semantic hypothesis exists.
                forward_attempts = []
                for attempt in instruction_resolver.relation_attempts:
                    candidate = attempt.get("candidate", {})
                    center = candidate.get("center_3d")
                    if (
                        str(attempt.get("entity_id", ""))
                        != unresolved_entity_id
                        or str(attempt.get("path_state", "")) != "UNKNOWN"
                        or not attempt.get("bound_anchors")
                        or not isinstance(center, Sequence)
                        or isinstance(center, (str, bytes))
                        or len(center) < 2
                    ):
                        continue
                    progress = sum(
                        (float(center[index]) - float(origin[index]))
                        * float(direction[index])
                        for index in (0, 1)
                    )
                    if progress >= 0.0:
                        forward_attempts.append((progress, attempt))
                if forward_attempts:
                    _progress, attempt = max(
                        forward_attempts,
                        key=lambda value: (
                            value[0],
                            -len(value[1].get("candidate", {}).get("evidence", ())),
                        ),
                    )
                    candidate = dict(attempt["candidate"])
                    anchors = list(attempt["bound_anchors"])
                    candidate.update({
                        "probe_relation": str(
                            attempt.get("relation", {}).get("predicate", "")
                        ).lower(),
                        "probe_anchor_object_ids": [
                            int(value["object_id"]) for value in anchors
                        ],
                        "probe_source_candidate_id": int(
                            candidate["object_id"]
                        ),
                        "probe_target_class": str(
                            instruction_resolver.entities.get(
                                unresolved_entity_id, {}
                            ).get("class_name", "relation_target")
                        ),
                    })
                    candidate["evidence"] = [
                        *candidate.get("evidence", ()),
                        *(
                            evidence
                            for anchor in anchors
                            for evidence in anchor.get("evidence", ())
                        ),
                    ]
                    result["probe_object"] = candidate
                else:
                    forward_probe = {
                        "class_label": "observe",
                        "target_kind": "instruction_forward_relation_search",
                        "probe_relation": "instruction_forward_relation_search",
                        "observation_objective": {
                            **continuation_objective,
                            "probe_target_class": str(
                                prior_probe.get("probe_target_class", "")
                            ),
                        },
                    }
                    try:
                        deferred_id = int(prior_probe.get("object_id"))
                    except (TypeError, ValueError):
                        deferred_id = None
                    if deferred_id is not None and deferred_id >= 0:
                        forward_probe["observation_objective"][
                            "deferred_backward_probe_object_id"
                        ] = deferred_id
                    result["probe_object"] = forward_probe
        if (
            str(result["probe_object"].get("class_label", "")).lower()
            == "observe"
            and current_step_index >= 2
        ):
            objective = dict(
                result["probe_object"].get("observation_objective", {})
            )
            objective.update(continuation_objective)
            result["probe_object"]["observation_objective"] = objective
        navigation_target = _probe_navigation_target(result["probe_object"])
        if navigation_target is not None:
            result["probe_object"]["navigation_target_xy"] = navigation_target
        if trajectory_directives:
            result["probe_object"]["known_trajectory_directives"] = [
                dict(value) for value in trajectory_directives
            ]
        trajectory_directives = []
    else:
        # A resolved semantic step owns the action envelope.  Resolver-local
        # probe candidates are useful while binding, but retaining one beside
        # an executable trajectory would recreate the old mixed
        # instruction-hypothesis/probe state.
        result["probe_object"] = None
        active = normalized_steps[current_step_index]
        result["current_step_status"] = str(active["status"])
        result["current_step_ready"] = bool(trajectory_directives)
        result["current_step_binding"] = {
            "target_slot": str(active["target_slot"]),
            "object_id": (
                int(active["bound_object_id"])
                if active.get("bound_object_id") is not None else None
            ),
            "entity_id": str(active["target_entity"]),
            "class_name": str(
                entities[str(active["target_entity"])]["class_name"]
            ),
            "anchor_slots": list(active.get("anchor_slots", ())),
            "anchor_object_ids": list(
                active.get("bound_anchor_object_ids", ())
            ),
            "binding_revision": copy.deepcopy(
                active.get("binding_revision")
            ),
            "between_anchor_only_binding": bool(
                active.get("between_anchor_only_binding", False)
            ),
        }
        result["failed_constraints"] = []

    result["execution_steps"] = normalized_steps
    result["trajectory_directives"] = trajectory_directives
    result["instruction_candidate_selection"] = instruction_candidate_selection
    result["relation_verifications"] = list(instruction_resolver.verifications)
    result["candidate_domains"] = dict(instruction_resolver.candidate_domains)
    result["ranking_domains"] = ranking_domains
    result["selector_resolutions"] = list(
        instruction_resolver.selector_resolutions
    )
    trajectory_evidence_ids = {
        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
        for trajectory_directive in trajectory_directives
        for source in (trajectory_directive["object"],)
        for evidence in source.get("evidence", ())
        if evidence.get("acquisition_id") and evidence.get("observation_id")
    }
    probe_source = result.get("probe_object")
    probe_evidence_ids = {
        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
        for evidence in (
            probe_source.get("evidence", ())
            if isinstance(probe_source, Mapping) else ()
        )
        if evidence.get("acquisition_id") and evidence.get("observation_id")
    }
    result["evidence_ids"] = sorted({
        *(
            str(value) for value in result.get("evidence_ids", ())
            if str(value).strip()
        ),
        *trajectory_evidence_ids,
        *probe_evidence_ids,
    })
    bound_entities = {
        str(step["target_entity"])
        for step in normalized_steps
        if (
            step.get("bound_object_id") is not None
            or str(step.get("status"))
            in {
                StepStatus.READY.value,
                StepStatus.EXECUTING.value,
                StepStatus.SATISFIED.value,
            }
        )
    }
    remaining_entity_ids = list(dict.fromkeys(
        str(value)
        for step in normalized_steps[current_step_index:]
        for value in (step["target_entity"], *step.get("anchor_entities", ()))
        if str(value) not in bound_entities
    ))
    result["unresolved_slots"] = [
        str(step["target_slot"])
        for step in normalized_steps[current_step_index:]
        if str(step.get("status")) not in {
            StepStatus.READY.value,
            StepStatus.EXECUTING.value,
            StepStatus.SATISFIED.value,
        }
    ]
    result["remaining_required_classes"] = list(dict.fromkeys(
        str(entities[value]["class_name"])
        for value in remaining_entity_ids
        if value in entities
    ))
    result["failed_constraints"] = list(dict.fromkeys(
        str(value) for value in result["failed_constraints"]
    ))
    result["active_constraint_order"] = int(current_step_index)
    result["remaining_constraint_orders"] = list(
        range(current_step_index, len(normalized_steps))
    )
    return result
