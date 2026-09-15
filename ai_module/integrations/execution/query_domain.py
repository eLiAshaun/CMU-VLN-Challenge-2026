"""Geometric query candidates over the current persistent SceneMemory tracks.

Every acquisition produces a new snapshot of canonical object tracks. Ranking
relations are therefore recomputed from that snapshot; a previously selected
object is only the current winner and never closes the candidate domain.
Semantic verification and track quality remain evidence fields, not a hidden
replacement for the relation's geometric operator.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence


def _valid_geometry(value: Mapping[str, Any]) -> bool:
    center = value.get("center_3d")
    extent = value.get("bbox_3d")
    if not (
        isinstance(center, Sequence) and not isinstance(center, (str, bytes))
        and len(center) == 3
        and isinstance(extent, Sequence) and not isinstance(extent, (str, bytes))
        and len(extent) == 3
    ):
        return False
    try:
        values = [float(item) for item in (*center, *extent)]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in values) and all(item > 0.0 for item in values[3:])


def relation_identity_blockers(
    value: Mapping[str, Any],
    *,
    identity_ambiguous: bool = False,
) -> list[str]:
    """Describe why one track is not yet a canonical relation identity.

    High-recall discovery tracks remain in the candidate domain.  This helper
    only separates those tracks from the evidence that is allowed to close a
    geometric relation or bind a terminal waypoint.  It is a property of the
    current SceneMemory evidence, not a run preflight or a retry/token gate.
    """
    blockers: list[str] = []
    if str(value.get("status", "confirmed")).lower() != "confirmed":
        blockers.append("physical_identity_pending")
    if str(value.get("semantic_status", "unverified")).lower() != "verified":
        blockers.append("semantic_identity_pending")
    if value.get("identity_association_hypotheses"):
        blockers.append("association_identity_ambiguous")
    if identity_ambiguous:
        blockers.append("identity_ambiguity_group_open")
    identity_state = str(value.get("identity_state", "")).upper()
    if "AMBIGU" in identity_state:
        blockers.append("association_identity_ambiguous")
    if str(value.get("geometry_evidence_policy", "")).lower() == "all_evidence_provisional":
        blockers.append("geometry_consensus_pending")
    return list(dict.fromkeys(blockers))


def relation_identity_ready(
    value: Mapping[str, Any],
    *,
    identity_ambiguous: bool = False,
) -> bool:
    """Whether SceneMemory can currently serve this track as relation truth."""
    return not relation_identity_blockers(
        value,
        identity_ambiguous=identity_ambiguous,
    )


def _class_names(entity: Mapping[str, Any]) -> set[str]:
    names = {str(entity.get("class_name", "")).strip().lower()}
    names.update(
        str(value).strip().lower()
        for value in entity.get("aliases", ())
        if str(value).strip()
    )
    names.discard("")
    if "lamp" in names or any(name.endswith(" lamp") for name in names):
        names.update({"lamp", "table lamp", "floor lamp", "wall lamp", "bedside lamp"})
    return names


def _matches_entity(value: Mapping[str, Any], entity: Mapping[str, Any]) -> bool:
    label = str(value.get("class_label", value.get("canonical_class", ""))).strip().lower()
    names = _class_names(entity)
    return label in names or ("lamp" in names and (label == "lamp" or label.endswith(" lamp")))


def _center_sigma_xy(value: Mapping[str, Any]) -> float:
    covariance = value.get("center_cov")
    if isinstance(covariance, Sequence) and len(covariance) >= 2:
        try:
            variance = max(0.0, float(covariance[0][0])) + max(0.0, float(covariance[1][1]))
            if math.isfinite(variance):
                return math.sqrt(variance)
        except (TypeError, ValueError, IndexError):
            pass
    return 0.25


def _distance_xy(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_center = first.get("center_3d", ())
    second_center = second.get("center_3d", ())
    return math.hypot(
        float(first_center[0]) - float(second_center[0]),
        float(first_center[1]) - float(second_center[1]),
    )


def _track_quality(value: Mapping[str, Any]) -> float:
    """Return a continuous quality signal for ranking relation hypotheses.

    This is deliberately not a membership gate.  Low-recall discovery tracks
    stay in the candidate domain; their uncertainty simply makes a geometric
    selector less willing to call them the current winner.
    """
    point_count = value.get("geometry_point_count", value.get("point_count", 0))
    try:
        point_count = max(0.0, float(point_count or 0.0))
    except (TypeError, ValueError):
        point_count = 0.0
    views = value.get("independent_viewpoint_count", 0)
    try:
        views = max(0.0, float(views or 0.0))
    except (TypeError, ValueError):
        views = 0.0
    semantic = value.get("semantic_probability", 0.0)
    try:
        semantic = max(0.0, min(1.0, float(semantic or 0.0)))
    except (TypeError, ValueError):
        semantic = 0.0
    sigma = _center_sigma_xy(value)
    point_support = 1.0 - math.exp(-point_count / 24.0)
    view_support = 1.0 - math.exp(-views / 1.5)
    geometric = math.exp(-sigma / 0.75)
    status_support = 1.0 if str(value.get("status", "")) == "confirmed" else 0.5
    return (
        0.30 * point_support
        + 0.25 * view_support
        + 0.20 * geometric
        + 0.15 * semantic
        + 0.10 * status_support
    )


def _distance_record(
    subject: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    distance = _distance_xy(subject, anchor)
    uncertainty = _center_sigma_xy(subject) + _center_sigma_xy(anchor)
    # A relation selector needs a conservative world-space interval.  This
    # does not discard a noisy candidate; it records why that candidate is not
    # yet allowed to overwrite a better-supported identity.
    interval_radius = 2.0 * uncertainty
    return {
        "subject_object_id": int(subject["object_id"]),
        "anchor_object_id": int(anchor["object_id"]),
        "horizontal_distance_m": distance,
        "uncertainty_m": uncertainty,
        "lower_bound_m": max(0.0, distance - interval_radius),
        "upper_bound_m": distance + interval_radius,
        "subject_track_quality": _track_quality(subject),
        "anchor_track_quality": _track_quality(anchor),
        "selection_metric": "horizontal_distance_m",
        "uncertainty_model": "two_sigma_center_sum",
    }


def _selector_hypothesis(
    predicate: str,
    anchor: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    if predicate == "farthest":
        ordered = sorted(
            records,
            key=lambda value: (
                float(value.get("lower_bound_m", 0.0)),
                float(value.get("horizontal_distance_m", 0.0)),
                float(value.get("subject_track_quality", 0.0)),
                -int(value["subject_object_id"]),
            ),
            reverse=True,
        )
        winner = ordered[0]
        competitor_upper = max(
            (
                float(value.get("upper_bound_m", 0.0))
                for value in ordered[1:]
            ),
            default=None,
        )
        stability_margin = (
            float(winner.get("lower_bound_m", 0.0)) - competitor_upper
            if competitor_upper is not None
            else None
        )
    else:
        ordered = sorted(
            records,
            key=lambda value: (
                float(value.get("upper_bound_m", 0.0)),
                float(value.get("horizontal_distance_m", 0.0)),
                -float(value.get("subject_track_quality", 0.0)),
                int(value["subject_object_id"]),
            ),
        )
        winner = ordered[0]
        competitor_lower = min(
            (
                float(value.get("lower_bound_m", 0.0))
                for value in ordered[1:]
            ),
            default=None,
        )
        stability_margin = (
            competitor_lower - float(winner.get("upper_bound_m", 0.0))
            if competitor_lower is not None
            else None
        )
    return {
        "anchor_object_id": int(anchor["object_id"]),
        "selected_object_id": int(winner["subject_object_id"]),
        "raw_distance_m": float(winner.get("horizontal_distance_m", 0.0)),
        "conservative_distance_m": (
            float(winner.get("lower_bound_m", 0.0))
            if predicate == "farthest"
            else float(winner.get("upper_bound_m", 0.0))
        ),
        "stability_margin_m": stability_margin,
        "winner_stable": bool(
            stability_margin is not None and stability_margin >= 0.0
        ),
        "candidate_count": len(records),
        "candidate_object_ids": [
            int(value["subject_object_id"]) for value in records
        ],
    }


def _candidate_objects(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(value)
        for value in snapshot.get("objects", ())
        if (
            isinstance(value, Mapping)
            and value.get("status", "confirmed") in {"tentative", "confirmed"}
            and str(value.get("cardinality_role", "")) not in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }
            and _valid_geometry(value)
        )
    ]


def attach_query_domain(
    snapshot: Mapping[str, Any],
    task_ir: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach a recomputed geometric winner to an open snapshot.

    The distance anchor comes from the relation itself (for example, the
    lamp in ``pillow farthest from the lamp``), not from the robot's start
    pose.  A start-pose relation would need to be represented explicitly in
    the task IR rather than silently substituted here.
    """
    result = copy.deepcopy(snapshot)
    objects = _candidate_objects(result)
    entities = {
        str(value.get("id")): value
        for value in task_ir.get("entities", ())
        if isinstance(value, Mapping)
    }
    selector_closures: dict[str, dict[str, Any]] = {}
    relations = [
        value for value in task_ir.get("relations", ())
        if isinstance(value, Mapping)
    ]

    for relation in relations:
        predicate = str(relation.get("predicate", "")).strip().lower()
        if predicate not in {"closest", "farthest", "furthest"}:
            continue
        if predicate == "furthest":
            predicate = "farthest"
        relation_id = str(relation.get("id", ""))
        subject_entity = entities.get(str(relation.get("subject_entity", "")), {})
        subject_candidates = [value for value in objects if _matches_entity(value, subject_entity)]
        anchor_entities = [entities.get(str(value), {}) for value in relation.get("object_entities", ())]
        anchors = [
            value for value in objects
            if any(_matches_entity(value, entity) for entity in anchor_entities)
        ]
        ambiguous_ids = {
            int(raw_id)
            for raw_group in result.get("identity_ambiguity_groups", ())
            if isinstance(raw_group, Mapping)
            for raw_id in raw_group.get("canonical_object_ids", ())
            if str(raw_id).lstrip("-").isdigit()
        }
        subject_identity = {
            int(value["object_id"]): relation_identity_blockers(
                value,
                identity_ambiguous=int(value["object_id"]) in ambiguous_ids,
            )
            for value in subject_candidates
        }
        anchor_identity = {
            int(value["object_id"]): relation_identity_blockers(
                value,
                identity_ambiguous=int(value["object_id"]) in ambiguous_ids,
            )
            for value in anchors
        }
        # Keep every high-recall anchor in the joint hypothesis domain. Track
        # quality and identity state are continuous support terms; they must
        # not silently turn one raw anchor into an irreversible preselection.
        selection_anchors = list(anchors)
        def semantic_support(value: Mapping[str, Any]) -> float:
            try:
                probability = max(
                    0.0,
                    min(1.0, float(value.get("semantic_probability", 0.0) or 0.0)),
                )
            except (TypeError, ValueError):
                probability = 0.0
            verified = 1.0 if str(value.get("semantic_status", "")).lower() == "verified" else 0.0
            return float(0.65 * probability + 0.35 * verified)

        def identity_support(value: Mapping[str, Any]) -> float:
            blockers = relation_identity_blockers(
                value,
                identity_ambiguous=int(value["object_id"]) in ambiguous_ids,
            )
            return float(
                max(0.0, min(1.0, _track_quality(value)))
                * math.exp(-0.22 * len(blockers))
            )

        selector_hypotheses: list[dict[str, Any]] = []
        for subject in subject_candidates:
            for anchor in selection_anchors:
                record = _distance_record(subject, anchor)
                target_semantic_support = semantic_support(subject)
                anchor_semantic_support = semantic_support(anchor)
                target_identity_support = identity_support(subject)
                anchor_identity_support = identity_support(anchor)
                pair_support = float(
                    0.30 * target_semantic_support
                    + 0.20 * anchor_semantic_support
                    + 0.25 * target_identity_support
                    + 0.25 * anchor_identity_support
                )
                uncertainty = float(record["uncertainty_m"])
                distance = float(record["horizontal_distance_m"])
                geometry_uncertainty = max(
                    0.0,
                    min(1.0, uncertainty / max(0.50, distance)),
                )
                selector_score = (
                    float(record["lower_bound_m"])
                    if predicate == "farthest"
                    else -float(record["upper_bound_m"])
                )
                selector_score += 0.20 * pair_support
                selector_score -= 0.15 * geometry_uncertainty
                selector_hypotheses.append({
                    **record,
                    "target_semantic_support": target_semantic_support,
                    "anchor_semantic_support": anchor_semantic_support,
                    "target_identity_support": target_identity_support,
                    "anchor_identity_support": anchor_identity_support,
                    "identity_support": float(
                        0.5 * (target_identity_support + anchor_identity_support)
                    ),
                    "pair_support": pair_support,
                    "geometry_uncertainty": geometry_uncertainty,
                    "identity_ambiguity": float(
                        int(
                            int(subject["object_id"]) in ambiguous_ids
                            or int(anchor["object_id"]) in ambiguous_ids
                        )
                    ),
                    "selector_score": float(selector_score),
                })
        ordered_hypotheses = sorted(
            selector_hypotheses,
            key=lambda value: (
                float(value.get("selector_score", float("-inf"))),
                float(value.get("pair_support", 0.0)),
                -int(value["subject_object_id"]),
                -int(value["anchor_object_id"]),
            ),
            reverse=True,
        )
        selected_hypothesis = ordered_hypotheses[0] if ordered_hypotheses else None
        score_margin = (
            float(ordered_hypotheses[0]["selector_score"])
            - float(ordered_hypotheses[1]["selector_score"])
            if len(ordered_hypotheses) > 1 else 0.50
        )
        if selected_hypothesis is not None and len(ordered_hypotheses) > 1:
            if predicate == "farthest":
                competitor_bound = max(
                    float(value.get("upper_bound_m", 0.0))
                    for value in ordered_hypotheses[1:]
                )
                selector_margin = float(
                    selected_hypothesis.get("lower_bound_m", 0.0)
                ) - competitor_bound
            else:
                competitor_bound = min(
                    float(value.get("lower_bound_m", 0.0))
                    for value in ordered_hypotheses[1:]
                )
                selector_margin = competitor_bound - float(
                    selected_hypothesis.get("upper_bound_m", 0.0)
                )
        else:
            selector_margin = 0.50
        if selected_hypothesis is not None:
            selected_hypothesis["stability_margin_m"] = selector_margin
            selected_hypothesis["winner_stable"] = bool(selector_margin >= 0.0)
            selected_hypothesis["selector_score_margin"] = score_margin
        distance_records = selector_hypotheses
        records_by_anchor = {
            int(anchor["object_id"]): [
                value for value in distance_records
                if int(value["anchor_object_id"]) == int(anchor["object_id"])
            ]
            for anchor in anchors
        }
        anchor_hypotheses = [
            hypothesis
            for anchor in selection_anchors
            for hypothesis in [
                _selector_hypothesis(
                    predicate,
                    anchor,
                    records_by_anchor.get(int(anchor["object_id"]), ()),
                )
            ]
            if hypothesis is not None
        ]
        selected_object_id = (
            int(selected_hypothesis["subject_object_id"])
            if selected_hypothesis is not None else None
        )
        selected_anchor_id = (
            int(selected_hypothesis["anchor_object_id"])
            if selected_hypothesis is not None else None
        )
        stable_across_anchors = bool(
            selected_hypothesis is not None
            and selected_hypothesis.get("winner_stable") is True
        )
        closure_reasons = ["persistent_scene_memory_open"]
        if selected_hypothesis is not None and float(selector_margin) < 0.0:
            closure_reasons.append("distance_uncertainty_overlap")
        unresolved_subject_ids = sorted(
            object_id for object_id, blockers in subject_identity.items() if blockers
        )
        unresolved_anchor_ids = sorted(
            object_id for object_id, blockers in anchor_identity.items() if blockers
        )
        if unresolved_subject_ids or unresolved_anchor_ids:
            closure_reasons.append("identity_hypothesis_open")
        if unresolved_subject_ids:
            closure_reasons.append("subject_identity_pending")
        if unresolved_anchor_ids:
            closure_reasons.append("anchor_identity_pending")
        selected_records = [
            value for value in distance_records
            if selected_anchor_id is not None
            and int(value["anchor_object_id"]) == selected_anchor_id
        ]
        selected_winner = selected_hypothesis
        selected_subject = next(
            (
                value for value in subject_candidates
                if selected_object_id is not None
                and int(value["object_id"]) == selected_object_id
            ),
            None,
        )
        selected_anchor_value = next(
            (
                value for value in anchors
                if selected_anchor_id is not None
                and int(value["object_id"]) == selected_anchor_id
            ),
            None,
        )
        identity_ready = bool(
            selected_subject is not None
            and selected_anchor_value is not None
            and not subject_identity.get(int(selected_subject["object_id"]), ["missing_subject"])
            and not anchor_identity.get(int(selected_anchor_value["object_id"]), ["missing_anchor"])
        )
        selector_stable = bool(
            selected_hypothesis is not None
            and selected_hypothesis.get("winner_stable") is True
            and identity_ready
        )
        if not identity_ready and selected_winner is not None:
            closure_reasons.append("selected_identity_not_confirmed")
        selected_pair_support = float(
            selected_hypothesis.get("pair_support", 0.0)
            if selected_hypothesis else 0.0
        )
        selected_identity_support = float(
            selected_hypothesis.get("identity_support", 0.0)
            if selected_hypothesis else 0.0
        )
        selector_margin_support = max(
            0.0,
            min(1.0, float(selector_margin) / 0.25),
        )
        geometry_uncertainty = float(
            selected_hypothesis.get("geometry_uncertainty", 1.0)
            if selected_hypothesis else 1.0
        )
        identity_ambiguity = float(
            selected_hypothesis.get("identity_ambiguity", 1.0)
            if selected_hypothesis else 1.0
        )
        selector_closures[relation_id] = {
            "relation_id": relation_id,
            "predicate": predicate,
            "selector_state": (
                "STABLE_GEOMETRIC_WINNER"
                if selector_stable
                else "PROVISIONAL_GEOMETRIC_WINNER"
                if selected_winner
                else "UNRESOLVED"
            ),
            "candidate_domain_state": "OPEN_PERSISTENT_TRACKS",
            "candidate_domain_closed": False,
            "closure_gate": False,
            "selected_object_id": selected_object_id,
            "selected_anchor_id": selected_anchor_id,
            "anchor_object_ids": [
                int(value["object_id"]) for value in anchors
            ],
            "selection_anchor_object_ids": [
                int(value["object_id"]) for value in selection_anchors
            ],
            "candidate_object_ids": [int(value["object_id"]) for value in subject_candidates],
            "canonical_candidate_object_ids": [
                int(value["object_id"])
                for value in subject_candidates
                if not subject_identity.get(int(value["object_id"]))
            ],
            "unresolved_candidate_object_ids": unresolved_subject_ids,
            "unresolved_anchor_object_ids": unresolved_anchor_ids,
            "identity_blockers": {
                "subjects": {
                    str(object_id): blockers
                    for object_id, blockers in subject_identity.items()
                    if blockers
                },
                "anchors": {
                    str(object_id): blockers
                    for object_id, blockers in anchor_identity.items()
                    if blockers
                },
            },
            "distance_records": distance_records,
            "candidate_distances_m": {
                str(value["subject_object_id"]): float(value["horizontal_distance_m"])
                for value in selected_records
            },
            "anchor_hypotheses": anchor_hypotheses,
            "selector_hypotheses": ordered_hypotheses,
            "anchor_sensitive": len(anchor_hypotheses) > 1,
            "stable_winner_across_anchor_hypotheses": stable_across_anchors,
            "current_winner_stable": bool(
                selected_hypothesis and selected_hypothesis.get("winner_stable")
            ),
            "identity_ready_for_selection": identity_ready,
            "current_winner_stability_margin_m": (
                selected_hypothesis.get("stability_margin_m")
                if selected_hypothesis else None
            ),
            "selector_evidence_complete": selector_stable,
            "selector_evidence_features": {
                "relation_support": selected_pair_support,
                "identity_support": selected_identity_support,
                "selector_margin_support": selector_margin_support,
                "geometry_uncertainty": geometry_uncertainty,
                "identity_ambiguity": identity_ambiguity,
            },
            "distance_metric": "horizontal_xy",
            "selection_mode": "recomputed_horizontal_distance",
            "selection_recomputed_from_scene_version": int(
                result.get("scene_version", 0)
            ),
            "closure_reasons": closure_reasons,
        }

    # Keep the three decisions separate.  A geometric selector is provisional
    # because SceneMemory may acquire another instance; that says nothing
    # about whether a numerical count has an independent answer-ready
    # observation.
    task_type = str(task_ir.get("task_type", "")).strip().lower()
    relation_selector_open = bool(selector_closures)
    result["scene_memory_open"] = True
    result["scene_memory_state"] = "OPEN_PERSISTENT_TRACKS"
    result["relation_selector_domain_open"] = relation_selector_open
    result["relation_selector_domain_state"] = (
        "OPEN_PERSISTENT_TRACKS"
        if relation_selector_open else "NOT_APPLICABLE"
    )
    count_ready = result.get("numerical_count_answer_ready") is True
    result["numerical_count_domain_open"] = bool(
        task_type == "numerical" and not count_ready
    )
    result["numerical_count_domain_state"] = (
        "OPEN_PENDING_COUNT_AUTHORITY"
        if task_type == "numerical" and not count_ready
        else "READY"
        if task_type == "numerical"
        else "NOT_APPLICABLE"
    )
    result["query_domain"] = {
        "schema_version": "query_domain_v5_open_geometric",
        "belief_mode": "open_persistent_tracks",
        "closure_gate": False,
        "scene_memory_open": True,
        "relation_selector_domain_open": relation_selector_open,
        "numerical_count_domain_open": result["numerical_count_domain_open"],
        "selector_closures": selector_closures,
    }
    result["candidate_domain"] = {
        "state": (
            "OPEN_PERSISTENT_TRACKS"
            if relation_selector_open else "TASK_SCOPED_CANDIDATES"
        ),
        "object_ids": [int(value["object_id"]) for value in objects],
        "unresolved_classes": [],
    }
    return result
