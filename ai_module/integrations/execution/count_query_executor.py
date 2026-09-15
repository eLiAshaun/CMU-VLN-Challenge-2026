"""Tri-state execution for COUNT_DISTINCT query graphs.

Candidate identity comes exclusively from the persistent world snapshot.  A
relation evaluator (the live chain supplies Qwen) is invoked for exact object
ID tuples; missing visual support is preserved as UNKNOWN rather than being
converted to a negative relation.
"""

from __future__ import annotations

import copy
import math
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from integrations.semantics.count_query_graph import validate_count_query_graph
from orchestration.observation_contract import INVALID, NO, UNKNOWN, YES


_STATES = {YES, NO, UNKNOWN, INVALID}

RelationEvaluator = Callable[
    [Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


def _canonical_binding_id(
    binding: Mapping[str, Any] | None,
    *,
    relation_id: str = "",
) -> str:
    values: list[str] = []
    for key, value in sorted((binding or {}).items(), key=lambda item: str(item[0])):
        try:
            object_id: object = int(value)
        except (TypeError, ValueError):
            object_id = str(value)
        values.append(f"{key}={object_id}")
    return f"{str(relation_id).strip()}|" + ",".join(values)


def _unbounded_distance(value: object) -> float:
    """Use an in-memory ordering value without leaking non-finite JSON.

    ``None`` means that no comparison point exists (for example, there has
    not yet been a failed navigation attempt).  It is maximally novel for
    selection, but must remain ``None`` in persisted evidence rather than
    becoming ``Infinity``.
    """
    distance = _finite_distance_or_none(value)
    return distance if distance is not None else math.inf


def _finite_distance_or_none(value: object) -> float | None:
    """Normalize a distance for persisted diagnostics without non-finite values."""
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None
    return distance if math.isfinite(distance) else None


def _optional_distance(
    point: tuple[float, float],
    references: Sequence[tuple[float, float]],
) -> float | None:
    """Return a finite clearance, or ``None`` when no reference exists."""
    if not references:
        return None
    distance = min(math.dist(point, reference) for reference in references)
    return float(distance) if math.isfinite(distance) else None


def _center_uncertainty_xy(value: Mapping[str, Any]) -> float:
    """Return one-sigma XY centre uncertainty for ranking intervals."""
    covariance = value.get("center_cov")
    if isinstance(covariance, Sequence) and len(covariance) >= 2:
        try:
            variance = max(0.0, float(covariance[0][0])) + max(
                0.0, float(covariance[1][1])
            )
            if math.isfinite(variance):
                return float(math.sqrt(variance))
        except (TypeError, ValueError, IndexError):
            pass
    return 0.25


@dataclass
class CountClosureState:
    """Episode-persistent convergence state for one CountQuery.

    This is evidence about the queried world region, not a confidence gate
    around an answer.  Acquisition uses it to decide what evidence is still
    useful; the Count Graph remains the only answer producer.
    """

    query_key: str
    confirmed_ids: list[int] = field(default_factory=list)
    refuted_ids: list[int] = field(default_factory=list)
    pending_identity_ids: list[int] = field(default_factory=list)
    unresolved_target_ids: list[int] = field(default_factory=list)
    unknown_relation_ids: list[str] = field(default_factory=list)
    invalid_relation_ids: list[str] = field(default_factory=list)
    identity_ambiguous_groups: list[dict[str, Any]] = field(default_factory=list)
    covered_viewpoints: list[str] = field(default_factory=list)
    covered_regions: list[str] = field(default_factory=list)
    recent_new_instance_counts: list[int] = field(default_factory=list)
    last_discovery_time: float | None = None
    new_instance_total: int = 0
    valid_view_count: int = 0
    independent_view_count: int = 0
    consecutive_no_new_views: int = 0
    coverage_threshold: int = 2
    coverage_score: float = 0.0
    closed: bool = False
    closure_reasons: list[str] = field(default_factory=list)
    next_acquisition: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "count_closure_state_v1",
            "query_key": self.query_key,
            "confirmed_ids": list(self.confirmed_ids),
            "refuted_ids": list(self.refuted_ids),
            "pending_identity_ids": list(self.pending_identity_ids),
            "unresolved_target_ids": list(self.unresolved_target_ids),
            "unknown_relation_ids": list(self.unknown_relation_ids),
            "invalid_relation_ids": list(self.invalid_relation_ids),
            "identity_ambiguous_groups": [
                dict(value) for value in self.identity_ambiguous_groups
            ],
            "covered_viewpoints": list(self.covered_viewpoints),
            "covered_regions": list(self.covered_regions),
            "recent_new_instance_counts": list(self.recent_new_instance_counts),
            "last_discovery_time": self.last_discovery_time,
            "new_instance_total": int(self.new_instance_total),
            "valid_view_count": int(self.valid_view_count),
            "independent_view_count": int(self.independent_view_count),
            "consecutive_no_new_views": int(self.consecutive_no_new_views),
            "coverage_threshold": int(self.coverage_threshold),
            "coverage_score": float(self.coverage_score),
            "closed": bool(self.closed),
            "closure_reasons": list(self.closure_reasons),
            "next_acquisition": dict(self.next_acquisition),
        }


def _object_domain(
    variable: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = {
        str(variable.get("class_name", "")).strip().lower(),
        *(
            str(value).strip().lower()
            for value in variable.get("aliases", ())
        ),
    }
    names.discard("")
    matches = [
        dict(obj)
        for obj in objects
        if obj.get("status") in {"tentative", "confirmed"}
        and str(obj.get("cardinality_role", "ATOMIC") or "ATOMIC") not in {
            "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
        }
        and str(obj.get("semantic_status", "")).strip().lower()
        != "rejected"
        and str(obj.get("class_label", "")).strip().lower() in names
    ]
    # Representation roles are owned by SceneMemory.  An aggregate or partial
    # proposal is evidence about existing entities, never another query
    # entity. UNKNOWN_CARDINALITY remains in the domain and therefore keeps
    # exact count unresolved until the owner has enough evidence.
    return sorted(matches, key=lambda value: int(value["object_id"]))


def _canonical_count_identity(value: Mapping[str, Any]) -> bool:
    """Return whether a physical track is also semantically confirmed.

    Discovery remains high-recall and keeps unverified candidates in the
    graph.  They become countable only after SceneMemory has both a canonical
    multi-view identity and positive semantic support; otherwise a detector
    label would be able to inflate the lower bound.
    """
    return bool(
        value.get("status") == "confirmed"
        and str(value.get("semantic_status", "")).strip().lower() == "verified"
    )


def _dependency_state(
    node: Mapping[str, Any],
    binding: Mapping[str, int],
    node_results: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    overall = YES
    for dependency_id in node.get("depends_on", ()):
        dependency = node_results[str(dependency_id)]
        compatible = []
        for result in dependency.get("tuple_results", ()):
            result_binding = result.get("binding", {})
            shared = set(binding).intersection(result_binding)
            if all(
                int(binding[key]) == int(result_binding[key]) for key in shared
            ):
                compatible.append(result)
        states = {
            str(value.get("state", UNKNOWN)).upper()
            for value in compatible
        }
        if INVALID in states:
            state = INVALID
        elif "CONFLICTED" in states:
            # Persistent conflict is unresolved, not a negative dependency
            # and never permission to inherit a YES branch.
            state = UNKNOWN
        elif YES in states and NO in states:
            # Alternative identity bindings that disagree cannot authorize a
            # dependent tuple.  Keep the dependency unresolved until the
            # binding itself is disambiguated.
            state = UNKNOWN
        elif YES in states:
            state = YES
        elif UNKNOWN in states or not compatible:
            state = UNKNOWN
        else:
            state = NO
        diagnostics.append({
            "relation_node_id": str(dependency_id),
            "state": state,
            "compatible_tuple_count": len(compatible),
        })
        if state == INVALID:
            overall = INVALID
        elif state == NO:
            overall = NO
        elif state == UNKNOWN and overall not in {NO, INVALID}:
            overall = UNKNOWN
    return overall, diagnostics


def _normalized_verdict(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    state = str(result.get("state", UNKNOWN)).upper()
    if state not in _STATES:
        state = UNKNOWN
        result["reason_code"] = "relation_evaluator_state_invalid"
    result["state"] = state
    result["evidence_ids"] = list(dict.fromkeys(
        str(item)
        for item in result.get("evidence_ids", ())
        if str(item).strip()
    ))
    result.setdefault("reason_code", "relation_verifier_no_reason")
    return result


class RelationEvidenceProvider:
    """Supply unfused live and persistent exact-ID tuple observations."""

    def __init__(
        self,
        live_evaluator: RelationEvaluator | None,
        persistent_relation_summary: Mapping[str, Any] | None = None,
        object_id_aliases: Mapping[str, Any] | None = None,
        *,
        identity_version: int | None = None,
        geometry_version: int | None = None,
    ) -> None:
        self.live_evaluator = live_evaluator
        self.persistent_relation_summary = (
            dict(persistent_relation_summary)
            if isinstance(persistent_relation_summary, Mapping)
            else {}
        )
        self.object_id_aliases = (
            dict(object_id_aliases)
            if isinstance(object_id_aliases, Mapping)
            else {}
        )
        self.identity_version = (
            int(identity_version) if identity_version is not None else None
        )
        self.geometry_version = (
            int(geometry_version) if geometry_version is not None else None
        )
        self.persistent_evidence_use_count = 0
        self.live_verification_count = 0
        self.stale_persistent_record_count = 0

    @staticmethod
    def _unknown(reason: str) -> dict[str, Any]:
        return {
            "state": UNKNOWN,
            "reason_code": str(reason),
            "evidence_ids": [],
        }

    def _canonical_object_id(self, value: object) -> int | None:
        try:
            current = int(value)
        except (TypeError, ValueError):
            return None
        visited: set[int] = set()
        while current not in visited:
            visited.add(current)
            next_value = self.object_id_aliases.get(
                str(current), self.object_id_aliases.get(current)
            )
            if next_value is None:
                break
            try:
                current = int(next_value)
            except (TypeError, ValueError):
                break
        return current

    def _persistent_verdict(
        self,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        relations = self.persistent_relation_summary.get("relations", {})
        if not isinstance(relations, Mapping):
            return None
        relation = relations.get(str(node.get("id", "")))
        if not isinstance(relation, Mapping):
            return None
        try:
            subject_id = self._canonical_object_id(subject["object_id"])
            object_ids = tuple(sorted(
                self._canonical_object_id(value["object_id"])
                for value in objects
            ))
        except (KeyError, TypeError, ValueError):
            return None
        if subject_id is None or any(value is None for value in object_ids):
            return None
        records = relation.get("candidate_records", ())
        if not isinstance(records, Sequence) or isinstance(
            records, (str, bytes, bytearray)
        ):
            return None
        matched: list[Mapping[str, Any]] = []
        for raw in records:
            if not isinstance(raw, Mapping):
                continue
            try:
                raw_subject = self._canonical_object_id(
                    raw.get("subject_object_id")
                )
                raw_objects = tuple(sorted(
                    self._canonical_object_id(value)
                    for value in raw.get("anchor_object_ids", ())
                ))
            except (TypeError, ValueError):
                continue
            if raw_subject is None or any(value is None for value in raw_objects):
                continue
            if raw_subject == subject_id and raw_objects == object_ids:
                matched.append(raw)
        if not matched:
            return None

        # A relation answer is attached to a particular SceneMemory geometry
        # and identity revision.  Once a new observation has updated either
        # revision, an old YES/NO is historical evidence only; it cannot be
        # silently reused as the current tuple state.  The live verifier gets
        # the first opportunity to recompute the tuple at the current station.
        current_versioned: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
        if self.identity_version is not None or self.geometry_version is not None:
            for raw in matched:
                fresh: list[Mapping[str, Any]] = []
                raw_evidence = raw.get("evidence_records", ())
                if isinstance(raw_evidence, Sequence) and not isinstance(
                    raw_evidence, (str, bytes, bytearray)
                ):
                    for value in raw_evidence:
                        if not isinstance(value, Mapping):
                            continue
                        try:
                            raw_identity = int(value.get("identity_version"))
                            raw_geometry = int(value.get("geometry_version"))
                        except (TypeError, ValueError):
                            continue
                        if (
                            self.identity_version is not None
                            and raw_identity != self.identity_version
                        ):
                            continue
                        if (
                            self.geometry_version is not None
                            and raw_geometry != self.geometry_version
                        ):
                            continue
                        fresh.append(value)
                if fresh:
                    current_versioned.append((raw, fresh))
                else:
                    self.stale_persistent_record_count += 1
            if not current_versioned:
                return None
        else:
            current_versioned = [
                (raw, [value for value in raw.get("evidence_records", ()) if isinstance(value, Mapping)])
                for raw in matched
            ]

        matched = [raw for raw, _fresh in current_versioned]
        versioned_evidence = [
            value for _raw, fresh in current_versioned for value in fresh
        ]
        states = {
            str(value.get("state", UNKNOWN)).upper()
            for value in versioned_evidence
        }
        if not states:
            states = {
                str(raw.get("state", UNKNOWN)).upper() for raw in matched
            }
        # The provider exposes categorical observations but never fuses them
        # into a verdict. RelationEngine owns that transition.
        state = UNKNOWN
        state_reason = "persistent_relation_observations_available"
        evidence_ids = list(dict.fromkeys(
            str(item)
            for value in versioned_evidence
            for item in value.get("source_observation_ids", ())
            if str(item).strip()
        ))
        if not evidence_ids:
            evidence_ids = list(dict.fromkeys(
                str(value)
                for raw in matched
                for value in raw.get("observation_ids", ())
                if str(value).strip()
            ))
        latest_evidence = versioned_evidence[-1] if versioned_evidence else {}
        source_observation_ids = list(dict.fromkeys(
            str(value)
            for value in latest_evidence.get("source_observation_ids", ())
            if str(value).strip()
        ))
        return {
            "state": state,
            "reason_code": state_reason,
            "persistent_observation_states": sorted(states),
            "evidence_ids": evidence_ids,
            "station_id": str(latest_evidence.get("station_id", "")).strip(),
            "timestamp": latest_evidence.get("timestamp"),
            "source_observation_ids": source_observation_ids,
            "identity_version": latest_evidence.get("identity_version"),
            "geometry_version": latest_evidence.get("geometry_version"),
            "persistent_state_inherited": True,
            "qwen": {
                "persistent_posterior": max(
                    float(raw.get("posterior_probability", 0.5) or 0.5)
                    for raw in matched
                ),
                "persistent_station_ids": sorted({
                    str(station)
                    for raw in matched
                    for station in raw.get("station_ids", ())
                    if str(station).strip()
                }),
                "persistent_evidence_versions": [
                    {
                        "identity_version": value.get("identity_version"),
                        "geometry_version": value.get("geometry_version"),
                    }
                    for value in versioned_evidence
                ],
            },
        }

    @staticmethod
    def _decisive(value: Mapping[str, Any] | None) -> bool:
        return str((value or {}).get("state", UNKNOWN)).upper() in {YES, NO}

    @staticmethod
    def _decorate_live_result(
        result: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Retain the exact observations that support one live tuple.

        The relation verifier normally adds this provenance itself.  The
        graph adapter also adds provenance because batch adapters and world
        geometry implementations may return a valid state without copying
        the participant evidence fields.  This is only applied to live
        results; inherited persistent states must not acquire fresh-looking
        observation provenance.
        """
        decorated = dict(result)
        source_ids = [
            str(value).strip()
            for value in decorated.get("source_observation_ids", ())
            if str(value).strip()
        ]
        for participant in (subject, *objects):
            if not isinstance(participant, Mapping):
                continue
            for evidence in participant.get("evidence", ()):
                if not isinstance(evidence, Mapping):
                    continue
                observation_id = str(
                    evidence.get("observation_id", "")
                ).strip()
                if observation_id:
                    source_ids.append(observation_id)
        if source_ids:
            decorated["source_observation_ids"] = list(dict.fromkeys(source_ids))
        return decorated

    def __call__(
        self,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        live_result: dict[str, Any] | None = None
        if callable(self.live_evaluator):
            self.live_verification_count += 1
            try:
                raw = self.live_evaluator(node, subject, objects)
                if isinstance(raw, Mapping):
                    live_result = self._decorate_live_result(
                        raw, subject, objects
                    )
            except Exception as exc:
                live_result = self._unknown(
                    f"live_relation_evaluator_exception:{type(exc).__name__}"
                )
        if live_result is not None and self._decisive(live_result):
            return live_result
        if live_result is not None and str(live_result.get("state", UNKNOWN)).upper() == INVALID:
            return live_result
        persistent = self._persistent_verdict(node, subject, objects)
        if live_result is not None:
            if persistent is not None:
                live_result = dict(live_result)
                live_result["persistent_evidence"] = persistent
                self.persistent_evidence_use_count += 1
            return live_result
        return persistent or self._unknown("relation_evidence_unavailable")

    def verify_many(
        self,
        items: Sequence[tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Sequence[Mapping[str, Any]],
        ]],
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any] | None] = [None] * len(items)
        live_many = getattr(self.live_evaluator, "verify_many", None)
        if callable(live_many) and items:
            self.live_verification_count += len(items)
            try:
                raw_results = list(live_many(items))
            except Exception:
                raw_results = []
            for index, raw in enumerate(raw_results[:len(items)]):
                if isinstance(raw, Mapping):
                    node, subject, objects = items[index]
                    outputs[index] = self._decorate_live_result(
                        raw, subject, objects
                    )
        elif callable(self.live_evaluator):
            for index, (node, subject, objects) in enumerate(items):
                outputs[index] = self(
                    node, subject, objects
                )
        for index, (node, subject, objects) in enumerate(items):
            current = outputs[index]
            if current is not None and str(current.get("state", UNKNOWN)).upper() == INVALID:
                continue
            if current is not None and self._decisive(current):
                continue
            persistent = self._persistent_verdict(node, subject, objects)
            if persistent is not None and current is not None:
                merged = dict(current)
                merged["persistent_evidence"] = persistent
                outputs[index] = merged
                self.persistent_evidence_use_count += 1
            elif current is None:
                outputs[index] = persistent or self._unknown(
                    "relation_evidence_unavailable"
                )
        return [dict(value or self._unknown("relation_evidence_unavailable")) for value in outputs]


def count_query_key(graph: Mapping[str, Any]) -> str:
    """Return a readable stable key for one compiled CountQuery."""
    variables = ";".join(
        f"{value.get('entity_id','')}:{value.get('class_name','')}"
        for value in graph.get("variables", ())
        if isinstance(value, Mapping)
    )
    relations = ";".join(
        f"{value.get('id','')}:{value.get('predicate','')}:{value.get('subject_entity','')}:"
        f"{','.join(str(item) for item in value.get('object_entities', ())) }"
        for value in graph.get("relation_nodes", ())
        if isinstance(value, Mapping)
    )
    task_id = str(graph.get("task_id", "")).strip() or "count-query"
    return f"{task_id}|target={graph.get('target_entity','')}|vars={variables}|relations={relations}"


def _object_lookup(snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in snapshot.get("objects", ()):
        if not isinstance(raw, Mapping):
            continue
        try:
            result[int(raw["object_id"])] = dict(raw)
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _history_position(value: Mapping[str, Any]) -> tuple[float, float, float] | None:
    raw = value.get("viewpoint_position_map")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
        return None
    try:
        result = tuple(float(item) for item in raw[:3])
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _history_region(value: Mapping[str, Any]) -> str:
    explicit = str(value.get("coverage_region", "")).strip()
    if explicit:
        return explicit
    position = _history_position(value)
    if position is None:
        return ""
    return f"xy:{math.floor(position[0]):d}:{math.floor(position[1]):d}"


def _identity_groups(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = [
        dict(value)
        for value in snapshot.get("identity_ambiguity_groups", ())
        if isinstance(value, Mapping)
    ]
    if groups:
        return groups
    reconstructed: list[dict[str, Any]] = []
    for index, raw in enumerate(snapshot.get("ambiguous_observations", ())):
        if not isinstance(raw, Mapping):
            continue
        try:
            ids = sorted({int(value) for value in raw.get("candidate_object_ids", ())})
        except (TypeError, ValueError):
            continue
        if len(ids) > 1:
            reconstructed.append({
                "group_id": f"identity-observation:{index}",
                "canonical_object_ids": ids,
            })
    return reconstructed


def _navigation_failure_geometry(
    snapshot: Mapping[str, Any],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return failed physical targets and the real poses where they failed.

    A navigation failure is evidence about the physical observability of a
    proposed query viewpoint.  It is not evidence that the query is solved,
    and it is not an attempt counter.  Keeping the coordinates in the query
    planner lets the next acquisition choose a different baseline when the
    official waypoint converter projected the previous request onto the same
    local patch.
    """
    failed_targets: list[tuple[float, float]] = []
    failed_poses: list[tuple[float, float]] = []
    for raw in snapshot.get("navigation_history", ()):
        if not isinstance(raw, Mapping) or str(raw.get("status", "")) != "failed":
            continue
        requested = raw.get("requested_waypoint_pose")
        if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
            context = raw.get("waypoint_context")
            if isinstance(context, Mapping):
                requested = context.get("requested_waypoint_pose")
        if isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)):
            try:
                point = (float(requested[0]), float(requested[1]))
            except (IndexError, TypeError, ValueError):
                point = None
            if point is not None and all(math.isfinite(value) for value in point):
                failed_targets.append(point)
        physical = raw.get("physical_waypoint_pose")
        if not isinstance(physical, Sequence) or isinstance(physical, (str, bytes)):
            context = raw.get("waypoint_context")
            if isinstance(context, Mapping):
                physical = context.get("physical_waypoint_pose")
        if not isinstance(physical, Sequence) or isinstance(physical, (str, bytes)):
            physical = raw.get("target_pose")
        if isinstance(physical, Sequence) and not isinstance(physical, (str, bytes)):
            try:
                point = (float(physical[0]), float(physical[1]))
            except (IndexError, TypeError, ValueError):
                point = None
            if point is not None and all(math.isfinite(value) for value in point):
                failed_targets.append(point)
        actual = raw.get("actual_arrival_pose")
        if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
            try:
                point = (float(actual[0]), float(actual[1]))
            except (IndexError, TypeError, ValueError):
                point = None
            if point is not None and all(math.isfinite(value) for value in point):
                failed_poses.append(point)
    return failed_targets, failed_poses


def _identity_viewpoint_candidates(
    object_ids: Sequence[int],
    objects: Mapping[int, Mapping[str, Any]],
    previous_viewpoints: Sequence[tuple[float, float]],
    failed_targets: Sequence[tuple[float, float]],
    failed_poses: Sequence[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Generate query-conditioned baselines for an identity ambiguity group.

    The group members are the only semantic target of this acquisition.  We
    sample viewpoints around their world-space support and rank them by new
    baseline and distance from failed converter projections.  This keeps the
    acquisition about resolving the identity pair rather than re-scanning the
    room or replaying a failed coordinate.
    """
    centers = [
        objects[object_id].get("center_3d")
        for object_id in object_ids
        if object_id in objects
        and isinstance(objects[object_id].get("center_3d"), Sequence)
        and len(objects[object_id]["center_3d"]) >= 2
    ]
    if not centers:
        return []
    center = (
        sum(float(value[0]) for value in centers) / len(centers),
        sum(float(value[1]) for value in centers) / len(centers),
    )
    baseline = max(
        0.60,
        max(
            (
                math.dist(
                    (float(first[0]), float(first[1])),
                    (float(second[0]), float(second[1])),
                )
                for first, second in itertools.combinations(centers, 2)
            ),
            default=0.0,
        ),
    )
    radius = max(0.90, min(1.35, 0.80 + 0.35 * baseline))
    raw_candidates = {
        (round(center[0] + radius * math.cos(angle), 5),
         round(center[1] + radius * math.sin(angle), 5))
        for angle in [index * math.pi / 4.0 for index in range(8)]
    }
    candidates: list[dict[str, Any]] = []
    for point in sorted(raw_candidates):
        previous_clearance = _optional_distance(point, previous_viewpoints)
        failed_clearance = _optional_distance(point, failed_targets)
        current_clearance = _optional_distance(point, failed_poses)
        candidates.append({
            "target_object_ids": [int(value) for value in object_ids],
            "navigation_target_xy": [float(point[0]), float(point[1])],
            "identity_baseline_m": float(baseline),
            "novel_baseline_m": previous_clearance,
            "failed_target_clearance_m": failed_clearance,
            "failed_pose_clearance_m": current_clearance,
            "already_observed_group": bool(
                previous_clearance is not None and previous_clearance < 0.30
            ),
        })
    return candidates


def _decorate_targeted_acquisition(
    graph: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach probe lineage metadata without changing CountGraph decisions."""
    result = copy.deepcopy(dict(acquisition))
    result.setdefault("query_key", count_query_key(graph))
    relation_id = str(
        result.get("relation_id")
        or next(iter(result.get("relation_ids", ())), "")
    )
    binding = result.get("unknown_binding", {})
    result["binding_id"] = str(
        result.get("binding_id")
        or _canonical_binding_id(
            binding if isinstance(binding, Mapping) else None,
            relation_id=relation_id,
        )
    )
    objects = _object_lookup(snapshot)
    anchor_ids = [
        int(value) for value in result.get("anchor_object_ids", ())
        if str(value).lstrip("-").isdigit()
    ]
    anchor_domain_ids: list[int] = []
    for entity_id, variable in {
        str(value.get("entity_id", "")): value
        for value in graph.get("variables", ())
        if isinstance(value, Mapping)
    }.items():
        if entity_id == str(graph.get("target_entity", "")):
            continue
        names = {
            str(variable.get("class_name", "")).strip().lower(),
            *(
                str(value).strip().lower()
                for value in variable.get("aliases", ())
            ),
        }
        anchor_domain_ids.extend(
            object_id for object_id, value in objects.items()
            if str(value.get("class_label", "")).strip().lower() in names
        )
    anchor_domain_ids = list(dict.fromkeys(anchor_domain_ids))
    anchor_quality: dict[str, Any] = {}
    low_quality_ids: list[int] = []
    for object_id in anchor_ids:
        value = objects.get(object_id, {})
        low_quality = bool(
            value.get("status") != "confirmed"
            or str(value.get("semantic_status", "")) != "verified"
            or not isinstance(value.get("center_3d"), Sequence)
            or any(
                isinstance(evidence, Mapping)
                and evidence.get("bearing_only") is True
                for evidence in value.get("evidence", ())
            )
        )
        if low_quality:
            low_quality_ids.append(object_id)
    anchor_quality.update({
        "anchor_domain_object_ids": anchor_domain_ids,
        "low_quality_object_ids": low_quality_ids,
        "localization_required": bool(low_quality_ids),
        "relation_probe_eligible": not bool(low_quality_ids),
    })
    result["anchor_domain_object_ids"] = anchor_domain_ids
    result["anchor_quality"] = anchor_quality
    result["anchor_localization_required"] = bool(low_quality_ids)
    result["probe_eligible"] = not bool(low_quality_ids)
    result["probe_type"] = (
        "ANCHOR_LOCALIZATION"
        if bool(low_quality_ids)
        else "RELATION_PROBE"
    )
    result["relation_id"] = relation_id
    return result


def build_count_query_targeted_acquisition(
    graph: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn UNKNOWN tuples into a relation-specific observation request."""
    validate_count_query_graph(graph)
    variables = {
        str(value["entity_id"]): dict(value)
        for value in graph.get("variables", ())
        if isinstance(value, Mapping)
    }
    target_entity = str(graph.get("target_entity", ""))
    objects = _object_lookup(snapshot)
    identity_groups = _identity_groups(snapshot)
    domains = {
        entity_id: _object_domain(variable, list(objects.values()))
        for entity_id, variable in variables.items()
    }
    resolved_singular_domains = {}
    if isinstance(execution, Mapping):
        raw_resolved = execution.get("resolved_singular_domains", {})
        if isinstance(raw_resolved, Mapping):
            for entity_id, values in raw_resolved.items():
                if not isinstance(values, Sequence) or isinstance(
                    values, (str, bytes, bytearray)
                ):
                    continue
                try:
                    resolved_singular_domains[str(entity_id)] = sorted({
                        int(value) for value in values
                    })
                except (TypeError, ValueError):
                    continue
    singular_domain_ids = {
        int(object_id)
        for entity_id, variable in variables.items()
        if str(variable.get("quantifier", "")) == "SINGULAR"
        for object_id in resolved_singular_domains.get(
            entity_id,
            [int(value["object_id"]) for value in domains[entity_id]],
        )
    }
    ambiguous_ids = {
        int(object_id)
        for group in identity_groups
        for object_id in group.get("canonical_object_ids", ())
        if str(object_id).lstrip("-").isdigit()
    }
    unknown_target_ids = {
        int(value)
        for value in (
            *list((execution or {}).get("unknown_target_ids", ())),
            *list((execution or {}).get("pending_identity_ids", ())),
        )
        if str(value).lstrip("-").isdigit()
    }
    relation_tuples = [
        (str(node.get("relation_node_id", "")), node, tuple_result)
        for node in (execution or {}).get("node_results", ())
        if isinstance(node, Mapping)
        for tuple_result in node.get("tuple_results", ())
        if isinstance(tuple_result, Mapping)
    ]
    unknown_tuples = [
        value for value in relation_tuples
        if str(value[2].get("state", UNKNOWN)).upper() == UNKNOWN
    ]
    failed_navigation_targets, failed_navigation_poses = _navigation_failure_geometry(
        snapshot
    )
    previous_viewpoints = [
        tuple(float(value) for value in position[:2])
        for record in snapshot.get("viewpoint_history", ())
        if isinstance(record, Mapping)
        for position in (record.get("viewpoint_position_map"),)
        if isinstance(position, Sequence)
        and not isinstance(position, (str, bytes))
        and len(position) >= 2
        and all(math.isfinite(float(value)) for value in position[:2])
    ]
    previous_viewpoints.extend(failed_navigation_poses)

    # Association ambiguity is itself an acquisition target.  If a later
    # observation may decide whether two canonical IDs are one object, send
    # the robot to a view that contains that group (and the relation tuple's
    # other participants when available).  Otherwise the same UNKNOWN tuple
    # would be requested forever while the ambiguous pair is never tested.
    for group in sorted(
        identity_groups,
        key=lambda value: 0 if any(
            str(object_id).lstrip("-").isdigit()
            and int(object_id) in singular_domain_ids
            for object_id in value.get("canonical_object_ids", ())
        ) else 1,
    ):
        if not isinstance(group, Mapping):
            continue
        group_ids = sorted({
            int(value) for value in group.get("canonical_object_ids", ())
            if str(value).lstrip("-").isdigit() and int(value) in objects
        })
        if len(group_ids) < 2:
            continue
        group_entities = [
            entity_id for entity_id, domain in domains.items()
            if any(int(value["object_id"]) in group_ids for value in domain)
        ]
        singular_group = any(
            str(variables.get(entity_id, {}).get("quantifier", "")) == "SINGULAR"
            for entity_id in group_entities
        )
        # A SINGULAR anchor ambiguity is a prerequisite for every dependent
        # relation tuple. Resolve/merge that group with one joint view before
        # letting the UNKNOWN-tuple branch enumerate more target hypotheses.
        # Non-anchor identity groups still yield to the exact tuple branch.
        if unknown_tuples and not singular_group:
            continue
        group_is_target = target_entity in group_entities
        if not singular_group and not unknown_target_ids.intersection(group_ids):
            continue
        context = next(
            (
                (relation_id, node, tuple_result)
                for relation_id, node, tuple_result in unknown_tuples
                if any(
                    int(value) in group_ids
                    for value in tuple_result.get("binding", {}).values()
                    if str(value).lstrip("-").isdigit()
                )
            ),
            None,
        )
        binding = (
            dict(context[2].get("binding", {}))
            if context is not None
            else {}
        )
        relation_id = context[0] if context is not None else ""
        relation_node = context[1] if context is not None else {}
        if not relation_node:
            relation_candidate = next(
                (
                    value for value in graph.get("relation_nodes", ())
                    if isinstance(value, Mapping)
                ),
                {},
            )
            relation_node = dict(relation_candidate)
            relation_id = str(relation_node.get("relation_node_id", relation_node.get("id", "")))
        target_id = binding.get(target_entity)
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            target_id = None
        # Both identity-group branches feed the common pair-selection and
        # return path below.  Keep their shared variables explicit so a
        # singular ambiguity group cannot leave the anchor metadata
        # unbound after it has already selected its canonical IDs.
        anchor_entity_ids: list[str] = []
        anchor_domain_ambiguous = False
        if singular_group:
            anchor_ids = [] if group_is_target else group_ids[:]
            visible_ids = group_ids[:]
            if target_id is not None and target_id not in visible_ids:
                visible_ids.insert(0, target_id)
            target_entity_ids = list(dict.fromkeys(
                [target_entity]
                + [entity_id for entity_id in group_entities]
            ) )
        else:
            anchor_domain_ambiguous = False
            anchor_ids = [
                int(value)
                for value in binding.values()
                if str(value).lstrip("-").isdigit()
                and int(value) not in group_ids
            ]
            anchor_entity_ids: list[str] = []
            if anchor_ids:
                anchor_entity_ids = [
                    entity_id
                    for entity_id, domain in domains.items()
                    if entity_id != target_entity
                    and any(
                        int(candidate["object_id"]) in anchor_ids
                        for candidate in domain
                    )
                ]
            # An ambiguity observation can be identity-relevant even when
            # the relation graph did not retain an UNKNOWN tuple for that
            # particular provisional target.  Do not degrade that request to
            # target-only observation: recover the current singular anchor
            # domain from the graph and keep the acquisition relation-aware.
            if not anchor_ids:
                for entity_id, variable in variables.items():
                    if entity_id == target_entity:
                        continue
                    if str(variable.get("quantifier", "")) != "SINGULAR":
                        continue
                    candidate_domain = domains.get(entity_id, ())
                    # SINGULAR is a language cardinality, not permission to
                    # discard tentative identity hypotheses.  Keep the full
                    # anchor domain here too; otherwise targeted acquisition
                    # could silently undo the complete product evaluated by
                    # the Count Graph whenever one confirmed anchor exists.
                    anchor_ids = [
                        int(candidate["object_id"])
                        for candidate in candidate_domain
                    ]
                    anchor_domain_ambiguous = len(anchor_ids) > 1
                    anchor_entity_ids = [entity_id] if anchor_ids else []
                    if anchor_ids:
                        break
            else:
                # The UNKNOWN tuple records one current binding, but a
                # singular anchor domain may still contain other canonical
                # candidates.  Keep the binding as a preference and expose
                # the full anchor domain to the pair selector so an
                # unobserved target/anchor hypothesis can be acquired next.
                for entity_id in anchor_entity_ids:
                    variable = variables.get(entity_id, {})
                    if str(variable.get("quantifier", "")) != "SINGULAR":
                        continue
                    candidate_ids = resolved_singular_domains.get(
                        entity_id,
                        [
                            int(candidate["object_id"])
                            for candidate in domains.get(entity_id, ())
                        ],
                    )
                    if len(candidate_ids) > 1:
                        anchor_domain_ambiguous = True
                        anchor_ids = [
                            int(object_id)
                            for object_id in candidate_ids
                            if int(object_id) in objects
                        ]
                    break
        # A singular anchor domain can contain several physical candidates.
        # Averaging every candidate produces a point that may be observable
        # from none of the tuple hypotheses.  Acquire one target/anchor
        # hypothesis at a time and choose the pair whose joint midpoint gives
        # the most useful new baseline.  The next call can select another
        # pair after this view has been recorded; no attempt counter is used
        # as a closure condition.
        viewpoint_candidates: list[dict[str, Any]] = []
        selected_anchor_id: int | None = None
        selected_navigation_target_xy: list[float] | None = None
        if not singular_group and anchor_ids:
            target_centers = [
                objects[object_id].get("center_3d")
                for object_id in group_ids
                if object_id in objects
                and isinstance(objects[object_id].get("center_3d"), Sequence)
            ]
            target_center = (
                [
                    sum(float(center[axis]) for center in target_centers)
                    / len(target_centers)
                    for axis in range(2)
                ]
                if target_centers
                else None
            )
            def observed_ids(record: Mapping[str, Any]) -> set[int]:
                result: set[int] = set()
                by_class = record.get("object_ids_by_class", {})
                if isinstance(by_class, Mapping):
                    for values in by_class.values():
                        if isinstance(values, Sequence) and not isinstance(
                            values, (str, bytes)
                        ):
                            result.update(
                                int(value)
                                for value in values
                                if str(value).lstrip("-").isdigit()
                            )
                result.update(
                    int(value)
                    for value in record.get("new_instance_ids", ())
                    if str(value).lstrip("-").isdigit()
                )
                return result

            history_pairs = [
                observed_ids(record)
                for record in snapshot.get("viewpoint_history", ())
                if isinstance(record, Mapping)
            ]
            bound_anchor_ids = {
                int(value)
                for value in binding.values()
                if str(value).lstrip("-").isdigit()
                and int(value) in anchor_ids
            }

            def anchor_needs_localization(object_id: int) -> bool:
                value = objects.get(object_id, {})
                return bool(
                    value.get("status") != "confirmed"
                    or str(value.get("semantic_status", "")) != "verified"
                    or not isinstance(value.get("center_3d"), Sequence)
                    or any(
                        isinstance(evidence, Mapping)
                        and evidence.get("bearing_only") is True
                        for evidence in value.get("evidence", ())
                    )
                )

            for anchor_id in anchor_ids:
                anchor_center = objects.get(anchor_id, {}).get("center_3d")
                if (
                    target_center is None
                    or not isinstance(anchor_center, Sequence)
                    or len(anchor_center) < 2
                ):
                    continue
                pair_center = [
                    0.5 * (target_center[axis] + float(anchor_center[axis]))
                    for axis in range(2)
                ]
                covered = any(
                    bool(set(group_ids).intersection(ids)) and anchor_id in ids
                    for ids in history_pairs
                )
                novelty = _optional_distance(pair_center, previous_viewpoints)
                viewpoint_candidates.append({
                    "anchor_object_id": int(anchor_id),
                    "target_object_ids": list(group_ids),
                    "navigation_target_xy": pair_center,
                    "already_observed_pair": bool(covered),
                    "novel_baseline_m": novelty,
                })
            if viewpoint_candidates:
                selected = max(
                    viewpoint_candidates,
                    key=lambda value: (
                        not anchor_needs_localization(
                            int(value["anchor_object_id"])
                        ),
                        not bool(value["already_observed_pair"]),
                        int(value["anchor_object_id"]) in bound_anchor_ids,
                        _unbounded_distance(value.get("novel_baseline_m")),
                        -int(value["anchor_object_id"]),
                    ),
                )
                selected_anchor_id = int(selected["anchor_object_id"])
                selected_navigation_target_xy = [
                    float(value) for value in selected["navigation_target_xy"]
                ]
                anchor_ids = [selected_anchor_id]

        if singular_group:
            # A singular identity ambiguity has no relation tuple with two
            # different semantic entities to supply a midpoint.  Resolve it
            # from the world-space separation of the ambiguous canonical
            # tracks and acquire a genuinely new baseline around that group.
            viewpoint_candidates = _identity_viewpoint_candidates(
                group_ids,
                objects,
                previous_viewpoints,
                failed_navigation_targets,
                failed_navigation_poses,
            )
            if viewpoint_candidates:
                selected = max(
                    viewpoint_candidates,
                    key=lambda value: (
                        _unbounded_distance(value.get("failed_target_clearance_m")),
                        _unbounded_distance(value.get("failed_pose_clearance_m")),
                        not bool(value["already_observed_group"]),
                        _unbounded_distance(value.get("novel_baseline_m")),
                        -float(value["navigation_target_xy"][0]),
                        -float(value["navigation_target_xy"][1]),
                    ),
                )
                selected_navigation_target_xy = [
                    float(value)
                    for value in selected["navigation_target_xy"]
                ]

        non_group_binding_ids = [
            int(value)
            for value in binding.values()
            if str(value).lstrip("-").isdigit()
            and int(value) in objects
            and int(value) not in group_ids
        ]
        target_domain_ids = {
            int(candidate["object_id"])
            for candidate in domains.get(target_entity, ())
        }
        target_focus_ids: list[int] = []
        if singular_group and not group_is_target:
            # Anchor-first identity disambiguation must not require one
            # arbitrary target to be visible at the same station.  The group
            # itself is the evidence target; the original target/anchor
            # binding remains attached below for the subsequent relation
            # recomputation after SceneMemory can merge or split the group.
            visible_ids = list(group_ids)
            target_entity_ids = list(dict.fromkeys(
                [target_entity, *group_entities, *anchor_entity_ids]
            ))
        elif singular_group:
            anchor_ids = list(dict.fromkeys(
                [*anchor_ids, *non_group_binding_ids]
            ))
            visible_ids = list(dict.fromkeys(group_ids + anchor_ids))
            target_entity_ids = list(dict.fromkeys(
                [*group_entities, *anchor_entity_ids]
            ))
        else:
            visible_ids = list(dict.fromkeys(group_ids + anchor_ids))
            target_entity_ids = list(dict.fromkeys(
                [*(group_entities or [target_entity]), *anchor_entity_ids]
            ))
        if group_ids:
            binding = dict(binding)
            if target_focus_ids:
                binding[target_entity] = target_focus_ids[0]
            elif group_is_target:
                binding.setdefault(target_entity, group_ids[0])
            for entity_id in group_entities:
                binding.setdefault(entity_id, group_ids[0])
            for entity_id in anchor_entity_ids:
                if anchor_ids:
                    binding[entity_id] = anchor_ids[0]
        for candidate in viewpoint_candidates:
            candidate["required_visible_object_ids"] = list(visible_ids)
        centers = [
            objects[object_id].get("center_3d")
            for object_id in visible_ids
            if object_id in objects
            and isinstance(objects[object_id].get("center_3d"), Sequence)
        ]
        center_xy = selected_navigation_target_xy or (
            [
                sum(float(center[axis]) for center in centers) / len(centers)
                for axis in range(2)
            ]
            if centers else None
        )
        return _decorate_targeted_acquisition(graph, snapshot, {
            "schema_version": "count_query_targeted_acquisition_v1",
            "mode": "identity_targeted",
            "reason": "identity_association_ambiguity",
            "relation_ids": [relation_id] if relation_id else [
                str(value.get("id", ""))
                for value in graph.get("relation_nodes", ())
                if isinstance(value, Mapping)
            ],
            "target_entity_ids": target_entity_ids,
            "target_object_ids": visible_ids,
            "required_visible_object_ids": visible_ids,
            "anchor_object_ids": anchor_ids,
            "anchor_first": bool(
                singular_group or (not singular_group and anchor_domain_ambiguous)
            ),
            "joint_visibility_required": bool(relation_node or graph.get("relation_nodes")),
            "navigation_target_xy": center_xy,
            "requires_new_station": True,
            "identity_ambiguity_group_id": str(group.get("group_id", "")),
            "identity_ambiguity_object_ids": group_ids,
            "identity_disambiguation": bool(singular_group and not group_is_target),
            "viewpoint_candidates": viewpoint_candidates,
            "viewpoint_selection": (
                "identity_group_new_baseline_away_from_failed_projection"
                if singular_group and viewpoint_candidates
                else "unobserved_pair_then_maximum_baseline"
                if viewpoint_candidates
                else "tuple_participant_centroid"
            ),
            "predicate": str(relation_node.get("predicate", "")),
            "unknown_binding": binding,
        })
    candidates: list[dict[str, Any]] = []

    def anchor_needs_localization(object_id: int) -> bool:
        """Keep a weak anchor in localization before relation probing."""
        value = objects.get(object_id, {})
        return bool(
            value.get("status") != "confirmed"
            or str(value.get("semantic_status", "")) != "verified"
            or not isinstance(value.get("center_3d"), Sequence)
            or any(
                isinstance(evidence, Mapping)
                and evidence.get("bearing_only") is True
                for evidence in value.get("evidence", ())
            )
        )

    for raw_node in (execution or {}).get("node_results", ()):
        if not isinstance(raw_node, Mapping):
            continue
        relation_id = str(raw_node.get("relation_node_id", ""))
        argument_entities = [
            str(value) for value in raw_node.get("argument_entities", ())
        ]
        for raw_tuple in raw_node.get("tuple_results", ()):
            if not isinstance(raw_tuple, Mapping) or str(raw_tuple.get("state", UNKNOWN)) != UNKNOWN:
                continue
            binding = raw_tuple.get("binding", {})
            if not isinstance(binding, Mapping):
                continue
            participant_ids: list[int] = []
            participant_entities: list[str] = []
            for entity_id in argument_entities:
                try:
                    object_id = int(binding[entity_id])
                except (KeyError, TypeError, ValueError):
                    continue
                if object_id not in participant_ids:
                    participant_ids.append(object_id)
                    participant_entities.append(entity_id)
            target_id = binding.get(str(graph.get("target_entity", "")))
            try:
                target_id_int = int(target_id)
            except (TypeError, ValueError):
                target_id_int = None
            anchor_ids = [value for value in participant_ids if value != target_id_int]
            if target_id_int is None and unknown_target_ids:
                target_id_int = min(unknown_target_ids)
            if not participant_ids and target_id_int is None:
                continue
            visible_ids = list(dict.fromkeys(
                ([target_id_int] if target_id_int is not None else []) + anchor_ids
            ))
            centers = [
                objects[object_id].get("center_3d")
                for object_id in visible_ids
                if object_id in objects
                and isinstance(objects[object_id].get("center_3d"), Sequence)
            ]
            center_xy = None
            if centers:
                center_xy = [
                    sum(float(center[axis]) for center in centers) / len(centers)
                    for axis in range(2)
                ]
            viewpoint_candidates = _identity_viewpoint_candidates(
                visible_ids,
                objects,
                previous_viewpoints,
                failed_navigation_targets,
                failed_navigation_poses,
            )
            if viewpoint_candidates:
                selected = max(
                    viewpoint_candidates,
                    key=lambda value: (
                        _unbounded_distance(value.get("failed_target_clearance_m")),
                        _unbounded_distance(value.get("failed_pose_clearance_m")),
                        not bool(value["already_observed_group"]),
                        _unbounded_distance(value.get("novel_baseline_m")),
                    ),
                )
                center_xy = [
                    float(value)
                    for value in selected["navigation_target_xy"]
                ]
            anchor_first = any(
                object_id in ambiguous_ids
                or anchor_needs_localization(object_id)
                for object_id in anchor_ids
            )
            novelty_values = [
                distance
                for value in viewpoint_candidates
                if (distance := _finite_distance_or_none(
                    value.get("novel_baseline_m")
                )) is not None
            ]
            candidates.append({
                "schema_version": "count_query_targeted_acquisition_v1",
                "mode": "relation_targeted",
                "reason": "unknown_relation_tuple",
                "relation_ids": [relation_id],
                "target_entity_ids": list(dict.fromkeys(participant_entities)),
                "target_object_ids": visible_ids,
                "required_visible_object_ids": visible_ids,
                "anchor_object_ids": anchor_ids,
                "anchor_first": bool(anchor_first),
                "joint_visibility_required": True,
                "navigation_target_xy": center_xy,
                "requires_new_station": True,
                "viewpoint_candidates": viewpoint_candidates,
                "viewpoint_selection": (
                    "joint_visibility_new_baseline_away_from_failed_projection"
                    if viewpoint_candidates else "tuple_participant_centroid"
                ),
                "predicate": str(raw_node.get("predicate", "")),
                "unknown_binding": dict(binding),
                "tuple_probe_priority": {
                    "anchor_requires_localization": any(
                        anchor_needs_localization(object_id)
                        for object_id in anchor_ids
                    ),
                    "focus_target": bool(
                        target_id_int is not None
                        and target_id_int in unknown_target_ids
                    ),
                    "ambiguous_participant_count": sum(
                        object_id in ambiguous_ids
                        for object_id in visible_ids
                    ),
                    "tentative_participant_count": sum(
                        objects.get(object_id, {}).get("status") != "confirmed"
                        for object_id in visible_ids
                    ),
                    "new_baseline_m": (
                        max(novelty_values) if novelty_values else None
                    ),
                },
            })

    if candidates:
        # UNKNOWN tuples are all admissible information targets.  Select the
        # one that can resolve the current count/identity frontier and offers
        # a new joint subject-anchor baseline; raw tuple enumeration order is
        # not an acquisition policy.
        return _decorate_targeted_acquisition(graph, snapshot, max(
            candidates,
            key=lambda value: (
                not bool(value["tuple_probe_priority"][
                    "anchor_requires_localization"
                ]),
                bool(value["tuple_probe_priority"]["focus_target"]),
                int(value["tuple_probe_priority"]["ambiguous_participant_count"]),
                int(value["tuple_probe_priority"]["tentative_participant_count"]),
                _unbounded_distance(
                    value["tuple_probe_priority"].get("new_baseline_m")
                ),
                -sum(int(item) for item in value.get("target_object_ids", ())),
            ),
        ))

    target_entity = str(graph.get("target_entity", ""))
    target_domain = _object_domain(variables.get(target_entity, {}), list(objects.values()))
    all_target_ids = [int(value["object_id"]) for value in target_domain]
    target_state_by_id = {
        int(value.get("object_id")): str(
            value.get("relation_state", value.get("state", UNKNOWN))
        ).upper()
        for value in (execution or {}).get("target_id_states", ())
        if isinstance(value, Mapping)
        and str(value.get("object_id", "")).lstrip("-").isdigit()
    }
    previous_viewpoints = [
        tuple(float(value) for value in position[:2])
        for record in snapshot.get("viewpoint_history", ())
        if isinstance(record, Mapping)
        for position in (record.get("viewpoint_position_map"),)
        if isinstance(position, Sequence)
        and not isinstance(position, (str, bytes))
        and len(position) >= 2
        and all(math.isfinite(float(value)) for value in position[:2])
    ]
    previous_viewpoints.extend(failed_navigation_poses)

    def novelty(object_id: int) -> float:
        center = objects.get(object_id, {}).get("center_3d")
        if not (
            isinstance(center, Sequence)
            and not isinstance(center, (str, bytes))
            and len(center) >= 2
            and previous_viewpoints
        ):
            return 0.0
        return min(
            math.dist(
                (float(center[0]), float(center[1])),
                viewpoint,
            )
            for viewpoint in previous_viewpoints
        )

    # The discovery branch still keeps the full graph domain for counting,
    # but a single acquisition should focus on the most informative current
    # candidate and its relation anchor.  Passing every room candidate here
    # made targeted acquisition indistinguishable from a global panorama.
    focus_ids = sorted(
        object_id for object_id in unknown_target_ids
        if object_id in all_target_ids
    )
    if not focus_ids:
        focus_candidates = [
            object_id for object_id in all_target_ids
            if (
                target_state_by_id.get(object_id, UNKNOWN) not in {YES, NO}
                or objects.get(object_id, {}).get("status") != "confirmed"
            )
        ]
        focus_candidates = focus_candidates or list(all_target_ids)
        if focus_candidates:
            focus_ids = [
                max(
                    focus_candidates,
                    key=lambda object_id: (
                        novelty(object_id),
                        objects.get(object_id, {}).get("status") != "confirmed",
                        -int(objects.get(object_id, {}).get("observation_count", 0) or 0),
                        -object_id,
                    ),
                )
            ]

    anchor_ids: list[int] = []
    anchor_selection_hypotheses: list[dict[str, Any]] = []
    for entity_id, variable in variables.items():
        if entity_id == target_entity:
            continue
        domain = _object_domain(variable, list(objects.values()))
        if not domain:
            continue
        resolved_ids = resolved_singular_domains.get(entity_id, ())
        if str(variable.get("quantifier", "")) == "SINGULAR" and resolved_ids:
            candidate_anchor_ids = [
                int(value) for value in resolved_ids if int(value) in objects
            ]
        else:
            candidate_anchor_ids = [
                int(value["object_id"]) for value in domain
            ]
        focus_set = set(focus_ids)
        for anchor_id in candidate_anchor_ids:
            matching_tuples = []
            for relation_id, node, tuple_result in relation_tuples:
                binding = tuple_result.get("binding", {})
                if not isinstance(binding, Mapping):
                    continue
                try:
                    bound_target = int(binding[target_entity])
                    bound_anchor = int(binding[entity_id])
                except (KeyError, TypeError, ValueError):
                    continue
                if bound_anchor != anchor_id or (
                    focus_set and bound_target not in focus_set
                ):
                    continue
                state = str(tuple_result.get("state", UNKNOWN)).upper()
                try:
                    probability = float(
                        tuple_result.get(
                            "relation_probability",
                            tuple_result.get("probability", 0.5),
                        )
                    )
                except (TypeError, ValueError):
                    probability = 0.5
                matching_tuples.append({
                    "relation_id": relation_id,
                    "predicate": str(node.get("predicate", "")),
                    "target_object_id": bound_target,
                    "state": state,
                    "relation_probability": max(
                        0.0, min(1.0, probability)
                    ),
                    "evidence_count": len(
                        tuple_result.get("evidence_ids", ())
                    ),
                })
            state_rank = {
                YES: 3,
                UNKNOWN: 2,
                INVALID: 1,
                NO: 0,
            }
            best_tuple = max(
                matching_tuples,
                key=lambda value: (
                    state_rank.get(str(value["state"]), 0),
                    float(value["relation_probability"]),
                    int(value["evidence_count"]),
                ),
                default=None,
            )
            anchor_center = objects.get(anchor_id, {}).get("center_3d")
            focus_distances = [
                math.dist(
                    [float(objects[target_id]["center_3d"][axis]) for axis in range(2)],
                    [float(anchor_center[axis]) for axis in range(2)],
                )
                for target_id in focus_ids
                if target_id in objects
                and isinstance(objects[target_id].get("center_3d"), Sequence)
                and len(objects[target_id]["center_3d"]) >= 2
                and isinstance(anchor_center, Sequence)
                and len(anchor_center) >= 2
            ]
            anchor_selection_hypotheses.append({
                "anchor_entity_id": entity_id,
                "anchor_object_id": anchor_id,
                "best_tuple_state": (
                    str(best_tuple["state"])
                    if best_tuple is not None else UNKNOWN
                ),
                "best_relation_probability": (
                    float(best_tuple["relation_probability"])
                    if best_tuple is not None else 0.5
                ),
                "matching_tuple_count": len(matching_tuples),
                "matching_tuples": matching_tuples,
                "nearest_focus_distance_m": (
                    min(focus_distances) if focus_distances else None
                ),
                "identity_status": str(
                    objects.get(anchor_id, {}).get("status", "")
                ),
                "semantic_status": str(
                    objects.get(anchor_id, {}).get("semantic_status", "")
                ),
            })
        if candidate_anchor_ids:
            best_anchor = max(
                anchor_selection_hypotheses,
                key=lambda value: (
                    state_rank.get(str(value["best_tuple_state"]), 0),
                    float(value["best_relation_probability"]),
                    int(value["matching_tuple_count"]),
                    value["semantic_status"] == "verified",
                    value["identity_status"] == "confirmed",
                    -float(value["nearest_focus_distance_m"])
                    if isinstance(value["nearest_focus_distance_m"], (int, float))
                    else float("-inf"),
                    -int(value["anchor_object_id"]),
                ),
            )
            anchor_ids = [int(best_anchor["anchor_object_id"])]
        if anchor_ids:
            break
    relation_ids = [
        str(value.get("id", ""))
        for value in graph.get("relation_nodes", ())
        if isinstance(value, Mapping)
    ]
    visible_ids = list(dict.fromkeys([*focus_ids, *anchor_ids]))
    centers = [
        objects[object_id].get("center_3d")
        for object_id in dict.fromkeys(visible_ids)
        if object_id in objects and isinstance(objects[object_id].get("center_3d"), Sequence)
    ]
    center_xy = (
        [sum(float(center[axis]) for center in centers) / len(centers) for axis in range(2)]
        if centers else None
    )
    viewpoint_candidates: list[dict[str, Any]] = []
    if (
        visible_ids
        and (failed_navigation_targets or failed_navigation_poses)
    ):
        viewpoint_candidates = _identity_viewpoint_candidates(
            visible_ids,
            objects,
            previous_viewpoints,
            failed_navigation_targets,
            failed_navigation_poses,
        )
        if viewpoint_candidates:
            selected = max(
                viewpoint_candidates,
                key=lambda value: (
                    _unbounded_distance(value.get("failed_target_clearance_m")),
                    _unbounded_distance(value.get("failed_pose_clearance_m")),
                    not bool(value["already_observed_group"]),
                    _unbounded_distance(value.get("novel_baseline_m")),
                ),
            )
            center_xy = [
                float(value) for value in selected["navigation_target_xy"]
            ]
    return _decorate_targeted_acquisition(graph, snapshot, {
        "schema_version": "count_query_targeted_acquisition_v1",
        "mode": "discovery_targeted",
        "reason": "count_discovery_not_converged",
        "relation_ids": relation_ids,
        "target_entity_ids": list(variables),
        "target_object_ids": list(dict.fromkeys(visible_ids)),
        "required_visible_object_ids": list(dict.fromkeys(visible_ids)),
        "anchor_object_ids": anchor_ids,
        "anchor_first": bool(
            anchor_ids
            and any(
                objects.get(object_id, {}).get("status") != "confirmed"
                for object_id in anchor_ids
            )
        ),
        "joint_visibility_required": bool(relation_ids),
        "navigation_target_xy": center_xy,
        "requires_new_station": False,
        "viewpoint_candidates": viewpoint_candidates,
        "viewpoint_selection": (
            "discovery_new_baseline_away_from_failed_projection"
            if viewpoint_candidates else "participant_centroid"
        ),
        "anchor_selection_mode": (
            "joint_target_anchor_tuple_evidence"
            if anchor_selection_hypotheses else "not_applicable"
        ),
        "anchor_selection_hypotheses": anchor_selection_hypotheses,
    })


def assess_count_query_domain(
    graph: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess discovery convergence for one CountQuery.

    A panorama is retained as provenance only.  Closure comes from independent
    valid viewpoints, target-region coverage, discovery saturation, confirmed
    canonical identities, and resolved relation tuples.
    """
    validate_count_query_graph(graph)
    active_objects = [
        dict(value)
        for value in snapshot.get("objects", ())
        if isinstance(value, Mapping)
        and value.get("status") in {"tentative", "confirmed"}
    ]
    variables = {
        str(value["entity_id"]): dict(value)
        for value in graph.get("variables", ())
        if isinstance(value, Mapping)
    }
    domains = {
        entity_id: _object_domain(variable, active_objects)
        for entity_id, variable in variables.items()
    }
    query_key = count_query_key(graph)
    object_lookup = _object_lookup(snapshot)
    target_entity = str(graph.get("target_entity", ""))
    target_ids = {
        int(value["object_id"]) for value in domains.get(target_entity, ())
    }
    canonical_confirmed_ids = sorted(
        object_id for object_id in target_ids
        if _canonical_count_identity(object_lookup.get(object_id, {}))
    )
    confirmed_ids = list(canonical_confirmed_ids)

    valid_records: list[dict[str, Any]] = []
    independent_positions: list[tuple[float, float, float]] = []
    covered_viewpoints: list[str] = []
    covered_regions: list[str] = []
    recent_new_counts: list[int] = []
    last_discovery_time: float | None = None
    new_instance_total = 0
    for raw in snapshot.get("viewpoint_history", ()):
        if not isinstance(raw, Mapping):
            continue
        position = _history_position(raw)
        valid = raw.get("valid_for_count_closure") is True or bool(
            position is not None
            and int(raw.get("rejected_observation_count", 0) or 0) == 0
            and isinstance(raw.get("object_ids_by_class"), Mapping)
        )
        if not valid:
            continue
        record = dict(raw)
        if position is not None:
            independent = record.get("independent_viewpoint")
            if not isinstance(independent, bool):
                independent = not independent_positions or all(
                    math.dist(position, prior) >= 0.30
                    for prior in independent_positions
                )
            if independent:
                independent_positions.append(position)
        else:
            independent = False
        record["_independent"] = bool(independent)
        valid_records.append(record)
        acquisition_id = str(record.get("acquisition_id", "")).strip()
        if acquisition_id:
            covered_viewpoints.append(acquisition_id)
        region = _history_region(record)
        if region:
            covered_regions.append(region)
        new_ids = [
            int(value) for value in record.get("new_instance_ids", ())
            if str(value).lstrip("-").isdigit()
        ]
        new_count = sum(object_id in target_ids for object_id in new_ids)
        if not new_ids and isinstance(record.get("new_instance_ids_by_class"), Mapping):
            target_names = {
                str(variables.get(target_entity, {}).get("class_name", "")).lower(),
                *(
                    str(value).lower()
                    for value in variables.get(target_entity, {}).get("aliases", ())
                ),
            }
            new_count = sum(
                len(value)
                for key, value in record["new_instance_ids_by_class"].items()
                if str(key).lower() in target_names and isinstance(value, Sequence)
            )
        new_count = max(0, int(new_count))
        recent_new_counts.append(new_count)
        new_instance_total += new_count
        if new_count > 0:
            try:
                last_discovery_time = float(
                    record.get("recorded_at_unix", last_discovery_time)
                )
            except (TypeError, ValueError):
                pass

    recent_new_counts = recent_new_counts[-8:]
    consecutive_no_new = 0
    for record, new_count in zip(
        reversed(valid_records[-8:]), reversed(recent_new_counts)
    ):
        if record.get("_independent") is not True:
            continue
        if new_count != 0:
            break
        consecutive_no_new += 1
    independent_view_count = sum(
        bool(record.get("_independent")) for record in valid_records
    )
    covered_viewpoints = list(dict.fromkeys(covered_viewpoints))
    covered_regions = list(dict.fromkeys(covered_regions))
    coverage_threshold = 2
    coverage_score = min(
        1.0,
        independent_view_count / coverage_threshold,
        len(covered_regions) / coverage_threshold,
        consecutive_no_new / 2.0,
    )

    domain_ids = {
        int(value["object_id"])
        for domain in domains.values()
        for value in domain
    }
    all_ambiguous_groups = [
        dict(group) for group in _identity_groups(snapshot)
        if domain_ids.intersection({
            int(value) for value in group.get("canonical_object_ids", ())
            if str(value).lstrip("-").isdigit()
        })
    ]
    unconfirmed_ids = sorted(
        object_id for object_id in domain_ids
        if not _canonical_count_identity(object_lookup.get(object_id, {}))
    )

    unknown_relation_ids: list[str]
    invalid_relation_ids: list[str] = []
    invalid_relation_tuples: list[dict[str, Any]] = []
    unknown_relation_tuples: list[dict[str, Any]] = []
    if execution is not None:
        unknown_relation_ids = sorted({
            str(value.get("relation_node_id", ""))
            for value in execution.get("node_results", ())
            if isinstance(value, Mapping) and int(value.get("unknown_tuple_count", 0) or 0) > 0
            and str(value.get("relation_node_id", ""))
        })
        invalid_relation_ids = sorted({
            str(value.get("relation_node_id", ""))
            for value in execution.get("node_results", ())
            if isinstance(value, Mapping)
            and int(value.get("invalid_tuple_count", 0) or 0) > 0
            and str(value.get("relation_node_id", ""))
        })
        for node in execution.get("node_results", ()):
            if not isinstance(node, Mapping):
                continue
            for tuple_result in node.get("tuple_results", ()):
                if isinstance(tuple_result, Mapping) and str(
                    tuple_result.get("state", UNKNOWN)
                ) == UNKNOWN:
                    unknown_relation_tuples.append({
                        "relation_node_id": str(node.get("relation_node_id", "")),
                        "binding": dict(tuple_result.get("binding", {})),
                    })
                elif isinstance(tuple_result, Mapping) and str(
                    tuple_result.get("state", UNKNOWN)
                ).upper() == INVALID:
                    invalid_relation_tuples.append({
                        "relation_node_id": str(node.get("relation_node_id", "")),
                        "binding": dict(tuple_result.get("binding", {})),
                        "reason_code": str(
                            tuple_result.get("reason_code", "relation_evidence_invalid")
                        ),
                    })
    else:
        unknown_relation_ids = []

    # Relation filtering may refute a high-recall proposal without ever
    # making it a canonical identity.  Only a positive/unknown candidate can
    # still affect the count; a relation-NO proposal is already removed by
    # the Count Graph and must not keep closure open by itself.
    relation_yes_ids: set[int] = set()
    refuted_ids: set[int] = set()
    pending_identity_ids: set[int] = set()
    unresolved_target_ids: set[int] = set()
    target_state_by_id: dict[int, dict[str, Any]] = {}
    if execution is not None:
        for raw_state in execution.get("target_id_states", ()):
            if not isinstance(raw_state, Mapping):
                continue
            try:
                object_id = int(raw_state.get("object_id"))
            except (TypeError, ValueError):
                continue
            target_state_by_id[object_id] = dict(raw_state)
            relation_state = str(
                raw_state.get("relation_state", raw_state.get("state", UNKNOWN))
            ).upper()
            identity_status = str(
                raw_state.get("identity_status", "tentative")
            ).lower()
            if relation_state == NO:
                refuted_ids.add(object_id)
            elif relation_state == YES:
                relation_yes_ids.add(object_id)
                if (
                    identity_status != "confirmed"
                    or not _canonical_count_identity(object_lookup.get(object_id, {}))
                ):
                    pending_identity_ids.add(object_id)
            else:
                unresolved_target_ids.add(object_id)

    if execution is None:
        unresolved_target_ids.update(
            int(value) for value in unconfirmed_ids
        )
    unresolved_target_ids.update(pending_identity_ids)
    # Positive support for a canonical target wins over rejected alternative
    # identity hypotheses.  The two sets describe different graph states and
    # must never contain the same object ID.
    refuted_ids.difference_update(relation_yes_ids)
    if execution is not None:
        # In the closure state, "confirmed" means a canonical target that is
        # also positively supported by the Count Graph.  Keep the broader
        # identity-only set in confirmed_candidate_ids below; otherwise an
        # identity-confirmed but relation-refuted object appears contradictory
        # in the query-local state.
        confirmed_ids = sorted(
            set(canonical_confirmed_ids).intersection(relation_yes_ids)
        )
    unresolved_target_ids.difference_update(refuted_ids)

    resolved_singular_domains: dict[str, list[int]] = {}
    if isinstance(execution, Mapping):
        raw_resolved = execution.get("resolved_singular_domains", {})
        if isinstance(raw_resolved, Mapping):
            for entity_id, values in raw_resolved.items():
                if not isinstance(values, Sequence) or isinstance(
                    values, (str, bytes, bytearray)
                ):
                    continue
                try:
                    resolved_singular_domains[str(entity_id)] = sorted({
                        int(value) for value in values
                    })
                except (TypeError, ValueError):
                    continue
    singular_domain_ids = {
        int(object_id)
        for entity_id, variable in variables.items()
        if str(variable.get("quantifier", "")) == "SINGULAR"
        for object_id in resolved_singular_domains.get(
            entity_id,
            [int(value["object_id"]) for value in domains[entity_id]],
        )
    }
    relevant_identity_ids = singular_domain_ids | unresolved_target_ids | pending_identity_ids
    ambiguous_groups = [
        group for group in all_ambiguous_groups
        if relevant_identity_ids.intersection({
            int(value) for value in group.get("canonical_object_ids", ())
            if str(value).lstrip("-").isdigit()
        })
    ]

    singular_confirmed_domains: dict[str, list[int]] = {}
    for entity_id, variable in variables.items():
        if str(variable.get("quantifier", "")) != "SINGULAR":
            continue
        candidate_ids = resolved_singular_domains.get(
            entity_id,
            [int(value["object_id"]) for value in domains[entity_id]],
        )
        singular_confirmed_domains[entity_id] = [
            object_id
            for object_id in candidate_ids
            if _canonical_count_identity(object_lookup.get(object_id, {}))
        ]

    blockers: list[str] = []
    for entity_id, variable in variables.items():
        if not domains[entity_id]:
            blockers.append(f"count_entity_domain_empty:{entity_id}")
        if str(variable.get("quantifier", "")) == "SINGULAR":
            confirmed_domain = singular_confirmed_domains[entity_id]
            if not confirmed_domain:
                blockers.append(f"singular_entity_identity_pending:{entity_id}")
            elif len(confirmed_domain) != 1:
                blockers.append(f"singular_entity_domain_ambiguous:{entity_id}")
    if independent_view_count < 2:
        blockers.append("independent_viewpoint_coverage_pending")
    if len(covered_regions) < coverage_threshold:
        blockers.append("query_region_coverage_pending")
    if consecutive_no_new < 2:
        blockers.append("discovery_not_saturated")
    if unresolved_target_ids:
        blockers.append("canonical_identity_confirmation_pending")
    if ambiguous_groups:
        blockers.append("identity_association_ambiguous")
    if unknown_relation_ids:
        blockers.extend(
            f"unknown_relation:{value}" for value in unknown_relation_ids
        )
    if invalid_relation_ids:
        blockers.extend(
            f"invalid_relation:{value}" for value in invalid_relation_ids
        )

    closed = not blockers
    next_acquisition = {}
    if not closed and invalid_relation_ids:
        # INVALID is a pipeline inconsistency, not missing world evidence.  It
        # must be recomputed at the current station before any navigation is
        # considered; the acquisition planner must not turn it into a probe.
        next_acquisition = {
            "schema_version": "count_query_repair_v1",
            "mode": "recompute_current_station",
            "reason": "relation_evidence_invalid",
            "relation_ids": list(invalid_relation_ids),
            "navigation_required": False,
            "force_perception": True,
        }
    elif not closed:
        next_acquisition = (
            build_count_query_targeted_acquisition(graph, snapshot, execution)
            if execution is not None else {}
        )
        if not next_acquisition:
            next_acquisition = {
                "schema_version": "count_query_targeted_acquisition_v1",
                "mode": "discovery_targeted",
                "reason": "count_discovery_not_converged",
                "target_entity_ids": list(variables),
                "target_object_ids": sorted(target_ids),
                "required_visible_object_ids": sorted(target_ids),
                "relation_ids": [
                    str(value.get("id", "")) for value in graph.get("relation_nodes", ())
                    if isinstance(value, Mapping)
                ],
                "anchor_object_ids": [],
                "anchor_first": False,
                "joint_visibility_required": bool(graph.get("relation_nodes")),
            }
    state = CountClosureState(
        query_key=query_key,
        confirmed_ids=confirmed_ids,
        refuted_ids=sorted(refuted_ids),
        pending_identity_ids=sorted(pending_identity_ids),
        unresolved_target_ids=sorted(unresolved_target_ids),
        unknown_relation_ids=unknown_relation_ids,
        invalid_relation_ids=invalid_relation_ids,
        identity_ambiguous_groups=ambiguous_groups,
        covered_viewpoints=covered_viewpoints,
        covered_regions=covered_regions,
        recent_new_instance_counts=recent_new_counts,
        last_discovery_time=last_discovery_time,
        new_instance_total=new_instance_total,
        valid_view_count=len(valid_records),
        independent_view_count=independent_view_count,
        consecutive_no_new_views=consecutive_no_new,
        coverage_threshold=coverage_threshold,
        coverage_score=coverage_score,
        closed=closed,
        closure_reasons=list(dict.fromkeys(blockers)),
        next_acquisition=next_acquisition,
    )
    result = state.to_dict()
    result.update({
        "schema_version": "count_query_domain_v2",
        "closed": closed,
        "state": "CLOSED_COUNT_QUERY_CONVERGED" if closed else "COUNT_QUERY_CONVERGING",
        "authority": "scene_memory_count_query_convergence" if closed else None,
        "active_object_count": len(active_objects),
        "candidate_domains": {
            entity_id: [int(value["object_id"]) for value in values]
            for entity_id, values in domains.items()
        },
        "confirmed_candidate_ids": canonical_confirmed_ids,
        "count_graph_confirmed_ids": confirmed_ids,
        "unconfirmed_candidate_ids": unconfirmed_ids,
        "relation_yes_target_ids": sorted(relation_yes_ids),
        "refuted_target_ids": sorted(refuted_ids),
        "pending_identity_target_ids": sorted(pending_identity_ids),
        "unresolved_target_ids": sorted(unresolved_target_ids),
        "target_id_states": list(target_state_by_id.values()),
        "unknown_relation_tuples": unknown_relation_tuples,
        "invalid_relation_ids": invalid_relation_ids,
        "invalid_relation_tuples": invalid_relation_tuples,
        "invalid_relation_evidence_current_revision": bool(
            invalid_relation_ids
        ),
        "full_panorama_observed": any(
            isinstance(value, Mapping) and value.get("full_panorama") is True
            for value in snapshot.get("viewpoint_history", ())
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "closure_state": result.copy(),
    })
    return result


def _evidence_provenance(value: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    source_views: set[str] = set()
    optical_centers: set[str] = set()
    for item in value.get("evidence", ()):
        source_views.update(
            str(view_id)
            for view_id in item.get("source_view_ids", ())
            if str(view_id).strip()
        )
        optical_center = str(item.get("optical_center_group", "")).strip()
        if optical_center:
            optical_centers.add(optical_center)
    return source_views, optical_centers


def _binding_support_rank(
    value: Mapping[str, Any],
    *,
    evaluation_candidate: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Bind an argument to the candidate's own visual provenance.

    Relation arguments must describe the same observed scene context.  Global
    object confidence is only a tie-breaker; otherwise one highly persistent
    anchor is incorrectly paired with every target in the room.
    """
    evidence = [dict(item) for item in value.get("evidence", ())]
    candidate_views, candidate_centers = _evidence_provenance(
        evaluation_candidate
    )
    value_views, value_centers = _evidence_provenance(value)
    shared_view_count = len(candidate_views.intersection(value_views))
    shared_center_count = len(candidate_centers.intersection(value_centers))
    qualified_count = sum(
        bool(item.get("qwen_verified"))
        or isinstance(item.get("bearing_observation"), Mapping)
        for item in evidence
    )
    source_view_count = sum(
        len(item.get("source_view_ids", ())) for item in evidence
    )
    geometry_support = sum(
        float(item.get("geometry_confidence", 0.0)) for item in evidence
    )
    return (
        shared_view_count > 0,
        shared_view_count,
        shared_center_count > 0,
        shared_center_count,
        qualified_count > 0,
        value.get("status") == "confirmed",
        int(value.get("independent_viewpoint_count", 0)),
        qualified_count,
        source_view_count,
        geometry_support,
        len(evidence),
        -int(value["object_id"]),
    )


def _all_bindings_per_candidate(
    *,
    argument_entities: Sequence[str],
    domains: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluation_entity: str,
) -> list[tuple[dict[str, int], list[dict[str, Any]]]]:
    """Enumerate exact identity/relation hypotheses for each candidate.

    A singular reference is a query variable, not a permission to discard
    alternative object IDs before relation evidence is evaluated.  The graph
    must be able to compare ``target_i`` against every compatible anchor
    hypothesis and then use YES/NO/UNKNOWN evidence to resolve the domain.
    """
    results: list[tuple[dict[str, int], list[dict[str, Any]]]] = []
    other_entities = list(dict.fromkeys(
        str(entity_id)
        for entity_id in argument_entities
        if str(entity_id) != str(evaluation_entity)
    ))
    for raw_candidate in domains.get(evaluation_entity, ()):
        candidate = dict(raw_candidate)
        candidate_options = [
            [dict(value) for value in domains.get(entity_id, ())]
            for entity_id in other_entities
        ]
        combinations = itertools.product(*candidate_options) if candidate_options else [()]
        emitted = False
        for combination in combinations:
            selected: dict[str, dict[str, Any]] = {
                str(evaluation_entity): candidate,
            }
            selected.update({
                entity_id: value
                for entity_id, value in zip(other_entities, combination)
            })
            selected_ids = [
                int(selected[entity_id]["object_id"])
                for entity_id in argument_entities
                if entity_id in selected
            ]
            if len(selected_ids) != len(set(selected_ids)):
                # A target cannot be its own relation anchor hypothesis.
                continue
            if any(entity_id not in selected for entity_id in argument_entities):
                continue
            emitted = True
            results.append((
                {
                    entity_id: int(selected[entity_id]["object_id"])
                    for entity_id in argument_entities
                },
                [selected[entity_id] for entity_id in argument_entities],
            ))
        if not emitted:
            results.append((
                {evaluation_entity: int(candidate["object_id"])},
                [],
            ))
    return results


def _ranking_postprocess(
    node: Mapping[str, Any],
    tuple_results: list[dict[str, Any]],
) -> None:
    """Resolve ARGMIN/ARGMAX from complete horizontal metric evidence.

    The relation verifier may describe every candidate as visually plausible;
    that is not a ranking.  Once the graph has enumerated the complete
    candidate domain, horizontal map-frame centers are the authoritative
    metric evidence.  XY uncertainty must still separate the winner from its
    competitors; otherwise the ranking remains UNKNOWN.  Each anchor is a
    separate joint target/anchor hypothesis, so unresolved anchor identities
    cannot be collapsed into one global XYZ winner.
    """
    eligible = [
        value
        for value in tuple_results
        if value.get("dependency_state") == YES
    ]
    if any(
        value.get("dependency_state") in {UNKNOWN, INVALID}
        for value in tuple_results
    ):
        for value in eligible:
            value.update(
                state=UNKNOWN,
                reason_code="ranking_dependency_domain_incomplete",
            )
        return
    for value in eligible:
        subject = value.pop("_subject")
        objects = value.pop("_objects")
        if len(objects) != 1:
            value.update(
                state=UNKNOWN,
                reason_code="ranking_reference_arity_invalid",
            )
            continue
        try:
            subject_center = [float(component) for component in subject["center_3d"]]
            object_center = [float(component) for component in objects[0]["center_3d"]]
        except (KeyError, TypeError, ValueError):
            value.update(
                state=UNKNOWN,
                reason_code="ranking_world_geometry_missing",
            )
            continue
        if (
            len(subject_center) < 3
            or len(object_center) < 3
            or not all(math.isfinite(component) for component in subject_center)
            or not all(math.isfinite(component) for component in object_center)
        ):
            value.update(
                state=UNKNOWN,
                reason_code="ranking_world_geometry_invalid",
            )
            continue
        distance = math.hypot(
            subject_center[0] - object_center[0],
            subject_center[1] - object_center[1],
        )
        uncertainty = _center_uncertainty_xy(subject) + _center_uncertainty_xy(
            objects[0]
        )
        value["ranking_distance_m"] = float(distance)
        value["ranking_distance_uncertainty_m"] = float(uncertainty)
        value["ranking_lower_bound_m"] = max(0.0, distance - 2.0 * uncertainty)
        value["ranking_upper_bound_m"] = distance + 2.0 * uncertainty
        value["ranking_metric"] = "horizontal_xy"
        value["ranking_anchor_object_id"] = int(objects[0]["object_id"])
    if not eligible:
        return
    if any(
        value.get("reason_code")
        in {
            "ranking_reference_arity_invalid",
            "ranking_world_geometry_missing",
            "ranking_world_geometry_invalid",
        }
        for value in eligible
    ):
        for value in eligible:
            value.update(
                state=UNKNOWN,
                reason_code="ranking_world_geometry_incomplete",
            )
        return
    groups: dict[int, list[dict[str, Any]]] = {}
    for value in eligible:
        groups.setdefault(int(value["ranking_anchor_object_id"]), []).append(value)
    for group in groups.values():
        if any(value.get("state") == INVALID for value in group):
            for value in group:
                if value.get("state") != INVALID:
                    value.update(
                        state=UNKNOWN,
                        reason_code="ranking_visual_evidence_invalid",
                    )
            continue
        if len(group) == 1:
            winner = group[0]
        elif node["operator"] == "ARGMIN":
            winner = min(
                group,
                key=lambda value: float(value["ranking_upper_bound_m"]),
            )
            competitor_lower = min(
                float(value["ranking_lower_bound_m"])
                for value in group if value is not winner
            )
            if float(winner["ranking_upper_bound_m"]) >= competitor_lower:
                for value in group:
                    value.update(
                        state=UNKNOWN,
                        reason_code="ranking_distance_uncertainty_overlap",
                    )
                continue
        else:
            winner = max(
                group,
                key=lambda value: float(value["ranking_lower_bound_m"]),
            )
            competitor_upper = max(
                float(value["ranking_upper_bound_m"])
                for value in group if value is not winner
            )
            if float(winner["ranking_lower_bound_m"]) <= competitor_upper:
                for value in group:
                    value.update(
                        state=UNKNOWN,
                        reason_code="ranking_distance_uncertainty_overlap",
                    )
                continue
        winner_id = int(winner["subject_object_id"])
        for value in group:
            visual_state = str(value.get("state", UNKNOWN))
            value["visual_state"] = visual_state
            value["state"] = YES if int(value["subject_object_id"]) == winner_id else NO
            value["reason_code"] = "world_geometry_ranking_horizontal_xy"
            value["evidence_source"] = "world_geometry"


def execute_count_query_graph(
    graph: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    relation_evaluator: RelationEvaluator,
) -> dict[str, Any]:
    """Evaluate a numerical graph and return a cardinality interval.

    The lower bound is the number of persistent target IDs proven YES.  When
    the candidate domain is closed, the upper bound additionally includes
    every existing target ID whose relation state is UNKNOWN.  An open domain
    has no finite upper bound.
    """
    validate_count_query_graph(graph)
    active_objects = [
        dict(value)
        for value in snapshot.get("objects", ())
        if value.get("status") in {"tentative", "confirmed"}
    ]
    variables = {
        str(value["entity_id"]): dict(value)
        for value in graph.get("variables", ())
    }
    domains = {
        entity_id: _object_domain(variable, active_objects)
        for entity_id, variable in variables.items()
    }
    singular_domain_candidates = {
        entity_id: [int(value["object_id"]) for value in domains[entity_id]]
        for entity_id, variable in variables.items()
        if str(variable.get("quantifier", "")) == "SINGULAR"
    }
    singular_domain_ambiguities = {
        entity_id: [int(value["object_id"]) for value in domains[entity_id]]
        for entity_id, variable in variables.items()
        if str(variable.get("quantifier", "")) == "SINGULAR"
        and len(domains[entity_id]) != 1
    }
    # Count-graph closure is task-specific.  SceneMemory remains open and
    # relation selector openness must never close or veto this graph.
    domain_closed = bool(snapshot.get("numerical_count_domain_closed"))
    nodes = {
        str(value["id"]): dict(value)
        for value in graph.get("relation_nodes", ())
    }
    node_results: dict[str, dict[str, Any]] = {}

    for node_id in graph.get("dependency_order", ()):
        node = nodes[str(node_id)]
        argument_entities = [
            str(node["subject_entity"]),
            *(str(value) for value in node.get("object_entities", ())),
        ]
        tuple_results: list[dict[str, Any]] = []
        target_entity = str(graph["target_entity"])
        evaluation_entity = (
            target_entity
            if target_entity in argument_entities
            else str(node["subject_entity"])
        )
        participant_bindings = _all_bindings_per_candidate(
            argument_entities=argument_entities,
            domains=domains,
            evaluation_entity=evaluation_entity,
        )
        wave3_items = []
        wave3_indexes = []
        dependency_by_index = {}
        for index, (binding, participants) in enumerate(participant_bindings):
            if not participants:
                continue
            dependency_state, dependency_diagnostics = _dependency_state(
                node, binding, node_results
            )
            dependency_by_index[index] = (
                dependency_state, dependency_diagnostics
            )
            if dependency_state == YES:
                wave3_indexes.append(index)
                wave3_items.append((
                    node, participants[0], participants[1:]
                ))
        wave3_verdicts = {}
        verify_many = getattr(relation_evaluator, "verify_many", None)
        if wave3_items and callable(verify_many):
            try:
                wave3_verdicts = dict(zip(
                    wave3_indexes, verify_many(wave3_items)
                ))
            except Exception:
                wave3_verdicts = {}
        for index, (binding, participants) in enumerate(participant_bindings):
            if not participants:
                tuple_results.append({
                    "binding": binding,
                    "subject_object_id": int(binding[evaluation_entity]),
                    "object_ids": [],
                    "object_instance_versions": [],
                    "dependency_state": UNKNOWN,
                    "dependency_diagnostics": [],
                    "state": UNKNOWN,
                    "reason_code": "relation_argument_domain_empty",
                    "evidence_ids": [],
                })
                continue
            object_ids = [int(value["object_id"]) for value in participants]
            dependency_state, dependency_diagnostics = dependency_by_index.get(
                index,
                _dependency_state(node, binding, node_results),
            )
            if dependency_state != YES:
                verdict = {
                    "state": dependency_state,
                    "reason_code": (
                        "relation_dependency_refuted"
                        if dependency_state == NO
                        else "relation_dependency_invalid"
                        if dependency_state == INVALID
                        else "relation_dependency_unknown"
                    ),
                    "evidence_ids": [],
                }
            else:
                try:
                    verdict = _normalized_verdict(
                        wave3_verdicts[index]
                        if index in wave3_verdicts
                        else relation_evaluator(
                            node, participants[0], participants[1:]
                        )
                    )
                except Exception as exc:
                    verdict = {
                        "state": UNKNOWN,
                        "reason_code": (
                            f"relation_evaluator_exception:{type(exc).__name__}:"
                            f"{str(exc)[:200]}"
                        ),
                        "evidence_ids": [],
                    }
            tuple_results.append({
                "binding": binding,
                "subject_object_id": object_ids[0],
                "object_ids": object_ids[1:],
                "object_instance_versions": [
                    int(value.get("instance_version", 0))
                    for value in participants
                ],
                "dependency_state": dependency_state,
                "dependency_diagnostics": dependency_diagnostics,
                **verdict,
                "_subject": participants[0],
                "_objects": list(participants[1:]),
            })

        if node.get("operator") in {"ARGMIN", "ARGMAX"}:
            _ranking_postprocess(node, tuple_results)
        for value in tuple_results:
            value.pop("_subject", None)
            value.pop("_objects", None)
        node_complete = all(
            value.get("state") not in {UNKNOWN, INVALID}
            for value in tuple_results
        )
        node_results[str(node_id)] = {
            "relation_node_id": str(node_id),
            "predicate": str(node["predicate"]),
            "operator": str(node["operator"]),
            "argument_entities": argument_entities,
            "argument_domain_object_ids": {
                entity_id: [int(value["object_id"]) for value in domains[entity_id]]
                for entity_id in argument_entities
            },
            "tuple_results": tuple_results,
            "complete": node_complete,
            "evaluation_deferred": False,
            "binding_policy": "all_identity_relation_hypotheses",
            "binding_entity": evaluation_entity,
            "yes_tuple_count": sum(value.get("state") == YES for value in tuple_results),
            "no_tuple_count": sum(value.get("state") == NO for value in tuple_results),
            "unknown_tuple_count": sum(
                value.get("state") == UNKNOWN for value in tuple_results
            ),
            "invalid_tuple_count": sum(
                value.get("state") == INVALID for value in tuple_results
            ),
        }

    # A singular class domain can contain detector hypotheses that are later
    # distinguished by the relation itself.  Resolve it only when exactly one
    # object ID has positive relation support and every other ID has no
    # unresolved tuple.  This is relation evidence resolving a variable, not
    # a hard-coded preference for one ID.
    resolved_singular_domains: dict[str, list[int]] = {}
    for entity_id, candidate_ids in singular_domain_candidates.items():
        if len(candidate_ids) == 1:
            resolved_singular_domains[entity_id] = list(candidate_ids)
            singular_domain_ambiguities.pop(entity_id, None)
            continue
        yes_ids: set[int] = set()
        uncertain_ids: set[int] = set()
        participates_in_relation = False
        for node_result in node_results.values():
            if entity_id not in node_result.get("argument_entities", ()):
                continue
            participates_in_relation = True
            for tuple_result in node_result.get("tuple_results", ()):
                binding = tuple_result.get("binding", {})
                if not isinstance(binding, Mapping) or entity_id not in binding:
                    continue
                try:
                    object_id = int(binding[entity_id])
                except (TypeError, ValueError):
                    continue
                state = str(tuple_result.get("state", UNKNOWN)).upper()
                if state == YES:
                    yes_ids.add(object_id)
                elif state in {UNKNOWN, INVALID}:
                    uncertain_ids.add(object_id)
        other_uncertain = uncertain_ids.difference(yes_ids)
        if participates_in_relation and len(yes_ids) == 1 and not other_uncertain:
            resolved_singular_domains[entity_id] = sorted(yes_ids)
            singular_domain_ambiguities.pop(entity_id, None)

    target_entity = str(graph["target_entity"])
    target_ids = [int(value["object_id"]) for value in domains[target_entity]]
    required_nodes = [
        str(value) for value in graph.get("root", {}).get(
            "required_relation_nodes", ()
        )
    ]
    target_has_attributes = bool(
        graph.get("root", {}).get("entity_constraint", {}).get("attributes")
    )
    object_status_by_id = {
        int(value["object_id"]): str(value.get("status", "tentative"))
        for value in active_objects
    }
    object_semantic_status_by_id = {
        int(value["object_id"]): str(value.get("semantic_status", "unverified"))
        for value in active_objects
    }

    def binding_matches_resolved_singular_domains(
        binding: Mapping[str, Any],
    ) -> bool:
        """Discard identity hypotheses rejected by a resolved singular slot."""
        for entity_id, object_ids in resolved_singular_domains.items():
            if entity_id not in binding or len(object_ids) != 1:
                continue
            try:
                if int(binding[entity_id]) not in {
                    int(value) for value in object_ids
                }:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    target_states = []
    for target_id in target_ids:
        relation_states = []
        for node_id in required_nodes:
            node_result = node_results[node_id]
            relevant = [
                value
                for value in node_result["tuple_results"]
                if int(value.get("binding", {}).get(target_entity, -1)) == target_id
                and isinstance(value.get("binding"), Mapping)
                and binding_matches_resolved_singular_domains(
                    value["binding"]
                )
            ]
            states = {
                str(value.get("state", UNKNOWN)).upper()
                for value in relevant
            }
            if INVALID in states:
                state = INVALID
            elif "CONFLICTED" in states:
                state = UNKNOWN
            elif YES in states and NO in states:
                # If the singular anchor/target identity is still unresolved,
                # contradictory relation bindings cannot make this target
                # countable.  A resolved singular domain has already removed
                # rejected bindings above; any remaining YES/NO mixture is a
                # genuine unresolved relation result.
                state = UNKNOWN
            elif YES in states:
                state = YES
            elif UNKNOWN in states:
                state = UNKNOWN
            elif relevant and node_result["complete"]:
                state = NO
            else:
                state = UNKNOWN
            relation_states.append({"relation_node_id": node_id, "state": state})
        if not required_nodes:
            state = UNKNOWN if target_has_attributes else YES
        elif any(value["state"] == INVALID for value in relation_states):
            state = INVALID
        elif any(value["state"] == NO for value in relation_states):
            state = NO
        elif all(value["state"] == YES for value in relation_states):
            state = YES
        else:
            state = UNKNOWN
        identity_status = object_status_by_id.get(target_id, "tentative")
        semantic_status = object_semantic_status_by_id.get(
            target_id,
            "unverified",
        )
        relation_state = state
        target_states.append({
            "object_id": target_id,
            "state": relation_state,
            "relation_state": relation_state,
            "identity_status": identity_status,
            "semantic_status": semantic_status,
            "countable": bool(
                relation_state == YES
                and identity_status == "confirmed"
                and semantic_status.strip().lower() == "verified"
            ),
            "relation_states": relation_states,
        })

    counted_ids = sorted(
        value["object_id"]
        for value in target_states
        if value.get("state") == YES and value.get("countable") is True
    )
    refuted_ids = sorted(
        value["object_id"] for value in target_states if value["state"] == NO
    )
    unknown_ids = sorted(
        value["object_id"]
        for value in target_states
        if value["state"] == UNKNOWN
    )
    invalid_ids = sorted(
        value["object_id"]
        for value in target_states
        if value["state"] == INVALID
    )
    pending_identity_ids = sorted(
        value["object_id"]
        for value in target_states
        if value.get("state") == YES
        and value.get("countable") is not True
    )
    all_nodes_complete = all(
        value["complete"] for value in node_results.values()
    )
    # Exhaustively refuting only the IDs currently recalled does not prove a
    # zero when the candidate domain is still open.  Preserve that missing
    # recall mass as UNKNOWN; positive proven IDs remain a usable lower bound.
    open_domain_zero_unknown = bool(
        not domain_closed and not counted_ids
    )
    identity_complete = (
        domain_closed
        and all_nodes_complete
        and not unknown_ids
        and not pending_identity_ids
        and not open_domain_zero_unknown
        and not singular_domain_ambiguities
        and not invalid_ids
    )
    object_cardinality_lower = len(counted_ids)
    object_cardinality_upper = len(counted_ids) + len(unknown_ids)
    cardinality_complete = bool(identity_complete)
    evidence_ids = list(dict.fromkeys(
        evidence_id
        for node_result in node_results.values()
        for result in node_result["tuple_results"]
        for evidence_id in result.get("evidence_ids", ())
    ))
    failure_reasons = []
    if unknown_ids:
        failure_reasons.append("target_relation_unknown")
    if invalid_ids:
        failure_reasons.append("relation_evidence_invalid")
    if any(
        int(value.get("invalid_tuple_count", 0) or 0) > 0
        for value in node_results.values()
    ):
        failure_reasons.append("relation_evidence_invalid_current_revision")
    if pending_identity_ids:
        failure_reasons.append("canonical_identity_confirmation_pending")
    if (
        any(not value["complete"] for value in node_results.values())
    ):
        failure_reasons.append("relation_graph_incomplete")
    if target_has_attributes and not required_nodes:
        failure_reasons.append("target_attribute_unverified")
    if open_domain_zero_unknown:
        failure_reasons.append("open_candidate_domain_cannot_prove_zero")
    if not domain_closed:
        failure_reasons.append("open_candidate_domain_cannot_prove_exact_count")
    if singular_domain_ambiguities:
        failure_reasons.append("singular_reference_unresolved")

    resolved_answer = len(counted_ids) if identity_complete else None

    relation_verifications = []
    for node_result in node_results.values():
        for tuple_result in node_result.get("tuple_results", ()):
            if tuple_result.get("dependency_state") != YES:
                # A dependency refutation is not fresh evidence for this
                # relation and must not be persisted as if Qwen observed it.
                continue
            object_ids = [
                int(value) for value in tuple_result.get("object_ids", ())
            ]
            if not object_ids:
                continue
            relation_verifications.append({
                "relation_id": str(node_result["relation_node_id"]),
                "predicate": str(node_result["predicate"]),
                "binding": dict(tuple_result.get("binding", {})),
                "subject_object_id": int(tuple_result["subject_object_id"]),
                "object_ids": object_ids,
                "state": str(tuple_result.get("state", UNKNOWN)),
                "reason_code": str(
                    tuple_result.get("reason_code", "relation_state_missing")
                ),
                "relation_probability": tuple_result.get(
                    "relation_probability"
                ),
                "evidence_ids": list(tuple_result.get("evidence_ids", ())),
                "geometry": dict(tuple_result.get("geometry", {})),
                "qwen": dict(tuple_result.get("qwen", {})),
                "persistent_state_inherited": bool(
                    tuple_result.get("persistent_state_inherited")
                ),
                "station_id": str(tuple_result.get("station_id", "")),
                "timestamp": tuple_result.get("timestamp"),
                "source_observation_ids": list(
                    tuple_result.get("source_observation_ids", ())
                ),
                "identity_version": tuple_result.get("identity_version"),
                "geometry_version": tuple_result.get("geometry_version"),
            })

    return {
        "schema_version": "count_query_execution_v1",
        "graph_schema_version": str(graph["schema_version"]),
        "scene_version": int(snapshot.get("scene_version", 0)),
        "target_entity": target_entity,
        "candidate_domain_closed": domain_closed,
        "numerical_count_domain_closed": domain_closed,
        "object_identity_domain_closed": domain_closed,
        "scene_memory_open": bool(snapshot.get("scene_memory_open", True)),
        "candidate_domains": {
            entity_id: [int(value["object_id"]) for value in values]
            for entity_id, values in domains.items()
        },
        "node_results": [
            node_results[str(value)] for value in graph.get("dependency_order", ())
        ],
        "resolved_singular_domains": {
            entity_id: list(object_ids)
            for entity_id, object_ids in resolved_singular_domains.items()
        },
        "target_id_states": target_states,
        "counted_target_ids": counted_ids,
        "refuted_target_ids": refuted_ids,
        "unknown_target_ids": unknown_ids,
        "invalid_target_ids": invalid_ids,
        "pending_identity_ids": pending_identity_ids,
        "invalid_relation_ids": sorted({
            str(node_result.get("relation_node_id", ""))
            for node_result in node_results.values()
            if int(node_result.get("invalid_tuple_count", 0) or 0) > 0
        }),
        "cardinality_lower_bound": object_cardinality_lower,
        "cardinality_upper_bound": (
            object_cardinality_upper + len(pending_identity_ids)
            if domain_closed else None
        ),
        "complete": cardinality_complete,
        "identity_complete": identity_complete,
        "cardinality_complete": cardinality_complete,
        "answer": resolved_answer,
        "evidence_ids": evidence_ids,
        "singular_domain_ambiguities": singular_domain_ambiguities,
        "cardinality_observation": None,
        "cardinality_resolution_mode": (
            "persistent_identity_graph"
            if identity_complete
            else "unresolved"
        ),
        "relation_verifications": relation_verifications,
        "invalid_relation_evidence_current_revision": bool(
            invalid_ids
            or any(
                int(value.get("invalid_tuple_count", 0) or 0) > 0
                for value in node_results.values()
            )
        ),
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
        "diagnostics": (
            []
            if domain_closed
            else ["object_identity_domain_open"]
        ),
    }


def finalize_count_query_execution_domain(
    execution: Mapping[str, Any],
    *,
    domain_closed: bool,
) -> dict[str, Any]:
    """Finalize cardinality from already-evaluated tuples without rerunning them."""
    result = copy.deepcopy(dict(execution))
    counted_ids = [int(value) for value in result.get("counted_target_ids", ())]
    unknown_ids = [int(value) for value in result.get("unknown_target_ids", ())]
    invalid_ids = [int(value) for value in result.get("invalid_target_ids", ())]
    pending_identity_ids = [
        int(value) for value in result.get("pending_identity_ids", ())
    ]
    singular_ambiguities = result.get("singular_domain_ambiguities", {})
    node_results = [
        value for value in result.get("node_results", ())
        if isinstance(value, Mapping)
    ]
    all_nodes_complete = all(
        value.get("complete") is True for value in node_results
    )
    complete = bool(
        domain_closed
        and all_nodes_complete
        and not unknown_ids
        and not invalid_ids
        and not pending_identity_ids
        and not singular_ambiguities
    )
    result.update({
        "candidate_domain_closed": bool(domain_closed),
        "numerical_count_domain_closed": bool(domain_closed),
        "object_identity_domain_closed": bool(domain_closed),
        "cardinality_upper_bound": (
            len(counted_ids) + len(unknown_ids) + len(pending_identity_ids)
            if domain_closed else None
        ),
        "complete": complete,
        "identity_complete": complete,
        "cardinality_complete": complete,
        "answer": len(counted_ids) if complete else None,
        "cardinality_resolution_mode": (
            "persistent_identity_graph" if complete else "unresolved"
        ),
        "diagnostics": [] if domain_closed else ["object_identity_domain_open"],
    })
    reasons = [
        str(value) for value in result.get("failure_reasons", ())
        if str(value) not in {
            "open_candidate_domain_cannot_prove_zero",
            "open_candidate_domain_cannot_prove_exact_count",
        }
    ]
    if not domain_closed:
        if not counted_ids:
            reasons.append("open_candidate_domain_cannot_prove_zero")
        reasons.append("open_candidate_domain_cannot_prove_exact_count")
    result["failure_reasons"] = list(dict.fromkeys(reasons))
    return result
