"""Single semantic owner for turning EvidenceNeed into observation work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from integrations.execution.resolver_contracts import EvidenceNeed


def _revision(value: Mapping[str, Any], key: str, *aliases: str) -> int:
    for name in (key, *aliases):
        try:
            return max(0, int(value.get(name, 0)))
        except (TypeError, ValueError):
            continue
    return 0


@dataclass(frozen=True)
class ObservationIntent:
    evidence_need: EvidenceNeed
    target: dict[str, Any]
    required_visible_object_ids: tuple[int, ...]
    joint_visibility_required: bool
    provenance: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "observation_intent_v1",
            "intent": "ACQUIRE_EVIDENCE",
            "evidence_need": self.evidence_need.to_dict(),
            "target": dict(self.target),
            "required_visible_object_ids": list(
                self.required_visible_object_ids
            ),
            "joint_visibility_required": bool(
                self.joint_visibility_required
            ),
            "provenance": list(self.provenance),
            "metadata": dict(self.metadata),
        }


class EvidenceAcquisitionCoordinator:
    """Own evidence targeting and post-arrival transaction completion.

    It owns no task answer, relation verdict or physical navigation state.
    """

    @staticmethod
    def _object_lookup(snapshot: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for value in snapshot.get("objects", ()):
            if not isinstance(value, Mapping):
                continue
            if str(value.get("cardinality_role", "")) in {
                "AGGREGATE_COVERAGE", "PARTIAL_FRAGMENT"
            }:
                continue
            try:
                result[int(value["object_id"])] = dict(value)
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def plan(
        self,
        need: EvidenceNeed,
        snapshot: Mapping[str, Any],
    ) -> ObservationIntent:
        lookup = self._object_lookup(snapshot)
        context = need.priority_context
        suggested = context.get("suggested_target")
        target = dict(suggested) if isinstance(suggested, Mapping) else {}
        focus_ids = tuple(dict.fromkeys((*need.target_ids, *need.anchor_ids)))
        if not target:
            target = next(
                (dict(lookup[value]) for value in focus_ids if value in lookup),
                {},
            )
        if not target:
            frontiers = [
                value for value in snapshot.get("observation_frontier_regions", ())
                if isinstance(value, Mapping)
            ]
            if frontiers:
                frontier = max(
                    frontiers,
                    key=lambda value: (
                        float(value.get("information_gain", 0.0) or 0.0),
                        float(value.get("area_m2", 0.0) or 0.0),
                    ),
                )
                target = {
                    "class_label": "observe",
                    "target_kind": "unobserved_world_region",
                    "navigation_target_xy": list(
                        frontier.get("representative_xy", ())
                    ),
                    "frontier_region_id": str(frontier.get("region_id", "")),
                }
            else:
                target = {
                    "class_label": "observe",
                    "target_kind": "current_world_observation",
                }
        target.update({
            "observation_reason": need.reason.value,
            "required_visible_object_ids": list(focus_ids),
            "joint_visibility_required": bool(need.anchor_ids),
            "probe_relation": need.predicate or "",
            "probe_candidate_object_ids": list(need.target_ids),
            "probe_anchor_object_ids": list(need.anchor_ids),
            "evidence_need": need.to_dict(),
        })
        return ObservationIntent(
            evidence_need=need,
            target=target,
            required_visible_object_ids=focus_ids,
            joint_visibility_required=bool(need.anchor_ids),
            provenance=("ResolverResult", "EvidenceAcquisitionCoordinator"),
            metadata={"planner_state": "STATELESS"},
        )

    def validate_transaction(
        self,
        intent: ObservationIntent,
        *,
        navigation_arrived: bool,
        fresh_observation: bool,
        scene_snapshot: Mapping[str, Any],
        transaction: Mapping[str, Any],
    ) -> dict[str, Any]:
        need = intent.evidence_need
        current_transaction = scene_snapshot.get("last_observation_transaction", {})
        if not isinstance(current_transaction, Mapping):
            current_transaction = {}
        associated = {
            int(value)
            for value in current_transaction.get("associated_object_ids", ())
            if str(value).lstrip("-").isdigit()
        }
        visible = set(intent.required_visible_object_ids).issubset(associated)
        revision_progressed = bool(
            _revision(scene_snapshot, "scene_version") > need.scene_version
            or _revision(scene_snapshot, "identity_revision")
            > need.identity_revision
            or _revision(
                scene_snapshot, "geometry_version", "geometry_revision"
            ) > need.geometry_version
            or _revision(scene_snapshot, "relation_revision")
            > need.relation_revision
        )
        relation_consumed = bool(
            not need.predicate
            or _revision(scene_snapshot, "relation_revision")
            > need.relation_revision
        )
        committed = bool(
            transaction.get("scene_memory_committed", True)
            and current_transaction.get("acquisition_id")
            == transaction.get("acquisition_id")
        )
        conditions = {
            "navigation_arrived": bool(navigation_arrived),
            "fresh_observation_after_arrival": bool(fresh_observation),
            "scene_memory_transaction_committed": committed,
            "requested_world_revision_consumed": revision_progressed,
            "required_objects_observed": bool(
                visible or not intent.required_visible_object_ids
            ),
            "requested_relation_recomputed": relation_consumed,
        }
        completed = all(conditions.values())
        return {
            "schema_version": "evidence_transaction_result_v1",
            "completed": completed,
            "conditions": conditions,
            "evidence_need": need.to_dict(),
            "acquisition_id": str(transaction.get("acquisition_id", "")),
            "failure_reasons": [
                key for key, value in conditions.items() if not value
            ],
        }

    @staticmethod
    def intent_from_dict(value: Mapping[str, Any]) -> ObservationIntent:
        need = EvidenceNeed.from_dict(value.get("evidence_need", {}))
        return ObservationIntent(
            evidence_need=need,
            target=dict(value.get("target", {})),
            required_visible_object_ids=tuple(
                int(item)
                for item in value.get("required_visible_object_ids", ())
                if str(item).lstrip("-").isdigit()
            ),
            joint_visibility_required=bool(
                value.get("joint_visibility_required")
            ),
            provenance=tuple(str(item) for item in value.get("provenance", ())),
            metadata=dict(value.get("metadata", {})),
        )
