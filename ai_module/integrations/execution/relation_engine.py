"""The sole module allowed to turn spatial evidence into relation verdicts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from integrations.execution.count_query_executor import RelationEvidenceProvider


RELATION_STATES = frozenset({"YES", "NO", "UNKNOWN", "INVALID"})


def _object_id(value: Mapping[str, Any]) -> int:
    try:
        return int(value.get("object_id", -1))
    except (TypeError, ValueError):
        return -1


class RelationEngine:
    """Fuse live, geometric and persistent evidence behind one verdict API.

    Providers may calculate features or propose categorical observations.  No
    caller is allowed to reinterpret those observations into another final
    relation state after this class returns.
    """

    def __init__(
        self,
        live_evaluator=None,
        persistent_relation_summary: Mapping[str, Any] | None = None,
        object_id_aliases: Mapping[str, Any] | None = None,
        *,
        identity_version: int | None = None,
        geometry_version: int | None = None,
    ) -> None:
        self._provider = RelationEvidenceProvider(
            live_evaluator,
            persistent_relation_summary,
            object_id_aliases,
            identity_version=identity_version,
            geometry_version=geometry_version,
        )
        self._cache: dict[tuple[str, int, tuple[int, ...]], dict[str, Any]] = {}

    @staticmethod
    def _key(
        relation: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> tuple[str, int, tuple[int, ...]]:
        return (
            str(relation.get("id", relation.get("predicate", ""))),
            _object_id(subject),
            tuple(sorted(_object_id(value) for value in objects)),
        )

    @staticmethod
    def _canonical_verdict(value: Mapping[str, Any] | None) -> dict[str, Any]:
        result = dict(value or {})
        raw_state = str(result.get("state", "UNKNOWN")).upper()
        persistent = result.get("persistent_evidence")
        persistent = persistent if isinstance(persistent, Mapping) else result
        observations = {
            str(item).upper()
            for item in persistent.get("persistent_observation_states", ())
        }
        if raw_state == "UNKNOWN" and observations:
            if "INVALID" in observations:
                raw_state = "INVALID"
                result["reason_code"] = "persistent_observation_invalid"
            elif "CONFLICTED" in observations or (
                "YES" in observations and "NO" in observations
            ):
                raw_state = "UNKNOWN"
                result["reason_code"] = "persistent_observations_conflicted"
            elif "YES" in observations:
                raw_state = "YES"
                result["reason_code"] = "persistent_observations_support_yes"
            elif "NO" in observations:
                raw_state = "NO"
                result["reason_code"] = "persistent_observations_support_no"
        if raw_state == "CONFLICTED":
            raw_state = "UNKNOWN"
            result.setdefault("reason_code", "relation_evidence_conflicted")
        if raw_state not in RELATION_STATES:
            raw_state = "UNKNOWN"
            result.setdefault("reason_code", "relation_evidence_state_invalid")
        result["state"] = raw_state
        result["verdict_authority"] = "RelationEngine"
        result.setdefault("evidence_ids", [])
        result.setdefault("geometry", {})
        result.setdefault("qwen", {})
        return result

    def evaluate(
        self,
        relation: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        key = self._key(relation, subject, objects)
        cached = self._cache.get(key)
        if cached is not None:
            return dict(cached)
        result = self._canonical_verdict(
            self._provider(relation, subject, objects)
        )
        self._cache[key] = dict(result)
        return result

    def __call__(
        self,
        relation: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self.evaluate(relation, subject, objects)

    def evaluate_selector(
        self,
        relation: Mapping[str, Any],
        subject: Mapping[str, Any],
        anchor: Mapping[str, Any],
        *,
        domain_complete: bool,
        distance_m: float,
    ) -> dict[str, Any]:
        """Issue the canonical verdict for a metric selector winner."""
        predicate = str(relation.get("predicate", "")).lower()
        try:
            distance = float(distance_m)
        except (TypeError, ValueError):
            distance = float("nan")
        state = (
            "YES"
            if predicate in {"closest", "farthest"}
            and domain_complete
            and _object_id(subject) >= 0
            and _object_id(anchor) >= 0
            and math.isfinite(distance)
            and distance >= 0.0
            else "INVALID"
            if predicate not in {"closest", "farthest"} or not math.isfinite(distance)
            else "UNKNOWN"
        )
        return self._canonical_verdict({
            "state": state,
            "reason_code": (
                f"metric_{predicate}_resolved"
                if state == "YES"
                else "metric_selector_provisional"
                if state == "UNKNOWN"
                else "metric_selector_invalid"
            ),
            "evidence_ids": [],
            "geometry": {"distance_m": distance},
            "qwen": {},
        })

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
        outputs: list[dict[str, Any] | None] = [None] * len(items)
        pending_items = []
        pending_indexes = []
        for index, item in enumerate(items):
            cached = self._cache.get(self._key(*item))
            if cached is not None:
                outputs[index] = dict(cached)
            else:
                pending_indexes.append(index)
                pending_items.append(item)
        if pending_items:
            provided = self._provider.verify_many(pending_items)
            for index, item, raw in zip(pending_indexes, pending_items, provided):
                verdict = self._canonical_verdict(raw)
                self._cache[self._key(*item)] = dict(verdict)
                outputs[index] = verdict
        return [
            dict(value or self._canonical_verdict(None)) for value in outputs
        ]

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "verdict_authority": "RelationEngine",
            "cached_tuple_count": len(self._cache),
            "live_verification_count": int(
                self._provider.live_verification_count
            ),
            "persistent_evidence_consumption_count": int(
                self._provider.persistent_evidence_use_count
            ),
            "stale_persistent_record_count": int(
                self._provider.stale_persistent_record_count
            ),
        }
