"""Persistent aggregation for the existing relation evaluator output.

This module stores only relation evidence that is anchored by SceneMemory
object IDs.  It does not infer spatial relations and therefore does not
duplicate the project's geometry or Qwen relation evaluator.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "persistent_relation_evidence_v1"
_STATES = frozenset({"YES", "NO", "UNKNOWN", "INVALID", "CONFLICTED"})
_LOG_ODDS_LIMIT = 8.0


@dataclass
class NormalizedRelationEvidence:
    relation_id: str
    subject_object_id: int
    predicate: str
    anchor_object_ids: list[int]
    state: str
    probability: float
    geometric_margin: float | None
    semantic_support: float | None
    geometry_quality: float
    acquisition_id: str
    station_id: str
    observation_ids: list[str] = field(default_factory=list)
    timestamp: float | None = None
    identity_version: int | None = None
    geometry_version: int | None = None
    reason_code: str = ""

    @property
    def record_key(self) -> str:
        anchors = ",".join(str(value) for value in self.anchor_object_ids)
        return (
            f"{self.relation_id}|subject={self.subject_object_id}"
            f"|anchors={anchors}"
        )


@dataclass
class PersistentRelationRecord:
    relation_id: str
    subject_object_id: int
    predicate: str
    anchor_object_ids: list[int]
    state: str = "UNKNOWN"
    posterior_probability: float = 0.5
    geometric_margin: float | None = None
    semantic_support: float | None = None
    positive_weight: float = 0.0
    negative_weight: float = 0.0
    positive_evidence_count: int = 0
    negative_evidence_count: int = 0
    unknown_evidence_count: int = 0
    conflicted_evidence_count: int = 0
    acquisition_ids: list[str] = field(default_factory=list)
    station_ids: list[str] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)
    first_seen_monotonic: float | None = None
    last_updated_monotonic: float | None = None
    conflict_reason: str | None = None
    positive_station_ids: list[str] = field(default_factory=list)
    negative_station_ids: list[str] = field(default_factory=list)
    positive_quality_max: float = 0.0
    negative_quality_max: float = 0.0
    invalid_evidence_count: int = 0
    evidence_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def record_key(self) -> str:
        anchors = ",".join(str(value) for value in self.anchor_object_ids)
        return (
            f"{self.relation_id}|subject={self.subject_object_id}"
            f"|anchors={anchors}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "subject_object_id": int(self.subject_object_id),
            "predicate": self.predicate,
            "anchor_object_ids": list(self.anchor_object_ids),
            "state": self.state,
            "posterior_probability": float(self.posterior_probability),
            "geometric_margin": self.geometric_margin,
            "semantic_support": self.semantic_support,
            "positive_weight": float(self.positive_weight),
            "negative_weight": float(self.negative_weight),
            "positive_evidence_count": int(self.positive_evidence_count),
            "negative_evidence_count": int(self.negative_evidence_count),
            "unknown_evidence_count": int(self.unknown_evidence_count),
            "conflicted_evidence_count": int(self.conflicted_evidence_count),
            "acquisition_ids": list(self.acquisition_ids),
            "station_ids": list(self.station_ids),
            "observation_ids": list(self.observation_ids),
            "first_seen_monotonic": self.first_seen_monotonic,
            "last_updated_monotonic": self.last_updated_monotonic,
            "conflict_reason": self.conflict_reason,
            "positive_station_ids": list(self.positive_station_ids),
            "negative_station_ids": list(self.negative_station_ids),
            "positive_quality_max": float(self.positive_quality_max),
            "negative_quality_max": float(self.negative_quality_max),
            "invalid_evidence_count": int(self.invalid_evidence_count),
            "evidence_records": [
                dict(value) for value in self.evidence_records
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        relation_id: str | None = None,
    ) -> "PersistentRelationRecord | None":
        try:
            subject_object_id = int(raw.get("subject_object_id"))
            anchor_object_ids = sorted({
                int(value) for value in raw.get("anchor_object_ids", ())
            })
        except (TypeError, ValueError):
            return None
        record = cls(
            relation_id=str(relation_id or raw.get("relation_id", "")).strip(),
            subject_object_id=subject_object_id,
            predicate=str(raw.get("predicate", "")).strip().lower(),
            anchor_object_ids=anchor_object_ids,
            state=(
                "CONFLICTED"
                if str(raw.get("state", "UNKNOWN")).upper() == "CONFLICTED"
                else "UNKNOWN"
            ),
            posterior_probability=_bounded_probability(
                raw.get("posterior_probability", 0.5)
            ),
            geometric_margin=_optional_float(raw.get("geometric_margin")),
            semantic_support=_optional_probability(raw.get("semantic_support")),
            positive_weight=_finite_nonnegative(raw.get("positive_weight", 0.0)),
            negative_weight=_finite_nonnegative(raw.get("negative_weight", 0.0)),
            positive_evidence_count=_nonnegative_int(
                raw.get("positive_evidence_count", 0)
            ),
            negative_evidence_count=_nonnegative_int(
                raw.get("negative_evidence_count", 0)
            ),
            unknown_evidence_count=_nonnegative_int(
                raw.get("unknown_evidence_count", 0)
            ),
            conflicted_evidence_count=_nonnegative_int(
                raw.get("conflicted_evidence_count", 0)
            ),
            acquisition_ids=_ordered_unique(raw.get("acquisition_ids", ())),
            station_ids=_ordered_unique(raw.get("station_ids", ())),
            observation_ids=_ordered_unique(raw.get("observation_ids", ())),
            first_seen_monotonic=_optional_float(raw.get("first_seen_monotonic")),
            last_updated_monotonic=_optional_float(raw.get("last_updated_monotonic")),
            conflict_reason=(
                str(raw.get("conflict_reason"))
                if raw.get("conflict_reason") not in (None, "")
                else None
            ),
            positive_station_ids=_ordered_unique(raw.get("positive_station_ids", ())),
            negative_station_ids=_ordered_unique(raw.get("negative_station_ids", ())),
            positive_quality_max=_bounded_probability(
                raw.get("positive_quality_max", 0.0)
            ),
            negative_quality_max=_bounded_probability(
                raw.get("negative_quality_max", 0.0)
            ),
            invalid_evidence_count=_nonnegative_int(
                raw.get("invalid_evidence_count", 0)
            ),
            evidence_records=[
                dict(value) for value in raw.get("evidence_records", ())
                if isinstance(value, Mapping)
            ][-64:],
        )
        if record.state == "CONFLICTED" and record.conflicted_evidence_count <= 0:
            # Preserve a previously persisted conflict across schema upgrades;
            # it must not be silently downgraded by the next merge.
            record.conflicted_evidence_count = 1
        if (
            subject_object_id < 0
            or any(value < 0 for value in anchor_object_ids)
            or not record.relation_id
            or not record.predicate
            or not anchor_object_ids
        ):
            return None
        return record


def _empty_store() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "records": {},
        "relation_summaries": {},
        "relation_revision": 0,
        "last_updated_monotonic": 0.0,
    }


def _ordered_unique(values: object) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_probability(value: object, default: float = 0.5) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not math.isfinite(result):
        result = default
    return max(0.0, min(1.0, result))


def _optional_probability(value: object) -> float | None:
    if value is None:
        return None
    return _bounded_probability(value)


def _finite_nonnegative(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result >= 0.0 else 0.0


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _clamp(value: object, default: float = 0.5) -> float:
    return _bounded_probability(value, default=default)


def _state(raw: object) -> str:
    value = str(raw or "UNKNOWN").strip().upper()
    return value if value in _STATES else "UNKNOWN"


def _numeric_from(*values: object) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _quality(
    *,
    geometry_quality: float,
    semantic_support: float | None,
    probability: float,
) -> float:
    semantic = 0.5 if semantic_support is None else _clamp(semantic_support)
    # Missing fields intentionally remain neutral.  Geometry and semantic
    # support each contribute, while probability prevents a weak verdict from
    # becoming a strong persistent confirmation.
    return _clamp(
        0.40 * _clamp(geometry_quality)
        + 0.30 * semantic
        + 0.30 * max(probability, 1.0 - probability)
    )


def _relation_probability(
    raw: Mapping[str, Any],
    state: str,
    qwen: Mapping[str, Any],
) -> float:
    # These fields are explicit P(relation=true) fields in the relation
    # contract.  Qwen's ``confidence`` instead expresses confidence in its
    # reported state and is inverted for a NO verdict.
    explicit = _numeric_from(
        raw.get("relation_probability"),
        raw.get("p_true"),
        raw.get("probability"),
        qwen.get("relation_probability"),
        qwen.get("p_true"),
    )
    if explicit is not None:
        return _bounded_probability(explicit)
    confidence = _numeric_from(
        raw.get("confidence"),
        qwen.get("confidence"),
        qwen.get("semantic_support"),
    )
    if confidence is None:
        return 0.75 if state in {"YES", "NO"} else 0.5
    confidence = _bounded_probability(confidence)
    return (
        confidence
        if state == "YES"
        else 1.0 - confidence
        if state == "NO"
        else 0.5
    )


def _geometry_quality(raw: Mapping[str, Any], geometry: Mapping[str, Any]) -> float:
    explicit = _numeric_from(
        raw.get("geometry_quality"),
        geometry.get("geometry_quality"),
        geometry.get("quality"),
    )
    if explicit is not None:
        return _clamp(explicit)
    if geometry.get("answer_authority") is True:
        return 0.90
    if geometry:
        return 0.55
    return 0.50


def normalize_ephemeral_relation_evidence(
    raw: Mapping[str, Any],
    *,
    acquisition_id: str = "",
    station_id: str = "",
) -> NormalizedRelationEvidence | None:
    """Adapt one real ``relation_verifications`` row to the persistent form."""
    if not isinstance(raw, Mapping):
        return None
    if raw.get("persistent_state_inherited") is True:
        # A posterior read is not a new camera observation.  Re-ingesting it
        # under a later acquisition/station would manufacture independent
        # support and make the store reinforce itself without new pixels.
        return None
    relation_id = str(raw.get("relation_id", "")).strip()
    predicate = str(raw.get("predicate", "")).strip().lower()
    if not relation_id or not predicate:
        return None
    try:
        subject_object_id = int(raw.get("subject_object_id"))
        raw_anchor_ids = raw.get("object_ids") or raw.get(
            "anchor_object_ids", ()
        )
        anchor_object_ids = sorted({
            int(value) for value in raw_anchor_ids
        })
    except (TypeError, ValueError):
        return None
    if subject_object_id < 0 or not anchor_object_ids or any(
        value < 0 for value in anchor_object_ids
    ):
        # No stable physical tuple means diagnostic-only evidence.  An
        # acquisition-local observation number is never substituted here.
        return None
    evidence_ids = _ordered_unique(raw.get("evidence_ids", ()))
    evidence_acquisition_ids = {
        value.split(":", 1)[0].strip()
        for value in evidence_ids
        if ":" in value and value.split(":", 1)[0].strip()
    }
    evidence_acquisition_id = (
        next(iter(evidence_acquisition_ids))
        if len(evidence_acquisition_ids) == 1
        else ""
    )
    normalized_acquisition_id = (
        str(raw.get("evidence_acquisition_id", "")).strip()
        or evidence_acquisition_id
        or str(raw.get("acquisition_id", "")).strip()
        or str(acquisition_id).strip()
    )
    normalized_station_id = (
        str(raw.get("evidence_station_id", "")).strip()
        or (
            f"station:{normalized_acquisition_id}"
            if evidence_acquisition_id
            else ""
        )
        or str(raw.get("station_id", "")).strip()
        or str(station_id).strip()
    )
    if not normalized_acquisition_id or not normalized_station_id:
        # Without both provenance keys we cannot decide whether a later row is
        # an independent viewpoint.  Persisting a synthetic "unknown" key
        # would make unrelated acquisitions look like duplicates.
        return None
    state = _state(raw.get("state", raw.get("path_state", "UNKNOWN")))
    geometry = raw.get("geometry", {})
    geometry = geometry if isinstance(geometry, Mapping) else {}
    qwen = raw.get("qwen", {})
    qwen = qwen if isinstance(qwen, Mapping) else {}
    observation_ids = _ordered_unique([
        *(
            str(value) for value in raw.get("source_observation_ids", ())
        ),
        *(
            str(value) for value in raw.get("observation_ids", ())
        ),
        *(
            str(value) for value in raw.get("evidence_ids", ())
        ),
    ])
    timestamp = _optional_float(raw.get("timestamp"))
    try:
        identity_version = (
            int(raw.get("identity_version"))
            if raw.get("identity_version") is not None else None
        )
        geometry_version = (
            int(raw.get("geometry_version"))
            if raw.get("geometry_version") is not None else None
        )
    except (TypeError, ValueError):
        identity_version = None
        geometry_version = None
    return NormalizedRelationEvidence(
        relation_id=relation_id,
        subject_object_id=subject_object_id,
        predicate=predicate,
        anchor_object_ids=anchor_object_ids,
        state=state,
        probability=_relation_probability(raw, state, qwen),
        geometric_margin=_numeric_from(
            raw.get("geometric_margin"),
            geometry.get("geometric_margin"),
        ),
        semantic_support=_numeric_from(
            raw.get("semantic_support"),
            qwen.get("semantic_support"),
            qwen.get("confidence") if state in {"YES", "NO"} else None,
        ),
        geometry_quality=_geometry_quality(raw, geometry),
        acquisition_id=normalized_acquisition_id,
        station_id=normalized_station_id,
        observation_ids=observation_ids,
        timestamp=timestamp,
        identity_version=identity_version,
        geometry_version=geometry_version,
        reason_code=str(raw.get("reason_code", "")).strip(),
    )


def _load_store_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return _empty_store()
    if str(raw.get("schema_version", "")) not in {"", SCHEMA_VERSION}:
        return _empty_store()
    store = _empty_store()
    store.update(dict(raw))
    store["schema_version"] = SCHEMA_VERSION
    store["records"] = (
        dict(store["records"])
        if isinstance(store.get("records"), Mapping)
        else {}
    )
    try:
        store["relation_revision"] = max(0, int(store.get("relation_revision", 0)))
    except (TypeError, ValueError):
        store["relation_revision"] = 0
    return store


def _record_mapping(
    store: Mapping[str, Any],
) -> dict[str, PersistentRelationRecord]:
    result: dict[str, PersistentRelationRecord] = {}
    raw_records = store.get("records", {})
    if not isinstance(raw_records, Mapping):
        return result
    for key, raw in raw_records.items():
        if not isinstance(raw, Mapping):
            continue
        record = PersistentRelationRecord.from_mapping(raw)
        if record is not None:
            result[str(key)] = record
    return result


def _store_from_records(
    store: Mapping[str, Any],
    records: Mapping[str, PersistentRelationRecord],
) -> dict[str, Any]:
    result = _load_store_payload(store)
    result["records"] = {
        key: record.to_dict() for key, record in sorted(records.items())
    }
    return result


def _merge_quality_candidate(
    candidates: dict[str, tuple[NormalizedRelationEvidence, float]],
    evidence: NormalizedRelationEvidence,
) -> None:
    quality = _quality(
        geometry_quality=evidence.geometry_quality,
        semantic_support=evidence.semantic_support,
        probability=evidence.probability,
    )
    previous = candidates.get(evidence.record_key)
    if (
        previous is None
        or (
            evidence.state == "CONFLICTED"
            and previous[0].state != "CONFLICTED"
        )
        or (
            evidence.state == "INVALID"
            and previous[0].state not in {"INVALID", "CONFLICTED"}
        )
        or quality > previous[1]
    ):
        candidates[evidence.record_key] = (evidence, quality)


def _logit(probability: float) -> float:
    p = max(0.01, min(0.99, float(probability)))
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    bounded = max(-_LOG_ODDS_LIMIT, min(_LOG_ODDS_LIMIT, float(value)))
    return 1.0 / (1.0 + math.exp(-bounded))


def _update_state(record: PersistentRelationRecord) -> None:
    """Update evidence statistics without issuing a relation verdict."""
    if record.conflicted_evidence_count > 0:
        record.state = "CONFLICTED"
        record.conflict_reason = "explicit_conflicted_evidence"
        return
    posterior = _sigmoid(record.positive_weight - record.negative_weight)
    record.posterior_probability = posterior
    positive_stations = set(record.positive_station_ids)
    negative_stations = set(record.negative_station_ids)
    separated_conflict = (
        record.positive_quality_max >= 0.80
        and record.negative_quality_max >= 0.80
        and bool(positive_stations)
        and bool(negative_stations)
        and positive_stations.isdisjoint(negative_stations)
        and 0.0 < posterior < 1.0
    )
    if separated_conflict:
        record.state = "CONFLICTED"
        record.conflict_reason = "high_quality_opposite_stations"
        return
    record.conflict_reason = None
    # Persistence stores observations and posterior features only.
    # RelationEngine is the sole module that may convert them to YES/NO.
    record.state = "UNKNOWN"


def merge_persistent_relation_evidence(
    *,
    store: Mapping[str, Any],
    ephemeral_evidence: Sequence[Mapping[str, Any]] | None,
    task_ir: Mapping[str, Any] | None = None,
    acquisition_id: str = "",
    station_id: str = "",
    now_monotonic: float | None = None,
) -> dict[str, Any]:
    """Merge one acquisition, deduplicating exact tuples within that acquisition."""
    result = _load_store_payload(copy.deepcopy(dict(store)))
    records = _record_mapping(result)
    candidates: dict[str, tuple[NormalizedRelationEvidence, float]] = {}
    for raw in ephemeral_evidence or ():
        if not isinstance(raw, Mapping):
            continue
        evidence = normalize_ephemeral_relation_evidence(
            raw,
            acquisition_id=acquisition_id,
            station_id=station_id,
        )
        if evidence is not None:
            _merge_quality_candidate(candidates, evidence)
    now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    accepted_evidence_count = 0
    for key, (evidence, quality) in candidates.items():
        record = records.get(key)
        if record is None:
            record = PersistentRelationRecord(
                relation_id=evidence.relation_id,
                subject_object_id=evidence.subject_object_id,
                predicate=evidence.predicate,
                anchor_object_ids=list(evidence.anchor_object_ids),
            )
            records[key] = record
        # Re-merging a tuple from the same acquisition is not an independent
        # viewpoint.  The within-acquisition quality winner was selected above.
        if evidence.acquisition_id in record.acquisition_ids:
            continue
        accepted_evidence_count += 1
        record.acquisition_ids.append(evidence.acquisition_id)
        if evidence.station_id not in record.station_ids:
            record.station_ids.append(evidence.station_id)
        record.observation_ids = list(dict.fromkeys([
            *record.observation_ids,
            *evidence.observation_ids,
        ]))
        record.evidence_records = [
            *record.evidence_records,
            {
                "state": evidence.state,
                "acquisition_id": evidence.acquisition_id,
                "station_id": evidence.station_id,
                "timestamp": evidence.timestamp,
                "source_observation_ids": list(evidence.observation_ids),
                "identity_version": evidence.identity_version,
                "geometry_version": evidence.geometry_version,
                "reason_code": evidence.reason_code,
            },
        ][-64:]
        if record.first_seen_monotonic is None:
            record.first_seen_monotonic = now
        record.last_updated_monotonic = now
        record.geometric_margin = evidence.geometric_margin
        record.semantic_support = evidence.semantic_support
        if evidence.state == "UNKNOWN":
            record.unknown_evidence_count += 1
            continue
        if evidence.state == "INVALID":
            record.invalid_evidence_count += 1
            record.state = "INVALID"
            record.conflict_reason = "invalid_relation_evidence_requires_recompute"
            continue
        if evidence.state == "CONFLICTED":
            record.conflicted_evidence_count += 1
            record.conflict_reason = "explicit_conflicted_evidence"
            record.state = "CONFLICTED"
            continue
        contribution = abs(_logit(evidence.probability)) * quality
        if evidence.probability >= 0.5:
            record.positive_weight = min(
                _LOG_ODDS_LIMIT,
                record.positive_weight + contribution,
            )
            record.positive_evidence_count += 1
            record.positive_quality_max = max(
                record.positive_quality_max, quality
            )
            if evidence.station_id not in record.positive_station_ids:
                record.positive_station_ids.append(evidence.station_id)
        else:
            record.negative_weight = min(
                _LOG_ODDS_LIMIT,
                record.negative_weight + contribution,
            )
            record.negative_evidence_count += 1
            record.negative_quality_max = max(
                record.negative_quality_max, quality
            )
            if evidence.station_id not in record.negative_station_ids:
                record.negative_station_ids.append(evidence.station_id)
        _update_state(record)
    result = _store_from_records(result, records)
    if accepted_evidence_count:
        result["relation_revision"] = int(result.get("relation_revision", 0)) + 1
    result["last_updated_monotonic"] = now
    if task_ir is not None:
        result["relation_summaries"] = summarize_persistent_relation_evidence(
            result, task_ir
        ).get("relations", {})
    return result


def _summary_state(records: Sequence[PersistentRelationRecord]) -> str:
    if any(record.state == "INVALID" for record in records):
        return "INVALID"
    if any(record.state == "YES" for record in records):
        return "YES"
    if any(record.state == "CONFLICTED" for record in records):
        return "CONFLICTED"
    if any(record.state == "UNKNOWN" for record in records):
        return "UNKNOWN"
    return "NO" if records else "UNKNOWN"


def summarize_persistent_relation_evidence(
    store: Mapping[str, Any],
    task_ir: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose tuple-level states for scheduling and query resolution."""
    records = list(_record_mapping(store).values())
    by_relation: dict[str, list[PersistentRelationRecord]] = {}
    for record in records:
        by_relation.setdefault(record.relation_id, []).append(record)
    relations_ir = {
        str(value.get("id", "")): value
        for value in task_ir.get("relations", ())
        if isinstance(value, Mapping) and str(value.get("id", "")).strip()
    }
    summaries: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    invalid: list[str] = []
    conflicted: list[str] = []
    for relation_id, relation in relations_ir.items():
        relation_records = sorted(
            by_relation.get(relation_id, ()),
            key=lambda value: (
                value.posterior_probability,
                -value.subject_object_id,
            ),
            reverse=True,
        )
        verified = sorted({
            record.subject_object_id
            for record in relation_records if record.state == "YES"
        })
        rejected = sorted({
            record.subject_object_id
            for record in relation_records if record.state == "NO"
        })
        unknown = sorted({
            record.subject_object_id
            for record in relation_records if record.state == "UNKNOWN"
        })
        conflicted_ids = sorted({
            record.subject_object_id
            for record in relation_records if record.state == "CONFLICTED"
        })
        invalid_ids = sorted({
            record.subject_object_id
            for record in relation_records if record.state == "INVALID"
        })
        state = _summary_state(relation_records)
        if state == "UNKNOWN":
            unresolved.append(relation_id)
        if state == "CONFLICTED":
            conflicted.append(relation_id)
        if state == "INVALID":
            invalid.append(relation_id)
        dependencies = [str(value) for value in relation.get("depends_on", ())]
        qualified_anchor_ids = sorted({
            subject_id
            for dependency_id in dependencies
            for dependency_record in by_relation.get(dependency_id, ())
            if dependency_record.state == "YES"
            for subject_id in (dependency_record.subject_object_id,)
        })
        summaries[relation_id] = {
            "relation_id": relation_id,
            "predicate": str(relation.get("predicate", "")).strip().lower(),
            "subject_entity": str(relation.get("subject_entity", "")),
            "object_entities": [
                str(value) for value in relation.get("object_entities", ())
            ],
            "depends_on": dependencies,
            "state": state,
            "verified_subject_object_ids": verified,
            "rejected_subject_object_ids": rejected,
            "unknown_subject_object_ids": unknown,
            "conflicted_subject_object_ids": conflicted_ids,
            "invalid_subject_object_ids": invalid_ids,
            "best_subject_object_id": (
                relation_records[0].subject_object_id
                if relation_records else None
            ),
            "best_posterior": (
                relation_records[0].posterior_probability
                if relation_records else 0.0
            ),
            "qualified_anchor_object_ids": qualified_anchor_ids,
            "candidate_count": len(relation_records),
            "candidate_records": [record.to_dict() for record in relation_records],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "relations": summaries,
        "unresolved_relation_ids": sorted(set(unresolved)),
        "conflicted_relation_ids": sorted(set(conflicted)),
        "invalid_relation_ids": sorted(set(invalid)),
    }


__all__ = [
    "NormalizedRelationEvidence",
    "PersistentRelationRecord",
    "merge_persistent_relation_evidence",
    "normalize_ephemeral_relation_evidence",
    "summarize_persistent_relation_evidence",
]
