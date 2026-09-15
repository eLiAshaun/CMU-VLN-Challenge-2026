"""Final semantic-control contracts shared by every challenge task."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ResolverStatus(str, Enum):
    FINALIZABLE = "FINALIZABLE"
    NEED_EVIDENCE = "NEED_EVIDENCE"
    NEED_EXECUTION = "NEED_EXECUTION"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"


class EvidenceReason(str, Enum):
    DISCOVER_OBJECT = "DISCOVER_OBJECT"
    VERIFY_SEMANTIC = "VERIFY_SEMANTIC"
    LOCALIZE_ENTITY = "LOCALIZE_ENTITY"
    IDENTITY_DISAMBIGUATION = "IDENTITY_DISAMBIGUATION"
    REFINE_GEOMETRY = "REFINE_GEOMETRY"
    RELATION_EVIDENCE = "RELATION_EVIDENCE"


def _unique_ints(values: Sequence[object]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item >= 0 and item not in result:
            result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class EvidenceNeed:
    query_key: str
    task_type: str
    reason: EvidenceReason
    target_ids: tuple[int, ...] = ()
    anchor_ids: tuple[int, ...] = ()
    predicate: str | None = None
    required_observability: dict[str, Any] = field(default_factory=dict)
    scene_version: int = 0
    identity_revision: int = 0
    geometry_version: int = 0
    relation_revision: int = 0
    priority_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.query_key).strip():
            raise ValueError("evidence_need_query_key_empty")
        if not str(self.task_type).strip():
            raise ValueError("evidence_need_task_type_empty")
        object.__setattr__(self, "target_ids", _unique_ints(self.target_ids))
        object.__setattr__(self, "anchor_ids", _unique_ints(self.anchor_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "evidence_need_v1",
            "query_key": self.query_key,
            "task_type": self.task_type,
            "reason": self.reason.value,
            "target_ids": list(self.target_ids),
            "anchor_ids": list(self.anchor_ids),
            "predicate": self.predicate,
            "required_observability": dict(self.required_observability),
            "scene_version": int(self.scene_version),
            "identity_revision": int(self.identity_revision),
            "geometry_version": int(self.geometry_version),
            "relation_revision": int(self.relation_revision),
            "priority_context": dict(self.priority_context),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceNeed":
        return cls(
            query_key=str(value.get("query_key", "")),
            task_type=str(value.get("task_type", "")),
            reason=EvidenceReason(str(value.get("reason", "DISCOVER_OBJECT"))),
            target_ids=tuple(value.get("target_ids", ())),
            anchor_ids=tuple(value.get("anchor_ids", ())),
            predicate=(
                str(value["predicate"])
                if value.get("predicate") is not None else None
            ),
            required_observability=dict(value.get("required_observability", {})),
            scene_version=int(value.get("scene_version", 0)),
            identity_revision=int(value.get("identity_revision", 0)),
            geometry_version=int(value.get("geometry_version", 0)),
            relation_revision=int(value.get("relation_revision", 0)),
            priority_context=dict(value.get("priority_context", {})),
        )


@dataclass(frozen=True)
class ExecutionNeed:
    task_type: str
    intent: str
    directives: tuple[dict[str, Any], ...]
    current_step_index: int
    provenance: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "execution_need_v1",
            "task_type": self.task_type,
            "intent": self.intent,
            "directives": [dict(value) for value in self.directives],
            "current_step_index": int(self.current_step_index),
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class ResolverResult:
    task_type: str
    status: ResolverStatus
    scene_version: int
    identity_revision: int
    geometry_version: int
    relation_revision: int
    final_payload: dict[str, Any] | None = None
    evidence_need: EvidenceNeed | None = None
    execution_need: ExecutionNeed | None = None
    provenance: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        populated = sum(
            value is not None
            for value in (
                self.final_payload,
                self.evidence_need,
                self.execution_need,
            )
        )
        expected = 0 if self.status is ResolverStatus.SYSTEM_FAILURE else 1
        if populated != expected:
            raise ValueError("resolver_result_payload_status_mismatch")
        if self.status is ResolverStatus.FINALIZABLE and self.final_payload is None:
            raise ValueError("resolver_result_final_payload_missing")
        if self.status is ResolverStatus.NEED_EVIDENCE and self.evidence_need is None:
            raise ValueError("resolver_result_evidence_need_missing")
        if self.status is ResolverStatus.NEED_EXECUTION and self.execution_need is None:
            raise ValueError("resolver_result_execution_need_missing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "resolver_result_v1",
            "task_type": self.task_type,
            "status": self.status.value,
            "final_payload": (
                dict(self.final_payload) if self.final_payload is not None else None
            ),
            "evidence_need": (
                self.evidence_need.to_dict()
                if self.evidence_need is not None else None
            ),
            "execution_need": (
                self.execution_need.to_dict()
                if self.execution_need is not None else None
            ),
            "provenance": list(self.provenance),
            "scene_version": int(self.scene_version),
            "identity_revision": int(self.identity_revision),
            "geometry_version": int(self.geometry_version),
            "relation_revision": int(self.relation_revision),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolverResult":
        evidence = value.get("evidence_need")
        execution = value.get("execution_need")
        return cls(
            task_type=str(value.get("task_type", "")),
            status=ResolverStatus(str(value.get("status", "SYSTEM_FAILURE"))),
            final_payload=(
                dict(value["final_payload"])
                if isinstance(value.get("final_payload"), Mapping) else None
            ),
            evidence_need=(
                EvidenceNeed.from_dict(evidence)
                if isinstance(evidence, Mapping) else None
            ),
            execution_need=(
                ExecutionNeed(
                    task_type=str(execution.get("task_type", "")),
                    intent=str(execution.get("intent", "")),
                    directives=tuple(
                        dict(item) for item in execution.get("directives", ())
                        if isinstance(item, Mapping)
                    ),
                    current_step_index=int(execution.get("current_step_index", 0)),
                    provenance=tuple(
                        str(item) for item in execution.get("provenance", ())
                    ),
                )
                if isinstance(execution, Mapping) else None
            ),
            provenance=tuple(str(item) for item in value.get("provenance", ())),
            scene_version=int(value.get("scene_version", 0)),
            identity_revision=int(value.get("identity_revision", 0)),
            geometry_version=int(value.get("geometry_version", 0)),
            relation_revision=int(value.get("relation_revision", 0)),
            diagnostics=dict(value.get("diagnostics", {})),
        )
