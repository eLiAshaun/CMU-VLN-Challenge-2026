"""Provider-neutral VQA contracts; no provider is selected in this release."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol


class VQAVerdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VQARequest:
    request_id: str
    episode_id: str
    acquisition_id: str
    operation: str
    subject_object_id: int
    anchor_object_ids: tuple[int, ...]
    relation: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.request_id,
                self.episode_id,
                self.acquisition_id,
                self.operation,
                self.relation,
            )
        ):
            raise ValueError("VQA request identity fields must be non-empty")
        if self.subject_object_id < 0 or any(
            value < 0 for value in self.anchor_object_ids
        ):
            raise ValueError("VQA object IDs must be non-negative")


@dataclass(frozen=True)
class VQAResult:
    verdict: VQAVerdict
    confidence: float
    provider: str
    model: str
    error_code: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("VQA confidence must be finite and in [0, 1]")
        if not self.provider.strip():
            raise ValueError("VQA result provider must be explicit")


@dataclass(frozen=True)
class ObjectGroundedRelationRequest:
    request_id: str
    episode_id: str
    query_node_id: str
    world_snapshot_version: int
    acquisition_id: str
    subject_id: int
    subject_instance_version: int
    object_ids: tuple[int, ...]
    object_instance_versions: tuple[int, ...]
    predicate: str
    parameter_roles: tuple[str, ...]
    subject_description: str
    object_descriptions: tuple[str, ...]
    evidence_policy: str
    negative_evidence_policy: str
    verification_instruction: str
    geometry_diagnostic: Mapping[str, object]
    subject_bbox_or_mask: Mapping[str, object]
    object_bboxes_or_masks: tuple[Mapping[str, object], ...]
    camera_pose: tuple[float, ...]
    evidence_provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            not str(value).strip()
            for value in (
                self.request_id,
                self.episode_id,
                self.query_node_id,
                self.acquisition_id,
                self.predicate,
                self.subject_description,
                self.evidence_policy,
                self.negative_evidence_policy,
                self.verification_instruction,
            )
        ):
            raise ValueError("grounded Qwen request identity is incomplete")
        if self.world_snapshot_version < 0 or self.subject_id < 0:
            raise ValueError("grounded Qwen request IDs must be nonnegative")
        if not self.object_ids or any(value < 0 for value in self.object_ids):
            raise ValueError("grounded Qwen request requires object IDs")
        if len(self.object_ids) != len(self.object_instance_versions):
            raise ValueError("Qwen object ID/version lengths mismatch")
        if len(self.object_ids) != len(self.object_bboxes_or_masks):
            raise ValueError("Qwen object ID/grounding lengths mismatch")
        if len(self.object_ids) != len(self.object_descriptions):
            raise ValueError("Qwen object ID/description lengths mismatch")
        if len(self.parameter_roles) != 1 + len(self.object_ids):
            raise ValueError("Qwen relation parameter-role lengths mismatch")
        if any(not str(value).strip() for value in self.parameter_roles):
            raise ValueError("Qwen relation parameter roles must be explicit")
        if self.predicate.upper() == "BETWEEN" and len(self.object_ids) != 2:
            raise ValueError("Qwen BETWEEN requires two anchor IDs")
        if not isinstance(self.geometry_diagnostic, Mapping):
            raise ValueError("Qwen relation geometry diagnostic must be a mapping")

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "query_node_id": self.query_node_id,
            "world_snapshot_version": self.world_snapshot_version,
            "acquisition_id": self.acquisition_id,
            "subject_id": self.subject_id,
            "subject_instance_version": self.subject_instance_version,
            "object_ids": list(self.object_ids),
            "object_instance_versions": list(
                self.object_instance_versions
            ),
            "predicate": self.predicate.upper(),
            "parameter_roles": list(self.parameter_roles),
            "subject_description": self.subject_description,
            "object_descriptions": list(self.object_descriptions),
            "evidence_policy": self.evidence_policy,
            "negative_evidence_policy": self.negative_evidence_policy,
            "verification_instruction": self.verification_instruction,
            "geometry_diagnostic": dict(self.geometry_diagnostic),
            "subject_bbox_or_mask": dict(self.subject_bbox_or_mask),
            "object_bboxes_or_masks": [
                dict(value) for value in self.object_bboxes_or_masks
            ],
            "camera_pose": list(self.camera_pose),
            "evidence_provenance": list(self.evidence_provenance),
        }


@dataclass(frozen=True)
class ObjectGroundedRelationResult:
    subject_id: int
    predicate: str
    object_ids: tuple[int, ...]
    subject_role_state: str
    object_role_states: tuple[str, ...]
    state: str
    confidence: float
    visible_subject: bool
    visible_objects: tuple[bool, ...]
    jointly_observable: bool
    occlusion: str
    reason_code: str

    @classmethod
    def from_json(
        cls,
        payload,
        *,
        expected: ObjectGroundedRelationRequest | None = None,
    ) -> "ObjectGroundedRelationResult":
        required = {
            "subject_id",
            "predicate",
            "object_ids",
            "subject_role_state",
            "object_role_states",
            "state",
            "confidence",
            "visible_subject",
            "visible_objects",
            "jointly_observable",
            "occlusion",
            "reason_code",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("qwen_relation_json_schema_invalid")
        object_ids = tuple(int(item) for item in payload["object_ids"])
        raw_object_role_states = payload["object_role_states"]
        if not isinstance(raw_object_role_states, (list, tuple)):
            raise ValueError("qwen_relation_object_role_states_type_invalid")
        raw_visible_objects = payload["visible_objects"]
        if isinstance(raw_visible_objects, bool):
            if len(object_ids) != 1:
                raise ValueError("qwen_relation_visibility_length_invalid")
            visible_objects = (raw_visible_objects,)
        elif isinstance(raw_visible_objects, (list, tuple)):
            visible_objects = tuple(bool(item) for item in raw_visible_objects)
        else:
            raise ValueError("qwen_relation_visibility_type_invalid")
        value = cls(
            subject_id=int(payload["subject_id"]),
            predicate=str(payload["predicate"]).upper(),
            object_ids=object_ids,
            subject_role_state=str(payload["subject_role_state"]).upper(),
            object_role_states=tuple(
                str(item).upper() for item in raw_object_role_states
            ),
            state=str(payload["state"]).lower(),
            confidence=float(payload["confidence"]),
            visible_subject=bool(payload["visible_subject"]),
            visible_objects=visible_objects,
            jointly_observable=bool(payload["jointly_observable"]),
            occlusion=str(payload["occlusion"]).lower(),
            reason_code=str(payload["reason_code"]),
        )
        value._validate(expected)
        return value

    def _validate(
        self, expected: ObjectGroundedRelationRequest | None
    ) -> None:
        if self.state not in {"supported", "refuted", "uncertain"}:
            raise ValueError("qwen_relation_state_invalid")
        if self.subject_role_state not in {"YES", "NO", "UNKNOWN"}:
            raise ValueError("qwen_relation_subject_role_state_invalid")
        if (
            len(self.object_role_states) != len(self.object_ids)
            or any(
                value not in {"YES", "NO", "UNKNOWN"}
                for value in self.object_role_states
            )
        ):
            raise ValueError("qwen_relation_object_role_states_invalid")
        if self.occlusion not in {"none", "partial", "severe"}:
            raise ValueError("qwen_relation_occlusion_invalid")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("qwen_relation_confidence_invalid")
        if len(self.object_ids) != len(self.visible_objects):
            raise ValueError("qwen_relation_visibility_length_invalid")
        if expected is not None and (
            self.subject_id != expected.subject_id
            or self.predicate != expected.predicate.upper()
            or self.object_ids != expected.object_ids
        ):
            raise ValueError("qwen_relation_object_id_mismatch")


class VisualQuestionVerifier(Protocol):
    def verify(self, request: VQARequest) -> VQAResult:
        """Return supplemental evidence without controlling deterministic gates."""
        ...
