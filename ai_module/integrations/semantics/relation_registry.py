"""Canonical directed-relation contracts for semantic query execution.

The registry describes language and evidence roles.  It deliberately does not
contain answer thresholds: geometry and Qwen produce evidence, while the count
query graph owns variable binding and completion semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence


class OperatorKind(str, Enum):
    """Execution semantics, independent from the surface predicate name."""

    TRAJECTORY_ACTION = "trajectory_action"
    RELATION_FILTER = "relation_filter"
    SET_SELECTOR = "set_selector"


@dataclass(frozen=True)
class RelationSpec:
    predicate: str
    operator: str
    object_arity: int
    directed: bool
    symmetric: bool
    inverse_predicate: str | None
    subject_role: str
    object_roles: tuple[str, ...]
    evidence_roles: tuple[str, ...]
    evidence_policy: str
    geometry_policy: str
    negative_evidence_policy: str
    qwen_instruction: str
    operator_kind: OperatorKind
    candidate_scope: str
    requires_relation_roi_provenance: bool
    requires_qwen_tuple_verification: bool

    @property
    def parameter_roles(self) -> tuple[str, ...]:
        return (self.subject_role, *self.object_roles)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["operator_kind"] = self.operator_kind.value
        value["object_roles"] = list(self.object_roles)
        value["evidence_roles"] = list(self.evidence_roles)
        value["parameter_roles"] = list(self.parameter_roles)
        return value


def _binary(
    predicate: str,
    *,
    operator: str = "FILTER",
    directed: bool = True,
    symmetric: bool = False,
    inverse: str | None = None,
    subject_role: str,
    object_role: str,
    evidence_policy: str,
    geometry_policy: str,
    negative_policy: str,
    qwen_instruction: str,
) -> RelationSpec:
    operator_kind = (
        OperatorKind.SET_SELECTOR
        if operator in {"ARGMIN", "ARGMAX"}
        else OperatorKind.RELATION_FILTER
    )
    return RelationSpec(
        predicate=predicate,
        operator=operator,
        object_arity=1,
        directed=directed,
        symmetric=symmetric,
        inverse_predicate=inverse,
        subject_role=subject_role,
        object_roles=(object_role,),
        evidence_roles=("subject", "objects", "joint_context"),
        evidence_policy=evidence_policy,
        geometry_policy=geometry_policy,
        negative_evidence_policy=negative_policy,
        qwen_instruction=qwen_instruction,
        operator_kind=operator_kind,
        candidate_scope="persistent_global",
        requires_relation_roi_provenance=False,
        requires_qwen_tuple_verification=(
            operator_kind is OperatorKind.RELATION_FILTER
        ),
    )


_REGISTRY = {
    "on": _binary(
        "on",
        subject_role="supported_entity",
        object_role="support_surface",
        evidence_policy="surface_subject_object_and_joint_context",
        geometry_policy="support_surface_contact_with_horizontal_overlap",
        negative_policy="refute_only_when_both_entities_and_the_relevant_surface_are_jointly_visible",
        qwen_instruction=(
            "Verify that the subject is physically supported by or attached to the "
            "specified object surface; proximity alone is not ON."
        ),
    ),
    "above": _binary(
        "above",
        inverse="below",
        subject_role="upper_entity",
        object_role="lower_reference",
        evidence_policy="subject_object_and_wide_vertical_context",
        geometry_policy="gravity_aligned_vertical_and_horizontal_projection_evidence",
        negative_policy="refute only from joint visibility with contradictory vertical order",
        qwen_instruction=(
            "Evaluate ABOVE(subject, object), never ABOVE(object, subject). "
            "Supported means the grounded subject is above the grounded object "
            "in the scene. Reconcile the visible order with the supplied 3D "
            "gravity-axis direction and horizontal-projection evidence; no "
            "single diagnostic field has answer authority."
        ),
    ),
    "below": _binary(
        "below",
        inverse="above",
        subject_role="lower_entity",
        object_role="upper_reference",
        evidence_policy="subject_object_and_wide_vertical_context",
        geometry_policy="gravity_aligned_vertical_and_horizontal_projection_evidence",
        negative_policy="refute only from joint visibility with contradictory vertical order",
        qwen_instruction=(
            "Evaluate BELOW(subject, object), never BELOW(object, subject). "
            "Supported means the grounded subject is below the grounded object "
            "in the scene. Reconcile the visible order with the supplied 3D "
            "gravity-axis direction and horizontal-projection evidence; no "
            "single diagnostic field has answer authority."
        ),
    ),
    "near": _binary(
        "near",
        directed=False,
        symmetric=True,
        subject_role="first_entity",
        object_role="second_entity",
        evidence_policy="separate_identity_crops_plus_wide_metric_context",
        geometry_policy="map_frame_distance_with_object_scale_diagnostic",
        negative_policy="a tight crop or missing co-visibility can never establish NOT_NEAR",
        qwen_instruction=(
            "Use the wide scene context and map geometry. Do not infer NEAR from "
            "two separately cropped objects or from image-plane adjacency alone."
        ),
    ),
    "between": RelationSpec(
        predicate="between",
        operator="TERNARY_FILTER",
        object_arity=2,
        directed=True,
        symmetric=False,
        inverse_predicate=None,
        subject_role="middle_entity",
        object_roles=("boundary_a", "boundary_b"),
        evidence_roles=("subject", "objects", "joint_context"),
        evidence_policy="subject_two_boundaries_and_shared_wide_context",
        geometry_policy="map_frame_segment_or_corridor_membership",
        negative_evidence_policy=(
            "refute only when the subject and both boundaries are jointly grounded"
        ),
        qwen_instruction=(
            "Decide whether the subject's physical body occupies the intervening "
            "gap or corridor bounded by both specified objects. Reconstruct the "
            "layout from physical centers, visible bases, floor contact, depth/side "
            "ordering, and the supplied 3D observations together. Being closer to "
            "one endpoint does not prevent BETWEEN. Image-plane box ordering is "
            "not sufficient: the subject must lie in the spatial corridor joining "
            "the two physical boundary objects. If a labelled anchor box covers "
            "the wrong object or its physical location is not recoverable, return "
            "UNKNOWN. A real bottom-to-top support contact with one boundary "
            "refutes it, but overlap, adjacency, or partial occlusion alone does "
            "not prove support or BETWEEN."
        ),
        operator_kind=OperatorKind.RELATION_FILTER,
        candidate_scope="persistent_global",
        requires_relation_roi_provenance=False,
        requires_qwen_tuple_verification=True,
    ),
    "closest": _binary(
        "closest",
        operator="ARGMIN",
        subject_role="ranked_candidate",
        object_role="reference_entity",
        evidence_policy="candidate_reference_identity_plus_complete_comparison_context",
        geometry_policy=(
            "argmin_horizontal_xy_distance_over_joint_target_anchor_domain"
        ),
        negative_policy="pairwise absence is not refutation; ranking requires a closed comparison domain",
        qwen_instruction=(
            "Ground the exact candidate and reference IDs. Categorical selection is "
            "valid only after comparison with the complete candidate domain."
        ),
    ),
    "farthest": _binary(
        "farthest",
        operator="ARGMAX",
        subject_role="ranked_candidate",
        object_role="reference_entity",
        evidence_policy="candidate_reference_identity_plus_complete_comparison_context",
        geometry_policy=(
            "argmax_horizontal_xy_distance_over_joint_target_anchor_domain"
        ),
        negative_policy="pairwise absence is not refutation; ranking requires a closed comparison domain",
        qwen_instruction=(
            "Ground the exact candidate and reference IDs. Categorical selection is "
            "valid only after comparison with the complete candidate domain."
        ),
    ),
    "in": _binary(
        "in",
        subject_role="contained_entity",
        object_role="container_entity",
        evidence_policy="contained_entity_container_boundary_and_interior_context",
        geometry_policy="contained_bbox_or_mask_inside_container_volume",
        negative_policy="an unseen or occluded container interior is UNKNOWN, never NOT_IN",
        qwen_instruction=(
            "Verify that the subject lies inside the specified container boundary; "
            "overlap in a 2D crop without visible container context is insufficient."
        ),
    ),
}

RELATION_REGISTRY: Mapping[str, RelationSpec] = MappingProxyType(_REGISTRY)
RELATION_REGISTRY_SCHEMA_VERSION = "relation_registry_v3"


@dataclass(frozen=True)
class TrajectoryActionSpec:
    action: str
    operator_kind: OperatorKind
    binds_landmark: bool
    terminal_capable: bool
    trajectory_constraint: str
    region_kind: str
    forbidden: bool = False


_TRAJECTORY_ACTIONS = {
    "go_near": TrajectoryActionSpec(
        "go_near", OperatorKind.TRAJECTORY_ACTION, True, False,
        "ENTER_REGION", "NEAR_REGION",
    ),
    "go_to": TrajectoryActionSpec(
        "go_to", OperatorKind.TRAJECTORY_ACTION, True, True,
        "ENTER_REGION", "STOP_REGION",
    ),
    "pass_near": TrajectoryActionSpec(
        "pass_near", OperatorKind.TRAJECTORY_ACTION, True, False,
        "PASS_THROUGH", "NEAR_PATH_REGION",
    ),
    "pass_between": TrajectoryActionSpec(
        "pass_between", OperatorKind.TRAJECTORY_ACTION, True, False,
        "PASS_THROUGH", "BETWEEN_PATH_REGION",
    ),
    "avoid_near": TrajectoryActionSpec(
        "avoid_near", OperatorKind.TRAJECTORY_ACTION, True, False,
        "AVOID_REGION", "NEAR_PATH_REGION", True,
    ),
    "stop_at": TrajectoryActionSpec(
        "stop_at", OperatorKind.TRAJECTORY_ACTION, True, True,
        "TERMINATE_INSIDE", "STOP_REGION",
    ),
}
TRAJECTORY_ACTION_REGISTRY: Mapping[str, TrajectoryActionSpec] = MappingProxyType(
    _TRAJECTORY_ACTIONS
)


def relation_spec(predicate: str) -> RelationSpec:
    normalized = str(predicate).strip().lower()
    try:
        return RELATION_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"relation_predicate_unregistered:{normalized}") from exc


def trajectory_action_spec(action: str) -> TrajectoryActionSpec:
    normalized = str(action).strip().lower()
    try:
        return TRAJECTORY_ACTION_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"trajectory_action_unregistered:{normalized}") from exc


def validate_relation_arguments(
    predicate: str,
    object_entities: Sequence[object],
) -> RelationSpec:
    spec = relation_spec(predicate)
    if len(tuple(object_entities)) != spec.object_arity:
        raise ValueError(
            f"relation_arity_invalid:{spec.predicate}:"
            f"expected={spec.object_arity}:actual={len(tuple(object_entities))}"
        )
    return spec


def registry_manifest() -> dict[str, object]:
    return {
        "schema_version": RELATION_REGISTRY_SCHEMA_VERSION,
        "relations": {
            predicate: spec.to_dict()
            for predicate, spec in RELATION_REGISTRY.items()
        },
        "trajectory_actions": {
            action: {
                **asdict(spec),
                "operator_kind": spec.operator_kind.value,
            }
            for action, spec in TRAJECTORY_ACTION_REGISTRY.items()
        },
    }
