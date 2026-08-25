"""Predicate/role-specific visual evidence and object-ID Qwen verification."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
import uuid
from dataclasses import replace

from integrations.qwen3vl.relation_types import ObjectGroundedRelationRequest
from orchestration.observation_contract import (
    INVALID,
    NO,
    UNKNOWN,
    YES,
    MASK_PANORAMA_PIXELS,
    relation_record_metadata,
)


def _geometry_center_size(
    obj: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    # Relation geometry lives in the persistent map frame.  A last-frame
    # monocular measurement can be a partial mask or even share a lifted box
    # with its support object; it must not overwrite the fused LiDAR/multiview
    # identity centre used for cross-object metric reasoning.
    center = [
        float(value)
        for value in obj["center_3d"]
    ]
    size = [
        float(value)
        for value in obj["bbox_3d"]
    ]
    return center, size


def _bounds(obj: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    center, size = _geometry_center_size(obj)
    half = [0.5 * value for value in size]
    return (
        [center[index] - half[index] for index in range(3)],
        [center[index] + half[index] for index in range(3)],
    )


def _axis_covariance_variance(value: Mapping[str, Any], axis: int) -> float:
    try:
        covariance = value.get("center_cov")
        result = float(covariance[axis][axis])
    except (TypeError, ValueError, IndexError):
        return 0.0
    return max(0.0, result) if math.isfinite(result) else 0.0


def _combined_axis_uncertainty_m(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    axis: int,
) -> float:
    variance = (
        _axis_covariance_variance(first, axis)
        + _axis_covariance_variance(second, axis)
    )
    return max(0.02, min(0.75, 2.5 * math.sqrt(variance)))


def relation_geometry_diagnostic(
    node: Mapping[str, Any],
    subject: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return predicate-specific geometry diagnostics without answer authority."""
    predicate = str(node["predicate"]).lower()
    result: dict[str, Any] = {
        "policy": str(node.get("geometry_policy", "")),
        "answer_authority": False,
    }
    if not objects:
        return result
    subject_center, subject_size = _geometry_center_size(subject)
    object_geometry = [_geometry_center_size(obj) for obj in objects]
    result["geometry_reliability"] = _relation_geometry_weight(
        subject, objects
    )
    result["participant_geometry_support"] = [
        {
            "object_id": int(value.get("object_id", -1)),
            "covariance_mode": str(value.get("covariance_mode", "")),
            "center_covariance_provenance": str(
                value.get("center_covariance_provenance", "")
            ),
            "geometry_point_count": int(
                value.get("geometry_point_count", 0) or 0
            ),
        }
        for value in (subject, *objects)
    ]
    if predicate in {"closest", "farthest"}:
        distances = [
            math.hypot(
                subject_center[0] - center[0],
                subject_center[1] - center[1],
            )
            for center, _size in object_geometry
        ]
    else:
        distances = [
            math.dist(subject_center, center)
            for center, _size in object_geometry
        ]
    result["distances_m"] = distances
    if predicate in {"closest", "farthest", "near"}:
        subject_diagonal = math.sqrt(sum(
            float(value) ** 2 for value in subject_size
        ))
        object_diagonals = [
            math.sqrt(sum(float(value) ** 2 for value in size))
            for _center, size in object_geometry
        ]
        combined_radii = [
            0.5 * (subject_diagonal + object_diagonal)
            for object_diagonal in object_diagonals
        ]
        result.update({
            "subject_bbox_diagonal_m": subject_diagonal,
            "object_bbox_diagonals_m": object_diagonals,
            "bounding_sphere_surface_gaps_m": [
                max(0.0, distance - radius)
                for distance, radius in zip(distances, combined_radii)
            ],
            "center_distance_over_combined_bbox_radius": [
                distance / max(radius, 1e-6)
                for distance, radius in zip(distances, combined_radii)
            ],
        })
        return result
    if predicate in {
        "left", "left_of", "right", "right_of",
        "front", "in_front_of", "behind", "behind_of",
    } and len(objects) == 1:
        subject_low, subject_high = _bounds(subject)
        object_low, object_high = _bounds(objects[0])
        if predicate in {"left", "left_of", "right", "right_of"}:
            axis = 0
            direction = "left" if predicate in {"left", "left_of"} else "right"
        else:
            axis = 1
            direction = "front" if predicate in {"front", "in_front_of"} else "behind"
        result.update({
            "world_axis": axis,
            "world_direction": direction,
            "subject_axis_interval": [subject_low[axis], subject_high[axis]],
            "object_axis_interval": [object_low[axis], object_high[axis]],
            "axis_separation_m": (
                object_low[axis] - subject_high[axis]
                if direction in {"left", "front"}
                else subject_low[axis] - object_high[axis]
            ),
        })
        return result
    if predicate in {"above", "below", "on"} and len(objects) == 1:
        subject_low, subject_high = _bounds(subject)
        object_low, object_high = _bounds(objects[0])
        overlap = [
            float(
                min(subject_high[index], object_high[index])
                - max(subject_low[index], object_low[index])
            )
            for index in (0, 1)
        ]
        horizontal_uncertainty = [
            _combined_axis_uncertainty_m(subject, objects[0], index)
            for index in (0, 1)
        ]
        vertical_delta = float(
            subject_center[2] - object_geometry[0][0][2]
        )
        result["horizontal_projection_axis_overlap_m"] = overlap
        result["vertical_center_delta_m"] = vertical_delta
        result["vertical_gap_m"] = float(subject_low[2] - object_high[2])
        result["subject_bounds"] = [subject_low, subject_high]
        result["object_bounds"] = [object_low, object_high]
        result["horizontal_overlap_uncertainty_m"] = horizontal_uncertainty
        return result
    if predicate == "in" and len(objects) == 1:
        subject_low, subject_high = _bounds(subject)
        object_low, object_high = _bounds(objects[0])
        result["subject_bounds"] = [subject_low, subject_high]
        result["object_bounds"] = [object_low, object_high]
        result["containment_margins_m"] = [
            [
                float(subject_low[index] - object_low[index]),
                float(object_high[index] - subject_high[index]),
            ]
            for index in range(3)
        ]
        return result
    if predicate == "between" and len(objects) == 2:
        point = subject_center[:2]
        first = object_geometry[0][0][:2]
        second = object_geometry[1][0][:2]
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared > 0.0:
            length = math.sqrt(length_squared)
            position = (
                (point[0] - first[0]) * dx + (point[1] - first[1]) * dy
            ) / length_squared
            projection = [first[0] + position * dx, first[1] + position * dy]
            subject_half_x = 0.5 * subject_size[0]
            subject_half_y = 0.5 * subject_size[1]
            normal_x, normal_y = -dy / length, dx / length
            subject_normal_radius = (
                abs(normal_x) * subject_half_x
                + abs(normal_y) * subject_half_y
            )
            boundary_normal_radii = [
                abs(normal_x) * 0.5 * size[0]
                + abs(normal_y) * 0.5 * size[1]
                for _center, size in object_geometry
            ]
            projected_half_extent = (
                abs(dx) * subject_half_x + abs(dy) * subject_half_y
            ) / length_squared
            subject_low, _subject_high = _bounds(subject)
            boundary_bounds = [_bounds(value) for value in objects]
            result.update({
                "segment_position": position,
                "segment_position_interval": [
                    position - projected_half_extent,
                    position + projected_half_extent,
                ],
                "perpendicular_distance_m": math.dist(point, projection),
                "corridor_half_width_m": (
                    subject_normal_radius + max(boundary_normal_radii)
                ),
                "vertical_center_deltas_m": [
                    subject_center[2] - center[2]
                    for center, _size in object_geometry
                ],
                "subject_bottom_minus_boundary_top_m": [
                    float(subject_low[2] - bounds[1][2])
                    for bounds in boundary_bounds
                ],
                "subject_horizontal_overlap_with_boundaries": [
                    bool(
                        subject_low[0] <= bounds[1][0]
                        and bounds[0][0] <= _subject_high[0]
                        and subject_low[1] <= bounds[1][1]
                        and bounds[0][1] <= _subject_high[1]
                    )
                    for bounds in boundary_bounds
                ],
            })
        return result
    return result


def _attach_shared_image_layout(
    geometry: dict[str, Any],
    node: Mapping[str, Any],
    evidence: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> None:
    """Attach exact same-view 2D topology used by the visual verifier."""
    predicate = str(node.get("predicate", "")).lower()
    if predicate not in {"above", "below", "on"} or len(objects) != 1:
        return
    subject_box = evidence.get("subject_grounding", {}).get(
        "focused_roi_bbox_xyxy"
    )
    object_groundings = evidence.get("object_groundings", ())
    object_box = (
        object_groundings[0].get("focused_roi_bbox_xyxy")
        if isinstance(object_groundings, Sequence)
        and len(object_groundings) == 1
        and isinstance(object_groundings[0], Mapping)
        else None
    )
    if not (
        isinstance(subject_box, Sequence)
        and not isinstance(subject_box, (str, bytes))
        and len(subject_box) == 4
        and isinstance(object_box, Sequence)
        and not isinstance(object_box, (str, bytes))
        and len(object_box) == 4
    ):
        return
    try:
        sx1, sy1, sx2, sy2 = (float(value) for value in subject_box)
        ox1, oy1, ox2, oy2 = (float(value) for value in object_box)
    except (TypeError, ValueError):
        return
    if not all(math.isfinite(value) for value in (
        sx1, sy1, sx2, sy2, ox1, oy1, ox2, oy2
    )):
        return
    subject_width = max(1e-6, sx2 - sx1)
    subject_height = max(1e-6, sy2 - sy1)
    object_height = max(1e-6, oy2 - oy1)
    overlap_width = max(0.0, min(sx2, ox2) - max(sx1, ox1))
    overlap_height = max(0.0, min(sy2, oy2) - max(sy1, oy1))
    try:
        image_width = float(object_groundings[0].get("image_width"))
        image_height = float(object_groundings[0].get("image_height"))
    except (TypeError, ValueError):
        image_width = 0.0
        image_height = 0.0
    exact_perspective_pixels = str(evidence.get("pixel_source", "")) in {
        "shared_perspective_view",
        "original_perspective_view",
    }
    support_box_clipped = bool(
        exact_perspective_pixels
        and image_width > 0.0
        and image_height > 0.0
        and (
            ox1 <= 0.0
            or oy1 <= 0.0
            or ox2 >= image_width
            or oy2 >= image_height
        )
    )
    if predicate in {"above", "below"}:
        subject_center_y = 0.5 * (sy1 + sy2)
        object_center_y = 0.5 * (oy1 + oy2)
        geometry["shared_image_vertical_layout"] = {
            "subject_bbox_xyxy": [sx1, sy1, sx2, sy2],
            "object_bbox_xyxy": [ox1, oy1, ox2, oy2],
            "horizontal_overlap_px": overlap_width,
            "horizontal_overlap_over_min_width": float(
                overlap_width / min(subject_width, max(1e-6, ox2 - ox1))
            ),
            "horizontal_compatible": bool(overlap_width > 0.0),
            "vertical_center_delta_px": float(subject_center_y - object_center_y),
            "vertical_ordering_compatible": bool(
                subject_center_y < object_center_y
                if predicate == "above"
                else subject_center_y > object_center_y
            ),
            "jointly_visible": evidence.get("jointly_observable") is True,
            "pixel_source": str(evidence.get("pixel_source", "")),
        }
        return
    geometry["shared_image_on_layout"] = {
        "subject_bbox_xyxy": [sx1, sy1, sx2, sy2],
        "object_bbox_xyxy": [ox1, oy1, ox2, oy2],
        "subject_containment_ratio": float(
            overlap_width * overlap_height
            / (subject_width * subject_height)
        ),
        "subject_bottom_position_in_object_height": float(
            (sy2 - oy1) / object_height
        ),
        "subject_starts_above_object_top": bool(sy1 < oy1),
        "subject_horizontal_overlap_with_support": bool(overlap_width > 0.0),
        "subject_box_strictly_inside_object_box": bool(
            sx1 > ox1 and sy1 > oy1 and sx2 < ox2 and sy2 < oy2
        ),
        "support_object_box_clipped_to_image": support_box_clipped,
        "image_size": [image_width, image_height],
        "support_object_class": str(objects[0].get("class_label", "")),
        "pixel_source": str(evidence.get("pixel_source", "")),
    }


def _geometry_consistency(
    node: Mapping[str, Any],
    geometry: Mapping[str, Any],
    subject: Mapping[str, Any] | None = None,
    objects: Sequence[Mapping[str, Any]] = (),
    qwen: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Fuse explicit 3D direction/projection evidence with a Qwen YES.

    This is relation evidence, not an answer-readiness gate.  A supported
    directed relation still needs the supplied map-frame gravity direction to
    agree.  ``below``/``above`` horizontal footprint separation is not by
    itself a contradiction: wall-mounted objects can be above furniture while
    remaining separated in depth.  Exact joint image evidence may resolve that
    layout; a missing diagnostic remains UNKNOWN.
    """
    predicate = str(node.get("predicate", "")).lower()
    if predicate in {"near", "next_to"}:
        threshold = node.get("near_threshold_m", node.get("metric_threshold_m"))
        if not _world_geometry_usable(subject, objects):
            return "UNKNOWN", "near_world_geometry_insufficient"
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            threshold_value = None
        gaps = geometry.get("bounding_sphere_surface_gaps_m")
        if not isinstance(gaps, (list, tuple)) or len(gaps) != len(objects):
            return "UNKNOWN", "near_surface_gap_missing"
        try:
            gap = min(float(value) for value in gaps)
        except (TypeError, ValueError):
            return "UNKNOWN", "near_surface_gap_invalid"
        if not math.isfinite(gap):
            return "UNKNOWN", "near_surface_gap_invalid"
        if threshold_value is None:
            qwen_evidence = qwen if isinstance(qwen, Mapping) else {}
            role_states = [
                str(qwen_evidence.get("subject_role_state", "UNKNOWN")).upper(),
                *(
                    str(value).upper()
                    for value in qwen_evidence.get("object_role_states", ())
                ),
            ]
            metric_contact_support = bool(
                qwen_evidence.get("state") == "supported"
                and qwen_evidence.get("jointly_observable") is True
                and len(role_states) == 1 + len(objects)
                and all(value == "YES" for value in role_states)
                and _relation_geometry_reliable(subject, objects)
                and gap <= 0.0
            )
            return (
                ("YES", "near_joint_metric_surface_contact")
                if metric_contact_support
                else ("UNKNOWN", "near_metric_threshold_not_declared")
            )
        if not math.isfinite(threshold_value):
            return "UNKNOWN", "near_surface_gap_invalid"
        return (
            ("YES", "near_world_geometry_within_declared_threshold")
            if gap <= max(0.0, threshold_value)
            else ("NO", "near_world_geometry_outside_declared_threshold")
        )
    if predicate == "between":
        if not _world_geometry_usable(subject, objects):
            return "UNKNOWN", "between_world_geometry_insufficient"
        distances = geometry.get("distances_m")
        if isinstance(distances, (list, tuple)):
            try:
                same_support = any(
                    math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-6)
                    for value in distances
                )
            except (TypeError, ValueError):
                same_support = False
            if same_support:
                return "NO", "between_subject_anchor_same_3d_support"
        interval = geometry.get("segment_position_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            return "UNKNOWN", "between_segment_geometry_missing"
        try:
            lower, upper = (float(value) for value in interval)
        except (TypeError, ValueError):
            return "UNKNOWN", "between_segment_geometry_invalid"
        if not (math.isfinite(lower) and math.isfinite(upper)):
            return "UNKNOWN", "between_segment_geometry_invalid"
        anchor_span = 0.0
        try:
            anchor_span = math.dist(
                objects[0]["center_3d"][:2],
                objects[1]["center_3d"][:2],
            )
        except (KeyError, TypeError, ValueError):
            anchor_span = 0.0
        normalized_uncertainty = min(
            0.75,
            (
                _center_uncertainty_radius(subject or {})
                + sum(_center_uncertainty_radius(value) for value in objects)
            ) / max(0.25, anchor_span),
        )
        if upper < 0.0 or lower > 1.0:
            outside_margin = max(0.0, -upper, lower - 1.0)
            if outside_margin > normalized_uncertainty:
                return "NO", "between_subject_outside_anchor_segment"
            return "UNKNOWN", "between_subject_segment_boundary_uncertain"
        if lower < 0.0 or upper > 1.0:
            # The subject extent/covariance overlaps an endpoint of the anchor
            # segment.  Its center may lie inside, but the available geometry
            # does not establish that the object is wholly between both
            # boundaries.
            return "UNKNOWN", "between_subject_segment_boundary_uncertain"
        try:
            perpendicular = float(geometry["perpendicular_distance_m"])
            corridor = float(geometry["corridor_half_width_m"])
        except (KeyError, TypeError, ValueError):
            return "UNKNOWN", "between_corridor_geometry_missing"
        if not (math.isfinite(perpendicular) and math.isfinite(corridor)):
            return "UNKNOWN", "between_corridor_geometry_invalid"
        if perpendicular > corridor:
            corridor_margin = perpendicular - corridor
            if corridor_margin > sum(
                _center_uncertainty_radius(value)
                for value in (subject or {}, *objects)
            ):
                return "NO", "between_subject_outside_anchor_corridor"
            return "UNKNOWN", "between_corridor_boundary_uncertain"
        return "YES", "between_segment_geometry_consistent"
    if predicate in {
        "left", "left_of", "right", "right_of",
        "front", "in_front_of", "behind", "behind_of",
    }:
        if not _world_geometry_usable(subject, objects):
            return "UNKNOWN", "direction_world_geometry_insufficient"
        separation = geometry.get("axis_separation_m")
        try:
            separation_value = float(separation)
        except (TypeError, ValueError):
            return "UNKNOWN", "direction_world_geometry_missing"
        if not math.isfinite(separation_value):
            return "UNKNOWN", "direction_world_geometry_invalid"
        if separation_value > 0.0:
            return "YES", "direction_world_geometry_separated"
        # Intersecting intervals do not establish the opposite directed
        # relation.  They remain epistemically unknown until another view or
        # a valid visual relation resolves the ordering.
        return "UNKNOWN", "direction_world_intervals_overlap"
    if predicate == "in":
        if not _world_geometry_usable(subject, objects) or len(objects) != 1:
            return "UNKNOWN", "containment_world_geometry_insufficient"
        margins = geometry.get("containment_margins_m")
        if not isinstance(margins, Sequence) or len(margins) != 3:
            return "UNKNOWN", "containment_geometry_missing"
        try:
            values = [float(item) for pair in margins for item in pair]
        except (TypeError, ValueError):
            return "UNKNOWN", "containment_geometry_invalid"
        if not all(math.isfinite(value) for value in values):
            return "UNKNOWN", "containment_geometry_invalid"
        if all(value >= 0.0 for value in values):
            return "YES", "containment_world_geometry_confirmed"
        subject_low, subject_high = _bounds(subject)
        object_low, object_high = _bounds(objects[0])
        if any(
            subject_high[index] < object_low[index]
            or subject_low[index] > object_high[index]
            for index in range(3)
        ):
            return "NO", "containment_world_geometry_disjoint"
        return "UNKNOWN", "containment_world_geometry_partial"
    # ``above``/``below`` and ``on`` have necessary map-frame geometry.  A
    # probabilistic/bearing-only lift may guide acquisition, but it must not
    # be turned into a categorical answer without usable metric support.
    if predicate not in {"above", "below", "on"}:
        return "YES", "geometry_not_required_for_predicate"
    if not _world_geometry_usable(subject, objects):
        return "UNKNOWN", "vertical_world_geometry_insufficient"
    raw_delta = geometry.get("vertical_center_delta_m")
    raw_overlap = geometry.get("horizontal_projection_axis_overlap_m")
    try:
        delta = float(raw_delta)
        overlap = [float(value) for value in raw_overlap]
    except (TypeError, ValueError):
        return "UNKNOWN", "direction_or_projection_evidence_missing"
    if not math.isfinite(delta) or len(overlap) != 2 or not all(
        math.isfinite(value) for value in overlap
    ):
        return "UNKNOWN", "direction_or_projection_evidence_invalid"
    if predicate == "on":
        if any(value <= 0.0 for value in overlap):
            layout = geometry.get("shared_image_on_layout")
            if not isinstance(layout, Mapping):
                # An observed point-cloud box describes only the visible
                # support geometry.  Projection disjointness is categorical
                # only after the relevant surface is jointly visible, as the
                # ON negative-evidence contract requires.
                return "UNKNOWN", "on_joint_support_surface_evidence_missing"
            if layout.get("support_object_box_clipped_to_image") is True:
                return "UNKNOWN", "on_support_surface_extent_truncated"
            return "NO", "on_horizontal_projection_disjoint"
        if delta < 0.0:
            return "NO", "on_subject_below_support_center"
        if delta == 0.0:
            return "UNKNOWN", "vertical_direction_ambiguous"
        if "vertical_gap_m" not in geometry:
            return "UNKNOWN", "on_vertical_gap_missing"
        if not math.isfinite(vertical_gap):
            return "UNKNOWN", "on_vertical_gap_invalid"
        # Contact tolerance is derived from the participant uncertainties and
        # is only used after horizontal overlap/order are established.  A large
        # gap is missing support geometry, not proof that ON is false.
        tolerance = max(
            0.12,
            _center_uncertainty_radius(subject or {})
            + _center_uncertainty_radius(objects[0]),
        )
        if vertical_gap > tolerance:
            return "UNKNOWN", "on_support_surface_gap_unresolved"
        return "YES", "on_gravity_projection_and_contact_consistent"
    if delta == 0.0:
        return "UNKNOWN", "vertical_direction_ambiguous"
    vertical_uncertainty = _center_uncertainty_radius(subject or {}) + sum(
        _center_uncertainty_radius(value) for value in objects
    )
    if (predicate == "below" and delta > 0.0) or (
        predicate == "above" and delta < 0.0
    ):
        if abs(delta) > vertical_uncertainty:
            return "NO", "vertical_direction_contradiction"
        return "UNKNOWN", "vertical_direction_boundary_uncertain"
    if any(value <= 0.0 for value in overlap):
        projection_gaps = [max(0.0, -value) for value in overlap]
        raw_horizontal_uncertainty = geometry.get(
            "horizontal_overlap_uncertainty_m", ()
        )
        try:
            horizontal_uncertainty = [
                float(value) for value in raw_horizontal_uncertainty
            ]
        except (TypeError, ValueError):
            horizontal_uncertainty = []
        if len(horizontal_uncertainty) != 2:
            horizontal_uncertainty = [vertical_uncertainty, vertical_uncertainty]
        layout = geometry.get("shared_image_vertical_layout", {})
        qwen_evidence = qwen if isinstance(qwen, Mapping) else {}
        qwen_state = str(qwen_evidence.get("state", "")).strip().lower()
        role_states = [
            str(qwen_evidence.get("subject_role_state", "UNKNOWN")).upper(),
            *(
                str(value).upper()
                for value in qwen_evidence.get("object_role_states", ())
            ),
        ]
        visual_boundary_support = bool(
            isinstance(layout, Mapping)
            and layout.get("jointly_visible") is True
            and layout.get("horizontal_compatible") is True
            and layout.get("vertical_ordering_compatible") is True
            and qwen_state == "supported"
            and len(role_states) == 1 + len(objects)
            and all(value == "YES" for value in role_states)
        )
        if visual_boundary_support:
            return (
                "YES",
                "gravity_direction_and_joint_visual_projection_consistent",
            )
        if any(
            gap > uncertainty
            for gap, uncertainty in zip(
                projection_gaps, horizontal_uncertainty
            )
        ):
            return "UNKNOWN", "horizontal_projection_requires_joint_visual_evidence"
        return "UNKNOWN", "horizontal_projection_boundary_uncertain"
    return "YES", "gravity_direction_and_horizontal_projection_consistent"


def _world_geometry_usable(
    subject: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
) -> bool:
    """Whether the map geometry can participate in a categorical relation.

    This is based on the actual contract and support provenance, not a global
    attempt/token/viewpoint gate.  A fused probabilistic box with real depth
    support is usable; a bearing-only hypothesis without support is not.
    """
    participants = [value for value in (subject, *objects) if isinstance(value, Mapping)]
    if not participants or len(participants) != 1 + len(objects):
        return False
    for value in participants:
        try:
            center = [float(item) for item in value.get("center_3d", ())]
            extent = [float(item) for item in value.get("bbox_3d", ())]
        except (TypeError, ValueError):
            return False
        if len(center) != 3 or len(extent) != 3 or not all(
            math.isfinite(item) for item in (*center, *extent)
        ) or any(item <= 0.0 for item in extent):
            return False
        mode = str(value.get("covariance_mode", "")).lower()
        provenance = str(
            value.get("center_covariance_provenance", "")
        ).lower()
        try:
            support = int(value.get("geometry_point_count", 0) or 0)
        except (TypeError, ValueError):
            support = 0
        if support <= 0 and (
            mode == "bearing_only"
            or provenance in {"bearing_only", "bearing_model"}
        ):
            return False
        if support <= 0 and str(value.get("geometry_source", "")).lower() in {
            "bearing_only", "mask_bearing_sparse_lidar",
        }:
            return False
    return True


def _center_uncertainty_radius(value: Mapping[str, Any]) -> float:
    try:
        covariance = value.get("center_cov")
        matrix = [[float(item) for item in row] for row in covariance]
        if len(matrix) == 3 and all(len(row) == 3 for row in matrix):
            diagonal = [max(0.0, matrix[index][index]) for index in range(3)]
            return min(0.75, max(0.04, 2.5 * math.sqrt(max(diagonal))))
    except (TypeError, ValueError, IndexError):
        pass
    return 0.10


def _relation_geometry_reliable(
    subject: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
) -> bool:
    """Whether every tuple participant has usable metric extent evidence."""
    participants = [
        value for value in (subject, *objects)
        if isinstance(value, Mapping)
    ]
    return (
        bool(participants)
        and len(participants) == 1 + len(objects)
        and all(
            str(value.get("covariance_mode", "")).lower() == "strict"
            and str(value.get("center_covariance_provenance", "")).lower()
            not in {
                "",
                "bearing_only",
                "extent_derived_fallback",
                "single_point_fallback",
            }
            for value in participants
        )
        and _relation_geometry_weight(subject, objects) >= 0.50
    )


def _relation_geometry_weight(
    subject: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
) -> float:
    """Continuous trust weight for metric relation evidence."""
    participants = [
        value for value in (subject, *objects)
        if isinstance(value, Mapping)
    ]
    if not participants or len(participants) != 1 + len(objects):
        return 0.0
    weights: list[float] = []
    for value in participants:
        mode = str(value.get("covariance_mode", "")).lower()
        mode_weight = {
            "strict": 1.0,
            "probabilistic": 0.70,
            "bearing_only": 0.20,
        }.get(mode, 0.35)
        provenance = str(
            value.get("center_covariance_provenance", "")
        ).lower()
        provenance_weight = {
            "fused_statistical": 0.95,
            "multi_view_statistical": 0.90,
            "extent_derived": 0.70,
            "extent_derived_fallback": 0.45,
            "single_point_fallback": 0.25,
            "bearing_only": 0.15,
        }.get(provenance, 0.40)
        try:
            point_count = max(0, int(value.get("geometry_point_count", 0) or 0))
        except (TypeError, ValueError):
            point_count = 0
        # Sparse small-object masks regularly attach a handful of returns to
        # the background wall or support surface. Confidence should grow
        # gradually across dozens of returns, not become near-certain at the
        # traditional three-point minimum.
        point_weight = 1.0 - math.exp(-float(point_count) / 50.0)
        weights.append(mode_weight * provenance_weight * point_weight)
    return max(0.0, min(1.0, sum(weights) / len(weights)))


def _reported_true_probability(
    state: str,
    qwen: Mapping[str, Any],
) -> float:
    """Convert confidence-in-reported-state to P(relation=true)."""
    try:
        confidence = float(qwen.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75
    if not math.isfinite(confidence):
        confidence = 0.75
    confidence = max(0.01, min(0.99, confidence))
    normalized = str(state).upper()
    if normalized == "YES":
        return confidence
    if normalized == "NO":
        return 1.0 - confidence
    return 0.5


def _relation_true_probability(
    node: Mapping[str, Any],
    geometry: Mapping[str, Any],
    subject: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
    state: str,
    qwen: Mapping[str, Any],
) -> float:
    """Fuse visual semantics and metric layout without a fixed NEAR radius."""
    visual_probability = _reported_true_probability(state, qwen)
    if str(node.get("predicate", "")).lower() != "near":
        return visual_probability
    geometry_weight = _relation_geometry_weight(subject, objects)
    if geometry_weight <= 0.0:
        return visual_probability
    raw_gaps = geometry.get("bounding_sphere_surface_gaps_m", ())
    raw_distances = geometry.get("distances_m", ())
    if (
        not isinstance(raw_gaps, (list, tuple))
        or not isinstance(raw_distances, (list, tuple))
        or not raw_gaps
        or len(raw_gaps) != len(raw_distances)
    ):
        return visual_probability
    proximity_probabilities: list[float] = []
    for raw_gap, raw_distance in zip(raw_gaps, raw_distances):
        try:
            gap = max(0.0, float(raw_gap))
            distance = max(0.0, float(raw_distance))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(gap) and math.isfinite(distance)):
            continue
        combined_radius = max(0.0, distance - gap)
        # Object extent defines the natural spatial scale.  The 1 m floor
        # prevents tiny/noisy reconstructions from creating an accidental
        # near-zero scale; the Cauchy curve remains continuous everywhere.
        scale = max(1.0, 2.0 * combined_radius)
        proximity_probabilities.append(1.0 / (1.0 + (gap / scale) ** 2))
    if not proximity_probabilities:
        return visual_probability
    # NEAR has one object argument in the public task grammar.  ``max`` keeps
    # the helper well-defined for a future any-of anchor list without adding
    # a hidden all-objects gate.
    proximity_probability = max(proximity_probabilities)
    # Uncertain metric reconstruction softens the geometric contribution
    # continuously instead of disabling it or turning it into a veto. A
    # three-return plant mask therefore remains visually decidable, while a
    # dense reconstruction can strongly discount a truly distant tuple.
    weighted_proximity = (
        (1.0 - geometry_weight)
        + geometry_weight * proximity_probability
    )
    return max(0.01, min(0.99, visual_probability * weighted_proximity))


def _panorama_for_mask(mask_path: Path) -> Path | None:
    for parent in (mask_path.parent, *mask_path.parents[:5]):
        candidate = parent / "camera_panorama.png"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _usable_evidence(obj: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in obj.get("evidence", ()):
        # Only an explicitly projected panorama mask can index the panorama.
        # Perspective masks are handled by the exact shared-view path below;
        # their parent directory is not evidence that they are panorama data.
        if str(raw.get("panorama_mask_coordinate_frame", "")) != MASK_PANORAMA_PIXELS:
            continue
        mask_path = Path(str(raw.get("panorama_mask_path", ""))).resolve()
        if not mask_path.is_file():
            continue
        declared_panorama = Path(
            str(raw.get("panorama_image_path", ""))
        ).resolve()
        panorama_path = (
            declared_panorama
            if declared_panorama.is_file()
            else _panorama_for_mask(mask_path)
        )
        if panorama_path is None:
            continue
        candidate = {
            **dict(raw),
            "mask_path": str(mask_path),
            "panorama_path": str(panorama_path),
        }
        # World-model relifting can refresh an old observation in a newer
        # acquisition.  The panorama path, not that mutable refresh label, is
        # the co-visibility identity for relation evidence.
        evidence_view_key = str(panorama_path)
        previous = result.get(evidence_view_key)
        if previous is None or (
            float(candidate.get("geometry_confidence", 0.0)),
            len(candidate.get("source_view_ids", ())),
        ) > (
            float(previous.get("geometry_confidence", 0.0)),
            len(previous.get("source_view_ids", ())),
        ):
            result[evidence_view_key] = candidate
    return result


def _mask_bbox(mask) -> list[int] | None:
    import numpy as np

    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _union_bbox(values: Sequence[Sequence[int]]) -> list[int]:
    return [
        min(value[0] for value in values),
        min(value[1] for value in values),
        max(value[2] for value in values),
        max(value[3] for value in values),
    ]


def _expanded_bbox(
    bbox: Sequence[int],
    *,
    width: int,
    height: int,
    predicate: str,
    full_context: bool,
) -> list[int]:
    if full_context:
        return [0, 0, width, height]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    horizontal = 0.75 if predicate in {"above", "below"} else 0.50
    vertical = 1.00 if predicate == "on" else 0.75
    return [
        max(0, int(round(x1 - horizontal * box_width))),
        max(0, int(round(y1 - vertical * box_height))),
        min(width, int(round(x2 + horizontal * box_width))),
        min(height, int(round(y2 + vertical * box_height))),
    ]


def _patch_aligned(image, maximum_pixels: int):
    import cv2

    height, width = image.shape[:2]
    # Focused relation ROIs are often deliberately small. Preserve the pixels
    # selected by the dependency-bound search but enlarge tiny evidence so the
    # model can read the marked instance identity (for example book spines).
    detail_scale = max(1.0, 512.0 / max(1, min(height, width)))
    scale = min(
        detail_scale,
        math.sqrt(max(1, int(maximum_pixels)) / (height * width)),
    )
    target_width = max(32, int(math.floor(width * scale / 32.0)) * 32)
    target_height = max(32, int(math.floor(height * scale / 32.0)) * 32)
    if (target_width, target_height) != (width, height):
        image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
    height, width = image.shape[:2]
    pad_right = (-width) % 32
    pad_bottom = (-height) % 32
    if pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return image


def _annotated_crop(
    panorama,
    records: Sequence[tuple[str, int, Mapping[str, Any]]],
    crop_bbox: Sequence[int],
):
    import cv2

    x1, y1, x2, y2 = (int(value) for value in crop_bbox)
    crop = panorama[y1:y2, x1:x2].copy()
    colors = {
        "subject": (32, 32, 255),
        "object_1": (255, 96, 32),
        "object_2": (32, 200, 32),
    }
    for role, object_id, record in records:
        mask = cv2.imread(str(record["mask_path"]), cv2.IMREAD_GRAYSCALE)
        bbox = _mask_bbox(mask) if mask is not None else None
        if bbox is None:
            continue
        left = max(0, bbox[0] - x1)
        top = max(0, bbox[1] - y1)
        right = min(crop.shape[1] - 1, bbox[2] - x1)
        bottom = min(crop.shape[0] - 1, bbox[3] - y1)
        color = colors.get(role, (255, 255, 32))
        thickness = max(2, int(round(min(crop.shape[:2]) / 240.0)))
        cv2.rectangle(crop, (left, top), (right, bottom), color, thickness)
        label = f"S:{object_id}" if role == "subject" else f"O{role[-1]}:{object_id}"
        cv2.putText(
            crop,
            label,
            (left, max(18, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    return crop


def _pixel_bbox(
    normalized: object, *, width: int, height: int
) -> list[int] | None:
    if not isinstance(normalized, Sequence) or isinstance(
        normalized, (str, bytes, bytearray)
    ) or len(normalized) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in normalized)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return [left, top, right, bottom]


def _focused_evidence_priority(value: Mapping[str, Any]) -> tuple[float, float, str]:
    """Prefer the view where the grounded subject occupies most real pixels."""
    bbox = value.get("relation_roi_subject_bbox_xyxy_normalized")
    area = 0.0
    if (
        isinstance(bbox, Sequence)
        and not isinstance(bbox, (str, bytes, bytearray))
        and len(bbox) == 4
    ):
        try:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
        except (TypeError, ValueError):
            area = 0.0
    return (
        area,
        float(value.get("geometry_confidence", 0.0)),
        str(value.get("relation_roi_view_id", "")),
    )


def _annotated_boxes(
    image,
    records: Sequence[tuple[str, int, Sequence[int]]],
):
    import cv2

    result = image.copy()
    colors = {
        "subject": (32, 32, 255),
        "object_1": (255, 96, 32),
        "object_2": (32, 200, 32),
    }
    thickness = max(2, int(round(min(result.shape[:2]) / 180.0)))
    for role, object_id, bbox in records:
        left, top, right, bottom = (int(value) for value in bbox)
        color = colors.get(role, (255, 255, 32))
        cv2.rectangle(
            result,
            (left, top),
            (max(left, right - 1), max(top, bottom - 1)),
            color,
            thickness,
        )
        label = f"S:{object_id}" if role == "subject" else f"O{role[-1]}:{object_id}"
        cv2.putText(
            result,
            label,
            (left, max(18, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    return result


def _role_crop(
    image,
    records: Sequence[tuple[str, int, Sequence[int]]],
    focus_bboxes: Sequence[Sequence[int]],
    *,
    predicate: str,
):
    """Crop one role closely while preserving its ID annotations.

    The grounding ROI is already the correct dependency-bound visual region,
    but reusing the whole ROI for every role makes small subjects occupy only a
    few model patches.  Subject and object views therefore focus their own
    pixels; the joint view remains responsible for the predicate itself.
    """
    height, width = image.shape[:2]
    crop_bbox = _expanded_bbox(
        _union_bbox(focus_bboxes),
        width=width,
        height=height,
        predicate=predicate,
        full_context=False,
    )
    left, top, right, bottom = crop_bbox
    shifted = [
        (
            role,
            object_id,
            [
                max(0, int(bbox[0]) - left),
                max(0, int(bbox[1]) - top),
                min(right - left, int(bbox[2]) - left),
                min(bottom - top, int(bbox[3]) - top),
            ],
        )
        for role, object_id, bbox in records
    ]
    return _annotated_boxes(image[top:bottom, left:right], shifted)


class RelationEvidenceBuilder:
    """Create audit-visible subject/object/joint images for one ID tuple."""

    def __init__(self, output_dir: Path, *, maximum_pixels: int) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.maximum_pixels = max(32 * 32, int(maximum_pixels))

    def _shared_perspective_view(
        self,
        *,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
        tuple_dir: Path,
    ) -> dict[str, Any] | None:
        """Build tuple evidence in the pixel frame that owns the boxes.

        LiDAR observations keep perspective-view masks and boxes. Treating
        those 1024x768 masks as if they indexed the 1920x640 panorama silently
        moved annotations onto unrelated pixels. Prefer an exact perspective
        view shared by every persistent-ID participant before considering a
        true panorama mask.
        """
        import cv2

        participants = [subject, *objects]
        records_by_view: list[dict[str, list[Mapping[str, Any]]]] = []
        for participant in participants:
            by_view: dict[str, list[Mapping[str, Any]]] = {}
            for raw in participant.get("evidence", ()):
                representative_view_id = str(
                    raw.get("representative_view_id", "")
                ).strip()
                representative_image_path = Path(
                    str(raw.get("representative_view_image", ""))
                ).resolve()
                candidates = [{
                    "view_id": representative_view_id,
                    "image_path": representative_image_path,
                    "bbox": raw.get("representative_bbox_xyxy"),
                }]
                # A station-fused identity retains an exact box for every
                # source perspective, while ``representative_*`` names only
                # one of them. Requiring all participants to choose the same
                # representative discarded valid book/cabinet and
                # plant/book tuples even though their source boxes shared a
                # view. Recover those exact per-view boxes here.
                for source_box in raw.get("source_view_boxes", ()):
                    if not isinstance(source_box, Mapping):
                        continue
                    source_view_id = str(
                        source_box.get("view_id", "")
                    ).strip()
                    suffix = (
                        source_view_id.split("__", 1)[1]
                        if "__" in source_view_id else ""
                    )
                    source_image_path = (
                        representative_image_path.parent / f"{suffix}.png"
                        if suffix and representative_image_path.parent.is_dir()
                        else Path("")
                    ).resolve()
                    candidates.append({
                        "view_id": source_view_id,
                        "image_path": source_image_path,
                        "bbox": source_box.get("bbox_xyxy"),
                    })
                for candidate in candidates:
                    view_id = str(candidate["view_id"])
                    image_path = Path(candidate["image_path"])
                    bbox = candidate["bbox"]
                    if (
                        not view_id
                        or not image_path.is_file()
                        or not isinstance(bbox, Sequence)
                        or isinstance(bbox, (str, bytes, bytearray))
                        or len(bbox) != 4
                    ):
                        continue
                    normalized = {
                        **dict(raw),
                        "representative_view_id": view_id,
                        "representative_view_image": str(image_path),
                        "representative_bbox_xyxy": list(bbox),
                    }
                    by_view.setdefault(view_id, []).append(normalized)
            if not by_view:
                return None
            records_by_view.append(by_view)

        common_views = set(records_by_view[0])
        for values in records_by_view[1:]:
            common_views.intersection_update(values)
        if not common_views:
            return None

        def record_score(raw: Mapping[str, Any]) -> tuple[float, float, str]:
            bbox = raw.get("representative_bbox_xyxy", ())
            try:
                area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                    0.0, float(bbox[3]) - float(bbox[1])
                )
            except (TypeError, ValueError, IndexError):
                area = 0.0
            return (
                area,
                float(raw.get("geometry_confidence", 0.0)),
                str(raw.get("observation_id", "")),
            )

        selected_by_view: dict[str, list[Mapping[str, Any]]] = {
            view_id: [
                max(values[view_id], key=record_score)
                for values in records_by_view
            ]
            for view_id in common_views
        }

        def view_score(view_id: str) -> tuple[float, float, str]:
            scores = [
                record_score(value) for value in selected_by_view[view_id]
            ]
            return (
                min(value[0] for value in scores),
                sum(value[0] for value in scores),
                view_id,
            )

        view_id = max(common_views, key=view_score)
        selected = selected_by_view[view_id]
        image_paths = {
            str(Path(str(value["representative_view_image"])).resolve())
            for value in selected
        }
        if len(image_paths) != 1:
            return None
        image_path = Path(next(iter(image_paths)))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        bboxes: list[list[int]] = []
        for raw in selected:
            try:
                x1, y1, x2, y2 = (
                    int(round(float(value)))
                    for value in raw["representative_bbox_xyxy"]
                )
            except (KeyError, TypeError, ValueError):
                return None
            bbox = [
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
                max(1, min(width, x2)),
                max(1, min(height, y2)),
            ]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None
            bboxes.append(bbox)

        subject_record = ("subject", int(subject["object_id"]), bboxes[0])
        object_records = [
            (f"object_{index}", int(obj["object_id"]), bboxes[index])
            for index, obj in enumerate(objects, start=1)
        ]
        image_specs = []
        for role, annotations, focus_bboxes in (
            ("subject", [subject_record], [bboxes[0]]),
            ("objects", object_records, bboxes[1:]),
            (
                "joint_context",
                [subject_record, *object_records],
                bboxes,
            ),
        ):
            rendered = _role_crop(
                image,
                annotations,
                focus_bboxes,
                predicate=str(node["predicate"]).lower(),
            )
            rendered = _patch_aligned(rendered, self.maximum_pixels)
            path = tuple_dir / f"{role}.png"
            if not cv2.imwrite(str(path), rendered):
                raise RuntimeError("relation_evidence_image_write_failed")
            image_specs.append({
                "role": role,
                "image_path": str(path),
                "preprocessed_once": True,
            })

        acquisition_id = view_id.split("__", 1)[0]
        evidence_ids = list(dict.fromkeys(
            f"{acquisition_id}:{raw.get('observation_id', '')}"
            for raw in selected
        ))
        camera_pose: list[float] = []
        source_view_ids = [
            str(value) for value in selected[0].get("source_view_ids", ())
        ]
        source_poses = selected[0].get("source_cam2w_maps", ())
        try:
            pose_index = source_view_ids.index(view_id)
            camera_pose = [
                float(value)
                for row in source_poses[pose_index]
                for value in row
            ]
        except (ValueError, IndexError, TypeError):
            camera_pose = []

        result = {
            "status": "completed",
            "reason": None,
            "jointly_observable": True,
            "acquisition_id": acquisition_id,
            "panorama_path": None,
            "perspective_view_path": str(image_path),
            "perspective_view_id": view_id,
            "pixel_source": "shared_perspective_view",
            "images": image_specs,
            "subject_grounding": {
                "object_id": int(subject["object_id"]),
                "instance_version": int(subject.get("instance_version", 0)),
                "acquisition_id": acquisition_id,
                "observation_id": str(selected[0].get("observation_id", "")),
                "focused_roi_bbox_xyxy": bboxes[0],
                "image_width": width,
                "image_height": height,
            },
            "object_groundings": [
                {
                    "object_id": int(obj["object_id"]),
                    "instance_version": int(obj.get("instance_version", 0)),
                    "acquisition_id": acquisition_id,
                    "observation_id": str(selected[index].get(
                        "observation_id", ""
                    )),
                    "focused_roi_bbox_xyxy": bboxes[index],
                    "image_width": width,
                    "image_height": height,
                }
                for index, obj in enumerate(objects, start=1)
            ],
            "camera_pose": camera_pose,
            "evidence_ids": evidence_ids,
            "evidence_policy": (
                f"{node.get('evidence_policy', '')}:shared_perspective_view"
            ),
        }
        (tuple_dir / "manifest.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    def _focused_relation_roi(
        self,
        *,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
        tuple_dir: Path,
    ) -> dict[str, Any] | None:
        """Use the original grounding ROI before panorama reprojection.

        The ROI is the pixel surface on which the target proposal was made.
        Keeping it here prevents tiny objects from being reduced to a handful
        of panorama pixels before Qwen verifies their class and relation.
        """
        import cv2

        if len(objects) > 1:
            expected_classes = [
                str(value.get("class_label", "")).strip().lower()
                for value in objects
            ]
            for raw in sorted(
                subject.get("evidence", ()),
                key=_focused_evidence_priority,
                reverse=True,
            ):
                roi_path = Path(
                    str(raw.get("relation_roi_image_path", ""))
                ).resolve()
                roi_view_id = str(raw.get("relation_roi_view_id", "")).strip()
                anchor_classes = [
                    str(value).strip().lower()
                    for value in raw.get("relation_roi_anchor_classes", ())
                ]
                normalized_anchor_boxes = raw.get(
                    "relation_roi_anchor_bboxes_xyxy_normalized", ()
                )
                if (
                    not roi_path.is_file()
                    or not roi_view_id
                    or anchor_classes != expected_classes
                    or len(normalized_anchor_boxes) != len(objects)
                ):
                    continue
                counterpart_records = []
                for counterpart in objects:
                    record = next((
                        value for value in counterpart.get("evidence", ())
                        if roi_view_id in {
                            str(view_id)
                            for view_id in value.get("source_view_ids", ())
                        }
                    ), None)
                    if record is None:
                        counterpart_records = []
                        break
                    counterpart_records.append(record)
                if len(counterpart_records) != len(objects):
                    continue
                # Class identity cannot be audited reliably from the tight
                # relation crop alone: a detector box may cover only one part
                # of a larger body (for example, a seat without its back).
                # Prefer the untouched perspective frame when every tuple
                # participant has an exact box in that same view.  The close
                # role crops below still retain small-object detail, while the
                # joint image keeps the complete physical-object context.
                raw_view_path = Path(str(
                    raw.get("representative_view_image", "")
                )).resolve()
                participant_records = [raw, *counterpart_records]
                participant_view_paths = [
                    Path(str(value.get(
                        "representative_view_image", ""
                    ))).resolve()
                    for value in participant_records
                ]
                participant_view_bboxes = [
                    value.get("representative_bbox_xyxy")
                    for value in participant_records
                ]
                use_raw_view = bool(
                    raw_view_path.is_file()
                    and all(value == raw_view_path for value in participant_view_paths)
                    and all(
                        str(value.get("representative_view_id", ""))
                        == roi_view_id
                        for value in participant_records
                    )
                    and all(
                        isinstance(value, Sequence)
                        and not isinstance(value, (str, bytes, bytearray))
                        and len(value) == 4
                        for value in participant_view_bboxes
                    )
                )
                roi = cv2.imread(
                    str(raw_view_path if use_raw_view else roi_path),
                    cv2.IMREAD_COLOR,
                )
                if roi is None:
                    continue
                height, width = roi.shape[:2]
                if use_raw_view:
                    try:
                        concrete_view_bboxes = [
                            [int(round(float(coordinate))) for coordinate in value]
                            for value in participant_view_bboxes
                        ]
                    except (TypeError, ValueError):
                        continue
                    subject_bbox = concrete_view_bboxes[0]
                    object_bboxes = concrete_view_bboxes[1:]
                else:
                    subject_bbox = _pixel_bbox(
                        raw.get("relation_roi_subject_bbox_xyxy_normalized"),
                        width=width,
                        height=height,
                    )
                    object_bboxes = [
                        _pixel_bbox(value, width=width, height=height)
                        for value in normalized_anchor_boxes
                    ]
                if subject_bbox is None or any(
                    value is None for value in object_bboxes
                ):
                    continue
                concrete_object_bboxes = [
                    list(value) for value in object_bboxes if value is not None
                ]
                subject_record = (
                    "subject", int(subject["object_id"]), subject_bbox
                )
                object_records = [
                    (f"object_{index}", int(obj["object_id"]), bbox)
                    for index, (obj, bbox) in enumerate(
                        zip(objects, concrete_object_bboxes), start=1
                    )
                ]
                image_specs = []
                for role, annotations, focus_bboxes in (
                    ("subject", [subject_record], [subject_bbox]),
                    (
                        "objects",
                        object_records,
                        concrete_object_bboxes,
                    ),
                    (
                        "joint_context",
                        [subject_record, *object_records],
                        [subject_bbox, *concrete_object_bboxes],
                    ),
                ):
                    image = _role_crop(
                        roi,
                        annotations,
                        focus_bboxes,
                        predicate=str(node["predicate"]).lower(),
                    )
                    image = _patch_aligned(image, self.maximum_pixels)
                    path = tuple_dir / f"{role}.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(
                            "relation_evidence_image_write_failed"
                        )
                    image_specs.append({
                        "role": role,
                        "image_path": str(path),
                        "preprocessed_once": True,
                    })
                acquisition_id = roi_view_id.split("__", 1)[0]
                evidence_ids = list(dict.fromkeys([
                    f"{acquisition_id}:{raw.get('observation_id', '')}",
                    *(
                        f"{acquisition_id}:{record.get('observation_id', '')}"
                        for record in counterpart_records
                    ),
                ]))
                result = {
                    "status": "completed",
                    "reason": None,
                    "jointly_observable": True,
                    "acquisition_id": acquisition_id,
                    "panorama_path": None,
                    "focused_relation_roi_path": str(roi_path),
                    "pixel_source": (
                        "original_perspective_view"
                        if use_raw_view
                        else "original_multi_anchor_relation_roi"
                    ),
                    "images": image_specs,
                    "subject_grounding": {
                        "object_id": int(subject["object_id"]),
                        "instance_version": int(
                            subject.get("instance_version", 0)
                        ),
                        "acquisition_id": acquisition_id,
                        "observation_id": str(raw.get("observation_id", "")),
                        "focused_roi_bbox_xyxy": subject_bbox,
                        "image_width": width,
                        "image_height": height,
                    },
                    "object_groundings": [
                        {
                            "object_id": int(obj["object_id"]),
                            "instance_version": int(
                                obj.get("instance_version", 0)
                            ),
                            "acquisition_id": acquisition_id,
                            "observation_id": str(
                                record.get("observation_id", "")
                            ),
                            "focused_roi_bbox_xyxy": bbox,
                            "image_width": width,
                            "image_height": height,
                        }
                        for obj, record, bbox in zip(
                            objects,
                            counterpart_records,
                            concrete_object_bboxes,
                        )
                    ],
                    "camera_pose": [],
                    "evidence_ids": evidence_ids,
                    "evidence_policy": (
                        f"{node.get('evidence_policy', '')}:"
                        "focused_multi_anchor_relation_roi"
                    ),
                }
                (tuple_dir / "manifest.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                return result
            return None
        if len(objects) != 1:
            return None
        participants = [subject, objects[0]]
        for provider_index, provider in enumerate(participants):
            counterpart = participants[1 - provider_index]
            for raw in sorted(
                provider.get("evidence", ()),
                key=_focused_evidence_priority,
                reverse=True,
            ):
                roi_path = Path(
                    str(raw.get("relation_roi_image_path", ""))
                ).resolve()
                roi_view_id = str(raw.get("relation_roi_view_id", "")).strip()
                anchor_class = str(
                    raw.get("relation_roi_anchor_class", "")
                ).strip().lower()
                if (
                    not roi_path.is_file()
                    or not roi_view_id
                    or anchor_class
                    != str(counterpart.get("class_label", "")).strip().lower()
                ):
                    continue
                counterpart_record = next((
                    value
                    for value in counterpart.get("evidence", ())
                    if roi_view_id in {
                        str(view_id)
                        for view_id in value.get("source_view_ids", ())
                    }
                ), None)
                if counterpart_record is None:
                    continue
                # Prefer the untouched perspective frame for role identity.
                # Grounding ROIs may contain every candidate/anchor annotation;
                # reusing them can paint labels and boxes over the exact small
                # object Qwen is being asked to classify.
                raw_view_path = Path(str(
                    raw.get("representative_view_image", "")
                )).resolve()
                counterpart_view_path = Path(str(
                    counterpart_record.get("representative_view_image", "")
                )).resolve()
                raw_view_bbox = raw.get("representative_bbox_xyxy")
                counterpart_view_bbox = counterpart_record.get(
                    "representative_bbox_xyxy"
                )
                use_raw_view = bool(
                    raw_view_path.is_file()
                    and raw_view_path == counterpart_view_path
                    and str(raw.get("representative_view_id", "")) == roi_view_id
                    and str(counterpart_record.get(
                        "representative_view_id", ""
                    )) == roi_view_id
                    and isinstance(raw_view_bbox, Sequence)
                    and len(raw_view_bbox) == 4
                    and isinstance(counterpart_view_bbox, Sequence)
                    and len(counterpart_view_bbox) == 4
                )
                roi = cv2.imread(
                    str(raw_view_path if use_raw_view else roi_path),
                    cv2.IMREAD_COLOR,
                )
                if roi is None:
                    continue
                height, width = roi.shape[:2]
                if use_raw_view:
                    try:
                        candidate_bbox = [
                            int(round(float(value))) for value in raw_view_bbox
                        ]
                        anchor_bbox = [
                            int(round(float(value)))
                            for value in counterpart_view_bbox
                        ]
                    except (TypeError, ValueError):
                        continue
                else:
                    candidate_bbox = _pixel_bbox(
                        raw.get(
                            "relation_roi_subject_bbox_xyxy_normalized"
                        ),
                        width=width,
                        height=height,
                    )
                    anchor_bbox = _pixel_bbox(
                        raw.get("relation_roi_anchor_bbox_xyxy_normalized"),
                        width=width,
                        height=height,
                    )
                if candidate_bbox is None or anchor_bbox is None:
                    continue
                if provider_index == 0:
                    subject_bbox, object_bbox = candidate_bbox, anchor_bbox
                else:
                    subject_bbox, object_bbox = anchor_bbox, candidate_bbox
                subject_record = (
                    "subject", int(subject["object_id"]), subject_bbox
                )
                object_record = (
                    "object_1", int(objects[0]["object_id"]), object_bbox
                )
                image_specs = []
                for role, annotations, focus_bboxes in (
                    ("subject", [subject_record], [subject_bbox]),
                    ("objects", [object_record], [object_bbox]),
                    (
                        "joint_context",
                        [subject_record, object_record],
                        [subject_bbox, object_bbox],
                    ),
                ):
                    image = _role_crop(
                        roi,
                        annotations,
                        focus_bboxes,
                        predicate=str(node["predicate"]).lower(),
                    )
                    image = _patch_aligned(image, self.maximum_pixels)
                    path = tuple_dir / f"{role}.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(
                            "relation_evidence_image_write_failed"
                        )
                    image_specs.append({
                        "role": role,
                        "image_path": str(path),
                        "preprocessed_once": True,
                    })

                acquisition_id = roi_view_id.split("__", 1)[0]
                source_view_ids = [
                    str(value)
                    for value in raw.get("source_view_ids", ())
                ]
                source_poses = raw.get("source_cam2w_maps", ())
                camera_pose: list[float] = []
                try:
                    pose_index = source_view_ids.index(roi_view_id)
                    camera_pose = [
                        float(value)
                        for row in source_poses[pose_index]
                        for value in row
                    ]
                except (ValueError, IndexError, TypeError):
                    camera_pose = []
                evidence_ids = list(dict.fromkeys((
                    f"{acquisition_id}:{raw.get('observation_id', '')}",
                    f"{acquisition_id}:{counterpart_record.get('observation_id', '')}",
                )))
                result = {
                    "status": "completed",
                    "reason": None,
                    "jointly_observable": True,
                    "acquisition_id": acquisition_id,
                    "panorama_path": None,
                    "focused_relation_roi_path": str(roi_path),
                    "pixel_source": (
                        "original_perspective_view"
                        if use_raw_view
                        else "original_focused_relation_roi"
                    ),
                    "images": image_specs,
                    "subject_grounding": {
                        "object_id": int(subject["object_id"]),
                        "instance_version": int(
                            subject.get("instance_version", 0)
                        ),
                        "acquisition_id": acquisition_id,
                        "observation_id": str(
                            raw.get("observation_id", "")
                            if provider_index == 0
                            else counterpart_record.get("observation_id", "")
                        ),
                        "focused_roi_bbox_xyxy": subject_bbox,
                        "image_width": width,
                        "image_height": height,
                    },
                    "object_groundings": [{
                        "object_id": int(objects[0]["object_id"]),
                        "instance_version": int(
                            objects[0].get("instance_version", 0)
                        ),
                        "acquisition_id": acquisition_id,
                        "observation_id": str(
                            counterpart_record.get("observation_id", "")
                            if provider_index == 0
                            else raw.get("observation_id", "")
                        ),
                        "focused_roi_bbox_xyxy": object_bbox,
                        "image_width": width,
                        "image_height": height,
                    }],
                    "camera_pose": camera_pose,
                    "evidence_ids": evidence_ids,
                    "evidence_policy": (
                        f"{node.get('evidence_policy', '')}:focused_relation_roi"
                    ),
                }
                (tuple_dir / "manifest.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                return result
        return None

    def build(
        self,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        import cv2

        participants = [subject, *objects]
        evidence_maps = [_usable_evidence(value) for value in participants]
        common = set(evidence_maps[0]) if evidence_maps else set()
        for values in evidence_maps[1:]:
            common.intersection_update(values)
        predicate = str(node["predicate"]).lower()
        tuple_name = "_".join((
            re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(node["id"])),
            f"s{int(subject['object_id'])}",
            *(f"o{int(value['object_id'])}" for value in objects),
        ))
        tuple_dir = self.output_dir / tuple_name
        tuple_dir.mkdir(parents=True, exist_ok=True)
        focused = self._focused_relation_roi(
            node=node,
            subject=subject,
            objects=objects,
            tuple_dir=tuple_dir,
        )
        if focused is not None:
            return focused
        shared_perspective = self._shared_perspective_view(
            node=node,
            subject=subject,
            objects=objects,
            tuple_dir=tuple_dir,
        )
        if shared_perspective is not None:
            return shared_perspective
        if not common:
            result = {
                "status": "incomplete",
                "reason": "relation_arguments_not_jointly_observable",
                "jointly_observable": False,
                "images": [],
                "subject_grounding": {},
                "object_groundings": [],
                "camera_pose": [],
                "evidence_ids": [],
            }
            (tuple_dir / "manifest.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result

        def acquisition_score(evidence_view_key: str) -> tuple[float, int, str]:
            records = [values[evidence_view_key] for values in evidence_maps]
            return (
                sum(float(value.get("geometry_confidence", 0.0)) for value in records),
                sum(len(value.get("source_view_ids", ())) for value in records),
                evidence_view_key,
            )

        evidence_view_key = max(common, key=acquisition_score)
        selected = [values[evidence_view_key] for values in evidence_maps]
        common_source_views: set[str] = set()
        if predicate == "between":
            common_source_views = {
                str(value)
                for value in selected[0].get("source_view_ids", ())
            }
            for record in selected[1:]:
                common_source_views.intersection_update(
                    str(value) for value in record.get("source_view_ids", ())
                )
            # A stitched panorama is one simultaneous station observation.
            # Requiring every role to fall inside the same perspective tile
            # discarded valid wide-baseline BETWEEN tuples even though all
            # object-ID masks were present in the same panorama.  Preserve
            # tile co-visibility as provenance, but let the role-grounded
            # panorama and map-frame geometry provide the actual evidence.
        panorama_paths = {str(value["panorama_path"]) for value in selected}
        if len(panorama_paths) != 1:
            result = {
                "status": "incomplete",
                "reason": "joint_acquisition_panorama_mismatch",
                "jointly_observable": False,
                "images": [],
                "subject_grounding": {},
                "object_groundings": [],
                "camera_pose": [],
                "evidence_ids": [],
            }
            (tuple_dir / "manifest.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result
        panorama_path = Path(next(iter(panorama_paths)))
        acquisition_id = panorama_path.parent.name
        panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
        masks = [
            cv2.imread(str(value["mask_path"]), cv2.IMREAD_GRAYSCALE)
            for value in selected
        ]
        if panorama is None or any(value is None for value in masks):
            result = {
                "status": "invalid",
                "state": INVALID,
                "reason": "relation_visual_evidence_unreadable",
                "jointly_observable": False,
                "images": [],
                "subject_grounding": {},
                "object_groundings": [],
                "camera_pose": [],
                "evidence_ids": [],
            }
            (tuple_dir / "manifest.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result
        if any(value.shape[:2] != panorama.shape[:2] for value in masks):
            result = {
                "status": "invalid",
                "state": INVALID,
                "reason": "relation_mask_coordinate_frame_or_shape_invalid",
                "jointly_observable": False,
                "mask_coordinate_frame": MASK_PANORAMA_PIXELS,
                "panorama_shape": list(panorama.shape[:2]),
                "mask_shapes": [list(value.shape[:2]) for value in masks],
                "images": [],
                "subject_grounding": {},
                "object_groundings": [],
                "camera_pose": [],
                "evidence_ids": [],
            }
            (tuple_dir / "manifest.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result
        bboxes = [_mask_bbox(value) for value in masks]
        if any(value is None for value in bboxes):
            result = {
                "status": "invalid",
                "state": INVALID,
                "reason": "relation_mask_empty",
                "jointly_observable": False,
                "images": [],
                "subject_grounding": {},
                "object_groundings": [],
                "camera_pose": [],
                "evidence_ids": [],
            }
            (tuple_dir / "manifest.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            return result
        concrete_bboxes = [list(value) for value in bboxes if value is not None]
        height, width = panorama.shape[:2]
        records = [
            ("subject", int(subject["object_id"]), selected[0]),
            *(
                (f"object_{index}", int(obj["object_id"]), selected[index])
                for index, obj in enumerate(objects, start=1)
            ),
        ]
        subject_bbox = _expanded_bbox(
            concrete_bboxes[0],
            width=width,
            height=height,
            predicate=predicate,
            full_context=False,
        )
        object_union = _union_bbox(concrete_bboxes[1:])
        object_bbox = _expanded_bbox(
            object_union,
            width=width,
            height=height,
            predicate=predicate,
            full_context=False,
        )
        joint_bbox = _expanded_bbox(
            _union_bbox(concrete_bboxes),
            width=width,
            height=height,
            predicate=predicate,
            full_context=predicate in {"near", "between", "closest", "farthest"},
        )
        image_specs = []
        for role, bbox, visible_records in (
            ("subject", subject_bbox, records[:1]),
            ("objects", object_bbox, records[1:]),
            ("joint_context", joint_bbox, records),
        ):
            image = _annotated_crop(panorama, visible_records, bbox)
            image = _patch_aligned(image, self.maximum_pixels)
            path = tuple_dir / f"{role}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("relation_evidence_image_write_failed")
            image_specs.append({
                "role": role,
                "image_path": str(path),
                "preprocessed_once": True,
            })

        def grounding(
            obj: Mapping[str, Any],
            record: Mapping[str, Any],
            bbox: Sequence[int],
        ) -> dict[str, Any]:
            return {
                "object_id": int(obj["object_id"]),
                "instance_version": int(obj.get("instance_version", 0)),
                "acquisition_id": acquisition_id,
                "observation_id": str(record.get("observation_id", "")),
                "mask_path": str(record["mask_path"]),
                "panorama_bbox_xyxy": [int(value) for value in bbox],
            }

        camera_pose: list[float] = []
        source_poses = selected[0].get("source_cam2w_maps", ())
        if source_poses:
            try:
                camera_pose = [
                    float(value)
                    for row in source_poses[0]
                    for value in row
                ]
            except (TypeError, ValueError):
                camera_pose = []
        evidence_ids = [
            f"{acquisition_id}:{record.get('observation_id', '')}"
            for record in selected
        ]
        result = {
            "status": "completed",
            "reason": None,
            "jointly_observable": True,
            "same_perspective_source_view_ids": sorted(common_source_views),
            "acquisition_id": acquisition_id,
            "panorama_path": str(panorama_path),
            "images": image_specs,
            "subject_grounding": grounding(
                subject, selected[0], concrete_bboxes[0]
            ),
            "object_groundings": [
                grounding(obj, selected[index], concrete_bboxes[index])
                for index, obj in enumerate(objects, start=1)
            ],
            "camera_pose": camera_pose,
            "evidence_ids": evidence_ids,
            "evidence_policy": str(node.get("evidence_policy", "")),
        }
        (tuple_dir / "manifest.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result


class QwenRelationTupleVerifier:
    """Call Qwen once per persistent-ID tuple and preserve UNKNOWN failures."""

    def __init__(
        self,
        *,
        endpoint: str,
        episode_id: str,
        acquisition_id: str,
        world_snapshot_version: int,
        evidence_builder: RelationEvidenceBuilder,
        socket_request: Callable[[str, dict, float], dict],
        deadline_unix: float | None,
        answer_reserve_seconds: float,
        station_id: str = "",
        identity_version: int = 0,
        geometry_version: int = 0,
    ) -> None:
        self.endpoint = str(endpoint)
        self.episode_id = str(episode_id)
        self.acquisition_id = str(acquisition_id)
        self.world_snapshot_version = int(world_snapshot_version)
        self.evidence_builder = evidence_builder
        self.socket_request = socket_request
        self.deadline_unix = deadline_unix
        self.answer_reserve_seconds = max(0.0, float(answer_reserve_seconds))
        self.station_id = str(station_id or acquisition_id)
        self.identity_version = max(0, int(identity_version))
        self.geometry_version = max(0, int(geometry_version))
        self.batch_call_count = 0
        self.batch_tuple_count = 0
        self.request_count = 0
        self.cache_hit_count = 0

    def _unknown(
        self,
        reason: str,
        *,
        evidence_ids: Sequence[str] = (),
        geometry: Mapping[str, Any] | None = None,
        qwen: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "state": UNKNOWN,
            "reason_code": str(reason),
            "evidence_ids": list(dict.fromkeys(str(value) for value in evidence_ids)),
            "geometry": dict(geometry or {}),
            "qwen": dict(qwen or {}),
        }
        result.update(relation_record_metadata(
            state=UNKNOWN,
            station_id=self.station_id,
            timestamp=time.time(),
            source_observation_ids=result["evidence_ids"],
            identity_version=self.identity_version,
            geometry_version=self.geometry_version,
        ))
        return result

    def _invalid(
        self,
        reason: str,
        *,
        evidence_ids: Sequence[str] = (),
        geometry: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "state": INVALID,
            "reason_code": str(reason),
            "evidence_ids": list(dict.fromkeys(str(value) for value in evidence_ids)),
            "geometry": dict(geometry or {}),
            "qwen": dict(details or {}),
        }
        result.update(relation_record_metadata(
            state=INVALID,
            station_id=self.station_id,
            timestamp=time.time(),
            source_observation_ids=result["evidence_ids"],
            identity_version=self.identity_version,
            geometry_version=self.geometry_version,
        ))
        return result

    def _decorate_result(
        self,
        result: Mapping[str, Any],
        *,
        evidence: Mapping[str, Any] | None = None,
        subject: Mapping[str, Any] | None = None,
        objects: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        decorated = dict(result)
        source_ids: list[str] = [
            str(value) for value in decorated.get(
                "source_observation_ids", ()
            )
            if str(value).strip()
        ]
        source_ids.extend(
            str(value) for value in decorated.get("evidence_ids", ())
            if str(value).strip()
        )
        for participant in (subject, *objects):
            if not isinstance(participant, Mapping):
                continue
            for raw in participant.get("evidence", ()):
                if not isinstance(raw, Mapping):
                    continue
                observation_id = str(raw.get("observation_id", "")).strip()
                if observation_id:
                    source_ids.append(observation_id)
        station_id = str(
            (evidence or {}).get("station_id", "")
            or self.station_id
        )
        decorated.update(relation_record_metadata(
            state=decorated.get("state", UNKNOWN),
            station_id=station_id,
            timestamp=time.time(),
            source_observation_ids=source_ids,
            identity_version=self.identity_version,
            geometry_version=self.geometry_version,
        ))
        decorated["source_observation_ids"] = list(dict.fromkeys(source_ids))
        return decorated

    def _world_first(
        self,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
        geometry: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        state, reason = _geometry_consistency(
            node, geometry, subject, objects
        )
        # A geometric contradiction can close the spatial predicate without a
        # visual call.  Geometric support cannot validate that the grounded
        # participants have the requested semantic roles, so a YES must pass
        # through the structured role audit before it becomes relation
        # evidence.
        if state != NO or reason == "geometry_not_required_for_predicate":
            return None
        return self._decorate_result({
            "state": state,
            "relation_probability": 0.96 if state == YES else 0.04,
            "reason_code": f"world_geometry:{reason}",
            "evidence_ids": [],
            "geometry": dict(geometry),
            "qwen": {},
            "evidence_source": "world_geometry",
        }, subject=subject, objects=objects)

    @staticmethod
    def _specific_role_contradiction(
        reason_code: str,
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Accept role NO only when Qwen names a different visible class."""
        normalized = re.sub(
            r"[^a-z0-9_]+", "_", str(reason_code).strip().lower()
        )
        if "_is_" not in normalized or "_not_" in normalized:
            return False
        assertions: list[tuple[str, str]] = []
        subject_match = re.search(
            r"(?:^|_)subject_is_(.+?)(?=_object_\d+_is_|$)",
            normalized,
        )
        if subject_match is not None:
            assertions.append((
                str(subject.get("class_label", "")),
                subject_match.group(1),
            ))
        for match in re.finditer(
            r"(?:^|_)object_(\d+)_is_(.+?)(?=_object_\d+_is_|_subject_is_|$)",
            normalized,
        ):
            role_number = int(match.group(1))
            # Qwen prompts label relation objects O1, O2, ... .  Preserve a
            # zero-based spelling as a tolerant alternative without letting
            # an unknown number become evidence.
            index = (
                role_number - 1
                if 1 <= role_number <= len(objects)
                else role_number
                if 0 <= role_number < len(objects)
                else None
            )
            if index is not None:
                assertions.append((
                    str(objects[index].get("class_label", "")),
                    match.group(2),
                ))
        if not assertions:
            return False
        # Spatial explanations such as ``subject_is_resting_on_object_1``
        # describe the predicate, not a competing visual class.  They must not
        # be misparsed as identity contradictions merely because they contain
        # the token ``_is_``.
        spatial_markers = (
            "resting_on",
            "on_object",
            "above_object",
            "below_object",
            "between",
            "outside",
            "inside",
            "near_object",
            "far_from",
            "closest",
            "farthest",
            "touching",
            "overlap",
        )
        for expected_raw, observed_raw in assertions:
            expected = re.sub(
                r"[^a-z0-9]+", "_", expected_raw.strip().lower()
            ).strip("_")
            observed = observed_raw.strip("_")
            # IDs and role numbers are not visible object classes.  Treat a
            # response such as ``object_2_is_8`` as malformed role evidence so
            # the predicate-only reconsideration can run.
            if not re.search(r"[a-z]", observed):
                continue
            if any(marker in observed for marker in spatial_markers):
                continue
            same_class = bool(
                expected
                and observed
                and (
                    observed == expected
                    or observed.endswith("_" + expected)
                    or expected.endswith("_" + observed)
                )
            )
            if expected and observed and not same_class:
                return True
        return False

    @staticmethod
    def _support_claim_geometry_contradiction(
        reason_code: str,
        objects: Sequence[Mapping[str, Any]],
        geometry: Mapping[str, Any],
    ) -> bool:
        """Detect only physically impossible top-support explanations.

        Qwen sometimes writes the persistent object ID and sometimes the O1/O2
        role number in ``subject_resting_on_object_*``.  A subject whose center
        is not above the referenced boundary center cannot be resting on that
        boundary's top.  This invalidates that explanation only; it does not
        assert that BETWEEN is true.
        """
        normalized = re.sub(
            r"[^a-z0-9_]+", "_", str(reason_code).strip().lower()
        )
        match = re.search(r"resting_on_object_(\d+)", normalized)
        if match is None:
            generic_support_claim = (
                "top_support" in normalized
                or "resting_on_boundary" in normalized
                or "supported_by_boundary" in normalized
                or (
                    "support" in normalized
                    and (
                        "boundary" in normalized
                        or "object" in normalized
                    )
                )
            )
            conditions = geometry.get(
                "top_support_necessary_conditions", ()
            )
            if not generic_support_claim:
                return False
            if geometry.get(
                "top_support_possible_on_any_boundary"
            ) is False:
                return True
            if not isinstance(conditions, (list, tuple)) or not conditions:
                return False
            return not any(
                item.get("subject_above_boundary_center") is True
                and item.get("horizontal_image_overlap") is True
                for item in conditions
                if isinstance(item, Mapping)
            )
        deltas = geometry.get("vertical_center_deltas_m", ())
        if not isinstance(deltas, (list, tuple)) or len(deltas) != len(objects):
            return False
        reference = int(match.group(1))
        index = next((
            position for position, obj in enumerate(objects)
            if int(obj.get("object_id", -1)) == reference
        ), None)
        if index is None and 1 <= reference <= len(objects):
            index = reference - 1
        if index is None and 0 <= reference < len(objects):
            index = reference
        if index is None:
            return False
        try:
            delta = float(deltas[index])
        except (TypeError, ValueError, IndexError):
            return False
        return math.isfinite(delta) and delta <= 0.0

    def __call__(
        self,
        node: Mapping[str, Any],
        subject: Mapping[str, Any],
        objects: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        geometry = relation_geometry_diagnostic(node, subject, objects)
        world_result = self._world_first(node, subject, objects, geometry)
        if world_result is not None:
            return world_result
        evidence = self.evidence_builder.build(node, subject, objects)
        _attach_shared_image_layout(geometry, node, evidence, objects)
        predicate = str(node.get("predicate", "")).lower()
        if str(evidence.get("state", "")).upper() == INVALID:
            return self._invalid(
                str(evidence.get("reason", "relation_visual_evidence_invalid")),
                evidence_ids=evidence.get("evidence_ids", ()),
                geometry=geometry,
                details=evidence,
            )
        if evidence.get("jointly_observable") is not True:
            return self._unknown(
                str(evidence.get("reason", "relation_joint_evidence_missing")),
                evidence_ids=evidence.get("evidence_ids", ()),
                geometry=geometry,
            )
        timeout = 180.0
        if self.deadline_unix is not None and self.deadline_unix > 0.0:
            available = (
                self.deadline_unix - time.monotonic() - self.answer_reserve_seconds
            )
            if available <= 0.0:
                return self._unknown(
                    "relation_verification_episode_budget_exhausted",
                    evidence_ids=evidence.get("evidence_ids", ()),
                    geometry=geometry,
                )
            timeout = min(timeout, max(1.0, available))

        request_id = uuid.uuid4().hex
        parameter_roles = tuple(str(value) for value in node["parameter_roles"])
        descriptions = [
            str(value) for value in node.get("argument_descriptions", ())
        ]
        if str(node.get("predicate", "")).lower() == "between":
            subject_bbox = evidence.get("subject_grounding", {}).get(
                "focused_roi_bbox_xyxy"
            )
            object_groundings = evidence.get("object_groundings", ())
            deltas = geometry.get("vertical_center_deltas_m", ())
            if (
                isinstance(subject_bbox, (list, tuple))
                and len(subject_bbox) == 4
                and isinstance(deltas, (list, tuple))
                and len(deltas) == len(objects)
                and len(object_groundings) == len(objects)
            ):
                conditions = []
                for index, (obj, grounding) in enumerate(zip(
                    objects, object_groundings
                )):
                    object_bbox = grounding.get("focused_roi_bbox_xyxy")
                    horizontal_overlap = None
                    if isinstance(object_bbox, (list, tuple)) and len(object_bbox) == 4:
                        horizontal_overlap = bool(
                            max(float(subject_bbox[0]), float(object_bbox[0]))
                            < min(float(subject_bbox[2]), float(object_bbox[2]))
                        )
                    conditions.append({
                        "object_id": int(obj["object_id"]),
                        "subject_above_boundary_center": bool(
                            float(deltas[index]) > 0.0
                        ),
                        "horizontal_image_overlap": horizontal_overlap,
                    })
                geometry["top_support_necessary_conditions"] = conditions
                geometry["top_support_possible_on_any_boundary"] = any(
                    item["subject_above_boundary_center"] is True
                    and item["horizontal_image_overlap"] is True
                    for item in conditions
                )
                object_center_x = [
                    0.5 * (
                        float(grounding["focused_roi_bbox_xyxy"][0])
                        + float(grounding["focused_roi_bbox_xyxy"][2])
                    )
                    for grounding in object_groundings
                    if isinstance(
                        grounding.get("focused_roi_bbox_xyxy"),
                        (list, tuple),
                    )
                    and len(grounding["focused_roi_bbox_xyxy"]) == 4
                ]
                if len(object_center_x) == 2:
                    subject_center_x = 0.5 * (
                        float(subject_bbox[0]) + float(subject_bbox[2])
                    )
                    interval_low = min(object_center_x)
                    interval_high = max(object_center_x)
                    geometry["shared_image_horizontal_order"] = {
                        "subject_center_x": subject_center_x,
                        "object_center_x": object_center_x,
                        "subject_center_between_object_centers": bool(
                            interval_low <= subject_center_x <= interval_high
                        ),
                        "subject_footprint_intersects_object_center_interval": bool(
                            float(subject_bbox[2]) >= interval_low
                            and float(subject_bbox[0]) <= interval_high
                        ),
                    }
        qwen_geometry_evidence = {
            key: value
            for key, value in geometry.items()
            if key != "answer_authority"
        }
        request = ObjectGroundedRelationRequest(
            request_id=request_id,
            episode_id=self.episode_id,
            query_node_id=str(node["id"]),
            world_snapshot_version=self.world_snapshot_version,
            acquisition_id=str(evidence["acquisition_id"]),
            subject_id=int(subject["object_id"]),
            subject_instance_version=int(subject.get("instance_version", 0)),
            object_ids=tuple(int(value["object_id"]) for value in objects),
            object_instance_versions=tuple(
                int(value.get("instance_version", 0)) for value in objects
            ),
            predicate=str(node["predicate"]),
            parameter_roles=parameter_roles,
            subject_description=descriptions[0],
            object_descriptions=tuple(descriptions[1:]),
            evidence_policy=str(node.get("evidence_policy", "")),
            negative_evidence_policy=str(
                node.get("negative_evidence_policy", "")
            ),
            verification_instruction=str(node.get("qwen_instruction", "")),
            geometry_diagnostic=qwen_geometry_evidence,
            subject_bbox_or_mask=dict(evidence["subject_grounding"]),
            object_bboxes_or_masks=tuple(
                dict(value) for value in evidence["object_groundings"]
            ),
            camera_pose=tuple(float(value) for value in evidence["camera_pose"]),
            evidence_provenance=tuple(
                str(value) for value in evidence["evidence_ids"]
            ),
        )
        stable_acquisition_id = "relation-" + "-".join((
            self.episode_id,
            str(node["id"]),
            str(self.world_snapshot_version),
            f"s{request.subject_id}v{request.subject_instance_version}",
            *(
                f"o{object_id}v{version}"
                for object_id, version in zip(
                    request.object_ids, request.object_instance_versions
                )
            ),
        ))
        try:
            response = self.socket_request(
                self.endpoint,
                {
                    "request_id": request_id,
                    "acquisition_id": stable_acquisition_id,
                    "backend": "qwen3vl",
                    "operation": "verify_relation",
                    "input_handles": [],
                    "parameters": {
                        "grounded_relation_request": request.to_dict(),
                        "relation_images": list(evidence["images"]),
                    },
                },
                timeout,
            )
        except Exception as exc:
            return self._unknown(
                f"qwen_relation_transport:{type(exc).__name__}:{str(exc)[:200]}",
                evidence_ids=evidence["evidence_ids"],
                geometry=geometry,
            )
        if response.get("ok") is not True:
            return self._unknown(
                f"qwen_relation_error:{response.get('error_code', 'unknown')}",
                evidence_ids=evidence["evidence_ids"],
                geometry=geometry,
                qwen=response,
            )
        metadata = dict(response.get("metadata", {}))
        state = str(metadata.get("state", "uncertain")).lower()
        role_states = [
            str(metadata.get("subject_role_state", "UNKNOWN")).upper(),
            *(
                str(value).upper()
                for value in metadata.get("object_role_states", ())
            ),
        ]
        model_reason = str(
            metadata.get("reason_code", "qwen_relation_uncertain")
        )
        support_reason_geometry_invalid = (
            self._support_claim_geometry_contradiction(
                model_reason, objects, geometry
            )
        )
        horizontal_order = geometry.get("shared_image_horizontal_order", {})
        between_refutation_conflicts_with_evidence = bool(
            str(node.get("predicate", "")).lower() == "between"
            and state == "refuted"
            and role_states
            and all(value == "YES" for value in role_states)
            and isinstance(horizontal_order, Mapping)
            and horizontal_order.get(
                "subject_center_between_object_centers"
            ) is True
            and geometry.get("top_support_possible_on_any_boundary") is False
            and isinstance(geometry.get("segment_position"), (int, float))
            and 0.0 < float(geometry["segment_position"]) < 1.0
        )
        role_negative_contract_invalid = bool(
            any(value == "NO" for value in role_states)
            and not self._specific_role_contradiction(
                model_reason, subject, objects
            )
        )
        role_restatement_contract_invalid = bool(
            state in {"supported", "refuted"}
            and role_states
            and all(value == "YES" for value in role_states)
            and "_is_" in model_reason.lower()
            and not self._specific_role_contradiction(
                model_reason, subject, objects
            )
        )
        predicate_contract_invalid = bool(
            state == "refuted"
            and role_states
            and all(value == "YES" for value in role_states)
            and (
                support_reason_geometry_invalid
                or between_refutation_conflicts_with_evidence
            )
        )
        retry_mode = (
            "role" if role_negative_contract_invalid
            else "evidence" if role_restatement_contract_invalid
            else "predicate" if predicate_contract_invalid
            else ""
        )
        retry_contract_invalid = bool(
            role_negative_contract_invalid
            or predicate_contract_invalid
            or role_restatement_contract_invalid
        )
        if retry_contract_invalid:
            if self.deadline_unix is not None and self.deadline_unix > 0.0:
                available = (
                    self.deadline_unix
                    - time.monotonic()
                    - self.answer_reserve_seconds
                )
            else:
                available = timeout
            if available > 0.0:
                retry_id = uuid.uuid4().hex
                horizontal_order = geometry.get(
                    "shared_image_horizontal_order", {}
                )
                horizontal_guidance = ""
                if (
                    str(node.get("predicate", "")).lower() == "between"
                    and isinstance(horizontal_order, Mapping)
                ):
                    horizontal_guidance = (
                        " The shared_image_horizontal_order diagnostic describes "
                        "the same labelled joint image. Absence of boundary "
                        "support does not prove BETWEEN: also verify that the "
                        "subject occupies the interval between both object "
                        "centers. If both reported interval booleans are false, "
                        "the subject is visibly outside that horizontal gap; do "
                        "not return supported unless independent depth evidence "
                        "clearly resolves the apparent ordering."
                    )
                if role_negative_contract_invalid:
                    retry_instruction = (
                        request.verification_instruction
                        + " The previous response violated the role-negative "
                        "contract. Reinspect every grounded ID. A role may be NO "
                        "only when the pixels positively identify a different "
                        "visible class, and reason_code must name that class. A "
                        "phrase such as subject_is_not_<requested_class> is not "
                        "evidence; use UNKNOWN when identity pixels are "
                        "insufficient, otherwise continue to the spatial "
                        "predicate."
                    )
                elif role_restatement_contract_invalid:
                    retry_instruction = (
                        request.verification_instruction
                        + " The previous supported response violated the "
                        "relation-evidence contract because reason_code only "
                        "restated a requested class. Reinspect every grounded ID "
                        "against its supplied visual description and then "
                        "re-evaluate the spatial predicate. Role YES requires "
                        "visible class-defining parts belonging to that same "
                        "grounded instance; a plausible location or silhouette "
                        "is insufficient. A supported reason_code must describe "
                        "the visible spatial relation rather than repeat a class "
                        "name."
                    )
                else:
                    retry_instruction = (
                        request.verification_instruction
                        + " The previous response was contract-invalid: every "
                        "role was YES but the refutation reason was not a valid "
                        "visible predicate contradiction. Re-evaluate only the "
                        "spatial predicate for these same IDs. Inspect the actual "
                        "physical layout and name only a contradiction visible in "
                        "the supplied evidence."
                        + (
                            (
                                " The diagnostic "
                                "top_support_possible_on_any_boundary is false "
                                "for this exact tuple: the subject cannot be "
                                "top-supported by either boundary. Do not use any "
                                "support explanation as a refutation. Inspect "
                                "whether the floor-standing subject occupies the "
                                "intervening gap."
                                if geometry.get(
                                    "top_support_possible_on_any_boundary"
                                ) is False
                                else " The prior top-support explanation "
                                "conflicts with the supplied observations. Do not "
                                "repeat it; inspect the intervening layout."
                            )
                            if support_reason_geometry_invalid
                            else (
                                " The previous BETWEEN refutation conflicts "
                                "with the supplied tuple evidence: the 3D "
                                "segment position lies between O1 and O2, the "
                                "joint image places S between both object "
                                "centers, and no boundary top-support is "
                                "possible. Reinspect the labelled physical "
                                "bodies and reconcile that evidence before "
                                "deciding the predicate. If the subject "
                                "physically occupies the intervening space, "
                                "return supported; if the subject is outside "
                                "that horizontal interval, return refuted with "
                                "the visible ordering contradiction."
                                if between_refutation_conflicts_with_evidence
                                else " Do not restate a correct role class as "
                                "a relation refutation."
                            )
                        )
                        + horizontal_guidance
                    )
                retry_request = replace(
                    request,
                    request_id=retry_id,
                    verification_instruction=retry_instruction,
                )
                try:
                    retry_response = self.socket_request(
                        self.endpoint,
                        {
                            "request_id": retry_id,
                            "acquisition_id": (
                                stable_acquisition_id
                                + (
                                    "-role-retry"
                                    if retry_mode == "role"
                                    else "-evidence-retry"
                                    if retry_mode == "evidence"
                                    else "-predicate-retry"
                                )
                            ),
                            "backend": "qwen3vl",
                            "operation": "verify_relation",
                            "input_handles": [],
                            "parameters": {
                                "grounded_relation_request": (
                                    retry_request.to_dict()
                                ),
                                "relation_images": list(evidence["images"]),
                            },
                        },
                        min(timeout, max(1.0, available)),
                    )
                    if retry_response.get("ok") is True:
                        metadata = dict(
                            retry_response.get("metadata", {})
                        )
                        metadata["contract_retry"] = (
                            retry_mode
                        )
                        state = str(
                            metadata.get("state", "uncertain")
                        ).lower()
                        role_states = [
                            str(metadata.get(
                                "subject_role_state", "UNKNOWN"
                            )).upper(),
                            *(
                                str(value).upper()
                                for value in metadata.get(
                                    "object_role_states", ()
                                )
                            ),
                        ]
                        model_reason = str(metadata.get(
                            "reason_code", "qwen_relation_uncertain"
                        ))
                except Exception:
                    pass
        # When a supported response merely restated the class, the first retry
        # deliberately rechecks identity.  If every exact ID still remains YES,
        # use one concise joint-context pass for the spatial predicate alone.
        # A wrong-class candidate stops before this step.
        evidence_retry_needs_predicate = bool(
            role_states
            and all(value == "YES" for value in role_states)
            and metadata.get("jointly_observable") is True
            and (
                state != "supported"
                or (
                    "_is_" in model_reason.lower()
                    and not self._specific_role_contradiction(
                        model_reason, subject, objects
                    )
                )
            )
        )
        if evidence_retry_needs_predicate:
            if self.deadline_unix is not None and self.deadline_unix > 0.0:
                second_available = (
                    self.deadline_unix
                    - time.monotonic()
                    - self.answer_reserve_seconds
                )
            else:
                second_available = timeout
            if second_available > 0.0:
                second_id = uuid.uuid4().hex
                second_horizontal_guidance = ""
                horizontal_order = geometry.get(
                    "shared_image_horizontal_order", {}
                )
                if (
                    str(node.get("predicate", "")).lower() == "between"
                    and isinstance(horizontal_order, Mapping)
                ):
                    second_horizontal_guidance = (
                        " Use shared_image_horizontal_order to check whether the "
                        "subject occupies the interval between both boundaries; "
                        "absence of support alone does not prove BETWEEN."
                    )
                second_request = replace(
                    request,
                    request_id=second_id,
                    verification_instruction=(
                        request.verification_instruction
                        + " The previous response was contract-invalid: every "
                        "role was YES, but the relation state or reason remained "
                        "unresolved after identity reconsideration. Re-evaluate "
                        "only the spatial predicate for these same IDs. A correct "
                        "role class is not a predicate reason."
                        + second_horizontal_guidance
                    ),
                )
                try:
                    second_response = self.socket_request(
                        self.endpoint,
                        {
                            "request_id": second_id,
                            "acquisition_id": (
                                stable_acquisition_id
                                + "-evidence-predicate-retry"
                            ),
                            "backend": "qwen3vl",
                            "operation": "verify_relation",
                            "input_handles": [],
                            "parameters": {
                                "grounded_relation_request": (
                                    second_request.to_dict()
                                ),
                                "relation_images": list(evidence["images"]),
                            },
                        },
                        min(timeout, max(1.0, second_available)),
                    )
                    if second_response.get("ok") is True:
                        metadata = dict(
                            second_response.get("metadata", {})
                        )
                        metadata["contract_retry"] = (
                            "evidence_then_predicate"
                        )
                        state = str(
                            metadata.get("state", "uncertain")
                        ).lower()
                        role_states = [
                            str(metadata.get(
                                "subject_role_state", "UNKNOWN"
                            )).upper(),
                            *(
                                str(value).upper()
                                for value in metadata.get(
                                    "object_role_states", ()
                                )
                            ),
                        ]
                        model_reason = str(metadata.get(
                            "reason_code", "qwen_relation_uncertain"
                        ))
                except Exception:
                    pass
        support_reason_geometry_invalid = (
            self._support_claim_geometry_contradiction(
                model_reason, objects, geometry
            )
        )
        role_contract_complete = len(role_states) == 1 + len(objects)
        # A role crop is only one noisy semantic observation.  In particular,
        # small books seen edge-on can look like part of the cabinet in a tight
        # crop.  A role disagreement therefore keeps the exact tuple uncertain
        # and eligible for a new view; it is not categorical proof that the
        # detector's class-labelled instance does not exist.
        if not role_contract_complete:
            final_state = "UNKNOWN"
            reason_code = f"qwen_argument_role_contract_incomplete:{model_reason}"
        elif any(value == "NO" for value in role_states):
            final_state = "UNKNOWN"
            reason_code = f"qwen_argument_role_conflict:{model_reason}"
        elif any(value != "YES" for value in role_states):
            final_state = "UNKNOWN"
            reason_code = f"qwen_argument_role_unresolved:{model_reason}"
        elif metadata.get("jointly_observable") is not True:
            final_state = "UNKNOWN"
            reason_code = f"qwen_joint_layout_unavailable:{model_reason}"
        elif state == "supported":
            final_state = "YES"
            reason_code = model_reason
        elif state == "refuted":
            if str(model_reason).strip().upper() in {"", "NO", "UNKNOWN"}:
                final_state = "UNKNOWN"
                reason_code = f"qwen_relation_negative_without_reason:{model_reason}"
            elif support_reason_geometry_invalid:
                final_state = "UNKNOWN"
                reason_code = (
                    "qwen_relation_negative_geometry_inconsistent:"
                    f"{model_reason}"
                )
            elif "_is_" in model_reason.lower() and not self._specific_role_contradiction(
                model_reason, subject, objects
            ):
                final_state = "UNKNOWN"
                reason_code = (
                    "qwen_relation_negative_only_restates_expected_role:"
                    f"{model_reason}"
                )
            else:
                final_state = "NO"
                reason_code = model_reason
        else:
            final_state = "UNKNOWN"
            reason_code = model_reason
        geometry_state, geometry_reason = _geometry_consistency(
            node, geometry, subject, objects, metadata
        )
        if geometry_state == "NO" and final_state != "INVALID":
            # Reliable map geometry is categorical negative evidence even when
            # the visual predicate call is inconclusive.  Leaving this as
            # UNKNOWN would violate the relation contract: an explicit,
            # uncertainty-cleared contradiction is already a NO.
            final_state = "NO"
            reason_code = f"geometry_contradiction_{geometry_reason}"
        elif (
            geometry_state == "YES"
            and role_contract_complete
            and all(value == "YES" for value in role_states)
            and metadata.get("jointly_observable") is True
        ):
            # Preserve world geometry as predicate authority only after Qwen
            # has validated the exact persistent-ID participant roles.
            final_state = "YES"
            reason_code = f"world_geometry_after_role_audit:{geometry_reason}"
        elif final_state == "YES" and geometry_state != "YES":
            final_state = geometry_state
            reason_code = f"qwen_geometry_{geometry_reason}"
        return self._decorate_result({
            "state": final_state,
            "relation_probability": _relation_true_probability(
                node, geometry, subject, objects, final_state, metadata
            ),
            "reason_code": reason_code,
            "evidence_ids": list(evidence["evidence_ids"]),
            "geometry": geometry,
            "qwen": metadata,
            "evidence_source": "qwen_visual",
        }, evidence=evidence, subject=subject, objects=objects)

    def verify_many(
        self,
        items: Sequence[tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Sequence[Mapping[str, Any]],
        ]],
    ) -> list[dict[str, Any]]:
        """Wave 3: verify independent persistent-ID tuples in GPU batches."""
        self.request_count += len(items)
        outputs: list[dict[str, Any] | None] = [None] * len(items)
        prepared: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], Sequence[Mapping[str, Any]]]] = []
        for index, (node, subject, objects) in enumerate(items):
            geometry = relation_geometry_diagnostic(node, subject, objects)
            world_result = self._world_first(node, subject, objects, geometry)
            if world_result is not None:
                outputs[index] = world_result
                continue
            evidence = self.evidence_builder.build(node, subject, objects)
            _attach_shared_image_layout(geometry, node, evidence, objects)
            if str(evidence.get("state", "")).upper() == INVALID:
                outputs[index] = self._invalid(
                    str(evidence.get("reason", "relation_visual_evidence_invalid")),
                    evidence_ids=evidence.get("evidence_ids", ()),
                    geometry=geometry,
                    details=evidence,
                )
                continue
            if evidence.get("jointly_observable") is not True:
                outputs[index] = self._unknown(
                    str(evidence.get("reason", "relation_joint_evidence_missing")),
                    evidence_ids=evidence.get("evidence_ids", ()),
                    geometry=geometry,
                )
                continue
            descriptions = [
                str(value) for value in node.get("argument_descriptions", ())
            ]
            request_id = uuid.uuid4().hex
            request = ObjectGroundedRelationRequest(
                request_id=request_id,
                episode_id=self.episode_id,
                query_node_id=str(node["id"]),
                world_snapshot_version=self.world_snapshot_version,
                acquisition_id=str(evidence["acquisition_id"]),
                subject_id=int(subject["object_id"]),
                subject_instance_version=int(subject.get("instance_version", 0)),
                object_ids=tuple(int(value["object_id"]) for value in objects),
                object_instance_versions=tuple(
                    int(value.get("instance_version", 0)) for value in objects
                ),
                predicate=str(node["predicate"]),
                parameter_roles=tuple(
                    str(value) for value in node["parameter_roles"]
                ),
                subject_description=descriptions[0],
                object_descriptions=tuple(descriptions[1:]),
                evidence_policy=str(node.get("evidence_policy", "")),
                negative_evidence_policy=str(
                    node.get("negative_evidence_policy", "")
                ),
                verification_instruction=str(node.get("qwen_instruction", "")),
                geometry_diagnostic={
                    key: value for key, value in geometry.items()
                    if key != "answer_authority"
                },
                subject_bbox_or_mask=dict(evidence["subject_grounding"]),
                object_bboxes_or_masks=tuple(
                    dict(value) for value in evidence["object_groundings"]
                ),
                camera_pose=tuple(float(value) for value in evidence["camera_pose"]),
                evidence_provenance=tuple(
                    str(value) for value in evidence["evidence_ids"]
                ),
            )
            prepared.append((
                index,
                dict(node),
                dict(subject),
                {
                    "grounded_relation_request": request.to_dict(),
                    "relation_images": list(evidence["images"]),
                },
                list(objects),
            ))
        for start in range(0, len(prepared), 8):
            chunk = prepared[start:start + 8]
            def call_pass(
                values: Sequence[tuple[
                    int,
                    dict[str, Any],
                    dict[str, Any],
                    dict[str, Any],
                    Sequence[Mapping[str, Any]],
                ]],
                pass_kind: str,
            ) -> list[dict[str, Any]]:
                if not values:
                    return []
                if self.deadline_unix is not None and self.deadline_unix > 0.0:
                    available = (
                        self.deadline_unix - time.monotonic()
                        - self.answer_reserve_seconds
                    )
                    if available <= 0.0:
                        return [{
                            "ok": False,
                            "error_code": (
                                "relation_verification_episode_budget_exhausted"
                            ),
                            "metadata": {},
                        }] * len(values)
                    timeout = min(180.0, max(1.0, available))
                else:
                    timeout = 180.0
                batch_id = uuid.uuid4().hex
                try:
                    self.batch_call_count += 1
                    self.batch_tuple_count += len(values)
                    response = self.socket_request(
                        self.endpoint,
                        {
                            "request_id": batch_id,
                            "acquisition_id": (
                                f"relation-{pass_kind}-wave-{batch_id}"
                            ),
                            "backend": "qwen3vl",
                            "operation": "verify_relation_batch",
                            "input_handles": [],
                            "parameters": {
                                "requests": [value[3] for value in values],
                                "verification_pass": pass_kind,
                            },
                        },
                        timeout,
                    )
                    rows = list(
                        response.get("metadata", {}).get("results", ())
                    )
                except Exception as exc:
                    rows = [{
                        "ok": False,
                        "error_code": type(exc).__name__,
                        "metadata": {"error_detail": str(exc)[:300]},
                    }] * len(values)
                if len(rows) != len(values):
                    return [{
                        "ok": False,
                        "error_code": "relation_batch_length_mismatch",
                        "metadata": {},
                    }] * len(values)
                return [dict(value) for value in rows]

            role_rows = call_pass(chunk, "roles")
            predicate_chunk = []
            role_metadata_by_index: dict[int, dict[str, Any]] = {}
            for (
                index,
                node,
                subject,
                payload,
                objects,
            ), role_row in zip(chunk, role_rows):
                evidence_ids = payload[
                    "grounded_relation_request"
                ]["evidence_provenance"]
                geometry = dict(payload["grounded_relation_request"].get(
                    "geometry_diagnostic", {}
                ))
                geometry.setdefault("answer_authority", False)
                if role_row.get("ok") is not True:
                    outputs[index] = self._unknown(
                        "qwen_role_audit_error:"
                        f"{role_row.get('error_code', 'unknown')}",
                        evidence_ids=evidence_ids,
                        geometry=geometry,
                        qwen={"role_audit": role_row},
                    )
                    continue
                role_metadata = dict(role_row.get("metadata", {}))
                audited_states = [
                    str(role_metadata.get(
                        "subject_role_state", "UNKNOWN"
                    )).upper(),
                    *(
                        str(value).upper()
                        for value in role_metadata.get(
                            "object_role_states", ()
                        )
                    ),
                ]
                role_reason = str(
                    role_metadata.get(
                        "reason_code", "role_identity_uncertain"
                    )
                )
                role_qwen = {
                    **role_metadata,
                    "role_audit": dict(role_metadata),
                }
                if len(audited_states) != 1 + len(objects):
                    outputs[index] = self._unknown(
                        "qwen_role_audit_contract_incomplete:"
                        + role_reason,
                        evidence_ids=evidence_ids,
                        geometry=geometry,
                        qwen=role_qwen,
                    )
                    continue
                if any(value == "NO" for value in audited_states):
                    outputs[index] = self._unknown(
                        "qwen_role_audit_conflict:" + role_reason,
                        evidence_ids=evidence_ids,
                        geometry=geometry,
                        qwen=role_qwen,
                    )
                    continue
                if any(value != "YES" for value in audited_states):
                    outputs[index] = self._unknown(
                        "qwen_role_audit_unresolved:" + role_reason,
                        evidence_ids=evidence_ids,
                        geometry=geometry,
                        qwen=role_qwen,
                    )
                    continue
                geometry_state, geometry_reason = _geometry_consistency(
                    node, geometry, subject, objects
                )
                if geometry_state == YES:
                    outputs[index] = self._decorate_result({
                        "state": YES,
                        "relation_probability": 0.96,
                        "reason_code": (
                            "world_geometry_after_role_audit:"
                            + geometry_reason
                        ),
                        "evidence_ids": list(evidence_ids),
                        "geometry": geometry,
                        "qwen": {
                            **role_metadata,
                            "role_audit": dict(role_metadata),
                        },
                        "evidence_source": (
                            "world_geometry_with_qwen_role_audit"
                        ),
                    }, subject=subject, objects=objects)
                    continue
                role_metadata_by_index[index] = role_metadata
                predicate_chunk.append((
                    index, node, subject, payload, objects
                ))

            predicate_rows = call_pass(predicate_chunk, "predicate")
            for (
                index,
                node,
                subject,
                payload,
                objects,
            ), row in zip(predicate_chunk, predicate_rows):
                evidence_ids = payload[
                    "grounded_relation_request"
                ]["evidence_provenance"]
                geometry = dict(payload["grounded_relation_request"].get(
                    "geometry_diagnostic", {}
                ))
                geometry.setdefault("answer_authority", False)
                role_metadata = role_metadata_by_index[index]
                if row.get("ok") is not True:
                    outputs[index] = self._unknown(
                        "qwen_relation_error:"
                        f"{row.get('error_code', 'unknown')}",
                        evidence_ids=evidence_ids,
                        geometry=geometry,
                        qwen={
                            "role_audit": role_metadata,
                            "predicate_verification": row,
                        },
                    )
                    continue
                predicate_metadata = dict(row.get("metadata", {}))
                metadata = {
                    **predicate_metadata,
                    "subject_role_state": role_metadata[
                        "subject_role_state"
                    ],
                    "object_role_states": list(
                        role_metadata["object_role_states"]
                    ),
                    "role_audit": role_metadata,
                    "predicate_role_report": {
                        "subject_role_state": predicate_metadata.get(
                            "subject_role_state", "UNKNOWN"
                        ),
                        "object_role_states": list(
                            predicate_metadata.get(
                                "object_role_states", ()
                            )
                        ),
                    },
                }
                role_states = [
                    str(metadata.get("subject_role_state", "UNKNOWN")).upper(),
                    *(
                        str(value).upper()
                        for value in metadata.get("object_role_states", ())
                    ),
                ]
                model_state = str(metadata.get("state", "uncertain")).lower()
                reason = str(metadata.get("reason_code", "qwen_relation_uncertain"))
                if len(role_states) != 1 + len(objects):
                    state = "UNKNOWN"
                    reason = f"qwen_argument_role_contract_incomplete:{reason}"
                elif any(value == "NO" for value in role_states):
                    state = "UNKNOWN"
                    reason = f"qwen_argument_role_conflict:{reason}"
                elif any(value != "YES" for value in role_states):
                    state = "UNKNOWN"
                    reason = f"qwen_argument_role_unresolved:{reason}"
                elif metadata.get("jointly_observable") is not True:
                    state = "UNKNOWN"
                    reason = f"qwen_joint_layout_unavailable:{reason}"
                elif model_state == "supported":
                    state = "YES"
                elif model_state == "refuted":
                    # Keep the batch path's tri-state contract identical to
                    # the single-tuple verifier.  A malformed/role-only
                    # refutation is not reliable negative spatial evidence;
                    # it remains UNKNOWN until a valid predicate contradiction
                    # is observed.
                    support_reason_geometry_invalid = (
                        self._support_claim_geometry_contradiction(
                            reason, objects, geometry
                        )
                    )
                    horizontal_order = geometry.get(
                        "shared_image_horizontal_order", {}
                    )
                    between_refutation_conflict = bool(
                        str(node.get("predicate", "")).lower() == "between"
                        and isinstance(horizontal_order, Mapping)
                        and horizontal_order.get(
                            "subject_center_between_object_centers"
                        ) is True
                        and geometry.get(
                            "top_support_possible_on_any_boundary"
                        ) is False
                        and isinstance(geometry.get("segment_position"), (int, float))
                        and 0.0 < float(geometry["segment_position"]) < 1.0
                    )
                    if str(reason).strip().upper() in {"", "NO", "UNKNOWN"}:
                        state = "UNKNOWN"
                        reason = f"qwen_relation_negative_without_reason:{reason}"
                    elif support_reason_geometry_invalid or between_refutation_conflict:
                        state = "UNKNOWN"
                        reason = f"qwen_relation_negative_geometry_inconsistent:{reason}"
                    elif "_is_" in reason.lower() and not self._specific_role_contradiction(
                        reason, subject, objects
                    ):
                        state = "UNKNOWN"
                        reason = (
                            "qwen_relation_negative_only_restates_expected_role:"
                            f"{reason}"
                        )
                    else:
                        state = "NO"
                else:
                    state = "UNKNOWN"
                geometry_state, geometry_reason = _geometry_consistency(
                    node, geometry, subject, objects, metadata
                )
                if geometry_state == "NO":
                    # Keep the batch path identical to the single-tuple path:
                    # an explicit reliable geometric contradiction is NO, not
                    # an UNKNOWN caused by an inconclusive visual predicate.
                    state = "NO"
                    reason = f"geometry_contradiction_{geometry_reason}"
                elif state == "YES" and geometry_state != "YES":
                    state = geometry_state
                    reason = f"qwen_geometry_{geometry_reason}"
                outputs[index] = self._decorate_result({
                    "state": state,
                    "relation_probability": _relation_true_probability(
                        node, geometry, subject, objects, state, metadata
                    ),
                    "reason_code": reason,
                    "evidence_ids": list(evidence_ids),
                    "geometry": geometry,
                    "qwen": metadata,
                    "evidence_source": "qwen_visual_batch",
                }, subject=subject, objects=objects)
        return [
            value if value is not None else self._unknown(
                "relation_batch_result_missing"
            )
            for value in outputs
        ]
