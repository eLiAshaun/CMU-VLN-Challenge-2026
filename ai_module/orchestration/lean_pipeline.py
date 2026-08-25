"""Lean production orchestration for one CMU-VLN acquisition.

This module intentionally contains only the real online path:

    task-conditioned Qwen grounding -> SAM2 masks -> LiDAR map-frame lift
    -> persistent SceneMemory -> query execution -> waypoint projection
    -> RootFinalizer.

It does not own a second episode clock, a second scene memory, a mock sensor
stage, or a substitute decision schema.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from integrations.execution.query_domain import attach_query_domain
from integrations.execution.count_query_executor import (
    assess_count_query_domain,
    execute_count_query_graph,
    finalize_count_query_execution_domain,
)
from integrations.execution.query_executor import (
    evaluate_required_relation_tuples,
)
from integrations.execution.evidence_acquisition import (
    EvidenceAcquisitionCoordinator,
)
from integrations.execution.persistent_relation_evidence import (
    merge_persistent_relation_evidence,
    summarize_persistent_relation_evidence,
)
from integrations.execution.relation_evidence import (
    QwenRelationTupleVerifier,
    RelationEvidenceBuilder,
)
from integrations.execution.relation_engine import RelationEngine
from integrations.execution.resolver_contracts import (
    ExecutionNeed,
    ResolverResult,
    ResolverStatus,
)
from integrations.execution.required_entity_coverage import compute_required_entity_coverage
from integrations.execution.root_finalizer import (
    finalize_resolver_result,
    finalize_system_failure,
)
from integrations.execution.task_resolvers import resolver_for
from integrations.execution.scene_memory import (
    apply_scene_memory_semantic_verifications,
    load_scene_memory_snapshot,
    materialize_query_view,
    relation_role_semantic_verifications,
    save_scene_relation_evidence,
    update_scene_memory,
)
from integrations.execution.trajectory_geometry import compile_directive_geometries
from integrations.model_socket_client import socket_request
from navigation.waypoint_planning import semantic_waypoint_segment_output
from orchestration.contracts import AcquisitionContext, PipelineSummary, TIME_BUDGET
from orchestration.lidar_geometry import lift_detections_to_observations
from orchestration.perception_stage import execute_perception_pipeline


class LeanPipeline:
    """Execute one acquisition against episode-owned persistent state."""

    def __init__(self, runtime_config: Mapping[str, Any] | None = None) -> None:
        self.runtime = dict(runtime_config or {})
        self.ai_root = Path(__file__).resolve().parents[1]

    @staticmethod
    def _elapsed(started: float) -> float:
        return max(0.0, time.monotonic() - started)

    @staticmethod
    def _stage(
        status: str,
        started: float,
        **metadata: Any,
    ) -> dict[str, Any]:
        return {
            "status": str(status),
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            **metadata,
        }

    @staticmethod
    def _failure_decision(
        ctx: AcquisitionContext,
        *,
        reason: str,
        scene_version: int = 0,
    ) -> dict[str, Any]:
        return finalize_system_failure(
            ctx.task_ir,
            reason=reason,
            scene_version=scene_version,
        )

    def _resolved_path(self, value: object) -> Path:
        path = Path(str(value or ""))
        return path if path.is_absolute() else (self.ai_root / path).resolve()

    def _validate_online_inputs(
        self,
        ctx: AcquisitionContext,
        *,
        force_perception: bool = False,
        perception_mode: str | None = None,
    ) -> dict[str, Any]:
        geometry = ctx.competition_geometry
        required: dict[str, Path] = {
            "state_estimation": Path(str(geometry.get("state_estimation_path", ""))),
        }
        observation_required = bool(
            (ctx.station_is_new or force_perception)
            and str(perception_mode or ctx.perception_mode) != "decision_only"
            and ctx.perception_view_limit > 0
        )
        if observation_required:
            required.update({
                "panorama": ctx.image_path,
                "sensor_scan": Path(str(geometry.get("sensor_scan_path", ""))),
            })
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("online_input_missing:" + ",".join(missing))
        return {
            "station_id": ctx.station_id,
            "station_is_new": bool(ctx.station_is_new),
            "perception_mode": str(perception_mode or ctx.perception_mode),
            "force_perception": bool(force_perception),
            "observation_required": observation_required,
            "image_path": str(ctx.image_path),
            "sensor_scan_path": str(geometry.get("sensor_scan_path", "")),
            "registered_scan_path": str(geometry.get("registered_scan_path", "")),
            "state_estimation_path": str(geometry.get("state_estimation_path", "")),
            "terrain_map_path": str(geometry.get("terrain_map_path", "")),
            "terrain_map_ext_path": str(geometry.get("terrain_map_ext_path", "")),
        }

    @staticmethod
    def _frontier_probe_object(
        snapshot: Mapping[str, Any],
        preferred_region_ids: Sequence[str] = (),
        objective: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        preferred = {str(value) for value in preferred_region_ids if str(value)}
        regions = [
            dict(value)
            for value in snapshot.get("observation_frontier_regions", ())
            if isinstance(value, Mapping) and value.get("points_xy")
        ]
        if preferred:
            matching = [value for value in regions if str(value.get("region_id", "")) in preferred]
            # A relation objective can outlive the scene version that created
            # its frontier IDs.  The current SceneMemory frontier is the
            # authoritative geometric domain; stale IDs must not force the
            # pipeline into the old representative-point alternative.
            if matching:
                regions = matching
        if not regions:
            return None

        def finite_xy(value: object) -> list[float] | None:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return None
            if len(value) < 2:
                return None
            try:
                result = [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return None
            return result if all(math.isfinite(component) for component in result) else None

        objects_by_id = {
            int(value["object_id"]): value
            for value in snapshot.get("objects", ())
            if isinstance(value, Mapping)
            and str(value.get("object_id", "")).lstrip("-").isdigit()
        }

        latest_viewpoint = None
        for raw_view in reversed(snapshot.get("viewpoint_history", ())):
            if not isinstance(raw_view, Mapping):
                continue
            latest_viewpoint = finite_xy(raw_view.get("viewpoint_position_map"))
            if latest_viewpoint is not None:
                break
        current_xy = latest_viewpoint or [0.0, 0.0]
        objective = dict(objective or {})

        def object_center(object_id: object) -> list[float] | None:
            try:
                item = objects_by_id.get(int(object_id))
            except (TypeError, ValueError):
                item = None
            if not item:
                return None
            return finite_xy(item.get("center_3d"))

        def hypothesis_centers(
            values: object,
            additional_ids: object = (),
        ) -> list[tuple[int | None, list[float]]]:
            result: list[tuple[int | None, list[float]]] = []
            seen: set[int] = set()
            raw_values = values if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes)
            ) else ()
            for raw in raw_values:
                if isinstance(raw, Mapping):
                    raw_id = raw.get("object_id")
                    center = finite_xy(raw.get("center_xy"))
                else:
                    raw_id = raw
                    center = None
                try:
                    object_id = int(raw_id)
                except (TypeError, ValueError):
                    object_id = None
                if center is None and object_id is not None:
                    center = object_center(object_id)
                if center is None:
                    continue
                if object_id is not None:
                    if object_id in seen:
                        continue
                    seen.add(object_id)
                result.append((object_id, center))
            if isinstance(additional_ids, Sequence) and not isinstance(additional_ids, (str, bytes)):
                for raw_id in additional_ids:
                    try:
                        object_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if object_id in seen:
                        continue
                    center = object_center(object_id)
                    if center is not None:
                        seen.add(object_id)
                        result.append((object_id, center))
            return result

        selected_subject = objective.get("selected_subject_object_id")
        selected_anchor_ids = objective.get("selected_anchor_object_ids", ())
        candidate_values = objective.get("candidate_hypotheses", ())
        candidate_ids = objective.get("candidate_object_ids", ())
        anchor_values = objective.get("anchor_hypotheses", ())
        anchor_ids = objective.get("anchor_object_ids", ())
        candidates = hypothesis_centers(candidate_values, candidate_ids)
        anchors = hypothesis_centers(anchor_values, anchor_ids)

        # Build viewpoint hypotheses from the actual target/anchor geometry.
        # The offset is on the observer side of the participant tuple, so the
        # probe does not drive the robot onto an object center and preserves a
        # useful parallax baseline for the next relation verification.
        viewpoint_hypotheses: list[tuple[float, float, float]] = []

        def add_pair(subject: list[float], pair_anchors: Sequence[list[float]], priority: float) -> None:
            participants = [subject, *pair_anchors]
            if not participants:
                return
            locus = [
                sum(point[index] for point in participants) / len(participants)
                for index in (0, 1)
            ]
            if len(participants) >= 2:
                axis = (
                    participants[-1][0] - participants[0][0],
                    participants[-1][1] - participants[0][1],
                )
            else:
                axis = (0.0, 0.0)
            axis_norm = math.hypot(axis[0], axis[1])
            if axis_norm > 0.15:
                normal = (-axis[1] / axis_norm, axis[0] / axis_norm)
            else:
                outward = (current_xy[0] - locus[0], current_xy[1] - locus[1])
                outward_norm = math.hypot(outward[0], outward[1])
                normal = (
                    outward[0] / outward_norm,
                    outward[1] / outward_norm,
                ) if outward_norm > 1e-6 else (0.0, -1.0)
            toward_current = (
                current_xy[0] - locus[0], current_xy[1] - locus[1]
            )
            if normal[0] * toward_current[0] + normal[1] * toward_current[1] < 0.0:
                normal = (-normal[0], -normal[1])
            spread = max(
                math.dist(participants[0], point) for point in participants[1:]
            ) if len(participants) > 1 else 0.0
            distance = max(0.9, min(1.6, 1.05 + 0.25 * spread))
            viewpoint_hypotheses.append((
                locus[0] + normal[0] * distance,
                locus[1] + normal[1] * distance,
                float(priority),
            ))

        selected_subject_id = None
        try:
            selected_subject_id = int(selected_subject)
        except (TypeError, ValueError):
            pass
        selected_subject_center = object_center(selected_subject_id)
        selected_anchor_centers = [
            center for _object_id, center in hypothesis_centers((), selected_anchor_ids)
        ]
        if selected_subject_center is not None and selected_anchor_centers:
            add_pair(selected_subject_center, selected_anchor_centers, 0.0)
        for subject_id, subject_center in candidates:
            if selected_subject_id is not None and subject_id == selected_subject_id:
                continue
            if anchors:
                add_pair(subject_center, [anchors[0][1]], 3.0)
            else:
                add_pair(subject_center, (), 3.0)
        if not viewpoint_hypotheses and candidates:
            add_pair(candidates[0][1], [anchors[0][1]] if anchors else (), 3.0)

        region_points: list[tuple[dict[str, Any], list[list[float]]]] = []
        for raw_region in regions:
            points = [
                [float(point[0]), float(point[1])]
                for point in raw_region.get("points_xy", ())
                if finite_xy(point) is not None
            ]
            if points:
                region_points.append((dict(raw_region), points))
        if not region_points:
            return None

        selected_viewpoint: list[float] | None = None
        selected_region: dict[str, Any] | None = None
        selected_region_distance = float("inf")
        selected_region_point: list[float] | None = None
        if viewpoint_hypotheses:
            scored_regions = []
            for region, points in region_points:
                best = min(
                    (
                        math.dist(point, [hypothesis[0], hypothesis[1]])
                        + hypothesis[2]
                        for point in points
                        for hypothesis in viewpoint_hypotheses
                    ),
                    default=float("inf"),
                )
                _best_point, best_hypothesis = min(
                    (
                        (point, hypothesis)
                        for point in points
                        for hypothesis in viewpoint_hypotheses
                    ),
                    key=lambda pair: math.dist(
                        pair[0], [pair[1][0], pair[1][1]]
                    ) + pair[1][2],
                )
                scored_regions.append((
                    best,
                    math.dist(_best_point, current_xy),
                    -len(points),
                    region,
                    list(_best_point),
                ))
            _score, _travel, _size, selected_region, selected_region_point = min(
                scored_regions,
                key=lambda value: (value[0], value[1], value[2], str(value[3].get("region_id", ""))),
            )
            selected_region_distance = float(_score)
            selected_viewpoint = [
                float(selected_region_point[0]),
                float(selected_region_point[1]),
            ]

            # If the map has no frontier near the actual participants, a
            # direct common-viewpoint request is more informative than walking
            # to an unrelated frontier.  Terrain projection remains the
            # downstream authority for the physical waypoint.
            if selected_region_distance > 2.5:
                selected_hypothesis = min(
                    viewpoint_hypotheses,
                    key=lambda hypothesis: (
                        hypothesis[2],
                        math.dist(
                            [hypothesis[0], hypothesis[1]], current_xy
                        ),
                    ),
                )
                selected_viewpoint = [
                    float(selected_hypothesis[0]),
                    float(selected_hypothesis[1]),
                ]
                selected_region = None

        if selected_viewpoint is None:
            selected_region, points = max(
                region_points,
                key=lambda value: len(value[1]),
            )
            selected_viewpoint = [
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            ]
        return {
            "class_label": "observe",
            "target_kind": (
                "relation_common_observable_viewpoint"
                if viewpoint_hypotheses and selected_region is None
                else "observation_frontier"
            ),
            "navigation_target_xy": selected_viewpoint,
            "observation_objective": {
                "unexplored_regions_that_can_change_result": (
                    [selected_region] if selected_region is not None else []
                ),
                "viewpoint_selection": (
                    "participant_geometry"
                    if selected_region is None
                    else "participant_geometry_nearest_frontier"
                    if viewpoint_hypotheses
                    else "largest_frontier"
                ),
                "participant_viewpoint_distance_m": (
                    float(selected_region_distance)
                    if math.isfinite(selected_region_distance)
                    else None
                ),
            },
            "frontier_region_id": (
                str(selected_region.get("region_id", ""))
                if selected_region is not None else ""
            ),
        }

    @staticmethod
    def _joint_relation_probe_object(
        snapshot: Mapping[str, Any],
        probe: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Build an observation viewpoint for one subject/anchor hypothesis."""
        objects_by_id = {
            int(value["object_id"]): dict(value)
            for value in snapshot.get("objects", ())
            if isinstance(value, Mapping)
            and str(value.get("object_id", "")).lstrip("-").isdigit()
        }

        def center_xy(value: Mapping[str, Any] | None) -> list[float] | None:
            if not isinstance(value, Mapping):
                return None
            center = value.get("center_3d")
            if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) < 2:
                return None
            try:
                result = [float(center[0]), float(center[1])]
            except (TypeError, ValueError):
                return None
            return result if all(math.isfinite(item) for item in result) else None

        try:
            source_id = int(probe.get("probe_source_candidate_id"))
        except (TypeError, ValueError):
            source_id = None
        # A joint relation probe is only valid when the subject identity is
        # explicit.  Falling back to the probe object's own center here makes
        # an anchor-only acquisition look like a subject/anchor observation
        # and can silently turn a missing relation participant into evidence.
        subject = objects_by_id.get(source_id) if source_id is not None else None
        subject_center = center_xy(subject)
        if subject_center is None:
            return None
        anchor_ids = [
            int(value)
            for value in probe.get("probe_anchor_object_ids", ())
            if str(value).lstrip("-").isdigit() and int(value) in objects_by_id
        ]
        if source_id is None or source_id in set(anchor_ids):
            return None
        anchors = [
            objects_by_id[object_id]
            for object_id in dict.fromkeys(anchor_ids)
            if center_xy(objects_by_id[object_id]) is not None
        ]
        if not anchors:
            return None
        anchor_centers = [center_xy(value) for value in anchors]
        anchor_centers = [value for value in anchor_centers if value is not None]
        if not anchor_centers:
            return None
        candidate_hypotheses: list[dict[str, Any]] = []
        raw_candidate_hypotheses = probe.get(
            "probe_candidate_hypotheses", ()
        )
        if isinstance(raw_candidate_hypotheses, Sequence) and not isinstance(
            raw_candidate_hypotheses, (str, bytes)
        ):
            for raw_candidate in raw_candidate_hypotheses:
                if not isinstance(raw_candidate, Mapping):
                    continue
                try:
                    candidate_id = int(raw_candidate.get("object_id"))
                except (TypeError, ValueError):
                    continue
                candidate_center = center_xy(objects_by_id.get(candidate_id))
                if candidate_center is None:
                    candidate_center = center_xy(raw_candidate)
                if candidate_center is None or candidate_id in set(anchor_ids):
                    continue
                candidate_hypotheses.append({
                    "object_id": candidate_id,
                    "center_xy": candidate_center,
                    "anchor_object_id": raw_candidate.get(
                        "anchor_object_id"
                    ),
                })
        if source_id is not None and source_id in objects_by_id and not any(
            int(value["object_id"]) == int(source_id)
            for value in candidate_hypotheses
        ):
            candidate_hypotheses.insert(0, {
                "object_id": int(source_id),
                "center_xy": list(subject_center),
            })
        subject_centers: list[list[float]] = [list(subject_center)]
        for value in candidate_hypotheses:
            candidate_center = value["center_xy"]
            if not any(
                math.dist(candidate_center, existing) <= 0.05
                for existing in subject_centers
            ):
                subject_centers.append(list(candidate_center))
        participants = [*subject_centers, *anchor_centers]
        locus = [
            sum(value[axis] for value in participants) / len(participants)
            for axis in (0, 1)
        ]
        anchor_locus = [
            sum(value[axis] for value in anchor_centers) / len(anchor_centers)
            for axis in (0, 1)
        ]
        axis = (
            anchor_locus[0] - subject_center[0],
            anchor_locus[1] - subject_center[1],
        )
        axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm <= 0.15:
            axis = (
                anchor_centers[-1][0] - anchor_centers[0][0],
                anchor_centers[-1][1] - anchor_centers[0][1],
            )
            axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm > 0.15:
            axis_unit = (axis[0] / axis_norm, axis[1] / axis_norm)
            normal = (-axis_unit[1], axis_unit[0])
        else:
            axis_unit = (1.0, 0.0)
            normal = (0.0, -1.0)

        history_xy: list[list[float]] = []

        def add_history(value: object) -> None:
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) < 2
            ):
                return
            try:
                point = [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return
            if not all(math.isfinite(item) for item in point):
                return
            if not any(math.dist(point, prior) <= 0.05 for prior in history_xy):
                history_xy.append(point)

        for raw in snapshot.get("viewpoint_history", ()):
            if isinstance(raw, Mapping):
                add_history(raw.get("viewpoint_position_map"))
        for raw in snapshot.get("navigation_history", ()):
            if not isinstance(raw, Mapping):
                continue
            for key in (
                "actual_arrival_pose",
                "physical_waypoint_pose",
                "requested_waypoint_pose",
            ):
                add_history(raw.get(key))
        current_xy = history_xy[-1] if history_xy else None

        spread = max(
            math.dist(locus, value) for value in participants
        ) if participants else 0.0
        standoff = max(1.0, min(1.8, 1.05 + 0.25 * spread))

        def object_footprint(value: Mapping[str, Any]) -> list[list[float]]:
            center = center_xy(value)
            extent = value.get("bbox_3d")
            if (
                center is None
                or not isinstance(extent, Sequence)
                or isinstance(extent, (str, bytes))
                or len(extent) < 2
            ):
                return []
            try:
                half_x = max(0.05, 0.5 * float(extent[0]) + 0.12)
                half_y = max(0.05, 0.5 * float(extent[1]) + 0.12)
            except (TypeError, ValueError):
                return []
            return [
                [center[0] - half_x, center[1] - half_y],
                [center[0] + half_x, center[1] - half_y],
                [center[0] + half_x, center[1] + half_y],
                [center[0] - half_x, center[1] + half_y],
            ]

        participant_objects = [
            value
            for object_id in dict.fromkeys([
                *(
                    int(value["object_id"])
                    for value in candidate_hypotheses
                ),
                *anchor_ids,
            ])
            for value in [objects_by_id.get(object_id)]
            if isinstance(value, Mapping)
        ]
        if isinstance(subject, Mapping) and subject not in participant_objects:
            participant_objects.insert(0, subject)
        participant_footprints = [
            footprint
            for footprint in (
                object_footprint(value) for value in participant_objects
            )
            if footprint
        ]

        def inside_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
            inside = False
            prior = polygon[-1]
            for current in polygon:
                if (
                    (float(current[1]) > float(point[1]))
                    != (float(prior[1]) > float(point[1]))
                ):
                    crossing_x = (
                        (float(prior[0]) - float(current[0]))
                        * (float(point[1]) - float(current[1]))
                        / (float(prior[1]) - float(current[1]))
                        + float(current[0])
                    )
                    if float(point[0]) < crossing_x:
                        inside = not inside
                prior = current
            return inside

        viewpoint_candidates: list[list[float]] = []

        def add_candidate(value: object) -> None:
            candidate = center_xy({"center_3d": value})
            if candidate is None:
                return
            if any(
                inside_polygon(candidate, footprint)
                for footprint in participant_footprints
            ):
                return
            if not any(math.dist(candidate, prior) <= 0.05 for prior in viewpoint_candidates):
                viewpoint_candidates.append(candidate)

        for raw_candidate in probe.get("probe_viewpoint_candidates_xy", ()):
            add_candidate(raw_candidate)
        for direction in (normal, (-normal[0], -normal[1])):
            add_candidate([
                locus[0] + direction[0] * standoff,
                locus[1] + direction[1] * standoff,
            ])
        # Degenerate or elongated tuples can hide both perpendicular points in
        # a support footprint.  Two axial alternatives keep the goal set small
        # while still avoiding generic frontier search.
        for direction in (axis_unit, (-axis_unit[0], -axis_unit[1])):
            add_candidate([
                locus[0] + direction[0] * standoff,
                locus[1] + direction[1] * standoff,
            ])
        if not viewpoint_candidates:
            return None

        next_target = None
        for directive in probe.get("known_trajectory_directives", ()):
            if not isinstance(directive, Mapping):
                continue
            obj = directive.get("object", {})
            next_target = center_xy(obj if isinstance(obj, Mapping) else None)
            if next_target is not None:
                break

        def alignment_cost(candidate: Sequence[float]) -> float:
            if current_xy is None or next_target is None:
                return 0.5
            first = (
                float(candidate[0]) - current_xy[0],
                float(candidate[1]) - current_xy[1],
            )
            second = (
                next_target[0] - float(candidate[0]),
                next_target[1] - float(candidate[1]),
            )
            first_norm = math.hypot(*first)
            second_norm = math.hypot(*second)
            if first_norm <= 1e-6 or second_norm <= 1e-6:
                return 0.5
            cosine = (
                first[0] * second[0] + first[1] * second[1]
            ) / (first_norm * second_norm)
            return 0.5 * (1.0 - max(-1.0, min(1.0, cosine)))

        def viewpoint_cost(candidate: Sequence[float]) -> tuple[float, float, float]:
            nearest_history = (
                min(math.dist(candidate, prior) for prior in history_xy)
                if history_xy else 2.0
            )
            repeated_side_cost = math.exp(-nearest_history / 0.75)
            travel = (
                min(1.0, math.dist(current_xy, candidate) / 5.0)
                if current_xy is not None else 0.5
            )
            route_deviation = alignment_cost(candidate)
            return (
                0.50 * repeated_side_cost
                + 0.30 * travel
                + 0.20 * route_deviation,
                travel,
                float(candidate[0]),
            )

        viewpoint_candidates.sort(key=viewpoint_cost)
        navigation_target_xy = list(viewpoint_candidates[0])
        candidate_object_ids = [
            int(value["object_id"]) for value in candidate_hypotheses
        ]
        anchor_hypotheses = [
            {
                "object_id": int(value["object_id"]),
                "center_xy": list(center_xy(value)),
            }
            for value in anchors
        ]
        objective = {
            "mode": "joint_relation_observation",
            "relation_predicate": str(probe.get("probe_relation", "")),
            "probe_relation_ids": list(probe.get("probe_relation_ids", ())),
            "joint_visibility_required": True,
            "required_visible_object_ids": list(dict.fromkeys(
                [*candidate_object_ids, *anchor_ids]
            )),
            "selected_subject_object_id": source_id,
            "selected_anchor_object_ids": anchor_ids,
            "candidate_object_ids": candidate_object_ids,
            "anchor_object_ids": anchor_ids,
            "candidate_hypotheses": candidate_hypotheses,
            "anchor_hypotheses": anchor_hypotheses,
            "viewpoint_selection": (
                "subject_anchor_joint_unseen_side_midpoint_standoff"
            ),
            "viewpoint_candidates_xy": copy.deepcopy(viewpoint_candidates),
            "participant_footprints_xy": copy.deepcopy(participant_footprints),
        }
        return {
            **dict(probe),
            "class_label": "observe",
            "target_kind": "relation_common_observable_viewpoint",
            "navigation_target_xy": navigation_target_xy,
            "probe_viewpoint_candidates_xy": copy.deepcopy(
                viewpoint_candidates
            ),
            "probe_look_at_xy": list(locus),
            "probe_relation_subject_xy": list(subject_center),
            "probe_relation_anchor_locus_xy": list(anchor_locus),
            "probe_participant_footprints_xy": copy.deepcopy(
                participant_footprints
            ),
            "probe_object_ids": list(dict.fromkeys(candidate_object_ids)),
            "required_visible_object_ids": list(dict.fromkeys(
                [*candidate_object_ids, *anchor_ids]
            )),
            "joint_visibility_required": True,
            "observation_objective": objective,
            "probe_intent": "PROBE_EVIDENCE",
            "probe_viewpoint": navigation_target_xy,
        }

    def _observation_directives(
        self,
        execution: dict[str, Any],
        snapshot: Mapping[str, Any],
        ctx: AcquisitionContext,
    ) -> list[dict[str, Any]]:
        raw_probe = execution.get("probe_object")
        if not isinstance(raw_probe, Mapping):
            return []
        probe = dict(raw_probe)
        probe_contract = {
            key: probe[key]
            for key in (
                "selector_provisional",
                "selector_state",
                "selector_relation_id",
                "selector_evidence_complete",
            )
            if key in probe
        }
        joint_probe_applied = False
        joint_probe_requested = bool(
            str(probe.get("probe_relation", "")).strip()
            or probe.get("probe_anchor_object_ids")
            or (
                isinstance(probe.get("observation_objective"), Mapping)
                and str(
                    probe["observation_objective"].get("mode", "")
                ) == "joint_relation_observation"
            )
        )
        if joint_probe_requested:
            joint_probe = self._joint_relation_probe_object(snapshot, probe)
            if joint_probe is not None:
                probe = joint_probe
                joint_probe_applied = True
            else:
                # A relation probe without a viable joint geometry must not
                # silently become a generic frontier.  The frontier is only
                # valid when neither relation participant has a usable track;
                # with one participant available, keep the acquisition local
                # to that participant so the missing role can be discovered.
                objects_by_id = {
                    int(value["object_id"]): value
                    for value in snapshot.get("objects", ())
                    if isinstance(value, Mapping)
                    and str(value.get("object_id", "")).lstrip("-").isdigit()
                }

                def finite_center(value: object) -> bool:
                    if not isinstance(value, Mapping):
                        return False
                    center = value.get("center_3d")
                    if (
                        not isinstance(center, Sequence)
                        or isinstance(center, (str, bytes))
                        or len(center) < 2
                    ):
                        return False
                    try:
                        return all(math.isfinite(float(item)) for item in center[:2])
                    except (TypeError, ValueError):
                        return False

                try:
                    source_id = int(probe.get("probe_source_candidate_id"))
                except (TypeError, ValueError):
                    source_id = None
                anchor_ids = [
                    int(value)
                    for value in probe.get("probe_anchor_object_ids", ())
                    if str(value).lstrip("-").isdigit()
                ]

                def integer_values(value: object) -> list[int]:
                    if not isinstance(value, Sequence) or isinstance(
                        value, (str, bytes)
                    ):
                        return []
                    return [
                        int(item)
                        for item in value
                        if str(item).lstrip("-").isdigit()
                    ]

                candidate_ids = integer_values(
                    probe.get("probe_candidate_object_ids", ())
                )
                candidate_ids.extend(
                    integer_values(probe.get("probe_object_ids", ()))
                )
                candidate_hypotheses = probe.get(
                    "probe_candidate_hypotheses", ()
                )
                if isinstance(candidate_hypotheses, Sequence) and not isinstance(
                    candidate_hypotheses, (str, bytes)
                ):
                    candidate_ids.extend(
                        int(value.get("object_id"))
                        for value in candidate_hypotheses
                        if isinstance(value, Mapping)
                        and str(value.get("object_id", "")).lstrip("-").isdigit()
                    )
                subject_ids = list(dict.fromkeys(
                    value for value in [source_id, *candidate_ids]
                    if value is not None
                ))
                viable_subject_ids = [
                    value for value in subject_ids
                    if finite_center(objects_by_id.get(value))
                ]
                viable_anchor_ids = [
                    value for value in dict.fromkeys(anchor_ids)
                    if finite_center(objects_by_id.get(value))
                ]
                if viable_subject_ids and viable_anchor_ids:
                    return []
                if viable_subject_ids or viable_anchor_ids:
                    participant_id = (
                        viable_subject_ids[0]
                        if viable_subject_ids
                        else viable_anchor_ids[0]
                    )
                    participant = dict(objects_by_id[participant_id])
                    relation_metadata = dict(probe)
                    for key in {
                        "object_id",
                        "class_label",
                        "status",
                        "semantic_status",
                        "center_3d",
                        "bbox_3d",
                        "center_cov",
                    }:
                        relation_metadata.pop(key, None)
                    participant.update(relation_metadata)
                    participant["class_label"] = "relation_participant_probe"
                    participant["target_kind"] = (
                        "relation_subject_neighborhood"
                        if viable_subject_ids
                        else "relation_anchor_neighborhood"
                    )
                    probe = participant
                else:
                    probe["class_label"] = "observe"

        if (
            not joint_probe_applied
            and str(probe.get("class_label", "")).strip().lower() == "observe"
        ):
            objective = probe.get("observation_objective", {})
            preferred_ids = []
            if isinstance(objective, Mapping):
                preferred_ids = [
                    str(value.get("region_id", ""))
                    for value in objective.get("unexplored_regions_that_can_change_result", ())
                    if isinstance(value, Mapping)
                ]
            frontier = self._frontier_probe_object(
                snapshot,
                preferred_ids,
                objective=objective,
            )
            if frontier is None and preferred_ids and isinstance(objective, Mapping):
                for region in objective.get(
                    "unexplored_regions_that_can_change_result", ()
                ):
                    if not isinstance(region, Mapping):
                        continue
                    if str(region.get("region_id", "")) not in preferred_ids:
                        continue
                    representative = region.get("representative_xy")
                    if not (
                        isinstance(representative, Sequence)
                        and not isinstance(representative, (str, bytes))
                        and len(representative) >= 2
                        and all(
                            math.isfinite(float(value))
                            for value in representative[:2]
                        )
                    ):
                        continue
                    center_xy = [
                        float(representative[0]),
                        float(representative[1]),
                    ]
                    frontier = {
                        "class_label": "observe",
                        "target_kind": "observation_frontier",
                        "navigation_target_xy": list(center_xy),
                        "frontier_region_id": str(region.get("region_id", "")),
                    }
                    break
            if frontier is None:
                return []
            frontier.update({
                key: value
                for key, value in probe.items()
                if key not in {"center_3d", "bbox_3d", "navigation_target_xy", "evidence"}
            })
            probe = frontier

        navigation_target = probe.get("navigation_target_xy")
        if not (
            isinstance(navigation_target, Sequence)
            and not isinstance(navigation_target, (str, bytes))
            and len(navigation_target) >= 2
        ):
            center = probe.get("center_3d")
            if not (
                isinstance(center, Sequence)
                and not isinstance(center, (str, bytes))
                and len(center) >= 2
            ):
                return []
            center_xy = [float(center[0]), float(center[1])]
            current_xy = None
            for raw_view in reversed(snapshot.get("viewpoint_history", ())):
                if not isinstance(raw_view, Mapping):
                    continue
                position = raw_view.get("viewpoint_position_map")
                if (
                    isinstance(position, Sequence)
                    and not isinstance(position, (str, bytes))
                    and len(position) >= 2
                ):
                    try:
                        candidate_xy = [float(position[0]), float(position[1])]
                    except (TypeError, ValueError):
                        continue
                    if all(math.isfinite(value) for value in candidate_xy):
                        current_xy = candidate_xy
                        break
            if current_xy is None:
                current_xy = [center_xy[0], center_xy[1] - 1.0]
            outward = (
                current_xy[0] - center_xy[0],
                current_xy[1] - center_xy[1],
            )
            outward_norm = math.hypot(*outward)
            normal = (
                outward[0] / outward_norm,
                outward[1] / outward_norm,
            ) if outward_norm > 0.15 else (0.0, -1.0)
            extent = probe.get("bbox_3d", ())
            try:
                standoff = max(
                    1.0,
                    min(1.8, 0.9 + 0.5 * max(
                        float(extent[0]), float(extent[1])
                    )),
                )
            except (IndexError, TypeError, ValueError):
                standoff = 1.2
            navigation_target = [
                float(center_xy[0] + normal[0] * standoff),
                float(center_xy[1] + normal[1] * standoff),
            ]
            if bool(probe.get("requires_new_station", False)):
                # Semantic verification needs an independently acquired view,
                # but the start-facing standoff can fall inside the physical
                # arrival/separation envelope when the object is already
                # nearby.  Preserve the object as semantic authority and give
                # the waypoint planner a view-orbit goal set around it.  Its
                # continuous terrain/travel cost remains the sole selector.
                orbit_candidates = []
                for index in range(8):
                    angle = math.tau * float(index) / 8.0
                    cosine = math.cos(angle)
                    sine = math.sin(angle)
                    orbit_normal = (
                        normal[0] * cosine - normal[1] * sine,
                        normal[0] * sine + normal[1] * cosine,
                    )
                    orbit_candidates.append([
                        float(center_xy[0] + orbit_normal[0] * standoff),
                        float(center_xy[1] + orbit_normal[1] * standoff),
                    ])
                existing_candidates = probe.get(
                    "probe_viewpoint_candidates_xy", ()
                )
                if not isinstance(existing_candidates, Sequence) or isinstance(
                    existing_candidates, (str, bytes)
                ):
                    existing_candidates = ()
                probe["probe_viewpoint_candidates_xy"] = [
                    list(value)
                    for value in [*existing_candidates, *orbit_candidates]
                    if isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                    and len(value) >= 2
                ]
                probe["probe_look_at_xy"] = list(center_xy)

        known = [
            dict(value)
            for value in probe.get("known_trajectory_directives", ())
            if isinstance(value, Mapping)
        ]
        # A probe is always an evidence viewpoint.  A semantic hint carried in
        # the probe provenance must not be converted into the current
        # provisional terminal object's approach region.
        directive = {
            "order": int(ctx.current_step_index),
            "action": "probe",
            "terminal": False,
            "forbidden": False,
            "object": probe,
            "navigation_target_xy": [
                float(navigation_target[0]),
                float(navigation_target[1]),
            ],
            "intent": "PROBE_EVIDENCE",
            "evidence_viewpoint": True,
            "observation_waypoint_role": "OBSERVATION",
        }
        directive.update(probe_contract)
        # Keep all known forbidden regions in the segment projection. A
        # semantic hint is the active route, while the probe object remains
        # available as acquisition provenance.
        constraints = [
            value for value in known
            if bool(value.get("forbidden"))
            or isinstance(value.get("trajectory_region"), Mapping)
        ]
        return [directive, *constraints]

    def _waypoint_selection(
        self,
        execution: dict[str, Any],
        snapshot: Mapping[str, Any],
        ctx: AcquisitionContext,
    ) -> dict[str, Any]:
        directives = [
            dict(value)
            for value in execution.get("trajectory_directives", ())
            if isinstance(value, Mapping)
        ]
        probe = False
        instruction_hypothesis = False
        if directives:
            instruction_hypothesis = True
        else:
            directives = self._observation_directives(execution, snapshot, ctx)
            probe = bool(directives)
        if not directives:
            return {
                "schema_version": "semantic_waypoint_segment_v1",
                "status": "blocked",
                "reason": "no_semantic_or_probe_waypoint",
                "waypoints": [],
                "probe": False,
                "instruction_hypothesis": False,
            }

        # The active semantic step owns the constraint namespace.  Reusing the
        # previous step's monitor ID makes terminal routes look like step 0 in
        # runtime evidence and lets stale corridor metadata leak forward.
        constraint_set_id = f"{ctx.episode_id}:step:{int(ctx.current_step_index)}"
        result = semantic_waypoint_segment_output(
            directives,
            constraint_set_id=constraint_set_id,
            active_directive_index=0,
            competition_geometry=dict(ctx.competition_geometry),
            navigation_config=dict(ctx.navigation_config),
            navigation_history=list(ctx.navigation_history),
        )
        result = dict(result)
        active_step = next(
            (
                value for value in execution.get("execution_steps", ())
                if isinstance(value, Mapping)
                and str(value.get("step_index", "")).lstrip("-").isdigit()
                and int(value.get("step_index", -1)) == int(ctx.current_step_index)
            ),
            {},
        )
        active_directive = directives[0] if directives else {}
        active_object = active_directive.get("object", {})
        active_region = active_directive.get("trajectory_region", {})
        target_object_id = (
            active_object.get("object_id")
            if isinstance(active_object, Mapping)
            else None
        )
        target_object_ids = []
        if target_object_id is not None:
            try:
                target_object_ids = [int(target_object_id)]
            except (TypeError, ValueError):
                target_object_ids = []
        anchor_object_ids = [
            int(value["object_id"])
            for value in active_directive.get("anchor_objects", ())
            if isinstance(value, Mapping)
            and str(value.get("object_id", "")).lstrip("-").isdigit()
        ]
        if not anchor_object_ids and isinstance(active_object, Mapping):
            anchor_object_ids = [
                int(value)
                for value in active_object.get("probe_anchor_object_ids", ())
                if str(value).lstrip("-").isdigit()
            ]
        required_visible_object_ids = []
        probe_object_ids = []
        if isinstance(active_object, Mapping):
            required_visible_object_ids = [
                int(value)
                for value in active_object.get("required_visible_object_ids", ())
                if str(value).lstrip("-").isdigit()
            ]
            probe_object_ids = [
                int(value)
                for value in active_object.get("probe_object_ids", ())
                if str(value).lstrip("-").isdigit()
            ]
        if probe and probe_object_ids:
            # The raw numerical plan may acquire an anchor first and keep it
            # in ``object_id``.  For a joint evidence viewpoint the semantic
            # target side of the context is the explicit subject candidate;
            # the anchor remains in its own field and both are preserved in
            # required_visible_object_ids.
            target_object_ids = list(dict.fromkeys(probe_object_ids))
        explicit_navigation_target = (
            active_directive.get("navigation_target_xy")
            if isinstance(active_directive.get("navigation_target_xy"), Sequence)
            and not isinstance(active_directive.get("navigation_target_xy"), (str, bytes))
            else active_object.get("navigation_target_xy")
            if isinstance(active_object, Mapping)
            else None
        )
        action_name = str(
            active_directive.get(
                "action", active_step.get("action", "probe")
            )
        ).lower()
        terminal = bool(
            active_directive.get(
                "terminal", active_step.get("is_terminal", False)
            )
        )
        target_name = (
            str(target_object_id)
            if target_object_id is not None
            else str(active_step.get("target_entity", "probe"))
        )
        prior_attempts = [
            value for value in ctx.navigation_history
            if isinstance(value, Mapping)
            and str(value.get("step_index", "")).lstrip("-").isdigit()
            and int(value.get("step_index", -1)) == int(ctx.current_step_index)
        ]
        attempt = len(prior_attempts) + 1
        waypoint_pairs = ";".join(
            f"({float(value[0]):.3f},{float(value[1]):.3f})"
            for value in result.get("waypoints", ())
            if isinstance(value, Sequence) and len(value) >= 2
        )
        base_command_signature = (
            f"step={int(ctx.current_step_index)}|target={target_name}|"
            f"wps={waypoint_pairs}"
        )

        result["command_signature"] = base_command_signature
        context = dict(result.get("waypoint_context", {}))
        context.update({
            "episode_id": str(ctx.episode_id),
            "corridor_id": (
                f"step={int(ctx.current_step_index)}|action={action_name}|"
                f"target={target_name}|attempt={attempt}"
            ),
            "step_index": int(ctx.current_step_index),
            "action": action_name,
            "selector_provisional": bool(
                active_directive.get(
                    "selector_provisional",
                    active_step.get("selector_provisional", False),
                )
            ),
            "selector_state": str(
                active_directive.get(
                    "selector_state", active_step.get("selector_state", "")
                )
            ),
            "selector_relation_id": str(
                active_directive.get(
                    "selector_relation_id",
                    active_step.get("selector_relation_id", ""),
                )
            ),
            "target_entity_id": str(active_step.get("target_entity", "")),
            "target_object_ids": target_object_ids,
            "anchor_entity_ids": [
                str(value) for value in active_step.get("anchor_entities", ())
            ],
            "anchor_object_ids": anchor_object_ids,
            "probe_object_ids": probe_object_ids,
            "probe_source_candidate_id": (
                int(active_object.get("probe_source_candidate_id"))
                if isinstance(active_object, Mapping)
                and str(active_object.get("probe_source_candidate_id", ""))
                .lstrip("-")
                .isdigit()
                else None
            ),
            "required_visible_object_ids": required_visible_object_ids,
            "joint_visibility_required": bool(
                active_object.get("joint_visibility_required", False)
                if isinstance(active_object, Mapping)
                else False
            ),
            "requires_new_station": bool(
                active_object.get("requires_new_station", False)
                if isinstance(active_object, Mapping)
                else False
            ),
            "anchor_first": bool(
                isinstance(active_object, Mapping)
                and isinstance(active_object.get("observation_objective"), Mapping)
                and active_object["observation_objective"].get("anchor_first") is True
            ),
            "navigation_target_xy": (
                [float(explicit_navigation_target[0]), float(explicit_navigation_target[1])]
                if isinstance(explicit_navigation_target, Sequence)
                and not isinstance(explicit_navigation_target, (str, bytes))
                and len(explicit_navigation_target) >= 2
                else None
            ),
            "is_terminal": terminal,
            "generated_scene_version": int(snapshot.get("scene_version", 0)),
            "semantic_region": (
                dict(active_region) if isinstance(active_region, Mapping) else {}
            ),
            "command_signature": (
                f"step={int(ctx.current_step_index)}|target={target_name}|"
                f"wps={waypoint_pairs}"
            ),
            "expected_checkpoint": (
                "terminal_completion" if terminal else "semantic_transition"
            ),
            "local_waypoint_count": len(result.get("waypoints", ())),
            "post_satisfaction_tail_count": 0,
            "attempt": attempt,
        })
        # Preserve the semantic object region separately from any explicit
        # terrain-adjusted terminal region.  A projected legal endpoint must
        # remain visible in runtime evidence; it must never look like the
        # semantic center was reached exactly.
        terminal_projection = result.get("terminal_projection")
        if isinstance(terminal_projection, Mapping):
            context["terminal_projection"] = dict(terminal_projection)
        terrain_projection = result.get("traversable_projection")
        if isinstance(terrain_projection, Mapping):
            context["terrain_projection"] = dict(terrain_projection)
        result["waypoint_context"] = context
        result["episode_id"] = str(ctx.episode_id)
        result["generated_scene_version"] = int(snapshot.get("scene_version", 0))
        semantic_route = bool(directives and not probe)
        result["probe"] = bool(
            probe and not semantic_route and result.get("status") == "completed"
        )
        result["instruction_hypothesis"] = bool(
            semantic_route
            and result.get("status") == "completed"
        )
        result["intent"] = (
            "EXECUTE_ORDERED_CONSTRAINT"
            if semantic_route
            else "ACQUIRE_EVIDENCE"
            if result.get("status") == "completed"
            else ""
        )
        selector_resolutions = [
            dict(value)
            for value in execution.get("selector_resolutions", ())
            if isinstance(value, Mapping)
        ]
        trace_selector_resolutions = list(selector_resolutions)
        active_selector = next(
            (
                value for value in trace_selector_resolutions
                if str(value.get("relation_id", ""))
                == str(
                    active_directive.get(
                        "selector_relation_id",
                        active_object.get("selector_relation_id", "")
                        if isinstance(active_object, Mapping) else "",
                    )
                )
            ),
            (
                trace_selector_resolutions[0]
                if trace_selector_resolutions else {}
            ),
        )
        selector_hypotheses = [
            dict(hypothesis)
            for resolution in trace_selector_resolutions
            for hypothesis in resolution.get("selector_hypotheses", ())
            if isinstance(hypothesis, Mapping)
        ]

        def compact_selector_hypothesis(
            value: object,
        ) -> dict[str, Any] | None:
            if not isinstance(value, Mapping):
                return None
            raw_anchor_ids = value.get("anchor_object_ids", ())
            anchor_id = value.get("anchor_object_id")
            if anchor_id is None and isinstance(
                raw_anchor_ids, Sequence
            ) and not isinstance(raw_anchor_ids, (str, bytes)):
                anchor_id = next(iter(raw_anchor_ids), None)
            return {
                "target_id": value.get(
                    "subject_object_id", value.get("selected_object_id")
                ),
                "anchor_id": anchor_id,
                "distance_m": value.get(
                    "horizontal_distance_m", value.get("raw_distance_m")
                ),
                "distance_uncertainty_m": value.get("uncertainty_m"),
                "identity_support": value.get("identity_support"),
                "relation_support": value.get(
                    "pair_support", value.get("relation_posterior")
                ),
                "selector_score": value.get("selector_score"),
                "stability_margin_m": value.get("stability_margin_m"),
            }

        compact_selector_hypotheses = [
            compact
            for compact in (
                compact_selector_hypothesis(value)
                for value in selector_hypotheses
            )
            if compact is not None
        ]
        compact_selector_resolutions = [
            {
                "relation_id": value.get("relation_id"),
                "selector_state": value.get("selector_state"),
                "selected_object_id": value.get("selected_object_id"),
                "selected_anchor_id": value.get("selected_anchor_id"),
                "selector_evidence_complete": value.get(
                    "selector_evidence_complete"
                ),
            }
            for value in trace_selector_resolutions
        ]
        trajectory_monitor = (
            ctx.trajectory_monitor
            if isinstance(ctx.trajectory_monitor, Mapping)
            else {}
        )
        trajectory_step_status = str(active_step.get("status", ""))
        try:
            terminal_dwell_samples = int(
                trajectory_monitor.get("terminal_dwell_samples", 0) or 0
            )
        except (TypeError, ValueError):
            terminal_dwell_samples = 0
        result["decision_trace"] = {
            "scene_revision": int(snapshot.get("scene_version", 0)),
            "identity_revision": int(snapshot.get("identity_revision", 0)),
            "geometry_revision": int(snapshot.get("geometry_version", 0)),
            "current_step": int(ctx.current_step_index),
            "raw_entity_domains": copy.deepcopy(
                execution.get("candidate_domains", {})
            ),
            "canonical_identity_groups": copy.deepcopy(
                snapshot.get("identity_clusters", ())
            ),
            "selector_hypotheses": compact_selector_hypotheses,
            "selector_resolutions": compact_selector_resolutions,
            "selector_evidence_features": copy.deepcopy(
                active_selector.get("selector_evidence_features", {})
            ),
            "intent": result["intent"],
            "final_intent": result["intent"],
            "probe_reason": str(
                active_object.get("probe_reason", "")
                if isinstance(active_object, Mapping) else ""
            ),
            "probe_target_ids": list(dict.fromkeys(
                [*target_object_ids, *anchor_object_ids, *probe_object_ids]
            )),
            "probe_viewpoint": (
                list(result.get("waypoints", ())[0])
                if result["intent"] == "ACQUIRE_EVIDENCE"
                and result.get("waypoints") else None
            ),
            "semantic_object_id": (
                int(target_object_id)
                if result["intent"] == "EXECUTE_ORDERED_CONSTRAINT"
                and target_object_id is not None
                else None
            ),
            "published_waypoint": list(result.get("waypoints", ())[0])
            if result.get("waypoints") else None,
            "route_active": bool(
                trajectory_monitor.get("route_active", False)
            ),
            "semantic_region_satisfied": bool(
                trajectory_step_status == "SATISFIED"
            ),
            "terminal_dwell_samples": terminal_dwell_samples,
            "trajectory_step_status": trajectory_step_status,
        }
        return result

    def run_single_acquisition(
        self,
        ctx: AcquisitionContext,
        output_dir: Path,
        *,
        summary: PipelineSummary | None = None,
    ) -> PipelineSummary:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        total_started = time.monotonic()
        summary = summary or PipelineSummary(
            episode_id=ctx.episode_id,
            acquisition_id=ctx.acquisition_id,
            question=ctx.question,
            task_ir=ctx.task_ir,
        )

        snapshot: dict[str, Any] = load_scene_memory_snapshot(
            ctx.scene_memory_path,
            acquisition_id=ctx.acquisition_id,
        )
        snapshot["arrival_transaction"] = copy.deepcopy(ctx.arrival_transaction)
        # Navigation history is runtime provenance for query-conditioned
        # viewpoint selection.  It is intentionally not SceneMemory truth:
        # the query planner uses failed physical targets to choose a new
        # baseline, while SceneMemory continues to own observed objects and
        # their world evidence.
        snapshot["navigation_history"] = [
            copy.deepcopy(dict(value))
            for value in ctx.navigation_history
            if isinstance(value, Mapping)
        ]
        perception_result: dict[str, Any] | None = None
        geometry_result: dict[str, Any] | None = None
        persistent_store: dict[str, Any] = {}
        persistent_relation_summary: dict[str, Any] = {}

        try:
            snapshot["episode_time_remaining_seconds"] = ctx.deadline.remaining()
            initial_coverage = compute_required_entity_coverage(
                snapshot,
                ctx.task_ir,
                current_step_index=int(ctx.current_step_index),
                entity_bindings=ctx.semantic_entity_bindings,
            )
            persistent_store = copy.deepcopy(
                snapshot.get("relation_evidence", {})
            )
            snapshot["relation_revision"] = int(
                persistent_store.get("relation_revision", 0)
            )
            persistent_relation_summary = summarize_persistent_relation_evidence(
                persistent_store,
                ctx.task_ir,
            )
            snapshot["required_entity_coverage"] = initial_coverage
            snapshot["persistent_relation_summary"] = persistent_relation_summary
            summary.set_stage(
                "evidence_acquisition_input",
                {
                    "status": "completed",
                    "evidence_need": copy.deepcopy(ctx.evidence_need),
                    "coordinator": "EvidenceAcquisitionCoordinator",
                },
            )

            effective_mode = ctx.perception_mode
            perception_task_ir: Mapping[str, Any] = ctx.task_ir
            perception_entity_ids = list(ctx.scene_observation_entity_ids)

            gate_started = time.monotonic()
            gate = self._validate_online_inputs(
                ctx,
                force_perception=bool(ctx.evidence_need),
                perception_mode=effective_mode,
            )
            summary.set_stage("input_gate", self._stage("completed", gate_started, **gate))

            qwen = dict(self.runtime.get("qwen3vl", {}))
            sam2 = dict(self.runtime.get("sam2", {}))
            yolo = dict(self.runtime.get("yolo_world", {}))
            pipeline_config = dict(self.runtime.get("pipeline", {}))
            configured_yaws = tuple(
                float(value)
                for value in qwen.get(
                    "semantic_perspective_yaws_deg",
                    (0, 45, 90, 135, 180, 225, 270, 315),
                )
            )
            requested_yaws = ctx.perception_yaws_deg or configured_yaws
            view_limit = ctx.perception_view_limit or len(requested_yaws)
            perspective_yaws = tuple(requested_yaws[: max(0, view_limit)])
            mode_default_cap = {
                "initial_full": TIME_BUDGET["initial_perception_cap_seconds"],
                "evidence_targeted": TIME_BUDGET["evidence_perception_cap_seconds"],
                "world_refresh": TIME_BUDGET["world_refresh_cap_seconds"],
                "decision_only": 0.0,
            }.get(effective_mode, TIME_BUDGET["evidence_perception_cap_seconds"])
            perception_cap = ctx.perception_cap_seconds or mode_default_cap
            should_observe = bool(
                (ctx.station_is_new or bool(ctx.evidence_need))
                and effective_mode != "decision_only"
                and perspective_yaws
                and perception_cap > 0.0
            )
            perception_timeout = ctx.deadline.stage_timeout(perception_cap)
            can_observe = bool(should_observe and perception_timeout >= 3.0)
            if should_observe and not can_observe:
                summary.set_stage(
                    "perception",
                    {
                        "status": "skipped",
                        "elapsed_seconds": 0.0,
                        "reason": "episode_deadline_dispatch_reserve_protected",
                        "perception_mode": effective_mode,
                        "requested_view_count": len(perspective_yaws),
                        "detections": [],
                        "views": [],
                    },
                )
            elif not should_observe:
                summary.set_stage(
                    "perception",
                    {
                        "status": "skipped",
                        "elapsed_seconds": 0.0,
                        "reason": (
                            "decision_only_uses_persistent_memory"
                            if effective_mode == "decision_only"
                            else "same_optical_station_reuses_persistent_memory"
                            if not ctx.station_is_new
                            else "perception_policy_has_no_views"
                        ),
                        "perception_mode": effective_mode,
                        "detections": [],
                        "views": [],
                    },
                )
            else:
                perception_started = time.monotonic()
                perception_result = execute_perception_pipeline(
                    panorama_path=ctx.image_path,
                    task_ir=perception_task_ir,
                    scene_observation_entity_ids=perception_entity_ids,
                    worker_endpoints={
                        "qwen": str(qwen.get("endpoint", "@mast3r_qwen3vl")),
                        "sam2": str(sam2.get("endpoint", "@mast3r_sam2")),
                        "yolo": str(yolo.get("endpoint", "@mast3r_yolo")),
                    },
                    output_dir=output_dir,
                    timeout=max(3.0, perception_timeout),
                    acquisition_id=ctx.acquisition_id,
                    detection_threshold=float(yolo.get("score_threshold", 0.20)),
                    perspective_width=int(qwen.get("semantic_perspective_width", 1024)),
                    perspective_height=int(qwen.get("semantic_perspective_height", 768)),
                    perspective_yaws_deg=perspective_yaws,
                    perception_mode=effective_mode,
                    skip_semantic_verification=(
                        effective_mode == "emergency_targeted"
                        or bool(ctx.perception_policy.get("skip_semantic_verification", False))
                    ),
                )
                summary.set_stage(
                    "perception",
                    self._stage(
                        str(perception_result.get("status", "error")),
                        perception_started,
                        reason=str(perception_result.get("reason", "")),
                        perception_mode=effective_mode,
                        requested_yaws_deg=list(perspective_yaws),
                        recovery_diagnostics=copy.deepcopy(
                            perception_result.get("recovery_diagnostics", {})
                        ) if isinstance(perception_result.get("recovery_diagnostics"), Mapping) else {},
                        detection_count=len(perception_result.get("detections", ())),
                        view_count=len(perception_result.get("views", ())),
                        counts_by_class=dict(perception_result.get("counts_by_class", {})),
                        proposal_sources=dict(
                            perception_result.get("proposal_sources", {})
                        ),
                        warnings=list(perception_result.get("warnings", ())),
                        proposals_path=str(perception_result.get("proposals_path", "")),
                        verified_detections_path=str(
                            perception_result.get("verified_detections_path", "")
                        ),
                        views_manifest_path=str(
                            perception_result.get("views_manifest_path", "")
                        ),
                    ),
                )
                if str(perception_result.get("status")) == "completed":
                    geometry_started = time.monotonic()
                    calibration = self._resolved_path(
                        self.runtime.get("competition_geometry", {}).get(
                            "calibration", "configs/sensor_at_scan_to_panorama.json"
                        )
                    )
                    geometry_result = lift_detections_to_observations(
                        detections=perception_result.get("detections", ()),
                        views=perception_result.get("views", ()),
                        competition_geometry=ctx.competition_geometry,
                        station_id=ctx.station_id,
                        acquisition_id=ctx.acquisition_id,
                        output_dir=output_dir,
                        calibration_path=calibration,
                        minimum_support_points=6,
                    )
                    geometry_status = (
                        "completed"
                        if geometry_result.get("observations")
                        else "blocked"
                    )
                    summary.set_stage(
                        "lidar_geometry",
                        self._stage(
                            geometry_status,
                            geometry_started,
                            reason=(
                                ""
                                if geometry_status == "completed"
                                else "no_mask_backed_lidar_observations"
                            ),
                            frame="map",
                            observation_count=len(geometry_result.get("observations", ())),
                            rejected_count=len(geometry_result.get("rejected", ())),
                            observations_path=str(
                                output_dir / "05_lidar_geometry" / "observations.json"
                            ),
                            rejected=list(geometry_result.get("rejected", ())),
                        ),
                    )

                    if geometry_result.get("observations"):
                        memory_started = time.monotonic()
                        navigation = dict(ctx.navigation_config)
                        snapshot = update_scene_memory(
                            ctx.scene_memory_path,
                            geometry_result["observations"],
                            acquisition_id=ctx.acquisition_id,
                            minimum_viewpoint_separation_m=float(
                                self.runtime.get("semantic_execution", {}).get(
                                    "minimum_independent_viewpoint_separation_m", 0.30
                                )
                            ),
                            terrain_map_path=str(
                                ctx.competition_geometry.get("terrain_map_path", "")
                            ),
                            accumulated_terrain_map_path=str(
                                ctx.competition_geometry.get("accumulated_terrain_path")
                                or ctx.competition_geometry.get("terrain_map_ext_path", "")
                            ),
                            terrain_obstacle_height_threshold=float(
                                navigation.get("terrain_obstacle_height_threshold", 0.05)
                            ),
                            terrain_voxel_size_m=float(
                                navigation.get("terrain_voxel_size_m", 0.05)
                            ),
                            viewpoint_position_map=ctx.viewpoint_position_map,
                            geometry_manifest_path=None,
                            panorama_view_count=len(
                                perception_result.get("views", ())
                            ),
                        )
                        snapshot["navigation_history"] = [
                            copy.deepcopy(dict(value))
                            for value in ctx.navigation_history
                            if isinstance(value, Mapping)
                        ]
                        summary.set_stage(
                            "scene_memory",
                            self._stage(
                                "completed",
                                memory_started,
                                scene_version=int(snapshot.get("scene_version", 0)),
                                associated_object_ids=list(
                                    snapshot.get("associated_object_ids", ())
                                ),
                                object_count=len(snapshot.get("objects", ())),
                                snapshot=snapshot,
                            ),
                        )
                    else:
                        summary.set_stage(
                            "scene_memory",
                            {
                                "status": "skipped",
                                "elapsed_seconds": 0.0,
                                "reason": "no_valid_3d_observation_to_ingest",
                                "scene_version": int(snapshot.get("scene_version", 0)),
                                "object_count": len(snapshot.get("objects", ())),
                                "snapshot": snapshot,
                            },
                        )
                else:
                    summary.set_stage(
                        "lidar_geometry",
                        {
                            "status": "skipped",
                            "elapsed_seconds": 0.0,
                            "reason": "perception_not_completed",
                            "observation_count": 0,
                        },
                    )
                    summary.set_stage(
                        "scene_memory",
                        {
                            "status": "skipped",
                            "elapsed_seconds": 0.0,
                            "reason": "persistent_memory_unchanged",
                            "scene_version": int(snapshot.get("scene_version", 0)),
                            "object_count": len(snapshot.get("objects", ())),
                            "snapshot": snapshot,
                        },
                    )

            if "lidar_geometry" not in summary.stages:
                summary.set_stage(
                    "lidar_geometry",
                    {
                        "status": "skipped",
                        "elapsed_seconds": 0.0,
                        "reason": "no_new_station_geometry",
                        "observation_count": 0,
                    },
                )
            if "scene_memory" not in summary.stages:
                summary.set_stage(
                    "scene_memory",
                    {
                        "status": "completed",
                        "elapsed_seconds": 0.0,
                        "reason": "persistent_snapshot_loaded",
                        "scene_version": int(snapshot.get("scene_version", 0)),
                        "object_count": len(snapshot.get("objects", ())),
                        "snapshot": snapshot,
                    },
                )

            # Keep the state dimensions independent.  SceneMemory is open for
            # every task; only relation selectors inherit that openness, while
            # numerical answer authority is decided by query execution.
            snapshot = dict(snapshot)
            if str(ctx.task_ir.get("task_type", "")) in {
                "numerical", "object_reference", "instruction_following",
            }:
                entity_view = materialize_query_view(
                    ctx.task_ir,
                    snapshot.get("observation_ledger", {}),
                    current_snapshot=snapshot,
                    minimum_viewpoint_separation_m=float(
                        self.runtime.get("semantic_execution", {}).get(
                            "minimum_independent_viewpoint_separation_m", 0.30
                        )
                    ),
                    terrain_voxel_size_m=float(
                        ctx.navigation_config.get("terrain_voxel_size_m", 0.05)
                    ),
                )
                for key in (
                    "scene_version",
                    "identity_revision",
                    "geometry_version",
                    "objects",
                    "object_id_aliases",
                    "identity_clusters",
                    "identity_constraints",
                    "identity_ambiguity_groups",
                    "ambiguous_observations",
                    "viewpoint_history",
                    "observation_ledger_size",
                    "cardinality_summary",
                ):
                    snapshot[key] = copy.deepcopy(entity_view[key])
                snapshot["query_entity_view"] = copy.deepcopy(entity_view)
                snapshot["scene_memory_authority"] = "ObservationLedger+QueryProgram"
            selector_relation_ids = [
                str(relation.get("id", ""))
                for relation in ctx.task_ir.get("relations", ())
                if str(relation.get("predicate", "")).strip().lower()
                in {"closest", "farthest", "furthest"}
            ]
            relation_selector_open = bool(selector_relation_ids)
            snapshot["scene_memory_open"] = True
            snapshot["scene_memory_state"] = "OPEN_PERSISTENT_TRACKS"
            snapshot["relation_selector_domain_open"] = relation_selector_open
            snapshot["relation_selector_domain_state"] = (
                "OPEN_PERSISTENT_TRACKS"
                if relation_selector_open else "NOT_APPLICABLE"
            )
            snapshot["acquisition_closure_diagnostics"] = {
                "mode": "open_persistent_tracks",
                "scene_memory_open": True,
                "relation_selector_domain_open": relation_selector_open,
                "relation_selector_relation_ids": selector_relation_ids,
                "ranking_recomputed_each_acquisition": True,
            }
            acquisition_closure_reasons: list[str] = []

            # Compute required entity coverage
            snapshot["episode_time_remaining_seconds"] = ctx.deadline.remaining()
            entity_coverage = compute_required_entity_coverage(
                snapshot,
                ctx.task_ir,
                current_step_index=int(ctx.current_step_index),
                entity_bindings=ctx.semantic_entity_bindings,
            )
            snapshot["required_entity_coverage"] = entity_coverage
            snapshot["persistent_relation_summary"] = persistent_relation_summary

            # Never claim acquisition closure when post-ingestion coverage
            # disproves decision-only.  The next parent-owned acquisition will
            # re-enter the scheduler with these entity-level facts.
            if ctx.perception_mode == "decision_only" and not entity_coverage["decision_only_allowed"]:
                snapshot.setdefault("continuous_belief_diagnostics", {})[
                    "decision_only_observation_gap"
                ] = list(entity_coverage["decision_only_blocked_reasons"])

            query_domain_started = time.monotonic()
            minimum_separation = float(
                self.runtime.get("semantic_execution", {}).get(
                    "minimum_independent_viewpoint_separation_m", 0.30
                )
            )
            snapshot = attach_query_domain(
                snapshot,
                ctx.task_ir,
            )
            count_graph = (
                ctx.task_ir.get("count_query_graph")
                if str(ctx.task_ir.get("task_type", "")) == "numerical"
                else None
            )
            if isinstance(count_graph, Mapping):
                count_domain = assess_count_query_domain(
                    count_graph,
                    snapshot,
                )
                snapshot["count_query_domain"] = count_domain
                snapshot["numerical_count_domain_closed"] = bool(
                    count_domain.get("closed") is True
                )
                snapshot["count_query_targeted_acquisition"] = dict(
                    count_domain.get("next_acquisition", {})
                )
            elif str(ctx.task_ir.get("task_type", "")) == "numerical":
                count_domain = {
                    "schema_version": "count_query_domain_v1",
                    "closed": False,
                    "state": "OPEN_COUNT_QUERY_GRAPH_MISSING",
                    "authority": None,
                    "blockers": ["count_query_graph_missing"],
                }
                snapshot["count_query_domain"] = count_domain
                snapshot["numerical_count_domain_closed"] = False
                snapshot["count_query_targeted_acquisition"] = {}
            snapshot["acquisition_closure_diagnostics"] = {
                **dict(snapshot.get("acquisition_closure_diagnostics", {})),
                "pre_query_blocking_reasons": acquisition_closure_reasons,
            }
            summary.set_stage(
                "query_domain",
                self._stage(
                    "completed",
                    query_domain_started,
                    scene_memory_open=bool(snapshot.get("scene_memory_open", True)),
                    relation_selector_domain_open=bool(
                        snapshot.get("relation_selector_domain_open", False)
                    ),
                    numerical_count_domain_open=bool(
                        snapshot.get("numerical_count_domain_open", False)
                    ),
                    numerical_count_domain_closed=bool(
                        snapshot.get("numerical_count_domain_closed", False)
                    ),
                    count_query_domain=dict(
                        snapshot.get("count_query_domain", {})
                    ),
                    numerical_count_answer_ready=bool(
                        snapshot.get("numerical_count_answer_ready", False)
                    ),
                    selector_closures=dict(
                        snapshot.get("query_domain", {}).get("selector_closures", {})
                    ),
                ),
            )
            summary.stages["scene_memory"]["snapshot"] = snapshot
            summary.stages["scene_memory"]["state_separation"] = {
                "scene_memory_open": bool(snapshot.get("scene_memory_open", True)),
                "relation_selector_domain_open": bool(
                    snapshot.get("relation_selector_domain_open", False)
                ),
                "numerical_count_domain_open": bool(
                    snapshot.get("numerical_count_domain_open", False)
                ),
                "numerical_count_answer_ready": bool(
                    snapshot.get("numerical_count_answer_ready", False)
                ),
            }

            query_started = time.monotonic()
            relation_verifier = None
            relation_engine = None
            task_type = str(ctx.task_ir.get("task_type", ""))
            count_graph_has_relations = bool(
                isinstance(count_graph, Mapping)
                and count_graph.get("relation_nodes")
            )
            if (
                task_type in {
                    "object_reference",
                    "instruction_following",
                }
                or task_type == "numerical" and count_graph_has_relations
            ):
                qwen = dict(self.runtime.get("qwen3vl", {}))
                # Relation verification is part of the Count Graph truth
                # path.  The model worker may return UNKNOWN when its socket
                # call runs out of episode time, but a configuration switch or
                # a pre-call time check must not silently replace the live
                # verifier with persistent-only evidence.
                relation_verifier = QwenRelationTupleVerifier(
                    endpoint=str(qwen.get("endpoint", "@mast3r_qwen3vl")),
                    episode_id=ctx.episode_id,
                    acquisition_id=ctx.acquisition_id,
                    world_snapshot_version=int(
                        snapshot.get("scene_version", 0)
                    ),
                    evidence_builder=RelationEvidenceBuilder(
                        output_dir / "06_relation_evidence",
                        maximum_pixels=int(
                            qwen.get("max_pixels", 1024 * 1024)
                        ),
                    ),
                    socket_request=socket_request,
                    deadline_unix=(
                        time.monotonic() + ctx.deadline.remaining()
                    ),
                    answer_reserve_seconds=ctx.deadline.mandatory_reserve,
                    station_id=ctx.station_id,
                    identity_version=int(
                        snapshot.get("identity_revision", 0)
                    ),
                    geometry_version=int(
                        snapshot.get("geometry_version", 0)
                    ),
                )
            relation_engine = RelationEngine(
                relation_verifier,
                persistent_relation_summary,
                snapshot.get("object_id_aliases", {}),
                identity_version=int(snapshot.get("identity_revision", 0)),
                geometry_version=int(snapshot.get("geometry_version", 0)),
            )
            if task_type == "numerical" and isinstance(count_graph, Mapping):
                count_graph_execution = execute_count_query_graph(
                    count_graph,
                    snapshot,
                    relation_engine,
                )
                refreshed_count_domain = assess_count_query_domain(
                    count_graph,
                    snapshot,
                    execution=count_graph_execution,
                )
                closure_state = dict(
                    refreshed_count_domain.get("closure_state", {})
                )
                closure_state["unknown_relation_tuples"] = list(
                    refreshed_count_domain.get("unknown_relation_tuples", ())
                )
                snapshot["count_resolution_state"] = closure_state
                snapshot["count_query_domain"] = refreshed_count_domain
                snapshot["count_query_targeted_acquisition"] = dict(
                    refreshed_count_domain.get("next_acquisition", {})
                )
                snapshot["numerical_count_domain_closed"] = bool(
                    refreshed_count_domain.get("closed") is True
                )
                # Domain closure is metadata over the tuple results already
                # evaluated for this frozen revision.  Finalize those results
                # in memory; rerunning the graph would call the live relation
                # verifier twice and create two worlds inside one acquisition.
                count_graph_execution = finalize_count_query_execution_domain(
                    count_graph_execution,
                    domain_closed=bool(
                        snapshot["numerical_count_domain_closed"]
                    ),
                )
                snapshot["count_query_execution"] = count_graph_execution
                snapshot["numerical_count_domain_open"] = not bool(
                    refreshed_count_domain.get("closed") is True
                )
                snapshot["numerical_count_domain_state"] = (
                    "READY"
                    if count_graph_execution.get("complete") is True
                    else "OPEN_PENDING_COUNT_AUTHORITY"
                )
                snapshot["numerical_count_answer_ready"] = bool(
                    count_graph_execution.get("complete") is True
                )
                snapshot["query_domain"] = {
                    **dict(snapshot.get("query_domain", {})),
                    "numerical_count_domain_open": bool(
                        snapshot.get("numerical_count_domain_open", False)
                    ),
                    "numerical_count_domain_state": str(
                        snapshot.get(
                            "numerical_count_domain_state",
                            "OPEN_PENDING_COUNT_AUTHORITY",
                        )
                    ),
                }
                summary.set_stage(
                    "count_query_graph",
                    self._stage(
                        "completed",
                        query_started,
                        domain=dict(snapshot.get("count_query_domain", {})),
                        complete=bool(count_graph_execution.get("complete")),
                        cardinality_lower_bound=count_graph_execution.get(
                            "cardinality_lower_bound"
                        ),
                        cardinality_upper_bound=count_graph_execution.get(
                            "cardinality_upper_bound"
                        ),
                        counted_target_ids=list(
                            count_graph_execution.get("counted_target_ids", ())
                        ),
                        unknown_target_ids=list(
                            count_graph_execution.get("unknown_target_ids", ())
                        ),
                        failure_reasons=list(
                            count_graph_execution.get("failure_reasons", ())
                        ),
                        relation_verification_count=len(
                            count_graph_execution.get(
                                "relation_verifications", ()
                            )
                        ),
                        relation_verifier_created=bool(
                            relation_verifier is not None
                        ),
                        relation_verifier_endpoint=(
                            str(getattr(relation_verifier, "endpoint", ""))
                            if relation_verifier is not None else ""
                        ),
                        relation_engine=relation_engine.diagnostics,
                    ),
                )
                summary.stages["scene_memory"]["snapshot"] = snapshot
                summary.stages["scene_memory"]["state_separation"].update({
                    "numerical_count_domain_open": bool(
                        snapshot.get("numerical_count_domain_open", False)
                    ),
                    "numerical_count_domain_closed": bool(
                        snapshot.get("numerical_count_domain_closed", False)
                    ),
                    "numerical_count_answer_ready": bool(
                        snapshot.get("numerical_count_answer_ready", False)
                    ),
                })
            revision_vector = {
                "scene_revision": int(snapshot.get("scene_version", 0)),
                "identity_revision": int(snapshot.get("identity_revision", 0)),
                "geometry_revision": int(snapshot.get("geometry_version", 0)),
            }
            tuple_evaluation: dict[str, Any] = {
                "relation_verifications": [],
                "candidate_domains": {},
                "selector_resolutions": [],
                "evaluated_entity_ids": [],
            }
            if task_type == "numerical" and isinstance(count_graph, Mapping):
                tuple_evaluation["relation_verifications"] = [
                    dict(value)
                    for value in snapshot.get("count_query_execution", {}).get(
                        "relation_verifications", ()
                    )
                    if isinstance(value, Mapping)
                    and value.get("persistent_state_inherited") is not True
                ]
            elif task_type in {"object_reference", "instruction_following"}:
                tuple_evaluation = evaluate_required_relation_tuples(
                    ctx.task_ir,
                    snapshot,
                    relation_engine,
                    entity_bindings=ctx.semantic_entity_bindings,
                    execution_steps=ctx.execution_steps or None,
                    current_step_index=int(ctx.current_step_index),
                    navigation_history=ctx.navigation_history,
                )
            current_relation_evidence = [
                dict(value)
                for value in tuple_evaluation.get("relation_verifications", ())
                if isinstance(value, Mapping)
            ]
            if ctx.task_ir.get("relations"):
                # Merge current-revision tuples in memory before policy.  The
                # history file is deliberately not written until after the
                # sole task solve has produced the action.
                persistent_store = merge_persistent_relation_evidence(
                    store=persistent_store,
                    ephemeral_evidence=current_relation_evidence,
                    task_ir=ctx.task_ir,
                    acquisition_id=ctx.acquisition_id,
                    station_id=ctx.station_id,
                    now_monotonic=time.monotonic(),
                )
                persistent_relation_summary = summarize_persistent_relation_evidence(
                    persistent_store,
                    ctx.task_ir,
                )
                snapshot["relation_revision"] = int(
                    persistent_store.get("relation_revision", 0)
                )
                persistent_store["relation_summaries"] = (
                    persistent_relation_summary.get("relations", {})
                )
                snapshot["relation_evidence"] = copy.deepcopy(
                    persistent_store
                )
            snapshot["persistent_relation_summary"] = persistent_relation_summary
            snapshot["relation_revision_transaction"] = {
                **revision_vector,
                "relation_revision": int(snapshot.get("relation_revision", 0)),
                "acquisition_id": str(ctx.acquisition_id),
                "evaluated_entity_ids": list(
                    tuple_evaluation.get("evaluated_entity_ids", ())
                ),
                "current_relation_record_count": len(current_relation_evidence),
                "task_solver_call_count": 1,
                "count_graph_evaluation_count": int(
                    task_type == "numerical" and isinstance(count_graph, Mapping)
                ),
                "persistent_write_timing": "after_action",
            }
            resolver = resolver_for(task_type, relation_engine)
            resolver_result = resolver.resolve(
                ctx.task_ir,
                snapshot,
                entity_bindings=ctx.semantic_entity_bindings,
                execution_steps=ctx.execution_steps or None,
                current_step_index=int(ctx.current_step_index),
                navigation_history=ctx.navigation_history,
                trajectory_monitor=ctx.trajectory_monitor,
            )
            if task_type == "numerical":
                certificate = resolver_result.diagnostics.get(
                    "count_certificate", {}
                )
                snapshot["count_certificate"] = dict(certificate)
                summary.stages["count_query_graph"]["count_certificate"] = dict(
                    certificate
                )
            execution = dict(
                resolver_result.diagnostics.get("task_evidence", {})
            )
            observed_revision_vector = {
                "scene_revision": int(snapshot.get("scene_version", 0)),
                "identity_revision": int(snapshot.get("identity_revision", 0)),
                "geometry_revision": int(snapshot.get("geometry_version", 0)),
            }
            if observed_revision_vector != revision_vector:
                raise RuntimeError("acquisition_revision_changed_during_decision")
            if current_relation_evidence:
                execution["relation_verifications"] = current_relation_evidence
            execution["persistent_relation_summary"] = persistent_relation_summary
            execution["relation_revision_transaction"] = copy.deepcopy(
                snapshot["relation_revision_transaction"]
            )
            summary.stages["scene_memory"]["snapshot"] = snapshot
            if str(ctx.task_ir.get("task_type")) == "instruction_following":
                raw_directives = execution.get("trajectory_directives", ())
                if raw_directives:
                    execution["trajectory_directives"] = (
                        compile_directive_geometries(
                            [
                                value
                                for value in raw_directives
                                if isinstance(value, Mapping)
                            ],
                            clearance_m=float(
                                ctx.navigation_config.get(
                                    "terrain_obstacle_clearance_m", 0.75
                                )
                            ),
                            acceptance_radius_m=float(
                                ctx.navigation_config.get(
                                    "terrain_waypoint_acceptance_radius_m", 0.30
                                )
                            ),
                        )
                    )
                    if resolver_result.status is ResolverStatus.NEED_EXECUTION:
                        resolver_result = replace(
                            resolver_result,
                            execution_need=ExecutionNeed(
                                task_type=task_type,
                                intent="EXECUTE_ORDERED_CONSTRAINT",
                                directives=tuple(
                                    dict(value)
                                    for value in execution["trajectory_directives"]
                                ),
                                current_step_index=int(ctx.current_step_index),
                                provenance=resolver_result.provenance,
                            ),
                        )
            summary.set_stage(
                "query_execution",
                self._stage(
                    "completed",
                    query_started,
                    execution=execution,
                ),
            )
            summary.set_stage(
                "task_resolution",
                {
                    "status": "completed",
                    "resolver": type(resolver).__name__,
                    "result": resolver_result.to_dict(),
                    "relation_engine": relation_engine.diagnostics,
                },
            )

            waypoint_started = time.monotonic()
            observation_intent = None
            execution_for_waypoint = dict(execution)
            if resolver_result.status is ResolverStatus.NEED_EVIDENCE:
                coordinator = EvidenceAcquisitionCoordinator()
                observation_intent = coordinator.plan(
                    resolver_result.evidence_need,
                    snapshot,
                )
                execution_for_waypoint["probe_object"] = dict(
                    observation_intent.target
                )
                execution_for_waypoint["trajectory_directives"] = []
            elif resolver_result.status is ResolverStatus.NEED_EXECUTION:
                execution_for_waypoint["probe_object"] = None
                execution_for_waypoint["trajectory_directives"] = [
                    dict(value)
                    for value in resolver_result.execution_need.directives
                ]
            waypoint_selection = self._waypoint_selection(
                execution_for_waypoint,
                snapshot,
                ctx,
            )
            if observation_intent is not None:
                waypoint_selection["observation_intent"] = (
                    observation_intent.to_dict()
                )
            if resolver_result.execution_need is not None:
                waypoint_selection["execution_intent"] = (
                    resolver_result.execution_need.to_dict()
                )
            waypoint_selection["elapsed_seconds"] = self._elapsed(waypoint_started)
            summary.set_stage("waypoint_selection", waypoint_selection)

            decision = finalize_resolver_result(
                ctx.task_ir,
                resolver_result,
                waypoint_selection=waypoint_selection,
            )
            summary.set_root_decision(decision)
            if ctx.task_ir.get("relations"):
                # Relation persistence is history only.  The decision above
                # was already produced from the frozen in-memory revision and
                # can never be changed by this write.
                relation_persistence_status = "completed"
                relation_persistence_error = ""
                try:
                    save_scene_relation_evidence(
                        ctx.scene_memory_path,
                        persistent_store,
                    )
                except Exception as exc:
                    # A history-write fault is observable diagnostics, never
                    # a reason to recompute or replace the action already
                    # derived from the in-memory revision transaction.
                    relation_persistence_status = "error"
                    relation_persistence_error = (
                        f"{type(exc).__name__}:{str(exc)[:240]}"
                    )
                summary.set_stage(
                    "scene_relation_evidence",
                    {
                        "status": relation_persistence_status,
                        "error": relation_persistence_error,
                        "path": str(ctx.scene_memory_path.resolve()),
                        "write_timing": "after_action",
                        "revision_transaction": copy.deepcopy(revision_vector),
                        "relation_revision": int(
                            snapshot.get("relation_revision", 0)
                        ),
                        "ephemeral_record_count": len(current_relation_evidence),
                        "record_count": len(persistent_store.get("records", {})),
                        "unresolved_relation_ids": list(
                            persistent_relation_summary.get(
                                "unresolved_relation_ids", ()
                            )
                        ),
                        "conflicted_relation_ids": list(
                            persistent_relation_summary.get(
                                "conflicted_relation_ids", ()
                            )
                        ),
                        "relations": persistent_relation_summary.get(
                            "relations", {}
                        ),
                    },
                )
                role_verifications = relation_role_semantic_verifications(
                    current_relation_evidence
                )
                semantic_feedback_status = "completed"
                semantic_feedback_error = ""
                semantic_feedback_touched: dict[int, str] = {}
                try:
                    semantic_feedback_touched = (
                        apply_scene_memory_semantic_verifications(
                            ctx.scene_memory_path,
                            role_verifications,
                        )
                        if role_verifications else {}
                    )
                except Exception as exc:
                    # The current action already belongs to the frozen input
                    # revision. Expose the failed feedback write and let the
                    # next acquisition remain fail-closed.
                    semantic_feedback_status = "error"
                    semantic_feedback_error = (
                        f"{type(exc).__name__}:{str(exc)[:240]}"
                    )
                summary.set_stage(
                    "semantic_role_feedback",
                    {
                        "status": semantic_feedback_status,
                        "error": semantic_feedback_error,
                        "write_timing": "after_action",
                        "verification_count": len(role_verifications),
                        "touched_object_states": {
                            str(key): value
                            for key, value in semantic_feedback_touched.items()
                        },
                    },
                )

        except Exception as exc:
            summary.set_root_decision(
                self._failure_decision(
                    ctx,
                    reason=(
                        f"lean_pipeline_exception:{type(exc).__name__}:"
                        f"{str(exc)[:300]}"
                    ),
                    scene_version=int(snapshot.get("scene_version", 0)),
                )
            )

        summary.timing = {
            "total_elapsed_seconds": self._elapsed(total_started),
            "remaining_seconds": ctx.deadline.remaining(),
            "mandatory_reserve_seconds": ctx.deadline.mandatory_reserve,
            "clock_owner": "ros_parent_monotonic_episode_clock",
        }
        summary.deadline.update({
            "remaining_seconds_after_child": ctx.deadline.remaining(),
            "elapsed_seconds_in_child": self._elapsed(total_started),
        })
        return summary
