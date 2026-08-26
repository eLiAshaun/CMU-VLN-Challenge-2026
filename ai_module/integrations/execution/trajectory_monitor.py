"""Pure state transition authority for instruction-following trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .trajectory_geometry import point_in_region, segment_intersects_region


@dataclass(frozen=True)
class TrajectoryUpdate:
    violation: dict[str, Any] | None
    progressed: bool
    became_complete: bool
    transition_sample_recorded: bool
    current_step_index: int
    satisfied_orders: tuple[int, ...]
    accepted_sample: bool
    sample_rejection_reason: str | None = None
    sample_gap_seconds: float | None = None
    sample_speed_mps: float | None = None
    terminal_relation_recheck: bool = False


def _update(
    episode_state: dict[str, Any],
    *,
    violation: dict[str, Any] | None = None,
    progressed: bool = False,
    became_complete: bool = False,
    transition_sample_recorded: bool = False,
    accepted_sample: bool = False,
    sample_rejection_reason: str | None = None,
    sample_gap_seconds: float | None = None,
    sample_speed_mps: float | None = None,
    terminal_relation_recheck: bool = False,
) -> TrajectoryUpdate:
    monitor = episode_state.get("trajectory_monitor", {})
    return TrajectoryUpdate(
        violation=violation,
        progressed=progressed,
        became_complete=became_complete,
        transition_sample_recorded=transition_sample_recorded,
        current_step_index=int(episode_state.get("current_step_index", 0)),
        satisfied_orders=tuple(monitor.get("satisfied_orders", ())),
        accepted_sample=accepted_sample,
        sample_rejection_reason=sample_rejection_reason,
        sample_gap_seconds=sample_gap_seconds,
        sample_speed_mps=sample_speed_mps,
        terminal_relation_recheck=terminal_relation_recheck,
    )


def _finite_pose(actual: list[float]) -> bool:
    return len(actual) >= 3 and all(math.isfinite(float(value)) for value in actual[:3])


def _region_contains(
    point: list[float],
    region: Mapping[str, Any],
    *,
    inner: bool = False,
) -> bool:
    if not inner:
        return point_in_region(point, region)
    inner_radius = region.get("approach_inner_radius_m", region.get("inner_radius_m"))
    if inner_radius is None:
        return point_in_region(point, region)
    try:
        radius = float(inner_radius)
    except (TypeError, ValueError):
        return False
    if radius <= 0.0:
        return point_in_region(point, region)
    try:
        outer_radius = float(region.get("radius_m", -1.0))
    except (TypeError, ValueError):
        outer_radius = -1.0
    if outer_radius < 0.0 or radius >= outer_radius:
        return point_in_region(point, region)
    inner_region = dict(region)
    inner_region["radius_m"] = radius
    return bool(
        point_in_region(point, region)
        and not point_in_region(point, inner_region)
    )


def terminal_region_contains(
    point: Sequence[float],
    region: Mapping[str, Any],
) -> bool:
    """Return whether an actual map pose is inside the stable stop region."""
    return _region_contains(point, region, inner=True)


def _segment_hits_approach_region(
    start: list[float],
    end: list[float],
    region: Mapping[str, Any],
) -> bool:
    inner_radius = region.get("approach_inner_radius_m", region.get("inner_radius_m"))
    if inner_radius is None:
        return segment_intersects_region(start, end, region)
    try:
        radius = float(inner_radius)
    except (TypeError, ValueError):
        return False
    if radius <= 0.0:
        return segment_intersects_region(start, end, region)
    try:
        outer_radius = float(region.get("radius_m", -1.0))
    except (TypeError, ValueError):
        outer_radius = -1.0
    if outer_radius < 0.0 or radius >= outer_radius:
        return segment_intersects_region(start, end, region)
    inner_region = dict(region)
    inner_region["radius_m"] = radius
    if not segment_intersects_region(start, end, region):
        return False
    samples = max(2, int(math.ceil(math.dist(start[:2], end[:2]) / 0.05)))
    return any(
        point_in_region(
            [
                float(start[0]) + (float(end[0]) - float(start[0])) * index / samples,
                float(start[1]) + (float(end[1]) - float(start[1])) * index / samples,
            ],
            region,
        )
        and not point_in_region(
            [
                float(start[0]) + (float(end[0]) - float(start[0])) * index / samples,
                float(start[1]) + (float(end[1]) - float(start[1])) * index / samples,
            ],
            inner_region,
        )
        for index in range(samples + 1)
    )


def _semantic_traversed(
    previous: list[float] | None,
    actual: list[float],
    region: Mapping[str, Any],
    action: str,
    monitor: dict[str, Any] | None = None,
) -> bool:
    if action in {"between", "pass_between"}:
        if previous is None or not isinstance(monitor, dict):
            return False
        ingress = region.get("ingress_xy")
        egress = region.get("egress_xy")
        if not (
            isinstance(ingress, Sequence)
            and len(ingress) >= 2
            and isinstance(egress, Sequence)
            and len(egress) >= 2
        ):
            return False
        dx = float(egress[0]) - float(ingress[0])
        dy = float(egress[1]) - float(ingress[1])
        scale = dx * dx + dy * dy
        if scale <= 1e-8:
            return False

        def progress(point: Sequence[float]) -> float:
            return (
                (float(point[0]) - float(ingress[0])) * dx
                + (float(point[1]) - float(ingress[1])) * dy
            ) / scale

        intersects = _segment_hits_approach_region(
            previous, actual, region
        )
        if intersects:
            monitor["between_min_progress"] = min(
                float(monitor.get("between_min_progress", 1.0)),
                progress(previous),
                progress(actual),
            )
            monitor["between_max_progress"] = max(
                float(monitor.get("between_max_progress", 0.0)),
                progress(previous),
                progress(actual),
            )
        crossed = bool(
            float(monitor.get("between_min_progress", 1.0)) <= 0.30
            and float(monitor.get("between_max_progress", 0.0)) >= 0.70
        )
        if crossed:
            monitor["semantic_ingress_seen"] = True
        return crossed
    if action in {"pass_near", "pass_by", "path_near"}:
        if previous is None:
            return False
        # ingress_xy/egress_xy are navigation-planning hints.  They are not
        # semantic evidence: a converter may project either point around an
        # obstacle or onto a legal local route that never reaches the object.
        # The ordered instruction is satisfied only by the actual accepted
        # state-estimation segment intersecting the compiled relation region.
        # This keeps waypoint arrival and semantic path evidence separate.
        traversed = bool(_segment_hits_approach_region(previous, actual, region))
        if traversed and isinstance(monitor, dict):
            monitor["semantic_ingress_seen"] = True
        return traversed
    if previous is not None and _region_contains(actual, region, inner=True):
        monitor_length = float(monitor.get("semantic_region_length_m", 0.0)) if isinstance(monitor, dict) else 0.0
        if isinstance(monitor, dict):
            monitor["semantic_region_length_m"] = monitor_length + math.dist(previous[:2], actual[:2])
            if str(region.get("kind", "")) == "approach_band":
                return monitor["semantic_region_length_m"] >= 0.20
    return _region_contains(actual, region, inner=True)


def _stop_ready(
    monitor: dict[str, Any],
    *,
    stamp_seconds: float,
    speed_mps: float | None,
    dwell_seconds: float,
    min_samples: int,
    stop_speed_mps: float,
) -> bool:
    if speed_mps is None or speed_mps > stop_speed_mps:
        monitor["terminal_dwell_since_seconds"] = None
        monitor["terminal_dwell_samples"] = 0
        return False
    since = monitor.get("terminal_dwell_since_seconds")
    samples = int(monitor.get("terminal_dwell_samples", 0))
    if since is None:
        since = float(stamp_seconds)
        samples = 0
    samples += 1
    monitor["terminal_dwell_since_seconds"] = float(since)
    monitor["terminal_dwell_samples"] = samples
    return (
        float(stamp_seconds) - float(since) >= max(0.0, dwell_seconds)
        and samples >= max(1, min_samples)
    )


def _requires_terminal_stop(action: str) -> bool:
    """Return whether a terminal semantic action requires a real stop.

    ``go_to`` is a terminal pose-entry predicate.  ``stop_at``
    (and the compatible ``stop_near`` spelling) are the only terminal
    actions that own the dwell/low-speed requirement.
    """
    return str(action).lower() in {"stop_at", "stop_near"}


def apply_actual_pose(
    episode_state: dict[str, Any],
    actual: list[float],
    stamp_seconds: float,
    *,
    audit_sample_recorded: bool,
    allow_terminal_completion: bool = True,
    linear_speed_mps: float | None = None,
    sample_frame_id: str | None = None,
    sample_episode_id: str | None = None,
    sample_config: Mapping[str, Any] | None = None,
) -> TrajectoryUpdate:
    """Apply one accepted ``/state_estimation`` pose without ROS I/O.

    Local route semantics start only after ``route_active`` is latched by the
    real waypoint publication. Samples before that boundary are still
    validated and retained as telemetry, but cannot satisfy a semantic step.
    """
    monitor = episode_state.setdefault("trajectory_monitor", {})
    if not isinstance(monitor, dict):
        episode_state["trajectory_monitor"] = monitor = {}
    config = dict(sample_config or {})
    expected_frame = str(
        config.get("state_estimation_frame", config.get("map_frame", "map"))
    ).strip().lstrip("/")
    max_speed = max(
        0.1,
        float(config.get("odometry_max_speed_mps", 3.0)),
    )
    max_gap = max(
        0.1,
        float(
            config.get(
                "odometry_max_gap_s",
                config.get("odometry_max_gap_seconds", 1.5),
            )
        ),
    )
    max_jump = max(0.5, float(config.get("odometry_max_jump_m", 2.5)))
    stop_speed = max(0.0, float(config.get("terminal_stop_speed_mps", 0.10)))
    dwell_seconds = max(0.0, float(config.get("terminal_dwell_seconds", 0.15)))
    min_samples = max(1, int(config.get("terminal_min_samples", 2)))

    if not _finite_pose(actual):
        monitor["last_sample_rejection_reason"] = "pose_nonfinite"
        return _update(
            episode_state,
            accepted_sample=False,
            sample_rejection_reason="pose_nonfinite",
        )
    try:
        stamp = float(stamp_seconds)
    except (TypeError, ValueError):
        stamp = float("nan")
    previous_stamp = monitor.get("last_accepted_stamp_seconds")
    last_seen_stamp = monitor.get("last_sample_stamp_seconds")
    previous_pose = monitor.get("last_accepted_pose_map")
    rejection = None
    gap = None
    normalized_frame = (
        str(sample_frame_id).strip().lstrip("/")
        if sample_frame_id is not None
        else ""
    )
    if not normalized_frame:
        rejection = "sample_frame_missing"
    elif normalized_frame != expected_frame:
        rejection = "sample_frame_mismatch"
    if not math.isfinite(stamp) or stamp <= 0.0:
        rejection = rejection or "timestamp_invalid"
    elif isinstance(last_seen_stamp, (int, float)) and stamp <= float(last_seen_stamp):
        rejection = rejection or "timestamp_not_monotonic"
    elif isinstance(previous_stamp, (int, float)):
        gap = stamp - float(previous_stamp)
        if gap > max_gap:
            rejection = rejection or "sample_gap_exceeded"
    if rejection is None and isinstance(previous_pose, list) and len(previous_pose) >= 3:
        try:
            jump = math.dist(actual[:2], previous_pose[:2])
        except (TypeError, ValueError):
            jump = float("inf")
        if jump > max_jump:
            rejection = "sample_jump_exceeded"
    speed = None
    if linear_speed_mps is not None:
        try:
            candidate_speed = float(linear_speed_mps)
        except (TypeError, ValueError):
            candidate_speed = float("nan")
        if math.isfinite(candidate_speed) and candidate_speed >= 0.0:
            speed = candidate_speed
    if speed is None and gap is not None and gap > 0.0 and isinstance(previous_pose, list):
        try:
            speed = math.dist(actual[:2], previous_pose[:2]) / gap
        except (TypeError, ValueError, ZeroDivisionError):
            speed = None
    if rejection is None and speed is not None and speed > max_speed:
        rejection = "sample_speed_exceeded"
    if sample_episode_id is not None and monitor.get("episode_id"):
        if str(sample_episode_id) != str(monitor.get("episode_id")):
            rejection = rejection or "sample_episode_mismatch"
    if math.isfinite(stamp) and stamp > 0.0:
        monitor["last_sample_stamp_seconds"] = stamp
    if rejection is not None:
        monitor["last_sample_rejection_reason"] = rejection
        diagnostics = monitor.setdefault("sample_diagnostics", {})
        diagnostics["rejected_count"] = int(diagnostics.get("rejected_count", 0)) + 1
        diagnostics["last_rejection"] = rejection
        if rejection in {
            "sample_gap_exceeded",
            "sample_jump_exceeded",
            "sample_speed_exceeded",
        }:
            # Do not connect the next accepted segment to a sample across the
            # rejected discontinuity.
            monitor["last_accepted_stamp_seconds"] = None
            monitor["last_accepted_pose_map"] = None
            monitor["last_accepted_speed_mps"] = None
            monitor["last_accepted_gap_seconds"] = None
            monitor["last_pose_map"] = None
            monitor["last_pose_stamp_seconds"] = None
        return _update(
            episode_state,
            accepted_sample=False,
            sample_rejection_reason=rejection,
            sample_gap_seconds=gap,
            sample_speed_mps=speed,
        )

    actual_pose = [float(value) for value in actual[:3]]
    monitor["last_accepted_stamp_seconds"] = stamp
    monitor["last_accepted_pose_map"] = actual_pose
    monitor["last_accepted_speed_mps"] = speed
    monitor["last_accepted_gap_seconds"] = gap
    monitor["last_sample_rejection_reason"] = None
    diagnostics = monitor.setdefault("sample_diagnostics", {})
    diagnostics["accepted_count"] = int(diagnostics.get("accepted_count", 0)) + 1
    diagnostics["last_accepted_stamp_seconds"] = stamp
    diagnostics["last_gap_seconds"] = gap
    diagnostics["last_speed_mps"] = speed

    if monitor.get("completed") is True:
        return _update(
            episode_state,
            accepted_sample=True,
            sample_gap_seconds=gap,
            sample_speed_mps=speed,
        )

    route_active = bool(monitor.get("route_active", False))
    activation_stamp = monitor.get("activation_stamp_seconds")
    if not route_active or (
        isinstance(activation_stamp, (int, float)) and stamp <= float(activation_stamp)
    ):
        monitor["last_pose_map"] = None
        monitor["last_pose_stamp_seconds"] = None
        return _update(
            episode_state,
            accepted_sample=True,
            sample_gap_seconds=gap,
            sample_speed_mps=speed,
        )

    previous = monitor.get("last_pose_map")
    previous_actual = previous if isinstance(previous, list) and len(previous) >= 3 else None
    previous_semantic_stamp = monitor.get("last_pose_stamp_seconds")

    def traversed(region: Mapping[str, Any]) -> bool:
        return bool(
            point_in_region(actual_pose, region)
            or (
                isinstance(previous_actual, list)
                and segment_intersects_region(previous_actual, actual_pose, region)
            )
        )

    constraints = list(monitor.get("constraints", ()))
    for region in constraints:
        if not isinstance(region, Mapping) or region.get("forbidden") is not True:
            continue
        if not traversed(region):
            continue
        violation = {
            "order": region.get("order"),
            "semantic_object_id": region.get("semantic_object_id"),
            "pose_map": actual_pose,
            "stamp_seconds": stamp,
            "sample_gap_seconds": gap,
            "sample_speed_mps": speed,
        }
        monitor["forbidden_violation"] = violation
        monitor["last_pose_map"] = actual_pose
        monitor["last_pose_stamp_seconds"] = stamp
        return _update(
            episode_state,
            violation=violation,
            accepted_sample=True,
            sample_gap_seconds=gap,
            sample_speed_mps=speed,
        )

    steps = episode_state.get("execution_steps", [])
    progressed = False
    terminal_relation_recheck = False
    # One accepted sample may advance at most the current active step.  The
    # next step must receive a later accepted sample, even if the current pose
    # also lies inside its region.
    if int(episode_state.get("current_step_index", 0)) < len(steps):
        step_index = int(episode_state["current_step_index"])
        step = steps[step_index]
        region = next((
            value for value in constraints
            if isinstance(value, Mapping)
            and value.get("forbidden") is not True
            and int(value.get("order", -1)) == step_index
        ), None)
        if region is not None and str(step.get("status")) in {"READY", "EXECUTING"}:
            action = str(step.get("action", region.get("action", ""))).lower()
            terminal = bool(step.get("is_terminal", False) or region.get("terminal", False))
            if not (terminal and not allow_terminal_completion):
                if action in {
                    "pass_near", "pass_by", "between", "pass_between", "path_near"
                }:
                    satisfied = _semantic_traversed(previous_actual, actual_pose, region, action, monitor)
                else:
                    satisfied = _region_contains(actual_pose, region, inner=True)
                if not satisfied:
                    if terminal:
                        monitor["terminal_dwell_since_seconds"] = None
                        monitor["terminal_dwell_samples"] = 0
                elif terminal and bool(region.get("selector_provisional", False)):
                    # Arrival at a provisional metric winner is real physical
                    # evidence, but it is not yet semantic completion.  Keep
                    # the active step unresolved and let the normal arrival
                    # transaction reacquire the scene so a new identity or
                    # anchor can rewrite the relation winner.
                    relation_id = str(region.get("selector_relation_id", ""))
                    existing_pending = monitor.get("terminal_relation_recheck_pending")
                    same_pending = (
                        isinstance(existing_pending, Mapping)
                        and int(existing_pending.get("step_index", -1)) == step_index
                        and str(existing_pending.get("relation_id", "")) == relation_id
                    )
                    # A stationary robot can generate many accepted odometry
                    # samples while it remains in the provisional terminal
                    # region.  Emit one acquisition event per active route;
                    # repeated samples are still physical evidence but must
                    # not retrigger the same transaction.
                    if not same_pending:
                        terminal_relation_recheck = True
                        monitor["terminal_relation_recheck_pending"] = {
                            "step_index": step_index,
                            "relation_id": relation_id,
                            "selector_state": str(region.get("selector_state", "")),
                            "pose_map": list(actual_pose),
                            "stamp_seconds": stamp,
                        }
                    monitor["terminal_dwell_since_seconds"] = None
                    monitor["terminal_dwell_samples"] = 0
                elif not (terminal and _requires_terminal_stop(action)) or _stop_ready(
                    monitor,
                    stamp_seconds=stamp,
                    speed_mps=speed,
                    dwell_seconds=dwell_seconds,
                    min_samples=min_samples,
                    stop_speed_mps=stop_speed,
                ):
                    step["status"] = "SATISFIED"
                    step["satisfaction_source"] = "actual_state_estimation_trajectory"
                    step["satisfied_pose_map"] = actual_pose
                    step["satisfied_stamp_seconds"] = stamp
                    step["satisfied_speed_mps"] = speed
                    region_center = region.get("center_xy")
                    minimum_distance = None
                    if isinstance(region_center, list) and len(region_center) >= 2:
                        try:
                            minimum_distance = math.dist(actual_pose[:2], region_center[:2])
                        except (TypeError, ValueError):
                            minimum_distance = None
                    evidence = {
                        "step_index": step_index,
                        "predicate": action,
                        "target_object_id": region.get("semantic_object_id"),
                        "target_geometry_version": region.get("geometry_version"),
                        "segment_start_pose": list(previous_actual) if previous_actual else None,
                        "segment_end_pose": list(actual_pose),
                        "segment_start_stamp_seconds": (
                            previous_semantic_stamp
                            if previous_actual else None
                        ),
                        "segment_end_stamp_seconds": stamp,
                        "minimum_distance_m": minimum_distance,
                        "intersection_result": bool(
                            action
                            in {
                                "pass_near",
                                "pass_by",
                                "between",
                                "pass_between",
                                "path_near",
                            }
                        ),
                        "region_parameters": dict(region),
                    }
                    step.setdefault("actual_trajectory_evidence", []).append(evidence)
                    satisfied_orders = monitor.setdefault("satisfied_orders", [])
                    if step_index not in satisfied_orders:
                        satisfied_orders.append(step_index)
                    episode_state["current_step_index"] = step_index + 1
                    progressed = True
                    monitor["last_pose_stamp_seconds"] = stamp

    transition_sample_recorded = False
    if progressed and not audit_sample_recorded:
        episode_state.setdefault("actual_trajectory", []).append({
            "pose_map": actual_pose,
            "stamp_seconds": stamp,
            "kind": "constraint_transition",
            "speed_mps": speed,
        })
        transition_sample_recorded = True

    became_complete = bool(
        int(episode_state.get("current_step_index", 0)) == len(steps)
        and steps
        and bool(steps[-1].get("is_terminal", False))
        and monitor.get("forbidden_violation") is None
    )
    if became_complete:
        monitor["completed"] = True
        monitor["completion_pose_map"] = actual_pose
        monitor["completion_stamp_seconds"] = stamp
        monitor["completion_speed_mps"] = speed
    monitor["last_pose_map"] = actual_pose
    monitor["last_pose_stamp_seconds"] = stamp
    return _update(
        episode_state,
        progressed=progressed,
        became_complete=became_complete,
        transition_sample_recorded=transition_sample_recorded,
        accepted_sample=True,
        sample_gap_seconds=gap,
        sample_speed_mps=speed,
        terminal_relation_recheck=terminal_relation_recheck,
    )
