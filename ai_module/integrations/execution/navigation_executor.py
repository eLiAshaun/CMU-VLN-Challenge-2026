"""ROS-free lifecycle for one semantic corridor with local waypoint chaining."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
import uuid
from typing import Callable, Mapping, Sequence


class NavigationState(str, Enum):
    CREATED = "CREATED"
    DISPATCHED = "DISPATCHED"
    ACTIVE = "ACTIVE"
    ARRIVED = "ARRIVED"
    NAVIGATION_FAILED = "NAVIGATION_FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class _Goal:
    goal_id: str
    pose: tuple[float, float, float]
    start: tuple[float, float, float]
    purpose: str
    metadata: dict[str, object]
    timeout_s: float
    state: NavigationState = NavigationState.CREATED
    published_at: float | None = None
    acquisition_id: str | None = None
    failure_reason: str = ""
    displacement_m: float = 0.0
    distance_to_goal_m: float | None = None
    actual_arrival_pose: tuple[float, float, float] | None = None
    arrival_latched: bool = False
    arrival_candidate_since: float | None = None
    arrival_candidate_samples: int = 0
    arrival_candidate_last_distance_m: float | None = None
    arrival_source: str = ""
    best_distance_to_goal_m: float | None = None


def _pose(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError("map_pose_incomplete")
    result = tuple(float(item) for item in value[:3])
    if not all(math.isfinite(item) for item in result):
        raise ValueError("map_pose_nonfinite")
    return result


class NavigationExecutor:
    """Execute one semantic waypoint segment from actual state estimation."""

    def __init__(
        self,
        waypoint_sender: Callable[[float, float, float], bool],
        *,
        acquisition_requester: Callable[[str], None] | None = None,
        failure_requester: Callable[[str], None] | None = None,
        event_reporter: Callable[[str, dict[str, object]], None] | None = None,
        timeout_s: float = 120.0,
        original_goal_acceptance_radius_m: float = 0.30,
        converter_projection_distance_m: float = 0.50,
        odometry_max_gap_s: float = 1.5,
        odometry_max_jump_m: float = 2.5,
        odometry_max_speed_mps: float = 3.0,
        expected_frame_id: str = "map",
        local_arrival_dwell_s: float = 0.15,
        local_arrival_min_samples: int = 2,
        stopped_speed_threshold_mps: float = 0.10,
        ack_diagnostics_enabled: bool = False,
    ) -> None:
        self._waypoint_sender = waypoint_sender
        self._acquisition_requester = acquisition_requester
        self._failure_requester = failure_requester
        self._event_reporter = event_reporter
        self._timeout_s = max(1.0, float(timeout_s))
        self._original_goal_acceptance_radius_m = float(
            original_goal_acceptance_radius_m
        )
        if (
            not math.isfinite(self._original_goal_acceptance_radius_m)
            or self._original_goal_acceptance_radius_m <= 0.0
        ):
            raise ValueError("original_goal_acceptance_radius_invalid")
        self._converter_projection_distance_m = float(
            converter_projection_distance_m
        )
        if (
            not math.isfinite(self._converter_projection_distance_m)
            or self._converter_projection_distance_m <= 0.0
        ):
            raise ValueError("converter_projection_distance_invalid")
        self._episode_id = ""
        self._goal: _Goal | None = None
        self._last_pose: tuple[float, float, float] | None = None
        self._segment_id = ""
        self._segment_waypoints: list[tuple[float, float, float]] = []
        self._segment_index = 0
        self._segment_context: dict[str, object] = {}
        self._segment_purpose = ""
        self._segment_timeout_s = self._timeout_s
        self._waypoint_cycle_complete = False
        # The official waypoint converter may replace the semantic
        # ``/way_point_with_heading`` coordinate with a traversable point on
        # ``/way_point``.  Keep that physical target separate from the
        # requested coordinate.  It is still state estimation, not the
        # converter acknowledgement, that decides arrival.
        self._converter_target_pose: tuple[float, float, float] | None = None
        self._converter_target_frozen = False
        self._odometry_max_gap_s = max(0.1, float(odometry_max_gap_s))
        self._odometry_max_jump_m = max(0.5, float(odometry_max_jump_m))
        self._odometry_max_speed_mps = max(0.1, float(odometry_max_speed_mps))
        self._expected_frame_id = str(expected_frame_id).strip().lstrip("/") or "map"
        self._local_arrival_dwell_s = max(0.0, float(local_arrival_dwell_s))
        self._local_arrival_min_samples = max(1, int(local_arrival_min_samples))
        self._stopped_speed_threshold_mps = max(
            0.0, float(stopped_speed_threshold_mps)
        )
        self._ack_diagnostics_enabled = bool(ack_diagnostics_enabled)
        self._last_odom_stamp_s: float | None = None
        self._last_accepted_stamp_s: float | None = None
        self._last_accepted_pose: tuple[float, float, float] | None = None
        self._last_accepted_speed_mps: float | None = None
        self._last_accepted_gap_s: float | None = None
        self._odometry_samples_seen = 0
        self._odometry_samples_accepted = 0
        self._odometry_samples_rejected = 0
        self._last_odometry_rejection = ""
        self._ack_count = 0
        self._last_ack_distance_m: float | None = None
        self._lock = threading.RLock()

    def _report(self, state: str, goal: _Goal, **fields: object) -> None:
        if self._event_reporter is None:
            return
        detail = goal.metadata
        payload = {
            "goal_id": goal.goal_id,
            "purpose": goal.purpose,
            "navigation_state": goal.state.value,
            "target_pose": list(goal.pose),
            "requested_waypoint_pose": detail.get("requested_waypoint_pose"),
            "physical_waypoint_pose": detail.get("physical_waypoint_pose"),
            "physical_waypoint_source": detail.get(
                "physical_waypoint_source", ""
            ),
            "physical_waypoint_target_frozen": bool(
                detail.get("physical_waypoint_target_frozen", False)
            ),
            "start_pose": list(goal.start),
            "actual_pose": list(self._last_pose) if self._last_pose else None,
            "expected_information_gain": detail.get("expected_information_gain", ""),
            "semantic_order": detail.get("order"),
            "semantic_action": detail.get("action", ""),
            "semantic_object_id": detail.get("semantic_object_id"),
            "semantic_object_class": detail.get("semantic_object_class", ""),
            "constraint_set_id": detail.get("constraint_set_id", ""),
            "step_index": detail.get("step_index"),
            "is_terminal": bool(detail.get("is_terminal", False)),
            "observation_region_id": str(
                detail.get("observation_region_id", "")
            ),
            "goal_timeout_seconds": goal.timeout_s,
            "local_waypoint_index": detail.get("local_waypoint_index"),
            "local_waypoint_count": detail.get("local_waypoint_count"),
            "displacement_m": goal.displacement_m,
            "distance_to_goal_m": goal.distance_to_goal_m,
            "arrival_source": goal.arrival_source,
            "arrival_state_estimation_stamp_seconds": self._last_accepted_stamp_s,
            "waypoint_role": detail.get("observation_waypoint_role", "OBSERVATION"),
            "waypoint_roles": detail.get("waypoint_roles", ()),
            "waypoint_context": dict(goal.metadata),
            **fields,
        }
        for key in (
            "probe_relation",
            "probe_anchor_object_ids",
            "probe_object_ids",
            "required_visible_object_ids",
            "probe_source_candidate_id",
            "probe_target_class",
            "probe_dependency_entity_id",
            "probe_requested_by_entity_id",
            "probe_id",
            "query_key",
            "binding_id",
            "relation_id",
            "observation_intent",
            "observation_waypoint_role",
            "waypoint_roles",
        ):
            if key in detail:
                payload[key] = detail[key]
        self._event_reporter(state, payload)

    def begin_episode(self, episode_id: str) -> None:
        with self._lock:
            self._begin_episode_locked(episode_id)

    def _begin_episode_locked(self, episode_id: str) -> None:
        self.cancel("episode_reset")
        self._episode_id = str(episode_id).strip()
        if not self._episode_id:
            raise ValueError("episode_id_empty")
        self._goal = None
        self._segment_id = ""
        self._segment_waypoints = []
        self._segment_index = 0
        self._segment_context = {}
        self._segment_purpose = ""
        self._waypoint_cycle_complete = False
        self._converter_target_pose = None
        self._converter_target_frozen = False
        self._last_pose = None
        self._last_odom_stamp_s = None
        self._last_accepted_stamp_s = None
        self._last_accepted_pose = None
        self._last_accepted_speed_mps = None
        self._last_accepted_gap_s = None
        self._odometry_samples_seen = 0
        self._odometry_samples_accepted = 0
        self._odometry_samples_rejected = 0
        self._last_odometry_rejection = ""
        self._ack_count = 0
        self._last_ack_distance_m = None

    def _dispatch_segment_index(self, start: tuple[float, float, float]) -> bool:
        target = self._segment_waypoints[self._segment_index]
        metadata = dict(self._segment_context)
        waypoint_roles = metadata.get("waypoint_roles")
        if (
            isinstance(waypoint_roles, Sequence)
            and not isinstance(waypoint_roles, (str, bytes))
            and self._segment_index < len(waypoint_roles)
        ):
            # Numerical probes may need a short local corridor.  Preserve the
            # role of the waypoint actually being dispatched so a transit
            # arrival cannot be mistaken for an observation station.
            metadata["observation_waypoint_role"] = str(
                waypoint_roles[self._segment_index]
            ).upper()
        metadata.update({
            "segment_id": self._segment_id,
            "local_waypoint_index": self._segment_index,
            "local_waypoint_count": len(self._segment_waypoints),
            "requested_waypoint_pose": list(target),
            "physical_waypoint_pose": list(target),
            "physical_waypoint_source": "/way_point_with_heading",
            "physical_waypoint_target_frozen": False,
        })
        self._converter_target_pose = target
        self._converter_target_frozen = False
        self._goal = _Goal(
            goal_id=f"{self._segment_id}:{self._segment_index}",
            pose=target,
            start=start,
            purpose=self._segment_purpose,
            metadata=metadata,
            timeout_s=self._segment_timeout_s,
            distance_to_goal_m=math.dist(start[:2], target[:2]),
            best_distance_to_goal_m=math.dist(start[:2], target[:2]),
        )
        if not bool(self._waypoint_sender(*target)):
            self._goal.state = NavigationState.CANCELLED
            self._goal.failure_reason = "waypoint_publish_rejected"
            self._report(
                "navigation_failed", self._goal, reason=self._goal.failure_reason
            )
            if self._failure_requester is not None:
                self._failure_requester(self._goal.goal_id)
            return False
        self._goal.state = NavigationState.DISPATCHED
        self._goal.published_at = time.monotonic()
        self._report("navigation_dispatched", self._goal)
        return True

    def dispatch_segment(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        start_pose: Sequence[float] | None = None,
        purpose: str = "instruction",
        waypoint_context: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        with self._lock:
            return self._dispatch_segment_locked(
                waypoints,
                start_pose=start_pose,
                purpose=purpose,
                waypoint_context=waypoint_context,
                timeout_s=timeout_s,
            )

    def _dispatch_segment_locked(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        start_pose: Sequence[float] | None = None,
        purpose: str = "instruction",
        waypoint_context: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Publish one semantic corridor with at most two semantic points."""
        if self.has_active_goal or self.awaiting_arrival_acquisition:
            return False
        if not isinstance(waypoint_context, Mapping) or not waypoints or len(waypoints) > 2:
            return False
        start = _pose(start_pose) if start_pose is not None else self._last_pose
        if start is None:
            return False
        try:
            segment = [_pose(value) for value in waypoints]
        except (TypeError, ValueError):
            return False
        self._segment_id = f"{self._episode_id}:{uuid.uuid4().hex[:12]}"
        self._segment_waypoints = segment
        self._segment_index = 0
        self._segment_context = dict(waypoint_context)
        self._segment_purpose = str(purpose)
        self._segment_timeout_s = (
            self._timeout_s
            if timeout_s is None
            else min(self._timeout_s, max(1.0, float(timeout_s)))
        )
        self._waypoint_cycle_complete = False
        return self._dispatch_segment_index(start)

    def dispatch_waypoint(
        self,
        waypoint: Sequence[float],
        *,
        start_pose: Sequence[float] | None = None,
        purpose: str = "instruction",
        waypoint_context: Mapping[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Execute a genuinely single-point goal through the segment lifecycle."""
        return self.dispatch_segment(
            [waypoint],
            start_pose=start_pose,
            purpose=purpose,
            waypoint_context=waypoint_context,
            timeout_s=timeout_s,
        )

    def update_odometry(
        self,
        x: float,
        y: float,
        heading: float,
        *,
        timestamp_seconds: float | None = None,
        linear_speed_mps: float | None = None,
        frame_id: str | None = None,
    ) -> bool:
        with self._lock:
            return self._update_odometry_locked(
                x,
                y,
                heading,
                timestamp_seconds=timestamp_seconds,
                linear_speed_mps=linear_speed_mps,
                frame_id=frame_id,
            )

    def _update_odometry_locked(
        self,
        x: float,
        y: float,
        heading: float,
        *,
        timestamp_seconds: float | None = None,
        linear_speed_mps: float | None = None,
        frame_id: str | None = None,
    ) -> bool:
        pose = _pose((x, y, heading))
        self._odometry_samples_seen += 1
        stamp = (
            None
            if timestamp_seconds is None
            else float(timestamp_seconds)
        )
        if linear_speed_mps is None:
            speed = None
        else:
            try:
                candidate_speed = float(linear_speed_mps)
            except (TypeError, ValueError):
                candidate_speed = float("nan")
            speed = (
                candidate_speed
                if math.isfinite(candidate_speed) and candidate_speed >= 0.0
                else None
            )
        rejection = ""
        normalized_frame = (
            str(frame_id).strip().lstrip("/")
            if frame_id is not None
            else ""
        )
        if not normalized_frame:
            rejection = "odometry_frame_missing"
        elif normalized_frame != self._expected_frame_id:
            rejection = "odometry_frame_mismatch"
        elif stamp is None or not math.isfinite(stamp) or stamp <= 0.0:
            rejection = "odometry_timestamp_invalid"
        elif self._last_odom_stamp_s is not None and stamp <= self._last_odom_stamp_s:
            rejection = "odometry_timestamp_not_monotonic"
        elif (
            self._last_accepted_stamp_s is not None
            and stamp - self._last_accepted_stamp_s > self._odometry_max_gap_s
        ):
            rejection = "odometry_gap_exceeded"
        elif (
            self._last_accepted_pose is not None
            and stamp is not None
            and self._last_accepted_stamp_s is not None
            and math.dist(pose[:2], self._last_accepted_pose[:2])
            > self._odometry_max_jump_m
        ):
            rejection = "odometry_jump_exceeded"
        if rejection:
            self._odometry_samples_rejected += 1
            self._last_odometry_rejection = rejection
            if stamp is not None and math.isfinite(stamp) and stamp > 0.0:
                self._last_odom_stamp_s = stamp
            if rejection in {
                "odometry_gap_exceeded",
                "odometry_jump_exceeded",
            }:
                # Keep the discontinuity as a rejection, then let the next
                # monotonic sample establish a fresh accepted baseline.
                self._last_accepted_stamp_s = None
                self._last_accepted_pose = None
                self._last_accepted_speed_mps = None
                self._last_accepted_gap_s = None
            return False

        gap = (
            None
            if self._last_accepted_stamp_s is None or stamp is None
            else max(0.0, stamp - self._last_accepted_stamp_s)
        )
        if speed is None or not math.isfinite(speed) or speed < 0.0:
            speed = (
                None
                if gap is None or gap <= 0.0 or self._last_accepted_pose is None
                else math.dist(pose[:2], self._last_accepted_pose[:2]) / gap
            )
        if not rejection and speed is not None and speed > self._odometry_max_speed_mps:
            rejection = "odometry_speed_exceeded"
        if rejection:
            self._odometry_samples_rejected += 1
            self._last_odometry_rejection = rejection
            if stamp is not None and math.isfinite(stamp) and stamp > 0.0:
                self._last_odom_stamp_s = stamp
            if rejection in {
                "odometry_gap_exceeded",
                "odometry_jump_exceeded",
                "odometry_speed_exceeded",
            }:
                self._last_accepted_stamp_s = None
                self._last_accepted_pose = None
                self._last_accepted_speed_mps = None
                self._last_accepted_gap_s = None
            return False
        self._last_pose = pose
        self._last_odom_stamp_s = stamp
        self._odometry_samples_accepted += 1
        self._last_odometry_rejection = ""
        self._last_accepted_stamp_s = stamp
        self._last_accepted_pose = pose
        self._last_accepted_speed_mps = speed
        self._last_accepted_gap_s = gap
        goal = self._goal
        if not self.has_active_goal or goal is None:
            return self._waypoint_cycle_complete
        now = time.monotonic()
        goal.displacement_m = math.dist(goal.start[:2], self._last_pose[:2])
        goal.distance_to_goal_m = math.dist(goal.pose[:2], self._last_pose[:2])
        if goal.best_distance_to_goal_m is None:
            goal.best_distance_to_goal_m = goal.distance_to_goal_m
        else:
            goal.best_distance_to_goal_m = min(
                goal.best_distance_to_goal_m,
                goal.distance_to_goal_m,
            )
        if goal.published_at is not None and now - goal.published_at > goal.timeout_s:
            self._fail(goal, "goal_timeout")
            return False
        if goal.state == NavigationState.DISPATCHED:
            goal.state = NavigationState.ACTIVE
            self._report("navigation_active", goal)
        if self._local_waypoint_reached(goal, now):
            return self._complete_local_waypoint(goal)
        return False

    def _local_waypoint_reached(self, goal: _Goal, now: float) -> bool:
        if goal.arrival_latched or self._last_pose is None:
            return False
        distance = math.dist(goal.pose[:2], self._last_pose[:2])
        goal.distance_to_goal_m = distance
        if distance > self._original_goal_acceptance_radius_m:
            goal.arrival_candidate_since = None
            goal.arrival_candidate_samples = 0
            goal.arrival_candidate_last_distance_m = None
            return False
        goal.arrival_candidate_last_distance_m = distance
        if goal.arrival_candidate_since is None:
            goal.arrival_candidate_since = now
            goal.arrival_candidate_samples = 0
        goal.arrival_candidate_samples += 1
        dwell_ready = (
            now - goal.arrival_candidate_since >= self._local_arrival_dwell_s
            and goal.arrival_candidate_samples >= self._local_arrival_min_samples
        )
        if not dwell_ready:
            return False
        is_terminal = bool(goal.metadata.get("is_terminal", False))
        if is_terminal:
            # A missing speed is deliberately not interpreted as stopped.
            speed = self._last_accepted_speed_mps
            if speed is None or speed > self._stopped_speed_threshold_mps:
                return False
        goal.arrival_latched = True
        goal.arrival_source = "/state_estimation"
        return True

    def _complete_local_waypoint(self, goal: _Goal) -> bool:
        if self._last_pose is None or goal.state not in {
            NavigationState.ACTIVE,
            NavigationState.DISPATCHED,
        }:
            return False
        goal.state = NavigationState.ARRIVED
        goal.actual_arrival_pose = self._last_pose
        fields = {
            "arrival_mode": goal.arrival_source or "/state_estimation",
            "official_distance_to_original_goal_m": goal.distance_to_goal_m,
        }
        if self._segment_index + 1 < len(self._segment_waypoints):
            self._report("navigation_local_waypoint_arrived", goal, **fields)
            self._segment_index += 1
            return self._dispatch_segment_index(self._last_pose)
        self._report("navigation_arrived", goal, **fields)
        if self._acquisition_requester is not None:
            self._acquisition_requester(goal.goal_id)
        return True

    def record_waypoint_reached(
        self, distance_to_original_goal_m: float | None = None
    ) -> bool:
        with self._lock:
            return self._record_waypoint_reached_locked(
                distance_to_original_goal_m
            )

    def _record_waypoint_reached_locked(
        self, distance_to_original_goal_m: float | None = None
    ) -> bool:
        """Record an optional converter ACK as diagnostics only."""
        self._ack_count += 1
        self._last_ack_distance_m = (
            None
            if distance_to_original_goal_m is None
            else float(distance_to_original_goal_m)
        )
        goal = self._goal
        if goal is not None and self.has_active_goal:
            # The converter publishes a projected point on ``/way_point``
            # immediately after its ACK and then moves that point forward by
            # waypointProjDis.  Freeze the last physical target before that
            # projection reset.  This does not complete the goal, publish an
            # answer, or bypass state-estimation arrival validation.
            if (
                self._converter_target_pose is not None
                and self._converter_target_is_terminal_candidate(
                    goal, self._converter_target_pose
                )
            ):
                goal.pose = self._converter_target_pose
                goal.metadata["physical_waypoint_pose"] = list(
                    self._converter_target_pose
                )
                goal.metadata["physical_waypoint_source"] = "/way_point"
                goal.metadata["physical_waypoint_target_frozen"] = True
                self._converter_target_frozen = True
        if self._ack_diagnostics_enabled and goal is not None:
            self._report(
                "navigation_ack_diagnostic",
                goal,
                arrival_mode="/way_point_reached",
                official_distance_to_original_goal_m=self._last_ack_distance_m,
                correctness_authority="diagnostic_only",
            )
        return False

    def observe_converter_waypoint(
        self,
        x: float,
        y: float,
        *,
        frame_id: str | None = None,
    ) -> bool:
        """Track the physical target emitted by the official converter.

        ``/way_point`` is the coordinate consumed by the local planner after
        terrain adjustment.  It is a target update, not an arrival event.
        Once the converter reports its own projection boundary, the target is
        frozen so its post-arrival look-ahead point cannot move the AI goal.
        """
        with self._lock:
            return self._observe_converter_waypoint_locked(
                x,
                y,
                frame_id=frame_id,
            )

    def _observe_converter_waypoint_locked(
        self,
        x: float,
        y: float,
        *,
        frame_id: str | None = None,
    ) -> bool:
        goal = self._goal
        if goal is None or not self.has_active_goal or self._converter_target_frozen:
            return False
        normalized_frame = (
            str(frame_id).strip().lstrip("/")
            if frame_id is not None
            else ""
        )
        if normalized_frame != self._expected_frame_id:
            return False
        try:
            target_x, target_y = float(x), float(y)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in (target_x, target_y)):
            return False
        target = (target_x, target_y, goal.pose[2])
        # ``/way_point`` is a continuously changing local-planner target.  It
        # can still contain the preceding goal, or a rolling look-ahead point
        # near the robot, immediately after a new semantic waypoint is
        # dispatched.  Such a point is not evidence that the requested
        # observation waypoint was reached.  Only the converter's terminal
        # terrain projection may replace the requested arrival target.
        if not self._converter_target_is_terminal_candidate(goal, target):
            return False
        previous_target = self._converter_target_pose
        if self._last_pose is not None and previous_target is not None:
            previous_distance = math.dist(
                previous_target[:2], self._last_pose[:2]
            )
            target_distance = math.dist(target[:2], self._last_pose[:2])
            target_jump = math.dist(target[:2], previous_target[:2])
            # waypointConverter publishes a 0.5 m look-ahead point after its
            # own local radius is reached.  Recognise that target reset from
            # the converter stream, but still require accepted odometry to
            # complete the AI goal.  This is geometry synchronisation, not a
            # wider arrival threshold.
            if (
                previous_distance <= self._original_goal_acceptance_radius_m
                and target_distance > previous_distance + 0.10
                and target_distance
                <= self._converter_projection_distance_m
                + self._original_goal_acceptance_radius_m
                and target_jump >= 0.25
            ):
                self._converter_target_frozen = True
                goal.pose = previous_target
                goal.metadata["physical_waypoint_pose"] = list(previous_target)
                goal.metadata["physical_waypoint_source"] = "/way_point"
                goal.metadata["physical_waypoint_target_frozen"] = True
                return True
        self._converter_target_pose = target
        goal.pose = target
        goal.metadata["physical_waypoint_pose"] = list(target)
        goal.metadata["physical_waypoint_source"] = "/way_point"
        goal.metadata["physical_waypoint_target_frozen"] = False
        if self._last_pose is not None:
            goal.distance_to_goal_m = math.dist(
                target[:2], self._last_pose[:2]
            )
            if goal.best_distance_to_goal_m is None:
                goal.best_distance_to_goal_m = goal.distance_to_goal_m
            else:
                goal.best_distance_to_goal_m = min(
                    goal.best_distance_to_goal_m,
                    goal.distance_to_goal_m,
                )
        # A converter target update invalidates a dwell candidate established
        # against an older physical target.  The next accepted odometry
        # samples must establish dwell for this target.
        goal.arrival_candidate_since = None
        goal.arrival_candidate_samples = 0
        goal.arrival_candidate_last_distance_m = None
        return True

    def _converter_target_is_terminal_candidate(
        self,
        goal: _Goal,
        target: Sequence[float],
    ) -> bool:
        requested = goal.metadata.get("requested_waypoint_pose")
        if (
            not isinstance(requested, Sequence)
            or isinstance(requested, (str, bytes))
            or len(requested) < 2
            or len(target) < 2
        ):
            return False
        try:
            distance = math.dist(
                (float(requested[0]), float(requested[1])),
                (float(target[0]), float(target[1])),
            )
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(distance)
            and distance <= self._converter_projection_distance_m
        )

    def _fail(self, goal: _Goal, reason: str) -> None:
        goal.state = NavigationState.NAVIGATION_FAILED
        goal.failure_reason = reason
        goal.actual_arrival_pose = self._last_pose
        self._report("navigation_failed", goal, reason=reason)
        # A failed goal is not an arrival transaction.  Let the ROS owner
        # decide whether the current, real pose is useful for a recovery
        # observation; never feed a failed goal into arrival acquisition.
        if self._failure_requester is not None:
            self._failure_requester(goal.goal_id)

    def release_after_trajectory_completion(self, pose: Sequence[float]) -> bool:
        with self._lock:
            return self._release_after_trajectory_completion_locked(pose)

    def _release_after_trajectory_completion_locked(self, pose: Sequence[float]) -> bool:
        actual = _pose(pose)
        self._last_pose = actual
        goal = self._goal
        if goal is None or self._waypoint_cycle_complete:
            return False
        goal.state = NavigationState.ARRIVED
        goal.actual_arrival_pose = actual
        self._waypoint_cycle_complete = True
        self._segment_index = len(self._segment_waypoints)
        self._report("navigation_released_after_trajectory", goal, completion_authority="actual_state_estimation_trajectory")
        return True

    def release_after_constraint_satisfied(
        self, pose: Sequence[float]
    ) -> str | None:
        with self._lock:
            return self._release_after_constraint_satisfied_locked(pose)

    def _release_after_constraint_satisfied_locked(
        self, pose: Sequence[float]
    ) -> str | None:
        """Release the active waypoint when its path constraint is satisfied."""
        actual = _pose(pose)
        self._last_pose = actual
        goal = self._goal
        if goal is None or not self.has_active_goal:
            return None
        goal.state = NavigationState.ARRIVED
        goal.actual_arrival_pose = actual
        goal.distance_to_goal_m = math.dist(goal.pose[:2], actual[:2])
        self._waypoint_cycle_complete = True
        self._segment_index = len(self._segment_waypoints)
        self._report(
            "navigation_released_after_constraint",
            goal,
            completion_authority="actual_state_estimation_trajectory",
        )
        return goal.goal_id

    def record_arrival_acquisition(
        self,
        acquisition_id: str,
        *,
        evaluation_completed: bool,
    ) -> bool:
        with self._lock:
            return self._record_arrival_acquisition_locked(
                acquisition_id,
                evaluation_completed=evaluation_completed,
            )

    def _record_arrival_acquisition_locked(
        self,
        acquisition_id: str,
        *,
        evaluation_completed: bool,
    ) -> bool:
        goal = self._goal
        if goal is None or goal.state != NavigationState.ARRIVED:
            raise RuntimeError("arrival_acquisition_without_arrived_goal")
        if not evaluation_completed:
            self.cancel("arrival_evaluation_failed")
            return False
        goal.acquisition_id = str(acquisition_id)
        self._report("arrival_acquisition_recorded", goal, acquisition_id=goal.acquisition_id)
        self._waypoint_cycle_complete = True
        return True

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            self._cancel_locked(reason)

    def _cancel_locked(self, reason: str = "cancelled") -> None:
        goal = self._goal
        if goal is None or self._waypoint_cycle_complete:
            return
        if goal.state not in {NavigationState.CANCELLED, NavigationState.NAVIGATION_FAILED}:
            goal.state = NavigationState.CANCELLED
            goal.failure_reason = str(reason)
            self._report("navigation_cancelled", goal, reason=goal.failure_reason)

    @property
    def has_active_goal(self) -> bool:
        return self._goal is not None and self._goal.state in {
            NavigationState.CREATED,
            NavigationState.DISPATCHED,
            NavigationState.ACTIVE,
        }

    @property
    def awaiting_arrival_acquisition(self) -> bool:
        return bool(
            self._goal is not None
            and self._goal.state == NavigationState.ARRIVED
            and self._goal.acquisition_id is None
            and not self._waypoint_cycle_complete
        )

    @property
    def waypoint_cycle_complete(self) -> bool:
        return self._waypoint_cycle_complete

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def current_purpose(self) -> str:
        return self._goal.purpose if self._goal else ""

    @property
    def last_pose(self) -> tuple[float, float, float] | None:
        with self._lock:
            return self._last_pose

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        goal = self._goal
        goal_snapshot = None
        if goal is not None:
            goal_snapshot = {
                "goal_id": goal.goal_id,
                "pose": list(goal.pose),
                "requested_pose": goal.metadata.get("requested_waypoint_pose"),
                "physical_waypoint_pose": goal.metadata.get(
                    "physical_waypoint_pose"
                ),
                "physical_waypoint_source": goal.metadata.get(
                    "physical_waypoint_source"
                ),
                "physical_waypoint_target_frozen": bool(
                    goal.metadata.get("physical_waypoint_target_frozen", False)
                ),
                "start": list(goal.start),
                "state": goal.state.value,
                "purpose": goal.purpose,
                "semantic_order": goal.metadata.get("order"),
                "semantic_action": goal.metadata.get("action", ""),
                "semantic_object_id": goal.metadata.get("semantic_object_id"),
                "semantic_object_class": goal.metadata.get("semantic_object_class", ""),
                "constraint_set_id": goal.metadata.get("constraint_set_id", ""),
                "step_index": goal.metadata.get("step_index"),
                "is_terminal": bool(goal.metadata.get("is_terminal", False)),
                "observation_region_id": str(
                    goal.metadata.get("observation_region_id", "")
                ),
                "acquisition_id": goal.acquisition_id,
                "failure_reason": goal.failure_reason,
                "displacement_m": goal.displacement_m,
                "distance_to_goal_m": goal.distance_to_goal_m,
                "actual_arrival_pose": list(goal.actual_arrival_pose) if goal.actual_arrival_pose else None,
                "expected_information_gain": goal.metadata.get("expected_information_gain", ""),
                "goal_timeout_seconds": goal.timeout_s,
                "arrival_latched": goal.arrival_latched,
                "arrival_candidate_samples": goal.arrival_candidate_samples,
                "arrival_candidate_last_distance_m": goal.arrival_candidate_last_distance_m,
                "arrival_source": goal.arrival_source,
                "best_distance_to_goal_m": goal.best_distance_to_goal_m,
                "waypoint_context": dict(goal.metadata),
            }
        return {
            "schema_version": "navigation_snapshot_v2",
            "episode_id": self._episode_id,
            "waypoint_cycle_complete": self._waypoint_cycle_complete,
            "policy": {
                "timeout_seconds": self._timeout_s,
                "local_completion_source": "/state_estimation",
                "local_completion_authority": "accepted_state_estimation_pose",
                "waypoint_ack_role": "diagnostic_only",
                "waypoint_ack_diagnostics_enabled": self._ack_diagnostics_enabled,
                "original_goal_acceptance_radius_m": (
                    self._original_goal_acceptance_radius_m
                ),
                "local_arrival_dwell_seconds": self._local_arrival_dwell_s,
                "local_arrival_min_samples": self._local_arrival_min_samples,
                "stopped_speed_threshold_mps": self._stopped_speed_threshold_mps,
                "odometry_max_gap_seconds": self._odometry_max_gap_s,
                "odometry_max_jump_m": self._odometry_max_jump_m,
                "odometry_max_speed_mps": self._odometry_max_speed_mps,
                "state_estimation_frame": self._expected_frame_id,
                "semantic_completion_source": "/state_estimation",
                "dispatch_mode": "semantic_corridor_local_waypoint_chaining",
            },
            "segment": {
                "segment_id": self._segment_id,
                "local_waypoint_index": self._segment_index,
                "local_waypoint_count": len(self._segment_waypoints),
                "remaining_waypoints": [
                    list(value)
                    for value in self._segment_waypoints[self._segment_index :]
                ],
            },
            "goal": goal_snapshot,
            "odometry": {
                "samples_seen": self._odometry_samples_seen,
                "samples_accepted": self._odometry_samples_accepted,
                "samples_rejected": self._odometry_samples_rejected,
                "last_rejection": self._last_odometry_rejection,
                "last_accepted_stamp_seconds": self._last_accepted_stamp_s,
                "last_accepted_gap_seconds": self._last_accepted_gap_s,
                "last_accepted_speed_mps": self._last_accepted_speed_mps,
            },
            "ack_diagnostics": {
                "count": self._ack_count,
                "last_distance_to_original_goal_m": self._last_ack_distance_m,
                "role": "diagnostic_only",
            },
        }
