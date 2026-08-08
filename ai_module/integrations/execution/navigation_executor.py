"""ROS-free owner of the official waypoint lifecycle.

Adapted from the former SC-NAV NavigationExecutor.  Arrival never advances a
route until the ROS adapter records a fresh, successfully evaluated acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
import uuid
from typing import Callable, Sequence


class NavigationState(str, Enum):
    CREATED = "CREATED"
    DISPATCHED = "DISPATCHED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    MOVING = "MOVING"
    ARRIVED = "ARRIVED"
    FAILED_STUCK = "FAILED_STUCK"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    CANCELLED = "CANCELLED"


@dataclass
class _Goal:
    goal_id: str
    pose: tuple[float, float, float]
    start: tuple[float, float, float]
    purpose: str
    state: NavigationState = NavigationState.CREATED
    published_at: float | None = None
    arrival_started: float | None = None
    acquisition_id: str | None = None
    failure_reason: str = ""


def _pose(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError("map_pose_incomplete")
    result = tuple(float(item) for item in value[:3])
    if not all(math.isfinite(item) for item in result):
        raise ValueError("map_pose_nonfinite")
    return result


class NavigationExecutor:
    def __init__(
        self,
        waypoint_sender: Callable[[float, float, float], bool],
        *,
        stop_sender: Callable[[bool], None] | None = None,
        acquisition_requester: Callable[[str], None] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._waypoint_sender = waypoint_sender
        self._stop_sender = stop_sender
        self._acquisition_requester = acquisition_requester
        self._timeout_s = max(1.0, float(timeout_s))
        self._episode_id = ""
        self._goals: list[_Goal] = []
        self._index = 0
        self._last_pose: tuple[float, float, float] | None = None
        self._route_complete = False

    def begin_episode(self, episode_id: str) -> None:
        self.cancel("episode_reset")
        self._episode_id = str(episode_id).strip()
        if not self._episode_id:
            raise ValueError("episode_id_empty")

    def dispatch_route(self, waypoints: Sequence[Sequence[float]], *, start_pose: Sequence[float] | None = None, purpose: str = "instruction") -> bool:
        normalized = [_pose(value) for value in waypoints]
        if not normalized or self.has_active_goal:
            return False
        start = _pose(start_pose) if start_pose is not None else self._last_pose
        if start is None:
            return False
        self._goals = []
        for index, target in enumerate(normalized):
            self._goals.append(_Goal(
                goal_id=f"{self._episode_id}:{uuid.uuid4().hex[:12]}:{index}",
                pose=target,
                start=start if index == 0 else normalized[index - 1],
                purpose=str(purpose),
            ))
        self._index = 0
        self._route_complete = False
        return self._dispatch_current()

    def _dispatch_current(self) -> bool:
        goal = self._goals[self._index]
        if self._stop_sender is not None:
            self._stop_sender(False)
        accepted = bool(self._waypoint_sender(*goal.pose))
        if not accepted:
            goal.state = NavigationState.CANCELLED
            goal.failure_reason = "official_publish_rejected"
            return False
        goal.state = NavigationState.DISPATCHED
        goal.published_at = time.monotonic()
        return True

    def update_odometry(self, x: float, y: float, heading: float, *, speed_mps: float | None = None, reach_dist: float = 0.50, dwell_s: float = 1.0) -> bool:
        pose = _pose((x, y, heading))
        self._last_pose = pose
        if not self.has_active_goal:
            return self._route_complete
        goal = self._goals[self._index]
        now = time.monotonic()
        if goal.published_at is not None and now - goal.published_at > self._timeout_s:
            goal.state = NavigationState.FAILED_TIMEOUT
            goal.failure_reason = "goal_timeout"
            if self._stop_sender is not None:
                self._stop_sender(True)
            return False
        displacement = math.dist(goal.start[:2], pose[:2])
        if goal.state == NavigationState.DISPATCHED:
            goal.state = NavigationState.ACKNOWLEDGED
        if goal.state == NavigationState.ACKNOWLEDGED and displacement >= 0.05:
            goal.state = NavigationState.MOVING
        distance = math.dist(goal.pose[:2], pose[:2])
        stopped = speed_mps is None or abs(float(speed_mps)) <= 0.15
        enough_probe_baseline = goal.purpose != "probe" or displacement >= 0.35
        # The official challenge stack is allowed to project a waypoint onto
        # traversable terrain.  For an exploratory PROBE the projected pose is
        # not echoed back to this module, so a stable stop after a real spatial
        # baseline is the observable arrival contract.  Ordered instruction
        # waypoints keep the exact-distance condition.
        arrived = distance <= float(reach_dist) or (
            goal.purpose == "probe" and displacement >= 0.35 and stopped
        )
        if arrived and stopped and enough_probe_baseline:
            if goal.arrival_started is None:
                goal.arrival_started = now
            elif now - goal.arrival_started >= max(0.0, float(dwell_s)):
                goal.state = NavigationState.ARRIVED
                if self._stop_sender is not None:
                    self._stop_sender(True)
                if self._acquisition_requester is not None:
                    self._acquisition_requester(goal.goal_id)
        else:
            goal.arrival_started = None
        return False

    def record_arrival_acquisition(self, acquisition_id: str, *, semantic_revalidated: bool) -> bool:
        if not self._goals or self._goals[self._index].state != NavigationState.ARRIVED:
            raise RuntimeError("arrival_acquisition_without_arrived_goal")
        if not semantic_revalidated:
            self.cancel("arrival_semantic_revalidation_failed")
            return False
        goal = self._goals[self._index]
        goal.acquisition_id = str(acquisition_id)
        if self._index + 1 < len(self._goals):
            self._index += 1
            return self._dispatch_current()
        self._route_complete = True
        return True

    def cancel(self, reason: str = "cancelled") -> None:
        if self._goals and not self._route_complete:
            goal = self._goals[self._index]
            if goal.state not in {
                NavigationState.CANCELLED,
                NavigationState.FAILED_STUCK,
                NavigationState.FAILED_TIMEOUT,
            }:
                goal.state = NavigationState.CANCELLED
                goal.failure_reason = str(reason)
                if self._stop_sender is not None:
                    self._stop_sender(True)

    @property
    def has_active_goal(self) -> bool:
        return bool(self._goals) and self._goals[self._index].state in {
            NavigationState.CREATED,
            NavigationState.DISPATCHED,
            NavigationState.ACKNOWLEDGED,
            NavigationState.MOVING,
        }

    @property
    def awaiting_arrival_acquisition(self) -> bool:
        return bool(self._goals) and self._goals[self._index].state == NavigationState.ARRIVED and self._goals[self._index].acquisition_id is None

    @property
    def route_complete(self) -> bool:
        return self._route_complete

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def current_purpose(self) -> str:
        return self._goals[self._index].purpose if self._goals else ""

    def snapshot(self) -> dict:
        return {
            "episode_id": self._episode_id,
            "active_index": self._index,
            "route_complete": self._route_complete,
            "goals": [
                {
                    "goal_id": goal.goal_id,
                    "pose": list(goal.pose),
                    "start": list(goal.start),
                    "state": goal.state.value,
                    "purpose": goal.purpose,
                    "acquisition_id": goal.acquisition_id,
                    "failure_reason": goal.failure_reason,
                }
                for goal in self._goals
            ],
        }
