#!/usr/bin/env python3
"""Capture live competition questions and run the model chain fail-closed.

The simulator camera is a 1920x640 equirectangular panorama.  This node keeps
that image untouched.  It never presents the panorama to MASt3R as a pinhole
camera.  Every terminal answer is authorized by RootFinalizer.  Time
exhaustion is an event delivered to that authority and never creates a
fallback answer.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import math
from typing import Mapping, Sequence

import cv2
import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, String

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.ros.output_adapter import RosOutputAdapter
from integrations.execution.evidence_acquisition import EvidenceAcquisitionCoordinator
from integrations.execution.navigation_executor import NavigationExecutor
from integrations.execution.query_executor import normalize_execution_steps
from integrations.execution.resolver_contracts import ResolverResult, ResolverStatus
from integrations.execution.root_finalizer import (
    ROOT_DECISION_SCHEMA,
    finalize_resolver_result,
)
from integrations.execution.trajectory_monitor import (
    TrajectoryUpdate,
    apply_actual_pose,
    terminal_region_contains,
)
from integrations.semantics.task_compiler import compile_task


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class LiveTaskProbe(Node):
    def __init__(self, ai_root: Path, output_root: Path) -> None:
        super().__init__("mast3r_live_task_probe")
        self.ai_root = ai_root
        self.output_root = output_root

        from config import load_config, create_session_context, write_effective_config

        self.app_config = load_config(
            asset_manifest_path=self.ai_root / "configs" / "model_assets.json"
        )
        self.session = create_session_context()
        self.mast3r_runtime = {
            "min_conf_thr": self.app_config.mast3r.pointmap_confidence_threshold,
            "matching_conf_thr": self.app_config.mast3r.matching_confidence_threshold,
            "max_keyframe_stations": self.app_config.mast3r.max_keyframe_stations,
            "reconstruction_mode": self.app_config.mast3r.reconstruction_mode,
        }

        configured_budget = self.app_config.runtime.time_budget.question_time_budget_seconds
        self.question_time_budget_seconds = max(
            1.0,
            float(
                os.environ.get(
                    "CHALLENGE_QUESTION_TIME_BUDGET_SECONDS",
                    configured_budget,
                )
            ),
        )
        self.answer_reserve_seconds = max(
            0.0, self.app_config.runtime.time_budget.answer_reserve_seconds
        )
        self.initial_acquisition_estimate_seconds = max(
            1.0,
            self.app_config.runtime.time_budget.initial_acquisition_estimate_seconds,
        )
        self.navigation_goal_timeout_seconds = max(
            1.0, self.app_config.runtime.navigation.goal_timeout_seconds
        )
        self.pipeline_order = list(self.app_config.runtime.pipeline.stage_order)
        self.model_switches = {
            "mast3r": self.app_config.mast3r.enabled,
            "yolo_world": self.app_config.yolo_world.enabled,
            "sam2": self.app_config.sam2.enabled,
            "qwen3vl": self.app_config.qwen3vl.enabled,
        }
        self.mast3r_checkpoint_ready = (
            self.ai_root
            / "checkpoints"
            / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        ).is_file()

        navigation_config = {
            "goal_timeout_seconds": self.app_config.runtime.navigation.goal_timeout_seconds,
            "terrain_waypoint_acceptance_radius_m": self.app_config.runtime.navigation.acceptance_radius_m,
            "waypoint_projection_distance_m": 0.50,
            "terrain_obstacle_height_threshold": self.app_config.runtime.navigation.obstacle_height_threshold_m,
            "terrain_obstacle_clearance_m": self.app_config.runtime.navigation.obstacle_clearance_m,
            "terrain_voxel_size_m": self.app_config.runtime.navigation.terrain_voxel_size_m,
            "semantic_goal_roi_m": self.app_config.runtime.navigation.semantic_goal_roi_m,
            "stop_surface_distance_m": self.app_config.runtime.navigation.stop_surface_distance_m,
            "stop_band_width_m": self.app_config.runtime.navigation.stop_band_width_m,
            "near_surface_distance_m": self.app_config.runtime.navigation.near_surface_distance_m,
            "near_band_width_m": self.app_config.runtime.navigation.near_band_width_m,
            "terrain_distance_weight": self.app_config.runtime.navigation.terrain_distance_weight,
            "next_step_alignment_weight": self.app_config.runtime.navigation.next_step_alignment_weight,
            "geometry_uncertainty_weight": self.app_config.runtime.navigation.geometry_uncertainty_weight,
            "odometry_max_gap_seconds": 1.5,
            "odometry_max_jump_m": 2.5,
            "odometry_max_speed_mps": 3.0,
            "state_estimation_frame": "map",
            "local_arrival_dwell_seconds": 0.15,
            "local_arrival_min_samples": 2,
            "terminal_stop_speed_mps": 0.10,
            "terminal_dwell_seconds": 0.15,
            "terminal_min_samples": 2,
            "waypoint_ack_diagnostics_enabled": False,
        }
        navigation_config = dict(navigation_config)
        self.navigation_config = navigation_config
        self.waypoint_ack_diagnostics_enabled = bool(
            navigation_config.get("waypoint_ack_diagnostics_enabled", False)
        )
        self.latest_image: np.ndarray | None = None
        self.latest_image_stamp: dict[str, int] | None = None
        self.latest_sensor_scan: PointCloud2 | None = None
        self.latest_registered_scan: PointCloud2 | None = None
        self.latest_terrain_map: PointCloud2 | None = None
        self.latest_terrain_map_ext: PointCloud2 | None = None
        self.latest_state_estimation: Odometry | None = None
        self.last_sensor_scan_monotonic: float | None = None
        self.last_registered_scan_monotonic: float | None = None
        self.last_image_monotonic: float | None = None
        self.last_sensor_scan_stamp_seconds: float | None = None
        self.last_registered_scan_stamp_seconds: float | None = None
        self.last_terrain_map_monotonic: float | None = None
        self.last_terrain_map_ext_monotonic: float | None = None
        self.last_state_estimation_monotonic: float | None = None
        self.last_world_alignment_ready = False
        # The public evaluation contract starts a new system for every language
        # command.  Historical run directories remain audit artifacts only and
        # must never seed the next process' runtime state.
        self.running = False
        self.lock = threading.Lock()
        self.state_file_lock = threading.Lock()
        self.active_question = ""
        self.active_episode_id = ""
        self.pending_initial_question = ""
        self.pending_initial_question_started_monotonic: float | None = None
        self.pending_initial_question_deadline_unix: float | None = None
        self.pending_arrival_goal_id: str | None = None
        self.pending_navigation_recovery_goal_id: str | None = None
        self.arrival_context_by_goal: dict[str, dict[str, object]] = {}
        self.dispatch_in_progress = False
        self.deferred_arrival_goal_id: str | None = None
        self.scene_memory_path: Path | None = None
        self.reconstruction_keyframes: list[dict[str, object]] = []
        self.episode_started_monotonic: float | None = None
        self.episode_deadline_unix: float | None = None
        self.chain_durations: list[float] = []
        self.navigation_durations: list[float] = []
        self.navigation_started: dict[str, float] = {}
        self.navigation_attempts: list[dict[str, object]] = []
        self.episode_state: dict[str, object] | None = None
        self.last_trajectory_audit_stamp_seconds: float | None = None
        self.probe_count = 0
        self.evidence_coordinator = EvidenceAcquisitionCoordinator()
        self.episode_terminalized = False
        self.time_budget_finalized = False
        self.time_budget_finalization_in_progress = False
        self.last_fresh_arrival_diagnostics: dict[str, object] = {}
        self.output_adapter = RosOutputAdapter(self)
        self.navigation = NavigationExecutor(
            self.output_adapter.publish_waypoint,
            acquisition_requester=self._request_arrival_acquisition,
            failure_requester=self._request_navigation_recovery,
            event_reporter=self._publish_navigation_event,
            timeout_s=float(navigation_config.get("goal_timeout_seconds", 120.0)),
            original_goal_acceptance_radius_m=float(
                navigation_config.get("terrain_waypoint_acceptance_radius_m", 0.30)
            ),
            converter_projection_distance_m=float(
                navigation_config.get("waypoint_projection_distance_m", 0.50)
            ),
            odometry_max_gap_s=float(
                navigation_config.get("odometry_max_gap_seconds", 1.5)
            ),
            odometry_max_jump_m=float(
                navigation_config.get("odometry_max_jump_m", 2.5)
            ),
            odometry_max_speed_mps=float(
                navigation_config.get("odometry_max_speed_mps", 3.0)
            ),
            expected_frame_id=str(
                navigation_config.get("state_estimation_frame", "map")
            ),
            local_arrival_dwell_s=float(
                navigation_config.get("local_arrival_dwell_seconds", 0.15)
            ),
            local_arrival_min_samples=int(
                navigation_config.get("local_arrival_min_samples", 2)
            ),
            stopped_speed_threshold_mps=float(
                navigation_config.get("terminal_stop_speed_mps", 0.10)
            ),
            ack_diagnostics_enabled=bool(
                navigation_config.get("waypoint_ack_diagnostics_enabled", False)
            ),
        )
        self.status_publisher = self.create_publisher(
            String, "/mast3r_chain/status", 50
        )
        self.create_subscription(Image, "/camera/image", self._on_image, 10)
        self.create_subscription(PointCloud2, "/sensor_scan", self._on_sensor_scan, 10)
        self.create_subscription(PointCloud2, "/registered_scan", self._on_registered_scan, 10)
        self.create_subscription(PointCloud2, "/terrain_map", self._on_terrain_map, 10)
        self.create_subscription(
            PointCloud2, "/terrain_map_ext", self._on_terrain_map_ext, 10
        )
        self.create_subscription(Odometry, "/state_estimation", self._on_state_estimation, 20)
        # The official converter publishes the terrain-adjusted point that
        # the local planner actually follows.  It is target geometry only;
        # arrival remains exclusively state-estimation based.
        self.create_subscription(
            PointStamped, "/way_point", self._on_converter_waypoint, 20
        )
        # Keep the converter boundary signal for target lifecycle
        # synchronisation and optional diagnostics.  This callback never
        # advances navigation or starts an arrival acquisition by itself.
        self.create_subscription(
            Float32, "/way_point_reached", self._on_waypoint_reached, 10
        )
        self.create_subscription(String, "/challenge_question", self._on_question, 10)
        self.create_timer(5.0, self._publish_heartbeat)
        self._publish_status("ready", detail="waiting_for_question")

    def _persist_episode_state(self, run_dir: Path | None = None) -> None:
        if self.episode_state is None or self.scene_memory_path is None:
            return
        with self.state_file_lock:
            payload = copy.deepcopy(self.episode_state)
            _write_json(
                self.scene_memory_path.parent / "episode_runtime_state.json",
                payload,
            )
            if run_dir is not None:
                _write_json(run_dir / "episode_runtime_state.json", payload)

    def _publish_status(self, state: str, **fields: object) -> None:
        payload = {
            "state": state,
            "node": self.get_name(),
            "time_monotonic": time.monotonic(),
            **fields,
        }
        serialized = json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        )
        message = String()
        message.data = serialized
        self.status_publisher.publish(message)
        self.get_logger().info(serialized)

    @staticmethod
    def _optical_center_xy(
        pose_payload: object,
    ) -> tuple[float, float] | None:
        if not isinstance(pose_payload, dict):
            return None
        xyz = pose_payload.get("position_xyz")
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
            return None
        try:
            return (float(xyz[0]), float(xyz[1]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _reuse_station_id(
        pose_xy: tuple[float, float] | None,
        registry: Mapping[str, object],
        *,
        default_station_id: str,
    ) -> str:
        """Reuse the station whose optical center is within the existing
        0.3m independent-viewpoint separation (same value as
        terrain_waypoint_acceptance_radius_m / minimum_independent_viewpoint
        _separation_m).  A repeat acquisition at the same physical pose is
        not a new independent station."""
        if pose_xy is None:
            return default_station_id
        for station_id, entry in registry.items():
            if not isinstance(entry, dict):
                continue
            xyz = entry.get("position_xyz")
            if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
                continue
            try:
                distance = math.hypot(
                    pose_xy[0] - float(xyz[0]),
                    pose_xy[1] - float(xyz[1]),
                )
            except (TypeError, ValueError):
                continue
            if distance <= 0.3:
                return str(station_id)
        return default_station_id

    def _publish_navigation_event(
        self,
        state: str,
        fields: dict[str, object],
    ) -> None:
        goal_id = str(fields.get("goal_id", ""))
        if state == "navigation_dispatched" and goal_id:
            self.navigation_started[goal_id] = time.monotonic()
            if str(fields.get("purpose", "")) == "evidence":
                # This is an attempt budget, not a success counter. Counting
                # only arrivals allowed one infeasible short probe to be
                # dispatched forever after each immediate navigation failure.
                self.probe_count += 1
            waypoint_context = fields.get("waypoint_context")
            self.navigation_attempts.append({
                "goal_id": goal_id,
                "purpose": str(fields.get("purpose", "")),
                "target_pose": list(fields.get("target_pose", ())),
                "requested_waypoint_pose": fields.get("requested_waypoint_pose")
                or (
                    fields.get("waypoint_context", {}).get("requested_waypoint_pose")
                    if isinstance(fields.get("waypoint_context"), Mapping)
                    else None
                ),
                "physical_waypoint_pose": fields.get("physical_waypoint_pose")
                or (
                    fields.get("waypoint_context", {}).get("physical_waypoint_pose")
                    if isinstance(fields.get("waypoint_context"), Mapping)
                    else None
                ),
                "physical_waypoint_source": str(
                    fields.get("physical_waypoint_source", "")
                ),
                "start_pose": list(fields.get("start_pose", ())),
                "expected_information_gain": str(
                    fields.get("expected_information_gain", "")
                ),
                "semantic_order": fields.get("semantic_order"),
                "semantic_action": str(fields.get("semantic_action", "")),
                "semantic_object_id": fields.get("semantic_object_id"),
                "semantic_object_class": str(
                    fields.get("semantic_object_class", "")
                ),
                "constraint_set_id": str(fields.get("constraint_set_id", "")),
                "step_index": fields.get("step_index"),
                "local_waypoint_index": fields.get("local_waypoint_index"),
                "local_waypoint_count": fields.get("local_waypoint_count"),
                "is_terminal": bool(fields.get("is_terminal", False)),
                "observation_region_id": str(
                    fields.get("observation_region_id", "")
                ),
                "status": "dispatched",
                "actual_arrival_pose": None,
                "arrival_source": None,
                "waypoint_context": (
                    dict(waypoint_context)
                    if isinstance(waypoint_context, Mapping)
                    else None
                ),
            })
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
                if key in fields:
                    self.navigation_attempts[-1][key] = fields[key]
            if self.episode_state is not None:
                self.episode_state["active_waypoint"] = {
                    "constraint_set_id": str(fields.get("constraint_set_id", goal_id)),
                    "goal_id": goal_id,
                    "step_index": fields.get("step_index"),
                    "local_waypoint_index": fields.get(
                        "local_waypoint_index"
                    ),
                    "local_waypoint_count": fields.get(
                        "local_waypoint_count"
                    ),
                    "semantic_target_object_id": fields.get(
                        "semantic_object_id"
                    ),
                    "is_terminal": bool(fields.get("is_terminal", False)),
                    "purpose": str(fields.get("purpose", "")),
                    "status": "DISPATCHED",
                    "waypoint_context": (
                        dict(waypoint_context)
                        if isinstance(waypoint_context, Mapping)
                        else None
                    ),
                }
                self.episode_state["active_waypoint_context"] = (
                    dict(waypoint_context)
                    if isinstance(waypoint_context, Mapping)
                    else None
                )
                step_index = fields.get("step_index")
                steps = self.episode_state.get("execution_steps", [])
                if (
                    str(fields.get("purpose", "")) == "instruction"
                    and isinstance(step_index, int)
                    and 0 <= step_index < len(steps)
                    and int(self.episode_state.get("current_step_index", 0))
                    == step_index
                ):
                    step = steps[step_index]
                    if str(step.get("status")) == "BOUND":
                        step["status"] = "EXECUTING"
        elif state in {"navigation_local_waypoint_arrived", "navigation_arrived"} and goal_id:
            started = self.navigation_started.pop(goal_id, None)
            if started is not None:
                self.navigation_durations.append(max(0.0, time.monotonic() - started))
            active_context_before_arrival = (
                self.episode_state.get("active_waypoint_context")
                if self.episode_state is not None
                else None
            )
            terminal_region_entered = False
            arrival_pose_known = False
            if isinstance(active_context_before_arrival, Mapping):
                terminal_region = active_context_before_arrival.get(
                    "semantic_region"
                )
                arrival_pose = fields.get("actual_pose")
                if (
                    not isinstance(arrival_pose, Sequence)
                    or isinstance(arrival_pose, (str, bytes))
                    or len(arrival_pose) < 2
                ) and self.episode_state is not None:
                    monitor = self.episode_state.get("trajectory_monitor", {})
                    arrival_pose = (
                        monitor.get("last_pose_map")
                        if isinstance(monitor, Mapping) else None
                    )
                if (
                    isinstance(arrival_pose, Sequence)
                    and not isinstance(arrival_pose, (str, bytes))
                    and len(arrival_pose) >= 2
                ):
                    try:
                        arrival_pose_known = all(
                            math.isfinite(float(value))
                            for value in arrival_pose[:3]
                        )
                    except (TypeError, ValueError):
                        arrival_pose_known = False
                if (
                    isinstance(terminal_region, Mapping)
                    and isinstance(arrival_pose, Sequence)
                    and not isinstance(arrival_pose, (str, bytes))
                    and len(arrival_pose) >= 2
                ):
                    try:
                        terminal_region_entered = terminal_region_contains(
                            [float(value) for value in arrival_pose[:3]],
                            terminal_region,
                        )
                    except (TypeError, ValueError):
                        terminal_region_entered = False
            terminal_stop_dwell = bool(
                state == "navigation_arrived"
                and isinstance(active_context_before_arrival, Mapping)
                and bool(
                    active_context_before_arrival.get(
                        "is_terminal",
                        active_context_before_arrival.get("terminal", False),
                    )
                )
                and str(active_context_before_arrival.get("action", "")).lower()
                in {"stop_at", "stop_near"}
                # If the navigation event has no usable pose, preserve the
                # active terminal route and let the next accepted
                # /state_estimation sample decide whether the robot is in the
                # region and has completed dwell.  A known pose outside the
                # region still follows the normal replan path.
                and (terminal_region_entered or not arrival_pose_known)
                and not bool(
                    active_context_before_arrival.get(
                        "selector_provisional", False
                    )
                )
            )
            if self.episode_state is not None and state == "navigation_arrived":
                active_waypoint = self.episode_state.get("active_waypoint")
                if isinstance(active_waypoint, dict):
                    active_waypoint["status"] = (
                        "ARRIVED_DWELLING"
                        if terminal_stop_dwell
                        else "ARRIVED"
                    )
                if not terminal_stop_dwell:
                    self.episode_state["active_waypoint"] = None
            for attempt in reversed(self.navigation_attempts):
                if str(attempt.get("goal_id")) == goal_id:
                    attempt["status"] = (
                        "local_waypoint_arrived"
                        if state == "navigation_local_waypoint_arrived"
                        else "arrived"
                    )
                    attempt["arrival_distance_to_original_goal_m"] = fields.get(
                        "distance_to_goal_m"
                    )
                    attempt["actual_arrival_pose"] = fields.get("actual_pose")
                    attempt["requested_waypoint_pose"] = fields.get(
                        "requested_waypoint_pose",
                        attempt.get("requested_waypoint_pose"),
                    )
                    attempt["physical_waypoint_pose"] = fields.get(
                        "target_pose", attempt.get("physical_waypoint_pose")
                    )
                    attempt["physical_waypoint_source"] = fields.get(
                        "physical_waypoint_source",
                        attempt.get("physical_waypoint_source", ""),
                    )
                    attempt["actual_displacement_m"] = fields.get(
                        "displacement_m"
                    )
                    attempt["arrival_source"] = fields.get("arrival_source")
                    attempt["waypoint_context"] = fields.get("waypoint_context")
                    attempt["arrival_state_estimation_stamp_seconds"] = fields.get(
                        "arrival_state_estimation_stamp_seconds"
                    )
                    context = attempt.get("waypoint_context")
                    if (
                        str(attempt.get("purpose", "")) == "evidence"
                        and state == "navigation_arrived"
                    ):
                        attempt["navigation_subgoal_reached"] = True
                        self.arrival_context_by_goal[goal_id] = {
                            "goal_id": goal_id,
                            "actual_arrival_pose": fields.get("actual_pose"),
                            "arrival_state_estimation_stamp_seconds": fields.get(
                                "arrival_state_estimation_stamp_seconds"
                            ),
                            "arrival_received_monotonic": time.monotonic(),
                            "navigation_subgoal_reached": True,
                            "observation_intent": (
                                dict(context.get("observation_intent", {}))
                                if isinstance(context, Mapping)
                                and isinstance(context.get("observation_intent"), Mapping)
                                else None
                            ),
                            "waypoint_context": dict(context)
                            if isinstance(context, Mapping)
                            else {},
                        }
                    # Physical arrival and semantic-step completion are
                    # different events.  The trajectory monitor runs before
                    # NavigationExecutor on the same accepted odometry
                    # sample, so this is the authoritative post-arrival
                    # semantic state that the next waypoint selection must
                    # see.  Keeping it in navigation history lets the
                    # planner choose another member of the semantic goal set
                    # after an arrival outside that set instead of replaying
                    # the same physical point at the same station.
                    step_index = attempt.get("step_index")
                    semantic_progressed = False
                    semantic_status = None
                    semantic_region = None
                    if (
                        self.episode_state is not None
                        and isinstance(step_index, int)
                    ):
                        steps = self.episode_state.get("execution_steps", ())
                        if 0 <= step_index < len(steps):
                            step = steps[step_index]
                            if isinstance(step, Mapping):
                                semantic_status = str(step.get("status", ""))
                                semantic_progressed = semantic_status == "SATISFIED"
                        context = attempt.get("waypoint_context")
                        if isinstance(context, Mapping):
                            raw_region = context.get("semantic_region")
                            if isinstance(raw_region, Mapping):
                                semantic_region = dict(raw_region)
                    attempt["semantic_progressed"] = semantic_progressed
                    attempt["semantic_region_entered"] = bool(
                        terminal_region_entered
                    )
                    # Region entry is physical evidence only.  A stable
                    # stop-at terminal still needs state-estimation dwell,
                    # and a provisional selector cannot satisfy the step at
                    # all, so do not label either case as semantic success.
                    attempt["semantic_region_satisfied"] = bool(
                        semantic_progressed
                    )
                    attempt["semantic_step_status"] = semantic_status
                    attempt["semantic_region"] = semantic_region
                    break
            if self.episode_state is not None and state == "navigation_arrived":
                monitor = self.episode_state.get("trajectory_monitor", {})
                terminal_dwell_pending = False
                if isinstance(monitor, dict):
                    dwell_complete = monitor.get("completed") is True
                    terminal_dwell_pending = bool(
                        terminal_stop_dwell and not dwell_complete
                    )
                    monitor["route_active"] = terminal_dwell_pending
                active_context = self.episode_state.get("active_waypoint_context")
                if isinstance(active_context, Mapping):
                    self.episode_state["last_completed_waypoint_context"] = dict(
                        active_context
                    )
                if not terminal_dwell_pending:
                    self.episode_state["active_waypoint_context"] = None
        elif state == "navigation_released_after_constraint" and goal_id:
            started = self.navigation_started.pop(goal_id, None)
            if started is not None:
                self.navigation_durations.append(
                    max(0.0, time.monotonic() - started)
                )
            if self.episode_state is not None:
                self.episode_state["active_waypoint"] = None
                monitor = self.episode_state.get("trajectory_monitor", {})
                if isinstance(monitor, dict):
                    monitor["route_active"] = False
                active_context = self.episode_state.get("active_waypoint_context")
                if isinstance(active_context, Mapping):
                    self.episode_state["last_completed_waypoint_context"] = dict(
                        active_context
                    )
                self.episode_state["active_waypoint_context"] = None
            for attempt in reversed(self.navigation_attempts):
                if str(attempt.get("goal_id")) == goal_id:
                    attempt["status"] = "semantic_step_satisfied"
                    attempt["actual_arrival_pose"] = fields.get("actual_pose")
                    attempt["distance_to_goal_m_when_released"] = fields.get(
                        "distance_to_goal_m"
                    )
                    attempt["arrival_source"] = fields.get("completion_authority")
                    break
        elif state in {"navigation_released_after_trajectory", "arrival_acquisition_recorded"} and goal_id:
            if self.episode_state is not None:
                monitor = self.episode_state.get("trajectory_monitor", {})
                active_context = self.episode_state.get(
                    "active_waypoint_context"
                )
                terminal_dwell_pending = bool(
                    state == "arrival_acquisition_recorded"
                    and isinstance(monitor, Mapping)
                    and monitor.get("route_active") is True
                    and monitor.get("completed") is not True
                    and isinstance(active_context, Mapping)
                    and bool(
                        active_context.get(
                            "is_terminal",
                            active_context.get("terminal", False),
                        )
                    )
                    and str(active_context.get("action", "")).lower()
                    in {"stop_at", "stop_near"}
                    and not bool(
                        active_context.get("selector_provisional", False)
                    )
                )
                if isinstance(monitor, dict):
                    if not terminal_dwell_pending:
                        monitor["route_active"] = False
                if state == "navigation_released_after_trajectory":
                    active_waypoint = self.episode_state.get("active_waypoint")
                    if isinstance(active_waypoint, Mapping):
                        completed_context = active_waypoint.get(
                            "waypoint_context"
                        )
                        if isinstance(completed_context, Mapping):
                            self.episode_state[
                                "last_completed_waypoint_context"
                            ] = dict(completed_context)
                        active_waypoint = dict(active_waypoint)
                        active_waypoint["status"] = "TRAJECTORY_COMPLETE"
                        self.episode_state["active_waypoint"] = active_waypoint
                    self.episode_state["active_waypoint"] = None
                if not terminal_dwell_pending:
                    self.episode_state["active_waypoint_context"] = None
        elif state == "navigation_failed" and goal_id:
            started = self.navigation_started.pop(goal_id, None)
            if started is not None:
                self.navigation_durations.append(max(0.0, time.monotonic() - started))
            for attempt in reversed(self.navigation_attempts):
                if str(attempt.get("goal_id")) == goal_id:
                    attempt["status"] = "failed"
                    attempt["failure_reason"] = str(
                        fields.get("reason", "navigation_failed")
                    )
                    attempt["actual_arrival_pose"] = fields.get("actual_pose")
                    attempt["requested_waypoint_pose"] = fields.get(
                        "requested_waypoint_pose",
                        attempt.get("requested_waypoint_pose"),
                    )
                    attempt["physical_waypoint_pose"] = fields.get(
                        "target_pose", attempt.get("physical_waypoint_pose")
                    )
                    attempt["physical_waypoint_source"] = fields.get(
                        "physical_waypoint_source",
                        attempt.get("physical_waypoint_source", ""),
                    )
                    attempt["displacement_m"] = fields.get("displacement_m")
                    attempt["distance_to_goal_m"] = fields.get(
                        "distance_to_goal_m"
                    )
                    break
            if self.episode_state is not None:
                active_waypoint = self.episode_state.get("active_waypoint")
                if isinstance(active_waypoint, dict):
                    active_waypoint["status"] = "NAVIGATION_FAILED"
                self.episode_state["active_waypoint"] = None
                self.episode_state["active_waypoint_context"] = None
                monitor = self.episode_state.get("trajectory_monitor", {})
                if isinstance(monitor, dict):
                    monitor["route_active"] = False
        elif state == "navigation_ack_diagnostic":
            if self.episode_state is not None:
                diagnostics = self.episode_state.setdefault(
                    "ack_diagnostics", {}
                )
                if isinstance(diagnostics, dict):
                    diagnostics["count"] = int(diagnostics.get("count", 0)) + 1
                    diagnostics["last"] = dict(fields)
        self._persist_episode_state()
        self._publish_status(state, **fields)

    def _seconds_remaining(self) -> float | None:
        if self.episode_started_monotonic is None:
            return None
        elapsed_monotonic = max(
            0.0, time.monotonic() - self.episode_started_monotonic
        )
        return max(0.0, self.question_time_budget_seconds - elapsed_monotonic)

    def _estimated_next_probe_cycle_seconds(self) -> float:
        observed_chain = (
            max(self.chain_durations[-3:]) if self.chain_durations else 0.0
        )
        task_type = str(
            (self.episode_state or {}).get("task_ir", {}).get("task_type", "")
        )
        step_index = int((self.episode_state or {}).get("current_step_index", 0))
        perception_count = int(
            self._current_step_runtime(step_index).get("perception_count", 0) or 0
        )
        if perception_count <= 0:
            planned_chain = max(
                self.initial_acquisition_estimate_seconds,
                observed_chain,
            )
        elif task_type != "instruction_following":
            # The next evidence pass is a focused two-view or one-view audit,
            # not another eight-view initial panorama.
            planned_chain = 29.0 if perception_count == 1 else 17.0
        else:
            planned_chain = 23.0
        navigation_estimate = (
            max(self.navigation_durations[-3:])
            if self.navigation_durations
            else min(
                30.0,
                max(10.0, 0.25 * self.initial_acquisition_estimate_seconds),
            )
        )
        return max(1.0, planned_chain + navigation_estimate)

    def _estimated_chain_seconds(self) -> float:
        return max(
            1.0,
            max(self.chain_durations[-3:])
            if self.chain_durations
            else self.initial_acquisition_estimate_seconds,
        )

    def _current_step_runtime(self, step_index: int | None = None) -> dict[str, object]:
        if self.episode_state is None:
            return {
                "perception_count": 0,
                "probe_dispatch_count": 0,
                "navigation_dispatch_count": 0,
            }
        if step_index is None:
            step_index = int(self.episode_state.get("current_step_index", 0))
        runtimes = self.episode_state.setdefault("step_runtime", {})
        key = str(int(step_index))
        runtime = runtimes.setdefault(key, {
            "perception_count": 0,
            "probe_dispatch_count": 0,
            "navigation_dispatch_count": 0,
            "last_perception_mode": None,
            "last_acquisition_id": None,
        })
        return runtime

    @staticmethod
    def _pose_yaw_radians(pose_payload: Mapping[str, object] | None) -> float:
        if not isinstance(pose_payload, Mapping):
            return 0.0
        orientation = pose_payload.get("orientation_xyzw")
        if not isinstance(orientation, Sequence) or len(orientation) < 4:
            return 0.0
        try:
            x, y, z, w = (float(value) for value in orientation[:4])
        except (TypeError, ValueError):
            return 0.0
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _active_step_class_names(self) -> set[str]:
        if self.episode_state is None:
            return set()
        task_ir = self.episode_state.get("task_ir", {})
        entities = {
            str(value.get("id", "")): value
            for value in task_ir.get("entities", ())
            if isinstance(value, Mapping)
        }
        steps = self.episode_state.get("execution_steps", ())
        index = int(self.episode_state.get("current_step_index", 0))
        if not isinstance(steps, Sequence) or index >= len(steps):
            return set()
        step = steps[index]
        entity_ids = [
            step.get("target_entity"),
            *step.get("anchor_entities", ()),
        ]
        classes: set[str] = set()
        for entity_id in entity_ids:
            entity = entities.get(str(entity_id), {})
            class_name = str(entity.get("class_name", "")).strip().lower()
            if class_name:
                classes.add(class_name)
            classes.update(
                str(value).strip().lower()
                for value in entity.get("aliases", ())
                if str(value).strip()
            )
        if "lamp" in classes or any(value.endswith(" lamp") for value in classes):
            classes.update({"lamp", "table lamp", "floor lamp", "wall lamp", "bedside lamp"})
        return classes

    def _scene_memory_objects(self) -> list[dict[str, object]]:
        if self.scene_memory_path is None or not self.scene_memory_path.is_file():
            return []
        try:
            payload = json.loads(self.scene_memory_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []
        objects = payload.get("objects", ()) if isinstance(payload, Mapping) else ()
        return [dict(value) for value in objects if isinstance(value, Mapping)]

    def _targeted_focus_object_ids(self) -> set[int]:
        """Return exact IDs requested by the current EvidenceNeed."""
        need = (
            self.episode_state.get("evidence_need")
            if isinstance(self.episode_state, Mapping)
            else None
        )
        if isinstance(need, Mapping):
            requested: set[int] = set()
            for value in (*need.get("target_ids", ()), *need.get("anchor_ids", ())):
                try:
                    object_id = int(value)
                except (TypeError, ValueError):
                    continue
                if object_id >= 0:
                    requested.add(object_id)
            if requested:
                return requested
        for attempt in reversed(self.navigation_attempts):
            if str(attempt.get("status", "")) not in {
                "arrived", "dispatched", "failed"
            }:
                continue
            raw_values: list[object] = [
                attempt.get("semantic_object_id"),
                attempt.get("probe_source_candidate_id"),
                *(attempt.get("probe_anchor_object_ids", ()) or ()),
            ]
            waypoint_context = attempt.get("waypoint_context")
            if isinstance(waypoint_context, Mapping):
                raw_values.extend(
                    waypoint_context.get("target_object_ids", ()) or ()
                )
                raw_values.extend(
                    waypoint_context.get("anchor_object_ids", ()) or ()
                )
            result: set[int] = set()
            for value in raw_values:
                try:
                    object_id = int(value)
                except (TypeError, ValueError):
                    continue
                if object_id >= 0:
                    result.add(object_id)
            if result:
                return result
        return set()

    def _evidence_perception_yaws(
        self,
        pose_payload: Mapping[str, object] | None,
        *,
        limit: int,
    ) -> list[float]:
        limit = max(1, min(8, int(limit)))
        position = pose_payload.get("position_xyz") if isinstance(pose_payload, Mapping) else None
        if not isinstance(position, Sequence) or len(position) < 2:
            return [float(value) for value in (0, 90, 180, 270)[:limit]]
        try:
            robot_x, robot_y = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            return [float(value) for value in (0, 90, 180, 270)[:limit]]
        robot_yaw = self._pose_yaw_radians(pose_payload)
        relevant_classes = self._active_step_class_names()
        focus_object_ids = self._targeted_focus_object_ids()
        ranked: list[tuple[int, tuple[object, ...], float]] = []
        for obj in self._scene_memory_objects():
            class_label = str(obj.get("class_label", "")).strip().lower()
            if relevant_classes and class_label not in relevant_classes and not (
                "lamp" in relevant_classes and (class_label == "lamp" or class_label.endswith(" lamp"))
            ):
                continue
            center = obj.get("center_3d")
            if not isinstance(center, Sequence) or len(center) < 2:
                continue
            try:
                dx = float(center[0]) - robot_x
                dy = float(center[1]) - robot_y
            except (TypeError, ValueError):
                continue
            if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) < 0.10:
                continue
            # Perspective extraction uses clockwise-positive image yaw, while
            # ROS/map yaw is counter-clockwise-positive.  Convert the desired
            # world bearing into that sensor-relative convention.  Using the
            # ROS delta directly points the narrow reacquisition view to the
            # mirror side of the panorama after the robot has rotated.
            relative = math.degrees(
                robot_yaw - math.atan2(dy, dx)
            ) % 360.0
            quality = (
                int(str(obj.get("semantic_status", "")) == "verified"),
                int(str(obj.get("physical_status", obj.get("status", ""))) == "confirmed"),
                int(obj.get("independent_viewpoint_count", 0) or 0),
                len(obj.get("evidence", ())),
                float(obj.get("semantic_probability", 0.0) or 0.0),
            )
            try:
                object_id = int(obj.get("object_id", -1))
            except (TypeError, ValueError):
                object_id = -1
            ranked.append((int(object_id in focus_object_ids), quality, relative))
        ranked.sort(reverse=True)
        selected: list[float] = []
        # After navigation, reacquisition must look back at the exact semantic
        # hypothesis that caused that motion.  Ranking every task-class object
        # by semantic confidence previously spent both narrow views on easier,
        # unrelated instances elsewhere in the room.
        for focused, _quality, yaw in ranked:
            if focus_object_ids and not focused:
                continue
            if any(abs(((yaw - existing + 180.0) % 360.0) - 180.0) < 25.0 for existing in selected):
                continue
            selected.append(yaw)
            if len(selected) >= limit:
                break
        seed = selected[0] if selected else 0.0
        for offset in (35.0, -35.0):
            if len(selected) >= limit:
                break
            candidate = (seed + offset) % 360.0
            if any(abs(((candidate - existing + 180.0) % 360.0) - 180.0) < 25.0 for existing in selected):
                continue
            selected.append(candidate)
        for _focused, _quality, yaw in ranked:
            if len(selected) >= limit:
                break
            if any(abs(((yaw - existing + 180.0) % 360.0) - 180.0) < 25.0 for existing in selected):
                continue
            selected.append(yaw)
        seed = selected[0] if selected else 0.0
        for offset in (90.0, -90.0, 180.0, 135.0, -135.0):
            if len(selected) >= limit:
                break
            candidate = (seed + offset) % 360.0
            if any(abs(((candidate - existing + 180.0) % 360.0) - 180.0) < 25.0 for existing in selected):
                continue
            selected.append(candidate)
        return [round(value % 360.0, 3) for value in selected[:limit]]

    def _perception_policy_for_capture(
        self,
        *,
        station_is_new: bool,
        semantic_transition_capture: bool,
        pose_payload: Mapping[str, object] | None,
    ) -> dict[str, object]:
        remaining = self._seconds_remaining()
        remaining_value = (
            float(remaining)
            if remaining is not None
            else float(self.question_time_budget_seconds)
        )
        step_index = int((self.episode_state or {}).get("current_step_index", 0))
        runtime = self._current_step_runtime(step_index)
        perception_count = int(runtime.get("perception_count", 0) or 0)
        memory_has_objects = bool(self._scene_memory_objects())
        evidence_need = (
            self.episode_state.get("evidence_need")
            if isinstance(self.episode_state, Mapping)
            else None
        )
        if not station_is_new or semantic_transition_capture:
            mode = "decision_only"
            yaws: list[float] = []
            cap = 0.0
            reason = "world_state_unchanged"
        elif perception_count == 0 and not memory_has_objects:
            mode = "initial_full"
            yaws = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
            cap = min(
                95.0,
                max(24.0, remaining_value - self.answer_reserve_seconds),
            )
            reason = "initial_task_conditioned_panorama"
        elif isinstance(evidence_need, Mapping):
            relation_requested = bool(evidence_need.get("predicate"))
            yaws = self._evidence_perception_yaws(
                pose_payload,
                limit=4 if relation_requested else 2,
            )
            mode = "evidence_targeted"
            # Every new-station EvidenceNeed can introduce recall proposals,
            # so semantic verification needs the same budget regardless of
            # whether this particular request also names a relation predicate.
            cap = min(
                40.0,
                max(8.0, remaining_value - self.answer_reserve_seconds),
            )
            reason = str(evidence_need.get("reason", "requested_world_evidence"))
        else:
            mode = "world_refresh"
            yaws = self._evidence_perception_yaws(pose_payload, limit=2)
            cap = min(
                18.0,
                max(6.0, remaining_value - self.answer_reserve_seconds),
            )
            reason = "new_physical_station"
        return {
            "schema_version": "evidence_perception_policy_v1",
            "mode": mode,
            "step_index": step_index,
            "perception_count_before": perception_count,
            "view_limit": len(yaws),
            "yaws_deg": yaws,
            "cap_seconds": float(max(0.0, cap)),
            "skip_semantic_verification": False,
            "reason": reason,
            "remaining_seconds_at_capture": remaining,
        }

    def _record_completed_perception(
        self,
        request_path: Path,
        summary: Mapping[str, object],
    ) -> None:
        perception = summary.get("stages", {}).get("perception", {})
        if not isinstance(perception, Mapping) or str(perception.get("status")) != "completed":
            return
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        policy = request.get("perception_policy", {})
        if not isinstance(policy, Mapping) or str(policy.get("mode")) == "decision_only":
            return
        try:
            step_index = int(policy.get("step_index", 0))
        except (TypeError, ValueError):
            step_index = 0
        runtime = self._current_step_runtime(step_index)
        runtime["perception_count"] = int(runtime.get("perception_count", 0) or 0) + 1
        runtime["last_perception_mode"] = str(policy.get("mode", ""))
        runtime["last_acquisition_id"] = str(request.get("acquisition_id", ""))
        runtime["last_detection_count"] = int(perception.get("detection_count", 0) or 0)
        self._persist_episode_state()

    def _update_episode_state_from_summary(
        self, summary: Mapping[str, object]
    ) -> bool:
        """Apply one chain result to the sole episode-owned runtime state."""
        if self.episode_state is None:
            raise RuntimeError("episode_state_uninitialized")
        stages = summary.get("stages", {})
        if not isinstance(stages, Mapping):
            return False
        query = stages.get("query_execution", {})
        perception = stages.get("perception", {})
        geometry = stages.get("lidar_geometry", {})
        memory = stages.get("scene_memory", {})
        resolution = stages.get("task_resolution", {})
        if not isinstance(query, Mapping):
            return False
        execution = query.get("execution", {})
        if not isinstance(execution, Mapping):
            return False
        if isinstance(resolution, Mapping) and isinstance(
            resolution.get("result"), Mapping
        ):
            resolver_result = ResolverResult.from_dict(resolution["result"])
            self.episode_state["resolver_result"] = resolver_result.to_dict()
            self.episode_state["evidence_need"] = (
                resolver_result.evidence_need.to_dict()
                if resolver_result.status is ResolverStatus.NEED_EVIDENCE
                and resolver_result.evidence_need is not None
                else None
            )
        returned_steps = execution.get("execution_steps")
        if isinstance(returned_steps, list):
            owned_steps = self.episode_state.get("execution_steps", [])
            if len(returned_steps) != len(owned_steps):
                raise RuntimeError("execution_step_count_changed")
            for owned, returned in zip(owned_steps, returned_steps):
                old_status = str(owned["status"])
                new_status = str(returned["status"])
                if new_status == "SATISFIED" and old_status != "SATISFIED":
                    new_status = "BOUND"
                if (
                    old_status == "SATISFIED"
                    and new_status != "SATISFIED"
                ):
                    raise RuntimeError("execution_step_status_regressed")
                old_binding = owned.get("bound_object_id")
                new_binding = returned.get("bound_object_id")
                old_anchor_bindings = list(
                    owned.get("bound_anchor_object_ids", ())
                )
                new_anchor_bindings = list(
                    returned.get("bound_anchor_object_ids", ())
                )
                if (
                    old_status == "SATISFIED"
                    and old_binding is not None
                    and new_binding != old_binding
                ):
                    raise RuntimeError("execution_step_binding_changed")
                if (
                    old_status == "SATISFIED"
                    and
                    any(value is not None for value in old_anchor_bindings)
                    and new_anchor_bindings != old_anchor_bindings
                ):
                    raise RuntimeError("execution_anchor_binding_changed")
                if old_status != "SATISFIED":
                    owned["status"] = new_status
                    owned["bound_object_id"] = (
                        None if new_binding is None else int(new_binding)
                    )
                    owned["bound_anchor_object_ids"] = new_anchor_bindings
        if isinstance(geometry, Mapping) and geometry.get("status") == "completed":
            self.episode_state["geometry_manifest_path"] = str(
                geometry.get("observations_path", "")
            )
        if isinstance(memory, Mapping) and isinstance(memory.get("snapshot"), Mapping):
            self.episode_state["scene_memory"] = dict(memory["snapshot"])
        if isinstance(perception, Mapping) and perception.get("status") == "completed":
            for keyframe in reversed(self.reconstruction_keyframes):
                station_id = str(keyframe.get("optical_center_group", ""))
                registry = self.episode_state.get("station_registry", {})
                if station_id in registry and not registry[station_id].get("processed"):
                    registry[station_id]["processed"] = True
                    registry[station_id]["detection_count"] = int(
                        perception.get("detection_count", 0)
                    )
                    registry[station_id]["view_count"] = int(
                        perception.get("view_count", 0)
                    )
                    break
        selection = stages.get("waypoint_selection", {})
        before_replay = int(self.episode_state.get("current_step_index", 0))
        if isinstance(selection, Mapping) and isinstance(
            selection.get("trajectory_constraints"), list
        ):
            monitor = self.episode_state["trajectory_monitor"]
            active_context = self.episode_state.get("active_waypoint_context")
            terminal_dwell_pending = bool(
                isinstance(monitor, Mapping)
                and monitor.get("route_active") is True
                and monitor.get("completed") is not True
                and isinstance(active_context, Mapping)
                and bool(
                    active_context.get(
                        "is_terminal", active_context.get("terminal", False)
                    )
                )
                and str(active_context.get("action", "")).lower()
                in {"stop_at", "stop_near"}
                and not bool(
                    active_context.get("selector_provisional", False)
                )
            )
            if not terminal_dwell_pending:
                monitor["constraint_set_id"] = str(
                    selection.get("constraint_set_id", "")
                )
                monitor["constraints"] = [
                    dict(value) for value in selection["trajectory_constraints"]
                ]
                monitor["last_pose_map"] = None
                monitor["last_pose_stamp_seconds"] = None
                monitor["route_active"] = False
                monitor["activation_stamp_seconds"] = None
                monitor["semantic_previous_pose_map"] = None
                monitor["semantic_ingress_seen"] = False
                # A newly computed selection is only a route candidate. Semantic
                # evaluation starts after output_adapter confirms publication.
                self.episode_state["active_waypoint_context"] = None
        return int(self.episode_state.get("current_step_index", 0)) > before_replay

    def _finalize_time_budget(self, reason: str) -> bool:
        """Deliver deadline exhaustion to RootFinalizer without inventing evidence."""
        if (
            not self.active_question
            or self.episode_terminalized
            or self.episode_state is None
        ):
            return False
        remaining = self._seconds_remaining()
        guard_seconds = max(30.0, float(self.answer_reserve_seconds) + 10.0)
        if remaining is None or remaining > guard_seconds:
            return False
        raw_result = self.episode_state.get("resolver_result")
        task_ir = self.episode_state.get("task_ir")
        if not isinstance(raw_result, Mapping) or not isinstance(task_ir, Mapping):
            self._publish_status(
                "time_budget_finalization_pending",
                reason=str(reason),
                detail="resolver_result_unavailable",
                time_remaining_seconds=remaining,
            )
            return False
        with self.lock:
            if self.time_budget_finalization_in_progress:
                return False
            self.time_budget_finalization_in_progress = True
        try:
            result = ResolverResult.from_dict(raw_result)
            decision = finalize_resolver_result(
                task_ir,
                result,
                time_budget_exhausted=True,
            )
            self.navigation.cancel("time_budget_exhausted")
            published = self._execute_output_decision(decision)
            self.episode_terminalized = True
            self.time_budget_finalized = True
            self.episode_state["terminal_root_decision"] = copy.deepcopy(decision)
            self.episode_state["active_waypoint"] = None
            self.episode_state["active_waypoint_context"] = None
            monitor = self.episode_state.get("trajectory_monitor", {})
            if isinstance(monitor, dict):
                monitor["route_active"] = False
            self._persist_episode_state()
            self._publish_status(
                "time_budget_finalized",
                root_action=decision.get("action"),
                answer_published=published,
                resolver_status=result.status.value,
                reason=str(reason),
                time_remaining_seconds=remaining,
            )
            return True
        finally:
            self.dispatch_in_progress = False
            with self.lock:
                self.time_budget_finalization_in_progress = False

    def _publish_heartbeat(self) -> None:
        now = time.monotonic()
        sensor_scan_age = None if self.last_sensor_scan_monotonic is None else now - self.last_sensor_scan_monotonic
        registered_scan_age = None if self.last_registered_scan_monotonic is None else now - self.last_registered_scan_monotonic
        terrain_map_age = None if self.last_terrain_map_monotonic is None else now - self.last_terrain_map_monotonic
        terrain_map_ext_age = None if self.last_terrain_map_ext_monotonic is None else now - self.last_terrain_map_ext_monotonic
        state_estimation_age = None if self.last_state_estimation_monotonic is None else now - self.last_state_estimation_monotonic
        if self.episode_state is not None and not self.episode_terminalized:
            self._finalize_time_budget("episode_deadline_guard")
        self._publish_status(
            "busy" if self.running else "ready",
            camera_ready=self.latest_image is not None,
            sensor_scan_ready=sensor_scan_age is not None and sensor_scan_age < 1.0,
            registered_scan_ready=registered_scan_age is not None and registered_scan_age < 1.0,
            terrain_map_ready=terrain_map_age is not None and terrain_map_age < 1.0,
            terrain_map_ext_ready=(
                terrain_map_ext_age is not None and terrain_map_ext_age < 1.0
            ),
            state_estimation_ready=state_estimation_age is not None and state_estimation_age < 1.0,
            sensor_scan_age_seconds=sensor_scan_age,
            registered_scan_age_seconds=registered_scan_age,
            terrain_map_age_seconds=terrain_map_age,
            terrain_map_ext_age_seconds=terrain_map_ext_age,
            state_estimation_age_seconds=state_estimation_age,
            model_switches=self.model_switches,
            pipeline_order=self.pipeline_order,
            mast3r_ready=(
                self.latest_image is not None
                and self.model_switches["mast3r"]
                and self.mast3r_checkpoint_ready
            ),
            mast3r_execution="on_demand_per_question",
            mast3r_reconstruction_mode=self.mast3r_runtime["reconstruction_mode"],
            mast3r_keyframe_station_count=len(self.reconstruction_keyframes),
            mast3r_world_alignment_ready=self.last_world_alignment_ready,
            mast3r_world_alignment_scope="last_completed_question",
            probe_count=self.probe_count,
            episode_time_remaining_seconds=self._seconds_remaining(),
            resolver_status=(
                self.episode_state.get("resolver_result", {}).get("status")
                if isinstance(self.episode_state, Mapping)
                and isinstance(self.episode_state.get("resolver_result"), Mapping)
                else None
            ),
        )

    def _on_image(self, message: Image) -> None:
        if message.encoding.lower() != "bgr8":
            self._publish_status("input_rejected", reason=f"camera_encoding:{message.encoding}")
            return
        if message.width != 1920 or message.height != 640:
            self._publish_status(
                "input_rejected",
                reason="unexpected_camera_geometry",
                width=int(message.width),
                height=int(message.height),
            )
            return
        row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = row[:, : message.width * 3].reshape(message.height, message.width, 3).copy()
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = {
                "sec": int(message.header.stamp.sec),
                "nanosec": int(message.header.stamp.nanosec),
            }
            self.last_image_monotonic = time.monotonic()
        self._wake_pending_initial_question_from_inputs()

    def _on_sensor_scan(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_sensor_scan = message
            self.last_sensor_scan_monotonic = time.monotonic()
            self.last_sensor_scan_stamp_seconds = self._stamp_seconds(message)
        self._wake_pending_initial_question_from_inputs()

    def _on_registered_scan(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_registered_scan = message
            self.last_registered_scan_monotonic = time.monotonic()
            self.last_registered_scan_stamp_seconds = self._stamp_seconds(message)
        self._wake_pending_initial_question_from_inputs()

    @staticmethod
    def _pointcloud_message_has_points(message: PointCloud2 | None) -> bool:
        fields = (
            {str(field.name) for field in message.fields}
            if message is not None else set()
        )
        return bool(
            message is not None
            and int(message.width) * int(message.height) > 0
            and len(message.data) > 0
            and {"x", "y", "z"}.issubset(fields)
        )

    @staticmethod
    def _terrain_message_has_points(message: PointCloud2 | None) -> bool:
        fields = (
            {str(field.name) for field in message.fields}
            if message is not None else set()
        )
        return bool(
            message is not None
            and int(message.width) * int(message.height) > 0
            and len(message.data) > 0
            and {"x", "y", "z", "intensity"}.issubset(fields)
        )

    def _wake_pending_initial_question_from_inputs(self) -> None:
        with self.lock:
            question = self.pending_initial_question
            inputs_available = (
                self.latest_image is not None
                and self._pointcloud_message_has_points(
                    self.latest_sensor_scan
                )
                and self._pointcloud_message_has_points(
                    self.latest_registered_scan
                )
                and self.latest_state_estimation is not None
                and (
                    self._terrain_message_has_points(self.latest_terrain_map)
                    or self._terrain_message_has_points(
                        self.latest_terrain_map_ext
                    )
                )
            )
            if (
                not question
                or not inputs_available
                or self.running
                or bool(self.active_question)
            ):
                return
            self.pending_initial_question = ""
        self._publish_status(
            "initial_acquisition_ready",
            question=question,
            wake_source="required_input_topic_callback",
        )
        pending = String()
        pending.data = question
        self._on_question(pending)

    def _on_terrain_map(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_terrain_map = message
            self.last_terrain_map_monotonic = time.monotonic()
        self._wake_pending_initial_question_from_inputs()

    def _on_terrain_map_ext(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_terrain_map_ext = message
            self.last_terrain_map_ext_monotonic = time.monotonic()
        self._wake_pending_initial_question_from_inputs()

    def _on_state_estimation(self, message: Odometry) -> None:
        with self.lock:
            self.latest_state_estimation = message
            self.last_state_estimation_monotonic = time.monotonic()
        self._wake_pending_initial_question_from_inputs()
        update = self._monitor_actual_trajectory(message)
        # Navigation and semantic progression share exactly the same accepted
        # state-estimation sample.  A frame/episode/timestamp/jump/speed
        # rejection must not still advance a local waypoint.
        if update is not None and update.accepted_sample:
            self._feed_navigation_odometry(message)

    def _feed_navigation_odometry(self, message: Odometry) -> None:
        pose = message.pose.pose
        orientation = pose.orientation
        heading = math.atan2(
            2.0 * (
                float(orientation.w) * float(orientation.z)
                + float(orientation.x) * float(orientation.y)
            ),
            1.0 - 2.0 * (
                float(orientation.y) ** 2 + float(orientation.z) ** 2
            ),
        )
        linear = getattr(
            getattr(getattr(message, "twist", None), "twist", None),
            "linear",
            None,
        )
        speed = None
        if linear is not None:
            speed = math.sqrt(
                float(linear.x) ** 2
                + float(linear.y) ** 2
                + float(linear.z) ** 2
            )
        self.navigation.update_odometry(
            float(pose.position.x),
            float(pose.position.y),
            heading,
            timestamp_seconds=self._stamp_seconds(message),
            linear_speed_mps=speed,
            frame_id=str(message.header.frame_id),
        )

    def _on_waypoint_reached(self, message: Float32) -> None:
        """Record an optional converter signal; it never advances state."""
        self.navigation.record_waypoint_reached(
            float(message.data)
        )

    def _on_converter_waypoint(self, message: PointStamped) -> None:
        """Track the converter's physical target without treating it as arrival."""
        self.navigation.observe_converter_waypoint(
            float(message.point.x),
            float(message.point.y),
            frame_id=str(message.header.frame_id),
        )

    def _monitor_actual_trajectory(self, message: Odometry) -> TrajectoryUpdate | None:
        """The sole writer of ordered semantic constraint completion."""
        if self.episode_state is None:
            return None
        pose = message.pose.pose
        q = pose.orientation
        heading = math.atan2(
            2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
            1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
        )
        actual = [float(pose.position.x), float(pose.position.y), heading]
        stamp_seconds = self._stamp_seconds(message)
        linear = getattr(getattr(message, "twist", None), "twist", None)
        linear = getattr(linear, "linear", None)
        speed_mps = None
        if linear is not None:
            try:
                speed_mps = math.sqrt(
                    float(linear.x) ** 2
                    + float(linear.y) ** 2
                    + float(linear.z) ** 2
                )
            except (AttributeError, TypeError, ValueError):
                speed_mps = None
        frame_id = str(
            getattr(getattr(message, "header", None), "frame_id", "")
        ).strip().lstrip("/")
        self.episode_state["actual_trajectory_messages_seen"] = int(
            self.episode_state.get("actual_trajectory_messages_seen", 0)
        ) + 1
        previous_stamp = self.last_trajectory_audit_stamp_seconds
        audit_sample_recorded = bool(
            not isinstance(previous_stamp, (int, float))
            or stamp_seconds - float(previous_stamp) >= 0.1
        )
        update = self._evaluate_actual_trajectory_pose(
            actual,
            stamp_seconds,
            audit_sample_recorded=audit_sample_recorded,
            linear_speed_mps=speed_mps,
            sample_frame_id=frame_id,
            sample_episode_id=self.active_episode_id,
            sample_config=self.navigation_config,
        )
        if update is not None and update.accepted_sample and audit_sample_recorded:
            self.episode_state.setdefault("actual_trajectory", []).append({
                "pose_map": actual,
                "stamp_seconds": stamp_seconds,
                "speed_mps": update.sample_speed_mps,
                "sample_gap_seconds": update.sample_gap_seconds,
                "kind": "state_estimation_sample",
            })
            self.last_trajectory_audit_stamp_seconds = stamp_seconds
            self._persist_episode_state()
        return update

    def _replay_actual_trajectory(self) -> None:
        """Do not replay pre-publication poses into a newly bound corridor."""
        if self.episode_state is None:
            return
        monitor = self.episode_state.get("trajectory_monitor", {})
        if not isinstance(monitor, Mapping) or not monitor.get("route_active"):
            return
        monitor["last_pose_map"] = None
        monitor["last_pose_stamp_seconds"] = None

    def _evaluate_actual_trajectory_pose(
        self,
        actual: list[float],
        stamp_seconds: float,
        *,
        audit_sample_recorded: bool,
        allow_terminal_completion: bool = True,
        linear_speed_mps: float | None = None,
        sample_frame_id: str | None = None,
        sample_episode_id: str | None = None,
        sample_config: Mapping[str, object] | None = None,
    ):
        """Apply the pure trajectory transition, then perform ROS side effects."""
        if self.episode_state is None:
            return
        update = apply_actual_pose(
            self.episode_state,
            actual,
            stamp_seconds,
            audit_sample_recorded=audit_sample_recorded,
            allow_terminal_completion=allow_terminal_completion,
            linear_speed_mps=linear_speed_mps,
            sample_frame_id=sample_frame_id,
            sample_episode_id=sample_episode_id,
            sample_config=sample_config,
        )
        if update.transition_sample_recorded:
            self.last_trajectory_audit_stamp_seconds = stamp_seconds
        if update.violation is not None:
            self.navigation.cancel("forbidden_path_region_entered")
            monitor = self.episode_state.get("trajectory_monitor", {})
            if isinstance(monitor, dict):
                monitor["route_active"] = False
            self._persist_episode_state()
            self._publish_status(
                "instruction_trajectory_violation",
                violation=update.violation,
            )
            return update
        if update.progressed:
            monitor = self.episode_state.get("trajectory_monitor", {})
            if isinstance(monitor, dict):
                monitor["route_active"] = False
            self._persist_episode_state()
            self._publish_status(
                "trajectory_constraint_progress",
                current_step_index=update.current_step_index,
                satisfied_orders=list(update.satisfied_orders),
            )
            if not update.became_complete:
                released_goal_id = self.navigation.release_after_constraint_satisfied(
                    actual
                )
                if released_goal_id is not None:
                    self._request_arrival_acquisition(
                        f"semantic-transition:{released_goal_id}"
                    )
        if update.became_complete:
            steps = self.episode_state.get("execution_steps", [])
            self.navigation.release_after_trajectory_completion(actual)
            monitor = self.episode_state.get("trajectory_monitor", {})
            if isinstance(monitor, dict):
                monitor["route_active"] = False
            active_context = self.episode_state.get("active_waypoint_context")
            if isinstance(active_context, Mapping):
                self.episode_state["last_completed_waypoint_context"] = dict(
                    active_context
                )
            self.episode_state["active_waypoint_context"] = None
            self.episode_state["active_waypoint"] = None
            self._persist_episode_state()
            self._publish_status(
                "instruction_trajectory_complete_pending_finalization",
                completion_authority="actual_state_estimation_trajectory",
                current_step_index=len(steps),
                execution_steps=steps,
            )
            if not self.running and not self.episode_terminalized:
                message = String()
                message.data = self.active_question
                threading.Thread(
                    target=self._on_question,
                    args=(message,),
                    kwargs={"acquisition_role": "trajectory_resolution"},
                    daemon=True,
                ).start()
        return update
    def _request_evidence_replan(
        self,
        goal_id: str,
        *,
        reason: str,
    ) -> None:
        if self.episode_terminalized or not self.active_question:
            return
        if self.running or self.pending_arrival_goal_id is not None:
            return
        if self.pending_navigation_recovery_goal_id is not None:
            return
        self.pending_navigation_recovery_goal_id = str(goal_id)
        self.navigation.cancel(str(reason))
        if self.episode_state is not None:
            self.episode_state["active_waypoint"] = None
            self.episode_state["active_waypoint_context"] = None
            monitor = self.episode_state.get("trajectory_monitor", {})
            if isinstance(monitor, dict):
                monitor["route_active"] = False
        self._publish_status(
            "evidence_replan_requested",
            goal_id=str(goal_id),
            reason=str(reason),
            replan_source="evidence_transaction_incomplete",
        )
        message = String()
        message.data = self.active_question
        threading.Thread(
            target=self._on_question,
            args=(message,),
            kwargs={
                "acquisition_role": "evidence_replan",
                "recovery_goal_id": str(goal_id),
            },
            daemon=True,
        ).start()

    def _wait_for_fresh_arrival_inputs(
        self,
        arrival_transaction: Mapping[str, object] | None,
    ) -> bool:
        if not isinstance(arrival_transaction, Mapping):
            return True
        raw_arrival_stamp = arrival_transaction.get(
            "arrival_state_estimation_stamp_seconds"
        )
        try:
            arrival_stamp = float(raw_arrival_stamp)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(arrival_stamp):
            return True
        raw_arrival_received = arrival_transaction.get(
            "arrival_received_monotonic"
        )
        try:
            arrival_received = float(raw_arrival_received)
        except (TypeError, ValueError):
            arrival_received = float("nan")
        if not math.isfinite(arrival_received):
            arrival_received = None
        remaining = self._seconds_remaining()
        timeout = min(5.0, max(0.5, (remaining or 5.0) - self.answer_reserve_seconds))
        deadline = time.monotonic() + timeout
        diagnostics: dict[str, object] = {
            "arrival_stamp_seconds": arrival_stamp,
            "arrival_received_monotonic": arrival_received,
            "timeout_seconds": timeout,
            "latest_stamps_seconds": [None, None, None],
            "latest_received_monotonic": [None, None, None],
            "stamp_after_arrival": [False, False, False],
            "received_after_arrival": [False, False, False],
        }
        while time.monotonic() < deadline:
            with self.lock:
                image_stamp = None
                if isinstance(self.latest_image_stamp, Mapping):
                    try:
                        image_stamp = float(
                            self.latest_image_stamp.get("sec", 0)
                        ) + float(
                            self.latest_image_stamp.get("nanosec", 0)
                        ) * 1e-9
                    except (TypeError, ValueError):
                        image_stamp = None
                messages = (
                    image_stamp,
                    self.latest_sensor_scan,
                    self.latest_registered_scan,
                )
                received = [
                    self.last_image_monotonic,
                    self.last_sensor_scan_monotonic,
                    self.last_registered_scan_monotonic,
                ]
                stamps = []
                for index, message in enumerate(messages):
                    if message is None:
                        stamps.append(None)
                        continue
                    if index == 0:
                        stamps.append(float(message))
                        continue
                    try:
                        stamps.append(self._stamp_seconds(message))
                    except (AttributeError, TypeError, ValueError):
                        stamps.append(None)
            stamp_after_arrival = [
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > arrival_stamp
                for value in stamps
            ]
            received_after_arrival = [
                arrival_received is None
                or (
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) > arrival_received
                )
                for value in received
            ]
            diagnostics.update({
                "latest_stamps_seconds": list(stamps),
                "latest_received_monotonic": list(received),
                "stamp_after_arrival": stamp_after_arrival,
                "received_after_arrival": received_after_arrival,
            })
            if all(stamp_after_arrival) and all(received_after_arrival):
                with self.lock:
                    self.last_fresh_arrival_diagnostics = copy.deepcopy(
                        diagnostics
                    )
                return True
            time.sleep(0.05)
        with self.lock:
            self.last_fresh_arrival_diagnostics = copy.deepcopy(diagnostics)
        return False

    def _request_arrival_acquisition(self, goal_id: str) -> None:
        if self.dispatch_in_progress:
            self.deferred_arrival_goal_id = str(goal_id)
            return
        if self.episode_terminalized:
            self._publish_status(
                "arrival_event_after_controller_checkpoint",
                goal_id=str(goal_id),
                completion_authority="actual_state_estimation_trajectory",
            )
            return
        if not self.active_question or self.running or self.pending_arrival_goal_id is not None:
            self.navigation.cancel("arrival_acquisition_unavailable")
            return
        arrival_context = dict(self.arrival_context_by_goal.get(str(goal_id), {}))
        self.pending_arrival_goal_id = str(goal_id)
        message = String()
        message.data = self.active_question
        threading.Thread(
            target=self._on_question,
            args=(message, self.pending_arrival_goal_id),
            kwargs={
                "acquisition_role": "arrival_reacquisition",
                "arrival_transaction": arrival_context,
            },
            daemon=True,
        ).start()

    def _request_navigation_recovery(self, goal_id: str) -> None:
        """Replan from the actual pose after a navigation failure.

        This is deliberately separate from arrival acquisition.  The current
        pose can provide a real new observation station, but the failed goal
        must never be marked ARRIVED or recorded as its arrival transaction.
        """
        if self.episode_terminalized or not self.active_question:
            return
        if self.running or self.pending_arrival_goal_id is not None:
            return
        if self.pending_navigation_recovery_goal_id is not None:
            return
        self.pending_navigation_recovery_goal_id = str(goal_id)
        self._publish_status(
            "navigation_recovery_requested",
            goal_id=str(goal_id),
            recovery_source="actual_state_estimation_after_navigation_failure",
        )
        message = String()
        message.data = self.active_question
        threading.Thread(
            target=self._on_question,
            args=(message,),
            kwargs={
                "acquisition_role": "navigation_recovery",
                "recovery_goal_id": str(goal_id),
            },
            daemon=True,
        ).start()

    def _execute_output_decision(
        self,
        decision: Mapping[str, object],
    ) -> bool:
        """Dispatch an authorized decision without reinterpreting semantics."""
        if (
            decision.get("schema_version") != ROOT_DECISION_SCHEMA
            or decision.get("task_id") != self.active_question
        ):
            return False
        action = str(decision.get("action", ""))
        if action == "COMMIT":
            return self.output_adapter.publish_root_decision(
                decision,
                episode_id=self.active_episode_id,
            )
        if action == "SAFE_REJECT":
            # SAFE_REJECT publishes no competition output.  Keep the episode
            # live so the parent-owned deadline guard can still commit the
            # latest legal current-state result instead of silently timing out.
            return False
        if action == "SYSTEM_FAILURE":
            return True
        if action != "PROBE":
            return False
        waypoints = decision.get("waypoint_sequence")
        if not isinstance(waypoints, list) or not waypoints:
            return False
        intent = str(decision.get("intent", ""))
        if intent not in {"ACQUIRE_EVIDENCE", "EXECUTE_ORDERED_CONSTRAINT"}:
            return False
        context = decision.get("waypoint_context", {})
        context = dict(context) if isinstance(context, Mapping) else {}
        if intent == "ACQUIRE_EVIDENCE":
            observation_intent = decision.get("observation_intent")
            if not isinstance(observation_intent, Mapping):
                return False
            context["observation_intent"] = dict(observation_intent)
            purpose = "evidence"
        else:
            execution_intent = decision.get("execution_intent")
            if not isinstance(execution_intent, Mapping):
                return False
            context["execution_intent"] = dict(execution_intent)
            purpose = "instruction"
        self.dispatch_in_progress = True
        dispatched = self.navigation.dispatch_segment(
            waypoints,
            purpose=purpose,
            waypoint_context=context,
            timeout_s=self.navigation_goal_timeout_seconds,
        )
        if self.episode_state is not None:
            monitor = self.episode_state.get("trajectory_monitor", {})
            if isinstance(monitor, dict):
                monitor["route_active"] = bool(
                    dispatched and intent == "EXECUTE_ORDERED_CONSTRAINT"
                )
                if monitor["route_active"]:
                    activation_stamp = monitor.get("last_accepted_stamp_seconds")
                    monitor["activation_stamp_seconds"] = (
                        float(activation_stamp)
                        if isinstance(activation_stamp, (int, float))
                        else 0.0
                    )
                    monitor["last_pose_map"] = None
                    monitor["last_pose_stamp_seconds"] = None
                    monitor["terminal_dwell_since_seconds"] = None
                    monitor["terminal_dwell_samples"] = 0
            if dispatched:
                self.episode_state["active_waypoint_context"] = context
                runtime = self._current_step_runtime()
                counter = (
                    "navigation_dispatch_count"
                    if intent == "EXECUTE_ORDERED_CONSTRAINT"
                    else "evidence_dispatch_count"
                )
                runtime[counter] = int(runtime.get(counter, 0) or 0) + 1
                runtime["last_dispatch_action"] = action
                runtime["last_dispatch_monotonic"] = time.monotonic()
                self._persist_episode_state()
        return dispatched

    def _validate_arrival_evidence(
        self,
        request_path: Path,
        summary: Mapping[str, object],
        arrival_goal_id: str,
    ) -> dict[str, object] | None:
        """Validate one evidence transaction without task-specific probe state."""
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(request, Mapping):
            return None
        transaction = request.get("arrival_transaction", {})
        if not isinstance(transaction, Mapping):
            return None
        transaction = dict(transaction)
        raw_intent = request.get("observation_intent")
        if not isinstance(raw_intent, Mapping):
            raw_intent = transaction.get("observation_intent")
        if not isinstance(raw_intent, Mapping):
            context = self.arrival_context_by_goal.get(str(arrival_goal_id), {})
            raw_intent = context.get("observation_intent")
        if not isinstance(raw_intent, Mapping):
            return None
        intent = self.evidence_coordinator.intent_from_dict(raw_intent)
        stages = summary.get("stages", {}) if isinstance(summary, Mapping) else {}
        scene_stage = stages.get("scene_memory", {}) if isinstance(stages, Mapping) else {}
        scene_snapshot = (
            scene_stage.get("snapshot", {})
            if isinstance(scene_stage, Mapping)
            else {}
        )
        if not isinstance(scene_snapshot, Mapping):
            scene_snapshot = {}
        last_transaction = scene_snapshot.get("last_observation_transaction", {})
        if not isinstance(last_transaction, Mapping):
            last_transaction = {}
        transaction["scene_memory_committed"] = bool(
            str(last_transaction.get("acquisition_id", ""))
            == str(transaction.get("acquisition_id", ""))
        )
        validation = self.evidence_coordinator.validate_transaction(
            intent,
            navigation_arrived=True,
            fresh_observation=bool(transaction.get("fresh_observation")),
            scene_snapshot=scene_snapshot,
            transaction=transaction,
        )
        validation["goal_id"] = str(arrival_goal_id)
        for attempt in reversed(self.navigation_attempts):
            if str(attempt.get("goal_id")) == str(arrival_goal_id):
                attempt["evidence_transaction"] = copy.deepcopy(validation)
                if validation.get("completed") is not True:
                    attempt["failure_reason"] = ",".join(
                        str(value) for value in validation.get("failure_reasons", ())
                    )
                break
        _write_json(
            request_path.parent / "evidence_transaction.json",
            validation,
        )
        self._persist_episode_state(request_path.parent)
        self._publish_status(
            "evidence_transaction_completed"
            if validation.get("completed")
            else "evidence_transaction_incomplete",
            **validation,
        )
        return validation

    @staticmethod
    def _stamp_seconds(message) -> float:
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _pointcloud_xyz(message: PointCloud2) -> np.ndarray:
        rows = [
            (float(point[0]), float(point[1]), float(point[2]))
            for point in point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        ]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 3)

    @staticmethod
    def _terrain_pointcloud(message: PointCloud2) -> tuple[np.ndarray, list[str]]:
        field_names = {str(field.name) for field in message.fields}
        if "intensity" not in field_names:
            return LiveTaskProbe._pointcloud_xyz(message), ["x", "y", "z"]
        rows = [
            (float(point[0]), float(point[1]), float(point[2]), float(point[3]))
            for point in point_cloud2.read_points(
                message,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=True,
            )
        ]
        return np.asarray(rows, dtype=np.float32).reshape(-1, 4), [
            "x", "y", "z", "intensity"
        ]

    def _record_reconstruction_keyframe(
        self,
        record: dict[str, object],
    ) -> None:
        """Record every real waypoint-arrival acquisition as a scene station."""
        self.reconstruction_keyframes.append(record)
        maximum = int(self.mast3r_runtime["max_keyframe_stations"])
        self.reconstruction_keyframes[:] = self.reconstruction_keyframes[-maximum:]

    def _on_question(
        self,
        message: String,
        arrival_goal_id: str | None = None,
        *,
        acquisition_role: str | None = None,
        recovery_goal_id: str | None = None,
        arrival_transaction: Mapping[str, object] | None = None,
    ) -> None:
        question = message.data.strip()
        continuation_acquisition = bool(
            arrival_goal_id is not None or acquisition_role
        )
        if not question:
            self._publish_status("question_rejected", reason="empty_question")
            return
        if self.running:
            self._publish_status("question_rejected", reason="chain_busy", question=question)
            return
        if continuation_acquisition and self.episode_terminalized:
            # Navigation arrival and the semantic trajectory monitor consume
            # the same odometry sample on different callbacks.  If arrival
            # queued an acquisition just before the monitor committed the
            # terminal step, that late continuation must be recorded as a
            # post-checkpoint event rather than creating a new episode
            # snapshot and overwriting the completed state.
            late_goal_id = self.pending_arrival_goal_id or arrival_goal_id
            self.pending_arrival_goal_id = None
            self._publish_status(
                "arrival_event_after_controller_checkpoint",
                goal_id=late_goal_id,
                completion_authority="actual_state_estimation_trajectory",
            )
            return
        if not continuation_acquisition and (
            self.navigation.has_active_goal
            or self.navigation.awaiting_arrival_acquisition
        ):
            self._publish_status("question_rejected", reason="navigation_in_progress", question=question)
            return
        if not continuation_acquisition:
            with self.lock:
                pending_initial_question = self.pending_initial_question
            if pending_initial_question:
                self._publish_status(
                    "question_deferred",
                    reason=(
                        "waiting_for_initial_required_inputs"
                        if question == pending_initial_question
                        else "different_question_already_pending"
                    ),
                    question=question,
                    pending_question=pending_initial_question,
                )
                return
        if not continuation_acquisition and self.active_question:
            self._publish_status(
                "question_rejected",
                reason=(
                    "duplicate_active_question"
                    if question == self.active_question
                    else "episode_already_active"
                ),
                question=question,
                active_question=self.active_question,
                active_episode_id=self.active_episode_id,
                navigation=self.navigation.snapshot(),
            )
            return
        if not continuation_acquisition and not self.active_question:
            # The evaluator's ten-minute clock starts with the first question,
            # not with the first camera frame. Keep this monotonic origin even
            # when startup briefly has no image, so the deadline guard cannot
            # publish after the host evaluator has already stopped the node.
            with self.lock:
                if self.pending_initial_question_started_monotonic is None:
                    self.pending_initial_question_started_monotonic = time.monotonic()
        if arrival_goal_id is not None and isinstance(arrival_transaction, Mapping):
            if not self._wait_for_fresh_arrival_inputs(arrival_transaction):
                self.pending_arrival_goal_id = None
                self.navigation.cancel("fresh_observation_timeout")
                for attempt in reversed(self.navigation_attempts):
                    if str(attempt.get("goal_id")) == str(arrival_goal_id):
                        attempt["status"] = "arrived_fresh_observation_timeout"
                        attempt["probe_completed"] = False
                        attempt["failure_reason"] = "fresh_observation_after_arrival_timeout"
                        break
                self._publish_status(
                    "probe_fresh_observation_rejected",
                    goal_id=str(arrival_goal_id),
                    navigation_subgoal_reached=True,
                    semantic_observation_region_satisfied=True,
                    fresh_observation=False,
                    probe_completed=False,
                    failure_reason="fresh_observation_after_arrival_timeout",
                    fresh_observation_diagnostics=copy.deepcopy(
                        self.last_fresh_arrival_diagnostics
                    ),
                )
                self._request_evidence_replan(
                    str(arrival_goal_id),
                    reason="fresh_observation_after_arrival_timeout",
                )
                return
        with self.lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            stamp = None if self.latest_image_stamp is None else dict(self.latest_image_stamp)
            sensor_scan = self.latest_sensor_scan
            registered_scan = self.latest_registered_scan
            terrain_map = self.latest_terrain_map
            terrain_map_ext = self.latest_terrain_map_ext
            state_estimation = self.latest_state_estimation
            sensor_scan_age = None if self.last_sensor_scan_monotonic is None else time.monotonic() - self.last_sensor_scan_monotonic
            registered_scan_age = None if self.last_registered_scan_monotonic is None else time.monotonic() - self.last_registered_scan_monotonic
            terrain_map_age = None if self.last_terrain_map_monotonic is None else time.monotonic() - self.last_terrain_map_monotonic
            terrain_map_ext_age = None if self.last_terrain_map_ext_monotonic is None else time.monotonic() - self.last_terrain_map_ext_monotonic
            state_estimation_age = None if self.last_state_estimation_monotonic is None else time.monotonic() - self.last_state_estimation_monotonic
        if image is None:
            self._publish_status("question_rejected", reason="no_live_camera_frame", question=question)
            return
        if not continuation_acquisition and not (
            self._pointcloud_message_has_points(sensor_scan)
            and self._pointcloud_message_has_points(registered_scan)
            and state_estimation is not None
            and (
                self._terrain_message_has_points(terrain_map)
                or self._terrain_message_has_points(terrain_map_ext)
            )
        ):
            with self.lock:
                self.pending_initial_question = question
                if self.pending_initial_question_started_monotonic is None:
                    self.pending_initial_question_started_monotonic = time.monotonic()
            self._publish_status(
                "question_deferred",
                reason="waiting_for_initial_required_inputs",
                question=question,
                wake_source="required_input_topic_callback",
            )
            return
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_dir = (self.output_root / now).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        image_path = run_dir / "camera_panorama.png"
        if not cv2.imwrite(str(image_path), image):
            self._publish_status("capture_failed", reason="could_not_write_panorama")
            return
        sensor_scan_path = run_dir / "sensor_scan.npy"
        registered_scan_path = run_dir / "registered_scan.npy"
        terrain_map_path = run_dir / "terrain_map.npy"
        terrain_map_ext_path = run_dir / "terrain_map_ext.npy"
        state_estimation_path = run_dir / "state_estimation.json"
        sensor_points = self._pointcloud_xyz(sensor_scan) if sensor_scan is not None else np.zeros((0, 3), dtype=np.float32)
        registered_points = self._pointcloud_xyz(registered_scan) if registered_scan is not None else np.zeros((0, 3), dtype=np.float32)
        terrain_points, terrain_fields = (
            self._terrain_pointcloud(terrain_map)
            if terrain_map is not None
            else (np.zeros((0, 3), dtype=np.float32), ["x", "y", "z"])
        )
        terrain_ext_points, terrain_ext_fields = (
            self._terrain_pointcloud(terrain_map_ext)
            if terrain_map_ext is not None
            else (np.zeros((0, 3), dtype=np.float32), ["x", "y", "z"])
        )
        np.save(sensor_scan_path, sensor_points)
        np.save(registered_scan_path, registered_points)
        np.save(terrain_map_path, terrain_points)
        np.save(terrain_map_ext_path, terrain_ext_points)
        image_stamp_s = None if stamp is None else float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
        sensor_stamp_s = self._stamp_seconds(sensor_scan) if sensor_scan is not None else None
        registered_stamp_s = self._stamp_seconds(registered_scan) if registered_scan is not None else None
        terrain_stamp_s = self._stamp_seconds(terrain_map) if terrain_map is not None else None
        terrain_ext_stamp_s = self._stamp_seconds(terrain_map_ext) if terrain_map_ext is not None else None
        state_stamp_s = self._stamp_seconds(state_estimation) if state_estimation is not None else None
        captured_arrival_transaction = copy.deepcopy(
            dict(arrival_transaction) if isinstance(arrival_transaction, Mapping) else {}
        )
        if captured_arrival_transaction:
            arrival_stamp = captured_arrival_transaction.get(
                "arrival_state_estimation_stamp_seconds"
            )
            try:
                arrival_stamp_value = float(arrival_stamp)
            except (TypeError, ValueError):
                arrival_stamp_value = float("nan")
            fresh_observation = bool(
                math.isfinite(arrival_stamp_value)
                and image_stamp_s is not None
                and sensor_stamp_s is not None
                and registered_stamp_s is not None
                and image_stamp_s > arrival_stamp_value
                and sensor_stamp_s > arrival_stamp_value
                and registered_stamp_s > arrival_stamp_value
            )
            captured_arrival_transaction.update({
                "acquisition_id": run_dir.name,
                "observation_transaction_id": f"{run_dir.name}:fresh-observation",
                "image_stamp_seconds": image_stamp_s,
                "sensor_scan_stamp_seconds": sensor_stamp_s,
                "registered_scan_stamp_seconds": registered_stamp_s,
                "state_estimation_stamp_seconds": state_stamp_s,
                "fresh_observation": fresh_observation,
                "scene_memory_committed": False,
                "fresh_observation_diagnostics": copy.deepcopy(
                    self.last_fresh_arrival_diagnostics
                ),
            })
        pose_payload = None
        if state_estimation is not None:
            pose = state_estimation.pose.pose
            pose_payload = {
                "stamp_seconds": state_stamp_s,
                "frame_id": state_estimation.header.frame_id.lstrip("/"),
                "child_frame_id": state_estimation.child_frame_id.lstrip("/"),
                "position_xyz": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                "orientation_xyzw": [float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w)],
            }
        _write_json(state_estimation_path, pose_payload)
        if pose_payload is not None:
            episode_id = (
                self.active_episode_id
                if continuation_acquisition
                else run_dir.name
            )
            if not continuation_acquisition:
                self.active_question = question
                self.active_episode_id = episode_id
                self.episode_started_monotonic = (
                    self.pending_initial_question_started_monotonic
                    if self.pending_initial_question_started_monotonic is not None
                    else time.monotonic()
                )
                self.episode_deadline_unix = None  # Now tracked via monotonic + budget
                self.pending_initial_question_started_monotonic = None
                self.pending_initial_question_deadline_unix = None
                self.scene_memory_path = run_dir / "episode_scene_memory.json"
                self.reconstruction_keyframes.clear()
                self.chain_durations.clear()
                self.navigation_durations.clear()
                self.navigation_started.clear()
                self.navigation_attempts.clear()
                self.last_trajectory_audit_stamp_seconds = None
                self.probe_count = 0
                self.episode_terminalized = False
                self.time_budget_finalized = False
                self.time_budget_finalization_in_progress = False
                task_ir = compile_task(question)
                execution_steps = normalize_execution_steps(task_ir)
                self.episode_state = {
                    "episode_id": self.active_episode_id,
                    "task_ir": task_ir,
                    "execution_steps": execution_steps,
                    "current_step_index": 0,
                    "step_runtime": {
                        str(int(step.get("step_index", index))): {
                            "perception_count": 0,
                            "evidence_dispatch_count": 0,
                            "navigation_dispatch_count": 0,
                            "last_perception_mode": None,
                            "last_acquisition_id": None,
                        }
                        for index, step in enumerate(execution_steps)
                    },
                    "active_waypoint": None,
                    "active_waypoint_context": None,
                    "last_completed_waypoint_context": None,
                    "resolver_result": None,
                    "evidence_need": None,
                    "ack_diagnostics": {
                        "enabled": self.waypoint_ack_diagnostics_enabled,
                        "count": 0,
                        "last": None,
                    },
                    "station_registry": {},
                    "scene_memory": None,
                    "geometry_manifest_path": None,
                    "actual_trajectory": [],
                    "actual_trajectory_messages_seen": 0,
                    "actual_trajectory_audit_rate_hz": 10.0,
                    "trajectory_monitor": {
                        "completion_authority": "actual_state_estimation_trajectory",
                        "constraint_set_id": None,
                        "constraints": [],
                        "satisfied_orders": [],
                        "forbidden_violation": None,
                        "completed": False,
                        "last_pose_map": None,
                        "last_pose_stamp_seconds": None,
                        "episode_id": self.active_episode_id,
                        "route_active": False,
                        "activation_stamp_seconds": None,
                        "last_sample_stamp_seconds": None,
                        "last_accepted_stamp_seconds": None,
                        "last_accepted_pose_map": None,
                        "last_accepted_speed_mps": None,
                        "last_accepted_gap_seconds": None,
                        "last_sample_rejection_reason": None,
                        "sample_diagnostics": {
                            "accepted_count": 0,
                            "rejected_count": 0,
                            "last_rejection": None,
                        },
                        "terminal_dwell_since_seconds": None,
                        "terminal_dwell_samples": 0,
                    },
                    "task_ir_compile_count": 1,
                }
                self.output_adapter.begin_episode(self.active_episode_id)
                self.navigation.begin_episode(self.active_episode_id)
                # The capture already contains the official current pose. Seed
                # only NavigationExecutor's transport baseline so a fast child
                # decision can dispatch from the real current pose. The
                # trajectory monitor still starts its semantic baseline only
                # after the first published corridor.
                if state_estimation is not None:
                    self._feed_navigation_odometry(state_estimation)
            if self.scene_memory_path is None:
                self._publish_status(
                    "capture_failed", reason="episode_scene_memory_uninitialized"
                )
                return
            keyframe_record = {
                "episode_id": episode_id,
                "keyframe_id": run_dir.name,
                "panorama_path": str(image_path),
                "state_estimation_path": str(state_estimation_path),
                "sensor_scan_path": str(sensor_scan_path),
                "registered_scan_path": str(registered_scan_path),
                "state_estimation": pose_payload,
                "optical_center_group": f"station:{run_dir.name}",
            }
            semantic_transition_capture = str(
                arrival_goal_id or ""
            ).startswith((
                "semantic-transition:",
                "actual-trajectory-progress:",
            ))
            if semantic_transition_capture:
                if self.reconstruction_keyframes:
                    keyframe_record["optical_center_group"] = str(
                        self.reconstruction_keyframes[-1][
                            "optical_center_group"
                        ]
                    )
            else:
                # Stable physical station: reuse an existing station when
                # the optical center is within the 0.3m independent-viewpoint
                # separation instead of minting station:<run_dir> for every
                # acquisition.  Repeated scans of one pose must not count as
                # independent views in SceneMemory association.
                if self.episode_state is not None:
                    keyframe_record["optical_center_group"] = (
                        self._reuse_station_id(
                            self._optical_center_xy(pose_payload),
                            self.episode_state["station_registry"],
                            default_station_id=str(
                                keyframe_record["optical_center_group"]
                            ),
                        )
                    )
                self._record_reconstruction_keyframe(keyframe_record)
            if self.episode_state is None:
                raise RuntimeError("episode_state_uninitialized")
            registry = self.episode_state["station_registry"]
            station_id = str(keyframe_record["optical_center_group"])
            station_is_new = station_id not in registry
            if station_is_new:
                registry[station_id] = {
                    "station_id": station_id,
                    "keyframe_id": run_dir.name,
                    "processed": False,
                    "position_xyz": (
                        list(pose_payload["position_xyz"])
                        if isinstance(pose_payload, dict)
                        and isinstance(
                            pose_payload.get("position_xyz"),
                            (list, tuple),
                        )
                        else None
                    ),
                }
            if semantic_transition_capture:
                # The robot has not moved.  A newly satisfied trajectory
                # constraint changes query/trajectory progress, not visual evidence.
                # Replan from persistent SceneMemory now; if the next
                # referent is still unresolved, its instruction-consistent
                # trajectory will acquire a genuinely new station.  Re-running the
                # complete perception stack on the same optical center both
                # wastes the shared ten-minute budget and cannot add an
                # independent observation.
                station_is_new = False

        perception_policy = self._perception_policy_for_capture(
            station_is_new=station_is_new,
            semantic_transition_capture=semantic_transition_capture,
            pose_payload=pose_payload,
        )
        competition_geometry = {
            "sensor_scan_path": str(sensor_scan_path),
            "sensor_scan_frame": None if sensor_scan is None else sensor_scan.header.frame_id.lstrip("/"),
            "sensor_scan_stamp_seconds": sensor_stamp_s,
            "sensor_scan_age_seconds": sensor_scan_age,
            "sensor_scan_point_count": int(len(sensor_points)),
            "registered_scan_path": str(registered_scan_path),
            "registered_scan_frame": None if registered_scan is None else registered_scan.header.frame_id.lstrip("/"),
            "registered_scan_stamp_seconds": registered_stamp_s,
            "registered_scan_age_seconds": registered_scan_age,
            "registered_scan_point_count": int(len(registered_points)),
            "terrain_map_path": str(terrain_map_path),
            "terrain_map_frame": None if terrain_map is None else terrain_map.header.frame_id.lstrip("/"),
            "terrain_map_stamp_seconds": terrain_stamp_s,
            "terrain_map_age_seconds": terrain_map_age,
            "terrain_map_point_count": int(len(terrain_points)),
            "terrain_map_fields": terrain_fields,
            "terrain_map_ext_path": str(terrain_map_ext_path),
            "terrain_map_ext_frame": None if terrain_map_ext is None else terrain_map_ext.header.frame_id.lstrip("/"),
            "terrain_map_ext_stamp_seconds": terrain_ext_stamp_s,
            "terrain_map_ext_age_seconds": terrain_map_ext_age,
            "terrain_map_ext_point_count": int(len(terrain_ext_points)),
            "terrain_map_ext_fields": terrain_ext_fields,
            "state_estimation_path": str(state_estimation_path),
            "state_estimation_frame": None if state_estimation is None else state_estimation.header.frame_id.lstrip("/"),
            "state_estimation_child_frame": None if state_estimation is None else state_estimation.child_frame_id.lstrip("/"),
            "state_estimation_age_seconds": state_estimation_age,
            "camera_sensor_scan_offset_seconds": None if image_stamp_s is None or sensor_stamp_s is None else abs(image_stamp_s - sensor_stamp_s),
            "allowed_competition_inputs": [
                "/camera/image",
                "/sensor_scan",
                "/registered_scan",
                "/terrain_map",
                "/terrain_map_ext",
                "/state_estimation",
                "/challenge_question",
            ],
        }
        if self.episode_state is None:
            raise RuntimeError("episode_state_uninitialized")
        execution_steps = self.episode_state["execution_steps"]
        entity_bindings = {
            str(step["target_entity"]): int(step["bound_object_id"])
            for step in execution_steps
            if (
                str(step.get("status")) == "SATISFIED"
                and step.get("bound_object_id") is not None
            )
        }
        # A new station observes the current step and every unresolved future
        # referent that can still change an ordered instruction.  This is still
        # task-conditioned; it is not a fixed room-wide class vocabulary.
        request_execution_steps = [dict(step) for step in execution_steps]
        if station_is_new and str(perception_policy.get("mode")) != "decision_only":
            task_type = str(self.episode_state["task_ir"].get("task_type", ""))
            if task_type == "instruction_following":
                active_index = int(self.episode_state["current_step_index"])
                scene_observation_entity_ids = list(dict.fromkeys(
                    str(entity_id)
                    for step in execution_steps[active_index:]
                    for entity_id in (
                        step.get("target_entity"),
                        *step.get("anchor_entities", ()),
                    )
                    if str(entity_id).strip()
                ))
            else:
                scene_observation_entity_ids = [
                    str(value["id"])
                    for value in self.episode_state["task_ir"].get("entities", ())
                ]
        else:
            scene_observation_entity_ids = []
        request = {
            "schema_version": "1.0",
            "question": question,
            "image_path": str(image_path),
            "image_geometry": {
                "projection": "equirectangular",
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "encoding": "bgr8",
                "ros_stamp": stamp,
            },
            "competition_geometry": competition_geometry,
            "output_dir": str(run_dir),
            "episode_id": self.active_episode_id,
            "acquisition_id": run_dir.name,
            "episode_remaining_seconds": self._seconds_remaining(),
            # Protect only root finalization and ROS dispatch. Perception owns its
            # own cap through perception_policy and must not be blocked by the old
            # fixed 75-second reserve.
            "mandatory_reserve_seconds": max(
                6.0, min(12.0, float(self.answer_reserve_seconds))
            ),
            "perception_policy": perception_policy,
            "navigation_config": dict(self.navigation_config),
            "scene_memory_path": str(self.scene_memory_path),
            "arrival_goal_id": arrival_goal_id,
            "acquisition_role": acquisition_role,
            "recovery_goal_id": recovery_goal_id,
            "observation_intent": copy.deepcopy(
                captured_arrival_transaction.get("observation_intent")
                if captured_arrival_transaction
                else None
            ),
            "arrival_transaction": captured_arrival_transaction,
            "reconstruction_keyframes": list(self.reconstruction_keyframes),
            "navigation_history": [
                dict(item) for item in self.navigation_attempts
            ],
            "semantic_entity_bindings": entity_bindings,
            "evidence_need": copy.deepcopy(
                self.episode_state.get("evidence_need")
            ),
            "completed_semantic_orders": [
                int(step["step_index"])
                for step in execution_steps
                if str(step.get("status")) == "SATISFIED"
            ],
            "compiled_task_ir": self.episode_state["task_ir"],
            "task_ir_compile_count": int(
                self.episode_state["task_ir_compile_count"]
            ),
            "execution_steps": request_execution_steps,
            "current_step_index": int(
                self.episode_state["current_step_index"]
            ),
            "trajectory_monitor": copy.deepcopy(
                self.episode_state.get("trajectory_monitor", {})
            ),
            "active_waypoint_context": copy.deepcopy(
                self.episode_state.get("active_waypoint_context")
            ),
            "last_completed_waypoint_context": copy.deepcopy(
                self.episode_state.get("last_completed_waypoint_context")
            ),
            "scene_observation_entity_ids": scene_observation_entity_ids,
            "station_id": station_id,
            "station_is_new": station_is_new,
            "geometry_manifest_path": self.episode_state.get(
                "geometry_manifest_path"
            ),
            "input_contract": {
                "question_source": "/challenge_question",
                "sensor_sources": [
                    "/camera/image",
                    "/sensor_scan",
                    "/registered_scan",
                    "/terrain_map",
                    "/terrain_map_ext",
                    "/state_estimation",
                ],
                "episode_only": True,
                "historical_run_discovery": False,
            },
        }
        request_path = run_dir / "request.json"
        _write_json(request_path, request)
        self._persist_episode_state(run_dir)
        self.running = True
        self._publish_status(
            "captured",
            run_dir=str(run_dir),
            question=question,
            episode_id=self.active_episode_id,
            arrival_goal_id=arrival_goal_id,
            acquisition_role=(
                str(acquisition_role)
                if acquisition_role
                else "arrival_reacquisition"
                if arrival_goal_id is not None
                else "initial_question"
            ),
            perception_policy=perception_policy,
        )
        threading.Thread(
            target=self._run_chain,
            args=(request_path, run_dir, arrival_goal_id, acquisition_role, recovery_goal_id),
            daemon=True,
        ).start()

    def _run_chain(
        self,
        request_path: Path,
        run_dir: Path,
        arrival_goal_id: str | None = None,
        acquisition_role: str | None = None,
        recovery_goal_id: str | None = None,
    ) -> None:
        published = False
        trajectory_progressed = False
        probe_validation: dict[str, object] | None = None
        needs_evidence_replan = False
        chain_started = time.monotonic()
        chain_duration_recorded = False
        try:
            instruction_episode = bool(
                self.episode_state is not None
                and self.episode_state.get("task_ir", {}).get("task_type")
                == "instruction_following"
            )
            remaining = self._seconds_remaining()
            protected_reserve = max(
                6.0, min(12.0, float(self.answer_reserve_seconds))
            )
            chain_timeout = max(
                1.0,
                min(
                    525.0,
                    (remaining - protected_reserve)
                    if remaining is not None
                    else 525.0,
                ),
            )
            completed = subprocess.run(
                [
                    "python3",
                    str(self.ai_root / "tools" / "live_model_chain.py"),
                    "--request",
                    str(request_path),
                ],
                cwd=self.ai_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=chain_timeout,
                check=False,
            )
            self.chain_durations.append(
                max(0.0, time.monotonic() - chain_started)
            )
            chain_duration_recorded = True
            (run_dir / "chain.log").write_text(completed.stdout, encoding="utf-8")
            summary_path = run_dir / "summary.json"
            summary = {}
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self._record_completed_perception(request_path, summary)
                self.last_world_alignment_ready = bool(
                    summary.get("stages", {})
                    .get("lidar_geometry", {})
                    .get("status") == "completed"
                )
            semantic_path = run_dir / "04_verified_detections.json"
            with self.lock:
                for keyframe in self.reconstruction_keyframes:
                    if str(keyframe.get("keyframe_id")) == run_dir.name:
                        if summary_path.is_file():
                            keyframe["summary_path"] = str(summary_path)
                        if semantic_path.is_file():
                            keyframe["semantic_observations_path"] = str(semantic_path)
                        break
            decision = (
                summary.get("stages", {})
                .get("root_finalization", {})
                .get("decision", {})
            )
            decision = dict(decision) if isinstance(decision, Mapping) else {}
            _write_json(run_dir / "runtime_dispatch_decision.json", decision)
            if summary:
                self._update_episode_state_from_summary(summary)
                self._persist_episode_state(run_dir)
            evidence_validation = None
            if arrival_goal_id is not None:
                evidence_validation = self._validate_arrival_evidence(
                    request_path,
                    summary,
                    str(arrival_goal_id),
                )
            self.running = False
            decision_completed = bool(
                decision.get("schema_version") == ROOT_DECISION_SCHEMA
                and decision.get("task_id") == self.active_question
                and decision.get("action") in {
                    "COMMIT",
                    "PROBE",
                    "SAFE_REJECT",
                    "SYSTEM_FAILURE",
                }
            )
            if arrival_goal_id is not None:
                semantic_transition = str(arrival_goal_id).startswith(
                    "semantic-transition:"
                )
                if not semantic_transition:
                    try:
                        self.navigation.record_arrival_acquisition(
                            run_dir.name,
                            evaluation_completed=decision_completed,
                        )
                    except RuntimeError:
                        self._publish_status(
                            "arrival_lifecycle_mismatch",
                            goal_id=str(arrival_goal_id),
                        )
                self.pending_arrival_goal_id = None
            if decision_completed and not self.episode_terminalized:
                published = self._execute_output_decision(decision)
                if published and decision.get("action") in {
                    "COMMIT",
                    "SAFE_REJECT",
                    "SYSTEM_FAILURE",
                }:
                    self.episode_terminalized = True
                    if self.episode_state is not None:
                        self.episode_state["terminal_root_decision"] = copy.deepcopy(
                            decision
                        )
                        self._persist_episode_state(run_dir)
            if acquisition_role in {"navigation_recovery", "evidence_replan"}:
                self.pending_navigation_recovery_goal_id = None
            self._publish_status(
                "chain_finished" if decision_completed else "chain_failed",
                returncode=int(completed.returncode),
                root_decision_completed=decision_completed,
                root_action=decision.get("action"),
                output_dispatched=published,
                answer_published=bool(
                    published
                    and decision.get("action") == "COMMIT"
                    and decision.get("answer_authorized") is True
                ),
                evidence_transaction=evidence_validation,
                run_dir=str(run_dir),
                gates=summary.get("gates", {}),
                navigation=self.navigation.snapshot(),
            )
            if (
                published
                and decision.get("action") == "COMMIT"
                and decision.get("task_type") == "instruction_following"
                and decision.get("episode_complete") is True
            ):
                self._publish_status(
                    "arrival_revalidated",
                    route_complete=True,
                    root_action="COMMIT",
                    arrival_purpose="instruction",
                    completion_authority=(
                        "actual_state_estimation_trajectory"
                    ),
                    run_dir=str(run_dir),
                )
            self.dispatch_in_progress = False
            deferred_arrival_goal_id = self.deferred_arrival_goal_id
            if deferred_arrival_goal_id is not None:
                self.deferred_arrival_goal_id = None
                self._request_arrival_acquisition(deferred_arrival_goal_id)
        except subprocess.TimeoutExpired as exc:
            if not chain_duration_recorded:
                self.chain_durations.append(
                    max(0.0, time.monotonic() - chain_started)
                )
            self._publish_status(
                "chain_budget_exhausted",
                reason="episode_deadline_reserve",
                timeout_seconds=float(exc.timeout),
                time_remaining_seconds=self._seconds_remaining(),
                run_dir=str(run_dir),
            )
            self._finalize_time_budget("chain_budget_exhausted")
        except Exception as exc:
            if not chain_duration_recorded:
                self.chain_durations.append(
                    max(0.0, time.monotonic() - chain_started)
                )
            self._publish_status(
                "chain_failed",
                reason=type(exc).__name__,
                detail=str(exc)[:300],
                run_dir=str(run_dir),
            )
        finally:
            self.dispatch_in_progress = False
            self.running = False


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--ai-root", type=Path, default=default_root)
    parser.add_argument("--output-root", type=Path, default=default_root / "runs" / "live_robot")
    args = parser.parse_args()
    rclpy.init()
    node = LiveTaskProbe(args.ai_root.resolve(), args.output_root.resolve())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
