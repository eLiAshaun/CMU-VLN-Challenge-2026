#!/usr/bin/env python3
"""Capture live competition questions and run the model chain fail-closed.

The simulator camera is a 1920x640 equirectangular panorama.  This node keeps
that image untouched.  It never presents the panorama to MASt3R as a pinhole
camera and never publishes a competition answer from an incomplete chain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import math

import cv2
import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.ros.output_adapter import RosOutputAdapter


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
        config = json.loads(
            (self.ai_root / "configs" / "model_assets.json").read_text(encoding="utf-8")
        )
        runtime = config["runtime"]
        self.mast3r_runtime = dict(runtime["mast3r"])
        self.pipeline_order = list(runtime["pipeline"]["stage_order"])
        self.model_switches = {
            name: bool(runtime[name]["enabled"])
            for name in ("mast3r", "yolo_world", "groundingdino", "sam2", "qwen3vl")
        }
        self.mast3r_checkpoint_ready = (
            self.ai_root / config["models"]["mast3r"]["checkpoint"]
        ).is_file()
        self.latest_image: np.ndarray | None = None
        self.latest_image_stamp: dict[str, int] | None = None
        self.latest_sensor_scan: PointCloud2 | None = None
        self.latest_registered_scan: PointCloud2 | None = None
        self.latest_terrain_map: PointCloud2 | None = None
        self.latest_state_estimation: Odometry | None = None
        self.last_sensor_scan_monotonic: float | None = None
        self.last_registered_scan_monotonic: float | None = None
        self.last_state_estimation_monotonic: float | None = None
        self.last_world_alignment_ready = False
        completed_summaries = sorted(self.output_root.glob("*/summary.json"))
        if completed_summaries:
            try:
                previous = json.loads(completed_summaries[-1].read_text(encoding="utf-8"))
                self.last_world_alignment_ready = bool(
                    previous.get("gates", {}).get("mast3r_world_alignment", False)
                )
            except (OSError, ValueError):
                pass
        self.running = False
        self.lock = threading.Lock()
        self.active_question = ""
        self.active_episode_id = ""
        self.pending_arrival_goal_id: str | None = None
        self.session_id = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")
        self.world_model_path = self.output_root / f"{self.session_id}_world_model.json"
        self.reconstruction_keyframes: list[dict[str, object]] = []
        self.status_publisher = self.create_publisher(String, "/mast3r_chain/status", 10)
        self.output_adapter = RosOutputAdapter(
            self,
            acquisition_requester=self._request_arrival_acquisition,
        )
        self.create_subscription(Image, "/camera/image", self._on_image, 10)
        self.create_subscription(PointCloud2, "/sensor_scan", self._on_sensor_scan, 10)
        self.create_subscription(PointCloud2, "/registered_scan", self._on_registered_scan, 10)
        self.create_subscription(PointCloud2, "/terrain_map", self._on_terrain_map, 10)
        self.create_subscription(Odometry, "/state_estimation", self._on_state_estimation, 20)
        self.create_subscription(String, "/challenge_question", self._on_question, 10)
        self.create_timer(5.0, self._publish_heartbeat)
        self._publish_status("ready", detail="waiting_for_question")

    def _publish_status(self, state: str, **fields: object) -> None:
        payload = {
            "state": state,
            "node": self.get_name(),
            "time_unix": time.time(),
            **fields,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_publisher.publish(message)
        self.get_logger().info(message.data)

    def _publish_heartbeat(self) -> None:
        now = time.monotonic()
        sensor_scan_age = None if self.last_sensor_scan_monotonic is None else now - self.last_sensor_scan_monotonic
        registered_scan_age = None if self.last_registered_scan_monotonic is None else now - self.last_registered_scan_monotonic
        state_estimation_age = None if self.last_state_estimation_monotonic is None else now - self.last_state_estimation_monotonic
        self._publish_status(
            "busy" if self.running else "ready",
            camera_ready=self.latest_image is not None,
            sensor_scan_ready=sensor_scan_age is not None and sensor_scan_age < 1.0,
            registered_scan_ready=registered_scan_age is not None and registered_scan_age < 1.0,
            state_estimation_ready=state_estimation_age is not None and state_estimation_age < 1.0,
            sensor_scan_age_seconds=sensor_scan_age,
            registered_scan_age_seconds=registered_scan_age,
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

    def _on_sensor_scan(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_sensor_scan = message
            self.last_sensor_scan_monotonic = time.monotonic()

    def _on_registered_scan(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_registered_scan = message
            self.last_registered_scan_monotonic = time.monotonic()

    def _on_terrain_map(self, message: PointCloud2) -> None:
        with self.lock:
            self.latest_terrain_map = message

    def _on_state_estimation(self, message: Odometry) -> None:
        with self.lock:
            self.latest_state_estimation = message
            self.last_state_estimation_monotonic = time.monotonic()
        self.output_adapter.update_odometry(message)

    def _request_arrival_acquisition(self, goal_id: str) -> None:
        if not self.active_question or self.running or self.pending_arrival_goal_id is not None:
            self.output_adapter.navigation.cancel("arrival_acquisition_unavailable")
            return
        self.pending_arrival_goal_id = str(goal_id)
        message = String()
        message.data = self.active_question
        threading.Thread(
            target=self._on_question,
            args=(message, self.pending_arrival_goal_id),
            daemon=True,
        ).start()

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
    def _quaternion_angle_deg(first: list[float], second: list[float]) -> float:
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        return math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))

    def _record_reconstruction_keyframe(self, record: dict[str, object]) -> None:
        state = dict(record["state_estimation"])
        if self.reconstruction_keyframes:
            previous = dict(self.reconstruction_keyframes[-1]["state_estimation"])
            translation = float(np.linalg.norm(
                np.asarray(state["position_xyz"], dtype=np.float64)
                - np.asarray(previous["position_xyz"], dtype=np.float64)
            ))
            rotation = self._quaternion_angle_deg(
                list(state["orientation_xyzw"]), list(previous["orientation_xyzw"])
            )
            if (
                translation < float(self.mast3r_runtime["keyframe_translation_m"])
                and rotation < float(self.mast3r_runtime["keyframe_rotation_deg"])
            ):
                self.reconstruction_keyframes[-1] = record
                return
        self.reconstruction_keyframes.append(record)
        maximum = int(self.mast3r_runtime["max_keyframe_stations"])
        self.reconstruction_keyframes[:] = self.reconstruction_keyframes[-maximum:]

    def _on_question(self, message: String, arrival_goal_id: str | None = None) -> None:
        question = message.data.strip()
        if not question:
            self._publish_status("question_rejected", reason="empty_question")
            return
        if self.running:
            self._publish_status("question_rejected", reason="chain_busy", question=question)
            return
        if arrival_goal_id is None and (
            self.output_adapter.navigation.has_active_goal
            or self.output_adapter.navigation.awaiting_arrival_acquisition
        ):
            self._publish_status("question_rejected", reason="navigation_in_progress", question=question)
            return
        with self.lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            stamp = None if self.latest_image_stamp is None else dict(self.latest_image_stamp)
            sensor_scan = self.latest_sensor_scan
            registered_scan = self.latest_registered_scan
            terrain_map = self.latest_terrain_map
            state_estimation = self.latest_state_estimation
            sensor_scan_age = None if self.last_sensor_scan_monotonic is None else time.monotonic() - self.last_sensor_scan_monotonic
            registered_scan_age = None if self.last_registered_scan_monotonic is None else time.monotonic() - self.last_registered_scan_monotonic
            state_estimation_age = None if self.last_state_estimation_monotonic is None else time.monotonic() - self.last_state_estimation_monotonic
        if image is None:
            self._publish_status("question_rejected", reason="no_live_camera_frame", question=question)
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
        state_estimation_path = run_dir / "state_estimation.json"
        sensor_points = self._pointcloud_xyz(sensor_scan) if sensor_scan is not None else np.zeros((0, 3), dtype=np.float32)
        registered_points = self._pointcloud_xyz(registered_scan) if registered_scan is not None else np.zeros((0, 3), dtype=np.float32)
        terrain_points = self._pointcloud_xyz(terrain_map) if terrain_map is not None else np.zeros((0, 3), dtype=np.float32)
        np.save(sensor_scan_path, sensor_points)
        np.save(registered_scan_path, registered_points)
        np.save(terrain_map_path, terrain_points)
        image_stamp_s = None if stamp is None else float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
        sensor_stamp_s = self._stamp_seconds(sensor_scan) if sensor_scan is not None else None
        registered_stamp_s = self._stamp_seconds(registered_scan) if registered_scan is not None else None
        state_stamp_s = self._stamp_seconds(state_estimation) if state_estimation is not None else None
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
            self._record_reconstruction_keyframe({
                "keyframe_id": run_dir.name,
                "panorama_path": str(image_path),
                "state_estimation_path": str(state_estimation_path),
                "sensor_scan_path": str(sensor_scan_path),
                "registered_scan_path": str(registered_scan_path),
                "state_estimation": pose_payload,
                "optical_center_group": f"station:{run_dir.name}",
            })
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
            "terrain_map_point_count": int(len(terrain_points)),
            "state_estimation_path": str(state_estimation_path),
            "state_estimation_frame": None if state_estimation is None else state_estimation.header.frame_id.lstrip("/"),
            "state_estimation_child_frame": None if state_estimation is None else state_estimation.child_frame_id.lstrip("/"),
            "state_estimation_age_seconds": state_estimation_age,
            "camera_sensor_scan_offset_seconds": None if image_stamp_s is None or sensor_stamp_s is None else abs(image_stamp_s - sensor_stamp_s),
            "allowed_competition_inputs": ["/camera/image", "/sensor_scan", "/registered_scan", "/state_estimation"],
        }
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
            "episode_id": self.active_episode_id if arrival_goal_id is not None else run_dir.name,
            "world_model_path": str(self.world_model_path),
            "arrival_goal_id": arrival_goal_id,
            "reconstruction_keyframes": list(self.reconstruction_keyframes),
        }
        request_path = run_dir / "request.json"
        _write_json(request_path, request)
        self.running = True
        if arrival_goal_id is None:
            self.active_question = question
            self.active_episode_id = run_dir.name
            request["episode_id"] = self.active_episode_id
            _write_json(request_path, request)
            self.output_adapter.begin_episode(self.active_episode_id)
        self._publish_status("captured", run_dir=str(run_dir), question=question)
        threading.Thread(
            target=self._run_chain,
            args=(request_path, run_dir, arrival_goal_id),
            daemon=True,
        ).start()

    def _run_chain(self, request_path: Path, run_dir: Path, arrival_goal_id: str | None = None) -> None:
        published = False
        try:
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
                timeout=900,
                check=False,
            )
            (run_dir / "chain.log").write_text(completed.stdout, encoding="utf-8")
            summary_path = run_dir / "summary.json"
            summary = {}
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.last_world_alignment_ready = bool(
                    summary.get("gates", {}).get("mast3r_world_alignment", False)
                )
            semantic_path = run_dir / "07_qwen3vl_verification" / "verified_observations.json"
            if semantic_path.is_file():
                with self.lock:
                    for keyframe in self.reconstruction_keyframes:
                        if str(keyframe.get("keyframe_id")) == run_dir.name:
                            keyframe["semantic_observations_path"] = str(semantic_path)
                            break
            decision = (
                summary.get("stages", {})
                .get("root_finalization", {})
                .get("decision", {})
            )
            if arrival_goal_id is not None:
                arrival_purpose = self.output_adapter.navigation.current_purpose
                revalidated = (
                    completed.returncode == 0
                    and decision.get("action") == "COMMIT"
                    and decision.get("task_id") == self.active_question
                )
                try:
                    route_complete = self.output_adapter.navigation.record_arrival_acquisition(
                        run_dir.name,
                        semantic_revalidated=revalidated,
                    )
                except RuntimeError:
                    route_complete = False
                    revalidated = False
                if revalidated and route_complete and arrival_purpose == "probe":
                    published = self.output_adapter.execute(
                        decision,
                        episode_id=self.active_episode_id,
                    )
                self.pending_arrival_goal_id = None
                self._publish_status(
                    "arrival_revalidated" if revalidated else "arrival_revalidation_failed",
                    goal_id=arrival_goal_id,
                    route_complete=route_complete,
                    followup_output_dispatched=published,
                    run_dir=str(run_dir),
                )
            elif completed.returncode == 0:
                published = self.output_adapter.execute(
                    decision,
                    episode_id=self.active_episode_id,
                )
                self._publish_status(
                    (
                        "probe_dispatched"
                        if published and decision.get("action") == "PROBE"
                        else "answer_published"
                        if published
                        else "answer_suppressed"
                    ),
                    root_action=decision.get("action"),
                    task_type=decision.get("task_type"),
                    run_dir=str(run_dir),
                )
            self._publish_status(
                "chain_finished" if completed.returncode == 0 else "chain_failed",
                returncode=int(completed.returncode),
                run_dir=str(run_dir),
                publish_answers=bool(
                    published and decision.get("action") == "COMMIT"
                ),
                gates=summary.get("gates", {}),
            )
        except Exception as exc:
            self._publish_status(
                "chain_failed",
                reason=type(exc).__name__,
                detail=str(exc)[:300],
                run_dir=str(run_dir),
            )
        finally:
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
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
