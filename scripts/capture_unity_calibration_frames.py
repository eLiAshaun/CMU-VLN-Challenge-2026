#!/usr/bin/env python3
"""Capture synchronized Unity RGB/semantic/depth/pose calibration frames.

This tool is intentionally external to the submitted agent.  It consumes
simulator-only semantic rendering and must never be launched in an evaluation
run except while producing an offline held-out detector calibration dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys

import cv2
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


class CalibrationCapture(Node):
    def __init__(
        self,
        output: Path,
        *,
        minimum_interval_s: float,
        minimum_translation_m: float,
    ) -> None:
        super().__init__("scnav_unity_calibration_capture")
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.minimum_interval_s = float(minimum_interval_s)
        self.minimum_translation_m = float(minimum_translation_m)
        self.bridge = CvBridge()
        self.frames = []
        self.last_stamp_s = -math.inf
        self.last_position = None
        self.latest_pose = None
        self.latest_depth = None
        self.pose_subscription = self.create_subscription(
            Odometry, "/state_estimation", self._remember_pose, 20
        )
        self.depth_subscription = self.create_subscription(
            Image, "/camera/depth", self._remember_depth, 20
        )
        subscribers = (
            Subscriber(self, Image, "/camera/image"),
            Subscriber(self, Image, "/camera/semantic_image"),
        )
        synchronizer = ApproximateTimeSynchronizer(
            subscribers, queue_size=40, slop=0.15
        )
        synchronizer.registerCallback(self._capture)
        self.synchronizer = synchronizer

    @staticmethod
    def _stamp_s(message) -> float:
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _remember_pose(self, message) -> None:
        self.latest_pose = message

    def _remember_depth(self, message) -> None:
        self.latest_depth = message

    def _capture(self, rgb_msg, semantic_msg) -> None:
        pose_msg = self.latest_pose
        depth_msg = self.latest_depth
        if pose_msg is None:
            return
        stamp_s = self._stamp_s(rgb_msg)
        position = np.asarray((
            pose_msg.pose.pose.position.x,
            pose_msg.pose.pose.position.y,
            pose_msg.pose.pose.position.z,
        ), dtype=np.float64)
        moved = (
            self.last_position is None
            or float(np.linalg.norm(position - self.last_position))
            >= self.minimum_translation_m
        )
        elapsed = stamp_s - self.last_stamp_s
        if elapsed < self.minimum_interval_s or not moved:
            return

        index = len(self.frames)
        stem = f"frame_{index:06d}_{stamp_s:.9f}"
        rgb_path = self.output / f"{stem}_rgb.png"
        semantic_path = self.output / f"{stem}_semantic.png"
        depth_path = self.output / f"{stem}_depth.npy"
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        semantic = self.bridge.imgmsg_to_cv2(
            semantic_msg, desired_encoding="rgb8"
        )
        if not cv2.imwrite(
            str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"failed to save {rgb_path}")
        if not cv2.imwrite(
            str(semantic_path),
            cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError(f"failed to save {semantic_path}")
        depth_stamp_s = None
        if (
            depth_msg is not None
            and abs(self._stamp_s(depth_msg) - stamp_s) <= 0.25
        ):
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            np.save(depth_path, np.asarray(depth))
            depth_stamp_s = self._stamp_s(depth_msg)
        orientation = pose_msg.pose.pose.orientation
        self.frames.append({
            "source_stamp_s": stamp_s,
            "semantic_source_stamp_s": self._stamp_s(semantic_msg),
            "depth_source_stamp_s": depth_stamp_s,
            "rgb_image": rgb_path.name,
            "semantic_image": semantic_path.name,
            "depth_image": depth_path.name if depth_stamp_s is not None else None,
            "sensor_position_scene": position.tolist(),
            "sensor_orientation_scene_xyzw": [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ],
        })
        self.last_stamp_s = stamp_s
        self.last_position = position
        self.get_logger().info(
            f"captured {stem} at {position.tolist()}"
        )

    def write_manifest(self, *, scene_id: str) -> None:
        manifest = self.output / "frames_manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "scnav_unity_calibration_frames_v1",
            "scene_id": scene_id,
            "simulator_only_ground_truth": True,
            "independence_contract": (
                "translation_baseline_for_360_panorama"
            ),
            "minimum_translation_m": self.minimum_translation_m,
            "frames": self.frames,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.get_logger().info(
            f"wrote {manifest} with {len(self.frames)} frames"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--minimum-interval-s", type=float, default=0.4)
    parser.add_argument("--minimum-translation-m", type=float, default=0.50)
    args = parser.parse_args()
    rclpy.init()
    node = CalibrationCapture(
        args.output,
        minimum_interval_s=args.minimum_interval_s,
        minimum_translation_m=args.minimum_translation_m,
    )

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_manifest(scene_id=args.scene_id)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
