"""Validated ROS2 adapter for official challenge outputs."""

from __future__ import annotations

import math
from typing import Any, Mapping

from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int8, Int32
from visualization_msgs.msg import Marker

from integrations.execution.navigation_executor import NavigationExecutor


class RosOutputAdapter:
    def __init__(self, node, *, acquisition_requester=None) -> None:
        self.node = node
        self.numerical = node.create_publisher(Int32, "/numerical_response", 10)
        self.marker = node.create_publisher(Marker, "/selected_object_marker", 10)
        self.waypoint = node.create_publisher(Pose2D, "/way_point_with_heading", 10)
        self.stop = node.create_publisher(Int8, "/stop", 10)
        self.navigation = NavigationExecutor(
            self._publish_waypoint,
            stop_sender=self._publish_stop,
            acquisition_requester=acquisition_requester,
        )
        self._episode_id = ""
        self._terminal_fingerprint: tuple | None = None
        self._last_pose: tuple[float, float, float] | None = None

    def begin_episode(self, episode_id: str) -> None:
        self._episode_id = str(episode_id)
        self._terminal_fingerprint = None
        self.navigation.begin_episode(self._episode_id)

    def update_odometry(self, message) -> None:
        pose = message.pose.pose
        orientation = pose.orientation
        heading = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        self._last_pose = (float(pose.position.x), float(pose.position.y), heading)
        linear = message.twist.twist.linear
        speed = math.sqrt(float(linear.x) ** 2 + float(linear.y) ** 2 + float(linear.z) ** 2)
        self.navigation.update_odometry(*self._last_pose, speed_mps=speed)

    def execute(self, decision: Mapping[str, Any], *, episode_id: str) -> bool:
        if episode_id != self._episode_id:
            self.node.get_logger().error("Episode mismatch suppressed output")
            return False
        if decision.get("schema_version") != "root_decision_v1" or decision.get("action") not in {"COMMIT", "PROBE"}:
            return False
        action = str(decision["action"])
        task_type = str(decision.get("task_type", ""))
        fingerprint = (
            task_type,
            decision.get("answer"),
            repr(decision.get("selected_object")),
            repr(decision.get("waypoint_sequence")),
        )
        if action == "COMMIT" and task_type != "instruction_following" and self._terminal_fingerprint is not None:
            self.node.get_logger().warning("Duplicate terminal publication suppressed")
            return False
        if action == "PROBE":
            if self._last_pose is None or self.navigation.has_active_goal or self.navigation.awaiting_arrival_acquisition:
                return False
            return self.navigation.dispatch_route(
                decision.get("waypoint_sequence", ()),
                start_pose=self._last_pose,
                purpose="probe",
            )
        if task_type == "numerical":
            answer = decision.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, int) or answer < 0:
                return False
            self.numerical.publish(Int32(data=answer))
        elif task_type == "object_reference":
            if not self._publish_marker(decision.get("selected_object")):
                return False
        elif task_type == "instruction_following":
            if self._last_pose is None or self.navigation.has_active_goal or self.navigation.awaiting_arrival_acquisition:
                return False
            if not self.navigation.dispatch_route(
                decision.get("waypoint_sequence", ()),
                start_pose=self._last_pose,
                purpose="instruction",
            ):
                return False
        else:
            return False
        if task_type != "instruction_following":
            self._terminal_fingerprint = fingerprint
        return True

    def _publish_marker(self, obj: object) -> bool:
        if not isinstance(obj, Mapping):
            return False
        center = obj.get("center_3d")
        bbox = obj.get("bbox_3d")
        if not isinstance(center, list) or not isinstance(bbox, list) or len(center) != 3 or len(bbox) != 3:
            return False
        values = [float(value) for value in (*center, *bbox)]
        if not all(math.isfinite(value) for value in values) or any(value <= 0.0 for value in values[3:]):
            return False
        message = Marker()
        message.header.frame_id = "map"
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.ns = str(obj.get("class_label", "unknown"))
        message.id = int(obj.get("object_id", 0))
        message.action = Marker.ADD
        message.type = Marker.CUBE
        message.pose.position.x, message.pose.position.y, message.pose.position.z = values[:3]
        message.pose.orientation.w = 1.0
        message.scale.x, message.scale.y, message.scale.z = values[3:]
        message.color.a = 0.6
        message.color.g = 0.8
        self.marker.publish(message)
        return True

    def _publish_waypoint(self, x: float, y: float, heading: float) -> bool:
        values = (float(x), float(y), float(heading))
        if not all(math.isfinite(value) for value in values):
            return False
        self.waypoint.publish(Pose2D(x=values[0], y=values[1], theta=values[2]))
        return True

    def _publish_stop(self, hold: bool) -> None:
        self.stop.publish(Int8(data=2 if hold else 0))
