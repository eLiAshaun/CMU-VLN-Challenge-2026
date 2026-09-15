"""Pure ROS2 serialization for already-authorized root decisions."""

from __future__ import annotations

import math
from typing import Any, Mapping

from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

from integrations.execution.root_finalizer import ROOT_DECISION_SCHEMA


class RosOutputAdapter:
    def __init__(self, node) -> None:
        self.node = node
        self.numerical = node.create_publisher(Int32, "/numerical_response", 10)
        self.marker = node.create_publisher(Marker, "/selected_object_marker", 10)
        self.waypoint = node.create_publisher(Pose2D, "/way_point_with_heading", 10)
        self._episode_id = ""
        self._terminal_fingerprint: tuple[object, ...] | None = None

    def begin_episode(self, episode_id: str) -> None:
        self._episode_id = str(episode_id)
        self._terminal_fingerprint = None

    def publish_root_decision(
        self,
        decision: Mapping[str, Any],
        *,
        episode_id: str,
    ) -> bool:
        """Serialize one RootFinalizer-authorized COMMIT.

        Envelope validation is deliberately binary: an unauthorized decision
        is not reinterpreted or repaired by the adapter.
        """
        if episode_id != self._episode_id:
            return False
        if (
            decision.get("schema_version") != ROOT_DECISION_SCHEMA
            or decision.get("action") != "COMMIT"
            or decision.get("answer_authorized") is not True
        ):
            return False
        task_type = str(decision.get("task_type", ""))
        fingerprint = (
            task_type,
            decision.get("answer"),
            repr(decision.get("selected_object")),
        )
        if self._terminal_fingerprint is not None:
            return False
        if task_type == "numerical":
            answer = decision.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, int) or answer < 0:
                return False
            self.numerical.publish(Int32(data=int(answer)))
        elif task_type == "object_reference":
            if not self._publish_marker(decision.get("selected_object")):
                return False
        elif task_type == "instruction_following":
            # Instruction completion is represented by actual trajectory
            # facts and an authorized root decision; the challenge defines no
            # separate terminal ROS answer message for this task type.
            pass
        else:
            return False
        self._terminal_fingerprint = fingerprint
        return True

    def _publish_marker(self, obj: object) -> bool:
        if not isinstance(obj, Mapping):
            return False
        try:
            object_id = int(obj["object_id"])
            values = [
                float(value)
                for value in (*obj["center_3d"], *obj["bbox_3d"])
            ]
        except (KeyError, TypeError, ValueError):
            return False
        if (
            object_id < 0
            or len(values) != 6
            or not all(math.isfinite(value) for value in values)
        ):
            return False
        message = Marker()
        message.header.frame_id = "map"
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.ns = str(obj.get("class_label", "unknown"))
        message.id = object_id
        message.action = Marker.ADD
        message.type = Marker.CUBE
        message.pose.position.x, message.pose.position.y, message.pose.position.z = values[:3]
        message.pose.orientation.w = 1.0
        message.scale.x, message.scale.y, message.scale.z = values[3:]
        message.color.a = 0.6
        message.color.g = 0.8
        self.marker.publish(message)
        return True

    def publish_waypoint(self, x: float, y: float, heading: float) -> bool:
        try:
            values = (float(x), float(y), float(heading))
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values):
            return False
        self.waypoint.publish(
            Pose2D(x=values[0], y=values[1], theta=values[2])
        )
        return True
