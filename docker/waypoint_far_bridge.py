#!/usr/bin/env python3
"""System-side adapter from the official Pose2D output to FAR's goal input."""

import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class WaypointFarBridge(Node):
    """Forward only finite, official AI waypoints to FAR in map coordinates."""

    def __init__(self) -> None:
        super().__init__("waypoint_far_bridge")
        self._publisher = self.create_publisher(
            PointStamped, "/goal_point", RELIABLE_QOS
        )
        self._subscription = self.create_subscription(
            Pose2D,
            "/way_point_with_heading",
            self._on_waypoint,
            RELIABLE_QOS,
        )

    def _on_waypoint(self, waypoint: Pose2D) -> None:
        values = (
            float(waypoint.x),
            float(waypoint.y),
            float(waypoint.theta),
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("Suppressed non-finite FAR goal")
            return
        goal = PointStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.point.x = values[0]
        goal.point.y = values[1]
        goal.point.z = 0.0
        self._publisher.publish(goal)


def main() -> None:
    rclpy.init()
    node = WaypointFarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
