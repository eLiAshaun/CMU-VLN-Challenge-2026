"""Independent development subscriber for actual official ROS outputs."""
import argparse
import json
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker
from rclpy.qos import qos_profile_sensor_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=600)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('cmu_rebuild_output_probe')
    stream = path.open('a', buffering=1)
    last_pose = [0.0]

    def record(topic, **fields):
        stream.write(json.dumps({'received_time': time.time(), 'topic': topic, **fields})+'\n')

    def marker(message):
        p, s = message.pose.position, message.scale
        record('/selected_object_marker', frame=message.header.frame_id,
               center=[p.x, p.y, p.z], extent=[s.x, s.y, s.z], label=message.text)

    def pose(message):
        timestamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        if timestamp-last_pose[0] >= 0.1:
            p = message.pose.pose.position
            record('/state_estimation', stamp=timestamp, position=[p.x, p.y, p.z])
            last_pose[0] = timestamp

    node.create_subscription(Int32, '/numerical_response', lambda m: record('/numerical_response', value=m.data), 10)
    node.create_subscription(Marker, '/selected_object_marker', marker, 10)
    node.create_subscription(Pose2D, '/way_point_with_heading', lambda m: record('/way_point_with_heading', waypoint=[m.x, m.y, m.theta]), 10)
    node.create_subscription(Odometry, '/state_estimation', pose, qos_profile_sensor_data)
    end = time.monotonic()+args.duration
    try:
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        stream.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
