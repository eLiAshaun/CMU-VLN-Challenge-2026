"""Read-only ROS checks. Does not load models or send navigation/motor goals."""
import argparse
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener, TransformException
from .config import load_config
from .packet import packet_timing, timing_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--timeout', type=float, default=20.0)
    args, ros_args = parser.parse_known_args()
    config = load_config(args.config)
    rclpy.init(args=ros_args)
    node = Node('go2_ai_preflight')
    received = {}
    subscriptions = []
    for key, cls in [('rgb_topic', Image), ('depth_topic', Image), ('camera_info_topic', CameraInfo)]:
        subscriptions.append(node.create_subscription(cls, config[key],
            lambda message, key=key: received.update({key: message}), qos_profile_sensor_data))
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL if config['costmap_transient_local']
                    else DurabilityPolicy.VOLATILE)
    subscriptions.append(node.create_subscription(OccupancyGrid, config['costmap_topic'],
        lambda message: received.update({'costmap': message}), qos))
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    client = ActionClient(node, NavigateToPose, config['nav2_action'])
    end = time.monotonic()+args.timeout
    report = {}
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        report = {key: key in received for key in ['rgb_topic', 'depth_topic', 'camera_info_topic', 'costmap']}
        report['nav2_action_server'] = client.server_is_ready()
        report['image_time_tf'] = False
        report['aligned_rgbd_contract'] = False
        if all(report[key] for key in ['rgb_topic', 'depth_topic', 'camera_info_topic']):
            rgb, depth, info = [received[key] for key in ['rgb_topic', 'depth_topic', 'camera_info_topic']]
            timing = packet_timing((rgb, depth, info),
                node.get_clock().now().nanoseconds*1e-9,
                config['max_observation_age_seconds'], config['rgb_depth_sync_seconds'])
            report['rgbd_timing'] = timing_valid(timing)
            report['aligned_rgbd_contract'] = (
                rgb.header.frame_id == depth.header.frame_id == info.header.frame_id
                and (rgb.width, rgb.height) == (depth.width, depth.height) == (info.width, info.height)
                and info.k[0] > 0 and info.k[4] > 0
                and depth.encoding.upper() in {'16UC1', '32FC1'})
            try:
                query_time = Time.from_msg(rgb.header.stamp)
                buffer.lookup_transform(config['world_frame'], info.header.frame_id, query_time)
                buffer.lookup_transform(config['world_frame'], config['base_frame'], query_time)
                age = node.get_clock().now().nanoseconds*1e-9 - (rgb.header.stamp.sec+rgb.header.stamp.nanosec*1e-9)
                report['image_time_tf'] = -0.1 <= age <= config['max_observation_age_seconds']
            except TransformException:
                pass
        if all(report.values()):
            break
    passed = all(report.values())
    print(json.dumps({'read_only_preflight_passed': passed, 'checks': report,
                      'not_tested': ['GPU inference', 'Nav2 motion', 'Go2 hardware safety', 'semantic accuracy']}, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
