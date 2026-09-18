"""Go2 sensor/navigation binding. Shares the research Runtime, not CMU topics.

Prerequisites: a running Go2 Nav2 stack with calibrated map->base_link->camera
TF, and the RealSense ROS driver. This process never sends motor commands.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import time
import traceback

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException

from rebuild.runtime import Runtime
from rebuild.run_log import RunLog
from rebuild.task_ir import resolve, binding_supported
from rebuild.navigation import needs_broad_coverage
from .config import load_config
from .sensors import RGBDProjector, PinholeObservationAdapter, depth_metres, rgb_image
from .sensors import transform_matrix, rotation_xyzw, yaw_of, angle_difference
from .navigation import navigation_factory, costmap_points, HeadingSweep
from .nav2_transport import Nav2Transport


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


class Go2Node(Node):
    def __init__(self, config):
        super().__init__('go2_semantic_ai')
        self.config = config
        self.runtime = Runtime(config, observation_adapter=PinholeObservationAdapter(),
                               navigation_factory=navigation_factory)
        self.projector = RGBDProjector(config['depth_min_m'], config['depth_max_m'], config['depth_stride'])
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='go2-models')
        self.future, self.future_kind, self.future_epoch = None, '', 0
        self.epoch, self.question, self.episode, self.log = 0, '', None, None
        self.started, self.terminal = 0.0, True
        self.packets = deque(maxlen=8)
        self.pose, self.pose_stamp = None, -1.0
        self.costmap, self.terrain_key = None, None
        self.terrain = np.empty((0, 4), dtype=np.float32)
        self.sweep, self.sweep_position = None, None
        self.motion_kind, self.ready_after = None, -1.0
        self.last_pose_log, self.last_wait = -1.0, ''
        self.tf = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf, self)
        self.transport = Nav2Transport(self, config)
        self.status = self.create_publisher(String, config['status_topic'], 10)
        self.numerical = self.create_publisher(Int32, config['numerical_topic'], 10)
        self.marker = self.create_publisher(Marker, config['marker_topic'], 10)
        self.create_subscription(String, config['task_topic'], self.on_task, 10)
        self.rgb_sub = message_filters.Subscriber(self, Image, config['rgb_topic'], qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, config['depth_topic'], qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, config['camera_info_topic'], qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], 10, config['rgb_depth_sync_seconds'])
        self.sync.registerCallback(lambda rgb, depth, info: self.packets.append((rgb, depth, info)))
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=(DurabilityPolicy.TRANSIENT_LOCAL if config['costmap_transient_local']
                                     else DurabilityPolicy.VOLATILE))
        self.create_subscription(OccupancyGrid, config['costmap_topic'], self.on_costmap, qos)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"Go2 RGB-D AI listening on {config['task_topic']}; Nav2 must already be running")

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def event(self, kind, **fields):
        if self.log:
            self.log.event(kind, **fields)

    def waiting(self, reason):
        if reason != self.last_wait:
            self.last_wait = reason
            self.event('waiting', reason=reason)
            self.get_logger().info(reason)
            self.status.publish(String(data=json.dumps({'state': 'waiting', 'reason': reason})))

    def on_task(self, message):
        question = ' '.join(message.data.split())
        if not question or (question == self.question and not self.terminal):
            return
        if self.episode is not None:
            with self.episode.lock:
                self.episode.terminal = True
        self.transport.cancel()
        self.transport.error = None
        self.epoch += 1
        self.question, self.episode, self.terminal = question, None, False
        self.started = time.monotonic()
        self.sweep, self.motion_kind = None, None
        self.ready_after = self.now_seconds() + self.config['observation_settle_seconds']
        self.packets.clear()
        run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.log = RunLog(self.config['output_root'] + '/' + run)
        self.log.write('request.json', {'question': question, 'config': self.config,
                                      'source': self.config['task_topic'], 'epoch': self.epoch})
        self.event('question_received', question=question)

    def on_costmap(self, message):
        self.costmap = message

    def update_pose(self):
        transform = self.tf.lookup_transform(self.config['world_frame'], self.config['base_frame'], Time())
        timestamp = stamp(transform)
        age = self.now_seconds() - timestamp
        if age < -0.1 or age > self.config['max_pose_age_seconds']:
            raise ValueError(f'Localization TF is stale or clocks differ: age={age:.3f}s')
        self.pose, self.pose_stamp = transform_matrix(transform.transform), timestamp
        if self.episode is not None and timestamp > self.episode.navigation.stamp:
            self.episode.navigation.update_pose(self.pose[:3, 3].tolist(), timestamp)
        if self.log and timestamp-self.last_pose_log >= 0.1:
            self.event('trajectory', stamp=timestamp, position=self.pose[:3, 3].tolist(),
                       quaternion_source='full_tf', yaw=yaw_of(self.pose))
            self.last_pose_log = timestamp

    def update_terrain(self):
        message = self.costmap
        if message is None:
            return False
        key = id(message)
        if key == self.terrain_key:
            return True
        origin = message.info.origin
        q = origin.orientation
        T_map_grid = np.eye(4)
        T_map_grid[:3, :3] = rotation_xyzw([q.x, q.y, q.z, q.w])
        T_map_grid[:3, 3] = [origin.position.x, origin.position.y, origin.position.z]
        if message.header.frame_id.lstrip('/') != self.config['world_frame']:
            transform = self.tf.lookup_transform(self.config['world_frame'], message.header.frame_id,
                                                 Time.from_msg(message.header.stamp))
            T_map_grid = transform_matrix(transform.transform) @ T_map_grid
        self.terrain = costmap_points(message.data, message.info.width, message.info.height,
                                      message.info.resolution, T_map_grid,
                                      self.config['costmap_free_cost'])
        self.terrain_key = key
        return True

    def capture(self):
        for rgb, depth, info in reversed(self.packets):
            timestamp = stamp(rgb)
            if timestamp < self.ready_after or (self.episode and timestamp <= self.episode.last_image_stamp):
                continue
            age = self.now_seconds()-timestamp
            if age < -0.1 or age > self.config['max_observation_age_seconds']:
                continue
            frame_id = info.header.frame_id
            if rgb.header.frame_id != frame_id or depth.header.frame_id != frame_id:
                raise ValueError('RGB, aligned depth and color CameraInfo must use the same optical frame')
            if (info.width, info.height) != (rgb.width, rgb.height):
                raise ValueError('CameraInfo resolution does not match RGB')
            try:
                query_time = Time.from_msg(rgb.header.stamp)
                camera = self.tf.lookup_transform(self.config['world_frame'], frame_id, query_time)
                base = self.tf.lookup_transform(self.config['world_frame'], self.config['base_frame'], query_time)
            except TransformException:
                continue
            data = depth_metres(depth.data, depth.width, depth.height, depth.step,
                                depth.encoding, bool(depth.is_bigendian), self.config['depth_16u_metres_per_unit'])
            result = self.projector.frame(
                f'rgbd_{self.epoch}_{timestamp:.6f}', timestamp, rgb_image(rgb), data,
                np.array(info.k).reshape(3, 3), info.d, info.distortion_model,
                transform_matrix(base.transform), transform_matrix(camera.transform), stamp(depth))
            if not len(result.registered_points_map):
                self.waiting('Waiting for valid measured RealSense depth')
                return None
            self.last_wait = ''
            return result
        self.waiting('Waiting for fresh synchronized RGB-D, color CameraInfo and image-time TF')
        return None

    def submit_work(self, kind, function, *args):
        self.future_kind, self.future_epoch = kind, self.epoch
        self.future = self.pool.submit(function, *args)

    def consume_work(self):
        if self.future is None or not self.future.done():
            return
        future, epoch, kind = self.future, self.future_epoch, self.future_kind
        self.future = None
        try:
            value = future.result()
        except Exception:
            if epoch == self.epoch and not self.terminal:
                raise
            return
        if epoch != self.epoch or self.terminal:
            return
        if kind == 'compile':
            self.episode = value
        else:
            self.episode.navigation.note_observation(
                self.episode.last_observation_position, self.episode.last_image_stamp)
            self.sweep.consumed()
            self.event('heading_observation_consumed', heading_index=self.sweep.index,
                       image_stamp=self.episode.last_image_stamp)

    def new_sweep(self):
        self.sweep = HeadingSweep(self.config['scan_yaws_deg'])
        self.sweep.reset(yaw_of(self.pose))
        self.sweep_position = self.pose[:2, 3].copy()
        self.ready_after = self.now_seconds() + self.config['observation_settle_seconds']

    def tick(self):
        try:
            self._tick()
        except Exception as exc:
            self.event('runtime_error', error=str(exc), traceback=traceback.format_exc())
            self.finish('runtime_failure', error=str(exc))

    def _tick(self):
        self.transport.pump()
        self.consume_work()
        if self.terminal:
            return
        remaining = self.config['question_time_budget_seconds'] - (time.monotonic()-self.started)
        if remaining <= 0:
            self.finish('time_budget_exceeded')
            return
        if self.transport.error:
            self.finish('navigation_failure', error=self.transport.error)
            return
        try:
            self.update_pose()
        except (TransformException, ValueError) as exc:
            # Stop a currently requested mission when its localization stops updating.
            if self.transport.busy:
                self.transport.cancel()
                self.finish('localization_unavailable', error=str(exc))
            else:
                self.waiting(f'Waiting for map localization: {exc}')
            return
        if self.episode is None:
            if self.future is None and not self.transport.busy:
                self.submit_work('compile', self.runtime.begin, self.question, self.log, self.started)
            return
        if self.future is not None or self.transport.busy:
            return
        if self.motion_kind:
            if self.transport.status != 'succeeded':
                self.waiting('Waiting for Nav2 action server/result')
                return
            self.event('navigation_goal_reached', goal=self.transport.last_success, kind=self.motion_kind)
            if self.motion_kind == 'move':
                self.new_sweep()
            else:
                self.ready_after = self.transport.completed_stamp + self.config['observation_settle_seconds']
            self.motion_kind = None
        if self.sweep is None:
            self.new_sweep()
        if not self.sweep.complete:
            yaw = self.sweep.target
            if abs(angle_difference(yaw_of(self.pose), yaw)) > math.radians(self.config['scan_yaw_tolerance_deg']):
                self.transport.submit((*self.sweep_position, yaw))
                self.motion_kind = 'turn'
                return
            frame = self.capture()
            if frame is not None:
                self.submit_work('observe', self.runtime.observe, self.episode, frame)
            return
        try:
            if not self.update_terrain():
                self.waiting('Waiting for Nav2 OccupancyGrid costmap')
                return
        except TransformException as exc:
            self.waiting(f'Waiting for costmap TF: {exc}')
            return
        episode = self.episode
        with episode.lock:
            scene = episode.scene
        result = resolve(episode.task, scene.records)
        # Our intentional stop-and-observe phase is not a locomotion stall.
        episode.navigation.progress_stamp = episode.navigation.stamp
        navigation = episode.navigation.plan(episode.task, scene.records, self.terrain,
            remaining, result, scene.proposals, scene.snapshot.get('aliases', {}))
        task_type = episode.task['task_type']
        unique_reference = (task_type == 'object_reference' and len(result.object_ids) == 1
                            and binding_supported(episode.task['expression'], result)
                            and not needs_broad_coverage(episode.task['expression']))
        answer_due = remaining <= self.config['answer_reserve_seconds']
        if navigation.get('complete') or unique_reference or answer_due:
            if self.publish_final(result, navigation, scene):
                return
        waypoint = navigation.get('waypoint')
        if waypoint is not None:
            x, y = waypoint[:2]
            yaw = math.atan2(y-self.pose[1, 3], x-self.pose[0, 3])
            goal = (float(x), float(y), yaw)
            if self.transport.last_success is not None and np.linalg.norm(
                    np.asarray(goal[:2])-self.transport.last_success[:2]) < 0.05:
                self.waiting('Semantic planner has not produced a new destination')
                return
            self.transport.submit(goal)
            self.motion_kind = 'move'
            self.event('navigation_goal_sent', goal=goal, navigation=navigation)
        else:
            self.waiting('More semantic evidence or a reachable exploration destination is needed')

    def publish_final(self, result, navigation, scene):
        kind = self.episode.task['task_type']
        if kind == 'numerical' and isinstance(result.value, int):
            self.numerical.publish(Int32(data=result.value))
        elif kind == 'object_reference' and len(result.object_ids) == 1:
            record = scene.records[result.object_ids[0]]
            if record.center is None or record.bbox is None or np.any(record.bbox <= 0):
                return False
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = 'map', self.get_clock().now().to_msg()
            marker.ns, marker.id, marker.type, marker.action = 'go2_selected_object', 0, Marker.CUBE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, record.center)
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = map(float, record.bbox)
            marker.color.g, marker.color.a = 1.0, 0.6
            marker.text = max(record.class_scores, key=record.class_scores.get)
            self.marker.publish(marker)
        elif kind != 'instruction' or not navigation.get('complete'):
            return False
        self.log.write('final_object_store.json', scene.snapshot)
        self.finish(kind + '_output', query_result=asdict(result), navigation=navigation,
                    numerical_answer=result.value, object_ids=result.object_ids,
                    observations=scene.observations, evidence_complete=result.complete,
                    image_stamp=scene.image_stamp)
        return True

    def finish(self, decision, **fields):
        if self.terminal:
            return
        self.terminal = True
        if self.episode:
            with self.episode.lock:
                self.episode.terminal = True
        self.transport.cancel()
        payload = {'question': self.question, 'decision': decision,
                   'elapsed_seconds': time.monotonic()-self.started,
                   'nav2_state': self.transport.status, **fields}
        if self.log:
            self.log.write('summary.json', payload)
        self.status.publish(String(data=json.dumps(payload, default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))))
        self.get_logger().info(f'Go2 episode finished: {decision}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Go2Node(load_config(args.config))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish('operator_shutdown')
        node.transport.cancel()
        end = time.monotonic() + 3.0
        while rclpy.ok() and node.transport.busy and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.transport.busy:
            node.get_logger().error('Nav2 cancellation not confirmed; stop the robot with its operator control')
        node.pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
