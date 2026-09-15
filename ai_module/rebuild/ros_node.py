"""Official ROS interface with independent reception and one model executor."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import math
import time
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker

from .config import load_config
from .contracts import Frame
from .navigation import needs_broad_coverage
from .run_log import RunLog
from .runtime import Runtime
from .task_ir import binding_supported, resolve


def stamp(message) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def quaternion_rotation(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def image_rgb(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}[encoding]
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    pixels = rows[:, :message.width * channels].reshape(message.height, message.width, channels)
    if channels == 1:
        return np.repeat(pixels, 3, axis=2)
    pixels = pixels[:, :, :3]
    return pixels[:, :, ::-1].copy() if encoding.startswith('bgr') else pixels.copy()


def cloud_points(message: PointCloud2, intensity: bool = False) -> np.ndarray:
    fields = ['x', 'y', 'z']
    if intensity and any(field.name == 'intensity' for field in message.fields):
        fields.append('intensity')
    points = point_cloud2.read_points_numpy(message, field_names=fields, skip_nans=True)
    return np.asarray(points, dtype=np.float32).reshape(-1, len(fields)).copy()


class RebuildNode(Node):
    def __init__(self, config: dict):
        super().__init__('cmu_ai_rebuild')
        self.config = config
        self.process_started = time.monotonic()
        self.runtime = Runtime(config)
        self.executor_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cmu-models')
        self.future = None
        self.future_kind = ''
        self.future_epoch = 0
        self.epoch = 0
        self.question = ''
        self.episode = None
        self.log = None
        self.started = 0.0
        self.failure = None
        self.latest_image = None
        self.scans = deque(maxlen=12)
        self.sensor_scan = None
        self.poses = deque(maxlen=1200)
        self.terrain = np.empty((0, 4), dtype=np.float32)
        self.terrain_maps = {}
        self.latest_pose = None
        self.last_trajectory_stamp = -1.0
        self.last_waypoint = None
        self.last_sensor_wait = ''
        self.numerical = self.create_publisher(Int32, '/numerical_response', 10)
        self.marker = self.create_publisher(Marker, '/selected_object_marker', 10)
        self.waypoint = self.create_publisher(Pose2D, '/way_point_with_heading', 10)
        self.create_subscription(String, '/challenge_question', self.on_question, 10)
        self.create_subscription(Image, '/camera/image', self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/state_estimation', self.on_pose, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/registered_scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/sensor_scan', self.on_sensor_scan, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/terrain_map', lambda m: self.on_terrain(m, 'local'), qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/terrain_map_ext', lambda m: self.on_terrain(m, 'extended'), qos_profile_sensor_data)
        self.create_timer(0.2, self.tick)
        self.get_logger().info('GroundingDINO / SAM2.1 chain listening on /challenge_question')

    def on_question(self, message: String) -> None:
        question = ' '.join(message.data.split())
        if not question:
            return
        if question == self.question:
            if self.log:
                self.log.event('duplicate_question_ignored')
            return
        if self.episode and not self.episode.terminal:
            self.episode.log.event('episode_replaced_by_new_question')
            with self.episode.lock:
                self.episode.terminal = True
        self.epoch += 1
        self.question, self.episode, self.failure = question, None, None
        self.started = self.process_started if self.epoch == 1 else time.monotonic()
        self.last_waypoint = None
        self.latest_image, self.sensor_scan, self.latest_pose = None, None, None
        self.scans.clear()
        self.poses.clear()
        self.terrain_maps.clear()
        self.terrain = np.empty((0, 4), dtype=np.float32)
        self.last_trajectory_stamp = -1.0
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.log = RunLog(self.config['output_root'] + '/' + run_id)
        self.log.write('request.json', {'question': question, 'source': '/challenge_question',
                                      'config': self.config, 'epoch': self.epoch})
        self.log.event('question_received', question=question)
        self.get_logger().info(f'New episode {run_id}: {question}')

    def on_image(self, message: Image) -> None:
        self.latest_image = message

    def on_scan(self, message: PointCloud2) -> None:
        if message.header.frame_id.lstrip('/') == 'map':
            self.scans.append(message)

    def on_sensor_scan(self, message: PointCloud2) -> None:
        self.sensor_scan = message  # recorded evidence; map scan owns geometry

    def on_terrain(self, message: PointCloud2, source: str) -> None:
        if message.header.frame_id.lstrip('/') == 'map':
            points = cloud_points(message, intensity=True)
            self.terrain_maps[source] = points
            maps = list(self.terrain_maps.values())
            self.terrain = np.concatenate(maps) if len({p.shape[1] for p in maps}) == 1 else points

    def on_pose(self, message: Odometry) -> None:
        if message.header.frame_id.lstrip('/') != 'map' or message.child_frame_id.lstrip('/') != 'sensor':
            return
        p, q = message.pose.pose.position, message.pose.pose.orientation
        position, quaternion = [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        if not np.isfinite(position + quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
            return
        timestamp = stamp(message)
        if self.poses and timestamp < self.poses[-1][0]:
            self.poses.clear()  # simulator restarted; never interpolate across clocks
        self.poses.append((timestamp, np.array(position), np.array(quaternion)))
        self.latest_pose = (timestamp, position)
        if self.episode:
            self.episode.navigation.update_pose(position, timestamp)
        if self.log and timestamp - self.last_trajectory_stamp >= 0.1:
            self.log.event('trajectory', stamp=timestamp, position=position)
            self.last_trajectory_stamp = timestamp

    def pose_at(self, timestamp: float) -> np.ndarray | None:
        samples = list(self.poses)
        if not samples or timestamp < samples[0][0] or timestamp > samples[-1][0]:
            return None
        before = samples[0]
        for after in samples:
            if after[0] >= timestamp:
                alpha = (timestamp-before[0]) / max(after[0]-before[0], 1e-12)
                p = before[1] * (1-alpha) + after[1] * alpha
                qa, qb = before[2], after[2]
                if np.dot(qa, qb) < 0:
                    qb = -qb
                # Normalized interpolation at 100-200 Hz, before any inference.
                transform = np.eye(4)
                transform[:3, :3] = quaternion_rotation((1-alpha)*qa + alpha*qb)
                transform[:3, 3] = p
                return transform
            before = after
        return None

    def capture(self) -> Frame | None:
        if self.latest_image is None or not self.scans:
            self.sensor_wait('waiting_for_camera_and_registered_scan')
            return None
        image_stamp = stamp(self.latest_image)
        transform = self.pose_at(image_stamp)
        if transform is None:
            self.sensor_wait('waiting_for_image_time_pose')
            return None
        scan = min(self.scans, key=lambda item: abs(stamp(item)-image_stamp))
        if abs(stamp(scan)-image_stamp) > self.config['max_camera_lidar_offset_seconds']:
            self.sensor_wait('waiting_for_synchronized_registered_scan')
            return None
        if self.episode and image_stamp <= self.episode.last_image_stamp:
            return None
        self.last_sensor_wait = ''
        frame = Frame(f'image_{self.epoch}_{image_stamp:.6f}', image_stamp,
                      image_rgb(self.latest_image), transform, cloud_points(scan), stamp(scan))
        if self.log:
            self.log.event('sensor_snapshot', image_stamp=image_stamp, registered_scan_stamp=stamp(scan),
                           sensor_scan_stamp=stamp(self.sensor_scan) if self.sensor_scan is not None else None,
                           registered_frame=scan.header.frame_id,
                           panorama_shape=frame.panorama_rgb.shape,
                           registered_point_count=len(frame.registered_points_map))
        return frame

    def sensor_wait(self, reason: str) -> None:
        if reason != self.last_sensor_wait and self.log:
            self.log.event('observation_pending', reason=reason)
        self.last_sensor_wait = reason

    def question_deadline(self) -> float:
        return self.started + float(self.config['question_time_budget_seconds'])

    def deadline_stage(self) -> str:
        if self.episode is None:
            return 'compile'
        if self.future is not None:
            return {
                'compile': 'compile',
                'perception_geometry_memory': 'perception',
            }.get(self.future_kind, self.future_kind or 'worker')
        return 'navigation'

    def fail_deadline(self, stage: str) -> None:
        if self.failure or (self.episode is not None and self.episode.terminal):
            if self.future is not None:
                self.future.cancel()
            return
        elapsed = max(0.0, time.monotonic() - self.started)
        if self.log:
            self.log.event('question_deadline_exceeded', stage=stage,
                           elapsed_seconds=elapsed, ros_published=False)
        # ThreadPoolExecutor cannot interrupt a model call that is already
        # running. cancel() still prevents a queued call from starting; the
        # terminal/failure state below prevents either kind of late worker
        # result from producing an output.
        if self.future is not None:
            self.future.cancel()
        self.record_failure(stage, TimeoutError('question_deadline_exceeded'))

    def tick(self) -> None:
        try:
            self._tick()
        except Exception as exc:
            self.record_failure('ros_execution', exc)

    def _tick(self) -> None:
        if self.future is not None and self.future.done():
            future, epoch, kind = self.future, self.future_epoch, self.future_kind
            self.future = None
            stale = (epoch != self.epoch or bool(self.failure) or
                     (self.episode is not None and self.episode.terminal))
            late = epoch == self.epoch and not stale and time.monotonic() >= self.question_deadline()
            try:
                result = future.result()
                if stale or late:
                    if self.log:
                        self.log.event('late_worker_discarded', worker_kind=kind,
                                       worker_epoch=epoch, current_epoch=self.epoch,
                                       deadline_exceeded=late)
                    if late:
                        self.fail_deadline('compile' if kind == 'compile' else 'perception')
                elif epoch == self.epoch and kind == 'compile':
                    self.episode = result
                    if self.latest_pose:
                        self.episode.navigation.update_pose(self.latest_pose[1], self.latest_pose[0])
                elif epoch == self.epoch and self.episode is not None:
                    self.episode.navigation.note_observation(
                        self.episode.last_observation_position, self.episode.last_image_stamp)
            except Exception as exc:
                if stale or late:
                    if self.log:
                        self.log.event('late_worker_discarded', worker_kind=kind,
                                       worker_epoch=epoch, current_epoch=self.epoch,
                                       deadline_exceeded=late, error=str(exc))
                    if late:
                        self.fail_deadline('compile' if kind == 'compile' else 'perception')
                elif epoch == self.epoch:
                    if self.episode is None or not self.episode.terminal:
                        self.record_failure(kind, exc)
                    else:
                        self.log.event('post_completion_worker_error', error=str(exc))
        if not self.question or self.failure:
            return
        # A successful output is already terminal; its 530 s reserve-based
        # publication must not be rewritten as a 600 s deadline failure.
        if self.episode is not None and self.episode.terminal:
            if self.future is not None:
                self.future.cancel()
            return
        if time.monotonic() >= self.question_deadline():
            self.fail_deadline(self.deadline_stage())
            return
        if self.episode is None:
            if self.future is None:
                self.future_kind, self.future_epoch = 'compile', self.epoch
                self.future = self.executor_pool.submit(self.runtime.begin, self.question, self.log, self.started)
            return
        episode = self.episode
        if episode.terminal:
            return
        remaining = self.question_deadline() - time.monotonic()
        if remaining <= 0:
            self.fail_deadline('navigation')
            return
        with episode.lock:
            scene = episode.scene
        result = resolve(episode.task, scene.records)
        navigation = (episode.navigation.plan(episode.task, scene.records, self.terrain,
                                              remaining, result, scene.proposals,
                                              scene.snapshot.get('aliases', {})) if scene.observations else
                      {'waypoint': None, 'complete': False, 'progress': 'initial_observation'})
        if time.monotonic() >= self.question_deadline():
            self.fail_deadline('navigation')
            return
        task_type = episode.task['task_type']
        if navigation.get('motion_event'):
            self.log.event('navigation_motion_issue', **navigation['motion_event'])
        exhausted = remaining <= self.config['answer_reserve_seconds']
        instruction_done = task_type == 'instruction' and navigation.get('complete', False)
        # Counting/extrema need translated coverage. Unique reference can end
        # once its AST binds a concrete object; the marker remains its bbox.
        unique_reference = (task_type == 'object_reference' and binding_supported(episode.task['expression'], result)
                            and len(result.object_ids) == 1
                            and not needs_broad_coverage(episode.task['expression']))
        coverage_done = bool(navigation.get('coverage_complete', False))
        if scene.observations and (instruction_done or unique_reference or exhausted or coverage_done):
            if self.publish_final(result, navigation, scene):
                return
        if time.monotonic() >= self.question_deadline():
            self.fail_deadline('navigation')
            return
        waypoint = navigation.get('waypoint')
        # A stalled command invalidates the bridge's previous waypoint state.
        # FAR's converter may have projected that command and stopped at its
        # adjusted point; publishing the same replacement waypoint once is
        # therefore meaningful and re-arms the bridge.  Without this, a
        # replan that happens to choose the same first point is suppressed by
        # the transport dedupe and the controller remains stopped forever.
        force_republish = navigation.get('motion_event') is not None
        if waypoint is not None and (waypoint != self.last_waypoint or force_republish):
            values = [float(value) for value in waypoint]
            if len(values) != 3 or not np.isfinite(values).all():
                raise ValueError('navigation_returned_invalid_pose2d')
            self.waypoint.publish(Pose2D(x=values[0], y=values[1], theta=values[2]))
            self.last_waypoint = waypoint
            self.log.event('waypoint_published', waypoint=waypoint, progress=navigation.get('progress'),
                           evidence_need=navigation.get('evidence_need'), ros_topic='/way_point_with_heading',
                           observation_id=scene.observation_id, image_stamp=scene.image_stamp)
        if self.future is None:
            position = np.array(self.latest_pose[1]) if self.latest_pose else None
            last = episode.last_observation_position
            moved = position is not None and last is not None and np.linalg.norm(position[:2]-last[:2]) >= self.config['observation_translation_m']
            if episode.observations == 0 or moved:
                frame = self.capture()
                if frame is not None:
                    self.future_kind, self.future_epoch = 'perception_geometry_memory', self.epoch
                    self.future = self.executor_pool.submit(self.runtime.observe, episode, frame)

    def publish_final(self, result, navigation, scene) -> bool:
        episode = self.episode
        task_type = episode.task['task_type']
        if task_type == 'numerical' and isinstance(result.value, int):
            self.numerical.publish(Int32(data=result.value))
            self.finish('numerical_output', True, result, navigation, scene)
            return True
        if task_type == 'object_reference' and len(result.object_ids) == 1:
            record = scene.records[result.object_ids[0]]
            if record.center is None or record.bbox is None:
                return False
            if not np.isfinite(np.r_[record.center, record.bbox]).all() or np.any(record.bbox <= 0):
                return False
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns, marker.id = 'cmu_selected_object', 0
            marker.type, marker.action = Marker.CUBE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, record.center)
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = map(float, record.bbox)
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 1.0, 0.2, 0.6
            marker.text = max(record.class_scores, key=record.class_scores.get)
            self.marker.publish(marker)
            self.finish('object_reference_output', True, result, navigation, scene)
            return True
        if task_type == 'instruction' and navigation.get('complete', False):
            self.finish('instruction_trajectory_complete', self.last_waypoint is not None, result, navigation, scene)
            return True
        return False

    def finish(self, decision: str, published: bool, result, navigation, scene) -> None:
        with self.episode.lock:
            self.episode.terminal = True
        # Use the exact scene that produced this decision, even if the worker
        # completed another observation between planning and publication.
        self.log.write('final_object_store.json', scene.snapshot)
        summary = {'question': self.question, 'task_type': self.episode.task['task_type'],
                   'root_decision': decision, 'numerical_answer': result.value,
                   'object_ids': result.object_ids, 'ros_published': published,
                   'failure_stage': None, 'query_result': asdict(result), 'navigation': navigation,
                   'observations': scene.observations,
                   'last_consumed_observation_id': scene.observation_id,
                   'last_consumed_image_stamp': scene.image_stamp,
                   'unconfirmed_region_count': len(scene.proposals),
                   'elapsed_seconds': time.monotonic()-self.started,
                   'run_dir': str(self.log.directory), 'evaluator_acceptance': 'unproven'}
        self.log.write('summary.json', summary)
        self.log.event('episode_finished', **summary)
        self.get_logger().info(f'Episode output: {decision}; run={self.log.directory}')

    def record_failure(self, stage: str, exc: Exception) -> None:
        self.failure = f'{type(exc).__name__}: {exc}'
        if self.episode is not None:
            with self.episode.lock:
                self.episode.terminal = True
        self.get_logger().error(f'{stage}: {self.failure}')
        if self.log:
            self.log.event('runtime_failure', stage=stage, error=self.failure, traceback=traceback.format_exc())
            self.log.write('summary.json', {
                'question': self.question,
                'task_type': self.episode.task['task_type'] if self.episode is not None else None,
                'root_decision': 'runtime_failure',
                'numerical_answer': None,
                'object_ids': [],
                'ros_published': False,
                'failure_stage': stage,
                'error': self.failure,
                'elapsed_seconds': (max(0.0, time.monotonic() - self.started)
                                    if self.started else None),
                'run_dir': str(self.log.directory),
                'evaluator_acceptance': 'unproven',
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = RebuildNode(load_config(args.config))
    try:
        rclpy.spin(node)
    finally:
        node.executor_pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
