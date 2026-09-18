"""Real ROS Jazzy DDS/TF/actions with synthetic sensors and a dummy Nav2 server.

The fixture runs in a separate process. No physical robot, motion controller,
RealSense hardware, model weights or GPU is involved. Use an isolated ROS domain.
"""
from __future__ import annotations
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from go2.config import load_config
from go2.ros_node import Go2Node


def fixture_process(ready, commands):
    rclpy.init()
    node=Node('go2_synthetic_sensor_and_nav2_fixture')
    group=ReentrantCallbackGroup()
    def execute(handle):
        x=handle.request.pose.pose.position.x
        result=NavigateToPose.Result()
        if x == 99:
            handle.abort()
            result.error_code=42
            result.error_msg='intentional_test_abort'
            return result
        started=time.monotonic()
        while rclpy.ok():
            if handle.is_cancel_requested:
                handle.canceled()
                return result
            if x >= 0 and time.monotonic()-started > .15:
                handle.succeed()
                return result
            time.sleep(.01)
        return result
    server=ActionServer(node,NavigateToPose,'/go2_test/navigate_to_pose',
        execute_callback=execute,goal_callback=lambda request: GoalResponse.ACCEPT,
        cancel_callback=lambda handle: CancelResponse.ACCEPT,callback_group=group)
    publishers={name:node.create_publisher(cls,'/go2_test/'+name,qos_profile_sensor_data)
                for name,cls in [('rgb',Image),('depth',Image),('info',CameraInfo)]}
    map_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
    full_pub=node.create_publisher(OccupancyGrid,'/go2_test/costmap',map_qos)
    patch_pub=node.create_publisher(OccupancyGridUpdate,'/go2_test/costmap_updates',10)
    broadcaster=TransformBroadcaster(node)
    static=StaticTransformBroadcaster(node)
    mount=TransformStamped()
    mount.header.frame_id='base_link';mount.child_frame_id='camera_color_optical_frame'
    mount.header.stamp=node.get_clock().now().to_msg()
    mount.transform.translation.x=.2;mount.transform.translation.z=.3
    mount.transform.rotation.x=-.5;mount.transform.rotation.y=.5
    mount.transform.rotation.z=-.5;mount.transform.rotation.w=.5
    static.sendTransform(mount)
    mode=['normal']
    def publish_map():
        message=OccupancyGrid();message.header.frame_id='map'
        message.header.stamp=node.get_clock().now().to_msg()
        message.info.resolution=.25;message.info.width=5;message.info.height=5
        message.info.origin.orientation.w=1.;message.data=[0]*25
        full_pub.publish(message)
    publish_map()
    def publish():
        try:
            while True:
                command=commands.get_nowait()
                if command in {'normal','paused','stale_depth'}: mode[0]=command
                elif command=='reset_map': publish_map()
                elif command=='patch':
                    patch=OccupancyGridUpdate();patch.header.frame_id='map'
                    patch.header.stamp=node.get_clock().now().to_msg()
                    patch.x=2;patch.y=2;patch.width=1;patch.height=1;patch.data=[100]
                    patch_pub.publish(patch)
        except queue.Empty:
            pass
        stamp=node.get_clock().now().to_msg()
        transform=TransformStamped();transform.header.stamp=stamp
        transform.header.frame_id='map';transform.child_frame_id='base_link'
        transform.transform.translation.x=1.;transform.transform.translation.y=2.
        transform.transform.translation.z=.5;transform.transform.rotation.w=1.
        broadcaster.sendTransform(transform)
        if mode[0]=='paused': return
        rgb=Image();rgb.header.stamp=stamp;rgb.header.frame_id='camera_color_optical_frame'
        rgb.height=6;rgb.width=8;rgb.encoding='rgb8';rgb.step=24
        rgb.data=np.full((6,8,3),100,np.uint8).tobytes()
        depth=Image();depth.header.stamp=stamp;depth.header.frame_id=rgb.header.frame_id
        if mode[0]=='stale_depth':
            depth.header.stamp=node.get_clock().now().to_msg()
            depth.header.stamp.sec-=2
        depth.height=6;depth.width=8;depth.encoding='16UC1';depth.step=16
        depth.data=np.full((6,8),2000,dtype='<u2').tobytes()
        info=CameraInfo();info.header=rgb.header;info.width=8;info.height=6
        info.distortion_model='plumb_bob';info.d=[0.]*5
        info.k=[4.,0.,3.,0.,4.,2.,0.,0.,1.]
        info.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
        publishers['rgb'].publish(rgb);publishers['depth'].publish(depth);publishers['info'].publish(info)
    timer=node.create_timer(.03,publish,callback_group=group)
    executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
    ready.set()
    try:
        executor.spin()
    except ExternalShutdownException:
        pass
    finally:
        server.destroy(); node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class LiveInterfacesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        context=mp.get_context('spawn')
        cls.ready=context.Event();cls.commands=context.Queue()
        cls.fixture=context.Process(target=fixture_process,args=(cls.ready,cls.commands))
        cls.fixture.start()
        if not cls.ready.wait(15): raise RuntimeError('ROS fixture did not start')
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()
        cls.fixture.terminate();cls.fixture.join(5)
        if cls.fixture.is_alive(): cls.fixture.kill();cls.fixture.join()

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.config=load_config()
        self.config.update(rgb_topic='/go2_test/rgb',depth_topic='/go2_test/depth',
            camera_info_topic='/go2_test/info',costmap_topic='/go2_test/costmap',
            costmap_updates_topic='/go2_test/costmap_updates',nav2_action='/go2_test/navigate_to_pose',
            task_topic='/go2_test/task',status_topic='/go2_test/status',depth_stride=1,
            output_root=self.temporary.name,max_observation_age_seconds=.6)
        self.node=Go2Node(self.config)
        self.commands.put('normal');self.commands.put('reset_map')
        self.spin_until(lambda:self.node.transport.client.server_is_ready())

    def tearDown(self):
        self.node.transport.cancel()
        self.spin_until(lambda:not self.node.transport.busy)
        self.node.pool.shutdown(wait=True,cancel_futures=True)
        self.node.destroy_node();self.temporary.cleanup()
        self.commands.put('normal')

    def spin_until(self,predicate,timeout=10):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            rclpy.spin_once(self.node,timeout_sec=.02)
            value=predicate()
            if value: return value
        self.fail('Timed out waiting for real ROS interface condition')

    def spin_for(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(self.node,timeout_sec=.02)

    def test_01_rgbd_sync_image_time_tf_and_measured_geometry(self):
        frame=self.spin_until(self.node.capture)
        self.assertEqual(frame.registered_points_map.shape,(48,3))
        # Pixel (3,2) at optical depth 2m: base + camera offset + rotated ray.
        self.assertTrue(np.any(np.linalg.norm(frame.registered_points_map-[3.2,2.,.8],axis=1)<1e-5))
        self.node.update_pose()
        np.testing.assert_allclose(self.node.pose[:3,3],[1.,2.,.5])

    def test_02_incremental_costmap_changes_live_terrain(self):
        self.spin_until(lambda:self.node.costmap is not None and self.node.costmap.data[12]==0)
        self.node.update_terrain();previous=self.node.terrain_key
        self.commands.put('patch')
        self.spin_until(lambda:self.node.costmap.data[12]==100)
        self.node.update_terrain()
        self.assertGreater(self.node.terrain_key,previous)
        self.assertEqual(self.node.terrain[12,3],1.)

    def test_03_real_navigate_to_pose_success(self):
        self.node.transport.submit([1,2,math.pi/3])
        self.spin_until(lambda:self.node.transport.status=='succeeded')
        self.assertEqual(self.node.transport.last_success,(1.,2.,math.pi/3))

    def test_04_real_cancel_then_replace(self):
        self.node.transport.submit([-1,2,0])
        self.spin_until(lambda:self.node.transport.status=='active')
        self.node.transport.submit([2,3,0])
        self.spin_until(lambda:self.node.transport.status=='succeeded')
        self.assertEqual(self.node.transport.last_success,(2.,3.,0.))
        self.assertIsNone(self.node.transport.error)

    def test_05_real_action_abort(self):
        self.node.transport.submit([99,0,0])
        self.spin_until(lambda:self.node.transport.error)
        self.assertIn('42',self.node.transport.error)
        self.assertIsNone(self.node.transport.last_success)

    def test_06_explicit_cancellation_is_confirmed(self):
        self.node.transport.submit([-1,0,0])
        self.spin_until(lambda:self.node.transport.status=='active')
        self.node.transport.cancel()
        self.spin_until(lambda:not self.node.transport.busy)
        self.assertIsNone(self.node.transport.error)
        self.assertIsNone(self.node.transport.last_success)

    def test_07_stale_camera_is_not_captured(self):
        self.spin_until(self.node.capture)
        self.commands.put('paused');self.spin_for(1.)
        self.assertIsNone(self.node.capture())

    def run_preflight(self,timeout):
        path=Path(self.temporary.name)/'profile.json';path.write_text(json.dumps(self.config))
        return subprocess.run([sys.executable,'-m','go2.preflight','--config',str(path),
                              '--timeout',str(timeout)],capture_output=True,text=True,timeout=timeout+15)

    def test_08_preflight_over_real_dds(self):
        result=self.run_preflight(8)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('"read_only_preflight_passed": true',result.stdout)

    def test_09_preflight_does_not_accept_fresh_rgb_with_old_depth(self):
        self.commands.put('stale_depth');self.spin_for(.2)
        # A new DDS participant needs time to discover the existing publishers.
        # Assert sensor reception so a missing publisher cannot satisfy this test.
        result=self.run_preflight(8)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        report=json.loads(result.stdout)
        for key in ('rgb_topic', 'depth_topic', 'camera_info_topic'):
            self.assertTrue(report['checks'][key], result.stdout+result.stderr)
        self.assertFalse(report['checks']['rgbd_timing'])


if __name__=='__main__':
    if os.environ.get('ROS_DOMAIN_ID') != '197' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise SystemExit('Use ROS_DOMAIN_ID=197 ROS_LOCALHOST_ONLY=1 for this synthetic test')
    unittest.main()
