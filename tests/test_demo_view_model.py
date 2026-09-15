"""Unit tests for the read-only Mission Control view-model contract."""

from __future__ import annotations

import importlib.util
from array import array
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import types
import unittest
import threading
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "ai_module" / "tools"
FIXTURES = TOOLS / "viewer_assets" / "fixtures"
sys.path.insert(0, str(TOOLS))


def _module(name):
    value = types.ModuleType(name)
    sys.modules[name] = value
    return value


def load_viewer():
    # Other ROS-node tests may have installed only a partial rclpy stub during
    # collection.  Fill the message packages independently so this test does
    # not depend on pytest module ordering.
    if "nav_msgs" not in sys.modules:
        nav = _module("nav_msgs")
        nav.msg = _module("nav_msgs.msg")
        nav.msg.Odometry = type("Odometry", (), {})
        nav.msg.Path = type("Path", (), {})
    if "geometry_msgs" not in sys.modules:
        geometry = _module("geometry_msgs")
        geometry.msg = _module("geometry_msgs.msg")
        geometry.msg.Pose2D = type("Pose2D", (), {})
    if "sensor_msgs" not in sys.modules:
        sensor = _module("sensor_msgs")
        sensor.msg = _module("sensor_msgs.msg")
        sensor.msg.CompressedImage = type("CompressedImage", (), {})
        sensor.msg.PointCloud2 = type("PointCloud2", (), {})
    if "std_msgs" not in sys.modules:
        std = _module("std_msgs")
        std.msg = _module("std_msgs.msg")
        std.msg.String = type("String", (), {})
    if "visualization_msgs" not in sys.modules:
        visualization = _module("visualization_msgs")
        visualization.msg = _module("visualization_msgs.msg")
        visualization.msg.Marker = type("Marker", (), {})
    if "rclpy.executors" not in sys.modules:
        executors = _module("rclpy.executors")
        executors.SingleThreadedExecutor = type("SingleThreadedExecutor", (), {})
    if "rclpy.node" not in sys.modules:
        node = _module("rclpy.node")
        node.Node = type("Node", (), {})
    if "rclpy.qos" not in sys.modules:
        qos = _module("rclpy.qos")
        qos.DurabilityPolicy = type("DurabilityPolicy", (), {"VOLATILE": 0})
        qos.HistoryPolicy = type("HistoryPolicy", (), {"KEEP_LAST": 0})
        qos.ReliabilityPolicy = type("ReliabilityPolicy", (), {"RELIABLE": 0})
        qos.QoSProfile = type("QoSProfile", (), {"__init__": lambda self, **kwargs: None})
    if "rclpy" not in sys.modules:
        geometry = _module("geometry_msgs")
        geometry.msg = _module("geometry_msgs.msg")
        geometry.msg.Pose2D = type("Pose2D", (), {})
        nav = _module("nav_msgs")
        nav.msg = _module("nav_msgs.msg")
        nav.msg.Odometry = type("Odometry", (), {})
        nav.msg.Path = type("Path", (), {})
        sensor = _module("sensor_msgs")
        sensor.msg = _module("sensor_msgs.msg")
        sensor.msg.CompressedImage = type("CompressedImage", (), {})
        sensor.msg.PointCloud2 = type("PointCloud2", (), {})
        std = _module("std_msgs")
        std.msg = _module("std_msgs.msg")
        std.msg.String = type("String", (), {})
        visualization = _module("visualization_msgs")
        visualization.msg = _module("visualization_msgs.msg")
        visualization.msg.Marker = type("Marker", (), {})
        rclpy = _module("rclpy")
        rclpy.executors = _module("rclpy.executors")
        rclpy.executors.SingleThreadedExecutor = type("SingleThreadedExecutor", (), {})
        rclpy.node = _module("rclpy.node")
        rclpy.node.Node = type("Node", (), {})
        rclpy.qos = _module("rclpy.qos")
        rclpy.qos.DurabilityPolicy = type("DurabilityPolicy", (), {"VOLATILE": 0})
        rclpy.qos.HistoryPolicy = type("HistoryPolicy", (), {"KEEP_LAST": 0})
        rclpy.qos.ReliabilityPolicy = type("ReliabilityPolicy", (), {"RELIABLE": 0})
        rclpy.qos.QoSProfile = type("QoSProfile", (), {"__init__": lambda self, **kwargs: None})
    spec = importlib.util.spec_from_file_location("mission_control_web_viewer", TOOLS / "web_viewer.py")
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load web_viewer")
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    return viewer


VIEWER = load_viewer()
TOP_LEVEL = {
    "meta", "task", "phase", "perception", "spatial", "decision",
    "evidence", "execution", "sensors", "result", "timeline",
}


def build(demo, **overrides):
    args = {
        "presentation": {"activityDetail": "Waiting", "healthLabel": "Data healthy", "staleStreams": []},
        "last_state": None,
        "path_points": [],
        "planner_path_points": [],
        "scan_points": [],
        "last_waypoint": None,
        "selected_marker": None,
        "map_bounds": [-5, 5, -5, 5],
        "timeline": [],
        "frame_age_sec": 0.1,
        "frame_rate_hz": 10.0,
        "scan_age_sec": 0.1,
        "scan_rate_hz": 5.0,
        "state_age_sec": 0.02,
        "state_rate_hz": 100.0,
        "demo_status_age_sec": 0.2,
        "front_crop": {
            "source_width": 1920,
            "source_height": 640,
            "output_width": 960,
            "output_height": 540,
        },
    }
    args.update(overrides)
    return VIEWER.DemoViewModel.build(demo, **args)


class DemoViewModelTest(unittest.TestCase):
    def test_sparse_telemetry_degrades_to_complete_safe_contract(self):
        value = build({"episode_id": "sparse", "seq": 1, "phase": "ready"})
        self.assertEqual(set(value), TOP_LEVEL)
        self.assertEqual(value["task"]["type"], "UNKNOWN")
        self.assertEqual(value["task"]["logic"]["expression"], "暂无结构化逻辑")
        self.assertFalse(value["result"]["submitted"])
        self.assertEqual([item["id"] for item in value["sensors"]["items"]], ["camera", "lidar", "pose", "tf", "ai"])
        self.assertEqual(value["sensors"]["items"][3]["status"], "not_reported")

    def test_count_argmin_and_stable_ids_come_from_structured_fields(self):
        demo = {
            "episode_id": "count", "seq": 4, "phase": "decide",
            "task_type": "numerical",
            "constraints": {
                "kind": "numerical",
                "query_ir": {
                    "aggregation": "count_distinct",
                    "quantifiers": [{"kind": "exists"}],
                    "selectors": [{"kind": "argmin_distance"}],
                },
            },
            "evidence": {"selector_candidates": [{"object_id": 7, "class": "chair", "distance_m": 1.2}]},
        }
        first = build(demo)
        changed_probability = json.loads(json.dumps(demo))
        changed_probability["evidence"]["selector_candidates"][0]["probability"] = 0.91
        second = build(changed_probability)
        self.assertEqual(first["task"]["type"], "COUNT")
        self.assertIn("COUNT_DISTINCT", first["task"]["logic"]["expression"])
        self.assertIn("ARGMIN_DISTANCE", first["task"]["logic"]["expression"])
        self.assertEqual(first["evidence"]["selector_candidates"][0]["id"], second["evidence"]["selector_candidates"][0]["id"])

    def test_ordinary_reject_is_continuing_search_not_system_error(self):
        value = build({"episode_id": "reject", "seq": 3, "phase": "decide", "action": "REJECT"})
        self.assertEqual(value["decision"]["action_label"], "拒绝当前候选，继续搜索")
        self.assertFalse(value["phase"]["is_error"])
        continuing = build({"episode_id": "reject", "seq": 4, "phase": "error", "action": "REJECT"})
        self.assertEqual(continuing["phase"]["label"], "补充证据")
        self.assertFalse(continuing["phase"]["is_error"])
        terminal = build({
            "episode_id": "reject", "seq": 5, "phase": "error",
            "action": "REJECT", "terminal": True,
        })
        self.assertEqual(terminal["phase"]["label"], "任务完成")
        self.assertFalse(terminal["phase"]["is_error"])
        self.assertEqual(terminal["result"]["commit_class"], "no submission")

    def test_current_commit_tiers_and_terminal_metadata_are_preserved(self):
        from viewer_presentation import normalize_demo_status

        for tier, expected in (
            ("STRICT_PHYSICAL", "strict physical"),
            ("BEST_EFFORT_PHYSICAL", "best-effort physical"),
            ("PANORAMA_FALLBACK", "panorama fallback"),
        ):
            normalized = normalize_demo_status(
                None,
                {
                    "schema_version": "scnav_demo_status_v2",
                    "episode_id": tier,
                    "seq": 1,
                    "phase": "success",
                    "commit_tier": tier,
                    "stop_reason": "deadline",
                    "answer_source": "physical",
                    "navigation_outcome": "done",
                    "result": {"kind": "numerical", "answer": 2},
                },
                received_wall_time_s=1.0,
            )
            value = build(normalized)
            self.assertEqual(value["result"]["commit_class"], expected)
            self.assertEqual(value["result"]["stop_reason"], "deadline")
            self.assertEqual(normalized["navigation_outcome"], "done")
            self.assertFalse(value["phase"]["is_error"])

    def test_terminal_episode_rejects_later_same_episode_updates(self):
        shared = VIEWER.SharedState()
        message = type("Message", (), {})()
        message.data = json.dumps({
            "schema_version": "scnav_demo_status_v2",
            "episode_id": "freeze",
            "seq": 9,
            "phase": "success",
            "elapsed_time_s": 12.0,
            "result": {"kind": "numerical", "answer": 3},
            "terminal": True,
        })
        shared.update_demo_status(message)
        frozen = shared.snapshot()["demo_view_model"]
        message.data = json.dumps({
            "schema_version": "scnav_demo_status_v2",
            "episode_id": "freeze",
            "seq": 10,
            "phase": "error",
            "terminal": True,
        })
        shared.update_demo_status(message)
        after = shared.snapshot()["demo_view_model"]
        self.assertEqual(frozen, after)
        self.assertEqual(after["result"]["display"], "3")

    def test_evidence_frame_is_pinned_by_exact_ros_source_stamp(self):
        shared = VIEWER.SharedState()
        stamp = type("Stamp", (), {"sec": 12, "nanosec": 345})()
        header = type("Header", (), {"stamp": stamp})()
        frame = VIEWER._placeholder_jpeg()
        message = type("Frame", (), {
            "data": array("B", frame),
            "header": header,
        })()
        shared.update_frame(message)
        status = type("Message", (), {})()
        status.data = json.dumps({
            "schema_version": "scnav_demo_status_v2",
            "episode_id": "aligned",
            "seq": 1,
            "phase": "perceive",
            "perception": {
                "image_width": 960,
                "image_height": 540,
                "view_type": "pano",
                "source_stamp_ns": 12_000_000_345,
                "detections": [],
            },
        })
        shared.update_demo_status(status)
        self.assertEqual(shared.evidence_source_stamp_ns, 12_000_000_345)
        self.assertEqual(shared.evidence_frame_pano, frame)
        self.assertTrue(
            shared.snapshot()["demo_view_model"]["perception"]
            ["evidence_frame_available"]
        )

    def test_perception_normalizes_pixel_boxes_and_falls_back_to_english_label(self):
        value = build({
            "episode_id": "detections",
            "seq": 2,
            "phase": "perceive",
            "perception": {
                "image_width": 1280,
                "image_height": 720,
                "view_type": "front",
                "detections": [{
                    "id": "chair-7",
                    "label": "chair",
                    "score": 0.91,
                    "bbox_xyxy": [320, 180, 640, 540],
                    "role": "target",
                    "state": "confirmed",
                    "track_id": 7,
                }],
            },
        })
        self.assertEqual(value["perception"]["image_width"], 1280)
        self.assertEqual(value["perception"]["image_height"], 720)
        self.assertEqual(value["perception"]["view_type"], "front")
        detection = value["perception"]["detections"][0]
        self.assertEqual(detection["label_zh"], "chair")
        self.assertEqual(detection["bbox_xyxy"], [320.0, 180.0, 640.0, 540.0])
        self.assertEqual(detection["track_id"], "7")

    def test_object_fit_contain_mapping_stays_aligned_after_resize(self):
        first = VIEWER.map_bbox_to_contain(
            [320, 180, 640, 360],
            image_width=1280,
            image_height=720,
            container_width=800,
            container_height=600,
        )
        second = VIEWER.map_bbox_to_contain(
            [320, 180, 640, 360],
            image_width=1280,
            image_height=720,
            container_width=400,
            container_height=300,
        )
        self.assertEqual(first, {"x": 200.0, "y": 187.5, "width": 200.0, "height": 112.5})
        self.assertEqual(second, {"x": 100.0, "y": 93.75, "width": 100.0, "height": 56.25})

    def test_panorama_boxes_are_clipped_and_scaled_into_front_crop(self):
        value = build({
            "episode_id": "crop", "seq": 3, "phase": "perceive",
            "perception": {
                "image_width": 1920,
                "image_height": 640,
                "view_type": "pano",
                "source_stamp_ns": 42,
                "detections": [
                    {"id": "center", "label": "chair", "bbox_xyxy": [800, 200, 1120, 500]},
                    {"id": "outside", "label": "table", "bbox_xyxy": [0, 0, 100, 100]},
                ],
            },
        }, front_crop={
            "source_width": 1920,
            "source_height": 640,
            "crop_left": 693,
            "crop_top": 50,
            "crop_width": 533,
            "crop_height": 299,
            "output_width": 960,
            "output_height": 540,
        })
        perception = value["perception"]
        self.assertEqual(set(perception["views"]), {"front", "pano"})
        self.assertEqual(perception["source_stamp_ns"], 42)
        self.assertEqual(len(perception["views"]["front"]["detections"]), 1)
        front_box = perception["views"]["front"]["detections"][0]["bbox_xyxy"]
        self.assertAlmostEqual(front_box[0], (800 - 693) * 960 / 533, places=3)
        self.assertEqual(front_box[3], 540.0)


class FixtureContractTest(unittest.TestCase):
    def test_all_frontend_fixtures_are_complete_and_stable(self):
        expected = {
            "count", "argmin", "object-marker", "navigation", "sensor-error",
            "final-result", "camera-empty", "camera-single", "camera-multi",
            "camera-resize", "camera-label-fallback",
        }
        self.assertEqual({path.stem for path in FIXTURES.glob("*.json")}, expected)
        for path in FIXTURES.glob("*.json"):
            with self.subTest(fixture=path.stem):
                model = json.loads(path.read_text(encoding="utf-8"))["demo_view_model"]
                self.assertEqual(set(model), TOP_LEVEL)
                for section in ("perception", "spatial", "decision", "evidence", "sensors", "result", "timeline"):
                    self.assertIsNotNone(model[section])
                for collection in (
                    model["perception"].get("detections", []),
                    model["spatial"].get("objects", []),
                    model["decision"].get("evidence_status", []),
                    model["timeline"],
                ):
                    self.assertTrue(all(item.get("id") for item in collection))
                perception = model["perception"]
                self.assertGreater(perception["image_width"], 0)
                self.assertGreater(perception["image_height"], 0)
                self.assertIn(perception["view_type"], {"front", "pano"})
                required = {"id", "label", "label_zh", "confidence", "bbox_xyxy", "role", "state"}
                for detection in perception["detections"]:
                    self.assertTrue(required.issubset(detection))
                    self.assertEqual(len(detection["bbox_xyxy"]), 4)

    def test_camera_fixtures_cover_empty_single_multi_resize_and_label_fallback(self):
        def perception(name):
            payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
            return payload["demo_view_model"]["perception"]

        self.assertEqual(perception("camera-empty")["detections"], [])
        self.assertEqual(len(perception("camera-single")["detections"]), 1)
        multi = perception("camera-multi")["detections"]
        self.assertEqual({item["role"] for item in multi}, {"target", "anchor", "object"})
        self.assertTrue({"confirmed", "provisional", "rejected", "error"}.issubset(
            {item["state"] for item in multi}
        ))
        resize = perception("camera-resize")
        self.assertEqual(resize["detections"][0]["bbox_xyxy"], [320, 180, 640, 360])
        fallback = perception("camera-label-fallback")["detections"][0]
        self.assertEqual(fallback["label_zh"], fallback["label"])


class StaticPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        preview_path = ROOT / "demo" / "preview_mission_control.py"
        spec = importlib.util.spec_from_file_location("preview_mission_control", preview_path)
        if spec is None or spec.loader is None:
            raise AssertionError("cannot load preview server")
        cls.preview = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.preview)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), cls.preview.PreviewHandler)
        cls.server.fixture_name = "count"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_preview_needs_no_ros_and_serves_fixture_page(self):
        self.assertNotIn("rclpy", (ROOT / "demo" / "preview_mission_control.py").read_text(encoding="utf-8"))
        for path in ("/demo?fixture=count", "/demo.css", "/demo.js", "/demo-fixtures/count.json", "/snapshot_front.jpg"):
            with urlopen(self.base + path, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())


if __name__ == "__main__":
    unittest.main()
