import importlib.util
from pathlib import Path
import sqlite3


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_validation_inputs.py"
_SPEC = importlib.util.spec_from_file_location("audit_validation_inputs", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _bag(path, topics):
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
            CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER);
        """)
        for topic_id, (name, message_type, count) in enumerate(topics, 1):
            connection.execute(
                "INSERT INTO topics VALUES (?, ?, ?)",
                (topic_id, name, message_type),
            )
            connection.executemany(
                "INSERT INTO messages(topic_id) VALUES (?)",
                ((topic_id,) for _ in range(count)),
            )
        connection.commit()
    finally:
        connection.close()


def test_lidar_imu_bag_is_not_scene_memory_replay_ready(tmp_path):
    bag = tmp_path / "base.db3"
    _bag(bag, (
        ("/imu/data", "sensor_msgs/msg/Imu", 2),
        ("/lidar/scan", "livox_ros_driver2/msg/CustomMsg", 2),
    ))
    result = _MODULE.audit(tmp_path)
    assert result["official_sample_rosbag_available"] is True
    assert result["production_scene_memory_rosbag_available"] is False
    assert result["rosbag_contracts"][0][
        "production_scene_memory_replay_ready"
    ] is False


def test_complete_nonempty_production_bag_is_scene_memory_replay_ready(tmp_path):
    bag = tmp_path / "scene_memory.db3"
    _bag(bag, (
        ("/camera/image", "sensor_msgs/msg/Image", 2),
        ("/sensor_scan", "sensor_msgs/msg/PointCloud2", 2),
        ("/registered_scan", "sensor_msgs/msg/PointCloud2", 2),
        ("/state_estimation", "nav_msgs/msg/Odometry", 2),
    ))
    result = _MODULE.audit(tmp_path)
    assert result["production_scene_memory_rosbag_available"] is True
    contract = result["rosbag_contracts"][0]
    assert contract["required_topic_checks"] == {
        "/camera/image": True,
        "/sensor_scan": True,
        "/registered_scan": True,
        "/state_estimation": True,
    }


def test_empty_sensor_scan_is_not_replay_ready(tmp_path):
    bag = tmp_path / "empty_sensor_scan.db3"
    _bag(bag, (
        ("/camera/image", "sensor_msgs/msg/Image", 2),
        ("/sensor_scan", "sensor_msgs/msg/PointCloud2", 0),
        ("/registered_scan", "sensor_msgs/msg/PointCloud2", 2),
        ("/state_estimation", "nav_msgs/msg/Odometry", 2),
    ))
    result = _MODULE.audit(tmp_path)
    assert result["production_scene_memory_rosbag_available"] is False
    assert result["rosbag_contracts"][0]["required_topic_checks"][
        "/sensor_scan"
    ] is False
