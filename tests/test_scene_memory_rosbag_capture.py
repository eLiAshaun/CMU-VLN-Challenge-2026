import importlib.util
from pathlib import Path
import sqlite3


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/capture_scene_memory_rosbag.py"
)
SPEC = importlib.util.spec_from_file_location(
    "capture_scene_memory_rosbag", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _database(path, topics):
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
            CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT);
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER,
                data BLOB
            );
        """)
        for topic_id, (name, message_type, count) in enumerate(topics, 1):
            connection.execute(
                "INSERT INTO topics VALUES (?, ?, ?)",
                (topic_id, name, message_type),
            )
            connection.executemany(
                "INSERT INTO messages(topic_id, timestamp, data) "
                "VALUES (?, ?, ?)",
                ((topic_id, index, b"x") for index in range(count)),
            )
        connection.commit()
    finally:
        connection.close()


def test_inspection_requires_nonempty_rgb_sensor_geometry_and_pose(tmp_path):
    database = tmp_path / "capture.db3"
    _database(database, (
        ("/camera/image", "sensor_msgs/msg/Image", 2),
        ("/sensor_scan", "sensor_msgs/msg/PointCloud2", 2),
        ("/registered_scan", "sensor_msgs/msg/PointCloud2", 1),
        ("/state_estimation", "nav_msgs/msg/Odometry", 3),
    ))
    result = MODULE.inspect_sqlite(database)
    assert result["sqlite_integrity"] == "ok"
    assert result["production_scene_memory_replay_ready"] is True


def test_inspection_rejects_empty_or_wrong_type_required_topic(tmp_path):
    database = tmp_path / "capture.db3"
    _database(database, (
        ("/camera/image", "sensor_msgs/msg/CompressedImage", 2),
        ("/sensor_scan", "sensor_msgs/msg/PointCloud2", 0),
        ("/registered_scan", "sensor_msgs/msg/PointCloud2", 0),
        ("/state_estimation", "nav_msgs/msg/Odometry", 3),
    ))
    result = MODULE.inspect_sqlite(database)
    assert result["production_scene_memory_replay_ready"] is False
    assert result["required_topic_checks"] == {
        "/camera/image": False,
        "/sensor_scan": False,
        "/registered_scan": False,
        "/state_estimation": True,
    }
