#!/usr/bin/env python3
"""Capture and verify one replayable production SceneMemory ROS 2 bag."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sqlite3
import subprocess
import time


REQUIRED_TOPICS = {
    "/camera/image": "sensor_msgs/msg/Image",
    # Evaluation-mode grounding binds RGB to the sensor-frame scan.  The
    # registered map-frame scan is additional geometry evidence; it cannot be
    # substituted for this synchronized perception input.
    "/sensor_scan": "sensor_msgs/msg/PointCloud2",
    "/registered_scan": "sensor_msgs/msg/PointCloud2",
    "/state_estimation": "nav_msgs/msg/Odometry",
}
SUPPORT_TOPICS = (
    "/terrain_map",
    "/terrain_map_ext",
    "/tf",
    "/tf_static",
    "/challenge_question",
)


def inspect_sqlite(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        topics = {
            str(name): {
                "type": str(message_type),
                "count": int(count),
            }
            for name, message_type, count in connection.execute(
                "SELECT topics.name, topics.type, COUNT(messages.id) "
                "FROM topics LEFT JOIN messages "
                "ON messages.topic_id = topics.id "
                "GROUP BY topics.id ORDER BY topics.name"
            )
        }
    finally:
        connection.close()
    checks = {
        topic: bool(
            topic in topics
            and topics[topic]["type"] == message_type
            and topics[topic]["count"] > 0
        )
        for topic, message_type in REQUIRED_TOPICS.items()
    }
    return {
        "sqlite_path": str(path.resolve()),
        "sqlite_integrity": integrity,
        "topics": topics,
        "required_topic_checks": checks,
        "production_scene_memory_replay_ready": bool(
            integrity == "ok" and all(checks.values())
        ),
    }


def capture(output_dir: Path, duration_s: float) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing capture: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    topics = (*REQUIRED_TOPICS, *SUPPORT_TOPICS)
    command = (
        "ros2", "bag", "record", "--storage", "sqlite3",
        "--output", str(output_dir), "--topics", *topics,
    )
    started = time.monotonic()
    process = subprocess.Popen(command)
    try:
        deadline = started + duration_s
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        return_code = process.wait(timeout=30.0)
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    databases = tuple(sorted(output_dir.glob("*.db3")))
    if return_code != 0 or len(databases) != 1:
        raise RuntimeError(
            f"ros2 bag record failed exit={return_code}, db3={databases}"
        )
    inspection = inspect_sqlite(databases[0])
    result = {
        "schema": "scnav-scene-memory-rosbag-capture-v1",
        "output_dir": str(output_dir.resolve()),
        "duration_requested_s": duration_s,
        "duration_elapsed_s": time.monotonic() - started,
        "record_command": list(command),
        **inspection,
    }
    manifest = output_dir / "scene_memory_capture_manifest.json"
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["production_scene_memory_replay_ready"]:
        raise RuntimeError(
            "capture completed but required SceneMemory topics were absent; "
            f"see {manifest}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0.0:
        raise ValueError("duration must be finite and positive")
    result = capture(args.output_dir.resolve(), float(args.duration_s))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
