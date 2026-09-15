#!/usr/bin/env python3
"""Inventory local inputs for the external validation sequence in 1.txt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


_SCENE_MEMORY_REQUIRED_TOPICS = {
    "/camera/image": "sensor_msgs/msg/Image",
    "/sensor_scan": "sensor_msgs/msg/PointCloud2",
    "/registered_scan": "sensor_msgs/msg/PointCloud2",
    "/state_estimation": "nav_msgs/msg/Odometry",
}


def _sqlite_topics(path: Path) -> tuple[dict[str, dict[str, object]], str | None]:
    """Read ROS 2 sqlite metadata without requiring a ROS installation."""
    if path.suffix.lower() != ".db3":
        return {}, None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT topics.name, topics.type, COUNT(messages.id) "
                "FROM topics LEFT JOIN messages "
                "ON messages.topic_id = topics.id "
                "GROUP BY topics.id ORDER BY topics.name"
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return {
        str(name): {"type": str(message_type), "count": int(count)}
        for name, message_type, count in rows
    }, None


def _bag_contract(path: Path) -> dict[str, object]:
    topics, error = _sqlite_topics(path)
    checks = {
        name: bool(
            name in topics
            and topics[name]["type"] == message_type
            and topics[name]["count"] > 0
        )
        for name, message_type in _SCENE_MEMORY_REQUIRED_TOPICS.items()
    }
    return {
        "path": str(path),
        "topics": topics,
        "inspection_error": error,
        "required_topic_checks": checks,
        "production_scene_memory_replay_ready": bool(
            not error and all(checks.values())
        ),
    }


def audit(workspace: Path):
    unity_root = workspace / "Unity_environment_models"
    vla_root = workspace / "VLA-3D_dataset" / "Unity"
    unity_players = tuple(sorted(
        path for path in unity_root.glob("*/environment/Model.x86_64")
        if path.is_file()
    ))
    vla_scenes = tuple(sorted(
        path.name for path in vla_root.iterdir()
        if path.is_dir()
        and (path / f"{path.name}_object_result.csv").is_file()
    )) if vla_root.is_dir() else ()
    bag_files = set()
    for pattern in ("*.bag", "*.db3", "*.mcap"):
        bag_files.update(
            path for path in workspace.rglob(pattern) if path.is_file()
        )
    for path in workspace.rglob("metadata.yaml"):
        if any(
            token in str(parent).lower()
            for parent in (path.parent, *path.parents[:3])
            for token in ("bag", "rosbag")
        ):
            bag_files.add(path)
    bag_contracts = tuple(_bag_contract(path) for path in sorted(bag_files))
    return {
        "schema": "scnav-validation-input-audit-v2",
        "workspace": str(workspace),
        "unity_player_count": len(unity_players),
        "unity_scenes": [path.parents[1].name for path in unity_players],
        "vla3d_scene_count": len(vla_scenes),
        "vla3d_scenes": list(vla_scenes),
        "rosbag_file_count": len(bag_files),
        "rosbag_files": [str(path) for path in sorted(bag_files)],
        "official_sample_rosbag_available": bool(bag_files),
        "rosbag_contracts": list(bag_contracts),
        "production_scene_memory_rosbag_available": any(
            bool(value["production_scene_memory_replay_ready"])
            for value in bag_contracts
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: result[key] for key in (
            "unity_player_count",
            "vla3d_scene_count",
            "rosbag_file_count",
        )
    }))


if __name__ == "__main__":
    main()
