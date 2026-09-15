#!/usr/bin/env python3
"""Build a source-frame-pinned SceneMemory replay bag from a real capture.

The fixture does not synthesize perception evidence.  It copies one exact RGB
frame, its nearest sensor/map geometry, and a bounded real odometry/TF window
from a production capture.  A deterministic evaluator question precedes the
sensor window so the production node clears episode state before the only
eligible RGB/scan pair arrives.

Run this script in a ROS 2 image that provides ``rosbag2_py`` and the message
packages used by the source bag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_TOPICS = {
    "/camera/image": "sensor_msgs/msg/Image",
    "/sensor_scan": "sensor_msgs/msg/PointCloud2",
    "/registered_scan": "sensor_msgs/msg/PointCloud2",
    "/state_estimation": "nav_msgs/msg/Odometry",
}
OPTIONAL_NEAREST_TOPICS = (
    "/terrain_map",
    "/terrain_map_ext",
)
WINDOW_TOPICS = ("/state_estimation",)
STATIC_TOPICS: tuple[str, ...] = ()
QUESTION_TOPIC = "/challenge_question"
CRITICAL_REPEAT_TOPICS = frozenset({
    "/camera/image",
    "/sensor_scan",
})
CRITICAL_REPETITIONS = 5
SOURCE_STAMP_TOPICS = frozenset({
    "/camera/image",
    "/sensor_scan",
    "/registered_scan",
    "/state_estimation",
    *OPTIONAL_NEAREST_TOPICS,
})


def nearest_entry(
    entries: Sequence[tuple[int, int]], target_timestamp_ns: int
) -> tuple[int, int]:
    """Return ``(ordinal, transport stamp)`` with deterministic tie-breaking."""

    if not entries:
        raise ValueError("cannot select from an empty topic stream")
    target = int(target_timestamp_ns)
    return min(entries, key=lambda item: (abs(item[1] - target), item[1], item[0]))


def bounded_window(
    entries: Sequence[tuple[int, int]],
    target_timestamp_ns: int,
    *,
    radius_ns: int,
    maximum: int,
) -> tuple[tuple[int, int], ...]:
    """Select a chronological real-message window around one target stamp."""

    if radius_ns < 0 or maximum <= 0:
        raise ValueError("window radius must be nonnegative and maximum positive")
    target = int(target_timestamp_ns)
    within = [
        item for item in entries if abs(int(item[1]) - target) <= int(radius_ns)
    ]
    if not within:
        within = [nearest_entry(entries, target)]
    if len(within) > maximum:
        within = sorted(
            within,
            key=lambda item: (abs(item[1] - target), item[1], item[0]),
        )[:maximum]
    return tuple(sorted(within, key=lambda item: (item[1], item[0])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_stamp_ns(serialized: bytes, message_type: str) -> int:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    message = deserialize_message(serialized, get_message(message_type))
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        raise ValueError(f"{message_type} has no header stamp")
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value < 0:
        raise ValueError("source stamp must be nonnegative")
    return value


def _topic_inventory(reader) -> tuple[dict[str, object], dict[str, str]]:
    metadata = {item.name: item for item in reader.get_all_topics_and_types()}
    types = {name: item.type for name, item in metadata.items()}
    missing = {
        name: expected
        for name, expected in REQUIRED_TOPICS.items()
        if types.get(name) != expected
    }
    if missing:
        raise ValueError(f"source bag lacks required production topics: {missing}")
    return metadata, types


def build_fixture(
    source_bag: Path,
    output_dir: Path,
    *,
    image_ordinal: int,
    question: str,
    sync_tolerance_s: float,
) -> dict[str, object]:
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from std_msgs.msg import String

    source_bag = source_bag.resolve()
    output_dir = output_dir.resolve()
    if not source_bag.is_file():
        raise FileNotFoundError(source_bag)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {output_dir}")
    if image_ordinal < 0 or not question.strip():
        raise ValueError("image ordinal must be nonnegative and question nonempty")
    if not math.isfinite(sync_tolerance_s) or sync_tolerance_s <= 0.0:
        raise ValueError("sync tolerance must be finite and positive")

    storage = rosbag2_py.StorageOptions(
        uri=str(source_bag), storage_id="sqlite3"
    )
    converter = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    topic_metadata, topic_types = _topic_inventory(reader)

    entries: dict[str, list[tuple[int, int]]] = {}
    source_entries: dict[str, list[tuple[int, int]]] = {}
    transport_by_key: dict[tuple[str, int], int] = {}
    ordinals: dict[str, int] = {}
    while reader.has_next():
        topic, data, transport_stamp = reader.read_next()
        ordinal = ordinals.get(topic, 0)
        ordinals[topic] = ordinal + 1
        transport = int(transport_stamp)
        entries.setdefault(topic, []).append((ordinal, transport))
        transport_by_key[(topic, ordinal)] = transport
        if topic in SOURCE_STAMP_TOPICS:
            source_entries.setdefault(topic, []).append((
                ordinal,
                _source_stamp_ns(data, topic_types[topic]),
            ))

    images = source_entries.get("/camera/image", [])
    if image_ordinal >= len(images):
        raise ValueError(
            f"image ordinal {image_ordinal} outside stream of {len(images)}"
        )
    target_image = images[image_ordinal]
    target_source_ns = target_image[1]
    target_transport_ns = transport_by_key[("/camera/image", target_image[0])]
    selected_keys: set[tuple[str, int]] = {("/camera/image", target_image[0])}
    selected_summary: dict[str, list[dict[str, int]]] = {
        "/camera/image": [{
            "ordinal": target_image[0],
            "transport_stamp_ns": target_transport_ns,
            "source_stamp_ns": target_source_ns,
        }]
    }

    for topic in ("/sensor_scan", "/registered_scan", *OPTIONAL_NEAREST_TOPICS):
        if not source_entries.get(topic):
            continue
        selected = nearest_entry(source_entries[topic], target_source_ns)
        transport = transport_by_key[(topic, selected[0])]
        selected_keys.add((topic, selected[0]))
        selected_summary[topic] = [{
            "ordinal": selected[0],
            "transport_stamp_ns": transport,
            "source_stamp_ns": selected[1],
        }]

    for topic in WINDOW_TOPICS:
        if not entries.get(topic):
            continue
        radius_ns = 400_000_000 if topic == "/state_estimation" else 500_000_000
        maximum = 160 if topic == "/state_estimation" else 240
        authority_entries = (
            source_entries[topic]
            if topic in source_entries
            else entries[topic]
        )
        authority_target = (
            target_source_ns if topic in source_entries else target_transport_ns
        )
        chosen = bounded_window(
            authority_entries, authority_target,
            radius_ns=radius_ns, maximum=maximum,
        )
        selected_keys.update((topic, ordinal) for ordinal, _ in chosen)
        selected_summary[topic] = [
            {
                "ordinal": ordinal,
                "transport_stamp_ns": transport_by_key[(topic, ordinal)],
                **({"source_stamp_ns": stamp} if topic in source_entries else {}),
            }
            for ordinal, stamp in chosen
        ]

    for topic in STATIC_TOPICS:
        chosen = tuple(entries.get(topic, ()))
        selected_keys.update((topic, ordinal) for ordinal, _ in chosen)
        selected_summary[topic] = [
            {"ordinal": ordinal, "transport_stamp_ns": stamp}
            for ordinal, stamp in chosen
        ]

    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    collected: list[tuple[str, bytes, int, int]] = []
    ordinals.clear()
    while reader.has_next():
        topic, data, transport_stamp = reader.read_next()
        ordinal = ordinals.get(topic, 0)
        ordinals[topic] = ordinal + 1
        if (topic, ordinal) in selected_keys:
            collected.append((topic, data, int(transport_stamp), ordinal))

    if len(collected) != len(selected_keys):
        raise RuntimeError(
            f"selected {len(selected_keys)} messages but recovered {len(collected)}"
        )
    payload_by_topic = {topic: data for topic, data, _, _ in collected}
    image_source_ns = _source_stamp_ns(
        payload_by_topic["/camera/image"], topic_types["/camera/image"]
    )
    scan_source_ns = _source_stamp_ns(
        payload_by_topic["/sensor_scan"], topic_types["/sensor_scan"]
    )
    sync_residual_s = abs(image_source_ns - scan_source_ns) / 1e9
    if sync_residual_s > sync_tolerance_s:
        raise ValueError(
            f"nearest RGB/scan source residual {sync_residual_s:.6f}s exceeds "
            f"{sync_tolerance_s:.6f}s"
        )

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_dir), storage_id="sqlite3"),
        converter,
    )
    used_topics = sorted({topic for topic, _, _, _ in collected})
    for topic in used_topics:
        writer.create_topic(topic_metadata[topic])
    writer.create_topic(rosbag2_py.TopicMetadata(
        id=0,
        name=QUESTION_TOPIC,
        type="std_msgs/msg/String",
        serialization_format="cdr",
    ))

    question_message = String()
    question_message.data = question.strip()
    question_transport_ns = 1_000_000_000
    writer.write(
        QUESTION_TOPIC, serialize_message(question_message), question_transport_ns
    )
    odometry_rows = [row for row in collected if row[0] == "/state_estimation"]
    if not odometry_rows:
        raise RuntimeError("fixture requires a real odometry window")
    first_odometry_transport_ns = min(row[2] for row in odometry_rows)
    sensor_start_ns = question_transport_ns + 2_000_000_000
    support_start_ns = question_transport_ns + 3_200_000_000
    pair_start_ns = question_transport_ns + 3_700_000_000
    output_rows: list[tuple[int, str, bytes, int]] = []
    for topic, data, original_stamp, ordinal in collected:
        if topic == "/state_estimation":
            output_rows.append((
                sensor_start_ns + (original_stamp - first_odometry_transport_ns),
                topic, data, ordinal,
            ))
            continue
        if topic == "/sensor_scan":
            output_stamp = pair_start_ns
        elif topic == "/camera/image":
            output_stamp = pair_start_ns + 50_000_000
        else:
            support_order = {
                "/registered_scan": 0,
                "/terrain_map": 1,
                "/terrain_map_ext": 2,
            }.get(topic, 9)
            output_stamp = support_start_ns + support_order * 50_000_000
        repetitions = (
            CRITICAL_REPETITIONS if topic in CRITICAL_REPEAT_TOPICS else 1
        )
        for repeat_index in range(repetitions):
            output_rows.append((
                output_stamp + repeat_index * 250_000_000,
                topic,
                data,
                ordinal,
            ))
    for output_stamp, topic, data, ordinal in sorted(
        output_rows, key=lambda item: (item[0], item[1], item[3])
    ):
        writer.write(topic, data, output_stamp)
    del writer

    databases = tuple(sorted(output_dir.glob("*.db3")))
    if len(databases) != 1:
        raise RuntimeError(f"fixture writer produced unexpected files: {databases}")
    manifest = {
        "schema": "scnav-scene-memory-pinned-replay-v1",
        "source_bag": str(source_bag),
        "source_bag_sha256": _sha256(source_bag),
        "fixture_bag": str(databases[0]),
        "fixture_bag_sha256": _sha256(databases[0]),
        "question": question.strip(),
        "question_transport_stamp_ns": question_transport_ns,
        "sensor_start_transport_stamp_ns": sensor_start_ns,
        "support_start_transport_stamp_ns": support_start_ns,
        "pair_start_transport_stamp_ns": pair_start_ns,
        "target_image_ordinal": image_ordinal,
        "target_image_source_stamp_ns": image_source_ns,
        "target_scan_source_stamp_ns": scan_source_ns,
        "source_sync_residual_s": sync_residual_s,
        "selected_message_count": len(collected),
        "output_message_count": len(output_rows) + 1,
        "critical_payload_repetitions": CRITICAL_REPETITIONS,
        "repeated_payloads_preserve_source_stamp_and_cdr": True,
        "selected_topics": selected_summary,
        "production_sensor_payloads_synthetic": False,
        "question_injected_by_fixture": True,
    }
    manifest_path = output_dir / "scene_memory_pinned_replay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ordinal", type=int, default=50)
    parser.add_argument("--question", default="Find the TV.")
    parser.add_argument("--sync-tolerance-s", type=float, default=0.15)
    args = parser.parse_args()
    result = build_fixture(
        args.source_bag,
        args.output_dir,
        image_ordinal=args.image_ordinal,
        question=args.question,
        sync_tolerance_s=args.sync_tolerance_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
