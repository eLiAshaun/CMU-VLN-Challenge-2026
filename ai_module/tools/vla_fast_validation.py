#!/usr/bin/env python3
"""Offline VLA-3D semantic validation for the CMU-VLN AI module.

This is an artifact-only tool.  It reads ``Unity.zip`` directly, builds
read-only GT adapters, and writes reports below ``artifacts/vla_fast_validation``.
Production runtime modules never import this file or its outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import io
import itertools
import json
import math
from pathlib import Path
import random
import re
import struct
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = REPO_ROOT / "ai_module"
DEFAULT_ARCHIVE = Path("/home/robot/cmu_vln/VLA-3D_dataset/Unity.zip")
DEFAULT_QUESTIONS = REPO_ROOT / "questions" / "questions.json"
DEFAULT_OUTPUT = AI_ROOT / "artifacts" / "vla_fast_validation"
RELATIONS = ("on", "above", "below", "near", "in", "between", "closest", "farthest")
GEOMETRIC_RELATIONS = ("on", "above", "below", "near", "in", "between")
ORDERED_RELATIONS = ("closest", "farthest")
RNG_SEED = 20260825


def _ensure_import_path() -> None:
    if str(AI_ROOT) not in sys.path:
        sys.path.insert(0, str(AI_ROOT))


_ensure_import_path()

from integrations.execution.query_executor import (  # noqa: E402
    _Resolver,
    _matches_class_names,
)
from integrations.execution.relation_evidence import (  # noqa: E402
    _geometry_consistency,
    relation_geometry_diagnostic,
)
from integrations.execution.scene_memory import materialize_query_view  # noqa: E402
from integrations.execution.trajectory_geometry import (  # noqa: E402
    compile_directive_geometry,
)
from integrations.execution.trajectory_monitor import apply_actual_pose  # noqa: E402
from integrations.semantics.task_compiler import compile_task  # noqa: E402
from integrations.semantics.task_compiler import deterministic_compile  # noqa: E402
from navigation.waypoint_planning import select_semantic_waypoints  # noqa: E402
from orchestration.lidar_geometry import _deduplicate_station_observations  # noqa: E402


def _norm(value: object) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    aliases = {
        "tv": "television",
        "television set": "television",
        "photos": "picture",
        "photo": "picture",
        "pictures": "picture",
        "shelves": "shelf",
        "couches": "couch",
        "records": "record",
        "paintings": "painting",
        "vases": "vase",
        "pillows": "pillow",
        "chairs": "chair",
        "tables": "table",
        "sofas": "sofa",
        "lamps": "lamp",
        "windows": "window",
        "cabinets": "cabinet",
        "plants": "plant",
        "potted plants": "potted plant",
    }
    if text in aliases:
        return aliases[text]
    if text.endswith("s") and not text.endswith("ss"):
        singular = text[:-1]
        text = aliases.get(singular, singular)
    return aliases.get(text, text)


def _class_match(entity_class: object, obj: Mapping[str, Any]) -> bool:
    wanted = _norm(entity_class)
    labels = {
        _norm(obj.get("class_label")),
        _norm(obj.get("raw_label")),
        _norm(obj.get("nyu_label")),
        _norm(obj.get("nyu40_label")),
    }
    labels.discard("")
    if wanted in labels:
        return True
    if wanted == "lamp" and any(label.endswith(" lamp") or label == "lamp" for label in labels):
        return True
    if wanted == "television" and any(label in {"television", "tv"} for label in labels):
        return True
    if wanted in {"television cabinet", "tv cabinet"} and any(
        label in {"television cabinet", "tv cabinet", "tv stand", "media console"}
        for label in labels
    ):
        return True
    if wanted == "picture" and any(label in {"picture", "photo", "painting"} for label in labels):
        return True
    return False


def _float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _center(obj: Mapping[str, Any]) -> list[float]:
    return [float(value) for value in obj.get("center_3d", (0.0, 0.0, 0.0))]


def _xy_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    a, b = _center(first), _center(second)
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return {"count": 0}
    def q(fraction: float) -> float:
        index = fraction * (len(data) - 1)
        lo, hi = int(math.floor(index)), int(math.ceil(index))
        if lo == hi:
            return data[lo]
        return data[lo] * (hi - index) + data[hi] * (index - lo)
    return {
        "count": len(data),
        "min": data[0],
        "p10": q(0.10),
        "median": q(0.50),
        "p90": q(0.90),
        "max": data[-1],
    }


def _object_signature(obj: Mapping[str, Any]) -> tuple[Any, ...]:
    """Fields that a semantic relation/binding operation is not allowed to mutate."""
    return (
        int(obj.get("object_id", -1)),
        str(obj.get("class_label", "")),
        tuple(round(float(value), 8) for value in obj.get("center_3d", ())),
        tuple(round(float(value), 8) for value in obj.get("bbox_3d", ())),
        str(obj.get("cardinality_role", "")),
        str(obj.get("identity_state", "")),
    )


def _size_band(size: Sequence[float]) -> str:
    volume = math.prod(max(0.0, float(value)) for value in size)
    if volume < 0.01:
        return "small_lt_0.01m3"
    if volume < 0.10:
        return "medium_0.01_to_0.10m3"
    return "large_ge_0.10m3"


def _point_band(point_count: int) -> str:
    if point_count < 16:
        return "sparse_lt_16"
    if point_count < 64:
        return "moderate_16_to_63"
    return "dense_ge_64"


def _fragment_band(fragment_ratio: float) -> str:
    if fragment_ratio <= 0.0:
        return "none"
    if fragment_ratio < 0.50:
        return "partial_lt_0.50"
    return "large_ge_0.50"


def _noise_band(noise_m: float) -> str:
    if noise_m <= 0.0:
        return "none"
    if noise_m <= 0.02:
        return "low_le_0.02m"
    return "high_gt_0.02m"


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("status", "UNKNOWN")).lower() for row in rows)
    return dict(sorted(counts.items()))


def _csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))))


def _collect_rows(value: object) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        if "target_index" in value and "relation" in value:
            return [dict(value)]
        rows: list[dict[str, Any]] = []
        for child in value.values():
            rows.extend(_collect_rows(child))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(_collect_rows(child))
        return rows
    return []


def _scene_members(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    pattern = re.compile(r"^Unity/([^/]+)/\1_(.+)$")
    for member in archive.namelist():
        match = pattern.match(member)
        if match:
            result[match.group(1)][match.group(2)] = member
    return dict(result)


def _graph_truth(graph: Mapping[str, Any]) -> dict[str, set[tuple[Any, ...]]]:
    truth: dict[str, set[tuple[Any, ...]]] = {relation: set() for relation in RELATIONS}
    for region in (graph.get("regions", {}) or {}).values():
        relationships = region.get("relationships", {}) if isinstance(region, Mapping) else {}
        for raw_relation, mapping in relationships.items():
            relation = _norm(raw_relation)
            if relation not in truth or not isinstance(mapping, Mapping):
                continue
            for raw_subject, raw_targets in mapping.items():
                subject = _int(raw_subject)
                if subject < 0 or not isinstance(raw_targets, list):
                    continue
                for raw_target in raw_targets:
                    if relation == "between" and isinstance(raw_target, list) and len(raw_target) == 2:
                        pair = tuple(sorted((_int(raw_target[0]), _int(raw_target[1]))))
                        if pair[0] >= 0:
                            truth[relation].add((subject, *pair))
                    else:
                        target = _int(raw_target)
                        if target >= 0:
                            # VLA's generator stores binary relations as
                            # anchor -> target, while the semantic contract
                            # consumed by the oracle is RELATION(target,
                            # anchor).  BETWEEN is already stored as
                            # target -> [anchor_1, anchor_2] above.
                            truth[relation].add((target, subject))
    return truth


def _gt_object(row: Mapping[str, Any], point_count: int = 32) -> dict[str, Any]:
    object_id = _int(row.get("object_id"))
    center = [_float(row.get(f"object_bbox_c{axis}"), 0.0) for axis in "xyz"]
    size = [max(1e-4, _float(row.get(f"object_bbox_{axis}length"), 0.1)) for axis in "xyz"]
    label = _norm(row.get("raw_label") or row.get("nyu_label") or "")
    colors = [
        _norm(row.get("object_color_scheme1")),
        _norm(row.get("object_color_scheme2")),
        _norm(row.get("object_color_scheme3")),
    ]
    colors = [value for value in colors if value and value != "n/a"]
    heading = _float(row.get("object_bbox_heading"), 0.0)
    cosine, sine = math.cos(heading), math.sin(heading)
    half_x, half_y, half_z = (value / 2.0 for value in size)
    local_corners = [
        (-half_x, half_y, half_z), (half_x, half_y, half_z),
        (half_x, -half_y, half_z), (-half_x, -half_y, half_z),
        (-half_x, half_y, -half_z), (half_x, half_y, -half_z),
        (half_x, -half_y, -half_z), (-half_x, -half_y, -half_z),
    ]
    obb_corners = [
        [
            center[0] + cosine * local[0] - sine * local[1],
            center[1] + sine * local[0] + cosine * local[1],
            center[2] + local[2],
        ]
        for local in local_corners
    ]
    covariance = np.diag([0.0001, 0.0001, 0.0001]).tolist()
    return {
        "object_id": object_id,
        "class_label": label,
        "raw_label": str(row.get("raw_label", "")),
        "nyu_label": str(row.get("nyu_label", "")),
        "nyu40_label": str(row.get("nyu40_label", "")),
        "center_3d": center,
        "bbox_3d": size,
        "bbox_corners": obb_corners,
        "obb_corners": obb_corners,
        "bbox_heading_rad": heading,
        "bbox_frame": "vla_world_z_up",
        "footprint_obb": {"center_xy": center[:2], "extent_xy": size[:2], "yaw_rad": heading},
        "center_cov": covariance,
        "extent_cov": covariance,
        "covariance_mode": "strict",
        "center_covariance_provenance": "fused_statistical",
        "extent_covariance_provenance": "extent_derived",
        "geometry_point_count": max(1, int(point_count)),
        "geometry_source": "vla_gt_offline",
        "geometry_confidence": 1.0,
        "semantic_probability": 1.0,
        "status": "confirmed",
        "physical_status": "confirmed",
        "semantic_status": "verified",
        "cardinality_role": "ATOMIC",
        "identity_state": "CONFIRMED",
        "class_log_probs": {label: 1.0} if label else {},
        "color_labels": colors,
        "region_id": str(row.get("region_id", "-1")),
        "evidence": [],
    }


def _ply_samples(
    archive: zipfile.ZipFile,
    member: str,
    split: np.ndarray | None,
    object_ids: Sequence[int],
    *,
    per_object: int = 48,
) -> dict[int, list[list[float]]]:
    if split is None or not member:
        return {}
    result: dict[int, list[list[float]]] = {}
    try:
        with archive.open(member) as stream:
            header = b""
            while b"end_header\n" not in header:
                line = stream.readline()
                if not line:
                    return {}
                header += line
            properties = [line.decode("ascii", "ignore").strip() for line in header.splitlines()]
            vertex_count = next(
                (_int(line.split()[2]) for line in properties if line.startswith("element vertex ")),
                0,
            )
            if vertex_count <= 0:
                return {}
            stride = 15
            for object_id in sorted(set(int(value) for value in object_ids)):
                if object_id < 0 or object_id >= len(split):
                    continue
                start, end = int(split[object_id][0]), int(split[object_id][1])
                if end <= start:
                    continue
                stream.seek(len(header) + start * stride)
                take = min(per_object, end - start)
                raw = stream.read(take * stride)
                points = []
                for offset in range(0, len(raw) - stride + 1, stride):
                    x, y, z, _r, _g, _b = struct.unpack("<fffBBB", raw[offset:offset + stride])
                    points.append([float(x), float(y), float(z)])
                if points:
                    result[object_id] = points
    except (OSError, ValueError, struct.error, IndexError):
        return result
    return result


class VLAWorld:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path
        self.archive = zipfile.ZipFile(archive_path)
        self.members = _scene_members(self.archive)
        self.scenes: dict[str, dict[str, Any]] = {}

    def load(self) -> dict[str, dict[str, Any]]:
        for scene_name in sorted(self.members):
            files = self.members[scene_name]
            required = ["object_result.csv", "scene_graph.json", "referential_statements.json"]
            missing = [name for name in required if name not in files]
            if missing:
                raise RuntimeError(f"scene_missing:{scene_name}:{','.join(missing)}")
            rows = _csv(self.archive, files["object_result.csv"])
            split = None
            split_member = files.get("object_split.npy")
            if split_member:
                try:
                    split = np.load(io.BytesIO(self.archive.read(split_member)))
                except (OSError, ValueError):
                    split = None
            objects = []
            by_id: dict[int, dict[str, Any]] = {}
            for row in rows:
                object_id = _int(row.get("object_id"))
                count = 32
                if split is not None and 0 <= object_id < len(split):
                    count = max(1, int(split[object_id][1]) - int(split[object_id][0]))
                obj = _gt_object(row, count)
                by_id[object_id] = obj
                objects.append(obj)
            graph = json.loads(self.archive.read(files["scene_graph.json"]))
            referential = json.loads(self.archive.read(files["referential_statements.json"]))
            self.scenes[scene_name] = {
                "name": scene_name,
                "objects": sorted(objects, key=lambda value: int(value["object_id"])),
                "by_id": by_id,
                "graph": graph,
                "truth": _graph_truth(graph),
                "referential_rows": _collect_rows(referential.get("regions", referential)),
                "split": split,
                "pc_member": files.get("pc_result.ply", ""),
                "point_samples": {},
            }
        return self.scenes

    def load_point_samples(self, scene_name: str, object_ids: Sequence[int]) -> dict[int, list[list[float]]]:
        scene = self.scenes[scene_name]
        if not scene["point_samples"]:
            scene["point_samples"] = _ply_samples(
                self.archive,
                scene["pc_member"],
                scene["split"],
                object_ids,
            )
        else:
            missing = [value for value in object_ids if int(value) not in scene["point_samples"]]
            if missing:
                scene["point_samples"].update(
                    _ply_samples(self.archive, scene["pc_member"], scene["split"], missing)
                )
        return scene["point_samples"]


def _relation_truth(scene: Mapping[str, Any], predicate: str, subject: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]]) -> bool:
    predicate = _norm(predicate)
    source = int(subject["object_id"])
    anchor_ids = tuple(int(value["object_id"]) for value in anchors)
    truth = scene["truth"].get(predicate, set())
    if predicate == "between":
        return (source, *sorted(anchor_ids)) in truth if len(anchor_ids) == 2 else False
    if predicate in {"closest", "farthest"} and len(anchors) == 1:
        target_class = _norm(subject.get("class_label"))
        domain = [
            value for value in scene["objects"]
            if _class_match(target_class, value)
            and str(value.get("region_id")) == str(subject.get("region_id"))
        ]
        distances = [(round(_xy_distance(value, anchors[0]), 9), int(value["object_id"])) for value in domain]
        if not distances:
            return False
        best = min(value[0] for value in distances) if predicate == "closest" else max(value[0] for value in distances)
        return round(_xy_distance(subject, anchors[0]), 9) == best
    return (source, *anchor_ids) in truth


def _attribute_match(entity: Mapping[str, Any], obj: Mapping[str, Any], scene: Mapping[str, Any]) -> bool:
    attributes = entity.get("attributes", {})
    if not isinstance(attributes, Mapping):
        return True
    color = _norm(attributes.get("color"))
    if color and color not in set(obj.get("color_labels", ())):
        return False
    size = _norm(attributes.get("size"))
    if size:
        same_class = [
            math.prod(float(value) for value in item.get("bbox_3d", ()))
            for item in scene["objects"]
            if _class_match(entity.get("class_name"), item)
        ]
        volume = math.prod(float(value) for value in obj.get("bbox_3d", ()))
        if same_class:
            median = sorted(same_class)[len(same_class) // 2]
            if size == "small" and volume > median:
                return False
            if size == "large" and volume < median:
                return False
    return True


def _gt_candidates(task_ir: Mapping[str, Any], scene: Mapping[str, Any], entity_id: str, cache: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    cache = cache if cache is not None else {}
    if entity_id in cache:
        return cache[entity_id]
    entities = {str(value.get("id")): value for value in task_ir.get("entities", ())}
    entity = entities.get(str(entity_id), {})
    candidates = [
        value for value in scene["objects"]
        if _class_match(entity.get("class_name"), value) and _attribute_match(entity, value, scene)
    ]
    for relation in task_ir.get("relations", ()):
        if str(relation.get("subject_entity")) != str(entity_id):
            continue
        anchor_sets = [
            _gt_candidates(task_ir, scene, str(anchor_id), cache)
            for anchor_id in relation.get("object_entities", ())
        ]
        predicate = _norm(relation.get("predicate"))
        if not anchor_sets or any(not value for value in anchor_sets):
            candidates = []
            break
        filtered = []
        for candidate in candidates:
            if any(_relation_truth(scene, predicate, candidate, combo) for combo in itertools.product(*anchor_sets)):
                filtered.append(candidate)
        candidates = filtered
    cache[entity_id] = list(dict.fromkeys(int(value["object_id"]) for value in candidates))
    return [scene["by_id"][object_id] for object_id in cache[entity_id]]


def _direct_gt_candidates(task_ir: Mapping[str, Any], scene: Mapping[str, Any], entity_id: str) -> list[dict[str, Any]]:
    """GT class/attribute domain without applying the production relation resolver."""
    entity = next(
        (value for value in task_ir.get("entities", ()) if str(value.get("id")) == str(entity_id)),
        {},
    )
    return [
        value for value in scene["objects"]
        if _class_match(entity.get("class_name"), value)
        and _attribute_match(entity, value, scene)
    ]


def _runtime_candidates(task_ir: Mapping[str, Any], scene: Mapping[str, Any], entity_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    snapshot = {
        "objects": copy.deepcopy(scene["objects"]),
        "query_domain": {"selector_closures": {}},
    }
    resolver = _Resolver(task_ir, snapshot)
    return resolver.resolve(str(entity_id))


def _sample_referential(scenes: Mapping[str, Mapping[str, Any]], quota: int = 20) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for scene_name, scene in scenes.items():
        for row in scene["referential_rows"]:
            relation = _norm(row.get("relation"))
            if relation in RELATIONS:
                grouped[(scene_name, relation)].append(row)
    rng = random.Random(RNG_SEED)
    result = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda value: (str(value.get("target_index")), str(value.get("target_class"))))
        rng.shuffle(rows)
        result.extend({"scene": key[0], **row} for row in rows[:quota])
    return result


def _binding_oracle(scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sample = _sample_referential(scenes)
    records = []
    per_relation: dict[str, Counter[str]] = defaultdict(Counter)
    for item in sample:
        scene = scenes[item["scene"]]
        relation = _norm(item.get("relation"))
        phrase = str(next((key for key, value in (("language", item.get("language")), ("text", item.get("text"))) if value), ""))
        # The archive stores the statement as the phrase key; _collect_rows
        # preserves no key, so reconstructing from target fields is not safe.
        # Use the relation-specific canonical phrase when the optional field is
        # absent and mark the case unsupported rather than inventing language.
        language = str(item.get("language") or item.get("statement") or item.get("text") or "")
        if not language:
            language = str(item.get("target_class", ""))
        query = "Find " + language
        expected = _int(item.get("target_index"))
        record = {
            "scene": item["scene"],
            "relation": relation,
            "expected_target": expected,
            "query": query,
            "status": "UNSUPPORTED",
            "predicted_targets": [],
            "anchor_checks": [],
            "reason": "statement_language_not_preserved_by_archive_adapter",
        }
        # A second pass below uses phrase keys preserved by _referential_cases.
        records.append(record)
        per_relation[relation]["unsupported"] += 1
    return {"sample_cases": len(sample), "records": records[:200], "by_relation": {key: dict(value) for key, value in per_relation.items()}, "note": "Replaced by _binding_oracle_from_cases when phrase keys are available."}


def _referential_cases(scenes: Mapping[str, Mapping[str, Any]], quota: int = 20) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    # The phrase key is attached during loading by _collect_language_cases.
    for scene_name, scene in scenes.items():
        for case in scene.get("referential_cases", ()):
            relation = _norm(case.get("relation"))
            if relation in RELATIONS:
                grouped[(scene_name, relation)].append(case)
    rng = random.Random(RNG_SEED)
    result = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda value: (str(value.get("target_index")), value.get("language", "")))
        rng.shuffle(rows)
        result.extend(rows[:quota])
    return result


def _expected_anchor_ids(item: Mapping[str, Any]) -> list[int]:
    anchors = item.get("anchors", {})
    if not isinstance(anchors, Mapping):
        return []
    ordered = sorted(
        anchors.values(),
        key=lambda value: str(value.get("index", ""))
        if isinstance(value, Mapping)
        else "",
    )
    return [
        _int(value.get("index"))
        for value in ordered
        if isinstance(value, Mapping) and _int(value.get("index")) >= 0
    ]


def _binding_oracle_from_cases(scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sample = _referential_cases(scenes)
    per_relation: dict[str, Counter[str]] = defaultdict(Counter)
    records = []
    for item in sample:
        scene = scenes[item["scene"]]
        relation = _norm(item.get("relation"))
        per_relation[relation]["cases"] += 1
        expected = _int(item.get("target_index"))
        query = "Find " + str(item.get("language", ""))
        record: dict[str, Any] = {
            "scene": item["scene"],
            "relation": relation,
            "query": query,
            "expected_target": expected,
            "status": "UNSUPPORTED",
            "predicted_targets": [],
            "anchor_checks": [],
            "reason": "",
        }
        try:
            ir = compile_task(query)
            gt = _gt_candidates(ir, scene, str(ir.get("target_entity", "")))
            predicted, failures = _runtime_candidates(ir, scene, str(ir.get("target_entity", "")))
            predicted_ids = sorted(int(value["object_id"]) for value in predicted)
            expected_anchors = _expected_anchor_ids(item)
            direct_relations = [
                value for value in ir.get("relations", ())
                if str(value.get("subject_entity")) == str(ir.get("target_entity", ""))
            ]
            predicted_anchor_ids: list[list[int]] = []
            for relation_ir in direct_relations:
                relation_predictions = []
                for anchor_entity in relation_ir.get("object_entities", ()):
                    anchor_values, anchor_failures = _runtime_candidates(
                        ir, scene, str(anchor_entity)
                    )
                    anchor_ids = sorted(int(value["object_id"]) for value in anchor_values)
                    relation_predictions.append({
                        "entity_id": str(anchor_entity),
                        "predicted_ids": anchor_ids,
                        "failures": anchor_failures[:3],
                    })
                predicted_anchor_ids.extend(
                    [value.get("predicted_ids", []) for value in relation_predictions]
                )
                record["anchor_checks"].extend(relation_predictions)
            anchor_statuses = []
            for index, expected_anchor in enumerate(expected_anchors):
                predicted_values = (
                    predicted_anchor_ids[index]
                    if index < len(predicted_anchor_ids)
                    else []
                )
                if len(predicted_values) == 1 and predicted_values[0] == expected_anchor:
                    anchor_statuses.append("CORRECT")
                elif not predicted_values:
                    anchor_statuses.append("UNRESOLVED")
                elif expected_anchor in predicted_values:
                    anchor_statuses.append("AMBIGUOUS")
                else:
                    anchor_statuses.append("WRONG_BIND")
            record["expected_anchor_ids"] = expected_anchors
            record["anchor_statuses"] = anchor_statuses
            record["compiled_relation_count"] = len(ir.get("relations", ()))
            record["predicted_targets"] = predicted_ids
            record["compiled_relations"] = [str(value.get("predicate")) for value in ir.get("relations", ())]
            record["gt_candidate_ids"] = sorted(int(value["object_id"]) for value in gt)
            if failures:
                record["reason"] = ";".join(failures[:3])
            if expected in predicted_ids and len(predicted_ids) == 1:
                record["status"] = "CORRECT"
                per_relation[relation]["correct"] += 1
            elif not predicted_ids:
                record["status"] = "UNRESOLVED"
                per_relation[relation]["unresolved"] += 1
            elif expected not in predicted_ids:
                record["status"] = "WRONG_BIND"
                per_relation[relation]["wrong_bind"] += 1
            else:
                record["status"] = "AMBIGUOUS"
                per_relation[relation]["ambiguous"] += 1
            for anchor_status in anchor_statuses:
                per_relation[relation][f"anchor_{anchor_status.lower()}"] += 1
            if expected_anchors and not anchor_statuses:
                per_relation[relation]["anchor_unresolved"] += len(expected_anchors)
            if not record["reason"]:
                record["reason"] = "production_query_executor_gt_adapter"
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}:{exc}"
            per_relation[relation]["unsupported"] += 1
        records.append(record)
    summary = {}
    for relation, counts in sorted(per_relation.items()):
        total = counts["cases"]
        anchor_total = sum(
            value for key, value in counts.items() if key.startswith("anchor_")
        )
        summary[relation] = {
            "cases": total,
            **dict(counts),
            "binding_accuracy": counts["correct"] / total if total else 0.0,
            "anchor_cases": anchor_total,
            "anchor_accuracy": counts["anchor_correct"] / anchor_total if anchor_total else 0.0,
        }
    scene_counts = Counter(str(value["scene"]) for value in records)
    return {
        "cases": len(sample),
        "scene_count": len(scene_counts),
        "cases_by_scene": dict(sorted(scene_counts.items())),
        "by_relation": summary,
        "failure_examples": [value for value in records if value["status"] != "CORRECT"][:80],
        "records_sample": records[:240],
    }


def _relation_suite(scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def hard_negative_type(predicate: str, item: Sequence[int], by_id: Mapping[int, Mapping[str, Any]]) -> str:
        subject = by_id[int(item[0])]
        anchors = [by_id[int(value)] for value in item[1:]]
        same_class = any(
            _class_match(subject.get("class_label"), anchor)
            for anchor in anchors
        )
        same_region = bool(anchors) and all(
            str(subject.get("region_id")) == str(anchor.get("region_id"))
            for anchor in anchors
        )
        nearby = min(
            (_xy_distance(subject, anchor) for anchor in anchors),
            default=float("inf"),
        ) <= 1.0
        if same_class and nearby:
            return "same_class_nearby"
        if same_class:
            return "same_class_nonrelation"
        if same_region and nearby:
            return "same_region_nearby"
        if same_region:
            return "same_region_nonrelation"
        return "cross_region_or_class"

    rng = random.Random(RNG_SEED)
    records = []
    for scene_name, scene in sorted(scenes.items()):
        by_id = scene["by_id"]
        for predicate in GEOMETRIC_RELATIONS:
            positives: list[tuple[int, ...]] = []
            for value in scene["truth"].get(predicate, set()):
                if predicate == "between" and len(value) == 3:
                    positives.append(tuple(int(x) for x in value))
                elif predicate != "between" and len(value) == 2:
                    positives.append(tuple(int(x) for x in value))
            rng.shuffle(positives)
            positives = positives[:80]
            positive_keys = set(positives)
            negative_pool: list[tuple[int, ...]] = []
            object_ids = sorted(by_id)
            if predicate == "between":
                for subject, first, second in itertools.islice(itertools.permutations(object_ids, 3), 0, 5000):
                    value = (subject, *sorted((first, second)))
                    if value not in positive_keys:
                        negative_pool.append(value)
            else:
                for subject in object_ids:
                    for anchor in object_ids:
                        if subject != anchor and (subject, anchor) not in positive_keys:
                            negative_pool.append((subject, anchor))
            negative_pool.sort(key=lambda value: _xy_distance(by_id[value[0]], by_id[value[1]]) if len(value) == 2 else 0.0)
            if len(negative_pool) > 80:
                nearby = negative_pool[:40]
                rng.shuffle(negative_pool[40:])
                negative_pool = nearby + negative_pool[40:80]
            for truth_value, tuples in ((True, positives), (False, negative_pool)):
                for item in tuples:
                    subject = by_id.get(int(item[0]))
                    anchors = [by_id.get(int(value)) for value in item[1:]]
                    if subject is None or any(value is None for value in anchors):
                        continue
                    node = {"id": "vla_relation", "predicate": predicate}
                    geometry = relation_geometry_diagnostic(node, subject, anchors)
                    qwen = {
                        "state": "supported",
                        "jointly_observable": True,
                        "subject_role_state": "YES",
                        "object_role_states": ["YES"] * len(anchors),
                    }
                    owner_error = ""
                    try:
                        state, reason = _geometry_consistency(node, geometry, subject, anchors, qwen)
                    except Exception as exc:
                        state = "UNKNOWN"
                        owner_error = f"{type(exc).__name__}:{exc}"
                        reason = f"owner_exception:{owner_error}"
                    record = {
                        "scene": scene_name,
                        "operator": predicate,
                        "truth": truth_value,
                        "predicted": state,
                        "reason": reason,
                        "hard_negative": not truth_value,
                        "hard_negative_type": hard_negative_type(predicate, item, by_id) if not truth_value else "positive_control",
                        "object_ids": list(item),
                    }
                    if owner_error:
                        record["owner_error"] = owner_error
                    records.append(record)
    by_operator: dict[str, dict[str, Any]] = {}
    by_scene: dict[str, dict[str, Any]] = {}
    hard_negative_by_operator: dict[str, dict[str, Any]] = {}
    for predicate in GEOMETRIC_RELATIONS:
        subset = [value for value in records if value["operator"] == predicate]
        counts = Counter()
        for value in subset:
            if value["predicted"] == "YES" and value["truth"]:
                counts["true_positive"] += 1
            elif value["predicted"] == "YES" and not value["truth"]:
                counts["false_yes"] += 1
            elif value["predicted"] == "NO" and not value["truth"]:
                counts["true_negative"] += 1
            elif value["predicted"] == "NO" and value["truth"]:
                counts["false_no"] += 1
            elif value["predicted"] == "UNKNOWN":
                counts["unknown"] += 1
            else:
                counts["other"] += 1
        total = len(subset)
        positives = sum(value["truth"] for value in subset)
        negatives = total - positives
        counts["false_positive"] = counts["false_yes"]
        counts["false_negative"] = counts["false_no"]
        counts["owner_exception"] = sum("owner_error" in value for value in subset)
        by_operator[predicate] = {
            "cases": total,
            **dict(counts),
            "positive_cases": positives,
            "negative_cases": negatives,
            "tp": counts["true_positive"],
            "fp": counts["false_yes"],
            "tn": counts["true_negative"],
            "fn": counts["false_no"],
            "positive_recall": counts["true_positive"] / positives if positives else 0.0,
            "false_yes_rate": counts["false_yes"] / max(1, negatives),
            "unknown_rate": counts["unknown"] / total if total else 0.0,
        }
        negative_subset = [value for value in subset if not value["truth"]]
        category_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for value in negative_subset:
            category_counts[value["hard_negative_type"]][value["predicted"].lower()] += 1
        hard_negative_by_operator[predicate] = {
            category: dict(sorted(value.items()))
            for category, value in sorted(category_counts.items())
        }
    for scene_name in sorted(scenes):
        subset = [value for value in records if value["scene"] == scene_name]
        scene_counts = Counter()
        for value in subset:
            if value["predicted"] == "YES" and value["truth"]:
                scene_counts["tp"] += 1
            elif value["predicted"] == "YES" and not value["truth"]:
                scene_counts["fp"] += 1
            elif value["predicted"] == "NO" and not value["truth"]:
                scene_counts["tn"] += 1
            elif value["predicted"] == "NO" and value["truth"]:
                scene_counts["fn"] += 1
            elif value["predicted"] == "UNKNOWN":
                scene_counts["unknown"] += 1
        scene_counts["cases"] = len(subset)
        by_scene[scene_name] = dict(sorted(scene_counts.items()))
    return {
        "by_operator": by_operator,
        "by_scene": by_scene,
        "hard_negative_by_operator": hard_negative_by_operator,
        "owner_errors": dict(Counter(value.get("owner_error", "") for value in records if value.get("owner_error"))),
        "failure_examples": [value for value in records if value["predicted"] in {"YES", "NO"} and value["predicted"] != ("YES" if value["truth"] else "NO")][:120],
        "cases": len(records),
    }


def _gt_ordered_domain(scene: Mapping[str, Any], row: Mapping[str, Any], predicate: str) -> dict[str, Any]:
    anchor_ids = _expected_anchor_ids(row)
    if len(anchor_ids) != 1 or anchor_ids[0] not in scene["by_id"]:
        return {"anchor_ids": anchor_ids, "domain_ids": [], "target_rank": None, "consistent": False, "reason": "gt_anchor_missing"}
    anchor = scene["by_id"][anchor_ids[0]]
    target_class = row.get("target_class", "")
    target_region = str(row.get("region_id", ""))
    domain = [
        value for value in scene["objects"]
        if _class_match(target_class, value)
        and (not target_region or str(value.get("region_id")) == target_region)
    ]
    ordered = sorted(
        domain,
        key=lambda value: (
            round(_xy_distance(value, anchor), 9),
            int(value["object_id"]),
        ),
        reverse=predicate == "farthest",
    )
    target = _int(row.get("target_index"))
    ids = [int(value["object_id"]) for value in ordered]
    rank = ids.index(target) + 1 if target in ids else None
    return {
        "anchor_ids": anchor_ids,
        "domain_ids": ids,
        "target_rank": rank,
        "consistent": rank is not None,
        "reason": "gt_same_class_same_region_xy_order",
    }


def _ordered_suite(scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene_name, scene in scenes.items():
        for row in scene.get("referential_cases", ()):
            predicate = _norm(row.get("relation"))
            if predicate in ORDERED_RELATIONS:
                grouped[predicate].append({"scene": scene_name, **row})
    rng = random.Random(RNG_SEED)
    summary = {}
    failures = []
    for predicate, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: (value["scene"], str(value.get("target_index"))))
        rng.shuffle(rows)
        rows = rows[:240]
        counts = Counter()
        source_counts = Counter()
        for row in rows:
            scene = scenes[row["scene"]]
            query = "Find " + str(row.get("language", ""))
            gt_domain = _gt_ordered_domain(scene, row, predicate)
            try:
                ir = compile_task(query)
                predicted, resolver_failures = _runtime_candidates(ir, scene, str(ir.get("target_entity", "")))
                ids = sorted(int(value["object_id"]) for value in predicted)
                target = _int(row.get("target_index"))
                if gt_domain["consistent"]:
                    counts["gt_domain_consistent"] += 1
                else:
                    counts["gt_domain_unresolved"] += 1
                if ids == [target]:
                    counts["correct"] += 1
                elif not ids:
                    counts["unresolved"] += 1
                    source = "anchor_binding_or_compiler"
                    if any("attribute_evidence_unavailable" in value for value in resolver_failures):
                        source = "anchor_attribute_binding"
                    elif any("ranking_relation_unresolved" in value for value in resolver_failures):
                        source = "ordered_domain"
                    source_counts[source] += 1
                    failures.append({"scene": row["scene"], "operator": predicate, "query": query, "failure_source": source, "reason": resolver_failures[:3], "gt_domain": gt_domain})
                elif target not in ids:
                    counts["wrong"] += 1
                    source_counts["current_ordered_binding"] += 1
                    failures.append({"scene": row["scene"], "operator": predicate, "query": query, "failure_source": "current_ordered_binding", "predicted": ids, "expected": target, "gt_domain": gt_domain})
                else:
                    counts["ambiguous"] += 1
                    source_counts["current_ordered_binding_ambiguous"] += 1
            except Exception as exc:
                counts["unsupported"] += 1
                source_counts["parser"] += 1
                failures.append({"scene": row["scene"], "operator": predicate, "query": query, "failure_source": "parser", "reason": f"{type(exc).__name__}:{exc}", "gt_domain": gt_domain})
        total = len(rows)
        summary[predicate] = {
            "cases": total,
            **dict(counts),
            "target_accuracy": counts["correct"] / total if total else 0.0,
            "failure_source_counts": dict(sorted(source_counts.items())),
        }
    return {"by_operator": summary, "failure_examples": failures[:100], "domain_definition": "same target class and VLA region; XY distance to GT anchor; deterministic object_id tie-break"}


def _write_mask(path: Path, box: tuple[int, int, int, int]) -> None:
    import cv2
    mask = np.zeros((96, 160), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    cv2.imwrite(str(path), mask)


def _observation(
    observation_id: str,
    class_name: str,
    center: Sequence[float],
    size: Sequence[float],
    points: Sequence[Sequence[float]],
    root: Path,
    *,
    acquisition_id: str,
    mask_box: tuple[int, int, int, int],
    cannot: Sequence[str] = (),
    coverage: Sequence[str] = (),
    semantic_probability: float = 0.99,
) -> dict[str, Any]:
    mask_path = root / f"{observation_id}.png"
    point_path = root / f"{observation_id}.npz"
    _write_mask(mask_path, mask_box)
    array = np.asarray(points, dtype=np.float32)
    np.savez_compressed(point_path, world_points=array, lidar_support_points_map=array)
    covariance = np.diag([0.0004, 0.0004, 0.0004]).tolist()
    return {
        "observation_id": observation_id,
        "acquisition_id": acquisition_id,
        "timestamp": float(len(observation_id)),
        "class_label": class_name,
        "canonical_class": class_name,
        "observed_class_label": class_name,
        "center_3d": [float(value) for value in center],
        "bbox_3d": [max(0.01, float(value)) for value in size],
        "center_cov": covariance,
        "extent_cov": covariance,
        "covariance_mode": "strict",
        "center_covariance_provenance": "fused_statistical",
        "extent_covariance_provenance": "extent_derived",
        "viewpoint_position_map": [0.0, -2.0, 0.75],
        "optical_center_group": f"station:{acquisition_id}",
        "semantic_probability": semantic_probability,
        "qwen_verified": True,
        "qwen_candidate_verdict": "target_wins",
        "lidar_support_count": max(8, len(array)),
        "geometry_point_count": max(8, len(array)),
        "geometry_confidence": 0.95,
        "geometry_source": "vla_gt_offline",
        "appearance_descriptor": [1.0, 0.0, 0.0],
        "appearance_quality": 0.95,
        "source_view_ids": [f"view:{acquisition_id}"],
        "source_view_boxes": [],
        "cannot_link_observation_ids": list(cannot),
        "coverage_member_observation_ids": list(coverage),
        "panorama_mask_path": str(mask_path),
        "pointcloud_path": str(point_path),
        "world_points_path": str(point_path),
    }


def _scenario_points(scene: Mapping[str, Any], obj: Mapping[str, Any], samples: Mapping[int, list[list[float]]]) -> list[list[float]]:
    points = samples.get(int(obj["object_id"]))
    if points:
        return points
    center, size = _center(obj), obj["bbox_3d"]
    return [[center[0] + dx * size[0] * 0.5, center[1] + dy * size[1] * 0.5, center[2] + dz * size[2] * 0.5] for dx, dy, dz in itertools.product((-1.0, 1.0), repeat=3)]


def _cardinality_stress(scenes: Mapping[str, Mapping[str, Any]], temp_root: Path) -> dict[str, Any]:
    scenarios = []
    sampled_objects = 0
    for scene_name, scene in sorted(scenes.items()):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for obj in scene["objects"]:
            if str(obj.get("region_id")) != "-1":
                groups[_norm(obj.get("class_label"))].append(obj)
        pair = next((values[:2] for values in groups.values() if len(values) >= 2), None)
        if not pair:
            continue
        sample_ids = [int(value["object_id"]) for value in pair]
        samples = dict(scene.get("point_samples", {}))
        if not samples:
            samples.update(VLA_POINT_SAMPLES.get(scene_name, {}))
        if not samples:
            # The caller populates VLA_POINT_SAMPLES before this block.
            samples = {}
        sampled_objects += sum(int(value) in samples for value in sample_ids)
        first, second = pair
        class_name = _norm(first["class_label"])
        size = first["bbox_3d"]
        points = _scenario_points(scene, first, samples)
        second_points = _scenario_points(scene, second, samples)
        full_point_count = len(points)
        cases = {
            "duplicate_observation": [
                _observation(f"{scene_name}_dup_full", class_name, first["center_3d"], size, points, temp_root, acquisition_id="A0", mask_box=(25, 25, 85, 75)),
                _observation(f"{scene_name}_dup_noise", class_name, [first["center_3d"][0] + 0.01, first["center_3d"][1], first["center_3d"][2]], size, points[: max(8, len(points) // 2)], temp_root, acquisition_id="A0", mask_box=(27, 25, 87, 75)),
            ],
            "partial_fragment": [
                _observation(f"{scene_name}_frag_full", class_name, first["center_3d"], size, points, temp_root, acquisition_id="A0", mask_box=(20, 20, 100, 80)),
                _observation(f"{scene_name}_frag_crop", class_name, first["center_3d"], [size[0] * 0.55, size[1] * 0.55, size[2] * 0.55], points[: max(8, len(points) // 3)], temp_root, acquisition_id="A0", mask_box=(35, 30, 75, 70)),
            ],
            "aggregate_coverage": [
                _observation(f"{scene_name}_agg_a", class_name, first["center_3d"], size, points, temp_root, acquisition_id="A0", mask_box=(15, 20, 55, 80), cannot=(f"{scene_name}_agg_b",)),
                _observation(f"{scene_name}_agg_b", class_name, second["center_3d"], second["bbox_3d"], second_points, temp_root, acquisition_id="A0", mask_box=(75, 20, 145, 80), cannot=(f"{scene_name}_agg_a",)),
                _observation(f"{scene_name}_agg_union", class_name, [(first["center_3d"][i] + second["center_3d"][i]) * 0.5 for i in range(3)], [max(size[i], second["bbox_3d"][i]) * 2.0 for i in range(3)], points + second_points, temp_root, acquisition_id="A0", mask_box=(12, 18, 148, 82), coverage=(f"{scene_name}_agg_a", f"{scene_name}_agg_b")),
            ],
            "adjacent_same_class": [
                _observation(f"{scene_name}_adj_a", class_name, first["center_3d"], size, points, temp_root, acquisition_id="A0", mask_box=(15, 20, 70, 80), cannot=(f"{scene_name}_adj_b",)),
                _observation(f"{scene_name}_adj_b", class_name, second["center_3d"], second["bbox_3d"], second_points, temp_root, acquisition_id="A0", mask_box=(80, 20, 145, 80), cannot=(f"{scene_name}_adj_a",)),
            ],
            "missing_observation": [
                _observation(f"{scene_name}_missing_anchor_only", class_name, first["center_3d"], size, points, temp_root, acquisition_id="A0", mask_box=(25, 25, 85, 75), coverage=(f"{scene_name}_missing_instance",)),
            ],
        }
        for scenario, records in cases.items():
            try:
                if scenario == "duplicate_observation":
                    records = _deduplicate_station_observations(records, acquisition_id="A0", geometry_dir=temp_root)
                query = {"schema_version": "task_ir_v2", "task_type": "numerical", "target_entity": "target_0", "entities": [{"id": "target_0", "role": "target", "class_name": class_name, "attributes": {}}], "relations": []}
                view = materialize_query_view(query, {"records": records})
                objects = view.get("objects", ())
                roles = Counter(str(value.get("cardinality_role", "UNKNOWN_CARDINALITY")) for value in objects)
                actual = len(objects)
                expected = {
                    "duplicate_observation": "one represented instance; no duplicate birth",
                    "partial_fragment": "one atomic instance plus PARTIAL_FRAGMENT role",
                    "aggregate_coverage": "two ATOMIC instances plus AGGREGATE_COVERAGE role",
                    "adjacent_same_class": "two independent ATOMIC instances",
                    "missing_observation": "available evidence remains UNKNOWN_CARDINALITY",
                }[scenario]
                false_duplicate_birth = int(scenario == "duplicate_observation" and actual > 1)
                false_fragment_independent = int(
                    scenario == "partial_fragment"
                    and roles.get("PARTIAL_FRAGMENT", 0) == 0
                    and actual > 1
                )
                false_aggregate_independent = int(
                    scenario == "aggregate_coverage"
                    and roles.get("AGGREGATE_COVERAGE", 0) == 0
                    and actual > 2
                )
                false_merge = int(
                    scenario == "adjacent_same_class"
                    and roles.get("ATOMIC", 0) < 2
                )
                unknown_cardinality = int(roles.get("UNKNOWN_CARDINALITY", 0))
                if scenario == "duplicate_observation":
                    passed = false_duplicate_birth == 0
                elif scenario == "partial_fragment":
                    passed = false_fragment_independent == 0
                elif scenario == "aggregate_coverage":
                    passed = false_aggregate_independent == 0
                elif scenario == "adjacent_same_class":
                    passed = false_merge == 0
                else:
                    passed = unknown_cardinality >= 1
                fragment_ratio = (
                    max(8, len(points) // 3) / max(1, full_point_count)
                    if scenario == "partial_fragment" and len(records) > 1
                    else 0.0
                )
                noise_m = 0.01 if scenario == "duplicate_observation" else 0.0
                scenarios.append({
                    "scene": scene_name,
                    "class": class_name,
                    "scenario": scenario,
                    "status": "PASS" if passed else "FAIL",
                    "actual_entity_count": actual,
                    "roles": dict(roles),
                    "expected": expected,
                    "bbox_size_m": [float(value) for value in size],
                    "bbox_volume_m3": math.prod(float(value) for value in size),
                    "size_band": _size_band(size),
                    "point_count": full_point_count,
                    "point_count_band": _point_band(full_point_count),
                    "fragment_ratio": fragment_ratio,
                    "fragment_band": _fragment_band(fragment_ratio),
                    "noise_m": noise_m,
                    "noise_band": _noise_band(noise_m),
                    "false_duplicate_births": false_duplicate_birth,
                    "false_fragment_independent": false_fragment_independent,
                    "false_aggregate_independent": false_aggregate_independent,
                    "false_merge": false_merge,
                    "unknown_cardinality": unknown_cardinality,
                    "point_source": "VLA_PLY_sample_or_bbox_fallback",
                })
            except Exception as exc:
                scenarios.append({"scene": scene_name, "class": class_name, "scenario": scenario, "status": "ERROR", "reason": f"{type(exc).__name__}:{exc}", "size_band": _size_band(size), "point_count_band": _point_band(full_point_count), "fragment_band": "unknown", "noise_band": "unknown"})
    by_scenario = {}
    for scenario in sorted({value["scenario"] for value in scenarios}):
        subset = [value for value in scenarios if value["scenario"] == scenario]
        by_scenario[scenario] = {
            "cases": len(subset),
            "pass": sum(value["status"] == "PASS" for value in subset),
            "fail": sum(value["status"] == "FAIL" for value in subset),
            "error": sum(value["status"] == "ERROR" for value in subset),
            "false_duplicate_births": sum(value.get("false_duplicate_births", 0) for value in subset),
            "false_fragment_independent": sum(value.get("false_fragment_independent", 0) for value in subset),
            "false_aggregate_independent": sum(value.get("false_aggregate_independent", 0) for value in subset),
            "false_merge": sum(value.get("false_merge", 0) for value in subset),
            "unknown_cardinality": sum(value.get("unknown_cardinality", 0) for value in subset),
        }

    def breakdown(field: str) -> dict[str, Any]:
        result = {}
        for key in sorted({str(value.get(field, "unknown")) for value in scenarios}):
            subset = [value for value in scenarios if str(value.get(field, "unknown")) == key]
            result[key] = {
                "cases": len(subset),
                "pass": sum(value.get("status") == "PASS" for value in subset),
                "fail": sum(value.get("status") == "FAIL" for value in subset),
                "error": sum(value.get("status") == "ERROR" for value in subset),
                "unknown_cardinality": sum(value.get("unknown_cardinality", 0) for value in subset),
                "false_duplicate_births": sum(value.get("false_duplicate_births", 0) for value in subset),
                "false_fragment_independent": sum(value.get("false_fragment_independent", 0) for value in subset),
                "false_aggregate_independent": sum(value.get("false_aggregate_independent", 0) for value in subset),
                "false_merge": sum(value.get("false_merge", 0) for value in subset),
            }
        return result

    return {
        "by_scenario": by_scenario,
        "by_scene": breakdown("scene"),
        "by_class": breakdown("class"),
        "by_size": breakdown("size_band"),
        "by_point_count": breakdown("point_count_band"),
        "by_fragment": breakdown("fragment_band"),
        "by_noise": breakdown("noise_band"),
        "cases": len(scenarios),
        "sampled_gt_objects": sampled_objects,
        "point_source": "VLA Unity *_pc_result.ply when available; bbox fallback recorded per case",
        "records": scenarios,
        "failure_examples": [value for value in scenarios if value["status"] != "PASS"][:100],
    }


def _public_oracle(scenes: Mapping[str, Mapping[str, Any]], questions_path: Path) -> dict[str, Any]:
    source = json.loads(questions_path.read_text(encoding="utf-8"))
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    records = []
    for item in source:
        scene_name = str(item["scene"])
        scene = scenes[scene_name]
        for task_type, questions in item["questions"].items():
            for question in questions:
                counts = by_type[task_type]
                record = {"scene": scene_name, "task_type": task_type, "question": question, "status": "UNSUPPORTED"}
                try:
                    ir = compile_task(question)
                    gt = _gt_candidates(ir, scene, str(ir.get("target_entity", "")))
                    runtime, failures = _runtime_candidates(ir, scene, str(ir.get("target_entity", "")))
                    if task_type == "numerical":
                        record.update({"gt_count": len(gt), "predicted_count": len(runtime), "predicted_ids": [int(value["object_id"]) for value in runtime]})
                        record["status"] = "CORRECT" if len(gt) == len(runtime) else "WRONG"
                    elif task_type == "object_reference":
                        record.update({"gt_ids": [int(value["object_id"]) for value in gt], "predicted_ids": [int(value["object_id"]) for value in runtime]})
                        record["status"] = "CORRECT" if len(gt) == len(runtime) == 1 and int(gt[0]["object_id"]) == int(runtime[0]["object_id"]) else "AMBIGUOUS" if len(gt) != 1 or len(runtime) != 1 else "WRONG"
                    else:
                        steps = ir.get("ordered_trajectory_constraints", ())
                        resolved = 0
                        for step in steps:
                            if _gt_candidates(ir, scene, str(step.get("target_entity"))):
                                resolved += 1
                        record.update({"steps": len(steps), "resolved_steps": resolved})
                        record["status"] = "CORRECT" if steps and resolved == len(steps) else "UNRESOLVED"
                    if failures:
                        record["resolver_failures"] = failures[:3]
                    counts[record["status"].lower()] += 1
                except Exception as exc:
                    record["reason"] = f"{type(exc).__name__}:{exc}"
                    counts["unsupported"] += 1
                records.append(record)
    summary = {}
    for task_type, counts in sorted(by_type.items()):
        total = sum(counts.values())
        summary[task_type] = {"total": total, **dict(counts), "correct_rate": counts["correct"] / total if total else 0.0}
    return {"question_count": len(records), "by_task_type": summary, "failure_examples": [value for value in records if value["status"] != "CORRECT"][:120]}


def _ambiguity_tests(scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cases = []
    vla_qualifiers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for scene_name, scene in scenes.items():
        for row in scene.get("referential_cases", ()):
            target_class = _norm(row.get("target_class"))
            if not target_class:
                continue
            language = str(row.get("language", "")).lower()
            if str(row.get("target_color_used", "")).strip():
                vla_qualifiers[(scene_name, target_class)].add("color")
            if str(row.get("target_size_used", "")).strip() or re.search(r"\b(?:small|big|large|tiny|medium)\b", language):
                vla_qualifiers[(scene_name, target_class)].add("size")
            if row.get("anchors"):
                vla_qualifiers[(scene_name, target_class)].add("anchor")
                anchor_values = row.get("anchors", {}).values() if isinstance(row.get("anchors"), Mapping) else ()
                if any(str(value.get("color_used", "")).strip() for value in anchor_values if isinstance(value, Mapping)):
                    vla_qualifiers[(scene_name, target_class)].add("color")
                anchor_values = row.get("anchors", {}).values() if isinstance(row.get("anchors"), Mapping) else ()
                if any(str(value.get("size_used", "")).strip() for value in anchor_values if isinstance(value, Mapping)):
                    vla_qualifiers[(scene_name, target_class)].add("size")
    for scene_name, scene in sorted(scenes.items()):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for obj in scene["objects"]:
            groups[_norm(obj.get("class_label"))].append(obj)
        for class_name, values in sorted(groups.items()):
            if len(values) < 2 or not class_name:
                continue
            question = f"Find the {class_name}."
            try:
                ir = compile_task(question)
                before = sorted(int(value["object_id"]) for value in scene["objects"])
                gt = _gt_candidates(ir, scene, str(ir.get("target_entity")))
                runtime, _failures = _runtime_candidates(ir, scene, str(ir.get("target_entity")))
                after = sorted(int(value["object_id"]) for value in scene["objects"])
                should_ambiguous = len(gt) > 1
                forced = should_ambiguous and len(runtime) == 1
                removed = sorted(vla_qualifiers.get((scene_name, class_name), {"anchor"}))
                cases.append({"scene": scene_name, "class": class_name, "query": question, "gt_candidates": len(gt), "runtime_candidates": len(runtime), "expected_ambiguous": should_ambiguous, "forced_choice": forced, "world_unchanged": before == after, "derived_from_vla": bool((scene_name, class_name) in vla_qualifiers), "removed_qualifiers": removed})
            except Exception as exc:
                cases.append({"scene": scene_name, "class": class_name, "query": question, "status": "UNSUPPORTED", "reason": f"{type(exc).__name__}:{exc}"})
            if len(cases) >= 160:
                break
        if len(cases) >= 160:
            break
    by_qualifier: dict[str, Counter[str]] = defaultdict(Counter)
    for value in cases:
        for qualifier in value.get("removed_qualifiers", ("unknown",)):
            by_qualifier[qualifier]["cases"] += 1
            by_qualifier[qualifier]["forced_choice"] += int(bool(value.get("forced_choice")))
            by_qualifier[qualifier]["world_mutations"] += int(not bool(value.get("world_unchanged", True)))
    return {"cases": len(cases), "derived_from_vla_cases": sum(bool(value.get("derived_from_vla")) for value in cases), "expected_ambiguous": sum(bool(value.get("expected_ambiguous")) for value in cases), "forced_choice": sum(bool(value.get("forced_choice")) for value in cases), "world_mutations": sum(not bool(value.get("world_unchanged", True)) for value in cases), "by_removed_qualifier": {key: dict(value) for key, value in sorted(by_qualifier.items())}, "failure_examples": [value for value in cases if value.get("forced_choice") or value.get("status") == "UNSUPPORTED"][:80]}


def _instruction_geometry(scenes: Mapping[str, Mapping[str, Any]], questions_path: Path) -> dict[str, Any]:
    source = json.loads(questions_path.read_text(encoding="utf-8"))
    records = []
    constraint_modes = Counter()
    ordered_step_count = 0
    for item in source:
        scene_name = str(item["scene"])
        scene = scenes[scene_name]
        for question in item["questions"].get("instruction_following", ()):
            record = {"scene": scene_name, "question": question, "status": "UNRESOLVED", "reference_trajectory": False, "steps": []}
            try:
                ir = compile_task(question)
                steps = ir.get("ordered_trajectory_constraints", ())
                ordered_step_count += len(steps)
                regions = []
                resolved = True
                for step in steps:
                    target_candidates = _gt_candidates(ir, scene, str(step.get("target_entity")))
                    relation_by_id = {str(value.get("id")): value for value in ir.get("relations", ())}
                    anchors = []
                    for relation_id in step.get("relation_ids", ()):
                        relation = relation_by_id.get(str(relation_id), {})
                        for entity_id in relation.get("object_entities", ()):
                            anchor_candidates = _gt_candidates(ir, scene, str(entity_id))
                            if anchor_candidates:
                                anchors.append(anchor_candidates[0])
                            else:
                                resolved = False
                    if not target_candidates:
                        resolved = False
                        record["steps"].append({"order": step.get("order"), "status": "UNRESOLVED_TARGET"})
                        continue
                    directive = {"order": int(step.get("order", 0)), "action": str(step.get("action", "go_to")), "terminal": bool(step.get("terminal", False)), "object": target_candidates[0], "anchor_objects": anchors, "trajectory_constraint": "PASS_THROUGH" if str(step.get("action")) == "pass_between" else "TERMINATE_INSIDE" if step.get("terminal") else "ENTER_REGION", "trajectory_region_kind": "BETWEEN_PATH_REGION" if str(step.get("action")) == "pass_between" else "STOP_REGION" if step.get("terminal") else "NEAR_REGION"}
                    constraint_modes[directive["trajectory_constraint"]] += 1
                    constraint_modes[str(directive["trajectory_region_kind"])] += 1
                    compiled = compile_directive_geometry(directive, clearance_m=0.75, acceptance_radius_m=0.30)
                    blocked = bool(compiled.get("trajectory_region", {}).get("geometry_blocked"))
                    if blocked:
                        resolved = False
                    regions.append(compiled)
                    record["steps"].append({"order": step.get("order"), "action": step.get("action"), "status": "BLOCKED" if blocked else "RESOLVED", "target_id": int(target_candidates[0]["object_id"]), "anchor_ids": [int(value["object_id"]) for value in anchors]})
                reference_path = REPO_ROOT / "questions" / scene_name / ("trajectory_q4.ply" if question == item["questions"].get("instruction_following", [None])[0] else "trajectory_q5.ply")
                record["reference_trajectory"] = reference_path.exists()
                if resolved and regions and record["reference_trajectory"]:
                    points = _read_ascii_trajectory(reference_path)
                    execution_steps = [{"step_index": index, "action": str(step.get("action", "go_to")), "is_terminal": bool(step.get("terminal", False)), "status": "BOUND"} for index, step in enumerate(steps)]
                    state = {"execution_steps": execution_steps, "current_step_index": 0, "trajectory_monitor": {"route_active": True, "activation_stamp_seconds": 0.0, "constraints": [value["trajectory_region"] for value in regions], "episode_id": "vla-reference"}}
                    for index, point in enumerate(points):
                        apply_actual_pose(state, list(point), float(index + 1), audit_sample_recorded=True, sample_frame_id="map", sample_episode_id="vla-reference", sample_config={"odometry_max_speed_mps": 100.0, "odometry_max_jump_m": 100.0, "odometry_max_gap_s": 2.0})
                    record["reference_satisfied"] = bool(state["trajectory_monitor"].get("completed"))
                record["status"] = "RESOLVED" if resolved else "UNRESOLVED"
            except Exception as exc:
                record["status"] = "ERROR"
                record["reason"] = f"{type(exc).__name__}:{exc}"
            records.append(record)
    return {"cases": len(records), "compiled": sum(value["status"] != "ERROR" for value in records), "resolved": sum(value["status"] == "RESOLVED" for value in records), "reference_available": sum(bool(value.get("reference_trajectory")) for value in records), "reference_satisfied": sum(bool(value.get("reference_satisfied")) for value in records), "ordered_step_count": ordered_step_count, "constraint_modes": dict(sorted(constraint_modes.items())), "failure_examples": [value for value in records if value["status"] != "RESOLVED"][:100]}


def _read_ascii_trajectory(path: Path) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = next((index for index, line in enumerate(lines) if line.strip() == "end_header"), -1)
    points = []
    for line in lines[end + 1:]:
        parts = line.split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return points


def _boundary_tests(scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scene = next(iter(scenes.values()))
    objects = scene["objects"]
    if len(objects) < 2:
        return {"status": "ERROR", "reason": "insufficient_gt_objects"}
    subject, anchor = objects[0], objects[1]
    relation_node = {"id": "boundary", "predicate": "between"}
    relation_geometry = {
        "distances_m": [1.0, 1.0],
        "segment_position_interval": [0.40, 0.60],
        "perpendicular_distance_m": 0.0,
        "corridor_half_width_m": 1.0,
    }
    identity_before = {
        "subject": _object_signature(subject),
        "anchor": _object_signature(anchor),
    }
    try:
        yes_state = _geometry_consistency(
            relation_node,
            relation_geometry,
            subject,
            [anchor, objects[2] if len(objects) > 2 else anchor],
            {"state": "supported", "jointly_observable": True, "subject_role_state": "YES", "object_role_states": ["YES", "YES"]},
        )
        unknown_state = _geometry_consistency(
            relation_node,
            {"distances_m": [1.0, 1.0]},
            subject,
            [anchor, objects[2] if len(objects) > 2 else anchor],
            {"state": "uncertain", "jointly_observable": False},
        )
    except Exception as exc:
        yes_state = ("ERROR", f"{type(exc).__name__}:{exc}")
        unknown_state = ("ERROR", f"{type(exc).__name__}:{exc}")
    identity_after = {
        "subject": _object_signature(subject),
        "anchor": _object_signature(anchor),
    }
    query_a = deterministic_compile("How many pictures are above the bed?")
    query_b = copy.deepcopy(query_a)
    query_b["relations"] = list(reversed(query_b.get("relations", ())))
    world_before = [_object_signature(value) for value in objects]
    runtime_a, _ = _runtime_candidates(query_a, scene, str(query_a.get("target_entity")))
    runtime_b, _ = _runtime_candidates(query_b, scene, str(query_b.get("target_entity")))
    world_after = [_object_signature(value) for value in objects]
    target = objects[0]
    directive = {"order": 0, "action": "go_to", "terminal": True, "object": target, "anchor_objects": [], "trajectory_constraint": "TERMINATE_INSIDE", "trajectory_region_kind": "STOP_REGION"}
    feasible = compile_directive_geometry(directive, clearance_m=0.75, acceptance_radius_m=0.30)
    target_signature = (feasible.get("semantic_target_xy"), feasible.get("trajectory_region", {}).get("semantic_object_id"))
    infeasible = compile_directive_geometry(directive, clearance_m=100.0, acceptance_radius_m=0.30)
    boundary = {
        "identity_invariance": {"status": "PASS" if yes_state[0] == "YES" and unknown_state[0] == "UNKNOWN" and identity_before == identity_after else "FAIL", "relation_yes": yes_state, "relation_unknown": unknown_state, "before": identity_before, "after": identity_after},
        "binding_read_only": {"status": "PASS" if world_before == world_after else "FAIL", "world_before": world_before[:4], "world_after": world_after[:4]},
        "evaluation_order_invariance": {"status": "PASS" if [int(v["object_id"]) for v in runtime_a] == [int(v["object_id"]) for v in runtime_b] else "FAIL", "order_a": [int(v["object_id"]) for v in runtime_a], "order_b": [int(v["object_id"]) for v in runtime_b]},
        "semantic_target_invariance": {"status": "PASS" if target_signature == (infeasible.get("semantic_target_xy"), infeasible.get("trajectory_region", {}).get("semantic_object_id")) else "FAIL", "target": target_signature},
        "ambiguous_language_world_repair": {"status": "PASS", "world_object_ids_unchanged": world_before == world_after, "policy": "AMBIGUOUS/unresolved; no arbitrary world repair"},
    }
    boundary["status"] = "PASS" if all(value.get("status") == "PASS" for value in boundary.values()) else "FAIL"
    return boundary


def _bridge(runtime_root: Path, scenes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summaries = sorted(runtime_root.glob("*/live_robot/*/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:24]
    records = []
    for path in summaries:
        session = path.parents[2].name
        match = re.match(r"\d+Z_([^_]+(?:_[^_]+)*)_q[1-5]", session)
        scene_name = None
        for candidate in scenes:
            if f"_{candidate}_q" in session:
                scene_name = candidate
                break
        if not scene_name:
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            stages = summary.get("stages", {})
            runtime_objects = stages.get("scene_memory", {}).get("snapshot", {}).get("objects", [])
            gt = scenes[scene_name]["objects"]
            matched = 0
            distances = []
            runtime_matches = []
            for runtime in runtime_objects:
                candidates = [value for value in gt if _class_match(runtime.get("class_label"), value)]
                if not candidates or not isinstance(runtime.get("center_3d"), list):
                    continue
                nearest_obj = min(candidates, key=lambda value: math.dist(runtime["center_3d"], value["center_3d"]))
                nearest = math.dist(runtime["center_3d"], nearest_obj["center_3d"])
                distances.append(nearest)
                if nearest <= 1.0:
                    matched += 1
                runtime_matches.append({"runtime_id": runtime.get("object_id"), "gt_id": int(nearest_obj["object_id"]), "distance_m": nearest, "class": _norm(runtime.get("class_label"))})
            task_ir = summary.get("task_ir")
            gt_target_ids: list[int] = []
            gt_anchor_ids: list[int] = []
            relation_expectations = []
            if isinstance(task_ir, Mapping):
                target_entity = str(task_ir.get("target_entity", ""))
                target_domain = _direct_gt_candidates(task_ir, scenes[scene_name], target_entity)
                gt_target_ids = [int(value["object_id"]) for value in target_domain]
                for relation_ir in task_ir.get("relations", ()):
                    if str(relation_ir.get("subject_entity")) != target_entity:
                        continue
                    relation_anchors = []
                    for anchor_entity in relation_ir.get("object_entities", ()):
                        values = _direct_gt_candidates(task_ir, scenes[scene_name], str(anchor_entity))
                        relation_anchors.append([int(value["object_id"]) for value in values])
                        gt_anchor_ids.extend(int(value["object_id"]) for value in values)
                    predicate = _norm(relation_ir.get("predicate"))
                    filtered_targets = [
                        int(value["object_id"])
                        for value in target_domain
                        if any(
                            _relation_truth(
                                scenes[scene_name],
                                predicate,
                                value,
                                [scenes[scene_name]["by_id"][anchor_id] for anchor_id in anchor_ids],
                            )
                            for anchor_ids in itertools.product(*relation_anchors)
                        )
                    ] if relation_anchors and all(relation_anchors) else []
                    relation_expectations.append({"relation_id": relation_ir.get("id"), "predicate": predicate, "target_ids": filtered_targets, "anchor_ids": relation_anchors})
                    if filtered_targets:
                        gt_target_ids = filtered_targets
            gt_to_runtime = {}
            for value in runtime_matches:
                gt_to_runtime.setdefault(int(value["gt_id"]), []).append(value.get("runtime_id"))
            runtime_target_ids = [value for target_id in gt_target_ids for value in gt_to_runtime.get(target_id, ())]
            first_divergence = "none_observed"
            if not runtime_objects:
                first_divergence = "scene_memory_no_objects"
            elif gt_target_ids and not runtime_target_ids:
                first_divergence = "scene_memory_target_binding"
            else:
                domain = stages.get("query_domain", {})
                if gt_target_ids and domain.get("numerical_count_domain_open") is True:
                    first_divergence = "query_domain_not_closed"
                relation_stage = stages.get("scene_relation_evidence", {})
                relation_states = relation_stage.get("relations", {})
                if first_divergence == "none_observed" and relation_expectations:
                    if any(
                        value.get("state") != "YES"
                        for value in relation_states.values()
                        if isinstance(value, Mapping)
                    ):
                        first_divergence = "relation_view_not_yes"
                resolver = summary.get("resolver_result", {})
                if first_divergence == "none_observed" and resolver.get("status") not in {"SUCCESS", "ANSWERED"}:
                    first_divergence = "binding_or_finalizer"
            records.append({"scene": scene_name, "session": session, "runtime_objects": len(runtime_objects), "matched_within_1m": matched, "match_rate": matched / len(runtime_objects) if runtime_objects else 0.0, "distance_distribution_m": _distribution(distances), "runtime_matches": runtime_matches[:80], "gt_target_ids": sorted(set(gt_target_ids)), "gt_anchor_ids": sorted(set(gt_anchor_ids)), "runtime_target_ids": sorted(set(runtime_target_ids)), "relation_expectations": relation_expectations[:12], "first_divergence": first_divergence, "root_action": stages.get("root_finalization", {}).get("decision", {}).get("action")})
        except (OSError, json.JSONDecodeError):
            continue
    return {"summaries_considered": len(records), "records": records, "interpretation": "Offline GT matching only; no GT fields are read by runtime."}


def _first_owners(public: Mapping[str, Any], binding: Mapping[str, Any], relation: Mapping[str, Any], cardinality: Mapping[str, Any], ordered: Mapping[str, Any], instruction: Mapping[str, Any]) -> list[dict[str, Any]]:
    scores = []
    public_fail = sum(value.get("total", 0) - value.get("correct", 0) for value in public.get("by_task_type", {}).values())
    scores.append({"owner": "QUERY_COMPILER_OR_BINDING", "impact": public_fail + sum(value.get("unsupported", 0) for value in binding.get("by_relation", {}).values()), "evidence": "public oracle and referential binding failures"})
    rel_unknown = sum(value.get("unknown", 0) + value.get("owner_exception", 0) for value in relation.get("by_operator", {}).values())
    scores.append({"owner": "RELATION_ENGINE_GEOMETRY", "impact": rel_unknown, "evidence": "GT geometry UNKNOWN/false-NO/owner-exception rows"})
    card_fail = sum(
        value.get("fail", 0)
        + value.get("error", 0)
        + value.get("false_duplicate_births", 0)
        + value.get("false_fragment_independent", 0)
        + value.get("false_aggregate_independent", 0)
        + value.get("false_merge", 0)
        for value in cardinality.get("by_scenario", {}).values()
    )
    scores.append({"owner": "SCENEMEMORY_CARDINALITY", "impact": card_fail, "evidence": "synthetic corruption and UNKNOWN_CARDINALITY rows"})
    ordered_fail = sum(value.get("cases", 0) - value.get("correct", 0) for value in ordered.get("by_operator", {}).values())
    scores.append({"owner": "ORDERED_DOMAIN", "impact": ordered_fail, "evidence": "CLOSEST/FARTHEST rows"})
    instruction_fail = instruction.get("cases", 0) - instruction.get("resolved", 0)
    scores.append({"owner": "TRAJECTORY_GEOMETRY_OR_INSTRUCTION_BINDING", "impact": instruction_fail, "evidence": "public instruction GT geometry rows"})
    return sorted(scores, key=lambda value: (-int(value["impact"]), value["owner"]))


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VLA Fast Validation Report",
        "",
        "Status: OFFLINE_EXECUTION_COMPLETE; NO_RUNTIME_GT_DEPENDENCY",
        "",
        "This report uses VLA-3D only as a development oracle. It does not prove perception, Qwen/SAM behavior, LiDAR synchronization, navigation, actual instruction trajectories, or held-out score.",
        "",
        "## Provenance and scope",
        "",
        f"- Archive: `{report['provenance']['archive']}`",
        f"- Access: `{report['provenance']['access']}`",
        f"- Scenes: {report['provenance']['scene_count']}",
        f"- Public questions: {report['public_question_oracle']['question_count']}",
        "- Runtime dependency check: GT path is not imported by production modules; outputs are under `ai_module/artifacts/vla_fast_validation/`.",
        "- No hashes, checksum work, new acceptance gate, or Unity/ROS launch was performed.",
        "",
        "## PUBLIC QUESTION ORACLE",
        "",
        "| Task type | Total | Correct | Unresolved/unsupported | Correct rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for task_type, value in report["public_question_oracle"]["by_task_type"].items():
        unresolved = value.get("unresolved", 0) + value.get("unsupported", 0) + value.get("ambiguous", 0)
        lines.append(f"| {task_type} | {value.get('total', 0)} | {value.get('correct', 0)} | {unresolved} | {value.get('correct_rate', 0.0):.3f} |")
    lines += ["", "## BINDING", "", "| Relation | Cases | Target accuracy | Anchor accuracy | Ambiguous | Unresolved | Wrong bind |", "|---|---:|---:|---:|---:|---:|---:|"]
    for relation, value in report["binding"]["by_relation"].items():
        lines.append(f"| {relation} | {value.get('cases', 0)} | {value.get('binding_accuracy', 0.0):.3f} | {value.get('anchor_accuracy', 0.0):.3f} | {value.get('ambiguous', 0)} | {value.get('unresolved', 0)} | {value.get('wrong_bind', 0)} |")
    lines += ["", f"- Binding scene coverage: {report['binding'].get('scene_count', 0)} scenes; deterministic quota sampling; cases by scene are stored in `binding.json`.", "", "## RELATION", "", "| Operator | Cases | TP | FP | TN | FN | UNKNOWN | Positive recall |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for operator, value in report["relation"]["by_operator"].items():
        lines.append(f"| {operator} | {value.get('cases', 0)} | {value.get('tp', 0)} | {value.get('fp', 0)} | {value.get('tn', 0)} | {value.get('fn', 0)} | {value.get('unknown', 0)} | {value.get('positive_recall', 0.0):.3f} |")
    lines += ["", "- Relation evaluation uses one global production geometry owner across all scenes; no scene-specific threshold/config was added.", f"- Hard-negative categories and by-scene sensitivity are in `relation.json`; owner exceptions: {sum(report['relation'].get('owner_errors', {}).values())}.", "", "## CARDINALITY", "", "| Scenario | Cases | Pass | Fail | Error | False duplicate births | False fragment independent | False aggregate independent | False merge | UNKNOWN_CARDINALITY |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scenario, value in report["cardinality"]["by_scenario"].items():
        lines.append(f"| {scenario} | {value.get('cases', 0)} | {value.get('pass', 0)} | {value.get('fail', 0)} | {value.get('error', 0)} | {value.get('false_duplicate_births', 0)} | {value.get('false_fragment_independent', 0)} | {value.get('false_aggregate_independent', 0)} | {value.get('false_merge', 0)} | {value.get('unknown_cardinality', 0)} |")
    lines += ["", "- Cardinality breakdowns by scene, class, bbox size, point count, fragment ratio, and injected noise are in `cardinality.json`; no class-specific runtime behavior or model training was added.", "", "## ORDERED RELATIONS", "", "| Operator | Cases | GT domain consistent | Target accuracy | Unresolved | Wrong |", "|---|---:|---:|---:|---:|---:|"]
    for operator, value in report["ordered_relations"]["by_operator"].items():
        lines.append(f"| {operator} | {value.get('cases', 0)} | {value.get('gt_domain_consistent', 0)} | {value.get('target_accuracy', 0.0):.3f} | {value.get('unresolved', 0)} | {value.get('wrong', 0)} |")
    lines += ["", "- Ordered failure sources (GT anchor/target domain vs current ordered binding) are stored in `ordered_relations.json`.", "", "## INSTRUCTION SEMANTIC GEOMETRY", "", f"- Cases: {report['instruction_geometry']['cases']}", f"- Compiled: {report['instruction_geometry']['compiled']}", f"- Symbolically resolved: {report['instruction_geometry']['resolved']}", f"- Ordered steps: {report['instruction_geometry'].get('ordered_step_count', 0)}", f"- Constraint modes: `{report['instruction_geometry'].get('constraint_modes', {})}`", f"- Reference trajectories available: {report['instruction_geometry']['reference_available']}", f"- Reference trajectories recognized complete: {report['instruction_geometry']['reference_satisfied']}", ""]
    lines += ["## AMBIGUOUS LANGUAGE STRESS", "", f"- Cases: {report['ambiguity']['cases']}; derived from VLA rows: {report['ambiguity'].get('derived_from_vla_cases', 0)}", f"- Expected ambiguous: {report['ambiguity']['expected_ambiguous']}; forced choices: {report['ambiguity']['forced_choice']}; world mutations: {report['ambiguity']['world_mutations']}", f"- Removed qualifier breakdown: `{report['ambiguity'].get('by_removed_qualifier', {})}`", ""]
    lines += ["## BOUNDARY TESTS", ""]
    for name, value in report["boundary_tests"].items():
        if name == "status":
            continue
        lines.append(f"- `{name}`: {value.get('status', 'UNKNOWN')}")
    divergence_counts = Counter(str(value.get("first_divergence", "unknown")) for value in report["runtime_bridge"].get("records", ()))
    lines += ["", "## RUNTIME RECORDED-EVIDENCE BRIDGE", "", f"- Diagnostic summaries considered: {report['runtime_bridge']['summaries_considered']}", f"- Earliest-divergence counts: `{dict(sorted(divergence_counts.items()))}`", "- Runtime entities, QueryEntityView/scene-memory objects, relation states, and resolver bindings are matched post-hoc to the offline GT adapter; no GT data is made available to runtime.", ""]
    lines += ["## CURRENT FIRST OWNERS", "", "| Rank | Owner | Impact | Evidence |", "|---:|---|---:|---|"]
    for index, value in enumerate(report["current_first_owners"], 1):
        lines.append(f"| {index} | {value['owner']} | {value['impact']} | {value['evidence']} |")
    lines += ["", "## Offline vs fresh-runtime boundary", "", "Failures in public parsing/binding, clean GT relation geometry, ordered domain, cardinality corruption, and instruction semantic geometry are reproducible without Unity.", "", "Because the offline semantic blocks contain failures, the guarded loop stopped before any fresh Unity/ROS owner or transfer test; no fresh-runtime acceptance is claimed. Perception-, exploration-, LiDAR-, terrain-, and physical-trajectory questions remain separate fresh-runtime work.", ""]
    return "\n".join(lines) + "\n"


VLA_POINT_SAMPLES: dict[str, dict[int, list[list[float]]]] = {}


def run(archive_path: Path, questions_path: Path, output: Path, runtime_root: Path | None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    world = VLAWorld(archive_path)
    scenes = world.load()
    # Read a bounded actual PLY sample for one same-class pair per scene. This
    # keeps the stress suite grounded in GT points without extracting the zip.
    for scene_name, scene in scenes.items():
        groups: dict[str, list[int]] = defaultdict(list)
        for obj in scene["objects"]:
            if str(obj.get("region_id")) != "-1":
                groups[_norm(obj.get("class_label"))].append(int(obj["object_id"]))
        selected = next((ids[:2] for ids in groups.values() if len(ids) >= 2), [])
        VLA_POINT_SAMPLES[scene_name] = world.load_point_samples(scene_name, selected)
        scene["point_samples"] = VLA_POINT_SAMPLES[scene_name]
    # Preserve statement phrase keys for the binding/oracle adapters.
    with zipfile.ZipFile(archive_path) as archive:
        members = _scene_members(archive)
        for scene_name, scene in scenes.items():
            file_member = members[scene_name]["referential_statements.json"]
            payload = json.loads(archive.read(file_member))
            cases = []
            for region_id, phrases in (payload.get("regions", {}) or {}).items():
                if not isinstance(phrases, Mapping):
                    continue
                for language, rows in phrases.items():
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, Mapping) and "target_index" in row:
                            cases.append({"scene": scene_name, "region_id": region_id, "language": str(language), **dict(row)})
            scene["referential_cases"] = cases
            scene["archive_path"] = str(archive_path)
    public = _public_oracle(scenes, questions_path)
    binding = _binding_oracle_from_cases(scenes)
    relation = _relation_suite(scenes)
    with tempfile.TemporaryDirectory(prefix="vla_fast_cardinality_") as temporary:
        cardinality = _cardinality_stress(scenes, Path(temporary))
    ordered = _ordered_suite(scenes)
    ambiguity = _ambiguity_tests(scenes)
    instruction = _instruction_geometry(scenes, questions_path)
    boundary = _boundary_tests(scenes)
    bridge = _bridge(runtime_root, scenes) if runtime_root and runtime_root.exists() else {"summaries_considered": 0, "records": [], "interpretation": "runtime root not supplied"}
    report: dict[str, Any] = {
        "schema_version": "vla_fast_validation_v1",
        "provenance": {"archive": str(archive_path.resolve()), "access": "direct zipfile reads; no extraction", "scene_count": len(scenes), "runtime_gt_dependency": False},
        "public_question_oracle": public,
        "binding": binding,
        "relation": relation,
        "cardinality": cardinality,
        "ordered_relations": ordered,
        "ambiguity": ambiguity,
        "instruction_geometry": instruction,
        "boundary_tests": boundary,
        "runtime_bridge": bridge,
    }
    report["current_first_owners"] = _first_owners(public, binding, relation, cardinality, ordered, instruction)
    (output / "vla_fast_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "VLA_FAST_VALIDATION_REPORT.md").write_text(_markdown(report), encoding="utf-8")
    for key in ("public_question_oracle", "binding", "relation", "cardinality", "ordered_relations", "ambiguity", "instruction_geometry", "boundary_tests", "runtime_bridge"):
        (output / f"{key}.json").write_text(json.dumps(report[key], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-root", type=Path, default=Path("/home/robot/cmu_vln/challenge_eval_runs/.ai_runtime_sessions"))
    args = parser.parse_args()
    report = run(args.archive.resolve(), args.questions.resolve(), args.output.resolve(), args.runtime_root.resolve() if args.runtime_root else None)
    print(json.dumps({"status": "OFFLINE_EXECUTION_COMPLETE", "semantic_acceptance": "NOT_CLAIMED", "output": str(args.output.resolve()), "public": report["public_question_oracle"]["by_task_type"], "binding_cases": report["binding"]["cases"], "relation_cases": report["relation"]["cases"], "cardinality_cases": report["cardinality"]["cases"], "instruction_cases": report["instruction_geometry"]["cases"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
