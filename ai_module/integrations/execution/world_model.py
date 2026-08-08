"""Persistent object identity for map-frame MASt3R observations.

The store deliberately keeps ambiguous observations outside the canonical
object namespace.  A physical object is publishable only after compatible
observations from spatially distinct robot viewpoints agree.
"""

from __future__ import annotations

import copy
import fcntl
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mast3r_world_model_v1"


def _finite_vector(value: object, size: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"expected_{size}d_vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("nonfinite_geometry")
    if positive and any(item <= 0.0 for item in result):
        raise ValueError("degenerate_geometry")
    return result


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.dist(tuple(float(v) for v in first[:3]), tuple(float(v) for v in second[:3]))


def _task_key(task_ir: Mapping[str, Any]) -> str:
    identity = {
        "task_type": task_ir["task_type"],
        "entities": task_ir["entities"],
        "relations": task_ir["relations"],
        "ordered_subgoals": task_ir["ordered_subgoals"],
        "target_entity": task_ir["target_entity"],
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _empty_store() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 0,
        "next_object_id": 0,
        "objects": [],
        "ambiguous_observations": [],
        "task_histories": {},
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_store()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("world_model_schema_mismatch")
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("frame") != "map" or raw.get("world_aligned") is not True:
        raise ValueError("observation_not_in_map_frame")
    class_name = str(raw.get("canonical_class", "")).strip().lower()
    observation_id = str(raw.get("observation_id", "")).strip()
    if not class_name or not observation_id:
        raise ValueError("observation_identity_incomplete")
    center = _finite_vector(raw.get("center_3d"), 3)
    bbox = _finite_vector(raw.get("bbox_3d"), 3, positive=True)
    viewpoint = _finite_vector(raw.get("viewpoint_position_map"), 3)
    probability = raw.get("semantic_probability")
    if probability is None or not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("semantic_probability_invalid")
    return {
        "observation_id": observation_id,
        "class_label": class_name,
        "center_3d": center,
        "bbox_3d": bbox,
        "viewpoint_position_map": viewpoint,
        "optical_center_group": str(raw.get("optical_center_group", "")),
        "semantic_probability": float(probability),
        "pointcloud_path": str(raw.get("pointcloud_path", "")),
        "panorama_mask_path": str(raw.get("panorama_mask_path", "")),
        "reconstruction_id": str(raw.get("reconstruction_id", "")),
        "source_cam2w_maps": copy.deepcopy(raw.get("source_cam2w_maps", [])),
        "source_view_ids": list(raw.get("source_view_ids", ())),
        "geometry_confidence": float(raw.get("median_confidence", 0.0)),
        "view_center_dispersion_m": float(raw.get("view_center_dispersion_m", 0.0)),
        "registered_scan_median_distance_m": raw.get("registered_scan_median_distance_m"),
    }


def _association_limit(obj: Mapping[str, Any], observation: Mapping[str, Any]) -> float:
    old_radius = 0.5 * math.hypot(float(obj["bbox_3d"][0]), float(obj["bbox_3d"][1]))
    new_radius = 0.5 * math.hypot(float(observation["bbox_3d"][0]), float(observation["bbox_3d"][1]))
    return min(1.5, max(0.45, old_radius + new_radius + 0.20))


def _independent_view_count(evidence: Sequence[Mapping[str, Any]], minimum_separation_m: float) -> int:
    representatives: list[list[float]] = []
    for item in evidence:
        viewpoint = list(item["viewpoint_position_map"])
        if representatives and all(
            _distance(viewpoint, existing) < minimum_separation_m
            for existing in representatives
        ):
            continue
        representatives.append(viewpoint)
    return len(representatives)


def _merge_object(obj: dict[str, Any], observation: Mapping[str, Any], acquisition_id: str, minimum_separation_m: float) -> None:
    evidence = [
        item for item in obj.get("evidence", [])
        if str(item.get("observation_id")) != str(observation["observation_id"])
    ]
    evidence.append({**copy.deepcopy(dict(observation)), "acquisition_id": acquisition_id})
    weights = [
        max(1e-6, float(item["semantic_probability"]))
        * max(1e-3, float(item.get("geometry_confidence", 0.0)))
        * math.exp(-max(0.0, float(item.get("view_center_dispersion_m", 0.0))) / 0.5)
        for item in evidence
    ]
    total = sum(weights)
    obj["center_3d"] = [
        sum(weight * float(item["center_3d"][axis]) for weight, item in zip(weights, evidence)) / total
        for axis in range(3)
    ]
    obj["bbox_3d"] = [
        sorted(float(item["bbox_3d"][axis]) for item in evidence)[len(evidence) // 2]
        for axis in range(3)
    ]
    obj["semantic_probability"] = min(float(item["semantic_probability"]) for item in evidence)
    obj["evidence"] = evidence
    obj["instance_version"] = int(obj.get("instance_version", 0)) + 1
    independent = _independent_view_count(evidence, minimum_separation_m)
    obj["independent_viewpoint_count"] = independent
    obj["status"] = "confirmed" if independent >= 2 else "tentative"


def _new_object(store: dict[str, Any], observation: Mapping[str, Any], acquisition_id: str, minimum_separation_m: float) -> dict[str, Any]:
    object_id = int(store["next_object_id"])
    store["next_object_id"] = object_id + 1
    obj = {
        "object_id": object_id,
        "class_label": observation["class_label"],
        "center_3d": list(observation["center_3d"]),
        "bbox_3d": list(observation["bbox_3d"]),
        "semantic_probability": float(observation["semantic_probability"]),
        "instance_version": 0,
        "status": "tentative",
        "independent_viewpoint_count": 1,
        "evidence": [],
    }
    _merge_object(obj, observation, acquisition_id, minimum_separation_m)
    store["objects"].append(obj)
    return obj


def update_world_model(
    store_path: Path,
    observations: Sequence[Mapping[str, Any]],
    task_ir: Mapping[str, Any],
    *,
    acquisition_id: str,
    task_scope_id: str = "default",
    minimum_viewpoint_separation_m: float = 0.30,
) -> dict[str, Any]:
    """Atomically ingest one acquisition and return an immutable snapshot."""
    if not acquisition_id.strip():
        raise ValueError("acquisition_id_empty")
    store_path = store_path.resolve()
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        store = _load(store_path)
        validated = []
        rejected = []
        for raw in observations:
            try:
                validated.append(_validate_observation(raw))
            except (TypeError, ValueError) as exc:
                rejected.append({"observation_id": raw.get("observation_id"), "reason": str(exc)})

        associated_ids: list[int] = []
        ambiguous = []
        used_groups_by_object: dict[int, set[str]] = {}
        for observation in validated:
            ranked = []
            for obj in store["objects"]:
                object_id = int(obj["object_id"])
                if (
                    observation["optical_center_group"] in used_groups_by_object.get(object_id, set())
                    or obj["class_label"] != observation["class_label"]
                ):
                    continue
                distance = _distance(obj["center_3d"], observation["center_3d"])
                if distance <= _association_limit(obj, observation):
                    ranked.append((distance, obj))
            ranked.sort(key=lambda value: (value[0], int(value[1]["object_id"])))
            if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.25:
                record = {
                    "acquisition_id": acquisition_id,
                    "observation": observation,
                    "candidate_object_ids": [int(item[1]["object_id"]) for item in ranked],
                    "reason": "association_ambiguous",
                }
                ambiguous.append(record)
                store["ambiguous_observations"].append(record)
                continue
            obj = ranked[0][1] if ranked else _new_object(
                store, observation, acquisition_id, minimum_viewpoint_separation_m
            )
            if ranked:
                _merge_object(obj, observation, acquisition_id, minimum_viewpoint_separation_m)
            object_id = int(obj["object_id"])
            used_groups_by_object.setdefault(object_id, set()).add(
                observation["optical_center_group"]
            )
            associated_ids.append(object_id)

        store["version"] = int(store["version"]) + 1
        if not str(task_scope_id).strip():
            raise ValueError("task_scope_id_empty")
        key = f"{task_scope_id}:{_task_key(task_ir)}"
        target_id = str(task_ir["target_entity"])
        target_entity = next(item for item in task_ir["entities"] if item["id"] == target_id)
        target_class = str(target_entity["class_name"]).lower()
        target_ids = sorted(
            object_id for object_id in associated_ids
            if next(obj for obj in store["objects"] if int(obj["object_id"]) == object_id)["class_label"] == target_class
        )
        required_classes = [str(value).lower() for value in task_ir["required_classes"]]
        object_lookup = {int(obj["object_id"]): obj for obj in store["objects"]}
        object_ids_by_class = {
            class_name: sorted(
                object_id for object_id in associated_ids
                if object_lookup[object_id]["class_label"] == class_name
            )
            for class_name in required_classes
        }
        viewpoint = validated[0]["viewpoint_position_map"] if validated else None
        history = store["task_histories"].setdefault(key, [])
        history.append({
            "acquisition_id": acquisition_id,
            "viewpoint_position_map": viewpoint,
            "target_object_ids": target_ids,
            "object_ids_by_class": object_ids_by_class,
            "ambiguous": bool(ambiguous or rejected),
        })
        history[:] = history[-12:]

        closure = False
        closure_reasons = []
        missing_required = [
            class_name for class_name, object_ids in object_ids_by_class.items()
            if not object_ids
        ]
        if missing_required:
            closure_reasons.extend(
                f"required_candidate_set_empty:{class_name}"
                for class_name in missing_required
            )
        if ambiguous:
            closure_reasons.append("association_ambiguous")
        if rejected:
            closure_reasons.append("invalid_observation_rejected")
        if not missing_required and not closure_reasons:
            current = history[-1]
            for previous in reversed(history[:-1]):
                if previous["ambiguous"]:
                    continue
                # Closure requires each required class to have at least one
                # shared persistent object_id across two independent viewpoints.
                # Exact set equality is too strict: a different viewpoint will
                # often reveal one extra instance or miss an occluded one.
                prev_by_class = previous.get("object_ids_by_class", {})
                if not all(
                    set(prev_by_class.get(cls, [])) & set(object_ids_by_class.get(cls, []))
                    for cls in required_classes
                ):
                    continue
                if previous["viewpoint_position_map"] is None or viewpoint is None:
                    continue
                if _distance(previous["viewpoint_position_map"], viewpoint) >= minimum_viewpoint_separation_m:
                    closure = True
                    break
            if not closure:
                closure_reasons.append("independent_repeat_candidate_set_missing")

        by_id = object_lookup
        required_object_ids = {
            object_id
            for values in object_ids_by_class.values()
            for object_id in values
        }
        if closure and any(by_id[value]["status"] != "confirmed" for value in required_object_ids):
            closure = False
            closure_reasons.append("required_instance_not_confirmed")

        _atomic_write(store_path, store)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    return {
        "schema_version": "scene_snapshot_v1",
        "scene_version": int(store["version"]),
        "acquisition_id": acquisition_id,
        "frame": "map",
        "objects": copy.deepcopy(store["objects"]),
        "associated_object_ids": associated_ids,
        "ambiguous_observations": ambiguous,
        "rejected_observations": rejected,
        "candidate_set_closed": closure,
        "closure_reasons": closure_reasons,
        "world_model_path": str(store_path),
        "geometry_reconstruction_ids": sorted({
            item["reconstruction_id"] for item in validated
            if item["reconstruction_id"]
        }),
    }
