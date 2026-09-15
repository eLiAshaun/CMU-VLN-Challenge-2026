"""Small CPU object store for the rebuilt AI module.

The store associates observations with a conservative combination of geometry,
panorama-mask overlap, and appearance.  It intentionally keeps identity
separate from semantic relation evidence: a support relation can be updated in
``current_relation_evidence`` without changing an object's ID.

All persistent arrays live on CPU and are bounded by configuration.  The
``snapshot`` method converts every numpy value to ordinary Python values so it
can be passed directly to ``json.dumps``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

import numpy as np

from .contracts import ObjectRecord, Observation, VerifiedObservation


_DEFAULT_MAX_POINTS = 512
_DEFAULT_MAX_CROPS = 4
_DEFAULT_MAX_CROP_PIXELS = 64 * 1024
_DEFAULT_MAX_HISTORY = 32
_DEFAULT_MAX_MATCH_MASKS = 8
_DEFAULT_SAME_FRAME_IOU = 0.12
_DEFAULT_SAME_FRAME_CONTAINMENT = 0.40
_DEFAULT_ASSOCIATION_DISTANCE_M = 0.75
_MAX_ATTRIBUTE_DEPTH = 3
_MAX_ATTRIBUTE_ITEMS = 32
_MAX_ATTRIBUTE_TEXT = 512
_MAX_DESCRIPTOR_DIMENSIONS = 2048


def _normalise_label(value: object) -> str:
    return " ".join(
        str(value or "").strip().lower().replace("_", " ").replace("-", " ").split()
    )


def _canonical_class_label(value: object) -> str:
    """Normalize identity-safe aliases while preserving object categories."""

    label = _normalise_label(value)
    if (
        label in {"chair", "chairs"}
        or label.endswith(" chair")
        or label.endswith(" chairs")
    ):
        return "chair"
    if (
        label in {"pillow", "pillows", "cushion", "cushions"}
        or label.endswith(" pillow")
        or label.endswith(" pillows")
        or label.endswith(" cushion")
        or label.endswith(" cushions")
    ):
        return "pillow"
    if label in {"tv", "television"}:
        return "television"
    if label.endswith("s") and label[:-1] in {"chair", "pillow", "television"}:
        return label[:-1]
    # Keep compound categories such as ``tv cabinet`` separate from either TV
    # or cabinet.  The same rule protects other support/supported categories.
    return label


def _prompt_attributes(value: object) -> dict[str, str]:
    """Keep simple attribute words from open-vocabulary prompts."""

    label = _normalise_label(value)
    tokens = label.split()
    if len(tokens) >= 2 and tokens[-1] in {"chair", "chairs", "pillow", "pillows", "cushion", "cushions"}:
        color_words = {
            "black", "blue", "brown", "gray", "grey", "green", "orange",
            "pink", "purple", "red", "white", "yellow",
        }
        if tokens[-2] in color_words:
            return {"color": tokens[-2]}
    return {}


def _score(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(max(0.0, min(1.0, number)))


def _valid_points(value: object) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, 3), dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64)
    return np.ascontiguousarray(points[np.isfinite(points).all(axis=1)])


def _valid_center(value: object) -> np.ndarray | None:
    try:
        center = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if center.shape != (3,) or not np.isfinite(center).all():
        return None
    return center.copy()


def _normalise_bbox(value: object, center: np.ndarray | None = None) -> np.ndarray | None:
    """Return the contract's positive 3-value full axis-aligned extent."""

    try:
        bbox = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if bbox.shape != (3,):
        return None
    if not np.isfinite(bbox).all():
        return None
    extent = bbox
    if np.any(extent <= 0.0):
        return None
    return extent.astype(np.float64)


def _bbox_extent(bbox: np.ndarray | None) -> np.ndarray | None:
    if bbox is None or np.asarray(bbox).shape != (3,):
        return None
    array = np.asarray(bbox, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        return None
    return array.copy()


def _point_envelope(points: object) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return a compact lower/upper envelope before any sample thinning."""

    valid = _valid_points(points)
    if len(valid) == 0:
        return None
    return (
        np.min(valid, axis=0).astype(np.float64),
        np.max(valid, axis=0).astype(np.float64),
        int(len(valid)),
    )


def _reprojection_conflict_mask(
    observation: Observation, object_id: str
) -> np.ndarray | None:
    """Return measured points contradicted at historical surface pixels.

    Geometry provides the raster index for each retained measured point, while
    reprojection evidence provides only the pixels where a depth-inconsistent
    current return contradicts a historical track.  Matching these two
    existing sensor representations keeps unseen pixels eligible for new
    extent.
    """

    quality = observation.geometry_quality
    if not isinstance(quality, Mapping):
        return None
    evidence_map = quality.get("reprojection_evidence")
    if not isinstance(evidence_map, Mapping):
        return None
    evidence = evidence_map.get(object_id)
    if not isinstance(evidence, Mapping):
        return None
    raw_conflicts = evidence.get("reprojection_conflict_raster_pixels")
    raw_rasters = quality.get("measured_raster_pixels")
    if raw_conflicts is None or raw_rasters is None:
        return None
    try:
        conflicts = np.asarray(raw_conflicts, dtype=np.int64).reshape(-1)
        rasters = np.asarray(raw_rasters, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return None
    measured = _valid_points(observation.measured_points)
    if len(rasters) != len(measured):
        return None
    if not len(conflicts):
        return np.zeros(len(measured), dtype=bool)
    return np.isin(rasters, conflicts)


def _unique_points(points: np.ndarray) -> np.ndarray:
    """Remove exact duplicate coordinates while preserving first-seen order."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return np.ascontiguousarray(points.copy())
    _, first = np.unique(points, axis=0, return_index=True)
    return np.ascontiguousarray(points[np.sort(first)].copy())


def _bounded_points(points: np.ndarray, limit: int) -> np.ndarray:
    # Envelopes and their raw contribution counters are updated before this
    # bounded association/diagnostic sample is built.  Deduplicating here
    # therefore cannot erase evidence of how many admitted points contributed.
    points = _unique_points(_valid_points(points))
    if len(points) <= limit:
        return points.copy()
    # Keep every coordinate extremum that the bounded representation can hold.
    # The aggregate envelope is built from this sample after consolidation, so
    # uniform thinning alone could make a supported true boundary disappear.
    extrema = np.unique(np.concatenate((
        np.argmin(points, axis=0), np.argmax(points, axis=0),
    )))
    if len(extrema) > limit:
        mandatory = extrema[np.linspace(0, len(extrema) - 1, limit, dtype=np.int64)]
    else:
        mandatory = extrema
    remaining = int(limit - len(mandatory))
    if remaining > 0:
        candidates = np.setdiff1d(
            np.arange(len(points), dtype=np.int64), mandatory, assume_unique=True
        )
        if len(candidates) > remaining:
            extras = candidates[np.linspace(0, len(candidates) - 1, remaining, dtype=np.int64)]
        else:
            extras = candidates
        indices = np.sort(np.concatenate((mandatory, extras)))
    else:
        indices = np.sort(mandatory)
    return np.ascontiguousarray(points[indices].copy())


def _bounded_crop(crop: object, max_pixels: int) -> np.ndarray | None:
    if crop is None:
        return None
    try:
        image = np.asarray(crop)
    except (TypeError, ValueError):
        return None
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        return None
    image = np.ascontiguousarray(image.copy())
    pixels = int(image.shape[0] * image.shape[1])
    if pixels <= max_pixels:
        return image
    stride = int(math.ceil(math.sqrt(float(pixels) / float(max_pixels))))
    rows = np.linspace(0, image.shape[0] - 1, max(1, int(math.ceil(image.shape[0] / stride))), dtype=np.int64)
    columns = np.linspace(0, image.shape[1] - 1, max(1, int(math.ceil(image.shape[1] / stride))), dtype=np.int64)
    return np.ascontiguousarray(image[np.ix_(rows, columns)].copy())


def _compact_mask(mask: object, max_pixels: int) -> np.ndarray | None:
    if mask is None:
        return None
    try:
        array = np.asarray(mask)
    except (TypeError, ValueError):
        return None
    if array.ndim == 3:
        array = np.any(array > 0, axis=-1)
    if array.ndim != 2 or not array.size:
        return None
    binary = np.asarray(array > 0, dtype=np.uint8)
    if binary.size <= max_pixels:
        return np.ascontiguousarray(binary.copy())
    stride = int(math.ceil(math.sqrt(float(binary.size) / float(max_pixels))))
    padded_height = int(math.ceil(binary.shape[0] / stride) * stride)
    padded_width = int(math.ceil(binary.shape[1] / stride) * stride)
    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[: binary.shape[0], : binary.shape[1]] = binary
    pooled = padded.reshape(
        padded_height // stride, stride, padded_width // stride, stride
    ).max(axis=(1, 3))
    return np.ascontiguousarray(pooled.astype(np.uint8, copy=False))


def _mask_overlap(first: np.ndarray | None, second: np.ndarray | None) -> tuple[float, float]:
    if first is None or second is None or first.shape != second.shape:
        return 0.0, 0.0
    first_binary, second_binary = first > 0, second > 0
    first_area, second_area = int(first_binary.sum()), int(second_binary.sum())
    if not first_area or not second_area:
        return 0.0, 0.0
    intersection = int(np.logical_and(first_binary, second_binary).sum())
    union = first_area + second_area - intersection
    iou = float(intersection / union) if union else 0.0
    containment = float(intersection / min(first_area, second_area))
    return iou, containment


def _appearance_descriptor(observation: Observation) -> np.ndarray | None:
    raw = observation.appearance_descriptor
    if raw is None and isinstance(observation.attributes, Mapping):
        raw = observation.attributes.get("appearance_descriptor")
        if raw is None:
            raw = observation.attributes.get("appearance_embedding")
    try:
        descriptor = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not len(descriptor) or not np.isfinite(descriptor).all():
        return None
    if len(descriptor) > _MAX_DESCRIPTOR_DIMENSIONS:
        descriptor = descriptor[:_MAX_DESCRIPTOR_DIMENSIONS]
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-8:
        return None
    return np.ascontiguousarray((descriptor / norm).astype(np.float32))


def _descriptor_similarity(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 1e-8:
        return None
    value = float(np.dot(first, second) / norm)
    return max(-1.0, min(1.0, value))


def _role(attributes: Mapping[str, Any] | None) -> str | None:
    if not isinstance(attributes, Mapping):
        return None
    for key in ("semantic_role", "relation_role", "role"):
        value = _normalise_label(attributes.get(key))
        if value in {"support", "supported", "supporting", "supported_object"}:
            return "support" if value in {"support", "supporting"} else "supported"
    return None


def _roles_compatible(first: str | None, second: str | None) -> bool:
    return not (first is not None and second is not None and first != second)


def _bounded_value(value: object, depth: int = 0) -> Any:
    """Copy metadata into a small CPU/Python representation."""

    if depth > _MAX_ATTRIBUTE_DEPTH:
        return str(value)[:_MAX_ATTRIBUTE_TEXT]
    if isinstance(value, np.ndarray):
        if value.size > _MAX_ATTRIBUTE_ITEMS * _MAX_ATTRIBUTE_ITEMS:
            flat = value.reshape(-1)[: _MAX_ATTRIBUTE_ITEMS * _MAX_ATTRIBUTE_ITEMS]
            return {
                "truncated": True,
                "shape": [int(item) for item in value.shape],
                "values": [_bounded_value(item, depth + 1) for item in flat.tolist()],
            }
        return _bounded_value(value.tolist(), depth + 1)
    if isinstance(value, np.generic):
        return _bounded_value(value.item(), depth + 1)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in list(value.items())[:_MAX_ATTRIBUTE_ITEMS]:
            output[str(key)] = _bounded_value(item, depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, depth + 1) for item in list(value)[:_MAX_ATTRIBUTE_ITEMS]]
    if isinstance(value, (str, bytes)):
        text = value.decode(errors="replace") if isinstance(value, bytes) else value
        return text[:_MAX_ATTRIBUTE_TEXT]
    if isinstance(value, (bool, int, float)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)[:_MAX_ATTRIBUTE_TEXT]


def _merge_attribute_values(old: Any, new: Any) -> Any:
    if old is None:
        return new
    if new is None:
        return old
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = _merge_attribute_values(merged.get(key), value)
        return _bounded_value(merged)
    if isinstance(old, list) and isinstance(new, list):
        merged = list(old)
        for value in new:
            if value not in merged and len(merged) < _MAX_ATTRIBUTE_ITEMS:
                merged.append(value)
        return merged
    return old


def _merge_relation_evidence(record: ObjectRecord, attributes: Mapping[str, Any]) -> None:
    for key in ("current_relation_evidence", "relation_evidence"):
        value = attributes.get(key)
        if not isinstance(value, Mapping):
            continue
        for relation, evidence in list(value.items())[:_MAX_ATTRIBUTE_ITEMS]:
            record.current_relation_evidence[str(relation)] = _bounded_value(evidence)
        while len(record.current_relation_evidence) > _MAX_ATTRIBUTE_ITEMS:
            record.current_relation_evidence.pop(next(iter(record.current_relation_evidence)))


class ObjectStore:
    """Bounded CPU identity store with stable per-episode object IDs."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config) if isinstance(config, Mapping) else {}
        self.records: dict[str, ObjectRecord] = {}
        self.aliases: dict[str, str] = {}
        self._next_id = 1
        self._evidence: dict[str, list[dict[str, Any]]] = {}
        self._crop_scores: dict[str, list[float]] = {}
        # The point arrays on ObjectRecord are bounded association/diagnostic
        # samples.  Keep the observed spatial envelope independently, before
        # those arrays are thinned, so a later dense frame or a sample cap
        # cannot erase an already observed part of an object.  Only six small
        # vectors plus scalar counters are retained per record.
        self._envelopes: dict[str, dict[str, dict[str, Any]]] = {}
        self._max_points = self._limit(
            ("max_points_per_object", "max_points"), _DEFAULT_MAX_POINTS
        )
        self._max_measured_points = self._limit(
            ("max_measured_points",), self._max_points
        )
        self._max_estimated_points = self._limit(
            ("max_estimated_points",), self._max_points
        )
        self._max_crops = self._limit(
            ("max_representative_crops", "max_crops"), _DEFAULT_MAX_CROPS
        )
        self._max_crop_pixels = self._limit(
            ("max_crop_pixels",), _DEFAULT_MAX_CROP_PIXELS
        )
        self._max_history = self._limit(
            ("max_observations_per_object", "max_observation_history", "max_history"),
            _DEFAULT_MAX_HISTORY,
        )
        self._max_match_masks = self._limit(
            ("max_match_masks",), min(_DEFAULT_MAX_MATCH_MASKS, self._max_history)
        )
        self._same_frame_iou = self._float_config(
            ("same_frame_mask_iou",), _DEFAULT_SAME_FRAME_IOU, minimum=0.0, maximum=1.0
        )
        self._same_frame_containment = self._float_config(
            ("same_frame_mask_containment",),
            _DEFAULT_SAME_FRAME_CONTAINMENT,
            minimum=0.0,
            maximum=1.0,
        )
        self._association_distance = self._float_config(
            ("association_distance_m", "cross_frame_geometry_distance_m"),
            _DEFAULT_ASSOCIATION_DISTANCE_M,
            minimum=0.05,
            maximum=10.0,
        )

    @staticmethod
    def _new_envelope() -> dict[str, Any]:
        return {
            "low": None,
            "high": None,
            "point_count": 0,
            "observation_count": 0,
        }

    def _ensure_envelope(self, object_id: str) -> dict[str, dict[str, Any]]:
        state = self._envelopes.setdefault(
            object_id,
            {"measured": self._new_envelope(), "estimated": self._new_envelope()},
        )
        for source in ("measured", "estimated"):
            state.setdefault(source, self._new_envelope())
        return state

    def _update_envelope_source(
        self,
        object_id: str,
        source: str,
        points: object,
    ) -> None:
        envelope = self._ensure_envelope(object_id)[source]
        current = _point_envelope(points)
        if current is None:
            return
        low, high, count = current
        if envelope["low"] is None:
            envelope["low"] = low
            envelope["high"] = high
        else:
            envelope["low"] = np.minimum(envelope["low"], low)
            envelope["high"] = np.maximum(envelope["high"], high)
        envelope["point_count"] = int(envelope.get("point_count", 0)) + count
        envelope["observation_count"] = int(envelope.get("observation_count", 0)) + 1

    def _update_envelopes(self, object_id: str, observation: Observation) -> None:
        self._update_envelope_source(object_id, "measured", observation.measured_points)
        self._update_envelope_source(object_id, "estimated", observation.estimated_points)

    @staticmethod
    def _envelope_geometry(
        envelope: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not isinstance(envelope, Mapping):
            return None, None
        low = _valid_center(envelope.get("low"))
        high = _valid_center(envelope.get("high"))
        if low is None or high is None or np.any(high < low):
            return None, None
        try:
            count = int(envelope.get("point_count", 0))
        except (TypeError, ValueError):
            count = 0
        center = 0.5 * (low + high)
        # Two repeated contributions at one coordinate do not describe an
        # extent.  The positive marker floor is valid only after the envelope
        # has observed at least two distinct spatial positions.
        bbox = (
            np.maximum(high - low, np.asarray([0.03, 0.03, 0.03], dtype=np.float64))
            if np.any(high != low)
            else None
        )
        return center.astype(np.float64), None if bbox is None else bbox.astype(np.float64)

    def _envelope_snapshot(self, object_id: str) -> dict[str, Any]:
        state = self._envelopes.get(object_id, {})
        output: dict[str, Any] = {}
        for source in ("measured", "estimated"):
            envelope = state.get(source, {})
            item: dict[str, Any] = {
                "point_count": int(envelope.get("point_count", 0)),
                "observation_count": int(envelope.get("observation_count", 0)),
            }
            low = _valid_center(envelope.get("low"))
            high = _valid_center(envelope.get("high"))
            if low is not None and high is not None:
                item["lower"] = low.tolist()
                item["upper"] = high.tolist()
            output[source] = item
        return output

    def _refresh_bounds_quality(self, record: ObjectRecord, object_id: str) -> None:
        state = self._ensure_envelope(object_id)
        measured = state["measured"]
        estimated = state["estimated"]
        measured_count = int(measured.get("point_count", 0))
        estimated_count = int(estimated.get("point_count", 0))
        if measured_count and estimated_count:
            source = "measured_and_estimated"
        elif measured_count:
            source = "measured"
        elif estimated_count:
            source = "estimated"
        else:
            source = "unavailable"
        quality = record.geometry_quality
        if not isinstance(quality, dict):
            quality = {}
        quality.update(
            {
                "bounds_semantics": "observed_surface_envelope",
                "entity_extent_complete": False,
                "count_semantics": "raw admitted image-region contributions, not independent observations or confidence",
                "observed_surface_source": source,
                "observed_surface_envelope": self._envelope_snapshot(object_id),
                "instance_measured_point_count": measured_count,
                "instance_estimated_point_count": estimated_count,
                "instance_measured_observation_count": int(measured.get("observation_count", 0)),
                "instance_estimated_observation_count": int(estimated.get("observation_count", 0)),
                "sample_measured_point_count": int(len(_valid_points(record.measured_points))),
                "sample_estimated_point_count": int(len(_valid_points(record.estimated_points))),
            }
        )
        # This field is deliberately explicit: a shared view calibration is
        # evidence about the depth map, not measured support for this object.
        if quality.get("depth_calibration_scope") == "shared_view":
            quality["shared_view_depth_calibration"] = {
                "matches": quality.get("depth_calibration_matches"),
                "residual": quality.get("depth_calibration_residual"),
                "status": quality.get("depth_calibration_status"),
            }
        record.geometry_quality = _bounded_value(quality)

    def _derive_record_geometry(self, record: ObjectRecord, object_id: str) -> None:
        """Derive the observed envelope independently of association samples."""
        state = self._envelopes[object_id]
        measured_center, measured_bbox = self._envelope_geometry(state['measured'])
        estimated_center, estimated_bbox = self._envelope_geometry(state['estimated'])
        if measured_bbox is not None:
            record.center, record.bbox = measured_center, measured_bbox
            source = 'measured'
        elif estimated_bbox is not None:
            record.center, record.bbox = estimated_center, estimated_bbox
            source = 'estimated'
        elif measured_center is not None:
            record.center = measured_center
            record.bbox = None
            source = 'measured_point_only'
        elif estimated_center is not None:
            record.center = estimated_center
            record.bbox = None
            source = 'estimated_point_only'
        else:
            record.center = None
            record.bbox = None
            source = 'unavailable'
        self._refresh_bounds_quality(record, object_id)
        record.geometry_quality['bbox_source'] = source

    def _limit(self, keys: Sequence[str], default: int) -> int:
        for key in keys:
            if key not in self.config:
                continue
            try:
                value = int(self.config[key])
            except (TypeError, ValueError):
                break
            return max(1, value)
        return max(1, int(default))

    def _float_config(
        self,
        keys: Sequence[str],
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        for key in keys:
            if key not in self.config:
                continue
            try:
                value = float(self.config[key])
            except (TypeError, ValueError):
                break
            if math.isfinite(value):
                return max(minimum, min(maximum, value))
        return default

    def _new_id(self) -> str:
        prefix = _normalise_label(self.config.get("id_prefix", "object")) or "object"
        object_id = f"{prefix}_{self._next_id:06d}"
        self._next_id += 1
        while object_id in self.records:
            object_id = f"{prefix}_{self._next_id:06d}"
            self._next_id += 1
        return object_id

    def _observation_attributes(self, observation: Observation) -> dict[str, Any]:
        raw = (
            dict(observation.attributes)
            if isinstance(observation.attributes, Mapping)
            else {}
        )
        bounded = _bounded_value(raw)
        if not isinstance(bounded, dict):
            bounded = {}
        if isinstance(observation, VerifiedObservation):
            bounded['category_evidence'] = {observation.concept: dict(observation.category_evidence)}
        concepts = bounded.get("concepts")
        if not isinstance(concepts, list):
            concepts = []
        raw_concept = str(observation.concept)
        if raw_concept not in concepts and len(concepts) < _MAX_ATTRIBUTE_ITEMS:
            concepts.append(raw_concept)
        canonical = _canonical_class_label(observation.concept)
        if canonical and canonical not in concepts and len(concepts) < _MAX_ATTRIBUTE_ITEMS:
            concepts.append(canonical)
        bounded["concepts"] = concepts
        bounded.setdefault("canonical_class", canonical)
        for key, value in _prompt_attributes(observation.concept).items():
            bounded.setdefault(key, value)
        return bounded

    def _evidence_from_observation(self, observation: Observation) -> dict[str, Any]:
        center = _valid_center(observation.center)
        bbox = _normalise_bbox(observation.bbox, center)
        camera_position = _valid_center(observation.camera_position)
        quality = (
            observation.geometry_quality
            if isinstance(observation.geometry_quality, Mapping)
            else {}
        )
        measured_count = len(_valid_points(observation.measured_points))
        estimated_count = len(_valid_points(observation.estimated_points))
        try:
            stamp = float(observation.stamp)
        except (TypeError, ValueError):
            stamp = float("nan")
        return {
            "observation_id": str(observation.observation_id),
            "view_id": str(observation.view_id),
            "stamp": stamp if math.isfinite(stamp) else None,
            "mask": _compact_mask(observation.panorama_mask, max(1024, self._max_match_masks * 16384)),
            "box_2d": np.asarray(observation.box_2d, dtype=float),
            "center": center,
            "bbox": bbox,
            "descriptor": _appearance_descriptor(observation),
            "camera_position": camera_position,
            "score": _score(observation.score),
            "role": _role(observation.attributes),
            # Keep instance support separate from a depth calibration computed
            # over the whole view.  These fields are private matching evidence
            # and remain bounded by the existing history limit.
            "geometry_source": str(quality.get("source", "unavailable")),
            "measured_point_count": int(measured_count),
            "estimated_point_count": int(estimated_count),
            "depth_calibration_scope": quality.get("depth_calibration_scope"),
            "depth_calibration_matches": quality.get("depth_calibration_matches"),
            "depth_calibration_residual": quality.get("depth_calibration_residual"),
        }

    def _append_evidence(self, object_id: str, evidence: dict[str, Any]) -> None:
        values = self._evidence.setdefault(object_id, [])
        for index, old in enumerate(values):
            if old.get("observation_id") == evidence.get("observation_id") and old.get("view_id") == evidence.get("view_id"):
                if old.get('mask') is not None and evidence.get('mask') is not None:
                    evidence['mask'] = np.logical_or(old['mask'], evidence['mask'])
                if old.get('box_2d') is not None and evidence.get('box_2d') is not None:
                    a, b = old['box_2d'], evidence['box_2d']
                    evidence['box_2d'] = np.r_[np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])]
                values[index] = evidence
                return
        values.append(evidence)
        if len(values) > max(self._max_history, self._max_match_masks):
            del values[: -max(self._max_history, self._max_match_masks)]

    def _append_history(self, record: ObjectRecord, observation: Observation) -> None:
        record.observation_ids.append(str(observation.observation_id))
        camera_position = _valid_center(observation.camera_position)
        record.observation_poses.append(
            camera_position.astype(float).tolist() if camera_position is not None else []
        )
        try:
            timestamp = float(observation.stamp)
        except (TypeError, ValueError):
            timestamp = 0.0
        record.timestamps.append(timestamp if math.isfinite(timestamp) else 0.0)
        if len(record.observation_ids) > self._max_history:
            del record.observation_ids[: -self._max_history]
            del record.observation_poses[: -self._max_history]
            del record.timestamps[: -self._max_history]

    def _geometry_match(
        self,
        first_center: np.ndarray | None,
        first_bbox: np.ndarray | None,
        second_center: np.ndarray | None,
        second_bbox: np.ndarray | None,
    ) -> tuple[bool, float, float]:
        if first_center is None or second_center is None:
            return False, 0.0, float("inf")
        distance = float(np.linalg.norm(first_center - second_center))
        first_extent = _bbox_extent(first_bbox)
        second_extent = _bbox_extent(second_bbox)
        radius = 0.0
        iou = 0.0
        if first_extent is not None:
            radius += 0.5 * float(np.linalg.norm(first_extent))
        if second_extent is not None:
            radius += 0.5 * float(np.linalg.norm(second_extent))
        # The configured association distance is for missing extents. Small
        # objects with measured extents must not inherit a 0.75 m radius that
        # can merge two separate picture frames half a metre apart.
        limit = max(0.30, radius + 0.15) if first_extent is not None and second_extent is not None else self._association_distance
        if first_extent is not None and second_extent is not None:
            first_low = first_center - 0.5 * first_extent
            first_high = first_center + 0.5 * first_extent
            second_low = second_center - 0.5 * second_extent
            second_high = second_center + 0.5 * second_extent
            intersection = np.maximum(
                0.0, np.minimum(first_high, second_high) - np.maximum(first_low, second_low)
            )
            intersection_volume = float(np.prod(intersection))
            first_volume = float(np.prod(np.maximum(first_high - first_low, 0.0)))
            second_volume = float(np.prod(np.maximum(second_high - second_low, 0.0)))
            union = first_volume + second_volume - intersection_volume
            iou = intersection_volume / union if union > 1e-9 else 0.0
            # A long cabinet's width must not permit merging a different
            # cabinet above it. Compare separation along the actual AABB axes.
            gap = np.maximum(np.abs(first_center-second_center) - (first_extent+second_extent)/2, 0.0)
            if np.linalg.norm(gap) > self._association_distance:
                return False, iou, distance
        return distance <= limit, iou, distance

    def _same_frame_match(
        self, record: ObjectRecord, object_id: str, observation: Observation
    ) -> tuple[bool, float]:
        incoming = self._evidence_from_observation(observation)
        candidates = [
            value
            for value in self._evidence.get(object_id, [])
            if value.get("observation_id") == incoming.get("observation_id")
        ]
        if not candidates:
            return False, 0.0
        best = 0.0
        for old in candidates:
            if not _roles_compatible(old.get("role"), incoming.get("role")):
                continue
            iou, containment = _mask_overlap(old.get("mask"), incoming.get("mask"))
            box_containment = 0.0
            if old.get('view_id') == incoming.get('view_id') and old.get('box_2d') is not None:
                a, b = old['box_2d'], incoming['box_2d']
                intersection = np.prod(np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0))
                box_containment = float(intersection / max(min(np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2])), 1e-9))
            # A containing detector rectangle does not establish common
            # pixel ownership. Preserve fragment association only when their
            # actual SAM regions also overlap by the existing containment rule.
            contained_fragment = box_containment >= 0.9 and containment >= self._same_frame_containment
            if not contained_fragment and (iou < self._same_frame_iou or containment < self._same_frame_containment):
                continue
            geometry_ok, _geometry_iou, distance = self._geometry_match(
                old.get("center"), old.get("bbox"), incoming.get("center"), incoming.get("bbox")
            )
            both_measured = old.get('geometry_source') == 'measured' and incoming.get('geometry_source') == 'measured'
            if ((both_measured or not contained_fragment) and old.get("center") is not None
                    and incoming.get("center") is not None and not geometry_ok):
                continue
            # Panorama overlap is the decisive same-frame signal.  The small
            # geometric term only ranks multiple possible records.
            score = iou + 0.35 * max(containment, box_containment)
            if math.isfinite(distance):
                score += 0.15 * max(0.0, 1.0 - distance / max(self._association_distance, 0.3))
            best = max(best, score)
        return best > 0.0, best

    def _projection_support(self, object_id: str, observation: Observation) -> dict | None:
        evidence = observation.geometry_quality.get('reprojection_evidence', {}).get(object_id)
        if (evidence and evidence.get('source') == 'historical_measured_image_projection'
                and evidence.get('matched_pixels', 0) >= 2
                and evidence.get('matched_fraction', 0.0) >= self._same_frame_containment):
            return evidence
        return None

    def _cross_frame_match(
        self, record: ObjectRecord, object_id: str, observation: Observation
    ) -> tuple[bool, float]:
        incoming = self._evidence_from_observation(observation)
        if not self._evidence.get(object_id):
            return False, 0.0
        if not _roles_compatible(
            _role(record.attributes), incoming.get("role")
        ):
            return False, 0.0
        projected = self._projection_support(object_id, observation)
        if projected:
            # A missing current return is not an occlusion. Past measurements
            # aligned with this image can preserve identity when monocular
            # position is wrong. Geometry fusion checks conflicting measured
            # returns separately; identity is not permission to fuse them.
            return True, 1.0 + float(projected['matched_fraction'])
        geometry_ok, geometry_iou, distance = self._geometry_match(
            record.center,
            record.bbox,
            incoming.get("center"),
            incoming.get("bbox"),
        )
        if not geometry_ok:
            return False, 0.0
        appearance = _descriptor_similarity(record.appearance_descriptor, incoming.get("descriptor"))
        # Geometry remains mandatory.  Appearance strengthens a cross-frame
        # match when both descriptors are available; an unavailable descriptor
        # is not treated as semantic rejection metadata.
        # The compact color descriptor ranks geometrically plausible matches;
        # a partial view's different color distribution is not a hard veto.
        appearance_score = 0.5 if appearance is None else max(0.0, (appearance + 1.0) * 0.5)
        distance_score = max(0.0, 1.0 - distance / max(self._association_distance, 0.3))
        score = 0.65 * distance_score + 0.20 * geometry_iou + 0.15 * appearance_score
        return True, score

    def _find_match(self, observation: Observation) -> tuple[str | None, float]:
        canonical = _canonical_class_label(observation.concept)
        incoming_role = _role(observation.attributes)
        best_id: str | None = None
        best_score = 0.0
        same_frame_id: str | None = None
        same_frame_score = 0.0
        for object_id, record in self.records.items():
            if canonical not in record.class_scores:
                continue
            if not _roles_compatible(_role(record.attributes), incoming_role):
                continue
            if str(observation.observation_id) in record.observation_ids:
                matched, score = self._same_frame_match(record, object_id, observation)
                if matched and score > same_frame_score:
                    same_frame_id, same_frame_score = object_id, score
            else:
                matched, score = self._cross_frame_match(record, object_id, observation)
                if matched and score > best_score:
                    best_id, best_score = object_id, score
        return (same_frame_id, same_frame_score) if same_frame_id is not None else (best_id, best_score)

    def _update_attributes(
        self,
        record: ObjectRecord,
        observation: Observation,
        object_id: str,
    ) -> None:
        incoming = self._observation_attributes(observation)
        for key, value in incoming.items():
            record.attributes[key] = _merge_attribute_values(record.attributes.get(key), value)
        record.attributes = _bounded_value(record.attributes)
        if not isinstance(record.attributes, dict):
            record.attributes = {}
        _merge_relation_evidence(record, incoming)
        quality = (
            dict(observation.geometry_quality)
            if isinstance(observation.geometry_quality, Mapping)
            else {}
        )
        merged_quality = dict(record.geometry_quality)
        for key, value in quality.items():
            merged_quality[str(key)] = _bounded_value(value)
        measured_count = len(_valid_points(record.measured_points))
        estimated_count = len(_valid_points(record.estimated_points))
        if measured_count and estimated_count:
            merged_quality["source"] = "measured_and_estimated"
        elif measured_count:
            merged_quality["source"] = "measured"
        elif estimated_count:
            merged_quality["source"] = "estimated"
        else:
            merged_quality.setdefault("source", "unavailable")
        merged_quality["observation_count"] = len(record.observation_ids)
        merged_quality["measured_point_count"] = measured_count
        merged_quality["estimated_point_count"] = estimated_count
        record.geometry_quality = _bounded_value(merged_quality)
        self._refresh_bounds_quality(record, object_id)

    def _update_descriptor(self, record: ObjectRecord, observation: Observation) -> None:
        descriptor = _appearance_descriptor(observation)
        if descriptor is None:
            return
        if record.appearance_descriptor is None:
            record.appearance_descriptor = descriptor.copy()
            return
        if record.appearance_descriptor.shape != descriptor.shape:
            return
        old_weight = max(1.0, float(len(record.observation_ids) - 1))
        fused = (record.appearance_descriptor * old_weight + descriptor) / (old_weight + 1.0)
        norm = float(np.linalg.norm(fused))
        if norm > 1e-8:
            record.appearance_descriptor = np.ascontiguousarray((fused / norm).astype(np.float32))

    def _update_crops(self, object_id: str, record: ObjectRecord, observation: Observation) -> None:
        crop = _bounded_crop(observation.crop_rgb, self._max_crop_pixels)
        if crop is None:
            return
        quality = _score(observation.score) + min(0.25, float(crop.shape[0] * crop.shape[1]) / float(max(self._max_crop_pixels, 1)) * 0.25)
        scores = self._crop_scores.setdefault(object_id, [])
        if len(record.representative_crops) < self._max_crops:
            record.representative_crops.append(crop)
            scores.append(quality)
            return
        if not scores:
            scores.extend([0.0] * len(record.representative_crops))
        worst = int(np.argmin(np.asarray(scores, dtype=np.float64)))
        if quality > scores[worst]:
            record.representative_crops[worst] = crop
            scores[worst] = quality

    def _create_record(self, object_id: str, observation: Observation) -> ObjectRecord:
        canonical = _canonical_class_label(observation.concept)
        center = _valid_center(observation.center)
        bbox = _normalise_bbox(observation.bbox, center)
        self._ensure_envelope(object_id)
        self._update_envelopes(object_id, observation)
        measured = _bounded_points(
            _valid_points(observation.measured_points), self._max_measured_points
        )
        estimated = _bounded_points(
            _valid_points(observation.estimated_points), self._max_estimated_points
        )
        measured_center, measured_bbox = self._envelope_geometry(
            self._envelopes[object_id]["measured"]
        )
        estimated_center, estimated_bbox = self._envelope_geometry(
            self._envelopes[object_id]["estimated"]
        )
        if measured_bbox is not None:
            center, bbox = measured_center, measured_bbox
        elif bbox is None and estimated_bbox is not None:
            center, bbox = estimated_center, estimated_bbox
        elif center is None:
            center = measured_center if measured_center is not None else estimated_center
        quality = (
            _bounded_value(dict(observation.geometry_quality))
            if isinstance(observation.geometry_quality, Mapping)
            else {}
        )
        if not isinstance(quality, dict):
            quality = {}
        quality.setdefault("source", "unavailable")
        quality["observation_count"] = 1
        quality["measured_point_count"] = int(len(measured))
        quality["estimated_point_count"] = int(len(estimated))
        record = ObjectRecord(
            id=object_id,
            class_scores={canonical: _score(observation.score)},
            attributes=self._observation_attributes(observation),
            measured_points=measured,
            estimated_points=estimated,
            center=None if center is None else center.copy(),
            bbox=None if bbox is None else bbox.copy(),
            geometry_quality=quality,
            representative_crops=[],
            appearance_descriptor=(
                None
                if _appearance_descriptor(observation) is None
                else _appearance_descriptor(observation).copy()
            ),
            observation_ids=[],
            observation_poses=[],
            timestamps=[],
            current_relation_evidence={},
        )
        self._append_history(record, observation)
        _merge_relation_evidence(record, record.attributes)
        self._evidence[object_id] = []
        self._crop_scores[object_id] = []
        self._append_evidence(object_id, self._evidence_from_observation(observation))
        self._update_crops(object_id, record, observation)
        self._derive_record_geometry(record, object_id)
        return record

    def _merge_record(self, record: ObjectRecord, object_id: str, observation: Observation) -> None:
        observation.geometry_quality['memory_geometry_admission'] = True
        observation.geometry_quality['geometry_conflict'] = None
        canonical = _canonical_class_label(observation.concept)
        score = _score(observation.score)
        record.class_scores[canonical] = max(record.class_scores.get(canonical, 0.0), score)
        conflict_mask = _reprojection_conflict_mask(observation, object_id)
        if conflict_mask is not None and np.any(conflict_mask):
            measured = _valid_points(observation.measured_points)
            retained = np.ascontiguousarray(measured[~conflict_mask].copy())
            quality = observation.geometry_quality
            quality['reprojection_geometry_admission'] = {
                'status': 'historical_surface_conflict_points_rejected',
                'conflicting_point_count': int(np.count_nonzero(conflict_mask)),
                'retained_point_count': int(len(retained)),
                'unseen_points_preserved': True,
            }
            current = _point_envelope(retained)
            if current is not None:
                low, high, _ = current
                incoming_center = (low + high) * 0.5
                incoming_bbox = np.maximum(high - low, 0.03) if np.any(high != low) else None
            else:
                estimated = _point_envelope(observation.estimated_points)
                if estimated is not None:
                    low, high, _ = estimated
                    incoming_center = (low + high) * 0.5
                    incoming_bbox = np.maximum(high - low, 0.03) if np.any(high != low) else None
                else:
                    incoming_center, incoming_bbox = None, None
            raw_rasters = quality.get('measured_raster_pixels')
            try:
                rasters = np.asarray(raw_rasters, dtype=np.int64).reshape(-1)
            except (TypeError, ValueError):
                rasters = np.empty((0,), dtype=np.int64)
            if len(rasters) == len(measured):
                quality['measured_raster_pixels'] = rasters[~conflict_mask].tolist()
            quality['measured_point_count'] = int(len(retained))
            quality['bbox_available'] = bool(incoming_bbox is not None)
            if len(retained):
                quality['source'] = 'measured' if incoming_bbox is not None else 'measured_sparse'
            elif len(_valid_points(observation.estimated_points)):
                quality['source'] = 'estimated'
            else:
                quality['source'] = 'unavailable'
            observation = replace(
                observation,
                measured_points=retained,
                center=incoming_center,
                bbox=incoming_bbox,
            )
        incoming_measurement = _point_envelope(observation.measured_points)
        if self._projection_support(object_id, observation) and incoming_measurement is not None:
            low, high, _ = incoming_measurement
            center = (low + high) * 0.5
            bbox = np.maximum(high-low, 0.03) if np.any(high != low) else None
            agrees, _, distance = self._geometry_match(record.center, record.bbox, center, bbox)
            if not agrees:
                observation.geometry_quality['memory_geometry_admission'] = False
                observation.geometry_quality['geometry_conflict'] = {
                    'reason': 'current_measurement_disagrees_with_reprojected_track',
                    'matched_object_id': object_id, 'center_distance_m': distance,
                    'current_measured_center': center.tolist(),
                    'retained_center': record.center.tolist() if record.center is not None else None,
                }
                # Keep the current image/category/history, without adding an
                # inconsistent surface or treating it as a new physical body.
                observation = replace(observation, measured_points=np.empty((0, 3)),
                                      estimated_points=np.empty((0, 3)), center=None, bbox=None)
        self._update_envelopes(object_id, observation)
        measured = _valid_points(observation.measured_points)
        estimated = _valid_points(observation.estimated_points)
        if len(measured):
            record.measured_points = _bounded_points(
                np.concatenate((record.measured_points, measured), axis=0),
                self._max_measured_points,
            )
        if len(estimated):
            record.estimated_points = _bounded_points(
                np.concatenate((record.estimated_points, estimated), axis=0),
                self._max_estimated_points,
            )
        incoming_center = _valid_center(observation.center)
        incoming_bbox = _normalise_bbox(observation.bbox, incoming_center)
        if record.center is None and incoming_center is not None:
            record.center = incoming_center.copy()
        if record.bbox is None and incoming_bbox is not None:
            record.bbox = incoming_bbox.copy()
        self._derive_record_geometry(record, object_id)
        self._append_history(record, observation)
        self._update_attributes(record, observation, object_id)
        self._update_descriptor(record, observation)
        self._update_crops(object_id, record, observation)
        self._append_evidence(object_id, self._evidence_from_observation(observation))

    def canonical_id(self, object_id: str) -> str:
        while object_id in self.aliases:
            object_id = self.aliases[object_id]
        return object_id

    def _separated_in_same_view(self, first: str, second: str) -> bool:
        other = {(e['observation_id'], e['view_id']): e for e in self._evidence.get(second, [])}
        for evidence in self._evidence.get(first, []):
            match = other.get((evidence['observation_id'], evidence['view_id']))
            if match is not None:
                a, b = evidence.get('box_2d'), match.get('box_2d')
                if a is not None and b is not None and np.any(np.minimum(a[2:], b[2:]) <= np.maximum(a[:2], b[:2])):
                    return True
                if evidence.get('mask') is not None and match.get('mask') is not None:
                    iou, containment = _mask_overlap(
                        evidence.get('mask'), match.get('mask')
                    )
                    if iou < self._same_frame_iou and containment < self._same_frame_containment:
                        # Overlapping detector rectangles do not establish
                        # common pixel ownership. Keep the tracks separate
                        # even if later measured samples share a coarse voxel.
                        return True
        return False

    def _merge_envelopes(
        self,
        winner: str,
        loser: str,
        *,
        measured_points: np.ndarray | None = None,
        estimated_points: np.ndarray | None = None,
    ) -> None:
        keep = self._ensure_envelope(winner)
        remove = self._envelopes.get(loser)
        if remove is None:
            return
        for source in ("measured", "estimated"):
            target = keep[source]
            incoming = remove.get(source, {})
            override = measured_points if source == "measured" else estimated_points
            if override is not None:
                # Rebuild bounds from the loser's retained support sample so a
                # prior incompatible aggregate bound cannot pollute the
                # canonical winner.  _bounded_points preserves supported
                # extrema; retain the loser's raw counters separately.
                current = _point_envelope(override)
                if current is not None:
                    low, high, _ = current
                    if target["low"] is None:
                        target["low"], target["high"] = low, high
                    else:
                        target["low"] = np.minimum(target["low"], low)
                        target["high"] = np.maximum(target["high"], high)
                target["point_count"] = int(target.get("point_count", 0)) + int(
                    incoming.get("point_count", 0)
                )
                target["observation_count"] = int(target.get("observation_count", 0)) + int(
                    incoming.get("observation_count", 0)
                )
                continue
            low = _valid_center(incoming.get("low"))
            high = _valid_center(incoming.get("high"))
            if low is not None and high is not None:
                if target["low"] is None:
                    target["low"], target["high"] = low, high
                else:
                    target["low"] = np.minimum(target["low"], low)
                    target["high"] = np.maximum(target["high"], high)
            target["point_count"] = int(target.get("point_count", 0)) + int(
                incoming.get("point_count", 0)
            )
            target["observation_count"] = int(target.get("observation_count", 0)) + int(
                incoming.get("observation_count", 0)
            )
        del self._envelopes[loser]

    def _merge_existing(self, winner: str, loser: str) -> None:
        keep, remove = self.records[winner], self.records[loser]
        for label, score in remove.class_scores.items():
            keep.class_scores[label] = max(keep.class_scores.get(label, 0.0), score)
        categories = dict(keep.attributes.get('category_evidence', {}))
        for label, evidence in remove.attributes.get('category_evidence', {}).items():
            old = categories.get(label, {})
            if old.get('verdict') != 'yes':
                categories[label] = evidence
        keep.attributes = _merge_attribute_values(keep.attributes, remove.attributes)
        if categories:
            keep.attributes['category_evidence'] = categories
        measured = _valid_points(remove.measured_points)
        estimated = _valid_points(remove.estimated_points)
        # A loser's aggregate envelope may contain a point that never had
        # identity-compatible support.  Rebuild the canonical envelope from
        # the bounded retained samples while preserving the full loser bounds
        # as diagnostic alias evidence below.
        loser_envelope = self._envelope_snapshot(loser)
        self._merge_envelopes(
            winner,
            loser,
            measured_points=measured,
            estimated_points=estimated,
        )
        aliases = keep.geometry_quality.setdefault('consolidated_alias_envelopes', {})
        if isinstance(aliases, dict):
            aliases[loser] = loser_envelope
        for name, limit in [('measured_points', self._max_measured_points), ('estimated_points', self._max_estimated_points)]:
            incoming = measured if name == 'measured_points' else estimated
            setattr(keep, name, _bounded_points(np.concatenate((getattr(keep, name), incoming)), limit))
        self._derive_record_geometry(keep, winner)
        if keep.appearance_descriptor is None:
            keep.appearance_descriptor = remove.appearance_descriptor
        elif remove.appearance_descriptor is not None and keep.appearance_descriptor.shape == remove.appearance_descriptor.shape:
            descriptor = keep.appearance_descriptor + remove.appearance_descriptor
            keep.appearance_descriptor = descriptor / max(np.linalg.norm(descriptor), 1e-9)
        history = {}
        for record in (keep, remove):
            for oid, pose, timestamp in zip(record.observation_ids, record.observation_poses, record.timestamps):
                history[(timestamp, oid)] = pose
        entries = sorted(history)[-self._max_history:]
        keep.observation_ids = [oid for timestamp, oid in entries]
        keep.timestamps = [timestamp for timestamp, oid in entries]
        keep.observation_poses = [history[key] for key in entries]
        crops = []
        for oid, record in ((winner, keep), (loser, remove)):
            scores = self._crop_scores.get(oid, [0.0]*len(record.representative_crops))
            crops.extend(zip(scores, record.representative_crops))
        crops.sort(key=lambda pair: pair[0], reverse=True)
        keep.representative_crops = [crop for score, crop in crops[:self._max_crops]]
        self._crop_scores[winner] = [score for score, crop in crops[:self._max_crops]]
        keep.current_relation_evidence.update(remove.current_relation_evidence)
        for evidence in self._evidence.get(loser, []):
            self._append_evidence(winner, dict(evidence))
        self.aliases[loser] = winner
        del self.records[loser]
        self._evidence.pop(loser, None)
        self._crop_scores.pop(loser, None)
        self._refresh_bounds_quality(keep, winner)

    def _consolidate_shared_measurements(self) -> None:
        voxels = {oid: set(map(tuple, np.floor(record.measured_points/0.15).astype(int)))
                  for oid, record in self.records.items() if len(record.measured_points) >= 3}
        identities = sorted(voxels)
        for index, first in enumerate(identities):
            if first not in self.records:
                continue
            for second in identities[index+1:]:
                if second not in self.records or not self.records[first].class_scores.keys() & self.records[second].class_scores.keys():
                    continue
                if not _roles_compatible(_role(self.records[first].attributes), _role(self.records[second].attributes)):
                    continue
                support = min(len(voxels[first]), len(voxels[second]))
                if support < 3 or len(voxels[first] & voxels[second])/support < 0.5:
                    continue
                if self._separated_in_same_view(first, second):
                    continue
                self._merge_existing(first, second)
                # Rebuild from the retained sample. The canonical envelope may
                # intentionally omit unsupported parts of the loser bounds;
                # stale loser voxels must not authorize later consolidation.
                voxels[first] = set(
                    map(
                        tuple,
                        np.floor(self.records[first].measured_points / 0.15).astype(int),
                    )
                )
        if not self.aliases:
            return
        for record in self.records.values():
            rewritten = {}
            for key, value in record.current_relation_evidence.items():
                evidence = dict(value) if isinstance(value, Mapping) else value
                if isinstance(evidence, dict):
                    for field in ('subject_id', 'anchor_id'):
                        if isinstance(evidence.get(field), str):
                            evidence[field] = self.canonical_id(evidence[field])
                    if evidence.get('relation') in {'on', 'inside'} and evidence.get('subject_id') == evidence.get('anchor_id'):
                        continue
                if ':' in key:
                    relation, target = key.split(':', 1)
                    key = relation + ':' + self.canonical_id(target)
                rewritten[key] = evidence
            record.current_relation_evidence = rewritten

    def update(self, observations: list[VerifiedObservation]) -> list[str]:
        """Associate strongest existing matches first; return input-aligned IDs."""

        if observations is None:
            return []
        if not isinstance(observations, list):
            observations = list(observations)
        for observation in observations:
            if not isinstance(observation, VerifiedObservation):
                raise TypeError("ObjectStore requires current-image VerifiedObservation values")
        # View index is unrelated to identity quality. A weak edge proposal
        # must not mutate a track before another view's strong match gets its
        # turn. Reuse the existing association score, then apply the existing
        # same-frame mask constraints as each observation is incorporated.
        ordered = sorted(enumerate(observations),
                         key=lambda item: self._find_match(item[1])[1], reverse=True)
        ids = [''] * len(observations)
        for index, observation in ordered:
            object_id, _ = self._find_match(observation)
            if object_id is None:
                object_id = self._new_id()
                self.records[object_id] = self._create_record(object_id, observation)
            else:
                self._merge_record(self.records[object_id], object_id, observation)
            ids[index] = object_id
        self._consolidate_shared_measurements()
        return [self.canonical_id(oid) for oid in ids]

    @staticmethod
    def _jsonable(value: object) -> Any:
        if isinstance(value, np.ndarray):
            return ObjectStore._jsonable(value.tolist())
        if isinstance(value, np.generic):
            return ObjectStore._jsonable(value.item())
        if isinstance(value, Mapping):
            return {str(key): ObjectStore._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ObjectStore._jsonable(item) for item in value]
        if isinstance(value, float):
            return float(value) if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def _record_snapshot(self, record: ObjectRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "class_scores": self._jsonable(record.class_scores),
            "attributes": self._jsonable(record.attributes),
            "measured_points": self._jsonable(record.measured_points),
            "estimated_points": self._jsonable(record.estimated_points),
            "center": self._jsonable(record.center),
            "bbox": self._jsonable(record.bbox),
            "geometry_quality": self._jsonable(record.geometry_quality),
            "representative_crops": [
                {"shape": list(crop.shape), "dtype": str(crop.dtype)}
                for crop in record.representative_crops
            ],
            "appearance_descriptor": self._jsonable(record.appearance_descriptor),
            "observation_ids": self._jsonable(record.observation_ids),
            "observation_poses": self._jsonable(record.observation_poses),
            "timestamps": self._jsonable(record.timestamps),
            "current_relation_evidence": self._jsonable(record.current_relation_evidence),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a bounded JSON-serializable snapshot of all records."""

        serialized = {
            object_id: self._record_snapshot(record)
            for object_id, record in self.records.items()
        }
        return {
            "schema_version": "object_store_v1",
            "count": len(serialized),
            "records": serialized,
            "aliases": {oid: self.canonical_id(oid) for oid in self.aliases},
        }

    def clear(self) -> None:
        """Clear this episode's records and restart its local ID namespace."""

        self.records.clear()
        self.aliases.clear()
        self._evidence.clear()
        self._crop_scores.clear()
        self._envelopes.clear()
        self._next_id = 1


__all__ = ["ObjectStore"]
