"""Shared contracts for an instance observation and relation verdict.

The live chain has two different image frames: the generated perspective view
and the source equirectangular panorama.  This module keeps that distinction
explicit.  A mask can only be consumed in the frame named by
``mask_coordinate_frame``; changing its shape is never a coordinate
conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


OBSERVATION_CONTRACT_VERSION = "observation_contract_v1"

YES = "YES"
NO = "NO"
UNKNOWN = "UNKNOWN"
INVALID = "INVALID"
RELATION_STATES = frozenset({YES, NO, UNKNOWN, INVALID})

MASK_PERSPECTIVE_PIXELS = "perspective_view_pixels"
MASK_PANORAMA_PIXELS = "panorama_pixels"


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_vector(value: object, length: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("observation_vector_missing")
    if len(value) != length:
        raise ValueError("observation_vector_length_invalid")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("observation_vector_nonfinite")
    if positive and any(item <= 0.0 for item in result):
        raise ValueError("observation_extent_nonpositive")
    return result


def validate_relation_state(value: object) -> str:
    state = str(value or UNKNOWN).upper()
    if state not in RELATION_STATES:
        raise ValueError(f"relation_state_invalid:{state}")
    return state


def validate_mask_contract(
    *,
    mask_coordinate_frame: object,
    mask_shape: Sequence[int],
    native_image_width: object,
    native_image_height: object,
) -> tuple[str, int, int]:
    """Validate a mask without changing its raster.

    The returned dimensions are the native dimensions declared by the
    producer.  Callers must reject a shape mismatch instead of resizing it.
    """
    frame = str(mask_coordinate_frame or "").strip()
    if frame not in {MASK_PERSPECTIVE_PIXELS, MASK_PANORAMA_PIXELS}:
        raise ValueError("mask_coordinate_frame_invalid")
    try:
        width = int(native_image_width)
        height = int(native_image_height)
    except (TypeError, ValueError):
        raise ValueError("native_image_dimensions_invalid") from None
    if width <= 1 or height <= 1:
        raise ValueError("native_image_dimensions_invalid")
    if not isinstance(mask_shape, Sequence) or len(mask_shape) < 2:
        raise ValueError("mask_shape_invalid")
    try:
        mask_height, mask_width = int(mask_shape[0]), int(mask_shape[1])
    except (TypeError, ValueError):
        raise ValueError("mask_shape_invalid") from None
    if (mask_width, mask_height) != (width, height):
        raise ValueError(
            "mask_native_shape_mismatch:"
            f"expected={height}x{width}:actual={mask_height}x{mask_width}"
        )
    return frame, width, height


@dataclass(frozen=True)
class ObservationContract:
    """Validated, JSON-friendly representation of one detected instance."""

    observation_id: str
    station_id: str
    acquisition_id: str
    timestamp: float
    view_id: str
    camera_id: str
    native_image_width: int
    native_image_height: int
    camera_model: dict[str, Any]
    mask_path: str
    mask_coordinate_frame: str
    bbox_xyxy: list[float]
    depth_support: dict[str, Any]
    T_world_camera: list[list[float]]
    world_center: list[float]
    world_points_path: str
    world_extent: list[float]
    semantic_label: str
    semantic_confidence: float
    appearance_feature: list[float]
    canonical_object_id: int | None
    identity_hypotheses: list[dict[str, Any]]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ObservationContract":
        observation_id = str(raw.get("observation_id", "")).strip()
        station_id = str(raw.get("station_id", "")).strip()
        acquisition_id = str(raw.get("acquisition_id", "")).strip()
        view_id = str(raw.get("view_id", raw.get("representative_view_id", ""))).strip()
        if not observation_id or not station_id or not acquisition_id or not view_id:
            raise ValueError("observation_contract_identity_incomplete")
        timestamp = raw.get("timestamp", raw.get("timestamp_unix"))
        if not _finite_number(timestamp):
            raise ValueError("observation_contract_timestamp_invalid")
        frame, width, height = validate_mask_contract(
            mask_coordinate_frame=raw.get("mask_coordinate_frame"),
            mask_shape=(
                int(raw.get("mask_native_height", 0) or 0),
                int(raw.get("mask_native_width", 0) or 0),
            ),
            native_image_width=raw.get("native_image_width"),
            native_image_height=raw.get("native_image_height"),
        )
        # ``validate_mask_contract`` above verifies declared dimensions.  The
        # producer separately verifies the actual raster before construction;
        # this class remains JSON-only and therefore does not read image data.
        camera_model = raw.get("camera_model")
        if not isinstance(camera_model, Mapping):
            raise ValueError("observation_contract_camera_model_missing")
        transform = raw.get("T_world_camera")
        if not isinstance(transform, Sequence) or len(transform) != 4:
            raise ValueError("observation_contract_transform_invalid")
        transform_rows: list[list[float]] = []
        for row in transform:
            if not isinstance(row, Sequence) or len(row) != 4:
                raise ValueError("observation_contract_transform_invalid")
            values = [float(item) for item in row]
            if not all(math.isfinite(item) for item in values):
                raise ValueError("observation_contract_transform_nonfinite")
            transform_rows.append(values)
        bbox = _finite_vector(raw.get("bbox_xyxy"), 4)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("observation_contract_bbox_invalid")
        world_center = _finite_vector(raw.get("world_center"), 3)
        world_extent = _finite_vector(raw.get("world_extent"), 3, positive=True)
        semantic_label = str(raw.get("semantic_label", "")).strip()
        if not semantic_label:
            raise ValueError("observation_contract_semantic_label_missing")
        confidence = raw.get("semantic_confidence")
        if not _finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("observation_contract_semantic_confidence_invalid")
        canonical_id = raw.get("canonical_object_id")
        if canonical_id is not None:
            try:
                canonical_id = int(canonical_id)
            except (TypeError, ValueError):
                raise ValueError("observation_contract_canonical_id_invalid") from None
        return cls(
            observation_id=observation_id,
            station_id=station_id,
            acquisition_id=acquisition_id,
            timestamp=float(timestamp),
            view_id=view_id,
            camera_id=str(raw.get("camera_id", view_id)),
            native_image_width=width,
            native_image_height=height,
            camera_model=dict(camera_model),
            mask_path=str(raw.get("mask_path", "")),
            mask_coordinate_frame=frame,
            bbox_xyxy=bbox,
            depth_support=dict(raw.get("depth_support", {})),
            T_world_camera=transform_rows,
            world_center=world_center,
            world_points_path=str(raw.get("world_points_path", "")),
            world_extent=world_extent,
            semantic_label=semantic_label,
            semantic_confidence=float(confidence),
            appearance_feature=[
                float(item) for item in raw.get("appearance_feature", ())
            ],
            canonical_object_id=canonical_id,
            identity_hypotheses=[
                dict(item) for item in raw.get("identity_hypotheses", ())
                if isinstance(item, Mapping)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_schema_version": OBSERVATION_CONTRACT_VERSION,
            "observation_id": self.observation_id,
            "station_id": self.station_id,
            "acquisition_id": self.acquisition_id,
            "timestamp": self.timestamp,
            "timestamp_unix": self.timestamp,
            "view_id": self.view_id,
            "camera_id": self.camera_id,
            "native_image_width": self.native_image_width,
            "native_image_height": self.native_image_height,
            "camera_model": dict(self.camera_model),
            "mask_path": self.mask_path,
            "mask_coordinate_frame": self.mask_coordinate_frame,
            "bbox_xyxy": list(self.bbox_xyxy),
            "depth_support": dict(self.depth_support),
            "T_world_camera": [list(row) for row in self.T_world_camera],
            "world_center": list(self.world_center),
            "world_points_path": self.world_points_path,
            "world_extent": list(self.world_extent),
            "semantic_label": self.semantic_label,
            "semantic_confidence": self.semantic_confidence,
            "appearance_feature": list(self.appearance_feature),
            "canonical_object_id": self.canonical_object_id,
            "identity_hypotheses": [dict(item) for item in self.identity_hypotheses],
        }


def relation_record_metadata(
    *,
    state: object,
    station_id: object,
    timestamp: object,
    source_observation_ids: Sequence[object],
    identity_version: object,
    geometry_version: object,
) -> dict[str, Any]:
    """Build the common versioned envelope for relation evidence."""
    normalized_state = validate_relation_state(state)
    if not _finite_number(timestamp):
        raise ValueError("relation_evidence_timestamp_invalid")
    try:
        identity = int(identity_version)
        geometry = int(geometry_version)
    except (TypeError, ValueError):
        raise ValueError("relation_evidence_version_invalid") from None
    if identity < 0 or geometry < 0:
        raise ValueError("relation_evidence_version_invalid")
    return {
        "state": normalized_state,
        "station_id": str(station_id or ""),
        "timestamp": float(timestamp),
        "source_observation_ids": list(dict.fromkeys(
            str(value) for value in source_observation_ids if str(value).strip()
        )),
        "identity_version": identity,
        "geometry_version": geometry,
    }
