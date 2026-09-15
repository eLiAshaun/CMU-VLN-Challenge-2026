"""CPU data shared by perception, geometry, memory and task execution.

Transforms use column vectors and T_destination_source naming. View cameras
use optical axes (right, down, forward); map uses the official ROS frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    observation_id: str
    stamp: float
    panorama_rgb: np.ndarray
    T_map_sensor: np.ndarray
    registered_points_map: np.ndarray
    scan_stamp: float


@dataclass
class View:
    observation_id: str
    stamp: float
    view_id: str
    image_rgb: np.ndarray
    intrinsics: np.ndarray
    T_map_view: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    panorama_shape: tuple[int, int]


@dataclass
class Detection:
    observation_id: str
    stamp: float
    view_id: str
    concept: str
    box_2d: list[float]
    mask: np.ndarray
    score: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    observation_id: str
    stamp: float
    view_id: str
    concept: str
    score: float
    box_2d: list[float]
    panorama_mask: np.ndarray
    camera_position: np.ndarray
    measured_points: np.ndarray
    estimated_points: np.ndarray
    center: np.ndarray | None
    bbox: np.ndarray | None  # full axis-aligned extent, not a navigation goal
    geometry_quality: dict[str, Any]
    attributes: dict[str, Any] = field(default_factory=dict)
    appearance_descriptor: np.ndarray | None = None
    crop_rgb: np.ndarray | None = None


@dataclass(kw_only=True)
class VerifiedObservation(Observation):
    """One image region with its own positive category evidence."""

    category_evidence: dict[str, Any]


@dataclass
class ObjectRecord:
    id: str
    class_scores: dict[str, float]
    attributes: dict[str, Any]
    measured_points: np.ndarray
    estimated_points: np.ndarray
    center: np.ndarray | None
    bbox: np.ndarray | None
    geometry_quality: dict[str, Any]
    representative_crops: list[np.ndarray] = field(default_factory=list)
    appearance_descriptor: np.ndarray | None = None
    observation_ids: list[str] = field(default_factory=list)
    observation_poses: list[list[float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    current_relation_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    object_ids: list[str] = field(default_factory=list)
    value: int | None = None
    missing: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    details: dict[str, Any] = field(default_factory=dict)
