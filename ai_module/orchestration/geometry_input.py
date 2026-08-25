"""Geometry input contract per Stage 1 §15."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GeometryInput:
    """明确的数据结构用于 LiDAR geometry lift。
    
    不要在函数内部依靠隐式全局路径猜数据来源。
    """
    observation_id: str
    acquisition_id: str
    station_id: str
    view_id: str
    
    image_path: str
    mask_path: str
    bbox_xyxy: tuple[float, float, float, float]
    
    canonical_class: str
    semantic_probability: float
    
    sensor_scan_path: str
    registered_scan_path: str
    state_estimation_path: str
    camera_contract_path: str
