"""Typed configuration schema for the ai_module runtime.

This module defines the single source of truth for all configuration parameters.
Every configurable parameter must be defined here with:
- Type annotation
- Default value
- Validation rules
- Documentation

Configuration is immutable after loading. Runtime state must be kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class EpisodeState(Enum):
    """Explicit episode lifecycle states."""
    IDLE = "idle"
    QUESTION_RECEIVED = "question_received"
    PREFLIGHT = "preflight"
    READY = "ready"
    RUNNING = "running"
    FINISHING = "finishing"
    TIMEOUT = "timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(Enum):
    """Worker process health status."""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class MASt3RAny2FullConfig:
    """Any2Full depth refinement configuration."""
    enabled: bool = False
    encoder: str = "vitb"
    depth_scale: float = 100.0
    denoise: bool = False


@dataclass(frozen=True)
class MASt3RConfig:
    """MASt3R 3D reconstruction configuration."""
    enabled: bool = False
    required: bool = False
    reconstruction_mode: str = "disabled"
    accept_equirectangular: bool = False
    projection_adapter_enabled: bool = True
    panorama_vertical_fov_deg: float = 120.0
    perspective_width: int = 512
    perspective_height: int = 384
    perspective_horizontal_fov_deg: float = 90.0
    perspective_yaws_deg: tuple[float, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    perspective_pitches_deg: tuple[float, ...] = (0,)
    device: str = "cuda"
    coarse_iterations: int = 300
    refine_iterations: int = 300
    matching_confidence_threshold: float = 5.0
    pointmap_confidence_threshold: float = 1.5
    max_keyframe_stations: int = 6
    any2full: MASt3RAny2FullConfig = field(default_factory=MASt3RAny2FullConfig)


@dataclass(frozen=True)
class YOLOWorldConfig:
    """YOLO-World detector configuration."""
    enabled: bool = True
    required: bool = True
    role: str = "fixed_task_conditioned_proposal_ensemble"
    score_threshold: float = 0.20
    nms_iou_threshold: float = 0.65
    endpoint: str = "@mast3r_yolo"
    model_config_path: str = "third_party/YOLO-World/configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_cc3mlite_train_lvis_minival.py"
    checkpoint_path: str = "checkpoints/yolo_world/x_stage1-62b674ad.pth"
    text_model_path: str = "checkpoints/text_encoders/clip-vit-base-patch32"
    device: str = "cuda"
    startup_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class GroundingDINOConfig:
    """Grounding DINO detector configuration."""
    enabled: bool = False
    required: bool = False
    role: str = "disabled"
    box_threshold: float = 0.22
    text_threshold: float = 0.2


@dataclass(frozen=True)
class SAM2Config:
    """SAM2 segmentation configuration."""
    enabled: bool = True
    required: bool = True
    endpoint: str = "@mast3r_sam2"
    keep_alive: bool = True
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 30.0
    checkpoint_path: str = "checkpoints/sam2/sam2.1_hiera_base_plus.pt"
    config_name: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device: str = "cuda"


@dataclass(frozen=True)
class Qwen3VLConfig:
    """Qwen3-VL vision-language model configuration."""
    enabled: bool = True
    required: bool = True
    endpoint: str = "@mast3r_qwen3vl"
    checkpoint_path: str = "checkpoints/qwen3vl/Qwen3-VL-8B-Instruct"
    quantization: Literal["int8", "int4", "bf16"] = "int8"
    device: str = "cuda"
    max_pixels: int = 1048576  # 1024x1024
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 120.0
    batch_verification_enabled: bool = True
    batch_verification_size: int = 32
    batch_verification_timeout_seconds: float = 300.0
    batch_max_new_tokens: int = 256


@dataclass(frozen=True)
class ModelCheckpointsConfig:
    """Model checkpoint paths."""
    mast3r: str = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    qwen3vl: str = "checkpoints/qwen3vl/Qwen3-VL-8B-Instruct"
    sam2: str = "checkpoints/sam2/sam2.1_hiera_base_plus.pt"
    yolo_world: str = "checkpoints/yolo_world/x_stage1-62b674ad.pth"
    text_encoder: str = "checkpoints/text_encoders/clip-vit-base-patch32"


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline orchestration configuration."""
    stage_order: tuple[str, ...] = (
        "task_compilation",
        "task_conditioned_perception",
        "lidar_geometry",
        "scene_memory",
        "query_domain",
        "query_execution",
        "waypoint_selection",
        "root_finalization",
    )
    canonical_evidence: str = "raw_equirectangular_panorama"
    semantic_primary_input: str = "raw_panorama_then_high_resolution_perspective_views"
    proposal_backend: str = "qwen3vl"
    same_optical_center_counts_as_independent_evidence: bool = False
    geometry_authority: str = "official_lidar_and_state_estimation_map_frame"


@dataclass(frozen=True)
class NavigationConfig:
    """Navigation and waypoint selection configuration."""
    goal_timeout_seconds: float = 120.0
    acceptance_radius_m: float = 0.3
    obstacle_height_threshold_m: float = 0.05
    obstacle_clearance_m: float = 0.75
    terrain_voxel_size_m: float = 0.05
    semantic_goal_roi_m: float = 2.5
    stop_surface_distance_m: float = 0.9
    stop_band_width_m: float = 0.45
    near_surface_distance_m: float = 1.1
    near_band_width_m: float = 0.8
    terrain_distance_weight: float = 0.2
    next_step_alignment_weight: float = 0.15
    geometry_uncertainty_weight: float = 0.1
    terrain_paths: tuple[str, ...] = (
        "/terrain_map.npy",
        "/tmp/terrain_map.npy",
    )
    state_estimation_path: str = "/tmp/state_estimation.json"


@dataclass(frozen=True)
class TimeBudgetConfig:
    """Episode time-budget parameters; this class has no answer authority."""
    question_time_budget_seconds: float = 600.0
    answer_reserve_seconds: float = 10.0
    initial_acquisition_estimate_seconds: float = 90.0
    heartbeat_interval_seconds: float = 5.0


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime behavior configuration."""
    enabled: bool = True
    perception_container_image: str = "docker_ai_module:mast3r-live"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    time_budget: TimeBudgetConfig = field(default_factory=TimeBudgetConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    resume_previous_episode: bool = False
    max_run_directories: int = 100


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths configuration."""
    ai_root: Path = Path("/home/docker/ai_module")
    output_root: Path = Path("/home/docker/ai_module/runs/live_robot")
    cache_root: Path = Path("/home/docker/ai_module/cache")
    config_dir: Path = Path("/home/docker/ai_module/configs")
    checkpoint_dir: Path = Path("/home/docker/ai_module/checkpoints")


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration.

    This is the single source of truth for all runtime configuration.
    After loading, this object is immutable. All parameters are explicitly typed.

    Configuration precedence (highest to lowest):
    1. Explicit CLI arguments (passed to loader)
    2. Environment variables (explicit allowlist only)
    3. YAML config file (if provided)
    4. Defaults defined in this schema

    To modify behavior, change configuration at load time. Do not modify after loading.
    """
    schema_version: str = "2.1"

    # Paths
    paths: PathsConfig = field(default_factory=PathsConfig)

    # Models
    mast3r: MASt3RConfig = field(default_factory=MASt3RConfig)
    yolo_world: YOLOWorldConfig = field(default_factory=YOLOWorldConfig)
    groundingdino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    qwen3vl: Qwen3VLConfig = field(default_factory=Qwen3VLConfig)

    # Model checkpoints
    checkpoints: ModelCheckpointsConfig = field(default_factory=ModelCheckpointsConfig)

    # Runtime
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns empty list if configuration is valid.
        Raises no exceptions - caller decides whether to fail or warn.
        """
        errors = []

        # Validate quantization compatibility
        if self.qwen3vl.quantization not in ("int8", "int4", "bf16"):
            errors.append(
                f"Invalid qwen3vl.quantization: {self.qwen3vl.quantization}. "
                f"Must be int8, int4, or bf16."
            )

        # Validate required workers are enabled
        if self.sam2.required and not self.sam2.enabled:
            errors.append("sam2 is required but disabled")

        if self.qwen3vl.required and not self.qwen3vl.enabled:
            errors.append("qwen3vl is required but disabled")

        if self.mast3r.required and not self.mast3r.enabled:
            errors.append("mast3r is required but disabled")

        # Validate timeouts are positive
        if self.runtime.time_budget.question_time_budget_seconds <= 0:
            errors.append("question_time_budget_seconds must be positive")

        if self.qwen3vl.batch_verification_timeout_seconds <= 0:
            errors.append("batch_verification_timeout_seconds must be positive")

        # Validate max_pixels is reasonable
        if self.qwen3vl.max_pixels < 64 * 64:
            errors.append(f"qwen3vl.max_pixels too small: {self.qwen3vl.max_pixels}")
        if self.qwen3vl.max_pixels > 4096 * 4096:
            errors.append(f"qwen3vl.max_pixels too large: {self.qwen3vl.max_pixels}")

        # Validate device strings
        valid_devices = ("cuda", "cuda:0", "cuda:1", "cpu")
        for model_name, device in [
            ("qwen3vl", self.qwen3vl.device),
            ("sam2", self.sam2.device),
            ("yolo_world", self.yolo_world.device),
            ("mast3r", self.mast3r.device),
        ]:
            if device not in valid_devices:
                errors.append(
                    f"{model_name}.device invalid: {device}. "
                    f"Must be one of {valid_devices}"
                )

        # Validate paths exist (for checkpoints that are required)
        ai_root = self.paths.ai_root
        if self.qwen3vl.enabled:
            checkpoint_path = ai_root / self.qwen3vl.checkpoint_path
            if not checkpoint_path.exists():
                errors.append(f"Qwen3VL checkpoint not found: {checkpoint_path}")

        if self.sam2.enabled:
            checkpoint_path = ai_root / self.sam2.checkpoint_path
            if not checkpoint_path.exists():
                errors.append(f"SAM2 checkpoint not found: {checkpoint_path}")

        if self.mast3r.enabled:
            checkpoint_path = ai_root / self.checkpoints.mast3r
            if not checkpoint_path.exists():
                errors.append(f"MASt3R checkpoint not found: {checkpoint_path}")

        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Used for writing effective config snapshots.
        """
        from dataclasses import asdict
        return asdict(self)


def create_default_config() -> AppConfig:
    """Create configuration with all defaults from schema."""
    return AppConfig()
