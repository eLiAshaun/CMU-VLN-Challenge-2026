"""Configuration loader - Single Source of Truth for all configuration.

This module is the ONLY place that reads:
- Environment variables
- YAML files
- JSON files
- CLI arguments

All other modules receive pre-loaded, validated, immutable Config objects.

Configuration precedence (highest to lowest):
1. Explicit CLI arguments passed to load_config()
2. Environment variables (explicit allowlist only - see ALLOWED_ENV_VARS)
3. YAML config file (if --config argument provided)
4. Current model_assets.json asset manifest
5. Schema defaults

After loading, configuration is immutable. Runtime state must be kept separate.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - the ROS image provides PyYAML
    yaml = None

from .schema import (
    AppConfig,
    GroundingDINOConfig,
    MASt3RConfig,
    MASt3RAny2FullConfig,
    ModelCheckpointsConfig,
    NavigationConfig,
    PathsConfig,
    PipelineConfig,
    TimeBudgetConfig,
    Qwen3VLConfig,
    RuntimeConfig,
    SAM2Config,
    YOLOWorldConfig,
    create_default_config,
)


def _parse_bool(value: str) -> bool:
    """Parse boolean from string with strict validation.

    Fixes the critical bug where bool("false") == True in Python.
    Environment variables are always strings, so we need explicit parsing.

    Args:
        value: String value to parse

    Returns:
        Boolean value

    Raises:
        ValueError: If value is not a valid boolean string

    Examples:
        >>> _parse_bool("true")
        True
        >>> _parse_bool("false")
        False
        >>> _parse_bool("1")
        True
        >>> _parse_bool("0")
        False
    """
    value_lower = value.lower().strip()
    if value_lower in ("true", "1", "yes", "on"):
        return True
    elif value_lower in ("false", "0", "no", "off", ""):
        return False
    else:
        raise ValueError(
            f"Invalid boolean value: {value!r}. "
            f"Valid values: true/false, 1/0, yes/no, on/off"
        )


# Explicit allowlist of environment variables that can override configuration
# This is the ONLY place environment variables are read in the entire application
ALLOWED_ENV_VARS = {
    # Paths
    "MAST3R_AI_ROOT": ("paths", "ai_root", Path),
    "MAST3R_OUTPUT_ROOT": ("paths", "output_root", Path),

    # Qwen3VL
    "MAST3R_QWEN_CHECKPOINT": ("qwen3vl", "checkpoint_path", str),
    "MAST3R_QWEN_ENDPOINT": ("qwen3vl", "endpoint", str),
    "MAST3R_QWEN_QUANTIZATION": ("qwen3vl", "quantization", str),
    "MAST3R_QWEN_MAX_PIXELS": ("qwen3vl", "max_pixels", int),
    "MAST3R_QWEN_BATCH_MAX_NEW_TOKENS": (
        "qwen3vl", "batch_max_new_tokens", int
    ),

    # SAM2
    "MAST3R_SAM_CHECKPOINT": ("sam2", "checkpoint_path", str),
    "MAST3R_SAM_CONFIG": ("sam2", "config_name", str),
    "MAST3R_SAM_ENDPOINT": ("sam2", "endpoint", str),

    # YOLO World
    "MAST3R_YOLO_ENABLED": ("yolo_world", "enabled", _parse_bool),
    "MAST3R_YOLO_ENDPOINT": ("yolo_world", "endpoint", str),

    # Runtime
    "CHALLENGE_QUESTION_TIME_BUDGET_SECONDS": (
        "runtime.time_budget",
        "question_time_budget_seconds",
        float,
    ),
}


def _load_asset_manifest(json_path: Path) -> dict[str, Any]:
    """Load the current model asset and runtime manifest."""
    if not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load {json_path}: {e}") from e


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load one typed YAML configuration document."""
    if not config_path.is_file():
        raise ValueError(f"Configuration file not found: {config_path}")
    if yaml is None:
        raise RuntimeError(
            "YAML configuration requested but PyYAML is unavailable: "
            f"{config_path}"
        )
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to load {config_path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Configuration root must be a mapping: {config_path}"
        )
    return dict(payload)


def _coerce_typed_config_value(current: Any, raw: Any, path: str) -> Any:
    """Coerce one YAML value against the current schema value.

    The schema is dataclass-based, so this intentionally rejects unknown keys
    and ambiguous scalar types instead of silently dropping configuration.
    """
    if is_dataclass(current):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must be a mapping")
        known = {item.name for item in fields(current)}
        unknown = sorted(str(key) for key in raw if key not in known)
        if unknown:
            raise ValueError(
                f"Unknown configuration field(s) at {path}: {', '.join(unknown)}"
            )
        updates = {
            item.name: _coerce_typed_config_value(
                getattr(current, item.name), raw[item.name], f"{path}.{item.name}"
            )
            for item in fields(current)
            if item.name in raw
        }
        return replace(current, **updates) if updates else current
    if isinstance(current, Path):
        if not isinstance(raw, (str, Path)):
            raise ValueError(f"{path} must be a path string")
        return Path(raw)
    if isinstance(current, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return _parse_bool(raw)
        raise ValueError(f"{path} must be boolean")
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(raw, bool):
            raise ValueError(f"{path} must be an integer")
        try:
            converted = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be an integer") from exc
        if isinstance(raw, float) and converted != raw:
            raise ValueError(f"{path} must be an integer")
        return converted
    if isinstance(current, float):
        if isinstance(raw, bool):
            raise ValueError(f"{path} must be numeric")
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must be numeric") from exc
    if isinstance(current, tuple):
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"{path} must be a sequence")
        template = current[0] if current else None
        return tuple(
            _coerce_typed_config_value(template, value, f"{path}[{index}]")
            if template is not None
            else value
            for index, value in enumerate(raw)
        )
    if isinstance(current, str):
        if not isinstance(raw, str):
            raise ValueError(f"{path} must be a string")
        return raw
    return raw


def _apply_yaml_to_config(config: AppConfig, payload: Mapping[str, Any]) -> AppConfig:
    """Apply a schema-shaped YAML document to the immutable config."""
    return _coerce_typed_config_value(config, payload, "config")


def _apply_asset_manifest_to_config(config: AppConfig, manifest: dict) -> AppConfig:
    """Apply the current asset manifest to the immutable typed config."""
    runtime = manifest.get("runtime", {})

    # Extract nested configuration sections
    pipeline_cfg = runtime.get("pipeline", {})
    mast3r_cfg = runtime.get("mast3r", {})
    yolo_cfg = runtime.get("yolo_world", {})
    dino_cfg = runtime.get("groundingdino", {})
    sam2_cfg = runtime.get("sam2", {})
    qwen_cfg = runtime.get("qwen3vl", {})
    nav_cfg = runtime.get("navigation", {})
    budget_cfg = runtime.get("time_budget", {})

    # Build updated config using replace (preserves immutability)
    updates = {}

    # MASt3R
    if mast3r_cfg:
        any2full_cfg = mast3r_cfg.get("any2full", {})
        updates["mast3r"] = replace(
            config.mast3r,
            enabled=mast3r_cfg.get("enabled", config.mast3r.enabled),
            required=mast3r_cfg.get("required", config.mast3r.required),
            reconstruction_mode=mast3r_cfg.get(
                "reconstruction_mode", config.mast3r.reconstruction_mode
            ),
            device=mast3r_cfg.get("device", config.mast3r.device),
            perspective_width=int(mast3r_cfg.get("perspective_width", config.mast3r.perspective_width)),
            perspective_height=int(mast3r_cfg.get("perspective_height", config.mast3r.perspective_height)),
            perspective_yaws_deg=tuple(mast3r_cfg.get("perspective_yaws_deg", config.mast3r.perspective_yaws_deg)),
            any2full=replace(
                config.mast3r.any2full,
                enabled=any2full_cfg.get("enabled", config.mast3r.any2full.enabled),
                encoder=any2full_cfg.get("encoder", config.mast3r.any2full.encoder),
            ) if any2full_cfg else config.mast3r.any2full,
        )

    # YOLO World
    if yolo_cfg:
        updates["yolo_world"] = replace(
            config.yolo_world,
            enabled=yolo_cfg.get("enabled", config.yolo_world.enabled),
            required=yolo_cfg.get("required", config.yolo_world.required),
            role=yolo_cfg.get("role", config.yolo_world.role),
            endpoint=yolo_cfg.get("endpoint", config.yolo_world.endpoint),
            score_threshold=float(yolo_cfg.get("score_threshold", config.yolo_world.score_threshold)),
        )

    # Grounding DINO
    if dino_cfg:
        updates["groundingdino"] = replace(
            config.groundingdino,
            enabled=dino_cfg.get("enabled", config.groundingdino.enabled),
            required=dino_cfg.get("required", config.groundingdino.required),
            role=dino_cfg.get("role", config.groundingdino.role),
        )

    # SAM2
    if sam2_cfg:
        updates["sam2"] = replace(
            config.sam2,
            enabled=sam2_cfg.get("enabled", config.sam2.enabled),
            endpoint=sam2_cfg.get("endpoint", config.sam2.endpoint),
            startup_timeout_seconds=float(sam2_cfg.get("startup_timeout_seconds", config.sam2.startup_timeout_seconds)),
        )

    # Qwen3VL
    if qwen_cfg:
        updates["qwen3vl"] = replace(
            config.qwen3vl,
            enabled=qwen_cfg.get("enabled", config.qwen3vl.enabled),
            endpoint=qwen_cfg.get("endpoint", config.qwen3vl.endpoint),
            quantization=qwen_cfg.get("quantization", config.qwen3vl.quantization),
            max_pixels=int(qwen_cfg.get("max_pixels", config.qwen3vl.max_pixels)),
            batch_max_new_tokens=int(
                qwen_cfg.get(
                    "batch_max_new_tokens", config.qwen3vl.batch_max_new_tokens
                )
            ),
            startup_timeout_seconds=float(qwen_cfg.get("startup_timeout_seconds", config.qwen3vl.startup_timeout_seconds)),
        )

    # Navigation
    if nav_cfg:
        updates["runtime"] = replace(
            config.runtime,
            navigation=replace(
                config.runtime.navigation,
                goal_timeout_seconds=float(nav_cfg.get("goal_timeout_seconds", config.runtime.navigation.goal_timeout_seconds)),
                acceptance_radius_m=float(nav_cfg.get("terrain_waypoint_acceptance_radius_m", config.runtime.navigation.acceptance_radius_m)),
                obstacle_height_threshold_m=float(nav_cfg.get("terrain_obstacle_height_threshold", config.runtime.navigation.obstacle_height_threshold_m)),
                obstacle_clearance_m=float(nav_cfg.get("terrain_obstacle_clearance_m", config.runtime.navigation.obstacle_clearance_m)),
                terrain_voxel_size_m=float(nav_cfg.get("terrain_voxel_size_m", config.runtime.navigation.terrain_voxel_size_m)),
                semantic_goal_roi_m=float(nav_cfg.get("semantic_goal_roi_m", config.runtime.navigation.semantic_goal_roi_m)),
                stop_surface_distance_m=float(nav_cfg.get("stop_surface_distance_m", config.runtime.navigation.stop_surface_distance_m)),
                stop_band_width_m=float(nav_cfg.get("stop_band_width_m", config.runtime.navigation.stop_band_width_m)),
                near_surface_distance_m=float(nav_cfg.get("near_surface_distance_m", config.runtime.navigation.near_surface_distance_m)),
                near_band_width_m=float(nav_cfg.get("near_band_width_m", config.runtime.navigation.near_band_width_m)),
                terrain_distance_weight=float(nav_cfg.get("terrain_distance_weight", config.runtime.navigation.terrain_distance_weight)),
                next_step_alignment_weight=float(nav_cfg.get("next_step_alignment_weight", config.runtime.navigation.next_step_alignment_weight)),
                geometry_uncertainty_weight=float(nav_cfg.get("geometry_uncertainty_weight", config.runtime.navigation.geometry_uncertainty_weight)),
            )
        )

    # Time budget. These values schedule work but never authorize an answer.
    if budget_cfg:
        current_runtime = updates.get("runtime", config.runtime)
        updates["runtime"] = replace(
            current_runtime,
            time_budget=replace(
                current_runtime.time_budget,
                question_time_budget_seconds=float(budget_cfg.get("question_time_budget_seconds", current_runtime.time_budget.question_time_budget_seconds)),
                answer_reserve_seconds=float(budget_cfg.get("answer_reserve_seconds", current_runtime.time_budget.answer_reserve_seconds)),
                initial_acquisition_estimate_seconds=float(budget_cfg.get("initial_acquisition_estimate_seconds", current_runtime.time_budget.initial_acquisition_estimate_seconds)),
            )
        )

    # Pipeline
    if pipeline_cfg:
        stage_order = pipeline_cfg.get("stage_order")
        current_runtime = updates.get("runtime", config.runtime)
        updates["runtime"] = replace(
            current_runtime,
            pipeline=replace(
                current_runtime.pipeline,
                stage_order=tuple(stage_order) if stage_order else current_runtime.pipeline.stage_order,
                canonical_evidence=pipeline_cfg.get(
                    "canonical_evidence", current_runtime.pipeline.canonical_evidence
                ),
                semantic_primary_input=pipeline_cfg.get(
                    "semantic_primary_input", current_runtime.pipeline.semantic_primary_input
                ),
                proposal_backend=pipeline_cfg.get(
                    "proposal_backend", current_runtime.pipeline.proposal_backend
                ),
                same_optical_center_counts_as_independent_evidence=bool(
                    pipeline_cfg.get(
                        "same_optical_center_counts_as_independent_evidence",
                        current_runtime.pipeline.same_optical_center_counts_as_independent_evidence,
                    )
                ),
                geometry_authority=pipeline_cfg.get(
                    "geometry_authority", current_runtime.pipeline.geometry_authority
                ),
            )
        )

    return replace(config, **updates)


def _apply_environment_overrides(config: AppConfig) -> AppConfig:
    """Apply environment variable overrides from explicit allowlist.

    Only environment variables in ALLOWED_ENV_VARS are checked.
    All other environment variables are ignored.
    """
    updates = {}

    for env_var, (path, field, converter) in ALLOWED_ENV_VARS.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        # Parse value
        try:
            if callable(converter):
                parsed = converter(value)
            else:
                parsed = value
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to parse {env_var}={value!r}: {e}"
            ) from e

        # Apply to config
        # Handle the single allowed nested runtime override.
        if "." in path:
            parts = path.split(".")
            # This is a nested update, needs special handling
            # For simplicity, handle common cases explicitly
            if path == "runtime.time_budget":
                current_runtime = updates.get("runtime", config.runtime)
                updates["runtime"] = replace(
                    current_runtime,
                    time_budget=replace(
                        current_runtime.time_budget,
                        **{field: parsed}
                    )
                )
        else:
            # Top-level section update
            current_section = updates.get(path, getattr(config, path))
            updates[path] = replace(current_section, **{field: parsed})

    return replace(config, **updates) if updates else config


def load_config(
    *,
    config_file: Path | None = None,
    asset_manifest_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load and validate configuration from all sources.

    Args:
        config_file: Optional YAML config file path
        asset_manifest_path: Optional path to the current model_assets.json
        cli_overrides: Optional explicit overrides from CLI arguments

    Returns:
        Validated, immutable AppConfig

    Raises:
        ValueError: If configuration is invalid

    Configuration precedence (highest to lowest):
        1. cli_overrides
        2. Environment variables (ALLOWED_ENV_VARS only)
        3. YAML config_file (if provided)
        4. model_assets.json asset manifest (if provided)
        5. Schema defaults
    """
    # Start with schema defaults
    config = create_default_config()

    # Apply the current asset manifest.
    if asset_manifest_path:
        manifest = _load_asset_manifest(asset_manifest_path)
        config = _apply_asset_manifest_to_config(config, manifest)

    # Apply schema-shaped YAML config file.
    if config_file:
        config = _apply_yaml_to_config(config, _load_yaml_config(config_file))

    # Apply environment variable overrides
    config = _apply_environment_overrides(config)

    # Apply CLI overrides
    if cli_overrides:
        config = replace(config, **cli_overrides)

    # Validate
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)

    return config


def write_effective_config(config: AppConfig, output_dir: Path) -> None:
    """Write effective configuration snapshot for audit trail.

    Writes two files:
    - effective_config.json: Complete resolved configuration
    - config_sources.json: Explicit resolution metadata.  Per-field source
      history is not retained by the immutable AppConfig object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write main config
    config_path = output_dir / "effective_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    sources_path = output_dir / "config_sources.json"
    with open(sources_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": config.schema_version,
                "provenance_available": False,
                "source": "resolved_immutable_app_config",
                "source_layers": [
                    "schema_defaults",
                    "asset_manifest_if_supplied",
                    "allowlisted_environment_if_supplied",
                    "cli_overrides_if_supplied",
                ],
                "note": (
                    "AppConfig stores resolved values only; individual source "
                    "history is not available at write_effective_config time."
                ),
                "effective_as_of": "runtime_start",
            },
            f,
            indent=2,
        )


def load_config_for_worker(
    worker_name: str,
    *,
    asset_manifest_path: Path | None = None,
) -> AppConfig:
    """Load configuration specifically for a worker server.

    Workers should use this instead of load_config() to get consistent behavior.

    Args:
        worker_name: Name of worker (for logging)
        asset_manifest_path: Path to model_assets.json

    Returns:
        Validated AppConfig
    """
    return load_config(asset_manifest_path=asset_manifest_path)
