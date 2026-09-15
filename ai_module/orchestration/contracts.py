"""Runtime contracts for the lean CMU-VLN production chain.

The ROS parent owns the ten-minute episode clock. Every acquisition process
receives the remaining budget and an explicit perception policy. A child must
never recreate a fresh 600-second deadline or reserve a fixed 75 seconds after
that reserve is no longer available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Deadline:
    """Child-process view of the parent-owned episode deadline.

    ``mandatory_reserve`` is supplied by the ROS parent for the current
    acquisition. It protects only publication / dispatch and is intentionally
    small. Perception profiles protect their own expected execution budget.
    """

    remaining_at_start: float
    mandatory_reserve: float = 10.0
    started_monotonic: float = field(default_factory=time.monotonic)

    def remaining(self) -> float:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        return max(0.0, float(self.remaining_at_start) - elapsed)

    def stage_timeout(self, stage_cap: float, *, reserve: float | None = None) -> float:
        protected = self.mandatory_reserve if reserve is None else max(0.0, float(reserve))
        usable = max(0.0, self.remaining() - protected)
        return max(0.0, min(max(0.0, float(stage_cap)), usable))

@dataclass
class StageResult:
    stage_name: str
    status: Literal["completed", "blocked", "timeout", "error", "skipped"]
    elapsed: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    def is_success(self) -> bool:
        return self.status in {"completed", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "elapsed": float(self.elapsed),
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


@dataclass
class PipelineSummary:
    """Single source of truth consumed by ``live_task_probe``."""

    schema_version: str = "lean_pipeline_v3"
    episode_id: str = ""
    acquisition_id: str = ""
    question: str = ""
    task_ir: dict[str, Any] = field(default_factory=dict)
    input_provenance: dict[str, Any] = field(default_factory=dict)
    deadline: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def set_stage(self, name: str, result: StageResult | dict[str, Any]) -> None:
        self.stages[name] = result.to_dict() if isinstance(result, StageResult) else dict(result)

    def set_root_decision(self, decision: dict[str, Any]) -> None:
        root = {
            "status": "completed",
            "decision": dict(decision),
        }
        self.stages["root_finalization"] = root

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "acquisition_id": self.acquisition_id,
            "question": self.question,
            "task_ir": self.task_ir,
            "input_provenance": self.input_provenance,
            "deadline": self.deadline,
            "stages": self.stages,
            "timing": self.timing,
        }


@dataclass(frozen=True)
class AcquisitionContext:
    episode_id: str
    acquisition_id: str
    question: str
    task_ir: dict[str, Any]
    image_path: Path
    scene_memory_path: Path
    deadline: Deadline
    competition_geometry: dict[str, Any] = field(default_factory=dict)
    navigation_config: dict[str, Any] = field(default_factory=dict)
    execution_steps: list[dict[str, Any]] = field(default_factory=list)
    current_step_index: int = 0
    trajectory_monitor: dict[str, Any] = field(default_factory=dict)
    semantic_entity_bindings: dict[str, int] = field(default_factory=dict)
    navigation_history: list[dict[str, Any]] = field(default_factory=list)
    scene_observation_entity_ids: list[str] = field(default_factory=list)
    station_id: str = ""
    station_is_new: bool = True
    geometry_manifest_path: str | None = None
    reconstruction_keyframes: list[dict[str, Any]] = field(default_factory=list)
    perception_policy: dict[str, Any] = field(default_factory=dict)
    evidence_need: dict[str, Any] | None = None
    arrival_transaction: dict[str, Any] = field(default_factory=dict)

    @property
    def perception_mode(self) -> str:
        return str(self.perception_policy.get("mode", "initial_full"))

    @property
    def perception_yaws_deg(self) -> tuple[float, ...]:
        values = self.perception_policy.get("yaws_deg", ())
        if not isinstance(values, (list, tuple)):
            return ()
        output: list[float] = []
        for value in values:
            try:
                output.append(float(value) % 360.0)
            except (TypeError, ValueError):
                continue
        return tuple(output)

    @property
    def perception_view_limit(self) -> int:
        try:
            return max(0, min(8, int(self.perception_policy.get("view_limit", 8))))
        except (TypeError, ValueError):
            return 8

    @property
    def perception_cap_seconds(self) -> float:
        try:
            return max(0.0, float(self.perception_policy.get("cap_seconds", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    @property
    def viewpoint_position_map(self) -> list[float] | None:
        path = Path(str(self.competition_geometry.get("state_estimation_path", "")))
        if not path.is_file():
            return None
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            position = payload.get("position_xyz") if isinstance(payload, dict) else None
            if isinstance(position, list) and len(position) >= 3:
                return [float(position[0]), float(position[1]), float(position[2])]
        except (OSError, ValueError, TypeError):
            return None
        return None


TIME_BUDGET = {
    # Publication/dispatch reserve. It is not a perception reserve and must not
    # prevent a final 20-30 second emergency acquisition.
    "mandatory_final_reserve_seconds": 10.0,
    "initial_perception_cap_seconds": 95.0,
    "evidence_perception_cap_seconds": 45.0,
    "world_refresh_cap_seconds": 24.0,
    "geometry_cap_seconds": 20.0,
    "memory_and_query_cap_seconds": 12.0,
}
