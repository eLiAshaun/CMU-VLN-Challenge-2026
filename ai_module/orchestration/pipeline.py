"""Production entry point for the lean acquisition chain."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

from integrations.execution.root_finalizer import finalize_system_failure
from orchestration.contracts import AcquisitionContext, Deadline, PipelineSummary
from utils.validation import validate_episode_request


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remaining_from_request(request: Mapping[str, Any], runtime: Mapping[str, Any]) -> float:
    def finite_remaining(value: object) -> float | None:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(candidate):
            return None
        return max(0.0, candidate)

    direct = request.get("episode_remaining_seconds")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return finite_remaining(direct) or 0.0
    raise ValueError("episode_remaining_seconds_missing")


def _failure_decision(question: str, task_type: str, reason: str) -> dict[str, Any]:
    return finalize_system_failure(
        {
            "original_question": str(question),
            "task_type": str(task_type or "unknown"),
        },
        reason=reason,
    )


def _runtime_summary_blocks(
    request: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose episode-owned execution state beside stage diagnostics."""
    stages = payload.get("stages", {})
    if not isinstance(stages, Mapping):
        stages = {}
    query_stage = stages.get("query_execution", {})
    query_stage = query_stage if isinstance(query_stage, Mapping) else {}
    execution = query_stage.get("execution", {})
    execution = execution if isinstance(execution, Mapping) else {}
    selection = stages.get("waypoint_selection", {})
    selection = selection if isinstance(selection, Mapping) else {}
    root = stages.get("root_finalization", {})
    root = root if isinstance(root, Mapping) else {}
    decision = root.get("decision", {})
    decision = decision if isinstance(decision, Mapping) else {}
    steps = execution.get("execution_steps", request.get("execution_steps", ()))
    if isinstance(steps, tuple):
        steps = [dict(value) for value in steps if isinstance(value, Mapping)]
    elif not isinstance(steps, list):
        steps = []
    current_step = execution.get(
        "current_step_index", request.get("current_step_index", 0)
    )
    trajectory_monitor = request.get("trajectory_monitor", {})
    if not isinstance(trajectory_monitor, Mapping):
        trajectory_monitor = {}
    try:
        current_step_index = int(current_step)
    except (TypeError, ValueError):
        current_step_index = 0
    navigation_config = request.get("navigation_config", {})
    ack_enabled = bool(
        navigation_config.get("waypoint_ack_diagnostics_enabled", False)
        if isinstance(navigation_config, Mapping)
        else False
    )
    return {
        "instruction_execution": {
            "source": "episode_state.execution_steps",
            "episode_id": str(request.get("episode_id", "")),
            "current_step_index": current_step_index,
            "steps": steps,
            "completion_authority": "actual_state_estimation_trajectory",
        },
        "trajectory_monitor": {
            "authoritative_for_semantic_progress": True,
            "completion_authority": str(
                trajectory_monitor.get(
                    "completion_authority",
                    "actual_state_estimation_trajectory",
                )
            ),
            "state": dict(trajectory_monitor),
        },
        "waypoint_execution": {
            "source": "navigation_executor",
            "progress_authority": "/state_estimation",
            "way_point_reached_role": "diagnostic_only",
            "way_point_reached_diagnostics_enabled": ack_enabled,
            "waypoint_context": decision.get("waypoint_context"),
            "waypoints": decision.get("waypoint_sequence", []),
            "action": decision.get("action"),
        },
        "resolver_result": dict(
            stages.get("task_resolution", {}).get("result", {})
            if isinstance(stages.get("task_resolution"), Mapping)
            else {}
        ),
        "evidence_acquisition": dict(
            stages.get("evidence_acquisition_input", {})
            if isinstance(stages.get("evidence_acquisition_input"), Mapping)
            else {}
        ),
    }


def run_episode_chain(request: dict[str, Any], config: Mapping[str, Any] | None) -> dict[str, Any]:
    output = Path(str(request["output_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = dict((config or {}).get("runtime", {}))
    question = str(request.get("question", ""))
    task_ir = request.get("compiled_task_ir")
    missing_task_ir = not isinstance(task_ir, dict) or not str(
        task_ir.get("task_type", "")
    ).strip()
    if missing_task_ir:
        task_ir = {}
    else:
        task_ir = dict(task_ir)
        task_ir.setdefault("original_question", question)

    remaining = _remaining_from_request(request, runtime)
    if "mandatory_reserve_seconds" not in request:
        raise ValueError("mandatory_reserve_seconds_missing")
    mandatory_reserve = max(0.0, float(request["mandatory_reserve_seconds"]))

    summary = PipelineSummary(
        episode_id=str(request.get("episode_id", "")),
        acquisition_id=str(request.get("acquisition_id", output.name)),
        question=question,
        task_ir=task_ir,
        deadline={
            "remaining_seconds_at_child_start": remaining,
            "mandatory_reserve_seconds": mandatory_reserve,
            "clock_owner": "ros_parent_monotonic_episode_clock",
        },
    )
    started = time.monotonic()

    try:
        if missing_task_ir:
            raise ValueError("compiled_task_ir_missing")
        summary.input_provenance = validate_episode_request(request, output)
        scene_memory_path = Path(str(request["scene_memory_path"])).resolve()
        competition_geometry = dict(request.get("competition_geometry", {}))
        navigation_config = dict(runtime.get("navigation", {}))
        navigation_config.update(dict(request.get("navigation_config", {})))

        context = AcquisitionContext(
            episode_id=summary.episode_id,
            acquisition_id=summary.acquisition_id,
            question=question,
            task_ir=task_ir,
            image_path=Path(str(request["image_path"])).resolve(),
            scene_memory_path=scene_memory_path,
            deadline=Deadline(
                remaining_at_start=remaining,
                mandatory_reserve=mandatory_reserve,
            ),
            competition_geometry=competition_geometry,
            navigation_config=navigation_config,
            execution_steps=[dict(value) for value in request.get("execution_steps", ())],
            current_step_index=int(request.get("current_step_index", 0)),
            trajectory_monitor=dict(request.get("trajectory_monitor", {})),
            semantic_entity_bindings={
                str(key): int(value)
                for key, value in dict(request.get("semantic_entity_bindings", {})).items()
            },
            navigation_history=[dict(value) for value in request.get("navigation_history", ())],
            scene_observation_entity_ids=[
                str(value) for value in request.get("scene_observation_entity_ids", ())
            ],
            station_id=str(request.get("station_id", summary.acquisition_id)),
            station_is_new=bool(request.get("station_is_new", True)),
            geometry_manifest_path=(
                str(request["geometry_manifest_path"])
                if request.get("geometry_manifest_path") else None
            ),
            reconstruction_keyframes=[
                dict(value) for value in request.get("reconstruction_keyframes", ())
            ],
            perception_policy=dict(request.get("perception_policy", {})),
            evidence_need=(
                dict(request["evidence_need"])
                if isinstance(request.get("evidence_need"), Mapping)
                else None
            ),
            arrival_transaction=(
                dict(request["arrival_transaction"])
                if isinstance(request.get("arrival_transaction"), Mapping)
                else {}
            ),
        )

        from orchestration.lean_pipeline import LeanPipeline

        summary = LeanPipeline(runtime).run_single_acquisition(context, output, summary=summary)
    except Exception as exc:
        summary.set_root_decision(
            _failure_decision(
                question,
                str(task_ir.get("task_type", "unknown")),
                f"pipeline_exception:{type(exc).__name__}:{str(exc)[:300]}",
            )
        )
        summary.timing = {
            "total_elapsed_seconds": max(0.0, time.monotonic() - started),
            "remaining_seconds": max(0.0, remaining - (time.monotonic() - started)),
        }
        _atomic_json(output / "summary.json", summary.to_dict())

    payload = summary.to_dict()
    payload.update(_runtime_summary_blocks(request, payload))
    _atomic_json(output / "summary.json", payload)
    _atomic_json(
        output / "result.json",
        {
            "status": "completed",
            "episode_id": summary.episode_id,
            "acquisition_id": summary.acquisition_id,
            "output_dir": str(output),
            "root_decision": payload.get("stages", {}).get("root_finalization", {}).get("decision", {}),
        },
    )
    return payload
