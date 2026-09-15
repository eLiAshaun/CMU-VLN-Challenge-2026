"""Single receding-horizon policy for ordered Instruction execution.

The executor owns only task-local StepSpec state.  TaskIR, QueryEntityView,
RelationEngine, SceneMemory geometry, the existing waypoint planner, and the
actual-trajectory monitor remain the factual authorities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class StepStatus(str, Enum):
    UNBOUND = "UNBOUND"
    READY = "READY"
    EXECUTING = "EXECUTING"
    SATISFIED = "SATISFIED"
    INVALIDATED = "INVALIDATED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ActiveWindow:
    current_index: int
    lookahead_index: int | None


def _finite_xyz(value: object) -> bool:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return False
    try:
        return all(math.isfinite(float(component)) for component in value)
    except (TypeError, ValueError):
        return False


def _positive_extent(value: object) -> bool:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return False
    try:
        return all(
            math.isfinite(float(component)) and float(component) > 0.0
            for component in value
        )
    except (TypeError, ValueError):
        return False


def _revision(snapshot: Mapping[str, Any]) -> dict[str, int]:
    return {
        "scene_revision": int(snapshot.get("scene_version", 0)),
        "identity_revision": int(snapshot.get("identity_revision", 0)),
        "geometry_revision": int(snapshot.get("geometry_version", 0)),
        "relation_revision": int(snapshot.get("relation_revision", 0)),
    }


def _semantic_support(obj: Mapping[str, Any]) -> float:
    values = [float(obj.get("semantic_probability", 0.0) or 0.0)]
    for evidence in obj.get("evidence", ()):
        if not isinstance(evidence, Mapping):
            continue
        for raw in evidence.get("qwen_verification_probabilities", ()):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if evidence.get("qwen_verified") is True:
            values.append(1.0)
    return max(values, default=0.0)


def _geometry_uncertainty(obj: Mapping[str, Any]) -> float:
    covariance = obj.get("center_cov")
    if not isinstance(covariance, Sequence):
        return float("inf")
    diagonal = []
    for index in range(min(2, len(covariance))):
        row = covariance[index]
        if not isinstance(row, Sequence) or len(row) <= index:
            return float("inf")
        try:
            value = float(row[index])
        except (TypeError, ValueError):
            return float("inf")
        if not math.isfinite(value) or value < 0.0:
            return float("inf")
        diagonal.append(value)
    return sum(diagonal) if diagonal else float("inf")


class RecedingHorizonInstructionExecutor:
    """Pure StepSpec policy used by the canonical query/runtime owners."""

    valid_statuses = frozenset(value.value for value in StepStatus)

    @staticmethod
    def compile(task_ir: Mapping[str, Any]) -> list[dict[str, Any]]:
        if str(task_ir.get("task_type", "")) != "instruction_following":
            return []
        relations = {
            str(value.get("id", "")): value
            for value in task_ir.get("relations", ())
            if isinstance(value, Mapping)
        }
        positive = {
            int(value["order"]): value
            for value in (task_ir.get("trajectory_ir") or {}).get(
                "ordered_path_constraints", ()
            )
            if isinstance(value, Mapping)
        }
        forbidden_orders = {
            int(value["order"])
            for value in (task_ir.get("trajectory_ir") or {}).get(
                "forbidden_path_regions", ()
            )
            if isinstance(value, Mapping)
        }
        raw_steps = [
            value
            for value in sorted(
                task_ir.get("ordered_trajectory_constraints", ()),
                key=lambda item: int(item["order"]),
            )
            if int(value["order"]) not in forbidden_orders
        ]
        steps: list[dict[str, Any]] = []
        for step_index, raw in enumerate(raw_steps):
            relation_ids = [str(value) for value in raw.get("relation_ids", ())]
            step_relations = [
                relations[value] for value in relation_ids if value in relations
            ]
            predicates = [
                str(value.get("predicate", "")).strip().lower()
                for value in step_relations
            ]
            predicate = (
                "ARGMAX_DISTANCE"
                if "farthest" in predicates
                else "ARGMIN_DISTANCE"
                if "closest" in predicates
                else str(raw.get("action", "go_to")).strip().upper()
            )
            anchor_entities = list(
                dict.fromkeys(
                    str(entity_id)
                    for relation in step_relations
                    for entity_id in relation.get("object_entities", ())
                )
            )
            source_order = int(raw["order"])
            trajectory = positive.get(source_order, {})
            steps.append(
                {
                    "step_index": step_index,
                    "source_order": source_order,
                    "predicate": predicate,
                    "operator": predicate,
                    "action": str(raw.get("action", "go_to")),
                    "target_slot": f"ordered_step_{step_index}.target",
                    "anchor_slots": [
                        f"ordered_step_{step_index}.anchor_{index}"
                        for index in range(len(anchor_entities))
                    ],
                    "target_entity": str(raw["target_entity"]),
                    "anchor_entities": anchor_entities,
                    "relation_ids": relation_ids,
                    "is_terminal": bool(raw.get("terminal", False)),
                    "trajectory_constraint": str(
                        trajectory.get(
                            "constraint",
                            "TERMINATE_INSIDE"
                            if raw.get("terminal")
                            else "ENTER_REGION",
                        )
                    ),
                    "trajectory_region_kind": str(
                        trajectory.get(
                            "region_kind",
                            "STOP_REGION" if raw.get("terminal") else "NEAR_REGION",
                        )
                    ),
                    "forbidden": False,
                    "status": StepStatus.UNBOUND.value,
                    "bound_object_id": None,
                    "bound_anchor_object_ids": [None] * len(anchor_entities),
                    "binding_revision": None,
                    "geometry_used_for_execution": None,
                    "invalidation_count": 0,
                }
            )
        return steps

    @staticmethod
    def normalize_steps(
        task_ir: Mapping[str, Any],
        steps: Sequence[Mapping[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        compiled = RecedingHorizonInstructionExecutor.compile(task_ir)
        if steps is None:
            return compiled
        if len(steps) != len(compiled):
            raise ValueError("execution_step_count_changed")
        normalized: list[dict[str, Any]] = []
        for expected, raw in zip(compiled, steps):
            value = {**expected, **dict(raw)}
            status = str(value.get("status", StepStatus.UNBOUND.value))
            legacy = {"UNRESOLVED": "UNBOUND", "BOUND": "READY"}
            status = legacy.get(status, status)
            if status not in RecedingHorizonInstructionExecutor.valid_statuses:
                raise ValueError("execution_step_status_invalid")
            value["status"] = status
            normalized.append(value)
        return normalized

    @staticmethod
    def window(step_count: int, current_step_index: int) -> ActiveWindow:
        if current_step_index < 0 or current_step_index > step_count:
            raise ValueError("current_step_index_invalid")
        lookahead = (
            current_step_index + 1
            if current_step_index + 1 < step_count
            else None
        )
        return ActiveWindow(current_step_index, lookahead)

    @staticmethod
    def eligible(obj: Mapping[str, Any]) -> bool:
        # SceneMemory's semantic_status is one evidence field, not a deletion
        # authority.  Instruction execution must still rank a current ATOMIC
        # detector/geometry hypothesis when every same-class candidate has a
        # negative semantic verification; otherwise one noisy verifier erases
        # all physical partial credit.  Explicit object/physical rejection,
        # non-atomic cardinality, and unusable geometry remain hard exclusions.
        return bool(
            str(obj.get("cardinality_role", "ATOMIC")) == "ATOMIC"
            and str(obj.get("status", "tentative")) not in {"rejected", "removed"}
            and str(obj.get("physical_status", "tentative"))
            not in {"rejected", "removed"}
            and _finite_xyz(obj.get("center_3d"))
            and _positive_extent(obj.get("bbox_3d"))
        )

    @staticmethod
    def object_rank(obj: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            _semantic_support(obj),
            str(obj.get("semantic_status", "")) == "verified",
            str(obj.get("status", "")) == "confirmed",
            int(obj.get("independent_viewpoint_count", 0) or 0),
            len(obj.get("evidence", ())),
            -_geometry_uncertainty(obj),
            -int(obj["object_id"]),
        )

    @staticmethod
    def best_atomic(
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        eligible = [dict(value) for value in candidates if RecedingHorizonInstructionExecutor.eligible(value)]
        return max(eligible, key=RecedingHorizonInstructionExecutor.object_rank) if eligible else None

    @staticmethod
    def distinct_anchor_pair(
        first_domain: Sequence[Mapping[str, Any]],
        second_domain: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        pairs = []
        for first in first_domain:
            if not RecedingHorizonInstructionExecutor.eligible(first):
                continue
            for second in second_domain:
                if not RecedingHorizonInstructionExecutor.eligible(second):
                    continue
                first_id = int(first["object_id"])
                second_id = int(second["object_id"])
                if first_id == second_id:
                    continue
                ranks = sorted(
                    (
                        RecedingHorizonInstructionExecutor.object_rank(first),
                        RecedingHorizonInstructionExecutor.object_rank(second),
                    )
                )
                pairs.append((ranks[0], ranks[1], -first_id, -second_id, first, second))
        if not pairs:
            return []
        selected = max(pairs)
        return [dict(selected[4]), dict(selected[5])]

    @staticmethod
    def distance_winner(
        candidates: Sequence[Mapping[str, Any]],
        anchor: Mapping[str, Any],
        *,
        farthest: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not RecedingHorizonInstructionExecutor.eligible(anchor):
            return None, None
        ranked = []
        anchor_xy = [float(value) for value in anchor["center_3d"][:2]]
        for candidate in candidates:
            if not RecedingHorizonInstructionExecutor.eligible(candidate):
                continue
            distance = math.dist(
                [float(value) for value in candidate["center_3d"][:2]],
                anchor_xy,
            )
            metric = distance if farthest else -distance
            ranked.append(
                (
                    metric,
                    RecedingHorizonInstructionExecutor.object_rank(candidate),
                    -int(candidate["object_id"]),
                    dict(candidate),
                )
            )
        ranked.sort(reverse=True)
        winner = ranked[0][3] if ranked else None
        challenger = next(
            (
                value[3]
                for value in ranked[1:]
                if winner is not None
                and int(value[3]["object_id"]) != int(winner["object_id"])
            ),
            None,
        )
        return winner, challenger

    @staticmethod
    def bind(
        step: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        *,
        target: Mapping[str, Any] | None,
        anchors: Sequence[Mapping[str, Any]] = (),
        between_anchor_only: bool = False,
        lookahead_only: bool = False,
    ) -> dict[str, Any]:
        value = dict(step)
        anchor_ids = [int(item["object_id"]) for item in anchors]
        if between_anchor_only:
            target_id = None
        else:
            target_id = int(target["object_id"]) if target is not None else None
        value.update(
            status=(
                str(value.get("status"))
                if str(value.get("status")) == StepStatus.EXECUTING.value
                else StepStatus.READY.value
            ),
            bound_object_id=target_id,
            bound_anchor_object_ids=anchor_ids,
            binding_revision=_revision(snapshot),
            between_anchor_only_binding=bool(between_anchor_only),
        )
        if lookahead_only:
            value["lookahead_only"] = True
        return value

    @staticmethod
    def block(step: Mapping[str, Any], reason: str) -> dict[str, Any]:
        value = dict(step)
        if str(value.get("status")) != StepStatus.EXECUTING.value:
            value["status"] = StepStatus.BLOCKED.value
        value["blocked_reason"] = str(reason)
        return value

    @staticmethod
    def invalidate(step: Mapping[str, Any], reason: str) -> dict[str, Any]:
        value = dict(step)
        value["status"] = StepStatus.INVALIDATED.value
        value["invalidation_reason"] = str(reason)
        value["invalidation_count"] = int(value.get("invalidation_count", 0) or 0) + 1
        return value

    @staticmethod
    def execution_binding_valid(
        step: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> bool:
        object_by_id = {
            int(value["object_id"]): value
            for value in snapshot.get("objects", ())
            if isinstance(value, Mapping)
            and str(value.get("object_id", "")).lstrip("-").isdigit()
        }
        target_id = step.get("bound_object_id")
        if target_id is not None:
            target = object_by_id.get(int(target_id))
            if target is None or not RecedingHorizonInstructionExecutor.eligible(target):
                return False
        anchors = [value for value in step.get("bound_anchor_object_ids", ()) if value is not None]
        if str(step.get("action", "")) == "pass_between" and len(set(anchors)) != 2:
            return False
        return all(
            int(value) in object_by_id
            and RecedingHorizonInstructionExecutor.eligible(object_by_id[int(value)])
            for value in anchors
        )

    @staticmethod
    def revision(snapshot: Mapping[str, Any]) -> dict[str, int]:
        return _revision(snapshot)
