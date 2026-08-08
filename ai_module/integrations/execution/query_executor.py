"""Canonical count, unique-object, relation, and route-reference execution."""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence


def execute_semantic_relation_count(
    task_ir: Mapping[str, Any],
    anchor_counts: Sequence[Mapping[str, Any]],
    *,
    acquisition_id: str,
    probability_threshold: float,
    geometry_reconstruction_id: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a simple numerical relation directly from fused VLM evidence.

    This is the bring-up path: panorama fusion owns duplicate suppression and
    Qwen verifies both the target class and its requested local relation.  3D
    geometry remains diagnostic and does not block this numerical result.
    Model probabilities are retained as uncertainty metadata instead of being
    used as a hard answer gate.
    """
    if str(task_ir.get("task_type")) != "numerical":
        return None
    target_id = str(task_ir.get("target_entity", ""))
    entities = {str(item["id"]): item for item in task_ir.get("entities", ())}
    target_entity = entities.get(target_id)
    relations = [
        item for item in task_ir.get("relations", ())
        if str(item.get("subject_entity")) == target_id
    ]
    if target_entity is None or len(relations) != 1:
        return None
    relation = relations[0]
    if str(relation.get("predicate", "")).lower() != "on" or len(relation.get("object_entities", ())) != 1:
        return None
    usable = []
    for item in anchor_counts:
        metadata = item.get("response", {}).get("metadata", {})
        count = metadata.get("contained_instance_count")
        if (
            item.get("response", {}).get("ok") is True
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            and (
                count > 0
                or float(metadata.get("anchor_probability") or 0.0) > 0.0
            )
        ):
            usable.append(item)
    if not usable:
        return None
    selected = max(
        usable,
        key=lambda item: float(item.get("response", {}).get("metadata", {}).get("count_confidence") or 0.0),
    )
    answer = int(selected["response"]["metadata"]["contained_instance_count"])
    return {
        "schema_version": "task_execution_v1",
        "task_type": "numerical",
        "scene_version": 0,
        "resolved": True,
        "answer": answer,
        "selected_object": None,
        "route_objects": [],
        "probe_object": None,
        "evidence_ids": [f"{acquisition_id}:{selected['observation_id']}:anchor_local_count"],
        "failed_constraints": [],
        "resolution_mode": "semantic_relation_count_bringup",
        "anchor_observation_id": str(selected["observation_id"]),
        "anchor_local_count": answer,
        "uncertainty": {
            "target_probability": float(
                selected["response"]["metadata"].get("target_probability") or 0.0
            ),
            "anchor_probability": float(
                selected["response"]["metadata"].get("anchor_probability") or 0.0
            ),
            "count_confidence": float(
                selected["response"]["metadata"].get("count_confidence") or 0.0
            ),
            "configured_probability_reference": float(probability_threshold),
        },
        "geometry_reconstruction_ids": (
            [geometry_reconstruction_id] if geometry_reconstruction_id else []
        ),
    }


def _distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return math.dist(tuple(first["center_3d"]), tuple(second["center_3d"]))


def _bounds(obj: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    center = [float(value) for value in obj["center_3d"]]
    half = [0.5 * float(value) for value in obj["bbox_3d"]]
    return ([center[i] - half[i] for i in range(3)], [center[i] + half[i] for i in range(3)])


def _horizontal_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    a0, a1 = _bounds(first)
    b0, b1 = _bounds(second)
    return a0[0] <= b1[0] and b0[0] <= a1[0] and a0[1] <= b1[1] and b0[1] <= a1[1]


def _predicate_supported(predicate: str, subject: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]]) -> bool:
    predicate = predicate.lower()
    if predicate == "near" and len(anchors) == 1:
        scale = 0.5 * (
            math.hypot(*subject["bbox_3d"][:2]) + math.hypot(*anchors[0]["bbox_3d"][:2])
        )
        return _distance(subject, anchors[0]) <= max(1.25, scale + 0.5)
    if predicate in {"above", "below", "on", "in"} and len(anchors) == 1:
        subject_low, subject_high = _bounds(subject)
        anchor_low, anchor_high = _bounds(anchors[0])
        if predicate == "above":
            return subject["center_3d"][2] > anchors[0]["center_3d"][2] + 0.10
        if predicate == "below":
            return subject["center_3d"][2] < anchors[0]["center_3d"][2] - 0.10
        if predicate == "on":
            gap = abs(subject_low[2] - anchor_high[2])
            # Small-object 3D bboxes are noisy; ray-surface projection
            # (hybrid lift) gives a better Z but the anchor bbox still
            # comes from MASt3R.  Accept a wider Z tolerance with any XY
            # overlap evidence.
            overlap_ok = _horizontal_overlap(subject, anchors[0])
            if not overlap_ok:
                x_overlap = (
                    subject_low[0] <= anchor_high[0]
                    and subject_high[0] >= anchor_low[0]
                )
                y_overlap = (
                    subject_low[1] <= anchor_high[1]
                    and subject_high[1] >= anchor_low[1]
                )
                overlap_ok = x_overlap or y_overlap
            return gap <= 0.70 and overlap_ok
        tolerance = 0.15
        return all(subject_low[i] >= anchor_low[i] - tolerance and subject_high[i] <= anchor_high[i] + tolerance for i in range(3))
    if predicate == "between" and len(anchors) == 2:
        point = subject["center_3d"][:2]
        first = anchors[0]["center_3d"][:2]
        second = anchors[1]["center_3d"][:2]
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared < 0.25:
            return False
        t = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / length_squared
        projected = (first[0] + t * dx, first[1] + t * dy)
        perpendicular = math.hypot(point[0] - projected[0], point[1] - projected[1])
        return 0.10 <= t <= 0.90 and perpendicular <= max(0.60, 0.25 * math.sqrt(length_squared))
    return False


class _Resolver:
    def __init__(self, task_ir: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        self.task_ir = task_ir
        self.snapshot = snapshot
        self.entities = {str(item["id"]): item for item in task_ir["entities"]}
        self.relations = list(task_ir["relations"])
        self.objects = [
            item for item in snapshot["objects"]
            if item.get("status") in {"tentative", "confirmed"}
        ]
        self.cache: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
        self.stack: set[str] = set()

    def resolve(self, entity_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        if entity_id in self.cache:
            return self.cache[entity_id]
        if entity_id in self.stack:
            return [], [f"relation_dependency_cycle:{entity_id}"]
        entity = self.entities.get(entity_id)
        if entity is None:
            return [], [f"entity_missing:{entity_id}"]
        if entity.get("attributes"):
            return [], [f"attribute_evidence_unavailable:{entity_id}"]
        self.stack.add(entity_id)
        class_names = {str(entity["class_name"]).lower(), *(str(value).lower() for value in entity.get("aliases", []))}
        candidates = [item for item in self.objects if str(item["class_label"]).lower() in class_names]
        failures: list[str] = []
        for relation in self.relations:
            if relation["subject_entity"] != entity_id:
                continue
            anchor_sets = []
            for anchor_id in relation["object_entities"]:
                anchors, anchor_failures = self.resolve(str(anchor_id))
                failures.extend(anchor_failures)
                if not anchors:
                    failures.append(f"relation_anchor_missing:{relation['id']}:{anchor_id}")
                    anchor_sets.append([])
                    continue
                # Pass all anchors so the predicate check can try each
                # one instead of locking onto the highest-confidence pick.
                anchor_sets.append(sorted(
                    anchors,
                    key=lambda item: (
                        float(item.get("semantic_probability", 0.0)),
                        len(item.get("evidence", ())),
                        item.get("status") == "confirmed",
                        -int(item["object_id"]),
                    ),
                    reverse=True,
                ))
            if failures or any(not values for values in anchor_sets):
                candidates = []
                break
            predicate = str(relation["predicate"]).lower()
            if predicate in {"closest", "farthest"}:
                anchors_flat = [values[0] for values in anchor_sets]
                if len(anchors_flat) != 1 or not candidates:
                    failures.append(f"ranking_relation_unresolved:{relation['id']}")
                    candidates = []
                    break
                ranked = sorted(
                    ((_distance(item, anchors_flat[0]), int(item["object_id"]), item) for item in candidates),
                    reverse=predicate == "farthest",
                )
                candidates = [ranked[0][2]]
            else:
                # Try all anchor combinations: small-object 3D is noisy so
                # the highest-confidence anchor is not always the spatially
                # correct one.
                filtered: list[dict[str, Any]] = []
                for item in candidates:
                    for combo in itertools.product(*anchor_sets):
                        if _predicate_supported(predicate, item, list(combo)):
                            filtered.append(item)
                            break
                seen_ids = set()
                deduped = []
                for obj in filtered:
                    oid = int(obj.get("object_id", -1))
                    if oid not in seen_ids:
                        seen_ids.add(oid)
                        deduped.append(obj)
                candidates = deduped
        self.stack.remove(entity_id)
        result = (candidates, list(dict.fromkeys(failures)))
        self.cache[entity_id] = result
        return result


def execute_task(
    task_ir: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    allow_numerical_resolution: bool = True,
) -> dict[str, Any]:
    resolver = _Resolver(task_ir, snapshot)
    target_id = str(task_ir["target_entity"])
    candidates, failures = resolver.resolve(target_id)
    task_type = str(task_ir["task_type"])
    closure = bool(snapshot.get("candidate_set_closed"))
    evidence_ids = sorted({
        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
        for candidate in candidates
        for evidence in candidate.get("evidence", [])
    })
    result: dict[str, Any] = {
        "schema_version": "task_execution_v1",
        "task_type": task_type,
        "scene_version": int(snapshot["scene_version"]),
        "resolved": False,
        "answer": None,
        "selected_object": None,
        "route_objects": [],
        "probe_object": None,
        "evidence_ids": evidence_ids,
        "failed_constraints": list(dict.fromkeys(failures)),
        "geometry_reconstruction_ids": list(snapshot.get("geometry_reconstruction_ids", ())),
        "uncertainty": {
            "candidate_set_closed": closure,
            "closure_reasons": list(snapshot.get("closure_reasons", ())),
        },
    }
    target_entity = resolver.entities[target_id]
    if not target_entity.get("attributes"):
        target_names = {
            str(target_entity["class_name"]).lower(),
            *(str(value).lower() for value in target_entity.get("aliases", [])),
        }
        probe_candidates = [
            item for item in snapshot["objects"]
            if str(item.get("class_label", "")).lower() in target_names
            and item.get("status") in {"tentative", "confirmed"}
        ]
        if probe_candidates:
            result["probe_object"] = sorted(
                probe_candidates,
                key=lambda item: (-float(item.get("semantic_probability", 0.0)), int(item["object_id"])),
            )[0]
            if not result["evidence_ids"]:
                result["evidence_ids"] = sorted({
                    f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                    for evidence in result["probe_object"].get("evidence", [])
                })
    # When a required anchor is missing but another anchor exists in the
    # world model, probe towards the available anchor instead of rejecting.
    if (
        not candidates
        and not result.get("probe_object")
        and any(
            "relation_anchor_missing" in str(f) or "entity_missing" in str(f)
            for f in failures
        )
    ):
        anchor_ids_in_snapshot: set[str] = set()
        anchor_classes_in_snapshot = {
            str(obj["class_label"]).lower() for obj in snapshot["objects"]
            if obj.get("status") in {"tentative", "confirmed"}
        }
        for entity in resolver.entities.values():
            if entity["role"] != "anchor":
                continue
            if str(entity["class_name"]).lower() in anchor_classes_in_snapshot:
                anchor_ids_in_snapshot.add(str(entity["id"]))
        if anchor_ids_in_snapshot:
            best_anchor = None
            best_score = -1.0
            for aid in anchor_ids_in_snapshot:
                anchor_candidates, _anchor_failures = resolver.resolve(aid)
                for ac in anchor_candidates:
                    score = float(ac.get("semantic_probability", 0.0))
                    if score > best_score:
                        best_score = score
                        best_anchor = ac
            if best_anchor is not None:
                result["probe_object"] = best_anchor
                result["evidence_ids"] = sorted({
                    f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                    for evidence in best_anchor.get("evidence", [])
                })
                result["failed_constraints"] = [
                    f for f in failures
                    if "relation_anchor_missing" in str(f) or "entity_missing" in str(f)
                ] or ["active_close_view_required"]
    if task_type == "numerical":
        target_relations = [
            item for item in resolver.relations
            if str(item.get("subject_entity")) == target_id
        ]
        if not allow_numerical_resolution and len(target_relations) == 1:
            anchor_ids = list(target_relations[0].get("object_entities", ()))
            if len(anchor_ids) == 1:
                anchors, anchor_failures = resolver.resolve(str(anchor_ids[0]))
                if anchors and not anchor_failures:
                    result["resolved"] = False
                    result["answer"] = None
                    result["probe_object"] = max(
                        anchors,
                        key=lambda item: (
                            float(item.get("semantic_probability", 0.0)),
                            len(item.get("evidence", ())),
                            -int(item["object_id"]),
                        ),
                    )
                    result["evidence_ids"] = sorted({
                        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                        for evidence in result["probe_object"].get("evidence", ())
                    })
                    result["failed_constraints"] = ["active_close_view_required"]
                    return result
        if not candidates and not failures:
            if len(target_relations) == 1:
                anchor_ids = list(target_relations[0].get("object_entities", ()))
                if len(anchor_ids) == 1:
                    anchors, anchor_failures = resolver.resolve(str(anchor_ids[0]))
                    if anchors and not anchor_failures:
                        result["probe_object"] = max(
                            anchors,
                            key=lambda item: (
                                float(item.get("semantic_probability", 0.0)),
                                len(item.get("evidence", ())),
                                -int(item["object_id"]),
                            ),
                        )
                        result["evidence_ids"] = sorted({
                            f"{evidence['acquisition_id']}:{evidence['observation_id']}"
                            for evidence in result["probe_object"].get("evidence", ())
                        })
                        result["failed_constraints"] = ["active_close_view_required"]
                        return result
        if not failures:
            result.update(resolved=True, answer=len(candidates))
        return result
    if task_type == "object_reference":
        # If a probe_object is already queued (anchor available but relation
        # chain broken), defer the answer and let the PROBE path run first.
        if candidates:
            selected = max(
                candidates,
                key=lambda item: (
                    float(item.get("semantic_probability", 0.0)),
                    len(item.get("evidence", ())),
                    -int(item["object_id"]),
                ),
            )
            result.update(
                resolved=True,
                answer=int(selected["object_id"]),
                selected_object=selected,
            )
        elif not result.get("probe_object"):
            # No candidates and no anchor to probe — fall back to any
            # unverified target instance rather than rejecting outright.
            usable = [
                item for item in resolver.objects
                if str(item.get("class_label", "")).lower() in {
                    str(resolver.entities[target_id]["class_name"]).lower(),
                    *(str(a).lower() for a in resolver.entities[target_id].get("aliases", [])),
                }
                and item.get("status") in {"tentative", "confirmed"}
            ]
            if usable:
                selected = max(
                    usable,
                    key=lambda item: (
                        float(item.get("semantic_probability", 0.0)),
                        len(item.get("evidence", ())),
                        -int(item["object_id"]),
                    ),
                )
                result.update(
                    resolved=True,
                    answer=int(selected["object_id"]),
                    selected_object=selected,
                )
            else:
                result["failed_constraints"].append("object_reference_missing")
        # else: probe_object is set — let root_finalizer issue PROBE.
        return result
    if task_type != "instruction_following":
        result["failed_constraints"].append("unsupported_task_type")
        return result

    route_objects = []
    # Resolve each step's candidates independently, then search for the
    # combination that forms the longest coherent path.  This avoids
    # picking a cluster of nearby objects when the question describes a
    # real navigation path across the room.
    step_candidates: list[list[dict[str, Any]]] = []
    for step in sorted(task_ir["ordered_subgoals"], key=lambda value: int(value["order"])):
        objects, step_failures = resolver.resolve(str(step["target_entity"]))
        result["failed_constraints"].extend(step_failures)
        if not objects:
            result["failed_constraints"].append(f"route_step_missing:{step['order']}")
            step_candidates.append([])
        else:
            step_candidates.append(objects)

    if all(step_candidates) and len(step_candidates) >= 2:
        # Score every combination by total path length — prefer objects
        # that genuinely form a forward-moving path.
        import itertools as _itertools
        best_combo = None
        best_score = -1.0
        for combo in _itertools.product(*step_candidates):
            positions = [
                (float(obj["center_3d"][0]), float(obj["center_3d"][1]))
                for obj in combo
            ]
            total_dist = sum(
                math.hypot(positions[i+1][0] - positions[i][0],
                           positions[i+1][1] - positions[i][1])
                for i in range(len(positions) - 1)
            )
            avg_prob = sum(
                float(obj.get("semantic_probability", 0.0))
                for obj in combo
            ) / len(combo)
            score = total_dist + 0.3 * avg_prob
            if score > best_score:
                best_score = score
                best_combo = combo
        if best_combo is not None:
            for i, obj in enumerate(best_combo):
                route_objects.append({
                    "order": int(task_ir["ordered_subgoals"][i]["order"]),
                    "action": str(task_ir["ordered_subgoals"][i]["action"]),
                    "terminal": bool(task_ir["ordered_subgoals"][i]["terminal"]),
                    "object": obj,
                })
    else:
        for i, step in enumerate(sorted(task_ir["ordered_subgoals"], key=lambda value: int(value["order"]))):
            if i < len(step_candidates) and step_candidates[i]:
                selected = max(
                    step_candidates[i],
                    key=lambda item: (
                        float(item.get("semantic_probability", 0.0)),
                        len(item.get("evidence", ())),
                        -int(item["object_id"]),
                    ),
                )
                route_objects.append({
                    "order": int(step["order"]),
            "action": str(step["action"]),
            "terminal": bool(step["terminal"]),
            "object": selected,
        })
    result["route_objects"] = route_objects
    result["evidence_ids"] = sorted({
        f"{evidence['acquisition_id']}:{evidence['observation_id']}"
        for step in route_objects
        for evidence in step["object"].get("evidence", [])
    })
    result["failed_constraints"] = list(dict.fromkeys(result["failed_constraints"]))
    if not result["failed_constraints"] and len(route_objects) == len(task_ir["ordered_subgoals"]):
        result["resolved"] = True
    return result
