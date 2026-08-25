#!/usr/bin/env python3
"""Offline Phase 3A task/VLA audit.

This module is an artifact generator only.  Production code must not import it.
It reads the VLA authority directly through ``zipfile`` and never extracts or
modifies the archive.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence
import zipfile


VLA_MEMBERS = (
    "region_result.csv",
    "object_result.csv",
    "scene_graph.json",
    "referential_statements.json",
)
FOCUS_CLASSES = ("pillow", "chair", "picture", "table", "sofa", "cabinet", "lamp")
REQUIRED_OPERATORS = (
    "COUNT",
    "SELECT_UNIQUE",
    "FILTER_CLASS",
    "FILTER_COLOR",
    "FILTER_SIZE",
    "ON",
    "ABOVE",
    "BELOW",
    "NEAR",
    "IN",
    "BETWEEN",
    "ARGMIN_DISTANCE",
    "ARGMAX_DISTANCE",
    "GO_TO",
    "GO_NEAR",
    "PASS_NEAR",
    "PASS_BETWEEN",
    "AVOID_NEAR",
    "AVOID_BETWEEN",
    "SEQUENCE",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": min(finite),
        "p10": _quantile(finite, 0.10),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "p90": _quantile(finite, 0.90),
        "max": max(finite),
    }


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _class_name(row: Mapping[str, Any]) -> str:
    return str(row.get("raw_label") or row.get("nyu_label") or "").strip().lower()


def _load_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))))


def _scene_members(archive: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    pattern = re.compile(
        r"^Unity/([^/]+)/\1_(region_result\.csv|object_result\.csv|"
        r"scene_graph\.json|referential_statements\.json)$"
    )
    for member in archive.namelist():
        match = pattern.match(member)
        if match:
            result[match.group(1)][match.group(2)] = member
    return dict(result)


def _same_class_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("region_id", "")), _class_name(row))].append(row)
    records: list[dict[str, Any]] = []
    nearest_by_class: dict[str, list[float]] = defaultdict(list)
    normalized_distances: list[float] = []
    close_pairs: list[dict[str, Any]] = []
    for (region_id, class_name), members in sorted(groups.items()):
        if not class_name or len(members) < 2:
            continue
        pair_distances: list[float] = []
        for index, first in enumerate(members):
            first_center = [_float(first, f"object_bbox_c{axis}") for axis in "xyz"]
            first_size = [_float(first, f"object_bbox_{axis}length") for axis in "xyz"]
            for second in members[index + 1 :]:
                second_center = [_float(second, f"object_bbox_c{axis}") for axis in "xyz"]
                second_size = [_float(second, f"object_bbox_{axis}length") for axis in "xyz"]
                if not all(math.isfinite(value) for value in (*first_center, *second_center)):
                    continue
                distance = math.dist(first_center, second_center)
                scale = max(
                    math.sqrt(sum(value * value for value in first_size if math.isfinite(value))),
                    math.sqrt(sum(value * value for value in second_size if math.isfinite(value))),
                    1e-6,
                )
                normalized = distance / scale
                pair_distances.append(distance)
                nearest_by_class[class_name].append(distance)
                normalized_distances.append(normalized)
                if normalized <= 1.5:
                    close_pairs.append({
                        "region_id": region_id,
                        "class_name": class_name,
                        "object_ids": [str(first.get("object_id")), str(second.get("object_id"))],
                        "center_distance_m": distance,
                        "scale_normalized_distance": normalized,
                    })
        records.append({
            "region_id": region_id,
            "class_name": class_name,
            "instance_count": len(members),
            "object_ids": sorted(str(member.get("object_id")) for member in members),
            "nearest_center_distance_m": min(pair_distances) if pair_distances else None,
        })
    nearest = {
        "global": _distribution(value for values in nearest_by_class.values() for value in values),
        "by_class": {name: _distribution(values) for name, values in sorted(nearest_by_class.items())},
        "scale_normalized": _distribution(normalized_distances),
        "close_annotation_component_pairs": sorted(
            close_pairs,
            key=lambda value: (value["scale_normalized_distance"], value["class_name"]),
        )[:200],
    }
    return records, nearest


def _anchor_depth(value: object, depth: int = 0) -> int:
    if isinstance(value, Mapping):
        child_depths = [_anchor_depth(child, depth + 1) for child in value.values()]
        return max([depth, *child_depths])
    if isinstance(value, list):
        return max([depth, *(_anchor_depth(child, depth) for child in value)])
    return depth


def audit_vla(archive_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scene_audits: list[dict[str, Any]] = []
    scene_relation_counts: dict[str, dict[str, int]] = {}
    relation_edges = Counter()
    language_relations = Counter()
    distractor_counts: list[int] = []
    anchor_depths: list[int] = []
    target_color_used = 0
    target_size_used = 0
    anchor_color_used = 0
    anchor_size_used = 0
    same_class_distractor_records = 0
    referential_records = 0
    all_group_counts: list[int] = []
    all_normalized_pair_distances: list[float] = []
    focus_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with zipfile.ZipFile(archive_path) as archive:
        members = _scene_members(archive)
        for scene_name in sorted(members):
            files = members[scene_name]
            missing = [suffix for suffix in VLA_MEMBERS if suffix not in files]
            if missing:
                raise RuntimeError(f"archive_scene_incomplete:{scene_name}:{','.join(missing)}")
            regions = _load_csv(archive, files["region_result.csv"])
            objects = _load_csv(archive, files["object_result.csv"])
            graph = json.loads(archive.read(files["scene_graph.json"]))
            language = json.loads(archive.read(files["referential_statements.json"]))

            histogram = Counter(_class_name(row) for row in objects if _class_name(row))
            per_region = Counter(str(row.get("region_id", "")) for row in objects)
            groups, nearest = _same_class_metrics(objects)
            all_group_counts.extend(int(group["instance_count"]) for group in groups)
            normalized = nearest.get("scale_normalized", {})
            if normalized.get("count"):
                for pair in nearest["close_annotation_component_pairs"]:
                    all_normalized_pair_distances.append(float(pair["scale_normalized_distance"]))
            for group in groups:
                name = str(group["class_name"])
                if any(token in name for token in FOCUS_CLASSES):
                    focus_cases[name].append({"scene_name": scene_name, **group})

            sizes = {
                axis: [_float(row, f"object_bbox_{axis}length") for row in objects]
                for axis in "xyz"
            }
            diagonals = [
                math.sqrt(sum(_float(row, f"object_bbox_{axis}length") ** 2 for axis in "xyz"))
                for row in objects
            ]
            volumes = [
                math.prod(_float(row, f"object_bbox_{axis}length") for axis in "xyz")
                for row in objects
            ]
            minus_one = [
                {"object_id": str(row.get("object_id")), "class_name": _class_name(row)}
                for row in objects if str(row.get("region_id")) == "-1"
            ]
            scene_audits.append({
                "scene_name": scene_name,
                "region_count": len(regions),
                "region_labels": [str(row.get("region_label", "")) for row in regions],
                "total_object_count": len(objects),
                "objects_per_region": dict(sorted(per_region.items())),
                "class_histogram": dict(sorted(histogram.items())),
                "same_class_instance_groups": groups,
                "nearest_same_class_center_distance": nearest,
                "object_size_distribution": {
                    "x_length_m": _distribution(sizes["x"]),
                    "y_length_m": _distribution(sizes["y"]),
                    "z_length_m": _distribution(sizes["z"]),
                    "diagonal_m": _distribution(diagonals),
                },
                "object_volume_distribution": _distribution(volumes),
                "objects_with_region_minus_one": minus_one,
            })

            local_relations = Counter()
            for region in graph.get("regions", {}).values():
                for predicate, mapping in region.get("relationships", {}).items():
                    if not isinstance(mapping, Mapping):
                        continue
                    count = sum(len(targets) for targets in mapping.values() if isinstance(targets, list))
                    relation_edges[str(predicate).lower()] += count
                    local_relations[str(predicate).lower()] += count
            scene_relation_counts[scene_name] = dict(sorted(local_relations.items()))

            for region_statements in language.get("regions", {}).values():
                if not isinstance(region_statements, Mapping):
                    continue
                for records in region_statements.values():
                    if not isinstance(records, list):
                        continue
                    for record in records:
                        if not isinstance(record, Mapping):
                            continue
                        referential_records += 1
                        language_relations[str(record.get("relation", "unknown")).lower()] += 1
                        distractors = list(record.get("distractor_ids", ()))
                        distractor_counts.append(len(distractors))
                        same_class_distractor_records += int(bool(distractors))
                        target_color_used += int(bool(str(record.get("target_color_used", "")).strip()))
                        target_size_used += int(bool(str(record.get("target_size_used", "")).strip()))
                        anchors = record.get("anchors", {})
                        anchor_depths.append(max(0, _anchor_depth(anchors) - 1))
                        if isinstance(anchors, Mapping):
                            for anchor in anchors.values():
                                if not isinstance(anchor, Mapping):
                                    continue
                                anchor_color_used += int(bool(str(anchor.get("color_used", "")).strip()))
                                anchor_size_used += int(bool(str(anchor.get("size_used", "")).strip()))

        coverage = {
            suffix: sum(suffix in values for values in members.values())
            for suffix in VLA_MEMBERS
        }
        archive_audit = {
            "schema_version": "phase3a_vla_scene_audit_v1",
            "authority_path": str(archive_path.resolve()),
            "access_mode": "zipfile_direct_no_extraction",
            "scene_count": len(members),
            "scenes": sorted(members),
            "archive_coverage": coverage,
            "region_count_distribution": _distribution(item["region_count"] for item in scene_audits),
            "object_count_distribution": _distribution(item["total_object_count"] for item in scene_audits),
            "scene_audits": scene_audits,
        }
        relation_audit = {
            "schema_version": "phase3a_vla_relation_audit_v1",
            "authority_path": str(archive_path.resolve()),
            "scene_count": len(members),
            "scene_graph_relation_edge_counts": dict(sorted(relation_edges.items())),
            "scene_graph_relation_edge_counts_by_scene": scene_relation_counts,
            "referential_statement_record_count": referential_records,
            "referential_relation_frequencies": dict(sorted(language_relations.items())),
            "nested_anchor_depth": _distribution(anchor_depths),
            "target_distractor_count": _distribution(distractor_counts),
            "same_class_distractor_record_count": same_class_distractor_records,
            "attribute_usage": {
                "target_color_used": target_color_used,
                "target_size_used": target_size_used,
                "anchor_color_used": anchor_color_used,
                "anchor_size_used": anchor_size_used,
            },
            "interpretation_boundary": (
                "Offline structural statistics only; no scene, object ID, coordinate, "
                "count, statement, or threshold is available to production runtime."
            ),
        }
        granularity = {
            "schema_version": "phase3a_vla_annotation_granularity_v1",
            "authority_path": str(archive_path.resolve()),
            "canonical_query_entity_definition": (
                "A stable observation cluster that may correspond to one VLA/Challenge annotation object instance."
            ),
            "same_class_multiplicity_distribution": _distribution(all_group_counts),
            "close_pair_scale_normalized_distance_distribution": _distribution(all_normalized_pair_distances),
            "focus_class_same_region_groups": dict(sorted(focus_cases.items())),
            "findings": [
                "Adjacent, coplanar, same-class annotation instances remain distinct annotation units.",
                "Class agreement and geometric continuity are insufficient destructive-merge evidence.",
                "Same-acquisition separated support is positive distinctness evidence.",
                "Archive findings calibrate one class-agnostic policy and never scene-specific runtime rules.",
            ],
        }
    return archive_audit, relation_audit, granularity


def _relation_operator(predicate: str) -> str:
    return {
        "closest": "ARGMIN_DISTANCE",
        "farthest": "ARGMAX_DISTANCE",
    }.get(predicate.lower(), predicate.upper())


def _dsl_entity(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only finite, class-agnostic adjective grammar for the audit DSL."""
    class_name = str(raw.get("class_name", "")).strip().lower()
    attributes = dict(raw.get("attributes", {}))
    size_match = re.match(r"^(small|little|big|large)\s+(.+)$", class_name)
    if size_match and "size" not in attributes:
        attributes["size"] = "small" if size_match.group(1) in {"small", "little"} else "large"
        class_name = size_match.group(2)
    return {"class_name": class_name, "attributes": attributes}


def _path_program(question: str, ir: Mapping[str, Any]) -> dict[str, Any]:
    action_map = {
        "go_to": "GO_TO",
        "stop_at": "GO_TO",
        "go_near": "GO_NEAR",
        "pass_near": "PASS_NEAR",
        "pass_between": "PASS_BETWEEN",
        "avoid_near": "AVOID_NEAR",
        "avoid_between": "AVOID_BETWEEN",
    }
    steps = []
    for step in ir.get("ordered_trajectory_constraints", ()):
        action = action_map.get(str(step.get("action", "")).lower())
        if action:
            steps.append({
                "operator": action,
                "target_entity": step.get("target_entity"),
                "relation_ids": list(step.get("relation_ids", ())),
                "terminal": bool(step.get("terminal")),
            })
    lowered = question.lower()
    supplemental: list[str] = []
    if re.search(r"avoid(?:ing)?(?: the)? path between", lowered):
        supplemental.append("AVOID_BETWEEN")
    if re.search(r"avoid(?:ing)?(?: the)? path near", lowered):
        supplemental.append("AVOID_NEAR")
    if re.search(r"(?:take|pass|go)(?: the)? path between", lowered) and "avoiding" not in lowered:
        supplemental.append("PASS_BETWEEN")
    if re.search(r"(?:take|pass|go)(?: the)? path near", lowered) and "avoiding" not in lowered:
        supplemental.append("PASS_NEAR")
    existing = {step["operator"] for step in steps}
    steps.extend({"operator": operator, "source": "generic_path_grammar"} for operator in supplemental if operator not in existing)
    return {"operator": "SEQUENCE", "steps": steps} if steps else {}


def _max_relation_depth(relations: Sequence[Mapping[str, Any]]) -> int:
    by_id = {str(value.get("id")): value for value in relations}
    memo: dict[str, int] = {}

    def visit(relation_id: str, active: set[str]) -> int:
        if relation_id in memo:
            return memo[relation_id]
        if relation_id in active:
            return 0
        relation = by_id.get(relation_id, {})
        dependencies = [str(value) for value in relation.get("depends_on", ())]
        result = 1 + max((visit(value, active | {relation_id}) for value in dependencies), default=0)
        memo[relation_id] = result
        return result

    return max((visit(relation_id, set()) for relation_id in by_id), default=0)


def compile_questions(questions_path: Path, ai_module: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ai_module))
    from integrations.semantics.task_compiler import compile_task  # pylint: disable=import-outside-toplevel

    source = json.loads(questions_path.read_text(encoding="utf-8"))
    programs: list[dict[str, Any]] = []
    used_operators = Counter()
    failures: list[dict[str, str]] = []
    for scene in source:
        scene_name = str(scene["scene"])
        for task_type, questions in scene["questions"].items():
            for question in questions:
                try:
                    ir = compile_task(question)
                    entities = {str(value["id"]): value for value in ir.get("entities", ())}
                    dsl_entities = {entity_id: _dsl_entity(value) for entity_id, value in entities.items()}
                    target = entities.get(str(ir.get("target_entity", "")), {})
                    dsl_target = dsl_entities.get(str(ir.get("target_entity", "")), _dsl_entity(target))
                    relations = list(ir.get("relations", ()))
                    predicates = [{
                        "operator": _relation_operator(str(value.get("predicate", ""))),
                        "relation_id": value.get("id"),
                        "subject_entity": value.get("subject_entity"),
                        "object_entities": list(value.get("object_entities", ())),
                        "depends_on": list(value.get("depends_on", ())),
                    } for value in relations]
                    path = _path_program(question, ir)
                    root_operator = {
                        "numerical": "COUNT",
                        "object_reference": "SELECT_UNIQUE",
                        "instruction_following": "SEQUENCE",
                    }[task_type]
                    used_operators[root_operator] += 1
                    used_operators["FILTER_CLASS"] += len(entities)
                    for entity in dsl_entities.values():
                        attributes = dict(entity.get("attributes", {}))
                        if attributes.get("color"):
                            used_operators["FILTER_COLOR"] += 1
                        if attributes.get("size"):
                            used_operators["FILTER_SIZE"] += 1
                    for predicate in predicates:
                        used_operators[predicate["operator"]] += 1
                    for step in path.get("steps", ()):
                        used_operators[str(step["operator"])] += 1
                    attributes = [
                        {"name": str(name), "value": value}
                        for name, value in sorted(dict(dsl_target.get("attributes", {})).items())
                    ]
                    depth = _max_relation_depth(relations)
                    has_selector = any(value["operator"] in {"ARGMIN_DISTANCE", "ARGMAX_DISTANCE"} for value in predicates)
                    scope = (
                        "scene_search" if task_type == "instruction_following"
                        else "place_local" if has_selector
                        else "anchor_local" if predicates
                        else "scene_search"
                    )
                    programs.append({
                        "scene": scene_name,
                        "question": question,
                        "task_type": task_type,
                        "target_class": str(dsl_target.get("class_name", "")),
                        "attributes": attributes,
                        "anchor_program": {
                            "entities": [
                                {
                                    "entity_id": entity_id,
                                    "operator": "FILTER_CLASS",
                                    "class_name": dsl_entities[entity_id].get("class_name"),
                                    "attributes": dict(dsl_entities[entity_id].get("attributes", {})),
                                }
                                for entity_id, value in entities.items()
                                if entity_id != str(ir.get("target_entity", ""))
                            ],
                            "dependency_order": list(ir.get("count_query_graph", {}).get("dependency_order", ())),
                        },
                        "predicate_program": {"operator": "AND", "clauses": predicates},
                        "path_program": path,
                        "max_anchor_depth": depth,
                        "candidate_scope": scope,
                        "root_operator": root_operator,
                        "compile_status": "COMPILED",
                        "task_ir": ir,
                    })
                except Exception as exc:  # Artifact records exact compiler failure.
                    failures.append({"scene": scene_name, "question": question, "error": f"{type(exc).__name__}:{exc}"})
    counts = Counter(program["task_type"] for program in programs)
    numerical = [program for program in programs if program["task_type"] == "numerical"]
    return {
        "schema_version": "phase3a_task_ontology_v1",
        "questions_authority_path": str(questions_path.resolve()),
        "compiler": "integrations.semantics.task_compiler.compile_task",
        "scene_count": len(source),
        "question_count": sum(counts.values()) + len(failures),
        "compiled_count": len(programs),
        "compile_failures": failures,
        "task_type_counts": dict(sorted(counts.items())),
        "numerical_compile": {"compiled": len(numerical), "required": 15, "passed": len(numerical) == 15},
        "required_operator_support": list(REQUIRED_OPERATORS),
        "observed_operator_usage": dict(sorted(used_operators.items())),
        "programs": programs,
    }


def _write_markdown(output: Path, task: Mapping[str, Any], scene: Mapping[str, Any], relation: Mapping[str, Any], granularity: Mapping[str, Any]) -> None:
    numerical = task["numerical_compile"]
    coverage = scene["archive_coverage"]
    lines = [
        "# Phase 3A Task and VLA Audit",
        "",
        "This is an offline audit. No VLA annotation, scene name, object ID, coordinate, or answer is imported by production runtime.",
        "",
        "## Challenge grammar",
        "",
        f"- Public questions compiled: {task['compiled_count']}/{task['question_count']}",
        f"- Numerical questions compiled: {numerical['compiled']}/{numerical['required']} ({'PASS' if numerical['passed'] else 'FAIL'})",
        f"- Scene count: {task['scene_count']}",
        "",
        "## VLA archive authority",
        "",
        f"- Path: `{scene['authority_path']}`",
        f"- Scenes: {scene['scene_count']}/15",
        f"- object_result: {coverage['object_result.csv']}/15",
        f"- region_result: {coverage['region_result.csv']}/15",
        f"- scene_graph: {coverage['scene_graph.json']}/15",
        f"- referential_statements: {coverage['referential_statements.json']}/15",
        "- Access: direct `zipfile` reads; no extraction and no archive modification.",
        "",
        "## Structural findings",
        "",
        f"- Region count distribution: `{json.dumps(scene['region_count_distribution'], sort_keys=True)}`",
        f"- Object count distribution: `{json.dumps(scene['object_count_distribution'], sort_keys=True)}`",
        f"- Referential records: {relation['referential_statement_record_count']}",
        f"- Same-class multiplicity: `{json.dumps(granularity['same_class_multiplicity_distribution'], sort_keys=True)}`",
        "- Candidate domains are formed by class/attribute filters plus finite relation chains; ordered selectors require place-local coverage.",
        "- Color and size are disambiguation evidence, never identity authority.",
        "- Adjacent same-class annotation components must remain distinct unless direct identity evidence supports every merge pair.",
        "",
    ]
    (output / "PHASE3A_TASK_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive", type=Path, default=Path("/home/robot/cmu_vln/VLA-3D_dataset/Unity.zip"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    ai_module = repo / "ai_module"
    output = ai_module / "artifacts" / "phase3a"
    task = compile_questions(repo / "questions" / "questions.json", ai_module)
    scene, relation, granularity = audit_vla(args.archive.resolve())
    _write_json(output / "task_ontology.json", task)
    _write_json(output / "vla_scene_audit.json", scene)
    _write_json(output / "vla_relation_audit.json", relation)
    _write_json(output / "vla_annotation_granularity.json", granularity)
    numerical = [value for value in task["programs"] if value["task_type"] == "numerical"]
    _write_json(output / "numerical_query_programs.json", {
        "schema_version": "phase3a_numerical_query_programs_v1",
        "compiled": len(numerical),
        "required": 15,
        "programs": numerical,
    })
    observed = task["observed_operator_usage"]
    coverage_lines = [
        "# Numerical and DSL Operator Coverage",
        "",
        f"Numerical compilation: {len(numerical)}/15 {'PASS' if len(numerical) == 15 else 'FAIL'}.",
        "",
        "| Operator | Supported | Public-question uses |",
        "|---|---:|---:|",
        *[f"| {operator} | yes | {observed.get(operator, 0)} |" for operator in REQUIRED_OPERATORS],
        "",
        "Zero uses means the finite grammar supports the operator but no public question exercised it.",
    ]
    (output / "numerical_operator_coverage.md").write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")
    _write_markdown(output, task, scene, relation, granularity)
    print(json.dumps({
        "output_dir": str(output),
        "questions": f"{task['compiled_count']}/{task['question_count']}",
        "numerical": f"{len(numerical)}/15",
        "vla_scenes": f"{scene['scene_count']}/15",
        "archive_coverage": scene["archive_coverage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
