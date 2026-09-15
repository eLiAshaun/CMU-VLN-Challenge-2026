"""Compile numerical TaskIR into an explicit COUNT_DISTINCT query graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .relation_registry import (
    RELATION_REGISTRY_SCHEMA_VERSION,
    relation_spec,
    validate_relation_arguments,
)


SCHEMA_VERSION = "count_query_graph_v1"


def _descriptor(entity: Mapping[str, Any]) -> str:
    attributes = dict(entity.get("attributes", {}))
    modifiers = " ".join(
        str(value).strip() for value in attributes.values() if str(value).strip()
    )
    class_name = str(entity.get("class_name", "")).strip()
    return " ".join(value for value in (modifiers, class_name) if value)


def _semantic_description(
    entity: Mapping[str, Any],
    grounding: Mapping[str, Any],
) -> str:
    """Describe identity independently from the relation being evaluated."""
    descriptor = _descriptor(entity)
    visual_definition = str(
        grounding.get("visual_definition", descriptor)
    ).strip()
    visual_alternatives = [
        str(value).strip()
        for value in grounding.get("hard_negatives", ())
        if str(value).strip()
    ]
    parts = [descriptor]
    if visual_definition and visual_definition.lower() != descriptor.lower():
        parts.append(f"visible class definition: {visual_definition}")
    if visual_alternatives:
        parts.append(
            "visual alternatives to distinguish from using the supplied pixels: "
            + ", ".join(visual_alternatives)
        )
    return "; ".join(parts)


def _relation_arguments(relation: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(relation.get("subject_entity", "")),
        *(str(value) for value in relation.get("object_entities", ())),
    )


def _expanded_dependencies(
    relations: Sequence[Mapping[str, Any]],
    target_entity: str,
) -> dict[str, list[str]]:
    """Preserve explicit dependencies and recover omitted anchor qualifiers."""
    dependencies = {
        str(relation["id"]): list(
            dict.fromkeys(str(value) for value in relation.get("depends_on", ()))
        )
        for relation in relations
    }
    arguments = {
        str(relation["id"]): set(_relation_arguments(relation))
        for relation in relations
    }
    by_id = {str(relation["id"]): relation for relation in relations}
    for relation_id, relation in by_id.items():
        relation_args = arguments[relation_id]
        if target_entity not in relation_args:
            continue
        context_args = relation_args - {target_entity}
        for candidate_id, candidate_args in arguments.items():
            if candidate_id == relation_id or target_entity in candidate_args:
                continue
            if context_args.intersection(candidate_args):
                dependencies[relation_id].append(candidate_id)
        dependencies[relation_id] = list(dict.fromkeys(dependencies[relation_id]))
    return dependencies


def _topological_order(
    relation_ids: Sequence[str],
    dependencies: Mapping[str, Sequence[str]],
) -> list[str]:
    ordered_ids = list(dict.fromkeys(str(value) for value in relation_ids))
    known = set(ordered_ids)
    for relation_id, values in dependencies.items():
        missing = [value for value in values if value not in known]
        if missing:
            raise ValueError(
                f"count_query_dependency_missing:{relation_id}:{','.join(missing)}"
            )
    followers: dict[str, list[str]] = defaultdict(list)
    indegree = {relation_id: 0 for relation_id in ordered_ids}
    for relation_id, values in dependencies.items():
        for dependency in values:
            followers[dependency].append(relation_id)
            indegree[relation_id] += 1
    queue = [value for value in ordered_ids if indegree[value] == 0]
    result: list[str] = []
    while queue:
        relation_id = queue.pop(0)
        result.append(relation_id)
        for follower in followers.get(relation_id, ()):
            indegree[follower] -= 1
            if indegree[follower] == 0:
                queue.append(follower)
    if len(result) != len(ordered_ids):
        cyclic = [value for value in ordered_ids if indegree[value] > 0]
        raise ValueError("count_query_dependency_cycle:" + ",".join(cyclic))
    return result


def compile_count_query_graph(task_ir: Mapping[str, Any]) -> dict[str, Any]:
    if str(task_ir.get("task_type", "")) != "numerical":
        raise ValueError("count_query_graph_requires_numerical_task")
    target_entity = str(task_ir.get("target_entity", ""))
    entities = {
        str(entity.get("id", "")): dict(entity)
        for entity in task_ir.get("entities", ())
    }
    if not target_entity or target_entity not in entities:
        raise ValueError("count_query_target_missing")
    grounding_by_entity = {
        str(value.get("entity_id", "")): dict(value)
        for value in task_ir.get("grounding_plan", {}).get("entities", ())
    }
    relations = [dict(value) for value in task_ir.get("relations", ())]
    relation_ids = [str(value.get("id", "")) for value in relations]
    if any(not value for value in relation_ids) or len(set(relation_ids)) != len(
        relation_ids
    ):
        raise ValueError("count_query_relation_ids_invalid")
    dependencies = _expanded_dependencies(relations, target_entity)
    dependency_order = _topological_order(relation_ids, dependencies)

    variables = []
    for entity_id, entity in entities.items():
        grounding = grounding_by_entity.get(entity_id, {})
        variables.append({
            "id": f"var:{entity_id}",
            "entity_id": entity_id,
            "quantifier": (
                "COUNT_DISTINCT"
                if entity_id == target_entity
                else "SINGULAR"
                if str(entity.get("role", "anchor")) == "anchor"
                else "EXISTS"
            ),
            "role": str(entity.get("role", "anchor")),
            "class_name": str(entity.get("class_name", "")),
            "aliases": [str(value) for value in entity.get("aliases", ())],
            "attributes": dict(entity.get("attributes", {})),
            "descriptor": _descriptor(entity),
            "visual_definition": str(
                grounding.get("visual_definition", _descriptor(entity))
            ),
            "visual_alternatives": [
                str(value) for value in grounding.get("hard_negatives", ())
            ],
        })

    relation_nodes = []
    for relation in relations:
        relation_id = str(relation["id"])
        subject_entity = str(relation["subject_entity"])
        object_entities = [str(value) for value in relation["object_entities"]]
        if subject_entity not in entities or any(
            value not in entities for value in object_entities
        ):
            raise ValueError(f"count_query_relation_entity_missing:{relation_id}")
        spec = validate_relation_arguments(
            str(relation["predicate"]), object_entities
        )
        arguments = [subject_entity, *object_entities]
        counted_roles = [
            spec.parameter_roles[index]
            for index, entity_id in enumerate(arguments)
            if entity_id == target_entity
        ]
        relation_nodes.append({
            "id": relation_id,
            "node_type": "RELATION",
            "predicate": spec.predicate,
            "operator": spec.operator,
            "subject_variable": f"var:{subject_entity}",
            "object_variables": [f"var:{value}" for value in object_entities],
            "subject_entity": subject_entity,
            "object_entities": object_entities,
            "parameter_roles": list(spec.parameter_roles),
            "counted_parameter_roles": counted_roles,
            "depends_on": list(dependencies[relation_id]),
            "directed": spec.directed,
            "symmetric": spec.symmetric,
            "evidence_roles": list(spec.evidence_roles),
            "evidence_policy": spec.evidence_policy,
            "geometry_policy": spec.geometry_policy,
            "negative_evidence_policy": spec.negative_evidence_policy,
            "qwen_instruction": spec.qwen_instruction,
            "argument_descriptions": [
                _semantic_description(
                    entities[entity_id], grounding_by_entity.get(entity_id, {})
                )
                for entity_id in arguments
            ],
        })

    root_relations = [
        node["id"]
        for node in relation_nodes
        if target_entity
        in {node["subject_entity"], *node["object_entities"]}
    ]
    target = entities[target_entity]
    graph = {
        "schema_version": SCHEMA_VERSION,
        "registry_schema_version": RELATION_REGISTRY_SCHEMA_VERSION,
        "task_id": str(task_ir.get("original_question", "")),
        "target_entity": target_entity,
        "variables": variables,
        "relation_nodes": relation_nodes,
        "dependency_order": dependency_order,
        "root": {
            "operator": "COUNT_DISTINCT",
            "variable": f"var:{target_entity}",
            "entity_constraint": {
                "class_name": str(target.get("class_name", "")),
                "aliases": [str(value) for value in target.get("aliases", ())],
                "attributes": dict(target.get("attributes", {})),
                "descriptor": _descriptor(target),
                "visual_definition": str(
                    grounding_by_entity.get(target_entity, {}).get(
                        "visual_definition", _descriptor(target)
                    )
                ),
                "visual_alternatives": [
                    str(value)
                    for value in grounding_by_entity.get(target_entity, {}).get(
                        "hard_negatives", ()
                    )
                ],
            },
            "required_relation_nodes": root_relations,
            "cardinality_semantics": "unique_persistent_target_object_ids",
        },
    }
    validate_count_query_graph(graph)
    return graph


def validate_count_query_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("count_query_graph_schema_invalid")
    variables = list(graph.get("variables", ()))
    variable_ids = {str(value.get("id", "")) for value in variables}
    if str(graph.get("root", {}).get("variable", "")) not in variable_ids:
        raise ValueError("count_query_root_variable_missing")
    nodes = list(graph.get("relation_nodes", ()))
    node_ids = {str(value.get("id", "")) for value in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("count_query_relation_nodes_not_unique")
    for node in nodes:
        spec = relation_spec(str(node.get("predicate", "")))
        if node.get("operator") != spec.operator:
            raise ValueError(f"count_query_operator_invalid:{node.get('id')}")
        if str(node.get("subject_variable", "")) not in variable_ids or any(
            str(value) not in variable_ids
            for value in node.get("object_variables", ())
        ):
            raise ValueError(f"count_query_relation_variable_missing:{node.get('id')}")
        if any(str(value) not in node_ids for value in node.get("depends_on", ())):
            raise ValueError(f"count_query_relation_dependency_missing:{node.get('id')}")
    if set(graph.get("dependency_order", ())) != node_ids:
        raise ValueError("count_query_dependency_order_incomplete")
    if any(
        str(value) not in node_ids
        for value in graph.get("root", {}).get("required_relation_nodes", ())
    ):
        raise ValueError("count_query_root_relation_missing")
    return dict(graph)
