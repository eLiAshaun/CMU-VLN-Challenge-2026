"""Small executable TaskIR language and spatial query tools.

The parser accepts only structured JSON produced by the compiler prompt. It
never tries to solve a natural-language question with a second parser. The
evaluator operates on CPU ObjectRecord values and keeps unresolved evidence
visible to its caller.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import ObjectRecord, QueryResult


TASK_TYPES = frozenset({"numerical", "object_reference", "instruction"})
STEP_ACTIONS = frozenset({"pass_near", "go_to", "near_path", "between_path", "avoid_path", "avoid_region"})
AVOID_ACTIONS = frozenset({"avoid_path", "avoid_region"})
RELATIONAL_ATTRIBUTES = frozenset({
    "near", "on", "inside", "between", "below", "above", "same_room",
    "argmin_distance", "argmax_distance", "proximity", "distance", "closest", "farthest", "furthest",
})
RELATION_RETURN_ROLES = frozenset({"subject", "anchor"})
AST_OPERATORS = frozenset(
    {
        "all",
        "filter_class",
        "filter_attribute",
        "entity_ref",
        "object_ref",
        "bound_ref",
        "literal",
        "bind",
        "let",
        "select",
        "near",
        "on",
        "inside",
        "between",
        "same_room",
        "above",
        "below",
        "argmin_distance",
        "argmax_distance",
        "count_distinct",
        "union",
        "intersection",
        "difference",
        "sequence",
        "not",
        "true",
        "false",
    }
)


TASK_IR_SCHEMA: dict[str, Any] = {
    "schema_version": "task_ir_v1",
    "task_type": "numerical|object_reference|instruction",
    "instruction": "original question or instruction",
    "concepts": ["semantic classes from the expanded expression and steps"],
    "entities": [
        {
            "id": "target_or_anchor_name",
            "role": "target|anchor",
            "class": "semantic object class",
            "visual_class": "short base visual class (required)",
            "attributes": {},
        }
    ],
    "expression": "AST required for numerical/object_reference; derived from the final motion step for instruction",
    "steps": [
        {
            "action": "pass_near|go_to|near_path|between_path|avoid_path|avoid_region",
            "target": "object expression for object or near-path actions",
            "anchors": "two object expressions for between_path",
            "region": "region expression for avoid_region",
        }
    ],
}


def _parse_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def parse_task_ir(text: str) -> dict[str, Any]:
    """Parse and validate structured TaskIR JSON.

    JSON code fences are tolerated because local model wrappers often retain
    them, but natural-language text is never parsed as a task.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("TaskIR must be a non-empty JSON string")
    payload = text.strip()
    fence = chr(96) * 3
    if payload.startswith(fence) and payload.endswith(fence):
        lines = payload.splitlines()
        if len(lines) < 3:
            raise ValueError("empty TaskIR code fence")
        payload = "\n".join(lines[1:-1]).strip()
    try:
        raw = json.loads(payload, parse_constant=_parse_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("TaskIR must be valid structured JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("TaskIR root must be a JSON object")
    return _normalize_task(dict(raw))


def _normalize_task(raw: dict[str, Any]) -> dict[str, Any]:
    declaration = raw.get("task_type")
    if declaration is not None and not isinstance(declaration, str):
        raise ValueError("TaskIR task_type declaration must be a string when present")
    declared_task_type = declaration.strip().lower() if declaration is not None else None
    if declared_task_type == "instruction_following":
        declared_task_type = "instruction"

    instruction = raw.get("instruction", raw.get("original_question", ""))
    if instruction is None:
        instruction = ""
    if not isinstance(instruction, str):
        raise ValueError("TaskIR instruction must be a string")

    entities = _normalize_entities(raw.get("entities", []))
    steps_raw = raw.get("steps", raw.get("ordered_steps", []))
    if steps_raw is None:
        steps_raw = []
    if not isinstance(steps_raw, list):
        raise ValueError("TaskIR steps must be an array")
    steps = [_normalize_step(step, index) for index, step in enumerate(steps_raw)]
    entity_map = {entity["id"]: entity for entity in entities}
    steps = _expand_step_targets(steps, entity_map)
    motion_steps = [step for step in steps if step["action"] not in AVOID_ACTIONS]
    if motion_steps:
        task_type = "instruction"
        # Ordered steps are the only source of instruction targets. Rebuild the
        # resolver expression from the final motion target so two ASTs cannot
        # disagree about the destination.
        terminal = motion_steps[-1]
        if terminal["action"] == "between_path":
            # A region traversal has no target object. Its anchors provide the
            # scene-query vocabulary; navigation owns the traversal outcome.
            expression = {"op": "union", "items": terminal["anchors"]}
        else:
            expression = _normalize_expression(terminal["target"])
    else:
        expression_raw = raw.get("expression", raw.get("query", None))
        if expression_raw is None:
            raise ValueError(
                "TaskIR requires movement steps or a structured query expression"
            )
        expression = _normalize_expression(expression_raw)
        expression = _expand_entity_references(expression, entity_map)
        # The executable return type owns the output interface. In particular,
        # distance ranking selects object IDs even if the compiler calls the
        # calculation "numerical". Preserve that declaration for diagnosis.
        task_type = _expression_task_type(expression)

    concepts = _unique_strings(
        _concepts_from_expression(expression) + _concepts_from_steps(steps)
    )

    normalized = dict(raw)
    normalized.update(
        {
            "schema_version": str(raw.get("schema_version", "task_ir_v1")),
            "task_type": task_type,
            "compiler_declared_task_type": declared_task_type,
            "instruction": instruction.strip(),
            "entities": entities,
            "expression": expression,
            "steps": steps,
            "concepts": concepts,
        }
    )
    if not normalized["schema_version"]:
        raise ValueError("TaskIR schema_version must be non-empty")
    normalized["output_contract"] = {
        "numerical": "/numerical_response",
        "object_reference": "/selected_object_marker",
        "instruction": "/way_point_with_heading",
    }[task_type]
    return normalized


def _expression_task_type(
    node: Mapping[str, Any], bindings: Mapping[str, str] | None = None,
) -> str:
    """Follow the same value propagation as the evaluator, including scope."""
    local = dict(bindings or {})
    op = _normal_op(node.get("op", ""))
    if op == "count_distinct":
        return "numerical"
    if op in {"object_ref", "bound_ref", "entity_ref"}:
        return local.get(str(node.get("id", "")), "object_reference")
    if op == "let":
        for name, expression in node.get("bindings", {}).items():
            local[str(name)] = _expression_task_type(expression, local)
        return _expression_task_type(node["body"], local)
    if op == "bind":
        source_type = _expression_task_type(node["source"], local)
        local[str(node["name"])] = source_type
        return (_expression_task_type(node["body"], local)
                if "body" in node else source_type)
    if op == "select" and not any(key in node for key in ("id", "object_id", "ids")):
        return _expression_task_type(node["source"], local)
    if op in {"union", "intersection", "difference", "sequence"}:
        return _expression_task_type(node["items"][-1], local)
    return "object_reference"


def _normalize_entities(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = [dict(spec, id=str(entity_id)) for entity_id, spec in value.items()]
    if not isinstance(value, list):
        raise ValueError("TaskIR entities must be an array or object")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each TaskIR entity must be an object")
        entity = dict(item)
        entity_id = entity.get("id", entity.get("entity_id", entity.get("name")))
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("each TaskIR entity requires a non-empty id")
        entity_id = entity_id.strip()
        if entity_id in seen:
            raise ValueError(f"duplicate TaskIR entity id: {entity_id}")
        seen.add(entity_id)
        role = str(entity.get("role", "anchor")).strip().lower()
        if role not in {"target", "anchor", "region", "robot"}:
            raise ValueError(f"unsupported TaskIR entity role: {role!r}")
        class_name = entity.get(
            "class", entity.get("class_name", entity.get("concept", ""))
        )
        if class_name is None:
            class_name = ""
        if not isinstance(class_name, str):
            raise ValueError(f"entity {entity_id} class must be a string")
        if "visual_class" in entity:
            visual_class = entity["visual_class"]
            if not isinstance(visual_class, str) or not visual_class.strip():
                raise ValueError(
                    f"entity {entity_id} visual_class must be a non-empty string"
                )
            entity["visual_class"] = visual_class.strip().lower()
        attrs = entity.get("attributes", {})
        if attrs is None:
            attrs = {}
        if not isinstance(attrs, Mapping):
            raise ValueError(f"entity {entity_id} attributes must be an object")
        invalid_attributes = sorted(
            str(key).strip().lower()
            for key in attrs
            if str(key).strip().lower() in RELATIONAL_ATTRIBUTES
        )
        if invalid_attributes:
            raise ValueError(
                f"entity {entity_id} relation attributes are not allowed: "
                + ", ".join(invalid_attributes)
            )
        entity.update(
            {
                "id": entity_id,
                "role": role,
                "class": class_name.strip().lower(),
                "attributes": dict(attrs),
            }
        )
        result.append(entity)
    return result


def _normalize_step(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"TaskIR step {index} must be an object")
    step = dict(value)
    action = str(step.get("action", step.get("type", ""))).strip().lower()
    action = {
        "pass_by": "pass_near",
        "go_near": "pass_near",
        "avoid_near": "avoid_region",
    }.get(action, action)
    if action not in STEP_ACTIONS:
        raise ValueError(f"unsupported TaskIR step action: {action!r}")
    step["action"] = action
    step.setdefault("order", index)
    if step.get("order") != index:
        step["order"] = index
    if action == "between_path":
        anchors = step.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != 2:
            raise ValueError(f"TaskIR between_path step {index} requires exactly two anchors")
    elif action == "avoid_region":
        if not any(
            key in step
            for key in (
                "region",
                "region_expression",
                "polygon",
                "bounds",
                "center",
                "target",
            )
        ):
            raise ValueError(f"TaskIR avoid_region step {index} requires a region")
    elif not any(
        key in step
        for key in (
            "target",
            "target_expression",
            "target_entity",
            "object_id",
            "object_ids",
        )
    ):
        raise ValueError(f"TaskIR {action} step {index} requires a target")
    return step


def _expand_entity_references(
    node: Mapping[str, Any],
    entities: Mapping[str, Mapping[str, Any]],
    resolving: frozenset[str] = frozenset(),
    bound_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Inline symbolic entity references while preserving lexical bindings."""

    if not isinstance(node, Mapping):
        raise ValueError("TaskIR AST child must be an object")
    normalized = dict(node)
    op = _normal_op(normalized.get("op", normalized.get("operator", "")))
    normalized["op"] = op
    if op == "entity_ref":
        identity = str(normalized.get("id", "")).strip()
        if not identity:
            raise ValueError("entity_ref requires a non-empty id")
        if identity in bound_names:
            return normalized
        entity = entities.get(identity)
        if entity is None:
            explicit = normalized.get("expression")
            if isinstance(explicit, Mapping):
                return _expand_entity_references(
                    _normalize_expression(explicit),
                    entities,
                    resolving,
                    bound_names,
                )
            raise ValueError(f"entity_ref references missing entity id: {identity}")
        if identity in resolving:
            cycle = " -> ".join([*resolving, identity])
            raise ValueError(f"cyclic TaskIR entity reference: {cycle}")
        return _expand_entity_references(
            _entity_expression(entity),
            entities,
            resolving | {identity},
            bound_names,
        )
    if op == "bound_ref":
        identity = str(normalized.get("id", "")).strip()
        if identity not in bound_names:
            raise ValueError(f"bound_ref references missing binding: {identity}")
        return normalized
    if op == "bind":
        normalized["source"] = _expand_entity_references(
            normalized["source"], entities, resolving, bound_names
        )
        name = str(normalized["name"])
        if isinstance(normalized.get("body"), Mapping):
            normalized["body"] = _expand_entity_references(
                normalized["body"],
                entities,
                resolving,
                bound_names | {name},
            )
        return normalized
    if op == "let":
        scope = set(bound_names)
        bindings = normalized.get("bindings", {})
        expanded_bindings: dict[str, Any] = {}
        if not isinstance(bindings, Mapping):
            raise ValueError("let bindings must be an object")
        for name, child in bindings.items():
            expanded_bindings[str(name)] = _expand_entity_references(
                child, entities, resolving, frozenset(scope)
            )
            scope.add(str(name))
        normalized["bindings"] = expanded_bindings
        normalized["body"] = _expand_entity_references(
            normalized["body"], entities, resolving, frozenset(scope)
        )
        return normalized

    child_keys = {
        "source",
        "input",
        "operand",
        "subject",
        "anchor",
        "candidates",
        "body",
        "in",
        "items",
        "operands",
        "sources",
        "boundaries",
        "anchors",
    }
    for key in child_keys:
        value = normalized.get(key)
        if isinstance(value, Mapping):
            normalized[key] = _expand_entity_references(
                value, entities, resolving, bound_names
            )
        elif isinstance(value, list):
            normalized[key] = [
                _expand_entity_references(child, entities, resolving, bound_names)
                if isinstance(child, Mapping)
                else child
                for child in value
            ]
    return normalized


def _entity_expression(entity: Mapping[str, Any]) -> dict[str, Any]:
    explicit = entity.get("expression")
    if isinstance(explicit, Mapping):
        return _normalize_expression(explicit)
    class_name = entity.get("class", entity.get("class_name", entity.get("concept", "")))
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(
            f"entity {entity.get('id', '<unknown>')} has no class or expression"
        )
    if "visual_class" not in entity:
        raise ValueError(
            f"entity {entity.get('id', '<unknown>')} requires explicit visual_class"
        )
    visual_class = entity["visual_class"]
    if not isinstance(visual_class, str) or not visual_class.strip():
        raise ValueError(
            f"entity {entity.get('id', '<unknown>')} visual_class must be non-empty"
        )
    expression: dict[str, Any] = {
        "op": "filter_class",
        "class": class_name.strip().lower(),
        "visual_class": visual_class.strip().lower(),
    }
    attributes = entity.get("attributes", {})
    if isinstance(attributes, Mapping):
        for attribute, value in attributes.items():
            expression = {
                "op": "filter_attribute",
                "attribute": str(attribute).strip().lower(),
                "value": value,
                "source": expression,
            }
    return expression


def _expand_step_targets(
    steps: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for step in steps:
        normalized = dict(step)
        action = str(normalized.get("action", "")).lower()
        if action == "between_path":
            normalized["anchors"] = [
                _expand_step_expression(anchor, entities) for anchor in normalized["anchors"]
            ]
            result.append(normalized)
            continue
        target: Any = normalized.get("target")
        if target is None:
            target = normalized.get("target_expression", normalized.get("target_entity"))
        if target is None and "object_id" in normalized:
            target = {"op": "object_ref", "id": normalized["object_id"]}
        if target is None and "object_ids" in normalized:
            target = {"op": "literal", "ids": normalized["object_ids"]}
        if target is None and action == "avoid_region":
            target = normalized.get("region_expression", normalized.get("region"))
            if target is None:
                # Explicit geometric regions do not require a synthetic object
                # expression. The navigation region consumer owns these fields.
                result.append(normalized)
                continue
        if target is None:
            raise ValueError(f"TaskIR step {normalized.get('order')} has no target")
        expression = (_expand_region_expression(target, entities) if action == "avoid_region"
                      else _expand_step_expression(target, entities))
        if action == "avoid_region":
            normalized["region"] = expression
            normalized.pop("target", None)
        else:
            normalized["target"] = expression
        result.append(normalized)
    return result


def _expand_step_expression(value: Any, entities: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("TaskIR step target id must be non-empty")
        value = {"op": "entity_ref", "id": value.strip()}
    elif isinstance(value, Mapping) and "op" not in value:
        identity = value.get("entity", value.get("entity_id", value.get("id")))
        if identity is None:
            raise ValueError("structured step target requires op or entity id")
        value = {"op": "entity_ref", "id": identity}
    if not isinstance(value, Mapping):
        raise ValueError("TaskIR step target must be an AST object or entity id")
    return _expand_entity_references(_normalize_expression(value), entities)


def _expand_region_expression(value: Any, entities: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("op") == "between_region":
        anchors = value.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != 2:
            raise ValueError("between_region requires exactly two anchor selections")
        return {"op": "between_region", "anchors": [
            _expand_step_expression(anchor, entities) for anchor in anchors
        ]}
    return _expand_step_expression(value, entities)


def _normal_op(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("TaskIR AST op must be a string")
    op = value.strip().lower().replace("-", "_")
    aliases = {
        "filter_by_class": "filter_class",
        "class_filter": "filter_class",
        "filter_by_attribute": "filter_attribute",
        "attribute_filter": "filter_attribute",
        "closest": "argmin_distance",
        "farthest": "argmax_distance",
        "count": "count_distinct",
        "object": "object_ref",
        "entity": "entity_ref",
        "and": "intersection",
        "or": "union",
    }
    return aliases.get(op, op)


def _normalize_expression(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 48:
        raise ValueError("TaskIR AST is too deeply nested")
    if not isinstance(value, Mapping):
        raise ValueError("each TaskIR expression node must be an object")
    node = dict(value)
    op = _normal_op(node.get("op", node.get("operator", node.get("type", ""))))
    if op not in AST_OPERATORS:
        raise ValueError(f"unsupported TaskIR AST operator: {op!r}")
    node["op"] = op

    if op in {"all", "true", "false"}:
        return node
    if op in {"filter_class", "filter_attribute"}:
        if op == "filter_class":
            class_name = node.get(
                "class", node.get("class_name", node.get("concept", node.get("name", "")))
            )
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError("filter_class requires class")
            node["class"] = class_name.strip().lower()
            if "visual_class" not in node:
                raise ValueError(
                    "filter_class requires explicit visual_class"
                )
            visual_class = node["visual_class"]
            if not isinstance(visual_class, str) or not visual_class.strip():
                raise ValueError("filter_class visual_class must be a non-empty string")
            node["visual_class"] = visual_class.strip().lower()
        else:
            attribute = node.get("attribute", node.get("attr", ""))
            if not isinstance(attribute, str) or not attribute.strip():
                raise ValueError("filter_attribute requires attribute")
            if "value" not in node and "equals" not in node:
                raise ValueError("filter_attribute requires value")
            node["attribute"] = attribute.strip().lower()
            if node["attribute"] in RELATIONAL_ATTRIBUTES or node["attribute"] in {"class", "category"}:
                raise ValueError("filter_attribute requires an intrinsic attribute, not a relation or class")
            node.setdefault("value", node.get("equals"))
        source = node.get("source", node.get("input", node.get("over")))
        if source is not None:
            node["source"] = _normalize_expression(source, depth + 1)
        return node
    if op in {"entity_ref", "object_ref", "bound_ref"}:
        identity = node.get(
            "id", node.get("entity", node.get("entity_id", node.get("name")))
        )
        if identity is None:
            raise ValueError(f"{op} requires id/entity")
        if not isinstance(identity, (str, int)):
            raise ValueError(f"{op} id must be scalar")
        node["id"] = str(identity)
        if "source" in node:
            node["source"] = _normalize_expression(node["source"], depth + 1)
        return node
    if op == "literal":
        if "value" not in node and "ids" not in node:
            raise ValueError("literal requires value or ids")
        return node
    if op in {"bind", "let"}:
        if op == "bind":
            name = node.get("name", node.get("id"))
            source = node.get("source", node.get("value"))
            if not isinstance(name, str) or not name.strip() or source is None:
                raise ValueError("bind requires name and source")
            node["name"] = name.strip()
            node["source"] = _normalize_expression(source, depth + 1)
            body = node.get("body", node.get("in"))
            if body is not None:
                node["body"] = _normalize_expression(body, depth + 1)
        else:
            bindings = node.get("bindings", node.get("bind", {}))
            body = node.get("body", node.get("in"))
            if not isinstance(bindings, Mapping) or body is None:
                raise ValueError("let requires bindings and body")
            node["bindings"] = {
                str(name): _normalize_expression(expr, depth + 1)
                for name, expr in bindings.items()
            }
            node["body"] = _normalize_expression(body, depth + 1)
        return node
    if op == "select":
        source = node.get("source", node.get("input"))
        identity = node.get("id", node.get("object_id"))
        if source is None and identity is None and "ids" not in node:
            raise ValueError("select requires source or id")
        if source is not None:
            node["source"] = _normalize_expression(source, depth + 1)
        return node
    if op in {"union", "intersection", "difference", "sequence"}:
        items = node.get("items", node.get("operands", node.get("sources")))
        if not isinstance(items, list) or not items:
            raise ValueError(f"{op} requires a non-empty items array")
        node["items"] = [_normalize_expression(item, depth + 1) for item in items]
        return node
    if op == "not":
        child = node.get("source", node.get("input", node.get("operand")))
        if child is None:
            raise ValueError("not requires source")
        node["source"] = _normalize_expression(child, depth + 1)
        return node
    if op in {"near", "on", "inside", "same_room", "above", "below"}:
        subject, anchor = _relation_operands(node, depth)
        node["subject"] = subject
        node["anchor"] = anchor
        node["return_role"] = _relation_return_role(node)
        return node
    if op == "between":
        subject = node.get("subject", node.get("input", node.get("object")))
        anchors = node.get("anchors", node.get("boundaries", node.get("objects")))
        if subject is None or not isinstance(anchors, list) or len(anchors) != 2:
            raise ValueError("between requires subject and exactly two anchors")
        node["subject"] = _normalize_expression(subject, depth + 1)
        node["anchors"] = [_normalize_expression(item, depth + 1) for item in anchors]
        return node
    if op in {"argmin_distance", "argmax_distance"}:
        candidates = node.get("candidates", node.get("source", node.get("input")))
        anchor = node.get("anchor", node.get("reference", node.get("object")))
        if candidates is None or anchor is None:
            raise ValueError(f"{op} requires candidates and one explicit anchor")
        node["candidates"] = _normalize_expression(candidates, depth + 1)
        node["anchor"] = _normalize_expression(anchor, depth + 1)
        return node
    if op == "count_distinct":
        source = node.get("source", node.get("input", node.get("operand")))
        if source is None:
            raise ValueError("count_distinct requires source")
        node["source"] = _normalize_expression(source, depth + 1)
        node.setdefault("key", "id")
        return node
    return node


def _relation_operands(
    node: Mapping[str, Any], depth: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    subject = node.get(
        "subject", node.get("input", node.get("left", node.get("object")))
    )
    anchor = node.get("anchor", node.get("reference", node.get("right")))
    operands = node.get("operands")
    if (subject is None or anchor is None) and isinstance(operands, list) and len(operands) == 2:
        subject, anchor = operands
    if subject is None or anchor is None:
        raise ValueError("binary relation requires subject and anchor")
    return _normalize_expression(subject, depth + 1), _normalize_expression(anchor, depth + 1)


def _relation_return_role(node: Mapping[str, Any]) -> str:
    value = node.get("return_role", "subject")
    if not isinstance(value, str):
        raise ValueError("binary relation return_role must be subject or anchor")
    role = value.strip().lower()
    if role not in RELATION_RETURN_ROLES:
        raise ValueError(
            "binary relation return_role must be subject or anchor"
        )
    return role


def task_concepts(task: Mapping[str, Any]) -> list[str]:
    """Collect the finite visual vocabulary encoded by a normalized task."""

    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    values: list[str] = []
    expression = task.get("expression")
    if isinstance(expression, Mapping):
        values.extend(_concepts_from_expression(expression))
    values.extend(_concepts_from_steps(task.get("steps", [])))
    return _unique_strings(values)


def task_visual_queries(task: Mapping[str, Any]) -> dict[str, str]:
    """Collect semantic-to-visual detector queries from expanded AST nodes."""

    if not isinstance(task, Mapping):
        raise ValueError("task must be an object")
    queries: dict[str, str] = {}
    nested_keys = (
        "source",
        "input",
        "operand",
        "subject",
        "anchor",
        "candidates",
        "body",
        "in",
        "items",
        "operands",
        "sources",
        "bindings",
        "boundaries",
        "anchors",
    )

    def add_query(
        semantic: Any,
        visual: Any = None,
        *,
        require_visual: bool = True,
    ) -> None:
        if not isinstance(semantic, str) or not semantic.strip():
            raise ValueError("filter_class requires a non-empty semantic class")
        if visual is None:
            if require_visual:
                raise ValueError("filter_class requires explicit visual_class")
            visual = semantic
        if not isinstance(visual, str) or not visual.strip():
            raise ValueError("filter_class visual_class must be a non-empty string")
        semantic_key = semantic.strip().lower()
        queries.setdefault(semantic_key, visual.strip().lower())

    def visit(expression: Any, depth: int = 0) -> None:
        if depth > 48 or not isinstance(expression, Mapping):
            return
        raw_op = expression.get("op", expression.get("operator", expression.get("type")))
        if raw_op is not None and _normal_op(raw_op) == "filter_class":
            semantic = expression.get(
                "class", expression.get("class_name", expression.get("concept"))
            )
            add_query(semantic, expression.get("visual_class"))
        for key in nested_keys:
            value = expression.get(key)
            if isinstance(value, Mapping):
                if key == "bindings":
                    for child in value.values():
                        visit(child, depth + 1)
                else:
                    visit(value, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    visit(child, depth + 1)

    visit(task.get("expression"))
    steps = task.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, Mapping):
                for key in ("concept", "class", "class_name"):
                    if (key in step and isinstance(step[key], str)
                            and step[key].strip()):
                        add_query(
                            step[key],
                            step.get("visual_class"),
                            require_visual=False,
                        )
                for expression in _step_expressions(step):
                    visit(expression)
    return queries


def _concepts_from_entities(entities: Any) -> list[str]:
    if isinstance(entities, Mapping):
        entities = list(entities.values())
    if not isinstance(entities, list):
        return []
    result: list[str] = []
    for item in entities:
        if isinstance(item, Mapping):
            value = item.get("class", item.get("class_name", item.get("concept")))
            if isinstance(value, str) and value.strip():
                result.append(value.strip().lower())
    return result


def _concepts_from_expression(expression: Any, depth: int = 0) -> list[str]:
    if depth > 48 or not isinstance(expression, Mapping):
        return []
    result: list[str] = []
    op = str(expression.get("op", "")).lower()
    if op == "filter_class":
        value = expression.get(
            "class", expression.get("class_name", expression.get("concept"))
        )
        if isinstance(value, str) and value.strip():
            result.append(value.strip().lower())
    if op in {"entity_ref", "object_ref", "bound_ref"}:
        value = expression.get(
            "class", expression.get("class_name", expression.get("concept"))
        )
        if isinstance(value, str) and value.strip():
            result.append(value.strip().lower())
    nested_keys = {
        "source",
        "input",
        "operand",
        "subject",
        "anchor",
        "candidates",
        "body",
        "in",
        "items",
        "operands",
        "sources",
        "bindings",
        "boundaries",
        "anchors",
    }
    for key, value in expression.items():
        if key not in nested_keys:
            continue
        if isinstance(value, Mapping):
            if key == "bindings":
                for child in value.values():
                    result.extend(_concepts_from_expression(child, depth + 1))
            else:
                result.extend(_concepts_from_expression(value, depth + 1))
        elif isinstance(value, list):
            for child in value:
                result.extend(_concepts_from_expression(child, depth + 1))
    return result


def _concepts_from_steps(steps: Any) -> list[str]:
    if not isinstance(steps, list):
        return []
    result: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        for key in ("concept", "class", "class_name"):
            value = step.get(key)
            if isinstance(value, str) and value.strip():
                result.append(value.strip().lower())
        for expression in _step_expressions(step):
            result.extend(_concepts_from_expression(expression))
    return result


def _step_expressions(step: Mapping[str, Any]):
    for key in ("target", "region"):
        if isinstance(step.get(key), Mapping):
            yield step[key]
    for anchor in step.get("anchors", []):
        if isinstance(anchor, Mapping):
            yield anchor


def _unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


@dataclass
class _Selection:
    ids: list[str] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    value: int | None = None

    def merge_diagnostics(self, other: "_Selection") -> None:
        self.missing.extend(other.missing)
        self.unknown.extend(other.unknown)
        for key, value in other.details.items():
            self.details.setdefault(key, value)


def evaluate(
    expression: Mapping[str, Any],
    records: Mapping[str, ObjectRecord],
) -> QueryResult:
    """Evaluate a nested spatial expression against object records.

    The missing field contains absent observations and unresolved pair
    evidence. The same unresolved relation rows are also available in the
    details unknown field so the caller can request a translated view.
    """

    if not isinstance(expression, Mapping):
        raise ValueError("expression must be an AST object")
    if not isinstance(records, Mapping):
        raise ValueError("records must map object IDs to ObjectRecord values")
    record_map = _record_map(records)
    selected = _eval(dict(expression), record_map, {}, 0)
    object_ids = _unique_strings(selected.ids)
    details = dict(selected.details)
    details["unknown"] = list(selected.unknown)
    details.setdefault(
        "candidate_coverage",
        {
            "observed_record_count": len(record_map),
            "selected_record_count": len(object_ids),
            "unknown_count": len(selected.unknown),
            "missing_count": len(selected.missing),
            "complete": not selected.missing and not selected.unknown,
        },
    )
    value = selected.value
    if value is None and _normal_op(expression.get("op", "")) == "count_distinct":
        value = len(object_ids)
    return QueryResult(
        object_ids=object_ids,
        value=value,
        missing=[*selected.missing, *selected.unknown],
        complete=not selected.missing and not selected.unknown,
        details=details,
    )


def selection_supported(expression: Mapping[str, Any], result: QueryResult) -> bool:
    """Positive existential witnesses need not close unrelated pairs.

    Coverage remains in result.complete/missing. Extrema and exclusions are
    non-monotone and retain their complete-evidence requirement.
    """
    def non_monotone(value):
        if isinstance(value, Mapping):
            return value.get('op') in {'argmin_distance', 'argmax_distance', 'difference', 'not', 'count_distinct'} or any(non_monotone(v) for v in value.values())
        return isinstance(value, list) and any(non_monotone(v) for v in value)
    return bool(result.object_ids) and (result.complete or not non_monotone(expression))


def binding_supported(expression: Mapping[str, Any], result: QueryResult) -> bool:
    return len(result.object_ids) == 1 and selection_supported(expression, result)


def bind_target(expression: Mapping[str, Any], records: Mapping[str, Any]) -> QueryResult:
    """Bind a singular target using the AST and explicit object evidence."""
    result = evaluate(expression, records)
    if not result.complete or len(result.object_ids) <= 1:
        return result
    confirmed = [oid for oid in result.object_ids if any(
        value.get('verdict') == 'yes'
        for value in _get(records[oid], 'attributes', {}).get('category_evidence', {}).values()
        if isinstance(value, Mapping))]
    # Match the normal LiDAR support used before invoking sparse depth; one
    # stray foreground hit must not equal a consistently measured object.
    measured = [oid for oid in confirmed if len(_get(records[oid], 'measured_points', [])) >= 12]
    supported = measured or confirmed
    if len(supported) == 1:
        result.details['binding_candidates'] = list(result.object_ids)
        result.details['binding_basis'] = 'confirmed_category_with_measurement' if measured else 'confirmed_category'
        result.object_ids = supported
    return result


def resolve(
    task: Mapping[str, Any],
    records: Mapping[str, ObjectRecord],
) -> QueryResult:
    """Evaluate the expression embedded in one TaskIR object."""

    if not isinstance(task, Mapping):
        raise ValueError("task must be a TaskIR object")
    expression = task.get("expression")
    if not isinstance(expression, Mapping):
        raise ValueError("TaskIR requires a structured expression")
    result = (bind_target(expression, records) if task.get('task_type') in {'object_reference', 'instruction'}
              else evaluate(expression, records))
    result.details.setdefault("task_type", task.get("task_type"))
    result.details.setdefault("concepts", task_concepts(task))
    result.details.setdefault(
        "steps",
        list(task.get("steps", [])) if isinstance(task.get("steps", []), list) else [],
    )
    return result


def _record_map(records: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, record in records.items():
        identity = _record_id(record, key)
        if identity:
            result[identity] = record
    return result


def _record_id(record: Any, fallback: Any = "") -> str:
    value = _get(record, "id", fallback)
    return str(value).strip()


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _eval(
    node: Mapping[str, Any],
    records: Mapping[str, Any],
    env: dict[str, _Selection],
    depth: int,
) -> _Selection:
    if depth > 48:
        return _Selection(unknown=[{"reason": "ast_depth_exceeded"}])
    op = _normal_op(node.get("op", node.get("operator", "")))
    if op == "all":
        return _Selection(ids=list(records), details={"selected_by": "all"})
    if op == "true":
        return _Selection(ids=list(records))
    if op == "false":
        return _Selection()
    if op == "literal":
        raw = node.get("ids", node.get("value", []))
        values = raw if isinstance(raw, list) else [raw]
        identities = [str(value) for value in values]
        missing = [
            {"reason": "record_absent", "object_id": identity}
            for identity in identities
            if identity not in records
        ]
        return _Selection(
            ids=[identity for identity in identities if identity in records],
            missing=missing,
        )
    if op in {"object_ref", "bound_ref"}:
        identity = str(node.get("id", ""))
        if identity in env:
            return _copy_selection(env[identity])
        if identity in records:
            return _Selection(ids=[identity], details={"explicit_object_id": identity})
        return _Selection(
            missing=[{"reason": "bound_object_absent", "object_id": identity}]
        )
    if op == "entity_ref":
        identity = str(node.get("id", ""))
        if identity in env:
            return _copy_selection(env[identity])
        if identity in records:
            return _Selection(ids=[identity], details={"explicit_object_id": identity})
        class_name = node.get("class", node.get("class_name", node.get("concept")))
        if isinstance(class_name, str) and class_name.strip():
            return _filter_class(class_name, records, node.get("min_score"))
        return _Selection(
            unknown=[{"reason": "entity_unbound", "entity_id": identity}]
        )
    if op == "filter_class":
        source = _source_selection(node, records, env, depth)
        result = _filter_class(
            str(node.get("class", "")),
            records,
            node.get("min_score"),
            source,
        )
        result.merge_diagnostics(source)
        return result
    if op == "filter_attribute":
        source = _source_selection(node, records, env, depth)
        return _filter_attribute(
            str(node.get("attribute", "")),
            node.get("value"),
            records,
            source,
        )
    if op == "select":
        if "id" in node or "object_id" in node:
            return _eval(
                {
                    "op": "literal",
                    "ids": [node.get("id", node.get("object_id"))],
                },
                records,
                env,
                depth + 1,
            )
        if "ids" in node:
            return _eval({"op": "literal", "ids": node["ids"]}, records, env, depth + 1)
        return _eval(node.get("source", {}), records, env, depth + 1)
    if op == "bind":
        source = _eval(node["source"], records, env, depth + 1)
        local = dict(env)
        local[str(node["name"])] = _copy_selection(source)
        body = node.get("body")
        if isinstance(body, Mapping):
            return _eval(body, records, local, depth + 1)
        return source
    if op == "let":
        local = dict(env)
        diagnostics = _Selection()
        for name, child in node.get("bindings", {}).items():
            selected = _eval(child, records, local, depth + 1)
            local[str(name)] = selected
            diagnostics.merge_diagnostics(selected)
        result = _eval(node["body"], records, local, depth + 1)
        result.merge_diagnostics(diagnostics)
        return result
    if op in {"union", "intersection", "difference", "sequence"}:
        return _set_operation(op, node.get("items", []), records, env, depth)
    if op == "not":
        source = _eval(node["source"], records, env, depth + 1)
        return _Selection(
            ids=[identity for identity in records if identity not in source.ids],
            missing=source.missing,
            unknown=source.unknown,
            details={"negated": source.details},
        )
    if op in {"near", "on", "inside", "same_room", "above", "below"}:
        subject = _eval(node["subject"], records, env, depth + 1)
        anchor = _eval(node["anchor"], records, env, depth + 1)
        return _relation_filter(op, node, subject, anchor, records)
    if op == "between":
        subject = _eval(node["subject"], records, env, depth + 1)
        anchor_sets = [
            _eval(value, records, env, depth + 1) for value in node["anchors"]
        ]
        return _between_filter(node, subject, anchor_sets, records)
    if op in {"argmin_distance", "argmax_distance"}:
        candidates = _eval(node["candidates"], records, env, depth + 1)
        anchor = _eval(node["anchor"], records, env, depth + 1)
        return _rank(node, candidates, anchor, records)
    if op == "count_distinct":
        source = _eval(node["source"], records, env, depth + 1)
        ids = _unique_strings(source.ids)
        return _Selection(
            ids=ids,
            missing=source.missing,
            unknown=source.unknown,
            details={
                **source.details,
                "counted_ids": list(ids),
                "candidate_coverage": _coverage(source, records),
            },
            value=len(ids),
        )
    return _Selection(unknown=[{"reason": "unsupported_ast_operator", "op": op}])


def _copy_selection(value: _Selection) -> _Selection:
    return _Selection(
        ids=list(value.ids),
        missing=[dict(item) for item in value.missing],
        unknown=[dict(item) for item in value.unknown],
        details=dict(value.details),
        value=value.value,
    )


def _source_selection(
    node: Mapping[str, Any],
    records: Mapping[str, Any],
    env: dict[str, _Selection],
    depth: int,
) -> _Selection:
    source = node.get("source")
    if isinstance(source, Mapping):
        return _eval(source, records, env, depth + 1)
    return _Selection(ids=list(records), details={"selected_by": "all"})


_CLASS_ALIASES = {
    "tv": "television",
    "television set": "television",
    "couch": "sofa",
    "settee": "sofa",
    "cushion": "pillow",
    "photo": "picture",
    "photograph": "picture",
    "photos": "picture",
    "pictures": "picture",
    "bookcase": "bookshelf",
    "book shelf": "bookshelf",
}


def _label_forms(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if not text:
        return set()
    forms = {text, _CLASS_ALIASES.get(text, text)}
    words = text.split()
    if words and words[-1].endswith("s") and len(words[-1]) > 3:
        singular = " ".join(words[:-1] + [words[-1][:-1]])
        forms.update({singular, _CLASS_ALIASES.get(singular, singular)})
    return forms


def _filter_class(
    class_name: str,
    records: Mapping[str, Any],
    min_score: Any = None,
    source: _Selection | None = None,
) -> _Selection:
    threshold = 0.0 if min_score is None else _safe_float(min_score, 0.0)
    universe = list(source.ids) if source is not None else list(records)
    ids = [
        identity
        for identity in universe
        if (_class_score(records.get(identity), class_name) or 0.0) > threshold
    ]
    result = _Selection(
        ids=ids,
        details={
            "class": class_name,
            "class_observed": bool(ids),
            "candidate_count": len(ids),
        },
    )
    if not ids:
        # An empty candidate set is not a closed-world proof that the class
        # does not exist. Keep the query open until the requested category has
        # been observed (or the scene itself is unavailable), including when
        # other classes already populate the record map.
        result.unknown.append(
            {
                "reason": "class_not_observed" if records else "scene_records_unavailable",
                "class": class_name,
                "candidate_scope_count": len(universe),
            }
        )
    return result


def _class_score(record: Any, requested: str) -> float | None:
    forms = _label_forms(requested)
    attributes = _get(record, "attributes", {})
    category_evidence = attributes.get("category_evidence", {}) if isinstance(attributes, Mapping) else {}
    for label, evidence in category_evidence.items():
        if forms.intersection(_label_forms(label)) and evidence.get("verdict") == "no":
            return None
    scores = _get(record, "class_scores", {})
    found: list[float] = []
    if isinstance(scores, Mapping):
        for label, value in scores.items():
            if forms.intersection(_label_forms(label)):
                try:
                    found.append(float(value))
                except (TypeError, ValueError):
                    found.append(1.0 if value else 0.0)
    if found:
        return max(found)
    attrs = _get(record, "attributes", {})
    if isinstance(attrs, Mapping):
        for key in ("class", "class_name", "concept", "category", "label"):
            if forms.intersection(_label_forms(attrs.get(key))):
                return 1.0
    return None


def _filter_attribute(
    attribute: str,
    expected: Any,
    records: Mapping[str, Any],
    source: _Selection,
) -> _Selection:
    ids: list[str] = []
    missing: list[dict[str, Any]] = []
    for identity in source.ids:
        attrs = _get(records.get(identity), "attributes", {})
        values = _attribute_values(attrs, attribute) if isinstance(attrs, Mapping) else []
        if not values:
            missing.append(
                {
                    "reason": "attribute_not_observed",
                    "attribute": attribute,
                    "object_id": identity,
                }
            )
        elif any(_value_matches(value, expected) for value in values):
            ids.append(identity)
    return _Selection(
        ids=ids,
        missing=source.missing + missing,
        unknown=source.unknown,
        details={
            "attribute": attribute,
            "expected": expected,
            "candidate_count": len(ids),
        },
    )


def _attribute_values(attrs: Mapping[str, Any], attribute: str) -> list[Any]:
    wanted = attribute.lower().strip()
    values: list[Any] = []
    for key, value in attrs.items():
        key_norm = str(key).lower().strip()
        if key_norm == wanted or (
            wanted == "color" and key_norm in {"colour", "colors", "colours"}
        ):
            values.extend(value if isinstance(value, (list, tuple, set)) else [value])
    return values


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return bool(_label_forms(actual).intersection(_label_forms(expected)))
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return all(
            _value_matches(actual.get(key), value) for key, value in expected.items()
        )
    return actual == expected


def _set_operation(
    op: str,
    items: list[Any],
    records: Mapping[str, Any],
    env: dict[str, _Selection],
    depth: int,
) -> _Selection:
    selections = [_eval(item, records, env, depth + 1) for item in items]
    if not selections:
        return _Selection()
    if op in {"union", "sequence"}:
        ids = _unique_strings(
            identity for selected in selections for identity in selected.ids
        )
    elif op == "intersection":
        common = set(selections[0].ids)
        for selected in selections[1:]:
            common.intersection_update(selected.ids)
        ids = [identity for identity in records if identity in common]
    else:
        removed = {
            identity
            for selected in selections[1:]
            for identity in selected.ids
        }
        ids = [identity for identity in selections[0].ids if identity not in removed]
    result = _Selection(ids=ids, value=selections[-1].value)
    for selected in selections:
        result.merge_diagnostics(selected)
    result.details["operation"] = op
    return result


def _relation_filter(
    op: str,
    node: Mapping[str, Any],
    subject: _Selection,
    anchor: _Selection,
    records: Mapping[str, Any],
) -> _Selection:
    result = _Selection()
    result.merge_diagnostics(subject)
    result.merge_diagnostics(anchor)
    return_role = _relation_return_role(node)
    relation_rows: list[dict[str, Any]] = []
    anchor_ids = list(anchor.ids)
    if not anchor_ids and anchor.unknown:
        result.unknown.append(
            {
                "reason": "anchor_not_resolved",
                "relation": op,
                "anchor_diagnostics": [dict(item) for item in anchor.unknown],
            }
        )
    for subject_id in subject.ids:
        yes: list[str] = []
        unknown_for_subject: list[dict[str, Any]] = []
        for anchor_id in anchor_ids:
            state, evidence = _relation_state(op, node, subject_id, anchor_id, records)
            row = {
                "relation": op,
                "subject_id": subject_id,
                "anchor_id": anchor_id,
                "state": state,
            }
            if evidence:
                row["evidence"] = evidence
            relation_rows.append(row)
            if state == "YES":
                yes.append(anchor_id)
            elif state == "UNKNOWN":
                unknown_for_subject.append(
                    {
                        "reason": "relation_evidence_unknown",
                        "relation": op,
                        "subject_id": subject_id,
                        "anchor_id": anchor_id,
                        "evidence": evidence,
                    }
                )
        if yes:
            if return_role == "subject":
                result.ids.append(subject_id)
            else:
                # Project the qualifying side of the relation. This is a
                # half-join: R(subject, anchor) returns each anchor that has
                # at least one qualifying subject, while preserving the
                # subject -> anchor relation direction above.
                for anchor_id in yes:
                    if anchor_id not in result.ids:
                        result.ids.append(anchor_id)
            if (return_role == "subject" and len(anchor_ids) > 1
                    and not bool(node.get("allow_multiple_anchors", False))):
                result.unknown.append(
                    {
                        "reason": "multiple_anchor_candidates",
                        "relation": op,
                        "subject_id": subject_id,
                        "anchor_ids": list(anchor_ids),
                    }
                )
        elif unknown_for_subject:
            result.unknown.extend(unknown_for_subject)
            if op in {"on", "inside"}:
                result.missing.extend(unknown_for_subject)
    result.details["relation"] = op
    result.details["return_role"] = return_role
    result.details["relation_evidence"] = relation_rows
    result.details["anchor_candidates"] = list(anchor_ids)
    return result


def _between_filter(
    node: Mapping[str, Any],
    subject: _Selection,
    anchors: list[_Selection],
    records: Mapping[str, Any],
) -> _Selection:
    result = _Selection()
    result.merge_diagnostics(subject)
    for selected in anchors:
        result.merge_diagnostics(selected)
    if len(anchors) != 2:
        result.unknown.append({"reason": "between_requires_two_anchors"})
        return result
    # BETWEEN is symmetric in its anchors and requires two physical bodies.
    # Identical class expressions may legitimately resolve the same pair.
    anchor_pairs = sorted({tuple(sorted((left, right)))
                           for left in anchors[0].ids for right in anchors[1].ids
                           if left != right})
    if not anchor_pairs and all(value.ids for value in anchors):
        result.unknown.append({"reason": "between_requires_two_distinct_anchors",
                               "anchor_ids": [list(value.ids) for value in anchors]})
    rows: list[dict[str, Any]] = []
    for subject_id in subject.ids:
        any_yes = False
        any_unknown = False
        for left_id, right_id in anchor_pairs:
            if subject_id in (left_id, right_id):
                continue
            state, evidence = _relation_state(
                "between", node, subject_id, (left_id, right_id), records,
            )
            row = {"relation": "between", "subject_id": subject_id,
                   "anchor_ids": [left_id, right_id], "state": state}
            if evidence:
                row["evidence"] = evidence
            rows.append(row)
            any_yes |= state == "YES"
            any_unknown |= state == "UNKNOWN"
        if any_yes:
            result.ids.append(subject_id)
        elif any_unknown:
            result.unknown.append(
                {
                    "reason": "relation_evidence_unknown",
                    "relation": "between",
                    "subject_id": subject_id,
                    "anchor_ids": [list(value.ids) for value in anchors],
                }
            )
    result.details["relation"] = "between"
    result.details["relation_evidence"] = rows
    result.details["anchor_candidate_groups"] = [list(value.ids) for value in anchors]
    result.details["anchor_candidates"] = _unique_strings(
        identity for group in anchors for identity in group.ids
    )
    return result


def _rank(
    node: Mapping[str, Any],
    candidates: _Selection,
    anchor: _Selection,
    records: Mapping[str, Any],
) -> _Selection:
    result = _Selection()
    result.merge_diagnostics(candidates)
    result.merge_diagnostics(anchor)
    if not anchor.ids:
        result.unknown.append(
            {"reason": "anchor_not_resolved", "relation": node.get("op")}
        )
        result.details["anchor_candidates"] = []
        return result
    rankings: dict[str, dict[str, Any]] = {}
    geometry_unknown: list[dict[str, Any]] = []
    reverse = node.get("op") == "argmax_distance"
    for anchor_id in anchor.ids:
        distances: dict[str, float] = {}
        for candidate_id in candidates.ids:
            distance = _distance(records.get(candidate_id), records.get(anchor_id))
            if distance is None:
                geometry_unknown.append(
                    {"reason": "geometry_unknown", "relation": node.get("op"),
                     "subject_id": candidate_id, "anchor_id": anchor_id}
                )
            else:
                distances[candidate_id] = distance
        winners: list[str] = []
        if distances:
            winner_distance = (max if reverse else min)(distances.values())
            winners = [identity for identity, value in distances.items()
                       if abs(value-winner_distance) <= 1e-6]
        rankings[anchor_id] = {"distances_m": distances, "winner_ids": winners}
    result.unknown.extend(geometry_unknown)
    result.details.update(
        relation=node.get("op"),
        rankings_by_anchor=rankings,
        candidate_coverage={
            "observed_candidates": len(candidates.ids),
            "observed_anchors": len(anchor.ids),
            "geometry_unknown_pairs": len(geometry_unknown),
            "complete": bool(candidates.ids) and not geometry_unknown,
        },
    )
    # The anchor expression denotes an observed object set, as it does for
    # other spatial operators. Distance to that set is distance to its nearest
    # member. This works for multiple actual anchors and split observations
    # without choosing or merging an anchor identity.
    distances = {}
    nearest_anchors = {}
    for candidate_id in candidates.ids:
        values = [(row['distances_m'][candidate_id], anchor_id)
                  for anchor_id, row in rankings.items()
                  if candidate_id in row['distances_m']]
        if values:
            distance = min(value for value, _ in values)
            distances[candidate_id] = distance
            nearest_anchors[candidate_id] = [identity for value, identity in values
                                             if abs(value-distance) <= 1e-6]
    result.details.update(distances_m=distances,
                          nearest_anchor_ids=nearest_anchors,
                          anchor_candidates=list(anchor.ids),
                          anchor_resolution='distance_to_observed_anchor_set')
    if distances:
        best = (max if reverse else min)(distances.values())
        result.ids = [identity for identity, value in distances.items()
                      if abs(value-best) <= 1e-6]
        if len(result.ids) > 1:
            result.unknown.append({'reason': 'distance_tie', 'relation': node.get('op'),
                                   'object_ids': result.ids})
    elif not geometry_unknown:
        result.unknown.append({'reason': 'geometry_unknown', 'relation': node.get('op')})
    return result


def _relation_state(
    relation: str,
    node: Mapping[str, Any],
    subject_id: str,
    anchor: str | tuple[str, str],
    records: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if relation == "between":
        left_id, right_id = anchor
        subject = records.get(subject_id)
        left = records.get(left_id)
        right = records.get(right_id)
        if subject is None or left is None or right is None:
            return "UNKNOWN", {"reason": "record_absent"}
        pair = _pair_evidence(
            "between",
            subject_id,
            (left_id, right_id),
            subject,
            left,
            right,
        )
        geometry = _geometry_between(subject, left, right, node)
        return _combine_geometry_pair(geometry, pair, "between")

    anchor_id = str(anchor)
    subject = records.get(subject_id)
    anchor_record = records.get(anchor_id)
    if subject is None or anchor_record is None:
        return "UNKNOWN", {"reason": "record_absent"}
    pair = _pair_evidence(
        relation,
        subject_id,
        anchor_id,
        subject,
        anchor_record,
    )
    geometry = _geometry_relation(relation, subject, anchor_record, node)
    return _combine_geometry_pair(geometry, pair, relation)


def _combine_geometry_pair(
    geometry: tuple[str, dict[str, Any]],
    pair: tuple[str, dict[str, Any]],
    relation: str,
) -> tuple[str, dict[str, Any]]:
    geometry_state, geometry_detail = geometry
    pair_state, pair_detail = pair
    detail: dict[str, Any] = {
        "geometry_state": geometry_state,
        "image_state": pair_state,
    }
    if geometry_detail:
        detail["geometry"] = geometry_detail
    if pair_detail:
        detail["pair"] = pair_detail
    if pair_state == "YES" and geometry_state == "NO":
        return "UNKNOWN", {**detail, "reason": "image_geometry_disagree"}
    if pair_state == "NO" or geometry_state == "NO":
        return "NO", detail
    if relation in {"on", "inside"} and geometry_state != "YES":
        return "UNKNOWN", {**detail, "reason": "physical_geometry_evidence_needed"}
    if pair_state == "YES" and geometry_state in {"YES", "UNKNOWN"}:
        return "YES", detail
    if geometry_state == "YES" and relation in {"on", "inside"}:
        return "UNKNOWN", {**detail, "reason": "pair_image_evidence_needed"}
    if geometry_state == "YES":
        return "YES", detail
    return "UNKNOWN", detail or {"reason": f"{relation}_evidence_unavailable"}


def _geometry_relation(
    relation: str,
    subject: Any,
    anchor: Any,
    node: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if relation == "near":
        distance = _distance(subject, anchor)
        if distance is None:
            return "UNKNOWN", {"reason": "geometry_unknown"}
        # An explicit local policy, independent of the candidate's distance.
        # A threshold derived from distance + margin made nearly every tested
        # candidate satisfy NEAR. Task-provided metric distances still win.
        default = 1.5
        threshold = _safe_float(
            node.get(
                "max_distance_m",
                node.get("distance_m", node.get("threshold_m", default)),
            ),
            default,
        )
        return (
            "YES" if distance <= threshold else "NO",
            {"distance_m": distance, "threshold_m": threshold},
        )
    if relation in {"on", "inside", "above", "below"}:
        if relation == "on":
            state, detail = _geometry_on(subject, anchor)
        elif relation == "inside":
            state, detail = _geometry_inside(subject, anchor)
        else:
            state, detail = _geometry_vertical(relation, subject, anchor)
        # A visible surface is evidence about observed points, not a complete
        # entity extent. Its missing portions cannot geometrically disprove a
        # relation. Preserve uncertainty for the existing geometry/image
        # evidence combiner instead of treating partial bounds as a full box.
        subject_quality = _get(subject, "geometry_quality", {})
        anchor_quality = _get(anchor, "geometry_quality", {})
        complete_extents = (subject_quality.get("entity_extent_complete") is True
                            and anchor_quality.get("entity_extent_complete") is True)
        if state == "NO" and not complete_extents:
            return "UNKNOWN", {**detail, "reason": "entity_extent_incomplete",
                               "bounds_semantics": "observed_surface_envelope"}
        return state, detail
    if relation == "same_room":
        subject_room = _room(subject)
        anchor_room = _room(anchor)
        if subject_room is None or anchor_room is None:
            return "UNKNOWN", {"reason": "room_identity_unknown"}
        return (
            "YES" if subject_room == anchor_room else "NO",
            {"subject_room": subject_room, "anchor_room": anchor_room},
        )
    return "UNKNOWN", {
        "reason": "relation_geometry_unimplemented",
        "relation": relation,
    }


def _geometry_between(
    subject: Any,
    left: Any,
    right: Any,
    node: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    subject_point = _center(subject)
    left_point = _center(left)
    right_point = _center(right)
    if subject_point is None or left_point is None or right_point is None:
        return "UNKNOWN", {"reason": "geometry_unknown"}
    a = left_point[:2]
    b = right_point[:2]
    p = subject_point[:2]
    vector = b - a
    length_sq = float(np.dot(vector, vector))
    if length_sq <= 1e-9:
        return "UNKNOWN", {"reason": "between_anchors_coincident"}
    along_fraction = float(np.dot(p - a, vector) / length_sq)
    projection = a + along_fraction * vector
    cross_track = float(np.linalg.norm(p - projection))
    default_corridor = max(0.65, min(1.5, 0.2 * math.sqrt(length_sq)))
    corridor = _safe_float(
        node.get("corridor_width_m", node.get("tolerance_m", default_corridor)),
        default_corridor,
    )
    state = (
        "YES"
        if 0.0 <= along_fraction <= 1.0 and cross_track <= corridor
        else "NO"
    )
    return state, {
        "along_fraction": along_fraction,
        "cross_track_m": cross_track,
        "corridor_width_m": corridor,
    }


def _geometry_on(subject: Any, anchor: Any) -> tuple[str, dict[str, Any]]:
    subject_bounds = _bounds(subject)
    anchor_bounds = _bounds(anchor)
    if subject_bounds is None or anchor_bounds is None:
        return "UNKNOWN", {"reason": "support_geometry_unknown"}
    subject_points, subject_source = _surface_points(subject)
    anchor_points, anchor_source = _surface_points(anchor)
    if (subject_points is None or anchor_points is None
            or subject_points.shape[1] < 3 or anchor_points.shape[1] < 3):
        return "UNKNOWN", {"reason": "support_surface_unobserved"}
    # ON can describe support through an unobserved mount/container: visible
    # flowers need not touch the table beneath their vase. Geometry supplies
    # a compatible support column, not a fixed maximum vertical contact gap.
    # The independent image relation remains necessary to distinguish ON
    # from merely being above an object.
    center_z = float((subject_bounds[0][2] + subject_bounds[1][2]) * 0.5)
    below = anchor_points[anchor_points[:, 2] <= center_z]
    if not len(below):
        return "UNKNOWN", {"reason": "no_observed_anchor_surface_below_subject_center"}
    from scipy.spatial import cKDTree
    distances, indices = cKDTree(below[:, :2]).query(subject_points[:, :2], k=1)
    closest = int(np.argmin(distances))
    gap = float(distances[closest])
    state = "YES" if gap <= 0.35 else "NO"
    return state, {
        "support_column_gap_m": gap,
        "horizontal_tolerance_m": 0.35,
        "vertical_gap_m": float(subject_points[closest, 2]-below[int(indices[closest]), 2]),
        "geometry_basis": "observed_support_column_with_image_relation",
        "subject_point_source": subject_source,
        "anchor_point_source": anchor_source,
        "subject_surface_sample": subject_points[closest, :3].tolist(),
        "anchor_surface_sample": below[int(indices[closest]), :3].tolist(),
    }


def _geometry_inside(subject: Any, container: Any) -> tuple[str, dict[str, Any]]:
    subject_bounds = _bounds(subject)
    container_bounds = _bounds(container)
    if subject_bounds is None or container_bounds is None:
        return "UNKNOWN", {"reason": "containment_geometry_unknown"}
    s0, s1 = subject_bounds
    c0, c1 = container_bounds
    margin = 0.05
    state = "YES" if all(
        float(s0[index]) >= float(c0[index]) - margin
        and float(s1[index]) <= float(c1[index]) + margin
        for index in range(3)
    ) else "NO"
    return state, {"containment_margin_m": margin}


def _geometry_vertical(
    relation: str,
    subject: Any,
    anchor: Any,
) -> tuple[str, dict[str, Any]]:
    subject_bounds = _bounds(subject)
    anchor_bounds = _bounds(anchor)
    if subject_bounds is None or anchor_bounds is None:
        return "UNKNOWN", {"reason": "vertical_geometry_unknown"}
    s0, s1 = subject_bounds
    a0, a1 = anchor_bounds
    horizontal_gap = float(np.linalg.norm(np.maximum(
        np.maximum(s0[:2]-a1[:2], a0[:2]-s1[:2]), 0.0)))
    if relation == "above":
        vertical_gap = float(s0[2]-a1[2])
    else:
        vertical_gap = float(a0[2]-s1[2])
    # ABOVE/BELOW describe a spatial direction, not the global ordering of
    # heights anywhere in a room. Compare separation of the observed surfaces
    # along gravity with their lateral separation. This includes a wall object
    # just beyond a support's footprint without assigning an arbitrary search
    # radius to that support. Retain the existing measurement tolerance.
    tolerance = 0.05
    vertical_order = vertical_gap >= -tolerance
    vertical_direction = horizontal_gap <= max(vertical_gap, 0.0)+tolerance
    state = "YES" if vertical_order and vertical_direction else "NO"
    return state, {
        "subject_z": (s0[2] + s1[2]) / 2.0,
        "anchor_z": (a0[2] + a1[2]) / 2.0,
        "vertical_gap_m": vertical_gap,
        "horizontal_gap_m": horizontal_gap,
        "vertical_order": vertical_order,
        "vertical_direction": vertical_direction,
        "measurement_tolerance_m": tolerance,
        "geometry_basis": "vertical_separation_dominates_lateral_separation",
    }


def _room(record: Any) -> str | None:
    attrs = _get(record, "attributes", {})
    if isinstance(attrs, Mapping):
        for key in ("room_id", "room", "room_name", "same_room"):
            value = attrs.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().lower()
    evidence = _get(record, "current_relation_evidence", {})
    if isinstance(evidence, Mapping):
        value = evidence.get("room_id", evidence.get("room"))
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return None


def _pair_evidence(
    relation: str,
    subject_id: str,
    anchor: str | tuple[str, str],
    *records: Any,
) -> tuple[str, dict[str, Any]]:
    anchor_ids = [anchor] if isinstance(anchor, str) else list(anchor)
    found: list[tuple[str, dict[str, Any]]] = []
    for record in records:
        evidence = _get(record, "current_relation_evidence", {})
        if isinstance(evidence, Mapping):
            found.extend(
                _evidence_entries(evidence, relation, subject_id, anchor_ids)
            )
    states = {state for state, _ in found}
    if "YES" in states and "NO" in states:
        return "UNKNOWN", {
            "state": "UNKNOWN",
            "reason": "pair_evidence_conflict",
        }
    for wanted in ("YES", "NO", "UNKNOWN"):
        for state, payload in found:
            if state == wanted:
                return state, payload
    return "UNKNOWN", {}


def _evidence_entries(
    evidence: Mapping[str, Any],
    relation: str,
    subject_id: str,
    anchor_ids: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    relation_name = str(relation).lower()
    accepted = {
        relation_name,
        {"inside": "in", "on": "supported_by"}.get(relation_name, ""),
    }
    accepted.discard("")

    def add(value: Any) -> None:
        state = _evidence_state(value)
        if state is not None:
            payload = (
                dict(value) if isinstance(value, Mapping) else {"state": state}
            )
            payload.setdefault("state", state)
            result.append((state, payload))

    for key in accepted:
        relation_map = evidence.get(key)
        if isinstance(relation_map, Mapping):
            for anchor_id in anchor_ids:
                value = relation_map.get(anchor_id)
                if value is not None:
                    add(value)
        elif relation_map is not None and len(anchor_ids) == 1:
            add(relation_map)

    pairs = evidence.get("pairs", evidence.get("pair_evidence", {}))
    if isinstance(pairs, Mapping):
        relation_map = pairs.get(
            relation_name, pairs.get(relation_name.upper())
        )
        if isinstance(relation_map, Mapping):
            for anchor_id in anchor_ids:
                value = relation_map.get(anchor_id)
                if value is not None:
                    add(value)
        for anchor_id in anchor_ids:
            for key in (
                f"{relation_name}:{anchor_id}",
                f"{subject_id}:{relation_name}:{anchor_id}",
            ):
                if key in pairs:
                    add(pairs[key])

    for anchor_id in anchor_ids:
        for key in (
            f"{relation_name}:{anchor_id}",
            f"{relation_name}|{anchor_id}",
        ):
            if key in evidence:
                add(evidence[key])

    rows = evidence.get("pair_relations", evidence.get("relations", []))
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("relation", "")).lower() not in accepted:
                continue
            row_subject = row.get("subject_id", row.get("subject"))
            row_anchor = row.get("anchor_id", row.get("anchor"))
            if row_subject is not None and str(row_subject) != subject_id:
                continue
            if row_anchor is not None and str(row_anchor) not in anchor_ids:
                continue
            add(row)
    return result


def _evidence_state(value: Any) -> str | None:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, str):
        state = value.strip().upper()
        return state if state in {"YES", "NO", "UNKNOWN"} else None
    if not isinstance(value, Mapping):
        return None
    raw = value.get(
        "state",
        value.get("verdict", value.get("status", value.get("value"))),
    )
    if isinstance(raw, bool):
        return "YES" if raw else "NO"
    if isinstance(raw, str):
        state = raw.strip().upper()
        if state in {"YES", "NO", "UNKNOWN"}:
            return state
    return None


def _center(record: Any) -> np.ndarray | None:
    value = _get(record, "center", None)
    if value is not None:
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            array = np.empty(0)
        if len(array) >= 2 and np.all(np.isfinite(array[: min(len(array), 3)])):
            return array[:3] if len(array) >= 3 else array[:2]
    points = _points(record)
    if points is not None and len(points):
        return np.nanmedian(points[:, : min(3, points.shape[1])], axis=0)
    return None


def _extent(record: Any) -> np.ndarray | None:
    # ObjectRecord.bbox is the full [sx, sy, sz] extent. It is deliberately
    # not interpreted as an AABB or as min/max coordinates.
    value = _get(record, "bbox", None)
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) != 3 or not np.all(np.isfinite(array)):
        return None
    if np.any(array < 0.0):
        return None
    return array


def _bounds(record: Any) -> tuple[np.ndarray, np.ndarray] | None:
    center = _center(record)
    extent = _extent(record)
    if center is not None and len(center) >= 3 and extent is not None:
        half = extent / 2.0
        return center[:3] - half, center[:3] + half
    points = _points(record)
    if points is not None and points.shape[1] >= 3 and len(points):
        return np.nanmin(points[:, :3], axis=0), np.nanmax(points[:, :3], axis=0)
    return None


def measured_extent(record: Any) -> np.ndarray | None:
    """Estimated depth spread is uncertainty, not an observed physical body."""
    quality = _get(record, 'geometry_quality', {})
    if quality.get('bbox_source', quality.get('source')) != 'measured':
        return None
    return _extent(record)


def _points(record: Any) -> np.ndarray | None:
    return _surface_points(record)[0]


def _surface_points(record: Any) -> tuple[np.ndarray | None, str | None]:
    """Return observed samples and their measured/estimated provenance."""
    for key in ("measured_points", "estimated_points"):
        value = _get(record, key, None)
        if value is None:
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.ndim != 2 or array.shape[1] < 2 or not len(array):
            continue
        dimensions = min(array.shape[1], 3)
        finite = array[np.all(np.isfinite(array[:, :dimensions]), axis=1)]
        if len(finite):
            return finite, key
    return None, None


def _distance(first: Any, second: Any) -> float | None:
    first_bounds = _bounds(first) if measured_extent(first) is not None else None
    second_bounds = _bounds(second) if measured_extent(second) is not None else None
    if first_bounds is not None and second_bounds is not None:
        a0, a1 = first_bounds
        b0, b1 = second_bounds
        gaps = np.maximum(0.0, np.maximum(a0 - b1, b0 - a1))
        return float(np.linalg.norm(gaps))
    first_center = _center(first)
    second_center = _center(second)
    if first_center is None or second_center is None:
        return None
    dimensions = min(len(first_center), len(second_center), 3)
    return float(np.linalg.norm(first_center[:dimensions] - second_center[:dimensions]))


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _coverage(selection: _Selection, records: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observed_record_count": len(records),
        "selected_record_count": len(selection.ids),
        "unknown_count": len(selection.unknown),
        "missing_count": len(selection.missing),
        "complete": not selection.missing and not selection.unknown,
    }


__all__ = [
    "AST_OPERATORS",
    "TASK_IR_SCHEMA",
    "TASK_TYPES",
    "evaluate",
    "parse_task_ir",
    "resolve",
    "task_concepts",
    "task_visual_queries",
]
