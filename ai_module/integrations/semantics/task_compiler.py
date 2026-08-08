"""Compile a complete challenge question into a small, validated TaskIR.

DeepSeek is the primary semantic compiler.  The deterministic compiler is kept
as a bounded fallback and as a source of task-type/output-contract invariants.
Neither compiler is allowed to produce coordinates or robot commands.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import time
from typing import Any, Mapping


TASK_TYPES = {"numerical", "object_reference", "instruction_following"}
PREDICATES = {
    "near", "between", "closest", "farthest", "above", "below", "on", "in",
}
ACTIONS = {"go_near", "go_to", "pass_near", "pass_between", "stop_at"}
OUTPUT_BY_TASK = {
    "numerical": "/numerical_response",
    "object_reference": "/selected_object_marker",
    "instruction_following": "/way_point_with_heading",
}
CANONICAL = {
    "tv": ("television", ["tv"]),
    "television set": ("television", ["tv", "television set"]),
    "photos": ("picture", ["photo", "photos", "pictures"]),
    "photo": ("picture", ["photo", "photos", "pictures"]),
    "pictures": ("picture", ["photo", "photos", "pictures"]),
    "plant": ("potted plant", ["plant", "houseplant"]),
    "books": ("book", ["books"]),
    "cushion": ("pillow", ["cushion"]),
}
COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "gray", "grey", "white",
}

# Small, task-independent indoor ontology used only to broaden visual prompts.
# These hints are soft context: they never add a TaskIR constraint and never
# authorize an answer.  The same shape works for furniture, fixtures and
# containers instead of special-casing the television-cabinet question.
GROUNDING_HINTS: dict[str, dict[str, Any]] = {
    "picture": {
        "aliases": ["photo", "photograph", "framed photo", "framed picture"],
        "supporting_context": [],
        "confusers": [
            "wall painting",
            "wall art",
            "poster",
            "television screen content",
        ],
    },
    "television cabinet": {
        "aliases": ["tv cabinet", "tv stand", "media console", "entertainment unit"],
        "supporting_context": [("television", ["tv"], ["on", "above", "near"])],
        "confusers": ["sideboard", "storage cabinet", "bookshelf", "wall cabinet"],
    },
    "bookshelf": {
        "aliases": ["bookcase", "book shelf"],
        "supporting_context": [("book", ["books"], ["on", "in"])],
        "confusers": ["storage cabinet", "wall shelf"],
    },
    "sofa": {
        "aliases": ["couch", "settee"],
        "supporting_context": [("pillow", ["cushion"], ["on"])],
        "confusers": ["armchair", "bench", "bed"],
    },
    "nightstand": {
        "aliases": ["bedside table", "bedside cabinet"],
        "supporting_context": [
            ("bed", [], ["near"]),
            ("lamp", ["bedside lamp"], ["on", "above"]),
        ],
        "confusers": ["side table", "small cabinet"],
    },
    "desk": {
        "aliases": ["work desk", "computer desk"],
        "supporting_context": [
            ("monitor", ["computer monitor"], ["on", "above"]),
            ("keyboard", [], ["on"]),
            ("office chair", ["desk chair"], ["near"]),
        ],
        "confusers": ["dining table", "console table"],
    },
    "coffee table": {
        "aliases": ["cocktail table", "low table"],
        "supporting_context": [("sofa", ["couch"], ["near"])],
        "confusers": ["dining table", "desk", "side table"],
    },
    "dining table": {
        "aliases": ["dinner table"],
        "supporting_context": [("dining chair", ["chair"], ["near"])],
        "confusers": ["desk", "coffee table", "console table"],
    },
    "sink": {
        "aliases": ["washbasin", "basin"],
        "supporting_context": [("faucet", ["tap"], ["above", "near"])],
        "confusers": ["bowl", "bathtub"],
    },
    "bed": {
        "aliases": [],
        "supporting_context": [("pillow", ["cushion"], ["on"])],
        "confusers": ["sofa", "daybed"],
    },
    "potted plant": {
        "aliases": ["houseplant", "plant in pot", "plant pot", "potted plant"],
        "supporting_context": [],
        "confusers": ["vase", "artificial plant"],
    },
}

SYSTEM_PROMPT = """You are a deterministic task compiler for a competition robot.
Compile only the supplied question and return one JSON object with no prose.
Never output coordinates, waypoints, hidden reasoning, commands, URLs, or tools.

The exact top-level keys are: schema_version, task_type, original_question,
entities, relations, ordered_subgoals, target_entity, required_classes,
output_contract, parser_confidence. schema_version is "task_ir_v1". task_type is
numerical, object_reference, or instruction_following. entities contain exactly
id, role, class_name, aliases, attributes. role is target or anchor. attributes
is an object. relations contain exactly id, predicate, subject_entity,
object_entities, depends_on. Allowed predicates: near, between, closest,
farthest, above, below, on, in. ordered_subgoals contain exactly order, action,
target_entity, relation_ids, terminal. Allowed actions: go_near, go_to,
pass_near, pass_between, stop_at. Orders are contiguous from zero.
For numerical and object_reference tasks ordered_subgoals MUST be an empty list.
Only instruction_following tasks may contain ordered_subgoals.
For go_near and pass_near, set the landmark itself as target_entity and leave
relation_ids empty; the action already encodes the route relation. Never create
a relation whose subject is robot or robot_path. Object qualifiers such as
farthest(pillow,lamp) belong in relations and are referenced by the go_to step.

Preserve the grammatical target, every anchor, nested relations, comparisons,
and instruction order. Do not select the first noun. Required classes contain
the target and every anchor class. output_contract must be /numerical_response,
/selected_object_marker, or /way_point_with_heading according to task_type.

Examples:
How many photos are on the TV cabinet? => target picture; anchor television
cabinet; relation on(picture, television cabinet); numerical.
Find the vase between the cabinet and the stool. => target vase; anchors cabinet
and stool; relation between(vase, cabinet, stool); object_reference.
Take the path near the TV and go to the pillow farthest from the lamp. => first
pass_near television, then go_to pillow qualified by farthest(pillow,lamp).
"""


def _normalize_phrase(value: str) -> tuple[str, list[str], dict[str, str]]:
    words = re.sub(r"[^a-z0-9 -]", "", value.lower()).strip().split()
    while words and words[0] in {"the", "a", "an"}:
        words.pop(0)
    attributes: dict[str, str] = {}
    kept = []
    for word in words:
        if word in COLORS:
            attributes["color"] = "gray" if word == "grey" else word
        else:
            kept.append(word)
    phrase = " ".join(kept).strip()
    if phrase == "tv" or phrase.startswith("tv "):
        phrase = "television" + phrase[2:]
    if phrase.endswith("s") and phrase not in CANONICAL and not phrase.endswith("ss"):
        phrase = phrase[:-1]
    canonical, aliases = CANONICAL.get(phrase, (phrase, []))
    if not canonical:
        raise ValueError("empty_entity_class")
    return canonical, aliases, attributes


def _entity(entity_id: str, role: str, phrase: str) -> dict[str, Any]:
    class_name, aliases, attributes = _normalize_phrase(phrase)
    return {
        "id": entity_id,
        "role": role,
        "class_name": class_name,
        "aliases": aliases,
        "attributes": attributes,
    }


def _relation(relation_id: str, predicate: str, subject: str, objects: list[str], depends_on: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": relation_id,
        "predicate": predicate,
        "subject_entity": subject,
        "object_entities": objects,
        "depends_on": depends_on or [],
    }


def _relation_parts(text: str) -> tuple[str, list[tuple[str, str, list[str], list[str]]]]:
    """Return target phrase and relation tuples for count/reference grammar."""
    between = re.fullmatch(
        r"(.+?)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    nested = re.fullmatch(
        r"(.+?)\s+(near|closest to|farthest from)\s+(?:the\s+)?(.+?)\s+"
        r"(on|in|above|below|near)\s+(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    simple = re.fullmatch(
        r"(.+?)\s+(closest to|farthest from|on|in|above|below|near)\s+(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    if between:
        return between.group(1), [("between", "target_0", [between.group(2), between.group(3)], [])]
    if nested:
        target, outer, anchor, inner, inner_anchor = nested.groups()
        return target, [
            (inner, anchor, [inner_anchor], []),
            (outer, "target_0", [anchor], ["rel_0"]),
        ]
    if simple:
        target, predicate, anchor = simple.groups()
        return target, [(predicate, "target_0", [anchor], [])]
    return text, []


def _predicate(value: str) -> str:
    return {"closest to": "closest", "farthest from": "farthest"}.get(value.lower(), value.lower())


def _query_plan(question: str, task_type: str) -> dict[str, Any]:
    text = question.strip().rstrip(".?")
    if task_type == "numerical":
        body = re.sub(r"^(how many|count(?: the number of)?)\s+", "", text, flags=re.IGNORECASE)
        # Existential count questions carry no spatial qualifier.  Remove the
        # full copular tail before the generic relational "are on/in/..."
        # rewrite, otherwise "TVs are there" becomes the bogus class
        # "tvs there".
        body = re.sub(r"\s+are\s+there$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+are\s+", " ", body, count=1, flags=re.IGNORECASE)
    else:
        body = re.sub(r"^(find|locate|identify|select)\s+", "", text, flags=re.IGNORECASE)
    target_phrase, parts = _relation_parts(body)
    entities = [_entity("target_0", "target", target_phrase)]
    relations = []
    anchors_by_phrase: dict[str, str] = {}
    for index, (predicate, subject, anchor_phrases, depends) in enumerate(parts):
        subject_id = subject
        if subject != "target_0":
            subject_key = " ".join(subject.lower().split())
            if subject_key not in anchors_by_phrase:
                anchor_id = f"anchor_{len(anchors_by_phrase)}"
                anchors_by_phrase[subject_key] = anchor_id
                entities.append(_entity(anchor_id, "anchor", subject))
            subject_id = anchors_by_phrase[subject_key]
        object_ids = []
        for phrase in anchor_phrases:
            key = " ".join(phrase.lower().split())
            if key not in anchors_by_phrase:
                anchor_id = f"anchor_{len(anchors_by_phrase)}"
                anchors_by_phrase[key] = anchor_id
                entities.append(_entity(anchor_id, "anchor", phrase))
            object_ids.append(anchors_by_phrase[key])
        relations.append(_relation(f"rel_{index}", _predicate(predicate), subject_id, object_ids, depends))
    return _finish(question, task_type, entities, relations, [])


def _instruction_plan(question: str) -> dict[str, Any]:
    text = question.strip().rstrip(".?")
    clauses = re.split(r"\s*(?:,?\s+then\s+|,?\s+and\s+)(?=(?:go|take|stop)\b)", text, flags=re.IGNORECASE)
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    entity_by_phrase: dict[str, str] = {}

    def ensure(phrase: str, role: str = "anchor") -> str:
        key = " ".join(re.sub(r"^(?:the|a|an)\s+", "", phrase.lower()).split())
        if key not in entity_by_phrase:
            entity_id = f"entity_{len(entity_by_phrase)}"
            entity_by_phrase[key] = entity_id
            entities.append(_entity(entity_id, role, phrase))
        elif role == "target":
            for item in entities:
                if item["id"] == entity_by_phrase[key]:
                    item["role"] = "target"
        return entity_by_phrase[key]

    for raw_clause in clauses:
        clause = re.sub(r"^(first,?\s*)", "", raw_clause.strip(), flags=re.IGNORECASE)
        match = re.match(r"(?:take the path|go)\s+near\s+(?:the\s+)?(.+)$", clause, flags=re.IGNORECASE)
        stop = re.match(r"stop at\s+(?:the\s+)?(.+)$", clause, flags=re.IGNORECASE)
        go = re.match(r"go to\s+(?:the\s+)?(.+)$", clause, flags=re.IGNORECASE)
        action = "pass_near" if clause.lower().startswith("take the path") else "go_near"
        phrase = match.group(1) if match else stop.group(1) if stop else go.group(1) if go else ""
        if not phrase:
            raise ValueError(f"unsupported_instruction_clause:{clause}")
        target_phrase, parts = _relation_parts(phrase)
        target_id = ensure(target_phrase, "target")
        relation_ids = []
        for predicate, _subject, anchor_phrases, depends in parts:
            object_ids = [ensure(anchor) for anchor in anchor_phrases]
            relation_id = f"rel_{len(relations)}"
            translated_depends = [f"rel_{len(relations) - len(parts) + int(value.split('_')[-1])}" for value in depends]
            relations.append(_relation(relation_id, _predicate(predicate), target_id, object_ids, translated_depends))
            relation_ids.append(relation_id)
        if stop:
            action = "stop_at"
        elif go:
            action = "go_to"
        steps.append({
            "order": len(steps),
            "action": action,
            "target_entity": target_id,
            "relation_ids": relation_ids,
            "terminal": False,
        })
    if not steps:
        raise ValueError("instruction_has_no_steps")
    steps[-1]["terminal"] = True
    target_id = steps[-1]["target_entity"]
    return _finish(question, "instruction_following", entities, relations, steps, target_id)


def deterministic_compile(question: str) -> dict[str, Any]:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("empty_question")
    lower = normalized.lower()
    if re.match(r"^(how many|count)\b", lower):
        return _query_plan(normalized, "numerical")
    if re.match(r"^(find|locate|identify|select)\b", lower):
        return _query_plan(normalized, "object_reference")
    return _instruction_plan(normalized)


def _finish(question: str, task_type: str, entities: list[dict[str, Any]], relations: list[dict[str, Any]], steps: list[dict[str, Any]], target_entity: str = "target_0") -> dict[str, Any]:
    return {
        "schema_version": "task_ir_v1",
        "task_type": task_type,
        "original_question": " ".join(question.strip().split()),
        "entities": entities,
        "relations": relations,
        "ordered_subgoals": steps,
        "target_entity": target_entity,
        "required_classes": list(dict.fromkeys(item["class_name"] for item in entities)),
        "output_contract": OUTPUT_BY_TASK[task_type],
        "parser_confidence": 0.72,
    }


def validate_task_ir(payload: Mapping[str, Any], question: str) -> dict[str, Any]:
    required = {
        "schema_version", "task_type", "original_question", "entities", "relations",
        "ordered_subgoals", "target_entity", "required_classes", "output_contract",
        "parser_confidence",
    }
    if set(payload) != required or payload["schema_version"] != "task_ir_v1":
        raise ValueError("task_ir_top_level_schema_invalid")
    task_type = str(payload["task_type"])
    if task_type not in TASK_TYPES or payload["output_contract"] != OUTPUT_BY_TASK[task_type]:
        raise ValueError("task_ir_task_output_contract_invalid")
    if " ".join(str(payload["original_question"]).split()) != " ".join(question.strip().split()):
        raise ValueError("task_ir_question_mismatch")
    entities = list(payload["entities"])
    if not entities:
        raise ValueError("task_ir_entities_empty")
    entity_ids = set()
    for entity in entities:
        if set(entity) != {"id", "role", "class_name", "aliases", "attributes"}:
            raise ValueError("task_ir_entity_schema_invalid")
        if entity["role"] not in {"target", "anchor"} or not str(entity["class_name"]).strip():
            raise ValueError("task_ir_entity_invalid")
        entity_ids.add(str(entity["id"]))
    if payload["target_entity"] not in entity_ids:
        raise ValueError("task_ir_target_missing")
    relation_ids = set()
    for relation in payload["relations"]:
        if set(relation) != {"id", "predicate", "subject_entity", "object_entities", "depends_on"}:
            raise ValueError("task_ir_relation_schema_invalid")
        if relation["predicate"] not in PREDICATES or relation["subject_entity"] not in entity_ids:
            raise ValueError("task_ir_relation_invalid")
        if not relation["object_entities"] or any(item not in entity_ids for item in relation["object_entities"]):
            raise ValueError("task_ir_relation_object_invalid")
        relation_ids.add(relation["id"])
    if any(
        dependency not in relation_ids
        for relation in payload["relations"]
        for dependency in relation["depends_on"]
    ):
        raise ValueError("task_ir_relation_dependency_invalid")
    steps = list(payload["ordered_subgoals"])
    if task_type == "instruction_following" and not steps:
        raise ValueError("task_ir_instruction_steps_missing")
    if task_type != "instruction_following" and steps:
        raise ValueError("task_ir_query_has_steps")
    for index, step in enumerate(steps):
        if set(step) != {"order", "action", "target_entity", "relation_ids", "terminal"}:
            raise ValueError("task_ir_step_schema_invalid")
        if step["order"] != index or step["action"] not in ACTIONS or step["target_entity"] not in entity_ids:
            raise ValueError("task_ir_step_invalid")
        if any(item not in relation_ids for item in step["relation_ids"]):
            raise ValueError("task_ir_step_relation_invalid")
    if steps and steps[-1]["terminal"] is not True:
        raise ValueError("task_ir_terminal_step_missing")
    confidence = float(payload["parser_confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("task_ir_confidence_invalid")
    required_classes = list(dict.fromkeys(str(item).strip() for item in payload["required_classes"] if str(item).strip()))
    entity_classes = {str(item["class_name"]) for item in entities}
    if not entity_classes.issubset(set(required_classes)):
        raise ValueError("task_ir_required_classes_incomplete")
    result = dict(payload)
    result["required_classes"] = required_classes
    return result


def build_semantic_grounding_plan(task_ir: Mapping[str, Any]) -> dict[str, Any]:
    """Derive soft detector/VLM context from already validated TaskIR.

    DeepSeek remains responsible for parsing the words in the question.  This
    deterministic post-pass adds reusable visual aliases, nearby supporting
    concepts and hard negatives without changing the requested entities or
    relations.
    """
    entities = {str(item["id"]): dict(item) for item in task_ir["entities"]}
    plan_entities = []
    detector_classes: dict[str, list[str]] = {}

    for entity_id, entity in entities.items():
        class_name = str(entity["class_name"])
        hint = GROUNDING_HINTS.get(class_name, {})
        detector_aliases = list(dict.fromkeys([
            *entity.get("aliases", []),
            *hint.get("aliases", []),
        ]))
        detector_classes[class_name] = detector_aliases

        supporting_context = []
        seen_context = set()

        # Explicit entities from the question are useful context for each
        # other, while the relation remains exactly the compiler's relation.
        for relation in task_ir["relations"]:
            subject_id = str(relation["subject_entity"])
            object_ids = [str(item) for item in relation["object_entities"]]
            if entity_id == subject_id:
                related_ids = object_ids
                candidate_role = "subject"
            elif entity_id in object_ids:
                related_ids = [subject_id]
                candidate_role = "object"
            else:
                continue
            for related_id in related_ids:
                related = entities[related_id]
                related_class = str(related["class_name"])
                key = (related_class, str(relation["predicate"]), candidate_role)
                if key in seen_context:
                    continue
                seen_context.add(key)
                supporting_context.append({
                    "class_name": related_class,
                    "aliases": list(related.get("aliases", [])),
                    "relations": [str(relation["predicate"])],
                    "source": "question",
                    "candidate_role": candidate_role,
                })

        for context_class, context_aliases, relations in hint.get("supporting_context", []):
            key = (context_class, tuple(relations), "context_hint")
            if key in seen_context:
                continue
            seen_context.add(key)
            supporting_context.append({
                "class_name": context_class,
                "aliases": list(context_aliases),
                "relations": list(relations),
                "source": "indoor_context_hint",
                "candidate_role": "context",
            })
            detector_classes.setdefault(context_class, list(context_aliases))

        plan_entities.append({
            "entity_id": entity_id,
            "primary_class": class_name,
            "detector_aliases": detector_aliases,
            "supporting_context": supporting_context,
            "hard_negatives": list(hint.get("confusers", [])),
        })

    return {
        "schema_version": "semantic_grounding_plan_v1",
        "policy": "soft_context_only",
        "entities": plan_entities,
        "detector_classes": detector_classes,
    }


def _attach_grounding_plan(task_ir: dict[str, Any]) -> dict[str, Any]:
    result = dict(task_ir)
    result["grounding_plan"] = build_semantic_grounding_plan(result)
    return result


def _canonicalize_remote(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    normalized_entities = []
    for raw_entity in list(payload.get("entities", [])):
        entity = dict(raw_entity)
        original = str(entity.get("class_name", "")).replace("_", " ").strip()
        class_name, known_aliases, normalized_attributes = _normalize_phrase(original)
        supplied_aliases = [str(item).strip() for item in entity.get("aliases", []) if str(item).strip()]
        aliases = list(dict.fromkeys([*known_aliases, *supplied_aliases]))
        if original.lower() != class_name.lower():
            aliases.append(original)
        entity["class_name"] = class_name
        entity["aliases"] = list(dict.fromkeys(aliases))
        attributes = dict(entity.get("attributes", {}))
        attributes.update(normalized_attributes)
        entity["attributes"] = attributes
        normalized_entities.append(entity)
    result["entities"] = normalized_entities
    result["required_classes"] = list(
        dict.fromkeys(item["class_name"] for item in normalized_entities)
    )
    return result


def _deepseek_compile(question: str, config: Mapping[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get(str(config.get("api_key_env", "DEEPSEEK_API_KEY")), "").strip()
    if not api_key:
        raise RuntimeError("deepseek_credential_unavailable")
    endpoint = str(config.get("base_url", "https://api.deepseek.com")).rstrip("/") + "/chat/completions"
    body = {
        "model": str(config.get("model", "deepseek-v4-flash")),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": int(config.get("max_tokens", 4096)),
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(int(config.get("max_retries", 1)) + 1):
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=float(config.get("timeout_seconds", 30))) as response:
                provider_payload = json.loads(response.read().decode("utf-8"))
            content = str(provider_payload["choices"][0]["message"]["content"]).strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("deepseek_task_ir_not_object")
            return payload
        except (KeyError, ValueError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < int(config.get("max_retries", 1)):
                time.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


def compile_task(question: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compile with DeepSeek first, returning explicit diagnostics and fallback."""
    settings = dict(config or {})
    local = None
    local_error = ""
    try:
        local = validate_task_ir(deterministic_compile(question), question)
    except ValueError as exc:
        local_error = f"{type(exc).__name__}:{str(exc)[:160]}"
    diagnostics = {
        "primary_backend": "deepseek",
        "backend": "deterministic_v1",
        "deepseek_called": False,
        "deepseek_accepted": False,
        "local_parser_ready": local is not None,
        "local_parser_error": local_error,
        "fallback_reason": "deepseek_disabled",
    }
    if settings.get("enabled", True):
        try:
            diagnostics["deepseek_called"] = True
            remote = validate_task_ir(
                _canonicalize_remote(_deepseek_compile(question, settings)), question
            )
            # Task type and ROS output authority are deterministic invariants.
            if local is not None and (
                remote["task_type"] != local["task_type"]
                or remote["output_contract"] != local["output_contract"]
            ):
                raise ValueError("deepseek_local_task_contract_disagreement")
            if local is not None and set(remote["required_classes"]) != set(local["required_classes"]):
                raise ValueError("deepseek_local_entity_set_disagreement")
            remote["compiler_diagnostics"] = {
                **diagnostics,
                "backend": "deepseek",
                "deepseek_accepted": True,
                "fallback_reason": "",
            }
            return _attach_grounding_plan(remote)
        except (KeyError, ValueError, RuntimeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            diagnostics["fallback_reason"] = f"{type(exc).__name__}:{str(exc)[:160]}"
    if local is None:
        raise ValueError(
            "task_compilation_failed:"
            f"local={local_error or 'unavailable'};"
            f"deepseek={diagnostics['fallback_reason']}"
        )
    local["compiler_diagnostics"] = diagnostics
    return _attach_grounding_plan(local)
