"""Compile a complete challenge question into a small, validated TaskIR.

DeepSeek is the runtime semantic compiler.  The deterministic compiler is kept
only as a development utility for explicit local contract checks.
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
ACTIONS = {
    "go_near", "go_to", "pass_near", "pass_between", "avoid_near", "stop_at"
}
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
        "visual_definition": (
            "a physical photo or recognizable picture frame displaying an image, "
            "including small tabletop or furniture-top framed photos; exclude "
            "decorative paintings, wall art, posters, and abstract artwork"
        ),
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
        "visual_definition": (
            "a television cabinet, TV stand, media console, or visually similar "
            "cabinet/stand that may support a television; include the lower "
            "furniture stand even when the television is partly occluded"
        ),
        "supporting_context": [("television", ["tv"], ["on", "above", "near"])],
        "confusers": [],
    },
    "bookshelf": {
        "aliases": ["bookcase", "book shelf"],
        "supporting_context": [("book", ["books"], ["on", "in"])],
        "confusers": ["storage cabinet", "wall shelf"],
    },
    "pillow": {
        "aliases": ["cushion"],
        "visual_definition": (
            "a separate stuffed pillow or cushion with a volumetric soft body "
            "and fabric cover, used for head support or decoration; exclude "
            "flat folded towels or blankets and upholstery attached to furniture"
        ),
        "supporting_context": [],
        "confusers": [
            "towel",
            "folded blanket",
            "sofa armrest",
            "upholstered furniture backrest",
        ],
    },
    "sofa": {
        "aliases": ["couch", "settee"],
        "supporting_context": [("pillow", ["cushion"], ["on"])],
        "confusers": ["armchair", "bench", "bed"],
    },
    "window": {
        "aliases": [
            "window with grilles",
            "glazed window",
            "lattice window",
            "screened window",
            "window with blinds",
            "shuttered window",
            "closed window",
        ],
        "visual_definition": (
            "an architectural window set into a wall, including windows covered "
            "by grilles or lattice, blinds, shutters, screens, glazing, or a "
            "closed panel; exclude any human-passable doorway, archway, or "
            "walk-through opening"
        ),
        "supporting_context": [],
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
        "visual_definition": (
            "full-size sleeping furniture with a mattress or clearly padded "
            "sleeping platform large enough for a person; exclude narrow "
            "upholstered seating benches with exposed legs and no mattress"
        ),
        "supporting_context": [("pillow", ["cushion"], ["on"])],
        "confusers": ["sofa", "daybed", "bench"],
    },
    "book": {
        "aliases": ["books", "bound volume"],
        "visual_definition": (
            "a physical book or visible group of books, recognizable from a "
            "bound cover, page block, upright spine, or stacked volume; a "
            "partly occluded spine remains book evidence"
        ),
        "supporting_context": [],
        "confusers": [],
    },
    "potted plant": {
        "aliases": ["houseplant", "plant in pot", "plant pot", "potted plant"],
        "visual_definition": (
            "a physical plant growing from a container or pot, including a "
            "partly occluded potted plant whose leaves, stems, and container "
            "edge jointly identify the same instance"
        ),
        "supporting_context": [],
        "confusers": ["vase"],
    },
}

SYSTEM_PROMPT = """You are a deterministic task compiler for a competition robot.
Compile only the supplied question and return one JSON object with no prose.
Never output coordinates, waypoints, hidden reasoning, commands, URLs, or tools.

The exact top-level keys are: schema_version, task_type, original_question,
entities, relations, ordered_trajectory_constraints, target_entity, required_classes,
output_contract, parser_confidence, visual_grounding. schema_version is
"task_ir_v2". task_type is
numerical, object_reference, or instruction_following. entities contain exactly
id, role, class_name, aliases, attributes. role is target or anchor. attributes
is an object. relations contain exactly id, predicate, subject_entity,
object_entities, depends_on. Allowed predicates: near, between, closest,
farthest, above, below, on, in. ordered_trajectory_constraints contain exactly order, action,
target_entity, relation_ids, terminal. Allowed actions: go_near, go_to,
pass_near, pass_between, avoid_near, stop_at. Orders are contiguous from zero.
For numerical and object_reference tasks ordered_trajectory_constraints MUST be an empty list.
Only instruction_following tasks may contain ordered_trajectory_constraints.
For go_near and pass_near, set the landmark itself as target_entity and leave
relation_ids empty; the action already encodes the trajectory relation. Never create
a relation whose subject is robot or robot_path. Object qualifiers such as
farthest(pillow,lamp) belong in relations and are referenced by the go_to step.
Normalize the word under to the allowed predicate below. For example, "pillows
on the sofa under the pictures" means below(sofa, pictures) and on(pillows,
sofa); under the pictures selects the sofa and does not modify pillows.

Treat relations as an executable dependency graph, not as an unordered bag of
words. Parse noun-phrase heads and modifier attachment before emitting any
entity or relation. A spatial phrase inside an anchor noun phrase qualifies
that anchor; it does not automatically modify the outer target. Resolve every
qualified anchor first, then use that bound anchor in the relation that
identifies the target. Put only immediate prerequisite relation IDs in
depends_on and list relations in dependency order. For a generic phrase of the
form "X REL_A [Y REL_B Z]", emit REL_B(Y,Z) first and REL_A(X,Y) depending on
it. Apply the same rule recursively to deeper nesting, relative clauses, and
pronouns whose antecedent is a visible noun phrase. Do not flatten relational
modifiers into class_name, aliases, or attributes.

Preserve predicate direction and parameter roles exactly; never swap the
subject and object to make a relation easier to evaluate. BETWEEN has one
subject and exactly two separately named boundary objects in their grammatical
order; it is one ternary relation, never two NEAR relations. CLOSEST/FARTHEST
rank instances of the grammatical subject relative to the named object. A
plural noun changes candidate cardinality, not relation direction or modifier
attachment. The target entity is the object requested by the main question,
not the closest noun or the deepest nested anchor.

Preserve the grammatical target, every anchor, nested relations, comparisons,
and instruction order. Do not select the first noun. Required classes contain
the target and every anchor class. output_contract must be /numerical_response,
/selected_object_marker, or /way_point_with_heading according to task_type.

visual_grounding is an array with exactly one object per entity, each containing
exactly entity_id, visual_definition, and confusers. confusers is a JSON array
of ordinary indoor object classes that are visually plausible but categorically
distinct from the requested class. Describe the broad visible intrinsic
features that distinguish that physical class from its closest ordinary visual
confusers. Use shape, parts, material, mounting, and physical extent; do not use
the requested relation, likely room location, coordinates, or answer priors as
class evidence. Cover legitimate common variants and partial views instead of
defining one stereotyped appearance. These are soft visual semantics for a VLM,
not hard accept/reject rules.

Examples:
How many mugs are on the dining table? => target mug; anchor dining table;
relation on(mug, dining table); numerical.
Find the vase between the cabinet and the stool. => target vase; anchors cabinet
and stool; relation between(vase, cabinet, stool); object_reference.
Find the lamp near the magazines on the shelf. => r1 on(magazines,shelf), then
r2 near(lamp,magazines) with depends_on [r1]; object_reference. The class words
are illustrative; apply the same dependency rule to any nouns and predicates.
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
    nested_with = re.fullmatch(
        r"(.+?)\s+(near|closest to|farthest from|furthest from|on|in|above|below)\s+"
        r"(?:the\s+)?(.+?)\s+with\s+(?:(?:a|an|the)\s+)?(.+?)\s+"
        r"(on|in|above|below|under|near)\s+(?:it|them)",
        text, flags=re.IGNORECASE,
    )
    inverse_with = re.fullmatch(
        r"(.+?)\s+with\s+(?:(?:a|an|the)\s+)?(.+?)\s+"
        r"(on|in|above|below|under|near)\s+(?:it|them)",
        text, flags=re.IGNORECASE,
    )
    between = re.fullmatch(
        r"(.+?)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    nested = re.fullmatch(
        r"(.+?)\s+(near|closest to|farthest from|furthest from)\s+(?:the\s+)?(.+?)\s+"
        r"(on|in|above|below|under|near)\s+(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    nested_on = re.fullmatch(
        r"(.+?)\s+(on|in|above|below|near)\s+(?:the\s+)?(.+?)\s+"
        r"(closest to|farthest from|furthest from|on|in|above|below|under|near)\s+"
        r"(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    simple = re.fullmatch(
        r"(.+?)\s+(closest to|farthest from|furthest from|on|in|above|below|near)\s+(?:the\s+)?(.+)",
        text, flags=re.IGNORECASE,
    )
    if nested_with:
        target, outer, anchor, inner_subject, inner = nested_with.groups()
        return target, [
            (inner, inner_subject, [anchor], []),
            (outer, "target_0", [anchor], ["rel_0"]),
        ]
    if inverse_with:
        target, related_subject, predicate = inverse_with.groups()
        return target, [(predicate, related_subject, ["target_0"], [])]
    if between:
        return between.group(1), [("between", "target_0", [between.group(2), between.group(3)], [])]
    if nested:
        target, outer, anchor, inner, inner_anchor = nested.groups()
        return target, [
            (inner, anchor, [inner_anchor], []),
            (outer, "target_0", [anchor], ["rel_0"]),
        ]
    if nested_on:
        target, outer, anchor, inner, inner_anchor = nested_on.groups()
        return target, [
            (inner, anchor, [inner_anchor], []),
            (outer, "target_0", [anchor], ["rel_0"]),
        ]
    if simple:
        target, predicate, anchor = simple.groups()
        return target, [(predicate, "target_0", [anchor], [])]
    return text, []


def _predicate(value: str) -> str:
    return {
        "closest to": "closest",
        "farthest from": "farthest",
        "furthest from": "farthest",  # Normalize furthest to farthest
        "furthest": "farthest",
        "under": "below",
    }.get(value.lower(), value.lower())


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
            if phrase == "target_0":
                object_ids.append("target_0")
                continue
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
    text = re.sub(
        r"\s+and\s+then\s+to\s+",
        " then go to ",
        text,
        flags=re.IGNORECASE,
    )
    clauses = re.split(r"\s*(?:,?\s+then\s+|,?\s+and\s+)(?=(?:go|take|stop|avoid)\b)", text, flags=re.IGNORECASE)
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    entity_by_phrase: dict[str, str] = {}

    def next_entity_id() -> str:
        used_ids = {str(item.get("id", "")) for item in entities}
        index = 0
        while f"entity_{index}" in used_ids:
            index += 1
        return f"entity_{index}"

    def ensure(phrase: str, role: str = "anchor") -> str:
        key = " ".join(re.sub(r"^(?:the|a|an)\s+", "", phrase.lower()).split())
        if key not in entity_by_phrase:
            entity_id = next_entity_id()
            entity_by_phrase[key] = entity_id
            entities.append(_entity(entity_id, role, phrase))
        elif role == "target":
            for item in entities:
                if item["id"] == entity_by_phrase[key]:
                    item["role"] = "target"
        return entity_by_phrase[key]

    for raw_clause in clauses:
        clause = re.sub(r"^(first,?\s*)", "", raw_clause.strip(), flags=re.IGNORECASE)

        # Match "take the path between X and Y" or "go between X and Y"
        # Handle "between the two X" -> "between X and X"
        path_between = re.match(
            r"(?:take the path|go)\s+between\s+(?:the\s+)?(?:two\s+)?(.+?)\s+and\s+(?:the\s+)?(?:two\s+)?(.+)$",
            clause, flags=re.IGNORECASE
        )
        # Special case: "between the two X" means "between X and X"
        path_between_two = re.match(
            r"(?:take the path|go)\s+between\s+the\s+two\s+(.+)$",
            clause, flags=re.IGNORECASE
        )

        # Match "go near" or "take the path near"
        match = re.match(r"(?:take the path|go)\s+near\s+(?:the\s+)?(.+)$", clause, flags=re.IGNORECASE)
        stop = re.match(r"stop (?:at|by)\s+(?:the\s+)?(.+)$", clause, flags=re.IGNORECASE)
        go = re.match(r"go to\s+(?:the\s+)?(.+)$", clause, flags=re.IGNORECASE)
        avoid = re.match(
            r"avoid\s+(?:the\s+)?path\s+(?:near|between)\s+(?:the\s+)?(.+)$",
            clause,
            flags=re.IGNORECASE,
        )

        action = "pass_near" if clause.lower().startswith("take the path") else "go_near"
        phrase = ""

        # Handle "take the path between X and Y" or "go between X and Y"
        if path_between:
            boundary1, boundary2 = path_between.groups()
            # Create a synthetic phrase for "between X and Y"
            phrase = f"waypoint between {boundary1} and {boundary2}"
            action = "pass_between"
        elif path_between_two:
            # Handle "between the two X" -> treat as "between X and X"
            entity_type = path_between_two.group(1)
            phrase = f"waypoint between {entity_type} and {entity_type}"
            action = "pass_between"
        elif match:
            phrase = match.group(1)
        elif stop:
            phrase = stop.group(1)
        elif go:
            phrase = go.group(1)
        elif avoid:
            phrase = avoid.group(1)

        if not phrase:
            raise ValueError(f"unsupported_instruction_clause:{clause}")
        # Special handling for pass_between
        if action == "pass_between" and (path_between or path_between_two):
            # Extract the two boundary entities
            if path_between:
                boundary1, boundary2 = path_between.groups()
            else:  # path_between_two
                entity_type = path_between_two.group(1)
                boundary1 = boundary2 = entity_type
            # Create a virtual waypoint entity for "between X and Y"
            waypoint_id = f"waypoint_{len(entities)}"
            entities.append({
                "id": waypoint_id,
                "role": "anchor",
                "class_name": "waypoint",
                "aliases": [],
                "attributes": {},
            })
            # Create entities for the boundaries
            anchor1_id = ensure(boundary1)
            if path_between_two:
                # "between the two X" denotes two distinct instances from
                # the same class.  Keep two entity slots so the resolver can
                # bind a physical pair instead of collapsing the relation to
                # one anchor and failing trajectory geometry compilation.
                anchor2_id = next_entity_id()
                entities.append(_entity(anchor2_id, "anchor", boundary2))
            else:
                anchor2_id = ensure(boundary2)
            # Create between relation
            relation_id = f"rel_{len(relations)}"
            relations.append(_relation(relation_id, "between", waypoint_id, [anchor1_id, anchor2_id], []))
            relation_ids = [relation_id]
            target_id = waypoint_id
        else:
            target_phrase, parts = _relation_parts(phrase)
            target_id = ensure(target_phrase, "target")
            relation_ids = []
            batch_start = len(relations)  # Record where this batch starts
            for predicate, subject, anchor_phrases, depends in parts:
                subject_id = (
                    target_id if subject == "target_0" else ensure(subject)
                )
                object_ids = [
                    target_id if anchor == "target_0" else ensure(anchor)
                    for anchor in anchor_phrases
                ]
                relation_id = f"rel_{len(relations)}"
                # Translate depends: "rel_0" means "first relation of this batch"
                translated_depends = [f"rel_{batch_start + int(value.split('_')[-1])}" for value in depends]
                relations.append(_relation(relation_id, _predicate(predicate), subject_id, object_ids, translated_depends))
                relation_ids.append(relation_id)

        if stop:
            action = "stop_at"
        elif go:
            action = "go_to"
        elif avoid:
            action = "avoid_near"
        steps.append({
            "order": len(steps),
            "action": action,
            "target_entity": target_id,
            "relation_ids": relation_ids,
            "terminal": False,
        })
    if not steps:
        raise ValueError("instruction_has_no_steps")
    terminal_step = next(
        (value for value in reversed(steps) if value["action"] != "avoid_near"),
        None,
    )
    if terminal_step is None:
        raise ValueError("instruction_terminal_constraint_missing")
    terminal_step["terminal"] = True
    target_id = terminal_step["target_entity"]
    return _finish(question, "instruction_following", entities, relations, steps, target_id)


def deterministic_compile(question: str) -> dict[str, Any]:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("empty_question")
    lower = normalized.lower()

    # Check for numerical/count
    if re.match(r"^(how many|count)\b", lower):
        return _query_plan(normalized, "numerical")

    # Check for object_reference: "Find ..." or "The X ..."
    if re.match(r"^(find|locate|identify|select)\b", lower):
        return _query_plan(normalized, "object_reference")

    # Handle "The X ..." as object_reference (descriptive phrase)
    if re.match(r"^the\s+\w", lower):
        # Convert "The X ..." to "Find the X ..."
        return _query_plan("Find " + normalized, "object_reference")

    # Otherwise, instruction_following
    return _instruction_plan(normalized)


def _finish(question: str, task_type: str, entities: list[dict[str, Any]], relations: list[dict[str, Any]], steps: list[dict[str, Any]], target_entity: str = "target_0") -> dict[str, Any]:
    return {
        "schema_version": "task_ir_v2",
        "task_type": task_type,
        "original_question": " ".join(question.strip().split()),
        "entities": entities,
        "relations": relations,
        "ordered_trajectory_constraints": steps,
        "target_entity": target_entity,
        "required_classes": list(dict.fromkeys(item["class_name"] for item in entities)),
        "output_contract": OUTPUT_BY_TASK[task_type],
        "parser_confidence": 0.72,
    }


def validate_task_ir(payload: Mapping[str, Any], question: str) -> dict[str, Any]:
    required = {
        "schema_version", "task_type", "original_question", "entities", "relations",
        "ordered_trajectory_constraints", "target_entity", "required_classes", "output_contract",
        "parser_confidence",
    }
    if set(payload) != required or payload["schema_version"] != "task_ir_v2":
        raise ValueError("task_ir_top_level_schema_invalid")
    task_type = str(payload["task_type"])
    if task_type not in TASK_TYPES or payload["output_contract"] != OUTPUT_BY_TASK[task_type]:
        raise ValueError("task_ir_task_output_contract_invalid")

    # Normalize both questions for comparison (handle "Find " prefix addition)
    normalized_payload = " ".join(str(payload["original_question"]).split())
    normalized_question = " ".join(question.strip().split())

    # Allow match if payload is "Find " + question (for "The X ..." pattern)
    if not (normalized_payload == normalized_question or
            normalized_payload == "Find " + normalized_question):
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
    steps = list(payload["ordered_trajectory_constraints"])
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
    if steps and sum(bool(value["terminal"]) for value in steps) != 1:
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


def build_semantic_grounding_plan(
    task_ir: Mapping[str, Any],
    visual_definitions: Mapping[str, str] | None = None,
    visual_confusers: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Derive soft detector/VLM context from already validated TaskIR.

    DeepSeek remains responsible for parsing the words in the question.  This
    deterministic post-pass adds reusable visual aliases, nearby supporting
    concepts and hard negatives without changing the requested entities or
    relations.
    """
    entities = {str(item["id"]): dict(item) for item in task_ir["entities"]}
    deepseek_visual = {
        str(key): str(value).strip()
        for key, value in (visual_definitions or {}).items()
        if str(value).strip()
    }
    deepseek_confusers = {
        str(key): [
            str(item).strip() for item in values if str(item).strip()
        ]
        for key, values in (visual_confusers or {}).items()
        if isinstance(values, list)
    }
    candidate_generation_only = str(task_ir.get("task_type", "")) == "numerical"
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

        # Numerical grounding must recall the base-class domain without
        # applying relation or attribute constraints.  Non-numerical tasks
        # retain the established soft context behavior.
        for relation in (() if candidate_generation_only else task_ir["relations"]):
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

        for context_class, context_aliases, relations in (
            () if candidate_generation_only else hint.get("supporting_context", [])
        ):
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

        semantic_negatives = []
        target_names = {
            class_name.lower(),
            *(str(value).lower() for value in detector_aliases),
        }
        for value in (
            *deepseek_confusers.get(entity_id, ()),
            *hint.get("confusers", ()),
        ):
            raw_value = str(value).strip()
            if not raw_value or raw_value.lower() in target_names:
                continue
            target_tokens = set(class_name.lower().split())
            negative_tokens = set(
                re.sub(r"[^a-z0-9 ]", " ", raw_value.lower()).split()
            )
            # A phrase containing the requested class is ordinarily a subtype
            # or variant (for example floor lamp), not a categorical negative.
            if target_tokens and target_tokens.issubset(negative_tokens):
                continue
            try:
                normalized_negative, _, _ = _normalize_phrase(raw_value)
            except ValueError:
                normalized_negative = raw_value.lower()
            if normalized_negative == class_name:
                continue
            if raw_value not in semantic_negatives:
                semantic_negatives.append(raw_value)

        plan_entities.append({
            "entity_id": entity_id,
            "primary_class": class_name,
            "detector_aliases": detector_aliases,
            "visual_definition": str(
                hint.get(
                    "visual_definition",
                    deepseek_visual.get(entity_id, class_name),
                )
            ),
            "supporting_context": supporting_context,
            "hard_negatives": semantic_negatives,
        })

    return {
        "schema_version": "semantic_grounding_plan_v1",
        "policy": (
            "candidate_class_recall_only"
            if candidate_generation_only
            else "soft_context_only"
        ),
        "relation_filters_applied": False if candidate_generation_only else None,
        "entities": plan_entities,
        "detector_classes": detector_classes,
    }


def _attach_grounding_plan(
    task_ir: dict[str, Any],
    visual_definitions: Mapping[str, str] | None = None,
    visual_confusers: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    result = dict(task_ir)
    result["grounding_plan"] = build_semantic_grounding_plan(
        result, visual_definitions, visual_confusers
    )
    result["trajectory_ir"] = build_trajectory_ir(result)
    if result["task_type"] == "numerical":
        from .count_query_graph import compile_count_query_graph

        result["count_query_graph"] = compile_count_query_graph(result)
    return result


def build_trajectory_ir(task_ir: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compile language actions into trajectory constraints."""
    if str(task_ir.get("task_type", "")) != "instruction_following":
        return None
    from .relation_registry import trajectory_action_spec

    relations = {
        str(value["id"]): value for value in task_ir.get("relations", ())
    }
    referents: dict[str, dict[str, Any]] = {}
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    for raw in sorted(
        task_ir.get("ordered_trajectory_constraints", ()), key=lambda value: int(value["order"])
    ):
        action = str(raw["action"])
        spec = trajectory_action_spec(action)
        relation_ids = [str(value) for value in raw.get("relation_ids", ())]
        target_entity = str(raw["target_entity"])
        relation_expressions = [
            {
                "relation_id": relation_id,
                "predicate": str(relations[relation_id]["predicate"]),
                "anchor_entities": [
                    str(value)
                    for value in relations[relation_id].get("object_entities", ())
                ],
            }
            for relation_id in relation_ids
            if relation_id in relations
        ]
        referents[target_entity] = {
            "entity_id": target_entity,
            "relation_expressions": relation_expressions,
        }
        constraint = {
            "order": int(raw["order"]),
            "constraint": (
                "TERMINATE_INSIDE"
                if bool(raw.get("terminal")) and spec.terminal_capable
                else spec.trajectory_constraint
            ),
            "region_kind": spec.region_kind,
            "target_entity": target_entity,
            "relation_ids": relation_ids,
            "action": action,
            "terminal": bool(raw.get("terminal")),
        }
        if spec.forbidden:
            negative.append(constraint)
        else:
            positive.append(constraint)
        if constraint["constraint"] == "TERMINATE_INSIDE":
            terminal = constraint
    if terminal is None and positive:
        positive[-1]["constraint"] = "TERMINATE_INSIDE"
        positive[-1]["region_kind"] = "STOP_REGION"
        positive[-1]["terminal"] = True
        terminal = positive[-1]
    return {
        "schema_version": "trajectory_ir_v1",
        "referents": referents,
        "ordered_path_constraints": positive,
        "forbidden_path_regions": negative,
        "terminal_constraint_order": (
            int(terminal["order"]) if terminal is not None else None
        ),
        "completion_authority": "actual_state_estimation_trajectory",
    }


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
    result["relations"] = [
        {
            **dict(relation),
            "predicate": _predicate(str(relation.get("predicate", ""))),
        }
        for relation in payload.get("relations", [])
    ]
    result["required_classes"] = list(
        dict.fromkeys(item["class_name"] for item in normalized_entities)
    )
    return result


def _remote_visual_grounding(
    payload: Mapping[str, Any],
    entity_ids: set[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Read usable DeepSeek visual semantics without making them a gate."""
    result: dict[str, str] = {}
    confusers: dict[str, list[str]] = {}
    raw_values = payload.get("visual_grounding", ())
    if not isinstance(raw_values, list):
        return result, confusers
    for raw in raw_values:
        if not isinstance(raw, Mapping):
            continue
        entity_id = str(raw.get("entity_id", "")).strip()
        definition = " ".join(
            str(raw.get("visual_definition", "")).strip().split()
        )
        if entity_id in entity_ids and definition:
            result[entity_id] = definition[:1200]
            raw_confusers = raw.get("confusers", ())
            if isinstance(raw_confusers, list):
                confusers[entity_id] = list(dict.fromkeys(
                    str(value).strip()[:200]
                    for value in raw_confusers
                    if str(value).strip()
                ))
    return result, confusers


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
    """Compile the canonical question with the deterministic TaskIR grammar."""
    del config
    try:
        local_payload = deterministic_compile(question)
        validated = validate_task_ir(local_payload, question)
    except Exception as exc:
        raise ValueError(
            f"task_compilation_failed:template_parser:{type(exc).__name__}:{str(exc)[:200]}"
        ) from exc
    validated["compiler_diagnostics"] = {
        "primary_backend": "deterministic_local_template",
        "backend": "deterministic_local_template",
        "local_parser_succeeded": True,
        "remote_compiler": "disabled",
    }
    return _attach_grounding_plan(validated, {}, {})
