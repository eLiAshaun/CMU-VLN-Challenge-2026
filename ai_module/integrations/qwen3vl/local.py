"""Pinned, lazy local Qwen3-VL grounding and candidate verification.

Candidate verification supports both the legacy single-candidate request and
the newer per-view batch request.  The batch request uses one source image
with numbered red boxes and returns independently validated rows, so a bad
row does not become evidence for another candidate.  Model weights are loaded
only when the first request arrives so the worker can be scheduled
independently from the ROS vision process.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
from typing import Mapping, Sequence

from .semantic_verifier import SemanticVerification
from .relation_types import (
    ObjectGroundedRelationResult,
)


def _normalized_bbox(value) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError("candidate_bbox_must_have_four_numbers") from None
    if not (
        0.0 <= x1 < x2 <= 1.0
        and 0.0 <= y1 < y2 <= 1.0
    ):
        raise ValueError("candidate_bbox_must_be_normalized_xyxy")
    return x1, y1, x2, y2


def _candidate_images_from_pil(source, normalized_bbox):
    """Return a close candidate view and the full scene with a box overlay."""
    from PIL import Image, ImageDraw

    if not isinstance(source, Image.Image):
        raise TypeError("qwen3vl_source_must_be_pil_image")
    source = source.convert("RGB")
    width, height = source.size
    x1, y1, x2, y2 = _normalized_bbox(normalized_bbox)
    box = (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )
    box_width = max(2, box[2] - box[0])
    box_height = max(2, box[3] - box[1])

    object_pad_x = max(3, int(round(0.25 * box_width)))
    object_pad_y = max(3, int(round(0.25 * box_height)))
    object_bounds = (
        max(0, box[0] - object_pad_x),
        max(0, box[1] - object_pad_y),
        min(width, box[2] + object_pad_x),
        min(height, box[3] + object_pad_y),
    )

    object_crop = source.crop(object_bounds)
    context_crop = source.copy()
    context_box = box
    draw = ImageDraw.Draw(context_crop)
    stroke = max(2, int(round(min(context_crop.size) / 120.0)))
    draw.rectangle(context_box, outline=(255, 32, 32), width=stroke)
    return object_crop, context_crop


def _numbered_candidate_image_from_pil(source, candidates):
    """Annotate one source image with independently numbered candidate boxes."""
    from PIL import Image, ImageDraw

    if not isinstance(source, Image.Image):
        raise TypeError("qwen3vl_source_must_be_pil_image")
    annotated = source.convert("RGB").copy()
    width, height = annotated.size
    draw = ImageDraw.Draw(annotated)
    stroke = max(2, int(round(min(annotated.size) / 240.0)))
    for index, candidate in enumerate(candidates, start=1):
        bbox = _normalized_bbox(candidate.get("candidate_bbox", ()))
        x1, y1, x2, y2 = (
            int(round(bbox[0] * width)),
            int(round(bbox[1] * height)),
            int(round(bbox[2] * width)),
            int(round(bbox[3] * height)),
        )
        draw.rectangle((x1, y1, x2, y2), outline=(255, 32, 32), width=stroke)
        label = str(index)
        text_box = draw.textbbox((0, 0), label)
        label_width = max(16, text_box[2] - text_box[0] + 8)
        label_height = max(16, text_box[3] - text_box[1] + 6)
        label_x = max(0, min(width - label_width, x1))
        label_y = max(0, y1 - label_height)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=(255, 32, 32),
        )
        draw.text(
            (label_x + 4, label_y + 2),
            label,
            fill=(255, 255, 255),
        )
    return annotated


def _candidate_images(image_path: str, normalized_bbox):
    from PIL import Image

    return _candidate_images_from_pil(
        Image.open(str(image_path)), normalized_bbox
    )


def _request_source_image(parameters):
    from PIL import Image

    image_path = str(parameters.get("image_path", "")).strip()
    if image_path:
        if not Path(image_path).is_file():
            raise ValueError("qwen3vl_image_path_missing")
        return Image.open(image_path).convert("RGB")

    shared = parameters.get("shared_memory")
    if not isinstance(shared, Mapping):
        raise ValueError("qwen3vl_image_input_missing")
    required = {"name", "shape", "dtype"}
    if set(shared) != required:
        raise ValueError("qwen3vl_shared_memory_schema_invalid")
    from .shared_memory_io import SharedArrayStore

    image = SharedArrayStore().get(
        str(shared["name"]),
        tuple(int(value) for value in shared["shape"]),
        str(shared["dtype"]),
        unlink=False,
    )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("qwen3vl_shared_image_must_be_rgb")
    if str(image.dtype) != "uint8":
        raise ValueError("qwen3vl_shared_image_must_be_uint8")
    return Image.fromarray(image, mode="RGB")


def _verification_prompt(
    *,
    operation: str,
    query_concept: str,
    anchor_concept: str,
    hard_negatives: Sequence[str],
) -> str:
    role = (
        "target object"
        if operation in {"verify_object", "verify_object_batch"}
        else "anchor object"
    )
    negatives = ", ".join(
        str(item).strip() for item in hard_negatives if str(item).strip()
    ) or "none supplied"
    return (
        "You are a calibrated visual candidate verifier. "
        "Image 1 is a tight detector-candidate crop. Image 2 is the complete "
        "source view; the exact candidate is enclosed by the red rectangle. "
        f"Verify only that one {role}. Query concept: {query_concept!r}. "
        f"Anchor concept: {anchor_concept!r}. Explicit hard negatives: "
        f"{negatives}. Treat the target and every hard negative as mutually "
        "exclusive visual classes. A semantically related hard negative is "
        "not the target. Use visible shape, mounting, text, material, and "
        "requested attributes; do not accept a candidate merely because the "
        "query words appear in the prompt. In confuser_probabilities, include "
        "the strongest visually plausible alternative class for the red-"
        "rectangle object, even when it was not supplied as a hard negative. "
        "Never put the requested target class or one of its synonyms in the "
        "confuser map. Use a short concrete class name as its key. Also include any visually "
        "supported supplied hard negative. Emit at most the three strongest "
        "alternatives; use an empty object only when no alternative class is "
        "visually plausible. "
        "Do not count outside the red rectangle and do not answer the scene "
        "question. Treat blur or insufficient pixels as uncertainty, not "
        "false. "
        "In this generic class-verification operation, target_probability "
        "always means P(the red-box object belongs to the query concept), "
        "regardless of whether that object later serves as a task target or a "
        "relation anchor. anchor_probability is unused here and must be 0.0; "
        "never move class probability into that field because of the object's "
        "role in the question. "
        "Return exactly one JSON object and no markdown with exactly these "
        'keys: {"target_probability":0.0,"anchor_probability":0.0,'
        '"confuser_probabilities":{},"rationale_tags":[]}. '
        "Probabilities must be calibrated numbers in [0,1]."
    )


def _batch_verification_prompt(
    candidates: Sequence[Mapping[str, object]],
    *,
    anchor_concept: str,
) -> str:
    """Build the strict, indexed output contract for one annotated view."""
    descriptions = []
    for index, candidate in enumerate(candidates, start=1):
        query = str(candidate.get("query_concept", "")).strip()
        visual_definition = str(
            candidate.get("visual_definition", query)
        ).strip() or query
        negatives = ", ".join(
            str(value).strip()
            for value in (candidate.get("hard_negatives", ()) or ())
            if str(value).strip()
        ) or "none supplied"
        descriptions.append(
            f"Candidate {index}: query concept={query!r}; "
            f"visual definition={visual_definition!r}; "
            f"hard negatives={negatives}."
        )
    return (
        "You are a calibrated visual candidate verifier. The image is one "
        "complete source view with red rectangles numbered 1 through "
        f"{len(candidates)}. Each numbered rectangle identifies exactly one "
        "detector candidate. Verify every candidate independently; never "
        "transfer evidence from one numbered rectangle to another. "
        + (f"Common anchor concept: {anchor_concept!r}. " if anchor_concept else "")
        + "\n"
        + "\n".join(descriptions)
        + "\nFor each candidate, use visible shape, mounting, text, material, "
        "and requested attributes. A red rectangle is only a detector "
        "proposal, not evidence that the query object fills it. First "
        "identify the dominant visible object or surface inside that exact "
        "rectangle. If the rectangle is mostly background, a room surface, "
        "or an unrelated object with only a clipped boundary fragment of the "
        "query concept, assign probability to that concrete dominant "
        "alternative instead of inferring the query from the boundary. "
        "Treat the query and hard negatives as mutually exclusive visual "
        "classes. In confuser_probabilities, include the strongest visually "
        "plausible alternative class for each numbered rectangle even when "
        "it was not supplied as a hard negative, plus any visually supported "
        "supplied hard negative. Never put the requested query class or one "
        "of its synonyms in the confuser map. Use an empty object only when "
        "no alternative class is visually plausible. Treat blur or insufficient "
        "pixels as uncertainty, not false. Do not answer a scene-wide "
        "question. For every row, target_probability always means P(the "
        "numbered-box object belongs to that row's query concept), regardless "
        "of whether the class is called a target or anchor in the question. "
        "anchor_probability is unused in this generic batch and must be 0.0; "
        "never route class probability into it based on task role. Return "
        "exactly one compact JSON object and no markdown. "
        "The object has the single key v. Its value is an array containing "
        f"exactly {len(candidates)} rows, one for each candidate, in any order. "
        "Every row is exactly [index,target_probability,anchor_probability," 
        "confuser_probabilities,rationale_tags]. Example: "
        "{\"v\":[[1,0.8,0.1,{},[\"visible_frame\"]]]}. Use at most two "
        "short snake_case rationale tags. Use the "
        "numbered candidate index from 1 through the final candidate index. "
        "Probabilities must be calibrated numbers in [0,1]."
    )


def _strict_semantic_batch_json(
    text: str,
    expected_count: int,
) -> list[dict[str, object]]:
    """Parse every independently valid indexed row without semantic repair.

    Missing rows carry no verdict.  The perception owner already represents
    an absent index as ``unavailable``; discarding other schema-valid rows
    would instead erase real positive or confuser evidence from that view.
    """
    payload = None
    standalone_rows: list[object] = []
    for candidate in _decoded_json_values(text):
        if (
            isinstance(candidate, Mapping)
            and set(candidate) == {"v"}
            and isinstance(candidate.get("v"), list)
        ):
            payload = candidate
            break
        # ``raw_decode`` can still recover later complete rows when one row
        # makes the enclosing object invalid.  These are independent schema
        # units; accepting them does not repair or infer the malformed row.
        if isinstance(candidate, list) and len(candidate) == 5:
            standalone_rows.append(candidate)
    raw_rows = payload["v"] if payload is not None else standalone_rows
    if not raw_rows:
        raise ValueError("qwen3vl_batch_json_object_missing")

    results: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw in raw_rows:
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 5
        ):
            continue
        try:
            raw_index = raw[0]
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, int)
            ):
                continue
            index = int(raw_index)
            if not 1 <= index <= expected_count or index in seen:
                continue
            verification = SemanticVerification.from_json(
                {
                    "target_probability": raw[1],
                    "anchor_probability": raw[2],
                    "confuser_probabilities": raw[3],
                    "rationale_tags": raw[4],
                }
            )
        except (TypeError, ValueError, KeyError):
            continue
        seen.add(index)
        results.append({
            "index": index,
            "target_probability": verification.target_probability,
            "anchor_probability": verification.anchor_probability,
            "confuser_probabilities": dict(
                verification.confuser_probabilities
            ),
            "rationale_tags": list(verification.rationale_tags),
        })
    if not results:
        raise ValueError("qwen3vl_batch_verifications_empty")
    return sorted(results, key=lambda value: int(value["index"]))


def _json_syntax_candidates(text: str) -> list[str]:
    """Return syntax-only repairs without inventing any semantic field.

    Quantized generation can wrap JSON in fences, use typographic quotes, or
    stop immediately after a complete list item.  The repairs below only
    normalize delimiters/quotes and close an otherwise complete prefix.  Every
    caller still applies its exact schema validator afterwards.
    """
    raw = str(text).strip()
    unfenced = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    normalized = (
        unfenced.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\uff02", '"')
    )
    normalized = re.sub(r'(?<=\d)"(?=\s*[,}])', "", normalized)
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    candidates = [raw, unfenced, normalized]

    starts = [index for index, value in enumerate(normalized) if value in "[{"]
    for start in starts:
        fragment = normalized[start:].strip()
        stack: list[str] = []
        in_string = False
        escaped = False
        invalid = False
        for character in fragment:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "[{":
                stack.append(character)
            elif character in "]}":
                expected = "[" if character == "]" else "{"
                if not stack or stack[-1] != expected:
                    invalid = True
                    break
                stack.pop()
        if invalid or in_string or not stack:
            continue
        repairable = fragment.rstrip()
        # A cut-off proposal object may follow one or more complete objects.
        # Discard only that incomplete suffix, then close the enclosing list.
        if stack[-1] == "{" and "[" in stack:
            last_complete_object = repairable.rfind("}")
            first_array = repairable.find("[")
            if first_array <= last_complete_object:
                repairable = repairable[: last_complete_object + 1]
                stack = ["["]
        repairable = re.sub(r",\s*$", "", repairable)
        if not repairable or repairable[-1] in "[{:":
            continue
        closing = "".join("]" if value == "[" else "}" for value in reversed(stack))
        candidates.append(repairable + closing)
    return list(dict.fromkeys(value for value in candidates if value))


def _decoded_json_values(text: str):
    decoder = json.JSONDecoder()
    for candidate in _json_syntax_candidates(text):
        for index, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                payload, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            yield payload


def _strict_semantic_json(text: str) -> SemanticVerification:
    """Extract the first complete JSON object and validate the strict schema.

    Quantized instruction models occasionally prepend a sentence or emit a
    second example object even when asked for JSON-only output.  The previous
    greedy ``{.*}`` capture joined both objects into invalid JSON and discarded
    an otherwise valid calibrated verdict.  ``raw_decode`` preserves strict
    schema validation while tolerating harmless surrounding text.
    """
    saw_object_start = False
    last_schema_error: Exception | None = None
    for candidate in _json_syntax_candidates(text):
        saw_object_start = saw_object_start or "{" in candidate
    for payload in _decoded_json_values(text):
        if not isinstance(payload, Mapping):
            continue
        saw_object_start = True
        try:
            return SemanticVerification.from_json(payload)
        except (TypeError, ValueError) as exc:
            last_schema_error = exc
    if last_schema_error is not None:
        raise ValueError("qwen3vl_json_schema_invalid") from last_schema_error
    if saw_object_start:
        raise ValueError("qwen3vl_json_invalid")
    raise ValueError("qwen3vl_json_object_missing")


def _grounding_prompt(
    concepts: Sequence[Mapping[str, object]],
    *,
    view_id: str,
    scene_context: str,
) -> str:
    specification = []
    for concept in concepts:
        class_name = str(concept["class_name"])
        aliases = ", ".join(str(value) for value in concept.get("aliases", ())) or "none"
        visual_definition = str(
            concept.get("visual_definition", class_name)
        ).strip()
        context = json.dumps(
            list(concept.get("supporting_context", ())),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        negatives = ", ".join(
            str(value) for value in concept.get("hard_negatives", ())
        ) or "none"
        if concept.get("context_group") is True:
            reporting_instruction = (
                "emit one coarse union box around the complete visible context "
                "group rather than one box per member"
            )
        elif concept.get("complete_instance_box") is True:
            reporting_instruction = (
                "emit one box per visible physical instance; each box must "
                "cover that instance's complete visible extent, including "
                "class-defining attached parts, container, stem, stand, or "
                "base when they belong to the same object, while excluding "
                "supporting furniture and background"
            )
        else:
            reporting_instruction = "emit one box per visible physical instance"
        specification.append(
            f"- label {class_name!r}; aliases: {aliases}; relation-local "
            f"attention hints (never a membership requirement): {context}; "
            f"visible class definition: "
            f"{visual_definition!r}; exclude: {negatives}; reporting: "
            f"{reporting_instruction}"
        )
    return (
        "Locate every visible physical instance that belongs to the categories "
        "listed below. Report bbox coordinates in Qwen3-VL's native JSON format: "
        "a JSON list of objects like "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"category"}]. '
        "bbox_2d coordinates are integer relative coordinates from 0 to 1000. "
        "Use exactly one of the supplied label names in each label field. Return [] "
        "when none is visible. Do not output confidence scores, prose, markdown, "
        "duplicate boxes, screen content, reflections, or inferred hidden objects. "
        "A segmentation model will refine the coarse boxes. Apply the required "
        "visual definitions and exclusions when choosing instances. Search at "
        "multiple scales, including small or partly occluded instances and objects "
        "near image edges. Do not stop after finding one large or easy instance of "
        "a category. For every relation-local attention hint, inspect the "
        "neighborhood of each visible context object as well as the whole image, "
        "but report a visible category instance even when its relation is not yet "
        "known; relation verification happens later. Enumerate all visible "
        "instances that satisfy the supplied definitions.\n" + "\n".join(specification) +
        f"\nQuestion context: {scene_context}\nView id: {view_id}"
    )


def _strict_grounding_json(
    text: str,
    label_to_class: Mapping[str, str],
) -> list[dict[str, object]]:
    last_error: Exception | None = None
    for payload in _decoded_json_values(text):
        try:
            if isinstance(payload, list):
                raw_proposals = payload
            elif isinstance(payload, Mapping) and set(payload) == {"proposals"}:
                raw_proposals = payload["proposals"]
            else:
                raise ValueError("grounding_output_schema_invalid")
            if not isinstance(raw_proposals, list):
                raise ValueError("grounding_proposals_not_list")
            proposals = []
            for raw in raw_proposals:
                if not isinstance(raw, Mapping) or set(raw) != {
                    "label", "bbox_2d",
                }:
                    raise ValueError("grounding_proposal_schema_invalid")
                emitted_label = str(raw["label"]).strip().lower()
                class_name = label_to_class.get(emitted_label)
                if class_name is None:
                    # A narrowed candidate request can still make Qwen repeat
                    # another visible class from the shared task context.  An
                    # out-of-request label is not evidence that the requested
                    # proposals in the same JSON list are malformed.  Ignore
                    # that extra proposal and preserve the requested boxes.
                    continue
                bbox_1000 = [float(value) for value in raw["bbox_2d"]]
                bbox = list(_normalized_bbox([
                    value / 1000.0 for value in bbox_1000
                ]))
                proposals.append({
                    "class_name": class_name,
                    "bbox_xyxy_normalized": bbox,
                    "rationale_tags": ["qwen3vl_native_2d_grounding"],
                })
            unique_proposals = []
            seen_proposals = set()
            for proposal in proposals:
                key = (
                    str(proposal["class_name"]),
                    *(
                        round(float(value), 6)
                        for value in proposal["bbox_xyxy_normalized"]
                    ),
                )
                if key in seen_proposals:
                    continue
                seen_proposals.add(key)
                unique_proposals.append(proposal)
            return unique_proposals
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError("qwen3vl_grounding_json_invalid") from last_error


def _relation_images(parameters):
    from PIL import Image

    output = []
    path_values = parameters.get("relation_images")
    if path_values is not None:
        if not isinstance(path_values, list) or not 1 <= len(path_values) <= 3:
            raise ValueError("qwen3vl_relation_image_contract_invalid")
        for value in path_values:
            if not isinstance(value, Mapping) or set(value) != {
                "role", "image_path", "preprocessed_once"
            }:
                raise ValueError("qwen3vl_relation_image_path_schema_invalid")
            if value["preprocessed_once"] is not True:
                raise ValueError("qwen3vl_relation_image_not_preprocessed")
            path = Path(str(value["image_path"])).resolve()
            if not path.is_file():
                raise ValueError("qwen3vl_relation_image_missing")
            image = Image.open(path).convert("RGB")
            if image.width % 32 or image.height % 32:
                raise ValueError("qwen3vl_relation_image_profile_invalid")
            output.append((str(value["role"]), image))
        return tuple(output)

    values = parameters.get("shared_memories")
    if not isinstance(values, list) or not 1 <= len(values) <= 3:
        raise ValueError("qwen3vl_relation_image_contract_invalid")
    from .shared_memory_io import SharedArrayStore

    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "role", "name", "shape", "dtype", "preprocessed_once"
        }:
            raise ValueError("qwen3vl_relation_shared_memory_schema_invalid")
        if value["preprocessed_once"] is not True:
            raise ValueError("qwen3vl_relation_image_not_preprocessed")
        image = SharedArrayStore().get(
            str(value["name"]),
            tuple(int(item) for item in value["shape"]),
            str(value["dtype"]),
            unlink=False,
        )
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or str(image.dtype) != "uint8"
            or image.shape[0] % 32
            or image.shape[1] % 32
        ):
            raise ValueError("qwen3vl_relation_image_profile_invalid")
        output.append((str(value["role"]), Image.fromarray(image, mode="RGB")))
    return tuple(output)


def _relation_role_prompt(request: Mapping[str, object]) -> str:
    """Audit exact boxed role identities before asking about the predicate."""
    subject_id = int(request["subject_id"])
    object_ids = [int(value) for value in request["object_ids"]]
    object_descriptions = [
        str(value) for value in request["object_descriptions"]
    ]
    object_contract = "; ".join(
        f"O{index}:{object_id} must visibly be {description!r}"
        for index, (object_id, description) in enumerate(
            zip(object_ids, object_descriptions),
            start=1,
        )
    )
    return (
        "You are performing an independent object-role identity audit. "
        "Do not evaluate the spatial predicate in this pass and do not trust "
        "the upstream class labels. Treat each exact coloured box as the "
        "identity anchor for that role. Use the surrounding joint-context "
        "pixels only to recover the complete physical body connected to the "
        "boxed region (for example, whether a boxed surface is merely one "
        "part of a larger object); never use a separate nearby item to satisfy "
        "the role. The red box S:"
        f"{subject_id} must visibly be {str(request['subject_description'])!r}; "
        + object_contract
        + ". The subject image isolates S, the objects image isolates O1/O2, "
        "and joint_context shows which coloured boxes refer to the same or "
        "different physical bodies. Overlapping boxes do not validate one "
        "another; when two roles enclose the same body but request different "
        "classes, judge each requested class independently. Set a role to Y "
        "only when the complete physical object containing the boxed pixels "
        "belongs to the requested class and defining visual parts are "
        "positively visible. Set N "
        "only when a different physical class is positively visible, and name "
        "it in x using subject_is_<class> or object_<index>_is_<class>. Use U "
        "for blur, partial pixels, mixed bodies, or insufficient evidence. "
        "The relation field r is U in this identity-only pass. The x field "
        "must be one snake_case label, never a sentence: use roles_match when "
        "all roles are Y, role_unclear when any role is U, or the required "
        "*_is_<class> label when a role is N. Return the requested compact "
        "JSON only."
    )


def _relation_prompt(request: Mapping[str, object]) -> str:
    subject_id = int(request["subject_id"])
    object_ids = [int(value) for value in request["object_ids"]]
    predicate = str(request["predicate"]).upper()
    parameter_roles = [str(value) for value in request["parameter_roles"]]
    subject_description = str(request["subject_description"])
    object_descriptions = [
        str(value) for value in request["object_descriptions"]
    ]
    object_legend = ", ".join(
        f"O{index}:{object_id} is object argument {index}"
        for index, object_id in enumerate(object_ids, start=1)
    )
    geometry_diagnostic = json.dumps(
        request["geometry_diagnostic"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    vertical_contract = ""
    if predicate in {"ABOVE", "BELOW"}:
        vertical_contract = (
            f" Evaluate exactly {predicate}(S:{subject_id}, O1:{object_ids[0]}), "
            "never the inverse. Reconcile the visible directed order with the "
            "supplied 3D center delta, bounds, and horizontal-projection evidence. "
            "These are observations, not a threshold or veto: no single boolean, "
            "distance, overlap value, or missing field decides the result. A "
            "supported result must be compatible with the combined visual and 3D "
            "evidence. If credible cues conflict or remain insufficient, use "
            "uncertain; horizontal separation alone is not a refutation."
        )
    elif predicate == "ON":
        vertical_contract = (
            f" Evaluate exactly ON(S:{subject_id}, O1:{object_ids[0]}). "
            "Inspect whether the labelled subject visibly rests on, contacts, "
            "or is directly supported by a surface of the labelled object. "
            "Do not infer support from containment, box overlap, or relative "
            "image position alone. Use refuted only when the visible layout "
            "contradicts support, and uncertain when the support/contact surface "
            "cannot be resolved. Reconcile the image "
            "with vertical_gap_m and horizontal_projection_axis_overlap_m in "
            "proportion to geometry_reliability. Sparse small-object depth can "
            "attach to the rear wall or support and displace its map center; a "
            "low-reliability metric mismatch must not override clear physical "
            "support in the shared perspective image. Use uncertain when the "
            "contact surface is hidden and metric evidence is also unreliable."
        )
    elif predicate == "NEAR":
        vertical_contract = (
            f" Evaluate exactly NEAR(S:{subject_id}, O1:{object_ids[0]}). "
            "Joint visibility or membership in the same room/crop is not NEAR. "
            "Compare the physical separation of the two labelled bodies with "
            "their visible size and local support layout. They should be locally "
            "adjacent as physical objects, not merely visible at opposite sides "
            "of a wide context image. Reconcile the pixels with distances_m, "
            "bounding_sphere_surface_gaps_m, and "
            "center_distance_over_combined_bbox_radius as continuous diagnostic "
            "cues, never as a fixed threshold. geometry_reliability states how "
            "much those metric cues deserve; participant_geometry_support shows "
            "why. Sparse small-object masks can attach their few depth returns "
            "to a background wall or support, so a low-reliability large surface "
            "gap must not override clear local visual adjacency. If the labelled "
            "plant and book are visibly adjacent on the same cabinet, return "
            "supported even when sparse metric depth is displaced. A large "
            "scale-normalized gap is a conflict only in proportion to its "
            "reliability. Use uncertain when neither the visual layout nor "
            "reliable geometry resolves adjacency; never return supported solely "
            "because the tuple is jointly observable."
        )
    elif predicate == "BETWEEN":
        support_diagnostic = ""
        if request["geometry_diagnostic"].get(
            "top_support_possible_on_any_boundary"
        ) is False:
            support_diagnostic = (
                " For this exact tuple, the supplied 3D evidence says "
                "top_support_possible_on_any_boundary=false; therefore do not "
                "use a resting-on, supported-by, or sits-on-boundary explanation "
                "to refute BETWEEN."
            )
        vertical_contract = (
            f" Evaluate exactly BETWEEN(S:{subject_id}, O1:{object_ids[0]}, "
            f"O2:{object_ids[1]}). Recover the physical 3D layout, not merely the "
            "red/blue/green rectangle order. First locate the body centers and "
            "floor or support contact of all three labelled instances. Then ask "
            "whether the subject lies in the spatial corridor joining the two "
            "distinct boundary objects: the direction from O1 to S should "
            "continue toward O2, and S should be reasonably near that corridor. "
            "A floor-standing S inside the corridor is BETWEEN even when closer "
            "to one endpoint. A subject sitting on unrelated furniture is not "
            "BETWEEN merely because its 2D box center falls between two wide or "
            "partly occluded anchor boxes. When S visibly rests on the top "
            "surface of O1 or O2, that is a direct physical contradiction: "
            "return refuted even if a noisy map diagnostic places its center "
            "inside the corridor. If an anchor box visibly covers the "
            "wrong object or its physical center/support cannot be located, use "
            "uncertain rather than supported. Bounding-box overlap, adjacency, "
            "or partial occlusion alone proves neither BETWEEN nor top support. "
            "Reconcile the image with geometry_diagnostic: segment_position "
            "and segment_position_interval describe position along O1-to-O2, and "
            f"vertical_center_deltas_m is ordered [O1:{object_ids[0]}, "
            f"O2:{object_ids[1]}]. A non-positive delta invalidates top support "
            "on that boundary but does not alone prove BETWEEN. Use uncertain "
            "when the physical layout cannot be recovered. The "
            "shared_image_horizontal_order diagnostic describes only the same "
            "joint crop's 2D ordering; it is a cue, never sufficient evidence. "
            "Do not infer BETWEEN merely from horizontal box order or absence of "
            "support, and do not default to a support explanation from overlapping "
            "boxes."
            + support_diagnostic
        )
    predicate_retry = (
        "previous response was contract-invalid"
        in str(request["verification_instruction"]).lower()
    )
    if predicate_retry:
        return (
            "This is a predicate-only reconsideration of the same grounded IDs, "
            "not a new detection task. Inspect the supplied labelled images and "
            f"evaluate exactly subject_id={subject_id}, predicate={predicate}, "
            f"object_ids={object_ids}. Grounding legend: red S is subject_id "
            f"{subject_id}; {object_legend}. The preceding pass already "
            "classified every grounded role YES. Keep those role states YES "
            "unless the pixels positively show a different object class. A "
            "correct class name is never a spatial refutation. Predicate-specific "
            f"instruction: {request['verification_instruction']} The tuple's "
            f"read-only 3D observations are {geometry_diagnostic}."
            f"{vertical_contract} Decide state=supported when the exact physical "
            "layout satisfies the predicate, state=refuted only for a visible "
            "physical contradiction to the predicate, and state=uncertain when "
            "the evidence conflicts or is insufficient. For a refuted state, "
            "reason_code must be one short snake_case label describing that "
            "spatial contradiction, never the requested class. Keep reason_code "
            "under 48 characters. Return exactly one JSON object with exactly these "
            "keys: subject_id, predicate, object_ids, subject_role_state, "
            "object_role_states, state, confidence, reason_code, visible_subject, "
            "visible_objects, jointly_observable, occlusion. Repeat the supplied "
            "IDs as raw JSON integers exactly, with no S: or O: prefixes; role "
            "and visibility arrays must match object_ids. occlusion must be the "
            "string none, partial, or severe."
        )
    return (
        "You are an object-ID-grounded visual relation verifier. "
        "Use only the supplied crops and joint context. Never invent, merge, "
        "or substitute IDs. Treat missing pixels, occlusion, or absent joint "
        "visibility as uncertain. Verify this exact tuple: "
        f"subject_id={subject_id}, predicate={predicate}, "
        f"object_ids={object_ids}. Directed parameter roles are "
        f"{parameter_roles}. The subject must visibly satisfy the description "
        f"{subject_description!r}; the object descriptions are "
        f"{object_descriptions!r}. Evidence policy: "
        f"{request['evidence_policy']}. Predicate-specific instruction: "
        f"{request['verification_instruction']} Negative-evidence rule: "
        f"{request['negative_evidence_policy']}. The exact tuple's read-only 3D "
        f"diagnostic is geometry_diagnostic={geometry_diagnostic}."
        f"{vertical_contract} Grounding legend: red S:{subject_id} is the subject; "
        f"{object_legend}. The image labelled subject contains the red subject "
        "grounding, the image labelled objects contains the object groundings, "
        "and the image labelled joint_context contains all groundings together. "
        "The supplied IDs and descriptions come from upstream grounding; this "
        "call verifies their spatial predicate instead of repeating candidate "
        "generation. Use the role fields as a visible contradiction check. Set "
        "subject_role_state and each entry of object_role_states to exactly "
        "YES, NO, or UNKNOWN. YES requires positive visible evidence of the "
        "requested physical class and its defining parts; mere compatibility, "
        "a plausible location, or the upstream label is not enough. UNKNOWN "
        "means the crop is blurred, partial, too small, or otherwise lacks "
        "those defining visual features. NO requires visible evidence that the "
        "grounded ID is a different physical class. Inspect the pixels inside "
        "the exact labelled box: a nearby requested object outside that box "
        "does not validate the boxed ID. "
        "When a role is NO, reason_code must name the positively observed "
        "alternative using subject_is_<class> or object_<index>_is_<class>; "
        "a tautology such as subject_not_<requested_class>, object_not_<class>, "
        "NO, or not_visible is not negative evidence and must be UNKNOWN. "
        "Any listed visual alternatives are comparison cues, never an automatic "
        "exclusion or fixed rejection rule; blur, occlusion, or insufficient "
        "pixels is UNKNOWN. A plausible location or a visually plausible "
        "relation must never turn an ID into the requested object class. A "
        "relation can be supported only when every required role is YES; if a "
        "role is UNKNOWN, the relation state must be uncertain. Set "
        "jointly_observable=true when that joint image "
        "actually lets you compare the labelled subject and object layout, even "
        "when the relation is false. A missing object, a crop that "
        "does not show the joint layout, or failure to see the relation is "
        "uncertain rather than refuted. Do not emit a default response: inspect "
        "the pixels separately for this exact tuple. Choose relation state as "
        "exactly supported, refuted, or uncertain from the labelled joint "
        "layout. A role UNKNOWN does not erase a clearly visible spatial "
        "predicate because candidate identity was established upstream; a role "
        "NO refutes this exact binding. Choose occlusion as exactly none, "
        "partial, or severe. reason_code must be one short snake_case label under "
        "48 characters, not a sentence or copied diagnostic. Return exactly one "
        "JSON object, no prose and no "
        "markdown, with exactly these keys: subject_id, predicate, object_ids, "
        "subject_role_state, object_role_states, state, confidence, reason_code, "
        "visible_subject, visible_objects, jointly_observable, occlusion. "
        "object_role_states and visible_objects must be JSON arrays with "
        "exactly one entry for every object_id, in the same order. "
        "Repeat the supplied IDs exactly. "
        "confidence is confidence in the chosen state, not a scene answer."
    )


def _strict_relation_json(
    text: str,
    expected: Mapping[str, object],
) -> ObjectGroundedRelationResult:
    def normalized_id(value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        match = re.search(r"(?:^|:)(-?\d+)$", str(value).strip())
        return int(match.group(1)) if match is not None else value

    decoder = json.JSONDecoder()
    last_error = None
    for index, character in enumerate(str(text)):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(str(text)[index:])
            if isinstance(payload, Mapping):
                payload = dict(payload)
                if set(payload) == {"s", "o", "r", "j", "x", "c"}:
                    compact_state = str(payload["r"]).strip().upper()
                    # The batch template is shared across tuple arities and
                    # the model occasionally echoes extra role entries.  Keep
                    # exactly the requested arity: extra entries are dropped,
                    # missing ones become UNKNOWN.
                    if isinstance(payload.get("o"), list):
                        expected_arity = len(expected["object_ids"])
                        if len(payload["o"]) < expected_arity:
                            payload["o"] = [
                                *payload["o"],
                                *(["U"] * (expected_arity - len(payload["o"]))),
                            ]
                        elif len(payload["o"]) > expected_arity:
                            payload["o"] = payload["o"][:expected_arity]
                    compact_role = {
                        "Y": "YES", "N": "NO", "U": "UNKNOWN",
                        "YES": "YES", "NO": "NO", "UNKNOWN": "UNKNOWN",
                    }
                    compact_subject = compact_role.get(
                        str(payload["s"]).strip().upper(), "UNKNOWN"
                    )
                    compact_objects = [
                        compact_role.get(
                            str(value).strip().upper(), "UNKNOWN"
                        )
                        for value in payload["o"]
                    ]
                    payload = {
                        "subject_id": int(expected["subject_id"]),
                        "predicate": str(expected["predicate"]).upper(),
                        "object_ids": [
                            int(value) for value in expected["object_ids"]
                        ],
                        "subject_role_state": compact_subject,
                        "object_role_states": compact_objects,
                        "state": {
                            "Y": "supported",
                            "N": "refuted",
                            "U": "uncertain",
                        }.get(compact_state, "uncertain"),
                        "confidence": float(payload["c"]) / 100.0,
                        "reason_code": str(payload["x"]),
                        "visible_subject": compact_subject != "UNKNOWN",
                        "visible_objects": [
                            value != "UNKNOWN" for value in compact_objects
                        ],
                        "jointly_observable": bool(payload["j"]),
                        "occlusion": (
                            "partial"
                            if "UNKNOWN" in [compact_subject, *compact_objects]
                            else "none"
                        ),
                    }
                state = str(payload.get("state", "")).strip().lower()
                payload["state"] = {
                    "yes": "supported",
                    "no": "refuted",
                    "unknown": "uncertain",
                }.get(state, state)
                payload["subject_id"] = normalized_id(
                    payload.get("subject_id")
                )
                if isinstance(payload.get("object_ids"), (list, tuple)):
                    payload["object_ids"] = [
                        normalized_id(value)
                        for value in payload["object_ids"]
                    ]
                if isinstance(payload.get("occlusion"), bool):
                    payload["occlusion"] = (
                        "partial" if payload["occlusion"] else "none"
                    )
                if not str(payload.get("reason_code", "")).strip():
                    if payload["state"] == "supported":
                        payload["reason_code"] = "predicate_supported"
                    elif payload["state"] == "uncertain":
                        payload["reason_code"] = "evidence_insufficient"
            result = ObjectGroundedRelationResult.from_json(payload)
            if (
                result.subject_id != int(expected["subject_id"])
                or result.predicate != str(expected["predicate"]).upper()
                or result.object_ids != tuple(
                    int(value) for value in expected["object_ids"]
                )
            ):
                raise ValueError("qwen_relation_object_id_mismatch")
            return result
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError("qwen3vl_relation_json_invalid") from last_error


class LocalQwen3VLImplementation:
    """Lazy, memory-bounded Qwen3-VL callable for ``Qwen3VLBackend``.

    The detector/SAM process occupies most of a 32 GB competition GPU.  A
    second full-BF16 copy of Qwen3-VL can therefore be killed by the driver
    after the worker socket has already passed a superficial liveness check.
    INT8 keeps the verifier resident alongside the official vision process
    without changing its candidate-only contract.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        max_new_tokens: int = 128,
        max_pixels: int = 512 * 512,
        quantization: str = "bf16",
        batch_max_new_tokens: int = 256,
    ):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError("qwen3vl_checkpoint_directory_missing")
        self.checkpoint_path = str(checkpoint)
        self.device = str(device)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_max_new_tokens = int(batch_max_new_tokens)
        if self.max_new_tokens < 1 or self.batch_max_new_tokens < 1:
            raise ValueError("qwen3vl_token_budget_must_be_positive")
        self.max_pixels = max(224 * 224, int(max_pixels))
        quantization = str(quantization).strip().lower()
        if quantization not in {"int8", "int4", "bf16"}:
            raise ValueError("qwen3vl_quantization_must_be_int8_int4_or_bf16")
        self.quantization = quantization
        self._model = None
        self._processor = None
        self._active_deadline_monotonic = 0.0

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
            max_pixels=self.max_pixels,
        )
        # Qwen3-VL is decoder-only. Batched rows have different multimodal
        # prefix lengths, so right padding shifts generation and can truncate
        # the beginning of shorter rows. Transformers batch generation requires
        # left padding here.
        self._processor.tokenizer.padding_side = "left"
        compute_dtype = (
            torch.float16 if self.quantization in {"int8", "int4"}
            else torch.bfloat16
        )
        model_kwargs = {
            "local_files_only": True,
            "dtype": compute_dtype,
            "attn_implementation": "sdpa",
        }
        if self.quantization == "int8":
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": {"": self.device},
            })
        elif self.quantization == "int4":
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": {"": self.device},
            })
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint_path,
            **model_kwargs,
        )
        if self.quantization == "bf16":
            self._model = self._model.to(self.device)
        self._model.eval()

    def _generate(self, model_inputs, *, max_new_tokens: int):
        """Generate with cooperative deadline cancellation at token steps."""
        import torch

        deadline = float(self._active_deadline_monotonic)
        if deadline > 0.0 and deadline <= time.monotonic():
            raise TimeoutError("model_service_deadline_expired")
        stopping_criteria = None
        if deadline > 0.0:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _DeadlineStoppingCriteria(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return bool(time.monotonic() >= deadline)

            stopping_criteria = StoppingCriteriaList([
                _DeadlineStoppingCriteria()
            ])
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=False,
                **(
                    {"stopping_criteria": stopping_criteria}
                    if stopping_criteria is not None else {}
                ),
            )
        if deadline > 0.0 and deadline <= time.monotonic():
            raise TimeoutError("model_service_deadline_expired")
        return generated

    def healthcheck(self) -> Mapping[str, object]:
        """Load all weights so launch-time health proves runtime residency."""
        self._ensure_loaded()
        return {
            "ready": True,
            "device": str(self._model.device),
            "quantization": self.quantization,
            "max_pixels": self.max_pixels,
            "batch_max_new_tokens": self.batch_max_new_tokens,
            "deadline_cancellation": "cooperative_token_boundary",
            "checkpoint_path": self.checkpoint_path,
        }

    def __call__(self, request) -> Mapping[str, object]:
        parameters = dict(getattr(request, "parameters", {}) or {})
        self._active_deadline_monotonic = float(
            getattr(request, "deadline_monotonic", 0.0) or 0.0
        )
        operation = str(getattr(request, "operation", ""))
        if operation == "verify_relation":
            return self._verify_relation(parameters)
        if operation == "verify_relation_batch":
            return self._verify_relation_batch(parameters)
        if operation == "ground_objects":
            return self._ground_objects(parameters)
        if operation == "ground_objects_batch":
            return self._ground_objects_batch(parameters)
        if operation == "verify_object_batch":
            return self._verify_object_batch(parameters)
        if operation == "rank_candidates":
            return self._rank_candidates(parameters)
        bbox = _normalized_bbox(parameters.get("candidate_bbox", ()))
        query_concept = str(parameters.get("query_concept", "")).strip()
        if not query_concept:
            raise ValueError("qwen3vl_query_concept_missing")
        anchor_concept = str(parameters.get("anchor_concept", "")).strip()
        hard_negatives = tuple(parameters.get("hard_negatives", ()) or ())

        self._ensure_loaded()
        source_image = _request_source_image(parameters)
        object_crop, context_crop = _candidate_images_from_pil(
            source_image, bbox
        )
        prompt = _verification_prompt(
            operation=operation,
            query_concept=query_concept,
            anchor_concept=anchor_concept,
            hard_negatives=hard_negatives,
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": object_crop},
                {"type": "image", "image": context_crop},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            verification = _strict_semantic_json(decoded)
        except ValueError as exc:
            # The worker log is local and contains model output only (never
            # credentials or source image bytes).  Keep a bounded diagnostic
            # so a failed semantic verdict is distinguishable from transport
            # failure instead of silently appearing as generic "unknown".
            print(json.dumps({
                "event": "qwen3vl_invalid_output",
                "error": str(exc),
                "decoded": decoded[:500],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        return {
            "target_probability": verification.target_probability,
            "anchor_probability": verification.anchor_probability,
            "confuser_probabilities": dict(
                verification.confuser_probabilities
            ),
            "rationale_tags": list(verification.rationale_tags),
            "backend": "qwen3vl_local_candidate_verifier",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _verify_object_batch(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Verify up to eight candidates from one source view in one wave."""
        candidates = parameters.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("qwen3vl_verify_batch_candidates_missing")
        if len(candidates) > 8:
            raise ValueError("qwen3vl_verify_batch_max_eight_candidates")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("qwen3vl_verify_batch_candidate_invalid")
            if not str(candidate.get("query_concept", "")).strip():
                raise ValueError("qwen3vl_verify_batch_query_concept_missing")
            _normalized_bbox(candidate.get("candidate_bbox", ()))

        self._ensure_loaded()
        source_image = _request_source_image(parameters)
        annotated_image = _numbered_candidate_image_from_pil(
            source_image,
            candidates,
        )
        prompt = _batch_verification_prompt(
            candidates,
            anchor_concept=str(parameters.get("anchor_concept", "")).strip(),
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": annotated_image},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        # The compact response contains one independently validated row per
        # candidate.  A fixed 256-token ceiling truncates otherwise valid JSON
        # when a full view supplies seven or eight candidates, causing every
        # row in that view to become unavailable.  Scale only this object-batch
        # response with its explicit cardinality contract. Configured capacity
        # remains the floor; every returned row still passes the strict schema
        # parser, while a model-omitted index remains explicitly unavailable.
        verification_max_new_tokens = max(
            self.batch_max_new_tokens,
            64 * len(candidates),
        )
        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=verification_max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            verifications = _strict_semantic_batch_json(
                str(decoded),
                len(candidates),
            )
        except ValueError as exc:
            print(json.dumps({
                "event": "qwen3vl_invalid_batch_verification_output",
                "error": str(exc),
                "decoded": str(decoded)[:1000],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        returned_indices = {
            int(value["index"]) for value in verifications
        }
        missing_indices = [
            index for index in range(1, len(candidates) + 1)
            if index not in returned_indices
        ]
        if missing_indices:
            print(json.dumps({
                "event": "qwen3vl_partial_batch_verification",
                "missing_candidate_indices": missing_indices,
                "valid_candidate_indices": sorted(returned_indices),
            }, ensure_ascii=False), file=sys.stderr, flush=True)
        return {
            "verifications": verifications,
            "candidate_count": len(candidates),
            "missing_candidate_indices": missing_indices,
            "backend": "qwen3vl_local_candidate_verifier_batch",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _ground_objects(self, parameters: Mapping[str, object]) -> Mapping[str, object]:
        view_id = str(parameters.get("view_id", "")).strip()
        concepts = parameters.get("concepts")
        if not view_id:
            raise ValueError("qwen3vl_grounding_view_id_missing")
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("qwen3vl_grounding_concepts_missing")
        normalized_concepts = []
        label_to_class = {}
        for raw in concepts:
            if not isinstance(raw, Mapping):
                raise ValueError("qwen3vl_grounding_concept_invalid")
            class_name = str(raw.get("class_name", "")).strip().lower()
            if not class_name:
                raise ValueError("qwen3vl_grounding_class_missing")
            label_to_class[class_name] = class_name
            for alias in raw.get("aliases", ()):
                label_to_class[str(alias).strip().lower()] = class_name
            normalized_concepts.append({**dict(raw), "class_name": class_name})
        self._ensure_loaded()
        source_image = _request_source_image(parameters)
        prompt = _grounding_prompt(
            normalized_concepts,
            view_id=view_id,
            scene_context=str(parameters.get("scene_context", "")),
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": source_image},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # The model occasionally emits a truncated JSON array when it runs
        # out of tokens.  Repair the common pattern of a missing closing "]".
        decoded_for_parse = str(decoded)
        decoded_stripped = decoded_for_parse.rstrip()
        if decoded_stripped.endswith(",") or (
            decoded_stripped.count("[") > decoded_stripped.count("]")
        ):
            decoded_stripped = re.sub(r',\s*$', '', decoded_stripped)
            last_complete_object = decoded_stripped.rfind("}")
            first_array = decoded_stripped.find("[")
            if 0 <= first_array < last_complete_object:
                decoded_for_parse = (
                    decoded_stripped[: last_complete_object + 1].rstrip(",")
                    + "]"
                )
        try:
            proposals = _strict_grounding_json(
                decoded_for_parse,
                label_to_class,
            )
        except ValueError as exc:
            print(json.dumps({
                "event": "qwen3vl_invalid_grounding_output",
                "error": str(exc),
                "decoded": decoded[:1000],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        return {
            "view_id": view_id,
            "proposals": proposals,
            "proposal_count": len(proposals),
            "backend": "qwen3vl_query_conditioned_grounder",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _ground_objects_batch(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Ground objects across multiple views in a single GPU forward pass.

        Accepts ``views``: a list of {view_id, image_path, concepts,
        scene_context}.  All images are encoded together so the
        vision backbone runs once per batch instead of once per view.
        """
        views = parameters.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError("ground_objects_batch_views_missing")
        if len(views) > 8:
            raise ValueError("ground_objects_batch_max_eight_views")

        self._ensure_loaded()
        from PIL import Image

        per_view: list[dict] = []
        for v in views:
            view_id = str(v.get("view_id", "")).strip()
            concepts = v.get("concepts")
            if not view_id or not isinstance(concepts, list) or not concepts:
                raise ValueError("ground_objects_batch_view_incomplete")
            label_to_class: dict[str, str] = {}
            normalized = []
            for raw in concepts:
                if not isinstance(raw, dict):
                    raise ValueError("ground_objects_batch_concept_invalid")
                cn = str(raw.get("class_name", "")).strip().lower()
                if not cn:
                    raise ValueError("ground_objects_batch_class_missing")
                label_to_class[cn] = cn
                for alias in raw.get("aliases", ()):
                    label_to_class[str(alias).strip().lower()] = cn
                normalized.append({**dict(raw), "class_name": cn})
            prompt = _grounding_prompt(
                normalized,
                view_id=view_id,
                scene_context=str(v.get("scene_context", "")),
            )
            image = Image.open(str(v["image_path"])).convert("RGB")
            per_view.append({
                "view_id": view_id,
                "label_to_class": label_to_class,
                "image": image,
                "prompt": prompt,
            })

        # Each view is a separate batch row.  This avoids the previous
        # multi-image/single-conversation approximation, which could not
        # reliably associate a returned box with its source image.
        conversations = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": pv["image"]},
                {"type": "text", "text": pv["prompt"]},
            ],
        }] for pv in per_view]
        model_inputs = self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        import torch

        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=self.batch_max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded_rows = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results: list[dict] = []
        for pv, decoded in zip(per_view, decoded_rows):
            try:
                view_proposals = _strict_grounding_json(
                    str(decoded),
                    pv["label_to_class"],
                )
            except ValueError as exc:
                print(json.dumps({
                    "event": "qwen3vl_invalid_batch_grounding_output",
                    "view_id": pv["view_id"],
                    "error": str(exc),
                    "decoded": str(decoded)[:1000],
                }, ensure_ascii=False), file=sys.stderr, flush=True)
                results.append({
                    "view_id": pv["view_id"],
                    "proposals": [],
                    "proposal_count": 0,
                    "ok": False,
                    "error_code": (
                        f"ground_objects_batch_view_json_invalid:"
                        f"{pv['view_id']}"
                    ),
                    "error_detail": str(exc)[:300],
                    "backend": "qwen3vl_batch_multiview_grounder",
                    "checkpoint_path": self.checkpoint_path,
                    "quantization": self.quantization,
                })
                continue
            results.append({
                "view_id": pv["view_id"],
                "proposals": view_proposals,
                "proposal_count": len(view_proposals),
                "ok": True,
                "backend": "qwen3vl_batch_multiview_grounder",
                "checkpoint_path": self.checkpoint_path,
                "quantization": self.quantization,
            })

        return {
            "views": results,
            "backend": "qwen3vl_batch_multiview_grounder",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _rank_candidates(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Given a scene image and labelled candidates, return the index
        of the candidate that best matches the question description."""
        question = str(parameters.get("question", "")).strip()
        candidates = parameters.get("candidates")
        allow_none = bool(parameters.get("allow_none", False))
        if (
            not question
            or not isinstance(candidates, list)
            or not candidates
            or (len(candidates) < 2 and not allow_none)
        ):
            raise ValueError("rank_candidates_needs_question_and_2plus_candidates")

        self._ensure_loaded()
        from PIL import Image, ImageDraw

        source = Image.open(str(parameters["image_path"])).convert("RGB")
        width, height = source.size
        draw = ImageDraw.Draw(source)

        candidate_descriptions = []
        for i, cand in enumerate(candidates):
            label = chr(ord("A") + i)
            bbox = cand.get("bbox_xyxy")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)
                draw.text((x1 + 4, y1 + 4), label, fill=(255, 50, 50))
            cls = str(cand.get("class_name", ""))
            ctx = str(cand.get("context", ""))
            candidate_descriptions.append(
                f"Candidate {label}: {cls}"
                + (f" ({ctx})" if ctx else "")
            )

        selection_contract = (
            "Return NONE when every candidate contradicts the requested "
            "physical entity or the entity is not visibly identifiable. "
            "Otherwise return exactly one candidate letter."
            if allow_none else
            "Return exactly one candidate letter."
        )
        candidate_question = (
            "Which candidate, if any, best matches what the question asks "
            "for?" if allow_none else
            "Which candidate best matches what the question asks for?"
        )
        prompt = (
            f"Question: {question}\n\n"
            + "\n".join(candidate_descriptions)
            + "\n\n"
            "The image shows the scene with candidates labelled A,B,C... "
            "in red boxes. " + candidate_question + " "
            "Consider the full spatial context described in the question. "
            + selection_contract
            + " Return exactly one JSON object with key \"best_candidate\" "
            "(the letter or NONE) and \"confidence\" (0-1). No prose, no "
            "markdown."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": source},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse the response
        import json as _json, re as _re
        decoder = _json.JSONDecoder()
        best_idx: int | None = None if allow_none else 0
        confidence = 0.5
        for idx, ch in enumerate(str(decoded)):
            if ch != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(str(decoded)[idx:])
                letter = str(payload.get("best_candidate", "A")).strip().upper()
                if allow_none and letter in {"NONE", "NO_MATCH", "UNKNOWN"}:
                    best_idx = None
                elif len(letter) == 1 and "A" <= letter <= "Z":
                    best_idx = ord(letter) - ord("A")
                confidence = float(payload.get("confidence", 0.5))
                break
            except (_json.JSONDecodeError, ValueError, KeyError):
                continue

        if best_idx is not None and not 0 <= best_idx < len(candidates):
            # With an explicit NONE option, an out-of-domain label is an
            # unresolved visual binding, not permission to clamp onto an
            # unrelated endpoint candidate.
            best_idx = None if allow_none else max(
                0, min(best_idx, len(candidates) - 1)
            )
        return {
            "best_candidate_index": (
                None if best_idx is None else int(best_idx)
            ),
            "binding_state": "UNKNOWN" if best_idx is None else "YES",
            "confidence": float(max(0.0, min(1.0, confidence))),
            "backend": "qwen3vl_candidate_ranker",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _verify_relation(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        grounded = parameters.get("grounded_relation_request")
        if not isinstance(grounded, Mapping):
            raise ValueError("qwen3vl_grounded_relation_request_missing")
        required = {
            "request_id",
            "episode_id",
            "query_node_id",
            "world_snapshot_version",
            "acquisition_id",
            "subject_id",
            "subject_instance_version",
            "object_ids",
            "object_instance_versions",
            "predicate",
            "parameter_roles",
            "subject_description",
            "object_descriptions",
            "evidence_policy",
            "negative_evidence_policy",
            "verification_instruction",
            "geometry_diagnostic",
            "subject_bbox_or_mask",
            "object_bboxes_or_masks",
            "camera_pose",
            "evidence_provenance",
        }
        if set(grounded) != required:
            raise ValueError("qwen3vl_grounded_relation_schema_invalid")
        images = _relation_images(parameters)
        self._ensure_loaded()
        predicate_retry = (
            "previous response was contract-invalid"
            in str(grounded["verification_instruction"]).lower()
        )
        prompt_images = (
            tuple(
                (role, image) for role, image in images
                if role == "joint_context"
            )
            if predicate_retry else images
        )
        messages = [{
            "role": "user",
            "content": [
                *(
                    item
                    for role, image in prompt_images
                    for item in (
                        {"type": "text", "text": f"Evidence role: {role}"},
                        {"type": "image", "image": image},
                    )
                ),
                {"type": "text", "text": _relation_prompt(grounded)},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            result = _strict_relation_json(decoded, grounded)
        except ValueError as exc:
            print(json.dumps({
                "event": "qwen3vl_relation_invalid_output",
                "error": str(exc),
                "decoded": decoded[:500],
                "subject_id": grounded["subject_id"],
                "predicate": grounded["predicate"],
                "object_ids": grounded["object_ids"],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        return {
            "subject_id": result.subject_id,
            "predicate": result.predicate,
            "object_ids": list(result.object_ids),
            "subject_role_state": result.subject_role_state,
            "object_role_states": list(result.object_role_states),
            "state": result.state,
            "confidence": result.confidence,
            "visible_subject": result.visible_subject,
            "visible_objects": list(result.visible_objects),
            "jointly_observable": result.jointly_observable,
            "occlusion": result.occlusion,
            "reason_code": result.reason_code,
            "backend": "qwen3vl_local_object_grounded_relation_verifier",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _verify_relation_batch(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Verify independent ID tuples in one physical generation wave."""
        raw_requests = parameters.get("requests")
        verification_pass = str(
            parameters.get("verification_pass", "predicate")
        ).strip().lower()
        if verification_pass not in {"roles", "predicate"}:
            raise ValueError("qwen3vl_relation_batch_pass_invalid")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError("qwen3vl_relation_batch_requests_missing")
        if len(raw_requests) > 8:
            raise ValueError("qwen3vl_relation_batch_max_eight_requests")
        prepared = []
        conversations = []
        for raw in raw_requests:
            if not isinstance(raw, Mapping):
                raise ValueError("qwen3vl_relation_batch_request_invalid")
            grounded = raw.get("grounded_relation_request")
            if not isinstance(grounded, Mapping):
                raise ValueError("qwen3vl_grounded_relation_request_missing")
            images = _relation_images(raw)
            predicate_retry = (
                "previous response was contract-invalid"
                in str(grounded["verification_instruction"]).lower()
            )
            prompt_images = (
                tuple(
                    (role, image) for role, image in images
                    if role == "joint_context"
                )
                if predicate_retry else images
            )
            conversations.append([{
                "role": "user",
                "content": [
                    *(
                        item
                        for role, image in prompt_images
                        for item in (
                            {"type": "text", "text": f"Evidence role: {role}"},
                            {"type": "image", "image": image},
                        )
                    ),
                    {"type": "text", "text": (
                        (
                            _relation_role_prompt(grounded)
                            if verification_pass == "roles"
                            else _relation_prompt(grounded)
                        )
                        + " Batched-wave output override: return only compact "
                        "JSON {\"s\":\"Y|N|U\",\"o\":[\"Y|N|U\"],"
                        "\"r\":\"Y|N|U\",\"j\":true,\"x\":"
                        "\"short_reason\",\"c\":0}. Here s is subject role, "
                        "o has one entry per object role, r is predicate state, "
                        "j is joint observability, and c is integer confidence "
                        "0..100. Output no other keys or prose."
                    )},
                ],
            }])
            prepared.append(dict(grounded))
        self._ensure_loaded()
        model_inputs = self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._generate(
                model_inputs,
                max_new_tokens=self.batch_max_new_tokens,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded_rows = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results = []
        for grounded, decoded in zip(prepared, decoded_rows):
            try:
                result = _strict_relation_json(decoded, grounded)
                results.append({
                    "ok": True,
                    "metadata": {
                        "subject_id": result.subject_id,
                        "predicate": result.predicate,
                        "object_ids": list(result.object_ids),
                        "subject_role_state": result.subject_role_state,
                        "object_role_states": list(result.object_role_states),
                        "state": result.state,
                        "confidence": result.confidence,
                        "visible_subject": result.visible_subject,
                        "visible_objects": list(result.visible_objects),
                        "jointly_observable": result.jointly_observable,
                        "occlusion": result.occlusion,
                        "reason_code": result.reason_code,
                        "verification_pass": verification_pass,
                        "backend": (
                            "qwen3vl_local_object_grounded_relation_batch"
                        ),
                        "checkpoint_path": self.checkpoint_path,
                        "quantization": self.quantization,
                    },
                    "error_code": "",
                })
            except Exception as exc:
                print(json.dumps({
                    "event": "qwen3vl_relation_batch_row_invalid",
                    "error": str(exc),
                    "decoded": str(decoded)[:500],
                    "subject_id": grounded.get("subject_id"),
                    "predicate": grounded.get("predicate"),
                    "object_ids": grounded.get("object_ids"),
                }, ensure_ascii=False), file=sys.stderr, flush=True)
                results.append({
                    "ok": False,
                    "metadata": {"error_detail": str(exc)[:500]},
                    "error_code": type(exc).__name__,
                })
        return {
            "results": results,
            "batch_size": len(results),
            "backend": "qwen3vl_local_object_grounded_relation_batch",
        }
