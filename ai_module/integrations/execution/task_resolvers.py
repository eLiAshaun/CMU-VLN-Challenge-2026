"""Three thin task resolvers over the shared SceneSnapshot."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from integrations.execution.query_executor import derive_task_resolution_evidence
from integrations.execution.scene_memory import materialize_query_view
from integrations.execution.resolver_contracts import (
    EvidenceNeed,
    EvidenceReason,
    ExecutionNeed,
    ResolverResult,
    ResolverStatus,
)


def _revision(snapshot: Mapping[str, Any], key: str, *aliases: str) -> int:
    for name in (key, *aliases):
        try:
            return max(0, int(snapshot.get(name, 0)))
        except (TypeError, ValueError):
            continue
    return 0


def _provenance(raw: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(value) for value in raw.get("evidence_ids", ())
        if str(value).strip()
    ))


def _reason(raw: Mapping[str, Any]) -> EvidenceReason:
    failures = " ".join(str(value).lower() for value in raw.get("failed_constraints", ()))
    if "identity" in failures or "unique" in failures:
        return EvidenceReason.IDENTITY_DISAMBIGUATION
    if "relation" in failures or "selector" in failures:
        return EvidenceReason.RELATION_EVIDENCE
    if "geometry" in failures or "localiz" in failures:
        return EvidenceReason.LOCALIZE_ENTITY
    if "semantic" in failures or "verification" in failures:
        return EvidenceReason.VERIFY_SEMANTIC
    return EvidenceReason.DISCOVER_OBJECT


def _ids(values: object) -> tuple[int, ...]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return ()
    result: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item >= 0 and item not in result:
            result.append(item)
    return tuple(result)


def _probe_target_ids(probe: Mapping[str, Any]) -> tuple[int, ...]:
    candidate_ids = _ids(probe.get("probe_candidate_object_ids", ()))
    if candidate_ids:
        return candidate_ids
    return _ids((probe.get("probe_source_candidate_id"),))


def _semantic_evidence_ids(
    snapshot: Mapping[str, Any],
    object_ids: Sequence[object],
) -> tuple[int, ...]:
    requested = _ids(object_ids)
    if not requested:
        return ()
    statuses: dict[int, str] = {}
    for value in snapshot.get("objects", ()):
        if not isinstance(value, Mapping):
            continue
        try:
            object_id = int(value["object_id"])
        except (KeyError, TypeError, ValueError):
            continue
        statuses[object_id] = str(
            value.get("semantic_status", "unverified")
        ).strip().lower()
    return tuple(
        object_id for object_id in requested
        if statuses.get(object_id, "unverified") in {"unverified", "ambiguous"}
        and _probe_attempts(
            snapshot, "VERIFY_SEMANTIC", (object_id,), (), None
        ) < 2
    )


def _probe_attempts(
    snapshot: Mapping[str, Any],
    reason: str,
    target_ids: Sequence[object],
    anchor_ids: Sequence[object],
    predicate: str | None,
) -> int:
    signature = (
        str(reason), _ids(target_ids), _ids(anchor_ids), str(predicate or "")
    )
    attempts = 0
    for record in snapshot.get("navigation_history", ()):
        if not isinstance(record, Mapping):
            continue
        intent = record.get("observation_intent", {})
        if not isinstance(intent, Mapping):
            continue
        need = intent.get("evidence_need", {})
        if not isinstance(need, Mapping):
            continue
        observed = (
            str(need.get("reason", "")),
            _ids(need.get("target_ids", ())),
            _ids(need.get("anchor_ids", ())),
            str(need.get("predicate") or ""),
        )
        attempts += int(observed == signature)
    return attempts


@dataclass(frozen=True)
class CountCertificate:
    lower_bound: int
    upper_bound: int | None
    supported_entities: tuple[dict[str, Any], ...]
    provisional_singletons: tuple[int, ...]
    duplicate_risk_orphans: tuple[int, ...]
    unresolved_relations: tuple[str, ...]
    unseen_relevant_volume: dict[str, Any]
    strict_closed: bool
    best_evidence_answer: int
    remaining_uncertainties: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "count_certificate_v1",
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "supported_entities": [dict(value) for value in self.supported_entities],
            "provisional_singletons": list(self.provisional_singletons),
            "duplicate_risk_orphans": list(self.duplicate_risk_orphans),
            "unresolved_relations": list(self.unresolved_relations),
            "unseen_relevant_volume": dict(self.unseen_relevant_volume),
            "strict_closed": self.strict_closed,
            "best_evidence_answer": self.best_evidence_answer,
            "remaining_uncertainties": [
                dict(value) for value in self.remaining_uncertainties
            ],
            "why_this_integer_has_highest_support": (
                "Counts query-target entities with affirmative QueryProgram evidence; "
                "unresolved candidates remain uncertainty rather than votes."
            ),
        }


def _count_certificate(
    task_ir: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> CountCertificate:
    lower = graph.get("cardinality_lower_bound")
    lower = int(lower) if isinstance(lower, int) and not isinstance(lower, bool) else 0
    upper = graph.get("cardinality_upper_bound")
    upper = int(upper) if isinstance(upper, int) and not isinstance(upper, bool) else None
    counted = _ids(graph.get("counted_target_ids", ()))
    unknown = _ids(graph.get("unknown_target_ids", ()))
    domain = snapshot.get("count_query_domain", {})
    domain = domain if isinstance(domain, Mapping) else {}
    target_entity = str(graph.get("target_entity", "target_0"))
    variables = task_ir.get("count_query_graph", {}).get("variables", ())
    target_class = next((
        str(value.get("class_name", "")) for value in variables
        if isinstance(value, Mapping) and str(value.get("entity_id")) == target_entity
    ), "")
    relevant: list[Mapping[str, Any]] = []
    for value in snapshot.get("objects", ()):
        if not isinstance(value, Mapping):
            continue
        if target_class and str(value.get("class_label", "")) != target_class:
            continue
        relevant.append(value)
    lookup = {
        int(value["object_id"]): value for value in relevant
        if str(value.get("object_id", "")).lstrip("-").isdigit()
    }
    supported = tuple({
        "entity_id": object_id,
        "semantic_probability": float(lookup.get(object_id, {}).get("semantic_probability", 0.0) or 0.0),
        "independent_viewpoints": int(lookup.get(object_id, {}).get("independent_viewpoint_count", 0) or 0),
    } for object_id in counted)
    provisional = tuple(sorted(
        object_id for object_id, value in lookup.items()
        if str(value.get("entity_lifecycle")) == "NEW_SPACE_SINGLETON"
    ))
    orphans = tuple(sorted(
        object_id for object_id, value in lookup.items()
        if str(value.get("entity_lifecycle")) == "DUPLICATE_RISK_ORPHAN"
    ))
    relation_ids = tuple(sorted({
        str(value) for value in domain.get("unknown_relation_ids", ()) if str(value)
    }))
    coverage_closed = bool(domain.get("closed") is True)
    remaining: list[dict[str, Any]] = []
    if orphans:
        remaining.append({"variable_type": "IDENTITY", "entity_ids": list(orphans), "reason": "duplicate_risk"})
    if unknown:
        remaining.append({"variable_type": "RELATION", "entity_ids": list(unknown), "reason": "query_predicate_unknown"})
    ambiguities = domain.get("singular_domain_ambiguities", graph.get("singular_domain_ambiguities", {}))
    if isinstance(ambiguities, Mapping) and ambiguities:
        remaining.append({"variable_type": "ANCHOR", "entity_ids": [], "reason": "singular_reference_unresolved"})
    if not coverage_closed:
        remaining.append({"variable_type": "COVERAGE", "entity_ids": [], "reason": "relevant_volume_open"})
    strict = bool(
        upper is not None and lower == upper and not orphans
        and not relation_ids and not unknown and coverage_closed
    )
    return CountCertificate(
        lower_bound=lower,
        upper_bound=upper,
        supported_entities=supported,
        provisional_singletons=provisional,
        duplicate_risk_orphans=orphans,
        unresolved_relations=relation_ids,
        unseen_relevant_volume={
            "closed": coverage_closed,
            "coverage_score": float(domain.get("coverage_score", 0.0) or 0.0),
            "covered_viewpoints": len(domain.get("covered_viewpoints", ())),
            "covered_regions": len(domain.get("covered_regions", ())),
            "closure_reasons": list(domain.get("closure_reasons", ())),
            "finite_candidate_upper_bound": lower + len(unknown),
        },
        strict_closed=strict,
        best_evidence_answer=lower,
        remaining_uncertainties=tuple(remaining),
    )


class _TaskResolver:
    task_type = ""

    def __init__(self, relation_engine) -> None:
        self.relation_engine = relation_engine

    def _base(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "scene_version": _revision(snapshot, "scene_version"),
            "identity_revision": _revision(snapshot, "identity_revision"),
            "geometry_version": _revision(
                snapshot, "geometry_version", "geometry_revision"
            ),
            "relation_revision": _revision(snapshot, "relation_revision"),
        }

    def _need(
        self,
        task_ir: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        raw: Mapping[str, Any],
        *,
        reason: EvidenceReason | None = None,
        target_ids: Sequence[object] = (),
        anchor_ids: Sequence[object] = (),
        predicate: str | None = None,
        suggested_target: Mapping[str, Any] | None = None,
        include_count_query_request: bool = True,
    ) -> ResolverResult:
        base = self._base(snapshot)
        suggestion = (
            suggested_target
            if isinstance(suggested_target, Mapping)
            else raw.get("probe_object")
        )
        count_request = (
            snapshot.get("count_query_targeted_acquisition")
            if include_count_query_request else None
        )
        need = EvidenceNeed(
            query_key=str(task_ir.get("original_question", self.task_type)),
            task_type=self.task_type,
            reason=reason or _reason(raw),
            target_ids=_ids(target_ids),
            anchor_ids=_ids(anchor_ids),
            predicate=predicate,
            required_observability={
                "fresh_observation_after_arrival": True,
                "scene_memory_transaction_required": True,
                "relation_recompute_required": bool(predicate),
            },
            scene_version=base["scene_version"],
            identity_revision=base["identity_revision"],
            geometry_version=base["geometry_version"],
            relation_revision=base["relation_revision"],
            priority_context={
                "suggested_target": (
                    dict(suggestion) if isinstance(suggestion, Mapping) else None
                ),
                "count_query_request": (
                    dict(count_request) if isinstance(count_request, Mapping) else None
                ),
                "failed_constraints": list(raw.get("failed_constraints", ())),
            },
        )
        return ResolverResult(
            **base,
            status=ResolverStatus.NEED_EVIDENCE,
            evidence_need=need,
            provenance=_provenance(raw),
            diagnostics={"task_evidence": dict(raw)},
        )


class NumericalResolver(_TaskResolver):
    task_type = "numerical"

    def resolve(
        self,
        task_ir: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        **_: Any,
    ) -> ResolverResult:
        if str(snapshot.get("scene_memory_authority", "")) != "ObservationLedger+QueryProgram":
            entity_view = materialize_query_view(
                task_ir,
                snapshot.get("observation_ledger", {}),
                current_snapshot=snapshot,
            )
            snapshot = {
                **dict(snapshot),
                **{
                    key: entity_view[key]
                    for key in (
                        "scene_version", "identity_revision", "geometry_version",
                        "objects", "object_id_aliases", "identity_clusters",
                        "identity_constraints", "identity_ambiguity_groups",
                        "ambiguous_observations", "viewpoint_history",
                        "observation_ledger_size", "cardinality_summary",
                    )
                },
                "query_entity_view": entity_view,
                "scene_memory_authority": "ObservationLedger+QueryProgram",
            }
        raw = derive_task_resolution_evidence(task_ir, snapshot)
        graph = raw.get("count_query_execution")
        graph = graph if isinstance(graph, Mapping) else {}
        certificate = _count_certificate(task_ir, snapshot, graph).to_dict()
        graph_complete = bool(
            certificate["strict_closed"]
            and (
                graph.get("complete") is True
                or graph.get("cardinality_complete") is True
            )
        )
        answer = graph.get("answer") if isinstance(graph, Mapping) else None
        if (
            graph_complete
            and isinstance(answer, int)
            and not isinstance(answer, bool)
            and answer >= 0
        ):
            return ResolverResult(
                **self._base(snapshot),
                status=ResolverStatus.FINALIZABLE,
                final_payload={
                    "answer": int(answer),
                    "selected_object": None,
                    "evidence_ids": list(_provenance(raw)),
                    "resolution_mode": "count_query_graph",
                    "commit_mode": "STRICT_COMMIT",
                    "count_certificate": certificate,
                },
                provenance=_provenance(raw),
                diagnostics={"task_evidence": raw, "count_certificate": certificate},
            )
        request = snapshot.get("count_query_targeted_acquisition", {})
        target_ids = (
            request.get("target_object_ids", ())
            if isinstance(request, Mapping) else ()
        )
        anchor_ids = (
            request.get("anchor_object_ids", ())
            if isinstance(request, Mapping) else ()
        )
        predicate = (
            str(request.get("predicate"))
            if isinstance(request, Mapping) and request.get("predicate") else None
        )
        semantic_domain_ids: list[int] = []
        if isinstance(graph, Mapping):
            domains = graph.get("candidate_domains", {})
            if isinstance(domains, Mapping):
                target_entity = str(graph.get("target_entity", ""))
                ordered_entities = [
                    target_entity,
                    *(
                        str(entity_id) for entity_id in domains
                        if str(entity_id) != target_entity
                    ),
                ]
                for entity_id in ordered_entities:
                    semantic_domain_ids.extend(_ids(domains.get(entity_id, ())))
        identity_disambiguation = bool(
            isinstance(request, Mapping)
            and (
                request.get("identity_disambiguation") is True
                or str(request.get("mode", "")) == "identity_targeted"
                or str(request.get("reason", ""))
                == "identity_association_ambiguity"
            )
        )
        anchor_localization = bool(
            isinstance(request, Mapping)
            and str(request.get("probe_type", "")) == "ANCHOR_LOCALIZATION"
        )
        targeted_request = bool(
            isinstance(request, Mapping)
            and str(request.get("mode", "")) in {
                "relation_targeted", "identity_targeted"
            }
        )
        requested_semantic_ids = (
            *_ids(target_ids),
            *_ids(anchor_ids),
        )
        semantic_evidence_ids = (
            [] if anchor_localization else _semantic_evidence_ids(
                snapshot,
                (
                    *requested_semantic_ids,
                    *(() if targeted_request else semantic_domain_ids),
                ),
            )[:1]
        )
        semantic_target = None
        if semantic_evidence_ids:
            semantic_target = next(
                (
                    dict(value)
                    for value in snapshot.get("objects", ())
                    if isinstance(value, Mapping)
                    and str(value.get("object_id", "")).lstrip("-").isdigit()
                    and int(value["object_id"]) == semantic_evidence_ids[0]
                ),
                None,
            )
            if semantic_target is not None:
                # VERIFY_SEMANTIC is a new evidence contract, not a relabeling
                # of the prior relation/identity request.  Target the selected
                # unresolved object itself and remove stale tuple participants
                # and navigation hints from the superseded request.
                semantic_target.update({
                    "probe_candidate_object_ids": list(semantic_evidence_ids),
                    "probe_object_ids": list(semantic_evidence_ids),
                    "probe_source_candidate_id": semantic_evidence_ids[0],
                    "required_visible_object_ids": list(semantic_evidence_ids),
                    "probe_anchor_object_ids": [],
                    "probe_relation_ids": [],
                    "probe_relation": "",
                    "joint_visibility_required": False,
                    "requires_new_station": True,
                    "observation_objective": {
                        "mode": "semantic_verification",
                        "required_visible_object_ids": list(
                            semantic_evidence_ids
                        ),
                        "requires_new_station": True,
                    },
                })
                semantic_target.pop("navigation_target_xy", None)
                semantic_target.pop("count_query_request", None)
        selected_reason = (
            EvidenceReason.VERIFY_SEMANTIC
            if semantic_evidence_ids
            else EvidenceReason.IDENTITY_DISAMBIGUATION
            if identity_disambiguation
            else EvidenceReason.RELATION_EVIDENCE
            if predicate
            else EvidenceReason.DISCOVER_OBJECT
        )
        selected_targets = semantic_evidence_ids or _ids(target_ids)
        selected_anchors = (
            () if semantic_evidence_ids or identity_disambiguation
            or anchor_localization
            else _ids(anchor_ids)
        )
        selected_predicate = (
            None if semantic_evidence_ids or identity_disambiguation
            else predicate
        )
        attempts = _probe_attempts(
            snapshot, selected_reason.value, selected_targets,
            selected_anchors, selected_predicate,
        )
        result = self._need(
            task_ir,
            snapshot,
            raw,
            reason=selected_reason,
            target_ids=selected_targets,
            anchor_ids=selected_anchors,
            predicate=selected_predicate,
            suggested_target=semantic_target,
            include_count_query_request=not bool(semantic_evidence_ids),
        )
        uncertainty_type = {
            EvidenceReason.VERIFY_SEMANTIC: "SEMANTIC",
            EvidenceReason.IDENTITY_DISAMBIGUATION: "IDENTITY",
            EvidenceReason.RELATION_EVIDENCE: "RELATION",
        }.get(selected_reason, "COVERAGE")
        need = result.evidence_need
        if need is not None:
            context = dict(need.priority_context)
            context["uncertainty_variable"] = {
                "variable_type": uncertainty_type,
                "entity_ids": list(selected_targets),
                "anchor_ids": list(selected_anchors),
                "predicate": selected_predicate,
                "prior_probe_count": attempts,
                "maximum_probe_count": 2,
            }
            result = ResolverResult(
                **self._base(snapshot), status=result.status,
                evidence_need=EvidenceNeed(
                    **{**need.__dict__, "priority_context": context}
                ),
                provenance=result.provenance,
                diagnostics={
                    "task_evidence": raw,
                    "count_certificate": certificate,
                },
            )
        return result


class ObjectReferenceResolver(_TaskResolver):
    task_type = "object_reference"

    def resolve(
        self,
        task_ir: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        *,
        entity_bindings: Mapping[str, int] | None = None,
        navigation_history: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> ResolverResult:
        raw = derive_task_resolution_evidence(
            task_ir,
            snapshot,
            object_relation_verifier=self.relation_engine,
            entity_bindings=entity_bindings,
            navigation_history=navigation_history,
        )
        target = str(task_ir.get("target_entity", ""))
        domain = raw.get("candidate_domains", {}).get(target, ())
        relation_required = raw.get("relation_required") is True
        eligible = (
            _ids(raw.get("relation_eligible_object_ids", ()))
            if relation_required
            else _ids(
                value.get("object_id") for value in domain
                if isinstance(value, Mapping)
            )
        )
        selected = raw.get("selected_object")
        if not relation_required and len(eligible) == 1:
            selected = next(
                (
                    dict(value)
                    for value in domain
                    if isinstance(value, Mapping)
                    and str(value.get("object_id", "")).lstrip("-").isdigit()
                    and int(value["object_id"]) == eligible[0]
                ),
                None,
            )
        selected_id = (
            int(selected.get("object_id"))
            if isinstance(selected, Mapping)
            and str(selected.get("object_id", "")).lstrip("-").isdigit()
            else None
        )
        if selected_id is not None:
            selected = next(
                (
                    dict(value)
                    for value in snapshot.get("objects", ())
                    if isinstance(value, Mapping)
                    and str(value.get("object_id", "")).lstrip("-").isdigit()
                    and int(value["object_id"]) == selected_id
                    and str(value.get("cardinality_role", "ATOMIC"))
                    == "ATOMIC"
                ),
                None,
            )
        if (
            len(eligible) == 1
            and selected_id == eligible[0]
            and isinstance(selected, Mapping)
        ):
            return ResolverResult(
                **self._base(snapshot),
                status=ResolverStatus.FINALIZABLE,
                final_payload={
                    "answer": selected_id,
                    "selected_object": dict(selected),
                    "evidence_ids": list(_provenance(raw)),
                    "relation_required": relation_required,
                    "relation_state": (
                        raw.get("relation_candidate_states", {}).get(
                            str(selected_id), "YES"
                        )
                    ),
                },
                provenance=_provenance(raw),
                diagnostics={"task_evidence": raw},
            )
        relations = list(task_ir.get("relations", ()))
        predicate = (
            str(relations[0].get("predicate"))
            if relations and isinstance(relations[0], Mapping) else None
        )
        probe = raw.get("probe_object", {})
        return self._need(
            task_ir,
            snapshot,
            raw,
            reason=(
                EvidenceReason.RELATION_EVIDENCE
                if relation_required else EvidenceReason.IDENTITY_DISAMBIGUATION
            ),
            target_ids=eligible or (
                _probe_target_ids(probe)
                if isinstance(probe, Mapping) else ()
            ),
            anchor_ids=(
                probe.get("probe_anchor_object_ids", ())
                if isinstance(probe, Mapping) else ()
            ),
            predicate=predicate,
        )


class InstructionResolver(_TaskResolver):
    task_type = "instruction_following"

    def resolve(
        self,
        task_ir: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        *,
        entity_bindings: Mapping[str, int] | None = None,
        execution_steps: Sequence[Mapping[str, Any]] | None = None,
        current_step_index: int = 0,
        navigation_history: Sequence[Mapping[str, Any]] | None = None,
        trajectory_monitor: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> ResolverResult:
        raw = derive_task_resolution_evidence(
            task_ir,
            snapshot,
            object_relation_verifier=self.relation_engine,
            entity_bindings=entity_bindings,
            execution_steps=execution_steps,
            current_step_index=current_step_index,
            navigation_history=navigation_history,
        )
        steps = [dict(value) for value in raw.get("execution_steps", ())]
        actual_complete = bool(
            steps
            and current_step_index == len(steps)
            and all(str(value.get("status")) == "SATISFIED" for value in steps)
            and isinstance(trajectory_monitor, Mapping)
            and trajectory_monitor.get("completed") is True
            and trajectory_monitor.get("forbidden_violation") is None
        )
        if actual_complete:
            return ResolverResult(
                **self._base(snapshot),
                status=ResolverStatus.FINALIZABLE,
                final_payload={
                    "answer": None,
                    "selected_object": None,
                    "evidence_ids": list(_provenance(raw)),
                    "execution_steps": steps,
                    "actual_trajectory_complete": True,
                },
                provenance=_provenance(raw),
                diagnostics={"task_evidence": raw},
            )
        directives = tuple(
            dict(value) for value in raw.get("trajectory_directives", ())
            if isinstance(value, Mapping)
        )
        if directives:
            return ResolverResult(
                **self._base(snapshot),
                status=ResolverStatus.NEED_EXECUTION,
                execution_need=ExecutionNeed(
                    task_type=self.task_type,
                    intent="EXECUTE_ORDERED_CONSTRAINT",
                    directives=directives,
                    current_step_index=int(current_step_index),
                    provenance=_provenance(raw),
                ),
                provenance=_provenance(raw),
                diagnostics={"task_evidence": raw},
            )
        probe = raw.get("probe_object", {})
        return self._need(
            task_ir,
            snapshot,
            raw,
            target_ids=(
                _probe_target_ids(probe)
                if isinstance(probe, Mapping) else ()
            ),
            anchor_ids=(
                probe.get("probe_anchor_object_ids", ())
                if isinstance(probe, Mapping) else ()
            ),
            predicate=(
                str(probe.get("probe_relation"))
                if isinstance(probe, Mapping) and probe.get("probe_relation")
                else None
            ),
        )


def resolver_for(task_type: str, relation_engine):
    resolvers = {
        "numerical": NumericalResolver,
        "object_reference": ObjectReferenceResolver,
        "instruction_following": InstructionResolver,
    }
    try:
        return resolvers[str(task_type)](relation_engine)
    except KeyError as exc:
        raise ValueError(f"unsupported_task_type:{task_type}") from exc
