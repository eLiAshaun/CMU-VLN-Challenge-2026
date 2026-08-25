THIS FILE IS THE ONLY CURRENT AI MODULE ARCHITECTURE AUTHORITY.

Historical handoffs, exploration/answering phase designs,
legacy counting pipelines, previous implementation prompts,
and obsolete runtime documents are not part of the current architecture.

# Final AI Module Architecture

## Runtime call graph

```text
/challenge_question
        |
        v
      TaskIR
        |
        v
Perception -> SceneMemory -> Canonical Identity + Map Geometry
                         -> RelationEngine -> SceneSnapshot
                                                  |
                         +------------------------+------------------------+
                         |                        |                        |
                         v                        v                        v
                NumericalResolver      ObjectReferenceResolver   InstructionResolver
                  Count Query Graph       Unique Object             Ordered Constraint
                         |                Resolution                 Resolution
                         +------------------------+------------------------+
                                                  |
                                                  v
                                           ResolverResult
                         +------------------------+------------------------+
                         |                        |                        |
                  FINALIZABLE              NEED_EVIDENCE           NEED_EXECUTION
                         |                        |                        |
                         v                        v                        v
                   RootFinalizer       EvidenceAcquisition-       NavigationExecutor
                         |                 Coordinator                    |
                         v                        |                        |
                  OutputAdapter             ObservationIntent            |
                         |                        |                        |
                         v                        +-----------+------------+
                 official ROS output                         |
                                                            v
                                                   NavigationExecutor
                                                            |
                                                            v
                                                   /state_estimation
                                                            |
                                                            v
                                              fresh observation transaction
                                                            |
                                                            v
                                                       SceneMemory
                                                            |
                                                            +----> resolve again
```

## Authorities

`TaskIR` is the only parsed-question authority. Task parsing is implemented by
`integrations/semantics/task_compiler.py`; resolvers consume the resulting immutable task
description and do not reinterpret the question.

`SceneMemory` in `integrations/execution/scene_memory.py` is the only world-state
authority. It owns canonical object IDs, aliases, ambiguity groups, observation
transactions, identity reconciliation, map-frame geometry, semantic evidence,
relation-evidence provenance, and world revisions. Resolver retry state and
answer-bearing task state are not stored there.

`RelationEngine` in `integrations/execution/relation_engine.py` is the only
relation-verdict authority. Qwen tuple verification, geometry, visibility,
freshness, persistent relation observations, and selector distance are evidence
providers. Only the engine returns `YES`, `NO`, `UNKNOWN`, or `INVALID`.

Task resolution is implemented by the three thin resolvers in
`integrations/execution/task_resolvers.py`:

- `NumericalResolver` consumes the sole Count Query Graph result. An integer is
  finalizable only when that graph is complete over the current canonical
  domain.
- `ObjectReferenceResolver` resolves one stable canonical object and requires
  the current SceneMemory geometry needed for the marker.
- `InstructionResolver` derives ordered-constraint progress from the TaskIR,
  current SceneSnapshot, and the actual trajectory. A step counter alone never
  establishes completion.

Every resolver returns the common `ResolverResult` contract from
`integrations/execution/resolver_contracts.py`: `FINALIZABLE`, `NEED_EVIDENCE`,
`NEED_EXECUTION`, or `SYSTEM_FAILURE`. `FINALIZABLE` is the only status that may
carry a final payload.

All missing semantic evidence is expressed as the same `EvidenceNeed` contract.
`EvidenceAcquisitionCoordinator` in
`integrations/execution/evidence_acquisition.py` is the only semantic evidence
acquisition authority. It converts an `EvidenceNeed` into an
`ObservationIntent` and accepts completion only after physical arrival, a fresh
post-arrival observation, a consumed SceneMemory transaction, updated world
revisions, required visibility, and the relevant relation recomputation.

`NavigationExecutor` in `integrations/execution/navigation_executor.py` owns
only physical waypoint dispatch, arrival tracking, and execution telemetry for an
`ObservationIntent` or `ExecutionNeed`. It reports planned, moving, arrived, or
failed facts. It does not decide whether evidence, a relation, or a task is
complete.

`/state_estimation` is the sole actual-pose and actual-trajectory truth. The ROS
parent feeds those odometry samples to `NavigationExecutor` and the trajectory
monitor. `/way_point_reached` is diagnostic and cannot establish semantic
progress.

`RootFinalizer` in `integrations/execution/root_finalizer.py` is the sole final
authorization authority and the sole constructor of root decisions. Normal
completion, time-budget exhaustion, safe rejection, and system failure all pass
through it. A budget event cannot invent an answer; it can commit only an
already-current `FINALIZABLE` result.

`OutputAdapter` in `integrations/ros/output_adapter.py` only validates the root
envelope, performs message-type conversion, and dispatches the official ROS
message. It owns no semantic answer validation, task intent, fallback, or
navigation behavior.

## Fixed invariants

- No global exploration/answering phase.
- No second world model.
- No second relation truth source.
- No task-specific independent evidence state machine.
- No direct final-answer fallback.
- No semantic authority in ROS adapters.
- No semantic authority in `NavigationExecutor`.
- All semantic decisions are recomputed from `TaskIR + SceneSnapshot + actual
  trajectory`.
- All `NEED_EVIDENCE` results enter the shared evidence coordinator.
- All physical movement enters `NavigationExecutor`.
- All terminal paths enter `RootFinalizer`.

## Authority matrix

| Responsibility | Sole owner |
|---|---|
| Question parsing | `TaskIR` / `task_compiler.py` |
| World state and identity | `SceneMemory` |
| Object geometry | `SceneMemory` |
| Relation verdict | `RelationEngine` |
| Numerical resolution | `NumericalResolver` / Count Query Graph |
| Object-reference resolution | `ObjectReferenceResolver` |
| Instruction resolution | `InstructionResolver` |
| Evidence diagnosis | Resolver -> `EvidenceNeed` |
| Evidence acquisition | `EvidenceAcquisitionCoordinator` |
| Physical execution | `NavigationExecutor` |
| Actual pose and trajectory | `/state_estimation` |
| Final authorization | `RootFinalizer` |
| ROS serialization and publish | `OutputAdapter` |
