# Phase 2 Runtime Closure Report — 2026-08-22

This report separates source deployment, fresh-container runtime closure, observed ROS output, local-proxy scoring, and semantic correctness. `challenge_evaluator` is a transparent local proxy, not an official score or official acceptance result.

The machine-readable trace bundle is `runtime_validation/PHASE2_COMPACT_TRACE_20260822.json`.

## A. Docker

- Freeze point: branch `phase2/runtime-validation-20260822`, commit `8f262bf`, annotated tag `final-semantic-authority-cutover`.
- Explicit staging was used; weights, caches, runs, evaluator runs, temporary outputs, and four pre-existing untracked `artifacts/scene_memory_real_*` directories were excluded. `git add -A` was not used.
- Build command: `DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker-compose -f docker/compose_gpu_headless.yml build --no-cache ai_module`.
- The first BuildKit attempt failed before source compilation because the host lacked the buildx component. The legacy Docker builder then completed the requested no-cache build. Final image: `docker_ai_module:p1p2-final`, short image ID `fae3c64054b0`.
- Official AI entrypoint: `/home/docker/ai_module/docker/start_live_chain.sh`.
- The entrypoint started and health-probed Qwen3VL, SAM2, and YOLO-World before starting `integrations.ros.live_task_probe`. Camera, `/sensor_scan`, `/registered_scan`, both terrain maps, and `/state_estimation` were live in the evaluator preflight.
- Runtime assets were readable and loaded from the declared Qwen3VL, SAM2, YOLO-World, and text-encoder checkpoint paths. No model, source, or image hash validation was performed.
- Deleted-path dependency: PASS. Active Python/shell runtime code contains no reference to `old_1_stage_search`, `src/scnav_vln`, `numerical_probe_coordinator`, `terminal_first_window`, or `semantic_checkpoint_pending`. The only `best_supported_hypothesis` occurrence is in the forbidden-input denylist, not an answer path.
- Each accepted final run recorded `machine_verified=true`, `fresh_container_instances=true`, `fresh_process_started=true`, an empty isolated AI run mount before startup, and `stopped_after_command=true`. The only AI bind mount was the per-command `/home/docker/ai_module/runs` directory; source was not bind-mounted.
- Latest-image runs used fresh AI containers `f70e1a491e60` (Instruction) and `65c39d435201` (difficult Numerical).

`FRESH_CONTAINER_READY = true`

## B. Numerical

### Simple Numerical — terminal path PASS, correctness FAIL

- Canonical case: `hotel_room_1/q1`, “How many pillows are on the bed?”
- Evaluation run: `/home/robot/cmu_vln/challenge_eval_runs/20260822T143906Z_hotel_room_1_q1`.
- TaskIR: `pillow ON bed`, output contract `/numerical_response`.
- Same episode `20260822T143923_316442Z` produced three acquisitions. Revision vectors advanced from `scene/identity/geometry/relation = 1/5/1/1` to `3/10/3/3`.
- Resolver/root history: `NEED_EVIDENCE/PROBE` twice, then `FINALIZABLE/COMMIT` once.
- Physical feedback: two commanded evidence waypoints, two real navigation arrivals, 999 `/state_estimation` samples, 3.978 m topic travel, one incomplete and one completed evidence transaction.
- Observed ROS output: `/numerical_response=1` at 105.637 s. Evaluator termination: `numerical_response_received`.
- Known ground truth is 4. Local-proxy exact-match result: incorrect, `0/1`. This is a real terminal-path pass, not a correct semantic answer.

### Difficult relational Numerical — FAIL

- Required canonical case: `livingroom_3/q1`, “How many photos are on the TV cabinet?”
- Evaluation run: `/home/robot/cmu_vln/challenge_eval_runs/20260822T152417Z_livingroom_3_q1`.
- TaskIR: `picture/photo ON television cabinet`, output contract `/numerical_response`.
- Same episode `20260822T152433_677346Z` produced 14 acquisitions. Revisions advanced from `1/4/1/1` to `6/16/6/14`.
- The shared RelationEngine produced two photo `ON` YES tuples for one cabinet hypothesis. CountGraph did not publish that apparent count: it correctly moved to `IDENTITY_DISAMBIGUATION` for competing TV-cabinet canonical IDs.
- Physical feedback: 13 commanded evidence waypoints, 12 real arrival facts, 5,789 `/state_estimation` samples, 12.908 m topic travel, two completed and nine incomplete evidence transactions.
- Root history remained `PROBE` for the model acquisitions. The episode deadline guard passed the current `NEED_EVIDENCE` result to RootFinalizer, which emitted `SAFE_REJECT` at 588.272 s. The status field named `answer_published` means the safe-reject lifecycle was handled; no `/numerical_response` was observed.
- Evaluator termination: `time_limit_reached` at 600 s. Known ground truth is 2; observed answer is absent; local proxy `0/1`.

## C. Object Reference

- Primary simple case: `hotel_room_2/q2`, “Find the flowers near the window.”
- Evaluation run: `/home/robot/cmu_vln/challenge_eval_runs/20260822T150126Z_hotel_room_2_q2`.
- TaskIR: `flower NEAR window`, output contract `/selected_object_marker`.
- Same episode `20260822T150140_547605Z` produced 16 acquisitions. Revisions advanced from `1/2/1/1` to `10/16/10/15`.
- Every resolver result was `NEED_EVIDENCE`; every model root decision was `PROBE`. After the EvidenceNeed repair, each request explicitly carried both target and anchor canonical IDs, and target-missing acquisitions were correctly recorded as incomplete.
- Physical feedback: 15 commanded evidence waypoints, 12 real arrivals, three navigation timeouts, 5,779 `/state_estimation` samples, 23.677 m topic travel, five completed and seven incomplete evidence transactions.
- Relation evidence remained `UNKNOWN/NO`; no unique canonical physical object reached finalizable closure.
- RootFinalizer emitted deadline `SAFE_REJECT` at 588.467 s. No `/selected_object_marker` was observed. Evaluator termination: `time_limit_reached`; local proxy `0/2`.
- A second canonical diagnostic, `office_1/q2` (“Find the potted plant on the file cabinet”), also completed repeated real arrivals/reacquisitions after its call-contract repair but produced no unique relation YES and no marker in the full 600 s window.

Object Reference real E2E: **FAIL**.

## D. Instruction

- Canonical case: `japanese_room/q4`, “Go near the small table with a vase on it and then to the flowers near the jar.”
- Evaluation run: `/home/robot/cmu_vln/challenge_eval_runs/20260822T152125Z_japanese_room_q4`.
- Corrected TaskIR contains exactly two ordered constraints:
  1. nonterminal `go_near small table`, qualified by `vase ON small table`;
  2. terminal `go_to flower`, qualified by `flower NEAR jar`.
- Same episode `20260822T152140_522453Z` used four resolver acquisitions: `NEED_EXECUTION/PROBE` three times, then `FINALIZABLE/COMMIT`.
- Three physical waypoints entered the shared NavigationExecutor. The trajectory monitor advanced only from real `/state_estimation`:
  - step 0 `SATISFIED` at map pose `[0.2893, 0.0070, 0.0434]`;
  - step 1 `SATISFIED` at map pose `[0.8297, 0.1737, 0.4830]`.
- Both steps record `satisfaction_source=actual_state_estimation_trajectory`. There were 1,431 state samples and three arrival/release facts; `/way_point_reached` was not used as completion authority.
- RootFinalizer emitted `COMMIT` with `episode_complete=true`. The authorized ROS lifecycle latch published `arrival_revalidated`; the evaluator terminated successfully as `instruction_route_complete` at 149.916 s.
- Local proxy score is `0.5386/6`, reflecting weak similarity to its reference trajectory. It is not an official score and does not change the verified runtime closure.

Instruction real terminal path: **PASS**.

## E. Shared-loop verification

- TaskIR authority: `integrations/semantics/task_compiler.py`; compiled once per language episode.
- Shared world and identity: one episode-owned SceneMemory; all four final traces show monotonically consumed scene/identity/geometry revisions.
- Shared relation truth: `integrations/execution/relation_engine.py`; Numerical, Object Reference, and Instruction consume its YES/NO/UNKNOWN verdicts rather than task-specific relation shortcuts.
- Shared evidence: all `NEED_EVIDENCE` results enter `EvidenceAcquisitionCoordinator`; transaction validation requires fresh post-arrival observation, SceneMemory commit, requested revision progress, required-object visibility, and relation recomputation when applicable.
- Shared physical execution: all evidence and ordered-constraint motion enters `NavigationExecutor`; actual pose/trajectory truth is `/state_estimation`.
- Three thin resolvers remain task-specific only at resolution: Numerical/CountGraph, ObjectReferenceResolver, and InstructionResolver.
- Final authority is unique: all terminal decisions observed here came from RootFinalizer. OutputAdapter serialized only authorized COMMIT envelopes; the Instruction lifecycle latch only relayed an already-authorized `episode_complete` decision.
- No legacy exploration/answering phase, terminal-first controller, cached-answer commit, deadline answer, or ROS-parent semantic answer path was added.

## F. Repairs made

| Symptom | First divergence | Owner | Minimal fix | Real regression result |
|---|---|---|---|---|
| First real scan crashed with `_stamp_seconds()` argument mismatch | ROS scan callback before TaskIR/world update | `live_task_probe` ROS ingestion | Marked `_stamp_seconds` static | All later runs consumed thousands of real state/scan samples |
| Simple Numerical reached identity ambiguity but emitted a relation request with contradictory roles | NumericalResolver → EvidenceNeed | NumericalResolver | Mapped identity-targeted CountGraph requests to `IDENTITY_DISAMBIGUATION`, clearing relation anchors/predicate | Same simple case reached two arrivals and `/numerical_response` |
| Object Reference raised `unexpected keyword argument persistent_relation_summary` | LeanPipeline → tuple evaluator call | LeanPipeline call contract | Removed one stale keyword; persistent evidence remains owned by RelationEngine | Both Object Reference cases ran the full semantic/evidence loop without the exception |
| Relation probe had a canonical subject ID, but `EvidenceNeed.target_ids` was empty | Object/Instruction resolver → shared coordinator | Resolver contract normalization | Reused existing `probe_source_candidate_id` when the candidate list was absent | Hotel ObjectRef required `[target, anchor]`; missing-target transactions became incomplete rather than false-complete; Instruction passed |
| Canonical “and then to …” compiled as one terminal constraint | Language question → TaskIR | TaskIR compiler | Normalized the elided second verb to `then go to` before clause splitting | Same Q4 compiled to two constraints and both advanced from actual trajectory |
| Instruction reached FINALIZABLE/COMMIT but evaluator never saw terminal lifecycle | Authorized RootDecision → ROS status latch | ROS lifecycle latch | Relayed `arrival_revalidated` only after authorized Instruction COMMIT with `episode_complete=true` | Same Q4 terminated as `instruction_route_complete` at 149.916 s |

Repair commits after the freeze point:

- `3973cb7` — live ROS ingestion and Numerical identity-evidence routing;
- `c5a1ee1` — relation tuple evaluation call contract;
- `2e0338f` — probe target IDs in EvidenceNeed;
- `5cff727` — ordered Instruction compilation and terminal lifecycle.

## G. Remaining blockers

1. Simple Object Reference does not yet turn repeated fresh target/anchor observations into one unique relation-eligible canonical object. This is a shared evidence/SceneMemory association/RelationEngine closure blocker, not an OutputAdapter failure.
2. Simple Numerical closes and publishes, but its count is semantically wrong (`1` versus `4`), so perception/identity/count-domain completeness is not yet reliable.
3. Difficult relational Numerical obtains useful relation YES evidence but cannot merge or disambiguate the singular TV-cabinet anchor before the deadline; CountGraph therefore correctly withholds the answer.

No architecture redesign is proposed here.

## H. Decision

`RUNTIME_CLOSURE_NOT_YET_ACCEPTED`

Unique reason: the mandatory real three-task closure gate is not fully satisfied—Object Reference never produced its authorized marker, and Numerical correctness/relational identity closure is not yet reliable under the same real runtime.
