SUPERSEDED — HISTORICAL REFERENCE ONLY

# EVE v3.1 Dependency Map

Status: Draft for reviewer handoff  
Scope: Round641 through Round1020, plus next boundary work  
Current known baseline: PR #78, `1785 passed`

---

## 0. Purpose

This document exists because a large middle section of EVE v3 work was completed across many rounds after the earlier v3 baseline.

Known review gap:

```text
Known by earlier reviewer: round1~54, round71
Needs dependency review:   round641~1020
Current active line:       affect ladder + sensory/visual read-only contracts
```

The goal is not to re-explain every file. The goal is to show:

1. Which modules are read-only surfaces.
2. How data is intended to flow.
3. Which modules must not mutate runtime state.
4. Which invariants are already protected by tests/reports.
5. Which boundary modules are still missing.

---

## 1. Core EVE v3.1 Invariants

These are project-level invariants and must remain true unless an explicit operator-authorized activation round says otherwise.

```text
EVE is not a generic chatbot.
EVE is a deterministic Korean local digital-being project.
Runtime EVE core must remain non-LLM.
External LLM/code agents are development tools only.
```

Hard invariants:

```text
No uncontrolled runtime mutation.
No hidden persistence activation.
No memory write without gate/quarantine/preflight.
No self-model update from one observation.
No affect/hormone direct mutation from sensory input.
No AGP bypass.
No fallback bypass.
No vector/vocab/subset artifact read/load unless explicitly authorized.
No operator artifact/vector/zip/part file commits.
No test weakening, skipping, xfail, or deletion to force green.
Korean fixtures and literal "민석" must remain preserved.
```

Production defaults:

```text
runtime_mapping_enabled = false
runtime_mapping enforcement = disabled
persistence = NO-GO
runtime default = no-load
vector_contents_read = false
vectors_loaded = false
```

---

## 2. Current Round Band Summary

### 2.1 Earlier reviewed baseline

```text
round1~54:
- initial v3 constitution/skeleton/AGP baseline
- deterministic principles
- early tests and design constraints

round71:
- reviewed by earlier participant
- approximate known test count: 1154
```

### 2.2 Middle section needing dependency visibility

```text
round641~980:
- affect/hormone read-only ladder
- proposal, validation, payload, handoff, bridge, checkpoint/rollback dry-run surfaces
- future dry-run simulation request/preflight surfaces

round981~1000:
- sensory observation contract
- read-only multimodal sensory entry policy

round1001~1020:
- visual observation schema
- read-only external visual schema derived from sensory contract
```

Current PR status at time of this document:

```text
PR #77: sensory observation contract, 1727 passed
PR #78: visual observation schema, 1785 passed
```

---

## 3. Affect Ladder Dependency Map: Round641~980

The affect ladder is intended to remain read-only until a later explicit activation sequence.

Conceptual flow:

```text
read_only_appraisal_input
→ emotion transition contract
→ emotion transition validator
→ emotion transition gate
→ operator-authorized dry-run apply plan
→ affect/hormone neural rhythm registry
→ event→axis proposal map
→ affect event proposal validator
→ proposal→transition payload builder
→ operator handoff
→ reviewed payload dry-run bridge
→ checkpoint/rollback plan
→ checkpoint/rollback execution dry-run trace
→ operator decision packet
→ future dry-run simulation request packet
→ future dry-run simulation runner preflight
```

Known module chain:

```text
adapters/affect_event_proposal_validator.py
adapters/affect_proposal_transition_payload_builder.py
adapters/affect_transition_payload_operator_handoff.py
adapters/affect_reviewed_payload_dryrun_bridge.py
adapters/affect_transition_checkpoint_rollback_plan.py
adapters/affect_transition_checkpoint_rollback_dryrun_trace.py
adapters/affect_dryrun_trace_operator_decision_packet.py
adapters/affect_future_dryrun_simulation_request_packet.py
adapters/affect_future_dryrun_simulation_runner_preflight.py
```

The intended direction is one-way:

```text
proposal
→ validate
→ payload
→ handoff
→ bridge
→ plan
→ trace
→ decision
→ future request
→ future preflight
```

Required non-permissions across this chain:

```text
live_apply_allowed = false
dryrun_apply_executed = false unless explicitly dry-run trace only
memory_write_performed = false
self_model_update_allowed = false
runtime_mutation_performed = false
persistence_write_performed = false
vector_read_performed = false
vector_load_performed = false
agp_bypass_allowed = false
fallback_bypass_allowed = false
artifact_created_or_staged = false
```

Interpretation:

```text
The ladder prepares, validates, gates, and simulates.
It must not actually mutate live affect/hormone state yet.
```

---

## 4. Affect Ladder Operator Reports

The operator scripts are designed as compact JSON proof surfaces. They should report schema status and invariant proof, not mutate files or runtime state.

Known validation chain:

```text
scripts/operator_verify_round601_620_baseline.py
scripts/operator_lock_round621_640_baseline.py
scripts/operator_audit_round641_660_appraisal_agp_input.py
scripts/operator_report_round681_700_emotion_transition_contract.py
scripts/operator_validate_round701_720_emotion_transition_payloads.py
scripts/operator_gate_round721_740_emotion_transition.py
scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py
scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py
scripts/operator_report_round781_800_affect_event_proposal_map.py
scripts/operator_validate_round801_820_affect_event_proposals.py
scripts/operator_build_round821_840_affect_transition_payloads.py
scripts/operator_handoff_round841_860_affect_transition_payload_review.py
scripts/operator_bridge_round861_880_affect_reviewed_payload_to_dryrun.py
scripts/operator_plan_round881_900_affect_checkpoint_rollback.py
scripts/operator_trace_round901_920_affect_checkpoint_rollback_dryrun.py
scripts/operator_decision_round921_940_affect_dryrun_trace_packet.py
scripts/operator_request_round941_960_affect_future_dryrun_simulation.py
scripts/operator_preflight_round961_980_affect_future_dryrun_runner.py
```

Expected report properties:

```text
read-only
path-metadata-only when inspecting repository state
no write side effects
no audit file write unless explicitly authorized
no artifact creation
exactly one next recommendation where required
```

Reviewer check:

```text
Confirm each operator script emits proof data only.
Confirm no script creates hidden artifacts.
Confirm no script enables persistence/runtime mapping/enforcement.
```

---

## 5. Sensory Contract Map: Round981~1000

PR #77 added the first formal sensory entry contract.

Primary file:

```text
adapters/sensory_observation_contract.py
```

Supporting files:

```text
scripts/operator_report_round981_1000_sensory_observation_contract.py
docs/round981_1000_sensory_observation_contract.md
tests/test_v3_round981_1000_sensory_observation_contract.py
```

Conceptual role:

```text
text / visual / auditory / internal_state / tool_state / multimodal
→ normalized sensory observation
→ AGP input plan
→ memory candidate plan
```

It must not do:

```text
camera activation
microphone activation
OCR
speech-to-text
image/audio model load
raw image/audio/video/screen persistence
memory write
self-model update
affect/hormone transition
AGP/fallback bypass
runtime mutation
persistence/vector access
artifact creation
```

Supported modalities:

```text
text
visual
auditory
internal_state
tool_state
multimodal
```

Supported source families:

```text
visual:
- user_uploaded_image
- camera_frame_candidate
- screenshot_candidate
- screen_ui_candidate

auditory:
- voice_message_candidate
- microphone_frame_candidate
- speech_transcript_candidate
- environmental_sound_candidate

internal_state:
- battery
- temperature
- ram_pressure
- storage_pressure
- network_state
- process_health
- error_state
- latency_state

tool_state:
- file_change
- notification
- calendar_event
- telegram_event
- github_event
- test_result
- search_result
- app_state
```

Key invariant:

```text
Raw sensory data must not enter memory or thought directly.
Only normalized observations, event candidates, confidence metadata, privacy flags, and gated candidates may pass forward.
```

---

## 6. Visual Schema Map: Round1001~1020

PR #78 specialized the sensory contract for external visual inputs.

Primary file:

```text
adapters/visual_observation_schema.py
```

Supporting files:

```text
scripts/operator_report_round1001_1020_visual_observation_schema.py
docs/round1001_1020_visual_observation_schema.md
tests/test_v3_round1001_1020_visual_observation_schema.py
```

Conceptual role:

```text
visual source candidate
→ visual observation schema
→ sensory observation contract compatibility
→ event candidate plan
→ memory candidate plan
```

Supported visual sources:

```text
user_uploaded_image
camera_frame_candidate
screenshot_candidate
screen_ui_candidate
```

Observed feature groups:

```text
scene_features
object_features
person_presence_features
screen_ui_features
lighting_features
motion_features
safety_relevance_features
```

Inferred feature groups must remain candidate-only:

```text
person_identity_candidate
expression_candidate
posture_candidate
gaze_candidate
activity_candidate
place_candidate
screen_state_candidate
visual_salience_candidate
```

Visual schema must not assert:

```text
person identity
user emotion
user intent
relationship state
memory fact
```

Visual schema hard false flags:

```text
raw_data_persisted = false
raw_image_persisted = false
raw_video_persisted = false
raw_screen_recording_persisted = false
camera_activated = false
ocr_performed = false
vision_model_loaded = false
face_recognition_performed = false
memory_write_performed = false
long_term_memory_write_allowed = false
self_model_update_allowed = false
affect_transition_allowed = false
hormone_transition_allowed = false
agp_bypass_allowed = false
fallback_bypass_allowed = false
vector_read_performed = false
vector_load_performed = false
runtime_mutation_performed = false
persistence_write_performed = false
artifact_created_or_staged = false
```

Current verification reported:

```text
full suite: 1785 passed
focused visual tests: 58 passed
operator report: round1001_1020_read_only_visual_observation_schema_green
```

---

## 7. Read-Only vs Future Mutation-Capable Classification

### 7.1 Current read-only surfaces

These are design/test/contract surfaces only.

```text
Affect proposal and validator surfaces
Affect transition payload builder
Operator handoff surfaces
Checkpoint/rollback plan surfaces
Dry-run trace surfaces
Decision packet surfaces
Future simulation request/preflight surfaces
Sensory observation contract
Visual observation schema
Operator reports
Focused invariant tests
Documentation files
```

### 7.2 Explicitly not active yet

```text
Real affect/hormone apply
Real checkpoint creation
Real rollback execution
Real audit write
Real memory write
Real self-model update
Real camera/microphone activation
Real OCR/STT
Real image/audio model inference
Real face recognition
Real vector load
Real persistence activation
Real AGP/fallback route mutation
Real autonomous sensory loop
```

### 7.3 Future mutation-capable areas requiring separate activation ladders

```text
bounded affect/hormone apply
bounded memory write
bounded self-model update
camera/microphone adapters
OCR/STT adapters
screen/UI parser
file/tool action adapters
Telegram/GitHub action adapters
virtual world state mutation
long-running autonomous loop
```

Every mutation-capable area needs:

```text
read-only schema
validator
dry-run/preflight
checkpoint/rollback plan
operator decision packet
focused tests
full suite
explicit activation authorization
```

---

## 8. Invariant Coverage Map

### 8.1 Covered by current affect ladder

```text
No direct live affect/hormone mutation
No AGP/fallback bypass
No memory/self write from affect proposals
No all-axis/global synchrony activation
Operator-gated dry-run/preflight surfaces
Checkpoint/rollback planning without real persistence
```

### 8.2 Covered by sensory contract

```text
No raw sensory persistence
No camera/mic activation
No OCR/STT
No model loads
No vector access
No direct memory/self/affect/hormone mutation
No AGP/fallback bypass
Observation plans require appraisal/AGP/memory gate
Hardware/internal observations remain operational-only
```

### 8.3 Covered by visual schema

```text
No raw visual persistence
No camera activation
No OCR
No vision model load
No face recognition
Candidate-only visual inference
No emotion/intent/relationship/memory-fact assertion
Privacy flags for face/person/screen/private scene
Visual-to-sensory compatibility
Visual-to-event candidate-only plan
Visual-to-memory gate/quarantine plan
```

### 8.4 Still missing coverage

```text
origin/fact_status policy
virtual visual observation schema
auditory observation schema
memory replay provenance
imagination/simulation boundary
cross-modal binding policy
spatiotemporal episode builder
sensory attention/rate limit
deduplication/cooldown policy
privacy guard as standalone module
memory source provenance guard
sensory memory compression/decay
action candidate gate
operator permission gate for tool actions
virtual world state schema
```

---

## 9. Known Missing Boundary: Origin and Fact Status

This is the next required boundary.

Problem:

```text
External visual observation exists.
Virtual visual observation is planned.
Memory replay, imagination, simulation, and dream/DMN surfaces are planned.
Without origin/fact_status, these can be confused.
```

Required origins:

```text
external_visual
external_audio
screen_visual
tool_state
internal_state
virtual_world_visual
memory_replay
imagination
simulation
dream_dmn
```

Required fact statuses:

```text
observed_external
observed_internal_virtual
reconstructed_memory
imagined_candidate
simulated_future
symbolic_visualization
```

Hard rule:

```text
Only observed_external may become an external-world event candidate.
All other fact statuses require boundary labels and must not assert external fact.
```

Examples:

```text
virtual_world_visual + 민석_avatar_far
≠ "민석이 실제로 멀어졌다"

memory_replay + remembered_expression
≠ "현재 민석이 그런 표정이다"

imagination + future_conversation
≠ "그 일이 일어날 것이다"

simulation + possible_failure
≠ "실제 실패가 발생했다"
```

---

## 10. Next Recommended Round

Recommended next task:

```text
Round1021-1040:
read_only_observation_origin_and_fact_status_policy
```

Why this before virtual visual:

```text
External visual schema is already present.
Virtual visual, memory replay, imagination, and simulation will all need a shared boundary.
If virtual visual is added first, it may duplicate origin/fact rules locally.
A central policy avoids later divergence.
```

Target modules:

```text
adapters/observation_origin_fact_status_policy.py
scripts/operator_report_round1021_1040_observation_origin_fact_status_policy.py
docs/round1021_1040_observation_origin_fact_status_policy.md
tests/test_v3_round1021_1040_observation_origin_fact_status_policy.py
```

Required policy surfaces:

```text
build_observation_origin_fact_status_policy_summary()
build_origin_fact_status_record(origin, fact_status, source=None, metadata=None)
validate_origin_fact_status_record(record)
build_external_fact_assertion_guard(record)
build_memory_contamination_guard(record)
build_origin_fact_status_compatibility_plan(observation)
observation_origin_fact_status_policy_summary()
```

Required guards:

```text
external_fact_assertion_guard
virtual_world_reality_boundary_guard
memory_replay_provenance_guard
imagination_memory_contamination_guard
simulation_result_boundary_guard
dream_dmn_boundary_guard
```

Hard false flags:

```text
external_fact_asserted = false unless origin/fact_status permits it
memory_write_performed = false
self_model_update_allowed = false
affect_transition_allowed = false
hormone_transition_allowed = false
runtime_mutation_performed = false
persistence_write_performed = false
vector_read_performed = false
vector_load_performed = false
artifact_created_or_staged = false
agp_bypass_allowed = false
fallback_bypass_allowed = false
```

---

## 11. Reviewer Checklist

A reviewer who did not see Round641~1020 should check:

```text
1. Are all new modules pure data / pure functions?
2. Do imports remain read-only and acyclic enough?
3. Does any operator script write files or create artifacts?
4. Does any path enable runtime_mapping, enforcement, persistence, or vector loading?
5. Do tests assert actual false flags, not just key presence?
6. Do visual/sensory schemas only produce candidates/plans?
7. Are memory/self/affect/hormone writes still impossible by default?
8. Is AGP/fallback bypass explicitly false everywhere?
9. Are Korean fixtures and "민석" preserved?
10. Is the next missing boundary clearly origin/fact_status?
```

---

## 12. Current Safe Merge Interpretation

Based on the latest reported state:

```text
PR #77 sensory contract: merge OK
PR #78 visual schema: merge OK
```

Reason:

```text
Both are read-only schema/test/report surfaces.
Both preserve no-device/no-model/no-raw-persistence/no-memory/no-affect/no-runtime invariants.
Both report full suite pass.
Both add focused invariant tests.
```

Caveat:

```text
PR body wording for git status should always be checked.
If a PR body says files are untracked after commit, verify actual PR changed_files and final branch status before merging.
```

---

## 13. Design Direction After Round1021-1040

Recommended sequence:

```text
Round1021-1040:
read_only_observation_origin_and_fact_status_policy

Round1041-1060:
read_only_virtual_visual_observation_schema

Round1061-1080:
read_only_auditory_observation_schema

Round1081-1100:
read_only_virtual_world_state_schema

Round1101-1120:
read_only_cross_modal_binding_policy

Round1121-1140:
read_only_sensory_attention_priority_ranker

Round1141-1160:
read_only_sensory_privacy_guard

Round1161-1180:
read_only_sensory_memory_candidate_gate

Round1181-1200:
read_only_sensory_thought_candidate_builder
```

Do not jump directly to:

```text
camera adapter
microphone adapter
OCR/STT
vision model integration
face recognition
memory writes
self-model writes
live affect/hormone apply
autonomous loop activation
```

until the boundary/gate layers above are complete.

---

## 14. Final Summary

Round641~1020 appears to have built a controlled read-only ladder:

```text
affect/hormone proposal chain
+ dry-run/preflight surfaces
+ sensory observation contract
+ visual observation schema
```

The next missing architecture boundary is not another sensory adapter. It is:

```text
origin / fact_status / external fact assertion guard
```

This boundary must be inserted before virtual visual, memory replay, imagination, simulation, auditory, and cross-modal binding expand the system.
