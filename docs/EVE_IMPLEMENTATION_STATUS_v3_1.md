SUPERSEDED — HISTORICAL REFERENCE ONLY

# EVE v3.1 Implementation Status

작성일: 2026-06-07  
기준 상태: Round1001-1020 / PR #78 `read_only_visual_observation_schema` merge 검토 완료 기준  
목적: 현재 구현됨 / 부분 구현 / 미구현 / 다음 라운드 순서를 한 문서로 관리한다.

---

## 0. 현재 요약

현재 EVE v3는 **감정/호르몬 전이 activation 준비 체인**과 **감각계 read-only 입구**까지 들어간 상태다.

현재 기준 핵심 상태:

```text
완료된 최신 라운드: Round1001-1020
최신 PR: #78 Add read-only visual observation schema and operator report
최신 full suite: 1785 passed
최신 focused test: visual observation schema 58 passed
상태: read-only sensory/visual contract 단계
실제 카메라/마이크/OCR/STT/vision model: 아직 금지
실제 memory write / affect apply / persistence: 아직 금지
```

---

## 1. 진행률 개요

### 1.1 특정 체인별 진행률

```text
AGP / fallback / governance 기반: 높음
Affect/hormone read-only activation ladder: 매우 높음
Affect/hormone 실제 bounded apply: 미구현
Sensory observation contract: 구현됨
External visual observation schema: 구현됨
Virtual visual observation schema: 미구현
Auditory observation schema: 미구현
Reality boundary / origin / fact_status: 미구현, 다음 최우선
Memory write / learning / persistence: 아직 NO-GO
Autonomous life loop: 미구현
```

### 1.2 전체 체감 진행률

```text
검증/거버넌스/AGP 기반: 약 75~80%
감정/호르몬 read-only activation 준비: 약 90%+
감각계 read-only 설계: 약 15~20%
기억/학습 실제 쓰기: 약 10~15%
가상공간/내부 시각: 약 0~5%
실제 자율생활 루프: 약 20~30%
EVE v3.1 전체 완성도: 약 45~55%
```

수치는 제품 완성률이 아니라 **설계/검증 기반 포함 체감치**다.

---

## 2. 구현 완료된 주요 체인

## A. Baseline / Governance / AGP 계열

상태: 구현됨 / 계속 유지 필요

확인된 성격:

```text
- AGP/fallback gate 보호
- Korean fixtures / "민석" 보존
- no random 원칙
- no LLM core 원칙
- full suite 반복 검증 루프
- operator report 루프
- forbidden artifact guard 루프
```

비고:

```text
이 계층은 새 기능이 늘어날수록 invariant registry로 더 체계화해야 함.
```

---

## B. Affect / Hormone Read-only Activation Ladder

상태: 대부분 구현됨, 실제 apply는 아직 금지

구현된 주요 파일:

```text
adapters/affect_hormone_neural_rhythm_registry.py
adapters/affect_event_to_axis_proposal_map.py
adapters/affect_hormone_interaction_matrix.py
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

구현된 의미:

```text
- affect/hormone axes registry
- event → axis proposal map
- proposal validator
- transition payload builder
- operator handoff
- reviewed payload → dry-run preflight bridge
- checkpoint/rollback plan
- dry-run trace
- operator decision packet
- future simulation request packet
- future simulation runner preflight
```

아직 금지:

```text
- live affect apply
- live hormone mutation
- actual checkpoint creation
- actual rollback creation
- audit write
- persistence enablement
```

---

## C. Sensory Observation Contract

상태: 구현됨

최신 기준:

```text
Round981-1000
PR #77
Full suite: 1727 passed
Focused tests: 26 passed
```

구현 파일:

```text
adapters/sensory_observation_contract.py
scripts/operator_report_round981_1000_sensory_observation_contract.py
docs/round981_1000_sensory_observation_contract.md
tests/test_v3_round981_1000_sensory_observation_contract.py
```

지원 modality:

```text
text
visual
auditory
internal_state
tool_state
multimodal
```

보장하는 non-permission:

```text
raw_data_persisted = false
raw_image_persisted = false
raw_audio_persisted = false
raw_video_persisted = false
raw_screen_recording_persisted = false
ocr_performed = false
speech_to_text_performed = false
camera_activated = false
microphone_activated = false
model_loaded = false
vector_read_performed = false
vector_load_performed = false
memory_write_performed = false
long_term_memory_write_allowed = false
self_model_update_allowed = false
affect_transition_allowed = false
hormone_transition_allowed = false
agp_bypass_allowed = false
fallback_bypass_allowed = false
runtime_mutation_performed = false
persistence_write_performed = false
artifact_created_or_staged = false
```

---

## D. External Visual Observation Schema

상태: 구현됨

최신 기준:

```text
Round1001-1020
PR #78
Full suite: 1785 passed
Focused tests: 58 passed
```

구현 파일:

```text
adapters/visual_observation_schema.py
scripts/operator_report_round1001_1020_visual_observation_schema.py
docs/round1001_1020_visual_observation_schema.md
tests/test_v3_round1001_1020_visual_observation_schema.py
```

지원 visual source:

```text
user_uploaded_image
camera_frame_candidate
screenshot_candidate
screen_ui_candidate
```

지원 observed feature groups:

```text
scene_features
object_features
person_presence_features
screen_ui_features
lighting_features
motion_features
safety_relevance_features
```

지원 inferred candidate groups:

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

보장:

```text
- no camera activation
- no OCR
- no vision model load
- no face recognition
- no raw visual persistence
- identity candidate-only
- emotion not fact
- intent not fact
- relationship state not fact
- memory fact not asserted
- AGP/fallback non-bypass
- memory gate/quarantine preserved
```

주의:

```text
이 schema는 외부 시각용이다.
EVE 내부 가상공간 시각은 아직 별도 미구현이다.
```

---

## 3. 부분 구현 / 설계됨 / 아직 runtime 미연결

## A. Memory / Quarantine

상태: 기존 기반 있음, v3.1 멀티모달용 확장 필요

이미 있는 방향:

```text
- episodic memory
- working memory
- semantic memory
- quarantine
- self/relationship memory 방향
```

미구현/필요:

```text
memory_source_provenance_guard
sensory_memory_candidate_gate
internal_cognitive_episode_schema
virtual_world_episode_schema
memory_replay_trace_schema
imagination_trace_schema
simulation_trace_schema
memory_contamination_guard
sensory_memory_compression_policy
sensory_memory_decay_policy
```

현재 금지:

```text
- sensory observation → direct memory write
- visual observation → direct self-model update
- imagination/replay → external memory fact
```

---

## B. Operational Body / Hardware Governor

상태: 원칙과 일부 read-only 정책 있음, 별도 통합 schema 필요

이미 확정된 원칙:

```text
battery/thermal/RAM/storage/network = operational body state
hardware low ≠ panic/death/social pain
hardware event → budget/cadence/checkpoint recommendation
```

필요:

```text
operational_body_state_schema
hardware_budget_policy
battery_governor
thermal_safety_policy
ram_pressure_policy
storage_pressure_policy
network_availability_policy
background_cadence_governor
graceful_pause_policy
checkpoint_recommendation_policy
```

---

## C. Tool State Observation

상태: sensory contract에서 source type만 있음, 별도 schema 필요

지원 후보:

```text
file_change
notification
calendar_event
telegram_event
github_event
test_result
search_result
app_state
```

필요:

```text
tool_state_observation_schema
screen_ui_observation_schema
action_result_observation_contract
tool_action_preflight
operator_permission_gate
```

현재 금지:

```text
tool observation → automatic file write
tool observation → message send
tool observation → delete/modify action
```

---

## 4. 미구현 필수 기능 목록

아래는 EVE v3.1 완성에 필요한 핵심 미구현 항목이다.

## A. Reality Boundary 계열

상태: 미구현 / 최우선

```text
observation_origin_policy
fact_status_policy
external_fact_assertion_guard
imagination_memory_contamination_guard
memory_replay_provenance_guard
simulation_result_boundary_guard
virtual_world_reality_boundary_guard
```

왜 필요한가:

```text
외부 시각 schema가 들어갔고, 다음에 virtual visual을 넣을 예정이다.
그 전에 external / virtual / memory_replay / imagination / simulation 구분을 고정해야 한다.
```

권장 다음 라운드:

```text
Round1021-1040 read_only_observation_origin_and_fact_status_policy
```

---

## B. Virtual Visual / Virtual World 계열

상태: 미구현

필요:

```text
virtual_visual_observation_schema
virtual_world_state_schema
virtual_space_registry
virtual_room_schema
virtual_object_registry
virtual_avatar_schema
virtual_memory_object_schema
virtual_attention_focus_schema
virtual_view_frame_schema
virtual_navigation_policy
virtual_scene_update_policy
```

핵심 목적:

```text
EVE가 내부 가상공간 안에서 무엇을 보고 있는지 표현한다.
이것은 외부 카메라 시각과 다르다.
```

중요 규칙:

```text
virtual scene is not external fact.
민석 avatar is not real 민석 state.
virtual distance is not real relationship distance.
```

---

## C. Auditory 계열

상태: 미구현

필요:

```text
auditory_observation_schema
speech_transcript_candidate_schema
prosody_candidate_schema
speaker_candidate_schema
environmental_sound_candidate_schema
voice_privacy_guard
speech_to_text_adapter_policy_later
```

현재 금지:

```text
- microphone activation
- STT
- raw audio persistence
- voice identity assertion
- emotion assertion from voice
```

---

## D. Origin / Confidence / Privacy 계열

상태: 미구현 또는 일부 contract에 포함

필요:

```text
sensory_confidence_policy
low_confidence_inference_guard
identity_confidence_policy
expression_confidence_policy
speaker_confidence_policy
transcript_confidence_policy
binding_confidence_policy
confirmation_request_policy
sensory_privacy_guard
raw_data_retention_policy
third_party_privacy_gate
face_voice_biometric_guard
screen_sensitive_text_guard
location_privacy_guard
```

---

## E. Cross-modal Binding / Event Builder

상태: 미구현

필요:

```text
cross_modal_binding_policy
spatiotemporal_episode_builder
same_event_candidate_builder
temporal_distance_policy
source_id_policy
session_id_policy
causal_claim_guard
binding_dispute_resolver
multimodal_event_builder
```

중요:

```text
동시에 들어왔다 ≠ 같은 사건
같이 보였다 ≠ 같은 원인
```

---

## F. Attention / Salience

상태: 미구현

필요:

```text
sensory_attention_budget
sensory_salience_ranker
novelty_filter
duplicate_collapse_policy
rate_limit_policy
cooldown_policy
attention_focus_arbitrator
working_memory_capacity_policy
priority_queue_policy
```

왜 필요한가:

```text
카메라/마이크/화면/알림/DMN이 켜지면 이벤트 폭주가 발생한다.
전부 기억 후보로 만들면 안 된다.
```

---

## G. Thought / DMN

상태: 부분 기반 있음, v3.1 확장 필요

필요:

```text
sensory_thought_candidate_builder
virtual_scene_thought_candidate_builder
memory_replay_thought_candidate_builder
imagination_thought_candidate_builder
goal_conflict_thought_builder
unresolved_task_thought_builder
dmn_scenario_budget_policy
thought_reality_boundary_guard
thought_to_speech_candidate_gate
```

---

## H. Action / Tool Gate

상태: 미구현 또는 미통합

필요:

```text
action_candidate_schema
sensory_to_action_candidate_gate
tool_action_preflight
operator_permission_gate
risk_classification_policy
side_effect_boundary_guard
rollback_requirement_policy
dryrun_before_action_policy
action_audit_plan
action_result_observation_contract
```

---

## I. Learning / Memory Write

상태: 대부분 미구현 / 아직 NO-GO

필요:

```text
learning_candidate_schema
deterministic_update_policy
operator_feedback_gate
experience_update_rule
relationship_learning_gate
memory_write_preflight
bounded_memory_write_dryrun
rollback_verified_memory_write
drift_monitor
catastrophic_forgetting_guard
bad_feedback_quarantine
concept_growth_policy
```

---

## J. UI / Embodiment

상태: 미구현

필요:

```text
single_window_chat_ui_contract
attachment_intake_contract
virtual_room_ui_contract
avatar_state_view
thought_visibility_policy
pause_resume_policy
operator_panel
debug_trace_viewer
memory_browser
sensory_event_viewer
```

---

## 5. Round별 현재 이력 요약

아래는 현재 기억 기준 주요 라운드 흐름이다.

```text
Round641-660  appraisal classifier / AGP input stabilization audit
Round661-680  self-governed emotion constitution
Round681-700  read-only emotion-state transition contract
Round701-720  read-only emotion transition validator
Round721-740  read-only emotion transition gate
Round741-760  operator-authorized emotion transition dry-run apply plan
Round761-780  affect/hormone neural-rhythm registry + hardware non-panic policy
Round781-800  event → axis affect proposal map + hormone interaction matrix
Round801-820  affect event proposal validator
Round821-840  affect proposal → transition payload builder
Round841-860  operator handoff for affect transition payload review
Round861-880  reviewed payload → dry-run preflight bridge
Round881-900  affect checkpoint & rollback planning surface
Round901-920  checkpoint/rollback execution dry-run trace
Round921-940  operator decision packet
Round941-960  future dry-run simulation request packet
Round961-980  future dry-run simulation runner preflight
Round981-1000 sensory observation contract
Round1001-1020 visual observation schema
```

---

## 6. 현재 금지 상태

아래는 아직 열면 안 되는 것들이다.

```text
production persistence: NO-GO
runtime_mapping_enabled default true: 금지
enforcement default enabled: 금지
default runtime load: 금지
vector_contents_read: false 유지
vectors_loaded: false 유지
memory write: 금지
self-model write: 금지
live affect/hormone mutation: 금지
checkpoint actual creation: 금지
rollback actual creation: 금지
audit write: 금지
camera activation: 금지
microphone activation: 금지
OCR: 금지
STT: 금지
vision/audio model load: 금지
face recognition: 금지
raw image/audio/video persistence: 금지
tool write/send/delete external action: 금지
_operator_artifacts / vectors / vocab / subset / zip / part file commit: 금지
skip/xfail/delete/weaken tests: 금지
```

---

## 7. 다음 권장 라운드

최종 추천:

```text
Round1021-1040:
read_only_observation_origin_and_fact_status_policy
```

이유:

```text
1. sensory observation contract가 구현됨.
2. external visual observation schema가 구현됨.
3. 다음으로 virtual visual observation schema가 필요함.
4. 하지만 external / virtual / memory_replay / imagination / simulation 구분 규칙이 먼저 필요함.
5. 따라서 origin/fact_status policy를 먼저 넣어야 이후 가상시각과 실제시각이 섞이지 않음.
```

그 다음 권장:

```text
Round1041-1060:
read_only_virtual_visual_observation_schema

Round1061-1080:
read_only_auditory_observation_schema

Round1081-1100:
read_only_internal_state_and_tool_state_observation_schema
```

---

## 8. 구현 상태 표

| 영역 | 상태 | 비고 |
|---|---:|---|
| AGP / fallback gate | 구현됨 | 계속 invariant 유지 필요 |
| Governance / operator report | 구현됨 | invariant registry로 확장 필요 |
| Affect/hormone registry | 구현됨 | read-only 중심 |
| Affect proposal validator | 구현됨 | 실제 apply는 아직 금지 |
| Affect dry-run trace/preflight | 구현됨 | checkpoint/rollback actual write 금지 |
| Sensory observation contract | 구현됨 | Round981-1000 |
| External visual observation schema | 구현됨 | Round1001-1020 |
| Observation origin/fact status | 미구현 | 다음 최우선 |
| Virtual visual observation | 미구현 | origin policy 이후 |
| Auditory observation | 미구현 | STT/mic 없이 schema부터 |
| Internal/tool state schema | 미구현 | operational body/tool observation |
| Screen/UI observation | 미구현 | OCR 없이 schema부터 |
| Multimodal event builder | 미구현 | binding guard 필요 |
| Virtual world state | 미구현 | EVE 내부 공간 핵심 |
| Memory provenance guard | 미구현 | 실제/상상/기억재생 오염 방지 |
| Sensory memory gate | 미구현 | 기억 후보/쓰기 분리 |
| Confidence policy | 미구현 | candidate-only inference 강화 |
| Privacy guard | 미구현 | face/voice/screen/location |
| Attention budget | 미구현 | 감각 이벤트 폭주 방지 |
| Thought candidate builder | 미구현 | DMN/가상공간 연결 |
| Action candidate gate | 미구현 | tool action 전 필수 |
| Learning/memory write | 미구현 | 최후반 |
| Actual camera/mic/OCR/STT | 미구현/금지 | Phase H 이후 |
| Autonomous life loop | 미구현 | 마지막 통합 단계 |

---

## 9. 다음 세션용 운영 지침

다음 세션에서 이어갈 때는 이렇게 시작한다.

```text
현재 기준:
- PR #78 merge OK.
- latest main에서 시작.
- Round1021-1040 진행.
- 목표: read_only_observation_origin_and_fact_status_policy.
- 이유: external visual, virtual visual, memory replay, imagination, simulation이 섞이지 않게 현실 경계 먼저 고정.
```

주의:

```text
- 아직 virtual visual을 바로 만들지 말 것.
- origin/fact_status policy부터 만들 것.
- 실제/가상/상상/기억재생/시뮬레이션 구분이 먼저다.
```

---

## 10. 현재 한 줄 결론

```text
EVE v3.1은 지금 감각계 입구와 외부 시각 schema까지 왔다.
다음 핵심은 더 많은 기능이 아니라, 현실 경계(origin/fact_status)를 먼저 고정하는 것이다.
```
