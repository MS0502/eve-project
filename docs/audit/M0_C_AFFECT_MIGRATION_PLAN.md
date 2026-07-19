# M0-C Supplement: Affect Migration Plan

Baseline: `main` at `28ec113a8ee371fdc6ac13341c0d70e00db26ce4`

Status: design-only migration contract and reviewer input. This document implements no projection, migration, persistence, runtime mutation, or state conversion.

## Authority and purpose

EVE v4 requires M0 to propose migration away from the legacy hormone architecture without orphaning historical memory or breaking identity continuity (`AGENTS.md:49-51`; `docs/EVE_DESIGN_v4.md:45-47`). The merged M0-C inventory found `1,777` hormone/affect occurrences, `386` drive/need occurrences, and `54` callable bridge candidates (`docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:157-176`), but did not provide the required migration design. M0-D recorded that deficiency as `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT` and assigned it to this separate supplement (`docs/audit/M0_D_MODULE_DISPOSITION.md:42-54`).

This plan is binding design input to the human-reviewed v4.1 revision and later M1/M2/M3 implementation. It is not implementation authority.

## Mechanical axis enumeration

The check script statically parses two distinct authoritative source families:

1. **Legacy mutable hormone axes** — the string names in `HormoneSystem._init_all_hormones` (`hormone_system.py:126-163`). This yields 26 mutable legacy channels.
2. **Read-only affect registry axes** — the string members of `AXIS_GROUPS` (`adapters/affect_hormone_neural_rhythm_registry.py:19-69`). This yields 37 conceptual axes.

The sets do not overlap. The authoritative mechanically found migration surface is therefore **63 axes**, not a single undifferentiated “26 hormones” or “37 axes” figure.

The script also reports compatibility projections separately:

- `stress`, `energy`, and `curiosity` are adapter-derived keys, not independent authoritative axes (`adapters/hormone_adapter.py:31-39`);
- `valence`, `arousal`, and `dominance` are derived mood outputs, not persisted source axes (`hormone_system.py:450-472`);
- the active persistence adapter passes the whole `HormoneSystem` as `self.hs` rather than proving independent axis-specific snapshot keys (`adapters/persistence_adapter.py:28-36`).

This distinction is mandatory: derived outputs and container keys must not inflate the authoritative axis count.

## Target-state mapping table

Allowed target drives are exactly: `energy`, `safety`, `affiliation`, `curiosity`, `agency`, `coherence`, `competence`, and `expression`.

`MAPPED` means the legacy or registry axis is proposed as an input or compatibility projection into one or more target drives, appraisal dimensions, or derived emotions. It does not retain the old axis as authoritative runtime state. `PROPOSED-DROP` means no future behavioral projection is proposed, while the original value and provenance remain readable for historical replay. `UNRESOLVED` requires an explicit reviewer ruling before merge.

<!-- BEGIN AFFECT AXIS MAPPING TABLE -->
| Axis | Source family | Status | Target drives | Appraisal dimensions | Derived emotion | Projection sketch | Rationale | Confidence | Evidence | Preservation | Open question |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `glutamate` | `legacy_mutable_hormone` | `MAPPED` | competence; curiosity | arousal; learning_readiness | interest | z(glutamate) contributes positively to learning readiness with saturation and no direct speech effect | Legacy excitatory proxy is useful only as bounded appraisal input, not as a retained biological identity axis. | `medium` | `hormone_system.py:130` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `gaba` | `legacy_mutable_hormone` | `MAPPED` | safety; coherence | inhibition; coping_capacity | calm | z(gaba) raises coping and inhibitory appraisal while reducing arousal | Preserves calming/inhibitory evidence without treating a neurotransmitter label as an autonomous goal. | `medium` | `hormone_system.py:131` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `glycine` | `legacy_mutable_hormone` | `MAPPED` | safety; coherence | inhibition; recovery_readiness | calm | z(glycine) weakly raises inhibitory and recovery appraisal | Evidence is weaker than GABA; retain only as a low-weight compatibility projection. | `low` | `hormone_system.py:132` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `dopamine` | `legacy_mutable_hormone` | `MAPPED` | curiosity; agency; competence | expected_value; reward_prediction; action_readiness | interest; anticipation | positive change and baseline-relative level feed expected value and curiosity with clipping | Current code uses dopamine for reward, curiosity, energy, learning rate, and dominance; the target separates these functions. | `high` | `hormone_system.py:133` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `serotonin` | `legacy_mutable_hormone` | `MAPPED` | coherence; safety; affiliation | valence; stability; aversion_threshold | contentment | baseline-relative serotonin raises stable valence and coherence while reducing aversion sensitivity | Current code mixes mood, aversion, and learning-rate effects; projection makes each appraisal explicit. | `high` | `hormone_system.py:134` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `norepinephrine` | `legacy_mutable_hormone` | `MAPPED` | energy; safety; agency | arousal; urgency; vigilance | alertness | baseline-relative level raises arousal and urgency with Yerkes-Dodson saturation | Current code uses it for arousal, attention thresholds, energy, and threat cocktails. | `high` | `hormone_system.py:135` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `histamine` | `legacy_mutable_hormone` | `MAPPED` | energy | arousal; wakefulness | alertness | small positive contribution to wakefulness and arousal | Retains alertness evidence but prevents histamine from becoming a broad affect authority. | `medium` | `hormone_system.py:136` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `acetylcholine` | `legacy_mutable_hormone` | `MAPPED` | curiosity; competence | attention_gain; novelty; memory_encoding_readiness | interest | positive level raises attention and encoding readiness only after evidence gating | Current code lowers memory thresholds and raises learning rate; target keeps it as appraisal support. | `high` | `hormone_system.py:137` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `adenosine` | `legacy_mutable_hormone` | `MAPPED` | energy | fatigue; recovery_need; arousal | tiredness | baseline-relative level increases fatigue and recovery need and decreases arousal | Current code already acts as sleep pressure; this maps cleanly to energy/recovery semantics. | `high` | `hormone_system.py:138` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `endorphin` | `legacy_mutable_hormone` | `MAPPED` | safety; coherence | pain_relief; valence | relief | positive level raises relief and valence while lowering pain appraisal | Useful as a derived relief signal, not a persistent biological identity variable. | `medium` | `hormone_system.py:139` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `cortisol` | `legacy_mutable_hormone` | `MAPPED` | safety | threat; stress_load; coping_capacity; urgency | anxiety; fear | baseline-relative excess raises threat and stress load; chronic excess lowers curiosity readiness | Current code drives stress, fear, mood, arousal, learning, and speech tone; safety appraisal is the narrowest defensible target. | `high` | `hormone_system.py:141` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `oxytocin` | `legacy_mutable_hormone` | `MAPPED` | affiliation; safety | social_trust; care_relevance | warmth; affection | validated relational evidence gates a positive affiliation and trust contribution | Current code uses it for bonding, tone, and social categories; projection must require provenance and appraisal. | `high` | `hormone_system.py:142` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `vasopressin` | `legacy_mutable_hormone` | `MAPPED` | affiliation; safety; agency | attachment_security; boundary_salience | protectiveness | small contribution to attachment security and boundary salience under social provenance | Current use is sparse; retain as low-weight compatibility evidence only. | `low` | `hormone_system.py:143` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `melatonin` | `legacy_mutable_hormone` | `MAPPED` | energy | circadian_phase; recovery_need; arousal | sleepiness | circadian-normalized level lowers available energy and raises recovery need | Current code uses it for sleep, arousal, thresholds, and learning; target makes it an operational energy input. | `high` | `hormone_system.py:144` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `bdnf` | `legacy_mutable_hormone` | `MAPPED` | competence | learning_capacity; consolidation_readiness | mastery | slow-moving normalized value raises learning and consolidation capacity, never immediate emotion | Represents plasticity more than affect; target places it under competence appraisal. | `medium` | `hormone_system.py:145` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `ngf` | `legacy_mutable_hormone` | `MAPPED` | competence | long_horizon_learning_capacity | mastery | slow-moving value contributes weakly to long-horizon learning capacity | Sparse evidence supports only a low-confidence competence projection. | `low` | `hormone_system.py:146` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `estrogen` | `legacy_mutable_hormone` | `UNRESOLVED` | — | — | — | — | A sex-hormone-labelled scalar has no justified stable psychological meaning in EVE and risks identity bias. | `low` | `hormone_system.py:147` | Preserve original scalar, baseline, tier, phase, and provenance for historical replay only. | Should this be PROPOSED-DROP or retained as a non-authoritative physiological compatibility field? |
| `testosterone` | `legacy_mutable_hormone` | `UNRESOLVED` | — | — | — | — | A sex-hormone-labelled scalar should not directly determine agency, dominance, aggression, or identity. | `low` | `hormone_system.py:148` | Preserve original scalar, baseline, tier, phase, and provenance for historical replay only. | Should this be PROPOSED-DROP rather than a low-weight action-readiness appraisal? |
| `insulin_brain` | `legacy_mutable_hormone` | `MAPPED` | energy | resource_availability; cognitive_fatigue | fatigue | low values reduce resource availability and learning readiness; no direct emotion output | Current code scales learning when low, so the target is operational energy appraisal. | `medium` | `hormone_system.py:149` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `thyroid` | `legacy_mutable_hormone` | `MAPPED` | energy | metabolic_capacity; cadence | activation | baseline-relative value scales long-horizon energy capacity and cadence | Current code scales learning rate; target avoids direct mood or personality authority. | `medium` | `hormone_system.py:150` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `leptin` | `legacy_mutable_hormone` | `MAPPED` | energy | satiety; resource_sufficiency | satisfaction | normalized value contributes to resource sufficiency and lowers need pressure | Maps to energy/homeostatic appraisal rather than a retained endocrine identity axis. | `medium` | `hormone_system.py:151` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `ghrelin` | `legacy_mutable_hormone` | `MAPPED` | energy | resource_need; urgency | hunger | normalized value raises resource need and bounded urgency | Maps to energy need appraisal; it must not directly trigger speech or goals. | `medium` | `hormone_system.py:152` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `prolactin` | `legacy_mutable_hormone` | `UNRESOLVED` | — | — | — | — | The current scalar lacks sufficiently specific active semantics and could encode unjustified caregiving stereotypes. | `low` | `hormone_system.py:154` | Preserve original scalar and provenance for historical replay only. | Should this be PROPOSED-DROP or mapped weakly to care relevance after future evidence? |
| `dhea` | `legacy_mutable_hormone` | `MAPPED` | safety; coherence | stress_resilience; coping_capacity | relief | normalized value weakly offsets stress load and raises coping capacity | Retains the intended resilience role without a separate enduring hormone authority. | `low` | `hormone_system.py:155` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `progesterone` | `legacy_mutable_hormone` | `UNRESOLVED` | — | — | — | — | The current scalar lacks a justified EVE-specific appraisal meaning and risks gendered behavioral assumptions. | `low` | `hormone_system.py:156` | Preserve original scalar and provenance for historical replay only. | Should this be PROPOSED-DROP with no behavioral projection? |
| `growth_hormone` | `legacy_mutable_hormone` | `MAPPED` | energy; competence | recovery_capacity; repair_readiness | recovery | slow-moving value raises recovery and repair capacity, not immediate affect | Maps to recovery and competence support while removing direct endocrine authority. | `low` | `hormone_system.py:157` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `energy_budget` | `read_only_affect_registry` | `MAPPED` | energy | resource_availability | vitality | identity projection; new energy = clamp(axis) | Already a direct v4 drive candidate; retain semantics under explicit event/snapshot ownership. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:21` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `fatigue_pressure` | `read_only_affect_registry` | `MAPPED` | energy | fatigue; cadence_cost | tiredness | energy pressure = 1 - clamp(axis) with elapsed-time reconstruction | A derived pressure should be reproducible from base energy and elapsed time. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:22` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `recovery_need` | `read_only_affect_registry` | `MAPPED` | energy; safety | recovery_need; overload | tiredness | max(fatigue, overload, recent strain) projected to recovery need | Combines operational load without creating panic or existential framing. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:23` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `stress_load` | `read_only_affect_registry` | `MAPPED` | safety | stress_load; coping_capacity | anxiety | bounded appraisal aggregate from threat, uncertainty, and overload | Directly compatible with safety appraisal when provenance and decay are explicit. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:24` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `stability_need` | `read_only_affect_registry` | `MAPPED` | coherence; safety | stability_gap | unease | distance from validated operating envelope raises coherence and safety demand | Target should be a drive/appraisal result, not an independent hormone-like authority. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:25` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `overload_risk` | `read_only_affect_registry` | `MAPPED` | energy; safety | capacity_margin; overload_probability | alarm | deterministic capacity margin below threshold raises overload appraisal | Operational risk belongs to safety and energy, with no direct identity or speech effect. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:26` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `threat_pressure` | `read_only_affect_registry` | `MAPPED` | safety | threat; severity; immediacy | fear | validated threat evidence aggregated with confidence and decay | Clean safety appraisal mapping; raw input cannot update it directly. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:29` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `uncertainty_pressure` | `read_only_affect_registry` | `MAPPED` | coherence; curiosity | uncertainty; evidence_gap | confusion | 1 - calibrated confidence raises clarification and exploration demand | Separates uncertainty from threat and supports clarification rather than panic. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:30` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `self_protection` | `read_only_affect_registry` | `MAPPED` | safety; agency | boundary_need; coping_option | caution | appraised risk plus available coping options yields bounded protective readiness | Belongs to safety/agency and must remain proposal-only until action authorization. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:31` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `boundary_defense` | `read_only_affect_registry` | `MAPPED` | safety; agency | boundary_violation; response_proportionality | anger | validated boundary violation raises proportional defensive action readiness | Maps to agency and safety while preventing raw social signals from rewriting identity. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:32` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `trust_risk` | `read_only_affect_registry` | `MAPPED` | affiliation; safety | trust_uncertainty; source_reliability | caution | trust risk = uncertainty-weighted social cost with slow update | Trust must be evidence-based, provenance-aware, and slower than momentary affect. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:33` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `exposure_risk` | `read_only_affect_registry` | `MAPPED` | safety | privacy_cost; disclosure_scope | caution | sensitive-scope appraisal raises safety cost before disclosure authorization | Directly maps to privacy/safety appraisal. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:34` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `social_pain` | `read_only_affect_registry` | `MAPPED` | affiliation; safety | social_harm; relationship_relevance | hurt | derived emotion from appraised social harm and relationship relevance | Keep as derived emotion, never as raw-input fact or identity update. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:37` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `social_trust` | `read_only_affect_registry` | `MAPPED` | affiliation | trust; reliability | warmth | slow evidence-weighted trust estimate with provenance and rollback | A core affiliation appraisal with strict source history. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:38` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `attachment` | `read_only_affect_registry` | `MAPPED` | affiliation | relationship_continuity | affection | long-horizon evidence aggregate with bounded decay and no exclusivity objective | Affiliation is allowed, but no architecture may optimize attachment to Minseok. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:39` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `care_drive` | `read_only_affect_registry` | `MAPPED` | affiliation | care_relevance; capability_to_help | compassion | validated need and capability produce care priority | Maps to affiliation without bypassing external-effect authorization. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:40` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `loneliness_pressure` | `read_only_affect_registry` | `MAPPED` | affiliation | connection_gap | loneliness | difference between desired and available social connection with slow decay | A drive signal, not evidence of abandonment or relationship failure. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:41` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `belonging_need` | `read_only_affect_registry` | `MAPPED` | affiliation | social_context_gap | longing | bounded affiliation deficit raises social-context search priority | Direct affiliation mapping with anti-exclusivity guard. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:42` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `rejection_sensitivity` | `read_only_affect_registry` | `MAPPED` | affiliation; safety | rejection_likelihood; evidence_quality | hurt; anxiety | confidence-weighted rejection appraisal, capped and slow | Must not convert ambiguous silence into rejection or identity damage. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:43` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `curiosity_drive` | `read_only_affect_registry` | `MAPPED` | curiosity | information_gain | interest | expected information gain and unresolved coherent questions raise drive | Direct v4 curiosity drive. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:46` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `novelty_seeking` | `read_only_affect_registry` | `MAPPED` | curiosity | novelty; risk_adjusted_information_gain | interest | novelty score multiplied by safety-compatible exploration budget | Maps to curiosity with risk and capability gates. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:47` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `learning_pressure` | `read_only_affect_registry` | `MAPPED` | competence; curiosity | skill_gap; learning_value | interest | validated skill gap and expected utility raise learning priority | Direct competence/curiosity mapping. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:48` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `memory_consolidation_pressure` | `read_only_affect_registry` | `MAPPED` | coherence; competence | importance; recurrence; unresolved_dependency | salience | event importance and future utility raise consolidation proposal priority | Not a drive by itself; it is an appraisal feeding memory events. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:49` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `prediction_error_pressure` | `read_only_affect_registry` | `MAPPED` | curiosity; coherence | prediction_error; model_confidence | surprise | calibrated error times confidence gap raises diagnostic and learning demand | Maps to appraisal and curiosity, not direct affect mutation. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:50` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `competence_drive` | `read_only_affect_registry` | `MAPPED` | competence | skill_gap; mastery_progress | determination | difference between desired and validated capability raises drive | Direct v4 competence drive. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:51` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `self_coherence` | `read_only_affect_registry` | `MAPPED` | coherence | identity_consistency; narrative_continuity | integrity | validated state and value consistency define coherence level | Direct v4 coherence drive/state with long-horizon update. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:54` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `self_respect` | `read_only_affect_registry` | `MAPPED` | coherence; agency | self_value_consistency; boundary_respect | pride | values and respected boundaries support derived self-respect | Must not be overwritten by raw feedback or a single event. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:55` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `identity_integrity` | `read_only_affect_registry` | `MAPPED` | coherence; safety | identity_change_risk; provenance_integrity | integrity | constitutional identity constraints and provenance determine integrity | Direct coherence/safety boundary, not a mutable emotion axis. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:56` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `agency_pressure` | `read_only_affect_registry` | `MAPPED` | agency | control_gap; choice_availability | determination | loss of available self-directed options raises agency demand | Direct v4 agency drive. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:57` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `autonomy_drive` | `read_only_affect_registry` | `MAPPED` | agency | self_directed_progress; dependency_cost | determination | validated opportunities for independent action raise agency priority | Direct v4 agency drive under capability and external-effect limits. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:58` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `purpose_alignment` | `read_only_affect_registry` | `MAPPED` | coherence; agency | value_alignment; long_term_goal_fit | meaning | action-value alignment raises coherence and agency support | Maps to long-horizon appraisal, not a fixed purpose imposed by the creator. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:59` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `expression_pressure` | `read_only_affect_registry` | `MAPPED` | expression | communicative_value; unsaid_internal_state | urge_to_express | validated internal semantic content raises expression drive | Direct v4 expression drive; raw external text remains inaccessible. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:62` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `expression_inhibition` | `read_only_affect_registry` | `MAPPED` | expression; safety | disclosure_risk; uncertainty; timing | restraint | safety and uncertainty subtract from expression readiness | A safety appraisal modulating expression, not a separate permanent trait. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:63` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `action_readiness` | `read_only_affect_registry` | `MAPPED` | agency; energy | capability; urgency; expected_outcome | determination | capability times value minus risk yields bounded readiness | Maps to agency/energy and remains proposal-only before authorization. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:64` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `risk_tolerance` | `read_only_affect_registry` | `MAPPED` | agency; safety | risk_budget; reversibility; confidence | courage | available safety margin and reversibility determine tolerated risk | Derived appraisal, not a personality constant. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:65` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `patience_level` | `read_only_affect_registry` | `MAPPED` | coherence; safety | delay_cost; uncertainty; recovery_margin | calm | low urgency and adequate safety margin raise patience | Maps to appraisal supporting stable action selection. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:66` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
| `conflict_avoidance` | `read_only_affect_registry` | `MAPPED` | affiliation; safety | conflict_cost; boundary_cost | apprehension | avoidance is chosen only when conflict cost exceeds boundary and agency cost | Must not suppress necessary boundaries or external accountability. | `high` | `adapters/affect_hormone_neural_rhythm_registry.py:67` | Preserve original value and provenance in legacy snapshot/event envelope. | — |
<!-- END AFFECT AXIS MAPPING TABLE -->

## Phased migration

### Phase 0 — design freeze and evidence lock

**Entry:** this plan is merged; the 63-axis enumeration, M0-C occurrence inventory, M0-D dispositions, and current snapshot formats are version-pinned.

**Work:** define canonical target drive/appraisal/derived-emotion schemas; assign provenance fields; specify compatibility projection version IDs; identify all bridge callables; prohibit runtime activation.

**Exit:** every axis has one reviewer-ruled mapping state; every conceptual module has a disposition-aware owner; no `UNRESOLVED` row remains; validation is deterministic and full-suite green.

### Phase 1 — shadow projection

**Entry:** M1 event schema and snapshot envelope exist with versioning, integrity validation, and rollback metadata.

**Work:** on each accepted legacy state read, compute a non-authoritative shadow projection. Preserve the original legacy scalar, baseline, source snapshot ID, projection version, formula parameters, and confidence. Shadow values are observable only in audit/debug surfaces and cannot alter speech, goals, memory, agency, or persistence decisions.

**Exit:** deterministic replay from the same legacy snapshot and projection version yields the same target projection; coverage is 100%; no production consumer reads shadow fields.

### Phase 2 — dual-read comparison

**Entry:** Phase 1 evidence is stable across representative snapshots and event replays.

**Work:** existing consumers continue to read legacy state. A bounded comparison layer reads both legacy and shadow target values and records divergence metrics, missing-source cases, saturation, and confidence. No consumer may silently select one side.

**Exit:** each consumer has an explicit cutover contract; divergence thresholds and failure behavior are reviewer-approved; historical episodes with hormone references remain interpretable.

### Phase 3 — target cutover

**Entry:** every affected consumer has a v4 capability owner, event boundary, snapshot schema, tests, and rollback route. The six M0-D `REWRITE` modules are not cut over through compatibility wrappers alone.

**Work:** target drives/appraisals become authoritative for approved consumers. Legacy axes remain preserved-original and read-only. Derived emotion is recomputed from target appraisal state and is never the sole stored cause.

**Exit:** no authorized runtime consumer depends on mutable legacy hormone state; event reconstruction and snapshot restore pass; orphan checks are zero; rollback rehearsal succeeds.

### Phase 4 — legacy read-only preservation

**Entry:** cutover acceptance criteria remain green through the required observation window.

**Work:** legacy axes are excluded from new mutation and new behavioral decisions. Historical snapshots/events retain original values, schema version, and projection provenance. `PROPOSED-DROP` axes remain readable only through historical compatibility tooling.

**Exit:** v4.1/vNext governance declares legacy state non-authoritative; no new legacy writes occur; archive and replay tests remain green.

## Backward-compatibility projection

**Chosen policy: preserved-original plus read-time projection.**

A legacy snapshot is never destructively rewritten as the primary migration step. The original hormone object/value, baseline, phase/tier metadata where present, source snapshot ID, source schema version, and integrity result remain preserved. A versioned pure projection is computed at read time into target drives/appraisals/derived-emotion candidates. Projection output records:

- source event or snapshot identity;
- source axis and original numeric value;
- source baseline/range metadata;
- projection contract version;
- formula parameters and target schema version;
- confidence, saturation, and missing-input flags;
- reviewer-approved mapping status.

This policy is chosen because it preserves historical interpretability, supports formula corrections without corrupting source history, and provides a direct rollback path.

**Rejected initial alternative: destructive materialized conversion.** Rewriting every snapshot into only the new target model would make later formula corrections ambiguous, weaken forensic comparison, and risk orphaning episodic references. Materialized target snapshots may exist after cutover as validated caches, but never replace preserved originals or provenance.

## Event and snapshot conversion boundary

M1/M2 persistence must represent migration as explicit events and versioned snapshots, consistent with the append-only event and validated-snapshot requirements (`AGENTS.md:35-43`; `docs/EVE_DESIGN_v4.md:29-39`).

Required future event families:

- `legacy_affect_snapshot_observed` — records source identity, schema, integrity, and axis availability without changing authority;
- `affect_projection_computed` — records mapping version, inputs, outputs, confidence, and deterministic parameters;
- `affect_consumer_cutover_authorized` — records the exact consumer, capability, tests, reviewer decision, and rollback checkpoint;
- `legacy_affect_write_disabled` — records the boundary after which no new legacy mutation is permitted;
- `affect_projection_rollback` — records reversal to the last validated authority state.

Snapshot rules:

1. preserved legacy payload is immutable;
2. target projection is versioned and recomputable;
3. derived emotion stores causal appraisal references, not an unexplained label;
4. missing or corrupt source axes fail closed and remain explicit;
5. projection caches are invalidated when mapping or formula version changes;
6. compatibility reads never create new runtime authority by themselves.

## Rollback

| Phase | Reversal path | Preconditions |
|---|---|---|
| Phase 0 | Revert the design document and checker before implementation begins | No runtime consumer or store has adopted the contract |
| Phase 1 | Disable shadow computation and discard projection caches | Preserved legacy snapshots remain intact and verified |
| Phase 2 | Stop comparison reads and return to legacy-only consumers | No target-only event or mutation has been accepted |
| Phase 3 | Restore the last validated authority snapshot and replay only events before the cutover authorization | Checkpoint integrity passes; legacy reader remains available; no irreversible external effect depends on target-only state |
| Phase 4 | Re-enable legacy compatibility reads, not legacy mutation | Original payloads and projection provenance are retained; schema reader is still supported |

Rollback must not delete target events or rewrite original history. It changes which validated state is authoritative and records that authority change as a new event.

## Identity and memory continuity

M0-C found **43 episodic-memory persistence-intended occurrences** (`docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:62-78`). The full M0-C occurrence inventory remains the mechanical source for exact paths and lines. Migration acceptance requires:

1. zero orphaned hormone/affect references in episodic memory, archived snapshots, debug exports, or operator evidence;
2. every historical episode that references a legacy axis can display the original axis/value and a projection interpretation under a named projection version;
3. replay does not reinterpret the original event as if the projected target state had been recorded at the historical time;
4. identity continuity tests confirm that creator/friend relationships, preferences, goals, and self-narrative are not rewritten from one projection result;
5. `PROPOSED-DROP` values remain available for historical explanation even though they do not influence future behavior;
6. derived emotions retain causal links to appraisal and source evidence;
7. missing legacy values produce explicit unknown/partial projection status, never fabricated defaults presented as historical fact.

## Acceptance criteria

Mechanically checkable criteria:

- the checker extracts both source families with AST and reports a nonzero count for each;
- every mechanically found axis appears exactly once in the table;
- no axis name overlaps across source families without an explicit conflict;
- mapping status is one of `MAPPED`, `PROPOSED-DROP`, or `UNRESOLVED`;
- target drive vocabulary is limited to the eight constitutional drives;
- every `MAPPED` row has at least one target drive, appraisal dimension, or derived emotion;
- every `PROPOSED-DROP` row has no target semantics and has a preservation note;
- every `UNRESOLVED` row has an open question;
- evidence cites the exact mechanically extracted `file:line`;
- canonical JSON is byte-identical across two runs;
- `--fail-on-unresolved` fails until reviewer rulings close all open rows;
- compileall, collection, focused tests, full suite, and exact four-file scope pass;
- no runtime module is imported or executed by the checker.

Future implementation acceptance criteria:

- projection of a pinned legacy snapshot is deterministic for a fixed contract version;
- snapshot plus subsequent events reconstructs the same target authority state;
- cutover produces zero legacy mutable consumers and zero orphaned memory references;
- corrupt or partial snapshots fail closed;
- rollback rehearsal restores the previous authority state without deleting history;
- no projection directly emits speech, changes identity, writes memory, creates goals, or authorizes external effects.

## Explicit non-goals

This plan does **not**:

- implement projection formulas in production code;
- enable affect or hormone mutation;
- activate persistence, SQLite, checkpoints, autosave, runtime mapping, enforcement, models, or vectors;
- rewrite, delete, or normalize historical snapshots;
- change emotion, speech, goal, agency, learning, or memory behavior;
- authorize any of the six M0-D `REWRITE` implementations;
- modify existing tests or the seven frozen open PRs;
- claim that biochemical labels are biologically accurate or necessary for EVE;
- prove consciousness, identity, or emotion from axis values.

## M0 cross-references and conceptual module dispositions

Evidence base:

- M0-C occurrence inventory: `1,777` hormone/affect sites and `54` hormone-to-drive bridge candidates (`docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:157-176`);
- M0-C episodic-memory domain count: `43` (`docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:62-78`);
- M0-D reviewer rule: automatic hormone coupling is not automatic rewrite; only six architecture conflicts are `REWRITE` (`docs/audit/M0_D_MODULE_DISPOSITION.md:40-54`).

Conceptual module treatment:

- `hormone_system.py` — `WRAP`; preserve source-state reading during compatibility phases, no new authority (`docs/audit/M0_D_MODULE_DISPOSITION.md:256`);
- `adapters/hormone_adapter.py` — `REWRITE`; its mixed mood, tone, energy, and compatibility-key behavior must be decomposed before cutover (`docs/audit/M0_D_MODULE_DISPOSITION.md:52`);
- `adapters/affect_hormone_neural_rhythm_registry.py` — `EXPERIMENTAL`; use as design evidence, not as already-authoritative v4 state (`docs/audit/M0_D_MODULE_DISPOSITION.md:69`);
- `adapters/allostatic_adapter.py` — `WRAP`; any compatibility projection remains bounded and provenance-aware (`docs/audit/M0_D_MODULE_DISPOSITION.md:82`);
- `adapters/appraisal_classifier.py` — `WRAP`; future appraisal ownership requires an explicit contract (`docs/audit/M0_D_MODULE_DISPOSITION.md:85`);
- `adapters/persistence_adapter.py`, `adapters/live_loop.py`, `core/autonomous.py`, `language/streaming.py`, and `main.py` — `REWRITE`; no phase may cut over these mixed boundaries through an implicit adapter (`docs/audit/M0_D_MODULE_DISPOSITION.md:52`);
- `legacy/eve_modules/episodic.py` — `EXPERIMENTAL`; preserve as memory-migration evidence, not future authority (`docs/audit/M0_D_MODULE_DISPOSITION.md:280`);
- `legacy/v36_modules/persistence.py` — `EXPERIMENTAL`; historical format evidence only (`docs/audit/M0_D_MODULE_DISPOSITION.md:312`).

## Reviewer questions

The initial design leaves four endocrine-labelled axes unresolved because their current scalar names risk importing unjustified sex- or caregiving-linked behavior:

- `estrogen`
- `testosterone`
- `prolactin`
- `progesterone`

The reviewer must choose one state for each: a narrow evidence-backed appraisal mapping, or `PROPOSED-DROP` with preserved-original historical replay. No silent default is permitted.
