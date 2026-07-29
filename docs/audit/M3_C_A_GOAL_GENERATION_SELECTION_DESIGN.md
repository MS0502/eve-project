# M3-C-A Goal Generation and Selection Design

Baseline: `a9f70ef78b06744eba01a0b35c60371b10eaf672` — M2-E A-2 cutover squash merge (#215)

Status: reviewer-ruled, **documentation/checker/test only**. No goal runtime integration, no legacy goal-domain migration, no live drive mutation, no action execution, no scheduler activation, no speech generation, and no M3-E affect cutover.

## Authority and scope

M3-C-A is the first post-cutover goal-design slice. It integrates the already-merged M3-A eight-drive dynamics with deterministic goal proposal and selection while preserving every authority boundary established by #213/#215.

The persistence substrate available to future v4-native goal components is exactly the #215 digest-pinned event-kernel + SQLite authority:

```text
human authorization digest: 3844e4d0a836924eb881048d45d98d89d5041f87d15a836686119a2d8487efbf
event store role:            authoritative_persistence_substrate_for_v4_native_subsystems
m3_authority_open:           true
legacy runtime authority:    authoritative_per_domain_until_separate_domain_migration_gate
legacy goal authority:       unchanged until its own later migration gate
M3-E affect cutover:         false
```

This design does not claim that legacy `GoalManagement` has migrated. It remains the authority for the legacy goal domain until a separate implementation/migration gate explicitly transfers that domain.

Versions:

```text
eve.m3-c-a.goal-candidate.v1
eve.m3-c-a.goal-score.v1
eve.m3-c-a.goal-transition-predicate.v1
eve.m3-c-a.goal-selection-receipt.v1
eve.m3-c-a.goal-selection-check.v1
```

## Input contract

A future v4-native goal proposal may consume only EVE-internal semantic/affordance candidates plus provenance-bearing internal state. Raw external text is not a goal-generation capability.

Required inputs for one deterministic decision epoch:

1. the exact eight M3-A derived drive samples;
2. each sample's M3-A parameter/predicate version and replay-carried monotonic time relation;
3. validated internal semantic goal candidates with provenance and evidence digest;
4. bounded candidate attributes defined below;
5. the current goal-selection epoch and prior selected-candidate state, if any.

Continuous drive values and continuously recomputed scores are **not events**.

## Eight-drive integration table

<!-- BEGIN M3C DRIVE INTEGRATION TABLE -->
| Drive | M3-A source | Goal-selection interpretation | Direct action authority | Ruling | Open question |
|---|---|---|---|---|---|
| `energy` | `eve.m3-a.drive-dynamics.v1` | operating-capacity alignment; low energy may favor restorative candidates through negative alignment weights | none | `RESOLVED` | — |
| `safety` | `eve.m3-a.drive-dynamics.v1` | validated safety-margin alignment; threatened state may favor protective/recovery candidates | none | `RESOLVED` | — |
| `affiliation` | `eve.m3-a.drive-dynamics.v1` | bounded social-continuity relevance without exclusive-person reward authority | none | `RESOLVED` | — |
| `curiosity` | `eve.m3-a.drive-dynamics.v1` | information-gap/exploration alignment | none | `RESOLVED` | — |
| `agency` | `eve.m3-a.drive-dynamics.v1` | self-directed option/readiness alignment under capability and safety constraints | none | `RESOLVED` | — |
| `coherence` | `eve.m3-a.drive-dynamics.v1` | narrative/state-consistency alignment and conflict-resolution preference | none | `RESOLVED` | — |
| `competence` | `eve.m3-a.drive-dynamics.v1` | learning/mastery alignment relative to validated capability | none | `RESOLVED` | — |
| `expression` | `eve.m3-a.drive-dynamics.v1` | internal-content expression relevance only; it cannot directly authorize speech | none | `RESOLVED` | — |
<!-- END M3C DRIVE INTEGRATION TABLE -->

All 59 `MAPPED` axes in the merged 63-axis Affect Migration Plan land only through their already-ruled drive/appraisal/emotion targets. The four `PROPOSED-DROP` axes retain historical provenance but contribute no goal score. M3-C-A adds no new axis and does not reinterpret a dropped axis.

## Goal candidate schema

A candidate is an internal proposal, not an active goal, action, speech command, or permission.

<!-- BEGIN M3C CANDIDATE FIELD TABLE -->
| Field | Domain | Meaning | Ruling | Open question |
|---|---|---|---|---|
| `semantic_goal_id` | non-empty stable internal semantic id | what internal outcome is proposed | `RESOLVED` | — |
| `decision_epoch` | non-negative integer | replay-stable proposal/selection epoch | `RESOLVED` | — |
| `evidence_digest` | SHA-256 | provenance-bearing evidence identity | `RESOLVED` | — |
| `base_value` | `[-1,1]` | non-drive candidate value before dynamic modulation | `RESOLVED` | — |
| `expected_value` | `[-1,1]` | bounded expected outcome appraisal | `RESOLVED` | — |
| `urgency` | `[0,1]` | bounded time-sensitive relevance, not wall-clock authority | `RESOLVED` | — |
| `continuity` | `[-1,1]` | consistency with validated prior goal/narrative state | `RESOLVED` | — |
| `cost` | `[0,1]` | bounded operating/resource cost | `RESOLVED` | — |
| `risk` | `[0,1]` | bounded validated risk estimate | `RESOLVED` | — |
| `drive_alignment[8]` | each `[-1,1]` | signed compatibility with each of the eight M3-A drives | `RESOLVED` | — |
| `drive_confidence[8]` | each `[0,1]` | confidence of each alignment contribution | `RESOLVED` | — |
<!-- END M3C CANDIDATE FIELD TABLE -->

Candidate identity is deterministic:

```text
candidate_id = sha256(
  candidate_schema || semantic_goal_id || decision_epoch ||
  evidence_digest || scoring_policy_version
)
```

No randomness, wall-clock timestamp, user identity shortcut, or raw external string participates in candidate identity.

## Deterministic scoring

For each M3-A drive `d` with continuous value `x_d` and design bounds `[L_d,U_d]`:

```text
z_d = clip((2*x_d - (L_d+U_d)) / (U_d-L_d), -1, 1)
```

For candidate `g`, signed drive alignment `w_gd∈[-1,1]`, and alignment confidence `q_gd∈[0,1]`:

```text
drive_term_g = sum(q_gd*w_gd*z_d) / max(1, sum(q_gd*abs(w_gd)))
```

The exact v1 score is:

```text
score_g = clip(
    0.30*base_value
  + 0.30*drive_term_g
  + 0.15*expected_value
  + 0.10*urgency
  + 0.10*continuity
  - 0.10*cost
  - 0.15*risk,
  -1, 1
)
```

All arithmetic uses finite bounded inputs. Selection is `argmax(score_g)` with lexical `candidate_id` as the deterministic tie-break. Sampling/random choice is forbidden.

## Versioned proposal and selection predicates

<!-- BEGIN M3C POLICY TABLE -->
| Policy item | Exact v1 value | Meaning | Ruling | Open question |
|---|---:|---|---|---|
| proposal enter threshold | `0.20` | absent candidate may become proposed only at/above this score | `RESOLVED` | — |
| proposal exit threshold | `0.10` | proposed/eligible candidate may withdraw only at/below this score | `RESOLVED` | — |
| selection minimum score | `0.30` | no candidate may become selected below this score | `RESOLVED` | — |
| initial winner margin | `0.08` | winner must exceed runner-up by at least this margin | `RESOLVED` | — |
| switch margin | `0.12` | challenger must exceed currently selected candidate by this margin | `RESOLVED` | — |
| selection cooldown seconds | `30` | replay-carried monotonic elapsed time before a selected-goal switch | `RESOLVED` | — |
<!-- END M3C POLICY TABLE -->

The `0.20/0.10` proposal thresholds provide hysteresis. The larger `0.12` switch margin plus cooldown prevents drive jitter from repeatedly changing the named selected state.

## Named lifecycle and event boundary

<!-- BEGIN M3C LIFECYCLE TABLE -->
| From | To | Versioned trigger | Event eligible | Authority effect |
|---|---|---|---|---|
| `absent` | `proposed` | valid provenance and `score >= 0.20` | yes | proposal only |
| `proposed` | `validated` | validator recomputes candidate id, evidence, drive sample, policy, and score | yes | validation only |
| `proposed` | `rejected` | validation fails | yes | none |
| `proposed` | `expired` | source evidence becomes stale | yes | none |
| `validated` | `eligible` | `score >= 0.30` | yes | selection eligibility only |
| `validated` | `rejected` | selection preconditions fail permanently | yes | none |
| `eligible` | `selected` | deterministic winner, margin `>=0.08`, cooldown satisfied | yes | names selected goal proposal only |
| `eligible` | `withdrawn` | `score <= 0.10` before selection | yes | none |
| `selected` | `superseded` | validated challenger margin `>=0.12` and cooldown satisfied | yes | prior selection loses selected state |
| `selected` | `expired` | selected evidence becomes stale before downstream activation | yes | selected state removed |
| `rejected` | `absent` | rejection acknowledged in next epoch | yes | none |
| `expired` | `absent` | expiry acknowledged in next epoch | yes | none |
| `withdrawn` | `absent` | withdrawal acknowledged in next epoch | yes | none |
| `superseded` | `absent` | supersession acknowledged in next epoch | yes | none |
<!-- END M3C LIFECYCLE TABLE -->

A `selected` record in M3-C-A is still only a selected **goal proposal**. It does not execute an action, mutate memory, schedule work, emit speech, or transfer legacy goal-domain authority. Those require later separately reviewed M3 implementation gates.

## A9 no-duplicate proof

1. Drive samples and score recomputation emit zero events.
2. Every event corresponds to one named candidate/lifecycle transition under `eve.m3-c-a.goal-transition-predicate.v1`.
3. `candidate_id` is fixed to `decision_epoch`; the same semantic candidate cannot be re-issued as a new candidate while the epoch remains unchanged.
4. A candidate has exactly one current lifecycle state and may move at most one listed edge per logical step.
5. Proposal hysteresis (`0.20` enter / `0.10` exit) prevents threshold chatter.
6. Selected-goal switches require both a `0.12` challenger margin and 30 seconds of replay-carried monotonic cooldown.
7. The selected winner is deterministic (`argmax`, lexical candidate-id tie-break).
8. A repeated evaluation with unchanged candidate set, drive sample, prior state, policy version, and epoch produces no named transition and therefore no event.
9. Only a verified future event append may advance the persistent lifecycle/selection state.

Therefore continuous affect/drive variation cannot create continuous events, and a persistent named goal-selection state cannot emit duplicates.

## A9 compliance matrix

| Rule | Mechanical proof |
|---|---|
| no continuous-value event | score/drive samples are derived only |
| named semantic transition only | lifecycle table is exhaustive for M3-C-A |
| versioned predicates | fixed `eve.m3-c-a.goal-transition-predicate.v1` |
| bidirectional stability | proposal enter/exit hysteresis + selected supersession path |
| cooldown | exact 30 replay-seconds for selection switch |
| no duplicate | decision epoch + one lifecycle state + deterministic winner |
| replay time | monotonic elapsed time only; no wall clock |
| one transition per step | lifecycle state advances one listed edge only |

## Authoritative persistence design

After #215, future **v4-native** M3-C goal-candidate and goal-selection lifecycle events may use the event kernel + SQLite store as their authoritative persistence substrate. Each persisted event must retain:

```text
candidate_id
semantic_goal_id
decision_epoch
before_lifecycle_state
after_lifecycle_state
drive_sample_digest
candidate_set_digest
evidence_digest
scoring_policy_version
transition_predicate_version
score and runner-up margin when selection-relevant
prior selected candidate id when applicable
```

A future snapshot must use the existing validated snapshot/replay regime and reproduce the same selected-candidate state from event history.

This does **not** make the legacy goal domain non-authoritative. Legacy `GoalManagement` remains authoritative for that legacy domain until a separate per-domain migration implementation and cutover gate. M3-C-A introduces no writer, database, sidecar, or runtime append.

## Counterfactual verification — does affect-derived state change actual choice?

The required verification is causal and deterministic: keep semantic context, evidence, candidate set, policy version, candidate attributes, and decision epoch identical; vary only the validated M3-A drive sample produced from mapped affect/appraisal inputs. The selected winner must change when the drive state materially changes.

Both candidates use identical `base_value=0.30`, `expected_value=0`, `urgency=0`, `continuity=0`, `cost=0`, and `risk=0`.

<!-- BEGIN M3C COUNTERFACTUAL CANDIDATE TABLE -->
| Candidate | energy alignment | safety alignment | curiosity alignment | Other five alignments | Confidence all | Ruling |
|---|---:|---:|---:|---:|---:|---|
| `recover_operating_margin` | `-0.90` | `-0.80` | `-0.10` | `0` | `1.0` | `RESOLVED` |
| `explore_information_gap` | `0.30` | `0.10` | `0.90` | `0` | `1.0` | `RESOLVED` |
<!-- END M3C COUNTERFACTUAL CANDIDATE TABLE -->

The normalized M3-A drive samples are:

<!-- BEGIN M3C COUNTERFACTUAL DRIVE TABLE -->
| Condition | energy z | safety z | curiosity z | affiliation z | agency z | coherence z | competence z | expression z | Expected winner | Ruling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `strain_mapped_affect` | `-0.70` | `-0.80` | `-0.20` | `0` | `0` | `0` | `0` | `0` | `recover_operating_margin` | `RESOLVED` |
| `recovered_exploration` | `0.60` | `0.70` | `0.90` | `0` | `0` | `0` | `0` | `0` | `explore_information_gap` | `RESOLVED` |
<!-- END M3C COUNTERFACTUAL DRIVE TABLE -->

Under the exact v1 formula, the checker must independently recompute all candidate scores and prove both:

```text
strain_mapped_affect winner      = recover_operating_margin
recovered_exploration winner     = explore_information_gap
```

and each winning selection must satisfy the `0.30` minimum plus `0.08` runner-up margin.

This proves the intended M3-C property: **mapped affect/appraisal evidence can alter derived drive state, and the derived drive state can causally alter deterministic goal selection.** It does not claim subjective emotion, does not mutate legacy hormone state, and does not open M3-E.

## M3-E boundary

M3-E remains independently closed. M3-C-A may read future validated M3-A drive samples as inputs; it may not:

- make legacy hormone or affect-registry values authoritative M3 state;
- mutate live affect/hormone state;
- authorize affect persistence cutover;
- turn an emotion label into a direct goal/action trigger;
- bypass M3-B provenance/evidence requirements;
- use M3-E as a prerequisite already satisfied.

## Reviewer-ruling regime

Every design row is `RESOLVED` or `UNRESOLVED`; unresolved rows require an explicit open question. This version contains zero unresolved rulings. Numeric changes require a new version and append-only reviewed decision evidence rather than silent edits after acceptance.

## Verification

```text
python scripts/audit/m3_c_a_goal_selection_check.py --summary-only
python scripts/audit/m3_c_a_goal_selection_check.py --fail-on-unresolved
pytest -q tests/audit/test_m3_c_a_goal_selection_check.py
```

The checker must be standard-library only and must not import or execute the legacy goal runtime, drive runtime, persistence runtime, or M3-E code.

## Acceptance criteria

- exact eight M3-A drives integrated;
- merged 63-axis Affect Migration Plan rechecked as 59 `MAPPED` / 4 `PROPOSED-DROP` / 0 unresolved;
- deterministic bounded scoring formula and lexical tie-break fixed;
- exact proposal hysteresis, winner margins, and cooldown fixed;
- named lifecycle and A9 no-continuous/no-duplicate proof complete;
- #215 human-authorization digest and v4-native authoritative-store role pinned;
- legacy goal-domain authority explicitly remains legacy until its own migration gate;
- deterministic counterfactual proves a drive-state-only change flips actual selected goal proposal;
- no M3-E authority, live affect mutation, goal runtime integration, persistence write, speech, scheduler, action, memory/vector/model/AGP mutation, or production-default change.

## Explicit non-goals

No runtime implementation. No legacy `GoalManagement` rewrite. No legacy goal-domain authority transfer. No M3-E affect cutover. No M3-B observation-window completion claim. No action execution. No scheduler integration. No speech generation. No raw external text capability. No new persistence writer. No live database access. No production default change.

## Changed-file boundary

This M3-C-A slice is intended to contain only the design document, its static checker/test, the same-PR forward-additions registration if discovery requires one, the #215 validation-reuse record/reference, and the required STATUS update. It must not modify production runtime code.