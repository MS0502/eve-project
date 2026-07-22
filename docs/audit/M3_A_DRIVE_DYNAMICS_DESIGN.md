# M3-A Drive Dynamics Design

Baseline: `7697c1047bbf081295a01f630d63d8a3ad5c69b0`

Status: reviewer-ruled, documentation-only design. No runtime projection, mutation, event emission, persistence, scheduler, goal/expression integration, cutover, or M3-E authority.

## Authority and scope

This work is parallel to the M2-E observation window. It binds the merged Affect Migration Plan's 63 axes (59 `MAPPED`, 4 `PROPOSED-DROP`, 0 `UNRESOLVED`) and resolves A9 hysteresis/cooldown numerics. The legacy runtime remains authoritative; integration eligibility exists only after an explicit persistence cutover and later reviewed M3 implementation.

Versions: `eve.m3-a.drive-dynamics.v1`, `eve.m3-a.drive-sample.v1`, `eve.m3-a.named-transition-predicate.v1`, `eve.m3-a.transition-candidate.v1`, `eve.m3-a.candidate-lifecycle.v1`, `eve.m3-a.axis-landing-report.v1`, `eve.m3-a.drive-dynamics-check.v1`.

## Continuous dynamics

For drive `d`, `x_d∈[L_d,U_d]` is a derived continuous salience/readiness value, not an event, fact, goal, permission, action, or speech command. Inputs are provenance-bearing `(c_i,q_i,target_i)` contributions with `c_i∈[-1,1]`, confidence `q_i∈[0,1]`, and an Affect Plan target.

```text
a_d       = clip(sum(q_i*c_i)/max(1,sum(q_i)),-1,1)
target_d  = clip(beta_d+gain_d*a_d,L_d,U_d)
relaxed_d = target_d+(x_d-target_d)*exp(-Δt/tau_d)
x_next_d  = clip(x_d+clip(relaxed_d-x_d,-slew_d*Δt,+slew_d*Δt),L_d,U_d)
```

`Δt` is replay-carried monotonic elapsed time, never wall clock. With no input, `x_d` returns exponentially to `beta_d`. Continuous sampling emits zero events. Named state may move at most one adjacent state per logical step and only after cooldown.

## Drive parameter rulings

<!-- BEGIN M3A DRIVE TABLE -->
| Drive | Baseline beta | Decay tau seconds | Floor L | Ceiling U | Gain | Max slew per second | Ruling | Open question |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `energy` | 0.60 | 300 | 0.02 | 0.98 | 0.38 | 0.020 | `RESOLVED` | — |
| `safety` | 0.62 | 180 | 0.02 | 0.98 | 0.42 | 0.015 | `RESOLVED` | — |
| `affiliation` | 0.35 | 1800 | 0.01 | 0.95 | 0.30 | 0.005 | `RESOLVED` | — |
| `curiosity` | 0.32 | 240 | 0.01 | 0.97 | 0.45 | 0.020 | `RESOLVED` | — |
| `agency` | 0.50 | 420 | 0.01 | 0.97 | 0.40 | 0.015 | `RESOLVED` | — |
| `coherence` | 0.68 | 1200 | 0.03 | 0.99 | 0.28 | 0.005 | `RESOLVED` | — |
| `competence` | 0.42 | 1800 | 0.01 | 0.96 | 0.30 | 0.005 | `RESOLVED` | — |
| `expression` | 0.25 | 90 | 0.00 | 0.95 | 0.50 | 0.030 | `RESOLVED` | — |
<!-- END M3A DRIVE TABLE -->

These are versioned design rulings, not biological measurements. Changes require a new version and append-only A12 ruling.

## Semantic state catalog

Each row lists exact ordinal states `0;1;2;3` and matching meanings. The checker expands 8 rows into 32 state records.

<!-- BEGIN M3A STATE TABLE -->
| Drive | Ordered states | Ordered meanings | Ruling | Open question |
|---|---|---|---|---|
| `energy` | `depleted;guarded;available;abundant` | insufficient operating energy;energy conserved for bounded work;ordinary work capacity available;high reversible operating margin | `RESOLVED` | — |
| `safety` | `threatened;guarded;secure;resilient` | validated safety margin is low;protective appraisal is active;ordinary safety margin is available;high coping and reversibility margin | `RESOLVED` | — |
| `affiliation` | `withdrawn;receptive;connected;affiliative` | social engagement drive is low;bounded social contact is welcome;validated relationship continuity is salient;high non-exclusive connection motivation | `RESOLVED` | — |
| `curiosity` | `quiet;attentive;exploring;absorbed` | no material information-gap pressure;novelty or uncertainty is being monitored;bounded information-gain work is salient;sustained high-value exploration is active | `RESOLVED` | — |
| `agency` | `constrained;deliberative;self_directed;assertive` | available self-directed options are low;choices are being compared;validated autonomous action readiness is present;high boundary-aware self-direction is present | `RESOLVED` | — |
| `coherence` | `fragmented;reconciling;coherent;integrated` | validated state or narrative conflicts are material;conflicts are being integrated;identity and narrative constraints are consistent;high long-horizon consistency is retained | `RESOLVED` | — |
| `competence` | `uncertain;practicing;capable;mastering` | validated capability is below the current demand;bounded learning work is active;validated capability meets ordinary demand;sustained high-confidence skill growth is present | `RESOLVED` | — |
| `expression` | `silent;forming;ready;expressive` | no validated internal content requires expression;internal semantic content is being composed;bounded expression proposal is ready for validation;high communicative value is present without action authority | `RESOLVED` | — |
<!-- END M3A STATE TABLE -->

## Named transition catalog

Each boundary cell is `down/up/cooldown_seconds`. For every adjacent state pair, the checker expands both directions using predicate version `eve.m3-a.named-transition-predicate.v1`:

```text
up ID:        m3a.<drive>.<lower>_to_<upper>.v1
down ID:      m3a.<drive>.<upper>_to_<lower>.v1
up predicate: x_next >= up
down predicate: x_next <= down
candidate:    drive.<drive>.<destination>_candidate
width:        up-down
```

Thus 8 rows × 3 boundaries × 2 directions produce the exhaustive 48-transition catalog.

<!-- BEGIN M3A BOUNDARY TABLE -->
| Drive | Boundary 0↔1 down/up/cool | Boundary 1↔2 down/up/cool | Boundary 2↔3 down/up/cool | Ruling | Open question |
|---|---|---|---|---|---|
| `energy` | `0.20/0.28/30` | `0.45/0.55/45` | `0.72/0.82/90` | `RESOLVED` | — |
| `safety` | `0.22/0.30/20` | `0.48/0.58/60` | `0.74/0.84/120` | `RESOLVED` | — |
| `affiliation` | `0.17/0.25/120` | `0.42/0.52/300` | `0.68/0.80/600` | `RESOLVED` | — |
| `curiosity` | `0.16/0.24/30` | `0.40/0.50/60` | `0.66/0.78/120` | `RESOLVED` | — |
| `agency` | `0.20/0.28/45` | `0.46/0.56/90` | `0.70/0.82/180` | `RESOLVED` | — |
| `coherence` | `0.22/0.32/60` | `0.50/0.60/180` | `0.74/0.86/600` | `RESOLVED` | — |
| `competence` | `0.18/0.26/60` | `0.44/0.54/180` | `0.68/0.80/600` | `RESOLVED` | — |
| `expression` | `0.14/0.22/15` | `0.38/0.48/30` | `0.62/0.74/90` | `RESOLVED` | — |
<!-- END M3A BOUNDARY TABLE -->

Cooldown is accumulated monotonic time since the last accepted transition of the drive. While incomplete, named state remains unchanged. No transition targets speech, tools, scheduler work, goals, memory mutation, or persistence.

## Candidate lifecycle

<!-- BEGIN M3A LIFECYCLE TABLE -->
| From | To | Trigger | Authority effect |
|---|---|---|---|
| `absent` | `proposed` | one adjacent predicate true after cooldown; next epoch has no candidate | none |
| `proposed` | `validated` | future validator recomputes predicate, parameters, provenance, before-state, cooldown | none |
| `proposed` | `rejected` | validation fails | none |
| `proposed` | `expired` | source evidence becomes stale | none |
| `validated` | `emitted` | future authorized event append returns verified receipt | named-state event only |
| `validated` | `rejected` | append precondition fails | none |
| `emitted` | `absent` | reducer acknowledgement advances epoch | none |
| `rejected` | `absent` | rejection acknowledged | none |
| `expired` | `absent` | expiry acknowledged | none |
<!-- END M3A LIFECYCLE TABLE -->

Candidate identity is `sha256(schema||drive||from||to||next_state_epoch||predicate_version||parameter_version)`. No randomness or wall-clock timestamp participates.

## A9 no-duplicate proof

1. Continuous samples derive `x_next` only.
2. Only one adjacent edge is eligible per drive per logical step.
3. A drive has at most one non-terminal candidate.
4. Candidate identity is fixed to `next_state_epoch`; it cannot recur while state/epoch persist.
5. Cooldown prevents immediate reversal.
6. Only verified future append may advance state/epoch.
7. A reverse candidate therefore requires a prior accepted transition, new epoch, opposite threshold, and completed cooldown.

Duplicate issuance while the same named state persists is therefore impossible.

## A9 compliance matrix

| Rule | Mechanical proof |
|---|---|
| no continuous event | formula/scope token |
| versioned predicates | fixed predicate version |
| bidirectional | 8×3×2 exact expansion |
| hysteresis | `up>down`, width recomputation |
| cooldown | positive integer per boundary |
| no duplicate | epoch + one pending candidate + cooldown |
| replay time | monotonic `Δt` |
| one event per step | one adjacent edge |

## Affect Migration Plan landing

The checker parses `docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md` directly. Every `MAPPED` axis is reported as its exact `drive::<name>`, `appraisal::<name>`, and `emotion::<name>` targets. Every `PROPOSED-DROP` axis must have no future target and a historical-preservation ruling. Required totals are 63/59/4/0 and all 63 rows must be covered. Normalization, sign, confidence calibration, acquisition, and shadow projection remain M3-B work.

## Reviewer-ruling regime

Drive/state/boundary rows allow `RESOLVED` or `UNRESOLVED`; unresolved requires an open question. `--fail-on-unresolved` rejects unresolved M3-A or source-plan rows. This revision has zero unresolved M3-A rulings. Corrections require separate digest-linked A12 artifacts.

## Verification

```text
python scripts/audit/m3_a_drive_dynamics_check.py --summary-only
python scripts/audit/m3_a_drive_dynamics_check.py --fail-on-unresolved
pytest -q tests/audit/test_m3_a_drive_dynamics_check.py
```

The checker is standard-library only and imports no runtime module.

## Acceptance criteria

8 drives; 32 states; 24 boundaries; 48 bidirectional transitions; positive decay; bounded saturation/slew; exact hysteresis/cooldown; complete lifecycle/A9 proof; all 63 axes covered as 59 mapped/4 drop/0 unresolved; deterministic checker/tests; no runtime/production change; no M3-E authority or pre-cutover integration eligibility.

## Explicit non-goals

No runtime integration, SQLite access, dual read, recovery, cutover, observation-window activation, scheduler, M3-B/C/D/E implementation, affect/hormone/drive/goal/memory/vector/model/AGP mutation, speech, tool/external effect, raw-text capability, or production-default change.

## Changed-file boundary

`docs/audit/M3_A_DRIVE_DYNAMICS_DESIGN.md`; `scripts/audit/m3_a_drive_dynamics_check.py`; `tests/audit/test_m3_a_drive_dynamics_check.py`; `docs/EVE_IMPLEMENTATION_STATUS_v4.md`; `docs/audit/forward_additions/pr-<this-pr>.json`.
