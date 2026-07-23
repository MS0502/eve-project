# M3-B Read-Only Affect Projection Candidate

Baseline: `aa10a475cc29c8b1be7ffb598ea5999eee40a6db`

Status: bounded technical candidate. This work implements a disconnected, caller-invoked shadow projection only. It does not install a live observer, start or satisfy an observation window, read persistence, append events, mutate affect/drives/goals/memory, trigger speech or external action, authorize cutover, open M3-C, or grant M3-E authority.

## Authority boundary

The pre-kernel legacy runtime and legacy persistence remain authoritative. Every M3-B output is `shadow_only`, immutable, diagnostic, and derived from caller-supplied observations. Import and construction perform no I/O. No source module, runtime loop, scheduler, persistence adapter, SQLite store, event-kernel append surface, expression path, or production startup path imports this module.

M3-A merge pin: `6d581ba1cf11ffbefafe77beabd8f669102909d0`.

M3-A contract versions reused without modification:

```text
eve.m3-a.drive-dynamics.v1
eve.m3-a.named-transition-predicate.v1
eve.m3-a.transition-candidate.v1
```

M3-B schemas:

```text
eve.m3-b.affect-axis-observation.v1
eve.m3-b.affect-axis-mapping.v1
eve.m3-b.drive-shadow-projection.v1
eve.m3-b.affect-shadow-projection.v1
eve.m3-b.affect-shadow-check.v1
```

## Input acquisition contract

M3-B accepts only explicit `AxisObservation` values supplied by a caller. Each observation contains:

- exact source axis and source family;
- original scalar value, baseline, floor, and ceiling;
- source confidence;
- source snapshot identity and schema version;
- source integrity SHA-256;
- bounded ordered source metadata;
- fixed `caller_supplied_read_only` acquisition mode;
- fixed `shadow_only` authority.

The core does not import `hormone_system.py`, the read-only affect registry, adapters, `main.py`, live loops, or persistence. Acquiring a real legacy snapshot and proving read ownership remain later integration/observation-window work. Unknown axes, source-family mismatches, non-finite values, malformed digests, invalid ranges, out-of-range values, duplicate axes, and incomplete strict inputs fail closed.

## Mechanical 63-axis catalog

`scripts/audit/m3_b_affect_projection.py` parses the merged `docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md` table directly and constructs the runtime-neutral mapping catalog:

```text
axes:                    63
legacy mutable axes:     26
read-only registry axes: 37
MAPPED:                  59
PROPOSED-DROP:            4
UNRESOLVED:               0
```

The four proposed-drop axes remain exactly:

```text
estrogen
testosterone
prolactin
progesterone
```

Their original values, ranges, source identities, schemas, integrity digests, confidence, and metadata remain visible in the projection report. They produce zero drive contributions, zero appraisals, zero derived-emotion targets, zero candidates, and zero behavioral authority.

## Normalization and confidence calibration

For source value `v`, baseline `b`, floor `L`, and ceiling `U`:

```text
if v >= b: z = clip((v-b)/(U-b), -1, +1)
if v <  b: z = clip((v-b)/(b-L), -1, +1)
```

The input contract requires `L < b < U` and `L <= v <= U`; source-range violations fail closed rather than being silently normalized. `z=-1` means the declared floor, `z=0` the declared baseline, and `z=+1` the declared ceiling.

Source confidence `q_source∈[0,1]` is capped by the reviewer-ruled M0-C confidence:

```text
high   cap = 1.00
medium cap = 0.75
low    cap = 0.50
q = q_source * cap
```

Confidence affects aggregation weight only. It cannot grant authority, suppress provenance, or convert missing input into a value.

## Explicit polarity rulings

Every axis/drive target pair is positive by default. The following exact pairs are negative because the source semantic is a deficit, load, inhibition, threat, risk, pressure, or avoidance signal that lowers the achieved/readiness value of that target. Appraisal and derived-emotion labels remain separately visible.

| Axis | Negative target drives |
|---|---|
| `norepinephrine` | safety |
| `adenosine` | energy |
| `cortisol` | safety |
| `melatonin` | energy |
| `ghrelin` | energy |
| `fatigue_pressure` | energy |
| `recovery_need` | energy; safety |
| `stress_load` | safety |
| `stability_need` | coherence; safety |
| `overload_risk` | energy; safety |
| `threat_pressure` | safety |
| `uncertainty_pressure` | coherence |
| `self_protection` | safety |
| `boundary_defense` | safety |
| `trust_risk` | affiliation; safety |
| `exposure_risk` | safety |
| `social_pain` | affiliation; safety |
| `loneliness_pressure` | affiliation |
| `belonging_need` | affiliation |
| `rejection_sensitivity` | affiliation; safety |
| `learning_pressure` | competence |
| `memory_consolidation_pressure` | coherence |
| `prediction_error_pressure` | coherence |
| `competence_drive` | competence |
| `agency_pressure` | agency |
| `expression_inhibition` | expression; safety |
| `conflict_avoidance` | affiliation; safety |

The checker requires all 35 negative pairs to be actual M0-C target pairs. Corrections require a new projection/mapping version and append-only A12 decision evidence; silently changing signs under v1 is prohibited.

For target drive `d`:

```text
polarity(axis,d) ∈ {-1,+1}
contribution(axis,d) = z_axis * polarity(axis,d)
```

Mixed-polarity axes are explicit. For example, elevated `norepinephrine` contributes positively to `energy` and negatively to `safety`; elevated `uncertainty_pressure` contributes positively to `curiosity` and negatively to `coherence`.

## Drive projection

For each drive, M3-B reuses the exact M3-A baseline, gain, decay, floor, ceiling, slew limit, semantic states, hysteresis thresholds, and cooldowns.

Given contributions `(c_i,q_i)`:

```text
a_d       = clip(sum(q_i*c_i)/max(1,sum(q_i)),-1,1)
target_d  = clip(beta_d+gain_d*a_d,L_d,U_d)
relaxed_d = target_d+(x_d-target_d)*exp(-Δt/tau_d)
x_next_d  = clip(x_d+clip(relaxed_d-x_d,-slew_d*Δt,+slew_d*Δt),L_d,U_d)
```

`Δt` is caller-supplied non-negative replay elapsed time. Wall clock is not read. Missing contributions produce `a_d=0`; the drive relaxes toward the M3-A baseline. Every output reports aggregate input, target, relaxed and next values, contribution count, total confidence, slew limiting, and target saturation.

## Named-state diagnostic candidate

M3-B may derive at most one adjacent diagnostic transition candidate per drive when:

- the M3-A predicate is true;
- the exact cooldown has completed;
- no pending candidate identity already exists.

Candidate identity remains:

```text
sha256(schema || drive || from || to || next_state_epoch || predicate_version || parameter_version)
```

M3-B never validates, emits, appends, acknowledges, or reduces the candidate. The prior named state and state epoch are retained unchanged. Candidate fields permanently include:

```text
authority = shadow_only
diagnostic_only = true
event_append_authorized = false
```

A supplied pending candidate identity suppresses duplicate derivation. This is diagnostic evidence for later M3-C review, not an M3-C implementation or event authority.

## Missing-input behavior

Strict mode requires observations for every supplied mapping row and fails closed on any missing axis. Non-strict mode exists only for audit/debug surfaces: the missing axis is explicit, its confidence is zero, it creates no contribution, and no fallback value is invented.

A complete 63-axis acceptance run uses strict mode.

## Provenance and recalculability

Every axis projection preserves:

- original value/baseline/floor/ceiling;
- normalized value and calibrated confidence;
- exact per-drive polarity and contribution;
- mapping status, appraisal targets, emotion targets, and preservation ruling;
- source snapshot/schema/integrity metadata;
- observation digest.

The complete projection records sorted mapping and observation digests plus a canonical projection digest. Same observations, mappings, priors, elapsed time, and versions must produce identical material and digest. The audit harness constructs a complete synthetic 63-axis replay and recomputes it twice.

Synthetic evidence proves contract determinism, coverage, bounds, sign decisions, confidence caps, no-side-effect behavior, and digest recalculability. It does not claim representative real-world equivalence or complete the M3-B observation window.

## Observation-window boundary

This candidate does not connect to the M2-E phone habitat driver and does not start an affect observation window. A later explicitly reviewed observation step must supply real read-only snapshots under approved source ownership and measure:

- deterministic repeat projection for identical source snapshots;
- missing-source and integrity failures;
- source and target saturation;
- confidence distribution;
- bounded drive divergence;
- zero live affect/drive/goal/memory/expression effects;
- zero event append and zero persistence access.

Only accepted evidence from that separate window can satisfy the M3-B milestone exit. This technical candidate alone does not open M3-C.

## Validation

```text
python scripts/audit/m3_b_affect_projection.py --summary-only --pretty --strict
pytest -q tests/test_v4_m3_b_affect_projection.py
```

Focused tests cover:

- exact 63/59/4 catalog and 26/37 source split;
- all 35 negative polarity decisions;
- bounded baseline-centered normalization and malformed-source rejection;
- confidence caps and mixed-polarity axes;
- proposed-drop historical preservation with zero contributions;
- deterministic complete projection and digest;
- M3-A bounds and slew limits;
- strict/non-strict missing-input behavior;
- diagnostic-only candidate and pending-identity duplicate suppression;
- AST proof of no I/O, persistence, runtime, or live-mutation import surface.

## Validation reuse across chats

Prerequisite evidence is reused and must not be rerun merely because work continues in a new chat:

### M3-A accepted design

```text
exact head:       c9e2e1d8227a5b53f8a52784bd7d9cffc5202a3c
workflow run:     29934366609
artifact:         exact-head-validation-c9e2e1d8227a5b53f8a52784bd7d9cffc5202a3c
artifact SHA-256: 464a71c12982c7948b945629e177493207f472c0b2d1533f110976475083b8b1
merge SHA:        6d581ba1cf11ffbefafe77beabd8f669102909d0
```

### M2-E driver merge prerequisite

```text
exact head:       4b3ea6cfb256221f69171dd807cbd2c5ca2357d5
workflow run:     29976350971
artifact SHA-256: f74a613c2e4cb99a7c253689e0d2c0fd8460a03fdf14cf1f09324bd64aee939f
window run:       29976350970
merge SHA:        aa10a475cc29c8b1be7ffb598ea5999eee40a6db
```

Do not rerun either prerequisite while its exact head, artifact digest, and required validation scope remain unchanged. PR metadata, comments, Draft/Ready transitions, and chat changes are not code changes. Rerun only for a head change, artifact loss/corruption, digest mismatch, or validation-scope change.

The final M3-B exact-head run must execute once on the final post-M2-E branch head. Its pins will be added to the PR body and retained as the reusable source of truth for subsequent chats.

## Acceptance criteria

Technical candidate acceptance requires:

- 63 unique axes parsed directly from the merged plan;
- exact 59 mapped / 4 proposed-drop / 0 unresolved;
- exact 26 legacy / 37 registry source split;
- all mapping targets and all 35 negative polarity rulings mechanically valid;
- complete strict projection with zero missing axes;
- eight bounded drive outputs using exact M3-A dynamics;
- deterministic replay and digest equality;
- proposed-drop original/provenance preservation with zero contribution;
- no named-state mutation, event append, persistence access, live behavior change, cutover, M3-C promotion, or M3-E authority;
- focused and full tests green;
- M0-A through M0-D byte-identical;
- forward unregistered/stale/same-PR/new-parse errors all zero;
- clean exact final head.

## Explicit non-goals

No live source acquisition, observer installation, runtime integration, event append, SQLite or persistence use, target-state materialization, reducer authority, named-state mutation, goal proposal, scheduler work, memory or identity mutation, AGP/vector/model update, speech, tool/external effect, cutover, production-default change, M3-C implementation, M3-B observation-window acceptance, or M3-E authorization.

## Intended changed-file boundary

1. `core/m3_b_affect_projection.py`
2. `scripts/audit/m3_b_affect_projection.py`
3. `tests/test_v4_m3_b_affect_projection.py`
4. `docs/audit/M3_B_READ_ONLY_AFFECT_PROJECTION_CANDIDATE.md`
5. `docs/audit/forward_additions/pr-<this-pr>.json`
