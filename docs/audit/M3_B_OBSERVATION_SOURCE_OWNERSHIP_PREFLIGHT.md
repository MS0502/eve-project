# M3-B Observation Source Ownership Preflight

Baseline: `0d755c35c994fa5b1ed3f2768c7905cda83c9a95`

Status: static blocker-evidence candidate. This work does not install a source observer, read live state, start or satisfy an observation window, access persistence, append events, mutate affect/drives/goals/memory/expression, authorize cutover, open M3-C, or grant M3-E authority.

## Purpose

PR #170 merged the disconnected caller-supplied M3-B projection contract. That contract deliberately does not decide who owns a real observation for each of the 63 source axes. Before a real read-only observation window can start, source ownership must be mechanically proved rather than inferred from defaults, proposal deltas, container access, or historical persistence.

This preflight asks:

1. Are all 26 legacy mutable hormone axes readable from one concrete runtime container?
2. Does that read surface already provide the immutable source envelope required by M3-B?
3. What exists for the 37 registry axes: definitions, proposal metadata, current values, or a real value owner?

## Result

```text
legacy mutable hormone axes:       26
read-only affect registry axes:     37
total authoritative source axes:    63
strict observation-ready axes:       0
M3-B real observation ready:     false
M3-B complete:                   false
M3-C open:                      false
```

Exact blockers:

```text
LEGACY_IMMUTABLE_SOURCE_ENVELOPE_ABSENT
REGISTRY_OBSERVED_VALUE_OWNER_ABSENT
```

These blockers are the expected successful preflight result. They prevent a false-positive observation-window start.

## Legacy 26-axis finding

`HormoneAdapter` owns a reference to one `HormoneSystem`, and `HormoneAdapter.as_dict()` iterates `self.hs.hormones.items()` and reads every `h.level`. All 26 legacy current levels are mechanically readable from the live legacy container.

That existing method is not yet an M3-B observation envelope. It does not bind the read to:

- versioned source snapshot identity;
- source schema version;
- source integrity SHA-256;
- explicit original floor and ceiling;
- confidence and provenance metadata;
- before/after no-mutation proof;
- exact source-object identity;
- observation-window sample identity.

It also adds `stress`, `energy`, and `curiosity`. Those are derived compatibility keys rather than authoritative source axes and must not inflate 26 to 29.

The active persistence adapter passes the entire `HormoneSystem` as `self.hs`. Static inspection finds no independent axis-specific snapshot-key contract for the 26 axes. Existing persistence ownership is not reinterpreted as M3-B observation ownership.

```text
legacy_mutable_hormone = READABLE_UNVERSIONED_LEGACY_CONTAINER
```

## Registry 37-axis finding

The registry module defines exactly 37 unique axes with descriptions, defaults/baselines, min/max bounds, decay/spike/refractory policy, evidence requirements, and safety guards. Its factory returns detached definition data.

The first discovery scan also found an existing proposal layer:

- `adapters/affect_event_to_axis_proposal_map.py` defines bounded event-to-axis **delta proposals**;
- `adapters/affect_event_proposal_validator.py` validates detached proposals;
- the interaction matrix and registry are consumed as read-only policy data;
- module contracts explicitly prohibit live application, runtime mutation, persistence, speech, memory writes, and gate bypass.

The scan found exactly 27 axis-key proposal-rule literals in the event proposal map. These are important future producer-design inputs, but they are not current axis values. A rule such as `hardware_low_power → recovery_need +0.04` does not establish the current value of `recovery_need`, a value container, update history, snapshot identity, or ownership.

The checker therefore separates:

```text
proposal_rule_candidates          = 27
observed_value_store_candidates    = 0
observed_value_owner_found      = false
```

Registry defaults are schema values. Proposal deltas are detached future transition metadata. Neither may be materialized as real observations.

```text
read_only_affect_registry = PROPOSAL_METADATA_EXISTS_NO_OBSERVED_VALUE_OWNER
```

## Why observation-ready count is zero

The legacy values are readable but not yet admissible as A10-recalculable M3-B evidence because the immutable identity/schema/range/integrity/confidence envelope and no-mutation proof are absent.

The registry axes have definitions and bounded proposal rules, but no current-value state owner. The preflight therefore counts neither source family as observation-ready.

## Required next artifacts

### 1. Legacy 26-axis immutable capture

A later PR may add a separately invoked, after-the-fact read-only capture around an explicitly supplied exact `HormoneAdapter` or `HormoneSystem`. It must:

- verify the reviewed object/type before reading;
- read all and only the 26 authoritative axes;
- preserve level, baseline, tier, phase, and source identity evidence;
- bind a versioned schema and canonical integrity digest;
- prove all source values and object state unchanged before/after capture;
- exclude derived compatibility keys from the authoritative set;
- install no production hook and access no persistence;
- remain caller-invoked, `shadow_only`, and default-disabled.

### 2. Registry 37-axis current-value ownership design

The existing proposal map should be reused as bounded candidate metadata, not mistaken for state. A separate design must define for every axis:

- current-value container and lifecycle owner;
- deterministic initial-state rule distinct from observation evidence;
- accepted proposal sources and validator boundary;
- update/cadence ownership;
- absence/unknown behavior;
- range, confidence, and saturation policy;
- snapshot identity, schema, provenance, and integrity;
- A9 event boundary and no-duplicate rule;
- read-only observation boundary;
- no direct speech, goal, memory, self-model, or external-effect authority.

This preflight does not authorize that state owner or apply path.

### 3. Real M3-B observation window

Only after both source families are admissible may a separate packet prove:

- complete strict 63-axis coverage;
- repeated identical projection for identical source snapshots;
- source-integrity and missing-input fail-closed behavior;
- source and target saturation metrics;
- confidence distribution and bounded drive divergence;
- zero live affect/drive/goal/memory/expression effects;
- zero event append and zero persistence access;
- legacy runtime and persistence still authoritative.

## Static checker

`scripts/audit/m3_b_observation_source_ownership.py`:

- parses the 26 legacy definitions from `HormoneSystem._init_all_hormones`;
- parses the 37 registry definitions from `AXIS_GROUPS`;
- inspects `HormoneAdapter.__init__` and `as_dict` for current-level readability;
- checks legacy persistence only for exact legacy-axis snapshot keys;
- scans tracked Python for registry-factory calls and exact-axis stores;
- classifies reviewed proposal/config modules separately from current-value ownership;
- records tracked parse errors separately because they are not source-ownership evidence;
- emits deterministic canonical JSON with a recalculable report SHA-256;
- treats the exact two blockers as the expected successful result.

It imports no EVE production module and invokes no runtime or persistence behavior.

## Validation

```text
python scripts/audit/m3_b_observation_source_ownership.py --summary-only --pretty --strict
pytest -q tests/audit/test_m3_b_observation_source_ownership.py
```

Focused tests require:

- exact 26+37 catalogs with no overlap;
- full legacy readability but incomplete immutable envelope;
- whole-container persistence evidence without axis-specific snapshot ownership;
- registry defaults classified as definitions;
- exact 27 proposal-rule candidates classified as metadata;
- zero observed-value store candidates;
- exact two blockers and zero checker errors;
- deterministic recalculable output;
- no runtime import or live-action surface.

## Cross-chat validation reuse

Prerequisite evidence:

```text
M3-B exact head:       0ccce6b0a39fd0b0030181d3bfdb7c99f01b7c6c
M3-B exact run:        29977749600
M3-B artifact SHA-256: 4a73b08e5ae4b143ebbd84963c7a6803a8964a5a47db95805fdef9b42df49ca1
M3-B merge SHA:        0d755c35c994fa5b1ed3f2768c7905cda83c9a95
M2-E compatibility:    29977749585
```

Do not rerun prerequisite workflows merely because work moves to a new chat. Rerun only if the pinned head, artifact digest, or validation scope changes, or the artifact is lost/corrupt.

The final exact-head run for this preflight will be pinned in PR #171 and reused under the same rule.

## Explicit non-goals

No live capture, observer installation, runtime composition change, registry current-value container, proposal apply path, default materialization as observed state, persistence read/write, SQLite access, event append, drive update, named-state transition, goal proposal, scheduler, memory/self-model mutation, expression/speech effect, cutover, M3-C implementation, M3-B completion, or M3-E authority.

## Intended changed-file boundary

1. `scripts/audit/m3_b_observation_source_ownership.py`
2. `tests/audit/test_m3_b_observation_source_ownership.py`
3. `docs/audit/M3_B_OBSERVATION_SOURCE_OWNERSHIP_PREFLIGHT.md`
4. `docs/audit/forward_additions/pr-171.json`
