# M3-B Observation Source Ownership Preflight

Baseline: `0d755c35c994fa5b1ed3f2768c7905cda83c9a95`

Status: static blocker-evidence candidate. This document and its checker do not install a source observer, read live state, start or satisfy an observation window, access persistence, append events, mutate affect/drives/goals/memory/expression, authorize cutover, open M3-C, or grant M3-E authority.

## Purpose

PR #170 merged the disconnected caller-supplied M3-B projection contract. That contract deliberately does not decide who owns a real observation for each of the 63 source axes. Before a real read-only observation window can start, source ownership must be mechanically proved rather than inferred from defaults, container access, or historical persistence.

This preflight answers three narrower questions:

1. Are all 26 legacy mutable hormone axes currently readable from one concrete runtime container?
2. Does that read surface already produce the immutable identity/schema/integrity/range envelope required by M3-B?
3. Do the 37 read-only registry axes have actual observed runtime values and an owner, or only schema definitions/defaults?

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

These blockers are expected and are the successful result of the preflight. They prevent a false-positive observation-window start.

## Legacy 26-axis finding

`HormoneAdapter` owns a reference to one `HormoneSystem`, and `HormoneAdapter.as_dict()` iterates `self.hs.hormones.items()` and reads every `h.level`. Therefore all 26 legacy current levels are mechanically readable from the live legacy container.

That existing method is not yet an M3-B observation source envelope. It does not bind the read to:

- a versioned source snapshot identity;
- a source schema version;
- a source integrity SHA-256;
- explicit original floor and ceiling;
- confidence/provenance metadata;
- a before/after no-mutation proof;
- a fixed exact source object identity;
- an observation-window sample identity.

It also adds `stress`, `energy`, and `curiosity`, which are derived compatibility keys rather than authoritative source axes. They must never inflate 26 to 29.

The active persistence adapter passes the entire `HormoneSystem` into legacy persistence as `self.hs`. Static inspection finds no independent axis-specific snapshot-key contract for the 26 axes. Existing persistence ownership is not silently reinterpreted as M3-B observation ownership.

Result:

```text
legacy_mutable_hormone = READABLE_UNVERSIONED_LEGACY_CONTAINER
```

## Registry 37-axis finding

`adapters/affect_hormone_neural_rhythm_registry.py` defines:

- six axis groups containing exactly 37 unique names;
- descriptions;
- defaults/baselines;
- min/max bounds;
- decay/spike/refractory policy;
- evidence and safety guards;
- a factory that returns a deep-copied read-only definition registry.

Those `default` and `baseline` values are design/schema values. They are not proof that a runtime observed `energy_budget`, `social_trust`, `self_coherence`, `expression_pressure`, or any other registry axis currently exists.

The repository scan distinguishes:

- registry definition construction;
- audit/test/projection mentions;
- production calls to the registry factory;
- production stores keyed by exact registry-axis literals.

No actual observed-value owner for the 37 axes is found. Projection rules and default values are explicitly excluded from ownership evidence.

Result:

```text
read_only_affect_registry = DEFINITION_ONLY_NO_OBSERVED_VALUE_OWNER
```

## Why observation-ready count is zero rather than 26

The legacy levels are readable, but the M3-B contract requires immutable source identity, schema, range, integrity digest, confidence, and provenance. Until that exact envelope exists and proves no mutation, the 26 axes are not counted as observation-ready.

This does not mean the legacy values are inaccessible. It means they are not yet admissible as A10-recalculable M3-B observation evidence.

## Required next artifacts

### 1. Legacy 26-axis immutable capture

A later PR may add a separately invoked, after-the-fact read-only capture around an explicitly supplied exact `HormoneAdapter` or `HormoneSystem` object. It must:

- verify the exact reviewed object/type before reading;
- read all and only the 26 authoritative axes;
- preserve level, baseline, tier, phase, and source object identity evidence;
- bind a versioned schema and canonical integrity digest;
- prove the source object and all axis values are unchanged before/after capture;
- add no derived compatibility keys to the authoritative set;
- install no production hook and access no persistence;
- remain caller-invoked, `shadow_only`, and default-disabled.

### 2. Registry 37-axis producer/ownership design

A separate design decision must define where actual values come from. It must not use registry defaults as fake observations. It must define, for every axis:

- producer or deterministic derivation inputs;
- source evidence and provenance;
- update/cadence ownership;
- absence/unknown behavior;
- range and confidence;
- snapshot identity and integrity;
- A9 event boundary;
- read-only observation boundary;
- no direct speech, goal, memory, self-model, or external-effect authority.

The 37-axis producer design is not authorized by this preflight.

### 3. Real M3-B observation window

Only after both source families are admissible may a separate observation packet prove:

- complete strict 63-axis coverage;
- repeated identical projection for identical source snapshots;
- source-integrity and missing-input fail-closed behavior;
- source and target saturation metrics;
- confidence distribution and bounded drive divergence;
- zero live affect/drive/goal/memory/expression effects;
- zero event append and zero persistence access;
- legacy runtime/persistence still authoritative.

## Static checker

`scripts/audit/m3_b_observation_source_ownership.py`:

- parses the 26 legacy definitions from `HormoneSystem._init_all_hormones`;
- parses the 37 registry definitions from `AXIS_GROUPS`;
- inspects `HormoneAdapter.__init__` and `as_dict` for current-level readability;
- checks the legacy persistence adapter only for exact legacy-axis snapshot keys;
- scans tracked Python for registry-factory calls and exact-axis production value-store candidates;
- reports tracked parse errors separately because they are not source-ownership evidence;
- emits canonical deterministic JSON with a recalculable report SHA-256;
- treats the exact two blockers as the expected successful preflight result.

It imports no EVE production module and invokes no runtime/persistence behavior.

## Validation

```text
python scripts/audit/m3_b_observation_source_ownership.py --summary-only --pretty --strict
pytest -q tests/audit/test_m3_b_observation_source_ownership.py
```

Focused tests require:

- exact 26+37 catalogs and no overlap;
- complete legacy readability but incomplete immutable envelope;
- whole-container persistence evidence without axis-specific snapshot ownership;
- registry defaults classified as definitions, not observations;
- exact two blockers and zero checker errors;
- no production registry value-store owner;
- deterministic recalculable output;
- no runtime import or live-action surface.

## Cross-chat validation reuse

The prerequisite technical-candidate evidence is immutable:

```text
M3-B exact head:       0ccce6b0a39fd0b0030181d3bfdb7c99f01b7c6c
M3-B exact run:        29977749600
M3-B artifact SHA-256: 4a73b08e5ae4b143ebbd84963c7a6803a8964a5a47db95805fdef9b42df49ca1
M3-B merge SHA:        0d755c35c994fa5b1ed3f2768c7905cda83c9a95
M2-E compatibility:    29977749585
```

Do not rerun these prerequisite workflows because work moves to a new chat. Rerun only if the pinned head, artifact digest, or validation scope changes, or if the artifact is lost/corrupt.

The final exact-head run for this preflight will be pinned in its PR body and reused the same way.

## Explicit non-goals

No live capture implementation, observer installation, runtime composition change, registry-axis producer, default-value materialization as observed state, persistence read/write, SQLite access, event append, drive update, named-state transition, goal proposal, scheduler, memory/self-model mutation, expression/speech effect, cutover, M3-C implementation, M3-B completion, or M3-E authority.

## Intended changed-file boundary

1. `scripts/audit/m3_b_observation_source_ownership.py`
2. `tests/audit/test_m3_b_observation_source_ownership.py`
3. `docs/audit/M3_B_OBSERVATION_SOURCE_OWNERSHIP_PREFLIGHT.md`
4. `docs/audit/forward_additions/pr-<this-pr>.json`
