# M3-B Legacy 26-Axis Immutable Capture Candidate

Baseline: `97ac96b8bdf54c7fbc74d4b0346ceff49624eaec`

Status: disconnected technical candidate resolving only the legacy-source envelope blocker. It does not resolve registry current-value ownership, start or satisfy an observation window, access persistence, append events, mutate affect/drives/goals/memory/expression, authorize cutover, open M3-C, or grant M3-E authority.

## Context

PR #171 established two blockers before real M3-B observation:

```text
LEGACY_IMMUTABLE_SOURCE_ENVELOPE_ABSENT
REGISTRY_OBSERVED_VALUE_OWNER_ABSENT
```

This candidate addresses only the first blocker by adding an explicit, after-the-fact, caller-invoked read of one already-existing exact `HormoneSystem` object.

The registry blocker remains:

```text
REGISTRY_OBSERVED_VALUE_OWNER_ABSENT
```

M3-B remains incomplete and M3-C remains closed.

## Authority boundary

`core/m3_b_legacy_affect_capture.py` performs no work at import time. It constructs no `HormoneSystem`, starts no runtime, installs no observer, registers no callback, reads no persistence, writes no file, appends no event, and changes no production default.

Capture occurs only when a caller explicitly invokes:

```python
capture_legacy_hormone_state(
    existing_hormone_system,
    source_instance_id="...",
    source_snapshot_id="...",
)
```

The source must be the exact `hormone_system.HormoneSystem` type. Subclasses, adapters, mappings, and duck-typed substitutes fail closed.

All output remains:

```text
authority = shadow_only
acquisition_mode = explicit_after_the_fact_read_only
source_mutated = false
persistence_accessed = false
event_append_performed = false
live_behavior_changed = false
observation_window_started = false
m3_c_open = false
m3_e_authority_open = false
cutover_authorized = false
```

## Exact source contract

The capture requires the exact canonical 26-axis order from `HormoneSystem._init_all_hormones`:

```text
glutamate, gaba, glycine, dopamine, serotonin, norepinephrine,
histamine, acetylcholine, adenosine, endorphin, cortisol, oxytocin,
vasopressin, melatonin, bdnf, ngf, estrogen, testosterone,
insulin_brain, thyroid, leptin, ghrelin, prolactin, dhea,
progesterone, growth_hormone
```

For every axis the exact source object must be the legacy `Hormone` type and its `name` must match the dict key. Capture preserves:

- current `level`;
- current `baseline`;
- fixed source range `[0,1]`;
- `reactivity`;
- `decay_rate`;
- `tier`;
- activation `phase`;
- fixed direct-source confidence `1.0`.

Capture also preserves source phase, developmental stage, source time, simulated hour, active-hormone order, exact source type, caller-supplied source instance identity, and caller-supplied snapshot identity.

Derived `HormoneAdapter.as_dict()` compatibility keys `stress`, `energy`, and `curiosity` are never captured as authoritative axes.

## No-mutation proof

The capture algorithm:

1. verifies the exact source type and exact dict container;
2. retains the source container reference and all 26 `Hormone` object references;
3. reads and validates a complete canonical source-state materialization;
4. computes the source integrity SHA-256;
5. constructs detached frozen axis evidence from the first read;
6. reads the complete source state again;
7. verifies the source dict is the same object;
8. verifies all 26 axis objects are the same objects;
9. verifies before and after source material are identical;
10. fails closed if any value, metadata, container identity, or axis-object identity changed.

The function never calls `HormoneSystem.update`, `Hormone.stimulate`, `HormoneAdapter.tick`, or any persistence surface.

The no-mutation proof is an after-the-fact stability check. It does not claim lock-based atomicity and does not authorize production observation scheduling. A later observation-window design must define sampling ownership and concurrent-update handling.

## Deterministic evidence

Schemas:

```text
eve.m3-b.legacy-26-axis-capture.v1
eve.m3-b.legacy-axis-evidence.v1
eve.legacy-hormone-system.v32.capture.v1
eve.m3-b.legacy-26-axis-capture-check.v1
```

The source integrity digest is canonical SHA-256 over:

- exact source type;
- phase/stage/time/sim-hour;
- active-hormone order;
- all 26 axis names and fields.

The capture digest is canonical SHA-256 over the complete immutable capture envelope. No wall clock, memory address, random value, process ID, filesystem state, or persistence value enters either digest.

Identical source state plus identical caller-supplied identities must produce byte-equivalent material and equal digests.

## M3-B projection bridge

`LegacyHormoneCapture.to_axis_observations()` returns exactly 26 immutable `AxisObservation` values when every captured baseline is strictly inside its declared source range, as required by the merged M3-B v1 observation contract. The current default adult `HormoneSystem` satisfies that condition.

The immutable capture itself also accepts and exactly preserves valid developmental profiles whose baseline equals a source boundary. In particular, the newborn profile has estrogen and testosterone baseline/value `0.0`, equal to the exact source floor. The capture does not invent an epsilon or widen `[0,1]`.

Conversion of such a boundary-baseline capture to the stricter merged v1 `AxisObservation` contract fails closed with `AffectProjectionError`. The detached immutable source envelope remains valid; no substitute observation or partial projection input is emitted.

Each successful observation contains:

- `source_family = legacy_mutable_hormone`;
- exact original value, baseline, floor, and ceiling;
- confidence `1.0`;
- caller-supplied snapshot identity;
- capture schema version;
- complete source integrity digest;
- axis integrity digest;
- capture digest;
- tier, phase, reactivity, decay rate, and source instance identity metadata.

This conversion does not run projection, mutate drive state, derive a named transition, append an event, or start an observation window.

## Fail-closed conditions

Capture rejects:

- non-exact source type;
- non-exact source container type;
- missing, additional, reordered, or duplicated axes;
- non-exact `Hormone` objects or name/key mismatch;
- non-finite values;
- level/baseline outside `[0,1]`;
- negative dynamics;
- invalid tier or phase;
- invalid source phase/time/sim-hour;
- active-hormone mismatch;
- missing/bounded-identity violations;
- source container replacement during capture;
- any axis-object replacement during capture;
- any before/after source-state change.

Projection conversion separately rejects a preserved source baseline that is not strictly interior to the merged v1 observation range. No epsilon, widened range, fallback value, partial envelope, or partial observation tuple is emitted.

## Audit harness

`scripts/audit/m3_b_legacy_affect_capture.py`:

- parses the authoritative 26-axis order directly from `hormone_system.py`;
- constructs only a local synthetic default-adult legacy source for audit;
- captures it twice with identical deterministic identities;
- proves exact 26-axis and `AxisObservation` coverage for that admissible source;
- proves source state is byte-material equivalent before, between, and after captures;
- proves deterministic capture/digest equality;
- proves all authority and side-effect fields remain false;
- reports only the registry owner blocker as remaining.

The audit harness does not inspect or capture the production runtime object.

## Validation

```text
python scripts/audit/m3_b_legacy_affect_capture.py --summary-only --pretty --strict
pytest -q tests/test_v4_m3_b_legacy_affect_capture.py
```

Focused tests cover:

- exact source/catalog agreement;
- deterministic detached capture and source immutability;
- all preserved axis/source fields;
- exact 26-value M3-B observation bridge for the current default adult source;
- exact preservation and fail-closed bridge behavior for boundary-baseline developmental profiles;
- proof that update/stimulate surfaces are never called;
- exact-type and catalog-drift rejection;
- numeric and identity validation;
- fail-closed source-value change between reads;
- fail-closed axis-object replacement;
- deterministic audit and remaining blocker;
- AST proof of no I/O, persistence, observer, or activation surface.

## Cross-chat validation reuse

Prerequisite immutable pins:

```text
M3-B technical exact head:       0ccce6b0a39fd0b0030181d3bfdb7c99f01b7c6c
M3-B technical run:              29977749600
M3-B technical artifact SHA-256: 4a73b08e5ae4b143ebbd84963c7a6803a8964a5a47db95805fdef9b42df49ca1
M3-B technical merge:            0d755c35c994fa5b1ed3f2768c7905cda83c9a95

Source preflight exact head:       6c43f5ee87ed1104f4682fffcb1468701765a1aa
Source preflight run:              29979778124
Source preflight artifact SHA-256: 641c99378e5b6716708e62e144d0236c490b17f9a32c83bf9ece9108417b1ea8
Source preflight merge:            97ac96b8bdf54c7fbc74d4b0346ceff49624eaec
Source preflight M2-E run:         29979778146
```

Do not rerun prerequisite workflows because work moves to another chat or PR metadata changes. Rerun only after a code head change, artifact loss/corruption, digest mismatch, or validation-scope change.

The final exact-head run for this candidate will become the reusable pin for later work.

## Explicit non-goals

No production hook, automatic capture, observer installation, runtime scheduler, lock/atomic sampling implementation, registry-axis owner, registry current-value container, proposal application, full 63-axis observation, persistence read/write, SQLite access, event append, drive mutation, named-state mutation, goal/memory/self-model/expression integration, speech/external effect, observation-window acceptance, M3-B completion, M3-C, cutover, or M3-E authority.

## Intended changed-file boundary

1. `core/m3_b_legacy_affect_capture.py`
2. `scripts/audit/m3_b_legacy_affect_capture.py`
3. `tests/test_v4_m3_b_legacy_affect_capture.py`
4. `docs/audit/M3_B_LEGACY_26_AXIS_IMMUTABLE_CAPTURE_CANDIDATE.md`
5. `docs/audit/forward_additions/pr-172.json`
