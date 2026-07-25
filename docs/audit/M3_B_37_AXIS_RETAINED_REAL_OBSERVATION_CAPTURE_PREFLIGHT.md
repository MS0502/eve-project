# M3-B 37-Axis Retained Real Observation Capture Preflight

Baseline: `272ae3132547395e03af55731ba87736aa6535e8` (#185 squash merge).

## Proven boundary

All seven source-binding groups now compose to the canonical registry order:

```text
operational                     4  -> cumulative  4
appraised survival              2  -> cumulative  6
quarantined risk-defense        6  -> cumulative 12
quarantined social-relationship 7  -> cumulative 19
validated learning-exploration  6  -> cumulative 25
long-horizon self-identity       6  -> cumulative 31
AGP-bounded expression-action    6  -> cumulative 37
```

Therefore source binding is `37/37`.

That statement means only that every registry axis has a versioned, deterministic, fail-closed source validation/derivation contract. It does not prove that production data has ever been captured or retained.

## Production components still absent

The next authority boundary requires two separate components that this PR intentionally does not create:

```text
core/m3_b_registry_production_capture_adapter.py
core/m3_b_registry_retained_real_observation_sink.py
```

The first future component must acquire exact verified records from production-owned sources without synthesizing them from registry defaults, proposals, baselines, or test fixtures.

The second future component must retain immutable envelopes with enough source/raw/verification/retention integrity for deterministic replay. A transient derived value is not a retained real observation.

## Current exact state

```text
source bindings:                       37/37
production capture adapter:            absent
immutable retention sink:              absent
retained real observation:              0/37
positive-confidence real observation:   0/37
observation window eligible:            false
observation window started:             false
observation window satisfied:           false
M3-B complete:                           false
M3-C open:                               false
M3-E authority open:                     false
cutover authorized:                      false
```

## Exact blockers

```text
REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
REGISTRY_OBSERVATION_WINDOW_NOT_STARTED
```

`REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT` covers the capture boundary as a whole: either the production capture adapter or the immutable retention sink being absent is sufficient to keep the blocker active.

The observation window is not eligible merely because source binding reached 37/37. It can become eligible only after the real production capture path exists and retained positive-confidence real observations satisfy the later window-entry contract.

## Explicit non-authority

This preflight installs no runtime hook or scheduler, performs no persistence access or event append, mutates no registry owner, affect, drive, named state, goal, memory, self, or expression state, starts no window, and grants no M3-C/M3-E/cutover authority.

Audit fixtures and source-binding derivation examples remain non-production. They must never be reclassified as retained real observations.

## Validation reuse across chats

`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json` remains the handoff authority. A chat/session transition does not invalidate an exact green prerequisite. Only the recorded invalidators may trigger rerun. In particular, #185 validation must be reused while its exact head, artifact SHA, workflow/dependency scope, and merge ancestry remain valid.

## Next step after this preflight

Implement the production capture adapter and immutable retained-real-observation sink as a new capability-boundary PR. That implementation may provide the machinery to retain real observations, but it still must not claim positive-confidence 37/37 coverage or start the observation window until actual production-origin records exist and pass the source contracts.
