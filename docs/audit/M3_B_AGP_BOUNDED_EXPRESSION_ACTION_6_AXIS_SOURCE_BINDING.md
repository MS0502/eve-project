# M3-B AGP-Bounded Expression-Action Six-Axis Source Binding

Baseline: `4adbb575ffe517512af74c74fce58a49f7996128` (#184 squash merge).

## Scope

This binding closes the final six source-manifest slots:

- `expression_pressure`
- `expression_inhibition`
- `action_readiness`
- `risk_tolerance`
- `patience_level`
- `conflict_avoidance`

Each accepted record is detached and caller-supplied. It must bind an immutable source snapshot to a versioned AGP verification trace and a separate bounded appraisal. The appraisal input digest must be the exact AGP trace output digest.

Both `passed` and explicit `failed_bounded` AGP outcomes can be observed. An AGP failure is therefore evidence about inhibition/fallback pressure, not permission to bypass AGP.

## Fail-closed boundaries

Rejected inputs include:

- raw social feedback as an unvalidated source,
- direct hardware input,
- synthetic or proposal-only records,
- registry-owner circular sourcing,
- runtime-polled records,
- expression or action execution,
- memory writes,
- cutover authorization,
- broken AGP → appraisal digest chains,
- unverified AGP/appraisal traces,
- noncanonical raw-field order or versions.

## What 37/37 means

After this binding set, source binding coverage is exactly `37/37`. This means the repository has a deterministic validation/derivation contract for every registry axis when valid source records are supplied.

It does **not** mean that any retained production observation exists.

```text
source bindings:                       37/37
production capture:                    absent
retained real observation:              0/37
positive-confidence real observation:   0/37
observation window started:             false
M3-B complete:                           false
M3-C open:                               false
M3-E authority open:                     false
cutover authorized:                      false
```

The binding-set contract therefore keeps these blockers:

```text
REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

The next artifact is a separate 37-axis retained-real-observation capture preflight. That preflight must not fabricate production data, start the observation window, or authorize cutover.

## Validation discipline

The exact-head reuse ledger is authoritative across chat/session boundaries. A new chat is not a reason to rerun a green merged prerequisite. Discovery may run focused/M0/M2-B checks, but full-suite execution remains blocked until exact M2-B and forward registrations are complete. Only the final registered head may run the full suite once.
