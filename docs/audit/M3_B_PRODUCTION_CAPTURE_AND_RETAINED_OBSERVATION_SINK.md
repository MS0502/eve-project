# M3-B Production Capture Adapter + Immutable Retained-Observation Sink

Baseline: `b92a16a81e1591e490edc36d0171bc9a2c3bf065` (#186 squash merge).

## What this stage implements

This capability-boundary stage adds the two code surfaces identified by the retained-real-observation preflight:

```text
core/m3_b_registry_production_capture_adapter.py
core/m3_b_registry_retained_real_observation_sink.py
```

The retention sink reuses the already validated M2-A `SQLiteShadowStore` instead of introducing a second persistence engine. That store requires an explicit concrete path and explicit initialization, uses SQLite WAL + `synchronous=FULL`, maintains a SHA-256 event chain, rejects duplicate/out-of-order events, verifies readback before commit, and has SQL triggers that reject UPDATE/DELETE of append-only state.

The M3-B sink wraps that existing store with a registry retained-observation event contract. Persisted events remain `shadow_only`.

## Critical anti-fabrication boundary

Code presence is not production evidence.

A `ProductionSourceVerification` object alone is also not sufficient. `ProductionCaptureRecord` additionally requires its exact `source_contract_id -> (verifier_id, verifier_version)` pair to be present in the closed registration table:

```text
REGISTERED_PRODUCTION_SOURCE_VERIFIERS
```

This PR intentionally leaves that table empty.

Therefore this PR can prove that the capture adapter and durable immutable sink exist, but it **cannot produce or retain a real observation in repository/runtime state yet**. Tests may monkeypatch a temporary verifier registration only to exercise the append machinery against a temporary SQLite file; those fixtures are test evidence and must never be counted as production observations.

## Current exact state after this capability PR

```text
source bindings:                         37/37
production capture adapter:              present
immutable retention sink:                present
registered production source verifiers:  0/37
retained real observation:                0/37
positive-confidence real observation:     0/37
observation window eligible:              false
observation window started:               false
M3-B complete:                             false
M3-C open:                                 false
M3-E authority open:                       false
cutover authorized:                        false
```

The former blocker:

```text
REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT
```

is resolved at the machinery-presence level only. It is replaced by the stricter source-integration blocker:

```text
REGISTRY_PRODUCTION_SOURCE_VERIFIER_COVERAGE_INCOMPLETE
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
REGISTRY_OBSERVATION_WINDOW_NOT_STARTED
```

## What the capture adapter requires

For an axis to become retention-eligible, all of these must bind exactly:

- canonical registry axis and source-contract ID,
- source family from the 37-axis source manifest,
- source instance ID and source snapshot ID,
- source schema version,
- source integrity digest,
- raw observation digest,
- derived positive-confidence evidence digest,
- production verifier ID + version,
- verifier trace digest,
- verification logical tick,
- explicit production environment,
- non-synthetic, non-proposal, non-registry-owner origin.

The verifier pair must also be registered for that source contract. A test fixture is rejected even if its verifier name is registered.

## What the sink guarantees

`RetainedRealObservationSink`:

- performs no I/O on construction,
- never auto-initializes storage,
- accepts only exact `ProductionCaptureRecord` objects,
- serializes a deterministic `m3_b.registry.retained_real_observation` event,
- appends through the existing `SQLiteShadowStore`,
- requires exact envelope digest/readback evidence from the underlying append receipt,
- returns a frozen receipt proving one append and chain advancement,
- does not mutate the registry owner,
- does not start the observation window,
- does not complete M3-B,
- does not open M3-C/M3-E,
- does not authorize cutover.

## Why verifier registration is separate

The six source families require different production proofs. Hardware/operational metrics, quarantined risk/social appraisals, validated learning traces, long-horizon self review, and AGP-bounded expression/action traces do not share one trustworthy runtime origin.

Registering a generic verifier here would silently turn caller assertions into “real observations.” The next stage must instead add reviewed source-contract-specific runtime bridges/verifiers and prove their origin independently.

## Next stage

Start production verifier/source-bridge integration. The first successful retained observation may be counted only when it comes from an actually registered production bridge and survives the immutable sink. Test/audit fixtures remain excluded.

The observation window still cannot start until the later coverage/window-entry contract is satisfied by actual retained production-origin evidence.
