# M3-B Production Verifier Issuance Boundary

## Baseline

- PR #187 merge: `7f2d755e28732e49ac2849d108c1d6f1e5255b7c`
- PR #187 exact head: `1f6c1d7746a335ef92e67940a7d5b791eef0ccdf`
- exact run: `30144115559`
- focused: `19 passed`
- full: `3,109 passed`
- artifact SHA-256: `16e209c82ef329213f144fed5b4a139f7c2a06a0ffe0190084b355b889e2a0b9`
- M2-E run: `30144115571` (`6/6` jobs passed)

The exact-head evidence above is recorded in
`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json`. A new chat, operator session,
PR body edit, review, or Draft/Ready transition does not invalidate it. Do not rerun
PR #187 full-suite or M2-E unless one of the ledger invalidators applies.

## Problem closed by this PR

PR #187 correctly left `REGISTERED_PRODUCTION_SOURCE_VERIFIERS` empty, so it could
not create a retained real observation. Its future registration shape, however, was
only `source_contract_id -> (verifier_id, verifier_version)`. If a later PR merely
populated that metadata table, a caller could construct matching
`ProductionSourceVerification` fields without proving that the registered verifier
actually executed against a production source.

This PR closes that future issuance gap before any production source is registered.

## Exact boundary

`REGISTERED_PRODUCTION_SOURCE_VERIFIERS` now accepts only
`ProductionSourceVerifierRegistration` objects containing:

- exact source contract ID;
- verifier ID and version;
- an executable verifier callable;
- exact shadow-only registration schema.

A caller cannot create an acceptable `ProductionSourceVerification` from metadata
alone. The only supported issuance path is:

1. `execute_registered_production_verifier(...)` resolves the manifest contract;
2. it requires the exact executable registration;
3. it executes that verifier over explicit source material;
4. the callable must return an exact immutable `ProductionSourceVerifierResult`;
5. the result must bind the exact source identity and digests of the supplied
   positive-confidence evidence;
6. only then does the capture module issue the immutable
   `ProductionSourceVerification` used by `ProductionCaptureRecord`.

Direct caller construction fails closed before a capture can be created.

## State intentionally unchanged

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

This PR installs no runtime source hook, performs no source polling, initializes no
store, appends no event, mutates no registry owner/affect/drive/named state, and grants
no observation-window or cutover authority.

## Next reviewed source candidate

Repository inspection found one concrete production-path candidate worth a separate
source-integration PR: `prediction_error_pressure`.

The normal response path in `language/streaming.py` invokes
`AIAdapter.predict_before_response()` before response generation and invokes
`AIAdapter.observe_after_response(...)` after the actual response outcome is inferred.
The underlying active-inference implementation records prediction/error material.
The source manifest requires for `prediction_error_pressure`:

- `model_version`
- `normalized_error`
- `observed_value_digest`
- `predicted_value_digest`
- `verification_status`

This is only a candidate bridge. It is **not** registered by this PR and is not counted
as a retained real observation. A later PR must prove an exact runtime-source mapping,
implement the source-contract-specific executable verifier, and preserve the same
anti-fabrication boundary before any registration is allowed.
