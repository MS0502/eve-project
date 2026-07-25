# M3-B Prediction-Error Runtime Source Bridge

## Baseline and prerequisite reuse

- baseline / PR #188 squash merge: `9566c07f89b6aba01f2d875a12e68f7767ab2fdb`
- PR #188 exact head: `7c76276a1f1ec17afc31c3cc7a85f349ca6f3e8b`
- exact run: `30145022292`
- focused: `14 passed`
- full: `3,113 passed`
- artifact SHA-256: `07cd81a95cf920577776e72e9daa8de5449074c515488bff590ab82dab8a2c4c`
- M2-E run: `30145022290` (`6/6` jobs passed)

The evidence above is pinned in
`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json`. A chat/session change, PR body
edit, review, or Draft/Ready transition is not a reason to rerun PR #188 full-suite or
M2-E.

## Why this axis is a real runtime-source candidate

Repository inspection found an existing call chain for `prediction_error_pressure`:

```text
main/build_full_engine
  -> StreamingEngine(ai_adapter=AIAdapter(...))
  -> StreamingEngine._stream_from_plan(...)
       -> AIAdapter.predict_before_response()
          -> ActiveInference.predict()
       -> response generation
       -> AIAdapter.observe_after_response(...)
          -> ActiveInference.observe()
```

`ActiveInference.predict()` stores an exact prediction record. `observe()` marks that
prediction observed, calculates `mood_error` plus per-hormone error material, appends
an error record, and increments `observe_count`.

This means the repository already has runtime-shaped prediction/error material. It
does **not** mean that a retained real observation already exists.

## Bridge added by this PR

`core/m3_b_prediction_error_runtime_source_bridge.py` is a read-only bridge over an
already-completed trace.

It may:

1. read the current `AIAdapter._last_prediction` reference and latest
   `ActiveInference.errors[-1]` record;
2. require that the prediction is already observed and that prediction/error IDs match;
3. freeze both mappings into canonical immutable JSON material;
4. derive the source-contract fields required for `prediction_error_pressure`;
5. convert the snapshot into the existing detached `ValidatedLearningRawRecord`;
6. derive detached `RegistryAxisPositiveConfidenceEvidence` when at least two records
   with at least one logical-tick span are supplied.

It does **not** call `predict()`, `observe()`, `tick()`, or any learning/persistence
surface. It does not install itself into `StreamingEngine`.

## Exact raw mapping

The existing source manifest requires:

```text
model_version
normalized_error
observed_value_digest
predicted_value_digest
verification_status
```

The bridge maps them as follows:

- `model_version` = `eve.active-inference.prediction-error-trace.v1`
- `normalized_error` = `mood_error / 3.0`
- `predicted_value_digest` = SHA-256 of canonical prediction identity, expected state,
  expected outcome, confidence, and horizon material
- `observed_value_digest` = SHA-256 of canonical prediction-linked error material
- `verification_status` = `verified`

`mood_error / 3.0` is bounded because the current hormone mood contract clips valence
to `[-1, 1]` and arousal to `[0, 1]`; the current `ActiveInference.observe()` error is
the sum of absolute valence and arousal differences, so the exact maximum is `3.0`.

`verification_status=verified` means only that the detached trace structure/digests
satisfy the existing source-binding contract. It is **not** production-origin proof.

## Existing runtime quirk deliberately not changed

`StreamingEngine` currently supplies response outcomes through its existing inference
path, while `ActiveInference.observe()` normalizes unsupported outcome labels to its
own outcome vocabulary. This PR does not change that legacy behavior. The bridge's
`prediction_error_pressure` value uses the recorded mood-error signal, not a claim
that the outcome-label path is semantically corrected.

## Anti-fabrication boundary

The bridge deliberately has no trusted production-runtime provenance anchor.
Consequently:

```text
prediction_error source bridge:               present
runtime hook installed by bridge:              false
trusted production runtime provenance:         false
production source verifier registered:         false
retained real observation:                     0
positive-confidence real observation:          0
observation window eligible:                    false
observation window started:                     false
M3-B complete:                                  false
M3-C open:                                      false
M3-E authority open:                            false
cutover authorized:                             false
```

A caller may pass `fixture_only=False`, but that flag is not trusted provenance and
cannot set `production_origin_verified`. The snapshot always remains unverified for
production origin. The #188 production verifier registry also remains empty and
immutable.

The new explicit blocker is:

```text
PREDICTION_ERROR_PRODUCTION_RUNTIME_PROVENANCE_ANCHOR_ABSENT
```

The existing production-verifier, positive-confidence-real-coverage, and observation-
window blockers remain active.

## What a later PR must prove before registration

A later source-integration PR must establish a trusted production-runtime provenance
anchor that cannot be caller-authored or test-fixture-authored, bind it to the exact
runtime trace, and implement the reviewed executable verifier contract from PR #188.
Only then may the repository consider registering the
`prediction_error_pressure` production verifier.

Until that proof exists, this bridge is capability/preflight only and contributes
`0/37` production verifier coverage and `0/37` retained-real-observation coverage.
