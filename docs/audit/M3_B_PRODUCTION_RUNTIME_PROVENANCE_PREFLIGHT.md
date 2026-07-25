# M3-B Production Runtime Provenance Preflight

## Baseline and reused prerequisite

- baseline / PR #189 squash merge: `2e2ad2ffa4d320ad9f8439a6b13097fa72fa40bb`
- PR #189 exact head: `897e7e90e7acb9263e7ac103f4d9314474346f9b`
- exact run: `30145767939`
- focused: `7 passed`
- full: `3,120 passed`
- artifact SHA-256: `94c3fe69bae141ce140cea9a48c1292a253bb25fbb84040dfaa57cb5787ebfa2`
- M2-E run: `30145767966` (`6/6` jobs passed)

The exact-head evidence above is recorded in
`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json`. Chat/session changes, PR metadata,
comments/reviews, and Draft/Ready transitions do not invalidate it.

## Repository finding

The repository has an actual execution path for the `prediction_error_pressure` trace:
`main.py -> repl()/build_full_engine() -> StreamingEngine -> AIAdapter -> ActiveInference`.
That proves the source-shaped runtime trace exists in normal application execution.

The repository does **not** currently contain an independently trusted production
runtime provenance verifier or launch-attestation trust root. Therefore none of the
following may be treated as proof of production origin by itself:

- Python `__main__` execution;
- the `main.py:repl` path;
- a process ID;
- argv or environment flags;
- a caller-selected source/runtime instance ID;
- `fixture_only=False`;
- a digest produced by the observed runtime over its own launch metadata.

These are integrity or routing facts, not independent provenance.

## Contract added by this PR

`core/m3_b_production_runtime_provenance_preflight.py` separates an **untrusted
candidate** from an **issued verified provenance proof**.

A `RuntimeProvenanceCandidate` may describe:

- trust domain;
- runtime and source instance IDs;
- exact repository head SHA;
- entrypoint ID;
- launch-attestation ID and digest;
- logical tick;
- fixture classification.

Candidate construction never establishes production origin.

A future trusted proof can be issued only by
`execute_registered_runtime_provenance_verifier(...)`, which requires:

1. exact candidate type;
2. attestation material whose canonical digest exactly matches the candidate;
3. an exact executable verifier registration for the candidate trust domain;
4. verifier output that binds the exact runtime/source/head/entrypoint/attestation and
   candidate digest;
5. independent trust-root verification;
6. verified production launch provenance;
7. verified non-CI runtime status;
8. verification at or after the candidate logical tick.

The final `ProductionRuntimeProvenanceVerification` uses a private `InitVar` issuance
proof. Direct construction fails closed, and `dataclasses.replace(...)` cannot clone an
issued proof while carrying issuance authority forward.

## Deliberately empty trust registry

`REGISTERED_RUNTIME_PROVENANCE_VERIFIERS` is an immutable `MappingProxyType({})`.
This PR registers no trust domain and therefore cannot issue production provenance.
A process-local test may monkeypatch a test-only verifier to prove the boundary, but a
`test_fixture` verification explicitly reports `counts_as_production == false`.

The existing PR #188 production source-verifier registry also remains empty and
immutable.

## State after this preflight

```text
prediction_error runtime source bridge:          present
runtime provenance contract:                     present
registered runtime provenance verifiers:          0
verified production runtime anchors:              0
registered production source verifiers:           0/37
retained real observation:                        0/37
positive-confidence real observation:             0/37
observation window eligible:                       false
observation window started:                        false
M3-B complete:                                     false
M3-C open:                                         false
M3-E authority open:                               false
cutover authorized:                                false
```

New blocker:

`PRODUCTION_RUNTIME_PROVENANCE_VERIFIER_ABSENT`

Existing blockers remain, including
`PREDICTION_ERROR_PRODUCTION_RUNTIME_PROVENANCE_ANCHOR_ABSENT`.

## What must happen outside this preflight

The next step cannot honestly be `registered verifier = 1/37` until an independently
verifiable launch-attestation source exists for the actual deployment environment.
The future implementation must identify that trust source, verify it without relying
on self-authored runtime claims, bind it to the exact deployed repository head and
runtime/source instance, and only then register the provenance verifier.

Until then, GitHub/CI tests may prove the boundary but cannot manufacture the real
production event the boundary is designed to attest.
