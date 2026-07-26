# M3-B C2 Phone Prediction-Error Witness Preflight

## Baseline and validation reuse

Baseline is PR #194 squash merge `a09ebb8abbbf68c9235795a7c89d8b8ea5d75378`.
PR #194's accepted exact-head evidence remains reusable as the prerequisite because the
artifact is present with the recorded SHA-256 and the merge ancestry is intact. A new
chat/operator session is not an invalidator. This PR is a new code head and therefore
requires its own final exact-head validation after forward registration.

## Why this preflight exists

C1 created an operator-private HMAC trust root but intentionally registered zero real
phone attestations. The existing `prediction_error_pressure` bridge can read an actual
completed `AIAdapter -> ActiveInference` prediction/error trace, but it intentionally
cannot claim production origin.

The M2-E habitat supervisor is not a substitute: it runs a synthetic scripted shadow
workload. Its status/logs are M2-E continuity evidence and must never be relabeled as the
M3-B production runtime/source witness.

This preflight supplies the missing operator-side acquisition surface for one actual
phone full-engine session while keeping the C2 review/registration boundary closed.

## Operator witness behavior

`scripts/operator/m3_b_phone_prediction_error_witness.py`:

1. requires an exact expected Git head and a clean checkout;
2. requires the existing operator nonce from a file outside the repository with private
   permissions;
3. creates no nonce and stores no secret in repository state;
4. builds the normal full EVE engine with `main.build_full_engine()`;
5. executes exactly two operator-supplied real interactions through
   `StreamingEngine.chat_stream(...)`;
6. after each completed interaction, reads the already-existing AIAdapter prediction/error
   trace through the merged read-only bridge;
7. requires the two records to satisfy the canonical `prediction_error_pressure` minimum
   of two raw records spanning at least one logical tick;
8. builds the C1 public launch attestation and immediately recomputes its private binding
   locally;
9. writes recalculable raw trace/evidence material only into an operator-private companion
   outside the repository;
10. emits only a public-safe digest/review JSON object on stdout.

The fixed source identity is `runtime:ai-adapter:primary` by default. The runtime instance
ID and launch-attestation ID are explicit operator arguments. They are identifiers, not
trust roots. Production trust remains the C1 private-nonce binding plus later exact
repository review.

## Public/private boundary

Private companion material may contain:

- the public attestation record;
- two complete immutable prediction/error source snapshots;
- the detached positive-confidence evidence mapping;
- the local private-binding verification trace digest.

It never serializes the private nonce itself.

The public review object contains only:

- public C1 attestation fields/digests;
- local verification trace digest;
- source/evidence integrity digests;
- record count and observed tick;
- private-material digest/reference classification;
- explicit false values for registration, retention, observation-window, M3 completion,
  M3-C/M3-E, and cutover claims.

Raw prediction/error mappings and the private nonce must not be pasted into GitHub or chat.

## What this PR deliberately does not do

After this preflight merges, counters remain:

```text
reviewed real operator attestations:          0
registered runtime provenance verifiers:      0
verified production runtime anchors:          0
registered production source verifiers:       0/37
retained real observation:                    0/37
positive-confidence real observation:         0/37
M3-B observation window started:               false
M3-B complete:                                 false
M3-C open:                                     false
M3-E authority open:                           false
cutover authorized:                            false
```

A non-fixture witness produced by the script is still only review material until its exact
public attestation is reviewed and pinned by a later repository change. The script does not
mutate `REVIEWED_OPERATOR_ATTESTATIONS`, `REGISTERED_RUNTIME_PROVENANCE_VERIFIERS`, or
`REGISTERED_PRODUCTION_SOURCE_VERIFIERS`, and it does not invoke the retained-real-
observation sink.

## C2 entry after this preflight

After merge, the phone operator may run the witness against the exact merged head. The
resulting public-safe JSON can then be reviewed. Only a later exact-head PR carrying that
specific reviewed digest may:

1. add the exact reviewed attestation registration;
2. add an executable runtime-provenance verifier for the C1 trust domain;
3. add an executable production source verifier for
   `eve:m3-b:registry-source:prediction_error_pressure:v1`;
4. verify the exact witnessed evidence;
5. append exactly one eligible retained-real-observation record.

One retained observation remains one observation. It cannot be reported as 37/37
production coverage and cannot automatically open M3-C, M3-E, or cutover authority.
