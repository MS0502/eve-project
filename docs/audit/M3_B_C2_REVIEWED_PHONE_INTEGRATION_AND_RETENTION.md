# M3-B C2 Reviewed Phone Integration and Retention Boundary

## Reviewed real witness

The operator produced one non-CI full-engine phone witness from exact repository head `b4968be9aeb6eefc7274f9985ab333f08e470daf` using the merged #197/#198 witness path. The repository pins only the public-safe v2 review material in `M3_B_C2_REVIEWED_PHONE_WITNESS.json`; the private nonce and raw prediction/error snapshots remain in the operator-private companion.

Pinned identities:

```text
public review digest: 6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e
attestation digest:   85b55eee61618ad98476f71c4dadcb9b2e4383d79aefd93a41a2c34634efecda
evidence digest:      14549d2b9f37f2a8b00a5bc9de61dbdad8e12dbb8a4d4e08e254ef0e9848b3dc
source contract:      eve:m3-b:registry-source:prediction_error_pressure:v1
source instance:      runtime:ai-adapter:primary
fixture_only:         false
```

The witness has two raw records spanning one logical tick, positive confidence `1.0`, and a canonical public-review digest that is recomputed before any C2 verification is issued.

## Why C2 uses a versioned activation layer

The C1 and production-capture preflight modules intentionally contain immutable empty registries and executable tests proving that an unreviewed caller cannot self-promote production claims. Those artifacts remain historical fail-closed preflight evidence.

`core/m3_b_c2_reviewed_phone_integration.py` is the versioned reviewed activation layer above those preflights. It contains three immutable C2 registries:

- one exact reviewed operator attestation;
- one runtime-provenance verifier for `eve.operator-attestation.primary.v1`;
- one production-source verifier for `prediction_error_pressure`.

The module recomputes the complete public-review digest, reconstructs the exact C1 public attestation and positive-confidence evidence, verifies the pinned source-manifest contract, and issues token-protected runtime/source verification objects. Direct construction or `dataclasses.replace` cannot manufacture issued verification.

After this PR merges, the reviewed integration state is:

```text
reviewed real operator attestations:             1
registered C2 runtime provenance verifiers:      1
verified production runtime anchors:             1
registered C2 production source verifiers:       1/37
verified positive-confidence candidates:         1/37
retained real observations:                      0/37
M3-B observation window eligible:                false
M3-B observation window started:                 false
M3-B complete:                                   false
M3-C open:                                       false
M3-E authority open:                             false
cutover authorized:                              false
```

## Durable retention is a separate real-world boundary

A disposable CI SQLite database can prove append/readback mechanics, but it cannot honestly increment the project's retained-real-observation counter. The first real retained observation therefore remains operator-side work after this PR merges.

`scripts/operator/m3_b_c2_retain_reviewed_prediction_error.py`:

1. requires an exact clean post-merge repository head;
2. reads the already-produced public-review v2 JSON from outside the repository;
3. recomputes and verifies the reviewed C2 attestation/evidence;
4. opens a dedicated operator-private `SQLiteShadowStore`;
5. refuses retention if the C2 retained-observation stream is already non-empty;
6. appends exactly one `m3_b.registry.retained_real_observation` shadow-only event;
7. verifies exact durable readback and chain advancement;
8. writes and prints only a public-safe receipt.

The real receipt must be returned for a later repository pin before `retained real observation` can move from `0/37` to `1/37`. No CI test receipt, PR merge, verifier registration, or public witness alone may substitute for that phone append.

## Authority boundary

Neither reviewed verification nor the future one-event retention append starts the M3-B observation window. Neither operation mutates live affect, drive, goals, memory, self-model, expression, or registry-owner state. M3-C, M3-E, and cutover remain closed until their later explicit reviewed gates are satisfied.

The M2-E habitat window remains separate. Its scripted shadow workload cannot be relabeled as this C2 runtime/source observation.
