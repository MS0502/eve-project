# M3-B C2 Reviewed Fatigue-Pressure Integration and Sequence-Three Retention

## Baseline

Exact merged prerequisite: `1ac94c402d6fb8935614d0a72cda3e622b69ec82` — PR #205 squash merge.

The operator then executed the merged phone `fatigue_pressure` witness exactly once on that clean head. The public review supplied for repository review was independently canonical-digest checked before this PR:

```text
attestation digest: 421da78df1035dd994df3098c1345a448fca59b7a36f9d8cc2fb8c3dce0d4db8
evidence digest:    017e189e1a35a26ce47a0372fe558e069679bf03e438ff9767bf3e0f4196a707
public review:      4b88c7734234ac2982836b95bf392fe143bc928119d4af515e576b39e480af61
witness head:       1ac94c402d6fb8935614d0a72cda3e622b69ec82
axis:               fatigue_pressure
confidence:         0.9999916437617128
value:              0.24117046163321723
fixture_only:       false
synthetic:          false
```

The attestation mapping, evidence mapping, and complete public-review mapping were each independently serialized with sorted compact canonical JSON and SHA-256; all three recomputations matched the supplied digests exactly.

## Reviewed activation

This PR pins the exact public-safe witness and installs a third immutable C2 reviewed activation layer:

```text
reviewed real operator attestations:          3
registered runtime provenance verifiers:     3
verified production runtime anchors:         3
registered production source verifiers:      3/37
verified positive-confidence candidates:      3/37
retained real observations before phone run:  2/37
```

The activation verifies the exact PR #205 source contract:

```text
axis:                  fatigue_pressure
source contract:       eve:m3-b:registry-source:fatigue_pressure:v1
source family:         operational_metrics_or_appraised_load_trace
minimum records:       3
minimum logical span:  2
hardware direct:       true
appraisal required:    false
```

The reviewed evidence remains detached/read-only. Repository review does not itself append a retained observation.

## Sequence-three staging

The operator-private append path is staged as sequence 3 only:

```text
new event:       m3b:c2:retained:fatigue_pressure:000003
sequence:        3
prior event:     m3b:c2:retained:energy_budget:000002
prior envelope:  1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664
prior chain:     d4660b5cef058bad1b9d1b6b1cb2987c78ef9dbbee403c85562ab945535883e0
```

The staged append fails closed unless the private store contains exactly two retained events and both match the immutable public pins for sequence 1 `prediction_error_pressure` and sequence 2 `energy_budget`. It cannot recreate, replace, or replay either prior event.

A successful operator run must prove:

```text
store ordinal:                         3
store before -> after:                 2 -> 3
retained delta:                        1
retained count after append:           3
readback verified:                     true
observation window started:            false
M3-B complete:                          false
M3-C open:                              false
M3-E authority open:                    false
cutover authorized:                     false
```

## Private companion boundary

The repository contains only the reviewed public witness and exact cryptographic identities. It does not contain the private witness file, nonce, raw process CPU/wall timing, kernel load observations, context-switch counters, SQLite database, WAL, or private filesystem path.

## Validation reuse

PR #205 is a merged immutable prerequisite. Its final validation must be reused rather than repeated because work moved to this PR/chat:

```text
exact head:   26c3c0058b444ee94e6ddd7b8cf8590c901a761d
exact run:    30195049107
focused:      6 passed
full:         3,178 passed
artifact:     exact-head-validation-26c3c0058b444ee94e6ddd7b8cf8590c901a761d
artifact SHA: b6ea22b246cfbd79ade0832906e365a0122cfd35e9c017f796d9e95192f2a16f
M2-E run:     30195049097
M2-E:         6/6 passed
merge SHA:    1ac94c402d6fb8935614d0a72cda3e622b69ec82
```

The #205 discovery head is not reusable merge evidence; its full suite was skipped and its M2-E run was cancelled after supersession.

## Authority boundary

This PR does not claim the real sequence-three append has already happened. Until the operator executes the merged retention command and returns the public receipt, canonical retained coverage remains `2/37` even though reviewed/verifier/candidate coverage becomes `3/37`.

The M3-B observation window remains closed. M3-B is incomplete. M3-C, M3-E, and cutover remain closed.
