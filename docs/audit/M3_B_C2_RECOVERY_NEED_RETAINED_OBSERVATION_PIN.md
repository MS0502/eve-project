# M3-B C2 Recovery-Need Retained Observation Pin

## Baseline

Repository activation head: `715f0b6da087add988d9628d083354505ffc064d` — PR #209 squash merge.

GitHub `main` was directly compared with that SHA and was identical before this pin branch was created. The operator then ran the merged sequence-four retention command exactly once on the clean phone checkout. The command performed the real operator-private SQLite append/readback and returned the public-safe receipt now stored in `M3_B_C2_RECOVERY_NEED_RETENTION_RECEIPT.json`.

## Receipt verification

Pinned public-safe identities:

```text
axis:                                  recovery_need
public review digest:                  e46df034d01b13e768ce37d14261b8ed20fdec30101945bea492d97e482e4c33
attestation digest:                    ce8cda9955a415ed05200a83fd6b3e8d4cd4028bef29f73b8b17a1d5e3ad25e1
evidence digest:                       535495759c0140d875da628d2fe5cc9ffc0904d5f91fa9546a784dd51b3baa4b
capture digest:                        66daf0cfd4e37fb63bd98e5def7dde3118b5a965be32d93ff9ffa74b25554974
runtime provenance verification:       a2cffc3683640871acd1b682cc227ea26180b1a285e9412d676f64eb6a5e45eb
source verification:                   b6421cc70f20a3c3f3cd082507876079bbce1cb42c57e639d02c013ac79c65c4
prior event envelope digest:            f81d43bf40b4dc76130767f91b65ad2503bc70e61ef718fe3d0e446528d1a7e3
event envelope digest:                  7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691
store transition hash:                  f36add507ebab0913ce29b6c2911d169e634cc59e9902e775bb4ee21e5fb2385
store after chain digest:               16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f
receipt digest:                         e776859e16b34a9222264f3d500993e4bf56ce2397c73b132075cb51fe3967c6
```

The receipt digest was independently recomputed from the canonical receipt mapping using sorted compact JSON SHA-256 and matched `e776859e16b34a9222264f3d500993e4bf56ce2397c73b132075cb51fe3967c6` exactly.

The receipt proves:

```text
prior_event_id:                         m3b:c2:retained:fatigue_pressure:000003
event_id:                               m3b:c2:retained:recovery_need:000004
sequence:                               4
store_before_count:                     3
store_after_count:                      4
store_ordinal:                          4
readback_verified:                      true
retained_real_observation_delta:        1
retained_real_observation_count:        4/37
observation_window_started:             false
M3-B complete:                          false
M3-C open:                              false
M3-E authority open:                    false
cutover authorized:                     false
```

## Private companion boundary

The repository does **not** contain the SQLite database, WAL, raw operational measurements, raw witness material, private nonce, or private filesystem path. `database_location=operator_private_companion_only` is the only location disclosure retained publicly.

Sequences 1 through 4 are immutable prior history. None may be replayed to increase coverage.

## Exact current M3-B boundary after this pin

```text
source bindings:                                  37/37
reviewed real operator attestations (C2):          4
registered runtime provenance verifiers (C2):     4
verified production runtime anchors (C2):         4
registered production source verifiers (C2):      4/37
verified positive-confidence candidates:          4/37
retained real observation:                        4/37
retained positive-confidence real observation:    4/37
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                    false
M3-C open:                                        false
M3-E authority open:                              false
cutover authorized:                               false
```

This is a real counter advance from `3/37` to `4/37`, but it is not an observation-window start and grants no later authority.

## Validation reuse boundary

PR #209 was already validated on exact head `d1aebe1d7f0ed843fb34c8403ec9929fb8684e01`:

```text
exact run:    30201501483
focused:      5 passed
full:         3,195 passed
artifact:     exact-head-validation-d1aebe1d7f0ed843fb34c8403ec9929fb8684e01
artifact SHA: ad4bddb00b859da91d13068c4c671c257e6472c708e6d47ede4fbcf33a4b24be
M2-E run:     30201501481
M2-E:         6/6 passed
merge SHA:    715f0b6da087add988d9628d083354505ffc064d
```

The artifact was directly rechecked on GitHub as present, unexpired, and digest-matching, and both #209 workflows remain successful. Those #209 validations are reused as the prerequisite and must not be rerun because work moved to this PR/chat. This receipt-pin PR is a new repository head and may receive its own final-head validation according to the reuse policy; the prior #209 full/M2-E runs are not repeated.

## Next boundary

The direct-operational four-axis retained set is now complete: `prediction_error_pressure`, `energy_budget`, `fatigue_pressure`, and `recovery_need`. Continue M3-B with a new independently witnessed production-origin axis from the remaining 33. The next source-binding group is the appraised survival pair; `stress_load` is the next candidate only after a separate fail-closed production-origin witness/acquisition surface is reviewed. Do not reinterpret detached appraisal fixtures or caller-supplied test records as production evidence.

Any future retained append must be sequence 5 and prove exact continuity from the sequence-4 `recovery_need` event/chain above. Existing sequences 1 through 4 are never append targets again. The M3-B observation window remains closed until the later 37-axis retained positive-confidence coverage/window-entry contract is satisfied.
