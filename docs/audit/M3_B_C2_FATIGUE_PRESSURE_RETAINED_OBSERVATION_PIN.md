# M3-B C2 Fatigue-Pressure Retained Observation Pin

## Baseline

Repository activation head: `08ae20479ab387f8e8962e3b8cbf3cc182a66fca` — PR #206 squash merge.

GitHub `main` was directly compared with that SHA and was identical before this pin branch was created. The operator then ran the merged sequence-three retention command exactly once on the clean phone checkout. The command performed the real operator-private SQLite append/readback and returned the public-safe receipt now stored in `M3_B_C2_FATIGUE_PRESSURE_RETENTION_RECEIPT.json`.

## Receipt verification

Pinned public-safe identities:

```text
axis:                                  fatigue_pressure
public review digest:                  4b88c7734234ac2982836b95bf392fe143bc928119d4af515e576b39e480af61
attestation digest:                    421da78df1035dd994df3098c1345a448fca59b7a36f9d8cc2fb8c3dce0d4db8
evidence digest:                       017e189e1a35a26ce47a0372fe558e069679bf03e438ff9767bf3e0f4196a707
capture digest:                        6175cdf097fdf86b2bec71f8abd440b4ac977d03abd0f32378834f69543d7b84
runtime provenance verification:       d85649f06033a44e20d880a5f8acc3823e1a474306c437c19310acca4aeedb62
source verification:                   060560b0981aa6ee7cd7cdc6409a549fc531ae1cf0bfdcb561ff8c89041d030e
prior event envelope digest:            1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664
event envelope digest:                  f81d43bf40b4dc76130767f91b65ad2503bc70e61ef718fe3d0e446528d1a7e3
store transition hash:                  7d0e608b245506836722d3bbbe609e29f8bdac55f435958ac0f69e456aee4929
store after chain digest:               b73ec7ea2f5e6e4e8eda5b57b4f6464a17d94e56026718b5b2e15cbca9f2162f
receipt digest:                         cef1b731eb6b3b15ebef1106bea3f12d2b053afbfd18b35aebeaca46dd143f66
```

The receipt digest was independently recomputed from the canonical receipt mapping using sorted compact JSON SHA-256 and matched `cef1b731eb6b3b15ebef1106bea3f12d2b053afbfd18b35aebeaca46dd143f66` exactly.

The receipt proves:

```text
prior_event_id:                         m3b:c2:retained:energy_budget:000002
event_id:                               m3b:c2:retained:fatigue_pressure:000003
sequence:                               3
store_before_count:                     2
store_after_count:                      3
store_ordinal:                          3
readback_verified:                      true
retained_real_observation_delta:        1
retained_real_observation_count:        3/37
observation_window_started:             false
M3-B complete:                          false
M3-C open:                              false
M3-E authority open:                    false
cutover authorized:                     false
```

## Private companion boundary

The repository does **not** contain the SQLite database, WAL, raw operational measurements, raw witness material, private nonce, or private filesystem path. `database_location=operator_private_companion_only` is the only location disclosure retained publicly.

Sequences 1 through 3 are immutable prior history. None may be replayed to increase coverage.

## Exact current M3-B boundary after this pin

```text
source bindings:                                  37/37
reviewed real operator attestations (C2):          3
registered runtime provenance verifiers (C2):     3
verified production runtime anchors (C2):         3
registered production source verifiers (C2):      3/37
verified positive-confidence candidates:          3/37
retained real observation:                        3/37
retained positive-confidence real observation:    3/37
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                    false
M3-C open:                                        false
M3-E authority open:                              false
cutover authorized:                               false
```

This is a real counter advance from `2/37` to `3/37`, but it is not an observation-window start and grants no later authority.

## Validation reuse boundary

PR #206 was already validated on exact head `2e44d897d73c883a473b5e3f78568c607e090412`:

```text
exact run:    30197318887
focused:      5 passed
full:         3,183 passed
artifact:     exact-head-validation-2e44d897d73c883a473b5e3f78568c607e090412
artifact SHA: 7934f53f3dce02560261e54a1f711dab1074694e07dba41fdb3c3caad6fd1f40
M2-E run:     30197318903
M2-E:         6/6 passed
merge SHA:    08ae20479ab387f8e8962e3b8cbf3cc182a66fca
```

The artifact was rechecked on GitHub as present, unexpired, and digest-matching, and all six M2-E jobs remain successful. Those #206 validations are reused as the prerequisite and must not be rerun because work moved to this PR/chat. This receipt-pin PR is a new repository head and may receive its own final-head validation according to the reuse policy; the prior #206 full/M2-E runs are not repeated.

## Next boundary

Continue M3-B with a new independently witnessed production-origin axis. The next direct-operational axis is `recovery_need`; any future retained append must be sequence 4 and prove exact continuity from the sequence-3 `fatigue_pressure` event/chain above. Existing sequences 1, 2, or 3 are never append targets again. The M3-B observation window remains closed until the later 37-axis retained positive-confidence coverage/window-entry contract is satisfied.
