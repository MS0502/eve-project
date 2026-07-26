# M3-B C2 Energy-Budget Retained Observation Pin

## Baseline

Repository activation head: `9b8ceceb22e2eee08f940e1673b624cbaa9bcf1a` — PR #203 squash merge.

GitHub `main` was directly compared with that SHA and was identical before this pin branch was created. The operator then ran the merged sequence-two retention command exactly once on the clean phone checkout. The command performed the real operator-private SQLite append/readback and returned the public-safe receipt now stored in `M3_B_C2_ENERGY_BUDGET_RETENTION_RECEIPT.json`.

## Receipt verification

Pinned public-safe identities:

```text
axis:                                  energy_budget
public review digest:                  a2ce3d84111224e2009bf22d1e03a8f92acab0506e42515aac185ae05ff54ab4
attestation digest:                    5413c35e912f95d90a1c0a5b0b8731a243bffc00e7b6338c1b7d9e4056e1c07f
evidence digest:                       9d814295e3b59fb42294f3ba661866aa29c512866b946e80a3f397864974af13
capture digest:                        3f30321d3d0248af20cbaa861deab070fbb66ec2e4fa2204567ef203b506c310
runtime provenance verification:       7439844ff71fb7040bbc5092059e1586bb6d3710f93e060b979e4f66b3c32956
source verification:                   eb2449c0d38d39a130f0016b0d2045315d96dfbbd5b1b067d7d02a67774c9341
prior event envelope digest:            07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493
event envelope digest:                  1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664
store transition hash:                  1189dcb8ae01370c9095ad676f2f724b2d579184d48ccef861496b04011b57a6
store after chain digest:               d4660b5cef058bad1b9d1b6b1cb2987c78ef9dbbee403c85562ab945535883e0
receipt digest:                         56401653404f9dee07804ed6a1027368baf7f118dcf6ca6f24e85a050891e3df
```

The receipt digest was independently recomputed from the canonical receipt mapping using sorted compact JSON SHA-256 and matched `56401653404f9dee07804ed6a1027368baf7f118dcf6ca6f24e85a050891e3df` exactly.

The receipt proves:

```text
prior_event_id:                         m3b:c2:retained:prediction_error_pressure:000001
event_id:                               m3b:c2:retained:energy_budget:000002
sequence:                               2
store_before_count:                     1
store_after_count:                      2
store_ordinal:                          2
readback_verified:                      true
retained_real_observation_delta:        1
retained_real_observation_count:        2/37
observation_window_started:             false
M3-B complete:                          false
M3-C open:                              false
M3-E authority open:                    false
cutover authorized:                     false
```

## Private companion boundary

The repository does **not** contain the SQLite database, WAL, raw operational measurements, raw witness material, private nonce, or private filesystem path. `database_location=operator_private_companion_only` is the only location disclosure retained publicly.

Neither retained event may be replayed to increase coverage. Sequence 1 `prediction_error_pressure` and sequence 2 `energy_budget` are immutable prior history for later appends.

## Exact current M3-B boundary after this pin

```text
source bindings:                                  37/37
reviewed real operator attestations (C2):          2
registered runtime provenance verifiers (C2):     2
verified production runtime anchors (C2):         2
registered production source verifiers (C2):      2/37
verified positive-confidence candidates:          2/37
retained real observation:                        2/37
retained positive-confidence real observation:    2/37
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                    false
M3-C open:                                        false
M3-E authority open:                              false
cutover authorized:                               false
```

This is a real counter advance from `1/37` to `2/37`, but it is not an observation-window start and grants no later authority.

## Validation reuse boundary

PR #203 was already validated on exact head `d02a0f5604f55668dead8f6e5d304f63b6e9fb18`:

```text
exact run:    30190700284
focused:      6 passed
full:         3,172 passed
artifact:     exact-head-validation-d02a0f5604f55668dead8f6e5d304f63b6e9fb18
artifact SHA: 313e18d44e1805e27b56ca5e5305f26616a27be0714b2cf452e31a1514244ef2
M2-E run:     30190700292
M2-E:         6/6 passed
merge SHA:    9b8ceceb22e2eee08f940e1673b624cbaa9bcf1a
```

Those #203 validations are reused as the prerequisite and must not be rerun because work moved to this PR/chat. This receipt-pin PR is a new repository head and may receive its own final-head validation according to the reuse policy; the prior #203 full/M2-E runs are not repeated.

## Next boundary

Continue M3-B with a new independently witnessed production-origin axis. The next retained append must be sequence 3 and must prove continuity from the exact sequence-2 `energy_budget` event/chain above. Existing sequence 1 or 2 events are never append targets again. The M3-B observation window remains closed until the later 37-axis retained positive-confidence coverage/window-entry contract is satisfied.
