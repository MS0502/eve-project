# M3-B C2 First Retained Real Observation Pin

## Baseline

Repository activation head: `e100bbd26eb84aa65ecae4ecbc10af42fd778476` — PR #199 squash merge.

GitHub `main` was directly compared with that SHA before this pin and was identical. The operator then ran the merged one-shot retention command exactly once on the clean phone checkout. The command performed the real operator-private SQLite append/readback and returned the public-safe receipt now stored in `M3_B_C2_RETAINED_REAL_OBSERVATION_RECEIPT.json`.

## Receipt verification

Pinned public-safe identities:

```text
axis:                                  prediction_error_pressure
public review digest:                  6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e
attestation digest:                    85b55eee61618ad98476f71c4dadcb9b2e4383d79aefd93a41a2c34634efecda
evidence digest:                       14549d2b9f37f2a8b00a5bc9de61dbdad8e12dbb8a4d4e08e254ef0e9848b3dc
capture digest:                        c000002f81367512c914b2992998f0dc4a827d76eec0c95d340f8749521e9f17
runtime provenance verification:       a9f7ab20d1366e6aa3d9aa977afac3162b67d722bc24544d69cc83c2756ca962
source verification:                   8b90e0de2dc9380aad7ebdbd0e246a66dcf7f68520e200d9708bca0de18f0dd4
event envelope digest:                 07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493
store transition hash:                 c1f16e8a00fa36c7903f0a585b575176830455cee83b26d262a9c04b35013c70
store after chain digest:               d51406d84dc755f72bd2ab661563c75cf19244710bf98376dbe3174ff101c8ce
receipt digest:                         ba1c5495e663cc2f7b983e1e834c96ec123733ccab2e3bee3dd6779c6e589d66
```

The receipt digest was independently recomputed from the canonical receipt mapping using the same sorted, compact JSON SHA-256 rule implemented by `core/m3_b_c2_retention_activation.py`; it matches exactly.

The receipt proves:

```text
store_before_count:                     0
store_after_count:                      1
store_ordinal:                          1
readback_verified:                      true
retained_real_observation_delta:        1
retained_real_observation_count:        1/37
```

## Private companion boundary

The repository does **not** contain the SQLite database, WAL, raw prediction/error records, private nonce, or private filesystem path. `database_location=operator_private_companion_only` is the only location disclosure retained publicly.

The append must not be repeated. The merged #199 command rejects a non-empty C2 retention stream by design; duplicate execution is not a way to increase coverage.

## Exact current M3-B boundary after this pin

```text
source bindings:                                  37/37
reviewed real operator attestations (C2):          1
verified production runtime anchors (C2):         1
registered production source verifiers (C2):      1/37
verified positive-confidence candidates:          1/37
retained real observation:                        1/37
retained positive-confidence real observation:    1/37
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                    false
M3-C open:                                        false
M3-E authority open:                              false
cutover authorized:                               false
```

This is a real counter advance from `0/37` to `1/37`, but it is not an observation-window start and grants no later authority.

## Validation reuse boundary

PR #199 was already validated on exact head `34044f768bcd5bc6e5871043e97eea5fcc5df6e8`:

```text
exact run:    30184941987
focused:      5 passed
full:         3,155 passed
artifact SHA: 6e7c0ab8b5432e5ac1ddc9e82ca0c8732377f4306effc02367965864ccbbf9d4
M2-E run:     30184941994
M2-E:         6/6 passed
merge SHA:    e100bbd26eb84aa65ecae4ecbc10af42fd778476
```

Those #199 validations are reused as the prerequisite and must not be rerun merely because this work occurs in a new chat. This receipt-pin PR is a new repository head and receives its own validation at most once on its final head according to the exact-head reuse policy.

## Next boundary

The next M3-B work is **not** to append this event again. It is to widen reviewed production-source verifier/witness coverage beyond `prediction_error_pressure`, while preserving source-family-specific provenance and the same no-fabrication boundary. Actual retained coverage can advance only from new real production-origin observations that pass their own reviewed source contracts and are durably retained.
