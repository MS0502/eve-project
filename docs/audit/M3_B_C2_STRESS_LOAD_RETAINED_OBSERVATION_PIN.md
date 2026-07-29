# M3-B C2 — `stress_load` sequence-five retained-observation pin

## Execution boundary

The operator executed the already-staged retention command exactly once on
repository head:

```text
a9f70ef78b06744eba01a0b35c60371b10eaf672
```

No witness was replayed. No private SQLite/WAL, nonce, raw process timing,
kernel-load measurements, interaction text, or private filesystem path is
committed.

## Independent receipt verification

The supplied public-safe JSON has schema
`eve.m3-b.c2-stress-load-retention-public-review.v1`. Canonical sorted compact
JSON of its nested `receipt` mapping recomputes exactly to:

```text
919a2a17c40b82e741dca01f9b7acb9f32bc83cbbdfbc6bef97fdff44fd9009f
```

This equals the supplied `receipt_digest`.

## Exact retained transition

```text
axis:                        stress_load
sequence:                    5
prior event:                 m3b:c2:retained:recovery_need:000004
new event:                   m3b:c2:retained:stress_load:000005
store count:                 4 -> 5
store ordinal:               5
retained delta:              1
readback verified:           true
prior event envelope:        7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691
new event envelope:          c53d80c3bb8683671ea6936a6ebb7ea2783941902b90fb116708bac96756aca8
store-before chain:          16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f
store-after chain:           0b7e8908f7ef6d583a6839e1600c1ae2d780263d2bab8e22ffef2b7e902b193b
store transition hash:       0d828805f654bf807e85877322414abdabc53ed77ec4947bb4acfa506d9d2672
```

The exact reviewed source bindings also match #212:

```text
attestation:                 7191e3493c582a191db3dcd488b2452dd3b0f29774b8a3a3ffeaff3b53c525fa
evidence:                    5bceb97155a5614de72b2b359b861db5c57eb6e892c259b56981d6003fc14680
public review:               1ec63bb54cfed398b0e5b93af25667474c3255d5ca50a47602974d363cf5e03a
capture:                     8fa45f62b240645dd896d08f1ed411e52d8e42b13c9eb182242d1215a6a97b30
runtime provenance verify:   03466c9d5ee70ee1b5b0e8a83373762358bb4306a61a2dcc006c866520b1cae5
source verify:               6ff67fb8ee1933e77a20274f7275d0e42a199ffeb5d55e93f05aad96d7652726
```

## Authority interpretation

The receipt records `authority=shadow_only`, `cutover_authorized=false`,
`m3_c_open=false`, `m3_e_authority_open=false`, and
`observation_window_started=false`. These are immutable fields of the M3-B
retention event executed on the #215 merge head. They do not revoke the
separate #215 cutover decision, under which v4-native persistence authority and
`m3_authority_open` are active. Legacy authority remains per-domain and M3-E
remains closed.

## Counter result

On this pin's merge:

```text
reviewed/runtime/source/candidate coverage: 5/37
retained real-observation coverage:          5/37
M3-B observation window started:             false
M3-B complete:                               false
```

Sequences 1-5 are immutable and must not be appended again.
