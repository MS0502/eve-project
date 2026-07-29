# EVE v4 Implementation Status

Last repository rebaseline: **2026-07-29**  
Rebaseline base/prerequisite: `e9e2c4598d7d0042c3c6fd78f61804b23fea163f` — PR #220 squash merge  
Active constitution: **EVE v4.2**  
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**

The complete pre-sequence-five status is preserved byte-for-byte at
`docs/audit/EVE_IMPLEMENTATION_STATUS_v4_PRE_SEQUENCE5_ARCHIVE.md`.
This document is the current operational rebaseline.

## 1. Current authority and milestone state

The explicit A12 decision (#213) and digest-pinned execution (#215) are active.
The event kernel plus SQLite store are the authoritative persistence substrate
for **v4-native subsystems only**. Every legacy domain remains authoritative in
the legacy runtime until its own separately reviewed migration gate. The
minimum seven-day legacy-parallel/rollback-preservation interval does not
automatically transfer a domain, delete legacy persistence, or open M3-E.

| Milestone | Current state | Repository basis |
|---|---|---|
| M0 | complete | historical audit set + pinned regeneration/invariance |
| M1 | complete for mechanism verification | #145-#158, explicit human acceptance #158 |
| M2-A | merged | #161 |
| M2-B | merged | #162 |
| M2-C | merged | #164 |
| M2-D | merged | #165 |
| M2-E | operational v4-native cutover active | #166-#168, #192, #195-#196, #213, #215 |
| M3-A | complete, design-only | #169 |
| M3-B | in progress; reviewed/runtime/source/candidate `5/37`; retained `5/37` | #170-#212 plus sequence-five receipt pin #218 |
| M3-C | authority open; A design, B selection kernel, and C lifecycle kernel merged; D event/reducer preflight under review | #215, #217, #219, #220, current candidate |
| M3-D | closed | separate milestone; requires later M3-C continuity inputs |
| M3-E | closed | separate reviewed affect/goal cutover; no authority open |

Current authority facts:

```text
v4-native persistence substrate authoritative: true
m3_authority_open:                              true
legacy authority:                               per-domain until separate migration gate
legacy goal-domain authority transferred:       false
M3-E authority open:                            false
```

## 2. M3-B sequence-five retained observation

The operator executed the already-reviewed `stress_load` retention command once
on exact repository head
`a9f70ef78b06744eba01a0b35c60371b10eaf672`. The public-safe receipt was
independently canonicalized and its inner receipt mapping recomputed to:

```text
receipt digest:              919a2a17c40b82e741dca01f9b7acb9f32bc83cbbdfbc6bef97fdff44fd9009f
axis:                        stress_load
sequence:                    5
event:                       m3b:c2:retained:stress_load:000005
prior event:                 m3b:c2:retained:recovery_need:000004
store count:                 4 -> 5
store ordinal:               5
retained delta:              1
readback verified:           true
event envelope digest:       c53d80c3bb8683671ea6936a6ebb7ea2783941902b90fb116708bac96756aca8
store-before chain digest:    16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f
store-after chain digest:     0b7e8908f7ef6d583a6839e1600c1ae2d780263d2bab8e22ffef2b7e902b193b
store transition hash:       0d828805f654bf807e85877322414abdabc53ed77ec4947bb4acfa506d9d2672
```

The receipt's `m3_c_open=false`, `cutover_authorized=false`, and
`authority=shadow_only` are immutable execution-time fields of the M3-B
retention event. They do not roll back the separately merged #215 repository
authority. Current M3-C authority remains open, while M3-E remains closed.

```text
source bindings:                                  37/37
reviewed real operator attestations (C2):          5
registered runtime provenance verifiers (C2):     5
verified production runtime anchors (C2):         5
registered production source verifiers (C2):      5/37
verified positive-confidence candidates:          5/37
retained real observations:                        5/37
retained positive-confidence real observations:    5/37
stress_load real witness executed:                 true — exactly once
stress_load retained:                              true — sequence 5
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                    false
```

Sequences 1 through 5 are immutable. The #211 witness and all five retention
appends must not be replayed. Retained `5/37` does not complete M3-B and does not
start its separate observation window.

## 3. M3-C-A design boundary

M3-A drive-dynamics design status remains **documentation-only** over the
**63-axis Affect Migration Plan**, including **48 bidirectional named
transitions**, with **no runtime integration** and the historical rule that
**integration eligibility only after persistence cutover**. The cutover
eligibility condition is now satisfied by #215; that does not retroactively
turn #169 into runtime code.

PR #217 completed the M3-C-A design boundary with exact eight-drive integration,
59 `MAPPED` / 4 `PROPOSED-DROP` affect rulings, deterministic candidate scoring,
proposal/selection hysteresis, replay cooldown, exact lifecycle edges, A9
no-continuous/no-duplicate proof, and a drive-only counterfactual selection
flip. It changed no production runtime or authority.

## 4. M3-C-B merged pure selection kernel

PR #219 merged the #217 arithmetic as an isolated standard-library kernel.

```text
exact head:     39d118decc6f9353623da8c71724d7302fccc7ef
exact run:      30422131822
focused:        9 passed
full:           3,227 passed
artifact SHA:   49552c718cc501758276317ec139caf6d1fecedf393037f90b95a4ba7bf6a689
M2-E run:       30422131843
M2-E:           6/6 jobs passed
merge SHA:      3a09a6ddd2f3b64d5483fd8564be0e645043538f
```

It remains disconnected from production, persistence, actions, scheduling,
speech, legacy goal authority, and M3-E.

## 5. M3-C-C merged pure lifecycle kernel

PR #220 merged an isolated lifecycle transition-candidate kernel covering the
exact fourteen #217 edges. It derives at most one edge per logical step and
returns either a deterministic transition candidate or no-transition receipt.

```text
exact head:     b11951940467de43e30f00e86f3c3a409ec3d51f
exact run:      30423022041
focused:        19 passed
full:           3,246 passed
artifact SHA:   0c7bc2e484bd877b27c1900ae895d38d5ebc48f167f00cc3793225144f482a4d
M2-E run:       30423021990
M2-E:           6/6 jobs passed
merge SHA:      e9e2c4598d7d0042c3c6fd78f61804b23fea163f
```

The merged kernel performs no event append, persistence write, production
integration, action, scheduling, speech, legacy authority transfer, or M3-E
opening.

## 6. M3-C-D event-envelope/reducer preflight candidate

The current candidate converts an immutable M3-C-C transition into a canonical
`candidate_only` event envelope and replays ordered candidates into an
immutable in-memory reducer snapshot.

It proves deterministic event/payload/envelope/snapshot/receipt identities,
duplicate refusal, before-state and prior-transition continuity, per-candidate
logical-step monotonicity, ordered replay, and checkpoint/resume equality.

```text
authoritative EventKernel append:       false
SQLite/file persistence write:           false
production lifecycle integration:        false
action/scheduler/speech authority:       false
legacy goal-domain authority transfer:   false
M3-E authority open:                     false
```

This M3-C-D slice is not the separate M3-D milestone. M3-D remains closed.

## 7. Validation reuse and new-chat rule

Accepted exact-head evidence is reused when head, artifact digest, validation
scope/dependency, and ancestry still match.

Latest merged implementation prerequisite:

```text
PR:             #220
exact head:     b11951940467de43e30f00e86f3c3a409ec3d51f
exact run:      30423022041
focused:        19 passed
full:           3,246 passed
artifact:       exact-head-validation-b11951940467de43e30f00e86f3c3a409ec3d51f
artifact SHA:   0c7bc2e484bd877b27c1900ae895d38d5ebc48f167f00cc3793225144f482a4d
M2-E run:       30423021990
M2-E:           6/6 jobs passed
merge SHA:      e9e2c4598d7d0042c3c6fd78f61804b23fea163f
```

Mandatory reuse rule:

- chat/session/operator-session changes, PR body/title/comments/reviews, and
  Draft/Ready transitions are not invalidators;
- a real tree/head change, artifact loss/corruption/digest mismatch,
  validation-scope/dependency change, or ancestry break is an invalidator;
- discovery/intermediate heads are not merge evidence;
- full suite runs once on the final registered exact head;
- do not rerun #211 witness, sequences 1-5, or accepted #215-#220
  prerequisite validation merely because work moves to another chat.

## 8. Historical executable-audit compatibility

M1 status: **closed for mechanism verification**.

The canonical M1 human-acceptance record SHA-256 remains
`aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522`.
At that historical decision point, the status contract stated
**M2-A remains blocked until v4.2 approval**. That sentence is historical
acceptance evidence only: v4.2 was later approved and M2-A merged in #161.

The historical record named **open REWRITE PRs #109, #86, #84, and #82**.
Repository disposition is preserved: #82, #84, and #86 are closed/unmerged and
superseded; #109 remains open/unmerged. The **absorbed PRs #11, #7, and #4 are
closed**. These historical statements grant no current runtime or authority.

Executable-audit literal compatibility:

```text
M3-A drive-dynamics design status
documentation-only
63-axis Affect Migration Plan
48 bidirectional named transitions
no runtime integration
integration eligibility only after persistence cutover
absorbed PRs #11, #7, and #4 are closed
```

## 9. Private companion and reporting boundary

Raw phone companion contents, SQLite/WAL files, backups, nonces, interaction
text, CPU/wall/load/battery/context-switch observations, and private filesystem
paths must remain outside the public repository. Public records may contain
approved bounded schemas, method identifiers, counts, and cryptographic
digests needed for recomputation.

Machine-green evidence, retained observations, or elapsed parallel time cannot
automatically transfer a legacy domain, complete M3-B, open M3-E, or authorize
an affect/goal cutover.

## 10. Frozen PR register

The remaining open frozen legacy-lineage PR is:

```text
#109
```

Its residual scope remains unabsorbed and must not be closed as superseded
without a separate exact review.

## 11. Authoritative next steps

1. Review M3-C-D as a pure event-envelope/replay-reducer preflight only.
2. Keep EventKernel/SQLite append and production integration closed.
3. After acceptance, design a separately gated authoritative-substrate binding
   and rollback rehearsal before any live lifecycle writer exists.
4. Continue M3-B production-origin coverage separately. `5/37` is not M3-B
   completion.
5. Reuse exact validation pins across chat changes and run new validation only
   for an actual new tree/head.

The immediate project state is: v4-native persistence authority active,
legacy authority retained per domain, M3-B retained coverage `5/37`, M3-C-A/B/C
merged, M3-C-D event/reducer preflight under review, separate M3-D closed, and
M3-E closed.
