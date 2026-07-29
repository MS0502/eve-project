# EVE v4 Implementation Status

Last repository rebaseline: **2026-07-29**  
Rebaseline base/prerequisite: `3acc22c8527ce76377bee4a0075ac861d2565cad` — PR #217 squash merge  
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
| M3-B | in progress; reviewed/runtime/source/candidate `5/37`; retained `5/37` after this receipt pin | #170-#212 plus verified sequence-five operator receipt |
| M3-C | authority open; M3-C-A design complete; runtime implementation not started | #215, #217 |
| M3-D | closed | requires M3-C continuity inputs |
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

Exact current M3-B boundary after this receipt pin:

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

PR #217 completed the M3-C-A design boundary:

- exact eight drives: energy, safety, affiliation, curiosity, agency,
  coherence, competence, expression;
- Affect Migration Plan rechecked as 59 `MAPPED`, 4 `PROPOSED-DROP`,
  0 unresolved;
- deterministic bounded candidate identity/scoring;
- proposal hysteresis, selection margins, and replay-carried cooldown;
- named candidate/selection lifecycle and A9 no-continuous/no-duplicate proof;
- drive-state-only counterfactual that flips deterministic selection between
  `recover_operating_margin` and `explore_information_gap`.

M3-C-A changes no production runtime, persistence writer, scheduler, action,
speech, memory/vector/model/AGP state, legacy goal authority, or M3-E authority.

## 4. Validation reuse and new-chat rule

Accepted exact-head evidence is reused when head, artifact digest, validation
scope/dependency, and ancestry still match.

Latest merged design prerequisite:

```text
PR:             #217
exact head:     7d8d861d99901998b73475c873c39a4d6f32f8ce
exact run:      30419912632
focused:        3 passed
full:           3,218 passed
artifact:       exact-head-validation-7d8d861d99901998b73475c873c39a4d6f32f8ce
artifact SHA:   97e51ff0cc6df0dfbcd3dde96f48114e64303dc4e724083b7d3c0f3ad57f474c
M2-E run:       30419912626
M2-E:           6/6 jobs passed
merge SHA:      3acc22c8527ce76377bee4a0075ac861d2565cad
```

Mandatory reuse rule:

- chat/session/operator-session changes, PR body/title/comments/reviews, and
  Draft/Ready transitions are not invalidators;
- a real tree/head change, artifact loss/corruption/digest mismatch,
  validation-scope/dependency change, or ancestry break is an invalidator;
- discovery/intermediate heads are not merge evidence;
- full suite runs once on the final registered exact head;
- do not rerun #211 witness, sequences 1-5, or accepted #215/#216/#217
  prerequisite validation merely because work moves to another chat.

## 5. Historical executable-audit compatibility

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

## 6. Private companion and reporting boundary

Raw phone companion contents, SQLite/WAL files, backups, nonces, interaction
text, CPU/wall/load/battery/context-switch observations, and private filesystem
paths must remain outside the public repository. Public records may contain
approved bounded schemas, method identifiers, counts, and cryptographic
digests needed for recomputation.

Machine-green evidence, retained observations, or elapsed parallel time cannot
automatically transfer a legacy domain, complete M3-B, open M3-E, or authorize
an affect/goal cutover.

## 7. Frozen PR register

The remaining open frozen legacy-lineage PR is:

```text
#109
```

Its residual scope remains unabsorbed and must not be closed as superseded
without a separate exact review.

## 8. Authoritative next steps

1. Pin this verified sequence-five public receipt and advance the
   repository-visible retained count to `5/37`.
2. Begin M3-C-B as a separately validated runtime-preflight/implementation
   slice derived from the accepted #217 design.
3. Keep the first M3-C runtime slice bounded: deterministic candidate
   generation/selection only; no action execution, speech generation,
   scheduler integration, legacy `GoalManagement` authority transfer, or M3-E
   cutover.
4. Continue M3-B production-origin coverage separately. `5/37` is not M3-B
   completion.
5. Reuse exact validation pins across chat changes and run new validation only
   for an actual new tree/head.

The immediate project state is: v4-native persistence authority active,
legacy authority retained per domain, M3-B retained coverage `5/37`, M3-C-A
design complete, M3-C runtime not yet implemented, and M3-E closed.
