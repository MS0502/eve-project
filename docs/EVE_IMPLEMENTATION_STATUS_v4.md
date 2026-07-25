# EVE v4 Implementation Status

Last repository rebaseline: **2026-07-26**  
Rebaseline base/prerequisite: `5664fc3bc22054c2d39142b3125416aea6089c63` — PR #196 squash merge  
Active constitution: **EVE v4.2**  
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**

## 1. Current authority and milestone state

The pre-kernel legacy runtime remains authoritative. The merged event-store, migration, recovery, affect, provenance, and capture machinery does not by itself transfer runtime, persistence, affect, goal, expression, or cutover authority.

| Milestone | Current state | Repository basis |
|---|---|---|
| M0 | complete | historical audit set + pinned regeneration/invariance |
| M1 | complete for mechanism verification | #145-#158 lineage, explicit M1 human acceptance in #158 |
| M2-A | merged | #161 append-only SQLite shadow persistence |
| M2-B | merged | #162 read-capability extraction and exact technical decisions |
| M2-C | merged | #164 bounded migration/dual-read comparison |
| M2-D | merged | #165 bounded recovery/rollback rehearsal |
| M2-E | **A11 repair + wrapper-entrypoint hotfix merged; reviewed resume already succeeded; phone supervisor restart pending; cutover not authorized** | #166 candidate, #167 human acceptance record, #168 chaos + phone habitat driver, #192 guarded recovery/reviewed-resume, #195 A11 content-addressed repair, #196 script bootstrap hotfix |
| M3-A | **complete** | #169 drive-dynamics design |
| M3-B | **in progress** | #170-#190 merged shadow/read-only chain; #194 is the C1 trust-root change owner and contains no real attestation |
| M3-C | closed | requires stable/completed M3-B |
| M3-D | closed | requires M3-C continuity inputs |
| M3-E | closed | separate reviewed affect/goal cutover; no authority open |

`M2-E` acceptance is not a production cutover authorization. The reviewed resume was executed once after #195 and returned `resume_exit=0`; it must **not** be repeated. PR #196 fixed the subsequent supervisor script-entrypoint failure and merged as `5664fc3bc22054c2d39142b3125416aea6089c63`. The remaining M2-E phone action is checkout update plus supervisor restart only. The M3-B observation window is a separate gate and has **not started**.

### Executable-audit compatibility notes

Historical acceptance wording that is still asserted by executable audit tests remains part of the status contract; it is retained here without reverting the current milestone state above.

M1 status: **closed for mechanism verification**. This is the historical #158 human-acceptance boundary and does not itself authorize persistence cutover or later M-series authority.

The canonical M1 human-acceptance record SHA-256 remains `aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522`. At that decision point the status contract stated **M2-A remains blocked until v4.2 approval**; that sentence is retained as historical acceptance evidence only, because v4.2 was later approved and M2-A subsequently merged in #161. The absorbed PRs #11, #7, and #4 are closed; this is likewise retained as the historical M1 disposition record.

M3-A drive-dynamics design status: the merged #169 artifact remains a **documentation-only** design boundary over the **63-axis Affect Migration Plan**, including **48 bidirectional named transitions**, with **no runtime integration** in that design artifact and **integration eligibility only after persistence cutover**. M3-A is now complete as a design milestone, while its historical no-runtime-integration boundary remains authoritative for what #169 itself proved.

## 2. M3-B exact current boundary

Merged M3-B work now includes:

- bounded read-only affect projection (#170);
- source-ownership preflight (#171);
- immutable legacy 26-axis capture (#172);
- registry 37-axis current-value owner (#173);
- corrected combined 63-axis packet (#176);
- positive-confidence evidence contract (#177);
- exact 37-axis source manifest (#178);
- all seven source-binding groups, completing 37/37 structural source binding (#179-#185);
- retained-real-observation capture preflight (#186);
- production capture adapter + immutable retained-observation sink (#187);
- executable verifier issuance anti-forgery boundary (#188);
- real `prediction_error_pressure` runtime source bridge (#189);
- production-runtime provenance preflight with fixture-classification binding (#190).

The C1 lineage (#194) supplies a one-operator attestation trust-root contract and operator-local digest recomputation surface. It deliberately leaves the reviewed-attestation registry empty because no real phone launch attestation has been produced and reviewed yet. Whether #194 is Draft, Ready, or merged must be read from live PR metadata; that metadata-only state does not change the counters below.

The structural/real-observation boundary remains:

```text
source bindings:                              37/37
production capture adapter:                   present
immutable retention sink:                     present
verifier issuance anti-forgery boundary:      present
prediction_error runtime source bridge:       present
production runtime provenance preflight:      present
operator attestation trust-root:              supplied by #194 lineage
reviewed real operator attestations:           0
registered runtime provenance verifiers:      0
verified production runtime anchors:          0
registered production source verifiers:       0/37
retained real observation:                    0/37
positive-confidence real observation:         0/37
M3-B observation window eligible:             false
M3-B observation window started:              false
M3-B complete:                                false
M3-C open:                                    false
M3-E authority open:                          false
cutover authorized:                           false
```

No audit fixture, detached synthetic evidence, test verifier, self-authored runtime metadata, `fixture_only=False`, PID, argv/environment flag, caller identity, self-hashed launch metadata, or unreviewed public attestation digest may be reclassified as production evidence. Production provenance requires an independently reviewable trust root and an exact reviewed registration.

## 3. M2-E habitat incidents, A1 visibility, A11 Fix 2, and wrapper hotfix

The phone habitat incident on 2026-07-24 exposed a failure-visibility defect rather than a demonstrated store-integrity failure. The private companion evidence remains outside the public repository; only reviewed summaries/digests may be committed.

Repository/code inspection plus the operator evidence establish the following bounded finding:

- the durable store survived the incident and the operator-side integrity checks reported healthy state;
- a freeze occurred immediately after the last known normal sequence during the watchdog restore path;
- the prior habitat runtime could collapse restore-path failures into broad `unrecoverable_corruption` handling without preserving exception type/message/traceback identity;
- the prior `supervisor.sh` did not continuously redirect runtime stdout/stderr into `supervisor.log`, leaving a second visibility gap;
- the swallowed 2026-07-24 incident exception's exact type remains **unknown** and is not retroactively invented.

The merged PR #192 A1 implementation supplied the visibility/recovery repair without authorizing resume by itself:

1. the original #168 habitat CLI remains byte-for-byte intact for audit compatibility, while the supervisor executes a separate independently testable guarded runtime;
2. every caught exception that can cause a freeze records exception type, message, traceback digest, attempt, and an evidence digest in the append-only private raw stream before the freeze transition;
3. restore/integrity I/O receives three bounded backoffs (`1s`, `2s`, `4s`), exhausted healthy-store I/O is classified `io_failure`, and `unrecoverable_corruption` is reserved for a failed integrity report;
4. reviewed resume requires integrity success, exact event-count validity, recomputed restore digest, and—only for the existing one-row pending-commit case—an exact deterministic next-event match before reconciliation;
5. `freeze_reviewed_resume` references the immutable pre-resume freeze-review digest and does not erase earlier freeze/raw evidence;
6. supervisor status stderr plus worker stdout/stderr are continuously appended to private `supervisor.log`;
7. focused tests cover injected OSError retry/evidence, corruption-only classification, corruption resume refusal, same-count resume, one-pending-event reconciliation, and supervisor logging.

A1 then did its intended job on the first recurrence. The operator-private record on 2026-07-25 retained:

```text
InvalidEventEnvelope: event_material exceeds canonical size limit
context: append_snapshot_backup
sequence boundary: 280
```

The deterministic timing boundary also matches the prior incident: sequence 279 was observed at 2026-07-24 11:29 and the fixed stimulus interval is 300 seconds, placing the next attempt at 11:34, the original freeze time. This supports a same-boundary recurrence; it does **not** retroactively supply the exception type that the pre-A1 path failed to record.

Repository inspection identifies an A11 representation defect at that boundary. Habitat events contain the cumulative full shadow state in `before` and `after`; the frozen v1 SQLite persistence path then embeds the canonical `payload_json` string inside `event_material`, adding escaping overhead until the persistence representation crosses the unchanged 65,536-byte canonical limit at sequence 280. Snapshot rows also inline the growing full state and therefore have the same unbounded-growth design problem.

Habitat Fix 2 (#195) applies A11 rather than weakening the threshold: large persisted material moves to append-only SHA-256 content-addressed storage, while canonical event/snapshot persistence records retain only the digest and a versioned structural manifest. Logical `EventEnvelope` validation and `MAX_CANONICAL_JSON_BYTES` remain unchanged. Exact legacy-v1 stores retain their original single migration record and old inline snapshots remain readable through an explicit format branch; no old snapshot is rewritten. Replay computes the same canonical full-state digest as snapshot content. The existing one-pending-row path remains responsible for reconciling a durable sequence 280 against `window_state.json` at 279.

The reviewed resume command was executed once after #195 merged and the operator reported `resume_exit=0`. That successful resume is complete and must not be repeated merely because the supervisor later failed to start.

PR #196 closed the separate script-entrypoint defect: `m2_e_window_runtime_guarded_a11.py` now installs the repository-root bootstrap before `core...` imports, and a supervisor-equivalent subprocess regression executes the wrapper as a real script. Its accepted exact head was `4944c01df3b0978ae73ea3060abd39bee14e41c1`; the squash merge is `5664fc3bc22054c2d39142b3125416aea6089c63`.

## 4. Validation reuse and new-chat rule

`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json` plus immutable merged-PR validation records are the durable source of truth for validation reuse.

Mandatory rule:

- a new chat/operator session **must inspect the ledger and merged PR reuse records first**;
- reuse an exact-head validation when the exact head, artifact SHA-256, validation scope/dependency, and merge ancestry still match;
- chat/session changes, PR body/title edits, comments/reviews, and Draft/Ready transitions are **not** invalidators;
- a code-head change, artifact loss/corruption/digest mismatch, validation-scope/dependency change, or ancestry break **is** an invalidator;
- discovery/intermediate registration heads are not merge evidence;
- full-suite runs once on the final registered exact head after the forward gate passes.

PR #196 is the latest merged prerequisite pin:

```text
exact head:   4944c01df3b0978ae73ea3060abd39bee14e41c1
exact run:    30179233468
focused:      4 passed
full:         3,138 passed
artifact:     exact-head-validation-4944c01df3b0978ae73ea3060abd39bee14e41c1
artifact SHA: 4f803b16343871b6e20517676cd2a0a4ebce7231444fcd4158fb0926320181b8
M2-E run:     30179233476
M2-E:         6/6 jobs passed
merge SHA:    5664fc3bc22054c2d39142b3125416aea6089c63
```

PR #190 through #193 and #195 remain separately pinned by their accepted records/ledger entries. PR #196's merged metadata record explicitly states that a new chat or operator session does not justify rerunning its green evidence. C1 is a genuine new code head and therefore requires its own final validation; it does not invalidate or rerun #196 as a prerequisite.

## 5. Governance rules added by the #191 rebaseline

### 5.1 Same-PR STATUS update

Every merge-intended implementation/governance PR must update this STATUS document in the same PR when it changes milestone state, blockers, authority, operational state, or the authoritative next step. A later cleanup PR must not be relied on to repair a knowingly stale completion claim.

### 5.2 Reporting integrity

A PR number, merge state, exact head, workflow result, artifact digest, or completion claim may be written in completed form only after direct repository/workflow verification. Work that has not been pushed/created/verified must be described as planned, proposed, or pending.

On 2026-07-25, before the #191 governance rebaseline was opened, a status report incorrectly stated that PRs #190-#192 had been merged. Direct repository audit showed #190 was still Draft/Open and that no PR objects existed for #191 or #192 at that time. #190 was then independently reviewed, corrected, exact-head validated, and squash-merged as `5af3fc8f2041e54a33384c4a8d60bebccb5a6eb2`; #191 was later actually created, validated, and squash-merged as `9b2545795b681dd0c53a9d51820b6baa70df9482`; #192 was later actually created, validated, and squash-merged as `77443032eb3fe70eac8c8ca8a18909574de81063`. Those later repository events do not retroactively validate the earlier false completion report.

### 5.3 Private companion boundary

Raw phone companion contents, SQLite/WAL files, backups, private nonce material, and other non-public habitat evidence must not be copied into the public repository. Public records may contain only approved schemas, bounded summaries, and cryptographic digests/references needed for authorized recomputation.

### 5.4 No automatic authority promotion

Machine-green evidence, PR merge, operator attestation machinery, source registration, retained observations, or an observation-window seal cannot automatically open M3-C/M3-E or authorize cutover. Any authority transition remains a separate explicit reviewed decision.

## 6. PR registry — verified repository history through #196

This table is regenerated from repository PR state, not prior chat reports.

| PR | Repository state | Registry meaning |
|---:|---|---|
| #145 | merged | v4.1 forward regression gate |
| #146 | merged | M1-A event kernel |
| #147 | merged | M1-B registered shadow observer |
| #148 | merged | M1-C bounded replay equivalence |
| #149 | merged | M1-D lifecycle bridge contracts |
| #150 | merged | M1-E machine acceptance window |
| #151 | merged | M1 evidence-gap documentation |
| #152 | merged | controlled M1 observation evidence |
| #153 | merged | expanded controlled M1 evidence |
| #154 | closed, unmerged | temporary M1 fidelity execution PR |
| #155 | closed, unmerged | temporary M1 fidelity retry PR |
| #156 | closed, unmerged | temporary corrected bootstrap PR |
| #157 | closed, unmerged | temporary final fidelity correction PR |
| #158 | merged | explicit M1 human acceptance |
| #159 | closed, unmerged | temporary PR #158 manifest execution PR |
| #160 | merged | EVE v4.2 constitutional amendment A9-A12 |
| #161 | merged | M2-A append-only SQLite shadow persistence |
| #162 | merged | M2-B read-capability manifest/decisions |
| #163 | closed, unmerged | temporary M2-B diagnostic probe |
| #164 | merged | M2-C migration/dual-read comparison |
| #165 | merged | M2-D recovery/rollback rehearsal |
| #166 | merged | M2-E bounded cutover candidate contract |
| #167 | merged | explicit M2-E human acceptance record |
| #168 | merged | M2-E chaos + phone habitat window driver |
| #169 | merged | M3-A drive-dynamics design |
| #170 | merged | M3-B read-only affect projection |
| #171 | merged | M3-B observation source ownership preflight |
| #172 | merged | M3-B legacy 26-axis immutable capture |
| #173 | merged | M3-B registry 37-axis current-value owner |
| #174 | closed, unmerged | superseded combined 63-axis preflight |
| #175 | closed, unmerged | stale-head superseded combined preflight |
| #176 | merged | corrected combined 63-axis packet |
| #177 | merged | 37-axis positive-confidence evidence contract |
| #178 | merged | 37-axis source manifest |
| #179 | merged | operational 4-axis source binding |
| #180 | merged | survival 2-axis appraised binding |
| #181 | merged | risk-defense 6-axis quarantined binding |
| #182 | merged | social-relationship 7-axis quarantined binding |
| #183 | merged | learning-exploration 6-axis validated binding |
| #184 | merged | self-identity 6-axis binding |
| #185 | merged | expression-action 6-axis AGP-bounded binding; structural 37/37 reached |
| #186 | merged | retained-real-observation capture preflight |
| #187 | merged | production capture adapter + immutable sink |
| #188 | merged | production verifier issuance anti-forgery boundary |
| #189 | merged | prediction-error real runtime source bridge preflight |
| #190 | merged | production-runtime provenance preflight; fixture relabeling defect corrected before merge |
| #191 | merged | STATUS/reporting-integrity rebaseline + exact-head reuse governance |
| #192 | merged | A1 guarded M2-E habitat recovery + reviewed resume |
| #193 | merged | B2 STATUS/reuse rebaseline and frozen-PR disposition |
| #194 | live PR state authoritative | C1 operator-attestation trust-root lineage; reviewed-attestation registry intentionally empty |
| #195 | merged | A11 content-addressed habitat persistence repair; exact validation pinned in merged PR metadata |
| #196 | merged | A11 wrapper script-bootstrap hotfix; exact validation pinned in merged PR metadata |

## 7. Frozen PR register

Open frozen legacy-lineage PRs now remaining:

```text
#109
```

The historical M1 acceptance record named **open REWRITE PRs #109, #86, #84, and #82**. B2 preserves that historical statement but records the repository-verified current disposition instead of treating it as live state:

- #82 is closed/unmerged and fully superseded by merged #83 (`8b46050151860d462a09137cd3236dc10373845d`). Both lineages use the same four-file Round1081-1100 multimodal-event candidate scope; the supersession evidence is preserved in #82's discussion.
- #84 is closed/unmerged and fully superseded by merged #85 (`9f0e1112c14883c0ee7b41d2770af713ab91fce7`). Both lineages use the same four-file Round1101-1120 cross-modal binding preflight scope; the supersession evidence is preserved in #84's discussion.
- #86 is closed/unmerged and fully superseded by merged #88 (`7e5bcbc8f1e1c7849b89054f11a7df62d661acf1`). Both lineages use the same four-file Round1121-1140 memory-replay observation scope; the supersession evidence is preserved in #86's discussion.
- #109 remains open/unmerged. Its four-file Round1461-1480 virtual-world situation conclusion-candidate scope is absent from the current `main`; therefore an unabsorbed residual exists and B2 does **not** close or label it superseded merely to make the frozen register zero.

These dispositions grant no runtime, persistence, M3, or cutover authority.

## 8. Current next steps

Order is constrained by evidence and authority boundaries:

1. **Phone M2-E continuation:** update the phone checkout to merged `main` and restart only `scripts/habitat/supervisor.sh`; do **not** run `resume --reviewed` again. Continue the existing real habitat window from the already-resumed state.
2. **C1 trust root (#194):** this lineage supplies the one-operator contract; its Draft/Ready/merged state must be read live. No C2 production evidence may be accepted while #194 is unmerged, and merging #194 does not itself create a real attestation or production verifier.
3. **Real phone launch attestation:** only after #194 is merged, the operator may create and locally verify one real launch attestation using private companion nonce material. Raw nonce material stays private; only approved digest/public fields may enter review.
4. **C2 — First capability-forcing real observation:** only after an exact real attestation is reviewed/pinned, bind one real attested runtime/source verifier and retain the first positive-confidence observation honestly. Do not inflate one observation into 37/37 production coverage.
5. Continue M3-B real source-batch observations and its separate observation window. Only a completed/stable M3-B may open M3-C.

The immediate project state remains **M3-B in progress**, with reviewed real operator attestations at zero, real production verifier/observation counters at zero, and no cutover authority.
