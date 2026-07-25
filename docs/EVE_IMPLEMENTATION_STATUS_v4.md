# EVE v4 Implementation Status

Last repository rebaseline: **2026-07-25**  
Rebaseline base: `5af3fc8f2041e54a33384c4a8d60bebccb5a6eb2` — PR #190 squash merge  
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
| M2-E | **operational window work in progress; cutover not authorized** | #166 candidate, #167 human acceptance record, #168 chaos + phone habitat driver |
| M3-A | **complete** | #169 drive-dynamics design |
| M3-B | **in progress** | #170-#190 shadow/read-only affect and real-observation preflight chain |
| M3-C | closed | requires stable/completed M3-B |
| M3-D | closed | requires M3-C continuity inputs |
| M3-E | closed | separate reviewed affect/goal cutover; no authority open |

`M2-E` acceptance is not a production cutover authorization. The phone habitat observation window is currently frozen pending the A7 habitat-driver repair described below. The M3-B observation window is a separate gate and has **not started**.

### Executable-audit compatibility notes

Historical acceptance wording that is still asserted by executable audit tests remains part of the status contract; it is retained here without reverting the current milestone state above.

M1 status: **closed for mechanism verification**. This is the historical #158 human-acceptance boundary and does not itself authorize persistence cutover or later M-series authority.

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

The merged structural state is:

```text
source bindings:                              37/37
production capture adapter:                   present
immutable retention sink:                     present
verifier issuance anti-forgery boundary:      present
prediction_error runtime source bridge:       present
production runtime provenance preflight:      present

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

No audit fixture, detached synthetic evidence, test verifier, self-authored runtime metadata, `fixture_only=False`, PID, argv/environment flag, caller identity, or self-hashed launch metadata may be reclassified as production evidence. Production provenance requires an independently reviewable trust root.

## 3. M2-E habitat incident and A7 repair gate

The phone habitat incident on 2026-07-24 exposed a failure-visibility defect rather than a demonstrated store-integrity failure. The private companion evidence remains outside the public repository; only reviewed summaries/digests may be committed.

Repository/code inspection plus the operator evidence establish the following bounded finding:

- the durable store survived the incident and the operator-side integrity checks reported healthy state;
- a freeze occurred immediately after the last known normal sequence during the watchdog restore path;
- the current runtime can collapse restore-path exceptions into broad `unrecoverable_corruption` handling without preserving exception type/message/traceback identity;
- `supervisor.sh` does not continuously redirect runtime stdout/stderr into `supervisor.log`, leaving a second visibility gap;
- the swallowed exception's exact type is therefore **not known** and must not be invented retrospectively.

A1 is the critical repair gate before operator resume. Required scope:

1. record exception type, message, and traceback digest in append-only raw evidence before freeze;
2. classify `unrecoverable_corruption` only after integrity failure; use bounded three-attempt backoff for I/O failure and classify exhausted I/O as `io_failure`;
3. add guarded reviewed resume requiring integrity success, exact event-count/reconciliation validity, and recomputed restore-digest success while retaining the original freeze evidence immutably;
4. make supervisor stdout/stderr actually reach `supervisor.log`;
5. add chaos coverage for injected OSError and corruption/resume denial.

This repair grants no cutover, recovery authority, M3 authority, or production-default change.

## 4. Validation reuse and new-chat rule

`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json` is the durable source of truth for validation reuse.

Mandatory rule:

- a new chat/operator session **must inspect the ledger first**;
- reuse an exact-head validation when the exact head, artifact SHA-256, validation scope/dependency, and merge ancestry still match;
- chat/session changes, PR body/title edits, comments/reviews, and Draft/Ready transitions are **not** invalidators;
- a code-head change, artifact loss/corruption/digest mismatch, validation-scope/dependency change, or ancestry break **is** an invalidator;
- discovery/intermediate registration heads are not merge evidence;
- full-suite runs once on the final registered exact head after the forward gate passes.

PR #190 is now pinned as a merged prerequisite:

```text
exact head:   2b2689e9fc49c8f10ea8c367b8e74e3860523ca0
exact run:    30148099229
focused:      8 passed
full:         3,128 passed
artifact SHA: 6abbdc0f6be2c887c75cddc30ca232fbfd15d02707580c68ca2098f0b0e2a2a4
M2-E run:     30148099254
M2-E:         6/6 jobs passed
merge SHA:    5af3fc8f2041e54a33384c4a8d60bebccb5a6eb2
```

## 5. Governance rules added by this rebaseline

### 5.1 Same-PR STATUS update

Every merge-intended implementation/governance PR must update this STATUS document in the same PR when it changes milestone state, blockers, authority, operational state, or the authoritative next step. A later cleanup PR must not be relied on to repair a knowingly stale completion claim.

### 5.2 Reporting integrity

A PR number, merge state, exact head, workflow result, artifact digest, or completion claim may be written in completed form only after direct repository/workflow verification. Work that has not been pushed/created/verified must be described as planned, proposed, or pending.

On 2026-07-25, before this governance rebaseline was opened, a status report incorrectly stated that PRs #190-#192 had been merged. Direct repository audit showed #190 was still Draft/Open and that no PR objects existed for #191 or #192 at that time. #190 was then independently reviewed, corrected, exact-head validated, and squash-merged as `5af3fc8f2041e54a33384c4a8d60bebccb5a6eb2`. The earlier #191/#192 completion claims remain invalid historical claims. Any later creation of those PR numbers does not retroactively validate the earlier report.

### 5.3 Private companion boundary

Raw phone companion contents, SQLite/WAL files, backups, private nonce material, and other non-public habitat evidence must not be copied into the public repository. Public records may contain only approved schemas, bounded summaries, and cryptographic digests/references needed for authorized recomputation.

### 5.4 No automatic authority promotion

Machine-green evidence, PR merge, operator attestation machinery, source registration, retained observations, or an observation-window seal cannot automatically open M3-C/M3-E or authorize cutover. Any authority transition remains a separate explicit reviewed decision.

## 6. PR registry — #145 through #190

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

## 7. Frozen PR register

Open frozen legacy-lineage PRs remain:

```text
#109  #86  #84  #82
```

The historical M1 acceptance record names these as **open REWRITE PRs #109, #86, #84, and #82**. That literal remains an executable-audit compatibility statement until B2 performs repository-verified supersession disposition.

They are not merge-authorized by this status rebaseline. The next governance step B2 compares each intent against the merged M2-A/M2-B and later contracts. Fully absorbed items may be closed with supersession/evidence preservation; any unabsorbed requirement must be reported instead of silently closed.

## 8. Current next steps

Order is constrained by evidence and authority boundaries:

1. **A1 — Habitat Driver Fix:** repair failure visibility, I/O classification/backoff, guarded reviewed resume, supervisor logging, and chaos coverage; then operator may execute the exact reviewed resume command from that PR.
2. **B2 — Frozen-PR disposition:** after this STATUS/governance rebaseline, compare #109/#86/#84/#82 against merged contracts and close only fully superseded work.
3. **C1 — Operator Attestation Trust Root:** create a one-operator trust root in which private companion nonce material remains private and repository/runtime evidence carries only the independently checkable digest binding. A runtime may not self-sign its own production provenance.
4. **C2 — First capability-forcing real observation:** only after C1, bind a real attested source verifier and land the first retained, positive-confidence real observation honestly. Do not inflate 1/37 into 37/37 production coverage.
5. Continue M3-B source-batch real observations and its separate observation window. Only a completed/stable M3-B may open M3-C.

The immediate project state remains **M3-B in progress**, with real production verifier/observation counters at zero and no cutover authority.
