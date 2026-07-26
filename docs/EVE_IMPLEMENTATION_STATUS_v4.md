# EVE v4 Implementation Status

Last repository rebaseline: **2026-07-26**  
Rebaseline base/prerequisite: `f178b9e0b0fbaa341776fbef66e6c5c87fe5a157` — PR #201 squash merge  
Active constitution: **EVE v4.2**  
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**

## 1. Current authority and milestone state

The pre-kernel legacy runtime remains authoritative. The merged event-store, migration, recovery, affect, provenance, capture, reviewed-verifier, and retention machinery does not by itself transfer runtime, persistence, affect, goal, expression, or cutover authority.

| Milestone | Current state | Repository basis |
|---|---|---|
| M0 | complete | historical audit set + pinned regeneration/invariance |
| M1 | complete for mechanism verification | #145-#158 lineage, explicit M1 human acceptance in #158 |
| M2-A | merged | #161 append-only SQLite shadow persistence |
| M2-B | merged | #162 read-capability extraction and exact technical decisions |
| M2-C | merged | #164 bounded migration/dual-read comparison |
| M2-D | merged | #165 bounded recovery/rollback rehearsal |
| M2-E | **phone supervisor running; quota/circadian/midnight thresholds observed, final readiness/seal still false; cutover not authorized** | #166-#168, #192, #195, #196 plus operator-reported habitat continuation |
| M3-A | **complete** | #169 drive-dynamics design |
| M3-B | **in progress** | #170-#190 structural/read-only chain; #194 C1; #197/#198 witness surface; #199 reviewed activation + one-shot durable retention; #200 first receipt pin; #201 merged `energy_budget` v1 preflight; #202 live Android-compatible v2 hotfix |
| M3-C | closed | requires stable/completed M3-B |
| M3-D | closed | requires M3-C continuity inputs |
| M3-E | closed | separate reviewed affect/goal cutover; no authority open |

`M2-E` acceptance is not a production cutover authorization. The reviewed resume was executed once after #195 and returned `resume_exit=0`; it must **not** be repeated. PR #196 fixed the later supervisor script-entrypoint failure. The operator subsequently reported the guarded phone supervisor running at `events=288/288`, `runtime_sim_hours=24.86/24`, `midnights=3/3`, `deaths=0`, `divergence=0`, `unauthorized=0`, but `ready=false`. The acceptance contract independently requires all checks, including `target_day_met` (five calendar days), before sealing; therefore the repository does not infer readiness from quota/circadian/midnight completion alone. The M3-B observation window is separate and has **not started**.

### Executable-audit compatibility notes

Historical acceptance wording that is still asserted by executable audit tests remains part of the status contract; it is retained here without reverting the current milestone state above.

M1 status: **closed for mechanism verification**. This is the historical #158 human-acceptance boundary and does not itself authorize persistence cutover or later M-series authority.

The canonical M1 human-acceptance record SHA-256 remains `aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522`. At that decision point the status contract stated **M2-A remains blocked until v4.2 approval**; that sentence is retained as historical acceptance evidence only, because v4.2 was later approved and M2-A subsequently merged in #161. The absorbed PRs #11, #7, and #4 are closed; this is likewise retained as the historical M1 disposition record.

M3-A drive-dynamics design status: the merged #169 artifact remains a **documentation-only** design boundary over the **63-axis Affect Migration Plan**, including **48 bidirectional named transitions**, with **no runtime integration** in that design artifact and **integration eligibility only after persistence cutover**. M3-A is complete as a design milestone, while its historical no-runtime-integration boundary remains authoritative for what #169 itself proved.

## 2. M3-B exact current boundary

Merged M3-B work includes:

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
- production-runtime provenance preflight with fixture-classification binding (#190);
- one-operator attestation trust-root contract (#194);
- operator-side full-engine phone prediction-error witness preflight (#197);
- public-safe witness review v2 completeness (#198);
- reviewed real phone witness C2 activation + one-shot operator-private durable retention command (#199);
- first real retained-observation public receipt pin (#200);
- `energy_budget` full-engine phone operational witness v1 preflight (#201).

After #198 merged, the operator executed the exact full-engine phone witness on head `b4968be9aeb6eefc7274f9985ab333f08e470daf`. The public-safe v2 record pins:

```text
public review digest: 6a3d34120d9773f28544aa82d963cf2e65220f6f899aeab42c132660f87ad81e
attestation digest:   85b55eee61618ad98476f71c4dadcb9b2e4383d79aefd93a41a2c34634efecda
evidence digest:      14549d2b9f37f2a8b00a5bc9de61dbdad8e12dbb8a4d4e08e254ef0e9848b3dc
source:               runtime:ai-adapter:primary
axis:                 prediction_error_pressure
confidence:           1.0
fixture_only:         false
```

The raw prediction/error snapshots and private nonce remain outside the repository.

PR #199 is the merged C2 reviewed activation owner. It deliberately leaves the historical C1/preflight empty registries untouched and adds a versioned C2 activation layer that recomputes the exact public review, registers exactly one reviewed attestation, one runtime-provenance verifier, and one `prediction_error_pressure` production-source verifier, and issues token-protected verification/capture objects. It also provides the operator-only one-shot retention command whose duplicate refusal keeps one real witness from being counted more than once.

After #199 merged, the operator executed that retention command exactly once on clean head `e100bbd26eb84aa65ecae4ecbc10af42fd778476`. The public-safe receipt pinned by #200 proves:

```text
receipt digest:                         ba1c5495e663cc2f7b983e1e834c96ec123733ccab2e3bee3dd6779c6e589d66
event envelope digest:                  07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493
store transition hash:                  c1f16e8a00fa36c7903f0a585b575176830455cee83b26d262a9c04b35013c70
store after chain digest:               d51406d84dc755f72bd2ab661563c75cf19244710bf98376dbe3174ff101c8ce
store before -> after count:             0 -> 1
readback verified:                       true
retained real observation delta:         1
observation window started:              false
cutover authorized:                      false
```

The receipt digest was independently recomputed from its canonical receipt mapping and matches exactly. The private SQLite database, WAL, nonce, raw prediction/error records, and private filesystem path remain outside the repository.

The current merged boundary after #201 is:

```text
source bindings:                                  37/37
production capture adapter:                       present
immutable retention sink:                         present
verifier issuance anti-forgery boundary:          present
prediction_error runtime source bridge:           present
reviewed real operator attestations (C2):          1
registered runtime provenance verifiers (C2):     1
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

PR #201 prepared a **new** source contract path for `energy_budget`. Its v1 collector assumed aggregate `/proc/stat` CPU counters, `/proc/meminfo` available-memory counters, directly readable power-supply battery capacity, and measured witness-process CPU load. After #201 merged as `f178b9e0b0fbaa341776fbef66e6c5c87fe5a157`, the operator executed the exact v1 command on that clean head. Collection stopped before any interaction snapshot or witness serialization with `cannot read CPU counters from /proc/stat`. Therefore no `energy_budget` public review exists from that attempt and no counter advanced.

PR #202 is the live Android-compatibility hotfix. It leaves the failed v1 files intact for audit and adds a versioned v2 acquisition surface. v2 records the exact measurement method and may fall back only from blocked Android filesystem surfaces to real system/API observations: `/proc/stat` -> kernel one-minute load average normalized by visible CPU count; `/proc/meminfo` -> `sysconf` physical/available pages; power-supply sysfs -> `termux-battery-status`. Method-specific raw observations remain private, while the public review exposes method identifiers plus bounded evidence/digests. Missing fallback surfaces fail closed rather than fabricating values. #202 itself still does not review, verify, retain, or pre-count a future `energy_budget` witness.

The first real retained observation is evidenced, but one retained observation is not 37-axis coverage and does not start the M3-B observation window. The retained `prediction_error_pressure` event must not be appended again. Further coverage requires new real production-origin observations for additional source contracts.

No audit fixture, detached synthetic evidence, test verifier, self-authored runtime metadata, `fixture_only=False`, PID, argv/environment flag, caller identity, self-hashed launch metadata, or unreviewed public attestation digest may be reclassified as production evidence. The M2-E habitat driver remains a synthetic scripted shadow workload and cannot substitute for C2 production-source observations.

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

PR #196 closed the separate script-entrypoint defect: `m2_e_window_runtime_guarded_a11.py` installs the repository-root bootstrap before `core...` imports, and a supervisor-equivalent subprocess regression executes the wrapper as a real script. Its accepted exact head was `4944c01df3b0978ae73ea3060abd39bee14e41c1`; the squash merge is `5664fc3bc22054c2d39142b3125416aea6089c63`.

## 4. Validation reuse and new-chat rule

`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json` plus immutable merged-PR validation records are the durable source of truth for validation reuse.

Mandatory rule:

- a new chat/operator session **must inspect the ledger and merged PR reuse records first**;
- reuse an exact-head validation when the exact head, artifact SHA-256, validation scope/dependency, and merge ancestry still match;
- chat/session changes, PR body/title edits, comments/reviews, and Draft/Ready transitions are **not** invalidators;
- a code-head change, artifact loss/corruption/digest mismatch, validation-scope/dependency change, or ancestry break **is** an invalidator;
- discovery/intermediate registration heads are not merge evidence;
- full-suite runs once on the final registered exact head after the forward gate passes.

PR #201 is the latest merged prerequisite pin:

```text
exact head:   fce245e5c4e63f2224b6fe69d54375315896c177
exact run:    30187041821
focused:      5 passed
full:         3,160 passed
artifact:     exact-head-validation-fce245e5c4e63f2224b6fe69d54375315896c177
artifact SHA: ec8b3b6f045e9f39007bd98b0a0b55f680a78622038f5ef1c918abdd4457522c
M2-E run:     30187041822
M2-E:         6/6 jobs passed
merge SHA:    f178b9e0b0fbaa341776fbef66e6c5c87fe5a157
```

PR #201's merged PR body carries its permanent exact-head reuse record, and `main` was directly verified identical to its squash merge before #202 branch creation. The failed phone acquisition changed no repository state and did not invalidate #201's accepted repository validation. A later chat must reuse #201 when its policy conditions still match. #202 is a genuine new repository head and receives its own validation only on the final registered head; discovery/registration-only heads are not merge evidence and must not cause #201 validation to be repeated.

## 5. Governance rules added by the #191 rebaseline

### 5.1 Same-PR STATUS update

Every merge-intended implementation/governance PR must update this STATUS document in the same PR when it changes milestone state, blockers, authority, operational state, or the authoritative next step. A later cleanup PR must not be relied on to repair a knowingly stale completion claim.

### 5.2 Reporting integrity

A PR number, merge state, exact head, workflow result, artifact digest, or completion claim may be written in completed form only after direct repository/workflow verification. Work that has not been pushed/created/verified must be described as planned, proposed, or pending.

On 2026-07-25, before the #191 governance rebaseline was opened, a status report incorrectly stated that PRs #190-#192 had been merged. Direct repository audit showed #190 was still Draft/Open and that no PR objects existed for #191 or #192 at that time. #190 was then independently reviewed, corrected, exact-head validated, and squash-merged as `5af3fc8f2041e54a33384c4a8d60bebccb5a6eb2`; #191 was later actually created, validated, and squash-merged as `9b2545795b681dd0c53a9d51820b6baa70df9482`; #192 was later actually created, validated, and squash-merged as `77443032eb3fe70eac8c8ca8a18909574de81063`. Those later repository events do not retroactively validate the earlier false completion report.

### 5.3 Private companion boundary

Raw phone companion contents, SQLite/WAL files, backups, private nonce material, raw prediction/error witness mappings, raw operational counters, raw battery data, and other non-public habitat/runtime evidence must not be copied into the public repository. Public records may contain approved public evidence schemas, bounded summaries, and cryptographic digests/references needed for authorized recomputation. The `RegistryAxisPositiveConfidenceEvidence` mapping is review-safe because it contains bounded derived value/confidence and provenance/digests, not raw source mappings. A public retention receipt may expose event/chain/capture/verifier digests and counts but not the private SQLite database path or raw witness material.

### 5.4 No automatic authority promotion

Machine-green evidence, PR merge, operator attestation machinery, source registration, retained observations, or an observation-window seal cannot automatically open M3-C/M3-E or authorize cutover. Any authority transition remains a separate explicit reviewed decision.

## 6. PR registry — verified repository history through #202

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
| #194 | merged | C1 operator-attestation trust root |
| #195 | merged | A11 content-addressed habitat persistence repair |
| #196 | merged | A11 wrapper script-bootstrap hotfix |
| #197 | merged | C2 phone prediction-error runtime witness preflight |
| #198 | merged | C2 public-review v2 completeness hotfix |
| #199 | merged | reviewed real phone witness activation + one-shot durable retention command |
| #200 | merged | first real operator-private retention receipt pin; retained coverage `1/37` |
| #201 | merged | `energy_budget` full-engine phone operational witness v1 preflight; first real execution exposed Android `/proc/stat` access blocker and produced no witness |
| #202 | live PR state authoritative | versioned Android-compatible `energy_budget` v2 acquisition fallback; no counter advance until a new real reviewed witness/retention |

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

1. **M2-E habitat continuation:** leave the already-running guarded supervisor alone and do **not** run `resume --reviewed` again. `ready=false` remains authoritative until every acceptance check and sealing condition is actually satisfied.
2. **Do not duplicate the first retained event:** `prediction_error_pressure` sequence 1 is complete. The existing operator-private retention stream must not be appended again to manufacture coverage.
3. **Validate/merge #202 once on its final registered head:** reuse #201 exact-head/M2-E evidence as prerequisite. Discovery/registration heads are not merge evidence; #202 gets one accepted full-suite/M2-E pair only on its final registered exact head.
4. **After #202 merge, execute one new v2 phone witness:** run `scripts/operator/m3_b_phone_energy_budget_witness_v2.py` once with the exact merged head and exactly three real interaction inputs. Return only its final public-review v2 JSON. Keep raw CPU/memory/battery/process observations and nonce private.
5. **Do not invent blocked Android metrics:** the v2 collector may use only its declared real-kernel/API fallbacks. If battery sysfs is blocked and `termux-battery-status` is unavailable, install/use Termux:API or stop; do not substitute a manually typed battery percentage as production evidence.
6. **Review then retain only genuinely new evidence:** a later PR may pin the reviewed `energy_budget` v2 witness, register its exact source verifier/runtime provenance boundary, and append a new retained event only after the public review validates. Until then all M3-B counters remain `1/37`.
7. Satisfy the later 37-axis retained positive-confidence coverage/window-entry contract before starting the M3-B observation window. Only completed/stable M3-B may open M3-C.

The immediate project state remains **M3-B in progress**. The first real phone observation is durably retained at `1/37`. The failed #201 `energy_budget` execution produced no witness and changed no counter. #202 only repairs the acquisition surface and likewise does not pre-count any future output. The observation window is still not started. M3-C, M3-E, and cutover remain closed.
