# EVE v4 Implementation Status

Last repository rebaseline: **2026-07-27**  
Rebaseline base/prerequisite: `a9f70ef78b06744eba01a0b35c60371b10eaf672` — PR #215 squash merge  
Active constitution: **EVE v4.2**  
Constitution status: **ACTIVE CONSTITUTIONAL AUTHORITY**

## 1. Current authority and milestone state

The explicit A12 human cutover decision is merged in #213 and its separately validated A-2 execution is merged in #215. The event kernel plus SQLite store are now the authoritative persistence substrate for **v4-native subsystems only**. Every legacy domain remains authoritative in the legacy runtime until that domain passes its own separately reviewed migration gate. The minimum seven-day legacy-parallel/rollback-preservation interval is active and is not an automatic migration timer. `m3_authority_open=true`. M3-E affect cutover remains separately closed.

| Milestone | Current state | Repository basis |
|---|---|---|
| M0 | complete | historical audit set + pinned regeneration/invariance |
| M1 | complete for mechanism verification | #145-#158 lineage, explicit M1 human acceptance in #158 |
| M2-A | merged | #161 append-only SQLite shadow persistence |
| M2-B | merged | #162 read-capability extraction and exact technical decisions |
| M2-C | merged | #164 bounded migration/dual-read comparison |
| M2-D | merged | #165 bounded recovery/rollback rehearsal |
| M2-E | **sealed, explicitly human-authorized, operational v4-native cutover active** | #166-#168, #192, #195, #196, sealed habitat evidence, #213 A12 decision, #215 digest-pinned activation |
| M3-A | **complete** | #169 drive-dynamics design |
| M3-B | **in progress** | #170-#190 structural/read-only chain; #194 C1; #197-#212 witness/review/retention chain; canonical reviewed/runtime/source/candidate coverage `5/37`, retained coverage `4/37`; `stress_load` sequence 5 is staged but not yet appended |
| M3-C | **authority open; M3-C-A design under review in #217; runtime implementation not started** | #169 drive dynamics + #215 authority opening + #217 documentation/checker/test candidate |
| M3-D | closed | requires M3-C continuity inputs |
| M3-E | closed | separate reviewed affect/goal cutover; no authority open |

The M2-E habitat window is sealed. The accepted package is pinned by `seal_digest=5bfd2bae9a60107b5bd647eeec30b602a4d6bca922e467755f17a04c990dafbb`, acceptance `12/12`, events `288`, death recoveries `2`, and observed midnights `4`. PR #213 records the explicit human authorization and canonical decision digest `3844e4d0a836924eb881048d45d98d89d5041f87d15a836686119a2d8487efbf`. PR #215 activates only the v4-native persistence substrate and `m3_authority_open`; it does not transfer any legacy-domain authority and does not authorize M3-E. The M3-B observation window is separate and has **not started**.

### Executable-audit compatibility notes

Historical acceptance wording that is still asserted by executable audit tests remains part of the status contract; it is retained here without reverting the current milestone state above.

M1 status: **closed for mechanism verification**. This is the historical #158 human-acceptance boundary and does not itself authorize persistence cutover or later M-series authority.

The canonical M1 human-acceptance record SHA-256 remains `aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522`. At that decision point the status contract stated **M2-A remains blocked until v4.2 approval**; that sentence is retained as historical acceptance evidence only, because v4.2 was later approved and M2-A subsequently merged in #161. The absorbed PRs #11, #7, and #4 are closed; this is likewise retained as the historical M1 disposition record.

M3-A drive-dynamics design status: the merged #169 artifact remains a **documentation-only** design boundary over the **63-axis Affect Migration Plan**, including **48 bidirectional named transitions**, with **no runtime integration** in that design artifact and **integration eligibility only after persistence cutover**. M3-A is complete as a design milestone. The persistence-cutover eligibility condition is now satisfied by #215, but that does not retroactively turn #169 into a runtime implementation.

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
- `energy_budget` full-engine phone operational witness v1 preflight (#201);
- Android-compatible versioned `energy_budget` v2 acquisition fallback (#202);
- reviewed `energy_budget` witness activation + exact sequence-two retention staging (#203);
- public-safe sequence-two `energy_budget` retained-observation receipt pin and canonical retained-count advance to `2/37` (#204);
- `fatigue_pressure` full-engine phone operational witness preflight (#205);
- reviewed `fatigue_pressure` witness activation + exact sequence-three retention staging (#206);
- public-safe sequence-three `fatigue_pressure` retained-observation receipt pin and canonical retained-count advance to `3/37` (#207);
- `recovery_need` full-engine phone operational witness preflight (#208);
- reviewed `recovery_need` witness activation + exact sequence-four retention staging (#209);
- public-safe sequence-four `recovery_need` retained-observation receipt pin and canonical retained-count advance to `4/37` (#210);
- `stress_load` real-phone appraisal witness preflight and one exact real witness execution (#211);
- independent `stress_load` review, fifth reviewed/runtime/source/candidate activation, and exact sequence-five durable-retention staging without executing the private append (#212).

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

PR #201 prepared a **new** source contract path for `energy_budget`. Its v1 collector assumed aggregate `/proc/stat` CPU counters, `/proc/meminfo` available-memory counters, directly readable power-supply battery capacity, and measured witness-process CPU load. After #201 merged as `f178b9e0b0fbaa341776fbef66e6c5c87fe5a157`, the operator executed the exact v1 command on that clean head. Collection stopped before any interaction snapshot or witness serialization with `cannot read CPU counters from /proc/stat`. Therefore no `energy_budget` public review exists from that attempt and no counter advanced.

PR #202 merged as `1161bb15d7bba0629d4862c05e8a61126cdb12c0`. It leaves the failed v1 files intact for audit and adds a versioned v2 acquisition surface. v2 records the exact measurement method and may fall back only from blocked Android filesystem surfaces to real system/API observations: `/proc/stat` -> kernel one-minute load average normalized by visible CPU count; `/proc/meminfo` -> `sysconf` physical/available pages; power-supply sysfs -> `termux-battery-status`. Method-specific raw observations remain private, while the public review exposes method identifiers plus bounded evidence/digests. Missing fallback surfaces fail closed rather than fabricating values.

The operator then executed the exact v2 witness on clean merged head `1161bb15d7bba0629d4862c05e8a61126cdb12c0`. The public-safe record independently reviewed for #203 pins:

```text
public review digest:  a2ce3d84111224e2009bf22d1e03a8f92acab0506e42515aac185ae05ff54ab4
attestation digest:    5413c35e912f95d90a1c0a5b0b8731a243bffc00e7b6338c1b7d9e4056e1c07f
evidence digest:       9d814295e3b59fb42294f3ba661866aa29c512866b946e80a3f397864974af13
source:                runtime:phone-operational-energy:primary
axis:                  energy_budget
confidence:            0.9993993636238185
CPU method:            kernel_loadavg_1m_headroom_v1
memory method:         proc_meminfo_available_v1
battery method:        termux_api_battery_status_v1
fixture_only:          false
synthetic:             false
```

The canonical `public_review_digest` was independently recomputed and matched exactly. No private raw CPU/memory/battery/process values or nonce material were copied into the repository.

PR #203 merged as `9b8ceceb22e2eee08f940e1673b624cbaa9bcf1a`. It pins the reviewed witness, registers the second reviewed attestation/runtime-provenance/source-verifier path, issues token-protected `energy_budget` verification/capture objects, and stages durable retention as **sequence 2 only**. The staged path refuses execution unless the operator-private stream already contains the exact sequence-1 `prediction_error_pressure` event and public chain digest pinned by #200.

After #203 merged, the operator executed that sequence-two retention command exactly once on the clean merged head. The public-safe receipt pinned by #204 was independently canonical-digest verified and proves:

```text
axis:                                      energy_budget
prior event:                               m3b:c2:retained:prediction_error_pressure:000001
new event:                                 m3b:c2:retained:energy_budget:000002
sequence:                                  2
receipt digest:                            56401653404f9dee07804ed6a1027368baf7f118dcf6ca6f24e85a050891e3df
prior event envelope digest:               07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493
event envelope digest:                     1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664
store before -> after count:                1 -> 2
store before chain digest:                  d51406d84dc755f72bd2ab661563c75cf19244710bf98376dbe3174ff101c8ce
store after chain digest:                   d4660b5cef058bad1b9d1b6b1cb2987c78ef9dbbee403c85562ab945535883e0
store transition hash:                      1189dcb8ae01370c9095ad676f2f724b2d579184d48ccef861496b04011b57a6
readback verified:                          true
retained real observation delta:            1
observation window started:                 false
cutover authorized:                         false
```

The private SQLite database, WAL, raw operational measurements, private nonce, private witness companion, and private filesystem path remain outside the repository. Sequence 1 and sequence 2 are immutable prior history and must not be appended again.

PR #205 merged as `1ac94c402d6fb8935614d0a72cda3e622b69ec82`. The operator then executed its `fatigue_pressure` phone witness exactly once on that clean merged head. The public-safe witness supplied for #206 was independently canonical-digest verified:

```text
public review digest:  4b88c7734234ac2982836b95bf392fe143bc928119d4af515e576b39e480af61
attestation digest:    421da78df1035dd994df3098c1345a448fca59b7a36f9d8cc2fb8c3dce0d4db8
evidence digest:       017e189e1a35a26ce47a0372fe558e069679bf03e438ff9767bf3e0f4196a707
source:                runtime:phone-operational-fatigue:primary
axis:                  fatigue_pressure
confidence:            0.9999916437617128
value:                 0.24117046163321723
process CPU method:    os_times_process_cpu_v1
queue method:          kernel_loadavg_1m_normalized_v1
task-switch method:    getrusage_context_switch_delta_v1
fixture_only:          false
synthetic:             false
```

The exact attestation, evidence, and full public-review mappings each recompute to their supplied SHA-256 digest. Raw process CPU/wall time, load-average observations, context-switch counters, private witness material, and nonce remain outside the repository.

PR #206 merged as `08ae20479ab387f8e8962e3b8cbf3cc182a66fca`. It pins the reviewed `fatigue_pressure` witness, registers the third reviewed attestation/runtime-provenance/source-verifier path, issues token-protected verification/capture objects, and stages durable retention as **sequence 3 only**. The staged path refuses execution unless the operator-private stream already contains the exact immutable sequence-1 `prediction_error_pressure` plus sequence-2 `energy_budget` history.

After #206 merged, the operator executed that sequence-three retention command exactly once on the clean merged head. The public-safe receipt pinned by #207 was independently canonical-digest verified and proves:

```text
axis:                                      fatigue_pressure
prior event:                               m3b:c2:retained:energy_budget:000002
new event:                                 m3b:c2:retained:fatigue_pressure:000003
sequence:                                  3
receipt digest:                            cef1b731eb6b3b15ebef1106bea3f12d2b053afbfd18b35aebeaca46dd143f66
prior event envelope digest:               1e4bd659ef348ac39588ba2bc13440bd96a81a9c24a4cdf804bf9ef48b23f664
event envelope digest:                     f81d43bf40b4dc76130767f91b65ad2503bc70e61ef718fe3d0e446528d1a7e3
store before -> after count:                2 -> 3
store before chain digest:                  d4660b5cef058bad1b9d1b6b1cb2987c78ef9dbbee403c85562ab945535883e0
store after chain digest:                   b73ec7ea2f5e6e4e8eda5b57b4f6464a17d94e56026718b5b2e15cbca9f2162f
store transition hash:                      7d0e608b245506836722d3bbbe609e29f8bdac55f435958ac0f69e456aee4929
store ordinal:                              3
readback verified:                          true
retained real observation delta:            1
retained real observation count:            3/37
observation window started:                 false
M3-B complete:                              false
M3-C open:                                  false
M3-E authority open:                        false
cutover authorized:                         false
```

The receipt digest was independently recomputed from the canonical sorted compact receipt mapping and matches exactly. The private SQLite database, WAL, raw CPU/load/context-switch observations, private witness material, private nonce, and private filesystem path remain outside the repository. Sequences 1, 2, and 3 are immutable prior history and must not be appended again.

PR #207 merged as `90933501e1a3b15b4721d5ffd944c00b168daf4e`. It pins the public-safe sequence-three receipt above and makes canonical retained real-observation coverage `3/37`. The receipt pin itself does not add a fourth reviewed witness or verifier and does not start the M3-B observation window.

PR #208 merged as `f0edb05201671814fed131ccbb73d2cb3b8d3f59`. The operator then executed its `recovery_need` phone witness exactly once on that clean merged head. The public-safe record supplied for #209 was independently canonical-digest verified:

```text
public review digest:  e46df034d01b13e768ce37d14261b8ed20fdec30101945bea492d97e482e4c33
attestation digest:    ce8cda9955a415ed05200a83fd6b3e8d4cd4028bef29f73b8b17a1d5e3ad25e1
evidence digest:       535495759c0140d875da628d2fe5cc9ffc0904d5f91fa9546a784dd51b3baa4b
source:                runtime:phone-operational-recovery:primary
axis:                  recovery_need
confidence:            0.9999985112700722
value:                 0.2559025046410264
process CPU method:    os_times_process_cpu_v1
queue method:          kernel_loadavg_1m_capacity_comparison_v1
cooldown method:       fixed_post_interaction_quiet_window_1s_v1
overload-count method: loadavg_visible_cpu_capacity_breach_count_v1
recovery-count method: cpu_and_queue_nonincrease_indicator_count_v1
fixture_only:          false
synthetic:             false
```

The exact attestation, evidence, and full public-review mappings each recompute to their supplied SHA-256 digest. Raw process CPU/wall time, load-average observations, private witness material, private nonce, and private filesystem path remain outside the repository.

PR #209 merged as `715f0b6da087add988d9628d083354505ffc064d`. It pins the reviewed `recovery_need` witness, registers the fourth reviewed attestation/runtime-provenance/source-verifier path, issues token-protected verification/capture objects, and stages durable retention as **sequence 4 only**. The staged append refuses execution unless the operator-private stream contains exactly the immutable sequence-1 `prediction_error_pressure`, sequence-2 `energy_budget`, and sequence-3 `fatigue_pressure` history with the pinned sequence-three envelope and store-chain digest.

After #209 merged, the operator executed that sequence-four retention command exactly once on the clean merged head. The public-safe receipt pinned by #210 was independently canonical-digest verified and proves:

```text
axis:                                      recovery_need
prior event:                               m3b:c2:retained:fatigue_pressure:000003
new event:                                 m3b:c2:retained:recovery_need:000004
sequence:                                  4
receipt digest:                            e776859e16b34a9222264f3d500993e4bf56ce2397c73b132075cb51fe3967c6
prior event envelope digest:               f81d43bf40b4dc76130767f91b65ad2503bc70e61ef718fe3d0e446528d1a7e3
event envelope digest:                     7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691
store before -> after count:                3 -> 4
store before chain digest:                  b73ec7ea2f5e6e4e8eda5b57b4f6464a17d94e56026718b5b2e15cbca9f2162f
store after chain digest:                   16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f
store transition hash:                      f36add507ebab0913ce29b6c2911d169e634cc59e9902e775bb4ee21e5fb2385
store ordinal:                              4
readback verified:                          true
retained real observation delta:            1
retained real observation count:            4/37
observation window started:                 false
M3-B complete:                              false
M3-C open:                                  false
M3-E authority open:                        false
cutover authorized:                         false
```

The receipt digest was independently recomputed from the canonical sorted compact receipt mapping and matches exactly. The private SQLite database, WAL, raw CPU/wall/load values, private witness material, private nonce, and private filesystem path remain outside the repository. Sequences 1 through 4 are immutable prior history and must not be appended again.

PR #210 merged as `b613e570b4c27ed75ebdb93aaef5a4756ffb44a4`. It pins the public-safe sequence-four receipt above and makes canonical retained real-observation coverage `4/37`. It does not add a fifth reviewed witness or verifier and does not start the M3-B observation window.

PR #211 merged as `3298d3b9911c79b1551a1d8bfe83bae756880840`. Its real-phone `stress_load` witness was executed exactly once on that clean merge head. The operator-private CPU/wall/kernel-load observations were used only as input to the deterministic appraisal bridge; the detached canonical appraisal record remained `runtime_polled=false` and `hardware_direct_input=false`. The exact historical interaction outputs are immutable witness history and are not rerun to improve wording.

PR #212 merged as `cbc6458a532e78664c798563afef966caebc9167`. Independent review pinned the fifth `stress_load` witness:

```text
axis:                 stress_load
attestation digest:   7191e3493c582a191db3dcd488b2452dd3b0f29774b8a3a3ffeaff3b53c525fa
evidence digest:      5bceb97155a5614de72b2b359b861db5c57eb6e892c259b56981d6003fc14680
public review digest: 1ec63bb54cfed398b0e5b93af25667474c3255d5ca50a47602974d363cf5e03a
confidence:           0.999752717989861
value:                0.29720805203604805
fixture_only:         false
synthetic:            false
```

#212 registers the fifth reviewed attestation/runtime-provenance/source-verifier/candidate path and stages exactly `m3b:c2:retained:stress_load:000005` as sequence 5. The operator-private append has **not** been executed. It fails closed unless sequences 1-4, including sequence-four envelope `7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691` and store-chain digest `16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f`, are exact.

The exact current reviewed/retained boundary is:

```text
source bindings:                                  37/37
production capture adapter:                       present
immutable retention sink:                         present
verifier issuance anti-forgery boundary:          present
prediction_error runtime source bridge:           present
reviewed real operator attestations (C2):          5
registered runtime provenance verifiers (C2):     5
verified production runtime anchors (C2):         5
registered production source verifiers (C2):      5/37
verified positive-confidence candidates:          5/37
retained real observation:                        4/37
retained positive-confidence real observation:    4/37
stress_load witness preflight:                    present
stress_load real witness executed:                true — exactly once
stress_load reviewed/registered:                  true
stress_load retained:                             false — sequence 5 staged only
M3-B observation window eligible:                 false
M3-B observation window started:                  false
M3-B complete:                                    false
A12 human cutover decision accepted:              true — #213
v4-native operational cutover active:             true — #215
M3-C authority gate:                              open — #215
M3-C runtime implementation:                      false
M3-E authority open:                              false
```

No audit fixture, detached synthetic evidence, test verifier, self-authored runtime metadata, `fixture_only=False`, PID, argv/environment flag, caller identity, self-hashed launch metadata, or unreviewed public attestation digest may be reclassified as production evidence. For the appraised-survival path, raw runtime metrics also may not be relabeled as a detached #180 canonical record: the production-origin bridge and detached appraisal output must remain separately digested and reviewable. The M2-E habitat driver remains a synthetic scripted shadow workload and cannot substitute for C2 production-source observations.

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

The later sealed habitat package supersedes the earlier live-window readiness state for current authority purposes. The historical incident and repair text above remains immutable evidence and is not rewritten to pretend the window was already sealed at those earlier timestamps.

## 4. Validation reuse and new-chat rule

`docs/audit/EXACT_HEAD_VALIDATION_REUSE_LEDGER.json` plus immutable merged-PR validation records and per-PR durable reuse pins are the source of truth for validation reuse.

Mandatory rule:

- a new chat/operator session **must inspect the ledger and merged PR reuse records first**;
- reuse an exact-head validation when the exact head, artifact SHA-256, validation scope/dependency, and merge ancestry still match;
- chat/session changes, PR body/title edits, comments/reviews, and Draft/Ready transitions are **not** invalidators;
- a code-head change, artifact loss/corruption/digest mismatch, validation-scope/dependency change, or ancestry break **is** an invalidator;
- discovery/intermediate registration heads are not merge evidence;
- full-suite runs once on the final registered exact head after the forward gate passes.

PR #215 is the latest merged authority prerequisite pin:

```text
exact head:   03f5d2365aae46ebe6cd950bb234c8062c3cdc63
exact run:    30255739310
focused:      5 passed
full:         3,212 passed
artifact:     exact-head-validation-03f5d2365aae46ebe6cd950bb234c8062c3cdc63
artifact SHA: c646393008bdc7e6d40177c81e6d86b236f3cb0e1da5d40338b542a1fc3a56be
M2-E run:     30255739240
M2-E:         6/6 jobs passed
merge SHA:    a9f70ef78b06744eba01a0b35c60371b10eaf672
```

The #215 exact-head/M2-E evidence remains reusable unless one of the explicit invalidators above occurs. `docs/audit/M2_E_PR215_VALIDATION_REUSE_PIN.json` is introduced on the active post-cutover work branches to make that reuse durable across chat/session changes. Do not schedule #215 full-suite or M2-E again merely because work moves to Track B, M3-C-A, a receipt-pin PR, or another chat.

## 5. Governance rules added by the #191 rebaseline

### 5.1 Same-PR STATUS update

Every merge-intended implementation/governance PR must update this STATUS document in the same PR when it changes milestone state, blockers, authority, operational state, or the authoritative next step. A later cleanup PR must not be relied on to repair a knowingly stale completion claim.

### 5.2 Reporting integrity

A PR number, merge state, exact head, workflow result, artifact digest, or completion claim may be written in completed form only after direct repository/workflow verification. Work that has not been pushed/created/verified must be described as planned, proposed, or pending.

On 2026-07-25, before the #191 governance rebaseline was opened, a status report incorrectly stated that PRs #190-#192 had been merged. Direct repository audit showed #190 was still Draft/Open and that no PR objects existed for #191 or #192 at that time. #190 was then independently reviewed, corrected, exact-head validated, and squash-merged as `5af3fc8f2041e54a33384c4a8d60bebccb5a6eb2`; #191 was later actually created, validated, and squash-merged as `9b2545795b681dd0c53a9d51820b6baa70df9482`; #192 was later actually created, validated, and squash-merged as `77443032eb3fe70eac8c8ca8a18909574de81063`. Those later repository events do not retroactively validate the earlier false completion report.

### 5.3 Private companion boundary

Raw phone companion contents, SQLite/WAL files, backups, private nonce material, raw prediction/error witness mappings, raw operational counters, raw battery data, raw interaction text, process CPU/wall timing, load averages, process context-switch counters, and other non-public habitat/runtime evidence must not be copied into the public repository. Public records may contain approved public evidence schemas, bounded summaries, measurement-method identifiers, and cryptographic digests/references needed for authorized recomputation. The `RegistryAxisPositiveConfidenceEvidence` mapping is review-safe because it contains bounded derived value/confidence and provenance/digests, not raw source mappings. A public retention receipt may expose event/chain/capture/verifier digests and counts but not the private SQLite database path or raw witness material.

### 5.4 No automatic authority promotion

Machine-green evidence, PR merge, operator attestation machinery, source registration, retained observations, or an observation-window seal cannot automatically open M3-C/M3-E or authorize cutover. Any authority transition remains a separate explicit reviewed decision. The explicit #213 A12 human decision plus separately validated #215 execution authorize only the v4-native persistence substrate and M3 authority opening recorded there. Neither transfer a legacy domain nor open M3-E.

## 6. PR registry — verified repository history through #215 plus active #216/#217

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
| #202 | merged | versioned Android-compatible `energy_budget` v2 acquisition fallback; exact phone witness subsequently produced on its merge head |
| #203 | merged | reviewed `energy_budget` witness activation + sequence-two retention staging; post-merge real sequence-two append succeeded |
| #204 | merged | public-safe sequence-two retention receipt pin; canonical retained coverage `2/37` |
| #205 | merged | `fatigue_pressure` real-phone witness preflight; exact phone witness subsequently produced on its merge head |
| #206 | merged | reviewed `fatigue_pressure` witness activation + exact sequence-three retention staging; post-merge real sequence-three append succeeded |
| #207 | merged | public-safe sequence-three `fatigue_pressure` retention receipt pin; canonical retained coverage `3/37` |
| #208 | merged | `recovery_need` real-phone witness preflight; exact public-safe witness subsequently produced on its merge head |
| #209 | merged | reviewed `recovery_need` witness activation + exact sequence-four retention staging; post-merge real sequence-four append succeeded |
| #210 | merged | public-safe sequence-four `recovery_need` retention receipt pin; canonical retained coverage `4/37` |
| #211 | merged | `stress_load` real-phone appraisal witness preflight; exact phone witness subsequently executed once on its merge head |
| #212 | merged | independent `stress_load` witness review, fifth reviewed/runtime/source/candidate activation, sequence-five retention staging; retained coverage remains `4/37` |
| #213 | merged | explicit A12 human cutover authorization; legacy per-domain authority retained; M3-E explicitly not authorized |
| #214 | closed, unmerged | superseded pre-cutover-base Track B routing discovery; no accepted full-suite evidence |
| #215 | merged | digest-pinned v4-native persistence authority activation, seven-day legacy-parallel guard, tested private operational rollback; `m3_authority_open=true` |
| #216 | draft, unmerged | clean post-cutover Track B Layer-1 routing repair; full validation may complete but merge is held until the operator sequence-five append no longer requires main to remain at #215 |
| #217 | draft, unmerged | M3-C-A deterministic eight-drive goal generation/selection design; checker/focused discovery green, final forward registration/STATUS exact-head pending |

Current merge boundary: **do not change `main` away from `a9f70ef78b06744eba01a0b35c60371b10eaf672` until the operator executes the staged sequence-five `stress_load` append pinned to that exact head.** Branch work and validation may proceed without changing main.

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

Order is constrained by the exact phone evidence and authority boundaries:

1. **Operator sequence-five append first, while main remains exact #215 merge `a9f70ef78b06744eba01a0b35c60371b10eaf672`.** Execute only `scripts/operator/m3_b_c2_retain_reviewed_stress_load.py` against the already-reviewed public `stress_load` witness and the existing private retained DB. Do not rerun the #211 witness. Canonical retained coverage remains `4/37` until the resulting public-safe receipt is independently reviewed and pinned.
2. **Track B #216 remains routing-only.** It may be validated on its own branch, but do not merge it before step 1 because that would change the exact phone expected head. After sequence-five execution, merge only if its final exact-head/forward/full/M2-E evidence is green. `INTENT_POOLS` and SpeechHub structure remain untouched; pool teardown stays M6.
3. **Finish M3-C-A #217 as design-only.** Register only its static checker/test forward occurrences, keep the same-PR STATUS current, then run one final registered exact-head validation. Its design must integrate all eight M3-A drives, recheck the 59/4 Affect Plan boundary, prove A9 no-continuous/no-duplicate behavior, and show a drive-state-only counterfactual that flips the deterministic selected goal proposal. No M3-E authority or legacy goal-domain transfer.
4. **After the sequence-five receipt is reviewed, pin it publicly and only then advance retained real-observation coverage to `5/37`.** Private SQLite/WAL, nonce, raw witness values, and filesystem paths remain outside the repository.
5. **Do not duplicate accepted validation because of chat/session changes.** #212, #213, and #215 exact-head/full/M2-E evidence are durable prerequisites; only a real tree/head change, artifact loss/corruption/digest mismatch, validation-scope/dependency change, or ancestry break permits a rerun.
6. **Keep legacy authority per-domain and M3-E closed.** The seven-day parallel interval preserves rollback availability; reaching day seven cannot automatically transfer a legacy domain, delete a legacy persistence path, or open M3-E.

The immediate project state is **M2-E operationally cut over for v4-native persistence, M3-C authority open, M3-B retained sequence five awaiting the operator-private append, Track B repaired on a clean draft branch, and M3-C-A design under exact static review**.