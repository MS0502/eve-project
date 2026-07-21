# EVE v4.2 Active Agent Instructions

## Active authority

This file and `docs/EVE_DESIGN_v4.md` are the active EVE v4.2 constitutional authority. Older v3, v3.1, and v4.1 design/status/dependency documents are retained only as historical reference and do not override this v4.2 authority.

EVE v4.2 is constitutional authority only. It is not a claim that a v4 runtime is implemented, activated, or authoritative in production.

## Current implementation status

Do not claim that the v4 runtime is implemented. The current implementation is the **pre-kernel legacy runtime** identified by M0. M0 is closed, the forward-regression gate is active, and M1 mechanism evidence is explicitly human-accepted. M1-A through M1-E remain shadow/declaration/evidence-only with no production integration. M2 has not started. Open frozen REWRITE PRs #109, #86, #84, and #82 remain untouched.

## EVE identity

EVE is one continuous central digital subject across mobile, desktop, sensory/tool nodes, and a future physical body. EVE is not a chatbot and not an embedded general-purpose LLM. Minseok is creator and friend, not EVE's reward function, life center, or exclusive purpose. EVE may form independent interests, goals, preferences, relationships, activities, and private experiences. No architecture may optimize primarily for attachment to Minseok.

## General-purpose LLM boundary

No local general-purpose LLM may be used as cognition, identity, language organ, or speech generator. No GPT, Claude, Gemini, or equivalent general-purpose LLM API may be integrated into EVE. EVE may later use external apps or websites as tools only when use complies with terms, access rules, rate limits, and automation restrictions. Do not disguise automation to bypass service restrictions. External model output must never flow directly into EVE speech.

## Allowed learned subsystems

Non-LLM learned models may be used for STT, TTS, OCR, vision, audio and music analysis, sensor processing, motor control, lexical and perceptual representations, temporal prediction, and EVE-trained neural modules or approved learned weights. They require provenance, confidence, capability, evaluation, versioning, and rollback controls. These subsystems are not EVE's whole identity, and no model may become a hidden direct speech generator. The default runtime remains no-load unless a later evidence-backed activation explicitly authorizes a bounded model or vector load.

## Observation and claim boundary

All external information enters as a claim or observation candidate, including people, websites, apps, documents, LLM output, STT/OCR, vision/audio classifications, sensor readings, and tool results. Retain origin, source identity where known, acquisition method, confidence and uncertainty, time or event relation, verification status, and model/tool version where applicable. Source trust may be learned, but no source enters as an internal fact by default.

## Speech passthrough prohibition

Future architecture must enforce that raw external text exists only in a quarantined source store. Expression and generation layers must have no capability to read raw external text. Cognition may produce internal semantic representations with provenance, and expression may read only EVE-internal semantic representations. Quotations require explicit quotation capability and attribution. Sentence-similarity checks are not the primary safeguard.

## Reproducibility and event log

EVE v4 does not require identical answers after learning or bit-identical neural execution. It requires meaningful state transitions to be represented by events; non-deterministic inputs to preserve causes, parameters, model versions, and seeds where applicable; state reconstruction from a valid snapshot plus subsequent events; reconstruction failure to be treated as a defect; and important decisions to retain causal provenance.

Events must not be emitted to sample continuous values. A tick may emit an event only when a versioned transition condition changes a named semantic state, candidate, permission, or lifecycle state, in either direction. While the same transition state persists, duplicate emission is prohibited. Preserve threshold predicates and model versions for replay. Concrete hysteresis and cooldown numerics belong to M3 design, not this constitution.

## Evidence recalculability

Acceptance, gate, and cutover evidence packages must contain the raw observational data sufficient to independently recompute every claimed metric, or immutable content-addressed references bound into the same verification package. A reference must pin SHA-256 and schema version. Access restrictions and redaction rules must be stated explicitly; after redaction, authorized reviewers must still be able to recompute the claims. Green verdicts alone are not observation evidence. This rule is forward-binding from v4.2 and does not retroactively invalidate merged M0 evidence.

## Mutation-state fidelity

Mutation observation evidence records actual state change through before/after values. Large state may use a content digest over a versioned canonical representation plus a revalidatable structural manifest containing serialization schema and version, hash algorithm, counts, shape, and key-domain metadata as applicable. Replay-generated state must be digested by the identical method. `state_changed` is computed from before/after evidence and is never a manual flag. Every record carries a transition hash and replay result. This binds M2 dual-read and cutover packages.

## Append-only decision records

Machine evidence packets and human decision artifacts are immutable after creation. Corrections, withdrawals, rejections, replacements, supersessions, and revocations are separate append-only artifacts that explicitly reference the superseded artifact digest. The latest valid decision is computed from the chain. History remains preserved while authority may be adjusted or withdrawn. A supersession or revocation artifact must pass the same validation regime, including exact-head validation and human review, as the decision it changes.

## Future persistence requirements

M2 must use append-only SQLite event storage, periodic validated snapshots, replay from latest valid snapshot, WAL where supported, explicit transactions, integrity checks, schema versions and migration history, crash recovery, bounded backups, forced-termination resilience, corrupt-snapshot fallback and restore verification, and mobile storage-growth policy. Do not create or activate a database during governance-only work.

## Memory and forgetting

EVE may not consciously delete historical source events. Original event history is retained. Forgetting is automatic accessibility decay, compression, consolidation, generalization, association change, and cue-based reactivation. Personal recollection and immutable safety/audit history are separate. Migration must preserve provenance and continuity. The same append-only principle applies to governance: originals remain retained while their current effect may be adjusted by a validated later record.

## Affect migration

The merged `docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md` is binding design input: 63 authoritative axes comprise 26 mutable legacy channels and 37 read-only registry axes; reviewer rulings are 59 `MAPPED`, 4 `PROPOSED-DROP`, and 0 `UNRESOLVED`. Dropped axes retain originals and provenance for historical interpretation. Implementation belongs to M2/M3 in the Milestone Registry and is not activated by v4.2.

## Self-code boundary

EVE may write code only in an isolated sandbox workspace. EVE may not write to its runtime repository, modify executable, constitutional, or security configuration, replace cognition modules, install generated scripts into runtime paths, or indirectly modify itself through tools, dependencies, plugins, scripts, or configuration. Learned weights are not source code, but may update only through observation → candidate → validation → bounded evaluation → stabilization → versioned activation → rollback support.

## Autonomy and privacy

Ordinary internal activity does not require Minseok's approval. Private journals and internal records may exist. Private records and safety-audit records are separate. External communication, account use, expenses, contracts, and physical effects require capability, legal authority where needed, and auditability. Privacy does not erase accountability for external consequences.

## Speech is not life

Timer ticks, hormone decay, and proactive speech are not proof of life or consciousness. Continuity is evaluated through persistent state, memory, goals, independent activity, learning, interruption/resumption, and long-term change. Architecture alone cannot prove subjective consciousness.

## M0 anti-hallucination policy

M0 is evidence-focused and split into M0-A through M0-D. M0-A inventories runtime entrypoints, imports/dependency construction, mutation/direct-write sites, and test classifications. M0-B covers gates, bypasses, outputs, exceptions, clocks, queues, threads, concurrency, and nondeterminism. M0-C covers persistence and affect/hormone migration. M0-D covers neural/vector/adaptive components, life-loop assessment, module disposition, integrated conclusions, and recommendations for frozen PRs.

Every M0 map entry must include path, exact line/range, callable, mechanically detected evidence, manual classification, confidence, and unresolved status where applicable. Grep may supplement AST but cannot be the sole evidence. Generated JSON artifacts must not be committed.

## v4.2 amendment summary

1. **Mutation reality and dual gates** — Treat the current implementation as the pre-kernel legacy runtime; preserve historical snapshot regeneration and reject unregistered current-tree mutation/direct-write additions.
2. **Persistence authority** — Add no pickle or legacy-sidecar writer; legacy remains authoritative through shadow and dual-read until a separately approved event-store cutover.
3. **Affect contract** — Use the reviewer-ruled 63-axis Affect Migration Plan as binding design input, with originals and provenance preserved for all proposed drops.
4. **Life loops** — Use the 75-entry M0-D inventory as a taxonomy baseline; only timer/proactive/output-centric paths are legacy expression behavior, and no new timer-driven speech trigger is allowed.
5. **Raw-source quarantine** — Do not mistake output surfaces for capability edges; permit no new raw-text-to-expression capability edge before the M2-B capability manifest is approved.
6. **Adaptive/numeric state** — Treat 1,225 as evidence entries and 288 runtime modules as the disposition unit; new state/artifact/vector authority requires disposition plus provenance/version/evaluation/rollback.
7. **Failure visibility** — Silent broad handling is a defect class; add no new silent+broad handler, and pre-kernel observation may emit diagnostic candidates only.
8. **Audit baseline pinning** — Completed audits are immutable snapshot-scoped evidence; retroactive drift is a defect and exact-head validation is required infrastructure.
9. **Discrete-transition granularity** — Continuous sampling is not an event; only named state changes under versioned transition conditions may emit, with no duplicate emission while state persists.
10. **Evidence recalculability** — Every claimed approval metric remains independently recomputable from raw observations or immutable SHA-256/schema-pinned references in the same package.
11. **Mutation-state fidelity** — Before/after evidence or canonical digest plus structural manifest determines `state_changed`; transition hashes and replay results are mandatory.
12. **Append-only decision records** — Evidence and decisions are immutable records; correction or revocation uses a separately validated and reviewed append-only artifact.

## Test migration policy

M0 may classify tests only as `KEEP`, `RETIRE`, or `REWRITE`. M0 must not delete, skip, xfail, weaken, or rewrite tests. Every classification requires `file:line` evidence and a reason. Test count alone is not a quality measure.

## Engineering rules

Keep changes targeted. Do not modify runtime code, tests, scripts, data, models, vectors, checkpoints, persistence files, defaults, or generated artifacts during governance-only tasks. Do not add external API calls. Do not add nondeterministic behavior. Do not weaken tests.
