# EVE v4 Active Agent Instructions

## Active authority

After the EVE v4 Constitution Rebaseline merges, this file and `docs/EVE_DESIGN_v4.md` are the active constitutional authority. Older v3 and v3.1 design/status/dependency documents are retained only as historical reference and do not override this v4 authority.

EVE v4.0 is provisional pending M0. It may be revised to v4.1 after evidence from M0. Evidence-based revision is part of the process and is not a project failure.

## Current implementation status

Do not claim that the v4 runtime is implemented. This rebaseline is documentation and governance only. Existing implementation work is frozen through M0, including open PRs #109, #97, #86, #84, #82, #11, #7, #4, and #1.

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

Record discrete transitions such as input/observation acceptance, memory candidate and consolidation, goal create/suspend/resume, action selection, appraisal completion, skill update stabilization, permission change, and external effect authorization/execution. Do not record every timer tick or continuous decay step. Derived continuous values such as activation, accessibility, energy, or drive decay must be reproducible from base state, model version, parameters, and monotonic elapsed time.

## Future persistence requirements

M1/M2 must later use append-only SQLite event storage, periodic validated snapshots, replay from latest valid snapshot, WAL where supported, explicit transactions, integrity checks, schema versions and migration history, crash recovery, bounded backups, forced-termination resilience, corrupt-snapshot fallback and restore verification, and mobile storage-growth policy. Do not create or activate a database during the constitution rebaseline.

## Memory and forgetting

EVE may not consciously delete historical source events. Original event history is retained. Forgetting is automatic accessibility decay, compression, consolidation, generalization, association change, and cue-based reactivation. Personal recollection and immutable safety/audit history are separate. Migration must preserve provenance and continuity.

## Affect migration

The current 26-hormone architecture is not automatically retained. M0 must inventory all hormone state and mutation, persistence and memory snapshots, and hormone-dependent tests, speech, goals, agency, loops, and persistence behavior. M0 must propose migration toward core drives, appraisal, and derived emotion without orphaning historical memory or breaking identity continuity. Do not change affect implementation during this rebaseline.

## Self-code boundary

EVE may write code only in an isolated sandbox workspace. EVE may not write to its runtime repository, modify executable, constitutional, or security configuration, replace cognition modules, install generated scripts into runtime paths, or indirectly modify itself through tools, dependencies, plugins, scripts, or configuration. Learned weights are not source code, but may update only through observation → candidate → validation → bounded evaluation → stabilization → versioned activation → rollback support.

## Autonomy and privacy

Ordinary internal activity does not require Minseok's approval. Private journals and internal records may exist. Private records and safety-audit records are separate. External communication, account use, expenses, contracts, and physical effects require capability, legal authority where needed, and auditability. Privacy does not erase accountability for external consequences.

## Speech is not life

Timer ticks, hormone decay, and proactive speech are not proof of life or consciousness. Continuity is evaluated through persistent state, memory, goals, independent activity, learning, interruption/resumption, and long-term change. Architecture alone cannot prove subjective consciousness.

## M0 anti-hallucination policy

M0 is evidence-focused and split into M0-A through M0-D. M0-A inventories runtime entrypoints, imports/dependency construction, mutation/direct-write sites, and test classifications. M0-B covers gates, bypasses, outputs, exceptions, clocks, queues, threads, concurrency, and nondeterminism. M0-C covers persistence and affect/hormone migration. M0-D covers neural/vector/adaptive components, life-loop assessment, module disposition, integrated conclusions, and recommendations for frozen PRs.

Every M0 map entry must include path, exact line/range, callable, mechanically detected evidence, manual classification, confidence, and unresolved status where applicable. Grep may supplement AST but cannot be the sole evidence. Generated JSON artifacts must not be committed.

## Test migration policy

M0 may classify tests only as `KEEP`, `RETIRE`, or `REWRITE`. M0 must not delete, skip, xfail, weaken, or rewrite tests. Every classification requires `file:line` evidence and a reason. Test count alone is not a quality measure.

## Engineering rules

Keep changes targeted. Do not modify runtime code, tests, scripts, data, models, vectors, checkpoints, persistence files, defaults, or generated artifacts during governance-only tasks. Do not add external API calls. Do not add nondeterministic behavior. Do not weaken tests.
