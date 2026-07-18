# EVE v4 Architecture Rebaseline Plan

EVE v4.0 is provisional pending M0. It may be revised to v4.1 after evidence from M0. Evidence-based revision is part of the process and is not a project failure.

## Freeze boundary

The rebaseline is documentation and governance only. Do not start new cognition schema work, conclusion/decision candidate schema work, persistence activation, runtime mapping, enforcement, vector/model/device activation, event-kernel implementation, memory migration, or affect migration in this PR.

## M0 sequence

### M0-A

Inventory runtime entrypoints, imports and dependency construction, mutation and direct-write sites, and tests. Required future files are `scripts/audit/m0_a_runtime_inventory.py`, `tests/audit/test_m0_a_runtime_inventory.py`, `docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md`, and `docs/audit/M0_A_TEST_CLASSIFICATION.md`.

### M0-B

Audit gates, bypasses, outputs, exceptions, silent failures, clocks, queues, threads, concurrency, and nondeterminism.

### M0-C

Audit persistence and persistence-intended state, including pickle, JSON, JSONL, databases, vectors, vocabularies, checkpoints, autosave targets, debug exports, episodic/semantic memory, self-model, relationships, affect/hormones, goals, learned parameters, and operator artifacts. It must also inventory hormone-to-drive migration requirements.

### M0-D

Audit neural, vector, and adaptive components; life-loop evidence; module disposition; integrated conclusions; and recommendations for all frozen PRs.

## Evidence requirements

Every M0 map entry must include path, exact line/range, callable, mechanically detected evidence, manual classification, confidence, and unresolved status where applicable. Grep may supplement AST but cannot be the sole evidence. Manual-only entries must be marked `manual_only: true`, unsupported call paths must be marked `unresolved`, and generated JSON artifacts must not be committed.

## Test migration policy

M0 classifies relevant tests only as `KEEP`, `RETIRE`, or `REWRITE`. It must not delete, skip, xfail, weaken, or rewrite tests. Every classification requires file:line evidence and a reason.
