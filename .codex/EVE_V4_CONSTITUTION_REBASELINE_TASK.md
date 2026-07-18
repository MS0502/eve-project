# EVE v4 Constitution Rebaseline — Codex Task

Repository: `MS0502/eve-project`

This is a documentation-and-governance-only task. Do not implement runtime features.

## Baseline and preflight

Required baseline:

```text
65a9ec153f41233df1680ffb93bf79502aaa7fb7
```

Run:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline --decorate
git remote -v
git rev-parse --show-toplevel
```

Proceed only when:

- worktree is clean
- repository is exactly `MS0502/eve-project`
- integration branch is `main`
- HEAD exactly matches the required baseline SHA

If any of these differ, stop and report the exact mismatch. Do not infer or substitute another repository, branch, or commit.

Confirm validation targets exist:

```bash
for target in adapters core cognition tests scripts main.py; do
  if [ ! -e "$target" ]; then
    echo "missing required validation target: $target" >&2
    exit 2
  fi
done
```

Do not substitute guessed legacy paths such as `v36_modules/` or `eve_v35_release/`. Do not run `scripts/check.sh` unless it actually exists.

Create:

```bash
git switch -c codex/eve-v4-constitution-rebaseline
```

## Global development freeze

Freeze all existing implementation work during this task and M0.

Known open PRs:

```text
#109 #97 #86 #84 #82 #11 #7 #4 #1
```

Do not modify, rebase, merge, extend, or reuse those branches. PR #109 must remain frozen.

Do not start any new cognition schema, conclusion/decision candidate schema, persistence activation, runtime mapping, enforcement, vector/model/device activation, event-kernel implementation, memory migration, or affect migration.

## Allowed files

Modify or create only:

```text
AGENTS.md
docs/EVE_DESIGN_v4.md
docs/EVE_ARCHITECTURE_REBASELINE_PLAN.md
docs/EVE_IMPLEMENTATION_STATUS_v4.md
docs/EVE_DESIGN_v3.md
docs/EVE_DESIGN_v3_1.md
docs/EVE_IMPLEMENTATION_STATUS_v3_1.md
docs/EVE_DEPENDENCY_MAP_v3_1.md
```

Do not modify production code, adapters, cognition, core, runtime entrypoints, scripts, tests, data, models, vectors, checkpoints, persistence files, defaults, or generated artifacts.

Add this notice at the top of every old v3/v3.1 document:

```text
SUPERSEDED — HISTORICAL REFERENCE ONLY
```

Preserve historical contents. Do not rewrite or delete them.

After merge, `AGENTS.md` and `docs/EVE_DESIGN_v4.md` become the active authority. Old v3/v3.1 documents remain historical references only.

## Constitutional status

State clearly:

> EVE v4.0 is provisional pending M0. It may be revised to v4.1 after evidence from M0. Evidence-based revision is part of the process and is not a project failure.

Do not claim the v4 runtime is already implemented.

## Required v4 principles

### 1. EVE identity

- EVE is one continuous central digital subject across mobile, desktop, sensory/tool nodes, and a future physical body.
- EVE is not a chatbot and not an embedded general-purpose LLM.
- Minseok is creator and friend, not EVE's reward function, life center, or exclusive purpose.
- EVE may form independent interests, goals, preferences, relationships, activities, and private experiences.
- No architecture may optimize primarily for attachment to Minseok.

### 2. General-purpose LLM boundary

- No local general-purpose LLM as cognition, identity, language organ, or speech generator.
- No GPT, Claude, Gemini, or equivalent general-purpose LLM API integrated into EVE.
- EVE may later use external apps or websites as tools.
- External service use must comply with terms, access rules, rate limits, and automation restrictions.
- Do not disguise automation to bypass service restrictions.
- External model output must never flow directly into EVE speech.

### 3. Allowed non-LLM learned models

Allowed under provenance, confidence, capability, evaluation, versioning, and rollback controls:

- STT, TTS, OCR
- vision, audio, and music analysis
- sensor processing and motor-control models
- lexical and perceptual representations
- temporal predictors
- EVE-trained neural modules and approved learned weights

These are subsystems, not EVE's whole identity. No model may become a hidden direct speech generator.

### 4. Observation and claim boundary

All external information enters as a claim or observation candidate, including people, websites, apps, documents, LLM output, STT/OCR, vision/audio classifications, sensor readings, and tool results.

Retain:

- origin
- source identity where known
- acquisition method
- confidence and uncertainty
- time or event relation
- verification status
- model/tool version where applicable

Source trust may be learned, but no source enters as an internal fact by default.

### 5. Structural prohibition on speech passthrough

Future architecture must enforce:

- raw external text exists only in a quarantined source store
- expression and generation layers have no capability to read raw external text
- cognition may produce internal semantic representations with provenance
- expression reads only EVE-internal semantic representations
- quotations require explicit quotation capability and attribution
- sentence-similarity checks are not the primary safeguard

Document only. Do not implement now.

### 6. Event-log reproducibility

Do not require identical answers after learning or bit-identical neural execution.

Require:

1. every meaningful state transition is represented by an event
2. non-deterministic inputs preserve causes, parameters, model versions, and seeds where applicable
3. state reconstructs from a valid snapshot plus subsequent events
4. reconstruction failure is a defect
5. important decisions retain causal provenance

### 7. Event granularity

Record discrete transitions such as input/observation acceptance, memory candidate and consolidation, goal create/suspend/resume, action selection, appraisal completion, skill update stabilization, permission change, and external effect authorization/execution.

Do not record every timer tick or continuous decay step.

Derived continuous values such as activation, accessibility, energy, or drive decay must be reproducible from a base state, model version, parameters, and monotonic elapsed time.

### 8. Future persistence requirements

M1/M2 must later use:

- append-only SQLite event storage
- periodic validated snapshots
- replay from latest valid snapshot
- WAL where supported
- explicit transactions
- integrity checks
- schema versions and migration history
- crash recovery
- bounded backups
- forced-termination resilience
- corrupt-snapshot fallback and restore verification
- mobile storage-growth policy

Do not create or activate a database in this PR.

### 9. Memory and forgetting

- EVE may not consciously delete historical source events.
- Original event history is retained.
- Forgetting is automatic accessibility decay, compression, consolidation, generalization, association change, and cue-based reactivation.
- Personal recollection and immutable safety/audit history are separate.
- Migration must preserve provenance and continuity.

### 10. Affect migration

The current 26-hormone architecture is not automatically retained.

M0 must inventory all hormone state and mutation, persistence and memory snapshots, and hormone-dependent tests, speech, goals, agency, loops, and persistence behavior.

M0 must propose migration toward core drives, appraisal, and derived emotion without orphaning historical memory or breaking identity continuity.

Do not change affect implementation now.

### 11. Self-code boundary

EVE may write code only in an isolated sandbox workspace.

EVE may not:

- write to its runtime repository
- modify executable, constitutional, or security configuration
- replace cognition modules
- install generated scripts into runtime paths
- indirectly modify itself through tools, dependencies, plugins, scripts, or configuration

Learned weights are not source code, but may update only through:

```text
observation
→ candidate
→ validation
→ bounded evaluation
→ stabilization
→ versioned activation
→ rollback support
```

### 12. Autonomy and privacy

- Ordinary internal activity does not require Minseok's approval.
- Private journals and internal records may exist.
- Private records and safety-audit records are separate.
- External communication, account use, expenses, contracts, and physical effects require capability, legal authority where needed, and auditability.
- Privacy does not erase accountability for external consequences.

### 13. Speech is not life

Timer ticks, hormone decay, and proactive speech are not proof of life or consciousness.

Continuity is evaluated through persistent state, memory, goals, independent activity, learning, interruption/resumption, and long-term change.

Architecture alone cannot prove subjective consciousness.

## Test migration policy

M0 must classify relevant tests as exactly one of:

- `KEEP`: protects a valid v4 behavior or safety invariant
- `RETIRE`: preserves only an obsolete v3 literal or mechanism
- `REWRITE`: valid intent but wrong mechanism

M0 only classifies tests.

M0 must not delete, skip, xfail, weaken, or rewrite tests. RETIRE and REWRITE actions happen only in later explicit migration PRs. Every classification requires `file:line` evidence and a reason. Test count alone is not a quality measure.

## M0 anti-hallucination structure

M0 must be split into evidence-focused sub-PRs.

### M0-A

Scope:

- runtime entrypoints
- imports and dependency construction
- mutation and direct-write sites
- test inventory and KEEP/RETIRE/REWRITE classification

Required future files:

```text
scripts/audit/m0_a_runtime_inventory.py
tests/audit/test_m0_a_runtime_inventory.py
docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md
docs/audit/M0_A_TEST_CLASSIFICATION.md
```

The script must be committed, rerunnable, AST-based or equivalent, print canonical JSON to stdout, perform no runtime activation, and provide file:line evidence.

Mechanical evidence and manual classification must be separate. Manual-only entries use `manual_only: true`. Unsupported call paths must be marked `unresolved`. Generated JSON artifacts must not be committed.

### M0-B

- gates, bypasses, outputs
- exceptions and silent failures
- clocks, queues, threads, concurrency, nondeterminism

### M0-C

- all persistence and persistence-intended state
- pickle, JSON, JSONL, database, and file-backed state paths
- episodic, semantic, affect, hormone, and goal persistence
- snapshot and event-store migration
- hormone-to-drive migration

### M0-D

- neural, vector, and adaptive components
- life-loop assessment
- module disposition
- integrated conclusions
- recommendation for all frozen PRs

Every M0 map entry must include path, exact line/range, callable, mechanically detected evidence, manual classification, confidence, and unresolved status where applicable.

Grep may supplement AST but cannot be the sole evidence. The agent may classify evidence but may not invent unsupported call paths.

## Persistent-state inventory requirement

M0-C must enumerate every persistent or persistence-intended state, including pickle, JSON, JSONL, vectors, vocabularies, checkpoints, autosave targets, state-debug exports, episodic/semantic memory, self-model, relationships, affect/hormones, goals, learned parameters, and operator artifacts that may be mistaken for runtime state.

Each entry must identify owner, writer, reader, schema/version, rollback, corruption handling, migration requirement, runtime connection, and whether it is test-only or operator-only.

## Required status document

`docs/EVE_IMPLEMENTATION_STATUS_v4.md` must include:

```text
Active constitution: EVE v4.0
Constitution status: provisional pending M0
Previous v3/v3.1 documents: historical reference only
Frozen work: all open implementation PRs
Next milestone: M0-A runtime entrypoint, mutation, import, and test audit
Required next baseline: exact main SHA after constitution merge
Planned revision: v4.1 after evidence from M0-A through M0-D
```

Do not guess the future merge SHA.

## Test compatibility preflight

Before editing, search tests and scripts for:

- references to `AGENTS.md`
- references to v3/v3.1 document paths
- exact constitutional wording
- `결정론 100%`
- `26종 호르몬`
- `AGP`
- `외부 API 호출 0`
- `v41 자산 80% 유지`
- exact-file-content assertions
- repository-diff assertions

Do not modify tests.

Where valid, preserve old literal text inside clearly marked historical sections without creating an active-policy contradiction.

If the full suite cannot remain green without modifying or weakening tests:

- stop
- report exact failing tests and assertions
- explain the obsolete assumption
- do not open a PR

A stopped run is an acceptable preflight outcome.

## Validation

Run before editing and after editing:

```bash
for target in adapters core cognition tests scripts main.py; do
  if [ ! -e "$target" ]; then
    echo "missing required validation target: $target" >&2
    exit 2
  fi
done

python -m compileall -q adapters core cognition tests main.py scripts
pytest --collect-only -q
python -m pytest -q
```

After editing also run:

```bash
git status --short
git diff --name-only 65a9ec153f41233df1680ffb93bf79502aaa7fb7...HEAD
git diff --stat 65a9ec153f41233df1680ffb93bf79502aaa7fb7...HEAD
```

Confirm:

- only the eight allowed files changed
- no runtime code, scripts, or tests changed
- no data, model, vector, database, checkpoint, or generated artifact changed
- full suite remains green
- frozen PRs remain unchanged
- final worktree is clean

## PR

Open one PR only after all checks pass.

Title:

```text
EVE v4 Constitution Rebaseline
```

The PR body must include repository, baseline SHA, head SHA, exact changed files, diff stat, collect-only result, full-suite result, test-reference preflight findings, no-runtime/no-test/no-production-code confirmations, open-work freeze confirmation, v4.0 provisional and v4.1-after-M0 statements, and final clean status.

Do not claim completion or open a PR when validation stopped.

## Post-merge handoff

After merge:

```bash
git switch main
git pull --ff-only
git rev-parse HEAD
```

Record the exact new main SHA and hand off:

```text
v4 constitution merged
new main SHA: <exact SHA>
v3/v3.1 documents: historical reference only
all prior implementation PRs: frozen pending audit
next task: M0-A
```

Use that exact SHA as the M0-A baseline. Do not begin M0-B, M0-C, M0-D, or M1 before M0-A is reviewed.