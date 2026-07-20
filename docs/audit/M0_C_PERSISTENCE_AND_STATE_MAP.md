# M0-C Persistence and State Map

Baseline: `main` at `eea70c286e947cbc180db9565bfa5ddc062d1ac3`

Status: evidence-only audit. M0-C does not activate persistence, migrate state, change formats, alter defaults, write runtime artifacts, or modify tests.

## Reproducible inventory

Run:

```bash
python scripts/audit/m0_c_persistence_state_inventory.py --pretty
```

The command scans tracked Python files with `ast.parse` and emits canonical JSON to stdout. `--output` is permitted only for an ephemeral validation artifact. Generated JSON must not be committed.

The audit path and source-content universe is frozen to the merged M0-C snapshot `fe10cd954bdf445400ea6aa9708dd214ed761114`. Later Python additions, deletions, or edits cannot retroactively change this completed audit's canonical output. Fixture repositories without that snapshot continue to use their local files for focused tests.

Every entry records repository-relative path, exact line range, enclosing callable, mechanically detected evidence, detector, manual classification, confidence, unresolved state, and whether the entry is manual-only.

## Validated inventory snapshot

Validated after path, source-file, embedded-fixture, and generic sqlite-string false-positive removal:

```text
tracked Python files: 582
total occurrence entries: 7,669
persistence I/O occurrences: 856
artifact path occurrences: 519
persistence-intended state occurrences: 3,845
legacy hormone/affect state occurrences: 1,777
drive/need state occurrences: 386
hormone-to-drive bridge candidates: 54
test files: 230 KEEP / 0 REWRITE / 0 RETIRE
unresolved entries: 7,440
parse errors: 2
```

Counts are occurrence-level mechanical evidence. They are not counts of distinct authoritative state objects, active persistence channels, or architectural defects. Active runtime code, legacy snapshots, operator scripts, validation utilities, tests, and dry-run surfaces remain in the tracked Python scan.

The two tracked parse failures remain explicit unresolved evidence:

- `eve_foundation_v10_2.py:11557` — `'[' was never closed`;
- `eve_foundation_v12_0.py:11542` — `'[' was never closed`.

M0-C neither excludes nor repairs those legacy snapshots.

## Audit categories

| Category | Mechanical scope | Default classification |
|---|---|---|
| `persistence_io` | serialization, deserialization, file open modes, databases, checkpoint/vector formats, filesystem replacement/copy, and persistence-like save/load methods | persistence read, write, mutation, or serialization candidate |
| `artifact_path` | string literals naming checkpoint, autosave, debug, database, vector, JSON/JSONL, pickle, compressed, or operator artifact paths | `PERSISTENCE_ARTIFACT_PATH_CANDIDATE` |
| `state_domain` | assignments, dictionary keys, classes, and persistence method symbols associated with required state domains | `PERSISTENCE_INTENDED_STATE_CANDIDATE` |
| `hormone_state` | hormone, affect, emotion, mood, and named hormone state symbols | `LEGACY_HORMONE_OR_AFFECT_STATE_SITE` |
| `drive_state` | drive, need, desire, urge, motivation, homeostasis, allostasis, or priority state symbols | `DRIVE_OR_NEED_STATE_SITE` |
| `hormone_drive_bridge` | callable scopes containing both hormone/affect and drive/need symbols | `HORMONE_TO_DRIVE_MIGRATION_CANDIDATE` |
| `parse_error` | tracked Python that cannot be parsed | `UNRESOLVED_PARSE_ERROR` |

Artifact-path detection accepts recognized persistence suffixes directly. Directory-like strings require persistence markers such as state, checkpoint, vector, vocabulary, seed, operator artifact, debug export, or validation artifact. Source files, Markdown, shell files, pytest node IDs, URLs, commands, root-only strings, and unrelated media paths are excluded.

## State-domain distribution

```text
affect / hormones: 1,534
vectors / embeddings: 1,175
checkpoints / rollback state: 278
self-model / identity / preferences: 267
semantic / concept memory: 155
vocabularies / lexicons: 112
relationships / familiarity / trust: 88
goals / intentions / plans: 65
operator / validation artifacts: 60
episodic memory: 43
debug / state exports: 39
learned parameters / statistics: 17
autosave: 12
```

A symbol match is persistence-intended-state evidence, not proof that the current object is serialized, authoritative, restored, validated, or reachable from the active runtime.

## Persistence formats and operations

Validated format evidence:

```text
JSON: 544
text: 314
filesystem mutation/copy/replace: 149
NumPy: 118
path or persistence target: 104
file open: 43
bytes: 27
gzip: 17
checkpoint: 10
pickle: 10
JSONL: 8
YAML: 7
CSV: 4
pickle+gzip: 2
sqlite: 2
torch checkpoint: 2
JSON+gzip: 1
JSONL+gzip: 1
generic database: 1
joblib: 1
safetensors: 1
TSV: 1
```

Validated persistence-operation evidence:

```text
read: 234
write: 180
serialize: 187
deserialize: 106
create: 87
replace: 48
copy: 9
delete: 5
```

These are call/path occurrences. For example, `json.dumps` is serialization evidence but does not alone prove durable persistence.

## High-impact active and migration surfaces

### Runtime persistence adapter

`adapters/persistence_adapter.py` contains the clearest active persistence path:

- `PersistenceAdapter.save` calls legacy `Persistence.save` and writes a gzip/pickle sidecar;
- `pickle.dump` occurs at line 70;
- `_load_sidecar` recognizes `.v41sidecar.gz` and related `.gz` paths;
- `pickle.load` occurs at line 142;
- M0-A previously confirmed `save` at lines 54–80, `load` at 84–101, and sidecar restoration at 130–151.

This creates two persistence representations: legacy persistence plus a separate pickle sidecar. M0-C does not decide their replacement contract.

### Autosave and operator save/load

- `adapters/live_loop.py` contains autosave state and the active `_do_autosave` path confirmed by M0-A at lines 205–212;
- `main.py:419` configures `~/eve.ckpt` for the interactive runtime, with another occurrence at line 547;
- `main.py:718` invokes `persistence.save`;
- `main.py:724` invokes `persistence.load`.

Automatic autosave and explicit operator persistence are separate activation and migration boundaries.

### State debug and operator evidence

`adapters/state_debug_adapter.py` aggregates identity, integrated self, goals, concept memory, embedding state, affect, and additional diagnostics into a broad snapshot/export surface. Runtime-mapping dry-run and limited-sandbox modules separately define checkpoint, rollback, debug-export, vector, and operator-evidence artifacts. These surfaces are evidence/export contracts, not automatically authoritative runtime checkpoints.

### Memory, self, relationship, vector, and vocabulary state

The inventory finds persistence-intended symbols across episodic memory, concept/semantic memory, self-state, integrated-self views, user-presence and relationship state, goal state, learned statistics, vector stores, embedding stores, and vocabularies. Their high occurrence counts reflect distributed ownership and repeated schema/test evidence; they do not prove one coherent persistence envelope exists.

## Hormone-to-drive migration inventory

M0-C records:

```text
legacy hormone/affect sites: 1,777
drive/need sites: 386
callable bridge candidates: 54
```

High-impact bridge candidates include:

- `main.py:20-356` / `build_full_engine` — hormone, emotion, mood, allostatic, and urge composition;
- `adapters/live_loop.py:113-201` / `LiveLoop._run` — hormone updates and urge-driven proactive behavior;
- `adapters/allostatic_adapter.py:24-37` / `AllostaticAdapter.__init__` — hormone/allostatic coupling;
- `adapters/urge_adapter.py:45-93` / `UrgeAdapter.compute_urge` — dopamine/hormone inputs and urge output;
- `core/autonomous.py` — need detection, autonomous step, and tick bridges;
- `language/streaming.py` — chat, proactive, and sensory-processing bridges.

A bridge candidate is not automatically the correct migration boundary. Later design work must distinguish representation replacement, compatibility projection, read-only observation, proposal generation, and prohibited live mutation.

## Interpretation rules

- A serialization call does not prove durable persistence unless its surrounding path and lifecycle are confirmed.
- A path literal does not prove the artifact is created in the active runtime.
- A state-domain symbol does not prove that state is included in a checkpoint.
- A load call does not prove restored data is validated before mutation.
- A debug export may contain persistence-intended state without being an authoritative checkpoint.
- Pickle evidence is retained as a security, compatibility, and migration concern; M0-C does not replace it.
- Vector, vocabulary, and learned-parameter files are state even when described as caches or model assets.
- Autosave and operator save/load paths must be assessed separately.
- Runtime-mapping dry-run artifacts must not be confused with enabled production persistence.
- Parse failures are never silently excluded.

## Scope boundary

M0-C does not:

- enable persistence or change persistence defaults;
- change checkpoint, sidecar, database, vector, vocabulary, or debug-export formats;
- write, migrate, restore, or delete runtime state;
- alter autosave behavior or target paths;
- mutate memory, self-model, relationships, affect/hormones, goals, drives, learned parameters, vectors, or vocabularies;
- perform hormone-to-drive migration;
- delete, skip, xfail, weaken, or rewrite tests;
- commit generated JSON;
- decide M0-D module retirement or integrated architecture disposition.
