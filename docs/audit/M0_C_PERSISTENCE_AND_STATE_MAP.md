# M0-C Persistence and State Map

Baseline: `main` at `eea70c286e947cbc180db9565bfa5ddc062d1ac3`

Status: evidence-only audit. M0-C does not activate persistence, migrate state, change formats, alter defaults, write runtime artifacts, or modify tests.

## Reproducible inventory

Run:

```bash
python scripts/audit/m0_c_persistence_state_inventory.py --pretty
```

The command scans tracked Python files with `ast.parse` and emits canonical JSON to stdout. `--output` is permitted only for an ephemeral validation artifact. Generated JSON must not be committed.

Every entry records repository-relative path, exact line range, enclosing callable, mechanically detected evidence, detector, manual classification, confidence, unresolved state, and whether the entry is manual-only.

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

## Required state domains

The scanner mechanically inventories these M0-C domains:

- episodic memory;
- semantic/concept memory;
- self-model, identity, and preferences;
- relationships, familiarity, trust, and user presence;
- affect, emotion, mood, and hormones;
- goals, intentions, and plans;
- learned parameters, statistics, and model state;
- vectors, embeddings, and vector stores;
- vocabularies, lexicons, and token maps;
- checkpoints and rollback state;
- autosave targets;
- debug/state exports;
- operator, validation, audit, and report artifacts.

A symbol match is persistence-intended-state evidence, not proof that the current object is actually serialized or restored. Reachability and ownership remain unresolved until call-path review.

## Persistence formats and operations

The inventory distinguishes:

- pickle and compressed pickle;
- JSON, compressed JSON, and JSONL;
- sqlite and generic database paths;
- NumPy arrays and archives;
- torch checkpoints;
- safetensors;
- joblib;
- YAML;
- CSV/TSV;
- gzip, bz2, and lzma containers;
- generic checkpoint and artifact targets.

Operations are classified as read, write, read-write, serialize, deserialize, copy, replace, delete, create, or container access where mechanical evidence permits.

## Hormone-to-drive migration inventory

M0-C does not perform the migration. It records three evidence layers:

1. legacy hormone/affect state sites;
2. drive/need state sites;
3. callable scopes that reference both families and therefore may translate, modulate, aggregate, or couple them.

A bridge candidate is not automatically a correct migration boundary. Later design work must determine whether the relation is representation replacement, compatibility projection, read-only observation, proposal generation, or prohibited live mutation.

## Initial high-impact review targets

The measured report must be used to confirm and document at least these active surfaces:

- `adapters/persistence_adapter.py` save/load and sidecar behavior;
- `adapters/live_loop.py` autosave target and autosave call path;
- `main.py` default checkpoint target and operator `/save`/`/load` commands;
- episodic and semantic memory stores;
- self-state and integrated-self surfaces;
- relationship/user-presence state;
- hormone/affect and goal/need/drive state;
- vector and vocabulary stores;
- runtime mapping checkpoints, rollback, debug export, and operator evidence artifacts.

## Interpretation rules

- A serialization call does not prove durable persistence unless its surrounding path and lifecycle are confirmed.
- A path literal does not prove the artifact is created in the active runtime.
- A state-domain symbol does not prove that state is included in a checkpoint.
- A load call does not prove restored data is validated before mutation.
- A debug export may contain persistence-intended state without being an authoritative checkpoint.
- Pickle evidence is retained as a security, compatibility, and migration concern; M0-C does not replace it.
- Vector, vocabulary, and learned-parameter files are state even when they are described as caches or model assets.
- Autosave and operator save/load paths must be assessed separately.
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

Measured counts and exact high-impact findings will be added only after independent validation against the unchanged branch head.
