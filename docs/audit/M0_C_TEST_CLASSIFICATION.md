# M0-C Test Classification

Baseline: `main` at `eea70c286e947cbc180db9565bfa5ddc062d1ac3`

M0-C classifies persistence and state tests without deleting, skipping, xfail-marking, weakening, or rewriting any test.

Run the exact file-level inventory with:

```bash
python scripts/audit/m0_c_persistence_state_inventory.py --pretty
```

Generated JSON is an ephemeral audit artifact and must not be committed.

## KEEP

`KEEP` is the conservative default. Persistence tests remain executable evidence even when they assert behavior that is expected to change in a later migration.

This includes tests for:

- pickle, JSON, JSONL, sqlite, vectors, vocabularies, checkpoints, sidecars, and debug exports;
- autosave and explicit operator save/load paths;
- episodic and semantic memory state;
- self-model and relationship state;
- affect/hormone, goal, need, and drive state;
- learned parameters, vectors, and vocabulary persistence;
- checkpoint validation, rollback, and operator evidence artifacts.

`KEEP` does not endorse the format or architecture. It preserves migration, security, compatibility, and regression evidence.

## Legacy-format evidence

A test file that directly references pickle, sidecar, sqlite, `.ckpt`, `.pickle`, or `.pkl` remains `KEEP`, but the generated classification is medium-confidence and unresolved. The classification reason records that later manual review must decide whether the test remains unchanged or requires a behavior-preserving rewrite after the target persistence contract is approved.

M0-C does not automatically classify legacy-format tests as `REWRITE`; the replacement contract does not yet exist.

## REWRITE

M0-C mechanically classifies no test as `REWRITE` by default.

A later rewrite requires exact file:line evidence that the assertion is coupled to a superseded persistence representation rather than the behavior or safety property. Any rewrite must preserve the underlying assertion, such as atomicity, validation-before-mutation, rollback integrity, no-default-activation, deterministic export, or round-trip fidelity.

## RETIRE

M0-C retires no test by default. Retirement requires exact file:line evidence that the test has no behavioral, migration, historical, security, compatibility, or regression value.

## Required fields

Every test classification records:

- repository-relative path;
- exact evidence line;
- `KEEP`, `REWRITE`, or `RETIRE`;
- mechanical evidence and detector;
- classification reason;
- confidence;
- unresolved state;
- manual-only state.

## Limits

The file-level classifier records the first persistence-format signal. It does not infer transitive use through fixtures, imported helpers, generated paths, or dynamically chosen formats. It also does not decide whether a current test belongs to checkpoint, debug-export, cache, or authoritative-state semantics.

M0-C changes no test behavior. Measured classification counts will be added only after independent validation against the unchanged branch head.
