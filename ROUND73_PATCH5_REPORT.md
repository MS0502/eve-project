# ROUND73 PATCH5 REPORT

## Goal

Patch4 kept the full suite fast by removing eager SelfEmbedding bootstrap from `build_full_engine()`. Patch5 closes the resulting functional gap: SelfEmbedding should remain cheap during engine construction, but still become usable when an embedding consumer actually needs it.

## Changes

- Added `SelfEmbeddingAdapter.ensure_ready()`.
  - Does not run during `build_full_engine()`.
  - Trains immediately from already observed local data when enough tokens exist.
  - Falls back to one-shot `bootstrap_from_engine()` only when local data is insufficient.
  - Prevents repeated failed bootstrap/SVD attempts via `_bootstrap_attempted`.
- Added `bootstrap_attempted` to SelfEmbedding stats for auditability.
- Added ConceptMemory → SelfEmbedding observation bridge.
  - New concept definitions are buffered into SelfEmbedding.
  - Auto-train is suspended during this bridge, so learning concepts does not secretly trigger SVD in hot paths.
- Added guarded Compositor lazy preparation.
  - Empty embeddings do not trigger SVD during normal chat.
  - Lazy preparation is allowed only after concept-seeded data exists and the requested inputs are in that observed vocabulary.
- Added regression tests for lazy embedding behavior.

## Validation

Commands run:

```bash
pytest -q
python3 -m compileall -q .
```

Result:

```text
544 passed in 4.65s
compileall passed
```

## Safety notes

- No test expectations weakened.
- No semantic memory files modified.
- No LLM calls added.
- No random behavior added.
- No broad refactor.
