# EVE v3 Round40 Report — pre-swap audit

## Result

- Previous stable: v3 round39 (`865 passed`)
- Current stable: v3 round40 (`888 passed`)
- Added tests: `+23`
- `compileall`: passed
- Regression: `0`

## Scope

Round40 is the final safety audit before the round41 `main.py` / `engine.self_embedding` swap.

This round does not swap the active embedding owner. It validates that the six migrated surfaces can coexist around the current `SelfEmbeddingAdapter(PMI+SVD, 50d)` path while `FasttextEmbeddingAdapter` remains an explicit, unloaded-by-default parallel observation adapter.

## Files changed

- `adapters/external_seed_manifest.py`
  - Adds `assess_main_py_swap_readiness(engine) -> dict`.
  - Adds round40 audit constants for migrated surfaces and trace modules.
  - The readiness function is read-only and does not call `fasttext_embedding.load()`.
- `tests/test_v3_round40_pre_swap_audit.py`
  - Adds 23 audit tests.
  - Covers invariant checks, simultaneous trace behavior, fail-open safety, three-system coexistence, and swap readiness.
- `CURRENT_STATUS.md`
- `AGENTS.md`
- `ROUND_V3_R40_REPORT.md`

## Audit areas verified

### 1. Six-surface invariant

```text
state_debug_adapter = read-only debug exposure
attention_analyzer = parallel observation only
compositor_adapter = parallel observation only; AGP independent
concept_memory_adapter = parallel observation only; semantic decisions unchanged
situation_responder = parallel observation only; user-visible behavior unchanged
language/streaming = parallel observation only; chunk output/order unchanged
```

Verified:

- All six migration surfaces are present.
- Trace modules report `in_use_by_generation = "self_embedding"`.
- Trace modules report `fasttext_trace_cap = 1000`.
- Default fastText state remains unloaded.

### 2. Simultaneous operation

Verified:

- Explicit fastText load activates all trace modules.
- Traces are independent data structures.
- Trace records remain dictionary-shaped diagnostics.
- Decisions remain unchanged across attention, compositor, concept memory, situation responder, and streaming.

### 3. Fail-open safety

Verified:

- FastText load failure does not break modules.
- OOV/unknown lookup behavior does not break decisions.
- Exceptions during parallel observation record error traces and continue.
- Corrupted fastText adapter behavior does not propagate into the self_embedding path.
- Runtime unload stops further parallel observation gracefully while decisions continue.

### 4. AGP + fastText + self_embedding coexistence

Verified:

- AGP veto and fastText observation can coexist.
- AGP trace and fastText trace stay separate.
- `engine.self_embedding` remains `SelfEmbeddingAdapter(PMI+SVD, 50d)`.
- FastText observation does not change AGP mode, fallback behavior, or the active generation embedding source.

### 5. Round41 readiness

`assess_main_py_swap_readiness(engine)` returns:

```text
swap_readiness = ready
all_6_modules_migrated = True
fasttext_adapter_ready = True
self_embedding_is_pmi_svd_50d = True
concerns = []
read_only = True
```

The function also records:

```text
swap_strategy = round41: replace engine.self_embedding wiring only after audit passes
rollback_strategy = restore engine.self_embedding = SelfEmbeddingAdapter(engine); keep fasttext unloaded
main_py_affected_areas = engine init, self_embedding wiring, fasttext_embedding rollback boundary
```

## Non-goals preserved

- No `engine.self_embedding` swap.
- No `main.py` migration.
- No default fastText load.
- No `self_embedding_adapter.py` rewrite.
- No active fastText use in generation or decisions.
- No streaming chunk output/order/timing change.
- No situation response selection/output change.
- No concept-memory query/write/quarantine change.
- No semantic guard keyword addition.
- No memory/quarantine data file modification.

## Current migration state

```text
migration_progress = 6/7 + audit complete
next = round41 main.py / engine.self_embedding final swap
post-swap = round42+ AGP drift measurement
```

## Recommendation

Proceed to round41 only as a targeted final swap round:

1. Preserve rollback by keeping the old `SelfEmbeddingAdapter` path restorable.
2. Rewire `engine.self_embedding` deliberately.
3. Re-run all round35~40 invariants after swap.
4. Keep fastText unload behavior safe.
5. Do not start AGP drift measurement until round41 passes.
