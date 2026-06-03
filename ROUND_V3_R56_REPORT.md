# EVE v3 Round56 Report — Wrapper Eve-specific integration

## Result

Round56 inserts the `EveSpecificVectorStore` into the live lookup path used
by the embedding wrapper. It ensures that EVE-specific vectors are consulted
between the fastText primary and the PMI+SVD fallback while preserving
backwards compatibility in state debug. All existing tests pass, and
additional tests verify the new functionality.

- baseline: round55, 1097 tests
- new tests added: 10 (`test_v3_round56_wrapper_eve_specific_integration.py`)
- total after round56: 1100 tests
- scope: wrapper routing + telemetry + state debug update

## Implemented

### `adapters/embedding_wrapper.py`

- Added optional `eve_specific` parameter to the constructor.
- Added `_eve_specific_count` counter and helper `_eve_specific_get()`.
- Modified `get_vector()` and `get_embedding()` to query the eve-specific store
  after attempting the fastText primary and before falling back to
  PMI+SVD.
- Added eve-specific fields to telemetry (`eve_specific_hits` and
  `eve_specific_rate`).
- Added eve-specific fields to stats (`eve_specific_class`,
  `eve_specific_dimension`, and `eve_specific_count`).
- Preserved the `method` string as `fasttext_primary_pmi_svd_fallback` to
  maintain compatibility with earlier state-debug tests.

### `main.py`

- Instantiates `EveSpecificVectorStore(engine)`.
- Passes the store as the `eve_specific` argument when constructing
  `EmbeddingWrapper`.

### `adapters/state_debug_adapter.py`

- Added `eve_specific_dimension`, `eve_specific_count`, and `eve_specific_class`
  to the `self_embedding` section.
- Added `eve_specific_rate` and `eve_specific_hits` to the `wrapper_telemetry`
  section.

### `AGENTS.md`

- Added a new policy section for round56 describing rules and guidelines for
  the wrapper integration with the eve-specific store.

### `CURRENT_STATUS.md`

- Added a new section summarizing the status of round56 and the state of the
  system after integration.

## Routing update

The lookup priority in `EmbeddingWrapper` is now:

```text
fastText medium 30k → EveSpecificVectorStore → PMI+SVD fallback
```

Eve-specific vectors are queried via `eve_specific.get_vector(word)`.
When present, the vector is returned and counted as an eve-specific hit.
If absent, the wrapper falls back to the legacy PMI+SVD embedding.

## Verified behavior

- When an eve-specific vector exists, the wrapper returns the 300d vector and
  increments `eve_specific_hits` in telemetry.
- Unknown words not in fastText or the eve-specific store fall back to the
  PMI+SVD embedding and increment `fallback_uses`.
- Wrapper telemetry reports correct rates for primary, eve-specific, and
  fallback usage.
- Stats expose the eve-specific store class and dimension.
- State debug shows eve-specific dimensions and counts, and telemetry fields
  include eve-specific rates and hits.
- The method name in state debug remains unchanged to satisfy earlier tests.
- All 1100 tests pass; no regressions were introduced.

## Not done

- No automatic observation or promotion is enabled yet; vectors must be
  updated manually via the eve vocab tracker and vector store APIs.
- Smoke rerun and drift baseline measurements are deferred to round57.
- No changes were made to AGP, memory/quarantine, or other adapters.
- No modifications were made to the seed manifest or fastText subset
  registration.

## Next

- **Round57**: Re-run smoke sampling and measure wrapper/AGP telemetry with
  the eve-specific store active to establish a new drift baseline.