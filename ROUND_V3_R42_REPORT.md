# ROUND_V3_R42_REPORT

## Summary

v3 round42 completed wrapper telemetry and Appendix-D seed drift baseline setup.

```text
baseline: v3 round41, 903 passed
result:   v3 round42, 917 passed
compileall: passed
migration_progress: 7/7 + telemetry tracking active
```

## Files changed

- `adapters/embedding_wrapper.py`
- `adapters/external_seed_manifest.py`
- `adapters/state_debug_adapter.py`
- `tests/test_v3_round42_wrapper_telemetry_and_drift.py` NEW
- `CURRENT_STATUS.md`
- `AGENTS.md`
- `ROUND_V3_R42_REPORT.md` NEW

## Implementation

### EmbeddingWrapper telemetry

`EmbeddingWrapper.telemetry()` now returns a read-only data dict:

- `total_calls`
- `primary_hits`
- `fallback_uses`
- `errors`
- `primary_hit_rate`
- `fallback_rate`
- `error_rate`
- `oov_log_size`
- `oov_log_cap`
- `recent_oov_sample`

OOV samples are capped at 1000 entries.

### Seed drift baseline

`measure_seed_drift_baseline(engine)` records the first post-swap Appendix-D baseline:

```text
baseline_round = 42
fasttext_vector_unchanged = True
evespecific_learning_substrate = PMI+SVD (fallback path)
target = round 200+ avg drift > 0.3 (Draft, v3.1 precise threshold pending)
```

### AGP + wrapper correlation

`correlate_agp_and_wrapper_telemetry(engine)` provides a read-only correlation surface between AGP traces and wrapper telemetry.

It returns a data dict only. It does not recommend or apply threshold changes.

## State debug

`state_debug_adapter` now reports:

- `wrapper_telemetry`
- `seed_drift`

Both are read-only debug surfaces.

## Preserved invariants

- No PMI+SVD fallback removal.
- No wrapper threshold change.
- No AGP threshold auto-adjustment.
- No drift-based runtime change.
- No new subset extraction.
- No `self_embedding_adapter.py` rewrite.
- No memory/quarantine data file edits.
- No external fastText runtime package import.

## Validation

```text
917 passed
compileall passed
```

## Next

v3 round43 should perform runtime smoke / telemetry sampling:

- Run representative Korean user inputs.
- Record primary hit rate, fallback rate, OOV samples, and AGP pass behavior.
- Keep telemetry read-only.
- Do not remove fallback or tune thresholds based on a single early sample.
