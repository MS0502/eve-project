# ROUND_V3_R74_REPORT — explicit commit drift/telemetry delta

## Summary

Round74 compares the EveSpecific routing surface before and after the first real explicit commit path introduced in Round73.

This round does **not** change self-learning policy. It only exports a compact delta report around the existing explicit commit smoke.

```text
Round73:
observe_text -> commit_eve_specific_vectors -> EveSpecificVectorStore -> EmbeddingWrapper lookup

Round74:
baseline before -> pre-commit target lookup -> explicit commit -> post-commit target lookup -> baseline after -> delta report
```

## New artifact

```text
EVE_SPECIFIC_COMMIT_DELTA_R74.json
```

## New test

```text
tests/test_v3_round74_explicit_commit_delta_report.py
```

## Code changes

```text
adapters/runtime_smoke_runner.py
  + run_round74_explicit_commit_delta_report()
  + helper telemetry/store/route delta utilities

adapters/embedding_wrapper.py
  - removed one unreachable duplicate `return result` in get_embedding()
```

The wrapper cleanup is behavior-preserving. It only removes unreachable code after a successful fastText get_embedding hit.

## Delta result

Default target:

```text
target_word = 민석
observation_texts = 민석 오늘 / 민석 군대
commit_context_words = 오늘 / 군대
```

Measured result:

```text
store_delta = 1
audit_record_delta = 2
commit_created_target = True
wrapper_vector_found_after_commit = True
```

Target lookup route separation:

```text
before commit:
  eve_specific_hit_delta = 0
  fallback_delta = 1

after commit:
  eve_specific_hit_delta = 1
  fallback_delta = 0
```

Telemetry delta:

```text
total_calls +2
primary_hits +0
fallback_uses +1
eve_specific_hits +1
errors +0
```

Route distribution delta:

```text
total_calls +2
routed_hits_total +2
fastText primary hits +0
EveSpecific hits +1
PMI+SVD fallback uses +1
errors +0
```

## Policy preserved

```text
auto_observe_enabled = True
auto_promotion_enabled = False
commit_gate_enabled = True
min_observations_for_commit = 2
context_diversity_gate_enabled = True
automatic_rollback_enabled = False
thresholds_changed = False
context_diversity_gate_changed = False
drift_based_runtime_change = False
memory/quarantine mutation = False
fastText seed mutation = False
AGP bypass = False
```

## Validation

Focused:

```text
tests/test_v3_round74_explicit_commit_delta_report.py
4 passed
```

Adjacent focused sweep:

```text
tests/test_v3_round5*.py tests/test_v3_round6*.py tests/test_v3_round7*.py
166 passed
```

Full split-suite:

```text
collect-only: 1164 tests collected
split suite: 7/7 chunks passed
passed tests by chunk sum: 1164
failures: 0
timeouts: 0
compileall: passed
```

Full split-suite status is recorded in `ROUND74_SPLIT_SUITE_STATUS.json`.

## Next recommendation

Round75: commit audit replay/export consolidation.

Goal:

```text
- export the Round73/74 explicit commit smoke + delta path as replayable audit data
- verify replay is read-only
- keep gate/threshold/context-diversity policies unchanged
- no automatic promotion or rollback
```
