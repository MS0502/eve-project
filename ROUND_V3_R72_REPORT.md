# EVE v3 Round72 Report — Smoke / EveSpecific Drift Baseline Remeasurement

## Scope

Round72 is a measurement and route-clarification round. It does not change the active self-learning policy.

Goals:

1. Clarify `EmbeddingWrapper.get_embedding()` routing so the local PMI+SVD map precheck is not confused with a duplicate fallback API call.
2. Re-run smoke and EveSpecific drift/route baseline after the Round71 consolidation baseline.
3. Preserve read-only snapshots for operator review before the first explicit EveSpecific commit smoke.

## Changes

Added:

```text
+ tests/test_v3_round72_smoke_drift_baseline.py
+ ROUND_V3_R72_REPORT.md
+ EVE_SPECIFIC_BASELINE_R72.json
+ ROUND72_SPLIT_SUITE_STATUS.json
+ ROUND72_SPLIT_SUITE_BY_CHUNK_RESULTS.json
```

Modified:

```text
- adapters/embedding_wrapper.py
- adapters/external_seed_manifest.py
- adapters/runtime_smoke_runner.py
- CURRENT_STATUS.md
- AGENTS.md
```

## EmbeddingWrapper route clarification

`get_vector()` route remains:

```text
fastText.get_vector
→ EveSpecificVectorStore.get_vector
→ PMI+SVD.get_embedding
```

`get_embedding()` route is now explicitly documented and surfaced through `lookup_route_policy_snapshot()`:

```text
PMI+SVD.local_embeddings_map_precheck
→ fastText.get_embedding
→ EveSpecificVectorStore.get_vector
→ PMI+SVD.get_embedding_final_fallback_once
```

Important clarification:

```text
PMI+SVD.local_embeddings_map_precheck does not call fallback.get_embedding().
The fallback API boundary is called at most once per get_embedding() call.
```

This addresses the Round71 follow-up concern about apparent PMI+SVD duplication. Routing priority is not changed; the distinction is now explicit and tested.

## Smoke / drift baseline

Snapshot artifact:

```text
EVE_SPECIFIC_BASELINE_R72.json
```

Baseline summary from the probe smoke run:

```text
total_calls: 48
fastText primary hits: 37
EveSpecific hits: 2
PMI+SVD fallback uses: 9
errors: 0
fastText primary rate: 0.7708333333333334
EveSpecific rate: 0.041666666666666664
PMI+SVD fallback rate: 0.1875
```

The EveSpecific probe vectors were added directly to the isolated smoke engine's vector store only to measure routing. The self-learning commit path was not called.

```text
probe_vector_mutation_scope = isolated_smoke_engine_only_not_self_learning_commit_path
self_learning_commit_path_called = False
```

## Policy preservation

Round72 does not change active learning policy:

```text
auto_observe_enabled = True
auto_promotion_enabled = False
commit_gate_enabled = True
min_observations_for_commit = 2
context_diversity_gate_enabled = True
automatic_rollback_enabled = False
routing_changed_in_round72 = False
thresholds_changed = False
context_diversity_gate_changed = False
AGP bypass = False
memory/quarantine mutation = False
fastText seed mutation = False
drift-based runtime change = False
```

## Validation

Single `pytest tests` is not used as the source of truth in this sandbox because the process can exceed the 300s execution window. Round72 uses the same split-run strategy established in Round71.

```text
Round72 focused: included in split chunk 10 and passed
collect-only: 1157 tests collected
split suite: 121/121 test files passed by chunk
passed tests by chunk sum: 1157
failures: 0
timeouts: 0
compileall: passed
```

Artifacts:

```text
ROUND72_SPLIT_SUITE_STATUS.json
ROUND72_SPLIT_SUITE_BY_CHUNK_RESULTS.json
```

## Result

Round72 is safe as a measurement baseline. It clarifies lookup route semantics, remeasures the EveSpecific route distribution, and preserves the Round71 safety policy.

## Recommended Round73

First explicit EveSpecific commit smoke.

Scope:

```text
- use EveSelfLearningAdapter explicit commit path, not direct vector-store probe
- observe candidate across 2 context-diverse inputs
- verify gate pass and vector-store mutation only after explicit commit
- verify same-context repeated candidate remains blocked
- remeasure route distribution after commit
- no auto-promotion
```
