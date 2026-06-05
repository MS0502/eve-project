# EVE v3 Round201 Report — Expected Operator-Local Delta Report Schema

Round201 documents the expected operator-local delta report format emitted by `scripts/operator_remeasure_eve_self_learning.py`.

## Required top-level keys

- `version`
- `rounds_completed`
- `selected_cluster_id`
- `status`
- `success`
- `exit_code`
- `operator_validation_summary`
- `runtime_load_summary`
- `measurement`
- `expected_delta_report_schema`
- `broader_validation_delta`
- `blockers`

## Required measurement keys

- `target_word`
- `observation_texts`
- `context_words`
- `known_context_words`
- `observed_count`
- `is_eve_specific_candidate`
- `commit_gate_pass`
- `created_vectors`
- `rejected_words`
- `vector_store_count_before`
- `vector_store_count_after`
- `wrapper_eve_specific_hits_before`
- `wrapper_eve_specific_hits_after`
- `wrapper_primary_loaded`
- `in_memory_vector_mutation_attempted`
- `persistent_vector_artifacts_written`

## Expected operator-local green delta

- Operator validation succeeds with `--attempt-load`.
- Runtime medium30k load attaches to `engine.fasttext_embedding` through the existing guard.
- `민석` is observed at least twice in distinct Korean-first contexts.
- Commit gate passes with known medium30k context evidence.
- `created_vectors` contains `민석`.
- Vector-store count increases by at least 1 in memory only.
- Wrapper EVE-specific hit count increases by at least 1.
- Persistent vector artifacts remain unwritten.

## Expected Cloud-blocked delta

When real operator artifacts are unavailable, the expected status is blocked/partial with `operator_artifact_files_missing`, no engine build, and no measurement execution.
