# EVE v3 Rounds566–590 Report — no-load subset validation alignment

## Rounds completed

- **Round566–570:** Reproduced and consolidated the current focused baseline after proceeding from the latest available main/work tree.
- **Round571–575:** Classified failures by exact nodeid and root cause.
- **Round576–583:** Applied narrow test-policy updates for no-load runtime, honest missing-artifact reporting, and explicit operator-only vector loading.
- **Round584–590:** Ran focused and broader validation, recorded remaining taxonomy, and prepared the next recommendation.

## Baseline

User-provided Codespaces baseline:

```text
16 failed / 62 passed
```

Local latest-main/work baseline after PR #55 was already present in this container:

```text
41 failed / 37 passed
```

The local delta was caused by the newer no-load runtime line being present while additional legacy tests still assumed committed vector artifacts or default fastText load.

## Clusters analyzed

1. **round29_31_subset_artifact_expectations**
   - Legacy tests expected `seeds/subsets/**/vectors.npy` to be committed and loadable.
   - Current policy forbids committing/fabricating vectors and treats those files as operator-local artifacts.

2. **round34_fasttext_state_debug_no_load_alignment**
   - Legacy tests expected `build_full_engine()` to expose a loaded fastText adapter with populated vocab/vector shape.
   - Current policy requires state debug to expose the adapter honestly without calling `load()`.

3. **round51_medium_primary_no_load_alignment**
   - Legacy tests expected medium30k to be loaded by default with `vocab_size == 30000` and real vector lookup success.
   - Current policy keeps medium30k selected as the default subset/primary metadata only; actual vector loading requires explicit operator authorization and real artifacts.

4. **round32 readiness adjacency**
   - One full-suite adjacent test still expected readiness `ready` despite missing vectors.
   - Current policy requires `needs_more_audit` until the operator-local vector matrix is present and checksum-valid.

## Exact failing nodeids addressed

Local baseline addressed these failing nodeid families:

- `tests/test_v3_round29_subset_extraction_mini_1k.py::test_subset_directory_and_files_exist`
- `tests/test_v3_round29_subset_extraction_mini_1k.py::test_vectors_shape_dtype_and_checksum`
- `tests/test_v3_round30_subset_validation_audit.py::test_subset_checksums_still_match_manifest`
- `tests/test_v3_round30_subset_validation_audit.py::test_subset_shape_vocab_and_corruption_invariants`
- `tests/test_v3_round30_subset_validation_audit.py::test_runtime_not_using_subset_and_no_fasttext_import`
- `tests/test_v3_round30_subset_validation_audit.py::test_subset_audit_is_read_only_for_manifest_and_engine`
- `tests/test_v3_round30_subset_validation_audit.py::test_self_embedding_rewrite_readiness_assessment_returns_data_dict`
- `tests/test_v3_round31_subset_extraction_small_5k.py::test_small_subset_directory_and_files_exist`
- `tests/test_v3_round31_subset_extraction_small_5k.py::test_small_vectors_shape_dtype_and_checksum`
- `tests/test_v3_round31_subset_extraction_small_5k.py::test_small_subset_audit_valid_and_distinct_from_mini_fixture`
- `tests/test_v3_round31_subset_extraction_small_5k.py::test_readiness_now_prefers_medium_expanded_subset_without_auto_apply`
- `tests/test_v3_round32_self_embedding_rewrite_scaffold.py::test_readiness_acknowledges_scaffold_without_auto_apply`
- `tests/test_v3_round33_fasttext_load_and_lookup.py` load/lookup nodeids that required missing vector matrices.
- `tests/test_v3_round34_state_debug_fasttext_migration.py` nodeids that expected default loaded fastText state.
- `tests/test_v3_round50_subset_medium_30k.py` nodeids that expected committed medium30k vectors.
- `tests/test_v3_round51_wrapper_primary_medium_swap.py` nodeids that expected default medium30k load and vector lookup.

## Root cause

The failing tests encoded obsolete artifact/default-load assumptions:

- Manifest entries correctly record extracted subset provenance and checksums.
- `vocab.txt` and `subset_manifest.json` are committed metadata.
- `vectors.npy` is intentionally not committed and must not be fabricated.
- Default runtime must remain no-load.
- State-debug/export surfaces must report no-load state honestly.
- Medium30k selection is valid metadata/default subset selection, but `vocab_size == 30000` in runtime stats is valid only after explicit operator-authorized load of a real, checksum-valid artifact.

## Guarded fix summary

- Updated mini/small/medium subset tests to assert committed metadata presence, missing `vectors.npy`, manifest checksum provenance, and fail-closed audit errors.
- Updated fastText adapter tests to assert unloaded default state, explicit load refusal without vectors, and no fake zero-vector creation.
- Updated state-debug tests to assert wrapper-primary metadata with `loaded == False`, `vocab_size == 0`, and no load side effects.
- Updated medium primary tests to distinguish medium30k selected subset metadata from operator-authorized vector runtime loading.
- Updated round32 readiness to `needs_more_audit` while vectors are absent.
- Did not create, download, fabricate, stage, or commit vector artifacts.
- Did not enable production persistence, runtime mapping defaults, enforcement, AGP bypass, or default fastText loading.

## Focused test results

```text
77 passed in 0.94s
```

Command:

```bash
python -m pytest -q \
  tests/test_v3_round29_subset_extraction_mini_1k.py \
  tests/test_v3_round30_subset_validation_audit.py \
  tests/test_v3_round31_subset_extraction_small_5k.py \
  tests/test_v3_round33_fasttext_load_and_lookup.py \
  tests/test_v3_round34_state_debug_fasttext_migration.py \
  tests/test_v3_round50_subset_medium_30k.py \
  tests/test_v3_round51_wrapper_primary_medium_swap.py \
  --tb=short
```

## Broader validation delta

- `python -m compileall -q adapters tests main.py scripts`: passed.
- `pytest --collect-only -q`: passed, `1418 tests collected`.
- `python -m pytest -q`: passed, `1418 passed in 18.75s`.

## Remaining taxonomy

- **Artifact-dependent runtime lookup:** still requires operator-local real vector artifacts and explicit guarded load authorization.
- **No-load default runtime:** green and preserved.
- **Subset metadata/provenance:** green for committed metadata, with vector checksums kept as provenance only.
- **State-debug no-load honesty:** green.
- **Medium30k primary metadata:** green; runtime load remains explicitly gated.
- **Production persistence/runtime mapping/enforcement:** still disabled/no-go.

## Next recommendation

Round591+ should keep artifact-free CI focused on no-load invariants and add a separate operator-local validation lane for real vector artifacts. That lane should remain ignored/untracked, explicitly authorized, and fail closed on missing or checksum-mismatched vectors.
