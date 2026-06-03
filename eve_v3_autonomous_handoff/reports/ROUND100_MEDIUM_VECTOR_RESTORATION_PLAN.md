# Round100 Medium Vector Restoration / Validation Plan

## Goal

Round100 does not add generation behavior. It isolates and documents the artifact blocker identified in Round99 and adds a deterministic operator-supplied artifact audit path for `cc.ko.300.subset.medium.30k/vectors.npy` without putting a binary `.npy` file in the PR diff.

## Artifact paths inspected

The restoration helper scans the following subset artifact tiers:

| Tier | Required vector path | Validation meaning |
| --- | --- | --- |
| Medium 30k | `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` | Required for medium/full fastText validation and default runtime primary validation. |
| Small 5k | `seeds/subsets/cc.ko.300.subset.small.5k/vectors.npy` | May support code-only focused fallback only if the exact manifest-verified artifact exists. It is not a medium/full validation substitute. |
| Mini 1k | `seeds/subsets/cc.ko.300.subset.mini.1k/vectors.npy` | Fixture boundary only; not a runtime fallback for Round97/98. |

Current scan result in this checkout:

- Medium `vectors.npy`: absent.
- Small `vectors.npy`: absent.
- Mini `vectors.npy`: absent.
- Medium/full validation remains blocked.
- Small/focused fallback validation is also blocked in this checkout because the small vector artifact is absent.

## Implemented restoration/validation surface

Added `adapters/medium_vector_restoration.py` with read-only helpers:

- `scan_subset_artifact_paths(...)`: reports exact medium/small/mini file paths, checksums, shape/dtype, and validation tier status.
- `audit_operator_supplied_medium_vectors(candidate_path, ...)`: fail-closed audit for an operator-supplied medium `vectors.npy` candidate. It checks existence, checksum, shape `(30000, 300)`, and dtype `float32`.
- `build_round100_restoration_plan(...)`: combines the scan and optional candidate audit into a deterministic operator plan.
- `write_round100_restoration_status(...)`: writes JSON status only when explicitly called; it never writes vectors.

The module can also be run as:

```bash
python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy
```

A passing candidate audit means the operator may manually install that exact file at `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` and then rerun validation. The script itself performs no copy and creates no seed/vector artifact.

## Direct cause of Round97/98 and Round92~98 failures

Round97/98 focused tests fail before runtime mapping smoke logic is reached:

```text
assert "민석" in commit["created"]
```

The direct chain is:

1. `build_full_engine()` attempts to load the medium 30k fastText subset.
2. The medium vector file is absent, so the fastText adapter remains unloaded and records a code-only blocked state.
3. No small 5k vector artifact is present either, so the code-only small fallback path cannot load.
4. `EveSelfLearningAdapter.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])` requires known fastText context words.
5. Known context is empty because no subset vectors are loaded.
6. The commit gate correctly rejects `민석` with `insufficient_known_context`.
7. The Round97/98 fixture assertion fails.

The Round92~98 adjacent focused suite has the same direct cause because every fixture prepares the committed `민석` EveSpecific vector before running runtime mapping dry-run/proposal/enforcement/precheck/smoke logic.

This is an artifact validation blocker, not evidence of AGP bypass, runtime mapping persistence, or a need to weaken tests.

## Validation tier separation

Round100 records three separate validation tiers:

```text
medium/full validation = blocked until exact medium vectors.npy is restored and passes manifest checksum/shape/dtype audit
small/focused validation = possible only if exact small vectors.npy is restored and loaded as the code-only fallback
runtime mapping persistence = disabled until validation is honestly passed or the operator explicitly approves a partial-validation path
```

No test expectation was lowered. No dummy or fake vector file was created. No vector was treated as an AGP anchor.

## Operator restoration steps

1. Obtain the original operator-extracted `cc.ko.300.subset.medium.30k/vectors.npy` outside the PR diff.
2. Run:

   ```bash
   python -m adapters.medium_vector_restoration --candidate /path/to/vectors.npy
   ```

3. Confirm `acceptable_for_manual_install=true`.
4. Manually place that exact artifact at:

   ```text
   seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
   ```

5. Rerun medium artifact validation:

   ```bash
   pytest -q tests/test_v3_round50_subset_medium_30k.py tests/test_v3_round51_wrapper_primary_medium_swap.py
   ```

6. Rerun focused runtime mapping validation:

   ```bash
   pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
   pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py
   ```

## Round100 validation result

Passed:

- `python -m compileall -q adapters tests main.py`
- `pytest -q tests/test_v3_round100_medium_vector_restoration.py` — 5 passed.

Blocked/partial, honestly retained:

- `pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py` — 3 failures from absent subset vectors and empty known context.
- Round92~Round98 adjacent focused command — 14 failures from the same prerequisite setup blocker.

## Next decision

Do not proceed to AGP proof expansion or runtime mapping persistence until one of these happens:

1. The medium vector artifact is restored and validated against the recorded manifest checksum, shape, and dtype.
2. The operator explicitly approves a partial-validation experiment with runtime mapping persistence still disabled by default.
