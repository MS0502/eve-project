# EVE v3 Round351-365 Report — Captured failure taxonomy and guarded artifact-guard repair

Rounds351-365 proceed from the latest available PR #46 merge state in this repository.  No production persistence was enabled, `runtime_mapping_enabled` remains default-false, enforcement remains disabled, AGP was not bypassed, and no vectors or operator artifacts were created or committed.

## Round351 — captured 135-failure evidence

Operator-local Round336-350 artifact-dependent execution is consolidated as:

- result: `135 failed / 1279 passed`
- pytest failure nodeids are now captured before stdout/stderr tail truncation
- sample captured nodeids:
  - `tests/test_v3_round40_pre_swap_audit.py::test_round40_all_trace_modules_have_parallel_observation_option_unloaded`
  - `tests/test_v3_round41_final_swap.py::test_round41_main_py_contains_wrapper_swap_and_backup`
  - `tests/test_v3_round41_final_swap.py::test_round41_post_swap_assessment_ready`
  - `tests/test_v3_round41_final_swap.py::test_round41_wrapper_primary_used_for_known_word`
  - `tests/test_v3_round306_320_split_validation_execution.py::test_round306_320_cli_artifact_dependent_missing_artifacts_fails_closed`

The committed repository has the sample evidence and counts, not the full 135-nodeid list.  The missing full list is reported as unexpanded evidence rather than fabricated taxonomy.

## Rounds352-355 — concrete failure cluster taxonomy

The Round351-365 analyzer classifies captured nodeids into the requested concrete clusters:

| Cluster | Meaning |
| --- | --- |
| `round40_41_wrapper_swap_audit` | Round40/41 pre-swap/final-wrapper-swap audit failures. |
| `round94_98_runtime_mapping_enable_smoke` | Round94-98 runtime-mapping dry-run/precheck/controlled-smoke failures. |
| `round306_320_artifact_guard_regression` | Recent guarded artifact-dependent/split-validation failures. |
| `unclassified_pytest_failure` | Captured failures outside those concrete clusters. |

Applied to the operator-provided sample nodeids, the captured sample currently includes:

- `round40_41_wrapper_swap_audit`: 4 sample nodeids
- `round306_320_artifact_guard_regression`: 1 sample nodeid
- `round94_98_runtime_mapping_enable_smoke`: 0 sample nodeids in the provided sample
- `unclassified_pytest_failure`: 0 sample nodeids in the provided sample
- `unexpanded_failure_count`: 130, because only 5 of 135 nodeids were provided in-repo

## Rounds356-360 — selected repair cluster and implementation

Selected narrow recent cluster: `round306_320_artifact_guard_regression`.

Implementation attempted and completed:

- Added a read-only Round351-365 analyzer for captured 135-failure evidence, requested concrete taxonomy, selected guarded repair cluster, validation delta, and next recommendation.
- Extended the Round306-320 artifact-dependent runner's full-pytest summary capture to include:
  - bounded `sample_nodeids` before stdout/stderr tail truncation,
  - Round351-365 concrete cluster counts,
  - concrete cluster samples,
  - unexpanded failure count when pytest summary count exceeds captured nodeids.

This improves measurable behavior for the recent artifact-guard cluster without weakening tests or changing fail-closed behavior.

## Rounds361-365 — validation delta and recommendation

Focused tests for the new taxonomy and the existing Round306-350 split-validation path pass locally.  Artifact-free compile and collect-only pass locally.  The artifact-dependent command is feasible in this workspace only as a fail-closed readiness check because the required operator-local artifacts are intentionally absent; it exits with fail-closed readiness rather than reading or creating artifacts.

Full pytest remains red in this artifact-free workspace.  The observed local result is `206 failed / 1213 passed`, which is broader than the operator-local artifact-dependent `135 failed / 1279 passed` result because this workspace still lacks committed/operator-local vector artifacts.  The Round351-365 concrete taxonomy support is now available for future guarded artifact-dependent runs.

## Remaining taxonomy

- `round40_41_wrapper_swap_audit`: visible in captured samples; likely a substantial remaining cluster, but full count requires the complete 135-nodeid artifact-dependent log.
- `round94_98_runtime_mapping_enable_smoke`: requested cluster; no count can be assigned from the provided sample alone.
- `round306_320_artifact_guard_regression`: selected and repaired at the measurement/taxonomy boundary; focused tests pass.
- `unclassified_pytest_failure`: full count remains unknown until the complete 135-nodeid list is available.

## Next recommendation

Keep production persistence, runtime mapping defaults, enforcement, and AGP bypass disabled.  Re-run the guarded artifact-dependent command in the operator-local workspace and use the Round351-365 concrete taxonomy emitted in `pytest_failure_summary` to choose the next true code-repair cluster.  If the full taxonomy confirms the sample shape, prioritize the Round40/41 wrapper-swap audit cluster next, while keeping the Round94-98 runtime-mapping enable-smoke cluster separated from any persistence decision.
