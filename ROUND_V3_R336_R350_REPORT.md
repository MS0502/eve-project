# EVE v3 Round336-350 Report — Artifact-dependent failure taxonomy capture

## Rounds completed

Rounds336-350 are completed as a guarded measurement-improvement round.

## Round336 evidence consolidation

The operator-local artifact-dependent readiness evidence was consolidated as green:

- `ready: true`
- `missing: []`
- `unsafe: []`
- `vector_contents_read: false`
- executed result: `1274 passed`, `135 failed`, exit code `1`

No production persistence, runtime-mapping default enablement, enforcement enablement, AGP bypass, vector creation, vector mutation, or operator artifact staging was introduced.

## Rounds337-340 failure taxonomy

The prior artifact-dependent execution aggregate proves the failure count, but the existing Round306-335 runner retained only stdout/stderr tails. That means the 135 failures cannot be honestly expanded into per-nodeid clusters from the committed evidence alone.

Recorded taxonomy for the operator-provided aggregate:

```text
unparsed_operator_report_failures_needing_full_pytest_summary: 135
```

This is an explicit taxonomy cluster, not a fabricated parse. The implemented improvement captures full pytest failure summaries before tail truncation on the next guarded artifact-dependent execution, while still not retaining full stdout/stderr in JSON.

## Rounds341-345 selected repair cluster

Selected cluster:

```text
pytest_failure_taxonomy_capture_gap
```

Implementation action:

```text
measurement_improvement_capture_full_pytest_failure_summary_before_tail_truncation
```

This is the narrowest safe non-persistence cluster because the available operator-local evidence lacks failed nodeids. Repairing the measurement gap is required before selecting a real code-failure cluster without guessing.

## Implementation summary

- Added a read-only Round336-350 analyzer that parses pytest failure nodeids, extracts summary counts, classifies deterministic clusters, consolidates readiness/execution evidence, selects the safe repair cluster, and emits the validation delta.
- Updated the guarded artifact-dependent runner so future `python -m pytest -q` execution captures failure counts, classified cluster counts, and bounded cluster samples before stdout/stderr tail truncation.
- Added focused regression tests for the analyzer, 135-failure unparsed taxonomy preservation, and the new pytest failure-summary capture path.

## Validation delta

Focused validation is green for the new taxonomy/capture tests and the previous split-validation tests. Broader artifact-free full pytest remains red in this workspace because seed/subset vector artifacts are not present locally; that baseline is still artifact-dependent and must not be repaired by fabricating or committing vectors.

## Remaining taxonomy

- `unparsed_operator_report_failures_needing_full_pytest_summary: 135`
- local artifact-free full-suite red baseline remains artifact-dependent where seed/subset vectors are absent from this workspace.

## Next recommendation

Keep production persistence, `runtime_mapping_enabled` default, enforcement, and AGP bypass disabled. Re-run the guarded artifact-dependent command in the operator-local workspace with the new summary-capture behavior, then select the largest non-persistence concrete cluster from the captured nodeids for Round351+ repair.
