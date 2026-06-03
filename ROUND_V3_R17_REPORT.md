# EVE v3 round17 report

## Scope

AGP trace data first-pass analysis.

Round17 applies the read-only trace analyzer helpers created in rounds 10~12 to a stable first-pass report shape. It produces data only and does not modify thresholds, fallback, veto, routes, semantic guards, or memory.

## Base

- Base artifact: `eve_v3_round16_passed.zip`
- Previous status: v3 round16, `647 passed`

## Changes

- `adapters/agp_trace_analyzer.py`
  - Added `first_pass_report(...)`.
  - Added private `_fail_counts_by_reason()` helper.
  - Report sections:
    - `summary`
    - `by_layer`
    - `fail_patterns`
    - `threshold_analysis`
    - `data_quality`
  - Data quality gate:
    - `collect_more` for insufficient samples
    - `ready_for_tuning` when sample size is sufficient

- `tests/test_v3_round17_agp_trace_first_pass_analysis.py`
  - Added deterministic synthetic trace report test.
  - Added empty trace safety test.
  - Added tiny runtime simulation report test using `build_full_engine()`.
  - Added data-dict-not-text invariant test.
  - Added read-only invariant test.
  - Added sample-size data quality gate test.

- `CURRENT_STATUS.md`
  - Updated current round and analysis-only status.

## Analysis invariants verified

- First-pass report is a dict, not a natural-language recommendation.
- Threshold analysis remains advisory-only.
- Report generation does not mutate trace rows.
- Report generation does not change analyzer threshold metadata.
- Empty trace data is safe.
- Runtime simulated traces can be summarized from engine wiring.
- Sample-size gate blocks premature tuning.

## Non-goals

- No threshold tuning.
- No threshold persistence.
- No default veto activation.
- No fallback pool expansion.
- No new AGP reasons.
- No semantic guard keyword additions.
- No memory/quarantine changes.
- No fastText/Hyperbolic changes.

## Validation

- `pytest` → `653 passed`
- `compileall` → passed

## Next

v3 round18 should use the first-pass report data to make an explicit threshold tuning decision:

- manual/configurable only
- no analyzer recommendation auto-apply
- no fallback pool expansion
- no semantic keyword expansion
