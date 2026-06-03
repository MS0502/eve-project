# EVE v3 round18 report

## Scope

AGP threshold first-pass tuning decision.

Round18 is the first explicit data-backed decision workflow round. It records how AGP thresholds may be tuned from trace data, while preserving the rule that recommendations are data only and never auto-applied.

## Base

- Base artifact: `eve_v3_round17_passed.zip`
- Previous status: v3 round17, `653 passed`

## Decision

- action: `no_change` by default
- alternate action supported: `tuned` only when a reviewer supplies explicit threshold values
- old_thresholds: read from AGP trace report metadata
- new_thresholds: unchanged unless explicit proposal data is supplied
- automatic application: forbidden
- runtime application: requires explicit `AGPAdapter.set_thresholds(...)`

## Rationale

- sample_size: taken from `first_pass_report(...)["data_quality"]`
- insufficient data path: `no_change` + `collect_more`
- sufficient data without explicit proposal: `no_change` + `manual_review_or_collect_more`
- sufficient data with explicit proposal: `tuned` decision data
- recommendation_data: preserved as a data dict under `rationale_data`; it is not natural-language text and is not auto-applied

## Changes

- `adapters/agp_threshold_decision.py`
  - Added `build_threshold_tuning_decision(...)`.
  - Added `validate_threshold_tuning_decision(...)`.
  - Added stable decision constants:
    - `tuned`
    - `no_change`
    - `insufficient_data`
    - `manual_proposal`
    - `manual_review_no_change`

- `tests/test_v3_round18_agp_threshold_decision.py`
  - Added sufficient-data explicit tuning decision test.
  - Added insufficient-data no-change decision test.
  - Added recommendation-not-auto-applied test.
  - Added explicit `set_thresholds(...)` required test.
  - Added sufficient-data-without-explicit-thresholds no-change test.
  - Added report documentation test.
  - Added decision generation read-only test.

- `CURRENT_STATUS.md`
  - Updated current round and decision-workflow status.

## Test Verification

- `pytest` → `660 passed`
- `compileall` → passed

## Non-goals

- No automatic threshold application.
- No default veto activation.
- No fallback pool expansion.
- No new AGP reasons.
- No semantic guard keyword additions.
- No memory/quarantine changes.
- No fastText/Hyperbolic changes.

## Next

v3 round19 should stabilize the post-decision workflow:

- verify decision invariants across compositor and SpeechHub
- keep default observation mode
- keep threshold changes explicit/manual
- keep fallback pool and semantic guards frozen
