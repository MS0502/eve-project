# ROUND_V3_R48_REPORT — post-bridge smoke rerun + residual AGP analysis

## Baseline

```text
previous_round = v3 round47
previous_tests = 987 passed
round48_scope = post-bridge smoke rerun + pre/post comparison
```

## Files changed

```text
adapters/smoke_data_analyzer.py
tests/test_v3_round48_post_bridge_smoke.py
CURRENT_STATUS.md
AGENTS.md
ROUND_V3_R48_REPORT.md
```

## New analysis surfaces

```text
compare_pre_post_bridge(pre_data, post_data) -> dict
identify_residual_issues(post_data) -> dict
```

Both are read-only. They do not change thresholds, wrapper behavior, bridge behavior, subset policy, fallback policy, or runtime decisions.

## Post-bridge smoke results

```text
fixtures_count = 20
agp_trace_count = 14

AGP pass rate:
  pre_bridge = 0.0
  post_bridge = 1.0
  delta = +1.0

candidate categories:
  pre_non_empty_traces = 0
  post_non_empty_traces = 14
  post_zero_candidate_traces = 0

overlap:
  pre_zero_overlap_traces = 14
  post_zero_overlap_traces = 0
  post_weak_overlap_traces = 0

extraction_source_distribution:
  meaning_bridge = 14

response path:
  pre_fallback_used_due_to_veto = 14
  post_fallback_used_due_to_veto = 0
```

## Wrapper telemetry comparison

```text
primary_hit_rate_pre = 0.3542
primary_hit_rate_post = 0.2500
primary_hit_rate_delta = -0.1042
fallback_rate_pre = 0.6458
fallback_rate_post = 0.7500
fallback_rate_delta = +0.1042
bridge_expected_to_change_lexical_coverage = False
```

Interpretation: AGP bridge success is independent from lexical coverage. The small 5k subset still under-covers the runtime fixtures, but PMI+SVD fallback remains active and this round does not alter subset policy.

## Residual issues

```text
fixtures_without_bridge = []
fixtures_with_low_overlap = []
fixtures_with_hormone_mismatch = []
residual_failures = []
issue_flags = ["no_residual_agp_issue_detected"]
recommendation_priority = manual_decision_stable_then_lexical_coverage
```

## Invariants preserved

- No AGP threshold change.
- No wrapper threshold change.
- No medium 30k extraction or promotion.
- No bridge expansion beyond round47 behavior.
- No automatic decision application.
- No fallback removal.
- No memory/quarantine file edits.

## Tests

```text
new_tests = 12
expected_total = 999 passed
compileall = required
```

## Next recommendation

v3 round49 should move to lexical coverage planning only after preserving AGP bridge stability as a regression invariant. Recommended focus: compare medium 30k projection vs EVE-specific self-learning expansion before extracting a new subset.
