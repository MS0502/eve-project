# ROUND_V3_R44_REPORT — first-pass smoke data analysis

## Baseline

```text
input baseline: v3 round43
previous tests: 931 passed
scope: data-only analysis of round43 post-swap smoke telemetry
```

## Goal

Analyze the round43 runtime smoke baseline without changing runtime behavior. Separate lexical coverage signals, OOV/fallback patterns, and AGP unknown-category failures before any threshold, subset, or routing decision.

## Files changed

```text
adapters/smoke_data_analyzer.py                         NEW
tests/test_v3_round44_smoke_data_analysis.py            NEW
CURRENT_STATUS.md                                       updated
AGENTS.md                                               updated
ROUND_V3_R44_REPORT.md                                  NEW
```

## Analysis result

Representative round43 smoke data analyzed through `analyze_smoke_data(...)`:

```text
total_calls: 48
primary_hits: 17
fallback_uses: 31
errors: 0
primary_hit_rate: 0.3541666666666667
fallback_rate: 0.6458333333333334
error_rate: 0.0
coverage_status: low_coverage
small_5k_coverage_interpretation: small_5k_under_covers_runtime_fixtures
```

Interpretation is still data-only. No subset promotion, fallback removal, threshold adjustment, or AGP policy change was applied.

## OOV / fallback first-pass grouping

Recent bounded OOV sample:

```text
EVE
군대
그래
뭐야
어때
좋아해
코딩
```

Grouped first-pass categories:

```text
conversational_question_expression: 3
emotion_expression: 2
minsok_military_context: 1
project_identity_term: 2
project_or_coding_context: 2
```

Fixture-category mapping:

```text
daily:    어때
minsok:   EVE, 군대, 어때, 좋아해, 코딩
reasoning: 그래, 뭐야
```

The OOV sample is a bounded recent sample, not a complete vocabulary audit.

## AGP first-pass analysis

```text
agp_pass_rate: 0.0
reason_counts: {unknown_category: 14}
speech_hub total: 14
speech_hub failed: 14
all_failures_unknown_category: true
first_pass_interpretation: agp_anchor_coverage_gap_not_fasttext_runtime_error
```

Because wrapper `error_rate = 0.0`, the AGP failure pattern is recorded as an anchor/category coverage gap rather than a fastText runtime failure.

## Manual review flags

```text
primary_hit_rate_below_0_50
fallback_rate_above_0_50
agp_unknown_category_present
agp_all_failures_unknown_category
```

## Next-round candidates

```text
round45: category_coverage_breakdown
  reason: separate lexical coverage from AGP anchor coverage before tuning
  auto_apply: false

round45+: medium_30k_subset_evaluation_only
  reason: small_5k runtime fixture coverage below 0.50
  auto_apply: false

round45+: agp_unknown_category_root_cause_analysis
  reason: post-swap smoke produced unknown_category failures
  auto_apply: false
```

## Invariants preserved

```text
engine.self_embedding remains EmbeddingWrapper
fastText primary remains loaded
PMI+SVD fallback remains preserved
fallback removal = false
subset promotion = false
wrapper threshold change = false
AGP threshold change = false
drift-based runtime decision = false
self_embedding rewrite = false
new subset extraction = false
memory/quarantine data edits = false
```

## Tests added

14 tests in `tests/test_v3_round44_smoke_data_analysis.py`:

```text
analysis data dict shape
data-only/no-action invariants
telemetry rate preservation
low-coverage flag
OOV grouping
OOV fixture-category mapping
fixture category summary
AGP unknown-category analysis
runtime-error separation
manual-only next-round candidates
read-only smoke input preservation
engine state non-mutation
data quality baseline marker
medium subset evaluation-only invariant
```

## Validation

```text
945 passed
compileall passed
```

## Next recommendation

v3 round45: category coverage breakdown + AGP unknown-category root cause analysis.

Round45 should still be analysis-only. It should not promote the medium 30k subset, tune AGP thresholds, remove fallback, or apply fixture-derived decisions automatically.
