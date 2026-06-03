# ROUND_V3_R45_REPORT

## Summary

Round45 adds category coverage breakdown and AGP unknown-category root-cause analysis on top of the round43/44 smoke baseline. It is analysis-only and does not change runtime behavior.

## Baseline

```text
round44: 945 passed
round45: 959 passed
compileall: passed
regression: 0
```

## Files changed

```text
adapters/smoke_data_analyzer.py
tests/test_v3_round45_coverage_and_agp_root_cause.py
CURRENT_STATUS.md
AGENTS.md
ROUND_V3_R45_REPORT.md
```

## New analysis functions

```text
analyze_category_coverage(smoke_result, fixture_rows) -> dict
analyze_agp_unknown_category_root_cause(smoke_result, engine) -> dict
confirm_problem_separation() -> dict
```

All three are read-only.

## Problem separation

```text
problem_1_lexical_coverage:
  severity: medium
  blocker: False
  reason: PMI+SVD fallback keeps runtime path available despite low fastText primary coverage

problem_2_agp_anchor:
  severity: high
  blocker: True
  reason: AGP unknown_category prevents anchored generation from passing

priority: problem_2_first
recommendation_priority: manual_decision
```

## AGP root-cause hypotheses

```text
H1: AGP response category extraction returns no usable candidate categories
H2: SA activation is empty or too weak during fixture response time
H3: candidate categories and SA active categories have zero overlap
H4: category threshold is too strict for weak but relevant activation
```

The current smoke result exposes no candidate response categories, so H1 becomes the safest first probe for round46.

## Medium 30k policy

Medium 30k evaluation remains data-only. Extraction/promotion is forbidden until the AGP blocker is understood or resolved.

## Invariants preserved

- No AGP threshold change.
- No wrapper threshold change.
- No AGP runtime change.
- No SA mechanism change.
- No subset extraction or promotion.
- No fallback removal.
- No self_embedding rewrite.
- No memory/quarantine file changes.

## Next recommended round

Round46 should run one narrow manual AGP probe. Recommended first target: AGP response category extraction trace.
