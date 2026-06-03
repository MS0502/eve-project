# ROUND_V3_R49_REPORT — lexical coverage planning + C strategy confirmation

## Baseline

```text
previous_round = v3 round48
previous_tests = 999 passed
round49_scope = lexical coverage strategy planning only
```

## Result

```text
strategy = C_hybrid
sequence = A first, B continuous
A = medium 30k extraction for quick general Korean coverage
B = continuous self-learning for EVE/Minsok-specific drift
```

## Files changed

```text
adapters/smoke_data_analyzer.py
tests/test_v3_round49_lexical_coverage_planning.py
CURRENT_STATUS.md
AGENTS.md
ROUND_V3_R49_REPORT.md
```

## New analysis surfaces

```text
classify_oov_by_origin(oov_data) -> dict
project_medium_30k_impact(current_data, oov_patterns) -> dict
project_self_learning_impact(current_data) -> dict
compare_lexical_coverage_strategies(projections) -> dict
```

All are read-only. They do not extract a new subset, implement self-learning, adjust thresholds, alter wrapper behavior, remove fallback, or change runtime decisions.

## OOV origin split

```text
general_korean:
  - conversational expressions such as 어때 / 그래 / 뭐야
  - emotion expressions such as 좋아해
  - general/project-adjacent Korean such as 군대 / 코딩

eve_specific:
  - EVE
  - 민석
```

Interpretation: medium 30k is expected to improve general Korean coverage, but project/person-specific symbols still require EVE-specific learning.

## Strategy A — medium 30k

```text
current_subset = cc.ko.300.subset.small.5k
projected_subset = cc.ko.300.subset.medium.30k
projected_primary_hit_rate_range = 0.50~0.65
alignment_with_appendix_d_drift = low_to_medium_external_seed_dependency_increases
auto_extract_medium_30k = False in round49
```

## Strategy B — continuous self-learning

```text
substrate = SelfEmbeddingAdapter(PMI+SVD, 50d) fallback path
minsok_specific_strength = high
eve_specific_strength = high
general_korean_strength = partial_only_after_runtime_exposure
alignment_with_appendix_d_drift = high_eve_specific_distribution_moves_away_from_seed
auto_implement_self_learning = False in round49
```

## Strategy C — confirmed

```text
selected_strategy = C_hybrid
selected_by_minsok = True
decision_status = confirmed_at_round49
sequence = A_first_medium_30k_then_B_continuous_self_learning
round50_next = medium_30k_extraction_operator_task
round51_plus_next = self_learning_mechanism_start
```

## Invariants preserved

- No medium 30k extraction in round49.
- No self-learning mechanism implementation in round49.
- No wrapper threshold change.
- No AGP threshold change.
- No runtime decision change.
- No fallback removal.
- No memory/quarantine file edits.

## Tests

```text
new_tests = 12
expected_total = 1011 passed
compileall = required
```

## Next round

v3 round50 should perform the medium 30k extraction/registration workflow using the round29/31 deterministic subset pattern. Self-learning starts after that as a continuous mechanism, not in round50.
