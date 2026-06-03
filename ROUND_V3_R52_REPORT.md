# ROUND_V3_R52_REPORT — post-medium smoke rerun + primary_hit_rate measurement

## Baseline

```text
previous_round = v3 round51
previous_tests = 1039 passed
round52_scope = post-medium smoke rerun + telemetry measurement
runtime_primary = cc.ko.300.subset.medium.30k
self_learning_implementation = false
```

## Post-medium smoke result

```text
fixtures_count = 20
sampled_token_count = 36
wrapper_total_calls = 48
primary_hits = 37
fallback_uses = 11
errors = 0
primary_hit_rate = 0.7708
fallback_rate = 0.2292
error_rate = 0.0
AGP_pass_rate = 1.0
```

## Three-baseline comparison

```text
round44 pre-bridge small 5k primary_hit_rate  = 0.3542
round48 post-bridge small 5k primary_hit_rate = 0.2500
round52 post-medium 30k primary_hit_rate      = 0.7708

round48 -> round52 delta = +0.5208
```

Medium 30k exceeded the round49 projected range of 0.50–0.65. Projection status: `high_unexpected`.

## OOV pattern check

```text
round44_general_korean_oov = [어때, 그래, 뭐야, 좋아해, 군대, 코딩]
resolved_in_post_medium_smoke = 6/6
remaining_eve_specific_oov = [EVE]
```

`민석` remains a known EVE-specific self-learning target even though it was not present in the round52 fixture smoke sample.

## Policy

Round52 is measurement-only.

```text
self_learning_implementation = false
wrapper_change = false
AGP_change = false
subset_extraction = false
automatic_decision_application = false
small_5k_removed = false
PMI_SVD_fallback_removed = false
```

## Next

```text
round53+ = B continuous self-learning mechanism design/start
priority = high for EVE/Minsok-specific tokens
```
