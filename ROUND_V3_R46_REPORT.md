# ROUND_V3_R46_REPORT — AGP category extraction trace probe

## Baseline

```text
round45: 959 passed
round46 goal: AGP extraction trace probe only
```

Round46 is a read-only probe round. It adds trace visibility for AGP category extraction, SA activation, overlap, and threshold evidence. It does not change AGP thresholds, category extraction logic, SA activation, runtime decision policy, wrapper policy, fallback policy, or subset size.

## Files changed

```text
adapters/agp_adapter.py
adapters/runtime_smoke_runner.py
adapters/smoke_data_analyzer.py
tests/test_v3_round46_agp_extraction_trace.py
CURRENT_STATUS.md
AGENTS.md
ROUND_V3_R46_REPORT.md
```

## Added surfaces

```text
AGPAdapter.verify_with_trace(...)
run_conversation_smoke_with_agp_trace(engine, fixtures)
analyze_extraction_traces(traces)
```

All three are data-only/read-only probe surfaces.

## Trace result on round43 fixtures

```text
fixtures_count: 20
agp_trace_count: 14
candidate_counts: 0/14 all zero
sa_counts: 4/14 each trace
zero_overlap_count: 14/14
weak_overlap_count: 0/14
thresholds: category=0.3, hormone=0.5
```

Representative trace rows:

```text
안녕           speech_hub candidates=[] SA=[speech_hub,greeting_simple,calm,has_sub] overlap=[] result=unknown_category
안녕하세요     speech_hub candidates=[] SA=[speech_hub,greeting_simple,calm,core_only] overlap=[] result=unknown_category
좋아           speech_hub candidates=[] SA=[speech_hub,empathy_neutral,curious,has_sub] overlap=[] result=unknown_category
너는 누구야    speech_hub candidates=[] SA=[speech_hub,meta_self_default,energetic,has_sub] overlap=[] result=unknown_category
그게 뭐야      speech_hub candidates=[] SA=[speech_hub,unknown_inquiry,energetic,has_sub] overlap=[] result=unknown_category
```

## H1~H4 evidence

```text
H1 candidate extraction returns no categories:
  fixtures_with_zero_candidates: 14/14
  strength: strong

H2 SA activation empty or too weak:
  fixtures_with_zero_sa_active: 0/14
  avg_sa_count: 4.0
  strength: weak

H3 candidate ∩ SA = 0:
  fixtures_with_zero_overlap: 14/14
  strength: strong
  interpretation: derivative symptom of H1 because candidate_count is 0 everywhere

H4 threshold too strict:
  fixtures_with_weak_overlap: 0/14
  strength: weak
```

## Most likely root cause

```text
most_likely_root_cause: H1
```

AGP is failing because the response category extraction probe sees no candidate categories. SA is not empty; it is producing active categories such as `speech_hub`, `greeting_simple`, `empathy_neutral`, `meta_self_default`, and `unknown_inquiry`. Threshold relaxation is not supported by this data because there is no weak overlap to rescue.

## Next-round options

```text
A: inspect/fix AGP category extraction surface for speech_hub/compositor meaning objects
B: inspect SA propagation during fixture response
C: run one threshold relaxation experiment only after manual approval
D: verify fixture categories against EVE internal category vocabulary
```

Recommended priority: A. Do not touch thresholds first.

## Invariants preserved

```text
AGP threshold 변경 없음
AGP category extraction logic 수정 없음
SA activation 수정 없음
runtime response selection/output 변경 없음
medium 30k 추출 없음
fallback 제거 없음
wrapper threshold 변경 없음
self_embedding rewrite 없음
automatic application 없음
```

## Result

```text
959 passed -> 973 passed
compileall passed
```

## Next recommended round

v3 round47: AGP meaning bridge fix for speech_hub/compositor candidate extraction.

Scope should be narrow: make existing speech_hub/compositor meaning objects expose AGP-compatible candidate categories, then rerun the round43/46 smoke measurement. Do not change thresholds.
