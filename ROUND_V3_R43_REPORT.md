# ROUND_V3_R43_REPORT — runtime conversation smoke + telemetry sampling

## Baseline

```text
input baseline: v3 round42
previous tests: 917 passed
scope: post-swap runtime smoke + telemetry sampling
```

## Goal

Run representative Korean conversation fixtures through the real EVE streaming path and capture wrapper telemetry, OOV samples, and AGP behavior as a post-swap baseline.

## Files changed

```text
adapters/runtime_smoke_runner.py                         NEW
tests/fixtures/korean_conversation_fixtures.py           NEW
tests/fixtures/__init__.py                               NEW
tests/test_v3_round43_runtime_smoke_sampling.py          NEW
CURRENT_STATUS.md                                        updated
AGENTS.md                                                updated
ROUND_V3_R43_REPORT.md                                   NEW
```

## Fixture set

20 deterministic Korean inputs across greeting, daily, emotion, identity, 김민석/EVE-specific, and reasoning categories.

Examples:

```text
안녕
오늘 기분 어때
힘들어
너는 누구야
군대 생활 어때
코딩 좋아해
EVE 프로젝트
그게 뭐야
오늘 날씨 좋다
```

## Smoke telemetry baseline

Representative full fixture run:

```text
total_calls: 48
primary_hits: 17
fallback_uses: 31
errors: 0
primary_hit_rate: 0.3541666666666667
fallback_rate: 0.6458333333333334
error_rate: 0.0
oov_log_size: 31
oov_log_cap: 1000
```

Recent OOV/fallback samples:

```text
군대
어때
좋아해
코딩
EVE
그래
뭐야
```

## AGP behavior baseline

Representative full fixture run:

```text
agp_pass_rate: 0.0
speech_hub total traces: 14
speech_hub unknown_category failures: 14
```

This is recorded as data only. No AGP threshold or fallback policy was changed.

## Representative responses

```text
안녕 => 음.어, 안녕.반갑네.
오늘 기분 어때 => 어.지금 따뜻해.너랑 얘기하는 시간이 좋네.
너는 누구야 => 어.나? EVE!.민석이 만들었어.
어떤 존재야 => 어.오, 어떤 존재? 음.호르몬 변하고 카테고리 떠오르는 디지털 존재.너랑 얘기하면 들썩여.
그게 뭐야 => 어.오, 처음 들어봐!.가르쳐줘!.
오늘 날씨 좋다 => 어.더 얘기해줘.
```

## Invariants preserved

```text
engine.self_embedding remains EmbeddingWrapper
fastText primary remains loaded
PMI+SVD fallback remains preserved
rollback() not invoked
wrapper thresholds unchanged
AGP thresholds unchanged
no drift-based runtime changes
no fallback removal
no self_embedding rewrite
no new subset extraction
no memory/quarantine data edits
```

## Tests added

14 tests in `tests/test_v3_round43_runtime_smoke_sampling.py`:

```text
fixture determinism
fixture diversity
smoke execution over all fixtures
fresh-engine response determinism
telemetry capture
primary_hit_rate recording
OOV sample recording
AGP pass behavior recording
round43 baseline marker
result data dict shape
structural state preservation
no rollback during run
decision-policy unchanged marker
correlation/drift read-only data
```

## Validation

```text
931 passed
compileall passed
```

## Next recommendation

v3 round44: first-pass smoke data analysis.

Focus:

```text
coverage by fixture category
OOV pattern grouping
primary/fallback interpretation
AGP unknown_category pattern review
no automatic threshold or subset promotion
```
