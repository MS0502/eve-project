# ROUND_V3_R51_REPORT — wrapper primary swap to medium 30k

## Baseline

```text
previous_round = v3 round50
previous_tests = 1021 passed
round51_scope = narrow runtime primary swap
```

## Change

`FasttextEmbeddingAdapter` default subset changed from:

```text
cc.ko.300.subset.small.5k
```

to:

```text
cc.ko.300.subset.medium.30k
```

`main.py` still constructs `FasttextEmbeddingAdapter(engine=engine)` without a hard-coded subset override, so wrapper primary now uses medium 30k automatically.

## Runtime state

```text
engine.fasttext_embedding.subset_name = cc.ko.300.subset.medium.30k
engine.fasttext_embedding.loaded = true
engine.fasttext_embedding.vocab_size = 30000
engine.self_embedding = EmbeddingWrapper
engine.self_embedding.primary = FasttextEmbeddingAdapter(medium 30k)
engine.self_embedding.fallback = SelfEmbeddingAdapter(PMI+SVD)
```

## OOV lookup validation

Round44 general Korean OOV terms are now resolved by medium 30k:

```text
어때   -> vector found
그래   -> vector found
뭐야   -> vector found
좋아해 -> vector found
군대   -> vector found
코딩   -> vector found
```

```text
resolution = 6/6
```

Remaining EVE-specific OOV terms:

```text
EVE
민석
```

These remain targets for strategy B continuous self-learning.

## Not done

```text
smoke 재실행 없음
primary_hit_rate 측정은 round52
self-learning 구현 없음
wrapper logic 변경 없음
small 5k 제거 없음
AGP 변경 없음
PMI+SVD fallback 유지
```

## Next

```text
round52 = smoke rerun + primary_hit_rate measurement
round53+ = continuous self-learning mechanism
```

## Validation

```text
collected = 1039 tests
v3 split = 473 passed
non_v3 split = 566 passed
total = 1039 passed
compileall = passed
regression = 0
```

## Test evolution

Historical tests that assumed the runtime default subset was small 5k were updated to the round51 reality:

```text
FasttextEmbeddingAdapter default = cc.ko.300.subset.medium.30k
small 5k = extracted artifact, preserved
medium 30k = runtime primary, loaded
```
