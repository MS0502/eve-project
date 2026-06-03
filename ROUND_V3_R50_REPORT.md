# ROUND_V3_R50_REPORT — medium 30k subset extraction + registration

## Result

```text
baseline: v3 round49
round50_scope: cc.ko.300.subset.medium.30k registration only
status: passed
```

## Operator artifact

```text
name: cc.ko.300.subset.medium.30k
parent_seed: cc.ko.300.bin
parent_checksum: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
selection_method: fasttext_frequency_order_top_k
filter_rule: remove_words_containing_unicode_replacement_char
vocab_size: 30000
vector_dim: 300
vocab_checksum: SHA256:dc0e6985e767003f13d74bf1a16c257be29e3a125dabb16a320cd13082aef308
vectors_checksum: SHA256:f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05
subset_manifest_checksum: SHA256:28222ff89e1fa0d2c20463eeef79729d731bf0484f0f4f8161863088c2724c3a
purpose: production_lexical_seed_expanded
raw_vocab_size: 2000000
clean_vocab_size: 1999987
filtered_corrupted: 13
extracted_at: 2026-05-12
```

## Files added

```text
seeds/subsets/cc.ko.300.subset.medium.30k/vocab.txt
seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
seeds/subsets/cc.ko.300.subset.medium.30k/subset_manifest.json
```

## Code changes

```text
adapters/external_seed_manifest.py
- medium 30k constants
- fasttext_korean_subset_medium_30k_entry()
- round50_medium_30k_oov_resolution_data()
- production_lexical_seed_expanded allowed purpose
- assess_self_embedding_rewrite_readiness now sees mini/small/medium tiers
```

## Tests added

```text
tests/test_v3_round50_subset_medium_30k.py
- 10 tests
- directory/files exist
- vocab.txt = 30000 UTF-8 lines
- vectors.npy shape = (30000, 300), dtype float32
- checksums match
- parent linkage correct
- purpose = production_lexical_seed_expanded
- subset_state = extracted
- mini/small/medium coexist
- round44 general Korean OOV 6/6 resolution recorded
- wrapper primary remains small 5k until round51
```

## OOV resolution

```text
resolved_general_korean_oov = 6/6
resolved:
  어때
  그래
  뭐야
  좋아해
  군대
  코딩

remaining_eve_specific_oov:
  EVE
  민석
```

Interpretation: round49 projection was correct for the general Korean OOV sample. EVE-specific proper nouns still require strategy B, continuous self-learning.

## Invariants preserved

```text
wrapper primary swap: not done
engine.self_embedding: EmbeddingWrapper remains active
fasttext default subset: small 5k remains active until round51
self-learning mechanism: not implemented
small 5k removal: not done
mini/small/medium coexist
AGP: unchanged
PMI+SVD fallback: preserved
memory/quarantine data files: unchanged
```

## Next

```text
round51: wrapper primary swap to medium 30k
round52: smoke rerun + primary_hit_rate measurement
round53+: strategy B continuous self-learning
```
