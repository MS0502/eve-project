# EVE v3 Round 30 Report — Subset Validation Audit

## Result

- Status: passed
- Base: v3 round29 (`748 passed`)
- Current: v3 round30 (`759 passed`)
- Scope: subset validation audit + self_embedding rewrite readiness assessment

## Audit summary

- invariants verified: subset file checksums, vocab line count, vector shape/dtype, corrupted-word filter, subset manifest payload
- dependencies verified: parent seed linkage and parent checksum dependency
- runtime safety verified: subset is not loaded, fastText runtime is not imported, `self_embedding_adapter.py` is unchanged
- read-only verified: audit/readiness helpers do not mutate manifest, AGP mode, compositor mode, speech_hub mode, or seed state

## Readiness assessment

Current embedding state:

- method: `PMI+SVD`
- dimension: `50`
- runtime rewrite: not started

Available subset:

- name: `cc.ko.300.subset.mini.1k`
- method: `fasttext_extracted`
- dimension: `300`
- vocab size: `1000`
- state: `extracted`
- character: fixture-level; top frequency tokens are punctuation and short morphemes

Assessment:

- readiness: `ready_with_concerns`
- migration risk: `medium`
- primary concern: mini 1k validates the pipeline but is not production vocabulary

## Decision point for round31

option A:

- build a self_embedding 300d scaffold using mini 1k as fixture only
- do not treat mini 1k as production lexical memory

option B:

- extract small 5k first, then perform a more meaningful self_embedding rewrite
- preferred by readiness data because mini 1k is fixture-level

## Not doing

- not doing self_embedding rewrite
- not doing subset runtime load
- not doing small/medium extraction
- not doing AGP runtime change
- not doing seed/subset state transition
- not doing readiness recommendation auto-application
- not doing semantic guard keyword expansion

## Next recommendation

v3 round31 should choose one of:

1. small 5k extraction preparation, or
2. self_embedding 300d scaffold only with mini 1k as a fixture boundary.
