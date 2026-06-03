# EVE v3 Round31 Report — small 5k subset extraction

## Result

- Previous stable: v3 round30 (`759 passed`)
- Current stable: v3 round31 (`770 passed`)
- Scope: register and validate the operator-extracted small 5k fastText subset.

## What changed

- Added subset files:
  - `seeds/subsets/cc.ko.300.subset.small.5k/vocab.txt`
  - `seeds/subsets/cc.ko.300.subset.small.5k/vectors.npy`
  - `seeds/subsets/cc.ko.300.subset.small.5k/subset_manifest.json`
- Added `cc.ko.300.subset.small.5k` to `seeds/MANIFEST.yaml`.
- Extended `adapters/external_seed_manifest.py` with:
  - small 5k constants
  - `fasttext_korean_subset_small_5k_entry()`
  - subset `purpose` validation
  - readiness assessment that can now prefer the small 5k production lexical seed candidate.
- Added `tests/test_v3_round31_subset_extraction_small_5k.py`.
- Updated older mini/readiness tests to allow both mini 1k and small 5k subsets.

## Registered subset

```text
name: cc.ko.300.subset.small.5k
parent_seed: cc.ko.300.bin
parent_checksum: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
selection_method: fasttext_frequency_order_top_k
filter_rule: remove_words_containing_unicode_replacement_char
vocab_size: 5000
vector_dim: 300
vocab_checksum: SHA256:c8d0d5ab119e7da119b7ca14d28a48f13767d2b86f8860ce0edbd51158cb701b
vectors_checksum: SHA256:6b55eb8eef00a003164b3ae5ce74424673d3539a0007e94f27edf08683bb2309
subset_manifest_checksum: SHA256:fddb35096e6acb4b541a5a96f28de4cc0478b56b28d724f8a2718e846bb0a856
purpose: production_lexical_seed
status: extracted
imported_at_round: 31
```

## Current state

```text
external_seed_state = registered
subset_state(cc.ko.300.subset.mini.1k) = extracted
subset_state(cc.ko.300.subset.small.5k) = extracted
runtime fastText load = False
subset used by self_embedding = False
self_embedding rewrite = False
```

## Readiness assessment

- Previous round30 readiness: `ready_with_concerns` because only mini 1k existed.
- Round31 readiness: `ready` because small 5k exists and validates as `production_lexical_seed`.
- The recommendation is still data-only. No rewrite, load, or state transition is automatically applied.

## Not doing

not doing summary: runtime usage and embedding rewrite are intentionally deferred.


- No self_embedding rewrite.
- No subset runtime load.
- No medium 30k extraction.
- No fastText runtime import.
- No AGP runtime change.
- No threshold change.
- No fallback pool expansion.
- No semantic guard keyword addition.
- No memory/quarantine modification.

## Next recommended round

v3 round32:

- self_embedding rewrite scaffold / 300d boundary audit.
- Keep the small 5k subset available but unused until the scaffold passes.
