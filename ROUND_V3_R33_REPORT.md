# ROUND_V3_R33_REPORT

## Goal

Implement actual instance-level loading and lookup for `FasttextEmbeddingAdapter` using the registered small 5k fastText subset.

## Base

- Previous stable: v3 round32 (`780 passed`)
- Current round: v3 round33

## Changes

### `adapters/fasttext_embedding_adapter.py`

Implemented:

- `load()`
  - reads `seeds/subsets/cc.ko.300.subset.small.5k/vocab.txt`
  - reads `vectors.npy`
  - reuses manifest/subset audit checks
  - verifies file checksums before loading
  - loads vectors as `float32`, shape `(5000, 300)`
- `get_vector(word)`
  - returns a 300d `float32` copy for known words
  - returns `None` for OOV words
- `get_embedding(word)`
  - compatibility alias for `get_vector`
- `similarity(word_a, word_b)`
  - deterministic cosine similarity
  - returns `0.0` for OOV or zero-norm cases
- `most_similar(word, top_n, min_sim)`
  - deterministic score-descending, word-ascending tie break
- `text_embedding(text)`
  - whitespace-token mean only
  - OOV tokens skipped
  - no known tokens -> deterministic zero vector
- `text_similarity(text_a, text_b)`
  - deterministic cosine over text embeddings

### Tests

Added `tests/test_v3_round33_fasttext_load_and_lookup.py`.

Covers:

- actual load succeeds
- instance state transitions to loaded
- invalid subset directory fails closed
- vector lookup returns 300d `float32` copy
- OOV word lookup returns `None`
- `get_embedding` aliases `get_vector`
- similarity is deterministic
- self-similarity is approximately `1.0`
- OOV similarity returns `0.0`
- text embeddings are deterministic
- empty/OOV text returns zero vector
- subset manifest state remains `extracted`
- engine global `self_embedding` remains PMI+SVD 50d
- fastText runtime package is not imported

Updated round32 scaffold tests to the new round33 reality: adapter methods now require explicit load instead of raising `NotImplementedError` unconditionally.

## Current state

```text
external_seed_state = registered
subset_state(cc.ko.300.subset.small.5k) = extracted
FasttextEmbeddingAdapter actual load = explicit instance only
engine.self_embedding = SelfEmbeddingAdapter(PMI+SVD, 50d)
self_embedding rewrite = False
runtime fastText package import = False
```

## Not doing

- No global engine swap.
- No `self_embedding_adapter.py` rewrite.
- No module migration.
- No morph analyzer integration.
- No fastText subword inference.
- No AGP runtime change.
- No semantic guard keyword addition.
- No memory/quarantine modification.

## Next recommendation

v3 round34: first low-risk module migration target. Recommended: `state_debug_adapter` reporting-only path, because it is diagnostic rather than core generation logic.
