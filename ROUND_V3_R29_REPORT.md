# EVE v3 Round29 Report — Mini 1k Subset Registration

## Result

- Base: v3 round28 (`740 passed`)
- Current: v3 round29 (`748 passed`)
- Scope: register and verify the operator-extracted mini 1k subset artifact
- `compileall` passed
- Production runtime changes: none

## Registered subset

```yaml
name: cc.ko.300.subset.mini.1k
parent_seed: cc.ko.300.bin
parent_checksum: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
selection_method: fasttext_frequency_order_top_k
filter_rule: remove_words_containing_unicode_replacement_char
vocab_size: 1000
vector_dim: 300
vocab_checksum: SHA256:5e7eb8da5cb96f9c1d207846f8f4ef781f48f74a0f9b3eb4ee112cdc47d53064
vectors_checksum: SHA256:ac987683f9ad733cea97ed13ff5b39e73a2ed91ba5d489466f690b2e747d6f2a
subset_manifest_checksum: SHA256:137631552e9a0c6efca941c48e0fa4bd35854ba59519d87dcdcdfa29a38b1caa
extracted_at: 2026-05-11
imported_at_round: 29
imported_at_patch: v3_round29
status: extracted
```

Files committed:

```text
seeds/subsets/cc.ko.300.subset.mini.1k/
├── vocab.txt
├── vectors.npy
└── subset_manifest.json
```

## Purpose

The mini 1k subset is fixture-level. It proves the deterministic extraction,
manifest linkage, checksum verification, and subset-state workflow. It is not
intended to be the production EVE lexical map.

## State after round29

```text
external_seed_state = registered
subset_state(cc.ko.300.subset.mini.1k) = extracted
runtime fastText load = False
subset used by self_embedding = False
self_embedding rewrite = False
```

## Non-goals preserved

- No self-embedding rewrite.
- No runtime subset load.
- No AGP runtime change.
- No threshold change.
- No fallback pool expansion.
- No semantic guard keyword addition.
- No memory/quarantine modification.
- No small/medium subset extraction.

## Next recommendation

v3 round30:

- subset validation audit;
- confirm subset files, manifest entry, checksums, and parent linkage remain stable;
- keep self-embedding rewrite postponed until after the fixture subset is audited.
