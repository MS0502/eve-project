# EVE v3 Round28 — Subset Extraction Plan

## Purpose

This document plans the future extraction of an EVE-sized lexical seed subset from the registered Korean fastText crawl vector.

Round28 is planning only. It records external verification and defines the deterministic extraction boundary for round29+. It does not load fastText, does not extract vectors, and does not rewrite `self_embedding_adapter.py`.

## Verified source seed

```yaml
name: cc.ko.300.bin
source: https://fasttext.cc/docs/en/crawl-vectors.html
download_url: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz
license: CC-BY-SA-3.0
version: "2017-12"
downloaded_at: "2026-05-11"
checksum: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
file_size_bytes: 7243669409
file_location: external (Google Drive: /eve_seeds/cc.ko.300.bin.gz)
verification_status: verified
verification_match: true
verification_context: Colab external SHA256 verification, Python hashlib, no adapters import
```

## Extraction goal

Reduce the full `cc.ko.300.bin` seed into deterministic EVE subsets that can be committed or managed safely.

```text
full seed:     ~2M words, external artifact only
EVE subset:    1k / 5k / 30k words depending on round target
embedding dim: 300d fastText vectors
```

## Deterministic selection policy

Selection must be deterministic.

1. Load the verified external seed only in the operator environment.
2. Use the fastText vocabulary order as the primary order.
3. Treat the vocabulary order as frequency-derived model order.
4. Select the first N tokens after deterministic filters.
5. Do not sample.
6. Do not use random shuffling.
7. Do not select tokens from raw runtime conversations in round29.
8. Record the exact extraction script, N, checksum, and output checksum.

If a token filter is introduced later, it must be deterministic and documented before extraction.

## Extraction targets

| Target | Tokens | Estimated vector bytes | Use |
|---|---:|---:|---|
| mini | 1,000 | ~1.2 MB | CI smoke test / migration rehearsal |
| small | 5,000 | ~6 MB | first EVE lexical seed candidate |
| medium | 30,000 | ~36 MB | fuller Phase 1 lexical seed candidate |

Estimates assume `tokens × 300 × 4 bytes`, excluding vocab metadata and file headers.

## Output format options

Round29 should choose one primary format and document the reason.

### Option A: `vocab.txt` + `vectors.npy`

- deterministic
- easy to inspect
- numpy-native
- simple checksum
- recommended for first subset

### Option B: pickle bundle

- single file
- less transparent
- not preferred unless metadata complexity grows

### Option C: fastText subset `.bin`

- closest to runtime model format
- hardest to produce safely
- defer until loader usage is proven

## Proposed round29 output

```text
seeds/subsets/cc_ko_300_mini_vocab.txt
seeds/subsets/cc_ko_300_mini_vectors.npy
seeds/subsets/cc_ko_300_mini_manifest.json
```

The subset manifest records extraction metadata and output checksums.

The mini target should be extracted first. Small/medium subsets should follow only after mini extraction and checksum validation pass.

## Drift baseline policy

External Seed Policy says seed drift must be measured later. The subset should preserve a baseline copy of the original seed vectors so future EVE-specific updates can be compared against the original seed.

Minimum future drift record:

```yaml
seed_name: cc.ko.300.bin
subset_name: cc_ko_300_mini
baseline_checksum: SHA256:...
current_checksum: SHA256:...
mean_cosine_distance_from_seed: 0.0
round_created: 29
round_used_by_self_embedding: null
```

The initial drift value is `0.0` because the subset is still seed-derived and not yet EVE-updated.

## Round29 boundary

Round29 may extract a subset in Colab or another explicit operator environment.

Round29 still must not:

- rewrite `self_embedding_adapter.py`;
- use the subset in runtime generation;
- change AGP thresholds;
- expand fallback surface pools;
- add semantic guard keywords;
- mutate memory or quarantine data.

## Round30+ boundary

Only after subset extraction is verified should self-embedding migration begin.

Expected ladder:

```text
registered → externally_verified → subset_planned → subset_extracted → used
```

Round28 reaches `subset_planned` as documentation only. The manifest state remains `registered`.
