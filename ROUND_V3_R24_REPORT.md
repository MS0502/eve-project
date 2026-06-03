# EVE v3 Round24 Report — First Strict External Seed Registration

## Result

- Previous stable: v3 round23 (`698 passed`)
- Current stable: v3 round24 (`706 passed`)
- `compileall`: passed

## Scope

Round24 applies the External Seed Policy for the first time with a concrete registered seed entry.

This round registers only provenance in `seeds/MANIFEST.yaml`. It does not import, load, parse, or use the seed binary.

## Provenance registered

```yaml
- name: cc.ko.300.bin
  source: https://fasttext.cc/docs/en/crawl-vectors.html
  download_url: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz
  license: CC-BY-SA-3.0
  version: "2017-12"
  downloaded_at: "2026-05-11"
  checksum: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
  file_size_bytes: 7243669409
  file_location: external (Google Drive: /eve_seeds/cc.ko.300.bin.gz)
  imported_at_round: 24
  imported_at_patch: v3_round24
```

## Files changed

- `seeds/MANIFEST.yaml`
- `seeds/MANIFEST.yaml.backup`
- `adapters/external_seed_manifest.py`
- `tests/test_v3_round21_external_seed_manifest.py`
- `tests/test_v3_round22_external_seed_dryrun.py`
- `tests/test_v3_round23_external_seed_acquisition_workflow.py`
- `tests/test_v3_round24_first_seed_registration.py`
- `AGENTS.md`
- `CURRENT_STATUS.md`
- `ROUND_V3_R24_REPORT.md`

## New/updated seed API

`adapters/external_seed_manifest.py` now provides:

- `fasttext_korean_seed_registered_entry()`
- `load_manifest_file(...)`
- `register_seed_entry(...)`

The registration path uses:

- strict validation
- duplicate seed-name rejection
- backup creation before modification
- temp-file write + atomic replace
- no file loading
- no self-embedding rewrite

## State after round24

```text
external_seed_state(load_manifest_file(SEED_MANIFEST_PATH)) == "registered"
external_seed_loaded(load_manifest_file(SEED_MANIFEST_PATH)) == True
```

Meaning:

- the manifest now declares one valid seed entry
- runtime has still not loaded the seed
- self embeddings still do not use the seed

## Tests added

`tests/test_v3_round24_first_seed_registration.py` verifies:

- manifest entry registered for `cc.ko.300.bin`
- external seed state is now `registered`
- invalid strict entry does not modify manifest
- backup is created before write
- simulated atomic failure leaves manifest intact
- `self_embedding_adapter.py` remains unchanged
- fastText/gensim libraries are not imported
- engine AGP state remains unchanged

Existing round21~23 tests were evolved from placeholder-only assumptions to the new state ladder:

- empty manifest template remains the unloaded baseline
- dry-run remains read-only
- acquisition helper remains read-only after registration

## Invariants preserved

- No seed binary is committed.
- No seed binary is loaded, parsed, or used.
- No fastText or gensim runtime import.
- No self-embedding rewrite.
- No AGP runtime change.
- No threshold change.
- No fallback pool expansion.
- No new AGP reason.
- No semantic guard keyword expansion.
- No memory/quarantine modification.

## Decision

Round24 moves the External Seed Policy state from `unregistered` to `registered` using concrete provenance.

This is the first real seed registration, not the first seed load.

## Next recommendation

v3 round25:

- add fastText runtime integration scaffold or optional loader boundary
- verify registered seed can remain unused by default
- do not rewrite `self_embedding_adapter.py` yet
- keep AGP and semantic guard behavior unchanged
