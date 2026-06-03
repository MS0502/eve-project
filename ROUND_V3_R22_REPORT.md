# EVE v3 Round22 Report — External Seed Dry-Run Workflow

## Result

- Previous stable: v3 round21 (`683 passed`)
- Current stable: v3 round22 (`690 passed`)
- `compileall`: passed

## Scope

Round22 validates the External Seed Policy registration workflow before any heavy seed binary is downloaded, imported, or used.

This is a dry-run-only round:

- no external seed file import
- no fastText loading/training
- no `self_embedding_adapter.py` rewrite
- no AGP runtime behavior change
- no threshold tuning
- no fallback pool expansion
- no semantic guard keyword expansion
- no memory/quarantine change

## Files changed

- `adapters/external_seed_manifest.py`
- `tests/test_v3_round21_external_seed_manifest.py`
- `tests/test_v3_round22_external_seed_dryrun.py`
- `AGENTS.md`
- `CURRENT_STATUS.md`
- `ROUND_V3_R22_REPORT.md`

## New dry-run API

`adapters/external_seed_manifest.py` now provides:

- `dryrun_register_seed(candidate_entry, existing_manifest)`
- `fasttext_korean_seed_dryrun_entry()`

The dry-run result is structured data only. It includes:

- `valid`
- `errors`
- `warnings`
- `candidate_result`
- `existing_manifest_result`
- `would_be_added`
- `manifest_after_dryrun`
- `dry_run_only`
- `file_loaded`
- `manifest_file_modified`

## Planned fastText dry-run candidate

The planned seed candidate is:

```yaml
name: cc.ko.300.bin
source: https://fasttext.cc/docs/en/crawl-vectors.html
license: CC-BY-SA-3.0
version: "2017-12"
downloaded_at: "(future)"
checksum: "SHA256:(future)"
imported_at_round: 23
imported_at_patch: v3_round23
```

The checksum/date placeholders are valid only in dry-run. Real import must replace them with concrete values.

## Round22 tests

Added tests verify:

- fastText candidate entry passes dry-run format checks.
- strict validation still rejects placeholders.
- dry-run does not load files.
- dry-run does not modify `seeds/MANIFEST.yaml`.
- dry-run returns simulated `manifest_after_dryrun` only.
- invalid license is blocked.
- duplicate seed names are blocked.
- dry-run does not mutate engine AGP modes, thresholds, or traces.
- no external seed is loaded at the end of round22.
- `self_embedding_adapter.py` remains unchanged.

## Invariants preserved

- Current manifest remains placeholder-only: `seeds: []`.
- `external_seed_loaded(empty_manifest_template())` remains `False`.
- No fastText seed is imported.
- No embedding rewrite occurred.
- AGP default mode remains observation.
- Compositor and SpeechHub veto double-lock behavior remains unchanged.
- Thresholds remain unchanged.
- Fallback pool remains minimal.
- External generation-body markers remain rejected.

## Decision

Round22 confirms that External Seed Policy can simulate a future fastText seed registration without side effects.

Actual seed import remains deferred. The next round may replace placeholders with concrete provenance only if the seed file is actually acquired and checksummed.

## Next recommendation

v3 round23:

- actual fastText seed provenance registration or checksum acquisition workflow
- replace dry-run placeholders with real `downloaded_at` and SHA256 only after acquisition
- still avoid `self_embedding_adapter.py` rewrite until seed registration is concrete
