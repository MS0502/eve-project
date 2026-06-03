# EVE v3 Round23 Report — External Seed Acquisition Workflow

## Result

- Previous stable: v3 round22 (`690 passed`)
- Current stable: v3 round23 (`698 passed`)
- `compileall`: passed

## Scope

Round23 prepares the actual acquisition workflow for the planned Korean fastText seed.

The seed file `cc.ko.300.bin` is not present in the current workspace. Therefore round23 follows the **B branch**: acquisition workflow preparation only.

This round does **not**:

- download `cc.ko.300.bin`
- register a real seed in `seeds/MANIFEST.yaml`
- import or load fastText
- rewrite `self_embedding_adapter.py`
- start drift metrics
- change AGP runtime behavior
- change thresholds
- expand fallback pools
- add semantic guard keywords
- modify memory/quarantine

## Files changed

- `adapters/external_seed_manifest.py`
- `tests/test_v3_round23_external_seed_acquisition_workflow.py`
- `AGENTS.md`
- `CURRENT_STATUS.md`
- `ROUND_V3_R23_REPORT.md`

## New acquisition/state API

`adapters/external_seed_manifest.py` now provides:

- `is_valid_checksum_format(checksum)`
- `compute_seed_checksum(file_path)`
- `external_seed_state(manifest_data, loaded_seed_names=None, used_seed_names=None)`
- `fasttext_korean_acquisition_workflow(seed_file_path=None)`

## External seed state ladder

```text
unregistered → registered → loaded → used
```

Round23 current state:

```text
external_seed_state(empty_manifest_template()) == "unregistered"
external_seed_loaded(empty_manifest_template()) == False
```

The legacy `external_seed_loaded(...)` bool still means that a manifest declares seed entries. New migration work should prefer the explicit `external_seed_state(...)` ladder.

## Acquisition workflow result

Because `cc.ko.300.bin` is absent:

```text
workflow_state = acquisition_deferred
seed_file_present = False
download_required = True
actual_download_performed = False
manifest_file_modified = False
file_loaded = False
self_embedding_rewrite = False
valid_if_registered_now = False
```

The strict validation fails only because the planned dry-run entry still has placeholder provenance:

- `downloaded_at = (future)`
- `checksum = SHA256:(future)`

Those placeholders are allowed only in dry-run. A real registration round must replace them with concrete values.

## Round23 tests

Added tests verify:

- missing `cc.ko.300.bin` defers registration
- explicit state ladder behavior
- deterministic SHA256 checksum helper on a temp file
- missing seed file raises instead of producing fake checksum
- real `seeds/MANIFEST.yaml` remains placeholder-only
- acquisition workflow is read-only for engine AGP state
- `self_embedding_adapter.py` remains unrevised
- strict registration still requires concrete provenance

## Invariants preserved

- Current manifest remains placeholder-only: `seeds: []`.
- External seed state remains `unregistered`.
- No fastText seed is imported.
- No embedding rewrite occurred.
- AGP default mode remains observation.
- Compositor and SpeechHub veto double-lock behavior remains unchanged.
- Thresholds remain unchanged.
- Fallback pool remains minimal.
- External generation-body markers remain rejected.

## Decision

Round23 confirms that the fastText acquisition path is ready, but actual registration is deferred because the seed file is not present.

This is a valid result. No change is better than fake provenance.

## Next recommendation

v3 round24:

- If `cc.ko.300.bin` is provided: compute real checksum and register strict MANIFEST entry.
- If no file is provided: keep registration deferred and write operator acquisition instructions.
- Do not load fastText into runtime or rewrite `self_embedding_adapter.py` until manifest registration is concrete.
