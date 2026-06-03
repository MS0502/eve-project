# EVE v3 Round21 Report — External Seed Policy Infrastructure

## Result

- Previous stable: v3 round20 (`675 passed`)
- Current stable: v3 round21 (`683 passed`)
- `compileall`: passed

## Scope

Round21 converts the v3 External Seed Policy into a concrete manifest boundary before Task 2 fastText work.

This is an infrastructure-only round:

- no external seed file import
- no fastText training/loading
- no self-embedding rewrite
- no AGP runtime behavior change
- no threshold tuning
- no fallback pool expansion
- no semantic guard keyword expansion
- no memory/quarantine change

## Files added

- `seeds/MANIFEST.yaml`
- `seeds/README.md`
- `seeds/.gitkeep`
- `adapters/external_seed_manifest.py`
- `tests/test_v3_round21_external_seed_manifest.py`
- `ROUND_V3_R21_REPORT.md`

## Manifest contract

Current manifest is intentionally empty:

```yaml
seeds: []
```

Future entries must include:

- `name`
- `source`
- `license`
- `version`
- `downloaded_at`
- `checksum`
- `imported_at_round`
- `imported_at_patch`

## Validator contract

`adapters/external_seed_manifest.py` provides:

- `validate_seed_entry(entry)`
- `validate_manifest(manifest_data)`
- `external_seed_loaded(manifest_data)`
- `empty_manifest_template()`

Validation checks:

- manifest shape
- required provenance fields
- license whitelist
- source URL shape
- SHA256 checksum shape
- import round/patch metadata
- forbidden generation-body markers such as GPT, LLM, RWKV, Mamba, SSM, Transformer, Body

## Round21 tests

Added tests verify:

- manifest file exists
- manifest is placeholder-only in round21
- complete future seed entries validate
- missing required fields are rejected
- invalid licenses are rejected
- invalid checksums are rejected
- external generation-body markers are rejected
- validator is read-only
- no external seed is loaded in round21

## Invariants preserved

- AGP default mode remains observation.
- Compositor and SpeechHub double-lock behavior remains unchanged.
- Thresholds remain unchanged.
- Fallback pool remains minimal.
- Analyzer/decision data remains read-only.
- External seed manifest validation does not mutate AGP, traces, modes, thresholds, fallback pool, semantic guards, memory, or quarantine.

## Decision

Round21 confirms that External Seed Policy is now enforceable at the manifest level.

Task 2 can proceed only through the manifest gate. Actual seed import remains deferred.

## Next recommendation

v3 round22:

- seed registration dry-run or fastText provenance planning
- checksum/provenance workflow validation
- no heavy seed binary import until the manifest path is validated end-to-end
