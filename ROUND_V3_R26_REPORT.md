# EVE v3 Round26 Report — fastText checksum/load workflow

## Result

- Previous baseline: v3 round25 (`715 passed`)
- Current result: v3 round26 (`725 passed`)
- `compileall`: passed

## Scope

Round26 implements the explicit fastText checksum verification and load workflow. It does not load the registered external seed by default and does not rewrite `self_embedding_adapter.py`.

## Files changed

- `adapters/fasttext_loader.py`
- `tests/test_v3_round25_fasttext_loader_scaffold.py`
- `tests/test_v3_round26_fasttext_load_workflow.py`
- `AGENTS.md`
- `CURRENT_STATUS.md`
- `ROUND_V3_R26_REPORT.md`

## Implemented behavior

- `verify_fasttext_seed_checksum(...)`
  - computes SHA256 from a supplied file path
  - compares to expected manifest checksum
  - reads bytes only
  - does not import fastText
  - does not load a model

- `load_fasttext_seed(...)`
  - explicit call only
  - `FileNotFoundError` for missing path
  - `ValueError` for malformed expected checksum
  - `ChecksumMismatchError` for mismatch
  - returns `None` when optional fastText runtime is unavailable
  - imports `fasttext` only inside the explicit load function after file/checksum gates pass

- `mark_seed_as_loaded(...)`
  - returns an explicit `registered` → `loaded` state transition as data
  - requires a registered seed name and non-`None` model handle
  - does not mutate `seeds/MANIFEST.yaml`

- `unload_fasttext_seed(...)`
  - data-only unload boundary
  - no global state mutation

## Safety checks

- default seed state remains `registered`
- explicit load is required for runtime model loading
- explicit mark is required for loaded-state transition
- checksum mismatch fails closed before optional runtime import
- missing file fails closed
- optional dependency absence degrades safely
- manifest remains unchanged
- `self_embedding_adapter.py` remains unchanged
- engine AGP/compositor/speech_hub state remains unchanged

## Non-goals preserved

- no default fastText model load
- no seed binary commit
- no self-embedding rewrite
- no subset extraction
- no AGP runtime change
- no threshold change
- no fallback pool expansion
- no semantic guard keyword addition
- no memory/quarantine modification

## Next recommendation

`v3 round27`: add an explicit operator verification runner or path-based manual workflow for the real `cc.ko.300.bin` artifact. Keep the embedding rewrite postponed until the real-path load has been manually verified.
