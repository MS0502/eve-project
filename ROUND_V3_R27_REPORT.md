# EVE v3 Round27 Report — Seed Verification Runner

## Result

- Base: v3 round26 (`725 passed`)
- Current: v3 round27 (`735 passed`)
- `compileall` passed
- Scope: fastText seed checksum verification runner only

## Changes

- Extended `adapters/fasttext_loader.py`:
  - `EVE_FASTTEXT_SEED_PATH_ENV`
  - `get_configured_seed_path()`
  - `seed_verification_runner(...)`
- Added `docs/FASTTEXT_SEED_VERIFICATION_WORKFLOW.md`.
- Added `tests/test_v3_round27_seed_verification_runner.py`.
- Updated `CURRENT_STATUS.md`.
- Updated `AGENTS.md`.

## Verification runner contract

The runner:

1. resolves `explicit_path` first;
2. falls back to `EVE_FASTTEXT_SEED_PATH`;
3. returns `skipped` if no path is configured;
4. returns `missing_file` for absent paths;
5. returns `mismatch` for checksum mismatch;
6. returns `verified` only when the computed checksum matches the manifest checksum;
7. never imports or loads fastText;
8. never rewrites self embedding;
9. never mutates the manifest or runtime AGP state.

## Non-goals preserved

- No fastText model load.
- No seed binary commit.
- No self-embedding rewrite.
- No subset extraction.
- No AGP runtime change.
- No threshold change.
- No fallback pool expansion.
- No semantic guard keyword addition.
- No memory/quarantine modification.

## Next recommendation

Run the verification runner in an environment that can access the real `cc.ko.300.bin` path. If it returns `verified`, proceed to subset extraction planning in a later round.
