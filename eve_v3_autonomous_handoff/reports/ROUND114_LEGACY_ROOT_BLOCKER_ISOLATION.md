# Round114 — Legacy root blocker isolation

## Scope

Round114 isolates the root-level collection blocker reported during broader validation without weakening tests.

## Implementation

- Added `build_round114_legacy_root_blocker_isolation(...)`.
- The helper performs a static, read-only scan of root-level `test*.py` files.
- It reports legacy root tests that import the missing `spreading_activation` module.

## Findings

- Root-level legacy collection is blocked by missing `spreading_activation` imports.
- The blocker is outside the Round113-117 runtime mapping sandbox surfaces.
- Focused tests remain the source of truth for this PR.

## Safety results

- No tests were weakened or skipped.
- Production persistence remains disabled.
- Runtime mapping default remains disabled.
- Enforcement default remains disabled.
- No vectors or operator artifacts are produced.

## Validation

- `python -m compileall -q adapters tests main.py` — passed.
- `pytest -q --collect-only` — blocked by pre-existing root legacy imports of missing `spreading_activation`.

## Status

`eve_v3_autonomous_handoff/validation/ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_STATUS.json` records `legacy_root_blocker_isolated`.
