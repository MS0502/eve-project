# EVE v3 Round132 Report — NaturalLanguage v2 SystemExit diagnosis

Round132 diagnosed the next collection blocker after the Round127-131 working-memory recovery.

## Finding

`pytest --collect-only -q` previously reached `test_natural_lang_v2.py`, executed the legacy script-style validation during module import, and hit `sys.exit(1)` after the NaturalLanguage v2 validation reported 8/28 checks passing.

## Decision

Proceed to Round133 isolation. Do not weaken, skip, xfail, or delete the legacy validation.

## Validation JSON

See `eve_v3_autonomous_handoff/validation/ROUND132_NATURAL_LANG_V2_SYSTEM_EXIT_DIAGNOSIS_STATUS.json`.

## Boundaries

Production persistence remains disabled. Runtime mapping defaults and enforcement remain disabled. No AGP bypass or vector artifact changes were made.
