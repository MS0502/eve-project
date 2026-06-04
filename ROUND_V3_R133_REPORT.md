# EVE v3 Round133 Report — collection side-effect isolation

Round133 isolated `test_natural_lang_v2.py` collection-time execution without weakening its legacy validation intent.

## Change

The script body now lives in `run_natural_language_v2_validation(...)`; pytest import is collection-safe; `test_natural_language_v2_validation_behavior()` preserves the failure as a runtime test; direct script execution still exits non-zero when validation fails.

## Validation JSON

See `eve_v3_autonomous_handoff/validation/ROUND133_COLLECTION_SIDE_EFFECT_ISOLATION_STATUS.json`.

## Boundaries

No skips, xfails, assertion weakening, runtime behavior fakes, AGP bypass, production persistence enablement, runtime mapping default enablement, or enforcement enablement were added.
