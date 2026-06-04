# Round148 — Legacy root collection side-effect isolation

Round148 isolated the collection-time script side effects in the two legacy root validation files.

Changes:

- `test_eve_main_ab.py` now keeps its historical Korean validation body inside `run_legacy_validation()` and executes it only from a `__main__` guard.
- `test_eve_main_abc.py` now keeps its historical Korean validation body inside `run_legacy_validation()` and executes it only from a `__main__` guard.
- `tests/test_v3_round147_149_legacy_collection_side_effect_isolation.py` adds focused checks that importing both legacy modules is quiet and successful, and that explicit validation entrypoints/main guards remain present.

Isolation decision:

- The legacy validations remain available for intentional script execution.
- Pytest collection no longer loads operator-local belief files or runs legacy validation scenarios during import.
- Real runtime validation issues are not hidden: the AB missing external `/home/claude/eve/beliefs.json` dependency and ABC dict/object mismatch remain in the explicit script path.

Guardrails:

- No production persistence enablement.
- No runtime mapping default change.
- No enforcement enablement.
- No AGP bypass.
- No Korean behavior examples were translated or replaced.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND148_LEGACY_ROOT_COLLECTION_SIDE_EFFECT_ISOLATION_STATUS.json`.
