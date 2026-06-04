# Round147 — Legacy root collection side-effect diagnosis

Round147 diagnosed the next `pytest --collect-only -q` blockers after Round142-146 recovery.

Findings:

- `test_eve_main_ab.py` executed the historical v32 validation script at module import time. During pytest collection it instantiated `EVE_TierAB` and reached `learn_beliefs(path='/home/claude/eve/beliefs.json')`, which failed because that operator-local file is not present in this repository.
- `test_eve_main_abc.py` also executed historical validation script code at module import time. Collection reached a scenario that passed dict-shaped mini beliefs into `eve_main_abc.EVE_TierAB.learn_beliefs(...)`, which currently expects belief objects and raised `AttributeError: 'dict' object has no attribute 'is_innate'`.
- These were collection-time side effects, not pytest test failures from collected test functions. They hid downstream collection status.

Decision for Round148:

- Move historical script execution behind explicit `run_legacy_validation()` entrypoints and `if __name__ == "__main__"` guards.
- Preserve Korean examples and legacy validation intent. Do not skip, xfail, delete, or weaken the validation body.
- Do not fabricate `/home/claude/eve/beliefs.json` and do not patch runtime behavior around the dict/object mismatch in this round.

Guardrails:

- Production persistence remains NO-GO.
- `runtime_mapping_enabled` default remains false.
- `enforcement_enabled` remains false.
- AGP was not bypassed.
- No vectors, seed subsets, zip files, part files, or `_operator_artifacts` were added.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND147_LEGACY_ROOT_COLLECTION_SIDE_EFFECT_DIAGNOSIS_STATUS.json`.
