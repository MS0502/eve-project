# Round150 — Broader validation taxonomy refresh

Round150 refreshed broader validation after collect-only recovery.

Commands run:

- `python -m compileall -q adapters tests main.py` — passed.
- `python -m pytest -q tests/test_v3_round147_149_legacy_collection_side_effect_isolation.py` — passed (`2 passed`).
- `python -m pytest --collect-only -q` — passed (`1294 tests collected`).
- `python -m pytest -q` — failed with existing broader runtime failures (`212 failed, 1082 passed`).

Taxonomy:

1. Collection blockers: recovered for `test_eve_main_ab.py` and `test_eve_main_abc.py`.
2. Known Korean NaturalLanguage behavior failures remain visible, including `test_natural_lang_v2.py::test_natural_language_v2_validation_behavior` (`8 / 28 passed`) and `tests/test_round2_nl_sd.py::test_nl_intent_overrides_v41_default` (`neutral` vs expected `negative`). Korean inputs and expected behavior were preserved.
3. Seed/vector artifact dependent tests remain failing because vector artifacts such as `seeds/subsets/cc.ko.300.subset.mini.1k/vectors.npy` are absent and must not be fabricated or committed in this PR.
4. Many Round64+ / concept-mapping / runtime-mapping tests cascade from missing EVE-specific vector commit prerequisites. These remain real runtime failures and were not skipped, xfailed, hidden, or patched with dummy vectors.

Blocked/partial validation disposition:

- Broader validation is partial/blocked by real runtime and artifact-dependent failures.
- This PR improves collection hygiene only.
- Production persistence remains NO-GO.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND150_BROADER_VALIDATION_TAXONOMY_REFRESH_STATUS.json`.
