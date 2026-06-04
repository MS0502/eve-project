# Round152 — Broader failure taxonomy from full pytest

Round152 started from the post-PR #21 merged state available in this checkout. There is no configured `origin` remote or local `main` branch in this environment, so the current `work` branch at merge commit `6eeb02b` was treated as the latest merged baseline.

Commands run:

- `python -m pytest -q` — failed with existing broader runtime failures (`212 failed, 1082 passed in 31.21s`).

Taxonomy from the full pytest result:

| Category | Count | Evidence |
| --- | ---: | --- |
| Korean NaturalLanguage v2 behavior | 2 | `test_natural_lang_v2.py::test_natural_language_v2_validation_behavior` reported `8 / 28 passed`; `tests/test_round2_nl_sd.py::test_nl_intent_overrides_v41_default` reported `neutral` instead of Korean negative sentiment. |
| Seed/vector artifact and fastText wrapper cascade | 127 | Round29-44 and Round50-52 tests still depend on absent `vectors.npy` subset artifacts and fastText wrapper load/telemetry paths. No dummy vectors were created. |
| EVE-specific vector/self-learning cascade | 40 | Round54-75 tests cascade from known fastText context/vector-store prerequisites that remain unavailable while vector artifacts are absent. |
| Concept/runtime mapping cascade | 43 | Round78-98 concept and runtime-mapping surfaces cascade from the absent EVE-specific vector/concept commit prerequisites. Runtime mapping remains disabled. |

Disposition:

- Collection is still green from the prior recovery work, but full pytest remains red.
- Production persistence remains **NO-GO**.
- `runtime_mapping_enabled` default remains false.
- `enforcement_enabled` remains false.
- Korean behavior failures were preserved and made visible; no tests were weakened, skipped, xfailed, deleted, or translated.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND152_BROADER_FAILURE_TAXONOMY_STATUS.json`.
