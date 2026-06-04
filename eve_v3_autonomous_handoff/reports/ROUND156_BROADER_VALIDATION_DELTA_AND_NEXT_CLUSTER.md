# Round156 — Broader validation delta and next cluster recommendation

Round156 reran required validation after the focused NaturalLanguage v2 fix.

Commands run:

- `python -m compileall -q adapters tests main.py` — passed.
- `python -m pytest --collect-only -q` — passed (`1294 tests collected in 1.08s`).
- `python -m pytest -q test_natural_lang_v2.py tests/test_round2_nl_sd.py::test_nl_intent_overrides_v41_default` — passed (`2 passed`).
- `python -m pytest -q` — failed with remaining broader runtime/artifact failures (`210 failed, 1084 passed in 24.78s`).

Broader validation delta:

| Metric | Before Round154 | After Round154 | Delta |
| --- | ---: | ---: | ---: |
| Failed tests | 212 | 210 | -2 |
| Passed tests | 1082 | 1084 | +2 |
| Collected tests | 1294 | 1294 | 0 |

Failures fixed:

1. `test_natural_lang_v2.py::test_natural_language_v2_validation_behavior`
2. `tests/test_round2_nl_sd.py::test_nl_intent_overrides_v41_default`

Remaining taxonomy after Round156:

| Category | Remaining count | Recommendation |
| --- | ---: | --- |
| Seed/vector artifact and fastText wrapper cascade | 127 | Next safest cluster is an artifact-presence/readiness reporting improvement only, unless the operator restores real vectors outside the PR. Do not fabricate or commit vectors. |
| EVE-specific vector/self-learning cascade | 40 | Defer until real fastText context vector availability is resolved. |
| Concept/runtime mapping cascade | 43 | Defer; runtime mapping and enforcement remain disabled, and concept evidence depends on earlier vector/self-learning prerequisites. |

Final recommendation for the next failure cluster:

- Address the seed/vector artifact dependency cluster through honest readiness/error reporting or operator artifact restoration, not dummy vectors.
- If code changes are needed before artifact restoration, keep them diagnostic/read-only and avoid changing runtime mapping defaults, enforcement defaults, AGP thresholds, semantic memory, quarantine, or seed files.
- Production persistence remains **NO-GO**.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND156_BROADER_VALIDATION_DELTA_AND_NEXT_CLUSTER_STATUS.json`.
