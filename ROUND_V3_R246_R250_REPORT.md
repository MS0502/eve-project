# EVE v3 Round246-250 Internal Report — Validation Manifest

## Cluster selected

- `validation_manifest`
- Rounds: 246, 247, 248, 249, 250

## Work completed

- Added validation command manifest for the Round236-260 handoff line.
- Linked focused tests to the new handoff helper.
- Linked adjacent tests to the Round221-235 runtime-mapping acceptance lineage.
- Recorded broader validation commands for collect-only and compile checks.

## Validation commands recorded

- Focused: `python -m pytest -q tests/test_v3_round236_260_runtime_mapping_acceptance_handoff.py`
- Adjacent: `python -m pytest -q tests/test_v3_round221_225_runtime_mapping_acceptance_delta.py tests/test_v3_round226_230_runtime_mapping_acceptance_taxonomy.py tests/test_v3_round231_235_runtime_mapping_acceptance_repair_selection.py tests/test_v3_round236_260_runtime_mapping_acceptance_handoff.py`
- Collect-only: `python -m pytest --collect-only -q`
- Compile: `python -m compileall -q .`

## Handoff

- Continue to Round251-255 remaining taxonomy if the validation manifest remains aligned with the no-enable policy.
