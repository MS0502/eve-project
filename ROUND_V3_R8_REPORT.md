# EVE v3 Round8 Report

## Summary

Round8 adds AGP observation-mode integration to the compositor.

This is trace-only. It does not activate veto, fallback, route changes, or output changes.

## Files changed

- `adapters/agp_adapter.py`
  - Added explicit AGP mode constants.
  - `AGPAdapter` now defaults to `mode="observation"`.
- `adapters/compositor_adapter.py`
  - Added AGP observation trace fields.
  - `composition_phrase(...)` now calls AGP in observation mode after producing the same candidate phrase.
  - Added bounded in-memory `agp_trace`.
- `main.py`
  - Wires `engine.agp_adapter` in observation mode before compositor use.
- `tests/test_v3_round8_agp_compositor_observation.py`
  - Added compositor observation-mode tests.
- `CURRENT_STATUS.md`
  - Updated to v3 round8.

## Tests added

- `test_agp_mode_default_is_observation`
- `test_compositor_calls_agp_verify_in_observation_mode`
- `test_agp_result_recorded_in_trace_without_fallback_activation`
- `test_output_unchanged_when_agp_fails_in_observation_mode`
- `test_build_full_engine_wires_agp_for_compositor_observation_only`

## Guardrails

- No compositor output changes.
- No fallback activation.
- No veto activation.
- No speech_hub integration.
- No semantic guard keyword additions.
- No memory/quarantine changes.

## Next

v3 round9 should add speech_hub observation-mode integration with the same constraints:
trace only, output unchanged, fallback unused, veto inactive.
