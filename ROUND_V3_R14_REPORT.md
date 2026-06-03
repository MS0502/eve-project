# EVE v3 Round14 Report — Explicit Compositor AGP Veto

## Result

- Previous stable: v3 round13 — `623 passed`
- Current stable: v3 round14 — `630 passed`
- `compileall` passed
- Regression: 0

## Scope

Round14 is the first EVE v3 round where AGP can change an output, but only in compositor veto mode and only after explicit mode switches.

Default runtime remains observation-only.

## Files changed

- `adapters/agp_adapter.py`
- `adapters/compositor_adapter.py`
- `tests/test_v3_round14_agp_compositor_veto.py`
- `CURRENT_STATUS.md`
- `ROUND_V3_R14_REPORT.md`

## Implementation notes

### `AGPAdapter`

Added:

- mode validation
- `set_mode(mode)` explicit mode switch
- `AGP_FALLBACK_SURFACE_POOL`
- `fallback_to_surface(result)` minimal surface conversion

Preserved:

- default mode: `observation`
- default thresholds: `0.3` / `0.5`
- no analyzer recommendation auto-apply
- no automatic veto activation

### `CompositorAdapter`

Added:

- explicit compositor veto path
- `agp_veto_count`
- trace records effective mode: `observation` or `veto`
- veto requires both compositor and AGP adapter to be in `AGP_MODE_VETO`

Behavior:

- observation mode: output unchanged
- veto mode + AGP pass: output unchanged
- veto mode + AGP fail: output replaced with minimal fallback surface

## Tests added

`tests/test_v3_round14_agp_compositor_veto.py`

- `test_default_engine_and_compositor_remain_observation_mode`
- `test_observation_mode_keeps_existing_compositor_output_even_on_fail`
- `test_veto_mode_pass_keeps_candidate_output`
- `test_veto_mode_fail_replaces_candidate_with_fallback_surface`
- `test_veto_requires_explicit_mode_switch_on_adapter_and_compositor`
- `test_fallback_surface_pool_stays_minimal_and_raw_text_independent`
- `test_trace_records_veto_mode_without_touching_speech_hub_mode`

## Guardrails

- Default engine remains observation-only.
- Veto requires explicit mode switches.
- SpeechHub remains observation-only.
- Fallback surface pool is intentionally tiny.
- Fallback does not branch on raw candidate text.
- No semantic guard keyword expansion.
- No memory/quarantine changes.

## Next recommended round

v3 round15:

- speech_hub veto activation, explicitly gated by `AGP_MODE_VETO`
- default engine remains observation-only
- fallback surface pool remains minimal
- no semantic guard expansion
