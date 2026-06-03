# EVE v3 round15 report

## Scope

SpeechHub AGP veto activation with explicit double lock.

## Base

- Base artifact: `eve_v3_round14_passed.zip`
- Previous status: v3 round14, `630 passed`

## Changes

- `adapters/speech_hub.py`
  - Added AGP veto path mirroring compositor round14 pattern.
  - Default remains `AGP_MODE_OBSERVATION`.
  - Added `agp_veto_count`.
  - Veto requires both:
    - `engine.agp_adapter.mode == AGP_MODE_VETO`
    - `engine.speech_hub.agp_mode == AGP_MODE_VETO`
  - Added fallback surface replacement only in explicit veto mode.
  - Added metadata-based already-fallback detection to prevent double replacement.
  - Existing fallback surfaces are never inferred from raw text alone.

- `tests/test_v3_round15_agp_speech_hub_veto.py`
  - Default engine remains observation-only.
  - Veto pass keeps output.
  - Veto fail uses fallback surface.
  - Double-lock required.
  - Already-applied fallback is not reverified/replaced.
  - Fallback surface pool stays minimal.
  - SpeechHub veto does not change compositor mode.

## Non-goals

- No default veto activation.
- No fallback pool expansion.
- No new AGP reasons.
- No semantic guard keyword additions.
- No threshold tuning.
- No memory/quarantine changes.

## Validation

- `pytest`
- `compileall`

## Next

v3 round16 should begin data-backed AGP stabilization:

- verify trace behavior under both veto layers
- inspect pass/fail distributions
- keep threshold changes manual only
