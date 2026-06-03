# ROUND73 PATCH11 REPORT

## Result

- Tests: `563 passed`
- `compileall`: passed
- Test count: `560 -> 563`

## Main change

Patch11 extends the round73 false-route guard from positive/negative object appraisal to external fear, discomfort, and hazard appraisal.

## Fixed route family

These statements now route to `small_talk` through `semantic_guard`, instead of entering `emotional_share` or command fallback:

- `공포영화 무서워`
- `이 의자 불편해`
- `그 상황 위험해`
- `여기 위험하다`
- `이 코드 위험해`

## Preserved emotional cases

These still route to `emotional_share`:

- `나 무서워`
- `기분이 불편해`
- `마음이 불편해`
- `불안하고 걱정돼`
- `너 무서워`

## Files changed

- `adapters/orchestrator_adapter.py`
  - Added `_is_external_threat_or_discomfort_statement()`
  - Added semantic guard before regex and intent fallback
  - Added `불편` to emotional pattern for explicit inner-state cases
- `adapters/state_debug_adapter.py`
  - Added risk label for external threat/discomfort false-emotion cases
- `tests/test_round73_patch11_threat_discomfort_guard.py`
  - Added 3 regression tests

## Guard principle

External target appraisal should not activate empathy routing unless the user names their inner state or an interpersonal target.
