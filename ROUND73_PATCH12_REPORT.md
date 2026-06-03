# Round73 Patch12 Report

## Result

- Tests: `566 passed in 2.63s`
- `compileall`: passed
- Test count: `563 -> 566`

## Focus

Patch12 extends the semantic route guard from direct object appraisal into affective-tone appraisal.

## Fixed

External works/scenes that merely carry an affective tone no longer route as `emotional_share`:

- `영화 슬프다`
- `이 장면 슬퍼`
- `이 이야기는 우울하다`
- `그 노래 외로워`

These now route to:

- `small_talk`
- `via=semantic_guard`
- `matched_pattern=external_affective_tone_statement`

## Preserved

Actual user affect/reaction remains `emotional_share`:

- `나 슬퍼`
- `기분이 우울해`
- `영화 보고 슬펐어`
- `이 노래 들으니까 외로워`

## Notes

This also blocks a broad teaching-pattern capture where `이 이야기는 우울하다` could be misread as a definition/teaching statement.
