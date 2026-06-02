# TECHNICAL_MAP

이 파일은 코드 모르는 사용자가 구조를 따라갈 수 있게 만드는 지도다.

## Current source package

The latest generated source package is:

- `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip`

Important: this GitHub handoff directory currently stores the instructions and status. The full generated source package still has to be uploaded or expanded into the repository for Codex to edit it directly.

## Round95 / Round96 code areas

### `adapters/lex_concept_mapping_adapter.py`

Role:

- lexical-to-concept mapping planning
- runtime mapping dry-run
- operator acceptance fixture
- enable-smoke precheck

Added surfaces:

- `runtime_mapping_operator_acceptance_fixture(...)`
- `runtime_mapping_enable_smoke_precheck(...)`

### `adapters/runtime_smoke_runner.py`

Role:

- round-specific runner functions
- export writers
- read-only checks around smoke execution

Added surfaces:

- `run_round95_runtime_mapping_operator_acceptance_fixture(...)`
- `write_round95_runtime_mapping_operator_acceptance_fixture(...)`
- `run_round96_runtime_mapping_enable_smoke_precheck(...)`
- `write_round96_runtime_mapping_enable_smoke_precheck(...)`

### `adapters/state_debug_adapter.py`

Role:

- expose compact runtime state for debug and tests

Added fields:

- `runtime_mapping_operator_acceptance_fixture_version`
- `runtime_mapping_operator_acceptance_fixture_available`
- `runtime_mapping_enable_smoke_precheck_version`
- `runtime_mapping_enable_smoke_precheck_available`

### Tests

- `tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py`
- `tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py`

## Generated artifacts

- `ROUND_V3_R95_REPORT.md`
- `ROUND_V3_R96_REPORT.md`
- `LEXICAL_CONCEPT_RUNTIME_MAPPING_OPERATOR_ACCEPTANCE_FIXTURE_R95.json`
- `LEXICAL_CONCEPT_RUNTIME_MAPPING_ENABLE_SMOKE_PRECHECK_R96.json`
- `LEXICAL_CONCEPT_R96_STATUS.json`
- `ROUND96_VALIDATION_STATUS.json`

## Next code path

Round97 should be a controlled enable-smoke round. It must not silently persist broad runtime behavior. It should first create checkpoint and rollback artifacts, then enable only the minimal accepted mapping path for smoke verification.
