# NEXT_ACTIONS

## Current position

Latest known generated package:

- `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip`

Latest completed round:

- Round96 runtime mapping enable-smoke precheck

Status:

- `runtime_mapping_enabled=False`
- `enforcement_enabled=False`
- Ready token: `민석`
- Full suite still needs long-running validation outside the chat runtime.

## Immediate next steps

### Step 1 — Put source package into the repo

Upload or expand the latest Round96 package into this repository so Codex can edit the real source directly.

Recommended branch:

- `work/autonomous-loop`

### Step 2 — Run validation

Commands to run from the expanded source root:

```bash
python -m compileall adapters/lex_concept_mapping_adapter.py adapters/runtime_smoke_runner.py adapters/state_debug_adapter.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py
pytest -q tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py --disable-warnings --maxfail=1
pytest --collect-only -q
pytest -q --disable-warnings --maxfail=1
```

Expected known results from chat runtime:

- compileall passed
- focused/adjacent tests: 7 passed
- collect-only: 1217 tests collected
- full pytest not completed in chat runtime due time limit

### Step 3 — Round97 controlled runtime mapping enable smoke

Goal:

- Turn on the smallest possible runtime mapping path for the accepted fixture token only.
- Keep rollback and audit explicit.

Required before implementation:

- pre-mutation checkpoint
- rollback report
- invariant checklist
- focused tests
- adjacent tests
- validation JSON

Stop if:

- full validation cannot run
- enabling requires broad refactor
- enabling bypasses AGP
- enabling treats vector evidence as AGP anchor
- tests need weakening

## Phase after Round97

1. AGP proof object expansion
2. Progress/Feasibility check for autonomous loop
3. false coherence diagnostics
4. event/frame graph later
