# Round196 broader validation delta and next recommendation

Round196 records the expected validation delta. Focused guarded integration tests pass locally in the PR environment. Broader full pytest is still expected to remain red without operator-local artifacts and broader historical repairs. The operator-local validation command for real artifacts remains:

```bash
python scripts/operator_validate_medium30k.py --attempt-load
```

Next recommendation: in Codespaces where the real artifact is present and the command above is green, call `build_full_engine(operator_medium30k_validation=<green report>, operator_medium30k_load_authorized=True, operator_medium30k_artifact_dir=...)`, then remeasure the seed/vector and EVE-specific vector/self-learning cascades. Do not enable production persistence, runtime mapping defaults, or enforcement.

## Validation run in PR environment

- `python -m compileall -q adapters tests main.py scripts`: passed.
- `python -m pytest --collect-only -q`: passed, 1328 tests collected.
- `python -m pytest -q tests/test_v3_round192_196_guarded_medium30k_integration.py tests/test_v3_round187_191_operator_validate_medium30k.py tests/test_v3_round182_186_explicit_load_guard.py`: passed, 18 tests.
- `python scripts/operator_validate_medium30k.py --attempt-load`: blocked in Codex Cloud because real operator artifacts are absent; operator-local Codespaces evidence remains green.
- `python -m pytest -q`: red, 206 failed and 1122 passed. This is recorded honestly as the artifact/self-learning/runtime-mapping cascade remains unresolved in the PR environment.
