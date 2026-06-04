# Round194 guarded integration implementation

Round194 adds `adapters/medium30k_runtime_load_integration.py` and wires `main.build_full_engine` to call `apply_round194_guarded_medium30k_runtime_load(...)`.

Default behavior remains no-load. The helper calls `guarded_explicit_medium30k_load(..., attempt_load=True, attach_to_engine=True)` only when all of these are true:

1. Operator validation report is green.
2. Operator validation was run with `--attempt-load` and reported `explicit_load_succeeded`.
3. Caller passes `operator_medium30k_load_authorized=True`.

The integration attaches a loaded adapter only after the existing guard succeeds. It does not enable production persistence, runtime mapping, enforcement, or AGP bypass.
