# TECHNICAL_MAP

## Runtime mapping surfaces through Round98

- `adapters/lex_concept_mapping_adapter.py`
  - `runtime_mapping_enable_smoke_precheck(...)`: Round96 read-only readiness gate.
  - `controlled_runtime_mapping_enable_smoke(...)`: Round97 ephemeral enable smoke with rollback.
  - `runtime_mapping_persistence_gate_audit(...)`: Round98 read-only persistence gate audit.
  - `stats()`: exposes Round97/Round98 versions and availability.

- `adapters/runtime_smoke_runner.py`
  - `run_round97_controlled_runtime_mapping_enable_smoke(...)`
  - `write_round97_controlled_runtime_mapping_enable_smoke(...)`
  - `run_round98_runtime_mapping_persistence_gate_audit(...)`
  - `write_round98_runtime_mapping_persistence_gate_audit(...)`

- `adapters/state_debug_adapter.py`
  - exposes controlled enable smoke and persistence gate audit versions/availability.

- `main.py`
  - code-only handoff fallback: if the medium 30k vectors file is absent, focused runtime tests may load the preserved small 5k subset; medium/full validation remains blocked/partial and must be reported as such.

## Round97 invariants

- Runtime mapping may be true only during the smoke method.
- Runtime mapping must be false after rollback.
- Enforcement must remain false.
- Ephemeral mapping table must be cleared.
- No AGP verify call is made by the runtime mapping smoke.
- No embedding lookup or EveSpecific vector commit is made by the runtime mapping smoke.

## Round98 invariants

- Audit is read-only.
- Persistence is not applied.
- Operator approval and full validation remain required before persistence.
