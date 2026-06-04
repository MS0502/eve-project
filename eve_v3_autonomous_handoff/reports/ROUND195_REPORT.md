# Round195 focused integration tests

Round195 adds focused tests in `tests/test_v3_round192_196_guarded_medium30k_integration.py`. These tests use mocks/fakes only for guard-call and adapter-attachment behavior. They do not fabricate vector contents and do not require real artifacts in Codex Cloud.

Covered cases:

- Round192 entrypoint diagnosis.
- Round193 path-selection constraints.
- Round194 default no-load behavior.
- Round194 no-load-only validation blocking.
- Round194 green validation plus explicit authorization calling the guard.
- `main.build_full_engine` forwarding explicit operator authorization only.
