# EVE v3 Round64 Report — commit threshold enforcement

## Summary

Round64 explicitly applies the stricter EVE-specific vector commit threshold that
Round62 dry-ran and Round63 proposed. The active default
`min_observations_for_commit` is now `2`.

This is not automatic promotion and not case hardcoding. It is a general commit
gate hardening rule: an EVE-specific candidate must be observed at least twice
before explicit vector-store mutation can pass.

## Policy

```text
auto_observe_enabled = True
auto_promotion_enabled = False
commit_gate_enabled = True
min_observations_for_commit = 2
min_known_context_words_for_commit = 1
commit_threshold_enforcement_version = v3_round64_min_observations_2_enforced
AGP bypass = False
memory/quarantine mutation = False
fastText seed mutation = False
drift-based runtime change = False
```

## Changed files

- `adapters/eve_self_learning_adapter.py`
- `main.py`
- `adapters/state_debug_adapter.py`
- `adapters/external_seed_manifest.py`
- `tests/test_v3_round64_commit_threshold_enforcement.py`
- Round58–63 adjacent tests updated to reflect the new default threshold.
- `CURRENT_STATUS.md`
- `AGENTS.md`

## Validation

Validation performed in the patch environment:

```text
Round64 focused: 4 passed
Round58–64 focused: 28 passed
Round54–64 adjacent: 66 passed
compileall: passed
collect-only: 1131 tests collected
full suite: attempted, timed out in sandbox after partial progress with no Round64 failure observed
```

## Next

Round65 should add a bounded rollback/config snapshot for the active commit
threshold, still without automatic promotion.
