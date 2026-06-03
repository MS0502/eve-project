# EVE v3 Round65 Report — threshold config / rollback snapshot

## Summary

Round65 adds a read-only threshold policy snapshot for the active EVE-specific
vector commit gate. The active policy remains the Round64 enforcement:
`min_observations_for_commit = 2`.

This round does not roll back anything automatically. It only exposes the
current threshold config, the policy history, and a manual rollback reference so
an operator can inspect what would need to change in a later explicit patch.

## Policy

```text
auto_observe_enabled = True
auto_promotion_enabled = False
commit_gate_enabled = True
min_observations_for_commit = 2
min_known_context_words_for_commit = 1
active_policy_version = v3_round64_min_observations_2_enforced
commit_threshold_policy_snapshot_version = v3_round65_threshold_policy_snapshot
automatic_rollback_enabled = False
AGP bypass = False
memory/quarantine mutation = False
fastText seed mutation = False
drift-based runtime change = False
```

## Added functions

```text
threshold_policy_snapshot()
write_threshold_policy_snapshot(path)
```

`threshold_policy_snapshot()` returns:

```text
- current commit threshold config
- Round58→65 threshold policy history
- previous manual rollback target: Round59 gate, min_observations_for_commit=1
- explicit safety flags showing no auto rollback / no auto promotion
```

`write_threshold_policy_snapshot(path)` writes the same snapshot to JSON as an
operator artifact only. It does not change thresholds, append audit records, or
create vectors.

## Changed files

- `adapters/eve_self_learning_adapter.py`
- `adapters/state_debug_adapter.py`
- `adapters/external_seed_manifest.py`
- `tests/test_v3_round65_threshold_policy_snapshot.py`
- Updated Round62–64 adjacent tests to reflect latest round 65.
- `CURRENT_STATUS.md`
- `AGENTS.md`

## Validation

Validation performed in the patch environment:

```text
Round65 focused: 4 passed
Round58–65 focused: 32 passed
Round54–65 adjacent: 70 passed
compileall: passed
collect-only: 1135 tests collected
full suite: attempted, timed out/killed in sandbox during early progress with no Round65 failure observed
```

## Next

Round66 should add a bounded observation-evidence quality summary, such as
context diversity per candidate, without changing the active threshold or
enabling automatic promotion.
