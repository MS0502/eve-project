# EVE v3 Round66 Report — Observation Evidence Quality Summary

## Goal

Add a read-only quality surface for EVE-specific observation evidence before
considering any future context-diversity gate.

Round64 already enforced `min_observations_for_commit = 2`, and Round65 made the
threshold policy visible. Round66 keeps that policy unchanged and adds a summary
that answers: did the repeated observations come from varied contexts or just the
same context repeated?

## Changes

### `adapters/eve_self_learning_adapter.py`

Added:

```python
observation_evidence_quality_summary(
    words=None,
    context_sample_limit=3,
    peer_token_sample_limit=10,
)
```

The summary reports:

- observed count per candidate
- current threshold readiness
- unique context count
- unique source count
- unique text count
- unique peer-token count
- duplicate-context count
- evidence status
- bounded peer-token and context samples

Round metadata updated:

```text
implementation_phase = round66_observation_evidence_quality_summary
observation_evidence_quality_version = v3_round66_observation_evidence_quality_summary
stats.round = 66
stats.latest_round = 66
```

### `adapters/state_debug_adapter.py`

Exposes:

```text
observation_evidence_quality_version
observation_evidence_quality
```

under `state["eve_self_learning"]`.

### `adapters/external_seed_manifest.py`

`measure_eve_self_learning_drift_accumulation(engine)` now includes the
Round66 evidence-quality summary and version in the commit-gate section.

### Tests

Added:

```text
tests/test_v3_round66_observation_evidence_quality.py
```

Coverage:

1. context-diverse repeated observations are detected
2. repeated same-context observations are flagged
3. state-debug and drift report expose the Round66 surface

Older Round58–65 focused tests were updated only where they asserted the latest
round/implementation metadata.

## Safety properties

Round66 is read-only.

```text
min_observations_for_commit = 2 unchanged
auto_promotion_enabled = False unchanged
commit_gate_enabled = True unchanged
no vector-store mutation
no audit-record append
no fastText seed mutation
no memory/quarantine mutation
no AGP bypass
no drift-based runtime change
no automatic threshold adjustment
```

The new context-diversity status is only an observation signal. It is not used
to block or allow commits yet.

## Validation

```text
pytest -q tests/test_v3_round66_observation_evidence_quality.py
3 passed

pytest -q tests/test_v3_round58_continuous_eve_self_learning.py \
          tests/test_v3_round59_commit_gate.py \
          tests/test_v3_round60_commit_audit_export.py \
          tests/test_v3_round61_commit_audit_dashboard.py \
          tests/test_v3_round62_commit_threshold_dry_run.py \
          tests/test_v3_round63_threshold_proposal_report.py \
          tests/test_v3_round64_commit_threshold_enforcement.py \
          tests/test_v3_round65_threshold_policy_snapshot.py \
          tests/test_v3_round66_observation_evidence_quality.py
35 passed

pytest -q tests/test_v3_round54_eve_vocab_tracker_observe.py \
          tests/test_v3_round55_eve_vector_store.py \
          tests/test_v3_round56_wrapper_eve_specific_integration.py \
          tests/test_v3_round57_post_eve_specific_smoke.py \
          tests/test_v3_round58_continuous_eve_self_learning.py \
          tests/test_v3_round59_commit_gate.py \
          tests/test_v3_round60_commit_audit_export.py \
          tests/test_v3_round61_commit_audit_dashboard.py \
          tests/test_v3_round62_commit_threshold_dry_run.py \
          tests/test_v3_round63_threshold_proposal_report.py \
          tests/test_v3_round64_commit_threshold_enforcement.py \
          tests/test_v3_round65_threshold_policy_snapshot.py \
          tests/test_v3_round66_observation_evidence_quality.py
73 passed

python3 -m compileall -q .
passed

pytest --collect-only -q
1138 tests collected
```

Full suite was not completed in the sandbox. No Round66 failure was observed in
focused/adjacent validation.

## Next recommendation

Round67: add a read-only context-diversity gate dry-run/report. Do not enforce
context-diversity in the commit gate yet.
