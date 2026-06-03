"""EVE v3 round76 — self-learning v1 freeze baseline.

Round76 freezes the Round57-75 self-learning safety pipeline as v1 baseline.
It does not add a learning shortcut or change policy. The only allowed mutation
in the runner is the same Round75 proof-path explicit commit; the freeze layer
itself is read-only.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    build_round76_self_learning_v1_freeze_baseline,
    run_round76_self_learning_v1_freeze_baseline,
    write_round76_self_learning_v1_freeze_baseline,
)


def test_round76_freeze_baseline_locks_active_policy_and_artifacts() -> None:
    engine = build_full_engine()

    report = run_round76_self_learning_v1_freeze_baseline(engine)

    assert report["freeze_baseline_version"] == "v3_round76_self_learning_v1_freeze_baseline"
    assert report["baseline_round"] == 76
    assert report["baseline_name"] == "self_learning_v1"
    assert report["frozen"] is True
    assert report["source_replay_export_version"] == "v3_round75_commit_audit_replay_export"
    assert report["source_delta_report_version"] == "v3_round74_explicit_commit_drift_telemetry_delta"
    assert report["source_commit_smoke_version"] == "v3_round73_explicit_eve_specific_commit_smoke"

    active = report["active_policy"]
    assert active["auto_observe_enabled"] is True
    assert active["auto_promotion_enabled"] is False
    assert active["commit_gate_enabled"] is True
    assert active["min_observations_for_commit"] == 2
    assert active["min_known_context_words_for_commit"] == 1
    assert active["context_diversity_gate_enabled"] is True
    assert active["automatic_rollback_enabled"] is False
    assert active["agp_anchor_policy"] == "explicit_categories_plus_sa_activation_only"

    artifacts = set(report["artifact_manifest"])
    assert "SELF_LEARNING_POLICY_ROUND71.md" in artifacts
    assert "EVE_SPECIFIC_BASELINE_R72.json" in artifacts
    assert "EVE_SPECIFIC_COMMIT_SMOKE_R73.json" in artifacts
    assert "EVE_SPECIFIC_COMMIT_DELTA_R74.json" in artifacts
    assert "EVE_SPECIFIC_COMMIT_REPLAY_EXPORT_R75.json" in artifacts
    assert "SELF_LEARNING_V1_FREEZE_BASELINE_R76.json" in artifacts


def test_round76_freeze_invariants_and_read_only_checks_hold() -> None:
    engine = build_full_engine()

    report = run_round76_self_learning_v1_freeze_baseline(engine)
    invariants = report["freeze_invariants"]

    assert all(invariants.values())
    checks = report["freeze_read_only_checks"]
    assert checks["audit_records_unchanged_during_freeze"] is True
    assert checks["vector_store_unchanged_during_freeze"] is True
    assert checks["telemetry_unchanged_during_freeze"] is True
    assert checks["policy_changed_during_freeze"] is False

    policy = report["policy"]
    assert policy["freeze_does_not_call_commit"] is True
    assert policy["freeze_does_not_call_lookup"] is True
    assert policy["freeze_does_not_append_audit"] is True
    assert policy["freeze_does_not_mutate_vectors"] is True
    assert policy["thresholds_changed"] is False
    assert policy["context_diversity_gate_changed"] is False
    assert policy["auto_promotion_enabled"] is False
    assert policy["agp_bypass"] is False


def test_round76_learner_freeze_snapshot_is_role_and_gate_source_of_truth() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    audit_before = len(learner.commit_audit_records())
    store_before = engine.eve_specific_vector_store.stats().copy()

    snapshot = learner.self_learning_v1_freeze_policy_snapshot()

    assert snapshot["freeze_version"] == "v3_round76_self_learning_v1_freeze_baseline"
    assert snapshot["round"] == 76
    assert snapshot["frozen_baseline_name"] == "self_learning_v1"
    assert snapshot["active_policy_version"] == "v3_round69_context_diversity_gate_enforced"
    assert snapshot["component_roles"]["EveVocabTracker"] == "lexical_observation_only_no_vector_generation"
    assert snapshot["component_roles"]["EveSelfLearningAdapter"] == "continuous_observation_and_explicit_commit_gate_owner"
    assert snapshot["component_roles"]["AGPAdapter"] == "anchor_validation_from_explicit_categories_and_sa_activation_only"
    assert snapshot["locked_commit_gate"] == [
        "observed_count >= 2",
        "fastText-OOV / EVE-specific candidate",
        "known_fastText_context_words >= 1",
        "context_diverse == True",
        "explicit_commit_call_required",
    ]
    assert snapshot["read_only_checks"]["audit_records_unchanged"] is True
    assert snapshot["read_only_checks"]["vector_store_unchanged"] is True
    assert len(learner.commit_audit_records()) == audit_before
    assert engine.eve_specific_vector_store.stats() == store_before


def test_round76_build_and_write_export_do_not_recompute_or_mutate(tmp_path) -> None:
    engine = build_full_engine()
    report = run_round76_self_learning_v1_freeze_baseline(engine)
    audit_before = len(engine.eve_self_learning.commit_audit_records())
    store_before = engine.eve_specific_vector_store.stats().copy()
    telemetry_before = engine.self_embedding.telemetry().copy()

    rebuilt = build_round76_self_learning_v1_freeze_baseline(
        {"replay_export_version": "v3_round75_commit_audit_replay_export"},
        engine.eve_self_learning.self_learning_v1_freeze_policy_snapshot(),
        engine.self_embedding.lookup_route_policy_snapshot(),
    )
    assert rebuilt["freeze_baseline_version"] == "v3_round76_self_learning_v1_freeze_baseline"
    assert len(engine.eve_self_learning.commit_audit_records()) == audit_before
    assert engine.eve_specific_vector_store.stats() == store_before
    assert engine.self_embedding.telemetry() == telemetry_before

    path = tmp_path / "self_learning_v1_freeze.json"
    result = write_round76_self_learning_v1_freeze_baseline(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round76_self_learning_v1_freeze_baseline"
    assert result["recomputed"] is False
    assert result["commit_called"] is False
    assert result["vector_mutation"] is False
    assert result["audit_record_mutation"] is False
    assert result["policy_changed"] is False
    assert loaded["freeze_baseline_version"] == "v3_round76_self_learning_v1_freeze_baseline"
    assert len(engine.eve_self_learning.commit_audit_records()) == audit_before
    assert engine.eve_specific_vector_store.stats() == store_before
    assert engine.self_embedding.telemetry() == telemetry_before
