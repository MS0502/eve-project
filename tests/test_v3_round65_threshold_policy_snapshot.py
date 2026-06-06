from __future__ import annotations

import json

from main import build_full_engine


def test_round65_threshold_policy_snapshot_is_read_only_and_tracks_manual_rollback_target() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning

    before_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    before_records = len(learner.commit_audit_records())
    snapshot = learner.threshold_policy_snapshot()
    after_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    after_records = len(learner.commit_audit_records())

    assert snapshot["snapshot_version"] == "v3_round65_threshold_policy_snapshot"
    assert snapshot["round"] == 65
    assert snapshot["read_only"] is True
    assert snapshot["active_policy_version"] == "v3_round69_context_diversity_gate_enforced"
    assert snapshot["current_config"]["min_observations_for_commit"] == 2
    assert snapshot["current_config"]["auto_promotion_enabled"] is False
    assert snapshot["rollback_snapshot"]["manual_reference_only"] is True
    assert snapshot["rollback_snapshot"]["automatic_rollback_enabled"] is False
    assert snapshot["rollback_snapshot"]["requires_operator_patch"] is True
    assert snapshot["rollback_snapshot"]["previous_manual_target"]["min_observations_for_commit"] == 1
    assert snapshot["safety"]["no_auto_rollback"] is True
    assert snapshot["safety"]["no_auto_threshold_adjustment"] is True
    assert before_vectors == after_vectors
    assert before_records == after_records
    assert learner.stats()["min_observations_for_commit"] == 2


def test_round65_threshold_policy_snapshot_export_does_not_change_threshold_or_audit_records(tmp_path) -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")

    before_threshold = learner.stats()["min_observations_for_commit"]
    before_records = len(learner.commit_audit_records())
    before_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    out = tmp_path / "threshold_policy_snapshot.json"
    export = learner.write_threshold_policy_snapshot(out)
    after_threshold = learner.stats()["min_observations_for_commit"]
    after_records = len(learner.commit_audit_records())
    after_vectors = engine.eve_specific_vector_store.stats()["stored_count"]

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert export["snapshot_version"] == "v3_round65_threshold_policy_snapshot"
    assert export["threshold_changed"] is False
    assert export["automatic_rollback_enabled"] is False
    assert export["vector_mutation"] is False
    assert export["audit_record_mutation"] is False
    assert payload["snapshot_version"] == "v3_round65_threshold_policy_snapshot"
    assert payload["rollback_snapshot"]["automatic_rollback_enabled"] is False
    assert before_threshold == after_threshold == 2
    assert before_records == after_records
    assert before_vectors == after_vectors


def test_round65_state_debug_and_drift_report_surface_policy_snapshot() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    stats = learner.stats()
    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert stats["round"] == 71
    assert stats["latest_round"] == 71
    assert stats["implementation_phase"] == "round71_self_learning_policy_consolidation"
    assert stats["commit_threshold_policy_snapshot_version"] == "v3_round65_threshold_policy_snapshot"
    assert stats["commit_threshold_policy_snapshot"]["read_only"] is True
    assert state["eve_self_learning"]["commit_threshold_policy_snapshot_version"] == "v3_round65_threshold_policy_snapshot"
    assert state["eve_self_learning"]["commit_threshold_policy_snapshot"]["rollback_snapshot"]["manual_reference_only"] is True
    assert report["commit_gate"]["threshold_policy_snapshot_version"] == "v3_round65_threshold_policy_snapshot"
    assert report["commit_threshold_policy_snapshot"]["safety"]["no_auto_rollback"] is True
    assert report["policy"]["no_auto_rollback"] is True


def test_round65_snapshot_does_not_weaken_round64_commit_threshold() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.threshold_policy_snapshot()

    single = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    assert single["created"] == []
    assert single["rejected"] == ["민석"]
    assert "insufficient_observations" in single["audit_report"]["candidate_reports"][0]["reasons"]

    learner.observe_text("민석 군대", source="unit_b")
    double = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    assert double["created"] == []
    assert double["rejected"] == ["민석"]
    assert double["audit_report"]["min_observations_for_commit"] == 2
    assert "insufficient_known_context" in double["audit_report"]["candidate_reports"][0]["reasons"]
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False
