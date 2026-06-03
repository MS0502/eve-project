from __future__ import annotations

from main import build_full_engine


def test_round63_threshold_proposal_blocks_single_observation_without_policy_change() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")

    before_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    before_records = len(learner.commit_audit_records())
    proposal = learner.threshold_proposal_report(
        words=["민석"],
        context_words=["오늘", "군대"],
        target_observation_threshold=2,
    )
    after_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    after_records = len(learner.commit_audit_records())

    assert proposal["proposal_version"] == "v3_round63_threshold_proposal_report"
    assert proposal["round"] == 63
    assert proposal["read_only"] is True
    assert proposal["active_policy_unchanged"] is True
    assert proposal["audit_records_unchanged"] is True
    assert proposal["current_min_observations_for_commit"] == 2
    assert proposal["proposed_min_observations_for_commit"] == 2
    assert proposal["blocked_under_proposal"] == ["민석"]
    assert proposal["eligible_under_proposal"] == []
    assert proposal["recommendation"] == "already_at_or_above_target"
    assert proposal["operator_review_required"] is True
    assert proposal["policy"]["auto_promotion_enabled"] is False
    assert proposal["policy"]["no_threshold_change"] is True
    assert before_vectors == after_vectors
    assert before_records == after_records
    assert learner.stats()["min_observations_for_commit"] == 2


def test_round63_threshold_proposal_marks_ready_but_still_requires_operator_review() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.observe_text("민석", source="unit")

    proposal = learner.threshold_proposal_report(
        words=["민석"],
        context_words=["오늘", "군대"],
        target_observation_threshold=2,
    )

    assert proposal["eligible_under_proposal"] == ["민석"]
    assert proposal["blocked_under_proposal"] == []
    assert proposal["recommendation"] == "already_at_or_above_target"
    assert proposal["operator_review_required"] is True
    assert proposal["active_policy_unchanged"] is True
    assert learner.stats()["min_observations_for_commit"] == 2


def test_round63_state_debug_and_drift_report_surface_threshold_proposal() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    stats = learner.stats()
    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)
    dashboard = learner.commit_audit_dashboard_snapshot()

    assert stats["round"] == 71
    assert stats["latest_round"] == 71
    assert stats["commit_threshold_proposal_version"] == "v3_round63_threshold_proposal_report"
    assert stats["commit_threshold_proposal"]["read_only"] is True
    assert dashboard["threshold_proposal_version"] == "v3_round63_threshold_proposal_report"
    assert state["eve_self_learning"]["commit_threshold_proposal_version"] == "v3_round63_threshold_proposal_report"
    assert report["commit_gate"]["threshold_proposal_version"] == "v3_round63_threshold_proposal_report"
    assert report["commit_threshold_proposal"]["operator_review_required"] is True
