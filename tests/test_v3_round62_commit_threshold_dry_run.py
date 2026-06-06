from __future__ import annotations

from main import build_full_engine


def _threshold_report(dry_run: dict, threshold: int) -> dict:
    for report in dry_run["threshold_reports"]:
        if report["observation_threshold"] == threshold:
            return report
    raise AssertionError(f"missing threshold {threshold}")


def test_round62_threshold_dry_run_blocks_single_observation_without_policy_change() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")

    before_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    before_records = len(learner.commit_audit_records())
    dry_run = learner.dry_run_commit_thresholds(
        words=["민석"],
        context_words=["오늘", "군대"],
        observation_thresholds=[2, 3],
    )
    after_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    after_records = len(learner.commit_audit_records())

    assert dry_run["dry_run_version"] == "v3_round62_multi_observation_commit_threshold_dry_run"
    assert dry_run["read_only"] is True
    assert dry_run["active_policy_unchanged"] is True
    assert dry_run["active_min_observations_for_commit"] == 2
    assert dry_run["policy"]["auto_promotion_enabled"] is False
    assert dry_run["policy"]["no_threshold_change"] is True
    assert before_vectors == after_vectors
    assert before_records == after_records

    threshold2 = _threshold_report(dry_run, 2)
    assert threshold2["eligible_words"] == []
    assert threshold2["rejected_words"] == ["민석"]
    assert "insufficient_observations" in threshold2["candidate_reports"][0]["reasons"]


def test_round62_threshold_dry_run_still_requires_loaded_known_context() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.observe_text("민석", source="unit")

    dry_run = learner.dry_run_commit_thresholds(
        words=["민석"],
        context_words=["오늘", "군대"],
        observation_thresholds=[2, 3],
    )

    threshold2 = _threshold_report(dry_run, 2)
    threshold3 = _threshold_report(dry_run, 3)
    assert threshold2["eligible_words"] == []
    assert threshold2["rejected_words"] == ["민석"]
    assert threshold2["gate_pass"] is False
    assert "insufficient_known_context" in threshold2["candidate_reports"][0]["reasons"]
    assert threshold3["eligible_words"] == []
    assert threshold3["rejected_words"] == ["민석"]
    assert "insufficient_observations" in threshold3["candidate_reports"][0]["reasons"]
    assert learner.stats()["min_observations_for_commit"] == 2


def test_round62_dashboard_state_debug_and_drift_report_surface_threshold_readiness() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    dashboard = learner.commit_audit_dashboard_snapshot()
    stats = learner.stats()
    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert stats["round"] == 71
    assert stats["commit_threshold_dry_run_version"] == "v3_round62_multi_observation_commit_threshold_dry_run"
    assert stats["commit_threshold_readiness_version"] == "v3_round62_commit_threshold_readiness"
    assert dashboard["threshold_readiness_version"] == "v3_round62_commit_threshold_readiness"
    assert dashboard["threshold_readiness"]["active_policy_unchanged"] is True
    assert state["eve_self_learning"]["commit_threshold_readiness_version"] == "v3_round62_commit_threshold_readiness"
    assert report["commit_gate"]["threshold_dry_run_version"] == "v3_round62_multi_observation_commit_threshold_dry_run"
    assert report["commit_threshold_readiness"]["read_only"] is True
