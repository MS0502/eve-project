from __future__ import annotations

from main import build_full_engine


def _summary_for_word(summary: dict, word: str) -> dict:
    for item in summary["candidate_summaries"]:
        if item["word"] == word:
            return item
    raise AssertionError(f"missing summary for {word}")


def test_round66_evidence_quality_summary_is_read_only_and_detects_context_diversity() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    before_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    before_records = len(learner.commit_audit_records())
    summary = learner.observation_evidence_quality_summary(words=["민석"])
    after_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    after_records = len(learner.commit_audit_records())

    assert summary["summary_version"] == "v3_round66_observation_evidence_quality_summary"
    assert summary["round"] == 66
    assert summary["read_only"] is True
    assert summary["min_observations_for_commit"] == 2
    assert summary["audit_records_unchanged"] is True
    assert summary["vector_store_unchanged"] is True
    assert summary["policy"]["auto_promotion_enabled"] is False
    assert summary["policy"]["threshold_changed"] is False
    assert summary["policy"]["no_vector_mutation"] is True
    assert before_vectors == after_vectors
    assert before_records == after_records

    item = _summary_for_word(summary, "민석")
    assert item["observed_count"] == 2
    assert item["meets_current_observation_threshold"] is True
    assert item["context_diverse"] is True
    assert item["unique_context_count"] == 2
    assert item["unique_source_count"] == 2
    assert item["unique_peer_token_count"] >= 2
    assert item["evidence_status"] == "threshold_met_context_diverse"
    assert summary["context_diverse_count"] == 1
    assert summary["threshold_ready_count"] == 1


def test_round66_evidence_quality_flags_repeated_same_context_without_changing_commit_policy() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit")
    learner.observe_text("민석 오늘", source="unit")

    summary = learner.observation_evidence_quality_summary(words=["민석"])
    item = _summary_for_word(summary, "민석")

    assert item["observed_count"] == 2
    assert item["meets_current_observation_threshold"] is True
    assert item["context_diverse"] is False
    assert item["duplicate_context_count"] == 1
    assert item["evidence_status"] == "threshold_met_repeated_same_context"
    assert summary["repeated_same_context_count"] == 1
    assert learner.stats()["min_observations_for_commit"] == 2


def test_round66_state_debug_and_drift_report_surface_evidence_quality() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    stats = learner.stats()
    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert stats["round"] == 71
    assert stats["latest_round"] == 71
    assert stats["implementation_phase"] == "round71_self_learning_policy_consolidation"
    assert stats["observation_evidence_quality_version"] == "v3_round66_observation_evidence_quality_summary"
    assert stats["observation_evidence_quality"]["read_only"] is True
    assert state["eve_self_learning"]["observation_evidence_quality_version"] == "v3_round66_observation_evidence_quality_summary"
    assert state["eve_self_learning"]["observation_evidence_quality"]["context_diverse_count"] >= 1
    assert report["commit_gate"]["observation_evidence_quality_version"] == "v3_round66_observation_evidence_quality_summary"
    assert report["observation_evidence_quality"]["policy"]["no_auto_threshold_adjustment"] is True
    assert report["policy"]["auto_promotion_enabled"] is False
