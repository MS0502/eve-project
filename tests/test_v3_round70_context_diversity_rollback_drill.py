from __future__ import annotations

from main import build_full_engine


def _candidate(report: dict, word: str) -> dict:
    for item in report["candidate_reports"]:
        if item["word"] == word:
            return item
    raise AssertionError(f"missing candidate report for {word}")


def test_round70_context_diversity_blocked_candidate_report_aggregates_audit_records() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit")
    learner.observe_text("민석 오늘", source="unit")

    before_store_count = engine.eve_specific_vector_store.stats()["stored_count"]
    commit = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    before_records = len(learner.commit_audit_records())
    report = learner.context_diversity_blocked_candidate_report()
    after_records = len(learner.commit_audit_records())
    after_store_count = engine.eve_specific_vector_store.stats()["stored_count"]

    assert commit["created"] == []
    assert commit["rejected"] == ["민석"]
    assert report["report_version"] == "v3_round70_context_diversity_blocked_candidate_report"
    assert report["round"] == 70
    assert report["context_diversity_gate_enabled"] is True
    assert report["blocked_candidate_count"] == 1
    assert report["blocked_words"][0]["word"] == "민석"
    assert report["blocked_words"][0]["block_count"] >= 2  # audit + commit records
    assert "insufficient_context_diversity" in report["blocked_words"][0]["reasons"]
    assert "오늘" in report["rollback_context_words"]
    assert before_records == after_records
    assert before_store_count == after_store_count
    assert report["policy"]["automatic_rollback_enabled"] is False
    assert report["policy"]["no_vector_mutation"] is True


def test_round70_manual_rollback_drill_is_read_only_and_identifies_newly_unblocked_word() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit")
    learner.observe_text("민석 오늘", source="unit")
    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    before_records = len(learner.commit_audit_records())
    before_store_count = engine.eve_specific_vector_store.stats()["stored_count"]
    drill = learner.context_diversity_rollback_drill(words=["민석"], context_words=["오늘", "군대"])
    after_records = len(learner.commit_audit_records())
    after_store_count = engine.eve_specific_vector_store.stats()["stored_count"]

    assert drill["drill_version"] == "v3_round70_context_diversity_rollback_drill"
    assert drill["round"] == 70
    assert drill["active_context_diversity_gate_enabled"] is True
    assert drill["hypothetical_context_diversity_gate_enabled"] is False
    assert drill["active_policy_unchanged"] is True
    assert drill["newly_unblocked_by_manual_rollback"] == ["민석"]
    assert drill["rollback_eligible_words"] == ["민석"]
    item = _candidate(drill, "민석")
    assert item["active_gate_pass"] is False
    assert item["rollback_gate_pass"] is True
    assert item["newly_unblocked_by_manual_context_diversity_rollback"] is True
    assert "insufficient_context_diversity" in item["active_reasons"]
    assert item["rollback_reasons"] == []
    assert before_records == after_records
    assert before_store_count == after_store_count
    assert drill["policy"]["automatic_rollback_enabled"] is False
    assert drill["policy"]["no_context_diversity_policy_change"] is True
    assert drill["policy"]["no_vector_mutation"] is True


def test_round70_state_debug_and_drift_report_surface_blocked_candidate_report() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit")
    learner.observe_text("민석 오늘", source="unit")
    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    stats = learner.stats()
    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert stats["round"] == 71
    assert stats["latest_round"] == 71
    assert stats["implementation_phase"] == "round71_self_learning_policy_consolidation"
    assert stats["active_policy_version"] == "v3_round69_context_diversity_gate_enforced"
    assert stats["context_diversity_rollback_drill_version"] == "v3_round70_context_diversity_rollback_drill"
    assert stats["context_diversity_blocked_candidate_report_version"] == "v3_round70_context_diversity_blocked_candidate_report"
    assert stats["context_diversity_blocked_candidate_report"]["blocked_candidate_count"] == 1

    assert state["eve_self_learning"]["context_diversity_rollback_drill_version"] == "v3_round70_context_diversity_rollback_drill"
    assert state["eve_self_learning"]["context_diversity_blocked_candidate_report_version"] == "v3_round70_context_diversity_blocked_candidate_report"
    assert state["eve_self_learning"]["context_diversity_blocked_candidate_report"]["blocked_candidate_count"] == 1

    assert report["context_diversity_rollback_drill"]["drill_version"] == "v3_round70_context_diversity_rollback_drill"
    assert report["context_diversity_blocked_candidate_report"]["report_version"] == "v3_round70_context_diversity_blocked_candidate_report"
    assert report["policy"]["context_diversity_rollback_drill_read_only"] is True
    assert report["policy"]["context_diversity_blocked_candidate_report_read_only"] is True
    assert report["policy"]["agp_bypass"] is False
