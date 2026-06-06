from __future__ import annotations

from main import build_full_engine


def _candidate(report: dict, word: str) -> dict:
    for item in report["candidate_reports"]:
        if item["word"] == word:
            return item
    raise AssertionError(f"missing candidate report for {word}")


def test_round67_context_diversity_gate_dry_run_blocks_repeated_context_without_policy_change() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit")
    learner.observe_text("민석 오늘", source="unit")

    before_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    before_records = len(learner.commit_audit_records())
    report = learner.dry_run_context_diversity_gate(words=["민석"], context_words=["오늘", "군대"])
    after_vectors = engine.eve_specific_vector_store.stats()["stored_count"]
    after_records = len(learner.commit_audit_records())

    assert report["dry_run_version"] == "v3_round67_context_diversity_gate_dry_run"
    assert report["round"] == 67
    assert report["read_only"] is True
    assert report["active_policy_unchanged"] is True
    assert report["current_gate_context_diversity_required"] is True
    assert report["dry_run_context_diversity_required"] is True
    assert report["audit_records_unchanged"] is True
    assert report["vector_store_unchanged"] is True
    assert report["newly_blocked_by_context_diversity"] == []
    assert report["eligible_words"] == []
    assert before_vectors == after_vectors
    assert before_records == after_records

    item = _candidate(report, "민석")
    assert item["current_gate_pass"] is False
    assert item["dry_run_pass"] is False
    assert item["context_diverse"] is False
    assert "insufficient_context_diversity" in item["dry_run_reasons"]
    assert learner.stats()["min_observations_for_commit"] == 2


def test_round67_context_diversity_gate_dry_run_blocks_without_loaded_known_context() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    report = learner.dry_run_context_diversity_gate(words=["민석"], context_words=["오늘", "군대"])
    item = _candidate(report, "민석")

    assert report["eligible_words"] == []
    assert report["blocked_words"] == ["민석"]
    assert report["newly_blocked_by_context_diversity"] == []
    assert item["current_gate_pass"] is False
    assert item["dry_run_pass"] is False
    assert "insufficient_known_context" in item["dry_run_reasons"]
    assert item["context_diverse"] is True
    assert item["evidence_status"] == "threshold_met_context_diverse"
    assert report["policy"]["context_diversity_gate_enforced"] is True
    assert report["policy"]["no_context_diversity_policy_change"] is True


def test_round67_state_debug_and_drift_report_surface_context_diversity_dry_run() -> None:
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
    assert stats["context_diversity_gate_dry_run_version"] == "v3_round67_context_diversity_gate_dry_run"
    assert stats["context_diversity_gate_dry_run"]["read_only"] is True
    assert state["eve_self_learning"]["context_diversity_gate_dry_run_version"] == "v3_round67_context_diversity_gate_dry_run"
    assert report["commit_gate"]["context_diversity_gate_dry_run_version"] == "v3_round67_context_diversity_gate_dry_run"
    assert report["context_diversity_gate_dry_run"]["policy"]["context_diversity_gate_enforced"] is True
    assert report["policy"]["auto_promotion_enabled"] is False
    assert report["policy"]["context_diversity_gate_enforced"] is True
