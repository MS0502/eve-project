from __future__ import annotations

from main import build_full_engine


def _candidate(report: dict, word: str) -> dict:
    for item in report["candidate_reports"]:
        if item["word"] == word:
            return item
    raise AssertionError(f"missing candidate report for {word}")


def test_round69_repeated_same_context_is_blocked_by_actual_gate() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit")
    learner.observe_text("민석 오늘", source="unit")

    before = engine.eve_specific_vector_store.stats()["stored_count"]
    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    after = engine.eve_specific_vector_store.stats()["stored_count"]

    assert report["created"] == []
    assert report["rejected"] == ["민석"]
    assert before == after
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False

    audit = report["audit_report"]
    assert audit["policy_round"] == 70
    assert audit["context_diversity_gate_enabled"] is True
    assert audit["context_diversity_gate_enforcement_version"] == "v3_round69_context_diversity_gate_enforcement"
    item = _candidate(audit, "민석")
    assert item["context_diverse"] is False
    assert "insufficient_context_diversity" in item["reasons"]


def test_round69_diverse_context_candidate_still_requires_loaded_known_context() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    assert report["created"] == []
    assert report["rejected"] == ["민석"]
    assert report["audit_report"]["gate_pass"] is False
    item = _candidate(report["audit_report"], "민석")
    assert item["context_diverse"] is True
    assert item["evidence_status"] == "threshold_met_context_diverse"
    assert "insufficient_known_context" in item["reasons"]
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round69_state_debug_and_drift_report_surface_context_diversity_enforcement() -> None:
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
    assert stats["active_policy_version"] == "v3_round69_context_diversity_gate_enforced"
    assert stats["context_diversity_gate_enabled"] is True
    assert stats["context_diversity_gate_enforcement_version"] == "v3_round69_context_diversity_gate_enforcement"

    assert state["eve_self_learning"]["context_diversity_gate_enabled"] is True
    assert state["eve_self_learning"]["context_diversity_gate_enforcement_version"] == "v3_round69_context_diversity_gate_enforcement"

    assert report["commit_gate"]["context_diversity_gate_enabled"] is True
    assert report["commit_gate"]["context_diversity_gate_enforcement_version"] == "v3_round69_context_diversity_gate_enforcement"
    assert report["context_diversity_gate_enforcement"]["context_diversity_gate_enabled"] is True
    assert report["policy"]["context_diversity_gate_enforced"] is True
    assert report["policy"]["auto_promotion_enabled"] is False
    assert report["policy"]["agp_bypass"] is False
