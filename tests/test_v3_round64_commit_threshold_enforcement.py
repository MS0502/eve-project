from __future__ import annotations

from main import build_full_engine


def test_round64_default_commit_threshold_is_two_observations() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning

    assert learner.stats()["implementation_phase"] == "round71_self_learning_policy_consolidation"
    assert learner.stats()["min_observations_for_commit"] == 2
    assert learner.stats()["commit_threshold_enforcement_version"] == "v3_round69_context_diversity_gate_enforced"


def test_round64_single_observation_commit_is_blocked_without_vector_mutation() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")

    before = engine.eve_specific_vector_store.stats()["stored_count"]
    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])
    after = engine.eve_specific_vector_store.stats()["stored_count"]

    assert report["created"] == []
    assert report["rejected"] == ["민석"]
    assert report["audit_report"]["min_observations_for_commit"] == 2
    assert "insufficient_observations" in report["audit_report"]["candidate_reports"][0]["reasons"]
    assert before == after
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round64_two_observations_commit_passes_existing_gate() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    assert report["created"] == ["민석"]
    assert report["rejected"] == []
    assert report["audit_report"]["gate_pass"] is True
    assert report["audit_report"]["candidate_reports"][0]["observed_count"] == 2
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is True


def test_round64_state_debug_and_drift_report_surface_enforced_threshold() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석", source="unit")
    learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대"])

    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert state["eve_self_learning"]["min_observations_for_commit"] == 2
    assert state["eve_self_learning"]["commit_threshold_enforcement_version"] == "v3_round69_context_diversity_gate_enforced"
    assert state["eve_self_learning"]["active_policy_version"] == "v3_round69_context_diversity_gate_enforced"
    assert report["commit_gate"]["min_observations_for_commit"] == 2
    assert report["commit_gate"]["threshold_enforcement_version"] == "v3_round69_context_diversity_gate_enforced"
    assert report["commit_threshold_enforcement"]["read_only"] is True
    assert report["policy"]["auto_promotion_enabled"] is False
