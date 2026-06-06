from __future__ import annotations

from main import build_full_engine


def test_round59_commit_gate_audit_is_read_only_and_fails_closed_without_loaded_context() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 EVE 오늘", source="unit_a")
    learner.observe_text("민석 EVE 군대", source="unit_b")

    before = engine.eve_specific_vector_store.stats()["stored_count"]
    audit = learner.audit_commit_gate(words=["민석"], context_words=["오늘", "군대", "없는단어999"])
    after = engine.eve_specific_vector_store.stats()["stored_count"]

    assert audit["audit_version"] == "v3_round59_commit_gate_audit"
    assert audit["read_only"] is True
    assert audit["commit_gate_enabled"] is True
    assert audit["target_words"] == ["민석"]
    assert audit["eligible_words"] == []
    assert audit["rejected_words"] == ["민석"]
    assert audit["known_context_words"] == []
    assert "insufficient_known_context" in audit["candidate_reports"][0]["reasons"]
    assert before == after


def test_round59_commit_blocks_unobserved_candidate_before_vector_mutation() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning

    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    assert report["created"] == []
    assert report["rejected"] == ["민석"]
    assert "insufficient_observations" in report["audit_report"]["candidate_reports"][0]["reasons"]
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False
    assert learner.stats()["commit_gate_blocks"] == 1


def test_round59_commit_blocks_when_context_has_no_known_fasttext_words() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["없는단어999", "또없는단어888"])

    assert report["created"] == []
    assert report["rejected"] == ["민석"]
    reasons = report["audit_report"]["candidate_reports"][0]["reasons"]
    assert "insufficient_known_context" in reasons
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round59_commit_rejects_observed_candidate_when_fasttext_is_not_loaded() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")

    report = learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    assert report["created"] == []
    assert report["rejected"] == ["민석"]
    assert report["audit_report"]["gate_pass"] is False
    assert report["known_context_words"] == []
    assert "insufficient_known_context" in report["audit_report"]["candidate_reports"][0]["reasons"]
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round59_state_debug_and_drift_measurement_surface_commit_gate() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    engine.eve_self_learning.observe_text("민석 오늘", source="unit_a")
    engine.eve_self_learning.observe_text("민석 군대", source="unit_b")
    engine.eve_self_learning.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    state = engine.state_debug.snapshot_state()
    report = measure_eve_self_learning_drift_accumulation(engine)

    assert state["eve_self_learning"]["commit_gate_enabled"] is True
    assert state["eve_self_learning"]["audit_calls"] >= 1
    assert report["commit_gate"]["enabled"] is True
    assert report["commit_gate"]["mutation_requires_audit_pass"] is True
    assert report["policy"]["explicit_commit_gate"] is True
