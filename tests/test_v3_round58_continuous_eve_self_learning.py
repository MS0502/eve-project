from __future__ import annotations

from main import build_full_engine
from adapters.eve_self_learning_adapter import EveSelfLearningAdapter


def test_round58_adapter_observes_eve_specific_without_auto_promotion() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning

    result = learner.observe_text("민석 EVE 오늘 군대", source="unit")

    assert result["auto_observe_enabled"] is True
    assert result["auto_promotion_enabled"] is False
    assert {"민석", "EVE"}.issubset(set(result["eve_specific_candidates"]))
    assert engine.eve_vocab_tracker.is_eve_specific("민석") is True
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round58_explicit_commit_fails_closed_when_fasttext_context_is_not_loaded() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 EVE 오늘", source="unit_a")
    learner.observe_text("민석 EVE 군대", source="unit_b")

    report = learner.commit_eve_specific_vectors(context_words=["오늘", "군대", "없는단어999"])

    assert report["created"] == []
    assert {"민석", "EVE"}.issubset(set(report["rejected"]))
    assert report["known_context_words"] == []
    assert all("insufficient_known_context" in item["reasons"] for item in report["audit_report"]["candidate_reports"])
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False
    assert engine.eve_specific_vector_store.get_vector("민석") is None
    assert engine.eve_specific_vector_store.update_count("민석") == 0


def test_round58_chat_stream_records_observation_but_does_not_promote() -> None:
    engine = build_full_engine()

    list(engine.chat_stream("민석 EVE 오늘 군대"))

    stats = engine.eve_self_learning.stats()
    assert stats["observe_calls"] >= 1
    assert stats["auto_promotion_enabled"] is False
    assert engine.eve_vocab_tracker.get_observation_count("민석") >= 1
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round58_wrapper_remains_no_load_after_rejected_explicit_commit() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="unit_a")
    learner.observe_text("민석 군대", source="unit_b")
    learner.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    vector = engine.self_embedding.get_vector("민석")
    telemetry = engine.self_embedding.telemetry()

    assert vector is None
    assert telemetry["eve_specific_hits"] == 0
    assert telemetry["eve_specific_rate"] == 0.0
    assert engine.eve_specific_vector_store.is_eve_specific("민석") is False


def test_round58_adapter_tokenization_is_deterministic() -> None:
    adapter = EveSelfLearningAdapter(auto_observe_enabled=True, auto_promotion_enabled=False)

    assert adapter.tokenize("민석아, EVE-v3 round58!") == ["민석아", "EVE", "v3", "round58"]


def test_round58_drift_accumulation_report_includes_observation_and_commit_counts() -> None:
    from adapters.external_seed_manifest import measure_eve_self_learning_drift_accumulation

    engine = build_full_engine()
    engine.eve_self_learning.observe_text("민석 EVE 오늘", source="unit_a")
    engine.eve_self_learning.observe_text("민석 EVE 군대", source="unit_b")
    engine.eve_self_learning.commit_eve_specific_vectors(words=["민석"], context_words=["오늘", "군대"])

    report = measure_eve_self_learning_drift_accumulation(engine)

    assert report["measurement_version"] == "v3_round58_eve_self_learning_drift_accumulation"
    assert report["drift_accumulation"]["observed_eve_specific_count"] >= 1
    assert report["drift_accumulation"]["vectors_created"] == 0
    assert report["drift_accumulation"]["commit_gate_blocks"] >= 1
    assert report["policy"]["auto_observe_enabled"] is True
    assert report["policy"]["auto_promotion_enabled"] is False
    assert report["policy"]["agp_bypass"] is False


def test_round58_state_debug_surfaces_self_learning_status() -> None:
    engine = build_full_engine()
    engine.eve_self_learning.observe_text("민석 EVE", source="unit")

    state = engine.state_debug.snapshot_state()

    assert state["eve_self_learning"]["module"] == "EveSelfLearningAdapter"
    assert state["eve_self_learning"]["auto_observe_enabled"] is True
    assert state["eve_self_learning"]["auto_promotion_enabled"] is False
    assert state["eve_self_learning"]["observe_calls"] >= 1
