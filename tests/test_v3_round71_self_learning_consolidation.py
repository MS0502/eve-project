"""EVE v3 round71 — self-learning policy consolidation checks.

Round71 is a stabilization/verification round. It does not add a new learning
shortcut. These tests lock down the four review concerns raised after rounds
57-70:

1. EveVocabTracker and EveSelfLearningAdapter have separate roles.
2. Round55 vector store and Round56 wrapper routing are wired as intended.
3. AGP anchoring is still based on explicit meaning categories + SA activation,
   not seed/EveSpecific vector lookup.
4. Active self-learning policy remains explicit-commit-only with threshold and
   context-diversity gates enabled.
"""

from __future__ import annotations

import numpy as np

from main import build_full_engine
from adapters.agp_adapter import AGP_REASON_ANCHORED, AGP_REASON_UNKNOWN_CATEGORY


def test_round71_tracker_and_self_learning_roles_are_separate() -> None:
    engine = build_full_engine()
    tracker = engine.eve_vocab_tracker
    learner = engine.eve_self_learning

    assert learner.tracker is tracker
    assert learner.vector_store is engine.eve_specific_vector_store

    stats = tracker.stats()
    assert stats["role"] == "lexical_observation_only"
    assert stats["manual_observation_supported"] is True
    assert stats["continuous_observation_owner"] == "EveSelfLearningAdapter"
    assert stats["vector_commit_owner"] == "EveSelfLearningAdapter"
    assert stats["vector_generation_enabled"] is False
    assert stats["wrapper_integrated"] is False
    assert not hasattr(tracker, "commit_eve_specific_vectors")

    before = tracker.get_observation_count("민석")
    learner.observe_text("민석 오늘", source="round71_test")
    assert tracker.get_observation_count("민석") == before + 1


def test_round71_round55_56_wrapper_routing_is_explicit() -> None:
    engine = build_full_engine()
    wrapper = engine.self_embedding
    store = engine.eve_specific_vector_store

    assert wrapper.__class__.__name__ == "EmbeddingWrapper"
    assert wrapper.primary is engine.fasttext_embedding
    assert wrapper.fallback is engine.self_embedding_backup
    assert wrapper.eve_specific is store

    assert store.add_or_update_vector("민석", ["안녕"], engine=engine) is True
    vec = wrapper.get_vector("민석")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (300,)
    assert wrapper.telemetry()["eve_specific_hits"] >= 1


def test_round71_agp_anchor_does_not_use_eve_specific_vectors() -> None:
    engine = build_full_engine()
    agp = engine.agp_adapter
    wrapper = engine.self_embedding

    assert engine.eve_specific_vector_store.add_or_update_vector("민석", ["안녕"], engine=engine) is True
    before = wrapper.telemetry().copy()

    # If AGP used EveSpecific vectors as anchors, this would pass merely because
    # "민석" has a stored vector. It must fail because SA activation is empty.
    failed = agp.verify(
        "민석",
        meaning={"meaning_categories": ["민석"]},
        activated_categories=[],
    )
    assert failed.passed is False
    assert failed.reason == AGP_REASON_UNKNOWN_CATEGORY

    after = wrapper.telemetry().copy()
    assert after == before

    passed = agp.verify(
        "민석",
        meaning={"meaning_categories": ["민석"]},
        activated_categories=["민석"],
    )
    assert passed.passed is True
    assert passed.reason == AGP_REASON_ANCHORED


def test_round71_active_policy_is_still_guarded_explicit_commit_only() -> None:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    stats = learner.stats()

    assert stats["auto_observe_enabled"] is True
    assert stats["auto_promotion_enabled"] is False
    assert stats["commit_gate_enabled"] is True
    assert stats["min_observations_for_commit"] == 2
    assert stats["min_known_context_words_for_commit"] == 1
    assert stats["context_diversity_gate_enabled"] is True
    assert stats["automatic_rollback_enabled"] is False

    before = engine.eve_specific_vector_store.stats()["stored_count"]
    learner.observe_text("민석 오늘", source="round71_same_context")
    learner.observe_text("민석 오늘", source="round71_same_context")
    result = learner.commit_eve_specific_vectors(["민석"], context_words=["안녕"])
    after = engine.eve_specific_vector_store.stats()["stored_count"]

    assert after == before
    assert "민석" in result["rejected"]
    candidate = result["audit_report"]["candidate_reports"][0]
    assert "insufficient_context_diversity" in candidate["reasons"]
