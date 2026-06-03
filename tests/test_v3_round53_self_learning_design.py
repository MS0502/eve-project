"""EVE v3 round53 — B continuous self-learning design + scaffold."""

import pytest

from adapters.eve_vocab_tracker import EveVocabTracker
from adapters.eve_vector_store import EveSpecificVectorStore
from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    design_self_learning_architecture,
    redefine_drift_measurement_for_self_learning,
)
from main import build_full_engine


def test_round53_eve_vocab_tracker_module_exists():
    tracker = EveVocabTracker()
    assert tracker.__class__.__name__ == "EveVocabTracker"
    assert tracker.stats()["implementation_phase"] in {"scaffold (round53)", "round54_observation"}


def test_round53_eve_vector_store_module_exists():
    store = EveSpecificVectorStore()
    assert store.__class__.__name__ == "EveSpecificVectorStore"
    assert store.dimension == 300
    assert store.stats()["implementation_phase"] == "round55_vector_update"


def test_round53_scaffold_methods_raise_not_implemented():
    tracker = EveVocabTracker()
    store = EveSpecificVectorStore()
    # Round54 evolves the tracker scaffold into manual observation while the vector store remains scaffolded.
    tracker.observe_word("민석")
    assert tracker.get_observation_count("민석") == 1
    assert tracker.tracked_vocab() == ["민석"]
    assert tracker.is_eve_specific("민석") is True
    assert store.get_vector("민석") is None
    assert store.is_eve_specific("민석") is False
    assert store.stored_vocab() == []


def test_round53_stats_returns_data_dict():
    tracker_stats = EveVocabTracker().stats()
    store_stats = EveSpecificVectorStore().stats()
    assert isinstance(tracker_stats, dict)
    assert isinstance(store_stats, dict)
    assert tracker_stats["observed_count"] == 0
    assert store_stats["stored_count"] == 0
    assert tracker_stats["read_only"] is True
    assert store_stats["read_only"] is True


def test_round53_design_architecture_returns_data():
    design = design_self_learning_architecture()
    assert isinstance(design, dict)
    assert design["strategy"] == "B continuous after A first medium 30k"
    assert design["read_only"] is True
    assert "EveVocabTracker" in design["components"]
    assert "EveSpecificVectorStore" in design["components"]


def test_round53_design_specifies_phase_plan():
    design = design_self_learning_architecture()
    rounds = [step["round"] for step in design["phase_plan"]]
    assert rounds[:5] == [53, 54, 55, 56, 57]
    assert design["phase_plan"][0]["task"] == "design + scaffold"
    assert design["components"]["Wrapper integration"]["round_integration"] == 56


def test_round53_design_specifies_appendix_d_alignment():
    design = design_self_learning_architecture()
    drift = redefine_drift_measurement_for_self_learning()
    assert design["appendix_d_alignment"] == "high"
    assert drift["appendix_d_alignment"] == "high"
    assert "round 200+ avg drift > 0.3" in design["drift_measurement_refinement"]["target"]


def test_round53_drift_redefinition_data():
    drift = redefine_drift_measurement_for_self_learning()
    assert isinstance(drift, dict)
    assert drift["new_baseline_round"] == 53
    assert drift["implementation_baseline_round"] == 57
    assert "EveSpecificVectorStore.stats.stored_count" in drift["metric_components"]
    assert drift["no_runtime_change"] is True


def test_round53_no_actual_observation_yet():
    tracker = EveVocabTracker()
    tracker.observe_word("EVE", {"source": "test"})
    after = tracker.stats()
    assert after["observation_implemented"] is True
    assert after["observed_snapshot"] == {"EVE": 1}
    assert after["auto_observe_enabled"] is False


def test_round53_vector_store_now_implemented_in_round55_but_unrouted():
    store = EveSpecificVectorStore()
    before = store.stats()
    assert store.get_vector("EVE") is None
    after = store.stats()
    assert before == after
    assert after["vector_update_implemented"] is True
    assert after["wrapper_integrated"] is False


def test_round53_wrapper_unchanged():
    engine = build_full_engine()
    wrapper = engine.self_embedding
    assert wrapper.__class__.__name__ == "EmbeddingWrapper"
    assert not hasattr(wrapper, "eve_specific_store")
    design = design_self_learning_architecture()
    assert design["wrapper_integration_design"]["current_round53_runtime"] == "fasttext medium 30k primary + PMI+SVD fallback only"


def test_round53_fasttext_primary_unchanged():
    engine = build_full_engine()
    assert engine.fasttext_embedding.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert engine.self_embedding.primary is engine.fasttext_embedding
    assert engine.self_embedding.primary.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME


def test_round53_pmi_svd_fallback_unchanged():
    engine = build_full_engine()
    assert engine.self_embedding.fallback.__class__.__name__ == "SelfEmbeddingAdapter"
    assert engine.self_embedding_backup.__class__.__name__ == "SelfEmbeddingAdapter"
    assert engine.self_embedding.fallback is engine.self_embedding_backup


def test_round53_report_records_self_learning_scaffold():
    from pathlib import Path

    text = Path("ROUND_V3_R53_REPORT.md").read_text(encoding="utf-8")
    assert "B continuous" in text
    assert "EveVocabTracker" in text
    assert "round54" in text
    assert "round57" in text
