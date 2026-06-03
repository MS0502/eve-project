"""EVE v3 round55 — EveSpecificVectorStore deterministic vector update."""

from pathlib import Path

import numpy as np

from adapters.eve_vector_store import EveSpecificVectorStore
from adapters.external_seed_manifest import design_self_learning_architecture
from main import build_full_engine


def test_round55_default_dimension_300():
    store = EveSpecificVectorStore()
    assert store.dimension == 300
    assert store.stats()["implementation_phase"] == "round55_vector_update"


def test_round55_add_vector_manual_validates_and_stores_copy():
    store = EveSpecificVectorStore()
    vec = np.ones(300, dtype=np.float32)
    store.add_vector("EVE", vec)
    out = store.get_vector("EVE")
    assert out is not None
    assert out.dtype == np.float32
    assert out.shape == (300,)
    out[0] = 99.0
    assert store.get_vector("EVE")[0] == 1.0


def test_round55_add_vector_rejects_wrong_dimension():
    store = EveSpecificVectorStore()
    try:
        store.add_vector("EVE", np.ones(50, dtype=np.float32))
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong dimension accepted")


def test_round55_context_update_succeeds_with_known_words():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    assert store.add_or_update_vector("민석", ["군대", "코딩"]) is True
    vec = store.get_vector("민석")
    assert vec is not None
    assert vec.shape == (300,)
    assert vec.dtype == np.float32


def test_round55_context_update_matches_fasttext_mean():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    store.add_or_update_vector("민석", ["군대", "코딩"])
    expected = np.mean(
        np.stack([engine.fasttext_embedding.get_vector("군대"), engine.fasttext_embedding.get_vector("코딩")]),
        axis=0,
    ).astype(np.float32)
    assert np.allclose(store.get_vector("민석"), expected)


def test_round55_same_context_is_deterministic():
    engine = build_full_engine()
    a = EveSpecificVectorStore(engine=engine)
    b = EveSpecificVectorStore(engine=engine)
    assert a.add_or_update_vector("EVE", ["좋아해", "코딩"]) is True
    assert b.add_or_update_vector("EVE", ["좋아해", "코딩"]) is True
    assert np.array_equal(a.get_vector("EVE"), b.get_vector("EVE"))


def test_round55_empty_context_no_mutation():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    before = store.stats()
    assert store.add_or_update_vector("EVE", []) is False
    assert store.stats() == before


def test_round55_all_oov_context_no_mutation():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    assert store.add_or_update_vector("EVE", ["EVE", "민석"]) is False
    assert store.get_vector("EVE") is None
    assert store.stored_vocab() == []


def test_round55_partial_oov_context_uses_known_only():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    assert store.add_or_update_vector("EVE", ["민석", "군대", "EVE"]) is True
    assert np.array_equal(store.get_vector("EVE"), engine.fasttext_embedding.get_vector("군대"))
    assert store.stats()["last_context"]["EVE"] == ["군대"]


def test_round55_repeated_update_averages_old_and_new():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    assert store.add_or_update_vector("EVE", ["군대"]) is True
    first = store.get_vector("EVE")
    assert store.add_or_update_vector("EVE", ["코딩"]) is True
    expected = ((first + engine.fasttext_embedding.get_vector("코딩")) / 2.0).astype(np.float32)
    assert np.allclose(store.get_vector("EVE"), expected)
    assert store.update_count("EVE") == 2


def test_round55_stored_vocab_sorted():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    store.add_or_update_vector("민석", ["군대"])
    store.add_or_update_vector("EVE", ["코딩"])
    assert store.stored_vocab() == ["EVE", "민석"]


def test_round55_is_eve_specific_requires_stored_vector():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    assert store.is_eve_specific("민석") is False
    store.add_or_update_vector("민석", ["군대"])
    assert store.is_eve_specific("민석") is True


def test_round55_stats_records_vector_update_but_no_wrapper():
    engine = build_full_engine()
    store = EveSpecificVectorStore(engine=engine)
    store.add_or_update_vector("민석", ["군대"])
    stats = store.stats()
    assert stats["vector_update_implemented"] is True
    assert stats["wrapper_integrated"] is False
    assert stats["auto_observe_enabled"] is False
    assert stats["stored_count"] == 1


def test_round55_wrapper_not_integrated_yet():
    engine = build_full_engine()
    assert not hasattr(engine.self_embedding, "eve_specific_store")
    assert engine.self_embedding.primary is engine.fasttext_embedding


def test_round55_design_phase_marks_300d_context_mean():
    design = design_self_learning_architecture()
    assert design["components"]["EveSpecificVectorStore"]["dimension"] == 300
    assert "fastText 300d context" in design["deterministic_update_rule"]["vector_generation"]


def test_round55_no_random_used_in_vector_store():
    text = Path("adapters/eve_vector_store.py").read_text(encoding="utf-8")
    assert "random.random" not in text
    assert "np.random" not in text
