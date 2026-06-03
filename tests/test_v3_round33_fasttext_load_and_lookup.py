"""EVE v3 round33 — FasttextEmbeddingAdapter actual medium 30k load.

Round33 makes the separate fastText subset adapter usable as an explicit
instance-level component. It still must not swap ``engine.self_embedding`` or
rewrite any affected module. Round51 promotes the medium 30k subset to the runtime primary while preserving small 5k as an extracted artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    SUBSET_STATE_EXTRACTED,
    load_manifest_file,
    subset_state,
)
from adapters.fasttext_embedding_adapter import FasttextEmbeddingAdapter
from main import build_full_engine

SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")


def _loaded_adapter() -> FasttextEmbeddingAdapter:
    adapter = FasttextEmbeddingAdapter()
    assert adapter.load() is True
    return adapter


def test_adapter_load_succeeds_and_transitions_instance_to_loaded():
    adapter = FasttextEmbeddingAdapter()

    assert adapter.is_loaded() is False
    assert adapter.load() is True
    assert adapter.is_loaded() is True
    assert adapter.stats()["vocab_size"] == 30000
    assert adapter.stats()["vectors_shape"] == (30000, 300)
    assert adapter.stats()["runtime_used"] is False
    assert adapter.stats()["self_embedding_rewrite"] is False


def test_adapter_load_validates_files_and_rejects_bad_subset_dir(tmp_path):
    subset_dir = tmp_path / "bad_subset"
    subset_dir.mkdir()
    (subset_dir / "vocab.txt").write_text("친구\n", encoding="utf-8")
    np.save(subset_dir / "vectors.npy", np.zeros((1, 300), dtype=np.float32))
    (subset_dir / "subset_manifest.json").write_text("{}\n", encoding="utf-8")

    adapter = FasttextEmbeddingAdapter(subset_dir=subset_dir)
    with pytest.raises(ValueError):
        adapter.load()


def test_get_vector_returns_300d_float32_copy():
    adapter = _loaded_adapter()

    vec = adapter.get_vector("친구")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (300,)
    assert vec.dtype == np.float32

    # Return a copy so callers cannot mutate the adapter's stored vector.
    vec[0] = vec[0] + 123.0
    vec_again = adapter.get_vector("친구")
    assert not np.array_equal(vec, vec_again)


def test_get_vector_unknown_word_returns_none_and_alias_matches():
    adapter = _loaded_adapter()

    assert adapter.get_vector("없는단어xyz") is None
    assert np.array_equal(adapter.get_embedding("친구"), adapter.get_vector("친구"))


def test_similarity_is_deterministic_and_self_is_one():
    adapter = _loaded_adapter()

    score1 = adapter.similarity("친구", "사랑")
    score2 = adapter.similarity("친구", "사랑")

    assert isinstance(score1, float)
    assert score1 == score2
    assert adapter.similarity("친구", "친구") == pytest.approx(1.0, abs=1e-5)
    assert adapter.similarity("친구", "없는단어xyz") == 0.0


def test_most_similar_is_deterministic_and_excludes_query_word():
    adapter = _loaded_adapter()

    result1 = adapter.most_similar("친구", top_n=3)
    result2 = adapter.most_similar("친구", top_n=3)

    assert result1 == result2
    assert len(result1) <= 3
    assert all(word != "친구" for word, _ in result1)
    assert all(isinstance(word, str) and isinstance(score, float) for word, score in result1)


def test_text_embedding_uses_whitespace_mean_and_zero_for_unknowns():
    adapter = _loaded_adapter()

    vec1 = adapter.text_embedding("친구 사랑")
    vec2 = adapter.text_embedding("친구 사랑")
    zero = adapter.text_embedding("없는단어xyz 또없는단어")

    assert vec1.shape == (300,)
    assert vec1.dtype == np.float32
    assert np.array_equal(vec1, vec2)
    assert np.array_equal(zero, np.zeros(300, dtype=np.float32))


def test_text_similarity_returns_float_and_handles_empty_known_tokens():
    adapter = _loaded_adapter()

    assert isinstance(adapter.text_similarity("친구 사랑", "친구"), float)
    assert adapter.text_similarity("없는단어xyz", "친구") == 0.0


def test_subset_state_remains_extracted_without_global_transition():
    manifest = load_manifest_file()
    adapter = _loaded_adapter()

    assert adapter.is_loaded() is True
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED


def test_engine_self_embedding_unchanged_and_no_runtime_swap():
    before_fasttext_imported = "fasttext" in sys.modules
    engine = build_full_engine()
    text = SELF_EMBEDDING.read_text(encoding="utf-8")
    adapter = _loaded_adapter()

    assert adapter.is_loaded() is True
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding, "dim") == 300
    assert "FasttextEmbeddingAdapter" not in text
    assert "cc.ko.300.subset" not in text
    assert "vectors.npy" not in text
    assert ("fasttext" in sys.modules) == before_fasttext_imported
