"""EVE v3 round51 — wrapper primary swap to medium 30k.

Round51 promotes the registered medium 30k fastText subset to the runtime
primary by changing FasttextEmbeddingAdapter's default subset. It does not run
smoke sampling, remove mini/small artifacts, or implement self-learning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    SEED_MANIFEST_PATH,
    SUBSET_STATE_EXTRACTED,
    load_manifest_file,
    subset_state,
)
from adapters.fasttext_embedding_adapter import (
    FASTTEXT_EMBEDDING_DEFAULT_SUBSET,
    FasttextEmbeddingAdapter,
)
from main import build_full_engine

REPORT = Path("ROUND_V3_R51_REPORT.md")
STATUS = Path("CURRENT_STATUS.md")
AGENTS = Path("AGENTS.md")

ROUND44_GENERAL_OOV = ["어때", "그래", "뭐야", "좋아해", "군대", "코딩"]
EVE_SPECIFIC_OOV = ["EVE", "민석"]


def test_default_subset_is_medium_30k():
    adapter = FasttextEmbeddingAdapter()

    assert FASTTEXT_EMBEDDING_DEFAULT_SUBSET == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert adapter.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert "medium.30k" in adapter.subset_dir
    assert adapter.get_dimension() == 300


def test_engine_fasttext_uses_medium():
    engine = build_full_engine()

    assert engine.fasttext_embedding.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert engine.fasttext_embedding.is_loaded() is True
    assert engine.fasttext_embedding.stats()["vocab_size"] == 30000
    assert engine.fasttext_embedding.stats()["vectors_shape"] == (30000, 300)


def test_wrapper_primary_uses_medium():
    engine = build_full_engine()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert engine.self_embedding.primary is engine.fasttext_embedding
    assert getattr(engine.self_embedding.primary, "subset_name", None) == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert engine.self_embedding.stats()["primary_stats"]["subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME


def test_medium_30k_load_succeeds():
    adapter = FasttextEmbeddingAdapter()

    assert adapter.is_loaded() is False
    assert adapter.load() is True
    assert adapter.is_loaded() is True
    assert adapter.stats()["vocab_size"] == 30000
    assert adapter.stats()["vectors_shape"] == (30000, 300)


def test_medium_30k_lookup_works():
    adapter = FasttextEmbeddingAdapter()
    adapter.load()

    vec = adapter.get_vector("안녕")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (300,)
    assert vec.dtype == np.float32


@pytest.mark.parametrize("word", ROUND44_GENERAL_OOV)
def test_round44_general_oov_now_resolved(word: str):
    adapter = FasttextEmbeddingAdapter()
    adapter.load()

    vec = adapter.get_vector(word)
    assert isinstance(vec, np.ndarray), word
    assert vec.shape == (300,)


@pytest.mark.parametrize("word", EVE_SPECIFIC_OOV)
def test_eve_specific_still_oov(word: str):
    adapter = FasttextEmbeddingAdapter()
    adapter.load()

    assert adapter.get_vector(word) is None


def test_small_5k_still_extracted():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED


def test_mini_1k_still_extracted():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME) == SUBSET_STATE_EXTRACTED


def test_state_debug_shows_medium_primary():
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()

    assert state["fasttext_embedding"]["subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert state["fasttext_embedding"]["vocab_size"] == 30000
    assert state["main_engine"]["primary_subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert state["main_engine"]["primary_vocab_size"] == 30000


def test_no_smoke_run_in_round51():
    text = REPORT.read_text(encoding="utf-8")

    assert "smoke 재실행 없음" in text
    assert "round52" in text
    assert "primary_hit_rate 측정은 round52" in text


def test_no_self_learning_in_round51():
    text = REPORT.read_text(encoding="utf-8")

    assert "self-learning 구현 없음" in text
    assert "round53+" in text
    assert "PMI+SVD fallback 유지" in text
