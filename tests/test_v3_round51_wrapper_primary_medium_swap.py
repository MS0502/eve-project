"""EVE v3 round51 — medium primary selection without default vector load."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    SEED_MANIFEST_PATH,
    SUBSET_STATE_EXTRACTED,
    audit_subset_artifact,
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
MEDIUM_VECTORS = Path(FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH) / "vectors.npy"

ROUND44_GENERAL_OOV = ["어때", "그래", "뭐야", "좋아해", "군대", "코딩"]
EVE_SPECIFIC_OOV = ["EVE", "민석"]


def test_default_subset_is_medium_30k():
    adapter = FasttextEmbeddingAdapter()

    assert FASTTEXT_EMBEDDING_DEFAULT_SUBSET == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert adapter.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert "medium.30k" in adapter.subset_dir
    assert adapter.get_dimension() == 300


def test_engine_fasttext_uses_medium_but_remains_unloaded_by_default():
    engine = build_full_engine()

    assert engine.fasttext_embedding.subset_name == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert engine.fasttext_embedding.is_loaded() is False
    assert engine.fasttext_embedding.stats()["vocab_size"] == 0
    assert engine.fasttext_embedding.stats()["vectors_shape"] is None


def test_wrapper_primary_uses_medium_unloaded_adapter():
    engine = build_full_engine()

    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert engine.self_embedding.primary is engine.fasttext_embedding
    assert getattr(engine.self_embedding.primary, "subset_name", None) == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert engine.self_embedding.stats()["primary_stats"]["subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert engine.self_embedding.stats()["primary_stats"]["loaded"] is False


def _assert_vector_artifact_absent_or_ignored_untracked(path: Path) -> None:
    if not path.exists():
        return
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], check=False).returncode == 0
    assert subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0


def test_medium_30k_load_requires_operator_vectors_and_fails_closed_when_absent(tmp_path: Path):
    adapter = FasttextEmbeddingAdapter(subset_dir=tmp_path / "isolated_missing_medium30k")

    assert adapter.is_loaded() is False
    with pytest.raises(ValueError, match="missing_vectors_file"):
        adapter.load()
    assert adapter.is_loaded() is False
    assert adapter.stats()["vocab_size"] == 0


def test_medium_30k_lookup_requires_loaded_operator_artifact():
    adapter = FasttextEmbeddingAdapter()

    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.get_vector("안녕")


@pytest.mark.parametrize("word", ROUND44_GENERAL_OOV)
def test_round44_general_oov_is_manifest_resolved_but_not_runtime_loaded(word: str):
    adapter = FasttextEmbeddingAdapter()
    vocab = Path(adapter.subset_dir, "vocab.txt").read_text(encoding="utf-8").splitlines()

    assert word in vocab
    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.get_vector(word)


@pytest.mark.parametrize("word", EVE_SPECIFIC_OOV)
def test_eve_specific_still_absent_from_medium_vocab(word: str):
    adapter = FasttextEmbeddingAdapter()
    vocab = Path(adapter.subset_dir, "vocab.txt").read_text(encoding="utf-8").splitlines()

    assert word not in vocab
    with pytest.raises(RuntimeError, match="FasttextEmbeddingAdapter is not loaded"):
        adapter.get_vector(word)


def test_medium_audit_fails_closed_without_reading_operator_vectors():
    audit = audit_subset_artifact(subset_name=FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME)

    _assert_vector_artifact_absent_or_ignored_untracked(MEDIUM_VECTORS)
    assert audit["valid"] is False
    assert ({"missing_vectors_file", "vector_contents_not_read"} & set(audit["errors"]))
    assert audit["vocab"]["line_count"] == 30000
    assert audit["vectors"]["shape"] is None
    assert audit["vector_contents_read"] is False
    assert audit["runtime_loaded"] is False


def test_small_5k_still_extracted():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED


def test_mini_1k_still_extracted():
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME) == SUBSET_STATE_EXTRACTED


def test_state_debug_shows_medium_primary_no_load():
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()

    assert state["fasttext_embedding"]["subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert state["fasttext_embedding"]["loaded"] is False
    assert state["fasttext_embedding"]["vocab_size"] == 0
    assert state["main_engine"]["primary_subset_name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert state["main_engine"]["primary_vocab_size"] == 0


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


def test_current_policy_docs_preserve_no_default_load_boundary():
    status = STATUS.read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")

    assert "default runtime remains no-load" in agents.lower()
    assert "operator" in status.lower()
