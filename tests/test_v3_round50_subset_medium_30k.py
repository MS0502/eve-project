import json
import hashlib
from pathlib import Path

import numpy as np

from main import build_full_engine
from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SHA256,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_MANIFEST_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    SUBSET_STATE_EXTRACTED,
    assess_self_embedding_rewrite_readiness,
    audit_subset_artifact,
    compute_seed_checksum,
    fasttext_korean_subset_medium_30k_entry,
    load_manifest_file,
    round50_medium_30k_oov_resolution_data,
    subset_state,
)
from adapters.fasttext_embedding_adapter import FASTTEXT_EMBEDDING_DEFAULT_SUBSET


SUBSET_DIR = Path(FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH)
VOCAB = SUBSET_DIR / "vocab.txt"
VECTORS = SUBSET_DIR / "vectors.npy"
SUBSET_MANIFEST = SUBSET_DIR / "subset_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "SHA256:" + digest.hexdigest()


def test_round50_medium_subset_directory_and_files_exist():
    assert SUBSET_DIR.is_dir()
    assert VOCAB.is_file()
    assert VECTORS.is_file()
    assert SUBSET_MANIFEST.is_file()


def test_round50_medium_vocab_txt_has_30000_utf8_lines():
    vocab = VOCAB.read_text(encoding="utf-8").splitlines()
    assert len(vocab) == 30000
    assert all("\ufffd" not in word for word in vocab)
    assert "어때" in vocab
    assert "코딩" in vocab


def test_round50_medium_vectors_shape_and_dtype():
    vectors = np.load(VECTORS, mmap_mode="r")
    assert tuple(vectors.shape) == (30000, 300)
    assert str(vectors.dtype) == "float32"


def test_round50_medium_checksums_match_constants_and_manifest():
    entry = fasttext_korean_subset_medium_30k_entry()
    assert _sha256(VOCAB) == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM == entry["vocab_checksum"]
    assert _sha256(VECTORS) == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM == entry["vectors_checksum"]
    assert _sha256(SUBSET_MANIFEST) == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_MANIFEST_CHECKSUM == entry["subset_manifest_checksum"]
    assert compute_seed_checksum(VOCAB) == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM


def test_round50_medium_subset_manifest_payload_matches_operator_result():
    payload = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    assert payload["name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert payload["parent_seed"] == "cc.ko.300.bin"
    assert payload["parent_checksum"] == FASTTEXT_KOREAN_SHA256
    assert payload["vocab_size"] == 30000
    assert payload["vector_dim"] == 300
    assert payload["purpose"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE
    assert payload["supersedes_small_5k_for_lexical_coverage"] is True


def test_round50_medium_manifest_entry_parent_linkage_and_subset_state():
    manifest = load_manifest_file()
    matches = [entry for entry in manifest["seeds"] if entry.get("name") == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["parent_seed"] == "cc.ko.300.bin"
    assert entry["parent_checksum"] == FASTTEXT_KOREAN_SHA256
    assert entry["purpose"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE
    assert entry["status"] == "extracted"
    assert entry["imported_at_round"] == 50
    assert entry["supersedes_small_5k_for_lexical_coverage"] is True
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME) == SUBSET_STATE_EXTRACTED


def test_round50_medium_audit_validates_artifact_without_runtime_use():
    manifest = load_manifest_file()
    audit = audit_subset_artifact(manifest, subset_name=FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME)
    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["runtime_used"] is False
    assert audit["self_embedding_rewrite"] is False
    assert audit["vectors"]["shape"] == (30000, 300)
    assert audit["vocab"]["line_count"] == 30000


def test_round50_mini_small_medium_three_tier_coexist():
    manifest = load_manifest_file()
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME) == SUBSET_STATE_EXTRACTED
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME) == SUBSET_STATE_EXTRACTED
    assert subset_state(manifest, FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME) == SUBSET_STATE_EXTRACTED
    names = {entry.get("name") for entry in manifest["seeds"]}
    assert {FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME, FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME, FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME}.issubset(names)


def test_round50_oov_resolution_data_records_6_of_6_general_korean():
    data = round50_medium_30k_oov_resolution_data()
    assert data["oov_resolution"] == "6/6"
    assert data["resolution_ratio"] == 1.0
    assert set(data["resolved_general_korean_oov"]) == {"어때", "그래", "뭐야", "좋아해", "군대", "코딩"}
    assert set(data["eve_specific_oov_remaining"]) == {"EVE", "민석"}
    assert data["wrapper_primary_swap"] is False


def test_round50_readiness_lists_medium_as_preferred_but_no_wrapper_swap():
    engine = build_full_engine()
    readiness = assess_self_embedding_rewrite_readiness(engine)
    assert readiness["available_subset"]["name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
    assert readiness["available_subset"]["vocab_size"] == 30000
    assert any(item["name"] == FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME for item in readiness["available_subsets"])
    assert any(item["name"] == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME for item in readiness["available_subsets"])
    assert readiness["recommendation_data"]["automatic_application"] is False
    assert engine.self_embedding.__class__.__name__ == "EmbeddingWrapper"
    assert getattr(engine.self_embedding.primary, "subset_name", None) == FASTTEXT_EMBEDDING_DEFAULT_SUBSET
    assert FASTTEXT_EMBEDDING_DEFAULT_SUBSET == FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME
