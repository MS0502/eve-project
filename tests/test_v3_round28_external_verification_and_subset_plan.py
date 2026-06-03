"""EVE v3 round28 — external verification result and subset plan.

Round28 records the operator-side checksum verification result and plans subset
extraction. It must not extract vectors, load fastText, or rewrite self_embedding.
"""

from pathlib import Path

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    FASTTEXT_KOREAN_SHA256,
    SEED_MANIFEST_PATH,
    SEED_STATE_REGISTERED,
    external_seed_state,
    load_manifest_file,
)
from adapters.fasttext_loader import seed_verification_runner


REPORT = Path("ROUND_V3_R28_REPORT.md")
PLAN = Path("docs/SUBSET_EXTRACTION_PLAN.md")
SELF_EMBEDDING = Path("adapters/self_embedding_adapter.py")


def test_round28_external_verification_documented():
    """The Colab verification result must be recorded with exact checksum data."""
    text = REPORT.read_text(encoding="utf-8")

    assert "status: verified" in text
    assert "match: true" in text
    assert "path: /content/cc.ko.300.bin" in text
    assert f"actual: {FASTTEXT_KOREAN_SHA256}" in text
    assert f"expected: {FASTTEXT_KOREAN_SHA256}" in text
    assert "external SHA256 verification" in text
    assert "no adapters import" in text


def test_subset_extraction_plan_exists_and_has_targets():
    """Round28 must add the planning document without producing artifacts."""
    text = PLAN.read_text(encoding="utf-8")

    assert "mini" in text and "1,000" in text
    assert "small" in text and "5,000" in text
    assert "medium" in text and "30,000" in text
    assert "vocab.txt" in text
    assert "vectors.npy" in text
    assert "subset manifest" in text.lower()


def test_subset_plan_specifies_deterministic_selection():
    """The plan must forbid random subset selection."""
    text = PLAN.read_text(encoding="utf-8")

    assert "Use the fastText vocabulary order as the primary order." in text
    assert "Do not sample." in text
    assert "Do not use random shuffling." in text
    assert "Record the exact extraction script" in text


def test_no_runtime_load_from_external_verification_documentation():
    """Round28 verification documentation still does not load fastText at runtime."""
    result = seed_verification_runner()
    assert result["status"] == "skipped"
    assert result["fasttext_load_attempted"] is False
    assert result["runtime_package_imported"] is False


def test_state_still_registered_and_self_embedding_unchanged():
    """External verification does not advance registered -> loaded/used."""
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert any(entry.get("name") == FASTTEXT_KOREAN_SEED_NAME for entry in manifest["seeds"])

    self_embedding_text = SELF_EMBEDDING.read_text(encoding="utf-8")
    assert "fasttext" not in self_embedding_text.lower()
    assert "300d" not in self_embedding_text.lower()
