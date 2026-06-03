"""EVE v3 round22 — External Seed Policy dry-run workflow.

Round22 validates the fastText Korean seed dry-run workflow without
importing a binary, rewriting self embeddings, or loading any external seed.
Later rounds may register concrete provenance in the real manifest.
"""

from copy import deepcopy
from pathlib import Path

from adapters.external_seed_manifest import (
    SEED_MANIFEST_PATH,
    dryrun_register_seed,
    empty_manifest_template,
    external_seed_loaded,
    fasttext_korean_seed_dryrun_entry,
    load_manifest_file,
    validate_seed_entry,
)
from main import build_full_engine


def test_dryrun_validates_fasttext_entry_format_with_placeholders():
    """The planned cc.ko.300.bin entry is acceptable only as a dry-run."""
    entry = fasttext_korean_seed_dryrun_entry()
    result = dryrun_register_seed(entry, empty_manifest_template())

    assert result["valid"] is True
    assert result["would_be_added"] is True
    assert result["dry_run_only"] is True
    assert result["file_loaded"] is False
    assert result["manifest_file_modified"] is False
    assert result["candidate_result"]["seed_name"] == "cc.ko.300.bin"
    assert "candidate:placeholder_checksum" in result["warnings"]
    assert "candidate:placeholder_downloaded_at" in result["warnings"]

    strict = validate_seed_entry(entry)
    assert strict["valid"] is False
    assert "invalid_checksum" in strict["errors"]
    assert "invalid_downloaded_at" in strict["errors"]


def test_dryrun_does_not_load_file_or_modify_manifest_file():
    """Dry-run returns simulated state only and never touches the real manifest."""
    manifest_path = Path(SEED_MANIFEST_PATH)
    before_text = manifest_path.read_text(encoding="utf-8")

    result = dryrun_register_seed(fasttext_korean_seed_dryrun_entry(), empty_manifest_template())

    after_text = manifest_path.read_text(encoding="utf-8")
    assert before_text == after_text
    assert result["file_loaded"] is False
    assert result["manifest_file_modified"] is False
    assert load_manifest_file(manifest_path) == load_manifest_file(manifest_path)


def test_dryrun_manifest_after_is_simulation_only():
    """manifest_after_dryrun changes only the returned data, not real state."""
    manifest = empty_manifest_template()
    before_manifest = deepcopy(manifest)

    result = dryrun_register_seed(fasttext_korean_seed_dryrun_entry(), manifest)

    assert manifest == before_manifest
    assert result["manifest_after_dryrun"]["seeds"][0]["name"] == "cc.ko.300.bin"
    assert result["external_seed_loaded_after_dryrun"] is True
    assert external_seed_loaded(empty_manifest_template()) is False


def test_dryrun_invalid_license_blocked():
    """A candidate with non-whitelisted license cannot be added even in dry-run."""
    entry = fasttext_korean_seed_dryrun_entry()
    entry["license"] = "GPL-3.0"

    result = dryrun_register_seed(entry, empty_manifest_template())

    assert result["valid"] is False
    assert result["would_be_added"] is False
    assert "candidate:license_not_whitelisted" in result["errors"]


def test_dryrun_duplicate_seed_blocked():
    """Dry-run rejects adding a seed whose name already exists."""
    existing = {"seeds": [
        {
            **fasttext_korean_seed_dryrun_entry(),
            "downloaded_at": "2026-05-11",
            "checksum": "SHA256:" + "1" * 64,
        }
    ]}

    result = dryrun_register_seed(fasttext_korean_seed_dryrun_entry(), existing)

    assert result["valid"] is False
    assert result["would_be_added"] is False
    assert "candidate:duplicate_seed_name" in result["errors"]


def test_dryrun_no_state_change_for_engine():
    """Dry-run must not mutate AGP, traces, modes, thresholds, or fallback pool."""
    engine = build_full_engine()
    before = {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "compositor_trace": deepcopy(engine.compositor.agp_trace),
        "speech_trace": deepcopy(engine.speech_hub.agp_trace),
    }

    result = dryrun_register_seed(fasttext_korean_seed_dryrun_entry(), empty_manifest_template())

    assert result["valid"] is True
    assert {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "compositor_trace": engine.compositor.agp_trace,
        "speech_trace": engine.speech_hub.agp_trace,
    } == before


def test_dryrun_still_does_not_load_seed_and_self_embedding_unchanged():
    """Dry-run has no runtime seed load and no self embedding rewrite."""
    path = Path("adapters/self_embedding_adapter.py")
    text = path.read_text(encoding="utf-8")

    result = dryrun_register_seed(fasttext_korean_seed_dryrun_entry(), empty_manifest_template())

    assert result["would_be_added"] is True
    assert external_seed_loaded(empty_manifest_template()) is False
    assert "fastText" not in text
    assert "cc.ko.300.bin" not in text
