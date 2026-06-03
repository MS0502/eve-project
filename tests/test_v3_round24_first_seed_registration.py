"""EVE v3 round24 — first strict external seed registration.

Round24 registers only the verified cc.ko.300.bin provenance entry in
seeds/MANIFEST.yaml. The seed binary remains external; no fastText runtime import,
self-embedding rewrite, AGP runtime change, or seed loading happens here.
"""

from copy import deepcopy
from pathlib import Path
import sys

from adapters.agp_adapter import AGP_FALLBACK_SURFACE_POOL, AGP_MODE_OBSERVATION
from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_FILE_SIZE_BYTES,
    FASTTEXT_KOREAN_IMPORTED_PATCH,
    FASTTEXT_KOREAN_SHA256,
    SEED_MANIFEST_PATH,
    SEED_STATE_REGISTERED,
    external_seed_loaded,
    external_seed_state,
    fasttext_korean_seed_registered_entry,
    load_manifest_file,
    register_seed_entry,
    validate_manifest,
)
from main import build_full_engine


def test_manifest_entry_registered_for_ccko300():
    """Round24 advances MANIFEST.yaml from placeholder to concrete provenance."""
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    validation = validate_manifest(manifest)

    assert validation["valid"] is True
    assert validation["seed_entries_count"] == 1
    assert validation["subset_entries_count"] >= 0
    assert validation["external_seed_loaded"] is True
    seed_entry = [entry for entry in manifest["seeds"] if entry.get("name") == "cc.ko.300.bin"][0]
    assert seed_entry["checksum"] == FASTTEXT_KOREAN_SHA256
    assert seed_entry["file_size_bytes"] == FASTTEXT_KOREAN_FILE_SIZE_BYTES
    assert seed_entry["imported_at_round"] == 24
    assert seed_entry["imported_at_patch"] == FASTTEXT_KOREAN_IMPORTED_PATCH


def test_external_seed_state_now_registered_but_not_loaded_or_used():
    """Manifest registration is not runtime seed loading or adapter usage."""
    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert external_seed_loaded(manifest) is True
    assert external_seed_state(manifest, loaded_seed_names=set()) == SEED_STATE_REGISTERED
    assert external_seed_state(manifest, used_seed_names=set()) == SEED_STATE_REGISTERED


def test_strict_validation_blocks_invalid_entry_without_manifest_change(tmp_path):
    """Invalid concrete entries must not partially modify the manifest."""
    manifest_path = tmp_path / "MANIFEST.yaml"
    manifest_path.write_text("seeds: []\n", encoding="utf-8")
    before = manifest_path.read_text(encoding="utf-8")
    entry = fasttext_korean_seed_registered_entry()
    entry["license"] = "GPL-3.0"

    result = register_seed_entry(manifest_path, entry)

    assert result["registered"] is False
    assert result["manifest_file_modified"] is False
    assert "candidate:license_not_whitelisted" in result["errors"]
    assert manifest_path.read_text(encoding="utf-8") == before


def test_manifest_backup_created_before_write(tmp_path):
    """Registration creates rollback material before atomic replacement."""
    manifest_path = tmp_path / "MANIFEST.yaml"
    manifest_path.write_text("# old manifest\nseeds: []\n", encoding="utf-8")
    backup_path = tmp_path / "MANIFEST.yaml.backup"

    result = register_seed_entry(manifest_path, fasttext_korean_seed_registered_entry(), backup_path=backup_path)

    assert result["registered"] is True
    assert result["backup_created"] is True
    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == "# old manifest\nseeds: []\n"
    assert "cc.ko.300.bin" in manifest_path.read_text(encoding="utf-8")


def test_atomic_write_failure_keeps_manifest_intact(tmp_path):
    """A failed atomic write path must leave the original manifest untouched."""
    manifest_path = tmp_path / "MANIFEST.yaml"
    original = "# original\nseeds: []\n"
    manifest_path.write_text(original, encoding="utf-8")

    result = register_seed_entry(
        manifest_path,
        fasttext_korean_seed_registered_entry(),
        simulate_atomic_failure=True,
    )

    assert result["registered"] is False
    assert result["manifest_file_modified"] is False
    assert "simulated_atomic_write_failure" in result["errors"]
    assert manifest_path.read_text(encoding="utf-8") == original


def test_self_embedding_adapter_unchanged_for_registration_only():
    """Round24 does not start the fastText embedding rewrite."""
    text = Path("adapters/self_embedding_adapter.py").read_text(encoding="utf-8")

    assert "fastText" not in text
    assert "cc.ko.300.bin" not in text


def test_fasttext_library_not_imported_by_registration_round():
    """Registering provenance must not import fastText/gensim runtime code."""
    assert "fasttext" not in sys.modules
    assert "gensim" not in sys.modules


def test_external_seed_registered_but_not_loaded_yet_for_engine_state():
    """Engine AGP state remains untouched by the registered seed manifest."""
    engine = build_full_engine()
    before = {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "fallback_pool": tuple(AGP_FALLBACK_SURFACE_POOL),
        "compositor_trace": deepcopy(engine.compositor.agp_trace),
        "speech_trace": deepcopy(engine.speech_hub.agp_trace),
    }

    manifest = load_manifest_file(SEED_MANIFEST_PATH)

    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "fallback_pool": tuple(AGP_FALLBACK_SURFACE_POOL),
        "compositor_trace": engine.compositor.agp_trace,
        "speech_trace": engine.speech_hub.agp_trace,
    } == before
    assert engine.agp_adapter.mode == AGP_MODE_OBSERVATION
