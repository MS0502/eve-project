"""EVE v3 round21 — External Seed Policy manifest infrastructure.

Round21 began the External Seed Policy contract. Later rounds may register
provenance in the manifest, but registration still does not mean runtime load or use.
"""

from copy import deepcopy
from pathlib import Path

from adapters.agp_adapter import AGP_FALLBACK_SURFACE_POOL, AGP_MODE_OBSERVATION
from adapters.external_seed_manifest import (
    LICENSE_WHITELIST,
    REQUIRED_SEED_FIELDS,
    SEED_MANIFEST_PATH,
    SEED_POLICY_VERSION,
    empty_manifest_template,
    external_seed_state,
    load_manifest_file,
    external_seed_loaded,
    validate_manifest,
    validate_seed_entry,
)
from main import build_full_engine


VALID_SEED_ENTRY = {
    "name": "cc.ko.300.bin",
    "source": "https://fasttext.cc/docs/en/crawl-vectors.html",
    "license": "CC-BY-SA-3.0",
    "version": "2017-12",
    "downloaded_at": "2026-05-11",
    "checksum": "SHA256:" + "a" * 64,
    "imported_at_round": 22,
    "imported_at_patch": "v3_round22",
}


def test_seed_manifest_file_exists_and_is_valid_policy_shape():
    """The manifest exists and validates whether empty or round24-registered."""
    path = Path(SEED_MANIFEST_PATH)

    assert path.exists()
    manifest = load_manifest_file(path)
    result = validate_manifest(manifest)
    assert result["version"] == SEED_POLICY_VERSION
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["seeds_count"] >= 0
    assert result.get("seed_entries_count", 0) in {0, 1}
    assert result.get("subset_entries_count", 0) >= 0
    assert external_seed_state(manifest) in {"unregistered", "registered"}


def test_validate_manifest_format_accepts_complete_seed_entry():
    """A future seed entry is valid only with full provenance fields."""
    result = validate_manifest({"seeds": [dict(VALID_SEED_ENTRY)]})

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["seeds_count"] == 1
    assert result["external_seed_loaded"] is True
    assert result["entry_results"][0]["seed_name"] == "cc.ko.300.bin"
    assert REQUIRED_SEED_FIELDS.issubset(VALID_SEED_ENTRY.keys())
    assert "CC-BY-SA-3.0" in LICENSE_WHITELIST


def test_validate_missing_required_field_rejected():
    """Provenance/license/version/checksum fields are mandatory."""
    entry = dict(VALID_SEED_ENTRY)
    entry.pop("checksum")
    result = validate_seed_entry(entry)

    assert result["valid"] is False
    assert "missing_required_field:checksum" in result["errors"]
    assert "invalid_checksum" in result["errors"]


def test_validate_invalid_license_rejected():
    """External seeds with non-whitelisted licenses are rejected."""
    entry = dict(VALID_SEED_ENTRY)
    entry["license"] = "GPL-3.0"
    result = validate_seed_entry(entry)

    assert result["valid"] is False
    assert "license_not_whitelisted" in result["errors"]


def test_validate_invalid_checksum_format_rejected():
    """Checksums must be explicit SHA256 values."""
    entry = dict(VALID_SEED_ENTRY)
    entry["checksum"] = "md5:abc"
    result = validate_seed_entry(entry)

    assert result["valid"] is False
    assert "invalid_checksum" in result["errors"]


def test_validate_generation_body_marker_rejected():
    """Manifest infrastructure must not admit external generation bodies."""
    entry = dict(VALID_SEED_ENTRY)
    entry["name"] = "tiny-rwkv-body.bin"
    result = validate_seed_entry(entry)

    assert result["valid"] is False
    assert "forbidden_generation_body_marker" in result["errors"]


def test_validator_is_read_only_for_engine_and_inputs():
    """Manifest validation must not mutate AGP, traces, fallback pool, or inputs."""
    engine = build_full_engine()
    manifest = {"seeds": [dict(VALID_SEED_ENTRY)]}
    before_manifest = deepcopy(manifest)
    before = {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "fallback_pool": tuple(AGP_FALLBACK_SURFACE_POOL),
        "compositor_trace": deepcopy(engine.compositor.agp_trace),
        "speech_trace": deepcopy(engine.speech_hub.agp_trace),
    }

    result = validate_manifest(manifest)

    assert result["valid"] is True
    assert manifest == before_manifest
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


def test_empty_manifest_template_remains_unloaded_baseline():
    """The empty template still represents the no-seed baseline."""
    placeholder = empty_manifest_template()
    result = validate_manifest(placeholder)

    assert result["valid"] is True
    assert result["seeds_count"] == 0
    assert result["external_seed_loaded"] is False
    assert external_seed_loaded(placeholder) is False
