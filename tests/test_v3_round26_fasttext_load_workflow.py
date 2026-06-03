"""EVE v3 round26 — fastText checksum verification + explicit load workflow.

Round26 implements the fastText loader workflow, but still does not rewrite
self_embedding or load the registered external seed by default.
"""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    SEED_MANIFEST_PATH,
    SEED_STATE_LOADED,
    SEED_STATE_REGISTERED,
    compute_seed_checksum,
    external_seed_state,
    load_manifest_file,
)
import adapters.fasttext_loader as loader
from adapters.fasttext_loader import (
    ChecksumMismatchError,
    FASTTEXT_PYTHON_PACKAGE,
    fasttext_loader_boundary_status,
    load_fasttext_seed,
    mark_seed_as_loaded,
    unload_fasttext_seed,
    verify_fasttext_seed_checksum,
)
from main import build_full_engine


def test_verify_checksum_match_and_mismatch(tmp_path):
    """Checksum verification is deterministic and file-read only."""
    seed = tmp_path / "tiny-fasttext-like.bin"
    seed.write_bytes(b"eve-v3-round26-fasttext-checksum-fixture")
    expected = compute_seed_checksum(seed)
    wrong = "SHA256:" + "0" * 64

    match = verify_fasttext_seed_checksum(seed, expected_checksum=expected)
    mismatch = verify_fasttext_seed_checksum(seed, expected_checksum=wrong)

    assert match["verified"] is True
    assert match["match"] is True
    assert mismatch["verified"] is False
    assert mismatch["match"] is False
    assert match["file_loaded"] is False
    assert match["runtime_package_imported"] is False


def test_load_fasttext_seed_missing_file_fails_closed():
    """Missing seed files must not be treated as loaded."""
    with pytest.raises(FileNotFoundError):
        load_fasttext_seed("/does/not/exist/cc.ko.300.bin")


def test_load_fasttext_seed_checksum_mismatch_fails_closed(tmp_path, monkeypatch):
    """Checksum mismatch blocks load before optional runtime import."""
    seed = tmp_path / "tiny.bin"
    seed.write_bytes(b"actual bytes")
    wrong = "SHA256:" + "f" * 64
    monkeypatch.setattr(loader, "is_fasttext_available", lambda: True)

    with pytest.raises(ChecksumMismatchError):
        load_fasttext_seed(seed, expected_checksum=wrong)

    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules


def test_load_fasttext_seed_unavailable_runtime_returns_none(tmp_path, monkeypatch):
    """Optional dependency absence degrades gracefully after checksum passes."""
    seed = tmp_path / "tiny.bin"
    seed.write_bytes(b"tiny fasttext placeholder")
    expected = compute_seed_checksum(seed)
    monkeypatch.setattr(loader, "is_fasttext_available", lambda: False)
    sys.modules.pop(FASTTEXT_PYTHON_PACKAGE, None)

    result = load_fasttext_seed(seed, expected_checksum=expected)

    assert result is None
    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules


def test_mark_seed_as_loaded_returns_explicit_state_transition_without_manifest_write():
    """Loaded state requires explicit call and is returned as data only."""
    manifest_before = load_manifest_file(SEED_MANIFEST_PATH)
    before = deepcopy(manifest_before)

    result = mark_seed_as_loaded(FASTTEXT_KOREAN_SEED_NAME, model_handle=object())
    manifest_after = load_manifest_file(SEED_MANIFEST_PATH)

    assert result["valid"] is True
    assert result["state_before"] == SEED_STATE_REGISTERED
    assert result["state_after"] == SEED_STATE_LOADED
    assert result["loaded_seed_names"] == [FASTTEXT_KOREAN_SEED_NAME]
    assert result["manifest_file_modified"] is False
    assert manifest_after == before
    assert external_seed_state(manifest_after) == SEED_STATE_REGISTERED
    assert external_seed_state(manifest_after, loaded_seed_names={FASTTEXT_KOREAN_SEED_NAME}) == SEED_STATE_LOADED


def test_mark_seed_as_loaded_rejects_unregistered_or_missing_handle():
    """State transition fails closed for unregistered seed or missing handle."""
    unregistered = mark_seed_as_loaded("not-a-seed.bin", model_handle=object())
    missing_handle = mark_seed_as_loaded(FASTTEXT_KOREAN_SEED_NAME, model_handle=None)

    assert unregistered["valid"] is False
    assert "seed_not_registered" in unregistered["errors"]
    assert unregistered["state_after"] == unregistered["state_before"]
    assert missing_handle["valid"] is False
    assert "model_handle_required" in missing_handle["errors"]
    assert missing_handle["state_after"] == missing_handle["state_before"]


def test_state_unchanged_without_explicit_loaded_mark():
    """Round26 default state remains registered until explicit runtime mark."""
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    status = fasttext_loader_boundary_status()

    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert status["state_after_round26_without_explicit_load"] == SEED_STATE_REGISTERED
    assert status["actual_file_load_performed"] is False
    assert status["self_embedding_rewrite"] is False


def test_unload_fasttext_seed_is_explicit_data_only():
    """Unload boundary must not mutate manifest or embedding state."""
    manifest_before = load_manifest_file(SEED_MANIFEST_PATH)

    result = unload_fasttext_seed(object())

    assert result["unloaded"] is True
    assert result["model_handle_present"] is True
    assert result["self_embedding_rewrite"] is False
    assert load_manifest_file(SEED_MANIFEST_PATH) == manifest_before


def test_self_embedding_adapter_unchanged_for_round26_load_workflow():
    """Round26 must not start the 300d self-embedding rewrite."""
    text = Path("adapters/self_embedding_adapter.py").read_text(encoding="utf-8")

    assert "fastText" not in text
    assert "cc.ko.300.bin" not in text
    assert "fasttext_loader" not in text


def test_engine_and_agp_state_unchanged_by_load_workflow_status():
    """Boundary status and checksum helpers must not mutate runtime state."""
    engine = build_full_engine()
    before = {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "compositor_trace": deepcopy(engine.compositor.agp_trace),
        "speech_trace": deepcopy(engine.speech_hub.agp_trace),
    }

    _ = fasttext_loader_boundary_status()

    assert {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "compositor_trace": engine.compositor.agp_trace,
        "speech_trace": engine.speech_hub.agp_trace,
    } == before
