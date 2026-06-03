"""EVE v3 round27 — seed verification runner.

Round27 verifies the real-path checksum workflow without loading fastText and
without rewriting self_embedding.
"""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    SEED_MANIFEST_PATH,
    SEED_STATE_REGISTERED,
    compute_seed_checksum,
    external_seed_state,
    load_manifest_file,
)
import adapters.fasttext_loader as loader
from adapters.fasttext_loader import (
    EVE_FASTTEXT_SEED_PATH_ENV,
    FASTTEXT_PYTHON_PACKAGE,
    get_configured_seed_path,
    seed_verification_runner,
)
from main import build_full_engine


def _manifest_for(seed_path: Path) -> dict:
    return {
        "seeds": [
            {
                "name": FASTTEXT_KOREAN_SEED_NAME,
                "checksum": compute_seed_checksum(seed_path),
            }
        ]
    }


def test_runner_without_env_var_skipped(monkeypatch):
    """No configured path means graceful skipped result."""
    monkeypatch.delenv(EVE_FASTTEXT_SEED_PATH_ENV, raising=False)

    result = seed_verification_runner()

    assert get_configured_seed_path() is None
    assert result["status"] == "skipped"
    assert result["reason"] == "seed_path_not_configured"
    assert result["fasttext_load_attempted"] is False
    assert result["runtime_package_imported"] is False


def test_runner_explicit_path_takes_priority_over_env(tmp_path, monkeypatch):
    """explicit_path must override EVE_FASTTEXT_SEED_PATH."""
    explicit = tmp_path / "explicit.bin"
    explicit.write_bytes(b"explicit seed bytes")
    env_file = tmp_path / "env.bin"
    env_file.write_bytes(b"different env seed bytes")
    monkeypatch.setenv(EVE_FASTTEXT_SEED_PATH_ENV, str(env_file))

    result = seed_verification_runner(explicit_path=explicit, manifest_data=_manifest_for(explicit))

    assert result["status"] == "verified"
    assert result["path_resolved"] == str(explicit)
    assert result["match"] is True
    assert result["fasttext_load_attempted"] is False


def test_runner_with_missing_file_reports_missing_file(monkeypatch, tmp_path):
    """A configured but absent path must not be treated as verified or loaded."""
    monkeypatch.delenv(EVE_FASTTEXT_SEED_PATH_ENV, raising=False)
    missing = tmp_path / "missing-cc.ko.300.bin"

    result = seed_verification_runner(explicit_path=missing)

    assert result["status"] == "missing_file"
    assert result["errors"] == ["seed_file_missing"]
    assert result["checksum_computed"] is None
    assert result["fasttext_load_attempted"] is False


def test_runner_verification_only_does_not_need_fasttext_runtime(tmp_path, monkeypatch):
    """Checksum verification must work even if fastText runtime is unavailable."""
    seed = tmp_path / "seed.bin"
    seed.write_bytes(b"checksum only")
    monkeypatch.setattr(loader, "is_fasttext_available", lambda: (_ for _ in ()).throw(AssertionError("runtime check not allowed")))

    result = seed_verification_runner(explicit_path=seed, manifest_data=_manifest_for(seed))

    assert result["status"] == "verified"
    assert result["match"] is True
    assert result["fasttext_load_attempted"] is False
    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules


def test_runner_checksum_mismatch(tmp_path):
    """Wrong file contents must produce mismatch status, not load attempts."""
    seed = tmp_path / "actual.bin"
    seed.write_bytes(b"actual bytes")
    other = tmp_path / "expected.bin"
    other.write_bytes(b"expected bytes")

    result = seed_verification_runner(explicit_path=seed, manifest_data=_manifest_for(other))

    assert result["status"] == "mismatch"
    assert result["match"] is False
    assert result["errors"] == ["checksum_mismatch"]
    assert result["fasttext_load_attempted"] is False


def test_runner_result_is_data_dict_not_text(tmp_path):
    """The runner returns structured data and no natural-language recommendation."""
    seed = tmp_path / "seed.bin"
    seed.write_bytes(b"data dict")

    result = seed_verification_runner(explicit_path=seed, manifest_data=_manifest_for(seed))

    assert isinstance(result, dict)
    assert "status" in result
    assert "recommendation" not in result
    assert "message" not in result
    assert result["manifest_file_modified"] is False
    assert result["self_embedding_rewrite"] is False


def test_runner_idempotent_for_same_file(tmp_path):
    """Same file + same manifest must produce the same deterministic result."""
    seed = tmp_path / "seed.bin"
    seed.write_bytes(b"deterministic")
    manifest = _manifest_for(seed)

    first = seed_verification_runner(explicit_path=seed, manifest_data=manifest)
    second = seed_verification_runner(explicit_path=seed, manifest_data=manifest)

    assert first == second


def test_runner_no_self_embedding_or_engine_state_change(tmp_path):
    """Verification must not mutate engine, AGP, traces, or self-embedding."""
    seed = tmp_path / "seed.bin"
    seed.write_bytes(b"read only")
    engine = build_full_engine()
    before = {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "compositor_trace": deepcopy(engine.compositor.agp_trace),
        "speech_trace": deepcopy(engine.speech_hub.agp_trace),
        "self_embedding_text": Path("adapters/self_embedding_adapter.py").read_text(encoding="utf-8"),
    }

    _ = seed_verification_runner(explicit_path=seed, manifest_data=_manifest_for(seed))

    assert {
        "agp_mode": engine.agp_adapter.mode,
        "compositor_mode": engine.compositor.agp_mode,
        "speech_hub_mode": engine.speech_hub.agp_mode,
        "thresholds": (engine.agp_adapter.category_threshold, engine.agp_adapter.hormone_threshold),
        "compositor_trace": engine.compositor.agp_trace,
        "speech_trace": engine.speech_hub.agp_trace,
        "self_embedding_text": Path("adapters/self_embedding_adapter.py").read_text(encoding="utf-8"),
    } == before


def test_default_manifest_state_still_registered_after_runner(tmp_path):
    """Round27 verification does not advance registered -> loaded by default."""
    seed = tmp_path / "seed.bin"
    seed.write_bytes(b"registered stays registered")
    before = load_manifest_file(SEED_MANIFEST_PATH)

    _ = seed_verification_runner(explicit_path=seed, manifest_data=_manifest_for(seed))
    after = load_manifest_file(SEED_MANIFEST_PATH)

    assert after == before
    assert external_seed_state(after) == SEED_STATE_REGISTERED


def test_operator_workflow_documented():
    """Round27 must leave an operator workflow for Colab/local verification."""
    text = Path("docs/FASTTEXT_SEED_VERIFICATION_WORKFLOW.md").read_text(encoding="utf-8")

    assert EVE_FASTTEXT_SEED_PATH_ENV in text
    assert "seed_verification_runner" in text
    assert "does not load fastText" in text
