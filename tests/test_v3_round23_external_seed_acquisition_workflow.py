"""EVE v3 round23 — External seed acquisition workflow.

Round23 prepared the acquisition/state workflow. Round24 may register concrete
provenance in MANIFEST.yaml, but runtime state and self-embedding remain
unchanged.
"""

from copy import deepcopy
from pathlib import Path

from adapters.agp_adapter import AGP_FALLBACK_SURFACE_POOL, AGP_MODE_OBSERVATION
from adapters.external_seed_manifest import (
    SEED_MANIFEST_PATH,
    SEED_STATE_INVALID,
    SEED_STATE_LOADED,
    SEED_STATE_REGISTERED,
    SEED_STATE_UNREGISTERED,
    SEED_STATE_USED,
    compute_seed_checksum,
    empty_manifest_template,
    external_seed_loaded,
    external_seed_state,
    fasttext_korean_acquisition_workflow,
    fasttext_korean_seed_dryrun_entry,
    is_valid_checksum_format,
    load_manifest_file,
    validate_manifest,
    validate_seed_entry,
)
from main import build_full_engine


VALID_ENTRY = {
    **fasttext_korean_seed_dryrun_entry(),
    "downloaded_at": "2026-05-11",
    "checksum": "SHA256:" + "2" * 64,
}


def test_round23_detects_fasttext_seed_not_present_and_defers_registration():
    """The current workspace has no cc.ko.300.bin, so round23 stays deferred."""
    workflow = fasttext_korean_acquisition_workflow()

    assert workflow["seed_name"] == "cc.ko.300.bin"
    assert workflow["workflow_state"] == "acquisition_deferred"
    assert workflow["seed_file_present"] is False
    assert workflow["download_required"] is True
    assert workflow["actual_download_performed"] is False
    assert workflow["manifest_file_modified"] is False
    assert workflow["file_loaded"] is False
    assert workflow["self_embedding_rewrite"] is False
    assert workflow["next_valid_state"] == SEED_STATE_REGISTERED


def test_external_seed_state_ladder():
    """Round23 exposes explicit seed lifecycle states without loading anything."""
    registered_manifest = {"seeds": [dict(VALID_ENTRY)]}

    assert external_seed_state(empty_manifest_template()) == SEED_STATE_UNREGISTERED
    assert external_seed_state(registered_manifest) == SEED_STATE_REGISTERED
    assert external_seed_state(registered_manifest, loaded_seed_names={"cc.ko.300.bin"}) == SEED_STATE_LOADED
    assert external_seed_state(registered_manifest, used_seed_names={"cc.ko.300.bin"}) == SEED_STATE_USED
    assert external_seed_state({"seeds": [{"name": "tiny-rwkv-body.bin"}]}) == SEED_STATE_INVALID


def test_checksum_computation_is_deterministic_and_strict(tmp_path):
    """Checksum acquisition is deterministic and does not parse the seed."""
    seed = tmp_path / "candidate.bin"
    seed.write_bytes(b"eve-v3-seed-checksum-fixture")

    first = compute_seed_checksum(seed)
    second = compute_seed_checksum(seed)

    assert first == second
    assert is_valid_checksum_format(first) is True
    assert first.startswith("SHA256:")
    assert len(first) == len("SHA256:") + 64


def test_missing_seed_checksum_requires_real_file():
    """Future registration cannot silently use a fake checksum for a missing file."""
    missing = Path("seeds/cc.ko.300.bin")

    assert not missing.exists()
    try:
        compute_seed_checksum(missing)
    except FileNotFoundError as exc:
        assert "cc.ko.300.bin" in str(exc)
    else:
        raise AssertionError("missing seed file should raise FileNotFoundError")


def test_acquisition_workflow_read_only_even_after_round24_registration():
    """The acquisition helper itself still does not mutate the real manifest."""
    manifest_path = Path(SEED_MANIFEST_PATH)
    before = manifest_path.read_text(encoding="utf-8")

    workflow = fasttext_korean_acquisition_workflow()

    after = manifest_path.read_text(encoding="utf-8")
    assert before == after
    assert workflow["manifest_file_modified"] is False
    assert validate_manifest(load_manifest_file(manifest_path))["valid"] is True
    assert external_seed_loaded(empty_manifest_template()) is False


def test_acquisition_workflow_is_read_only_for_engine_state():
    """Acquisition planning must not mutate AGP, traces, thresholds, or modes."""
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

    workflow = fasttext_korean_acquisition_workflow()

    assert workflow["valid_if_registered_now"] is False
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


def test_self_embedding_adapter_still_not_rewritten_for_fasttext():
    """Round23 acquisition workflow must not start the embedding migration."""
    text = Path("adapters/self_embedding_adapter.py").read_text(encoding="utf-8")
    workflow = fasttext_korean_acquisition_workflow()

    assert workflow["self_embedding_rewrite"] is False
    assert "fastText" not in text
    assert "cc.ko.300.bin" not in text


def test_real_registration_entry_still_requires_concrete_provenance():
    """Dry-run placeholders are still invalid for strict manifest registration."""
    dryrun_entry = fasttext_korean_seed_dryrun_entry()
    strict = validate_seed_entry(dryrun_entry)
    concrete = validate_seed_entry(VALID_ENTRY)

    assert strict["valid"] is False
    assert "invalid_checksum" in strict["errors"]
    assert "invalid_downloaded_at" in strict["errors"]
    assert concrete["valid"] is True
    assert external_seed_state({"seeds": [dict(VALID_ENTRY)]}) == SEED_STATE_REGISTERED
