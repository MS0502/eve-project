"""EVE v3 round25/26 — fastText runtime loader boundary regression tests.

Round26 implements explicit checksum/load functions, but importing the module and
checking boundary status must still keep the registered seed unused by default.
"""

from copy import deepcopy
from pathlib import Path
import inspect
import sys

from adapters.agp_adapter import AGP_FALLBACK_SURFACE_POOL, AGP_MODE_OBSERVATION
from adapters.external_seed_manifest import (
    SEED_MANIFEST_PATH,
    SEED_STATE_REGISTERED,
    external_seed_state,
    load_manifest_file,
)
from adapters.fasttext_loader import (
    FASTTEXT_OPTIONAL_REQUIREMENT,
    FASTTEXT_PYTHON_PACKAGE,
    fasttext_loader_boundary_status,
    is_fasttext_available,
    load_fasttext_seed,
    mark_seed_as_loaded,
    unload_fasttext_seed,
    verify_fasttext_seed_checksum,
)
from main import build_full_engine


def test_fasttext_loader_module_exists_and_declares_optional_requirement():
    """Round25 introduces the loader boundary without installing/loading a seed."""
    loader_path = Path("adapters/fasttext_loader.py")
    optional_requirements = Path("requirements-optional.txt").read_text(encoding="utf-8")

    assert loader_path.exists()
    assert FASTTEXT_OPTIONAL_REQUIREMENT == "fasttext-wheel>=0.9.2"
    assert FASTTEXT_OPTIONAL_REQUIREMENT in optional_requirements


def test_is_fasttext_available_returns_bool_without_importing_runtime():
    """Availability checks must not import the optional fastText package."""
    sys.modules.pop(FASTTEXT_PYTHON_PACKAGE, None)

    available = is_fasttext_available()

    assert isinstance(available, bool)
    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules


def test_loader_functions_have_explicit_scaffold_signatures():
    """Future load/unload/state functions exist but are not implemented yet."""
    load_sig = inspect.signature(load_fasttext_seed)
    verify_sig = inspect.signature(verify_fasttext_seed_checksum)
    mark_sig = inspect.signature(mark_seed_as_loaded)

    assert "file_path" in load_sig.parameters
    assert load_sig.parameters["expected_checksum"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "expected_checksum" in verify_sig.parameters
    assert verify_sig.parameters["expected_checksum"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(mark_sig.parameters) == ["seed_name", "model_handle"]
    assert callable(unload_fasttext_seed)


def test_load_unload_and_loaded_transition_are_explicit_not_default():
    """Round26 implements explicit functions but keeps default state registered."""
    try:
        load_fasttext_seed("/does/not/exist/cc.ko.300.bin")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing fastText seed must fail closed")

    unload_result = unload_fasttext_seed(object())
    loaded_result = mark_seed_as_loaded("cc.ko.300.bin", object())

    assert unload_result["unloaded"] is True
    assert loaded_result["valid"] is True
    assert loaded_result["state_after"] == "loaded"
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    assert external_seed_state(manifest) == SEED_STATE_REGISTERED


def test_external_seed_state_still_registered_after_loader_scaffold():
    """Loader scaffold does not advance registered seed to loaded/used."""
    manifest = load_manifest_file(SEED_MANIFEST_PATH)
    status = fasttext_loader_boundary_status()

    assert external_seed_state(manifest) == SEED_STATE_REGISTERED
    assert status["state_after_round26_without_explicit_load"] == SEED_STATE_REGISTERED
    assert status["actual_file_load_performed"] is False
    assert status["runtime_package_imported"] is False
    assert status["self_embedding_rewrite"] is False


def test_optional_dependency_safe_when_missing_or_present():
    """The full project remains import-safe regardless of optional package state."""
    status = fasttext_loader_boundary_status()

    assert status["package_name"] == "fasttext"
    assert isinstance(status["available"], bool)
    assert status["actual_file_load_performed"] is False
    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules


def test_verify_checksum_helper_is_file_read_only_and_does_not_load_fasttext(tmp_path):
    """The future checksum gate reads bytes only and never imports fastText."""
    seed = tmp_path / "tiny-fasttext-fixture.bin"
    seed.write_bytes(b"eve-v3-round25-checksum-fixture")
    from adapters.external_seed_manifest import compute_seed_checksum

    expected = compute_seed_checksum(seed)
    sys.modules.pop(FASTTEXT_PYTHON_PACKAGE, None)

    result = verify_fasttext_seed_checksum(seed, expected_checksum=expected)

    assert result["matches"] is True
    assert result["file_loaded"] is False
    assert result["runtime_package_imported"] is False
    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules


def test_self_embedding_adapter_unchanged_for_loader_scaffold():
    """Round25 must not start the 300d self-embedding rewrite."""
    text = Path("adapters/self_embedding_adapter.py").read_text(encoding="utf-8")

    assert "fastText" not in text
    assert "cc.ko.300.bin" not in text
    assert "fasttext_loader" not in text


def test_no_seed_file_loaded_and_engine_state_unchanged():
    """The scaffold must not mutate AGP, traces, thresholds, or modes."""
    sys.modules.pop(FASTTEXT_PYTHON_PACKAGE, None)
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

    status = fasttext_loader_boundary_status()

    assert status["actual_file_load_performed"] is False
    assert status["runtime_package_imported"] is False
    assert FASTTEXT_PYTHON_PACKAGE not in sys.modules
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
