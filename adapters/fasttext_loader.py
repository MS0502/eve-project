"""EVE v3 round27 — fastText seed verification runner.

Round24 registered the Korean fastText seed provenance in ``seeds/MANIFEST.yaml``.
Round25 added the optional runtime boundary. Round26 implemented explicit
checksum/load helpers. Round27 adds an operator-facing verification runner while
preserving the External Seed Policy separation:

- importing this module does not import the fastText runtime package;
- checksum verification and the round27 runner are file-read only and do not load a model;
- model loading is an explicit function call only;
- if the optional fastText package is unavailable, loading degrades to ``None``;
- missing files and checksum mismatches fail closed;
- loaded/used state is returned as data, not silently written into the manifest;
- ``self_embedding_adapter.py`` remains unchanged in round26.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SEED_NAME,
    FASTTEXT_KOREAN_SHA256,
    SEED_STATE_LOADED,
    SEED_STATE_REGISTERED,
    compute_seed_checksum,
    external_seed_state,
    is_valid_checksum_format,
    load_manifest_file,
)

FASTTEXT_LOADER_POLICY_VERSION = "v3_round27_seed_verification_runner"
FASTTEXT_PYTHON_PACKAGE = "fasttext"
EVE_FASTTEXT_SEED_PATH_ENV = "EVE_FASTTEXT_SEED_PATH"
FASTTEXT_OPTIONAL_REQUIREMENT = "fasttext-wheel>=0.9.2"
FASTTEXT_LOAD_IMPLEMENTATION_ROUND = "v3 round26"


class ChecksumMismatchError(ValueError):
    """Raised when an external seed file checksum differs from manifest data."""


class FastTextRuntimeUnavailable(RuntimeError):
    """Reserved for callers that require fastText instead of graceful ``None``."""


def is_fasttext_available() -> bool:
    """Return whether the optional fastText Python package can be found.

    This function intentionally does not import ``fasttext``. It only inspects
    module availability so CI and Termux environments without the package keep
    working and registered seed provenance is not confused with runtime load.
    """
    return importlib.util.find_spec(FASTTEXT_PYTHON_PACKAGE) is not None




def get_configured_seed_path() -> str | None:
    """Return the explicit operator-provided fastText seed path, if configured.

    Round27 uses this environment variable only for checksum verification. The
    returned path is not loaded as a model by this helper.
    """
    value = os.environ.get(EVE_FASTTEXT_SEED_PATH_ENV)
    if value is None or not value.strip():
        return None
    return value


def seed_verification_runner(
    *,
    seed_name: str = FASTTEXT_KOREAN_SEED_NAME,
    explicit_path: str | Path | None = None,
    manifest_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a registered seed file path without loading fastText.

    This runner is the round27 operator workflow. It resolves an explicit path
    first, then ``EVE_FASTTEXT_SEED_PATH``. If no path is configured, it returns
    ``status='skipped'``. If a path is configured, it computes SHA256 and compares
    the result to the registered manifest checksum. It never imports fastText,
    never calls ``load_fasttext_seed()``, never mutates the manifest, and never
    touches ``self_embedding_adapter.py``.
    """
    manifest = manifest_data if manifest_data is not None else load_manifest_file()
    expected_checksum = _manifest_checksum_for_seed(manifest, seed_name)
    resolved_path = str(explicit_path) if explicit_path is not None else get_configured_seed_path()

    base: dict[str, Any] = {
        "version": FASTTEXT_LOADER_POLICY_VERSION,
        "seed_name": seed_name,
        "path_resolved": resolved_path,
        "checksum_expected": expected_checksum,
        "checksum_computed": None,
        "match": False,
        "fasttext_load_attempted": False,
        "runtime_package_imported": False,
        "manifest_file_modified": False,
        "self_embedding_rewrite": False,
    }

    if expected_checksum is None:
        return {**base, "status": "unregistered", "errors": ["seed_not_registered"]}
    if resolved_path is None:
        return {**base, "status": "skipped", "errors": [], "reason": "seed_path_not_configured"}

    path = Path(resolved_path)
    if not path.is_file():
        return {**base, "status": "missing_file", "errors": ["seed_file_missing"]}

    actual = compute_seed_checksum(path)
    match = actual == expected_checksum
    return {
        **base,
        "status": "verified" if match else "mismatch",
        "checksum_computed": actual,
        "match": match,
        "errors": [] if match else ["checksum_mismatch"],
    }


def _manifest_checksum_for_seed(manifest_data: dict[str, Any], seed_name: str) -> str | None:
    seeds = manifest_data.get("seeds") if isinstance(manifest_data, dict) else None
    if not isinstance(seeds, list):
        return None
    for entry in seeds:
        if isinstance(entry, dict) and entry.get("name") == seed_name:
            checksum = entry.get("checksum")
            return checksum if isinstance(checksum, str) else None
    return None


def fasttext_loader_boundary_status() -> dict[str, Any]:
    """Return the round26 loader boundary status as data only."""
    manifest = load_manifest_file()
    return {
        "version": FASTTEXT_LOADER_POLICY_VERSION,
        "optional_requirement": FASTTEXT_OPTIONAL_REQUIREMENT,
        "package_name": FASTTEXT_PYTHON_PACKAGE,
        "seed_path_env": EVE_FASTTEXT_SEED_PATH_ENV,
        "configured_seed_path_present": get_configured_seed_path() is not None,
        "available": is_fasttext_available(),
        "implementation_round": FASTTEXT_LOAD_IMPLEMENTATION_ROUND,
        "seed_name": FASTTEXT_KOREAN_SEED_NAME,
        "state_after_round26_without_explicit_load": external_seed_state(manifest),
        "actual_file_load_performed": False,
        "runtime_package_imported_at_module_import": False,
        "runtime_package_imported": False,  # round25 compatibility
        "self_embedding_rewrite": False,
    }


def verify_fasttext_seed_checksum(
    file_path: str | Path,
    *,
    expected_checksum: str,
) -> dict[str, Any]:
    """Compute SHA256 for a seed file and compare it to manifest checksum.

    This helper reads bytes only. It never imports fastText, never loads a model,
    and never mutates the seed manifest or embedding adapters.
    """
    if not is_valid_checksum_format(expected_checksum):
        raise ValueError("expected_checksum_must_be_SHA256_hex")

    actual = compute_seed_checksum(file_path)
    return {
        "version": FASTTEXT_LOADER_POLICY_VERSION,
        "file_path": str(file_path),
        "expected_checksum": expected_checksum,
        "actual_checksum": actual,
        "match": actual == expected_checksum,
        "matches": actual == expected_checksum,  # round25 compatibility
        "verified": actual == expected_checksum,
        "file_loaded": False,
        "runtime_package_imported": False,
    }


def load_fasttext_seed(
    file_path: str | Path,
    *,
    expected_checksum: str | None = None,
) -> Any | None:
    """Explicitly load a fastText seed model if the runtime is available.

    The function is intentionally not called by default project startup. It is a
    manual gateway for future rounds. Safety behavior:

    - missing path -> ``FileNotFoundError``;
    - malformed checksum -> ``ValueError``;
    - checksum mismatch -> ``ChecksumMismatchError``;
    - fastText package unavailable -> ``None`` (graceful optional dependency);
    - successful path -> ``fasttext.load_model(str(path))`` result.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    if expected_checksum is not None:
        result = verify_fasttext_seed_checksum(path, expected_checksum=expected_checksum)
        if not result["match"]:
            raise ChecksumMismatchError(
                f"fasttext_seed_checksum_mismatch: expected={result['expected_checksum']} actual={result['actual_checksum']}"
            )

    if not is_fasttext_available():
        return None

    fasttext_module = importlib.import_module(FASTTEXT_PYTHON_PACKAGE)
    return fasttext_module.load_model(str(path))


def unload_fasttext_seed(model: Any) -> dict[str, Any]:
    """Release a future fastText model handle by dropping references.

    Python fastText handles do not expose a stable close API. This function is a
    deterministic, explicit boundary for callers: it performs no global mutation
    and returns data documenting that the caller should discard the handle.
    """
    return {
        "version": FASTTEXT_LOADER_POLICY_VERSION,
        "unloaded": True,
        "model_handle_present": model is not None,
        "runtime_package_imported": False,
        "self_embedding_rewrite": False,
    }


def mark_seed_as_loaded(seed_name: str, model_handle: Any) -> dict[str, Any]:
    """Return an explicit registered -> loaded state transition as data.

    The manifest remains provenance-only. Runtime state must be passed explicitly
    to ``external_seed_state(..., loaded_seed_names={...})`` by future callers;
    this function does not silently persist anything.
    """
    manifest = load_manifest_file()
    state_before = external_seed_state(manifest)
    errors: list[str] = []

    registered_names = {
        entry.get("name")
        for entry in manifest.get("seeds", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if seed_name not in registered_names:
        errors.append("seed_not_registered")
    if model_handle is None:
        errors.append("model_handle_required")

    loaded_names = {seed_name} if not errors else set()
    state_after = external_seed_state(manifest, loaded_seed_names=loaded_names)

    return {
        "version": FASTTEXT_LOADER_POLICY_VERSION,
        "seed_name": seed_name,
        "marked": not errors,
        "valid": not errors,
        "errors": errors,
        "state_before": state_before,
        "state_after": state_after if not errors else state_before,
        "loaded_seed_names": sorted(loaded_names),
        "manifest_file_modified": False,
        "self_embedding_rewrite": False,
    }
