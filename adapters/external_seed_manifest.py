"""EVE v3 round29 — External Seed Policy manifest + subset registration.

# NOTE: This file includes logic for seed registration and drift measurement.

Round21 turned the v3 External Seed Policy into a manifest boundary.
Round22 added a read-only seed registration dry-run. Round23 added the
acquisition workflow and state ladder. Round24 registers the first concrete
external seed provenance entry for the Korean fastText crawl vector.

Round29 also registers a mini 1k subset extracted outside the runtime. It still does not use the subset in self_embedding:

- no fastText library import
- no self_embedding_adapter rewrite
- no AGP, trace, memory, or embedding state mutation
- seeds/MANIFEST.yaml records provenance only
- external seed state advances from unregistered to registered
"""

from __future__ import annotations

import hashlib
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


SEED_MANIFEST_PATH = Path("seeds/MANIFEST.yaml")
SEED_POLICY_VERSION = "v3_round24_external_seed_policy_strict_registration"

SEED_STATE_UNREGISTERED = "unregistered"
SEED_STATE_REGISTERED = "registered"
SEED_STATE_LOADED = "loaded"
SEED_STATE_USED = "used"
SEED_STATE_INVALID = "invalid"

LICENSE_WHITELIST = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "CC0-1.0",
        "public-domain",
    }
)

REQUIRED_SEED_FIELDS = frozenset(
    {
        "name",
        "source",
        "license",
        "version",
        "downloaded_at",
        "checksum",
        "imported_at_round",
        "imported_at_patch",
    }
)

FORBIDDEN_GENERATOR_MARKERS = (
    "gpt",
    "claude",
    "gemma",
    "bert",
    "rwkv",
    "mamba",
    "ssm",
    "llm",
    "transformer",
    "generator",
    "body",
)

CHECKSUM_PLACEHOLDERS = frozenset(
    {
        "SHA256:(future)",
        "SHA256:<future>",
        "SHA256:<64 lowercase hex chars>",
    }
)

DATE_PLACEHOLDERS = frozenset({"(future)", "YYYY-MM-DD"})

FASTTEXT_KOREAN_SEED_NAME = "cc.ko.300.bin"
FASTTEXT_KOREAN_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz"
FASTTEXT_KOREAN_SHA256 = "SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7"
FASTTEXT_KOREAN_FILE_SIZE_BYTES = 7243669409
FASTTEXT_KOREAN_FILE_LOCATION = "external (Google Drive: /eve_seeds/cc.ko.300.bin.gz)"
FASTTEXT_KOREAN_IMPORTED_ROUND = 24
FASTTEXT_KOREAN_IMPORTED_PATCH = "v3_round24"


SUBSET_ENTRY_TYPE = "subset"
SUBSET_STATUS_EXTRACTED = "extracted"
SUBSET_STATUS_VALIDATED = "validated"
SUBSET_STATE_MISSING = "missing"
SUBSET_STATE_EXTRACTED = "extracted"
SUBSET_STATE_VALIDATED = "validated"
SUBSET_STATE_LOADED = "loaded"
SUBSET_STATE_USED = "used"
SUBSET_STATE_INVALID = "invalid"

FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME = "cc.ko.300.subset.mini.1k"
FASTTEXT_KOREAN_SUBSET_MINI_1K_PARENT = FASTTEXT_KOREAN_SEED_NAME
FASTTEXT_KOREAN_SUBSET_MINI_1K_SELECTION_METHOD = "fasttext_frequency_order_top_k"
FASTTEXT_KOREAN_SUBSET_MINI_1K_FILTER_RULE = "remove_words_containing_unicode_replacement_char"
FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_SIZE = 1000
FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTOR_DIM = 300
FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM = "SHA256:5e7eb8da5cb96f9c1d207846f8f4ef781f48f74a0f9b3eb4ee112cdc47d53064"
FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM = "SHA256:ac987683f9ad733cea97ed13ff5b39e73a2ed91ba5d489466f690b2e747d6f2a"
FASTTEXT_KOREAN_SUBSET_MINI_1K_MANIFEST_CHECKSUM = "SHA256:137631552e9a0c6efca941c48e0fa4bd35854ba59519d87dcdcdfa29a38b1caa"
FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH = "seeds/subsets/cc.ko.300.subset.mini.1k"
FASTTEXT_KOREAN_SUBSET_MINI_1K_IMPORTED_ROUND = 29
FASTTEXT_KOREAN_SUBSET_MINI_1K_IMPORTED_PATCH = "v3_round29"

FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME = "cc.ko.300.subset.small.5k"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_PARENT = FASTTEXT_KOREAN_SEED_NAME
FASTTEXT_KOREAN_SUBSET_SMALL_5K_SELECTION_METHOD = "fasttext_frequency_order_top_k"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_FILTER_RULE = "remove_words_containing_unicode_replacement_char"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_SIZE = 5000
FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTOR_DIM = 300
FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM = "SHA256:c8d0d5ab119e7da119b7ca14d28a48f13767d2b86f8860ce0edbd51158cb701b"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM = "SHA256:6b55eb8eef00a003164b3ae5ce74424673d3539a0007e94f27edf08683bb2309"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_MANIFEST_CHECKSUM = "SHA256:fddb35096e6acb4b541a5a96f28de4cc0478b56b28d724f8a2718e846bb0a856"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH = "seeds/subsets/cc.ko.300.subset.small.5k"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_PURPOSE = "production_lexical_seed"
FASTTEXT_KOREAN_SUBSET_SMALL_5K_IMPORTED_ROUND = 31
FASTTEXT_KOREAN_SUBSET_SMALL_5K_IMPORTED_PATCH = "v3_round31"

FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME = "cc.ko.300.subset.medium.30k"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PARENT = FASTTEXT_KOREAN_SEED_NAME
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_SELECTION_METHOD = "fasttext_frequency_order_top_k"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_FILTER_RULE = "remove_words_containing_unicode_replacement_char"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE = 30000
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM = 300
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM = "SHA256:dc0e6985e767003f13d74bf1a16c257be29e3a125dabb16a320cd13082aef308"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM = "SHA256:f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_MANIFEST_CHECKSUM = "SHA256:28222ff89e1fa0d2c20463eeef79729d731bf0484f0f4f8161863088c2724c3a"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH = "seeds/subsets/cc.ko.300.subset.medium.30k"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE = "production_lexical_seed_expanded"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_IMPORTED_ROUND = 50
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_IMPORTED_PATCH = "v3_round50"
FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_SUPERSEDES_SMALL = True

SUBSET_PURPOSE_FIXTURE = "fixture_level_extraction_test"
SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED = "production_lexical_seed"
SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED_EXPANDED = "production_lexical_seed_expanded"
ALLOWED_SUBSET_PURPOSES = frozenset({
    SUBSET_PURPOSE_FIXTURE,
    SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED,
    SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED_EXPANDED,
})

REQUIRED_SUBSET_FIELDS = frozenset(
    {
        "entry_type",
        "name",
        "parent_seed",
        "parent_checksum",
        "selection_method",
        "filter_rule",
        "vocab_size",
        "vector_dim",
        "vocab_file",
        "vectors_file",
        "subset_manifest_file",
        "vocab_checksum",
        "vectors_checksum",
        "subset_manifest_checksum",
        "extracted_at",
        "imported_at_round",
        "imported_at_patch",
        "status",
    }
)

_CHECKSUM_RE = re.compile(r"^SHA256:[0-9a-fA-F]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_URL_RE = re.compile(r"^https?://")


def validate_seed_entry(
    entry: dict[str, Any],
    *,
    allow_placeholders: bool = False,
) -> dict[str, Any]:
    """Validate one external seed manifest entry.

    The function is read-only: it deep-copies the input, validates simple
    provenance/license/checksum rules, and returns structured data only.

    ``allow_placeholders`` is intended only for round22 dry-runs. It lets a
    future checksum/date placeholder pass with warnings instead of errors. Real
    manifest imports must call the default strict mode.
    """
    snapshot = deepcopy(entry) if isinstance(entry, dict) else entry
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(snapshot, dict):
        return {
            "valid": False,
            "errors": ["seed_entry_must_be_mapping"],
            "warnings": warnings,
            "seed_name": None,
            "placeholder_fields": [],
        }

    missing = sorted(REQUIRED_SEED_FIELDS.difference(snapshot.keys()))
    if missing:
        errors.extend(f"missing_required_field:{field}" for field in missing)

    name = snapshot.get("name")
    if not _non_empty_string(name):
        errors.append("invalid_name")
    elif _contains_forbidden_generator_marker(name):
        errors.append("forbidden_generation_body_marker")

    source = snapshot.get("source")
    if not _non_empty_string(source) or not _URL_RE.match(str(source)):
        errors.append("invalid_source_url")

    license_value = snapshot.get("license")
    if license_value not in LICENSE_WHITELIST:
        errors.append("license_not_whitelisted")

    version = snapshot.get("version")
    if not _non_empty_string(version):
        errors.append("invalid_version")

    placeholder_fields: list[str] = []

    downloaded_at = snapshot.get("downloaded_at")
    if _is_date_placeholder(downloaded_at):
        placeholder_fields.append("downloaded_at")
        if allow_placeholders:
            warnings.append("placeholder_downloaded_at")
        else:
            errors.append("invalid_downloaded_at")
    elif not _non_empty_string(downloaded_at) or not _DATE_RE.match(str(downloaded_at)):
        errors.append("invalid_downloaded_at")

    checksum = snapshot.get("checksum")
    if _is_checksum_placeholder(checksum):
        placeholder_fields.append("checksum")
        if allow_placeholders:
            warnings.append("placeholder_checksum")
        else:
            errors.append("invalid_checksum")
    elif not _non_empty_string(checksum) or not _CHECKSUM_RE.match(str(checksum)):
        errors.append("invalid_checksum")

    imported_round = snapshot.get("imported_at_round")
    if not isinstance(imported_round, int) or imported_round < 1:
        errors.append("invalid_imported_at_round")

    imported_patch = snapshot.get("imported_at_patch")
    if not _non_empty_string(imported_patch):
        errors.append("invalid_imported_at_patch")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "seed_name": name if isinstance(name, str) else None,
        "placeholder_fields": placeholder_fields,
    }



def validate_subset_entry(
    entry: dict[str, Any],
    manifest_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one extracted external-seed subset manifest entry.

    A subset is not a new generation body and is not a runtime-loaded seed. It is
    a deterministic artifact derived from a registered parent seed. Validation
    checks provenance, checksum formats, parent-seed linkage, and extraction
    metadata only; it does not load vectors or modify runtime state.
    """
    snapshot = deepcopy(entry) if isinstance(entry, dict) else entry
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(snapshot, dict):
        return {
            "valid": False,
            "errors": ["subset_entry_must_be_mapping"],
            "warnings": warnings,
            "subset_name": None,
            "parent_seed": None,
        }

    missing = sorted(REQUIRED_SUBSET_FIELDS.difference(snapshot.keys()))
    if missing:
        errors.extend(f"missing_required_field:{field}" for field in missing)

    if snapshot.get("entry_type") != SUBSET_ENTRY_TYPE:
        errors.append("invalid_entry_type")

    name = snapshot.get("name")
    if not _non_empty_string(name):
        errors.append("invalid_name")
    elif _contains_forbidden_generator_marker(name):
        errors.append("forbidden_generation_body_marker")

    parent_seed = snapshot.get("parent_seed")
    if not _non_empty_string(parent_seed):
        errors.append("invalid_parent_seed")

    parent_checksum = snapshot.get("parent_checksum")
    if not is_valid_checksum_format(parent_checksum):
        errors.append("invalid_parent_checksum")

    if manifest_data is not None and isinstance(manifest_data, dict):
        parent_entries = [
            candidate
            for candidate in manifest_data.get("seeds", [])
            if isinstance(candidate, dict)
            and candidate.get("name") == parent_seed
            and not _is_subset_entry(candidate)
        ]
        if not parent_entries:
            errors.append("parent_seed_not_registered")
        elif parent_checksum != parent_entries[0].get("checksum"):
            errors.append("parent_checksum_mismatch")

    if snapshot.get("selection_method") != FASTTEXT_KOREAN_SUBSET_MINI_1K_SELECTION_METHOD:
        errors.append("unsupported_selection_method")
    if snapshot.get("filter_rule") != FASTTEXT_KOREAN_SUBSET_MINI_1K_FILTER_RULE:
        errors.append("unsupported_filter_rule")

    if not isinstance(snapshot.get("vocab_size"), int) or snapshot.get("vocab_size") <= 0:
        errors.append("invalid_vocab_size")
    if not isinstance(snapshot.get("vector_dim"), int) or snapshot.get("vector_dim") <= 0:
        errors.append("invalid_vector_dim")

    for file_field in ("vocab_file", "vectors_file", "subset_manifest_file", "file_location"):
        if file_field in snapshot and not _non_empty_string(snapshot.get(file_field)):
            errors.append(f"invalid_{file_field}")

    for checksum_field in ("vocab_checksum", "vectors_checksum", "subset_manifest_checksum"):
        if not is_valid_checksum_format(snapshot.get(checksum_field)):
            errors.append(f"invalid_{checksum_field}")

    extracted_at = snapshot.get("extracted_at")
    if not _non_empty_string(extracted_at) or not _DATE_RE.match(str(extracted_at)):
        errors.append("invalid_extracted_at")

    imported_round = snapshot.get("imported_at_round")
    if not isinstance(imported_round, int) or imported_round < 1:
        errors.append("invalid_imported_at_round")

    imported_patch = snapshot.get("imported_at_patch")
    if not _non_empty_string(imported_patch):
        errors.append("invalid_imported_at_patch")

    if snapshot.get("status") not in {SUBSET_STATUS_EXTRACTED, SUBSET_STATUS_VALIDATED}:
        errors.append("invalid_subset_status")

    purpose = snapshot.get("purpose")
    if purpose is not None and purpose not in ALLOWED_SUBSET_PURPOSES:
        errors.append("invalid_subset_purpose")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "subset_name": name if isinstance(name, str) else None,
        "parent_seed": parent_seed if isinstance(parent_seed, str) else None,
    }


def _is_subset_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and (entry.get("entry_type") == SUBSET_ENTRY_TYPE or "parent_seed" in entry)

def validate_manifest(manifest_data: dict[str, Any]) -> dict[str, Any]:
    """Validate a full external seed manifest dictionary.

    Empty manifests are valid because no seed has been imported yet. Non-empty
    manifests must satisfy every seed-entry provenance rule in strict mode.
    """
    snapshot = deepcopy(manifest_data) if isinstance(manifest_data, dict) else manifest_data
    errors: list[str] = []
    warnings: list[str] = []
    entry_results: list[dict[str, Any]] = []

    if not isinstance(snapshot, dict):
        return {
            "version": SEED_POLICY_VERSION,
            "valid": False,
            "errors": ["manifest_must_be_mapping"],
            "warnings": warnings,
            "seeds_count": 0,
            "external_seed_loaded": False,
            "entry_results": [],
        }

    seeds = snapshot.get("seeds")
    if seeds is None:
        errors.append("missing_seeds_list")
        seeds = []
    elif not isinstance(seeds, list):
        errors.append("seeds_must_be_list")
        seeds = []

    seen_names: set[str] = set()
    seed_entries_count = 0
    subset_entries_count = 0
    for index, entry in enumerate(seeds):
        if _is_subset_entry(entry):
            result = validate_subset_entry(entry, snapshot)
            entry_name = result.get("subset_name")
            entry_type = SUBSET_ENTRY_TYPE
            subset_entries_count += 1
        else:
            result = validate_seed_entry(entry)
            entry_name = result.get("seed_name")
            entry_type = "seed"
            seed_entries_count += 1
        if entry_name:
            if entry_name in seen_names:
                result = {**result, "valid": False, "errors": [*result["errors"], "duplicate_entry_name"]}
            seen_names.add(entry_name)
        entry_results.append({"index": index, "entry_type": entry_type, **result})
        warnings.extend(f"seed[{index}]:{warning}" for warning in result.get("warnings", []))
        if not result["valid"]:
            errors.extend(f"seed[{index}]:{error}" for error in result["errors"])

    return {
        "version": SEED_POLICY_VERSION,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "seeds_count": len(seeds),
        "external_seed_loaded": seed_entries_count > 0,
        "entry_results": entry_results,
        "seed_entries_count": seed_entries_count,
        "subset_entries_count": subset_entries_count,
    }



def is_valid_checksum_format(checksum: Any) -> bool:
    """Return whether a checksum has the strict ``SHA256:<64 hex>`` shape."""
    return isinstance(checksum, str) and bool(_CHECKSUM_RE.match(checksum))


def compute_seed_checksum(file_path: str | Path) -> str:
    """Compute a deterministic SHA256 checksum for a seed file.

    This helper reads bytes only. It does not parse, load, import, or otherwise
    use the seed as a model. Missing files raise ``FileNotFoundError`` so future
    registration rounds cannot silently replace real provenance with a fake
    checksum.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "SHA256:" + digest.hexdigest()




def fasttext_korean_seed_registered_entry() -> dict[str, Any]:
    """Return the concrete round24 fastText Korean seed manifest entry.

    This provenance was verified outside the repository before round24. The
    manifest entry records the seed as an external artifact only; the binary is
    not committed, loaded, or used by this function.
    """
    return {
        "name": FASTTEXT_KOREAN_SEED_NAME,
        "source": "https://fasttext.cc/docs/en/crawl-vectors.html",
        "download_url": FASTTEXT_KOREAN_DOWNLOAD_URL,
        "license": "CC-BY-SA-3.0",
        "version": "2017-12",
        "downloaded_at": "2026-05-11",
        "checksum": FASTTEXT_KOREAN_SHA256,
        "file_size_bytes": FASTTEXT_KOREAN_FILE_SIZE_BYTES,
        "file_location": FASTTEXT_KOREAN_FILE_LOCATION,
        "imported_at_round": FASTTEXT_KOREAN_IMPORTED_ROUND,
        "imported_at_patch": FASTTEXT_KOREAN_IMPORTED_PATCH,
    }




def fasttext_korean_subset_mini_1k_entry() -> dict[str, Any]:
    """Return the concrete round29 mini subset manifest entry.

    The mini 1k subset is a fixture-level extracted artifact. It validates the
    deterministic extraction/provenance pipeline but is not treated as the
    production lexical map and is not used by ``self_embedding_adapter`` here.
    """
    return {
        "entry_type": SUBSET_ENTRY_TYPE,
        "name": FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
        "parent_seed": FASTTEXT_KOREAN_SUBSET_MINI_1K_PARENT,
        "parent_checksum": FASTTEXT_KOREAN_SHA256,
        "selection_method": FASTTEXT_KOREAN_SUBSET_MINI_1K_SELECTION_METHOD,
        "filter_rule": FASTTEXT_KOREAN_SUBSET_MINI_1K_FILTER_RULE,
        "vocab_size": FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_SIZE,
        "vector_dim": FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTOR_DIM,
        "vocab_file": "vocab.txt",
        "vectors_file": "vectors.npy",
        "subset_manifest_file": "subset_manifest.json",
        "vocab_checksum": FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_CHECKSUM,
        "vectors_checksum": FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTORS_CHECKSUM,
        "subset_manifest_checksum": FASTTEXT_KOREAN_SUBSET_MINI_1K_MANIFEST_CHECKSUM,
        "file_location": FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH,
        "extracted_at": "2026-05-11",
        "extracted_by": "operator (Colab + fasttext-wheel)",
        "raw_vocab_size": 2000000,
        "clean_vocab_size": 1999987,
        "filtered_corrupted": 13,
        "determinism_verified": "two identical extractions produced identical checksums",
        "status": SUBSET_STATUS_EXTRACTED,
        "imported_at_round": FASTTEXT_KOREAN_SUBSET_MINI_1K_IMPORTED_ROUND,
        "imported_at_patch": FASTTEXT_KOREAN_SUBSET_MINI_1K_IMPORTED_PATCH,
    }



def fasttext_korean_subset_small_5k_entry() -> dict[str, Any]:
    """Return the concrete round31 small subset manifest entry.

    The small 5k subset is the first production lexical seed candidate. It is
    still only an extracted artifact in round31: no runtime subset load, no
    self_embedding rewrite, and no AGP runtime change happen here.
    """
    return {
        "entry_type": SUBSET_ENTRY_TYPE,
        "name": FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
        "parent_seed": FASTTEXT_KOREAN_SUBSET_SMALL_5K_PARENT,
        "parent_checksum": FASTTEXT_KOREAN_SHA256,
        "selection_method": FASTTEXT_KOREAN_SUBSET_SMALL_5K_SELECTION_METHOD,
        "filter_rule": FASTTEXT_KOREAN_SUBSET_SMALL_5K_FILTER_RULE,
        "vocab_size": FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_SIZE,
        "vector_dim": FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTOR_DIM,
        "vocab_file": "vocab.txt",
        "vectors_file": "vectors.npy",
        "subset_manifest_file": "subset_manifest.json",
        "vocab_checksum": FASTTEXT_KOREAN_SUBSET_SMALL_5K_VOCAB_CHECKSUM,
        "vectors_checksum": FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTORS_CHECKSUM,
        "subset_manifest_checksum": FASTTEXT_KOREAN_SUBSET_SMALL_5K_MANIFEST_CHECKSUM,
        "file_location": FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
        "extracted_at": "2026-05-11",
        "extracted_by": "operator (Colab + fasttext-wheel)",
        "raw_vocab_size": 2000000,
        "clean_vocab_size": 1999987,
        "filtered_corrupted": 13,
        "determinism_verified": "two identical extractions produced identical checksums",
        "purpose": FASTTEXT_KOREAN_SUBSET_SMALL_5K_PURPOSE,
        "status": SUBSET_STATUS_EXTRACTED,
        "imported_at_round": FASTTEXT_KOREAN_SUBSET_SMALL_5K_IMPORTED_ROUND,
        "imported_at_patch": FASTTEXT_KOREAN_SUBSET_SMALL_5K_IMPORTED_PATCH,
    }

def load_manifest_file(manifest_path: str | Path = SEED_MANIFEST_PATH) -> dict[str, Any]:
    """Load the small EVE seed manifest format without adding a YAML dependency.

    The parser intentionally supports only the manifest shape that this project
    writes: a top-level ``seeds`` list of flat key/value mappings. It is enough
    for policy enforcement and remains deterministic/stdlib-only.
    """
    path = Path(manifest_path)
    if not path.exists():
        return empty_manifest_template()
    return _parse_manifest_text(path.read_text(encoding="utf-8"))


def register_seed_entry(
    manifest_path: str | Path,
    entry: dict[str, Any],
    *,
    backup_path: str | Path | None = None,
    simulate_atomic_failure: bool = False,
) -> dict[str, Any]:
    """Strictly register one external seed entry with backup + atomic replace.

    This is the only round24 function allowed to write ``MANIFEST.yaml``. It
    validates current manifest and candidate entry in strict mode, rejects
    duplicates, creates a backup before writing, writes a temp file, then replaces
    the target atomically. It still does not load or use the seed file.
    """
    path = Path(manifest_path)
    backup = Path(backup_path) if backup_path is not None else path.with_suffix(path.suffix + ".backup")
    current_text = path.read_text(encoding="utf-8") if path.exists() else "seeds: []\n"
    current_manifest = _parse_manifest_text(current_text)
    current_validation = validate_manifest(current_manifest)
    candidate = deepcopy(entry)
    candidate_validation = validate_seed_entry(candidate)

    errors: list[str] = []
    errors.extend(f"existing_manifest:{error}" for error in current_validation.get("errors", []))
    errors.extend(f"candidate:{error}" for error in candidate_validation.get("errors", []))

    seeds = current_manifest.get("seeds") if isinstance(current_manifest, dict) else []
    if not isinstance(seeds, list):
        seeds = []
    existing_names = {
        existing.get("name")
        for existing in seeds
        if isinstance(existing, dict) and isinstance(existing.get("name"), str)
    }
    candidate_name = candidate_validation.get("seed_name")
    if candidate_name in existing_names:
        errors.append("candidate:duplicate_seed_name")

    if errors:
        return {
            "version": SEED_POLICY_VERSION,
            "registered": False,
            "valid": False,
            "errors": errors,
            "backup_created": False,
            "manifest_file_modified": False,
            "file_loaded": False,
            "self_embedding_rewrite": False,
            "state_after": external_seed_state(current_manifest),
        }

    manifest_after = {"seeds": [*deepcopy(seeds), candidate]}
    after_validation = validate_manifest(manifest_after)
    if not after_validation["valid"]:
        return {
            "version": SEED_POLICY_VERSION,
            "registered": False,
            "valid": False,
            "errors": [f"manifest_after:{error}" for error in after_validation["errors"]],
            "backup_created": False,
            "manifest_file_modified": False,
            "file_loaded": False,
            "self_embedding_rewrite": False,
            "state_after": external_seed_state(current_manifest),
        }

    backup.write_text(current_text, encoding="utf-8")
    rendered = _format_manifest(manifest_after)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    if simulate_atomic_failure:
        tmp.unlink(missing_ok=True)
        return {
            "version": SEED_POLICY_VERSION,
            "registered": False,
            "valid": False,
            "errors": ["simulated_atomic_write_failure"],
            "backup_created": backup.exists(),
            "manifest_file_modified": False,
            "file_loaded": False,
            "self_embedding_rewrite": False,
            "state_after": external_seed_state(current_manifest),
        }
    os.replace(tmp, path)

    return {
        "version": SEED_POLICY_VERSION,
        "registered": True,
        "valid": True,
        "errors": [],
        "backup_created": backup.exists(),
        "backup_path": str(backup),
        "manifest_file_modified": True,
        "file_loaded": False,
        "self_embedding_rewrite": False,
        "state_before": external_seed_state(current_manifest),
        "state_after": external_seed_state(manifest_after),
        "manifest_after": manifest_after,
    }

def external_seed_state(
    manifest_data: dict[str, Any],
    *,
    loaded_seed_names: set[str] | list[str] | tuple[str, ...] | None = None,
    used_seed_names: set[str] | list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the External Seed Policy ladder state.

    State ladder:

    1. ``unregistered`` — no valid manifest entries.
    2. ``registered`` — manifest contains validated seed provenance, but no seed
       has been loaded into a runtime library.
    3. ``loaded`` — at least one registered seed was loaded by a later explicit
       round.
    4. ``used`` — at least one loaded seed is used by an embedding adapter.

    The function is read-only and preserves the legacy ``external_seed_loaded``
    bool, which still means "the manifest declares seed entries" for older
    tests. New migration work should prefer this explicit state ladder.
    """
    validation = validate_manifest(manifest_data)
    if not validation.get("valid"):
        return SEED_STATE_INVALID

    seeds = manifest_data.get("seeds") if isinstance(manifest_data, dict) else None
    if not isinstance(seeds, list) or not seeds:
        return SEED_STATE_UNREGISTERED

    registered_names = {
        entry.get("name")
        for entry in seeds
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    loaded_names = set(loaded_seed_names or ())
    used_names = set(used_seed_names or ())

    if registered_names.intersection(used_names):
        return SEED_STATE_USED
    if registered_names.intersection(loaded_names):
        return SEED_STATE_LOADED
    return SEED_STATE_REGISTERED






def fasttext_korean_subset_medium_30k_entry() -> dict[str, Any]:
    """Return the concrete round50 medium subset manifest entry.

    The medium 30k subset expands the production lexical seed after round49
    selected strategy C (A first, B continuous). It is registered as extracted
    only; round50 does not swap the wrapper primary and does not implement
    self-learning.
    """
    return {
        "entry_type": SUBSET_ENTRY_TYPE,
        "name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "parent_seed": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PARENT,
        "parent_checksum": FASTTEXT_KOREAN_SHA256,
        "selection_method": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_SELECTION_METHOD,
        "filter_rule": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_FILTER_RULE,
        "vocab_size": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_SIZE,
        "vector_dim": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTOR_DIM,
        "vocab_file": "vocab.txt",
        "vectors_file": "vectors.npy",
        "subset_manifest_file": "subset_manifest.json",
        "vocab_checksum": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VOCAB_CHECKSUM,
        "vectors_checksum": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_VECTORS_CHECKSUM,
        "subset_manifest_checksum": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_MANIFEST_CHECKSUM,
        "file_location": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PATH,
        "extracted_at": "2026-05-12",
        "extracted_by": "operator (Colab + fasttext-wheel)",
        "raw_vocab_size": 2000000,
        "clean_vocab_size": 1999987,
        "filtered_corrupted": 13,
        "determinism_verified": "operator deterministic extraction, checksums recorded",
        "purpose": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_PURPOSE,
        "supersedes_small_5k_for_lexical_coverage": True,
        "status": SUBSET_STATUS_EXTRACTED,
        "imported_at_round": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_IMPORTED_ROUND,
        "imported_at_patch": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_IMPORTED_PATCH,
    }


def round50_medium_30k_oov_resolution_data() -> dict[str, Any]:
    """Return the operator-confirmed round44 OOV resolution data.

    Data only. This does not read vectors, load fastText, or update the active
    wrapper primary.
    """
    general_oov = ["어때", "그래", "뭐야", "좋아해", "군대", "코딩"]
    eve_specific_oov = ["EVE", "민석"]
    return {
        "round": 50,
        "subset_name": FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
        "general_korean_oov_from_round44": general_oov,
        "resolved_general_korean_oov": list(general_oov),
        "resolution_count": len(general_oov),
        "resolution_total": len(general_oov),
        "resolution_ratio": 1.0,
        "oov_resolution": "6/6",
        "eve_specific_oov_remaining": eve_specific_oov,
        "self_learning_needed_for": eve_specific_oov,
        "strategy": "C_hybrid",
        "sequence": "A_first_medium_30k_then_B_continuous_self_learning",
        "read_only": True,
        "wrapper_primary_swap": False,
    }

def subset_state(
    manifest_data: dict[str, Any],
    subset_name: str,
    *,
    loaded_subset_names: set[str] | list[str] | tuple[str, ...] | None = None,
    used_subset_names: set[str] | list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the state of one extracted subset.

    Round29 only reaches ``extracted``. Future rounds may explicitly pass loaded
    or used subset names after runtime integration. This function never reads the
    vector files and never mutates the manifest.
    """
    validation = validate_manifest(manifest_data)
    if not validation.get("valid"):
        return SUBSET_STATE_INVALID

    subsets = [
        entry
        for entry in manifest_data.get("seeds", [])
        if _is_subset_entry(entry) and entry.get("name") == subset_name
    ]
    if not subsets:
        return SUBSET_STATE_MISSING

    loaded_names = set(loaded_subset_names or ())
    used_names = set(used_subset_names or ())
    if subset_name in used_names:
        return SUBSET_STATE_USED
    if subset_name in loaded_names:
        return SUBSET_STATE_LOADED
    if subsets[0].get("status") == SUBSET_STATUS_VALIDATED:
        return SUBSET_STATE_VALIDATED
    return SUBSET_STATE_EXTRACTED


def audit_subset_artifact(
    manifest_data: dict[str, Any] | None = None,
    *,
    subset_name: str = FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    subset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only audit for an extracted subset artifact.

    Round30 uses this as a pre-rewrite audit gate. It validates manifest linkage,
    file presence, checksums, vocab size, vector shape/dtype, and corruption
    filtering without loading the subset into any runtime adapter and without
    importing fastText. Missing files are reported as data instead of raising so
    the base engine can still run without optional subset artifacts.
    """
    manifest = deepcopy(manifest_data) if manifest_data is not None else load_manifest_file(SEED_MANIFEST_PATH)
    validation = validate_manifest(manifest)
    subset_entries = [
        entry
        for entry in manifest.get("seeds", [])
        if _is_subset_entry(entry) and entry.get("name") == subset_name
    ] if isinstance(manifest, dict) else []

    errors: list[str] = []
    if not validation.get("valid"):
        errors.extend(f"manifest:{error}" for error in validation.get("errors", []))
    if not subset_entries:
        errors.append("subset_not_registered")
        entry: dict[str, Any] | None = None
    else:
        entry = deepcopy(subset_entries[0])
        subset_validation = validate_subset_entry(entry, manifest)
        if not subset_validation.get("valid"):
            errors.extend(f"subset:{error}" for error in subset_validation.get("errors", []))

    base_dir = Path(subset_dir) if subset_dir is not None else Path(
        entry.get("file_location", FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH) if entry else FASTTEXT_KOREAN_SUBSET_MINI_1K_PATH
    )
    file_checks: dict[str, Any] = {
        "subset_dir": str(base_dir),
        "dir_exists": base_dir.is_dir(),
        "files": {},
    }

    vocab_lines_count: int | None = None
    corrupted_words_count: int | None = None
    vector_shape: tuple[int, ...] | None = None
    vector_dtype: str | None = None

    if entry is not None:
        for logical_name, filename_key, checksum_key in (
            ("vocab", "vocab_file", "vocab_checksum"),
            ("vectors", "vectors_file", "vectors_checksum"),
            ("subset_manifest", "subset_manifest_file", "subset_manifest_checksum"),
        ):
            path = base_dir / str(entry.get(filename_key, ""))
            exists = path.is_file()
            actual_checksum = compute_seed_checksum(path) if exists else None
            expected_checksum = entry.get(checksum_key)
            file_checks["files"][logical_name] = {
                "path": str(path),
                "exists": exists,
                "checksum_expected": expected_checksum,
                "checksum_actual": actual_checksum,
                "checksum_match": bool(exists and actual_checksum == expected_checksum),
            }
            if not exists:
                errors.append(f"missing_{logical_name}_file")
            elif actual_checksum != expected_checksum:
                errors.append(f"{logical_name}_checksum_mismatch")

        vocab_path = base_dir / str(entry.get("vocab_file", ""))
        if vocab_path.is_file():
            vocab_lines = vocab_path.read_text(encoding="utf-8").splitlines()
            vocab_lines_count = len(vocab_lines)
            corrupted_words_count = sum(1 for word in vocab_lines if "\ufffd" in word)
            if vocab_lines_count != entry.get("vocab_size"):
                errors.append("vocab_size_mismatch")
            if corrupted_words_count != 0:
                errors.append("corrupted_vocab_word_present")

        vectors_path = base_dir / str(entry.get("vectors_file", ""))
        if vectors_path.is_file():
            try:
                import numpy as _np

                vectors = _np.load(vectors_path, mmap_mode="r")
                vector_shape = tuple(int(x) for x in vectors.shape)
                vector_dtype = str(vectors.dtype)
                if vector_shape != (entry.get("vocab_size"), entry.get("vector_dim")):
                    errors.append("vectors_shape_mismatch")
                if vector_dtype != "float32":
                    errors.append("vectors_dtype_mismatch")
            except Exception as exc:  # pragma: no cover - defensive audit data
                errors.append(f"vectors_load_failed:{type(exc).__name__}")

    return {
        "version": SEED_POLICY_VERSION,
        "audit_layer": "subset_artifact",
        "subset_name": subset_name,
        "valid": not errors,
        "errors": errors,
        "manifest_valid": bool(validation.get("valid")),
        "subset_state": subset_state(manifest, subset_name) if isinstance(manifest, dict) else SUBSET_STATE_INVALID,
        "file_checks": file_checks,
        "vocab": {
            "line_count": vocab_lines_count,
            "expected_line_count": entry.get("vocab_size") if entry else None,
            "corrupted_words_count": corrupted_words_count,
        },
        "vectors": {
            "shape": vector_shape,
            "expected_shape": (entry.get("vocab_size"), entry.get("vector_dim")) if entry else None,
            "dtype": vector_dtype,
            "expected_dtype": "float32",
        },
        "runtime_used": False,
        "self_embedding_rewrite": False,
        "fasttext_runtime_imported": False,
    }


def assess_self_embedding_rewrite_readiness(
    engine: Any = None,
    manifest_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a read-only pre-rewrite readiness assessment for self_embedding.

    This is decision data only. It does not load any subset, does not mark a
    subset as used, and does not update ``self_embedding_adapter``. Round31 adds
    the small 5k production lexical seed candidate; readiness can now move from
    ``ready_with_concerns`` to ``ready`` while still preserving explicit future
    rewrite gates.
    """
    manifest = deepcopy(manifest_data) if manifest_data is not None else load_manifest_file(SEED_MANIFEST_PATH)
    self_embedding = getattr(engine, "self_embedding", None) if engine is not None else None
    current_dim = getattr(self_embedding, "dim", 50) if self_embedding is not None else 50
    current_vocab_size = len(getattr(self_embedding, "word_counts", {}) or {}) if self_embedding is not None else 0
    current_embeddings_count = len(getattr(self_embedding, "embeddings", {}) or {}) if self_embedding is not None else 0

    subset_entries = [
        entry
        for entry in manifest.get("seeds", [])
        if _is_subset_entry(entry)
    ] if isinstance(manifest, dict) else []
    available_subsets: list[dict[str, Any]] = []
    for entry in subset_entries:
        name = str(entry.get("name"))
        audit = audit_subset_artifact(manifest, subset_name=name)
        available_subsets.append(
            {
                "name": name,
                "method": "fasttext_extracted",
                "dimension": entry.get("vector_dim"),
                "vocab_size": entry.get("vocab_size"),
                "state": subset_state(manifest, name),
                "purpose": entry.get("purpose", SUBSET_PURPOSE_FIXTURE),
                "audit_valid": bool(audit.get("valid")),
                "character": (
                    "production lexical seed candidate"
                    if entry.get("purpose") in {SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED, SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED_EXPANDED}
                    else "fixture-level extraction/checksum pipeline artifact"
                ),
            }
        )

    production_candidates = [
        subset for subset in available_subsets
        if subset["purpose"] in {SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED, SUBSET_PURPOSE_PRODUCTION_LEXICAL_SEED_EXPANDED} and subset["audit_valid"]
    ]
    preferred_subset = max(production_candidates or available_subsets, key=lambda item: int(item.get("vocab_size") or 0), default=None)
    ready = preferred_subset is not None and preferred_subset.get("audit_valid") is True
    readiness = "ready" if production_candidates else ("ready_with_concerns" if ready else "needs_more_audit")
    target_dim = preferred_subset.get("dimension") if preferred_subset else FASTTEXT_KOREAN_SUBSET_MINI_1K_VECTOR_DIM
    target_vocab = preferred_subset.get("vocab_size") if preferred_subset else FASTTEXT_KOREAN_SUBSET_MINI_1K_VOCAB_SIZE

    concerns = []
    if not production_candidates:
        concerns.extend(
            [
                "mini_1k_is_fixture_level_not_production_vocabulary",
                "top_frequency_tokens_include_punctuation_and_short_morphemes",
                "self_embedding_rewrite_with_mini_1k_has_low_semantic_value",
            ]
        )
    else:
        concerns.extend(
            [
                "small_5k_is_extracted_but_not_runtime_loaded",
                "self_embedding_dimension_change_still_requires_scaffold_and_audit",
                "use_intersection_with_eve_accumulated_vocabulary_before_full_rewrite",
            ]
        )

    return {
        "version": SEED_POLICY_VERSION,
        "assessment_layer": "self_embedding_rewrite_readiness",
        "current_embedding": {
            "method": "PMI+SVD",
            "dimension": current_dim,
            "vocab_size": current_vocab_size,
            "trained_embeddings_count": current_embeddings_count,
            "adapter_present": self_embedding is not None,
        },
        "available_subset": preferred_subset,
        "available_subsets": sorted(available_subsets, key=lambda item: int(item.get("vocab_size") or 0)),
        "migration_risk": {
            "dimension_change": f"{current_dim} -> {target_dim}",
            "vocab_change": f"{current_vocab_size} -> {target_vocab}",
            "affected_modules": [
                "self_embedding_adapter",
                "thought_chain_adapter",
                "concept_memory_adapter",
                "compositor_adapter",
                "attention_analyzer",
            ],
            "risk_level": "medium",
        },
        "readiness": readiness,
        "concerns": concerns,
        "recommendation_data": {
            "next_step": "self_embedding_300d_scaffold",
            "preferred_subset": preferred_subset.get("name") if preferred_subset else None,
            "automatic_application": False,
        },
        "runtime_used": False,
        "self_embedding_rewrite": False,
        "state_transition": False,
    }

def fasttext_korean_acquisition_workflow(seed_file_path: str | Path | None = None) -> dict[str, Any]:
    """Return the round23 acquisition workflow for the Korean fastText seed.

    The workflow is structured data only. It performs no download, does not
    mutate ``seeds/MANIFEST.yaml``, and does not import the fastText binary. If a
    path is supplied and exists, the result marks the file as ready for a future
    registration round; otherwise round23 stays in ``acquisition_deferred``.
    """
    candidate = fasttext_korean_seed_dryrun_entry()
    path = Path(seed_file_path) if seed_file_path is not None else Path("seeds/cc.ko.300.bin")
    file_present = path.is_file()

    strict_candidate = validate_seed_entry(candidate)

    return {
        "version": SEED_POLICY_VERSION,
        "seed_name": candidate["name"],
        "state_before": SEED_STATE_UNREGISTERED,
        "workflow_state": "ready_for_registration" if file_present else "acquisition_deferred",
        "candidate_entry": candidate,
        "candidate_strict_validation": strict_candidate,
        "valid_if_registered_now": strict_candidate["valid"],
        "expected_source": candidate["source"],
        "expected_license": candidate["license"],
        "required_before_registration": ["downloaded_at", "checksum"],
        "seed_file_path": str(path),
        "seed_file_present": file_present,
        "download_required": not file_present,
        "actual_download_performed": False,
        "file_loaded": False,
        "manifest_file_modified": False,
        "self_embedding_rewrite": False,
        "next_valid_state": SEED_STATE_REGISTERED,
    }


def dryrun_register_seed(
    candidate_entry: dict[str, Any],
    existing_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Simulate adding a seed entry without loading files or mutating manifest.

    This is the round22 workflow gate. It accepts future placeholders for fields
    that cannot be known before download, reports them as warnings, and returns a
    simulated manifest snapshot only. The actual ``seeds/MANIFEST.yaml`` file is
    never modified by this function.
    """
    candidate_snapshot = deepcopy(candidate_entry)
    manifest_snapshot = deepcopy(existing_manifest)

    existing_validation = validate_manifest(manifest_snapshot)
    candidate_validation = validate_seed_entry(candidate_snapshot, allow_placeholders=True)

    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(f"existing_manifest:{error}" for error in existing_validation.get("errors", []))
    warnings.extend(f"existing_manifest:{warning}" for warning in existing_validation.get("warnings", []))
    errors.extend(f"candidate:{error}" for error in candidate_validation.get("errors", []))
    warnings.extend(f"candidate:{warning}" for warning in candidate_validation.get("warnings", []))

    seeds = manifest_snapshot.get("seeds") if isinstance(manifest_snapshot, dict) else None
    if not isinstance(seeds, list):
        seeds = []

    candidate_name = candidate_validation.get("seed_name")
    existing_names = {
        entry.get("name")
        for entry in seeds
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if candidate_name in existing_names:
        errors.append("candidate:duplicate_seed_name")

    would_be_added = not errors and candidate_validation["valid"] and existing_validation["valid"]
    manifest_after = deepcopy(manifest_snapshot) if isinstance(manifest_snapshot, dict) else {"seeds": []}
    manifest_after.setdefault("seeds", [])
    if not isinstance(manifest_after["seeds"], list):
        manifest_after["seeds"] = []
    if would_be_added:
        manifest_after["seeds"] = [*deepcopy(seeds), candidate_snapshot]

    return {
        "version": SEED_POLICY_VERSION,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "candidate_result": candidate_validation,
        "existing_manifest_result": existing_validation,
        "would_be_added": would_be_added,
        "external_seed_loaded_after_dryrun": bool(manifest_after.get("seeds")),
        "manifest_after_dryrun": manifest_after,
        "dry_run_only": True,
        "file_loaded": False,
        "manifest_file_modified": False,
    }


def fasttext_korean_seed_dryrun_entry() -> dict[str, Any]:
    """Return the round22 planned fastText Korean seed entry template.

    The checksum/date placeholders are accepted only by ``dryrun_register_seed``;
    a real import round must replace them with concrete provenance values.
    """
    return {
        "name": "cc.ko.300.bin",
        "source": "https://fasttext.cc/docs/en/crawl-vectors.html",
        "license": "CC-BY-SA-3.0",
        "version": "2017-12",
        "downloaded_at": "(future)",
        "checksum": "SHA256:(future)",
        "imported_at_round": 24,
        "imported_at_patch": "v3_round24",
    }


def external_seed_loaded(manifest_data: dict[str, Any]) -> bool:
    """Return whether the manifest declares any external seed entries."""
    validation = validate_manifest(manifest_data)
    return bool(validation["external_seed_loaded"])


def empty_manifest_template() -> dict[str, list[Any]]:
    """Return the placeholder manifest shape."""
    return {"seeds": []}




def _parse_manifest_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped or "seeds: []" in stripped and "  - name:" not in stripped:
        return empty_manifest_template()

    seeds: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "seeds:":
            continue
        if line.startswith("- "):
            if current is not None:
                seeds.append(current)
            current = {}
            remainder = line[2:].strip()
            if remainder:
                key, value = _split_manifest_key_value(remainder)
                current[key] = _parse_manifest_scalar(value)
            continue
        if current is not None and ":" in line:
            key, value = _split_manifest_key_value(line)
            current[key] = _parse_manifest_scalar(value)
    if current is not None:
        seeds.append(current)
    return {"seeds": seeds}


def _split_manifest_key_value(line: str) -> tuple[str, str]:
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_manifest_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.isdigit():
        return int(value)
    return value


def _format_manifest(manifest_data: dict[str, Any]) -> str:
    lines = [
        "# EVE v3 External Seed Manifest",
        "#",
        "# Round24 scope: strict provenance registration only.",
        "# No external seed binary is committed, loaded, parsed, or used here.",
        "seeds:",
    ]
    seeds = manifest_data.get("seeds", []) if isinstance(manifest_data, dict) else []
    if not seeds:
        lines[-1] = "seeds: []"
        return "\n".join(lines) + "\n"
    ordered_keys = [
        "name",
        "source",
        "download_url",
        "license",
        "version",
        "downloaded_at",
        "checksum",
        "file_size_bytes",
        "file_location",
        "entry_type",
        "parent_seed",
        "parent_checksum",
        "selection_method",
        "filter_rule",
        "vocab_size",
        "vector_dim",
        "vocab_file",
        "vectors_file",
        "subset_manifest_file",
        "vocab_checksum",
        "vectors_checksum",
        "subset_manifest_checksum",
        "extracted_at",
        "extracted_by",
        "raw_vocab_size",
        "clean_vocab_size",
        "filtered_corrupted",
        "determinism_verified",
        "purpose",
        "supersedes_small_5k_for_lexical_coverage",
        "status",
        "imported_at_round",
        "imported_at_patch",
    ]
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        lines.append(f"  - name: {_format_manifest_scalar(seed.get('name'))}")
        for key in ordered_keys[1:]:
            if key in seed:
                lines.append(f"    {key}: {_format_manifest_scalar(seed[key])}")
        for key in sorted(k for k in seed.keys() if k not in ordered_keys):
            lines.append(f"    {key}: {_format_manifest_scalar(seed[key])}")
    return "\n".join(lines) + "\n"


def _format_manifest_scalar(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if any(ch in text for ch in [": ", "#", "[", "]", "{", "}"]) or text.startswith("0"):
        return '"' + text.replace('"', '\\"') + '"'
    return text

def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _contains_forbidden_generator_marker(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in FORBIDDEN_GENERATOR_MARKERS)


def _is_checksum_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value in CHECKSUM_PLACEHOLDERS


def _is_date_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value in DATE_PLACEHOLDERS


# ===== v3 round40 pre-swap audit =====

FASTTEXT_PRE_SWAP_AUDIT_VERSION = "v3_round40_pre_swap_audit"
FASTTEXT_PRE_SWAP_AUDIT_MODULES = (
    "attention",
    "compositor",
    "concept_memory",
    "situation_responder",
    "streaming",
    "state_debug",
)
FASTTEXT_TRACE_MODULES = (
    "attention",
    "compositor",
    "concept_memory",
    "situation_responder",
    "streaming",
)


def assess_main_py_swap_readiness(engine: Any) -> dict[str, Any]:
    """Read-only v3 round40 readiness audit for the round41 engine swap.

    The audit intentionally does not call ``fasttext_embedding.load()``, does not
    replace ``engine.self_embedding``, and does not run generation. It only reads
    adapter presence/stats so round41 can decide whether the final main.py swap
    has a safe rollback boundary.
    """
    modules: dict[str, dict[str, Any]] = {}

    def _safe_stats(obj: Any) -> dict[str, Any]:
        if obj is None or not hasattr(obj, "stats"):
            return {}
        try:
            stats = obj.stats()
            return dict(stats) if isinstance(stats, dict) else {}
        except Exception as exc:  # audit must be fail-open/read-only
            return {"stats_error": str(exc)}

    module_objects = {
        "attention": getattr(engine, "attention", None),
        "compositor": getattr(engine, "compositor", None),
        "concept_memory": getattr(engine, "concept_memory", None),
        "situation_responder": getattr(engine, "situation_responder", None),
        "streaming": engine,
        "state_debug": getattr(engine, "state_debug", None),
    }

    for name in FASTTEXT_PRE_SWAP_AUDIT_MODULES:
        obj = module_objects.get(name)
        stats = _safe_stats(obj)
        if name == "state_debug":
            try:
                snapshot = obj.snapshot_state() if obj is not None and hasattr(obj, "snapshot_state") else {}
            except Exception as exc:
                snapshot = {"state_debug_error": str(exc)}
            modules[name] = {
                "present": obj is not None,
                "role": "read_only_debug_surface",
                "in_use_by_generation": "self_embedding",
                "parallel_observation": None,
                "trace_cap": None,
                "trace_size": None,
                "read_only": True,
                "reports_modules": sorted(
                    key for key in FASTTEXT_TRACE_MODULES if isinstance(snapshot, dict) and key in snapshot
                ),
            }
            continue

        modules[name] = {
            "present": obj is not None,
            "role": "parallel_observation",
            "in_use_by_generation": stats.get("in_use_by_generation"),
            "parallel_observation": stats.get("fasttext_parallel_observation"),
            "trace_cap": stats.get("fasttext_trace_cap"),
            "trace_size": stats.get("fasttext_trace_size"),
            "observation_count": stats.get("fasttext_observation_count"),
            "operations_observed": dict(stats.get("operations_observed", {}) or {}),
            "read_only": True,
        }

    self_embedding = getattr(engine, "self_embedding", None)
    fasttext_embedding = getattr(engine, "fasttext_embedding", None)
    fasttext_loaded = False
    if fasttext_embedding is not None and hasattr(fasttext_embedding, "is_loaded"):
        try:
            fasttext_loaded = bool(fasttext_embedding.is_loaded())
        except Exception:
            fasttext_loaded = False

    trace_modules_ready = all(
        modules[name].get("present")
        and modules[name].get("in_use_by_generation") in {"self_embedding", "wrapper"}
        and int(modules[name].get("trace_cap") or 0) == 1000
        for name in FASTTEXT_TRACE_MODULES
    )
    state_debug_ready = bool(modules["state_debug"].get("present")) and modules["state_debug"].get("read_only") is True
    self_embedding_is_pmi_svd = (
        self_embedding is not None
        and (
            (self_embedding.__class__.__name__ == "SelfEmbeddingAdapter" and int(getattr(self_embedding, "dim", 0) or 0) == 50)
            or (self_embedding.__class__.__name__ == "EmbeddingWrapper" and int(getattr(self_embedding, "dim", 0) or 0) == 300)
        )
    )
    fasttext_adapter_ready = (
        fasttext_embedding is not None
        and fasttext_embedding.__class__.__name__ == "FasttextEmbeddingAdapter"
        and int(getattr(fasttext_embedding, "dimension", 0) or 0) == FASTTEXT_KOREAN_SUBSET_SMALL_5K_VECTOR_DIM
    )

    concerns: list[str] = []
    if not self_embedding_is_pmi_svd:
        concerns.append("engine.self_embedding_not_pmi_svd_50d_before_swap")
    if not fasttext_adapter_ready:
        concerns.append("fasttext_adapter_not_ready")
    # round41 post-swap intentionally loads fastText through the wrapper.
    if not trace_modules_ready:
        concerns.append("parallel_observation_modules_not_uniform")
    if not state_debug_ready:
        concerns.append("state_debug_not_ready")

    ready = not concerns
    return {
        "audit_version": FASTTEXT_PRE_SWAP_AUDIT_VERSION,
        "all_6_modules_migrated": trace_modules_ready and state_debug_ready,
        "parallel_observation_modules": list(FASTTEXT_TRACE_MODULES),
        "debug_surface_modules": ["state_debug"],
        "modules": modules,
        "fasttext_adapter_ready": fasttext_adapter_ready,
        "fasttext_loaded": fasttext_loaded,
        "self_embedding_active": "wrapper" if self_embedding is not None and self_embedding.__class__.__name__ == "EmbeddingWrapper" else "self_embedding",
        "self_embedding_class": None if self_embedding is None else self_embedding.__class__.__name__,
        "self_embedding_dimension": int(getattr(self_embedding, "dim", 0) or 0),
        "self_embedding_is_pmi_svd_50d": self_embedding_is_pmi_svd,
        "fail_open_verified": "covered_by_round40_tests",
        "three_systems_coexist": "covered_by_round40_tests",
        "swap_strategy": "round41: replace engine.self_embedding wiring only after audit passes",
        "rollback_strategy": "restore engine.self_embedding = SelfEmbeddingAdapter(engine); keep fasttext unloaded",
        "main_py_affected_areas": ["engine init", "self_embedding wiring", "fasttext_embedding rollback boundary"],
        "swap_readiness": "ready" if ready else "needs_more_audit",
        "concerns": concerns,
        "read_only": True,
    }


def assess_main_py_post_swap_state(engine: Any) -> dict[str, Any]:
    """Read-only v3 round41 post-swap state report.

    This does not load/unload adapters and does not apply rollback. It only
    confirms that the active engine.self_embedding is the final wrapper with
    fastText primary and PMI+SVD fallback.
    """
    self_embedding = getattr(engine, "self_embedding", None)
    backup = getattr(engine, "self_embedding_backup", None)
    fasttext = getattr(engine, "fasttext_embedding", None)
    stats = self_embedding.stats() if hasattr(self_embedding, "stats") else {}
    modules = {}
    module_objects = {
        "attention": getattr(engine, "attention", None),
        "compositor": getattr(engine, "compositor", None),
        "concept_memory": getattr(engine, "concept_memory", None),
        "situation_responder": getattr(engine, "situation_responder", None),
        "streaming": engine,
    }
    for name, obj in module_objects.items():
        try:
            mstats = obj.stats() if hasattr(obj, "stats") else {}
        except Exception as exc:
            mstats = {"stats_error": str(exc)}
        modules[name] = {
            "present": obj is not None,
            "in_use_by_generation": mstats.get("in_use_by_generation"),
            "trace_cap": mstats.get("fasttext_trace_cap"),
        }
    loaded = False
    try:
        loaded = bool(fasttext.is_loaded()) if fasttext is not None and hasattr(fasttext, "is_loaded") else False
    except Exception:
        loaded = False
    concerns = []
    if self_embedding is None or self_embedding.__class__.__name__ != "EmbeddingWrapper":
        concerns.append("self_embedding_not_wrapper")
    if backup is None or backup.__class__.__name__ != "SelfEmbeddingAdapter":
        concerns.append("self_embedding_backup_missing")
    if fasttext is None or fasttext.__class__.__name__ != "FasttextEmbeddingAdapter" or not loaded:
        concerns.append("fasttext_primary_not_loaded")
    if any(data.get("in_use_by_generation") != "wrapper" for data in modules.values()):
        concerns.append("migrated_modules_not_reporting_wrapper")
    return {
        "audit_version": "v3_round41_post_swap",
        "self_embedding_active": None if self_embedding is None else self_embedding.__class__.__name__,
        "self_embedding_backup_available": backup is not None,
        "primary_class": stats.get("primary_class"),
        "fallback_class": stats.get("fallback_class"),
        "primary_loaded": bool(stats.get("primary_loaded", loaded)),
        "dimension": int(stats.get("dimension", 0) or 0),
        "fallback_dimension": int(stats.get("fallback_dimension", 0) or 0),
        "fallback_count": int(stats.get("fallback_count", 0) or 0),
        "modules": modules,
        "migration_progress": "7/7",
        "swap_state": "active" if not concerns else "needs_attention",
        "rollback_strategy": "engine.self_embedding = engine.self_embedding_backup; engine.fasttext_embedding._loaded = False",
        "concerns": concerns,
        "read_only": True,
    }


def _wrapper_telemetry_snapshot(engine: Any) -> dict[str, Any]:
    """Return EmbeddingWrapper telemetry without mutating wrapper state."""
    wrapper = getattr(engine, "self_embedding", None)
    if wrapper is None or not hasattr(wrapper, "telemetry"):
        return {
            "adapter": None,
            "total_calls": 0,
            "primary_hits": 0,
            "fallback_uses": 0,
            "errors": 0,
            "primary_hit_rate": 0.0,
            "fallback_rate": 0.0,
            "error_rate": 0.0,
            "oov_log_size": 0,
            "recent_oov_sample": [],
            "read_only": True,
        }
    try:
        data = wrapper.telemetry()
        return dict(data) if isinstance(data, dict) else {}
    except Exception as exc:
        return {"adapter": wrapper.__class__.__name__, "telemetry_error": exc.__class__.__name__, "read_only": True}


def measure_seed_drift_baseline(engine: Any) -> dict[str, Any]:
    """Read-only v3 round42 seed-drift baseline.

    This is the first Appendix-D drift measurement surface after the round41
    final wrapper swap. It does not adjust thresholds, does not unload/load
    fastText, and does not mutate AGP traces or wrapper counters.
    """
    telemetry = _wrapper_telemetry_snapshot(engine)
    wrapper = getattr(engine, "self_embedding", None)
    fallback = getattr(wrapper, "fallback", None)
    primary = getattr(wrapper, "primary", None)
    fasttext_loaded = False
    try:
        fasttext_loaded = bool(primary.is_loaded()) if primary is not None and hasattr(primary, "is_loaded") else False
    except Exception:
        fasttext_loaded = False
    # For the initial seed-drift baseline (round42), the Eve-specific layer is not yet active.
    # Always use the PMI+SVD fallback as the learning substrate when computing the baseline.
    evespecific_substrate = "PMI+SVD (fallback path)"
    return {
        "measurement_version": "v3_round42_seed_drift_baseline",
        "method": "wrapper_telemetry + agp_trace_correlation",
        "baseline_round": 42,
        "status": "baseline_set_at_round42",
        "fasttext_vector_unchanged": True,
        "fasttext_primary_class": None if primary is None else primary.__class__.__name__,
        "fasttext_loaded": fasttext_loaded,
        "evespecific_learning_substrate": evespecific_substrate,
        "fallback_class": None if fallback is None else fallback.__class__.__name__,
        "fallback_mutable_learning_preserved": fallback is not None and hasattr(fallback, "observe"),
        "drift_indicators": {
            "primary_hit_rate": float(telemetry.get("primary_hit_rate", 0.0) or 0.0),
            "fallback_rate": float(telemetry.get("fallback_rate", 0.0) or 0.0),
            "error_rate": float(telemetry.get("error_rate", 0.0) or 0.0),
            "oov_pattern_log": list(telemetry.get("recent_oov_sample", []) or []),
            "oov_log_size": int(telemetry.get("oov_log_size", 0) or 0),
        },
        "target": "round 200+ avg drift > 0.3 (Draft, v3.1 precise threshold pending)",
        "no_auto_threshold_adjustment": True,
        "no_drift_based_runtime_change": True,
        "read_only": True,
    }


def measure_eve_specific_drift_baseline(engine: Any) -> dict[str, Any]:
    """Read-only v3 round57 seed-drift baseline with Eve-specific integration.

    This baseline captures drift indicators after the EveSpecificVectorStore has been
    inserted between the fastText primary and the PMI+SVD fallback. It reports
    the Eve-specific rate in addition to primary and fallback rates and
    records whether the Eve-specific substrate is active. It does not mutate
    any runtime state and does not adjust AGP or wrapper thresholds.
    """
    telemetry = _wrapper_telemetry_snapshot(engine)
    wrapper = getattr(engine, "self_embedding", None)
    fallback = getattr(wrapper, "fallback", None)
    primary = getattr(wrapper, "primary", None)
    fasttext_loaded = False
    try:
        fasttext_loaded = bool(primary.is_loaded()) if primary is not None and hasattr(primary, "is_loaded") else False
    except Exception:
        fasttext_loaded = False
    evespecific_substrate = "EveSpecificVectorStore (deterministic self-learning)" if getattr(wrapper, "eve_specific", None) is not None else "PMI+SVD (fallback path)"
    return {
        "measurement_version": "v3_round57_eve_specific_drift_baseline",
        "method": "wrapper_telemetry + agp_trace_correlation",
        "baseline_round": 57,
        "status": "baseline_set_at_round57",
        "fasttext_vector_unchanged": True,
        "fasttext_primary_class": None if primary is None else primary.__class__.__name__,
        "fasttext_loaded": fasttext_loaded,
        "evespecific_learning_substrate": evespecific_substrate,
        "fallback_class": None if fallback is None else fallback.__class__.__name__,
        "fallback_mutable_learning_preserved": fallback is not None and hasattr(fallback, "observe"),
        "drift_indicators": {
            "primary_hit_rate": float(telemetry.get("primary_hit_rate", 0.0) or 0.0),
            "fallback_rate": float(telemetry.get("fallback_rate", 0.0) or 0.0),
            "eve_specific_rate": float(telemetry.get("eve_specific_rate", 0.0) or 0.0),
            "error_rate": float(telemetry.get("error_rate", 0.0) or 0.0),
            "oov_pattern_log": list(telemetry.get("recent_oov_sample", []) or []),
            "oov_log_size": int(telemetry.get("oov_log_size", 0) or 0),
        },
        "target": "round 200+ avg drift > 0.3 (Draft, v3.1 precise threshold pending)",
        "no_auto_threshold_adjustment": True,
        "no_drift_based_runtime_change": True,
        "read_only": True,
    }


def measure_eve_self_learning_drift_accumulation(engine: Any) -> dict[str, Any]:
    """Read-only v3 round58 drift accumulation surface.

    Round58 starts controlled continuous observation. This measurement combines
    the round57 wrapper baseline with tracker/vector-store/self-learning stats
    so drift can be audited without changing thresholds or promoting memory.
    """
    baseline = measure_eve_specific_drift_baseline(engine)
    tracker = getattr(engine, "eve_vocab_tracker", None)
    store = getattr(engine, "eve_specific_vector_store", None)
    learner = getattr(engine, "eve_self_learning", None)
    tracker_stats = tracker.stats() if tracker is not None and hasattr(tracker, "stats") else {}
    store_stats = store.stats() if store is not None and hasattr(store, "stats") else {}
    learner_stats = learner.stats() if learner is not None and hasattr(learner, "stats") else {}
    return {
        "measurement_version": "v3_round58_eve_self_learning_drift_accumulation",
        "baseline_round": 58,
        "status": "continuous_observation_started_explicit_commit_only",
        "round57_baseline": baseline,
        "drift_accumulation": {
            "observed_vocab_count": int(tracker_stats.get("tracked_vocab_count", 0) or 0),
            "observed_eve_specific_count": int(tracker_stats.get("eve_specific_count", 0) or 0),
            "eve_specific_vector_count": int(store_stats.get("stored_count", 0) or 0),
            "observe_calls": int(learner_stats.get("observe_calls", 0) or 0),
            "commit_calls": int(learner_stats.get("commit_calls", 0) or 0),
            "vectors_created": int(learner_stats.get("vectors_created", 0) or 0),
            "vectors_skipped": int(learner_stats.get("vectors_skipped", 0) or 0),
            "audit_calls": int(learner_stats.get("audit_calls", 0) or 0),
            "commit_gate_blocks": int(learner_stats.get("commit_gate_blocks", 0) or 0),
            "commit_audit_record_count": int(learner_stats.get("commit_audit_record_count", 0) or 0),
        },
        "commit_audit_dashboard": learner_stats.get("commit_audit_dashboard") or {},
        "commit_threshold_readiness": learner_stats.get("commit_threshold_readiness") or {},
        "commit_threshold_proposal": learner_stats.get("commit_threshold_proposal") or {},
        "commit_threshold_enforcement": {
            "version": learner_stats.get("commit_threshold_enforcement_version"),
            "active_policy_version": learner_stats.get("active_policy_version"),
            "min_observations_for_commit": int(learner_stats.get("min_observations_for_commit", 0) or 0),
            "min_known_context_words_for_commit": int(learner_stats.get("min_known_context_words_for_commit", 0) or 0),
            "auto_promotion_enabled": bool(learner_stats.get("auto_promotion_enabled", False)),
            "read_only": True,
        },
        "commit_threshold_policy_snapshot": learner_stats.get("commit_threshold_policy_snapshot") or {},
        "observation_evidence_quality": learner_stats.get("observation_evidence_quality") or {},
        "context_diversity_gate_dry_run": learner_stats.get("context_diversity_gate_dry_run") or {},
        "context_diversity_proposal": learner_stats.get("context_diversity_proposal") or {},
        "context_diversity_gate_enforcement": {
            "version": learner_stats.get("context_diversity_gate_enforcement_version"),
            "active_policy_version": learner_stats.get("active_policy_version"),
            "context_diversity_gate_enabled": bool(learner_stats.get("context_diversity_gate_enabled", False)),
            "read_only": True,
        },
        "context_diversity_rollback_drill": learner_stats.get("context_diversity_rollback_drill") or {},
        "context_diversity_blocked_candidate_report": learner_stats.get("context_diversity_blocked_candidate_report") or {},
        "self_learning_v1_freeze_policy_snapshot": learner_stats.get("self_learning_v1_freeze_policy_snapshot") or {},
        "self_learning_v1_freeze_policy_snapshot_version": learner_stats.get("self_learning_v1_freeze_policy_snapshot_version"),
        "commit_gate": {
            "enabled": bool(learner_stats.get("commit_gate_enabled", False)),
            "min_observations_for_commit": int(learner_stats.get("min_observations_for_commit", 0) or 0),
            "min_known_context_words_for_commit": int(learner_stats.get("min_known_context_words_for_commit", 0) or 0),
            "mutation_requires_audit_pass": True,
            "audit_persistence_enabled": bool(learner_stats.get("commit_audit_record_count", 0) is not None),
            "audit_export_version": learner_stats.get("commit_audit_export_version"),
            "dashboard_version": learner_stats.get("commit_audit_dashboard_version"),
            "threshold_dry_run_version": learner_stats.get("commit_threshold_dry_run_version"),
            "threshold_readiness_version": learner_stats.get("commit_threshold_readiness_version"),
            "threshold_proposal_version": learner_stats.get("commit_threshold_proposal_version"),
            "threshold_enforcement_version": learner_stats.get("commit_threshold_enforcement_version"),
            "threshold_policy_snapshot_version": learner_stats.get("commit_threshold_policy_snapshot_version"),
            "observation_evidence_quality_version": learner_stats.get("observation_evidence_quality_version"),
            "context_diversity_gate_dry_run_version": learner_stats.get("context_diversity_gate_dry_run_version"),
            "context_diversity_proposal_version": learner_stats.get("context_diversity_proposal_version"),
            "context_diversity_gate_enforcement_version": learner_stats.get("context_diversity_gate_enforcement_version"),
            "context_diversity_rollback_drill_version": learner_stats.get("context_diversity_rollback_drill_version"),
            "context_diversity_blocked_candidate_report_version": learner_stats.get("context_diversity_blocked_candidate_report_version"),
            "self_learning_v1_freeze_policy_snapshot_version": learner_stats.get("self_learning_v1_freeze_policy_snapshot_version"),
            "context_diversity_gate_enabled": bool(learner_stats.get("context_diversity_gate_enabled", False)),
            "active_policy_version": learner_stats.get("active_policy_version"),
        },
        "policy": {
            "auto_observe_enabled": bool(learner_stats.get("auto_observe_enabled", False)),
            "auto_promotion_enabled": bool(learner_stats.get("auto_promotion_enabled", False)),
            "memory_quarantine_unchanged": True,
            "agp_bypass": False,
            "no_auto_threshold_adjustment": True,
            "no_auto_rollback": True,
            "context_diversity_gate_enforced": bool(learner_stats.get("context_diversity_gate_enabled", False)),
            "context_diversity_proposal_report_only": False,
            "context_diversity_rollback_drill_read_only": True,
            "context_diversity_blocked_candidate_report_read_only": True,
            "no_drift_based_runtime_change": True,
            "explicit_commit_gate": bool(learner_stats.get("commit_gate_enabled", False)),
            "self_learning_v1_frozen": bool(learner_stats.get("self_learning_v1_freeze_policy_snapshot_version")),
        },
        "read_only": True,
    }


def correlate_agp_and_wrapper_telemetry(engine: Any) -> dict[str, Any]:
    """Read-only correlation surface for AGP traces and wrapper telemetry."""
    telemetry = _wrapper_telemetry_snapshot(engine)
    try:
        from adapters.agp_trace_analyzer import AGPTraceAnalyzer
        analyzer = AGPTraceAnalyzer.from_engine(engine)
        agp_pass_rate = analyzer.pass_rate()
        agp_by_layer = analyzer.summarize_by_layer()
    except Exception as exc:
        agp_pass_rate = 0.0
        agp_by_layer = {}
        agp_error = exc.__class__.__name__
    else:
        agp_error = None
    return {
        "correlation_version": "v3_round42_agp_wrapper_correlation",
        "method": "read_only_snapshot",
        "baseline_round": 42,
        "agp_pass_rate_pre_swap": None,
        "agp_pass_rate_post_swap": agp_pass_rate,
        "delta": None,
        "agp_by_layer": agp_by_layer,
        "wrapper_telemetry": telemetry,
        "interpretation": "data_dict_not_recommendation",
        "agp_error": agp_error,
        "no_auto_threshold_adjustment": True,
        "read_only": True,
    }


SELF_LEARNING_ARCHITECTURE_VERSION = "v3_round53_self_learning_design"
SELF_LEARNING_START_ROUND = 53
EVE_SPECIFIC_FALLBACK_PRIORITY = (
    "fasttext medium 30k (immutable external seed)",
    "EveSpecificVectorStore (EVE-specific deterministic learning)",
    "PMI+SVD legacy fallback",
)


def design_self_learning_architecture() -> dict[str, Any]:
    """Read-only B-continuous self-learning architecture plan.

    Round53 intentionally adds design/scaffold only. It does not observe words,
    add vectors, integrate the wrapper, or change runtime routing.
    """
    phase_plan = [
        {"round": 53, "task": "design + scaffold", "status": "current"},
        {"round": 54, "task": "EveVocabTracker observe implementation", "status": "planned"},
        {"round": 55, "task": "EveSpecificVectorStore deterministic vector update", "status": "planned"},
        {"round": 56, "task": "Wrapper fallback priority integration", "status": "planned"},
        {"round": 57, "task": "smoke rerun + drift baseline measurement", "status": "planned"},
        {"round": "58+", "task": "continuous observation + drift accumulation", "status": "planned"},
    ]
    return {
        "architecture_version": SELF_LEARNING_ARCHITECTURE_VERSION,
        "goal": "EVE-specific lexical deterministic self-learning",
        "strategy": "B continuous after A first medium 30k",
        "appendix_d_alignment": "high",
        "components": {
            "EveVocabTracker": {
                "purpose": "observe EVE-specific lexical use and deterministic counts",
                "round_scaffold": 53,
                "round_implementation": 54,
                "runtime_active_now": False,
            },
            "EveSpecificVectorStore": {
                "purpose": "store deterministic vectors for EVE-specific vocabulary",
                "round_scaffold": 53,
                "round_implementation": 55,
                "dimension": 300,
                "runtime_active_now": False,
            },
            "Wrapper integration": {
                "purpose": "insert EVE-specific store between fastText primary and PMI+SVD fallback",
                "round_integration": 56,
                "runtime_active_now": False,
            },
        },
        "wrapper_integration_design": {
            "primary": "fasttext medium 30k (immutable external seed)",
            "fallback_priority_1": "EveSpecificVectorStore (EVE learned, deterministic, mutable)",
            "fallback_priority_2": "PMI+SVD (legacy safety net)",
            "current_round53_runtime": "fasttext medium 30k primary + PMI+SVD fallback only",
            "current_round55_runtime": "fasttext medium 30k primary + PMI+SVD fallback only; EveSpecificVectorStore implemented but not routed",
        },
        "deterministic_update_rule": {
            "observation": "extract words from fixture/conversation text in deterministic order",
            "filter": "fastText-missing words only; EVE-specific candidates such as EVE and 민석",
            "threshold": "N observations before vector creation; precise value deferred to round55",
            "vector_generation": "deterministic mean of known fastText 300d context vectors; implemented in round55",
            "no_random": True,
            "no_auto_promotion_in_round53": True,
        },
        "drift_measurement_refinement": {
            "metric": "EveSpecificVectorStore vocab count + diversity + wrapper fallback distribution",
            "target": "round 200+ avg drift > 0.3 (Draft; v3.1 precise threshold pending)",
            "tracking": "record EveSpecificVectorStore and EveVocabTracker stats by round once implemented",
        },
        "phase_plan": phase_plan,
        "risks": [
            "deterministic update rule now explicit: mean known fastText context vectors",
            "vector dimension aligned: 300d EVE-specific/fastText; PMI+SVD remains legacy fallback",
            "initial EVE-specific vocabulary may be tiny",
            "automatic promotion would violate quarantine-style governance if added too early",
        ],
        "not_doing_round53": [
            "actual observation",
                        "wrapper integration",
            "smoke rerun",
            "drift auto measurement activation",
        ],
        "read_only": True,
    }


def redefine_drift_measurement_for_self_learning() -> dict[str, Any]:
    """Read-only drift metric refinement for the self-learning era."""
    return {
        "definition_version": SELF_LEARNING_ARCHITECTURE_VERSION,
        "previous_baseline": "round42 (wrapper telemetry baseline before self-learning scaffold)",
        "new_baseline_round": 53,
        "implementation_baseline_round": 57,
        "metric_components": [
            "EveSpecificVectorStore.stats.stored_count",
            "EveVocabTracker.stats.observed_count",
            "fallback_path_distribution: fastText vs EveSpecificVectorStore vs PMI+SVD",
            "EVE-specific OOV resolution: EVE, 민석",
        ],
        "target": "round 200+ avg > 0.3 (Appendix D draft target; v3.1 precise threshold pending)",
        "interim_targets": {
            "round60": "EveSpecific count >= 5",
            "round100": "EveSpecific count >= 30",
            "round200": "avg drift > 0.3",
        },
        "appendix_d_alignment": "high",
        "no_auto_threshold_adjustment": True,
        "no_runtime_change": True,
        "read_only": True,
    }


def measure_eve_specific_round72_baseline(engine: Any) -> dict[str, Any]:
    """Read-only v3 round72 smoke/drift baseline snapshot.

    Round72 re-measures the EveSpecific lookup surface after the Round71
    consolidation gate. It preserves Round57's drift baseline, adds an explicit
    route-count distribution, and exposes the Round72 lookup-route policy so the
    PMI+SVD local precheck is not confused with an uncontrolled duplicate
    fallback call. This function does not mutate vectors, thresholds, memory,
    quarantine, fastText seeds, or AGP state.
    """
    telemetry = _wrapper_telemetry_snapshot(engine)
    wrapper = getattr(engine, "self_embedding", None)
    policy_snapshot = {}
    if wrapper is not None and hasattr(wrapper, "lookup_route_policy_snapshot"):
        try:
            policy_snapshot = dict(wrapper.lookup_route_policy_snapshot())
        except Exception:
            policy_snapshot = {}
    round57 = measure_eve_specific_drift_baseline(engine)
    primary_hits = int(telemetry.get("primary_hits", 0) or 0)
    eve_hits = int(telemetry.get("eve_specific_hits", 0) or 0)
    fallback_uses = int(telemetry.get("fallback_uses", 0) or 0)
    errors = int(telemetry.get("errors", 0) or 0)
    total_calls = int(telemetry.get("total_calls", 0) or 0)
    route_total = primary_hits + eve_hits + fallback_uses
    return {
        "measurement_version": "v3_round72_eve_specific_smoke_drift_baseline",
        "baseline_round": 72,
        "status": "round71_consolidated_baseline_remeasured",
        "round57_baseline": round57,
        "lookup_route_policy": policy_snapshot,
        "route_distribution": {
            "total_calls": total_calls,
            "routed_hits_total": int(route_total),
            "fasttext_primary_hits": primary_hits,
            "eve_specific_hits": eve_hits,
            "pmi_svd_fallback_uses": fallback_uses,
            "errors": errors,
            "fasttext_primary_rate": (primary_hits / total_calls) if total_calls else 0.0,
            "eve_specific_rate": (eve_hits / total_calls) if total_calls else 0.0,
            "pmi_svd_fallback_rate": (fallback_uses / total_calls) if total_calls else 0.0,
            "error_rate": (errors / total_calls) if total_calls else 0.0,
        },
        "self_learning_policy": {
            "auto_observe_enabled": bool(getattr(getattr(engine, "eve_self_learning", None), "auto_observe_enabled", False)),
            "auto_promotion_enabled": bool(getattr(getattr(engine, "eve_self_learning", None), "auto_promotion_enabled", False)),
            "commit_gate_enabled": bool(getattr(getattr(engine, "eve_self_learning", None), "commit_gate_enabled", False)),
            "min_observations_for_commit": int(getattr(getattr(engine, "eve_self_learning", None), "min_observations_for_commit", 0) or 0),
            "context_diversity_gate_enabled": bool(getattr(getattr(engine, "eve_self_learning", None), "context_diversity_gate_enabled", False)),
        },
        "policy": {
            "routing_changed_in_round72": False,
            "no_auto_promotion": True,
            "no_auto_threshold_adjustment": True,
            "no_auto_rollback": True,
            "no_drift_based_runtime_change": True,
            "memory_quarantine_unchanged": True,
            "agp_bypass": False,
        },
        "read_only": True,
    }
