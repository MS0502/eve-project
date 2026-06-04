"""Rounds162-164 operator artifact restore contract and preflight gates.

These helpers are deterministic, read-only control surfaces for the registered
fastText subset ``vectors.npy`` artifacts. They never create, download, copy, or
commit vector files. Load-dependent repair remains hard-blocked until the
existing seed/vector artifact readiness gate is green.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from adapters.external_seed_manifest import (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    load_manifest_file,
)
from adapters.seed_vector_artifact_readiness import build_seed_vector_artifact_readiness_gate

ROUND162_RESTORE_CONTRACT_VERSION = "v3_round162_operator_artifact_restore_contract"
ROUND163_POST_RESTORE_VALIDATION_SCHEMA_VERSION = "v3_round163_post_restore_validation_schema"
ROUND164_LOAD_DEPENDENT_REPAIR_PREFLIGHT_VERSION = "v3_round164_load_dependent_repair_preflight"

REGISTERED_VECTOR_SUBSETS: tuple[str, ...] = (
    FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,
    FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
    FASTTEXT_KOREAN_SUBSET_MINI_1K_NAME,
)

NO_COMMIT_SAFETY_LIST: tuple[str, ...] = (
    "seeds/subsets/**/vectors.npy",
    "seeds/subsets/**",
    "*.zip",
    "*.part",
    "*_operator_artifacts/**",
    "_operator_artifacts/**",
)

REQUIRED_SUBSET_MANIFEST_FIELDS: tuple[str, ...] = (
    "name",
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
    "status",
    "imported_at_round",
    "imported_at_patch",
)


def _subset_entries_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("seeds", []) if isinstance(manifest, dict) else []:
        if isinstance(entry, dict) and entry.get("entry_type") == "subset" and isinstance(entry.get("name"), str):
            entries[str(entry["name"])] = deepcopy(entry)
    return entries


def _contract_row(entry: dict[str, Any] | None, subset_name: str) -> dict[str, Any]:
    file_location = str(entry.get("file_location")) if entry else str(Path("seeds") / "subsets" / subset_name)
    vocab_file = str(entry.get("vocab_file", "vocab.txt")) if entry else "vocab.txt"
    vectors_file = str(entry.get("vectors_file", "vectors.npy")) if entry else "vectors.npy"
    subset_manifest_file = str(entry.get("subset_manifest_file", "subset_manifest.json")) if entry else "subset_manifest.json"
    subset_dir = Path(file_location)
    expected_fields = {field: (entry.get(field) if entry else None) for field in REQUIRED_SUBSET_MANIFEST_FIELDS}
    missing_fields = [field for field in REQUIRED_SUBSET_MANIFEST_FIELDS if entry is None or field not in entry]
    return {
        "subset_name": subset_name,
        "registered": entry is not None,
        "expected_paths": {
            "subset_dir": file_location,
            "vocab": str(subset_dir / vocab_file),
            "vectors": str(subset_dir / vectors_file),
            "subset_manifest": str(subset_dir / subset_manifest_file),
        },
        "expected_manifest_fields": expected_fields,
        "missing_manifest_fields": missing_fields,
        "checksum_verification": {
            "algorithm": "SHA256",
            "vocab_expected": entry.get("vocab_checksum") if entry else None,
            "vectors_expected": entry.get("vectors_checksum") if entry else None,
            "subset_manifest_expected": entry.get("subset_manifest_checksum") if entry else None,
            "must_match_manifest_exactly": True,
        },
        "shape_dtype_verification": {
            "vectors_expected_shape": [entry.get("vocab_size"), entry.get("vector_dim")] if entry else None,
            "vectors_expected_dtype": "float32",
            "numpy_mmap_read_only_ok": True,
        },
        "operator_restore_steps": [
            f"Obtain the real operator-extracted {subset_name}/vectors.npy outside the PR diff.",
            f"Place the artifact at {subset_dir / vectors_file} without editing seeds/MANIFEST.yaml.",
            f"Verify SHA256 equals {entry.get('vectors_checksum') if entry else 'the registered manifest checksum'}.",
            f"Verify numpy shape is {(entry.get('vocab_size'), entry.get('vector_dim')) if entry else '(registered vocab_size, registered vector_dim)'} and dtype is float32.",
            "Rerun the readiness gate and load-dependent preflight before any repair that imports vectors.",
        ],
    }


def build_round162_operator_artifact_restore_contract(
    *,
    manifest_data: dict[str, Any] | None = None,
    subset_names: Iterable[str] = REGISTERED_VECTOR_SUBSETS,
) -> dict[str, Any]:
    """Return the exact read-only restore contract for registered vector files."""
    manifest = deepcopy(manifest_data) if manifest_data is not None else load_manifest_file()
    entries = _subset_entries_by_name(manifest)
    rows = [_contract_row(entries.get(subset_name), subset_name) for subset_name in subset_names]
    return {
        "version": ROUND162_RESTORE_CONTRACT_VERSION,
        "round": 162,
        "contract_layer": "operator_registered_vectors_restore_contract",
        "registered_vector_subsets": list(subset_names),
        "restore_contract_rows": rows,
        "no_commit_safety_list": list(NO_COMMIT_SAFETY_LIST),
        "required_verification": [
            "expected_path_present",
            "manifest_fields_present",
            "sha256_checksum_match",
            "vectors_shape_match",
            "vectors_dtype_float32",
            "readiness_gate_green_before_load_dependent_repair",
        ],
        "forbidden_actions": [
            "download_artifacts_inside_pr",
            "create_dummy_vectors",
            "fabricate_checksums",
            "commit_vectors_npy_or_seed_subsets",
            "enable_production_persistence",
            "enable_runtime_mapping_default",
            "enable_enforcement",
            "bypass_agp",
        ],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
        "artifacts_included": False,
        "read_only": True,
    }


def build_round163_post_restore_validation_schema(
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic validation checklist/schema without artifacts."""
    restore_contract = deepcopy(contract) if contract is not None else build_round162_operator_artifact_restore_contract()
    checklist = [
        {"id": "path_presence", "required": True, "description": "vocab.txt, vectors.npy, and subset_manifest.json exist at registered paths"},
        {"id": "manifest_field_match", "required": True, "description": "registered manifest fields match the restore contract"},
        {"id": "sha256_match", "required": True, "description": "all artifact SHA256 values match seeds/MANIFEST.yaml"},
        {"id": "shape_dtype_match", "required": True, "description": "vectors.npy loads read-only with registered shape and float32 dtype"},
        {"id": "readiness_gate_green", "required": True, "description": "seed/vector readiness gate reports ready before load-dependent work"},
        {"id": "no_commit_boundary_clean", "required": True, "description": "git diff/staging excludes vectors.npy, seeds/subsets, zip/part files, and operator artifacts"},
        {"id": "policy_flags_unchanged", "required": True, "description": "production persistence, runtime mapping default, enforcement, and AGP bypass remain disabled"},
    ]
    return {
        "version": ROUND163_POST_RESTORE_VALIDATION_SCHEMA_VERSION,
        "round": 163,
        "schema_layer": "deterministic_post_restore_validation_checklist",
        "contract_version": restore_contract.get("version"),
        "json_schema": {
            "type": "object",
            "required": ["version", "round", "subset_results", "overall_status", "policy_flags"],
            "properties": {
                "version": {"const": ROUND163_POST_RESTORE_VALIDATION_SCHEMA_VERSION},
                "round": {"const": 163},
                "subset_results": {"type": "array", "items": {"type": "object"}},
                "overall_status": {"enum": ["post_restore_validation_ready", "blocked_operator_artifact_required", "blocked_post_restore_validation_failed"]},
                "policy_flags": {"type": "object"},
            },
        },
        "checklist": checklist,
        "expected_validation_commands": [
            "python -m compileall -q adapters tests main.py",
            "python -m pytest --collect-only -q",
            "python -m pytest -q tests/test_v3_round162_164_restore_contract_preflight.py tests/test_v3_round159_seed_vector_artifact_gate.py",
        ],
        "contract_rows": restore_contract.get("restore_contract_rows", []),
        "artifacts_included": False,
        "vectors_written": False,
        "read_only": True,
    }


def build_round164_load_dependent_repair_preflight(
    *,
    readiness_gate: dict[str, Any] | None = None,
    manifest_data: dict[str, Any] | None = None,
    subset_names: Iterable[str] = (FASTTEXT_KOREAN_SUBSET_MEDIUM_30K_NAME,),
) -> dict[str, Any]:
    """Hard-block load-dependent repair unless the readiness gate is green."""
    gate = deepcopy(readiness_gate) if readiness_gate is not None else build_seed_vector_artifact_readiness_gate(
        manifest_data=manifest_data,
        subset_names=subset_names,
    )
    ready = bool(gate.get("ready") and gate.get("load_should_be_attempted") and gate.get("status") == "ready_for_explicit_operator_artifact_load")
    blockers = [] if ready else sorted(set(str(item) for item in gate.get("readiness_blockers", []) or ["readiness_gate_not_green"]))
    return {
        "version": ROUND164_LOAD_DEPENDENT_REPAIR_PREFLIGHT_VERSION,
        "round": 164,
        "preflight_layer": "load_dependent_repair_hard_block",
        "status": "load_dependent_repair_preflight_green" if ready else "hard_block_load_dependent_repair_until_artifacts_ready",
        "ready": ready,
        "hard_block": not ready,
        "readiness_gate_status": gate.get("status"),
        "readiness_gate_ready": bool(gate.get("ready")),
        "readiness_gate_load_should_be_attempted": bool(gate.get("load_should_be_attempted")),
        "blockers": blockers,
        "allowed_when_green": [
            "explicit FasttextEmbeddingAdapter.load() with registered artifact path",
            "focused fastText wrapper/load repair tests",
            "broader pytest validation after focused load-dependent behavior is fixed",
        ] if ready else [],
        "blocked_when_red": [
            "load-dependent repair",
            "wrapper/vector repair that assumes vectors.npy is present",
            "runtime mapping persistence enablement",
            "enforcement enablement",
            "test skips or dummy vector substitution",
        ],
        "readiness_gate": gate,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
        "agp_bypass_used": False,
        "vectors_written": False,
        "artifacts_included": False,
        "read_only": True,
    }


__all__ = [
    "NO_COMMIT_SAFETY_LIST",
    "REGISTERED_VECTOR_SUBSETS",
    "REQUIRED_SUBSET_MANIFEST_FIELDS",
    "ROUND162_RESTORE_CONTRACT_VERSION",
    "ROUND163_POST_RESTORE_VALIDATION_SCHEMA_VERSION",
    "ROUND164_LOAD_DEPENDENT_REPAIR_PREFLIGHT_VERSION",
    "build_round162_operator_artifact_restore_contract",
    "build_round163_post_restore_validation_schema",
    "build_round164_load_dependent_repair_preflight",
]
