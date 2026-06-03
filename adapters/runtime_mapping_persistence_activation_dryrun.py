"""Round107 runtime mapping persistence activation dry-run harness.

Round107 defines the checkpoint, rollback, audit-log, and state-debug surfaces
required before a later patch may enable persistent runtime mapping.  This module
is intentionally dry-run only: it never flips runtime mapping defaults, never
persists mapping state, and never mutates AGP/category/vector/memory state.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROUND107_ACTIVATION_DRYRUN_VERSION = "v3_round107_runtime_mapping_persistence_activation_dryrun"


FUTURE_ACTIVATION_TOUCH_FILES = [
    {
        "path": "adapters/lex_concept_mapping_adapter.py",
        "future_touch_reason": "load/apply an operator-approved persistent runtime mapping table behind explicit activation gates",
        "round107_touched_for_activation": False,
    },
    {
        "path": "adapters/state_debug_adapter.py",
        "future_touch_reason": "export activation/checkpoint/rollback/audit status after real activation is introduced",
        "round107_touched_for_activation": False,
    },
    {
        "path": "main.py",
        "future_touch_reason": "wire real activation only if a later explicit runtime startup path is approved",
        "round107_touched_for_activation": False,
    },
    {
        "path": "eve_v3_autonomous_handoff/validation/runtime_mapping_persistence_checkpoint.json",
        "future_touch_reason": "operator checkpoint artifact for a later activation patch",
        "round107_touched_for_activation": False,
    },
    {
        "path": "eve_v3_autonomous_handoff/validation/runtime_mapping_persistence_rollback.json",
        "future_touch_reason": "operator rollback artifact for a later activation patch",
        "round107_touched_for_activation": False,
    },
    {
        "path": "eve_v3_autonomous_handoff/validation/runtime_mapping_persistence_audit_log.jsonl",
        "future_touch_reason": "append-only activation audit log for a later activation patch",
        "round107_touched_for_activation": False,
    },
]

FUTURE_ACTIVATION_TOUCH_STATE = [
    "LexConceptMappingAdapter.runtime_mapping_enabled",
    "LexConceptMappingAdapter.enforcement_enabled",
    "LexConceptMappingAdapter persistent runtime mapping table",
    "StateDebugAdapter lex_concept_mapping.runtime_mapping_persistence_activation",
    "operator checkpoint JSON artifact",
    "operator rollback JSON artifact",
    "operator audit JSONL artifact",
]

FORBIDDEN_MUTATION_STATE = [
    "AGP thresholds or AGP bypass state",
    "SA activations",
    "concept categories",
    "concept commit audit records",
    "concept memory",
    "frame/hypergraph edges",
    "EveSpecificVectorStore vectors",
    "fastText seed/subset vectors.npy",
    "semantic memory or quarantine files",
]


def checkpoint_format() -> dict[str, Any]:
    """Return the future activation checkpoint format without writing it."""
    return {
        "schema_version": f"{ROUND107_ACTIVATION_DRYRUN_VERSION}.checkpoint.v1",
        "artifact_kind": "runtime_mapping_persistence_checkpoint",
        "required_fields": [
            "checkpoint_version",
            "created_by_round",
            "source_decision_version",
            "source_agp_proof_version",
            "mapped_tokens",
            "runtime_mapping_enabled_before",
            "enforcement_enabled_before",
            "concept_category_count_before",
            "concept_commit_audit_count_before",
            "eve_specific_vector_count_before",
            "wrapper_telemetry_before",
            "sa_active_categories_before",
            "activation_touch_files",
            "activation_touch_state",
            "rollback_artifact_path",
            "audit_log_path",
        ],
        "immutable_requirements": {
            "must_be_created_before_activation": True,
            "must_capture_disabled_defaults_before_activation": True,
            "must_not_embed_vectors_npy": True,
            "must_not_embed_semantic_memory": True,
            "must_be_json_only": True,
        },
        "read_only": True,
    }


def rollback_format() -> dict[str, Any]:
    """Return the future rollback artifact format without writing it."""
    return {
        "schema_version": f"{ROUND107_ACTIVATION_DRYRUN_VERSION}.rollback.v1",
        "artifact_kind": "runtime_mapping_persistence_rollback",
        "required_fields": [
            "rollback_version",
            "source_checkpoint_version",
            "restore_runtime_mapping_enabled",
            "restore_enforcement_enabled",
            "remove_or_disable_persistent_mapping_table",
            "verify_agp_anchor_boundary",
            "verify_vector_store_unchanged",
            "verify_concept_memory_unchanged",
            "post_rollback_tests",
        ],
        "rollback_actions": [
            "set LexConceptMappingAdapter.runtime_mapping_enabled=False",
            "set LexConceptMappingAdapter.enforcement_enabled=False",
            "remove later persistent runtime mapping table from startup/load path",
            "retain Round104~107 review artifacts as audit-only records",
            "rerun focused and adjacent validation before reporting rollback complete",
        ],
        "immutable_requirements": {
            "rollback_must_not_delete_concept_categories": True,
            "rollback_must_not_mutate_vectors": True,
            "rollback_must_not_bypass_agp": True,
            "rollback_must_restore_disabled_defaults": True,
        },
        "read_only": True,
    }


def audit_log_schema() -> dict[str, Any]:
    """Return the future append-only audit log schema without writing it."""
    return {
        "schema_version": f"{ROUND107_ACTIVATION_DRYRUN_VERSION}.audit_log.v1",
        "artifact_kind": "runtime_mapping_persistence_audit_log_schema",
        "format": "jsonl_append_only_for_future_activation_patch",
        "required_event_fields": [
            "event_version",
            "event_type",
            "event_index",
            "source_checkpoint_version",
            "runtime_mapping_enabled_before",
            "runtime_mapping_enabled_after",
            "enforcement_enabled_before",
            "enforcement_enabled_after",
            "mapped_tokens",
            "mutation_boundary_checks",
            "operator_reference",
        ],
        "allowed_event_types": [
            "activation_precheck_started",
            "checkpoint_written",
            "activation_applied",
            "activation_validation_passed",
            "rollback_started",
            "rollback_applied",
            "rollback_validation_passed",
        ],
        "forbidden_event_payloads": [
            "raw vectors.npy data",
            "semantic memory dumps",
            "quarantine contents",
            "generated AGP bypass anchors",
        ],
        "read_only": True,
    }


def future_activation_touch_plan() -> dict[str, Any]:
    """Enumerate exact files/state a later activation patch would touch."""
    return {
        "plan_version": f"{ROUND107_ACTIVATION_DRYRUN_VERSION}.touch_plan.v1",
        "future_activation_patch_required": True,
        "files": deepcopy(FUTURE_ACTIVATION_TOUCH_FILES),
        "state": list(FUTURE_ACTIVATION_TOUCH_STATE),
        "state_forbidden_to_touch": list(FORBIDDEN_MUTATION_STATE),
        "round107_actual_activation_touch": False,
        "read_only": True,
    }


def state_debug_export_surface() -> dict[str, Any]:
    """Return the state-debug export shape for this dry-run surface."""
    return {
        "surface_version": f"{ROUND107_ACTIVATION_DRYRUN_VERSION}.state_debug.v1",
        "state_key": "lex_concept_mapping.runtime_mapping_persistence_activation",
        "fields": [
            "dry_run_version",
            "round",
            "activation_dry_run_available",
            "checkpoint_schema_version",
            "rollback_schema_version",
            "audit_log_schema_version",
            "runtime_mapping_enabled_default",
            "enforcement_enabled_default",
            "future_touch_files",
            "future_touch_state",
            "mutation_proof",
        ],
        "does_not_call_activation": True,
        "does_not_write_artifacts": True,
        "read_only": True,
    }


def _safe_call(obj: Any, method_name: str, default: Any) -> Any:
    if obj is None or not hasattr(obj, method_name):
        return deepcopy(default)
    try:
        return deepcopy(getattr(obj, method_name)())
    except Exception as exc:  # pragma: no cover - defensive debug surface
        return {"error": exc.__class__.__name__, "read_only": True}


def _vector_store_stats(engine: Any | None) -> dict[str, Any]:
    store = getattr(engine, "eve_specific_vector_store", None) if engine is not None else None
    if store is not None and hasattr(store, "stats"):
        return _safe_call(store, "stats", {})
    if store is not None and hasattr(store, "count"):
        try:
            return {"count": int(store.count()), "read_only": True}
        except Exception as exc:  # pragma: no cover
            return {"error": exc.__class__.__name__, "read_only": True}
    return {"count": 0, "store_missing": store is None, "read_only": True}


def _wrapper_telemetry(engine: Any | None) -> dict[str, Any]:
    wrapper = getattr(engine, "self_embedding", None) if engine is not None else None
    return _safe_call(wrapper, "telemetry", {})


def _active_categories(engine: Any | None) -> Any:
    activation = getattr(engine, "activation_adapter", None) if engine is not None else None
    sa = getattr(activation, "sa", None)
    activations = getattr(sa, "activations", None)
    if isinstance(activations, dict):
        return sorted(str(k) for k, v in activations.items() if float(v or 0.0) > 0.0)
    return []


def _snapshot(engine: Any | None) -> dict[str, Any]:
    lcm = getattr(engine, "lex_concept_mapping", None) if engine is not None else None
    concept_memory = getattr(engine, "concept_memory", None) if engine is not None else None
    agp = getattr(engine, "agp", None) if engine is not None else None
    return {
        "runtime_mapping_enabled": bool(getattr(lcm, "runtime_mapping_enabled", False)),
        "enforcement_enabled": bool(getattr(lcm, "enforcement_enabled", False)),
        "concept_categories": _safe_call(lcm, "concept_categories_snapshot", {}),
        "concept_commit_records": _safe_call(lcm, "concept_commit_records", []),
        "eve_specific_vector_store": _vector_store_stats(engine),
        "wrapper_telemetry": _wrapper_telemetry(engine),
        "sa_active_categories": _active_categories(engine),
        "concept_memory_stats": _safe_call(concept_memory, "stats", {}),
        "agp_stats": _safe_call(agp, "stats", {}),
    }


def _derive_count(snapshot: dict[str, Any], key: str) -> int:
    value = snapshot.get(key)
    if isinstance(value, dict):
        if "count" in value:
            return int(value.get("count") or 0)
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 0


def build_runtime_mapping_persistence_activation_dryrun(
    *,
    engine: Any | None = None,
    source_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Round107 activation dry-run report without mutating state."""
    decision = deepcopy(source_decision or {})
    before = _snapshot(engine)

    checkpoint = checkpoint_format()
    rollback = rollback_format()
    audit_schema = audit_log_schema()
    touch_plan = future_activation_touch_plan()
    debug_surface = state_debug_export_surface()

    # Deliberately do not set runtime_mapping_enabled or enforcement_enabled,
    # do not write any checkpoint/rollback/audit artifacts, and do not call AGP
    # verification or embedding lookup.  The second snapshot proves this call is
    # data-only for the observable state surfaces.
    after = _snapshot(engine)

    mutation_checks = {
        "runtime_mapping_enabled_default_unchanged": before["runtime_mapping_enabled"] == after["runtime_mapping_enabled"] == False,
        "enforcement_enabled_default_unchanged": before["enforcement_enabled"] == after["enforcement_enabled"] == False,
        "concept_categories_unchanged": before["concept_categories"] == after["concept_categories"],
        "concept_commit_records_unchanged": before["concept_commit_records"] == after["concept_commit_records"],
        "eve_specific_vector_store_unchanged": before["eve_specific_vector_store"] == after["eve_specific_vector_store"],
        "wrapper_telemetry_unchanged": before["wrapper_telemetry"] == after["wrapper_telemetry"],
        "sa_active_categories_unchanged": before["sa_active_categories"] == after["sa_active_categories"],
        "concept_memory_stats_unchanged": before["concept_memory_stats"] == after["concept_memory_stats"],
        "agp_stats_unchanged": before["agp_stats"] == after["agp_stats"],
        "agp_verify_called": False,
        "embedding_lookup_called": False,
        "vectors_npy_committed": False,
        "persistence_enabled_now": False,
    }
    unchanged_checks = [
        "runtime_mapping_enabled_default_unchanged",
        "enforcement_enabled_default_unchanged",
        "concept_categories_unchanged",
        "concept_commit_records_unchanged",
        "eve_specific_vector_store_unchanged",
        "wrapper_telemetry_unchanged",
        "sa_active_categories_unchanged",
        "concept_memory_stats_unchanged",
        "agp_stats_unchanged",
    ]
    forbidden_action_checks = [
        "agp_verify_called",
        "embedding_lookup_called",
        "vectors_npy_committed",
        "persistence_enabled_now",
    ]
    no_mutation = all(bool(mutation_checks[key]) for key in unchanged_checks) and not any(
        bool(mutation_checks[key]) for key in forbidden_action_checks
    )

    return {
        "dry_run_version": ROUND107_ACTIVATION_DRYRUN_VERSION,
        "round": 107,
        "axis": "runtime_mapping_persistence_activation_dry_run",
        "source_decision_version": decision.get("decision_version"),
        "source_decision_status": decision.get("decision_status"),
        "source_decision_ready": bool(decision.get("decision_ready", False)),
        "mapped_tokens": sorted(str(t) for t in decision.get("mapped_tokens", []) or []),
        "activation_status": "dry_run_only_not_enabled",
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "runtime_mapping_enabled_now": after["runtime_mapping_enabled"],
        "enforcement_enabled_now": after["enforcement_enabled"],
        "checkpoint_format": checkpoint,
        "rollback_format": rollback,
        "audit_log_schema": audit_schema,
        "state_debug_export_surface": debug_surface,
        "future_activation_touch_plan": touch_plan,
        "dry_run_execution": {
            "before": before,
            "after": after,
            "mutation_checks": mutation_checks,
            "no_runtime_mapping_default_change": mutation_checks["runtime_mapping_enabled_default_unchanged"],
            "no_enforcement_default_change": mutation_checks["enforcement_enabled_default_unchanged"],
            "no_agp_vector_category_memory_mutation": no_mutation,
        },
        "checkpoint_preview": {
            "checkpoint_version": checkpoint["schema_version"],
            "created_by_round": 107,
            "source_decision_version": decision.get("decision_version"),
            "mapped_tokens": sorted(str(t) for t in decision.get("mapped_tokens", []) or []),
            "runtime_mapping_enabled_before": before["runtime_mapping_enabled"],
            "enforcement_enabled_before": before["enforcement_enabled"],
            "concept_category_count_before": _derive_count(before, "concept_categories"),
            "concept_commit_audit_count_before": _derive_count(before, "concept_commit_records"),
            "eve_specific_vector_count_before": int((before.get("eve_specific_vector_store") or {}).get("count", 0) or 0),
            "activation_touch_files": [row["path"] for row in FUTURE_ACTIVATION_TOUCH_FILES],
            "activation_touch_state": list(FUTURE_ACTIVATION_TOUCH_STATE),
            "json_written_now": False,
        },
        "rollback_preview": {
            "rollback_version": rollback["schema_version"],
            "restore_runtime_mapping_enabled": False,
            "restore_enforcement_enabled": False,
            "json_written_now": False,
        },
        "policy": {
            "dry_run_only": True,
            "actual_persistence_enablement_forbidden_in_round107": True,
            "runtime_mapping_enabled_default_change_forbidden": True,
            "enforcement_enabled_default_change_forbidden": True,
            "agp_bypass_forbidden": True,
            "vectors_npy_commit_forbidden": True,
            "test_weakening_forbidden": True,
        },
        "read_only": True,
    }


def write_round107_runtime_mapping_persistence_activation_dryrun(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    """Write the dry-run report JSON only; this is not activation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("dry_run_version", ROUND107_ACTIVATION_DRYRUN_VERSION),
        "path": str(out),
        "json_written": True,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "runtime_mapping_persisted": False,
        "read_only": True,
    }


__all__ = [
    "ROUND107_ACTIVATION_DRYRUN_VERSION",
    "audit_log_schema",
    "build_runtime_mapping_persistence_activation_dryrun",
    "checkpoint_format",
    "future_activation_touch_plan",
    "rollback_format",
    "state_debug_export_surface",
    "write_round107_runtime_mapping_persistence_activation_dryrun",
]
