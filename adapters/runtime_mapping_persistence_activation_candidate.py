"""Round108 guarded runtime mapping persistence activation candidate.

Round108 prepares the first guarded activation candidate for runtime mapping
persistence, but it still does not enable persistence by default.  Candidate
mutation is allowed only when an explicit operator approval guard passes, after a
JSON checkpoint and before-state debug export are written.  The candidate is then
rolled back and verified so repository/runtime defaults remain disabled.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from adapters.runtime_mapping_persistence_activation_dryrun import (
    ROUND107_ACTIVATION_DRYRUN_VERSION,
    build_runtime_mapping_persistence_activation_dryrun,
)
from adapters.runtime_mapping_persistence_decision import ROUND106_PERSISTENCE_DECISION_VERSION
from adapters.state_debug_adapter import StateDebugAdapter

ROUND108_ACTIVATION_CANDIDATE_VERSION = "v3_round108_runtime_mapping_persistence_activation_candidate"
ROUND108_OPERATOR_APPROVAL_TOKEN = "ROUND108_OPERATOR_APPROVED_RUNTIME_MAPPING_PERSISTENCE_CANDIDATE"

CHECKPOINT_FILENAME = "runtime_mapping_persistence_activation_candidate_checkpoint.json"
ROLLBACK_FILENAME = "runtime_mapping_persistence_activation_candidate_rollback.json"
AUDIT_LOG_FILENAME = "runtime_mapping_persistence_activation_candidate_audit_log.jsonl"
STATE_DEBUG_BEFORE_FILENAME = "runtime_mapping_persistence_activation_candidate_state_debug_before.json"
STATE_DEBUG_AFTER_FILENAME = "runtime_mapping_persistence_activation_candidate_state_debug_after.json"
STATE_DEBUG_ROLLBACK_FILENAME = "runtime_mapping_persistence_activation_candidate_state_debug_after_rollback.json"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append_audit(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_call(obj: Any, method_name: str, default: Any) -> Any:
    if obj is None or not hasattr(obj, method_name):
        return deepcopy(default)
    try:
        return _json_copy(getattr(obj, method_name)())
    except Exception as exc:  # pragma: no cover - defensive debug surface
        return {"error": exc.__class__.__name__, "read_only": True}


def _active_categories(engine: Any | None) -> list[str]:
    activation = getattr(engine, "activation_adapter", None) if engine is not None else None
    sa = getattr(activation, "sa", None)
    activations = getattr(sa, "activations", None)
    if isinstance(activations, dict):
        return sorted(str(k) for k, v in activations.items() if float(v or 0.0) > 0.0)
    return []


def capture_runtime_mapping_activation_state(engine: Any | None) -> dict[str, Any]:
    """Capture deterministic mutation-boundary state for checkpoint/rollback."""
    lcm = getattr(engine, "lex_concept_mapping", None) if engine is not None else None
    store = getattr(engine, "eve_specific_vector_store", None) if engine is not None else None
    concept_memory = getattr(engine, "concept_memory", None) if engine is not None else None
    agp = getattr(engine, "agp", None) if engine is not None else None
    wrapper = getattr(engine, "self_embedding", None) if engine is not None else None
    return {
        "runtime_mapping_enabled": bool(getattr(lcm, "runtime_mapping_enabled", False)),
        "enforcement_enabled": bool(getattr(lcm, "enforcement_enabled", False)),
        "concept_categories": _safe_call(lcm, "concept_categories_snapshot", {}),
        "concept_commit_records": _safe_call(lcm, "concept_commit_records", []),
        "eve_specific_vector_store": _safe_call(store, "stats", {"count": 0, "read_only": True}),
        "wrapper_telemetry": _safe_call(wrapper, "telemetry", {}),
        "sa_active_categories": _active_categories(engine),
        "concept_memory_stats": _safe_call(concept_memory, "stats", {}),
        "agp_stats": _safe_call(agp, "stats", {}),
    }


def build_operator_approval_guard(
    *,
    source_decision: dict[str, Any] | None,
    source_dry_run: dict[str, Any] | None,
    operator_approved: bool = False,
    operator_approval_token: str | None = None,
) -> dict[str, Any]:
    """Validate Round106, Round107, and explicit operator approval prerequisites."""
    decision = deepcopy(source_decision or {})
    dry_run = deepcopy(source_dry_run or {})
    checks = {
        "round106_decision_version_ok": decision.get("decision_version") == ROUND106_PERSISTENCE_DECISION_VERSION,
        "round106_decision_ready": bool(decision.get("decision_ready", False)),
        "round106_not_applied": bool(decision.get("runtime_mapping_persisted_now", False)) is False,
        "round107_dry_run_version_ok": dry_run.get("dry_run_version") == ROUND107_ACTIVATION_DRYRUN_VERSION,
        "round107_dry_run_not_enabled": dry_run.get("activation_status") == "dry_run_only_not_enabled",
        "round107_no_mutation": bool((dry_run.get("dry_run_execution") or {}).get("no_agp_vector_category_memory_mutation", False)),
        "operator_approved": bool(operator_approved),
        "operator_token_ok": operator_approval_token == ROUND108_OPERATOR_APPROVAL_TOKEN,
    }
    blocking_reasons = sorted(key for key, passed in checks.items() if not passed)
    return {
        "guard_version": f"{ROUND108_ACTIVATION_CANDIDATE_VERSION}.operator_guard.v1",
        "approval_required": True,
        "approval_token_required": ROUND108_OPERATOR_APPROVAL_TOKEN,
        "guard_passed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
        "source_decision_version": decision.get("decision_version"),
        "source_dry_run_version": dry_run.get("dry_run_version"),
        "read_only": True,
    }


def build_runtime_mapping_activation_checkpoint(
    *,
    engine: Any | None,
    source_decision: dict[str, Any],
    source_dry_run: dict[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Build a checkpoint payload before any candidate mutation."""
    state = capture_runtime_mapping_activation_state(engine)
    artifact_root = Path(artifact_dir)
    mapped_tokens = sorted(str(t) for t in (source_decision.get("mapped_tokens") or source_dry_run.get("mapped_tokens") or []))
    return {
        "checkpoint_version": f"{ROUND108_ACTIVATION_CANDIDATE_VERSION}.checkpoint.v1",
        "created_by_round": 108,
        "source_decision_version": source_decision.get("decision_version"),
        "source_dry_run_version": source_dry_run.get("dry_run_version"),
        "mapped_tokens": mapped_tokens,
        "runtime_mapping_enabled_before": state["runtime_mapping_enabled"],
        "enforcement_enabled_before": state["enforcement_enabled"],
        "state_before": state,
        "rollback_artifact_path": str(artifact_root / ROLLBACK_FILENAME),
        "audit_log_path": str(artifact_root / AUDIT_LOG_FILENAME),
        "state_debug_before_path": str(artifact_root / STATE_DEBUG_BEFORE_FILENAME),
        "state_debug_after_path": str(artifact_root / STATE_DEBUG_AFTER_FILENAME),
        "state_debug_after_rollback_path": str(artifact_root / STATE_DEBUG_ROLLBACK_FILENAME),
        "checkpoint_created_before_candidate_mutation": True,
        "contains_vectors_npy": False,
        "contains_semantic_memory_dump": False,
        "json_only": True,
    }


def verify_runtime_mapping_rollback(engine: Any | None, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Verify rollback restored disabled flags and preserved protected state."""
    after = capture_runtime_mapping_activation_state(engine)
    before = checkpoint.get("state_before", {}) or {}
    checks = {
        "runtime_mapping_enabled_restored_false": after["runtime_mapping_enabled"] is False,
        "enforcement_enabled_restored_false": after["enforcement_enabled"] is False,
        "concept_categories_unchanged": after["concept_categories"] == before.get("concept_categories"),
        "concept_commit_records_unchanged": after["concept_commit_records"] == before.get("concept_commit_records"),
        "eve_specific_vector_store_unchanged": after["eve_specific_vector_store"] == before.get("eve_specific_vector_store"),
        "sa_active_categories_unchanged": after["sa_active_categories"] == before.get("sa_active_categories"),
        "concept_memory_stats_unchanged": after["concept_memory_stats"] == before.get("concept_memory_stats"),
        "agp_stats_unchanged": after["agp_stats"] == before.get("agp_stats"),
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }
    passed = all(bool(value) for key, value in checks.items() if key not in {"vectors_npy_committed", "agp_bypass_used"}) and not checks["vectors_npy_committed"] and not checks["agp_bypass_used"]
    return {
        "rollback_verification_version": f"{ROUND108_ACTIVATION_CANDIDATE_VERSION}.rollback_verification.v1",
        "rollback_verified": passed,
        "checks": checks,
        "state_after_rollback": after,
    }


def run_runtime_mapping_persistence_activation_candidate(
    *,
    engine: Any | None = None,
    source_decision: dict[str, Any] | None = None,
    source_dry_run: dict[str, Any] | None = None,
    operator_approved: bool = False,
    operator_approval_token: str | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation",
    apply_candidate: bool = False,
) -> dict[str, Any]:
    """Run the guarded activation candidate.

    By default this is blocked/read-only.  With explicit approval and
    ``apply_candidate=True``, it writes checkpoint/audit/state-debug artifacts,
    performs an ephemeral candidate flag mutation, then rolls back and verifies
    the protected state surfaces.
    """
    decision = deepcopy(source_decision or {})
    dry_run = deepcopy(source_dry_run) if source_dry_run is not None else build_runtime_mapping_persistence_activation_dryrun(
        engine=engine,
        source_decision=decision,
    )
    artifact_root = Path(artifact_dir)
    checkpoint_path = artifact_root / CHECKPOINT_FILENAME
    rollback_path = artifact_root / ROLLBACK_FILENAME
    audit_path = artifact_root / AUDIT_LOG_FILENAME
    before_debug_path = artifact_root / STATE_DEBUG_BEFORE_FILENAME
    after_debug_path = artifact_root / STATE_DEBUG_AFTER_FILENAME
    rollback_debug_path = artifact_root / STATE_DEBUG_ROLLBACK_FILENAME

    guard = build_operator_approval_guard(
        source_decision=decision,
        source_dry_run=dry_run,
        operator_approved=operator_approved,
        operator_approval_token=operator_approval_token,
    )
    lcm = getattr(engine, "lex_concept_mapping", None) if engine is not None else None
    initial_state = capture_runtime_mapping_activation_state(engine)

    if not guard["guard_passed"] or not apply_candidate:
        return {
            "candidate_version": ROUND108_ACTIVATION_CANDIDATE_VERSION,
            "round": 108,
            "activation_status": "blocked_operator_guard_or_apply_flag_missing",
            "guard": guard,
            "apply_candidate_requested": bool(apply_candidate),
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled_default": False,
            "runtime_mapping_enabled_now": initial_state["runtime_mapping_enabled"],
            "enforcement_enabled_now": initial_state["enforcement_enabled"],
            "checkpoint_written_before_mutation": False,
            "audit_log_emitted": False,
            "state_debug_before_exported": False,
            "state_debug_after_exported": False,
            "rollback_verified": False,
            "candidate_mutation_applied": False,
            "vectors_npy_committed": False,
            "agp_bypass_used": False,
            "read_only": True,
        }

    event_index = 0
    before_debug = StateDebugAdapter(engine).snapshot_state()
    _write_json(before_debug_path, {"export_version": ROUND108_ACTIVATION_CANDIDATE_VERSION, "phase": "before_candidate", "state_debug": before_debug})
    _append_audit(audit_path, {
        "event_version": ROUND108_ACTIVATION_CANDIDATE_VERSION,
        "event_type": "activation_precheck_started",
        "event_index": event_index,
        "operator_reference": operator_approval_token,
        "runtime_mapping_enabled_before": initial_state["runtime_mapping_enabled"],
        "enforcement_enabled_before": initial_state["enforcement_enabled"],
        "mutation_boundary_checks": guard["checks"],
    })
    event_index += 1

    checkpoint = build_runtime_mapping_activation_checkpoint(
        engine=engine,
        source_decision=decision,
        source_dry_run=dry_run,
        artifact_dir=artifact_root,
    )
    _write_json(checkpoint_path, checkpoint)
    _append_audit(audit_path, {
        "event_version": ROUND108_ACTIVATION_CANDIDATE_VERSION,
        "event_type": "checkpoint_written",
        "event_index": event_index,
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "checkpoint_path": str(checkpoint_path),
        "runtime_mapping_enabled_before": checkpoint["runtime_mapping_enabled_before"],
        "enforcement_enabled_before": checkpoint["enforcement_enabled_before"],
        "mapped_tokens": checkpoint["mapped_tokens"],
        "mutation_boundary_checks": {"checkpoint_created_before_candidate_mutation": True},
        "operator_reference": operator_approval_token,
    })
    event_index += 1

    if lcm is not None:
        lcm.runtime_mapping_enabled = True
        lcm.enforcement_enabled = False
    after_activation_state = capture_runtime_mapping_activation_state(engine)
    after_debug = StateDebugAdapter(engine).snapshot_state()
    _write_json(after_debug_path, {"export_version": ROUND108_ACTIVATION_CANDIDATE_VERSION, "phase": "after_candidate_activation", "state_debug": after_debug})
    _append_audit(audit_path, {
        "event_version": ROUND108_ACTIVATION_CANDIDATE_VERSION,
        "event_type": "activation_candidate_applied",
        "event_index": event_index,
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "runtime_mapping_enabled_before": checkpoint["runtime_mapping_enabled_before"],
        "runtime_mapping_enabled_after": after_activation_state["runtime_mapping_enabled"],
        "enforcement_enabled_before": checkpoint["enforcement_enabled_before"],
        "enforcement_enabled_after": after_activation_state["enforcement_enabled"],
        "mapped_tokens": checkpoint["mapped_tokens"],
        "mutation_boundary_checks": {
            "only_runtime_mapping_flag_mutated": True,
            "enforcement_enabled": False,
            "vectors_npy_committed": False,
            "agp_bypass_used": False,
        },
        "operator_reference": operator_approval_token,
    })
    event_index += 1

    if lcm is not None:
        lcm.runtime_mapping_enabled = False
        lcm.enforcement_enabled = False
    rollback_verification = verify_runtime_mapping_rollback(engine, checkpoint)
    rollback_payload = {
        "rollback_version": f"{ROUND108_ACTIVATION_CANDIDATE_VERSION}.rollback.v1",
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "restore_runtime_mapping_enabled": False,
        "restore_enforcement_enabled": False,
        "rollback_verification": rollback_verification,
        "json_only": True,
    }
    _write_json(rollback_path, rollback_payload)
    rollback_debug = StateDebugAdapter(engine).snapshot_state()
    _write_json(rollback_debug_path, {"export_version": ROUND108_ACTIVATION_CANDIDATE_VERSION, "phase": "after_candidate_rollback", "state_debug": rollback_debug})
    _append_audit(audit_path, {
        "event_version": ROUND108_ACTIVATION_CANDIDATE_VERSION,
        "event_type": "rollback_validation_passed" if rollback_verification["rollback_verified"] else "rollback_validation_failed",
        "event_index": event_index,
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "runtime_mapping_enabled_after": rollback_verification["state_after_rollback"]["runtime_mapping_enabled"],
        "enforcement_enabled_after": rollback_verification["state_after_rollback"]["enforcement_enabled"],
        "mapped_tokens": checkpoint["mapped_tokens"],
        "mutation_boundary_checks": rollback_verification["checks"],
        "operator_reference": operator_approval_token,
    })

    return {
        "candidate_version": ROUND108_ACTIVATION_CANDIDATE_VERSION,
        "round": 108,
        "activation_status": "activation_candidate_applied_then_rolled_back" if rollback_verification["rollback_verified"] else "activation_candidate_rollback_failed",
        "guard": guard,
        "apply_candidate_requested": True,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "runtime_mapping_enabled_during_candidate": after_activation_state["runtime_mapping_enabled"],
        "enforcement_enabled_during_candidate": after_activation_state["enforcement_enabled"],
        "runtime_mapping_enabled_now": rollback_verification["state_after_rollback"]["runtime_mapping_enabled"],
        "enforcement_enabled_now": rollback_verification["state_after_rollback"]["enforcement_enabled"],
        "checkpoint_written_before_mutation": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "rollback_path": str(rollback_path),
        "audit_log_path": str(audit_path),
        "state_debug_before_path": str(before_debug_path),
        "state_debug_after_path": str(after_debug_path),
        "state_debug_after_rollback_path": str(rollback_debug_path),
        "audit_log_emitted": audit_path.exists(),
        "state_debug_before_exported": before_debug_path.exists(),
        "state_debug_after_exported": after_debug_path.exists(),
        "state_debug_after_rollback_exported": rollback_debug_path.exists(),
        "rollback_verified": rollback_verification["rollback_verified"],
        "rollback_verification": rollback_verification,
        "candidate_mutation_applied": True,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "protected_state_unchanged_after_rollback": rollback_verification["rollback_verified"],
        "read_only": False,
    }


def write_round108_runtime_mapping_persistence_activation_candidate(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    _write_json(out, report)
    return {
        "export_version": report.get("candidate_version", ROUND108_ACTIVATION_CANDIDATE_VERSION),
        "path": str(out),
        "json_written": True,
        "runtime_mapping_enabled": bool(report.get("runtime_mapping_enabled_now", False)),
        "enforcement_enabled": bool(report.get("enforcement_enabled_now", False)),
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }


__all__ = [
    "ROUND108_ACTIVATION_CANDIDATE_VERSION",
    "ROUND108_OPERATOR_APPROVAL_TOKEN",
    "build_operator_approval_guard",
    "build_runtime_mapping_activation_checkpoint",
    "capture_runtime_mapping_activation_state",
    "run_runtime_mapping_persistence_activation_candidate",
    "verify_runtime_mapping_rollback",
    "write_round108_runtime_mapping_persistence_activation_candidate",
]
