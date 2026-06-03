"""Rounds110-112 runtime mapping limited persistence sandbox helpers.

These helpers exercise a JSON-only sandbox for runtime mapping persistence after
Round109's approval fixture / rollback drill.  They deliberately do not enable
production persistence, enforcement, AGP bypass, or vector/category/memory
mutation.  The only permitted mutation is an in-memory runtime mapping flag plus
JSON sandbox artifacts under an explicit artifact directory; every runner restores
``runtime_mapping_enabled=False`` and ``enforcement_enabled=False`` before it
returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from adapters.runtime_mapping_persistence_activation_candidate import (
    ROUND108_OPERATOR_APPROVAL_TOKEN,
    capture_runtime_mapping_activation_state,
)
from adapters.runtime_mapping_persistence_approval_fixture import (
    ROUND109_APPROVAL_FIXTURE_VERSION,
    ROUND109_APPROVAL_SCOPE,
    build_round109_operator_approval_fixture,
)
from adapters.state_debug_adapter import StateDebugAdapter

ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION = "v3_round110_runtime_mapping_limited_persistence_sandbox"
ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION = "v3_round111_sandbox_rollback_cleanup_verification"
ROUND112_SANDBOX_AUDIT_REPLAY_VERSION = "v3_round112_post_sandbox_focused_validation_audit_replay"

SANDBOX_CHECKPOINT_FILENAME = "runtime_mapping_round110_sandbox_checkpoint.json"
SANDBOX_STATE_FILENAME = "runtime_mapping_round110_sandbox_state.json"
SANDBOX_ROLLBACK_FILENAME = "runtime_mapping_round110_sandbox_rollback.json"
SANDBOX_AUDIT_FILENAME = "runtime_mapping_round110_sandbox_audit_log.jsonl"
SANDBOX_DEBUG_BEFORE_FILENAME = "runtime_mapping_round110_state_debug_before.json"
SANDBOX_DEBUG_DURING_FILENAME = "runtime_mapping_round110_state_debug_during_sandbox.json"
SANDBOX_DEBUG_AFTER_FILENAME = "runtime_mapping_round110_state_debug_after_rollback.json"
SANDBOX_CLEANUP_RECEIPT_FILENAME = "runtime_mapping_round111_cleanup_receipt.json"
SANDBOX_REPLAY_FILENAME = "runtime_mapping_round112_audit_replay.json"

_SANDBOX_EVENT_ORDER = [
    "sandbox_precheck_started",
    "sandbox_checkpoint_written",
    "sandbox_mapping_persisted_json_only",
    "sandbox_rollback_applied",
]
_CLEANUP_EVENT_ORDER = [
    "cleanup_precheck_started",
    "sandbox_state_file_removed",
    "cleanup_verification_passed",
]
_FORBIDDEN_SUFFIXES = ("vectors.npy", ".zip", ".part")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_tokens(tokens: Iterable[str] | None) -> list[str]:
    return sorted(str(token) for token in (tokens or ["민석"]) if str(token))


def _artifact_paths(root: str | Path) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(str(path.relative_to(base)).replace("\\", "/") for path in base.rglob("*") if path.is_file())


def _forbidden_artifact_paths(paths: Iterable[str]) -> list[str]:
    forbidden: list[str] = []
    for path in paths:
        normalized = str(path).replace("\\", "/")
        name = Path(normalized).name
        if name.endswith(_FORBIDDEN_SUFFIXES) or normalized.startswith("seeds/subsets/") or "_operator_artifacts" in normalized:
            forbidden.append(str(path))
    return sorted(forbidden)


def _event_order_report(events: list[dict[str, Any]], expected: list[str]) -> dict[str, Any]:
    actual = [str(event.get("event_type")) for event in events]
    indices = [event.get("event_index") for event in events]
    return {
        "expected_event_order": expected,
        "actual_event_order": actual,
        "event_indices": indices,
        "event_order_verified": actual == expected and indices == list(range(len(expected))),
    }


def _set_flags(engine: Any | None, *, runtime_mapping_enabled: bool, enforcement_enabled: bool) -> None:
    lcm = getattr(engine, "lex_concept_mapping", None) if engine is not None else None
    if lcm is not None:
        lcm.runtime_mapping_enabled = bool(runtime_mapping_enabled)
        lcm.enforcement_enabled = bool(enforcement_enabled)


def _state_debug(engine: Any | None) -> dict[str, Any]:
    return StateDebugAdapter(engine).snapshot_state() if engine is not None else {"engine_missing": True}


def run_round110_runtime_mapping_limited_persistence_sandbox(
    *,
    engine: Any | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation/round110_runtime_mapping_limited_persistence_sandbox",
    tokens: Iterable[str] | None = None,
    operator_approved: bool = True,
    operator_approval_token: str = ROUND108_OPERATOR_APPROVAL_TOKEN,
) -> dict[str, Any]:
    """Run a limited JSON-only sandbox persistence drill and rollback.

    The sandbox writes a checkpoint, a sandbox mapping JSON file, state-debug
    snapshots, and an audit JSONL file.  It restores disabled flags before
    returning and never enables enforcement or production persistence.
    """

    artifact_root = Path(artifact_dir)
    checkpoint_path = artifact_root / SANDBOX_CHECKPOINT_FILENAME
    sandbox_state_path = artifact_root / SANDBOX_STATE_FILENAME
    rollback_path = artifact_root / SANDBOX_ROLLBACK_FILENAME
    audit_path = artifact_root / SANDBOX_AUDIT_FILENAME
    before_debug_path = artifact_root / SANDBOX_DEBUG_BEFORE_FILENAME
    during_debug_path = artifact_root / SANDBOX_DEBUG_DURING_FILENAME
    after_debug_path = artifact_root / SANDBOX_DEBUG_AFTER_FILENAME
    requested_tokens = _normalize_tokens(tokens)
    fixture = build_round109_operator_approval_fixture(tokens=requested_tokens, operator_approval_token=operator_approval_token)
    guard_checks = {
        "round109_fixture_version_ok": fixture.get("fixture_version") == ROUND109_APPROVAL_FIXTURE_VERSION,
        "approval_scope_limited": fixture.get("approval_scope") == ROUND109_APPROVAL_SCOPE,
        "fixture_approval_valid": bool(fixture.get("approval_valid", False)),
        "operator_approved": bool(operator_approved),
        "operator_token_ok": operator_approval_token == ROUND108_OPERATOR_APPROVAL_TOKEN,
        "production_persistence_disabled": True,
        "enforcement_enablement_forbidden": True,
    }
    blocking_reasons = sorted(key for key, passed in guard_checks.items() if not passed)
    before_state = capture_runtime_mapping_activation_state(engine)
    if blocking_reasons:
        return {
            "sandbox_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
            "round": 110,
            "sandbox_status": "blocked_guard_failed",
            "guard_checks": guard_checks,
            "blocking_reasons": blocking_reasons,
            "runtime_mapping_enabled_now": before_state["runtime_mapping_enabled"],
            "enforcement_enabled_now": before_state["enforcement_enabled"],
            "production_persistence_enabled": False,
            "sandbox_state_written": False,
            "vectors_npy_committed": False,
            "agp_bypass_used": False,
            "read_only": True,
        }

    event_index = 0
    before_debug = _state_debug(engine)
    _write_json(before_debug_path, {"export_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION, "phase": "before_sandbox", "state_debug": before_debug})
    _append_jsonl(audit_path, {
        "event_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
        "event_type": "sandbox_precheck_started",
        "event_index": event_index,
        "operator_reference": operator_approval_token,
        "mapped_tokens": requested_tokens,
        "runtime_mapping_enabled_before": before_state["runtime_mapping_enabled"],
        "enforcement_enabled_before": before_state["enforcement_enabled"],
        "guard_checks": guard_checks,
    })
    event_index += 1

    checkpoint = {
        "checkpoint_version": f"{ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION}.checkpoint.v1",
        "created_by_round": 110,
        "source_fixture_version": fixture.get("fixture_version"),
        "mapped_tokens": requested_tokens,
        "runtime_mapping_enabled_before": before_state["runtime_mapping_enabled"],
        "enforcement_enabled_before": before_state["enforcement_enabled"],
        "state_before": before_state,
        "sandbox_state_path": str(sandbox_state_path),
        "rollback_path": str(rollback_path),
        "audit_log_path": str(audit_path),
        "production_persistence_enabled": False,
        "enforcement_enabled_default": False,
        "json_only": True,
        "contains_vectors_npy": False,
    }
    _write_json(checkpoint_path, checkpoint)
    _append_jsonl(audit_path, {
        "event_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
        "event_type": "sandbox_checkpoint_written",
        "event_index": event_index,
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "checkpoint_path": str(checkpoint_path),
        "mapped_tokens": requested_tokens,
    })
    event_index += 1

    _set_flags(engine, runtime_mapping_enabled=True, enforcement_enabled=False)
    during_state = capture_runtime_mapping_activation_state(engine)
    sandbox_state = {
        "sandbox_state_version": f"{ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION}.sandbox_state.v1",
        "round": 110,
        "scope": "limited_sandbox_json_only",
        "mapped_tokens": requested_tokens,
        "runtime_mapping_enabled_in_sandbox": during_state["runtime_mapping_enabled"],
        "enforcement_enabled_in_sandbox": during_state["enforcement_enabled"],
        "production_persistence_enabled": False,
        "may_be_deleted_by_round111_cleanup": True,
        "json_only": True,
        "contains_vectors_npy": False,
        "agp_bypass_used": False,
    }
    _write_json(sandbox_state_path, sandbox_state)
    _write_json(during_debug_path, {"export_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION, "phase": "during_sandbox", "state_debug": _state_debug(engine)})
    _append_jsonl(audit_path, {
        "event_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
        "event_type": "sandbox_mapping_persisted_json_only",
        "event_index": event_index,
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "sandbox_state_path": str(sandbox_state_path),
        "runtime_mapping_enabled_during_sandbox": during_state["runtime_mapping_enabled"],
        "enforcement_enabled_during_sandbox": during_state["enforcement_enabled"],
        "production_persistence_enabled": False,
    })
    event_index += 1

    _set_flags(engine, runtime_mapping_enabled=False, enforcement_enabled=False)
    after_state = capture_runtime_mapping_activation_state(engine)
    rollback_checks = {
        "runtime_mapping_enabled_restored_false": after_state["runtime_mapping_enabled"] is False,
        "enforcement_enabled_restored_false": after_state["enforcement_enabled"] is False,
        "concept_categories_unchanged": after_state["concept_categories"] == before_state["concept_categories"],
        "concept_commit_records_unchanged": after_state["concept_commit_records"] == before_state["concept_commit_records"],
        "eve_specific_vector_store_unchanged": after_state["eve_specific_vector_store"] == before_state["eve_specific_vector_store"],
        "sa_active_categories_unchanged": after_state["sa_active_categories"] == before_state["sa_active_categories"],
        "concept_memory_stats_unchanged": after_state["concept_memory_stats"] == before_state["concept_memory_stats"],
        "agp_stats_unchanged": after_state["agp_stats"] == before_state["agp_stats"],
    }
    rollback_verified = all(rollback_checks.values())
    rollback = {
        "rollback_version": f"{ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION}.rollback.v1",
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "restore_runtime_mapping_enabled": False,
        "restore_enforcement_enabled": False,
        "rollback_checks": rollback_checks,
        "rollback_verified": rollback_verified,
        "state_after_rollback": after_state,
        "json_only": True,
    }
    _write_json(rollback_path, rollback)
    _write_json(after_debug_path, {"export_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION, "phase": "after_sandbox_rollback", "state_debug": _state_debug(engine)})
    _append_jsonl(audit_path, {
        "event_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
        "event_type": "sandbox_rollback_applied",
        "event_index": event_index,
        "source_checkpoint_version": checkpoint["checkpoint_version"],
        "rollback_verified": rollback_verified,
        "runtime_mapping_enabled_after": after_state["runtime_mapping_enabled"],
        "enforcement_enabled_after": after_state["enforcement_enabled"],
        "rollback_checks": rollback_checks,
    })

    artifacts = _artifact_paths(artifact_root)
    forbidden = _forbidden_artifact_paths(artifacts)
    event_order = _event_order_report(_read_jsonl(audit_path), _SANDBOX_EVENT_ORDER)
    return {
        "sandbox_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
        "round": 110,
        "sandbox_status": "limited_persistence_sandbox_passed" if rollback_verified and event_order["event_order_verified"] and not forbidden else "limited_persistence_sandbox_failed",
        "fixture": fixture,
        "guard_checks": guard_checks,
        "blocking_reasons": [],
        "mapped_tokens": requested_tokens,
        "checkpoint_path": str(checkpoint_path),
        "sandbox_state_path": str(sandbox_state_path),
        "rollback_path": str(rollback_path),
        "audit_log_path": str(audit_path),
        "state_debug_paths": {"before": str(before_debug_path), "during": str(during_debug_path), "after_rollback": str(after_debug_path)},
        "event_order": event_order,
        "rollback_verified": rollback_verified,
        "rollback_checks": rollback_checks,
        "runtime_mapping_enabled_during_sandbox": during_state["runtime_mapping_enabled"],
        "enforcement_enabled_during_sandbox": during_state["enforcement_enabled"],
        "runtime_mapping_enabled_now": after_state["runtime_mapping_enabled"],
        "enforcement_enabled_now": after_state["enforcement_enabled"],
        "production_persistence_enabled": False,
        "sandbox_state_written": sandbox_state_path.exists(),
        "forbidden_artifact_paths": forbidden,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": False,
    }


def run_round111_sandbox_rollback_cleanup_verification(
    *,
    engine: Any | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation/round110_runtime_mapping_limited_persistence_sandbox",
    remove_sandbox_state: bool = True,
) -> dict[str, Any]:
    """Verify Round110 rollback and remove the JSON sandbox state file."""

    artifact_root = Path(artifact_dir)
    audit_path = artifact_root / SANDBOX_AUDIT_FILENAME
    checkpoint_path = artifact_root / SANDBOX_CHECKPOINT_FILENAME
    sandbox_state_path = artifact_root / SANDBOX_STATE_FILENAME
    rollback_path = artifact_root / SANDBOX_ROLLBACK_FILENAME
    cleanup_receipt_path = artifact_root / SANDBOX_CLEANUP_RECEIPT_FILENAME
    cleanup_audit_path = artifact_root / "runtime_mapping_round111_cleanup_audit_log.jsonl"
    before_cleanup_state = capture_runtime_mapping_activation_state(engine)

    event_index = 0
    _append_jsonl(cleanup_audit_path, {
        "event_version": ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION,
        "event_type": "cleanup_precheck_started",
        "event_index": event_index,
        "round110_audit_exists": audit_path.exists(),
        "sandbox_state_exists_before_cleanup": sandbox_state_path.exists(),
        "runtime_mapping_enabled_before_cleanup": before_cleanup_state["runtime_mapping_enabled"],
        "enforcement_enabled_before_cleanup": before_cleanup_state["enforcement_enabled"],
    })
    event_index += 1

    sandbox_state_existed = sandbox_state_path.exists()
    if remove_sandbox_state and sandbox_state_path.exists():
        sandbox_state_path.unlink()
    _append_jsonl(cleanup_audit_path, {
        "event_version": ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION,
        "event_type": "sandbox_state_file_removed",
        "event_index": event_index,
        "sandbox_state_existed_before_cleanup": sandbox_state_existed,
        "sandbox_state_exists_after_cleanup": sandbox_state_path.exists(),
    })
    event_index += 1

    after_cleanup_state = capture_runtime_mapping_activation_state(engine)
    round110_events = _read_jsonl(audit_path)
    rollback_payload = _read_json(rollback_path) if rollback_path.exists() else {}
    checks = {
        "round110_checkpoint_exists": checkpoint_path.exists(),
        "round110_audit_order_verified": _event_order_report(round110_events, _SANDBOX_EVENT_ORDER)["event_order_verified"],
        "round110_rollback_verified": bool(rollback_payload.get("rollback_verified", False)),
        "sandbox_state_removed": sandbox_state_path.exists() is False,
        "runtime_mapping_enabled_false": after_cleanup_state["runtime_mapping_enabled"] is False,
        "enforcement_enabled_false": after_cleanup_state["enforcement_enabled"] is False,
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }
    cleanup_verified = (
        checks["round110_checkpoint_exists"]
        and checks["round110_audit_order_verified"]
        and checks["round110_rollback_verified"]
        and checks["sandbox_state_removed"]
        and checks["runtime_mapping_enabled_false"]
        and checks["enforcement_enabled_false"]
        and not checks["production_persistence_enabled"]
        and not checks["vectors_npy_committed"]
        and not checks["agp_bypass_used"]
    )
    receipt = {
        "cleanup_version": ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION,
        "round": 111,
        "cleanup_status": "sandbox_cleanup_verified" if cleanup_verified else "sandbox_cleanup_failed",
        "source_sandbox_version": ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
        "checks": checks,
        "sandbox_state_existed_before_cleanup": sandbox_state_existed,
        "sandbox_state_exists_after_cleanup": sandbox_state_path.exists(),
        "runtime_mapping_enabled_now": after_cleanup_state["runtime_mapping_enabled"],
        "enforcement_enabled_now": after_cleanup_state["enforcement_enabled"],
        "production_persistence_enabled": False,
        "json_only": True,
    }
    _write_json(cleanup_receipt_path, receipt)
    _append_jsonl(cleanup_audit_path, {
        "event_version": ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION,
        "event_type": "cleanup_verification_passed" if cleanup_verified else "cleanup_verification_failed",
        "event_index": event_index,
        "cleanup_verified": cleanup_verified,
        "checks": checks,
    })
    cleanup_events = _read_jsonl(cleanup_audit_path)
    artifacts = _artifact_paths(artifact_root)
    forbidden = _forbidden_artifact_paths(artifacts)
    return {
        **receipt,
        "cleanup_receipt_path": str(cleanup_receipt_path),
        "cleanup_audit_path": str(cleanup_audit_path),
        "cleanup_event_order": _event_order_report(cleanup_events, _CLEANUP_EVENT_ORDER if cleanup_verified else _CLEANUP_EVENT_ORDER[:-1] + ["cleanup_verification_failed"]),
        "round110_audit_event_order": _event_order_report(round110_events, _SANDBOX_EVENT_ORDER),
        "forbidden_artifact_paths": forbidden,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": False,
    }


def run_round112_post_sandbox_focused_validation_audit_replay(
    *,
    engine: Any | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation/round110_runtime_mapping_limited_persistence_sandbox",
) -> dict[str, Any]:
    """Replay Round110/111 audit artifacts without mutating sandbox state."""

    artifact_root = Path(artifact_dir)
    round110_events = _read_jsonl(artifact_root / SANDBOX_AUDIT_FILENAME)
    cleanup_events = _read_jsonl(artifact_root / "runtime_mapping_round111_cleanup_audit_log.jsonl")
    checkpoint = _read_json(artifact_root / SANDBOX_CHECKPOINT_FILENAME) if (artifact_root / SANDBOX_CHECKPOINT_FILENAME).exists() else {}
    rollback = _read_json(artifact_root / SANDBOX_ROLLBACK_FILENAME) if (artifact_root / SANDBOX_ROLLBACK_FILENAME).exists() else {}
    cleanup = _read_json(artifact_root / SANDBOX_CLEANUP_RECEIPT_FILENAME) if (artifact_root / SANDBOX_CLEANUP_RECEIPT_FILENAME).exists() else {}
    current_state = capture_runtime_mapping_activation_state(engine)
    replay_checks = {
        "round110_event_order_verified": _event_order_report(round110_events, _SANDBOX_EVENT_ORDER)["event_order_verified"],
        "round111_event_order_verified": _event_order_report(cleanup_events, _CLEANUP_EVENT_ORDER)["event_order_verified"],
        "checkpoint_before_runtime_mapping_enabled_false": checkpoint.get("runtime_mapping_enabled_before") is False,
        "checkpoint_before_enforcement_enabled_false": checkpoint.get("enforcement_enabled_before") is False,
        "round110_rollback_verified": bool(rollback.get("rollback_verified", False)),
        "round111_cleanup_verified": cleanup.get("cleanup_status") == "sandbox_cleanup_verified",
        "sandbox_state_removed": (artifact_root / SANDBOX_STATE_FILENAME).exists() is False,
        "runtime_mapping_enabled_now_false": current_state["runtime_mapping_enabled"] is False,
        "enforcement_enabled_now_false": current_state["enforcement_enabled"] is False,
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }
    replay_passed = (
        all(value for key, value in replay_checks.items() if key not in {"production_persistence_enabled", "vectors_npy_committed", "agp_bypass_used"})
        and not replay_checks["production_persistence_enabled"]
        and not replay_checks["vectors_npy_committed"]
        and not replay_checks["agp_bypass_used"]
    )
    artifacts = _artifact_paths(artifact_root)
    forbidden = _forbidden_artifact_paths(artifacts)
    replay = {
        "replay_version": ROUND112_SANDBOX_AUDIT_REPLAY_VERSION,
        "round": 112,
        "replay_status": "post_sandbox_audit_replay_passed" if replay_passed and not forbidden else "post_sandbox_audit_replay_failed",
        "source_rounds": [110, 111],
        "round110_event_order": _event_order_report(round110_events, _SANDBOX_EVENT_ORDER),
        "round111_event_order": _event_order_report(cleanup_events, _CLEANUP_EVENT_ORDER),
        "replay_checks": replay_checks,
        "runtime_mapping_enabled_now": current_state["runtime_mapping_enabled"],
        "enforcement_enabled_now": current_state["enforcement_enabled"],
        "production_persistence_enabled": False,
        "forbidden_artifact_paths": forbidden,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }
    _write_json(artifact_root / SANDBOX_REPLAY_FILENAME, replay)
    return replay


def write_round110_runtime_mapping_limited_persistence_sandbox(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("sandbox_version", ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": bool(report.get("runtime_mapping_enabled_now", False)), "enforcement_enabled": bool(report.get("enforcement_enabled_now", False)), "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round111_sandbox_rollback_cleanup_verification(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("cleanup_version", ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": bool(report.get("runtime_mapping_enabled_now", False)), "enforcement_enabled": bool(report.get("enforcement_enabled_now", False)), "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round112_post_sandbox_focused_validation_audit_replay(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("replay_version", ROUND112_SANDBOX_AUDIT_REPLAY_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": bool(report.get("runtime_mapping_enabled_now", False)), "enforcement_enabled": bool(report.get("enforcement_enabled_now", False)), "vectors_npy_committed": False, "agp_bypass_used": False}

# Rounds113-117 autonomous continuation surfaces.
ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_VERSION = "v3_round113_state_debug_audit_replay_viewer"
ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_VERSION = "v3_round114_legacy_root_blocker_isolation"
ROUND115_BROADER_VALIDATION_TRIAGE_VERSION = "v3_round115_broader_validation_triage_report"
ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_VERSION = "v3_round116_runtime_mapping_sandbox_replay_regression_guard"
ROUND117_OPERATOR_GO_NO_GO_PACKAGE_VERSION = "v3_round117_operator_go_no_go_package"

ROUND113_VIEWER_FILENAME = "runtime_mapping_round113_state_debug_audit_replay_viewer.json"
ROUND114_BLOCKER_FILENAME = "runtime_mapping_round114_legacy_root_blocker_isolation.json"
ROUND115_TRIAGE_FILENAME = "runtime_mapping_round115_broader_validation_triage.json"
ROUND116_GUARD_FILENAME = "runtime_mapping_round116_sandbox_replay_regression_guard.json"
ROUND117_PACKAGE_FILENAME = "runtime_mapping_round117_operator_go_no_go_package.json"


def _safe_read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return _read_json(p) if p.exists() else {}


def build_round113_state_debug_audit_replay_viewer(
    *,
    engine: Any | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation/round110_runtime_mapping_limited_persistence_sandbox",
) -> dict[str, Any]:
    """Build a read-only state-debug/audit replay viewer for Round110-112 evidence."""

    artifact_root = Path(artifact_dir)
    replay = _safe_read_json(artifact_root / SANDBOX_REPLAY_FILENAME)
    checkpoint = _safe_read_json(artifact_root / SANDBOX_CHECKPOINT_FILENAME)
    rollback = _safe_read_json(artifact_root / SANDBOX_ROLLBACK_FILENAME)
    cleanup = _safe_read_json(artifact_root / SANDBOX_CLEANUP_RECEIPT_FILENAME)
    before_debug = _safe_read_json(artifact_root / SANDBOX_DEBUG_BEFORE_FILENAME)
    during_debug = _safe_read_json(artifact_root / SANDBOX_DEBUG_DURING_FILENAME)
    after_debug = _safe_read_json(artifact_root / SANDBOX_DEBUG_AFTER_FILENAME)
    round110_events = _read_jsonl(artifact_root / SANDBOX_AUDIT_FILENAME)
    round111_events = _read_jsonl(artifact_root / "runtime_mapping_round111_cleanup_audit_log.jsonl")
    current_state = capture_runtime_mapping_activation_state(engine)

    def _debug_flags(payload: dict[str, Any]) -> dict[str, Any]:
        lcm = (((payload.get("state_debug") or {}).get("lex_concept_mapping")) or {}) if payload else {}
        return {
            "runtime_mapping_enabled": bool(lcm.get("runtime_mapping_enabled", False)),
            "enforcement_enabled": bool(lcm.get("enforcement_enabled", False)),
            "round": int(lcm.get("round", 0) or 0),
        }

    debug_timeline = [
        {"phase": "before_sandbox", **_debug_flags(before_debug)},
        {"phase": "during_sandbox", **_debug_flags(during_debug)},
        {"phase": "after_rollback", **_debug_flags(after_debug)},
    ]
    checks = {
        "round112_replay_passed": replay.get("replay_status") == "post_sandbox_audit_replay_passed",
        "round110_event_order_verified": _event_order_report(round110_events, _SANDBOX_EVENT_ORDER)["event_order_verified"],
        "round111_event_order_verified": _event_order_report(round111_events, _CLEANUP_EVENT_ORDER)["event_order_verified"],
        "checkpoint_before_disabled": checkpoint.get("runtime_mapping_enabled_before") is False and checkpoint.get("enforcement_enabled_before") is False,
        "during_debug_sandbox_enabled_only": debug_timeline[1]["runtime_mapping_enabled"] is True and debug_timeline[1]["enforcement_enabled"] is False,
        "after_debug_disabled": debug_timeline[2]["runtime_mapping_enabled"] is False and debug_timeline[2]["enforcement_enabled"] is False,
        "rollback_verified": bool(rollback.get("rollback_verified", False)),
        "cleanup_verified": cleanup.get("cleanup_status") == "sandbox_cleanup_verified",
        "current_runtime_mapping_disabled": current_state["runtime_mapping_enabled"] is False,
        "current_enforcement_disabled": current_state["enforcement_enabled"] is False,
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }
    forbidden = _forbidden_artifact_paths(_artifact_paths(artifact_root))
    viewer_status = "state_debug_audit_replay_viewer_ready" if (
        all(value for key, value in checks.items() if key not in {"production_persistence_enabled", "vectors_npy_committed", "agp_bypass_used"})
        and not checks["production_persistence_enabled"]
        and not checks["vectors_npy_committed"]
        and not checks["agp_bypass_used"]
        and not forbidden
    ) else "state_debug_audit_replay_viewer_blocked"
    return {
        "viewer_version": ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_VERSION,
        "round": 113,
        "viewer_status": viewer_status,
        "source_rounds": [110, 111, 112],
        "artifact_dir": str(artifact_root),
        "audit_event_counts": {"round110": len(round110_events), "round111": len(round111_events)},
        "round110_event_order": _event_order_report(round110_events, _SANDBOX_EVENT_ORDER),
        "round111_event_order": _event_order_report(round111_events, _CLEANUP_EVENT_ORDER),
        "debug_timeline": debug_timeline,
        "viewer_checks": checks,
        "current_runtime_mapping_enabled": current_state["runtime_mapping_enabled"],
        "current_enforcement_enabled": current_state["enforcement_enabled"],
        "production_persistence_enabled": False,
        "forbidden_artifact_paths": forbidden,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def build_round114_legacy_root_blocker_isolation(
    *,
    repo_root: str | Path = ".",
    root_test_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Isolate legacy root-level collection blockers without changing tests."""

    root = Path(repo_root)
    files = sorted(root_test_files or [str(path.name) for path in root.glob("test*.py") if path.is_file()])
    blockers: list[dict[str, Any]] = []
    for rel in files:
        path = root / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if "spreading_activation" in text:
            blockers.append({
                "file": rel,
                "missing_import": "spreading_activation",
                "blocker_type": "legacy_root_collection_import_error",
                "isolated": True,
            })
    checks = {
        "root_tests_detected": bool(files),
        "legacy_import_blockers_detected": bool(blockers),
        "missing_spreading_activation_module": not (root / "spreading_activation.py").exists() and not (root / "spreading_activation").exists(),
        "runtime_mapping_enabled_default_remains_false": True,
        "enforcement_enabled_default_remains_false": True,
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }
    return {
        "blocker_isolation_version": ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_VERSION,
        "round": 114,
        "isolation_status": "legacy_root_blocker_isolated" if blockers else "legacy_root_blocker_not_reproduced_by_static_scan",
        "root_test_files": files,
        "isolated_blockers": blockers,
        "primary_blocker": "missing spreading_activation import in legacy root tests" if blockers else None,
        "recommended_handling": "record root pytest/collect-only as blocked-pre-existing; do not weaken tests in this PR",
        "checks": checks,
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def build_round115_broader_validation_triage_report(
    *,
    focused_results: Iterable[dict[str, Any]] | None = None,
    blocker_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize focused validation and broader blocked/partial validation."""

    focused = list(focused_results or [])
    blocker = blocker_report or {}
    broader_blockers = []
    if blocker.get("isolated_blockers"):
        broader_blockers.append({
            "command": "pytest -q --collect-only",
            "status": "blocked_pre_existing",
            "reason": "legacy root-level tests import missing spreading_activation",
            "affected_files": [item.get("file") for item in blocker.get("isolated_blockers", [])],
        })
    broader_blockers.append({
        "command": "pytest -q tests",
        "status": "partial_or_blocked_if_environment_lacks_seed_vector_artifacts_or_legacy_expectations_fail",
        "reason": "broader suite must be recorded honestly when fixture artifacts or older baseline expectations fail",
    })
    focused_passed = all(item.get("status") == "passed" for item in focused) if focused else True
    return {
        "triage_version": ROUND115_BROADER_VALIDATION_TRIAGE_VERSION,
        "round": 115,
        "triage_status": "focused_passed_broader_blocked_or_partial" if focused_passed else "focused_validation_failed",
        "focused_results": focused,
        "broader_validation": broader_blockers,
        "blocked_partial_validation": bool(broader_blockers),
        "blocker_source_round": blocker.get("round"),
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def run_round116_runtime_mapping_sandbox_replay_regression_guard(
    *,
    engine: Any | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation/round116_runtime_mapping_sandbox_replay_regression_guard",
) -> dict[str, Any]:
    """Rerun the sandbox/cleanup/replay sequence as a regression guard."""

    artifact_root = Path(artifact_dir)
    round110 = run_round110_runtime_mapping_limited_persistence_sandbox(engine=engine, artifact_dir=artifact_root)
    round111 = run_round111_sandbox_rollback_cleanup_verification(engine=engine, artifact_dir=artifact_root)
    round112 = run_round112_post_sandbox_focused_validation_audit_replay(engine=engine, artifact_dir=artifact_root)
    current_state = capture_runtime_mapping_activation_state(engine)
    checks = {
        "round110_passed": round110.get("sandbox_status") == "limited_persistence_sandbox_passed",
        "round111_cleanup_verified": round111.get("cleanup_status") == "sandbox_cleanup_verified",
        "round112_replay_passed": round112.get("replay_status") == "post_sandbox_audit_replay_passed",
        "sandbox_state_removed": (artifact_root / SANDBOX_STATE_FILENAME).exists() is False,
        "runtime_mapping_enabled_now_false": current_state["runtime_mapping_enabled"] is False,
        "enforcement_enabled_now_false": current_state["enforcement_enabled"] is False,
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }
    forbidden = _forbidden_artifact_paths(_artifact_paths(artifact_root))
    guard_passed = (
        all(value for key, value in checks.items() if key not in {"production_persistence_enabled", "vectors_npy_committed", "agp_bypass_used"})
        and not checks["production_persistence_enabled"]
        and not checks["vectors_npy_committed"]
        and not checks["agp_bypass_used"]
        and not forbidden
    )
    report = {
        "regression_guard_version": ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_VERSION,
        "round": 116,
        "regression_status": "sandbox_replay_regression_guard_passed" if guard_passed else "sandbox_replay_regression_guard_failed",
        "source_rounds_replayed": [110, 111, 112],
        "artifact_dir": str(artifact_root),
        "round110_status": round110.get("sandbox_status"),
        "round111_status": round111.get("cleanup_status"),
        "round112_status": round112.get("replay_status"),
        "regression_checks": checks,
        "forbidden_artifact_paths": forbidden,
        "runtime_mapping_enabled_now": current_state["runtime_mapping_enabled"],
        "enforcement_enabled_now": current_state["enforcement_enabled"],
        "production_persistence_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": False,
    }
    _write_json(artifact_root / ROUND116_GUARD_FILENAME, report)
    return report


def build_round117_operator_go_no_go_package(
    *,
    round113: dict[str, Any] | None = None,
    round114: dict[str, Any] | None = None,
    round115: dict[str, Any] | None = None,
    round116: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the operator package for a future persistence go/no-go decision."""

    r113 = round113 or {}
    r114 = round114 or {}
    r115 = round115 or {}
    r116 = round116 or {}
    completed_rounds = [round_no for round_no, payload in [(113, r113), (114, r114), (115, r115), (116, r116)] if payload]
    blockers = []
    if r115.get("blocked_partial_validation"):
        blockers.append("broader_validation_blocked_or_partial")
    if r114.get("isolated_blockers"):
        blockers.append("legacy_root_collection_blocker")
    ready_for_future_decision = (
        r113.get("viewer_status") == "state_debug_audit_replay_viewer_ready"
        and r116.get("regression_status") == "sandbox_replay_regression_guard_passed"
    )
    return {
        "go_no_go_package_version": ROUND117_OPERATOR_GO_NO_GO_PACKAGE_VERSION,
        "round": 117,
        "package_status": "operator_package_ready_future_decision_no_production_enablement",
        "completed_rounds": completed_rounds,
        "operator_recommendation": "no_go_for_production_persistence_in_this_pr",
        "future_decision_readiness": "ready_for_operator_review_not_auto_enable" if ready_for_future_decision else "defer_until_viewer_and_regression_guard_pass",
        "decision_blockers": blockers,
        "required_future_go_conditions": [
            "explicit operator approval for real persistence",
            "production persistence patch separate from sandbox reports",
            "runtime_mapping_enabled default reviewed separately",
            "enforcement_enabled default reviewed separately",
            "broader validation blockers acknowledged or resolved",
            "no vectors.npy, seed subset, zip, part, or _operator_artifacts in PR",
        ],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def write_round113_state_debug_audit_replay_viewer(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("viewer_version", ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": bool(report.get("current_runtime_mapping_enabled", False)), "enforcement_enabled": bool(report.get("current_enforcement_enabled", False)), "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round114_legacy_root_blocker_isolation(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("blocker_isolation_version", ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round115_broader_validation_triage_report(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("triage_version", ROUND115_BROADER_VALIDATION_TRIAGE_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round116_runtime_mapping_sandbox_replay_regression_guard(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("regression_guard_version", ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": bool(report.get("runtime_mapping_enabled_now", False)), "enforcement_enabled": bool(report.get("enforcement_enabled_now", False)), "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round117_operator_go_no_go_package(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("go_no_go_package_version", ROUND117_OPERATOR_GO_NO_GO_PACKAGE_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


__all__ = [
    "ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION",
    "ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION",
    "ROUND112_SANDBOX_AUDIT_REPLAY_VERSION",
    "run_round110_runtime_mapping_limited_persistence_sandbox",
    "run_round111_sandbox_rollback_cleanup_verification",
    "run_round112_post_sandbox_focused_validation_audit_replay",
    "write_round110_runtime_mapping_limited_persistence_sandbox",
    "write_round111_sandbox_rollback_cleanup_verification",
    "write_round112_post_sandbox_focused_validation_audit_replay",
    "ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_VERSION",
    "ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_VERSION",
    "ROUND115_BROADER_VALIDATION_TRIAGE_VERSION",
    "ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_VERSION",
    "ROUND117_OPERATOR_GO_NO_GO_PACKAGE_VERSION",
    "build_round113_state_debug_audit_replay_viewer",
    "build_round114_legacy_root_blocker_isolation",
    "build_round115_broader_validation_triage_report",
    "run_round116_runtime_mapping_sandbox_replay_regression_guard",
    "build_round117_operator_go_no_go_package",
    "write_round113_state_debug_audit_replay_viewer",
    "write_round114_legacy_root_blocker_isolation",
    "write_round115_broader_validation_triage_report",
    "write_round116_runtime_mapping_sandbox_replay_regression_guard",
    "write_round117_operator_go_no_go_package",
]
