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
]
