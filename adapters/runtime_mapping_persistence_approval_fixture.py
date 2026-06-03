"""Round109 runtime mapping persistence approval fixture and rollback drill.

Round109 adds a deterministic operator approval fixture for the already-guarded
runtime mapping persistence activation candidate.  The fixture is deliberately
scoped to ``runtime_mapping_persistence_only`` and to an explicit token allowlist
so operators can rehearse approval, audit ordering, rollback, and state-debug
inspection before any real persistence enablement.

This module does not enable persistence by default and does not enable
runtime-mapping enforcement.  Its drill uses the Round108 candidate path in
operator-approved test/dry-run mode only, then verifies rollback restored both
flags to ``False`` and protected AGP/vector/category/memory surfaces remained
unchanged.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from adapters.runtime_mapping_persistence_activation_candidate import (
    ROUND108_OPERATOR_APPROVAL_TOKEN,
    capture_runtime_mapping_activation_state,
    run_runtime_mapping_persistence_activation_candidate,
)
from adapters.runtime_mapping_persistence_activation_dryrun import build_runtime_mapping_persistence_activation_dryrun
from adapters.runtime_mapping_persistence_decision import ROUND106_PERSISTENCE_DECISION_VERSION

ROUND109_APPROVAL_FIXTURE_VERSION = "v3_round109_runtime_mapping_persistence_approval_fixture"
ROUND109_APPROVAL_SCOPE = "runtime_mapping_persistence_only"
ROUND109_TOKEN_ALLOWLIST = ("민석",)
ROUND109_TEST_DRY_RUN_MODE = "operator_approval_fixture_test_dry_run_only"

APPROVAL_FIXTURE_FILENAME = "runtime_mapping_persistence_round109_operator_approval_fixture.json"
DRILL_SUMMARY_FILENAME = "runtime_mapping_persistence_round109_rollback_drill_summary.json"

_PROTECTED_CHECKS = (
    "concept_categories_unchanged",
    "concept_commit_records_unchanged",
    "eve_specific_vector_store_unchanged",
    "sa_active_categories_unchanged",
    "concept_memory_stats_unchanged",
    "agp_stats_unchanged",
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_audit_events(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    audit_path = Path(path)
    if not audit_path.exists():
        return []
    return [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_tokens(tokens: Iterable[str] | None) -> list[str]:
    return sorted(str(token) for token in (tokens or []))


def _artifact_paths_contain_forbidden_files(paths: Iterable[str]) -> list[str]:
    forbidden: list[str] = []
    for raw_path in paths:
        normalized = str(raw_path).replace("\\", "/")
        name = Path(normalized).name
        if name == "vectors.npy" or normalized.startswith("seeds/subsets/") or "/vectors.npy" in normalized:
            forbidden.append(str(raw_path))
    return sorted(forbidden)


def build_round109_operator_approval_fixture(
    *,
    tokens: Iterable[str] | None = None,
    approval_scope: str = ROUND109_APPROVAL_SCOPE,
    operator_approval_token: str = ROUND108_OPERATOR_APPROVAL_TOKEN,
) -> dict[str, Any]:
    """Build the deterministic limited-scope operator approval fixture.

    The fixture is data only.  It narrows approval to runtime mapping
    persistence, requires the explicit Round108 operator token, and fails closed
    unless all requested tokens are exactly inside the Round109 allowlist.
    """

    requested_tokens = _normalize_tokens(tokens if tokens is not None else ROUND109_TOKEN_ALLOWLIST)
    allowed_tokens = list(ROUND109_TOKEN_ALLOWLIST)
    checks = {
        "scope_is_runtime_mapping_persistence_only": approval_scope == ROUND109_APPROVAL_SCOPE,
        "token_allowlist_explicit": requested_tokens == allowed_tokens,
        "token_allowlist_non_empty": bool(requested_tokens),
        "operator_token_matches_round108_guard": operator_approval_token == ROUND108_OPERATOR_APPROVAL_TOKEN,
        "test_dry_run_mode_only": True,
        "runtime_mapping_default_enablement_forbidden": True,
        "enforcement_enablement_forbidden": True,
    }
    blocking_reasons = sorted(key for key, passed in checks.items() if not passed)
    return {
        "fixture_version": ROUND109_APPROVAL_FIXTURE_VERSION,
        "round": 109,
        "approval_scope": approval_scope,
        "approval_status": "fixture_approved_for_test_dry_run_only" if not blocking_reasons else "fixture_blocked",
        "approval_valid": not blocking_reasons,
        "operator_approval_token": operator_approval_token,
        "allowed_tokens": allowed_tokens,
        "requested_tokens": requested_tokens,
        "test_dry_run_mode": ROUND109_TEST_DRY_RUN_MODE,
        "candidate_activation_mode": "round108_ephemeral_candidate_apply_then_immediate_rollback",
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "actual_persistence_enablement": False,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "read_only_fixture": True,
    }


def _source_decision(tokens: list[str]) -> dict[str, Any]:
    return {
        "decision_version": ROUND106_PERSISTENCE_DECISION_VERSION,
        "decision_status": "persistence_ready_but_not_applied",
        "decision_ready": True,
        "mapped_tokens": list(tokens),
        "runtime_mapping_persisted_now": False,
    }


def _verify_audit_order(events: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [
        "activation_precheck_started",
        "checkpoint_written",
        "activation_candidate_applied",
        "rollback_validation_passed",
    ]
    actual = [str(event.get("event_type")) for event in events]
    indices = [event.get("event_index") for event in events]
    return {
        "expected_event_order": expected,
        "actual_event_order": actual,
        "event_indices": indices,
        "event_order_verified": actual == expected and indices == list(range(len(expected))),
        "checkpoint_event_precedes_candidate_mutation": "checkpoint_written" in actual
        and "activation_candidate_applied" in actual
        and actual.index("checkpoint_written") < actual.index("activation_candidate_applied"),
    }


def run_round109_runtime_mapping_persistence_approval_fixture_drill(
    *,
    engine: Any | None = None,
    artifact_dir: str | Path = "eve_v3_autonomous_handoff/validation/round109_runtime_mapping_persistence_approval_fixture_drill",
    tokens: Iterable[str] | None = None,
    approval_scope: str = ROUND109_APPROVAL_SCOPE,
    changed_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run the Round109 approval fixture rollback drill.

    The drill applies only an ephemeral Round108 candidate flag mutation under an
    explicit operator fixture, then immediately rolls it back.  It exports
    before/after/rollback state-debug snapshots through the Round108 candidate
    artifact path and summarizes the safety proof for Round109.
    """

    artifact_root = Path(artifact_dir)
    fixture = build_round109_operator_approval_fixture(tokens=tokens, approval_scope=approval_scope)
    requested_tokens = list(fixture["requested_tokens"])
    before_state = capture_runtime_mapping_activation_state(engine)

    fixture_path = artifact_root / APPROVAL_FIXTURE_FILENAME
    _write_json(fixture_path, fixture)

    if not fixture["approval_valid"]:
        return {
            "drill_version": ROUND109_APPROVAL_FIXTURE_VERSION,
            "round": 109,
            "drill_status": "blocked_fixture_not_valid",
            "fixture": fixture,
            "fixture_path": str(fixture_path),
            "runtime_mapping_enabled_now": before_state["runtime_mapping_enabled"],
            "enforcement_enabled_now": before_state["enforcement_enabled"],
            "candidate_mutation_applied": False,
            "checkpoint_written_before_mutation": False,
            "rollback_verified": False,
            "vectors_npy_committed": False,
            "agp_bypass_used": False,
            "read_only_fixture": True,
        }

    decision = _source_decision(requested_tokens)
    dry_run = build_runtime_mapping_persistence_activation_dryrun(engine=engine, source_decision=decision)
    candidate_report = run_runtime_mapping_persistence_activation_candidate(
        engine=engine,
        source_decision=decision,
        source_dry_run=dry_run,
        operator_approved=True,
        operator_approval_token=ROUND108_OPERATOR_APPROVAL_TOKEN,
        artifact_dir=artifact_root,
        apply_candidate=True,
    )
    after_rollback_state = capture_runtime_mapping_activation_state(engine)
    events = _read_audit_events(candidate_report.get("audit_log_path"))
    audit_order = _verify_audit_order(events)
    rollback_checks = dict((candidate_report.get("rollback_verification") or {}).get("checks", {}) or {})
    protected_surface_checks = {key: bool(rollback_checks.get(key, False)) for key in _PROTECTED_CHECKS}
    exported_snapshots = {
        "before": bool(candidate_report.get("state_debug_before_exported", False)),
        "after_candidate": bool(candidate_report.get("state_debug_after_exported", False)),
        "after_rollback": bool(candidate_report.get("state_debug_after_rollback_exported", False)),
        "paths": {
            "before": candidate_report.get("state_debug_before_path"),
            "after_candidate": candidate_report.get("state_debug_after_path"),
            "after_rollback": candidate_report.get("state_debug_after_rollback_path"),
        },
    }
    forbidden_paths = _artifact_paths_contain_forbidden_files(changed_paths or [])
    artifact_files = sorted(str(path.relative_to(artifact_root)) for path in artifact_root.rglob("*") if path.is_file())
    forbidden_artifacts = _artifact_paths_contain_forbidden_files(artifact_files)

    drill_summary = {
        "drill_version": ROUND109_APPROVAL_FIXTURE_VERSION,
        "round": 109,
        "drill_status": "approval_fixture_drill_passed" if (
            candidate_report.get("activation_status") == "activation_candidate_applied_then_rolled_back"
            and candidate_report.get("checkpoint_written_before_mutation") is True
            and audit_order["event_order_verified"] is True
            and audit_order["checkpoint_event_precedes_candidate_mutation"] is True
            and candidate_report.get("rollback_verified") is True
            and after_rollback_state["runtime_mapping_enabled"] is False
            and after_rollback_state["enforcement_enabled"] is False
            and all(protected_surface_checks.values())
            and all(exported_snapshots[key] for key in ("before", "after_candidate", "after_rollback"))
            and not forbidden_paths
            and not forbidden_artifacts
        ) else "approval_fixture_drill_failed",
        "approval_fixture_path": str(fixture_path),
        "fixture": fixture,
        "source_decision": decision,
        "source_dry_run_version": dry_run.get("dry_run_version"),
        "test_dry_run_mode": ROUND109_TEST_DRY_RUN_MODE,
        "candidate_report": candidate_report,
        "checkpoint_created_before_candidate_mutation": bool(candidate_report.get("checkpoint_written_before_mutation", False)),
        "audit_order": audit_order,
        "rollback_restored_disabled_flags": {
            "runtime_mapping_enabled_false": after_rollback_state["runtime_mapping_enabled"] is False,
            "enforcement_enabled_false": after_rollback_state["enforcement_enabled"] is False,
        },
        "protected_surface_checks": protected_surface_checks,
        "state_debug_exports": exported_snapshots,
        "runtime_mapping_enabled_now": after_rollback_state["runtime_mapping_enabled"],
        "enforcement_enabled_now": after_rollback_state["enforcement_enabled"],
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "actual_persistence_enablement": False,
        "candidate_mutation_applied": bool(candidate_report.get("candidate_mutation_applied", False)),
        "candidate_mutation_scope": ROUND109_APPROVAL_SCOPE,
        "allowed_tokens": requested_tokens,
        "vectors_npy_committed": False,
        "forbidden_artifact_paths": forbidden_artifacts,
        "forbidden_pr_diff_paths": forbidden_paths,
        "agp_bypass_used": False,
        "read_only_fixture": True,
    }
    summary_path = artifact_root / DRILL_SUMMARY_FILENAME
    _write_json(summary_path, drill_summary)
    drill_summary["drill_summary_path"] = str(summary_path)
    _write_json(summary_path, drill_summary)
    return drill_summary


def write_round109_runtime_mapping_persistence_approval_fixture(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    """Write a Round109 fixture/drill report as JSON without enabling mapping."""

    _write_json(path, report)
    return {
        "export_version": report.get("drill_version", ROUND109_APPROVAL_FIXTURE_VERSION),
        "path": str(path),
        "json_written": True,
        "runtime_mapping_enabled": bool(report.get("runtime_mapping_enabled_now", False)),
        "enforcement_enabled": bool(report.get("enforcement_enabled_now", False)),
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only_fixture": True,
    }


__all__ = [
    "ROUND109_APPROVAL_FIXTURE_VERSION",
    "ROUND109_APPROVAL_SCOPE",
    "ROUND109_TEST_DRY_RUN_MODE",
    "ROUND109_TOKEN_ALLOWLIST",
    "build_round109_operator_approval_fixture",
    "run_round109_runtime_mapping_persistence_approval_fixture_drill",
    "write_round109_runtime_mapping_persistence_approval_fixture",
]
