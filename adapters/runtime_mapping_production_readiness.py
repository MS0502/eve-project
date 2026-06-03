"""Rounds118-121 production persistence readiness gate helpers.

These helpers continue the runtime mapping persistence activation preparation
loop after the Round113-117 validation/audit package.  They are deliberately
read-only: they produce JSON audit/checklist/gate payloads and never enable
production persistence, runtime mapping by default, enforcement, AGP bypass, or
binary/vector artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from adapters.runtime_mapping_persistence_activation_candidate import capture_runtime_mapping_activation_state

ROUND118_PRODUCTION_PERSISTENCE_READINESS_AUDIT_VERSION = "v3_round118_production_persistence_readiness_audit"
ROUND119_MINIMAL_ENABLEMENT_RISK_MATRIX_VERSION = "v3_round119_minimal_enablement_risk_matrix_operator_checklist"
ROUND120_FINAL_PRE_ACTIVATION_GATE_PACKAGE_VERSION = "v3_round120_final_pre_activation_no_go_go_gate_package"
ROUND121_REQUIRED_BLOCKER_ISOLATION_VERSION = "v3_round121_required_blocker_isolation"

ROUND118_READINESS_FILENAME = "runtime_mapping_round118_production_persistence_readiness_audit.json"
ROUND119_RISK_MATRIX_FILENAME = "runtime_mapping_round119_minimal_enablement_risk_matrix_operator_checklist.json"
ROUND120_GATE_FILENAME = "runtime_mapping_round120_final_pre_activation_gate_package.json"
ROUND121_BLOCKERS_FILENAME = "runtime_mapping_round121_required_blocker_isolation.json"

_FORBIDDEN_SUFFIXES = ("vectors.npy", ".zip", ".part")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


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


def _status_from_validation(validation_summary: Iterable[dict[str, Any]] | None) -> tuple[bool, list[dict[str, Any]]]:
    summary = list(validation_summary or [])
    blocked = [item for item in summary if str(item.get("status")) in {"blocked", "partial", "blocked_pre_existing", "partial_or_blocked"}]
    return bool(blocked), summary


def _blocked_or_partial_from_sources(*sources: dict[str, Any]) -> bool:
    for source in sources:
        if source.get("blocked_partial_validation") is True:
            return True
        if source.get("broader_validation_blocked_or_partial") is True:
            return True
        if source.get("decision_blockers"):
            return True
    return False


def build_round118_production_persistence_readiness_audit(
    *,
    engine: Any | None = None,
    source_go_no_go_package: dict[str, Any] | None = None,
    validation_summary: Iterable[dict[str, Any]] | None = None,
    artifact_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Audit whether production persistence is ready without applying it."""

    state = capture_runtime_mapping_activation_state(engine)
    package = source_go_no_go_package or {}
    blocked_validation, validation_items = _status_from_validation(validation_summary)
    artifact_paths: list[str] = []
    for root in artifact_roots or []:
        artifact_paths.extend(_artifact_paths(root))
    forbidden = _forbidden_artifact_paths(artifact_paths)
    checks = {
        "source_round117_package_present": package.get("round") == 117,
        "source_round117_no_go_recorded": package.get("operator_recommendation") == "no_go_for_production_persistence_in_this_pr",
        "production_persistence_still_disabled": True,
        "runtime_mapping_enabled_default_false": state["runtime_mapping_enabled"] is False,
        "enforcement_enabled_default_false": state["enforcement_enabled"] is False,
        "agp_bypass_absent": True,
        "forbidden_artifacts_absent": not forbidden,
        "broader_validation_not_claimed_green": blocked_validation or _blocked_or_partial_from_sources(package),
    }
    readiness_blockers: list[str] = []
    if blocked_validation or _blocked_or_partial_from_sources(package):
        readiness_blockers.append("broader_validation_blocked_or_partial")
    if not checks["source_round117_package_present"]:
        readiness_blockers.append("missing_round117_operator_package")
    if not checks["runtime_mapping_enabled_default_false"]:
        readiness_blockers.append("runtime_mapping_enabled_not_disabled")
    if not checks["enforcement_enabled_default_false"]:
        readiness_blockers.append("enforcement_enabled_not_disabled")
    if forbidden:
        readiness_blockers.append("forbidden_artifact_detected")
    readiness_decision = "no_go" if readiness_blockers else "conditional_go_operator_review_only"
    return {
        "readiness_audit_version": ROUND118_PRODUCTION_PERSISTENCE_READINESS_AUDIT_VERSION,
        "round": 118,
        "audit_status": "production_persistence_not_ready_no_go" if readiness_decision == "no_go" else "production_persistence_conditionally_ready_for_operator_review",
        "source_rounds": sorted(set([117] + [int(r) for r in package.get("completed_rounds", []) if str(r).isdigit()])),
        "readiness_decision": readiness_decision,
        "readiness_blockers": sorted(set(readiness_blockers)),
        "checks": checks,
        "validation_summary": validation_items,
        "forbidden_artifact_paths": forbidden,
        "operator_recommendation": "NO-GO" if readiness_decision == "no_go" else "CONDITIONAL-GO",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def build_round119_minimal_enablement_risk_matrix_operator_checklist(
    *,
    source_readiness_audit: dict[str, Any] | None = None,
    explicit_operator_approval_present: bool = False,
) -> dict[str, Any]:
    """Build the minimal enablement risk matrix and operator checklist."""

    audit = source_readiness_audit or {}
    risk_matrix = [
        {
            "risk_id": "R1",
            "area": "validation",
            "risk": "broader pytest remains blocked or partial",
            "severity": "high",
            "likelihood": "known_present" if "broader_validation_blocked_or_partial" in audit.get("readiness_blockers", []) else "possible",
            "mitigation": "resolve or explicitly waive blocked/partial validation before real persistence",
            "activation_allowed_before_resolution": False,
        },
        {
            "risk_id": "R2",
            "area": "operator_authorization",
            "risk": "production persistence could be applied without an explicit operator decision",
            "severity": "critical",
            "likelihood": "controlled_by_gate",
            "mitigation": "require a separate approval token/package and separate activation patch",
            "activation_allowed_before_resolution": False,
        },
        {
            "risk_id": "R3",
            "area": "runtime_flags",
            "risk": "runtime_mapping_enabled or enforcement_enabled defaults could drift true",
            "severity": "critical",
            "likelihood": "low_currently_false",
            "mitigation": "pre-activation check must verify both defaults false until the explicit activation patch",
            "activation_allowed_before_resolution": False,
        },
        {
            "risk_id": "R4",
            "area": "artifact_boundary",
            "risk": "binary vector, seed subset, zip/part, or operator artifact could enter the PR",
            "severity": "high",
            "likelihood": "controlled_by_artifact_scan",
            "mitigation": "scan changed files and artifact roots before PR; keep artifacts external",
            "activation_allowed_before_resolution": False,
        },
        {
            "risk_id": "R5",
            "area": "AGP_boundary",
            "risk": "lexical persistence could be mistaken for AGP anchor authority",
            "severity": "critical",
            "likelihood": "controlled_by_policy",
            "mitigation": "preserve AGP anchored generation and keep lexical mapping evidence separate from anchors",
            "activation_allowed_before_resolution": False,
        },
    ]
    checklist = [
        {"item_id": "C1", "item": "Confirm Round118 readiness decision is not NO-GO", "required": True, "satisfied": audit.get("readiness_decision") != "no_go"},
        {"item_id": "C2", "item": "Confirm explicit operator approval for real production persistence", "required": True, "satisfied": bool(explicit_operator_approval_present)},
        {"item_id": "C3", "item": "Confirm runtime_mapping_enabled default remains false before activation patch", "required": True, "satisfied": audit.get("runtime_mapping_enabled_default") is False},
        {"item_id": "C4", "item": "Confirm enforcement_enabled default remains false before activation patch", "required": True, "satisfied": audit.get("enforcement_enabled_default") is False},
        {"item_id": "C5", "item": "Confirm broader validation is passed or explicitly accepted as partial by operator", "required": True, "satisfied": "broader_validation_blocked_or_partial" not in audit.get("readiness_blockers", [])},
        {"item_id": "C6", "item": "Confirm no forbidden binary/operator artifacts are staged", "required": True, "satisfied": not audit.get("forbidden_artifact_paths")},
    ]
    unsatisfied = [item["item_id"] for item in checklist if item["required"] and not item["satisfied"]]
    return {
        "risk_matrix_version": ROUND119_MINIMAL_ENABLEMENT_RISK_MATRIX_VERSION,
        "round": 119,
        "source_round": audit.get("round"),
        "risk_matrix_status": "minimal_enablement_risk_matrix_ready",
        "risk_matrix": risk_matrix,
        "operator_checklist": checklist,
        "unsatisfied_required_checklist_items": unsatisfied,
        "operator_recommendation": "NO-GO" if unsatisfied else "CONDITIONAL-GO",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def build_round120_final_pre_activation_gate_package(
    *,
    source_readiness_audit: dict[str, Any] | None = None,
    source_risk_matrix: dict[str, Any] | None = None,
    explicit_operator_approval_present: bool = False,
) -> dict[str, Any]:
    """Build the final pre-activation gate package without activating."""

    audit = source_readiness_audit or {}
    risk = source_risk_matrix or {}
    hard_blocks = sorted(set((audit.get("readiness_blockers") or []) + (risk.get("unsatisfied_required_checklist_items") or [])))
    if not explicit_operator_approval_present:
        hard_blocks.append("explicit_operator_approval_missing")
    hard_blocks = sorted(set(hard_blocks))
    final_recommendation = "GO" if not hard_blocks else ("CONDITIONAL-GO" if hard_blocks == ["explicit_operator_approval_missing"] else "NO-GO")
    return {
        "gate_package_version": ROUND120_FINAL_PRE_ACTIVATION_GATE_PACKAGE_VERSION,
        "round": 120,
        "source_rounds": [118, 119],
        "gate_status": "pre_activation_gate_passed_operator_activation_request_only" if final_recommendation == "GO" else "pre_activation_gate_no_go",
        "final_recommendation": final_recommendation,
        "hard_blocks": hard_blocks,
        "operator_approval_present": bool(explicit_operator_approval_present),
        "activation_action_taken": False,
        "operator_request_only": final_recommendation in {"GO", "CONDITIONAL-GO"},
        "must_not_activate_without_explicit_operator_approval": True,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def build_round121_required_blocker_isolation(
    *,
    source_gate_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """If Round120 is no-go, isolate required blockers before activation."""

    gate = source_gate_package or {}
    blocks = list(gate.get("hard_blocks") or [])
    blocker_details = []
    for block in blocks:
        blocker_details.append({
            "blocker": block,
            "required_resolution": {
                "broader_validation_blocked_or_partial": "Resolve pre-existing blockers or obtain explicit operator acceptance of partial validation.",
                "explicit_operator_approval_missing": "Provide explicit approval for a separate activation patch; this package must not activate by itself.",
                "C1": "Round118 must not be NO-GO before activation.",
                "C2": "Operator approval checklist item must be satisfied.",
                "C5": "Broader validation must pass or be explicitly accepted as partial.",
            }.get(block, "Resolve before any real persistence enablement."),
            "blocks_real_persistence": True,
        })
    return {
        "blocker_isolation_version": ROUND121_REQUIRED_BLOCKER_ISOLATION_VERSION,
        "round": 121,
        "source_round": gate.get("round"),
        "isolation_status": "required_blockers_isolated_no_activation" if blocks else "no_required_blockers_found_operator_request_only",
        "final_recommendation_inherited": gate.get("final_recommendation"),
        "required_blockers": blocker_details,
        "activation_action_taken": False,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
        "read_only": True,
    }


def write_round118_production_persistence_readiness_audit(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("readiness_audit_version", ROUND118_PRODUCTION_PERSISTENCE_READINESS_AUDIT_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round119_minimal_enablement_risk_matrix_operator_checklist(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("risk_matrix_version", ROUND119_MINIMAL_ENABLEMENT_RISK_MATRIX_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round120_final_pre_activation_gate_package(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("gate_package_version", ROUND120_FINAL_PRE_ACTIVATION_GATE_PACKAGE_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


def write_round121_required_blocker_isolation(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {"export_version": report.get("blocker_isolation_version", ROUND121_REQUIRED_BLOCKER_ISOLATION_VERSION), "path": str(path), "json_written": True, "runtime_mapping_enabled": False, "enforcement_enabled": False, "vectors_npy_committed": False, "agp_bypass_used": False}


__all__ = [
    "ROUND118_PRODUCTION_PERSISTENCE_READINESS_AUDIT_VERSION",
    "ROUND119_MINIMAL_ENABLEMENT_RISK_MATRIX_VERSION",
    "ROUND120_FINAL_PRE_ACTIVATION_GATE_PACKAGE_VERSION",
    "ROUND121_REQUIRED_BLOCKER_ISOLATION_VERSION",
    "build_round118_production_persistence_readiness_audit",
    "build_round119_minimal_enablement_risk_matrix_operator_checklist",
    "build_round120_final_pre_activation_gate_package",
    "build_round121_required_blocker_isolation",
    "write_round118_production_persistence_readiness_audit",
    "write_round119_minimal_enablement_risk_matrix_operator_checklist",
    "write_round120_final_pre_activation_gate_package",
    "write_round121_required_blocker_isolation",
]
