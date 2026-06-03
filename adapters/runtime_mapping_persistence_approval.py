"""Round104 runtime mapping persistence approval gate.

Round104 consumes operator-reported medium-vector validation results after the
manual artifact install.  It does not persist runtime mapping; it produces a
reviewable approval gate that remains separate from any later mutation patch.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROUND104_PERSISTENCE_APPROVAL_VERSION = "v3_round104_runtime_mapping_persistence_approval_gate"

REQUIRED_OPERATOR_COMMANDS: tuple[str, ...] = (
    "python -m compileall -q adapters tests main.py",
    "pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py",
    "pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py",
)


def round103_operator_unblocked_validation_status() -> dict[str, Any]:
    """Return the operator-reported Round103 unblocked validation record.

    This is not a local Codex execution claim.  It records the operator-provided
    Codespaces results that unblock the persistence approval gate while keeping
    the binary vector artifact outside the PR diff.
    """
    return {
        "version": "v3_round103_operator_unblocked_validation_status",
        "round": 103,
        "source": "operator_reported_codespaces_validation",
        "local_codex_execution": False,
        "artifact_state": {
            "medium_vector_artifact_manually_restored": True,
            "vectors_npy_gitignored": True,
            "operator_artifacts_temporary": True,
            "binary_artifact_committed": False,
            "dummy_vector_created": False,
        },
        "commands": [
            {
                "command": "pytest -q tests/test_v3_round97_98_runtime_mapping_enable_smoke.py",
                "result": "passed",
                "details": "operator reported: 3 passed",
            },
            {
                "command": "pytest -q tests/test_v3_round92_runtime_mapping_gate_dry_run.py tests/test_v3_round93_runtime_mapping_proposal_report.py tests/test_v3_round94_runtime_mapping_enforcement_dry_run.py tests/test_v3_round95_runtime_mapping_operator_acceptance_fixture.py tests/test_v3_round96_runtime_mapping_enable_smoke_precheck.py tests/test_v3_round97_98_runtime_mapping_enable_smoke.py",
                "result": "passed",
                "details": "operator reported: 14 passed",
            },
            {
                "command": "python -m compileall -q adapters tests main.py",
                "result": "passed",
                "details": "operator reported: passed",
            },
        ],
        "focused_runtime_mapping_validation_unblocked": True,
        "hard_stop_released": True,
    }


def _commands_passed(operator_validation: dict[str, Any]) -> dict[str, Any]:
    commands = list(operator_validation.get("commands", []) or [])
    by_command = {str(row.get("command", "")): row for row in commands}
    missing = [command for command in REQUIRED_OPERATOR_COMMANDS if command not in by_command]
    failed = [
        str(row.get("command", ""))
        for row in commands
        if str(row.get("command", "")) in REQUIRED_OPERATOR_COMMANDS and row.get("result") != "passed"
    ]
    return {
        "required_commands": list(REQUIRED_OPERATOR_COMMANDS),
        "missing_required_commands": missing,
        "failed_required_commands": failed,
        "all_required_commands_passed": not missing and not failed,
    }


def runtime_mapping_persistence_approval_gate(
    operator_validation: dict[str, Any] | None = None,
    *,
    source_round98_audit: dict[str, Any] | None = None,
    operator_approved: bool = False,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Build a read-only runtime mapping persistence approval gate.

    Passing this gate means the project may ask the operator for an explicit
    future persistence mutation patch.  This function never enables runtime
    mapping and never writes mapping state.
    """
    validation = deepcopy(operator_validation) if operator_validation is not None else round103_operator_unblocked_validation_status()
    command_audit = _commands_passed(validation)
    artifact_state = dict(validation.get("artifact_state", {}) or {})
    audit = deepcopy(source_round98_audit) if isinstance(source_round98_audit, dict) else None
    round98_ready = True if audit is None else audit.get("persistence_gate_status") == "ready_for_operator_persistence_decision"
    hard_stop_reasons: list[str] = []
    if not bool(validation.get("hard_stop_released", False)):
        hard_stop_reasons.append("round103_hard_stop_not_released")
    if not command_audit["all_required_commands_passed"]:
        hard_stop_reasons.append("operator_validation_commands_missing_or_failed")
    if not bool(artifact_state.get("vectors_npy_gitignored", False)):
        hard_stop_reasons.append("vectors_npy_gitignore_not_confirmed")
    if bool(artifact_state.get("binary_artifact_committed", False)):
        hard_stop_reasons.append("binary_artifact_committed")
    if bool(artifact_state.get("dummy_vector_created", False)):
        hard_stop_reasons.append("dummy_vector_created")
    if not round98_ready:
        hard_stop_reasons.append("round98_persistence_gate_not_ready")

    validation_ready = not hard_stop_reasons
    status = "ready_for_explicit_operator_persistence_approval" if validation_ready else "blocked_before_persistence_approval"
    if validation_ready and operator_approved:
        status = "operator_approved_for_future_persistence_patch"
    return {
        "version": ROUND104_PERSISTENCE_APPROVAL_VERSION,
        "round": 104,
        "status": status,
        "hard_stop": bool(hard_stop_reasons),
        "hard_stop_reasons": hard_stop_reasons,
        "operator_validation": validation,
        "command_audit": command_audit,
        "source_round98_audit_version": audit.get("audit_version") if audit else None,
        "operator_approval": {
            "operator_approved": bool(operator_approved),
            "approval_id": approval_id,
            "approval_required_before_mutation": True,
        },
        "persistence_decision": {
            "runtime_mapping_persisted_now": False,
            "runtime_mapping_enabled_default_after_gate": False,
            "enforcement_enabled_after_gate": False,
            "may_prepare_future_persistence_patch": bool(validation_ready and operator_approved),
            "may_proceed_to_agp_proof_expansion": bool(validation_ready),
        },
        "policy": {
            "approval_gate_only": True,
            "no_runtime_mapping_mutation": True,
            "no_enforcement_enablement": True,
            "no_agp_bypass": True,
            "lexical_vectors_are_evidence_not_agp_anchors": True,
            "vectors_npy_must_remain_out_of_pr_diff": True,
        },
        "read_only": True,
    }


def write_round104_persistence_approval_status(path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deepcopy(data), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"version": ROUND104_PERSISTENCE_APPROVAL_VERSION, "path": str(out), "json_written": True, "mutation_written": False}


__all__ = [
    "REQUIRED_OPERATOR_COMMANDS",
    "ROUND104_PERSISTENCE_APPROVAL_VERSION",
    "round103_operator_unblocked_validation_status",
    "runtime_mapping_persistence_approval_gate",
    "write_round104_persistence_approval_status",
]
