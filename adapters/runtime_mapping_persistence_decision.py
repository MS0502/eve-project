"""Round106 runtime mapping persistence decision package.

Round106 is an operator decision package only.  It structures the prerequisites,
approval schema, checkpoint/rollback/audit/state-debug requirements, and a dry-run
for a future persistence activation patch.  It never enables runtime mapping,
never enables enforcement, and never mutates AGP/vector/category/memory state.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from adapters.runtime_mapping_persistence_approval import runtime_mapping_persistence_approval_gate

ROUND106_PERSISTENCE_DECISION_VERSION = "v3_round106_runtime_mapping_persistence_decision_package"


def default_operator_approval_schema() -> dict[str, Any]:
    """Return the explicit schema required before any future persistence patch."""
    return {
        "schema_version": ROUND106_PERSISTENCE_DECISION_VERSION,
        "required_fields": {
            "operator_id": "non_empty_string",
            "approval_id": "non_empty_string",
            "approved_at_utc": "iso8601_string",
            "scope": "runtime_mapping_persistence_only",
            "approved_tokens": "non_empty_list_of_committed_runtime_mapping_tokens",
            "validation_status": "round103_104_105_validation_reviewed",
            "rollback_acknowledged": True,
            "binary_artifact_boundary_acknowledged": True,
            "agp_anchor_boundary_acknowledged": True,
        },
        "forbidden_fields": {
            "enable_enforcement": True,
            "bypass_agp": True,
            "commit_vectors_npy": True,
            "weaken_tests": True,
        },
        "approval_required_before_mutation": True,
    }


def runtime_mapping_persistence_activation_prerequisites(
    source_round104_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return read-only prerequisites for a future runtime-mapping activation."""
    gate = deepcopy(source_round104_gate) if source_round104_gate is not None else runtime_mapping_persistence_approval_gate()
    gate_ready = not bool(gate.get("hard_stop", True))
    command_audit = dict(gate.get("command_audit", {}) or {})
    operator_validation = dict(gate.get("operator_validation", {}) or {})
    artifact_state = dict(operator_validation.get("artifact_state", {}) or {})
    prerequisites = [
        {
            "id": "medium_artifact_validated",
            "description": "medium vector artifact manually installed and validation unblocked",
            "satisfied": bool(operator_validation.get("hard_stop_released", False)),
        },
        {
            "id": "focused_validation_passed",
            "description": "Round97/98 and Round92~98 focused validation passed",
            "satisfied": bool(command_audit.get("all_required_commands_passed", False)),
        },
        {
            "id": "binary_artifact_boundary_preserved",
            "description": "vectors.npy is ignored and no binary artifact is committed",
            "satisfied": bool(artifact_state.get("vectors_npy_gitignored", False))
            and not bool(artifact_state.get("binary_artifact_committed", False))
            and not bool(artifact_state.get("dummy_vector_created", False)),
        },
        {
            "id": "round104_gate_ready",
            "description": "Round104 approval gate has no hard stop",
            "satisfied": gate_ready,
        },
        {
            "id": "operator_explicit_approval_pending",
            "description": "operator approval is still required before any persistence mutation",
            "satisfied": False,
            "intentionally_pending": True,
        },
    ]
    return {
        "version": ROUND106_PERSISTENCE_DECISION_VERSION,
        "round": 106,
        "source_round104_status": gate.get("status"),
        "all_non_approval_prerequisites_satisfied": all(
            row["satisfied"] for row in prerequisites if not row.get("intentionally_pending", False)
        ),
        "operator_approval_pending": True,
        "prerequisites": prerequisites,
        "read_only": True,
    }


def checkpoint_rollback_audit_requirements() -> dict[str, Any]:
    """Return required controls for a future activation patch."""
    return {
        "version": ROUND106_PERSISTENCE_DECISION_VERSION,
        "checkpoint_requirements": [
            "snapshot runtime_mapping_enabled and enforcement_enabled before activation",
            "snapshot ephemeral/runtime mapping table before activation",
            "snapshot concept category records and concept commit audit records",
            "snapshot state-debug persistence surface before activation",
        ],
        "rollback_requirements": [
            "restore runtime_mapping_enabled to false",
            "restore enforcement_enabled to false",
            "clear or restore persisted runtime mapping table",
            "verify AGP/category/memory/vector snapshots are unchanged",
        ],
        "audit_requirements": [
            "record approval_id and operator_id",
            "record exact approved token/category mappings",
            "record pre/post/rollback flag states",
            "record validation command summaries",
            "record binary artifact boundary confirmation",
        ],
        "state_debug_requirements": [
            "surface persistence approval status",
            "surface runtime_mapping_enabled default state",
            "surface enforcement_enabled default state",
            "surface checkpoint id and rollback availability",
            "surface mutation counters as zero until activation patch",
        ],
        "required_before_activation": True,
    }


def runtime_mapping_persistence_activation_dry_run(
    source_round104_gate: dict[str, Any] | None = None,
    *,
    operator_approval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a dry-run decision package without activating persistence."""
    prereq = runtime_mapping_persistence_activation_prerequisites(source_round104_gate)
    controls = checkpoint_rollback_audit_requirements()
    approval_schema = default_operator_approval_schema()
    approval = deepcopy(operator_approval) if isinstance(operator_approval, dict) else None
    approval_present = bool(approval)
    approval_valid = False
    approval_missing_fields: list[str] = []
    if approval_present:
        required_keys = list(approval_schema["required_fields"].keys())
        approval_missing_fields = [key for key in required_keys if key not in approval or approval.get(key) in (None, "", [])]
        approval_valid = not approval_missing_fields
    non_approval_ready = bool(prereq["all_non_approval_prerequisites_satisfied"])
    would_allow_future_activation_patch = bool(non_approval_ready and approval_valid)
    return {
        "version": ROUND106_PERSISTENCE_DECISION_VERSION,
        "round": 106,
        "decision_status": (
            "future_activation_patch_ready_after_operator_approval"
            if would_allow_future_activation_patch
            else "awaiting_explicit_operator_approval_or_prerequisite_repair"
        ),
        "activation_dry_run_only": True,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "runtime_mapping_persisted_now": False,
        "prerequisites": prereq,
        "checkpoint_rollback_audit_requirements": controls,
        "operator_approval_schema": approval_schema,
        "operator_approval_supplied": approval_present,
        "operator_approval_valid": approval_valid,
        "operator_approval_missing_fields": approval_missing_fields,
        "would_allow_future_activation_patch": would_allow_future_activation_patch,
        "mutation_checks": {
            "runtime_mapping_enabled_changed": False,
            "enforcement_enabled_changed": False,
            "agp_verify_called": False,
            "agp_bypassed": False,
            "category_mutated": False,
            "concept_memory_mutated": False,
            "vector_committed": False,
            "vectors_npy_committed": False,
        },
        "policy": {
            "dry_run_only": True,
            "persistence_actual_activation": False,
            "tests_weakened": False,
            "agp_bypass": False,
            "undocumented_mutation": False,
        },
        "read_only": True,
    }


def write_round106_persistence_decision_status(path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deepcopy(data), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"version": ROUND106_PERSISTENCE_DECISION_VERSION, "path": str(out), "json_written": True, "mutation_written": False}


__all__ = [
    "ROUND106_PERSISTENCE_DECISION_VERSION",
    "checkpoint_rollback_audit_requirements",
    "default_operator_approval_schema",
    "runtime_mapping_persistence_activation_dry_run",
    "runtime_mapping_persistence_activation_prerequisites",
    "write_round106_persistence_decision_status",
]
