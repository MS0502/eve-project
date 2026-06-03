"""Round106 runtime mapping persistence decision packet.

Round106 records the operator decision boundary.  It can report that persistence
is ready, but it deliberately does not mutate runtime mapping state; applying a
persistent mapping remains a separate explicit implementation patch.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROUND106_PERSISTENCE_DECISION_VERSION = "v3_round106_runtime_mapping_persistence_decision"


def build_runtime_mapping_persistence_decision(
    *,
    source_approval: dict[str, Any] | None = None,
    source_agp_proof: dict[str, Any] | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    approval = deepcopy(source_approval or {})
    proof = deepcopy(source_agp_proof or {})
    approval_ready = bool(approval.get("approval_ready", False))
    proof_ready = bool(proof.get("proof_ready", False))
    reasons: list[str] = []
    if not approval_ready:
        reasons.append("round104_approval_not_ready")
    if not proof_ready:
        reasons.append("round105_agp_proof_not_ready")
    if persist:
        reasons.append("persistence_application_deferred_to_separate_patch")

    decision_ready = bool(approval_ready and proof_ready)
    return {
        "decision_version": ROUND106_PERSISTENCE_DECISION_VERSION,
        "round": 106,
        "source_approval_version": approval.get("approval_version"),
        "source_agp_proof_version": proof.get("proof_version"),
        "operator_requested_persist": bool(persist),
        "decision_ready": decision_ready,
        "decision_status": "persistence_ready_but_not_applied" if decision_ready else "blocked_persistence_decision_incomplete",
        "blocking_reasons": sorted(set(reasons)),
        "mapped_tokens": sorted(str(t) for t in approval.get("mapped_tokens", []) or []),
        "proof_count": int(proof.get("proof_count", 0) or 0),
        "runtime_mapping_enabled_now": False,
        "enforcement_enabled_now": False,
        "runtime_mapping_persisted_now": False,
        "rollback_plan": {
            "persistent_state_changed_in_round106": False,
            "if_future_patch_persists_mapping_restore_runtime_mapping_enabled_false": True,
            "keep_operator_approval_and_agp_proof_as_audit_artifacts": True,
        },
        "policy": {
            "decision_packet_only": True,
            "does_not_apply_persistence": True,
            "requires_separate_apply_patch": True,
            "does_not_weaken_tests": True,
            "does_not_bypass_agp": True,
        },
        "read_only": True,
    }


def write_round106_runtime_mapping_persistence_decision(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("decision_version", ROUND106_PERSISTENCE_DECISION_VERSION),
        "path": str(out),
        "json_written": True,
        "runtime_mapping_persisted": False,
        "read_only": True,
    }


__all__ = ["ROUND106_PERSISTENCE_DECISION_VERSION", "build_runtime_mapping_persistence_decision", "write_round106_runtime_mapping_persistence_decision"]
