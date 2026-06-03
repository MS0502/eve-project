"""Round104 runtime mapping persistence approval packet.

This module turns prior Round98/103 evidence into an operator-facing approval
packet.  It does not enable runtime mapping, persist mappings, call AGP verify,
or mutate concept/category/vector state.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROUND104_PERSISTENCE_APPROVAL_VERSION = "v3_round104_runtime_mapping_persistence_approval"


def build_runtime_mapping_persistence_approval(
    *,
    source_gate_audit: dict[str, Any] | None = None,
    source_manual_validation: dict[str, Any] | None = None,
    operator_approved: bool = False,
) -> dict[str, Any]:
    gate = deepcopy(source_gate_audit or {})
    manual = deepcopy(source_manual_validation or {})
    ready_gate = str(gate.get("persistence_gate_status", "")) == "ready_for_operator_persistence_decision"
    manual_ok = bool(manual.get("validation_passed", False))
    mapped_count = int(gate.get("mapped_count", 0) or 0)
    reasons: list[str] = []
    if not ready_gate:
        reasons.append("round98_persistence_gate_not_ready")
    if not manual_ok:
        reasons.append("round103_manual_medium_validation_not_passed")
    if mapped_count <= 0:
        reasons.append("no_runtime_mapping_rows_to_persist")
    if not operator_approved:
        reasons.append("operator_approval_not_granted")

    approval_ready = bool(ready_gate and manual_ok and mapped_count > 0 and operator_approved)
    return {
        "approval_version": ROUND104_PERSISTENCE_APPROVAL_VERSION,
        "round": 104,
        "source_gate_audit_version": gate.get("audit_version"),
        "source_manual_validation_version": manual.get("validation_version"),
        "operator_approved": bool(operator_approved),
        "mapped_count": mapped_count,
        "mapped_tokens": sorted(str(t) for t in gate.get("mapped_tokens", []) or []),
        "approval_ready": approval_ready,
        "approval_status": "approved_for_round106_decision_review" if approval_ready else "blocked_pending_operator_approval_or_validation",
        "blocking_reasons": sorted(set(reasons)),
        "runtime_mapping_enabled_now": False,
        "enforcement_enabled_now": False,
        "runtime_mapping_persisted_now": False,
        "required_followup": "round105_agp_proof_object_expansion" if approval_ready else "complete_gate_manual_validation_and_operator_approval",
        "policy": {
            "approval_packet_only": True,
            "mutates_runtime_mapping": False,
            "mutates_concept_memory": False,
            "mutates_vectors": False,
            "calls_agp_verify": False,
            "lexical_vectors_are_evidence_not_agp_anchors": True,
        },
        "read_only": True,
    }


def write_round104_runtime_mapping_persistence_approval(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("approval_version", ROUND104_PERSISTENCE_APPROVAL_VERSION),
        "path": str(out),
        "json_written": True,
        "runtime_mapping_persisted": False,
        "read_only": True,
    }


__all__ = ["ROUND104_PERSISTENCE_APPROVAL_VERSION", "build_runtime_mapping_persistence_approval", "write_round104_runtime_mapping_persistence_approval"]
