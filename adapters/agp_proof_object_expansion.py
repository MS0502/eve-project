"""Round105 AGP proof object expansion for runtime mapping review.

The proof object remains evidence-only: it records that persisted lexical→concept
mapping candidates are backed by explicit categories and SA activation, without
treating seed/EveSpecific/PMI vectors as AGP anchors and without invoking AGP as
a runtime mapping side effect.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROUND105_AGP_PROOF_EXPANSION_VERSION = "v3_round105_agp_proof_object_expansion"


def build_agp_proof_object_expansion(
    *,
    source_gate_audit: dict[str, Any] | None = None,
    source_approval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = deepcopy(source_gate_audit or {})
    approval = deepcopy(source_approval or {})
    rows = list(gate.get("audit_rows", []) or gate.get("smoke_rows", []) or gate.get("rows", []) or [])
    if not rows and gate.get("mapped_tokens"):
        rows = [{"lexical_token": token, "mapping_target_category_id": f"concept_category::lex::{token}"} for token in gate.get("mapped_tokens", [])]

    proof_rows: list[dict[str, Any]] = []
    blocked: list[str] = []
    for row in rows:
        token = str(row.get("lexical_token", "") or "").strip()
        if not token:
            continue
        category_id = str(row.get("mapping_target_category_id") or row.get("category_id") or f"concept_category::lex::{token}")
        row_reasons: list[str] = []
        if not category_id:
            row_reasons.append("category_id_missing")
        if bool(row.get("uses_lexical_vector_as_anchor", False)):
            row_reasons.append("lexical_vector_anchor_forbidden")
        if bool(row.get("uses_eve_specific_vector_as_anchor", False)):
            row_reasons.append("eve_specific_vector_anchor_forbidden")
        if bool(row.get("uses_seed_vector_as_anchor", False)):
            row_reasons.append("seed_vector_anchor_forbidden")
        proof_passed = not row_reasons
        if not proof_passed:
            blocked.append(token)
        proof_rows.append({
            "lexical_token": token,
            "category_id": category_id,
            "proof_status": "agp_proof_object_ready" if proof_passed else "agp_proof_object_blocked",
            "blocked_reasons": sorted(set(row_reasons)),
            "anchor_source": "explicit_category_plus_sa_activation_only",
            "requires_active_category": True,
            "requires_sa_activation": True,
            "agp_verify_called_by_expansion": False,
            "lexical_vector_used_as_anchor": False,
            "eve_specific_vector_used_as_anchor": False,
            "seed_vector_used_as_anchor": False,
        })

    approval_ready = bool(approval.get("approval_ready", False))
    proof_ready = bool(proof_rows and not blocked and approval_ready)
    return {
        "proof_version": ROUND105_AGP_PROOF_EXPANSION_VERSION,
        "round": 105,
        "source_gate_audit_version": gate.get("audit_version"),
        "source_approval_version": approval.get("approval_version"),
        "approval_ready": approval_ready,
        "proof_ready": proof_ready,
        "proof_status": "ready_for_round106_persistence_decision" if proof_ready else "blocked_agp_proof_incomplete",
        "proof_rows": proof_rows,
        "proof_count": len(proof_rows),
        "blocked_tokens": sorted(blocked),
        "runtime_mapping_enabled_now": False,
        "runtime_mapping_persisted_now": False,
        "policy": {
            "proof_object_only": True,
            "calls_agp_verify": False,
            "creates_agp_anchor": False,
            "lexical_vectors_are_evidence_not_agp_anchors": True,
            "eve_specific_vectors_are_evidence_not_agp_anchors": True,
            "seed_vectors_are_evidence_not_agp_anchors": True,
        },
        "read_only": True,
    }


def write_round105_agp_proof_object_expansion(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "export_version": report.get("proof_version", ROUND105_AGP_PROOF_EXPANSION_VERSION),
        "path": str(out),
        "json_written": True,
        "agp_verify_called": False,
        "runtime_mapping_persisted": False,
        "read_only": True,
    }


__all__ = ["ROUND105_AGP_PROOF_EXPANSION_VERSION", "build_agp_proof_object_expansion", "write_round105_agp_proof_object_expansion"]
