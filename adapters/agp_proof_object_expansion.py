"""Round105 AGP proof object expansion.

The proof object is data-only.  It expands the evidence chain after Round104
without calling AGP.verify, creating anchors, or changing runtime mapping.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from adapters.runtime_mapping_persistence_approval import runtime_mapping_persistence_approval_gate

ROUND105_AGP_PROOF_EXPANSION_VERSION = "v3_round105_agp_proof_object_expansion"


def expand_agp_proof_object(source_round104_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    gate = deepcopy(source_round104_gate) if source_round104_gate is not None else runtime_mapping_persistence_approval_gate()
    gate_ready = not bool(gate.get("hard_stop", True)) and bool(gate.get("persistence_decision", {}).get("may_proceed_to_agp_proof_expansion", False))
    proof_steps = [
        {
            "step": "medium_vector_artifact_validation",
            "status": "passed_operator_reported" if gate_ready else "blocked_or_missing",
            "vectors_as_agp_anchor": False,
        },
        {
            "step": "runtime_mapping_smoke_validation",
            "status": "passed_operator_reported" if gate_ready else "blocked_or_missing",
            "runtime_mapping_persisted": False,
        },
        {
            "step": "agp_anchor_boundary",
            "status": "expanded_data_only" if gate_ready else "blocked_or_missing",
            "valid_anchor_source": "explicit_category_plus_sa_activation_only",
            "invalid_anchor_sources": ["fastText_vector", "EveSpecific_vector", "PMI_SVD_vector", "raw_response_text"],
        },
    ]
    return {
        "version": ROUND105_AGP_PROOF_EXPANSION_VERSION,
        "round": 105,
        "status": "agp_proof_object_expanded" if gate_ready else "blocked_by_round104_gate",
        "hard_stop": not gate_ready,
        "source_round104_status": gate.get("status"),
        "proof_object": {
            "proof_id": "round105_runtime_mapping_agp_anchor_boundary_proof",
            "proof_steps": proof_steps,
            "required_for_future_agp_use": [
                "explicit_generation_time_meaning_categories",
                "matching_sa_activation_category",
                "hormone_compatible_agp_result",
                "no_vector_as_anchor_shortcut",
            ],
            "runtime_mapping_role": "candidate_category_lookup_only_after_explicit_persistence_patch",
            "agp_role": "verify_explicit_category_against_active_sa_and_hormone_context",
        },
        "mutation_checks": {
            "agp_verify_called": False,
            "agp_anchor_created": False,
            "runtime_mapping_persisted": False,
            "enforcement_enabled": False,
            "category_created": False,
            "concept_memory_mutated": False,
            "vectors_committed": False,
        },
        "policy": {
            "proof_object_only": True,
            "agp_bypass": False,
            "vectors_are_evidence_not_anchors": True,
            "explicit_category_and_sa_required": True,
        },
        "read_only": True,
    }


def write_round105_agp_proof_status(path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deepcopy(data), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"version": ROUND105_AGP_PROOF_EXPANSION_VERSION, "path": str(out), "json_written": True, "mutation_written": False}


__all__ = ["ROUND105_AGP_PROOF_EXPANSION_VERSION", "expand_agp_proof_object", "write_round105_agp_proof_status"]
