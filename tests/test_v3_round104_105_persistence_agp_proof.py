"""EVE v3 Rounds104~105 — persistence approval and AGP proof expansion."""

from pathlib import Path

from adapters.agp_proof_object_expansion import (
    ROUND105_AGP_PROOF_EXPANSION_VERSION,
    build_agp_proof_object_expansion,
    write_round105_agp_proof_object_expansion,
)
from adapters.runtime_mapping_persistence_approval import (
    ROUND104_PERSISTENCE_APPROVAL_VERSION,
    build_runtime_mapping_persistence_approval,
    write_round104_runtime_mapping_persistence_approval,
)


def _gate_audit() -> dict:
    return {
        "audit_version": "v3_round98_runtime_mapping_persistence_gate_audit",
        "persistence_gate_status": "ready_for_operator_persistence_decision",
        "mapped_count": 1,
        "blocked_count": 0,
        "mapped_tokens": ["민석"],
        "audit_rows": [
            {
                "lexical_token": "민석",
                "mapping_target_category_id": "concept_category::lex::민석",
                "uses_lexical_vector_as_anchor": False,
                "uses_eve_specific_vector_as_anchor": False,
                "uses_seed_vector_as_anchor": False,
            }
        ],
    }


def _manual_validation() -> dict:
    return {
        "validation_version": "v3_round103_manual_medium_vector_validation",
        "validation_passed": True,
        "status": "manual_medium_vector_validation_passed",
    }


def test_round104_approval_requires_operator_approval_after_validation() -> None:
    blocked = build_runtime_mapping_persistence_approval(
        source_gate_audit=_gate_audit(),
        source_manual_validation=_manual_validation(),
        operator_approved=False,
    )
    assert blocked["approval_version"] == ROUND104_PERSISTENCE_APPROVAL_VERSION
    assert blocked["approval_ready"] is False
    assert "operator_approval_not_granted" in blocked["blocking_reasons"]
    assert blocked["runtime_mapping_persisted_now"] is False

    approved = build_runtime_mapping_persistence_approval(
        source_gate_audit=_gate_audit(),
        source_manual_validation=_manual_validation(),
        operator_approved=True,
    )
    assert approved["approval_ready"] is True
    assert approved["approval_status"] == "approved_for_round106_decision_review"
    assert approved["mapped_tokens"] == ["민석"]
    assert approved["policy"]["calls_agp_verify"] is False


def test_round105_proof_expansion_preserves_agp_anchor_boundary() -> None:
    approval = build_runtime_mapping_persistence_approval(
        source_gate_audit=_gate_audit(),
        source_manual_validation=_manual_validation(),
        operator_approved=True,
    )
    proof = build_agp_proof_object_expansion(source_gate_audit=_gate_audit(), source_approval=approval)

    assert proof["proof_version"] == ROUND105_AGP_PROOF_EXPANSION_VERSION
    assert proof["proof_ready"] is True
    assert proof["proof_status"] == "ready_for_round106_persistence_decision"
    assert proof["runtime_mapping_persisted_now"] is False
    row = proof["proof_rows"][0]
    assert row["lexical_token"] == "민석"
    assert row["anchor_source"] == "explicit_category_plus_sa_activation_only"
    assert row["agp_verify_called_by_expansion"] is False
    assert row["lexical_vector_used_as_anchor"] is False
    assert row["eve_specific_vector_used_as_anchor"] is False
    assert row["seed_vector_used_as_anchor"] is False


def test_round104_105_exports_write_json_only(tmp_path: Path) -> None:
    approval = build_runtime_mapping_persistence_approval(
        source_gate_audit=_gate_audit(),
        source_manual_validation=_manual_validation(),
        operator_approved=True,
    )
    proof = build_agp_proof_object_expansion(source_gate_audit=_gate_audit(), source_approval=approval)

    approval_export = write_round104_runtime_mapping_persistence_approval(tmp_path / "approval.json", approval)
    proof_export = write_round105_agp_proof_object_expansion(tmp_path / "proof.json", proof)

    assert approval_export["export_version"] == ROUND104_PERSISTENCE_APPROVAL_VERSION
    assert approval_export["runtime_mapping_persisted"] is False
    assert proof_export["export_version"] == ROUND105_AGP_PROOF_EXPANSION_VERSION
    assert proof_export["agp_verify_called"] is False
    assert proof_export["runtime_mapping_persisted"] is False
