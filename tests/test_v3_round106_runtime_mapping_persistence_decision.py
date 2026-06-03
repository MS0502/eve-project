"""EVE v3 Round106 — runtime mapping persistence decision packet."""

from pathlib import Path

from adapters.agp_proof_object_expansion import build_agp_proof_object_expansion
from adapters.runtime_mapping_persistence_approval import build_runtime_mapping_persistence_approval
from adapters.runtime_mapping_persistence_decision import (
    ROUND106_PERSISTENCE_DECISION_VERSION,
    build_runtime_mapping_persistence_decision,
    write_round106_runtime_mapping_persistence_decision,
)


def _approved_sources() -> tuple[dict, dict]:
    gate = {
        "audit_version": "v3_round98_runtime_mapping_persistence_gate_audit",
        "persistence_gate_status": "ready_for_operator_persistence_decision",
        "mapped_count": 1,
        "mapped_tokens": ["민석"],
        "audit_rows": [{"lexical_token": "민석", "mapping_target_category_id": "concept_category::lex::민석"}],
    }
    manual = {"validation_version": "v3_round103_manual_medium_vector_validation", "validation_passed": True}
    approval = build_runtime_mapping_persistence_approval(
        source_gate_audit=gate,
        source_manual_validation=manual,
        operator_approved=True,
    )
    proof = build_agp_proof_object_expansion(source_gate_audit=gate, source_approval=approval)
    return approval, proof


def test_round106_records_ready_decision_without_applying_persistence() -> None:
    approval, proof = _approved_sources()
    decision = build_runtime_mapping_persistence_decision(source_approval=approval, source_agp_proof=proof, persist=False)

    assert decision["decision_version"] == ROUND106_PERSISTENCE_DECISION_VERSION
    assert decision["round"] == 106
    assert decision["decision_ready"] is True
    assert decision["decision_status"] == "persistence_ready_but_not_applied"
    assert decision["mapped_tokens"] == ["민석"]
    assert decision["runtime_mapping_enabled_now"] is False
    assert decision["runtime_mapping_persisted_now"] is False
    assert decision["policy"]["requires_separate_apply_patch"] is True


def test_round106_persist_request_is_deferred_to_separate_patch() -> None:
    approval, proof = _approved_sources()
    decision = build_runtime_mapping_persistence_decision(source_approval=approval, source_agp_proof=proof, persist=True)

    assert decision["decision_ready"] is True
    assert "persistence_application_deferred_to_separate_patch" in decision["blocking_reasons"]
    assert decision["runtime_mapping_persisted_now"] is False
    assert decision["rollback_plan"]["persistent_state_changed_in_round106"] is False


def test_round106_export_writes_json_only(tmp_path: Path) -> None:
    approval, proof = _approved_sources()
    decision = build_runtime_mapping_persistence_decision(source_approval=approval, source_agp_proof=proof)
    export = write_round106_runtime_mapping_persistence_decision(tmp_path / "decision.json", decision)

    assert export["export_version"] == ROUND106_PERSISTENCE_DECISION_VERSION
    assert export["json_written"] is True
    assert export["runtime_mapping_persisted"] is False
    assert export["read_only"] is True
