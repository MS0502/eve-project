from adapters.agp_proof_object_expansion import expand_agp_proof_object
from adapters.runtime_mapping_persistence_approval import (
    round103_operator_unblocked_validation_status,
    runtime_mapping_persistence_approval_gate,
)


def test_round104_gate_accepts_operator_unblocked_validation_without_persisting() -> None:
    gate = runtime_mapping_persistence_approval_gate(round103_operator_unblocked_validation_status())

    assert gate["round"] == 104
    assert gate["hard_stop"] is False
    assert gate["status"] == "ready_for_explicit_operator_persistence_approval"
    assert gate["persistence_decision"]["runtime_mapping_persisted_now"] is False
    assert gate["persistence_decision"]["runtime_mapping_enabled_default_after_gate"] is False
    assert gate["persistence_decision"]["may_proceed_to_agp_proof_expansion"] is True
    assert gate["policy"]["vectors_npy_must_remain_out_of_pr_diff"] is True


def test_round104_gate_blocks_missing_validation_command() -> None:
    validation = round103_operator_unblocked_validation_status()
    validation["commands"] = validation["commands"][:1]

    gate = runtime_mapping_persistence_approval_gate(validation)

    assert gate["hard_stop"] is True
    assert "operator_validation_commands_missing_or_failed" in gate["hard_stop_reasons"]
    assert gate["persistence_decision"]["may_proceed_to_agp_proof_expansion"] is False


def test_round105_expands_agp_proof_object_without_mutation() -> None:
    gate = runtime_mapping_persistence_approval_gate(round103_operator_unblocked_validation_status())
    proof = expand_agp_proof_object(gate)

    assert proof["round"] == 105
    assert proof["status"] == "agp_proof_object_expanded"
    assert proof["hard_stop"] is False
    assert proof["mutation_checks"]["agp_verify_called"] is False
    assert proof["mutation_checks"]["runtime_mapping_persisted"] is False
    assert proof["policy"]["vectors_are_evidence_not_anchors"] is True
    assert "fastText_vector" in proof["proof_object"]["proof_steps"][2]["invalid_anchor_sources"]
