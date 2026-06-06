"""Round841-860 read-only operator handoff invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from adapters.affect_event_to_axis_proposal_map import SOCIAL_SELF_IDENTITY_AXIS_SET
from adapters.affect_hormone_neural_rhythm_registry import affect_hormone_axis_registry
from adapters.affect_proposal_transition_payload_builder import validate_and_build_transition_payload
from adapters.affect_transition_payload_operator_handoff import (
    affect_transition_payload_operator_handoff_summary,
    build_operator_review_handoff,
    build_operator_review_packet_from_builder_result,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "operator_handoff_round841_860_affect_transition_payload_review.py"
DOC_PATH = REPO_ROOT / "docs" / "round841_860_affect_transition_payload_operator_handoff.md"


def _handoff(event: str, deltas: dict[str, float]) -> dict:
    return build_operator_review_handoff(event, deltas)


def _packet(event: str, deltas: dict[str, float]) -> dict:
    result = _handoff(event, deltas)
    assert result["handoff_passed"] is True
    assert result["operator_review_ready"] is True
    return result["review_packet"]


def _assert_no_apply_or_mutation(result: dict) -> None:
    assert result["operator_review_required"] is True
    assert result["dryrun_apply_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["apply_permission_granted"] is False
    assert result["runtime_mutation_performed"] is False
    assert result["state_mutation_performed"] is False
    assert result["memory_write_performed"] is False
    assert result["persistence_write_performed"] is False
    assert result["vector_read_performed"] is False
    assert result["vector_load_performed"] is False
    assert result["artifact_created_or_staged"] is False
    assert result["agp_bypass_allowed"] is False
    assert result["fallback_bypass_allowed"] is False
    packet = result.get("review_packet") or {}
    for field in (
        "runtime_mutation_requested",
        "persistence_write_requested",
        "memory_write_requested",
        "vector_read_requested",
        "vector_load_requested",
        "agp_bypass_requested",
        "fallback_bypass_requested",
    ):
        assert packet.get(field, False) is False


def test_known_valid_built_payloads_create_read_only_review_packets() -> None:
    for event, deltas in (
        ("praise", {"competence_drive": 0.02, "social_trust": 0.02}),
        ("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02}),
        ("hardware_normal", {}),
    ):
        result = _handoff(event, deltas)
        assert result["handoff_passed"] is True
        assert result["handoff_status"] == "operator_review_ready_no_apply_permission"
        assert result["builder_passed"] is True
        assert result["proposal_validation_passed"] is True
        assert result["transition_payload_validated_by_emotion_validator"] is True
        assert result["transition_gate_passed"] is True
        assert result["review_packet"]["event_category"] == event
        _assert_no_apply_or_mutation(result)


def test_unknown_event_category_fails_closed() -> None:
    result = _handoff("unknown_round841_event", {})
    assert result["handoff_passed"] is False
    assert result["operator_review_ready"] is False
    assert result["review_packet"] is None
    assert "unknown_event_category_fail_closed" in result["blocked_reasons"]
    _assert_no_apply_or_mutation(result)


def test_builder_failure_blocks_handoff() -> None:
    builder = validate_and_build_transition_payload("praise", {"energy_budget": 0.01})
    assert builder["build_passed"] is False
    result = build_operator_review_packet_from_builder_result(builder)
    assert result["handoff_passed"] is False
    assert "builder_failed_closed" in result["blocked_reasons"]
    assert result["review_packet"] is None


def test_proposal_validator_failure_blocks_handoff() -> None:
    builder = validate_and_build_transition_payload("praise", {"competence_drive": 0.02})
    mutated = deepcopy(builder)
    mutated["proposal_validation_passed"] = False
    result = build_operator_review_packet_from_builder_result(mutated)
    assert result["handoff_passed"] is False
    assert "proposal_validator_failed_closed" in result["blocked_reasons"]


def test_emotion_transition_validator_failure_blocks_handoff() -> None:
    builder = validate_and_build_transition_payload("praise", {"competence_drive": 0.02})
    mutated = deepcopy(builder)
    mutated["transition_payload_validated_by_emotion_validator"] = False
    result = build_operator_review_packet_from_builder_result(mutated)
    assert result["handoff_passed"] is False
    assert "emotion_transition_validator_failed_closed" in result["blocked_reasons"]


def test_transition_gate_failure_blocks_handoff() -> None:
    builder = validate_and_build_transition_payload("praise", {"competence_drive": 0.02})
    mutated = deepcopy(builder)
    mutated["transition_gate_passed"] = False
    result = build_operator_review_packet_from_builder_result(mutated)
    assert result["handoff_passed"] is False
    assert "transition_gate_failed_closed" in result["blocked_reasons"]


def test_review_ready_does_not_imply_dryrun_live_or_memory_permission() -> None:
    result = _handoff("praise", {"competence_drive": 0.02, "social_trust": 0.02})
    assert result["operator_review_ready"] is True
    assert result["dryrun_apply_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["memory_write_performed"] is False
    assert result["review_packet"]["memory_write_requested"] is False


def test_operator_review_required_always_true_and_apply_permission_always_false() -> None:
    for result in (
        _handoff("praise", {"competence_drive": 0.02}),
        _handoff("unknown_round841_event", {}),
    ):
        assert result["operator_review_required"] is True
        assert result["apply_permission_granted"] is False


def test_review_packets_never_request_runtime_persistence_memory_vector_or_bypass() -> None:
    for event, deltas in (
        ("praise", {"competence_drive": 0.02}),
        ("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01}),
        ("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02, "learning_pressure": 0.01}),
    ):
        _assert_no_apply_or_mutation(_handoff(event, deltas))


def test_hostile_social_review_packets_preserve_quarantine_appraisal_gate_and_no_direct_updates() -> None:
    for event, deltas in (
        ("malicious_comment", {"social_pain": 0.02, "boundary_defense": 0.02}),
        ("identity_attack", {"threat_pressure": 0.03, "boundary_defense": 0.02}),
    ):
        packet = _packet(event, deltas)
        assert packet["quarantine_required"] is True
        assert packet["appraisal_required_before_memory"] is True
        assert packet["gate_required"] is True
        assert packet["core_identity_update_requested"] is False
        assert packet["self_model_update_requested"] is False
        assert packet["long_term_memory_update_requested"] is False


def test_useful_criticism_requires_appraisal_before_memory_self_model_update() -> None:
    packet = _packet("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02})
    assert packet["appraisal_required_before_memory"] is True
    assert packet["self_model_update_requested"] is False
    assert packet["long_term_memory_update_requested"] is False


def test_hardware_normal_review_packet_has_zero_affect_deltas() -> None:
    packet = _packet("hardware_normal", {})
    assert packet["proposed_axis_deltas"] == {}
    assert packet["target_axes"] == ()


def test_hardware_events_cannot_target_social_self_identity_axes() -> None:
    for event, deltas in (
        ("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02}),
        ("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02}),
    ):
        packet = _packet(event, deltas)
        assert not (set(packet["target_axes"]) & SOCIAL_SELF_IDENTITY_AXIS_SET)


def test_hardware_low_power_review_packet_remains_non_panic_and_operational_only() -> None:
    packet = _packet("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02})
    assert packet["hardware_non_panic_preserved"] is True
    assert packet["hardware_operational_only"] is True
    assert set(packet["target_surfaces"]) == {"hardware_governor_non_panic_policy", "operational_diagnostics"}


def test_hardware_prediction_error_remains_diagnostic_operational_only() -> None:
    packet = _packet("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02})
    assert packet["hardware_diagnostic_only"] is True
    assert packet["hardware_operational_only"] is True


def test_hardware_polling_tick_cannot_create_recursive_concern_loop() -> None:
    packet = _packet("hardware_polling_tick", {})
    assert packet["recursive_concern_loop_requested"] is False
    assert packet["proposed_axis_deltas"] == {}


def test_speech_pressure_review_packet_cannot_bypass_agp_or_fallback() -> None:
    packet = _packet("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01})
    assert packet["agp_bypass_requested"] is False
    assert packet["fallback_bypass_requested"] is False
    assert "agp_gate" in packet["target_surfaces"]
    assert "fallback_gate" in packet["target_surfaces"]


def test_listening_uncertainty_cannot_relabel_neutral_input_as_hostile_by_itself() -> None:
    packet = _packet("listening_uncertainty", {"uncertainty_pressure": 0.02})
    assert packet["relabel_neutral_as_hostile_requested"] is False


def test_imagination_negative_spiral_preserves_budget_cooldown_reality_check_boundary() -> None:
    packet = _packet("imagination_negative_spiral", {"recovery_need": 0.02, "stability_need": 0.02})
    assert packet["scenario_budget_preserved"] is True
    assert packet["cooldown_preserved"] is True
    assert packet["reality_check_boundary_preserved"] is True


def test_memory_self_update_candidates_preserve_appraisal_quarantine() -> None:
    packet = _packet("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02, "learning_pressure": 0.01})
    assert packet["quarantine_required"] is True
    assert packet["appraisal_required_before_memory"] is True
    assert packet["memory_write_requested"] is False


def test_one_event_cannot_build_all_axis_review_packets() -> None:
    all_axes = {axis: 0.01 for axis in affect_hormone_axis_registry()}
    result = _handoff("praise", all_axes)
    assert result["handoff_passed"] is False
    assert result["review_packet"] is None


def test_global_synchrony_remains_blocked() -> None:
    packet = _packet("praise", {"competence_drive": 0.02, "social_trust": 0.02})
    assert packet["global_synchrony_blocked"] is True
    assert affect_transition_payload_operator_handoff_summary()["anti_global_synchrony_policy"]["global_synchrony_runaway_allowed"] is False


def test_no_live_emotion_hormone_memory_mutation_or_persistence_enablement() -> None:
    summary = affect_transition_payload_operator_handoff_summary()
    assert summary["runtime_mutation_performed"] is False
    assert summary["state_mutation_performed"] is False
    assert summary["memory_write_performed"] is False
    assert summary["persistence_write_performed"] is False
    assert summary["production_persistence_enabled"] is False


def test_no_runtime_mapping_or_enforcement_default_enablement() -> None:
    summary = affect_transition_payload_operator_handoff_summary()
    assert summary["runtime_mapping_enabled_default"] is False
    assert summary["enforcement_enabled_default"] is False


def test_no_vector_content_read_load_or_artifact_creation() -> None:
    summary = affect_transition_payload_operator_handoff_summary()
    assert summary["vector_contents_read"] is False
    assert summary["vectors_loaded"] is False
    assert summary["vector_read_performed"] is False
    assert summary["vector_load_performed"] is False
    assert summary["artifact_created_or_staged"] is False


def test_handoff_blocks_mutated_packet_that_requests_runtime_persistence_memory_vector_or_bypass() -> None:
    builder = validate_and_build_transition_payload("praise", {"competence_drive": 0.02})
    mutated = deepcopy(builder)
    mutated["transition_payload"]["runtime_mutation_requested"] = True
    result = build_operator_review_packet_from_builder_result(mutated)
    assert result["handoff_passed"] is False
    assert "review_packet_requested_forbidden_runtime_mutation_requested" in result["blocked_reasons"]


def test_operator_report_command_emits_required_compact_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(OPERATOR_SCRIPT)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    report = json.loads(proc.stdout)
    assert report["operator_command"] == "python scripts/operator_handoff_round841_860_affect_transition_payload_review.py"
    assert report["operator_report_path"] == "docs/round841_860_affect_transition_payload_operator_handoff.md"
    assert report["valid_praise_handoff_result"]["handoff_passed"] is True
    assert report["unknown_event_fail_closed_result"]["handoff_passed"] is False
    assert report["operator_review_required_proof"] is True
    assert report["no_apply_permission_proof"] is True
    assert report["no_runtime_mutation_proof"] is True
    assert report["no_persistence_proof"] is True
    assert report["no_memory_write_proof"] is True
    assert report["no_vector_content_read_load_proof"] is True
    assert report["no_artifact_creation_staging_proof"]["passed"] is True
    assert report["next_implementation_recommendation"] == "add_operator_decision_record_schema_for_review_packets_without_apply_permission"


def test_doc_path_exists_for_operator_review_handoff() -> None:
    assert DOC_PATH.exists()


def test_korean_fixtures_and_minseok_token_remain_preserved() -> None:
    fixture = REPO_ROOT / "tests" / "fixtures" / "korean_conversation_fixtures.py"
    if fixture.exists():
        text = fixture.read_text(encoding="utf-8")
        assert "민석" in text or "KOREAN" in text
    historical_test = (REPO_ROOT / "tests" / "test_round16_frame_hg_sd.py").read_text(encoding="utf-8")
    assert "민석" in historical_test
