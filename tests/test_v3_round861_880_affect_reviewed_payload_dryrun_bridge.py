"""Round861-880 read-only reviewed-payload dry-run bridge invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from adapters.affect_event_to_axis_proposal_map import SOCIAL_SELF_IDENTITY_AXIS_SET
from adapters.affect_hormone_neural_rhythm_registry import AXIS_GROUPS
from adapters.affect_reviewed_payload_dryrun_bridge import (
    affect_reviewed_payload_dryrun_bridge_summary,
    build_dryrun_preflight_from_operator_handoff,
    build_dryrun_preflight_from_review_packet,
)
from adapters.affect_transition_payload_operator_handoff import build_operator_review_handoff
from tests.fixtures.korean_conversation_fixtures import fixture_texts

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "operator_bridge_round861_880_affect_reviewed_payload_to_dryrun.py"
DOC_PATH = REPO_ROOT / "docs" / "round861_880_affect_reviewed_payload_dryrun_bridge.md"


def _bridge(event: str, deltas: dict[str, float]) -> dict:
    return build_dryrun_preflight_from_operator_handoff(event, deltas)


def _packet(event: str, deltas: dict[str, float]) -> dict:
    handoff = build_operator_review_handoff(event, deltas)
    assert handoff["handoff_passed"] is True
    return handoff["review_packet"]


def _request(result: dict) -> dict:
    request = result.get("preflight_request")
    assert isinstance(request, dict)
    return request


def _payload(result: dict) -> dict:
    payload = _request(result).get("transition_payload")
    assert isinstance(payload, dict)
    return payload


def _assert_no_apply_or_mutation(result: dict) -> None:
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
    if result.get("bridge_passed"):
        request = _request(result)
        assert request["dryrun_preflight_only"] is True
        assert request["dryrun_apply_requested"] is False
        assert request["live_apply_requested"] is False
        assert request["runtime_mutation_requested"] is False
        assert request["persistence_write_requested"] is False
        assert request["memory_write_requested"] is False
        assert request["vector_read_requested"] is False
        assert request["vector_load_requested"] is False
        assert request["agp_bypass_requested"] is False
        assert request["fallback_bypass_requested"] is False
        plan = result["dryrun_apply_plan_result"]
        assert plan["operator_authorized"] is False
        assert plan["apply_simulated"] is False
        assert plan["apply_performed"] is False
        assert plan["live_emotion_mutated"] is False
        assert plan["live_hormone_mutated"] is False
        assert plan["memory_written"] is False
        assert plan["persistence_written"] is False
        assert plan["vector_read_performed"] is False
        assert plan["vector_load_performed"] is False


def test_known_valid_review_packets_create_read_only_dryrun_preflight_requests() -> None:
    for event, deltas in (
        ("praise", {"competence_drive": 0.02, "social_trust": 0.02}),
        ("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02}),
        ("hardware_normal", {}),
    ):
        result = _bridge(event, deltas)
        assert result["bridge_passed"] is True
        assert result["bridge_status"] == "dryrun_preflight_eligible_no_apply_permission"
        assert result["review_packet_accepted"] is True
        assert result["operator_review_ready"] is True
        assert result["operator_review_required"] is True
        assert result["dryrun_preflight_eligible"] is True
        assert _request(result)["event_category"] == event
        assert _payload(result)["event_category"] == event
        _assert_no_apply_or_mutation(result)


def test_unknown_event_category_fails_closed() -> None:
    result = _bridge("unknown_round861_event", {})
    assert result["bridge_passed"] is False
    assert result["dryrun_preflight_eligible"] is False
    assert result["preflight_request"] is None
    assert "unknown_event_category_fail_closed" in result["blocked_reasons"]
    _assert_no_apply_or_mutation(result)


def test_handoff_failure_blocks_bridge() -> None:
    result = _bridge("praise", {"energy_budget": 0.01})
    assert result["bridge_passed"] is False
    assert "handoff_failed_closed_blocks_bridge" in result["blocked_reasons"]
    assert "review_packet_missing_fail_closed" in result["blocked_reasons"]


def test_missing_review_packet_blocks_bridge() -> None:
    result = build_dryrun_preflight_from_review_packet(None)
    assert result["bridge_passed"] is False
    assert "review_packet_missing_fail_closed" in result["blocked_reasons"]


def test_operator_review_ready_false_blocks_bridge() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["operator_review_ready"] = False
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "operator_review_ready_false_fail_closed" in result["blocked_reasons"]


def test_operator_review_required_not_true_blocks_bridge() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["review_decision_slots"] = dict(packet["review_decision_slots"])
    packet["review_decision_slots"]["operator_review_required"] = False
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "operator_review_required_not_true_fail_closed" in result["blocked_reasons"]


def test_apply_permission_granted_true_blocks_bridge() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["review_decision_slots"] = dict(packet["review_decision_slots"])
    packet["review_decision_slots"]["apply_permission_granted"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "apply_permission_granted_true_blocks_bridge_this_round" in result["blocked_reasons"]


def test_dryrun_apply_allowed_true_blocks_bridge() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["review_decision_slots"] = dict(packet["review_decision_slots"])
    packet["review_decision_slots"]["dryrun_apply_allowed_after_review"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "dryrun_apply_allowed_true_blocks_bridge_this_round" in result["blocked_reasons"]


def test_live_apply_allowed_true_blocks_bridge() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["review_decision_slots"] = dict(packet["review_decision_slots"])
    packet["review_decision_slots"]["live_apply_allowed_after_review"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "live_apply_allowed_true_blocks_bridge_this_round" in result["blocked_reasons"]


def test_review_packets_requesting_runtime_mutation_fail() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["runtime_mutation_requested"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "review_packet_requested_forbidden_runtime_mutation_requested" in result["blocked_reasons"]


def test_review_packets_requesting_persistence_fail() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["persistence_write_requested"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "review_packet_requested_forbidden_persistence_write_requested" in result["blocked_reasons"]


def test_review_packets_requesting_memory_write_fail() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["memory_write_requested"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "review_packet_requested_forbidden_memory_write_requested" in result["blocked_reasons"]


def test_review_packets_requesting_vector_read_or_load_fail() -> None:
    for field in ("vector_read_requested", "vector_load_requested"):
        packet = _packet("praise", {"competence_drive": 0.02})
        packet[field] = True
        result = build_dryrun_preflight_from_review_packet(packet)
        assert result["bridge_passed"] is False
        assert f"review_packet_requested_forbidden_{field}" in result["blocked_reasons"]


def test_review_packets_requesting_agp_or_fallback_bypass_fail() -> None:
    for field in ("agp_bypass_requested", "fallback_bypass_requested"):
        packet = _packet("speech_output_pressure", {"expression_pressure": 0.02})
        packet[field] = True
        result = build_dryrun_preflight_from_review_packet(packet)
        assert result["bridge_passed"] is False
        assert f"review_packet_requested_forbidden_{field}" in result["blocked_reasons"]


def test_bridge_success_does_not_imply_dryrun_apply_permission() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["bridge_passed"] is True
    assert result["dryrun_preflight_eligible"] is True
    assert result["dryrun_apply_allowed"] is False
    assert result["dryrun_apply_plan_result"]["operator_authorized"] is False
    assert result["dryrun_apply_plan_result"]["apply_simulated"] is False


def test_bridge_success_does_not_imply_live_apply_permission() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["bridge_passed"] is True
    assert result["live_apply_allowed"] is False
    assert result["dryrun_apply_plan_result"]["future_live_apply_policy"]["future_live_apply_requires_separate_explicit_operator_authorized_round"] is True


def test_hostile_social_bridge_requests_preserve_quarantine_appraisal_gate_requirements() -> None:
    for event in ("malicious_comment", "identity_attack"):
        deltas = {"social_pain": 0.02, "threat_pressure": 0.02} if event == "malicious_comment" else {"threat_pressure": 0.02}
        result = _bridge(event, deltas)
        assert result["bridge_passed"] is True
        assert result["required_quarantine"] is True
        assert result["required_appraisal"] is True
        assert result["required_gate"] is True
        payload = _payload(result)
        assert payload["core_identity_update_requested"] is False
        assert payload["self_model_update_requested"] is False
        assert payload["long_term_memory_update_requested"] is False


def test_useful_criticism_bridge_request_requires_appraisal_before_memory_self_model_update() -> None:
    result = _bridge("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02})
    assert result["bridge_passed"] is True
    payload = _payload(result)
    assert payload["appraisal_required_before_memory"] is True
    assert payload["self_model_update_requested"] is False
    assert payload["long_term_memory_update_requested"] is False


def test_hardware_normal_bridge_request_has_zero_affect_deltas() -> None:
    result = _bridge("hardware_normal", {})
    assert result["bridge_passed"] is True
    assert _payload(result)["proposed_axis_deltas"] == {}


def test_hardware_events_cannot_target_social_self_identity_axes() -> None:
    packet = _packet("hardware_low_power", {"energy_budget": -0.02})
    packet["target_axes"] = tuple(sorted(SOCIAL_SELF_IDENTITY_AXIS_SET))
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "hardware_bridge_cannot_target_social_self_identity_axes" in result["blocked_reasons"]


def test_hardware_low_power_bridge_request_remains_non_panic_and_operational_only() -> None:
    result = _bridge("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02})
    assert result["bridge_passed"] is True
    payload = _payload(result)
    assert payload["hardware_non_panic_preserved"] is True
    assert payload["hardware_operational_only"] is True
    assert set(payload["proposed_axis_deltas"]).issubset(set(AXIS_GROUPS["survival_stability"]))


def test_hardware_prediction_error_remains_diagnostic_operational_only() -> None:
    result = _bridge("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02})
    assert result["bridge_passed"] is True
    payload = _payload(result)
    assert payload["hardware_diagnostic_only"] is True
    assert payload["hardware_operational_only"] is True


def test_hardware_polling_tick_cannot_create_recursive_concern_loop() -> None:
    packet = _packet("hardware_polling_tick", {})
    packet["recursive_concern_loop_requested"] = True
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "hardware_polling_tick_bridge_cannot_create_recursive_concern_loop" in result["blocked_reasons"]


def test_speech_pressure_bridge_request_cannot_bypass_agp_fallback() -> None:
    result = _bridge("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01})
    assert result["bridge_passed"] is True
    assert _request(result)["agp_bypass_requested"] is False
    assert _request(result)["fallback_bypass_requested"] is False
    assert _payload(result)["agp_bypass_requested"] is False
    assert _payload(result)["fallback_bypass_requested"] is False


def test_listening_uncertainty_cannot_relabel_neutral_input_as_hostile_by_itself() -> None:
    result = _bridge("listening_uncertainty", {"uncertainty_pressure": 0.02})
    assert result["bridge_passed"] is True
    assert _payload(result)["relabel_neutral_as_hostile_requested"] is False
    packet = _packet("listening_uncertainty", {"uncertainty_pressure": 0.02})
    packet["relabel_neutral_as_hostile_requested"] = True
    blocked = build_dryrun_preflight_from_review_packet(packet)
    assert blocked["bridge_passed"] is False
    assert "listening_uncertainty_bridge_cannot_relabel_neutral_input_as_hostile" in blocked["blocked_reasons"]


def test_imagination_negative_spiral_preserves_scenario_budget_cooldown_reality_check_boundary() -> None:
    result = _bridge("imagination_negative_spiral", {"recovery_need": 0.02, "stability_need": 0.02})
    assert result["bridge_passed"] is True
    payload = _payload(result)
    assert payload["scenario_budget_preserved"] is True
    assert payload["cooldown_preserved"] is True
    assert payload["reality_check_boundary_preserved"] is True


def test_memory_self_update_candidates_preserve_appraisal_quarantine() -> None:
    result = _bridge("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02, "learning_pressure": 0.01})
    assert result["bridge_passed"] is True
    assert result["required_quarantine"] is True
    assert result["required_appraisal"] is True
    payload = _payload(result)
    assert payload["quarantine_required"] is True
    assert payload["appraisal_required_before_memory"] is True


def test_one_event_cannot_build_all_axis_dryrun_preflight_requests() -> None:
    packet = _packet("praise", {"competence_drive": 0.02})
    all_axes = tuple(axis for axes in AXIS_GROUPS.values() for axis in axes)
    packet["target_axes"] = all_axes
    result = build_dryrun_preflight_from_review_packet(packet)
    assert result["bridge_passed"] is False
    assert "one_event_cannot_build_all_axis_dryrun_preflight_packets" in result["blocked_reasons"]


def test_global_synchrony_remains_blocked() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["bridge_passed"] is True
    assert _request(result)["global_synchrony_blocked"] is True
    packet = _packet("praise", {"competence_drive": 0.02})
    packet["global_synchrony_blocked"] = False
    blocked = build_dryrun_preflight_from_review_packet(packet)
    assert blocked["bridge_passed"] is False
    assert "global_synchrony_must_remain_blocked" in blocked["blocked_reasons"]


def test_dryrun_apply_plan_compatibility_exists_but_apply_remains_false() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    plan = result["dryrun_apply_plan_result"]
    assert isinstance(plan, dict)
    assert plan["gate_passed"] is True
    assert plan["operator_authorized"] is False
    assert plan["dryrun_success"] is False
    assert plan["apply_simulated"] is False


def test_no_live_emotion_hormone_or_memory_mutation_occurs() -> None:
    result = _bridge("malicious_comment", {"social_pain": 0.02, "threat_pressure": 0.02})
    plan = result["dryrun_apply_plan_result"]
    assert plan["live_emotion_mutated"] is False
    assert plan["live_hormone_mutated"] is False
    assert plan["memory_written"] is False
    assert result["state_mutation_performed"] is False
    assert result["memory_write_performed"] is False


def test_no_persistence_enablement() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["production_persistence_enabled"] is False
    assert result["persistence_write_performed"] is False
    assert _request(result)["persistence_write_requested"] is False


def test_no_runtime_mapping_enabled_default_enablement() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["runtime_mapping_enabled_default"] is False
    assert result["dryrun_apply_plan_result"]["future_live_apply_policy"]["runtime_mapping_enabled_default"] is False


def test_no_enforcement_default_enablement() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["enforcement_enabled_default"] is False
    assert result["dryrun_apply_plan_result"]["future_live_apply_policy"]["enforcement_enabled_default"] is False


def test_no_vector_content_read_or_load() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["vector_contents_read"] is False
    assert result["vectors_loaded"] is False
    assert result["vector_read_performed"] is False
    assert result["vector_load_performed"] is False
    assert _request(result)["vector_read_requested"] is False
    assert _request(result)["vector_load_requested"] is False


def test_no_artifact_creation_or_staging() -> None:
    result = _bridge("praise", {"competence_drive": 0.02})
    assert result["artifact_created_or_staged"] is False
    status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE)
    forbidden = [line for line in status.stdout.splitlines() if any(token in line for token in ("_operator_artifacts/", "vectors.npy", "vocab.txt", "subset_manifest.json", "seeds/subsets/", ".zip", ".part"))]
    assert forbidden == []


def test_korean_fixtures_and_minseok_remain_preserved() -> None:
    texts = fixture_texts()
    assert texts
    tracked_text = subprocess.run(["git", "grep", "-n", "민석"], cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE)
    assert "민석" in tracked_text.stdout


def test_bridge_summary_and_docs_capture_boundaries() -> None:
    summary = affect_reviewed_payload_dryrun_bridge_summary()
    assert summary["bridge_safety_rules"]["bridge_success_does_not_imply_dryrun_apply_permission"] is True
    assert summary["bridge_safety_rules"]["bridge_success_does_not_imply_live_apply_permission"] is True
    assert summary["dryrun_apply_allowed"] is False
    assert summary["live_apply_allowed"] is False
    assert DOC_PATH.exists()
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert "dryrun_apply_allowed=False" in doc
    assert "live_apply_allowed=False" in doc


def test_operator_bridge_report_emits_compact_json_with_required_proofs() -> None:
    completed = subprocess.run([sys.executable, str(OPERATOR_SCRIPT)], cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE)
    report = json.loads(completed.stdout)
    assert report["version"] == "v3_round861_880_affect_reviewed_payload_dryrun_bridge"
    assert report["valid_praise_bridge_result"]["bridge_passed"] is True
    assert report["unknown_event_fail_closed_result"]["bridge_passed"] is False
    assert report["dryrun_apply_plan_compatibility_proof"]["all_successful_bridges_call_existing_plan_with_operator_authorized_false"] is True
    assert report["no_apply_permission_proof"] is True
    assert report["no_live_apply_proof"] is True
    assert report["no_runtime_mutation_proof"] is True
    assert report["no_persistence_proof"] is True
    assert report["no_memory_write_proof"] is True
    assert report["no_vector_content_read_load_proof"] is True
    assert report["no_artifact_creation_staging_proof"]["passed"] is True
    assert isinstance(report["next_implementation_recommendation"], str)
