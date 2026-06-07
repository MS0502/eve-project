"""Round861-880 compact operator report for reviewed-payload dry-run bridge."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any, Mapping

from adapters.affect_reviewed_payload_dryrun_bridge import (
    FEATURE_TRACK,
    ROUND861_880_VERSION,
    affect_reviewed_payload_dryrun_bridge_summary,
    build_dryrun_preflight_from_operator_handoff,
)
from adapters.affect_transition_payload_operator_handoff import build_operator_review_handoff

OPERATOR_COMMAND = "python scripts/operator_bridge_round861_880_affect_reviewed_payload_to_dryrun.py"
OPERATOR_REPORT_PATH = "_operator_artifacts/round861_880_affect_reviewed_payload_to_dryrun_bridge.json"
NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "Round881 should add an operator-authorized dry-run simulation acceptance fixture while keeping live apply permission false."
)
_FORBIDDEN_GIT_PATTERNS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets/",
    ".zip",
    ".part",
)


def _compact_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    request = result.get("preflight_request") if isinstance(result.get("preflight_request"), Mapping) else {}
    plan = result.get("dryrun_apply_plan_result") if isinstance(result.get("dryrun_apply_plan_result"), Mapping) else {}
    return {
        "bridge_passed": result.get("bridge_passed") is True,
        "bridge_status": result.get("bridge_status"),
        "event_category": result.get("event_category"),
        "review_packet_accepted": result.get("review_packet_accepted") is True,
        "operator_review_ready": result.get("operator_review_ready") is True,
        "operator_review_required": result.get("operator_review_required") is True,
        "operator_review_recorded": result.get("operator_review_recorded") is True,
        "dryrun_preflight_eligible": result.get("dryrun_preflight_eligible") is True,
        "dryrun_preflight_only": request.get("dryrun_preflight_only") is True,
        "dryrun_apply_allowed": result.get("dryrun_apply_allowed") is True,
        "live_apply_allowed": result.get("live_apply_allowed") is True,
        "apply_permission_granted": result.get("apply_permission_granted") is True,
        "dryrun_plan_operator_authorized": plan.get("operator_authorized") is True,
        "dryrun_plan_apply_simulated": plan.get("apply_simulated") is True,
        "dryrun_plan_status": plan.get("dryrun_status"),
        "required_quarantine": result.get("required_quarantine") is True,
        "required_appraisal": result.get("required_appraisal") is True,
        "required_gate": result.get("required_gate") is True,
        "blocked_reasons": list(result.get("blocked_reasons", ())),
        "warnings": list(result.get("warnings", ())),
        "runtime_mutation_performed": result.get("runtime_mutation_performed") is True,
        "state_mutation_performed": result.get("state_mutation_performed") is True,
        "memory_write_performed": result.get("memory_write_performed") is True,
        "persistence_write_performed": result.get("persistence_write_performed") is True,
        "vector_read_performed": result.get("vector_read_performed") is True,
        "vector_load_performed": result.get("vector_load_performed") is True,
        "artifact_created_or_staged": result.get("artifact_created_or_staged") is True,
        "agp_bypass_allowed": result.get("agp_bypass_allowed") is True,
        "fallback_bypass_allowed": result.get("fallback_bypass_allowed") is True,
    }


def _request(result: Mapping[str, Any]) -> Mapping[str, Any]:
    request = result.get("preflight_request")
    return request if isinstance(request, Mapping) else {}


def _payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _request(result).get("transition_payload")
    return payload if isinstance(payload, Mapping) else {}


def _git_artifact_safety() -> dict[str, Any]:
    completed = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = completed.stdout.splitlines()
    forbidden = [line for line in lines if any(pattern in line for pattern in _FORBIDDEN_GIT_PATTERNS)]
    return {
        "git_status_command": "git status --short",
        "git_status_command_succeeded": completed.returncode == 0,
        "forbidden_git_entries": forbidden,
        "passed": completed.returncode == 0 and not forbidden,
    }


def build_operator_report() -> dict[str, Any]:
    cases = {
        "valid_praise_bridge_result": build_dryrun_preflight_from_operator_handoff("praise", {"competence_drive": 0.02, "social_trust": 0.02}),
        "valid_useful_criticism_bridge_result": build_dryrun_preflight_from_operator_handoff("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02}),
        "malicious_comment_bridge_safety_result": build_dryrun_preflight_from_operator_handoff("malicious_comment", {"social_pain": 0.03, "threat_pressure": 0.02, "boundary_defense": 0.02}),
        "identity_attack_bridge_safety_result": build_dryrun_preflight_from_operator_handoff("identity_attack", {"threat_pressure": 0.03, "boundary_defense": 0.02}),
        "hardware_normal_zero_delta_bridge_result": build_dryrun_preflight_from_operator_handoff("hardware_normal", {}),
        "hardware_low_power_non_panic_bridge_result": build_dryrun_preflight_from_operator_handoff("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02}),
        "hardware_prediction_error_diagnostic_only_bridge_result": build_dryrun_preflight_from_operator_handoff("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02}),
        "unknown_event_fail_closed_result": build_dryrun_preflight_from_operator_handoff("unknown_round861_event", {}),
        "speech_pressure_agp_fallback_safe_bridge_result": build_dryrun_preflight_from_operator_handoff("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01}),
        "imagination_negative_spiral_boundary_preserving_bridge_result": build_dryrun_preflight_from_operator_handoff("imagination_negative_spiral", {"recovery_need": 0.02, "stability_need": 0.02}),
        "memory_self_update_quarantine_preserving_bridge_result": build_dryrun_preflight_from_operator_handoff("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02, "learning_pressure": 0.01}),
    }
    safe_cases = {name: _safe_result(result) for name, result in cases.items()}
    successful = [result for result in cases.values() if result.get("bridge_passed") is True]
    requests = [_request(result) for result in successful]
    payloads = [_payload(result) for result in successful]
    handoff = build_operator_review_handoff("praise", {"competence_drive": 0.02, "social_trust": 0.02})
    artifact_safety = _git_artifact_safety()

    return {
        "version": ROUND861_880_VERSION,
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "bridge_summary": affect_reviewed_payload_dryrun_bridge_summary(),
        "preflight_request_schema_summary": affect_reviewed_payload_dryrun_bridge_summary()["preflight_request_schema_summary"],
        **safe_cases,
        "handoff_compatibility_proof": {
            "handoff_passed_for_valid_praise": handoff.get("handoff_passed") is True,
            "bridge_accepts_handoff_review_packet": cases["valid_praise_bridge_result"].get("review_packet_accepted") is True,
        },
        "builder_compatibility_proof": {
            "all_successful_bridges_have_builder_trace": all(_request(result).get("review_packet_summary", {}).get("event_category") == result.get("event_category") for result in successful),
            "unknown_event_fails_closed_before_preflight_request": cases["unknown_event_fail_closed_result"].get("preflight_request") is None,
        },
        "proposal_validator_compatibility_proof": {"all_successful_bridges_preserve_review_packet_acceptance": all(result.get("review_packet_accepted") is True for result in successful)},
        "emotion_transition_validator_compatibility_proof": {"all_successful_dryrun_plans_reach_gate_validator": all(isinstance(result.get("dryrun_apply_plan_result"), Mapping) for result in successful)},
        "emotion_transition_gate_compatibility_proof": {"all_successful_dryrun_plans_gate_passed_before_operator_authorization_block": all(result.get("dryrun_apply_plan_result", {}).get("gate_passed") is True for result in successful)},
        "dryrun_apply_plan_compatibility_proof": {
            "all_successful_bridges_call_existing_plan_with_operator_authorized_false": all(result.get("dryrun_apply_plan_result", {}).get("operator_authorized") is False for result in successful),
            "all_successful_bridges_leave_apply_simulated_false": all(result.get("dryrun_apply_plan_result", {}).get("apply_simulated") is False for result in successful),
        },
        "hardware_non_panic_proof": {
            "hardware_normal_zero_delta_only": _payload(cases["hardware_normal_zero_delta_bridge_result"]).get("proposed_axis_deltas") == {},
            "hardware_low_power_non_panic_operational_only": _payload(cases["hardware_low_power_non_panic_bridge_result"]).get("hardware_non_panic_preserved") is True and _payload(cases["hardware_low_power_non_panic_bridge_result"]).get("hardware_operational_only") is True,
            "hardware_prediction_error_diagnostic_operational_only": _payload(cases["hardware_prediction_error_diagnostic_only_bridge_result"]).get("hardware_diagnostic_only") is True,
        },
        "anti_global_synchrony_proof": {"all_preflight_requests_block_global_synchrony": all(request.get("global_synchrony_blocked") is True for request in requests)},
        "AGP_fallback_non_bypass_proof": {
            "all_preflight_requests_request_no_agp_bypass": all(request.get("agp_bypass_requested") is False for request in requests),
            "all_preflight_requests_request_no_fallback_bypass": all(request.get("fallback_bypass_requested") is False for request in requests),
        },
        "operator_review_required_proof": all(result.get("operator_review_required") is True for result in cases.values()),
        "dryrun_preflight_only_proof": all(request.get("dryrun_preflight_only") is True and request.get("dryrun_apply_requested") is False for request in requests),
        "no_apply_permission_proof": all(result.get("apply_permission_granted") is False and result.get("dryrun_apply_allowed") is False for result in cases.values()),
        "no_live_apply_proof": all(result.get("live_apply_allowed") is False for result in cases.values()),
        "no_runtime_mutation_proof": all(result.get("runtime_mutation_performed") is False and request.get("runtime_mutation_requested", False) is False for result, request in zip(successful, requests)),
        "no_persistence_proof": all(result.get("persistence_write_performed") is False and request.get("persistence_write_requested", False) is False for result, request in zip(successful, requests)),
        "no_memory_write_proof": all(result.get("memory_write_performed") is False and request.get("memory_write_requested", False) is False for result, request in zip(successful, requests)),
        "no_vector_content_read_load_proof": all(
            result.get("vector_read_performed") is False and result.get("vector_load_performed") is False and request.get("vector_read_requested", False) is False and request.get("vector_load_requested", False) is False
            for result, request in zip(successful, requests)
        ),
        "no_artifact_creation_staging_proof": artifact_safety,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "payload_event_categories_proof": sorted(str(payload.get("event_category")) for payload in payloads),
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> int:
    print(_compact_json(build_operator_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
