#!/usr/bin/env python3
"""Emit compact Round841-860 read-only affect transition handoff JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_transition_payload_operator_handoff import (  # noqa: E402
    FEATURE_TRACK,
    ROUND841_860_VERSION,
    affect_transition_payload_operator_handoff_summary,
    build_operator_review_handoff,
)

OPERATOR_COMMAND = "python scripts/operator_handoff_round841_860_affect_transition_payload_review.py"
OPERATOR_REPORT_PATH = "docs/round841_860_affect_transition_payload_operator_handoff.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = "add_operator_decision_record_schema_for_review_packets_without_apply_permission"
FORBIDDEN_GIT_PATTERNS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets",
    ".zip",
    ".part",
)


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git_status_short(repo_root: Path) -> tuple[bool, list[str]]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode == 0, [line for line in proc.stdout.splitlines() if line.strip()]


def _forbidden_entries(lines: Sequence[str]) -> list[str]:
    return [line for line in lines if any(pattern in line for pattern in FORBIDDEN_GIT_PATTERNS)]


def _safe_handoff(result: Mapping[str, Any]) -> dict[str, Any]:
    packet = result.get("review_packet") if isinstance(result.get("review_packet"), Mapping) else {}
    return {
        "handoff_passed": result.get("handoff_passed"),
        "handoff_status": result.get("handoff_status"),
        "event_category": result.get("event_category"),
        "builder_passed": result.get("builder_passed"),
        "proposal_validation_passed": result.get("proposal_validation_passed"),
        "transition_payload_validated_by_emotion_validator": result.get("transition_payload_validated_by_emotion_validator"),
        "transition_gate_passed": result.get("transition_gate_passed"),
        "operator_review_ready": result.get("operator_review_ready"),
        "operator_review_required": result.get("operator_review_required"),
        "dryrun_apply_allowed": result.get("dryrun_apply_allowed"),
        "live_apply_allowed": result.get("live_apply_allowed"),
        "apply_permission_granted": result.get("apply_permission_granted"),
        "required_quarantine": result.get("required_quarantine"),
        "required_appraisal": result.get("required_appraisal"),
        "required_gate": result.get("required_gate"),
        "requires_operator_authorization_for_apply": result.get("requires_operator_authorization_for_apply"),
        "blocked_reasons": result.get("blocked_reasons"),
        "warnings": result.get("warnings"),
        "review_packet_summary": {
            "packet_version": packet.get("packet_version"),
            "event_category": packet.get("event_category"),
            "proposed_axis_deltas": packet.get("proposed_axis_deltas"),
            "target_axes": packet.get("target_axes"),
            "target_surfaces": packet.get("target_surfaces"),
            "quarantine_required": packet.get("quarantine_required"),
            "appraisal_required_before_memory": packet.get("appraisal_required_before_memory"),
            "gate_required": packet.get("gate_required"),
            "operator_authorization_required_for_apply": packet.get("operator_authorization_required_for_apply"),
            "core_identity_update_requested": packet.get("core_identity_update_requested"),
            "self_model_update_requested": packet.get("self_model_update_requested"),
            "long_term_memory_update_requested": packet.get("long_term_memory_update_requested"),
            "runtime_mutation_requested": packet.get("runtime_mutation_requested"),
            "persistence_write_requested": packet.get("persistence_write_requested"),
            "memory_write_requested": packet.get("memory_write_requested"),
            "vector_read_requested": packet.get("vector_read_requested"),
            "vector_load_requested": packet.get("vector_load_requested"),
            "agp_bypass_requested": packet.get("agp_bypass_requested"),
            "fallback_bypass_requested": packet.get("fallback_bypass_requested"),
            "hardware_non_panic_preserved": packet.get("hardware_non_panic_preserved"),
            "hardware_operational_only": packet.get("hardware_operational_only"),
            "hardware_diagnostic_only": packet.get("hardware_diagnostic_only"),
            "global_synchrony_blocked": packet.get("global_synchrony_blocked"),
            "review_decision_slots": packet.get("review_decision_slots"),
        },
        "runtime_mutation_performed": result.get("runtime_mutation_performed"),
        "state_mutation_performed": result.get("state_mutation_performed"),
        "memory_write_performed": result.get("memory_write_performed"),
        "persistence_write_performed": result.get("persistence_write_performed"),
        "vector_read_performed": result.get("vector_read_performed"),
        "vector_load_performed": result.get("vector_load_performed"),
        "artifact_created_or_staged": result.get("artifact_created_or_staged"),
        "agp_bypass_allowed": result.get("agp_bypass_allowed"),
        "fallback_bypass_allowed": result.get("fallback_bypass_allowed"),
    }


def _packet(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return result.get("review_packet") if isinstance(result.get("review_packet"), Mapping) else {}


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)

    cases = {
        "valid_praise_handoff_result": build_operator_review_handoff("praise", {"competence_drive": 0.02, "social_trust": 0.02}),
        "valid_useful_criticism_handoff_result": build_operator_review_handoff("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02}),
        "malicious_comment_review_packet_safety_result": build_operator_review_handoff("malicious_comment", {"social_pain": 0.02, "boundary_defense": 0.02}),
        "identity_attack_review_packet_safety_result": build_operator_review_handoff("identity_attack", {"threat_pressure": 0.03, "boundary_defense": 0.02}),
        "hardware_normal_zero_delta_handoff_result": build_operator_review_handoff("hardware_normal", {}),
        "hardware_low_power_non_panic_handoff_result": build_operator_review_handoff("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02}),
        "hardware_prediction_error_diagnostic_only_handoff_result": build_operator_review_handoff("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02}),
        "unknown_event_fail_closed_result": build_operator_review_handoff("unknown_round841_event", {}),
        "speech_pressure_agp_fallback_safe_handoff_result": build_operator_review_handoff("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01}),
        "listening_uncertainty_neutral_safe_handoff_result": build_operator_review_handoff("listening_uncertainty", {"uncertainty_pressure": 0.02}),
        "imagination_negative_spiral_boundary_preserving_handoff_result": build_operator_review_handoff("imagination_negative_spiral", {"recovery_need": 0.02, "stability_need": 0.02}),
        "memory_self_update_quarantine_preserving_handoff_result": build_operator_review_handoff("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02, "learning_pressure": 0.01}),
    }
    safe_cases = {name: _safe_handoff(result) for name, result in cases.items()}
    successful = [result for result in cases.values() if result.get("handoff_passed") is True]
    packets = [_packet(result) for result in successful]

    return {
        "version": ROUND841_860_VERSION,
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "handoff_summary": affect_transition_payload_operator_handoff_summary(),
        "review_packet_schema_summary": affect_transition_payload_operator_handoff_summary()["review_packet_schema_summary"],
        **safe_cases,
        "builder_compatibility_proof": {
            "all_successful_handoffs_builder_passed": all(result.get("builder_passed") is True for result in successful),
            "unknown_event_fails_closed_before_review_packet": cases["unknown_event_fail_closed_result"].get("review_packet") is None,
        },
        "proposal_validator_compatibility_proof": {
            "all_successful_handoffs_proposal_validator_passed": all(result.get("proposal_validation_passed") is True for result in successful),
        },
        "emotion_transition_validator_compatibility_proof": {
            "all_successful_handoffs_emotion_validator_passed": all(result.get("transition_payload_validated_by_emotion_validator") is True for result in successful),
        },
        "emotion_transition_gate_compatibility_proof": {
            "all_successful_handoffs_gate_passed": all(result.get("transition_gate_passed") is True for result in successful),
        },
        "hardware_non_panic_proof": {
            "hardware_normal_zero_delta_only": _packet(cases["hardware_normal_zero_delta_handoff_result"]).get("proposed_axis_deltas") == {},
            "hardware_low_power_non_panic_operational_only": _packet(cases["hardware_low_power_non_panic_handoff_result"]).get("hardware_non_panic_preserved") is True,
            "hardware_prediction_error_diagnostic_operational_only": _packet(cases["hardware_prediction_error_diagnostic_only_handoff_result"]).get("hardware_diagnostic_only") is True,
        },
        "anti_global_synchrony_proof": {
            "all_review_packets_block_global_synchrony": all(packet.get("global_synchrony_blocked") is True for packet in packets),
        },
        "agp_fallback_non_bypass_proof": {
            "all_review_packets_request_no_agp_bypass": all(packet.get("agp_bypass_requested") is False for packet in packets),
            "all_review_packets_request_no_fallback_bypass": all(packet.get("fallback_bypass_requested") is False for packet in packets),
        },
        "operator_review_required_proof": all(result.get("operator_review_required") is True for result in cases.values()),
        "no_apply_permission_proof": all(
            result.get("apply_permission_granted") is False and result.get("dryrun_apply_allowed") is False and result.get("live_apply_allowed") is False
            for result in cases.values()
        ),
        "no_runtime_mutation_proof": all(result.get("runtime_mutation_performed") is False and _packet(result).get("runtime_mutation_requested", False) is False for result in cases.values()),
        "no_persistence_proof": all(result.get("persistence_write_performed") is False and _packet(result).get("persistence_write_requested", False) is False for result in cases.values()),
        "no_memory_write_proof": all(result.get("memory_write_performed") is False and _packet(result).get("memory_write_requested", False) is False for result in cases.values()),
        "no_vector_content_read_load_proof": all(
            result.get("vector_read_performed") is False
            and result.get("vector_load_performed") is False
            and _packet(result).get("vector_read_requested", False) is False
            and _packet(result).get("vector_load_requested", False) is False
            for result in cases.values()
        ),
        "no_artifact_creation_staging_proof": {
            "git_status_command_succeeded": git_ok,
            "forbidden_git_entries": forbidden,
            "passed": git_ok and not forbidden,
        },
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "production_persistence_enabled": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> int:
    print(_compact_json(build_operator_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
