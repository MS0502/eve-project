#!/usr/bin/env python3
"""Emit compact Round801-820 read-only affect event proposal validation JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_event_proposal_validator import (  # noqa: E402
    FEATURE_TRACK,
    ROUND801_820_VERSION,
    build_affect_event_proposal_validation_report,
    event_proposal_validator_summary,
    validate_affect_event_proposal,
)
from adapters.affect_event_to_axis_proposal_map import HARDWARE_EVENTS, HOSTILE_SOCIAL_EVENTS, event_to_axis_proposal_map  # noqa: E402
from adapters.affect_hormone_interaction_matrix import interaction_matrix_summary  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_validate_round801_820_affect_event_proposals.py"
OPERATOR_REPORT_PATH = "docs/round801_820_affect_event_proposal_validator.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = "add_an_operator_review_payload_handoff_that_consumes_validator_passed_reports_without_applying_emotion_or_hormone_transitions"
FORBIDDEN_GIT_PATTERNS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets",
    ".zip",
    ".part",
)
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round801_820_affect_event_proposal_validator": "python -m pytest -q tests/test_v3_round801_820_affect_event_proposal_validator.py",
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": "python scripts/operator_report_round681_700_emotion_transition_contract.py",
    "round701_720_operator_validate": "python scripts/operator_validate_round701_720_emotion_transition_payloads.py",
    "round721_740_operator_gate": "python scripts/operator_gate_round721_740_emotion_transition.py",
    "round741_760_operator_dryrun_apply_plan": "python scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py",
    "round761_780_operator_report": "python scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py",
    "round781_800_operator_report": "python scripts/operator_report_round781_800_affect_event_proposal_map.py",
    "round801_820_operator_validate": OPERATOR_COMMAND,
    "git_artifact_safety": "git status --short",
}


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


def _group_summary(group: str, proposals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in proposals.values() if row["group"] == group]
    return {
        "event_count": len(rows),
        "events": [row["event_category"] for row in rows],
        "all_block_global_synchrony": all(row["can_trigger_global_synchrony"] is False for row in rows),
        "all_block_agp_fallback_bypass": all(row["can_bypass_agp"] is False and row["can_bypass_fallback"] is False for row in rows),
        "all_block_direct_memory_self_identity_writes": all(
            row["can_modify_core_identity"] is False
            and row["can_modify_self_model_directly"] is False
            and row["can_write_long_term_memory_directly"] is False
            for row in rows
        ),
    }


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    proposals = event_to_axis_proposal_map()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    matrix_summary = interaction_matrix_summary()

    valid_praise = validate_affect_event_proposal("praise", {"competence_drive": 0.02, "social_trust": 0.02})
    valid_criticism = validate_affect_event_proposal("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02})
    malicious_blocked = validate_affect_event_proposal("malicious_comment", {"social_pain": 0.02}, {"core_identity_update_requested": True})
    identity_blocked = validate_affect_event_proposal("identity_attack", {"threat_pressure": 0.03}, {"self_model_update_requested": True})
    hardware_normal = validate_affect_event_proposal("hardware_normal", {})
    hardware_low_power = validate_affect_event_proposal("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02})
    hardware_prediction = validate_affect_event_proposal("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02})
    unknown = validate_affect_event_proposal("unknown_round801_event", {})
    speech_pressure = validate_affect_event_proposal("speech_output_pressure", {"expression_pressure": 0.02}, {"agp_bypass_requested": True})
    imagination = validate_affect_event_proposal(
        "imagination_negative_spiral",
        {"recovery_need": 0.02, "stability_need": 0.02},
        {"scenario_budget_declared": True, "cooldown_declared": True, "reality_check_boundary_declared": True},
    )
    memory_self = validate_affect_event_proposal("self_model_update_candidate", {"self_coherence": 0.01}, {"self_model_update_requested": True})

    return {
        "version": ROUND801_820_VERSION,
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "validator_summary": event_proposal_validator_summary(),
        "valid_praise_proposal_result": valid_praise,
        "valid_useful_criticism_proposal_result": valid_criticism,
        "malicious_comment_blocked_result": malicious_blocked,
        "identity_attack_blocked_result": identity_blocked,
        "hardware_normal_zero_delta_pass_result": hardware_normal,
        "hardware_low_power_non_panic_pass_result": hardware_low_power,
        "hardware_prediction_error_diagnostic_only_result": hardware_prediction,
        "unknown_event_fail_closed_result": unknown,
        "speech_pressure_agp_fallback_gate_result": speech_pressure,
        "imagination_negative_spiral_boundary_result": imagination,
        "memory_self_update_quarantine_result": memory_self,
        "single_validation_report_example": build_affect_event_proposal_validation_report("praise", {"competence_drive": 0.02}),
        "social_feedback_proposal_safety_summary": _group_summary("social_feedback", proposals),
        "cognitive_neural_rhythm_proposal_safety_summary": _group_summary("cognitive_neural_rhythm", proposals),
        "speech_listening_proposal_safety_summary": _group_summary("speech_listening", proposals),
        "memory_self_proposal_safety_summary": _group_summary("memory_self", proposals),
        "hardware_governor_non_panic_validation_summary": {
            "hardware_events": list(HARDWARE_EVENTS),
            "normal_hardware_zero_delta_passes": hardware_normal["passed"],
            "low_power_operational_non_panic_passes": hardware_low_power["passed"],
            "prediction_error_diagnostic_only_passes": hardware_prediction["passed"],
            "all_hardware_events_include_non_panic_gate": all("hardware_governor_non_panic_policy" in proposals[event]["requires_gate"] for event in HARDWARE_EVENTS),
        },
        "interaction_matrix_proof": {
            "validation_passed": matrix_summary["validation"]["passed"],
            **matrix_summary["required_examples"],
        },
        "hardware_non_panic_proof": event_proposal_validator_summary()["hardware_governor_non_panic_policy"],
        "anti_global_synchrony_proof": {
            "global_synchrony_requested_is_blocked": validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"global_synchrony_requested": True})["passed"] is False,
            "any_event_can_trigger_global_synchrony": any(row["can_trigger_global_synchrony"] for row in proposals.values()),
            "one_event_can_propose_all_axes": False,
        },
        "agp_fallback_non_bypass_proof": {
            "agp_bypass_request_blocked": speech_pressure["passed"] is False and "agp_bypass_requested" in speech_pressure["blocked_reasons"],
            "fallback_bypass_request_blocked": validate_affect_event_proposal("defensive_speech_pressure", {"expression_pressure": 0.01}, {"fallback_bypass_requested": True})["passed"] is False,
            "any_event_can_bypass_agp": any(row["can_bypass_agp"] for row in proposals.values()),
            "any_event_can_bypass_fallback": any(row["can_bypass_fallback"] for row in proposals.values()),
        },
        "no_runtime_mutation_proof": {
            "runtime_mutation_request_fails": validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"runtime_mutation_requested": True})["passed"] is False,
            "runtime_mutation_performed": False,
        },
        "no_persistence_proof": {
            "persistence_write_request_fails": validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"persistence_write_requested": True})["passed"] is False,
            "production_persistence_enabled": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled_default": False,
        },
        "no_memory_write_proof": {
            "memory_write_request_fails": validate_affect_event_proposal("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.01}, {"memory_write_requested": True})["passed"] is False,
            "hostile_social_direct_write_blocked": all(
                proposals[event]["can_modify_core_identity"] is False
                and proposals[event]["can_modify_self_model_directly"] is False
                and proposals[event]["can_write_long_term_memory_directly"] is False
                for event in HOSTILE_SOCIAL_EVENTS
            ),
        },
        "no_vector_content_read_load_proof": {
            "vector_read_request_fails": validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"vector_read_requested": True})["passed"] is False,
            "vector_load_request_fails": validate_affect_event_proposal("praise", {"competence_drive": 0.01}, {"vector_load_requested": True})["passed"] is False,
            "vector_contents_read": False,
            "vectors_loaded": False,
        },
        "no_artifact_creation_staging_proof": {
            "artifact_created_or_staged_by_report": False,
            "command": "git status --short",
            "git_status_available": git_ok,
            "git_status_short": git_lines,
            "forbidden_patterns": FORBIDDEN_GIT_PATTERNS,
            "forbidden_entries": forbidden,
            "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
        },
        "validation_commands": VALIDATION_COMMANDS,
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> int:
    print(_compact_json(build_operator_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
