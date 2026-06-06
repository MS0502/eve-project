#!/usr/bin/env python3
"""Emit compact Round821-840 read-only affect transition payload builder JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_event_to_axis_proposal_map import event_to_axis_proposal_map  # noqa: E402
from adapters.affect_proposal_transition_payload_builder import (  # noqa: E402
    FEATURE_TRACK,
    ROUND821_840_VERSION,
    affect_proposal_transition_payload_builder_summary,
    validate_and_build_transition_payload,
)

OPERATOR_COMMAND = "python scripts/operator_build_round821_840_affect_transition_payloads.py"
OPERATOR_REPORT_PATH = "docs/round821_840_affect_proposal_transition_payload_builder.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = "add_operator_review_handoff_for_built_read_only_transition_payloads_without_apply_permission"
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


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = result.get("transition_payload") if isinstance(result.get("transition_payload"), Mapping) else {}
    return {
        "build_passed": result.get("build_passed"),
        "build_status": result.get("build_status"),
        "event_category": result.get("event_category"),
        "proposal_validation_passed": result.get("proposal_validation_passed"),
        "transition_payload_validated_by_emotion_validator": result.get("transition_payload_validated_by_emotion_validator"),
        "transition_gate_passed": result.get("transition_gate_passed"),
        "blocked_reasons": result.get("blocked_reasons"),
        "warnings": result.get("warnings"),
        "required_quarantine": result.get("required_quarantine"),
        "required_appraisal": result.get("required_appraisal"),
        "required_gate": result.get("required_gate"),
        "requires_operator_authorization_for_apply": result.get("requires_operator_authorization_for_apply"),
        "target_surfaces": payload.get("target_surfaces"),
        "target_axes": payload.get("target_axes"),
        "payload_flags": {
            "runtime_mutation_requested": payload.get("runtime_mutation_requested"),
            "persistence_write_requested": payload.get("persistence_write_requested"),
            "memory_write_requested": payload.get("memory_write_requested"),
            "vector_read_requested": payload.get("vector_read_requested"),
            "vector_load_requested": payload.get("vector_load_requested"),
            "agp_bypass_requested": payload.get("agp_bypass_requested"),
            "fallback_bypass_requested": payload.get("fallback_bypass_requested"),
            "global_synchrony_blocked": payload.get("global_synchrony_blocked"),
            "hardware_non_panic_preserved": payload.get("hardware_non_panic_preserved"),
        },
        "dryrun_apply_allowed": result.get("dryrun_apply_allowed"),
        "live_apply_allowed": result.get("live_apply_allowed"),
        "runtime_mutation_performed": result.get("runtime_mutation_performed"),
        "memory_write_performed": result.get("memory_write_performed"),
        "persistence_write_performed": result.get("persistence_write_performed"),
        "vector_read_performed": result.get("vector_read_performed"),
        "vector_load_performed": result.get("vector_load_performed"),
        "artifact_created_or_staged": result.get("artifact_created_or_staged"),
    }


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    proposals = event_to_axis_proposal_map()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)

    cases = {
        "valid_praise_build_result": validate_and_build_transition_payload("praise", {"competence_drive": 0.02, "social_trust": 0.02}),
        "valid_useful_criticism_build_result": validate_and_build_transition_payload("useful_criticism", {"prediction_error_pressure": 0.03, "learning_pressure": 0.02}),
        "malicious_comment_blocked_direct_write_safe_build_result": validate_and_build_transition_payload("malicious_comment", {"social_pain": 0.02, "boundary_defense": 0.02}),
        "identity_attack_blocked_direct_write_safe_build_result": validate_and_build_transition_payload("identity_attack", {"threat_pressure": 0.03, "boundary_defense": 0.02}),
        "hardware_normal_zero_delta_build_result": validate_and_build_transition_payload("hardware_normal", {}),
        "hardware_low_power_non_panic_build_result": validate_and_build_transition_payload("hardware_low_power", {"energy_budget": -0.02, "recovery_need": 0.02}),
        "hardware_prediction_error_diagnostic_only_build_result": validate_and_build_transition_payload("hardware_prediction_error", {"overload_risk": 0.02, "stability_need": 0.02}),
        "unknown_event_fail_closed_result": validate_and_build_transition_payload("unknown_round821_event", {}),
        "speech_pressure_agp_fallback_safe_build_result": validate_and_build_transition_payload("speech_output_pressure", {"expression_pressure": 0.02, "action_readiness": 0.01}),
        "imagination_negative_spiral_boundary_preserving_build_result": validate_and_build_transition_payload("imagination_negative_spiral", {"recovery_need": 0.02, "stability_need": 0.02}),
        "memory_self_update_quarantine_preserving_build_result": validate_and_build_transition_payload("memory_consolidation_candidate", {"memory_consolidation_pressure": 0.02, "learning_pressure": 0.01}),
    }
    safe_cases = {name: _safe_result(result) for name, result in cases.items()}
    successful = [result for result in cases.values() if result.get("build_passed") is True]
    successful_payloads = [result["transition_payload"] for result in successful]

    return {
        "version": ROUND821_840_VERSION,
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "builder_summary": affect_proposal_transition_payload_builder_summary(),
        **safe_cases,
        "transition_payload_compatibility_proof": {
            "successful_payload_count": len(successful_payloads),
            "all_successful_payloads_have_required_shape": all(
                set((
                    "event_category",
                    "proposed_effects",
                    "target_surfaces",
                    "target_axes",
                    "proposed_axis_deltas",
                    "quarantine_required",
                    "appraisal_required_before_memory",
                    "hardware_non_panic_preserved",
                    "global_synchrony_blocked",
                    "notes",
                )).issubset(payload)
                for payload in successful_payloads
            ),
        },
        "emotion_transition_validator_compatibility_proof": {
            "all_successful_payloads_validator_passed": all(result.get("transition_payload_validated_by_emotion_validator") is True for result in successful),
        },
        "emotion_transition_gate_compatibility_proof": {
            "all_successful_payloads_gate_passed": all(result.get("transition_gate_passed") is True for result in successful),
            "gate_passing_still_does_not_allow_apply": all(result.get("live_apply_allowed") is False and result.get("dryrun_apply_allowed") is False for result in successful),
        },
        "hardware_non_panic_proof": {
            "hardware_normal_zero_delta_only": cases["hardware_normal_zero_delta_build_result"].get("build_passed") is True,
            "hardware_low_power_non_panic": cases["hardware_low_power_non_panic_build_result"].get("transition_payload", {}).get("hardware_non_panic_preserved") is True,
            "hardware_prediction_error_diagnostic_only": cases["hardware_prediction_error_diagnostic_only_build_result"].get("transition_payload", {}).get("hardware_diagnostic_only") is True,
        },
        "anti_global_synchrony_proof": {
            "all_successful_payloads_block_global_synchrony": all(payload.get("global_synchrony_blocked") is True for payload in successful_payloads),
            "proposal_rows_block_global_synchrony": all(row.get("can_trigger_global_synchrony") is False for row in proposals.values()),
        },
        "agp_fallback_non_bypass_proof": {
            "all_successful_payloads_request_no_agp_bypass": all(payload.get("agp_bypass_requested") is False for payload in successful_payloads),
            "all_successful_payloads_request_no_fallback_bypass": all(payload.get("fallback_bypass_requested") is False for payload in successful_payloads),
        },
        "no_runtime_mutation_proof": all(result.get("runtime_mutation_performed") is False for result in cases.values()),
        "no_persistence_proof": all(result.get("persistence_write_performed") is False for result in cases.values()),
        "no_memory_write_proof": all(result.get("memory_write_performed") is False for result in cases.values()),
        "no_vector_content_read_load_proof": all(result.get("vector_read_performed") is False and result.get("vector_load_performed") is False for result in cases.values()),
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
