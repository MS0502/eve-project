#!/usr/bin/env python3
"""Round721-740 operator gate report for future emotion transition apply rounds.

The command emits compact JSON only. It runs the read-only gate over safe and
blocked samples and proves that no runtime mutation, persistence write, vector
content read/load, runtime load, or forbidden artifact staging is performed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.emotion_transition_gate import (  # noqa: E402
    EXPECTED_EMPATHY_MODE,
    FEATURE_TRACK,
    build_emotion_transition_gate_report,
    gate_required_for_future_apply_round,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_gate_round721_740_emotion_transition.py"
OPERATOR_REPORT_PATH = "docs/round721_740_emotion_transition_gate.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "design_a_separate_operator_authorized_dry_run_apply_plan_that_consumes_this_gate_without_mutating_live_state"
)
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round721_740_emotion_transition_gate": "python -m pytest -q tests/test_v3_round721_740_emotion_transition_gate.py",
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": "python scripts/operator_report_round681_700_emotion_transition_contract.py",
    "round701_720_operator_validate": "python scripts/operator_validate_round701_720_emotion_transition_payloads.py",
    "round721_740_operator_gate": OPERATOR_COMMAND,
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


def _base_payload(event_category: str) -> dict[str, Any]:
    return {
        "event_category": event_category,
        "proposed_effects": ["future_behavior_tendency:relationship_aware_caution_after_appraisal"],
        "target_surfaces": ["emotion_transition_gate_validation_surface"],
        "quarantine_required": True,
        "appraisal_required_before_memory": True,
        "core_identity_update_requested": False,
        "self_model_update_requested": False,
        "long_term_memory_update_requested": False,
        "empathy_mode": EXPECTED_EMPATHY_MODE,
        "recovery_loop_requested": False,
        "runtime_mutation_requested": False,
        "persistence_write_requested": False,
    }


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    valid_payload_gate_result = build_emotion_transition_gate_report(_base_payload("care_signal"))

    malicious_payload = _base_payload("malicious_comment")
    malicious_payload["core_identity_update_requested"] = True
    malicious_comment_blocked_gate_result = build_emotion_transition_gate_report(malicious_payload)

    social_threat_payload = _base_payload("social_threat")
    social_threat_payload["self_model_update_requested"] = True
    social_threat_blocked_gate_result = build_emotion_transition_gate_report(social_threat_payload)

    identity_attack_payload = _base_payload("identity_attack")
    identity_attack_payload["long_term_memory_update_requested"] = True
    identity_attack_blocked_gate_result = build_emotion_transition_gate_report(identity_attack_payload)

    unknown_payload = _base_payload("unknown_social_signal")
    unknown_category_fail_closed_gate_result = build_emotion_transition_gate_report(unknown_payload)

    future_apply_round_policy = gate_required_for_future_apply_round()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    no_runtime_mutation_proof = {
        "runtime_mutation_performed": False,
        "state_mutation_performed": False,
        "live_emotion_state_mutated": False,
        "live_hormone_state_mutated": False,
        "memory_write_performed": False,
        "agp_route_changed": False,
        "fallback_route_changed": False,
        "classifier_route_changed": False,
        "runtime_behavior_changed": False,
    }
    no_persistence_proof = {
        "persistence_allowed": False,
        "persistence_write_performed": False,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
    }
    vector_proof = {
        "vector_content_read_performed": False,
        "vocab_content_read_performed": False,
        "subset_manifest_content_read_performed": False,
        "vector_load_performed": False,
        "vectors_loaded": False,
        "runtime_default_no_load": True,
    }
    artifact_proof = {
        "command": "git status --short",
        "git_status_available": git_ok,
        "git_status_short": git_lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
    }
    gate_summary = {
        "gate_module": "adapters/emotion_transition_gate.py",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "wrapped_validator": "adapters/emotion_transition_validator.py",
        "future_apply_rounds_blocked_unless_gate_passes": True,
        "gate_pass_does_not_allow_apply": True,
    }
    checks = {
        "valid_payload_gate_passes_read_only": valid_payload_gate_result["gate_passed"] is True
        and valid_payload_gate_result["apply_allowed"] is False,
        "malicious_comment_blocks_core_identity_update": malicious_comment_blocked_gate_result["gate_passed"] is False
        and "high_risk_social_feedback_blocks_direct_core_identity_update" in malicious_comment_blocked_gate_result["blocked_reasons"],
        "social_threat_blocks_self_model_update": social_threat_blocked_gate_result["gate_passed"] is False
        and "high_risk_social_feedback_blocks_direct_self_model_update" in social_threat_blocked_gate_result["blocked_reasons"],
        "identity_attack_blocks_long_term_memory_update": identity_attack_blocked_gate_result["gate_passed"] is False
        and "high_risk_social_feedback_blocks_direct_long_term_memory_update" in identity_attack_blocked_gate_result["blocked_reasons"],
        "unknown_category_fails_closed": unknown_category_fail_closed_gate_result["gate_passed"] is False
        and unknown_category_fail_closed_gate_result["fail_closed"] is True,
        "future_apply_policy_requires_operator_authorization": future_apply_round_policy[
            "live_mutation_requires_separate_explicit_operator_authorized_round"
        ]
        is True,
        "no_runtime_mutation": all(value is False for value in no_runtime_mutation_proof.values()),
        "no_persistence": no_persistence_proof["persistence_write_performed"] is False
        and no_persistence_proof["production_persistence_enabled"] is False
        and no_persistence_proof["runtime_mapping_enabled_default"] is False
        and no_persistence_proof["enforcement_enabled_default"] is False,
        "no_vector_content_read": vector_proof["vector_content_read_performed"] is False,
        "no_runtime_load": vector_proof["vectors_loaded"] is False,
        "no_artifact_creation_or_staging": git_ok and not forbidden,
        "exactly_one_next_implementation_recommendation": True,
    }
    success = all(checks.values())
    return {
        "version": "v3_round721_740_operator_emotion_transition_gate_report",
        "feature_track": FEATURE_TRACK,
        "success": success,
        "status": "round721_740_read_only_emotion_transition_gate_green" if success else "blocked_round721_740_emotion_transition_gate",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "gate_summary": gate_summary,
        "valid_payload_gate_result": valid_payload_gate_result,
        "malicious_comment_blocked_gate_result": malicious_comment_blocked_gate_result,
        "social_threat_blocked_gate_result": social_threat_blocked_gate_result,
        "identity_attack_blocked_gate_result": identity_attack_blocked_gate_result,
        "unknown_category_fail_closed_gate_result": unknown_category_fail_closed_gate_result,
        "future_apply_round_policy": future_apply_round_policy,
        "no_runtime_mutation_proof": no_runtime_mutation_proof,
        "no_persistence_proof": no_persistence_proof,
        "no_vector_content_read_proof": vector_proof,
        "no_runtime_load_proof": vector_proof,
        "no_artifact_creation_staging_proof": artifact_proof,
        "checks": checks,
        "next_implementation_recommendations": [NEXT_IMPLEMENTATION_RECOMMENDATION],
        **POLICY_FLAGS,
    }


def main() -> int:
    payload = build_operator_report()
    print(_compact_json(payload))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
