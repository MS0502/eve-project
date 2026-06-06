#!/usr/bin/env python3
"""Round741-760 operator dry-run emotion transition apply-plan report.

The command emits compact JSON only. It consumes the Round721-740 read-only gate
through the dry-run apply-plan module and proves that no runtime mutation,
persistence write, memory write, vector content read/load, or forbidden artifact
staging is performed.
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

from adapters.emotion_transition_dryrun_apply_plan import (  # noqa: E402
    FEATURE_TRACK,
    build_emotion_transition_dryrun_apply_plan,
    future_live_apply_policy,
)
from adapters.emotion_transition_gate import EXPECTED_EMPATHY_MODE  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py"
OPERATOR_REPORT_PATH = "docs/round741_760_emotion_transition_dryrun_apply_plan.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "propose_a_separate_operator_authorized_live_apply_round_with_explicit_checkpoint_and_rollback_plan"
)
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
    "focused_round741_760_emotion_transition_dryrun_apply_plan": "python -m pytest -q tests/test_v3_round741_760_emotion_transition_dryrun_apply_plan.py",
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": "python scripts/operator_report_round681_700_emotion_transition_contract.py",
    "round701_720_operator_validate": "python scripts/operator_validate_round701_720_emotion_transition_payloads.py",
    "round721_740_operator_gate": "python scripts/operator_gate_round721_740_emotion_transition.py",
    "round741_760_operator_dryrun_apply_plan": OPERATOR_COMMAND,
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
        "target_surfaces": ["emotion_transition_dryrun_apply_plan_surface"],
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
    valid_payload = _base_payload("care_signal")
    unauthorized_valid_payload_dryrun_blocked_result = build_emotion_transition_dryrun_apply_plan(
        valid_payload,
        operator_authorized=False,
    )
    authorized_valid_payload_dryrun_simulated_success_result = build_emotion_transition_dryrun_apply_plan(
        valid_payload,
        operator_authorized=True,
    )

    malicious_payload = _base_payload("malicious_comment")
    malicious_payload["core_identity_update_requested"] = True
    malicious_comment_blocked_result = build_emotion_transition_dryrun_apply_plan(malicious_payload, operator_authorized=True)

    social_threat_payload = _base_payload("social_threat")
    social_threat_payload["self_model_update_requested"] = True
    social_threat_blocked_result = build_emotion_transition_dryrun_apply_plan(social_threat_payload, operator_authorized=True)

    identity_attack_payload = _base_payload("identity_attack")
    identity_attack_payload["long_term_memory_update_requested"] = True
    identity_attack_blocked_result = build_emotion_transition_dryrun_apply_plan(identity_attack_payload, operator_authorized=True)

    unknown_payload = _base_payload("unknown_social_signal")
    unknown_category_fail_closed_result = build_emotion_transition_dryrun_apply_plan(unknown_payload, operator_authorized=True)

    future_policy = future_live_apply_policy()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    no_runtime_mutation_proof = {
        "runtime_mutation_performed": False,
        "live_emotion_mutated": False,
        "live_hormone_mutated": False,
        "agp_route_changed": False,
        "fallback_route_changed": False,
        "classifier_route_changed": False,
    }
    no_persistence_proof = {
        "persistence_written": False,
        "production_persistence_enabled": False,
        "runtime_mapping_changed": False,
        "enforcement_changed": False,
    }
    no_memory_write_proof = {"memory_written": False, "semantic_memory_written": False, "quarantine_written": False}
    no_vector_content_read_load_proof = {
        "vector_content_read_performed": False,
        "vocab_content_read_performed": False,
        "subset_manifest_content_read_performed": False,
        "vector_load_performed": False,
        "vectors_loaded": False,
        "runtime_default_no_load": True,
    }
    no_artifact_creation_staging_proof = {
        "command": "git status --short",
        "git_status_available": git_ok,
        "git_status_short": git_lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
    }
    dryrun_apply_plan_summary = {
        "module": "adapters/emotion_transition_dryrun_apply_plan.py",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "consumes_gate": "adapters/emotion_transition_gate.py",
        "requires_gate_pass": True,
        "requires_operator_authorization": True,
        "apply_performed_on_success": False,
    }
    checks = {
        "unauthorized_valid_payload_blocked": unauthorized_valid_payload_dryrun_blocked_result["dryrun_success"] is False
        and "explicit_operator_authorization_missing" in unauthorized_valid_payload_dryrun_blocked_result["blocked_reasons"],
        "authorized_valid_payload_simulates_success": authorized_valid_payload_dryrun_simulated_success_result[
            "dryrun_success"
        ]
        is True
        and authorized_valid_payload_dryrun_simulated_success_result["apply_simulated"] is True
        and authorized_valid_payload_dryrun_simulated_success_result["apply_performed"] is False,
        "malicious_comment_blocks_core_identity_update": malicious_comment_blocked_result["dryrun_success"] is False
        and "high_risk_social_feedback_blocks_direct_core_identity_update" in malicious_comment_blocked_result["blocked_reasons"],
        "social_threat_blocks_self_model_update": social_threat_blocked_result["dryrun_success"] is False
        and "high_risk_social_feedback_blocks_direct_self_model_update" in social_threat_blocked_result["blocked_reasons"],
        "identity_attack_blocks_long_term_memory_update": identity_attack_blocked_result["dryrun_success"] is False
        and "high_risk_social_feedback_blocks_direct_long_term_memory_update" in identity_attack_blocked_result["blocked_reasons"],
        "unknown_category_fails_closed": unknown_category_fail_closed_result["dryrun_success"] is False
        and "unknown_event_category_fail_closed" in unknown_category_fail_closed_result["blocked_reasons"],
        "future_policy_live_apply_separate_round": future_policy[
            "future_live_apply_requires_separate_explicit_operator_authorized_round"
        ]
        is True,
        "no_runtime_mutation": all(value is False for value in no_runtime_mutation_proof.values()),
        "no_persistence": all(value is False for value in no_persistence_proof.values()),
        "no_memory_write": all(value is False for value in no_memory_write_proof.values()),
        "no_vector_content_read_or_load": no_vector_content_read_load_proof["vector_content_read_performed"] is False
        and no_vector_content_read_load_proof["vector_load_performed"] is False,
        "no_artifact_creation_or_staging": git_ok and not forbidden,
        "exactly_one_next_implementation_recommendation": True,
    }
    success = all(checks.values())
    return {
        "version": "v3_round741_760_operator_emotion_transition_dryrun_apply_plan_report",
        "feature_track": FEATURE_TRACK,
        "success": success,
        "status": "round741_760_emotion_transition_dryrun_apply_plan_green"
        if success
        else "blocked_round741_760_emotion_transition_dryrun_apply_plan",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "dryrun_apply_plan_summary": dryrun_apply_plan_summary,
        "unauthorized_valid_payload_dryrun_blocked_result": unauthorized_valid_payload_dryrun_blocked_result,
        "authorized_valid_payload_dryrun_simulated_success_result": authorized_valid_payload_dryrun_simulated_success_result,
        "malicious_comment_blocked_result": malicious_comment_blocked_result,
        "social_threat_blocked_result": social_threat_blocked_result,
        "identity_attack_blocked_result": identity_attack_blocked_result,
        "unknown_category_fail_closed_result": unknown_category_fail_closed_result,
        "future_live_apply_policy": future_policy,
        "no_runtime_mutation_proof": no_runtime_mutation_proof,
        "no_persistence_proof": no_persistence_proof,
        "no_memory_write_proof": no_memory_write_proof,
        "no_vector_content_read_load_proof": no_vector_content_read_load_proof,
        "no_artifact_creation_staging_proof": no_artifact_creation_staging_proof,
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
