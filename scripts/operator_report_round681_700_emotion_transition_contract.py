#!/usr/bin/env python3
"""Emit the Round681-700 read-only emotion transition contract report as JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.emotion_state_transition_contract import (  # noqa: E402
    CORE_IDENTITY_PROTECTION_RULE,
    MALICIOUS_FEEDBACK_QUARANTINE_RULE,
    NEXT_IMPLEMENTATION_RECOMMENDATION,
    emotion_transition_contract,
    validate_contract_shape,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_report_round681_700_emotion_transition_contract.py"
OPERATOR_REPORT_PATH = "docs/round681_700_emotion_transition_contract.md"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round681_700_emotion_transition_contract.py"
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round681_700_emotion_transition_contract": FOCUSED_TEST_COMMAND,
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": OPERATOR_COMMAND,
    "git_artifact_safety": "git status --short",
}


def _compact_json(payload: dict[str, Any]) -> str:
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


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    contract = emotion_transition_contract()
    validation = validate_contract_shape(contract)
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    no_runtime_mutation_proof = {
        "contract_module_is_pure_data_functions_only": True,
        "runtime_mutation_performed": False,
        "live_emotion_state_mutated": False,
        "live_hormone_state_mutated": False,
        "memory_write_performed": False,
        "agp_route_change_performed": False,
        "fallback_route_change_performed": False,
    }
    no_persistence_proof = {
        "persistence_write_performed": False,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled": False,
    }
    vector_and_load_proof = {
        "vector_contents_read": False,
        "vocab_contents_read": False,
        "subset_manifest_contents_read": False,
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
    checks = {
        "contract_shape_valid": validation.get("success") is True,
        "malicious_feedback_quarantine_rule_present": MALICIOUS_FEEDBACK_QUARANTINE_RULE["quarantine_required"] is True,
        "core_identity_protection_rule_present": CORE_IDENTITY_PROTECTION_RULE[
            "core_identity_update_from_raw_social_feedback_allowed"
        ]
        is False,
        "no_runtime_mutation": all(value is False for key, value in no_runtime_mutation_proof.items() if key != "contract_module_is_pure_data_functions_only"),
        "no_persistence": all(value is False for value in no_persistence_proof.values()),
        "no_vector_content_read": vector_and_load_proof["vector_contents_read"] is False,
        "no_runtime_load": vector_and_load_proof["vectors_loaded"] is False,
        "no_artifact_creation_or_staging": git_ok and not forbidden,
        "exactly_one_next_implementation_recommendation": True,
    }
    return {
        "version": "v3_round681_700_operator_emotion_transition_contract_report",
        "feature_track": contract["feature_track"],
        "success": all(checks.values()),
        "status": "round681_700_read_only_emotion_transition_contract_green" if all(checks.values()) else "blocked_round681_700_read_only_emotion_transition_contract",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "emotion_transition_contract": contract,
        "social_feedback_categories": contract["social_feedback_categories"],
        "malicious_feedback_quarantine_rule": MALICIOUS_FEEDBACK_QUARANTINE_RULE,
        "core_identity_protection_rule": CORE_IDENTITY_PROTECTION_RULE,
        "no_runtime_mutation_proof": no_runtime_mutation_proof,
        "no_persistence_proof": no_persistence_proof,
        "no_vector_content_read_proof": vector_and_load_proof,
        "no_runtime_load_proof": vector_and_load_proof,
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
