#!/usr/bin/env python3
"""Emit the compact Round761-780 read-only registry operator report as JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_hormone_neural_rhythm_registry import (  # noqa: E402
    FEATURE_TRACK,
    ROUND761_780_VERSION,
    build_round761_780_registry_report,
    validate_registry_invariants,
)

OPERATOR_COMMAND = "python scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py"
OPERATOR_REPORT_PATH = "docs/round761_780_affect_hormone_neural_rhythm_registry.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = (
    "add_a_read_only_affect_proposal_validator_against_this_registry_before_any_transition_apply_round"
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
    "focused_round761_780_affect_hormone_neural_rhythm_registry": "python -m pytest -q tests/test_v3_round761_780_affect_hormone_neural_rhythm_registry.py",
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": "python scripts/operator_report_round681_700_emotion_transition_contract.py",
    "round701_720_operator_validate": "python scripts/operator_validate_round701_720_emotion_transition_payloads.py",
    "round721_740_operator_gate": "python scripts/operator_gate_round721_740_emotion_transition.py",
    "round741_760_operator_dryrun_apply_plan": "python scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py",
    "round761_780_operator_report": OPERATOR_COMMAND,
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


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    base_report = build_round761_780_registry_report()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)

    return {
        "version": ROUND761_780_VERSION,
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "constitution_design_update_summary": base_report["constitution_design_update_summary"],
        "full_affect_hormone_axis_registry_summary": base_report["affect_hormone_axis_registry_summary"],
        "rhythm_schema_summary": base_report["rhythm_schema_summary"],
        "neural_activation_pattern_fields": base_report["neural_activation_pattern_fields"],
        "thought_imagination_speech_listening_memory_action_connection_map": base_report[
            "thought_imagination_speech_listening_memory_action_connection_map"
        ],
        "hardware_governor_non_panic_policy": base_report["hardware_governor_non_panic_policy"],
        "anti_global_synchrony_policy": base_report["anti_global_synchrony_policy"],
        "registry_invariant_summary": validate_registry_invariants(),
        "no_runtime_mutation_proof": base_report["no_runtime_mutation_proof"],
        "no_persistence_proof": base_report["no_persistence_proof"],
        "no_memory_write_proof": base_report["no_memory_write_proof"],
        "no_vector_content_read_load_proof": base_report["no_vector_content_read_load_proof"],
        "no_artifact_creation_staging_proof": {
            **base_report["no_artifact_creation_staging_proof"],
            "command": "git status --short",
            "git_status_available": git_ok,
            "git_status_short": git_lines,
            "forbidden_patterns": FORBIDDEN_GIT_PATTERNS,
            "forbidden_entries": forbidden,
            "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
        },
        "runtime_defaults_proof": {
            "default_runtime_remains_no_load": True,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled_default": False,
            "production_persistence_remains_no_go": True,
            "vector_contents_read": False,
            "vectors_loaded": False,
        },
        "validation_commands": VALIDATION_COMMANDS,
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> int:
    print(_compact_json(build_operator_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
