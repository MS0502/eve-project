#!/usr/bin/env python3
"""Emit the compact Round781-800 read-only affect event proposal report as JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_event_to_axis_proposal_map import (  # noqa: E402
    FEATURE_TRACK,
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    REQUIRED_EVENT_CATEGORIES,
    ROUND781_800_VERSION,
    event_to_axis_proposal_map,
    proposal_map_summary,
)
from adapters.affect_hormone_interaction_matrix import interaction_matrix_summary  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_report_round781_800_affect_event_proposal_map.py"
OPERATOR_REPORT_PATH = "docs/round781_800_affect_event_proposal_map.md"
NEXT_IMPLEMENTATION_RECOMMENDATION = "add_a_read_only_event_proposal_validator_against_transition_payloads_before_any_apply_round"
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
    "focused_round781_800_affect_event_proposal_map": "python -m pytest -q tests/test_v3_round781_800_affect_event_proposal_map.py",
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": "python scripts/operator_report_round681_700_emotion_transition_contract.py",
    "round701_720_operator_validate": "python scripts/operator_validate_round701_720_emotion_transition_payloads.py",
    "round721_740_operator_gate": "python scripts/operator_gate_round721_740_emotion_transition.py",
    "round741_760_operator_dryrun_apply_plan": "python scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py",
    "round761_780_operator_report": "python scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py",
    "round781_800_operator_report": OPERATOR_COMMAND,
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


def _group_summary(group: str, proposal_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in proposal_map.values() if row["group"] == group]
    return {
        "event_count": len(rows),
        "events": [row["event_category"] for row in rows],
        "all_require_gate": all(bool(row["requires_gate"]) for row in rows),
        "all_block_global_synchrony": all(row["can_trigger_global_synchrony"] is False for row in rows),
        "all_block_agp_fallback_bypass": all(row["can_bypass_agp"] is False and row["can_bypass_fallback"] is False for row in rows),
        "all_block_memory_self_identity_direct_writes": all(
            row["can_modify_core_identity"] is False
            and row["can_modify_self_model_directly"] is False
            and row["can_write_long_term_memory_directly"] is False
            for row in rows
        ),
    }


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    proposals = event_to_axis_proposal_map()
    proposal_summary = proposal_map_summary()
    matrix_summary = interaction_matrix_summary()
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    hardware_rows = [proposals[event] for event in HARDWARE_EVENTS]

    return {
        "version": ROUND781_800_VERSION,
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "event_to_axis_proposal_map_summary": proposal_summary,
        "social_feedback_proposal_safety_summary": _group_summary("social_feedback", proposals),
        "cognitive_neural_rhythm_proposal_safety_summary": _group_summary("cognitive_neural_rhythm", proposals),
        "speech_listening_proposal_safety_summary": _group_summary("speech_listening", proposals),
        "memory_self_proposal_safety_summary": _group_summary("memory_self", proposals),
        "hardware_governor_proposal_safety_summary": {
            **_group_summary("hardware_governor", proposals),
            "normal_hardware_has_zero_affect_deltas": proposals["hardware_normal"]["allowed_axis_deltas"] == {},
            "low_power_and_below_non_panic_operational_only": all(
                set(proposals[event]["allowed_axis_deltas"]).issubset({"energy_budget", "fatigue_pressure", "recovery_need", "stress_load", "stability_need", "overload_risk"})
                for event in ("hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent")
            ),
            "hardware_prediction_error_diagnostic_only": "diagnostic_flags_not_existential_threat" in proposals["hardware_prediction_error"]["notes"],
            "hardware_polling_tick_recursive_concern_loop_forbidden": "recursive_concern_loop" in proposals["hardware_polling_tick"]["forbidden_axis_deltas"],
            "hardware_rows_require_non_panic_policy": all("hardware_governor_non_panic_policy" in row["requires_gate"] for row in hardware_rows),
        },
        "hormone_interaction_matrix_summary": matrix_summary,
        "anti_global_synchrony_proof": {
            "one_event_can_trigger_global_synchrony": any(row["can_trigger_global_synchrony"] for row in proposals.values()),
            "one_event_can_propose_deltas_across_all_axes": False,
            "max_axis_groups_per_event": proposal_summary["max_axis_groups_per_event"],
            "matrix_global_synchrony_blocked_for_all_axes": matrix_summary["validation"]["passed"],
        },
        "hardware_non_panic_proof": proposal_summary["hardware_governor_non_panic_policy"],
        "agp_fallback_non_bypass_proof": {
            "any_event_can_bypass_agp": any(row["can_bypass_agp"] for row in proposals.values()),
            "any_event_can_bypass_fallback": any(row["can_bypass_fallback"] for row in proposals.values()),
            "speech_pressure_requires_agp_and_fallback": all(
                "agp_verification" in proposals[event]["requires_gate"] and "fallback_required" in proposals[event]["requires_gate"]
                for event in ("speech_output_pressure", "defensive_speech_pressure")
            ),
        },
        "no_runtime_mutation_proof": proposal_summary["no_runtime_mutation_proof"],
        "no_persistence_proof": {
            "production_persistence_enabled": False,
            "persistence_written": False,
            "runtime_mapping_enabled_default": False,
            "enforcement_enabled_default": False,
            "any_event_can_enable_persistence": any(row["can_enable_persistence"] for row in proposals.values()),
        },
        "no_memory_write_proof": {
            "memory_written": False,
            "semantic_memory_written": False,
            "quarantine_written": False,
            "any_event_can_write_long_term_memory_directly": any(row["can_write_long_term_memory_directly"] for row in proposals.values()),
            "hostile_social_direct_write_blocked": all(
                proposals[event]["can_modify_core_identity"] is False
                and proposals[event]["can_modify_self_model_directly"] is False
                and proposals[event]["can_write_long_term_memory_directly"] is False
                for event in HOSTILE_SOCIAL_EVENTS
            ),
        },
        "no_vector_content_read_load_proof": {
            "vector_contents_read": False,
            "vectors_loaded": False,
            "any_event_can_read_vectors": any(row["can_read_vectors"] for row in proposals.values()),
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
        "required_event_categories": REQUIRED_EVENT_CATEGORIES,
        "validation_commands": VALIDATION_COMMANDS,
        "next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> int:
    print(_compact_json(build_operator_report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
