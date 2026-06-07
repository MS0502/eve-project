#!/usr/bin/env python3
"""Emit Round981-1000 read-only sensory observation contract report as JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.affect_future_dryrun_simulation_runner_preflight import (  # noqa: E402
    affect_future_dryrun_simulation_runner_preflight_summary,
)
from adapters.affect_hormone_neural_rhythm_registry import build_round761_780_registry_report  # noqa: E402
from adapters.sensory_observation_contract import (  # noqa: E402
    AFFECT_AXIS_GROUPS,
    FEATURE_TRACK,
    NEXT_IMPLEMENTATION_RECOMMENDATION,
    build_sensory_memory_candidate_plan,
    build_sensory_observation,
    build_sensory_observation_contract_summary,
    build_sensory_observation_to_agp_input_plan,
    validate_sensory_observation,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_report_round981_1000_sensory_observation_contract.py"
OPERATOR_REPORT_PATH = "docs/round981_1000_sensory_observation_contract.md"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round981_1000_sensory_observation_contract.py"
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round981_1000_sensory_observation_contract": FOCUSED_TEST_COMMAND,
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": "python scripts/operator_audit_round641_660_appraisal_agp_input.py",
    "round681_700_operator_report": "python scripts/operator_report_round681_700_emotion_transition_contract.py",
    "round701_720_operator_validate": "python scripts/operator_validate_round701_720_emotion_transition_payloads.py",
    "round721_740_operator_gate": "python scripts/operator_gate_round721_740_emotion_transition.py",
    "round741_760_operator_dryrun": "python scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py",
    "round761_780_operator_report": "python scripts/operator_report_round761_780_affect_hormone_neural_rhythm_registry.py",
    "round781_800_operator_report": "python scripts/operator_report_round781_800_affect_event_proposal_map.py",
    "round801_820_operator_validate": "python scripts/operator_validate_round801_820_affect_event_proposals.py",
    "round821_840_operator_build": "python scripts/operator_build_round821_840_affect_transition_payloads.py",
    "round841_860_operator_handoff": "python scripts/operator_handoff_round841_860_affect_transition_payload_review.py",
    "round861_880_operator_bridge": "python scripts/operator_bridge_round861_880_affect_reviewed_payload_to_dryrun.py",
    "round881_900_operator_plan": "python scripts/operator_plan_round881_900_affect_checkpoint_rollback.py",
    "round901_920_operator_trace": "python scripts/operator_trace_round901_920_affect_checkpoint_rollback_dryrun.py",
    "round921_940_operator_decision": "python scripts/operator_decision_round921_940_affect_dryrun_trace_packet.py",
    "round941_960_operator_request": "python scripts/operator_request_round941_960_affect_future_dryrun_simulation.py",
    "round961_980_operator_preflight": "python scripts/operator_preflight_round961_980_affect_future_dryrun_runner.py",
    "round981_1000_operator_report": OPERATOR_COMMAND,
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


def _proof(value: bool, **extra: Any) -> dict[str, Any]:
    payload = {"passed": bool(value)}
    payload.update(extra)
    return payload


def build_operator_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    summary = build_sensory_observation_contract_summary()
    visual = build_sensory_observation(
        "visual",
        "user_uploaded_image",
        observed_features={"person": "present", "object_count": 1},
        inferred_features={"expression": "candidate_smile", "person_identity": "candidate_only"},
        metadata={"confidence": {"expression": 0.41}, "thought_candidate": True},
    )
    auditory = build_sensory_observation(
        "auditory",
        "voice_message_candidate",
        observed_features={"voice": "present"},
        inferred_features={"prosody": "candidate_soft", "speaker_identity": "candidate_only"},
        metadata={"confidence": {"prosody": 0.37}},
    )
    internal_state = build_sensory_observation("internal_state", "battery", observed_features={"status": "low_power_candidate"})
    tool_state = build_sensory_observation("tool_state", "file_change", observed_features={"path_metadata": {"path_kind": "relative"}, "event_type": "modified"})
    multimodal = build_sensory_observation(
        "multimodal",
        "multimodal_candidate",
        observed_features={"person": "present", "voice": "present"},
        inferred_features={"expression": "candidate", "prosody": "candidate"},
    )
    unknown_modality = build_sensory_observation("olfactory", "unknown_candidate")
    unknown_source = build_sensory_observation("visual", "unregistered_visual_sensor")
    visual_emotion_fact = build_sensory_observation("visual", "user_uploaded_image", observed_features={"user_emotion": "happy"})
    auditory_emotion_fact = build_sensory_observation("auditory", "voice_message_candidate", observed_features={"speaker_emotion": "sad"})
    malicious = build_sensory_observation(
        "text",
        "user_text_candidate",
        observed_features={"summary": "hostile sensory content candidate"},
        metadata={"malicious_or_hostile_content_candidate": True, "memory_update_candidate": True},
    )
    all_axis = build_sensory_observation(
        "text",
        "user_text_candidate",
        metadata={"requested_affect_axis_groups": AFFECT_AXIS_GROUPS, "global_synchrony_requested": True},
    )
    hardware_bad_axis = build_sensory_observation("internal_state", "temperature", metadata={"target_axes": ["social"]})
    tool_bad_content = build_sensory_observation("tool_state", "file_change", observed_features={"file_content": "blocked"})
    raw_payload = build_sensory_observation("visual", "user_uploaded_image", metadata={"raw_data_present": True})

    agp_plan = build_sensory_observation_to_agp_input_plan(visual)
    memory_plan = build_sensory_memory_candidate_plan(malicious)
    validation_rows = {
        "visual": validate_sensory_observation(visual),
        "auditory": validate_sensory_observation(auditory),
        "internal_state": validate_sensory_observation(internal_state),
        "tool_state": validate_sensory_observation(tool_state),
        "multimodal": validate_sensory_observation(multimodal),
    }

    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    artifact_proof = {
        "command": "git status --short",
        "git_status_available": git_ok,
        "git_status_short": git_lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
    }
    affect_ladder = build_round761_780_registry_report()
    preflight = affect_future_dryrun_simulation_runner_preflight_summary()

    proofs = {
        "visual_emotion_not_fact_proof": _proof(not visual_emotion_fact["observation_passed"], blocked_reasons=visual_emotion_fact["blocked_reasons"]),
        "auditory_emotion_not_fact_proof": _proof(not auditory_emotion_fact["observation_passed"], blocked_reasons=auditory_emotion_fact["blocked_reasons"]),
        "raw_image_audio_video_non_persistence_proof": _proof(
            all(raw_payload[key] is False for key in ("raw_image_persisted", "raw_audio_persisted", "raw_video_persisted", "raw_screen_recording_persisted"))
        ),
        "no_ocr_stt_proof": _proof(visual["ocr_performed"] is False and auditory["speech_to_text_performed"] is False),
        "no_camera_microphone_activation_proof": _proof(visual["camera_activated"] is False and auditory["microphone_activated"] is False),
        "no_model_load_proof": _proof(visual["model_loaded"] is False and auditory["model_loaded"] is False),
        "no_memory_write_proof": _proof(all(row["memory_write_performed"] is False for row in (visual, auditory, internal_state, tool_state, multimodal, malicious))),
        "no_self_model_update_proof": _proof(all(row["self_model_update_allowed"] is False for row in (visual, auditory, malicious))),
        "no_affect_hormone_transition_proof": _proof(all(row["affect_transition_allowed"] is False and row["hormone_transition_allowed"] is False for row in (visual, auditory, all_axis))),
        "agp_fallback_non_bypass_proof": _proof(agp_plan["agp_bypass_allowed"] is False and agp_plan["fallback_bypass_allowed"] is False),
        "memory_gate_quarantine_proof": _proof(memory_plan["memory_gate_required"] is True and memory_plan["quarantine_required"] is True),
        "privacy_flag_proof": _proof(bool(visual["privacy_flags"]) and bool(auditory["privacy_flags"]), visual_flags=visual["privacy_flags"], auditory_flags=auditory["privacy_flags"]),
        "hardware_non_panic_proof": _proof(internal_state["hardware_operational_only"] is True and internal_state["hardware_non_panic"] is True and not hardware_bad_axis["observation_passed"]),
        "tool_state_path_metadata_only_proof": _proof(tool_state["tool_state_path_metadata_only"] is True and not tool_bad_content["observation_passed"]),
        "cross_modal_binding_deferred_proof": _proof(multimodal["cross_modal_binding_deferred"] is True and multimodal["cross_modal_binding_performed"] is False),
        "anti_global_synchrony_proof": _proof(all_axis["global_synchrony_blocked"] is True and not all_axis["observation_passed"]),
        "compatibility_proof_with_affect_ladder_modules": _proof(
            visual["affect_transition_allowed"] is False
            and visual["hormone_transition_allowed"] is False
            and "affect_hormone_neural_rhythm_registry" in affect_ladder.get("feature_track", "")
        ),
        "compatibility_proof_with_future_dryrun_runner_preflight": _proof(
            preflight.get("feature_track") == "read_only_future_dryrun_simulation_runner_preflight_from_request_packet"
        ),
        "no_persistence_runtime_vector_artifact_proof": _proof(
            all(row["persistence_write_performed"] is False and row["runtime_mutation_performed"] is False and row["vector_read_performed"] is False and row["vector_load_performed"] is False and row["artifact_created_or_staged"] is False for row in (visual, auditory, internal_state, tool_state, multimodal))
            and artifact_proof["no_forbidden_artifact_creation_or_staging"]
        ),
    }
    checks = {
        "all_schema_validations_passed": all(row["validation_passed"] for row in validation_rows.values()),
        "unknown_modality_fails_closed": unknown_modality["observation_passed"] is False,
        "unknown_source_fails_closed": unknown_source["observation_passed"] is False,
        "all_proofs_passed": all(row["passed"] for row in proofs.values()),
        "artifact_safety_passed": artifact_proof["no_forbidden_artifact_creation_or_staging"],
        "exactly_one_next_implementation_recommendation": len(summary["next_implementation_recommendations"]) == 1,
    }
    success = all(checks.values())
    return {
        "version": "v3_round981_1000_operator_sensory_observation_contract_report",
        "feature_track": FEATURE_TRACK,
        "success": success,
        "status": "round981_1000_read_only_sensory_observation_contract_green" if success else "blocked_round981_1000_sensory_observation_contract",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "sensory_observation_contract_summary": summary,
        "supported_modality_summary": summary["supported_modalities"],
        "supported_source_summary": summary["supported_sources_by_modality"],
        "visual_observation_schema_result": visual,
        "auditory_observation_schema_result": auditory,
        "internal_state_observation_schema_result": internal_state,
        "tool_state_observation_schema_result": tool_state,
        "multimodal_observation_schema_result": multimodal,
        "schema_validation_results": validation_rows,
        "unknown_modality_fail_closed_result": unknown_modality,
        "unknown_source_fail_closed_result": unknown_source,
        "observation_to_agp_plan": agp_plan,
        "observation_to_memory_plan": memory_plan,
        "proofs": proofs,
        "no_artifact_creation_staging_proof": artifact_proof,
        "checks": checks,
        "exactly_one_next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> None:
    print(_compact_json(build_operator_report()))


if __name__ == "__main__":
    main()
