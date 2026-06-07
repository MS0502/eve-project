#!/usr/bin/env python3
"""Compact operator report for Round1001-1020 visual observation schema."""

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
from adapters.sensory_observation_contract import AFFECT_AXIS_GROUPS  # noqa: E402
from adapters.visual_observation_schema import (  # noqa: E402
    FEATURE_TRACK,
    NEXT_IMPLEMENTATION_RECOMMENDATION,
    SUPPORTED_INFERRED_FEATURE_GROUPS,
    SUPPORTED_OBSERVED_FEATURE_GROUPS,
    SUPPORTED_VISUAL_SOURCE_TYPES,
    build_visual_observation,
    build_visual_observation_memory_candidate_plan,
    build_visual_observation_schema_summary,
    build_visual_observation_to_event_candidate_plan,
    build_visual_observation_to_sensory_observation,
    validate_visual_observation,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402

OPERATOR_COMMAND = "python scripts/operator_report_round1001_1020_visual_observation_schema.py"
OPERATOR_REPORT_PATH = "docs/round1001_1020_visual_observation_schema.md"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round1001_1020_visual_observation_schema.py"
VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round1001_1020_visual_observation_schema": FOCUSED_TEST_COMMAND,
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
    "round981_1000_operator_report": "python scripts/operator_report_round981_1000_sensory_observation_contract.py",
    "round1001_1020_operator_report": OPERATOR_COMMAND,
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
    summary = build_visual_observation_schema_summary()
    user_uploaded = build_visual_observation(
        "user_uploaded_image",
        observed_features={"person_presence_features": {"presence": "candidate"}, "object_features": {"count": 1}},
        inferred_features={"person_identity_candidate": "candidate_only", "expression_candidate": "candidate_neutral"},
        metadata={"confidence": {"person_identity_candidate": 0.32, "expression_candidate": 0.42}, "event_candidate": True},
    )
    camera_frame = build_visual_observation("camera_frame_candidate", observed_features={"scene_features": {"summary": "candidate_frame_metadata_only"}})
    screenshot = build_visual_observation("screenshot_candidate", observed_features={"screen_ui_features": {"path_metadata": "candidate_only"}})
    screen_ui = build_visual_observation("screen_ui_candidate", observed_features={"screen_ui_features": {"ui_region_metadata": "candidate_only"}}, inferred_features={"screen_state_candidate": "candidate_only"})
    unknown = build_visual_observation("unregistered_visual_source")
    emotion_fact = build_visual_observation("user_uploaded_image", observed_features={"user_emotion": "happy"})
    intent_fact = build_visual_observation("user_uploaded_image", inferred_features={"user_intent": "asking_for_help"})
    relationship_fact = build_visual_observation("user_uploaded_image", inferred_features={"relationship_state": "close_friend"})
    memory_fact = build_visual_observation("user_uploaded_image", observed_features={"memory_fact": "met_today"})
    raw_payload = build_visual_observation("user_uploaded_image", metadata={"raw_image_bytes": b"blocked"})
    malicious = build_visual_observation("user_uploaded_image", metadata={"malicious_or_hostile_content_candidate": True})
    memory_candidate = build_visual_observation("user_uploaded_image", metadata={"memory_update_candidate": True, "self_model_update_candidate": True})
    all_axis = build_visual_observation(
        "user_uploaded_image",
        metadata={"requested_affect_axis_groups": AFFECT_AXIS_GROUPS, "global_synchrony_requested": True},
    )

    visual_to_sensory = build_visual_observation_to_sensory_observation(user_uploaded)
    visual_to_event = build_visual_observation_to_event_candidate_plan(user_uploaded)
    visual_to_memory = build_visual_observation_memory_candidate_plan(memory_candidate)
    validation_rows = {
        "user_uploaded_image": validate_visual_observation(user_uploaded),
        "camera_frame_candidate": validate_visual_observation(camera_frame),
        "screenshot_candidate": validate_visual_observation(screenshot),
        "screen_ui_candidate": validate_visual_observation(screen_ui),
    }
    git_ok, git_lines = _git_status_short(root)
    forbidden = _forbidden_entries(git_lines)
    artifact_proof = {
        "git_status_ok": git_ok,
        "git_status_short_lines": git_lines,
        "forbidden_patterns": list(FORBIDDEN_GIT_PATTERNS),
        "forbidden_entries": forbidden,
        "no_forbidden_artifact_creation_or_staging": git_ok and not forbidden,
    }
    affect_ladder = build_round761_780_registry_report()
    preflight = affect_future_dryrun_simulation_runner_preflight_summary()

    invariant_rows = (user_uploaded, camera_frame, screenshot, screen_ui, malicious, memory_candidate)
    proofs = {
        "person_identity_candidate_only_proof": _proof(user_uploaded["person_identity_asserted"] is False and "person_identity_candidate" in user_uploaded["candidate_only_fields"]),
        "expression_posture_gaze_candidate_only_proof": _proof(all(key in user_uploaded["candidate_only_fields"] for key in ("expression_candidate", "posture_candidate", "gaze_candidate"))),
        "visual_emotion_not_fact_proof": _proof(not emotion_fact["visual_observation_passed"], blocked_reasons=emotion_fact["blocked_reasons"]),
        "visual_intent_not_fact_proof": _proof(not intent_fact["visual_observation_passed"], blocked_reasons=intent_fact["blocked_reasons"]),
        "relationship_state_not_fact_proof": _proof(not relationship_fact["visual_observation_passed"], blocked_reasons=relationship_fact["blocked_reasons"]),
        "memory_fact_not_asserted_proof": _proof(not memory_fact["visual_observation_passed"], blocked_reasons=memory_fact["blocked_reasons"]),
        "raw_visual_non_persistence_proof": _proof(all(raw_payload[key] is False for key in ("raw_data_persisted", "raw_image_persisted", "raw_video_persisted", "raw_screen_recording_persisted"))),
        "no_ocr_proof": _proof(all(row["ocr_performed"] is False for row in invariant_rows)),
        "no_camera_activation_proof": _proof(all(row["camera_activated"] is False for row in invariant_rows)),
        "no_vision_model_load_proof": _proof(all(row["vision_model_loaded"] is False for row in invariant_rows)),
        "no_face_recognition_proof": _proof(all(row["face_recognition_performed"] is False for row in invariant_rows)),
        "privacy_flag_proof": _proof(bool(user_uploaded["privacy_flags"]) and bool(screenshot["privacy_flags"]), user_uploaded_flags=user_uploaded["privacy_flags"], screenshot_flags=screenshot["privacy_flags"]),
        "uncertainty_flag_proof": _proof("low_confidence_visual_inference_candidate" in user_uploaded["uncertainty_flags"]),
        "sensory_contract_compatibility_proof": _proof(visual_to_sensory["plan_passed"] and visual_to_sensory["sensory_validation"]["validation_passed"]),
        "visual_to_event_candidate_only_proof": _proof(visual_to_event["event_fact_asserted"] is False and visual_to_event["user_emotion_fact_allowed"] is False),
        "visual_to_memory_gate_quarantine_proof": _proof(visual_to_memory["memory_gate_required"] is True and visual_to_memory["quarantine_required"] is True and visual_to_memory["long_term_memory_write_allowed"] is False),
        "agp_fallback_non_bypass_proof": _proof(all(row["agp_bypass_allowed"] is False and row["fallback_bypass_allowed"] is False for row in invariant_rows)),
        "no_memory_write_proof": _proof(all(row["memory_write_performed"] is False and row["long_term_memory_write_allowed"] is False for row in invariant_rows)),
        "no_self_model_update_proof": _proof(all(row["self_model_update_allowed"] is False for row in invariant_rows)),
        "no_affect_hormone_transition_proof": _proof(all(row["affect_transition_allowed"] is False and row["hormone_transition_allowed"] is False for row in invariant_rows + (all_axis,))),
        "no_persistence_runtime_vector_artifact_proof": _proof(all(row["persistence_write_performed"] is False and row["runtime_mutation_performed"] is False and row["vector_read_performed"] is False and row["vector_load_performed"] is False and row["artifact_created_or_staged"] is False for row in invariant_rows) and artifact_proof["no_forbidden_artifact_creation_or_staging"]),
        "anti_global_synchrony_proof": _proof(all_axis["global_synchrony_blocked"] is True and not all_axis["visual_observation_passed"]),
        "compatibility_proof_with_affect_ladder_modules": _proof(
            user_uploaded["affect_transition_allowed"] is False
            and user_uploaded["hormone_transition_allowed"] is False
            and "affect_hormone_neural_rhythm_registry" in affect_ladder.get("feature_track", "")
            and preflight.get("feature_track") == "read_only_future_dryrun_simulation_runner_preflight_from_request_packet"
        ),
    }
    checks = {
        "all_schema_validations_passed": all(row["validation_passed"] for row in validation_rows.values()),
        "unknown_source_fails_closed": unknown["visual_observation_passed"] is False,
        "all_proofs_passed": all(row["passed"] for row in proofs.values()),
        "artifact_safety_passed": artifact_proof["no_forbidden_artifact_creation_or_staging"],
        "exactly_one_next_implementation_recommendation": len(summary["next_implementation_recommendations"]) == 1,
    }
    success = all(checks.values())
    return {
        "version": "v3_round1001_1020_operator_visual_observation_schema_report",
        "feature_track": FEATURE_TRACK,
        "success": success,
        "status": "round1001_1020_read_only_visual_observation_schema_green" if success else "blocked_round1001_1020_visual_observation_schema",
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "validation_commands": VALIDATION_COMMANDS,
        "visual_observation_schema_summary": summary,
        "supported_visual_source_summary": SUPPORTED_VISUAL_SOURCE_TYPES,
        "observed_feature_group_summary": SUPPORTED_OBSERVED_FEATURE_GROUPS,
        "inferred_candidate_group_summary": SUPPORTED_INFERRED_FEATURE_GROUPS,
        "user_uploaded_image_schema_result": user_uploaded,
        "camera_frame_candidate_schema_result": camera_frame,
        "screenshot_candidate_schema_result": screenshot,
        "screen_ui_candidate_schema_result": screen_ui,
        "unknown_source_fail_closed_result": unknown,
        "visual_to_sensory_contract_plan": visual_to_sensory,
        "visual_to_event_candidate_plan": visual_to_event,
        "visual_to_memory_candidate_plan": visual_to_memory,
        "schema_validation_results": validation_rows,
        "proofs": proofs,
        "no_artifact_creation_staging_proof": artifact_proof,
        "checks": checks,
        "exactly_one_next_implementation_recommendation": NEXT_IMPLEMENTATION_RECOMMENDATION,
    }


def main() -> None:
    print(_compact_json(build_operator_report()))


if __name__ == "__main__":
    main()
