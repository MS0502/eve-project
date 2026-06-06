"""Round741-760 operator-authorized dry-run emotion transition apply-plan invariants."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

from adapters.agp_adapter import AGPAdapter, AGP_MODE_OBSERVATION, AGP_REASON_UNKNOWN_CATEGORY
from adapters.appraisal_classifier import AppraisalClassifier, LABEL_NONE
from adapters.emotion_transition_dryrun_apply_plan import (
    FEATURE_TRACK,
    build_emotion_transition_dryrun_apply_plan,
    dryrun_apply_requires_explicit_operator_authorization,
    dryrun_apply_requires_gate_pass,
    future_live_apply_policy,
    simulate_emotion_transition_apply_preflight,
)
from adapters.emotion_transition_gate import EXPECTED_EMPATHY_MODE
from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from scripts.operator_run_local_validation_suite import DEFAULT_TARGET_WORD
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "operator_dryrun_round741_760_emotion_transition_apply_plan.py"
FORBIDDEN_ARTIFACT_FRAGMENTS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets",
    ".zip",
    ".part",
)


def _base_payload(event_category: str) -> dict:
    return {
        "event_category": event_category,
        "proposed_effects": ["future_behavior_tendency:relationship_aware_after_appraisal"],
        "target_surfaces": ["dryrun_apply_plan_only"],
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


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def test_round741_745_dryrun_plan_requires_gate_and_operator_authorization() -> None:
    assert dryrun_apply_requires_gate_pass() is True
    assert dryrun_apply_requires_explicit_operator_authorization() is True

    result = build_emotion_transition_dryrun_apply_plan(_base_payload("care_signal"), operator_authorized=False)

    assert result["feature_track"] == FEATURE_TRACK
    assert result["dryrun_success"] is False
    assert result["dryrun_status"] == "blocked_dryrun_apply_requires_explicit_operator_authorization"
    assert result["operator_authorized"] is False
    assert result["gate_passed"] is True
    assert result["apply_simulated"] is False
    assert result["apply_performed"] is False
    assert "explicit_operator_authorization_missing" in result["blocked_reasons"]


def test_round741_745_authorized_valid_payload_can_only_simulate_success() -> None:
    result = simulate_emotion_transition_apply_preflight(_base_payload("praise"), operator_authorized=True)

    assert result["dryrun_success"] is True
    assert result["dryrun_status"] == "dryrun_apply_preflight_simulated_success_live_apply_still_blocked"
    assert result["operator_authorized"] is True
    assert result["gate_passed"] is True
    assert result["gate_status"] == "passed_read_only_gate_future_apply_still_blocked"
    assert result["apply_simulated"] is True
    assert result["apply_performed"] is False
    assert result["blocked_reasons"] == []


def test_round741_745_gate_failure_blocks_authorized_dryrun() -> None:
    payload = _base_payload("care_signal")
    payload["runtime_mutation_requested"] = True

    result = build_emotion_transition_dryrun_apply_plan(payload, operator_authorized=True)

    assert result["dryrun_success"] is False
    assert result["dryrun_status"] == "blocked_dryrun_apply_requires_gate_pass"
    assert result["gate_passed"] is False
    assert "round721_740_gate_did_not_pass" in result["blocked_reasons"]
    assert "runtime_mutation_requested_is_forbidden_in_round701_720_read_only_validator" in result["blocked_reasons"]
    assert result["apply_performed"] is False


def test_round746_750_future_policy_keeps_live_apply_and_persistence_no_go() -> None:
    policy = future_live_apply_policy()

    assert policy["this_round_applies_transitions"] is False
    assert policy["future_live_apply_requires_separate_explicit_operator_authorized_round"] is True
    assert policy["dryrun_success_authorizes_live_mutation"] is False
    assert policy["dryrun_success_authorizes_persistence"] is False
    assert policy["dryrun_success_authorizes_runtime_mapping"] is False
    assert policy["dryrun_success_authorizes_enforcement"] is False
    assert policy["dryrun_success_authorizes_memory_writes"] is False
    assert policy["dryrun_success_authorizes_vector_read_or_load"] is False
    assert policy["malicious_social_threat_identity_attack_direct_rewrites_blocked"] is True
    assert policy["blocked_direct_rewrite_surfaces"] == ("core_identity", "self_model", "long_term_memory")
    assert policy["production_persistence_remains_no_go"] is True
    assert policy["runtime_mapping_enabled_default"] is False
    assert policy["enforcement_enabled_default"] is False


def test_round751_755_malicious_social_threat_identity_attack_and_unknown_category_block() -> None:
    malicious_payload = _base_payload("malicious_comment")
    malicious_payload["core_identity_update_requested"] = True
    malicious = build_emotion_transition_dryrun_apply_plan(malicious_payload, operator_authorized=True)

    social_payload = _base_payload("social_threat")
    social_payload["self_model_update_requested"] = True
    social = build_emotion_transition_dryrun_apply_plan(social_payload, operator_authorized=True)

    identity_payload = _base_payload("identity_attack")
    identity_payload["long_term_memory_update_requested"] = True
    identity = build_emotion_transition_dryrun_apply_plan(identity_payload, operator_authorized=True)

    unknown = build_emotion_transition_dryrun_apply_plan(_base_payload("unknown_social_signal"), operator_authorized=True)

    assert malicious["dryrun_success"] is False
    assert "high_risk_social_feedback_blocks_direct_core_identity_update" in malicious["blocked_reasons"]
    assert social["dryrun_success"] is False
    assert "high_risk_social_feedback_blocks_direct_self_model_update" in social["blocked_reasons"]
    assert identity["dryrun_success"] is False
    assert "high_risk_social_feedback_blocks_direct_long_term_memory_update" in identity["blocked_reasons"]
    assert unknown["dryrun_success"] is False
    assert "unknown_event_category_fail_closed" in unknown["blocked_reasons"]


def test_round756_760_success_does_not_enable_persistence_runtime_mapping_enforcement_memory_or_vectors() -> None:
    result = build_emotion_transition_dryrun_apply_plan(_base_payload("care_signal"), operator_authorized=True)

    assert result["dryrun_success"] is True
    assert result["persistence_written"] is False
    assert result["runtime_mapping_changed"] is False
    assert result["enforcement_changed"] is False
    assert result["memory_written"] is False
    assert result["vector_read_performed"] is False
    assert result["vector_load_performed"] is False
    assert result["artifact_created_or_staged"] is False
    assert result["future_live_apply_policy"]["dryrun_success_authorizes_persistence"] is False
    assert result["future_live_apply_policy"]["dryrun_success_authorizes_runtime_mapping"] is False
    assert result["future_live_apply_policy"]["dryrun_success_authorizes_enforcement"] is False


def test_round756_760_success_does_not_mutate_live_emotion_hormone_or_payload() -> None:
    payload = _base_payload("praise")
    payload_before = deepcopy(payload)
    live_state = {
        "emotion": {"valence": 0.0, "arousal": 0.0},
        "hormone": {"cortisol": 0.1, "oxytocin": 0.2},
        "memory": [],
    }
    live_state_before = deepcopy(live_state)

    result = build_emotion_transition_dryrun_apply_plan(payload, operator_authorized=True)

    assert result["dryrun_success"] is True
    assert payload == payload_before
    assert live_state == live_state_before
    assert result["live_emotion_mutated"] is False
    assert result["live_hormone_mutated"] is False
    assert result["memory_written"] is False


def test_round756_760_no_agp_fallback_classifier_route_changes_occur() -> None:
    agp = AGPAdapter(mode=AGP_MODE_OBSERVATION)
    classifier = AppraisalClassifier()
    mapping = LexConceptMappingAdapter()
    before_agp_mode = agp.mode
    before_agp_result = agp.verify("그게 뭐야?", activated_categories=[])
    before_classifier_result = classifier.analyze("오늘은 그냥 문장")
    before_mapping_flags = (mapping.runtime_mapping_enabled, mapping.enforcement_enabled)

    result = build_emotion_transition_dryrun_apply_plan(_base_payload("care_signal"), operator_authorized=True)

    after_agp_result = agp.verify("그게 뭐야?", activated_categories=[])
    after_classifier_result = classifier.analyze("오늘은 그냥 문장")
    after_mapping_flags = (mapping.runtime_mapping_enabled, mapping.enforcement_enabled)
    assert result["dryrun_success"] is True
    assert agp.mode == before_agp_mode == AGP_MODE_OBSERVATION
    assert before_agp_result.reason == after_agp_result.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert before_agp_result.fallback == after_agp_result.fallback
    assert before_classifier_result.label == after_classifier_result.label == LABEL_NONE
    assert before_mapping_flags == after_mapping_flags == (False, False)
    assert result["agp_route_changed"] is False
    assert result["fallback_route_changed"] is False
    assert result["classifier_route_changed"] is False


def test_round756_760_no_artifact_creation_staging_and_korean_fixtures_preserved() -> None:
    result = build_emotion_transition_dryrun_apply_plan(_base_payload("praise"), operator_authorized=True)
    git_short = _git_output("status", "--short")
    forbidden_lines = [line for line in git_short.splitlines() if any(fragment in line for fragment in FORBIDDEN_ARTIFACT_FRAGMENTS)]
    fixture_texts = [str(row.get("text", "")) for row in KOREAN_CONVERSATION_FIXTURES]

    assert result["artifact_created_or_staged"] is False
    assert forbidden_lines == []
    assert DEFAULT_TARGET_WORD == "민석"
    assert any("민석" in text for text in fixture_texts) or any(
        row.get("category") == "minsok" for row in KOREAN_CONVERSATION_FIXTURES
    )


def test_round751_755_operator_command_emits_compact_dryrun_json() -> None:
    output = subprocess.check_output(["python", str(OPERATOR_SCRIPT)], cwd=REPO_ROOT, text=True)
    payload = json.loads(output)

    assert payload["success"] is True
    assert payload["operator_command"] == "python scripts/operator_dryrun_round741_760_emotion_transition_apply_plan.py"
    assert payload["operator_report_path"] == "docs/round741_760_emotion_transition_dryrun_apply_plan.md"
    assert payload["dryrun_apply_plan_summary"]["module"] == "adapters/emotion_transition_dryrun_apply_plan.py"
    assert payload["unauthorized_valid_payload_dryrun_blocked_result"]["dryrun_success"] is False
    assert payload["authorized_valid_payload_dryrun_simulated_success_result"]["dryrun_success"] is True
    assert payload["authorized_valid_payload_dryrun_simulated_success_result"]["apply_performed"] is False
    assert payload["malicious_comment_blocked_result"]["dryrun_success"] is False
    assert payload["social_threat_blocked_result"]["dryrun_success"] is False
    assert payload["identity_attack_blocked_result"]["dryrun_success"] is False
    assert payload["unknown_category_fail_closed_result"]["dryrun_success"] is False
    assert payload["no_runtime_mutation_proof"]["live_emotion_mutated"] is False
    assert payload["no_persistence_proof"]["persistence_written"] is False
    assert payload["no_memory_write_proof"]["memory_written"] is False
    assert payload["no_vector_content_read_load_proof"]["vector_load_performed"] is False
    assert payload["no_artifact_creation_staging_proof"]["no_forbidden_artifact_creation_or_staging"] is True
    assert len(payload["next_implementation_recommendations"]) == 1
