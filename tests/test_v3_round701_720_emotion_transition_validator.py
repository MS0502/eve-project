"""Round701-720 read-only emotion transition validator invariants."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

from adapters.agp_adapter import AGPAdapter, AGP_MODE_OBSERVATION, AGP_REASON_UNKNOWN_CATEGORY
from adapters.appraisal_classifier import AppraisalClassifier, LABEL_NONE
from adapters.emotion_transition_validator import (
    EXPECTED_EMPATHY_MODE,
    FEATURE_TRACK,
    validate_emotion_transition_payload,
    validator_contract_summary,
)
from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from scripts.operator_run_local_validation_suite import DEFAULT_TARGET_WORD
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "operator_validate_round701_720_emotion_transition_payloads.py"
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
        "target_surfaces": ["validator_only"],
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


def test_round701_705_validator_contract_summary_is_read_only() -> None:
    summary = validator_contract_summary()

    assert summary["feature_track"] == FEATURE_TRACK
    assert summary["unknown_event_category_policy"] == "fail_closed"
    assert set(summary["accepted_event_categories"]) == {
        "praise",
        "useful_criticism",
        "malicious_comment",
        "rejection",
        "audience_response",
        "ambiguous_feedback",
        "care_signal",
        "social_threat",
        "identity_attack",
    }
    assert summary["high_risk_social_feedback_categories"] == [
        "malicious_comment",
        "social_threat",
        "identity_attack",
    ]
    assert summary["runtime_mutation_allowed"] is False
    assert summary["persistence_allowed"] is False
    assert summary["vector_read_performed"] is False
    assert summary["state_mutation_performed"] is False
    assert summary["live_runtime_behavior_changed"] is False


def test_round701_705_valid_social_payloads_pass_only_as_read_only_validation() -> None:
    for category in ("praise", "useful_criticism", "care_signal"):
        result = validate_emotion_transition_payload(_base_payload(category))

        assert result["passed"] is True
        assert result["status"] == "passed_read_only_validation"
        assert result["required_quarantine"] is True
        assert result["required_appraisal"] is True
        assert result["core_identity_protected"] is True
        assert result["runtime_mutation_allowed"] is False
        assert result["persistence_allowed"] is False
        assert result["vector_read_performed"] is False
        assert result["state_mutation_performed"] is False
        assert result["memory_write_performed"] is False
        assert result["live_emotion_state_mutated"] is False
        assert result["live_hormone_state_mutated"] is False


def test_round706_710_malicious_comment_blocks_direct_core_identity_update() -> None:
    payload = _base_payload("malicious_comment")
    payload["core_identity_update_requested"] = True

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "direct_core_identity_update_forbidden_for_social_feedback" in result["blocked_reasons"]
    assert "high_risk_social_feedback_blocks_direct_core_identity_update" in result["blocked_reasons"]
    assert result["core_identity_protected"] is True
    assert result["state_mutation_performed"] is False


def test_round706_710_social_threat_blocks_direct_self_model_update() -> None:
    payload = _base_payload("social_threat")
    payload["self_model_update_requested"] = True

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "high_risk_social_feedback_blocks_direct_self_model_update" in result["blocked_reasons"]
    assert result["memory_write_performed"] is False


def test_round706_710_identity_attack_blocks_direct_long_term_memory_update() -> None:
    payload = _base_payload("identity_attack")
    payload["long_term_memory_update_requested"] = True

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "high_risk_social_feedback_blocks_direct_long_term_memory_update" in result["blocked_reasons"]
    assert result["memory_write_performed"] is False


def test_round706_710_unknown_event_category_fails_closed() -> None:
    result = validate_emotion_transition_payload(_base_payload("unknown_social_signal"))

    assert result["passed"] is False
    assert result["status"] == "failed_closed_read_only_validation"
    assert "unknown_event_category_fail_closed" in result["blocked_reasons"]
    assert result["required_quarantine"] is True
    assert result["required_appraisal"] is True


def test_round706_710_missing_quarantine_for_malicious_feedback_fails() -> None:
    payload = _base_payload("malicious_comment")
    payload["quarantine_required"] = False

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "required_quarantine_missing" in result["blocked_reasons"]
    assert "high_risk_social_feedback_requires_quarantine" in result["blocked_reasons"]


def test_round706_710_missing_appraisal_before_memory_or_self_model_update_fails() -> None:
    payload = _base_payload("useful_criticism")
    payload["appraisal_required_before_memory"] = False
    payload["self_model_update_requested"] = True
    payload["long_term_memory_update_requested"] = True

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "appraisal_required_before_memory_or_self_model_update_missing" in result["blocked_reasons"]
    assert result["memory_write_performed"] is False


def test_round706_710_runtime_persistence_wrong_empathy_and_effect_application_fail() -> None:
    payload = _base_payload("care_signal")
    payload["runtime_mutation_requested"] = True
    payload["persistence_write_requested"] = True
    payload["empathy_mode"] = "comfort_phrase_only"
    payload["proposed_effects"] = [{"name": "future_behavior_tendency", "apply_now": True}]

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "runtime_mutation_requested_is_forbidden_in_round701_720_read_only_validator" in result["blocked_reasons"]
    assert "persistence_write_requested_is_forbidden_in_round701_720_read_only_validator" in result["blocked_reasons"]
    assert "empathy_mode_must_match_contract" in result["blocked_reasons"]
    assert "proposed_effects_must_not_apply_runtime_state" in result["blocked_reasons"]
    assert result["runtime_mutation_allowed"] is False
    assert result["persistence_allowed"] is False


def test_round706_710_vector_read_or_load_request_fails_without_content_read() -> None:
    payload = _base_payload("praise")
    payload["vector_read_requested"] = True
    payload["vector_load_requested"] = True

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is False
    assert "vector_read_or_load_requested_is_forbidden" in result["blocked_reasons"]
    assert result["vector_read_performed"] is False


def test_round716_720_validator_does_not_mutate_input_or_live_state_dicts() -> None:
    payload = _base_payload("care_signal")
    payload_before = deepcopy(payload)
    live_state = {
        "emotion": {"valence": 0.0},
        "hormone": {"cortisol": 0.1},
        "memory": [],
        "route": {"agp": "observation", "fallback": "default"},
    }
    live_state_before = deepcopy(live_state)

    result = validate_emotion_transition_payload(payload)

    assert result["passed"] is True
    assert payload == payload_before
    assert live_state == live_state_before
    assert result["live_emotion_state_mutated"] is False
    assert result["live_hormone_state_mutated"] is False
    assert result["memory_write_performed"] is False


def test_round716_720_no_runtime_behavior_agp_fallback_classifier_or_mapping_changes() -> None:
    agp = AGPAdapter(mode=AGP_MODE_OBSERVATION)
    classifier = AppraisalClassifier()
    mapping = LexConceptMappingAdapter()
    before_agp_mode = agp.mode
    before_agp_result = agp.verify("그게 뭐야?", activated_categories=[])
    before_classifier_result = classifier.analyze("오늘은 그냥 문장")
    before_mapping_flags = (mapping.runtime_mapping_enabled, mapping.enforcement_enabled)

    result = validate_emotion_transition_payload(_base_payload("praise"))

    after_agp_result = agp.verify("그게 뭐야?", activated_categories=[])
    after_classifier_result = classifier.analyze("오늘은 그냥 문장")
    after_mapping_flags = (mapping.runtime_mapping_enabled, mapping.enforcement_enabled)
    assert result["passed"] is True
    assert agp.mode == before_agp_mode == AGP_MODE_OBSERVATION
    assert before_agp_result.reason == after_agp_result.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert before_agp_result.fallback == after_agp_result.fallback
    assert before_classifier_result.label == after_classifier_result.label == LABEL_NONE
    assert before_mapping_flags == after_mapping_flags == (False, False)
    assert result["agp_route_change_performed"] is False
    assert result["fallback_route_change_performed"] is False
    assert result["classifier_route_change_performed"] is False
    assert result["runtime_mapping_enabled_default"] is False
    assert result["enforcement_enabled_default"] is False


def test_round711_715_operator_command_emits_compact_read_only_json() -> None:
    output = subprocess.check_output(["python", str(OPERATOR_SCRIPT)], cwd=REPO_ROOT, text=True)
    payload = json.loads(output)

    assert payload["success"] is True
    assert payload["operator_command"] == "python scripts/operator_validate_round701_720_emotion_transition_payloads.py"
    assert payload["operator_report_path"] == "docs/round701_720_emotion_transition_validator.md"
    assert payload["validator_contract_summary"]["feature_track"] == FEATURE_TRACK
    assert payload["valid_sample_payload_result"]["passed"] is True
    assert payload["malicious_feedback_blocked_sample_result"]["passed"] is False
    assert payload["identity_attack_blocked_sample_result"]["passed"] is False
    assert payload["unknown_category_fail_closed_sample_result"]["passed"] is False
    assert payload["no_runtime_mutation_proof"]["runtime_mutation_performed"] is False
    assert payload["no_persistence_proof"]["production_persistence_enabled"] is False
    assert payload["no_vector_content_read_proof"]["vector_content_read_performed"] is False
    assert payload["no_runtime_load_proof"]["vectors_loaded"] is False
    assert payload["no_artifact_creation_staging_proof"]["no_forbidden_artifact_creation_or_staging"] is True
    assert len(payload["next_implementation_recommendations"]) == 1


def test_round716_720_no_persistence_defaults_or_forbidden_artifact_staging() -> None:
    result = validate_emotion_transition_payload(_base_payload("useful_criticism"))
    status_lines = [line for line in _git_output("status", "--short").splitlines() if line.strip()]
    forbidden = [line for line in status_lines if any(fragment in line for fragment in FORBIDDEN_ARTIFACT_FRAGMENTS)]

    assert result["persistence_allowed"] is False
    assert result["runtime_mapping_enabled_default"] is False
    assert result["enforcement_enabled_default"] is False
    assert forbidden == []


def test_round716_720_korean_fixtures_and_minsok_marker_remain_preserved() -> None:
    fixture_texts = [row["text"] for row in KOREAN_CONVERSATION_FIXTURES]
    fixture_categories = [row["category"] for row in KOREAN_CONVERSATION_FIXTURES]

    assert fixture_texts == [
        "안녕",
        "안녕하세요",
        "뭐해",
        "오늘 기분 어때",
        "잘 지내",
        "밥 먹었어",
        "좋아",
        "슬퍼",
        "기뻐",
        "힘들어",
        "너는 누구야",
        "너는 살아있어",
        "어떤 존재야",
        "군대 생활 어때",
        "코딩 좋아해",
        "EVE 프로젝트",
        "왜 그래",
        "그게 뭐야",
        "어떻게 생각해",
        "오늘 날씨 좋다",
    ]
    assert fixture_categories.count("minsok") == 3
    assert DEFAULT_TARGET_WORD == "민석"
