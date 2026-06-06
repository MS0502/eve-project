"""Round681-700 read-only emotion-state transition contract invariants."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from adapters.agp_adapter import AGPAdapter, AGP_MODE_OBSERVATION, AGP_REASON_UNKNOWN_CATEGORY
from adapters.appraisal_classifier import AppraisalClassifier, LABEL_NONE
from adapters.emotion_state_transition_contract import (
    EMOTION_FUNCTIONS,
    FEATURE_TRACK,
    NEXT_IMPLEMENTATION_RECOMMENDATION,
    emotion_transition_contract,
    social_feedback_category_map,
    validate_contract_shape,
)
from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES

REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "operator_report_round681_700_emotion_transition_contract.py"
DOC_PATH = REPO_ROOT / "docs" / "round681_700_emotion_transition_contract.md"
FORBIDDEN_ARTIFACT_FRAGMENTS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets",
    ".zip",
    ".part",
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def test_round681_685_contract_exists_and_represents_required_emotion_functions() -> None:
    contract = emotion_transition_contract()
    validation = validate_contract_shape(contract)

    assert contract["feature_track"] == FEATURE_TRACK
    assert validation["success"] is True
    assert tuple(contract["emotion_functions"]) == EMOTION_FUNCTIONS
    rows = {row["function"]: row for row in contract["transition_contract_rows"]}
    assert set(rows) == set(EMOTION_FUNCTIONS)
    for required in EMOTION_FUNCTIONS:
        assert rows[required]["read_only_now"] is True
        assert rows[required]["future_state_delta_shape"]
        assert rows[required]["validation_rule"]

    inertness = contract["inertness_guarantees"]
    assert inertness["runtime_mutation_enabled"] is False
    assert inertness["persistence_write_enabled"] is False
    assert inertness["live_hormone_update_enabled"] is False
    assert inertness["live_emotion_update_enabled"] is False
    assert inertness["memory_write_enabled"] is False
    assert inertness["agp_route_change_enabled"] is False
    assert inertness["fallback_route_change_enabled"] is False


def test_round686_690_social_feedback_categories_and_quarantine_rules_exist() -> None:
    categories = social_feedback_category_map()

    assert set(categories) == {
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
    for row in categories.values():
        assert row["possible_emotional_impact"]
        assert row["quarantine_required"] is True
        assert row["core_identity_update_forbidden"] is True
        assert row["long_term_memory_update_requires_appraisal"] is True
        assert isinstance(row["future_behavior_may_be_influenced"], bool)
        assert isinstance(row["recovery_loop_may_be_required"], bool)

    assert categories["malicious_comment"]["core_identity_update_forbidden"] is True
    assert categories["identity_attack"]["core_identity_update_forbidden"] is True
    assert categories["malicious_comment"]["future_behavior_may_be_influenced"] is False
    assert categories["identity_attack"]["future_behavior_may_be_influenced"] is False


def test_round686_700_malicious_feedback_cannot_directly_rewrite_identity_or_memory() -> None:
    contract = emotion_transition_contract()
    malicious_rule = contract["malicious_feedback_quarantine_rule"]
    identity_rule = contract["core_identity_protection_rule"]

    assert malicious_rule["quarantine_required"] is True
    assert malicious_rule["direct_core_identity_rewrite_allowed"] is False
    assert malicious_rule["direct_self_model_update_allowed"] is False
    assert malicious_rule["direct_long_term_memory_write_allowed"] is False
    assert malicious_rule["future_processing_requirement"] == "quarantine_then_appraisal_before_any_memory_or_self_model_update"
    assert identity_rule["core_identity_update_from_raw_social_feedback_allowed"] is False
    assert identity_rule["minseok_not_owner_command_center"] is True
    assert identity_rule["eve_not_user_serving_agent"] is True


def test_round691_695_empathy_contract_matches_constitution_definition() -> None:
    empathy = emotion_transition_contract()["empathy_contract"]

    assert empathy["definition"] == "other_state_inference_plus_relationship_aware_action_selection"
    assert set(empathy["required_components"]) == {
        "other_state_inference",
        "relationship_aware_action_selection",
    }
    assert "mere_comforting_phrase_generation" in empathy["forbidden_shortcuts"]
    assert empathy["runtime_mutation_enabled"] is False


def test_round691_695_operator_report_emits_required_read_only_json() -> None:
    output = subprocess.check_output(["python", str(OPERATOR_SCRIPT)], cwd=REPO_ROOT, text=True)
    payload = json.loads(output)

    assert payload["success"] is True
    assert payload["operator_command"] == "python scripts/operator_report_round681_700_emotion_transition_contract.py"
    assert payload["operator_report_path"] == "docs/round681_700_emotion_transition_contract.md"
    assert payload["emotion_transition_contract"]["feature_track"] == FEATURE_TRACK
    assert payload["malicious_feedback_quarantine_rule"]["direct_core_identity_rewrite_allowed"] is False
    assert payload["core_identity_protection_rule"]["core_identity_update_from_raw_social_feedback_allowed"] is False
    assert payload["no_runtime_mutation_proof"]["runtime_mutation_performed"] is False
    assert payload["no_persistence_proof"]["persistence_write_performed"] is False
    assert payload["no_vector_content_read_proof"]["vector_contents_read"] is False
    assert payload["no_runtime_load_proof"]["vectors_loaded"] is False
    assert payload["no_artifact_creation_staging_proof"]["no_forbidden_artifact_creation_or_staging"] is True
    assert payload["next_implementation_recommendations"] == [NEXT_IMPLEMENTATION_RECOMMENDATION]


def test_round696_700_no_runtime_behavior_changes_or_live_state_mutation() -> None:
    classifier = AppraisalClassifier()
    agp = AGPAdapter()
    mapping = LexConceptMappingAdapter()

    before_user_affect = classifier.analyze("오늘 기분 좋아")
    before_unknown = agp.verify(
        candidate_response="새 범주를 말해.",
        meaning={"meaning_categories": ["novel_category"], "meaning_categories_are_explicit": True},
        activated_categories=["appraisal"],
        hormone_state={},
    )

    contract = emotion_transition_contract()
    validation = validate_contract_shape(contract)

    after_user_affect = classifier.analyze("오늘 기분 좋아")
    after_unknown = agp.verify(
        candidate_response="새 범주를 말해.",
        meaning={"meaning_categories": ["novel_category"], "meaning_categories_are_explicit": True},
        activated_categories=["appraisal"],
        hormone_state={},
    )

    assert validation["runtime_mutation_performed"] is False
    assert validation["persistence_write_performed"] is False
    assert before_user_affect == after_user_affect
    assert after_user_affect.label == LABEL_NONE
    assert after_user_affect.route_override is None
    assert before_unknown == after_unknown
    assert after_unknown.passed is False
    assert after_unknown.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert agp.fallback_to_surface(after_unknown) is not None
    assert agp.mode == AGP_MODE_OBSERVATION
    assert agp.category_threshold == 0.3
    assert agp.hormone_threshold == 0.5
    assert mapping.runtime_mapping_enabled is False
    assert mapping.enforcement_enabled is False
    assert LexConceptMappingAdapter.runtime_mapping_enabled is False
    assert LexConceptMappingAdapter.enforcement_enabled is False


def test_round696_700_no_persistence_runtime_mapping_enforcement_vector_or_artifact_enablement() -> None:
    contract = emotion_transition_contract()
    inertness = contract["inertness_guarantees"]
    status_lines = [line for line in _git_output("status", "--short").splitlines() if line.strip()]
    forbidden_status_entries = [
        line for line in status_lines if any(fragment in line for fragment in FORBIDDEN_ARTIFACT_FRAGMENTS)
    ]
    changed_forbidden = [
        line.strip()
        for line in _git_output(
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "_operator_artifacts",
            "vectors.npy",
            "vocab.txt",
            "subset_manifest.json",
            "seeds/subsets",
            "*.zip",
            "*.part",
        ).splitlines()
        if line.strip()
    ]

    assert inertness["production_persistence_enabled"] is False
    assert inertness["runtime_mapping_enabled_default"] is False
    assert inertness["enforcement_enabled"] is False
    assert inertness["vector_contents_read"] is False
    assert inertness["vectors_loaded"] is False
    assert inertness["operator_artifact_creation_enabled"] is False
    assert changed_forbidden == []
    assert forbidden_status_entries == []
    assert "no vector content read proof" in DOC_PATH.read_text(encoding="utf-8")
    assert "no runtime load proof" in DOC_PATH.read_text(encoding="utf-8")


def test_round696_700_preserves_korean_fixtures_and_minsok_literal_exactly() -> None:
    assert [row["text"] for row in KOREAN_CONVERSATION_FIXTURES] == [
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
    assert [row for row in KOREAN_CONVERSATION_FIXTURES if row["category"] == "minsok"] == [
        {"category": "minsok", "text": "군대 생활 어때"},
        {"category": "minsok", "text": "코딩 좋아해"},
        {"category": "minsok", "text": "EVE 프로젝트"},
    ]
    assert "민석" in (REPO_ROOT / "main.py").read_text(encoding="utf-8")
