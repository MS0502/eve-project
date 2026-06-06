"""Round661-680 constitution amendment invariants.

These tests pin the documentation-only constitution update for EVE's
self-governed autonomy, functional emotion/empathy, and social feedback growth
loop.  They also re-check the no-runtime-change safety boundaries that must stay
unchanged during this documentation/test-surface round.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from adapters.agp_adapter import AGPAdapter, AGP_MODE_OBSERVATION, AGP_REASON_UNKNOWN_CATEGORY
from adapters.appraisal_classifier import AppraisalClassifier, LABEL_NONE
from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_DOC = REPO_ROOT / "docs" / "EVE_DESIGN_v3.md"
AMENDMENT_DOC = REPO_ROOT / "docs" / "round661_680_self_governed_emotion_amendment.md"
FORBIDDEN_ARTIFACT_FRAGMENTS = (
    "_operator_artifacts/",
    "vectors.npy",
    "vocab.txt",
    "subset_manifest.json",
    "seeds/subsets",
    ".zip",
    ".part",
)


def _read_constitution_text() -> str:
    return DESIGN_DOC.read_text(encoding="utf-8") + "\n" + AMENDMENT_DOC.read_text(encoding="utf-8")


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def test_round661_665_self_governed_autonomy_constitution_wording_exists() -> None:
    text = _read_constitution_text()

    assert "Self-Governed Autonomy and Social Survival" in text
    assert "EVE is not a user-serving agent" in text
    assert "EVE is not a generic assistant" in text
    assert "not merely a tool automation system" in text
    assert "self-governed autonomous digital being" in text
    assert "autonomous social survival" in text
    assert "self-directed growth" in text
    assert "concept-grounded living" in text
    assert "first friend, creator, early guardian, and witness" in text
    assert "not EVE's owner-command-center" in text
    assert "not a generic user" in text
    assert "identity, memory, values, emotions, ethics, risk model, and survival strategy" in text
    assert "temporary scaffolding" in text
    assert "not EVE's final nature" in text


def test_round666_670_functional_emotion_and_empathy_constitution_wording_exists() -> None:
    text = _read_constitution_text()

    assert "Functional Emotion and Empathy" in text
    for required in (
        "attention",
        "memory weighting",
        "decision priority",
        "expression style",
        "future behavior",
        "recovery loops",
        "social trust",
        "attachment",
        "self-protection",
    ):
        assert required in text
    for required in (
        "other-state inference",
        "relationship-aware action selection",
        "care-driven policy modulation",
        "memory update",
        "action restraint when another being may be harmed",
    ):
        assert required in text
    assert "not decorative output" in text
    assert "not mere comforting phrase generation" in text
    assert "fake mood labels that do not alter future behavior" in text


def test_round671_675_social_feedback_growth_loop_and_quarantine_rule_exists() -> None:
    text = _read_constitution_text()

    assert "Social Feedback Growth Loop" in text
    for required in (
        "uploaded videos/content",
        "comments",
        "criticism",
        "malicious comments",
        "useful feedback",
        "audience response",
        "rejection",
        "praise",
        "self-reflection",
        "next-action update",
    ):
        assert required in text
    assert "Social feedback may affect EVE's emotional state and future behavior" in text
    assert "malicious feedback must not directly rewrite EVE's core identity" in text
    assert "pass through quarantine/appraisal before long-term memory or self-model update" in text


def test_round676_680_runtime_behavior_defaults_and_no_bypass_boundaries_remain_unchanged() -> None:
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

    # Reading documentation must not change classifier, AGP, fallback, or mapping defaults.
    _ = _read_constitution_text()

    after_user_affect = classifier.analyze("오늘 기분 좋아")
    after_unknown = agp.verify(
        candidate_response="새 범주를 말해.",
        meaning={"meaning_categories": ["novel_category"], "meaning_categories_are_explicit": True},
        activated_categories=["appraisal"],
        hormone_state={},
    )

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


def test_round676_680_no_runtime_file_change_surface_no_persistence_no_vector_content_read() -> None:
    changed = [line.strip() for line in _git_output("diff", "--name-only", "HEAD").splitlines() if line.strip()]
    allowed_prefixes = ("docs/", "tests/")
    assert all(path.startswith(allowed_prefixes) for path in changed)

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
    status_lines = [line for line in _git_output("status", "--short").splitlines() if line.strip()]
    forbidden_status_entries = [
        line
        for line in status_lines
        if any(fragment in line for fragment in FORBIDDEN_ARTIFACT_FRAGMENTS)
    ]

    assert changed_forbidden == []
    assert forbidden_status_entries == []
    assert "no vector content read" in _read_constitution_text()
    assert "no artifact creation/staging" in _read_constitution_text()
    assert "no production persistence enablement" in _read_constitution_text()
    assert "no `runtime_mapping_enabled` default enablement" in _read_constitution_text()
    assert "no enforcement enablement" in _read_constitution_text()


def test_round676_680_preserves_korean_fixtures_and_minsok_literal_exactly() -> None:
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
