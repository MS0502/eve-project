"""EVE v3 Round641-660 — read-only AppraisalClassifier / AGP input audit tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from adapters.agp_adapter import AGPAdapter, AGP_MODE_OBSERVATION, AGP_REASON_UNKNOWN_CATEGORY
from adapters.appraisal_classifier import AppraisalClassifier, LABEL_NONE
from scripts import operator_audit_round641_660_appraisal_agp_input as round641_660
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("_operator_artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_declared_path_placeholders(root: Path) -> None:
    paths = sorted(
        {path for group in round641_660.RELEVANT_MODULES.values() for path in group}
        | set(round641_660.RELEVANT_TESTS)
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder for path-metadata-only audit test\n", encoding="utf-8")


def _write_medium30k_sentinels(root: Path) -> dict[Path, bytes]:
    base = root / "_operator_artifacts" / "subset_medium_30k"
    base.mkdir(parents=True, exist_ok=True)
    files = {
        base / "vocab.txt": b"operator-local vocab sentinel; must not be parsed\n",
        base / "subset_manifest.json": b'{"sentinel":"must not be parsed"}\n',
        base / "vectors.npy": b"not-a-real-numpy-file; contents must not be read",
    }
    for path, content in files.items():
        path.write_bytes(content)
    return files


def test_round641_645_inventory_documents_classifier_agp_fallback_and_korean_fixtures() -> None:
    report = round641_660.build_round641_645_boundary_inventory()

    assert report["version"] == "v3_round641_645_boundary_inventory"
    assert report["success"] is True
    assert report["feature_track"] == "read_only_appraisal_classifier_agp_input_stabilization_audit"
    assert "adapters/appraisal_classifier.py" in report["relevant_modules"]["appraisal_classifier"]
    assert "adapters/agp_adapter.py" in report["relevant_modules"]["agp_input_and_verification"]
    assert "adapters/speech_hub.py" in report["relevant_modules"]["generation_meaning_bridge"]
    assert "adapters/compositor_adapter.py" in report["relevant_modules"]["generation_meaning_bridge"]
    assert report["current_input_classification_path"] == [
        "orchestrator_adapter._analyze_appraisal_statement(text)",
        "AppraisalClassifier.analyze(text)",
        "AppraisalResult.as_meaning_dict()",
        "orchestrator trace appraisal payload / small_talk route_override when target appraisal",
    ]
    assert report["current_agp_input_verification_path"][0] == "AGPAdapter.accept_input_meaning(meaning) normalizes appraisal meaning without veto"
    assert report["fallback_path"][0] == "AGPAdapter.verify returns AGPResult fallback data on failed anchor/hormone checks"
    assert report["semantic_guard_compatibility"]["frozen_marker_expansion_allowed"] is False
    assert report["semantic_guard_compatibility"]["new_keyword_lists_added"] is False
    assert report["korean_fixture_coverage"]["fixture_count"] == 20
    assert report["korean_fixture_coverage"]["minsok_rows"] == [
        {"category": "minsok", "text": "군대 생활 어때"},
        {"category": "minsok", "text": "코딩 좋아해"},
        {"category": "minsok", "text": "EVE 프로젝트"},
    ]


def test_round646_650_audit_is_read_only_and_does_not_read_vector_artifacts(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_declared_path_placeholders(tmp_path)
    sentinels = _write_medium30k_sentinels(tmp_path)
    before = {path: path.read_bytes() for path in sentinels}

    report = round641_660.build_round646_650_read_only_audit_report(repo_root=tmp_path)

    after = {path: path.read_bytes() for path in sentinels}
    assert after == before
    assert report["version"] == "v3_round646_650_read_only_audit_report"
    assert report["success"] is True
    assert report["checks"] == {
        "classifier_boundary_documented": True,
        "agp_input_boundary_documented": True,
        "fallback_boundary_documented": True,
        "semantic_guard_compatibility_documented": True,
        "no_behavior_mutation": True,
        "no_persistence": True,
        "no_vector_content_read": True,
        "no_runtime_load": True,
        "no_artifact_creation_or_staging": True,
    }
    assert report["artifact_safety"]["forbidden_entries"] == []
    assert report["artifact_safety"]["artifacts_created_by_workflow"] is False
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False
    assert report["vectors_created"] is False


def test_round651_655_regression_surface_pins_no_behavior_change_no_bypass_and_defaults() -> None:
    classifier = AppraisalClassifier()
    agp = AGPAdapter()
    before_positive = classifier.analyze("오늘 날씨 좋다")
    before_user_affect = classifier.analyze("오늘 기분 좋아")
    before_unknown = agp.verify(
        candidate_response="새 범주를 말해.",
        meaning={"meaning_categories": ["novel_category"], "meaning_categories_are_explicit": True},
        activated_categories=["appraisal"],
        hormone_state={},
    )

    report = round641_660.build_round651_655_regression_test_surface()

    after_positive = classifier.analyze("오늘 날씨 좋다")
    after_user_affect = classifier.analyze("오늘 기분 좋아")
    after_unknown = agp.verify(
        candidate_response="새 범주를 말해.",
        meaning={"meaning_categories": ["novel_category"], "meaning_categories_are_explicit": True},
        activated_categories=["appraisal"],
        hormone_state={},
    )
    assert report["version"] == "v3_round651_655_audit_regression_test_surface"
    assert report["success"] is True
    assert report["checks"]["audit_is_read_only"] is True
    assert report["checks"]["runtime_behavior_unchanged"] is True
    assert before_positive == after_positive
    assert before_user_affect == after_user_affect
    assert before_user_affect.label == LABEL_NONE
    assert before_user_affect.route_override is None
    assert agp.mode == AGP_MODE_OBSERVATION
    assert agp.category_threshold == 0.3
    assert agp.hormone_threshold == 0.5
    assert before_unknown == after_unknown
    assert after_unknown.passed is False
    assert after_unknown.reason == AGP_REASON_UNKNOWN_CATEGORY
    assert agp.fallback_to_surface(after_unknown) is not None


def test_round656_660_feature_entry_report_chooses_exactly_one_next_step_and_forbids_mutation() -> None:
    report = round641_660.build_round656_660_feature_entry_report()

    assert report["version"] == "v3_round656_660_feature_entry_report"
    assert report["success"] is True
    assert report["recommended_next_implementation_steps"] == [
        "add_read_only_AppraisalClassifier_boundary_contract_object_without_changing_routes_or_AGP_verification"
    ]
    assert report["recommended_next_implementation_step"] == report["recommended_next_implementation_steps"][0]
    assert "change_AppraisalClassifier_behavior_or_thresholds" in report["forbidden_next_step_actions"]
    assert "bypass_AGP" in report["forbidden_next_step_actions"]
    assert "bypass_fallback" in report["forbidden_next_step_actions"]
    assert "load_runtime_vectors_or_read_vector_contents" in report["forbidden_next_step_actions"]
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled"] is False
    assert report["vector_contents_read"] is False
    assert report["vectors_loaded"] is False


def test_round641_660_cli_emits_compact_json_and_creates_no_artifacts(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write_declared_path_placeholders(tmp_path)
    sentinels = _write_medium30k_sentinels(tmp_path)
    before_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    before = {path: path.read_bytes() for path in sentinels}

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/operator_audit_round641_660_appraisal_agp_input.py",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    after_files = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    after = {path: path.read_bytes() for path in sentinels}
    assert proc.returncode == 0, proc.stderr
    assert after == before
    assert after_files == before_files
    payload = json.loads(proc.stdout)
    assert payload["version"] == round641_660.ROUND641_660_VERSION
    assert payload["success"] is True
    assert payload["operator_command"] == round641_660.OPERATOR_COMMAND
    assert payload["operator_report_path"] == "docs/round641_660_appraisal_agp_input_audit.md"
    assert payload["git_artifact_safety_proof"]["forbidden_entries"] == []
    assert payload["vector_contents_read"] is False
    assert payload["vectors_loaded"] is False


def test_round641_660_preserves_korean_and_minsok_fixture_order_exactly() -> None:
    report = round641_660.build_round641_660_report()

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
    assert report["inventory"]["korean_fixture_coverage"]["minsok_rows"] == [
        {"category": "minsok", "text": "군대 생활 어때"},
        {"category": "minsok", "text": "코딩 좋아해"},
        {"category": "minsok", "text": "EVE 프로젝트"},
    ]
    assert report["inventory"]["korean_fixture_coverage"]["literal_minsok_runtime_hardcode_added"] is False
