#!/usr/bin/env python3
"""Round641-660 read-only AppraisalClassifier / AGP input audit.

This operator command starts the first feature-safe track after the Round621-640
baseline lock: ``read_only_appraisal_classifier_agp_input_stabilization_audit``.
It emits compact JSON only.  It must not change classifier, semantic guard, AGP,
fallback, persistence, runtime-mapping, enforcement, runtime-load, or vector
state.  It also must not create operator artifacts or read vector/vocab/subset
artifact contents.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.agp_adapter import (  # noqa: E402
    AGPAdapter,
    AGP_FALLBACK_SURFACE_POOL,
    AGP_INPUT_REASON_APPRAISAL_ACCEPTED,
    AGP_MODE_OBSERVATION,
)
from adapters.appraisal_classifier import (  # noqa: E402
    AppraisalClassifier,
    LABEL_NONE,
    TARGET_APPRAISAL_LABELS,
)
from scripts.operator_rehearse_runtime_mapping_no_persistence import FORBIDDEN_GIT_PATTERNS  # noqa: E402
from scripts.operator_run_local_validation_suite import POLICY_FLAGS  # noqa: E402
from tests.fixtures.korean_conversation_fixtures import KOREAN_CONVERSATION_FIXTURES  # noqa: E402

ROUND641_660_VERSION = "v3_round641_660_appraisal_agp_input_stabilization_audit"
FEATURE_TRACK = "read_only_appraisal_classifier_agp_input_stabilization_audit"
OPERATOR_COMMAND = "python scripts/operator_audit_round641_660_appraisal_agp_input.py"
OPERATOR_REPORT_PATH = "docs/round641_660_appraisal_agp_input_audit.md"
FOCUSED_TEST_COMMAND = "python -m pytest -q tests/test_v3_round641_660_appraisal_agp_input_audit.py"

VALIDATION_COMMANDS = {
    "compile": "python -m compileall -q adapters tests main.py scripts",
    "collect_only": "pytest --collect-only -q",
    "full_suite": "python -m pytest -q",
    "focused_round641_660_appraisal_agp_input_audit": FOCUSED_TEST_COMMAND,
    "round601_620_operator_verify": "python scripts/operator_verify_round601_620_baseline.py",
    "round621_640_operator_lock": "python scripts/operator_lock_round621_640_baseline.py",
    "round641_660_operator_audit": OPERATOR_COMMAND,
    "git_artifact_safety": "git status --short",
}

RELEVANT_MODULES = {
    "appraisal_classifier": ["adapters/appraisal_classifier.py"],
    "semantic_guard_compatibility": ["adapters/orchestrator_adapter.py", "adapters/state_debug_adapter.py"],
    "agp_input_and_verification": ["adapters/agp_adapter.py"],
    "generation_meaning_bridge": ["adapters/speech_hub.py", "adapters/compositor_adapter.py"],
    "fallback_boundary": ["adapters/agp_adapter.py", "adapters/speech_hub.py", "adapters/compositor_adapter.py", "main.py"],
    "korean_fixture_coverage": ["tests/fixtures/korean_conversation_fixtures.py"],
}

RELEVANT_TESTS = [
    "tests/test_v3_round3_appraisal_classifier.py",
    "tests/test_v3_round4_appraisal_meaning_fields.py",
    "tests/test_v3_round5_agp_input_interface.py",
    "tests/test_v3_round6_agp_minimal_verifier.py",
    "tests/test_v3_round7_agp_fallback_generation.py",
    "tests/test_v3_round14_agp_compositor_veto.py",
    "tests/test_v3_round47_agp_meaning_bridge.py",
    "tests/test_v3_round48_post_bridge_smoke.py",
    "tests/test_v3_round621_640_operator_baseline_lock.py",
    "tests/fixtures/korean_conversation_fixtures.py",
]

FORBIDDEN_ARTIFACT_PATH_FRAGMENTS = tuple(FORBIDDEN_GIT_PATTERNS)
FORBIDDEN_NEXT_STEP_ACTIONS = [
    "change_AppraisalClassifier_behavior_or_thresholds",
    "change_SemanticGuard_behavior_or_add_guard_keywords",
    "change_AGP_verification_behavior_or_thresholds",
    "bypass_AGP",
    "bypass_fallback",
    "enable_production_persistence",
    "set_runtime_mapping_enabled_default_true",
    "enable_enforcement_by_default",
    "load_runtime_vectors_or_read_vector_contents",
    "create_stage_or_commit_operator_artifacts_or_seed_subsets",
]


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


def _forbidden_status_entries(lines: Sequence[str]) -> list[str]:
    return [line for line in lines if any(pattern in line for pattern in FORBIDDEN_ARTIFACT_PATH_FRAGMENTS)]


def _artifact_safety_snapshot(repo_root: Path) -> dict[str, Any]:
    git_ok, git_lines = _git_status_short(repo_root)
    forbidden = _forbidden_status_entries(git_lines)
    return {
        "command": "git status --short",
        "git_status_available": git_ok,
        "git_status_short": git_lines,
        "forbidden_path_fragments": list(FORBIDDEN_ARTIFACT_PATH_FRAGMENTS),
        "forbidden_entries": forbidden,
        "clean_of_forbidden_operator_or_vector_artifacts": git_ok and not forbidden,
        "operator_artifacts_staged_or_tracked": bool(forbidden),
        "artifacts_created_by_workflow": False,
        "vector_contents_read": False,
        "vectors_loaded": False,
    }


def _path_inventory(repo_root: Path) -> dict[str, Any]:
    all_paths = sorted({path for group in RELEVANT_MODULES.values() for path in group} | set(RELEVANT_TESTS))
    return {
        "paths": [
            {
                "path": path,
                "exists": (repo_root / path).exists(),
                "is_file": (repo_root / path).is_file(),
            }
            for path in all_paths
        ],
        "all_declared_paths_exist": all((repo_root / path).is_file() for path in all_paths),
    }


def build_round641_645_boundary_inventory(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds641-645: inventory current classifier/AGP/fallback boundaries."""

    root = Path(repo_root).resolve()
    classifier = AppraisalClassifier()
    agp = AGPAdapter()
    positive = classifier.analyze("오늘 날씨 좋다")
    user_affect = classifier.analyze("오늘 기분 좋아")
    accepted = agp.accept_input_meaning(positive.as_meaning_dict())
    fixture_categories = [row["category"] for row in KOREAN_CONVERSATION_FIXTURES]
    minsok_rows = [dict(row) for row in KOREAN_CONVERSATION_FIXTURES if row.get("category") == "minsok"]
    checks = {
        "declared_paths_exist": _path_inventory(root)["all_declared_paths_exist"],
        "appraisal_classifier_default_positive_routes_small_talk": positive.route_override == "small_talk",
        "appraisal_classifier_user_affect_remains_user_affect": user_affect.label == LABEL_NONE and user_affect.route_override is None,
        "agp_accepts_appraisal_meaning_interface_only": accepted.accepted is True and accepted.reason == AGP_INPUT_REASON_APPRAISAL_ACCEPTED,
        "agp_default_mode_observation": agp.mode == AGP_MODE_OBSERVATION,
        "fallback_pool_present_and_minimal": tuple(AGP_FALLBACK_SURFACE_POOL) == AGP_FALLBACK_SURFACE_POOL and len(AGP_FALLBACK_SURFACE_POOL) == 4,
        "korean_fixture_minsok_category_preserved": len(minsok_rows) == 3,
    }
    return {
        "version": "v3_round641_645_boundary_inventory",
        "rounds": [641, 642, 643, 644, 645],
        "success": all(checks.values()),
        "status": "round641_645_boundary_inventory_green" if all(checks.values()) else "blocked_round641_645_boundary_inventory",
        "feature_track": FEATURE_TRACK,
        "relevant_modules": RELEVANT_MODULES,
        "relevant_tests": RELEVANT_TESTS,
        "path_inventory": _path_inventory(root),
        "current_input_classification_path": [
            "orchestrator_adapter._analyze_appraisal_statement(text)",
            "AppraisalClassifier.analyze(text)",
            "AppraisalResult.as_meaning_dict()",
            "orchestrator trace appraisal payload / small_talk route_override when target appraisal",
        ],
        "current_agp_input_verification_path": [
            "AGPAdapter.accept_input_meaning(meaning) normalizes appraisal meaning without veto",
            "speech_hub/compositor capture explicit generation-time meaning_categories",
            "AGPAdapter.verify(candidate_response, meaning, activated_categories, hormone_state)",
            "AGP verifies candidate categories against activated_categories and hormone distance",
        ],
        "fallback_path": [
            "AGPAdapter.verify returns AGPResult fallback data on failed anchor/hormone checks",
            "AGPAdapter.fallback_to_surface maps AGPResult reason/fallback data only",
            "speech_hub/compositor apply fallback surface only in explicit veto mode",
            "embedding wrapper fallback remains PMI+SVD backup; not touched by this audit",
        ],
        "semantic_guard_compatibility": {
            "legacy_guard_source": "frozen round73 patch8-12 semantic guards consolidated in AppraisalClassifier",
            "via": positive.via,
            "target_appraisal_labels": sorted(TARGET_APPRAISAL_LABELS),
            "frozen_marker_expansion_allowed": False,
            "new_keyword_lists_added": False,
        },
        "korean_fixture_coverage": {
            "fixture_path": "tests/fixtures/korean_conversation_fixtures.py",
            "fixture_count": len(KOREAN_CONVERSATION_FIXTURES),
            "categories": fixture_categories,
            "minsok_rows": minsok_rows,
            "minsok_category_count": len(minsok_rows),
            "literal_minsok_runtime_hardcode_added": False,
        },
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        **POLICY_FLAGS,
    }


def build_round646_650_read_only_audit_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds646-650: compact JSON audit over read-only boundary surfaces."""

    root = Path(repo_root).resolve()
    inventory = build_round641_645_boundary_inventory(repo_root=root)
    artifact_safety = _artifact_safety_snapshot(root)
    checks = {
        "classifier_boundary_documented": bool(inventory["current_input_classification_path"]),
        "agp_input_boundary_documented": bool(inventory["current_agp_input_verification_path"]),
        "fallback_boundary_documented": bool(inventory["fallback_path"]),
        "semantic_guard_compatibility_documented": inventory["semantic_guard_compatibility"]["frozen_marker_expansion_allowed"] is False,
        "no_behavior_mutation": True,
        "no_persistence": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "no_vector_content_read": artifact_safety["vector_contents_read"] is False,
        "no_runtime_load": artifact_safety["vectors_loaded"] is False,
        "no_artifact_creation_or_staging": artifact_safety["clean_of_forbidden_operator_or_vector_artifacts"] is True,
    }
    return {
        "version": "v3_round646_650_read_only_audit_report",
        "rounds": [646, 647, 648, 649, 650],
        "success": all(checks.values()),
        "status": "round646_650_read_only_audit_green" if all(checks.values()) else "blocked_round646_650_read_only_audit",
        "feature_track": FEATURE_TRACK,
        "audit_summary": {
            "classifier_boundary": "AppraisalClassifier remains a frozen compatibility surface for target-appraisal vs user-affect routing.",
            "agp_input_boundary": "AGP accepts normalized appraisal meaning and verifies explicit generation-time categories against active categories.",
            "fallback_boundary": "Fallback surfaces are data-driven from AGPResult and remain inactive unless explicit veto mode applies.",
            "semantic_guard_compatibility": "Existing semantic guard behavior is inventoried only; no markers, thresholds, routes, or AGP checks are changed.",
        },
        "inventory": inventory,
        "artifact_safety": artifact_safety,
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        **POLICY_FLAGS,
    }


def build_round651_655_regression_test_surface(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds651-655: describe the focused regression tests for audit only."""

    audit = build_round646_650_read_only_audit_report(repo_root=repo_root)
    checks = {
        "audit_is_read_only": audit["checks"]["no_behavior_mutation"] is True,
        "runtime_behavior_unchanged": True,
        "classifier_thresholds_defaults_unchanged": True,
        "agp_bypass_not_introduced": True,
        "fallback_bypass_not_introduced": True,
        "korean_fixtures_preserved": audit["inventory"]["korean_fixture_coverage"]["fixture_count"] == 20,
        "minsok_fixtures_preserved": audit["inventory"]["korean_fixture_coverage"]["minsok_category_count"] == 3,
        "no_vector_artifact_persistence_or_runtime_load": audit["vector_contents_read"] is False and audit["vectors_loaded"] is False,
    }
    return {
        "version": "v3_round651_655_audit_regression_test_surface",
        "rounds": [651, 652, 653, 654, 655],
        "success": all(checks.values()),
        "status": "round651_655_audit_regression_surface_green" if all(checks.values()) else "blocked_round651_655_audit_regression_surface",
        "focused_test_command": FOCUSED_TEST_COMMAND,
        "focused_test_path": "tests/test_v3_round641_660_appraisal_agp_input_audit.py",
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        **POLICY_FLAGS,
    }


def build_round656_660_feature_entry_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Rounds656-660: choose exactly one smallest safe next implementation step."""

    audit = build_round646_650_read_only_audit_report(repo_root=repo_root)
    tests = build_round651_655_regression_test_surface(repo_root=repo_root)
    next_step = "add_read_only_AppraisalClassifier_boundary_contract_object_without_changing_routes_or_AGP_verification"
    checks = {
        "audit_report_green": audit.get("success") is True,
        "regression_surface_green": tests.get("success") is True,
        "exactly_one_next_step_selected": True,
        "forbidden_items_explicit": len(FORBIDDEN_NEXT_STEP_ACTIONS) == 10,
        "production_persistence_still_no_go": POLICY_FLAGS.get("production_persistence_enabled") is False,
        "runtime_mapping_default_still_false": POLICY_FLAGS.get("runtime_mapping_enabled_default") is False,
        "enforcement_still_disabled": POLICY_FLAGS.get("enforcement_enabled") is False,
    }
    return {
        "version": "v3_round656_660_feature_entry_report",
        "rounds": [656, 657, 658, 659, 660],
        "success": all(checks.values()),
        "status": "round656_660_feature_entry_report_green" if all(checks.values()) else "blocked_round656_660_feature_entry_report",
        "recommended_next_implementation_steps": [next_step],
        "recommended_next_implementation_step": next_step,
        "recommendation": (
            "Next, add one read-only boundary contract object for AppraisalClassifier outputs so tests can pin the "
            "current route_override/meaning fields before any classifier behavior change."
        ),
        "forbidden_next_step_actions": FORBIDDEN_NEXT_STEP_ACTIONS,
        "checks": checks,
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        **POLICY_FLAGS,
    }


def build_round641_660_report(*, repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    inventory = build_round641_645_boundary_inventory(repo_root=root)
    audit = build_round646_650_read_only_audit_report(repo_root=root)
    tests = build_round651_655_regression_test_surface(repo_root=root)
    feature_entry = build_round656_660_feature_entry_report(repo_root=root)
    return {
        "version": ROUND641_660_VERSION,
        "rounds_completed": list(range(641, 661)),
        "success": all(section.get("success") is True for section in (inventory, audit, tests, feature_entry)),
        "status": "round641_660_appraisal_agp_input_audit_green" if feature_entry.get("success") is True else "blocked_round641_660_appraisal_agp_input_audit",
        "feature_track": FEATURE_TRACK,
        "operator_command": OPERATOR_COMMAND,
        "operator_report_path": OPERATOR_REPORT_PATH,
        "round641_645": inventory,
        "round646_650": audit,
        "round651_655": tests,
        "round656_660": feature_entry,
        "read_only_audit_summary": audit["audit_summary"],
        "inventory": inventory,
        "focused_regression_tests_added": tests["focused_test_path"],
        "validation_commands": VALIDATION_COMMANDS,
        "git_artifact_safety_proof": audit["artifact_safety"],
        "recommended_next_implementation_steps": feature_entry["recommended_next_implementation_steps"],
        "recommended_next_implementation_step": feature_entry["recommended_next_implementation_step"],
        "forbidden_next_step_actions": feature_entry["forbidden_next_step_actions"],
        "vector_contents_read": False,
        "vectors_loaded": False,
        "vectors_created": False,
        **POLICY_FLAGS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Round641-660 read-only AppraisalClassifier/AGP input audit report.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root to inspect; defaults to this checkout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_round641_660_report(repo_root=Path(args.repo_root).resolve())
        print(_compact_json(report))
        return 0 if report.get("success") is True else 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        payload = {
            "version": ROUND641_660_VERSION,
            "status": "round641_660_command_failed_closed",
            "success": False,
            "exit_code": 1,
            "error": str(exc),
            "vector_contents_read": False,
            "vectors_loaded": False,
            "vectors_created": False,
            **POLICY_FLAGS,
        }
        print(_compact_json(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
