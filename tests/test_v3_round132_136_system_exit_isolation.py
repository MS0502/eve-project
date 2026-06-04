from __future__ import annotations

import importlib.util
from pathlib import Path

from adapters.runtime_mapping_import_blocker_recovery import (
    ROUND132_NATURAL_LANG_V2_SYSTEM_EXIT_DIAGNOSIS_VERSION,
    ROUND133_COLLECTION_SIDE_EFFECT_ISOLATION_VERSION,
    ROUND134_COLLECT_ONLY_AFTER_SYSTEM_EXIT_ISOLATION_VERSION,
    ROUND135_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
    ROUND136_GO_NO_GO_REFRESH_AFTER_SYSTEM_EXIT_VERSION,
    build_round132_natural_lang_v2_system_exit_diagnosis,
    build_round133_collection_side_effect_isolation_decision,
    build_round134_collect_only_after_system_exit_isolation,
    build_round135_broader_validation_taxonomy_refresh,
    build_round136_go_no_go_refresh_after_system_exit,
    write_round_report,
)


def test_round132_diagnoses_module_level_system_exit_without_activation(tmp_path: Path) -> None:
    legacy = tmp_path / "test_natural_lang_v2.py"
    legacy.write_text("import sys\nresults = [False]\nif not all(results):\n    sys.exit(1)\n", encoding="utf-8")

    report = build_round132_natural_lang_v2_system_exit_diagnosis(
        test_path=legacy,
        collect_return_code=3,
        observed_system_exit_line=4,
    )

    assert report["diagnosis_version"] == ROUND132_NATURAL_LANG_V2_SYSTEM_EXIT_DIAGNOSIS_VERSION
    assert report["diagnosis_status"] == "collection_time_system_exit_active"
    assert report["module_level_sys_exit_lines"] == [4]
    assert report["recommended_round133_action"] == "move execution behind main guard and expose pytest-safe validation wrapper"
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled_default"] is False
    assert report["agp_bypass_used"] is False


def test_round133_natural_lang_v2_import_is_collection_safe_and_preserves_test_intent() -> None:
    spec = importlib.util.spec_from_file_location("round133_nlv2_import_probe", "test_natural_lang_v2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "run_natural_language_v2_validation")
    assert hasattr(module, "test_natural_language_v2_validation_behavior")

    diagnosis = build_round132_natural_lang_v2_system_exit_diagnosis(test_path="test_natural_lang_v2.py")
    decision = build_round133_collection_side_effect_isolation_decision(
        test_path="test_natural_lang_v2.py",
        source_round132_diagnosis=diagnosis,
        import_check_passed=True,
        legacy_script_exit_preserved=True,
    )

    assert diagnosis["diagnosis_status"] == "collection_time_system_exit_isolated_or_absent"
    assert decision["isolation_version"] == ROUND133_COLLECTION_SIDE_EFFECT_ISOLATION_VERSION
    assert decision["isolation_status"] == "collection_side_effect_isolated_test_intent_preserved"
    assert decision["module_level_sys_exit_lines_remaining"] == []
    assert decision["collection_safe_wrapper_present"] is True
    assert decision["pytest_behavior_test_present"] is True
    assert decision["test_intent_preserved"] is True
    assert decision["skip_or_xfail_added"] is False


def test_round134_records_collect_only_recovery_after_system_exit_isolation() -> None:
    collect = build_round134_collect_only_after_system_exit_isolation(
        return_code=0,
        collected_tests=1210,
        remaining_errors=[],
    )

    assert collect["collect_recovery_version"] == ROUND134_COLLECT_ONLY_AFTER_SYSTEM_EXIT_ISOLATION_VERSION
    assert collect["collect_recovery_status"] == "collect_only_recovered_after_system_exit_isolation"
    assert collect["system_exit_errors_remaining"] == 0
    assert collect["critical_blocker_improved"] is True
    assert collect["broader_validation_status"] == "collect_only_passed"
    assert collect["production_persistence_enabled"] is False


def test_round135_136_taxonomy_keeps_no_go_when_legacy_behavior_still_fails() -> None:
    collect = build_round134_collect_only_after_system_exit_isolation(return_code=0, collected_tests=1210)
    taxonomy = build_round135_broader_validation_taxonomy_refresh(
        source_round134_collect_recovery=collect,
        validation_items=[
            {"category": "compile_checks", "command": "python -m compileall -q adapters tests main.py", "status": "pass"},
            {"category": "focused_round132_134_tests", "command": "pytest -q tests/test_v3_round132_136_system_exit_isolation.py", "status": "pass"},
            {"category": "collect_only", "command": "pytest --collect-only -q", "status": "pass"},
            {"category": "legacy_behavior_tests", "command": "pytest -q test_natural_lang_v2.py", "status": "fail", "reason": "pre-existing NaturalLanguage v2 behavior assertions fail after collection isolation"},
        ],
    )
    refresh = build_round136_go_no_go_refresh_after_system_exit(
        source_round134_collect_recovery=collect,
        source_round135_taxonomy=taxonomy,
    )

    assert taxonomy["taxonomy_version"] == ROUND135_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION
    assert taxonomy["taxonomy_status"] == "broader_validation_partial_or_blocked"
    assert taxonomy["system_exit_blocker_recovered"] is True
    assert taxonomy["primary_remaining_blocker_family"] == "legacy_behavior_failure"
    assert refresh["go_no_go_refresh_version"] == ROUND136_GO_NO_GO_REFRESH_AFTER_SYSTEM_EXIT_VERSION
    assert refresh["final_recommendation"] == "NO-GO"
    assert refresh["collect_only_green"] is True
    assert "legacy_behavior_failure" in refresh["remaining_blockers"]
    assert refresh["production_persistence_enabled"] is False
    assert refresh["runtime_mapping_enabled_default"] is False
    assert refresh["enforcement_enabled_default"] is False


def test_round132_136_export_writes_json_without_activation(tmp_path: Path) -> None:
    report = build_round136_go_no_go_refresh_after_system_exit(
        source_round134_collect_recovery=build_round134_collect_only_after_system_exit_isolation(return_code=0),
        source_round135_taxonomy=build_round135_broader_validation_taxonomy_refresh(
            validation_items=[{"category": "legacy_behavior_tests", "status": "fail"}]
        ),
    )
    export = write_round_report(tmp_path / "round136.json", report)

    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert (tmp_path / "round136.json").read_text(encoding="utf-8").startswith("{\n")
