from __future__ import annotations

from pathlib import Path

from adapters.runtime_mapping_import_blocker_recovery import (
    ROUND137_DMN_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
    ROUND138_DMN_COMPAT_SHIM_VERSION,
    ROUND139_COLLECT_ONLY_AFTER_DMN_ISOLATION_VERSION,
    ROUND140_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
    ROUND141_GO_NO_GO_REFRESH_AFTER_DMN_ISOLATION_VERSION,
    build_round137_dmn_import_blocker_diagnosis,
    build_round138_dmn_import_compatibility_shim_decision,
    build_round139_collect_only_after_dmn_isolation,
    build_round140_broader_validation_taxonomy_refresh,
    build_round141_go_no_go_refresh_after_dmn_isolation,
    write_round_report,
)
from dmn import DefaultModeNetwork
from legacy.eve_modules.dmn import DefaultModeNetwork as LegacyDefaultModeNetwork


def test_round137_diagnoses_missing_root_dmn_import_without_activation(tmp_path: Path) -> None:
    (tmp_path / "test_eve_main_ab.py").write_text("from dmn import DefaultModeNetwork\n", encoding="utf-8")
    legacy_dir = tmp_path / "legacy" / "eve_modules"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "dmn.py").write_text("class DefaultModeNetwork: pass\n", encoding="utf-8")

    report = build_round137_dmn_import_blocker_diagnosis(repo_root=tmp_path, module_available=False)

    assert report["diagnosis_version"] == ROUND137_DMN_IMPORT_BLOCKER_DIAGNOSIS_VERSION
    assert report["diagnosis_status"] == "dmn_import_blocker_active"
    assert report["root_import_sites"] == ["test_eve_main_ab.py"]
    assert report["retained_legacy_candidate"] == "legacy/eve_modules/dmn.py"
    assert report["recommended_round138_action"] == "add_minimal_compatibility_shim"
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled_default"] is False
    assert report["agp_bypass_used"] is False


def test_round138_root_shim_reexports_retained_legacy_dmn() -> None:
    assert DefaultModeNetwork is LegacyDefaultModeNetwork
    assert hasattr(DefaultModeNetwork, "activate_one")
    assert hasattr(DefaultModeNetwork, "tick")
    assert hasattr(DefaultModeNetwork, "inner_voice")

    diagnosis = build_round137_dmn_import_blocker_diagnosis(module_available=True)
    decision = build_round138_dmn_import_compatibility_shim_decision(
        source_round137_diagnosis=diagnosis,
        import_check_passed=True,
    )

    assert decision["compatibility_version"] == ROUND138_DMN_COMPAT_SHIM_VERSION
    assert decision["decision_status"] == "minimal_compatibility_shim_applied"
    assert decision["shim_is_minimal_reexport"] is True
    assert decision["reexported_symbols"] == ["DefaultModeNetwork"]
    assert decision["fake_behavior_markers_present"] == []
    assert decision["production_persistence_enabled"] is False
    assert decision["runtime_mapping_enabled_default"] is False
    assert decision["enforcement_enabled_default"] is False


def test_round139_records_collect_only_recovery_after_dmn_isolation() -> None:
    collect = build_round139_collect_only_after_dmn_isolation(
        return_code=0,
        collected_tests=1287,
        remaining_errors=[],
    )

    assert collect["collect_recovery_version"] == ROUND139_COLLECT_ONLY_AFTER_DMN_ISOLATION_VERSION
    assert collect["collect_recovery_status"] == "collect_only_recovered_after_dmn_isolation"
    assert collect["dmn_import_errors_remaining"] == 0
    assert collect["critical_blocker_improved"] is True
    assert collect["broader_validation_status"] == "collect_only_passed"
    assert collect["production_persistence_enabled"] is False


def test_round140_141_taxonomy_and_go_no_go_keep_no_go_for_partial_broader_validation() -> None:
    collect = build_round139_collect_only_after_dmn_isolation(return_code=0, collected_tests=1287)
    taxonomy = build_round140_broader_validation_taxonomy_refresh(
        source_round139_collect_recovery=collect,
        validation_items=[
            {"category": "compile_checks", "command": "python -m compileall -q adapters tests main.py", "status": "pass"},
            {"category": "focused_round137_139_tests", "command": "pytest -q tests/test_v3_round137_139_dmn_import_recovery.py", "status": "pass"},
            {"category": "collect_only", "command": "pytest --collect-only -q", "status": "pass"},
            {"category": "legacy_behavior_tests", "command": "pytest -q test_natural_lang_v2.py", "status": "fail", "reason": "pre-existing NaturalLanguage v2 behavior assertions fail after collection isolation"},
        ],
    )
    refresh = build_round141_go_no_go_refresh_after_dmn_isolation(
        source_round139_collect_recovery=collect,
        source_round140_taxonomy=taxonomy,
    )

    assert taxonomy["taxonomy_version"] == ROUND140_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION
    assert taxonomy["taxonomy_status"] == "broader_validation_partial_or_blocked"
    assert taxonomy["dmn_blocker_recovered"] is True
    assert taxonomy["collect_only_green"] is True
    assert taxonomy["primary_remaining_blocker_family"] == "legacy_behavior_failure"
    assert refresh["go_no_go_refresh_version"] == ROUND141_GO_NO_GO_REFRESH_AFTER_DMN_ISOLATION_VERSION
    assert refresh["final_recommendation"] == "NO-GO"
    assert refresh["collect_only_green"] is True
    assert "legacy_behavior_failure" in refresh["remaining_blockers"]
    assert refresh["production_persistence_enabled"] is False
    assert refresh["runtime_mapping_enabled_default"] is False
    assert refresh["enforcement_enabled_default"] is False


def test_round137_141_export_writes_json_without_activation(tmp_path: Path) -> None:
    report = build_round141_go_no_go_refresh_after_dmn_isolation(
        source_round139_collect_recovery=build_round139_collect_only_after_dmn_isolation(return_code=0),
        source_round140_taxonomy=build_round140_broader_validation_taxonomy_refresh(validation_items=[]),
    )
    export = write_round_report(tmp_path / "round141.json", report)

    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert (tmp_path / "round141.json").read_text(encoding="utf-8").startswith("{\n")
