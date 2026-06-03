from __future__ import annotations

from pathlib import Path

from adapters.runtime_mapping_import_blocker_recovery import (
    ROUND127_WORKING_MEMORY_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
    ROUND128_WORKING_MEMORY_COMPAT_SHIM_VERSION,
    ROUND129_COLLECT_ONLY_AFTER_WORKING_MEMORY_VERIFICATION_VERSION,
    ROUND130_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
    ROUND131_GO_NO_GO_REFRESH_AFTER_WORKING_MEMORY_VERSION,
    build_round127_working_memory_import_blocker_diagnosis,
    build_round128_working_memory_import_compatibility_shim_decision,
    build_round129_collect_only_after_working_memory_verification,
    build_round130_broader_validation_taxonomy_refresh,
    build_round131_go_no_go_refresh_after_working_memory,
    write_round_report,
)
from legacy.eve_modules.working_memory import WorkingMemory as LegacyWorkingMemory
from legacy.eve_modules.working_memory import WMSlot as LegacyWMSlot
from working_memory import WMSlot, WorkingMemory


def test_round127_diagnoses_missing_root_working_memory_import_without_activation(tmp_path: Path) -> None:
    (tmp_path / "test_legacy.py").write_text("from working_memory import WorkingMemory\n", encoding="utf-8")
    report = build_round127_working_memory_import_blocker_diagnosis(repo_root=tmp_path, module_available=False)

    assert report["diagnosis_version"] == ROUND127_WORKING_MEMORY_IMPORT_BLOCKER_DIAGNOSIS_VERSION
    assert report["diagnosis_status"] == "working_memory_import_blocker_active"
    assert report["root_import_sites"] == ["test_legacy.py"]
    assert report["retained_legacy_candidate"] == "legacy/eve_modules/working_memory.py"
    assert report["recommended_round128_action"] == "add_minimal_compatibility_shim"
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled_default"] is False
    assert report["agp_bypass_used"] is False


def test_round128_root_shim_reexports_retained_legacy_working_memory() -> None:
    assert WorkingMemory is LegacyWorkingMemory
    assert WMSlot is LegacyWMSlot
    memory = WorkingMemory(base_capacity=2)
    assert memory.add("민석", salience=0.8, source="test") is True
    assert memory.get_focus() == "민석"

    diagnosis = build_round127_working_memory_import_blocker_diagnosis(module_available=True)
    decision = build_round128_working_memory_import_compatibility_shim_decision(
        source_round127_diagnosis=diagnosis,
        import_check_passed=True,
    )

    assert decision["compatibility_version"] == ROUND128_WORKING_MEMORY_COMPAT_SHIM_VERSION
    assert decision["decision_status"] == "minimal_compatibility_shim_applied"
    assert decision["shim_is_minimal_reexport"] is True
    assert decision["reexported_symbols"] == ["WMSlot", "WorkingMemory"]
    assert decision["fake_behavior_markers_present"] == []
    assert decision["production_persistence_enabled"] is False
    assert decision["runtime_mapping_enabled_default"] is False
    assert decision["enforcement_enabled_default"] is False


def test_round129_records_working_memory_recovery_and_next_legacy_side_effect_blocker() -> None:
    collect = build_round129_collect_only_after_working_memory_verification(
        return_code=3,
        collected_tests=None,
        remaining_errors=[
            {
                "path": "test_natural_lang_v2.py",
                "error_family": "legacy_collection_side_effect_system_exit",
                "detail": "module-level validation calls sys.exit(1) during pytest collection",
            }
        ],
    )

    assert collect["collect_recovery_version"] == ROUND129_COLLECT_ONLY_AFTER_WORKING_MEMORY_VERIFICATION_VERSION
    assert collect["collect_recovery_status"] == "collect_only_partial_new_legacy_side_effect_blocker_after_working_memory_recovery"
    assert collect["working_memory_import_errors_remaining"] == 0
    assert collect["critical_blocker_improved"] is True
    assert collect["next_blocker_family"] == "legacy_collection_side_effect_system_exit"
    assert collect["broader_validation_status"] == "blocked_partial"
    assert collect["production_persistence_enabled"] is False


def test_round130_131_taxonomy_and_go_no_go_remain_no_go_until_collect_only_recovers() -> None:
    collect = build_round129_collect_only_after_working_memory_verification(
        return_code=3,
        remaining_errors=[{"path": "test_natural_lang_v2.py", "error_family": "legacy_collection_side_effect_system_exit"}],
    )
    taxonomy = build_round130_broader_validation_taxonomy_refresh(
        source_round129_collect_recovery=collect,
        validation_items=[
            {"category": "compile_checks", "command": "python -m compileall -q adapters tests main.py", "status": "pass"},
            {"category": "focused_round127_129_tests", "command": "pytest -q tests/test_v3_round127_129_working_memory_import_recovery.py", "status": "pass"},
            {"category": "broader_validation", "command": "pytest -q", "status": "blocked_partial", "reason": "collect-only still blocked"},
        ],
    )
    refresh = build_round131_go_no_go_refresh_after_working_memory(
        source_round129_collect_recovery=collect,
        source_round130_taxonomy=taxonomy,
    )

    assert taxonomy["taxonomy_version"] == ROUND130_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION
    assert taxonomy["taxonomy_status"] == "broader_validation_partial_or_blocked"
    assert taxonomy["working_memory_blocker_recovered"] is True
    assert refresh["go_no_go_refresh_version"] == ROUND131_GO_NO_GO_REFRESH_AFTER_WORKING_MEMORY_VERSION
    assert refresh["final_recommendation"] == "NO-GO"
    assert refresh["critical_blocker_improved"] is True
    assert "legacy_collection_side_effect_system_exit" in refresh["remaining_blockers"]
    assert refresh["production_persistence_enabled"] is False
    assert refresh["runtime_mapping_enabled_default"] is False
    assert refresh["enforcement_enabled_default"] is False


def test_round127_131_export_writes_json_without_activation(tmp_path: Path) -> None:
    report = build_round131_go_no_go_refresh_after_working_memory(
        source_round129_collect_recovery=build_round129_collect_only_after_working_memory_verification(return_code=0),
        source_round130_taxonomy=build_round130_broader_validation_taxonomy_refresh(validation_items=[]),
    )
    export = write_round_report(tmp_path / "round131.json", report)

    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert (tmp_path / "round131.json").read_text(encoding="utf-8").startswith("{\n")
