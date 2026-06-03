from __future__ import annotations

from pathlib import Path

from adapters.runtime_mapping_import_blocker_recovery import (
    ROUND122_LEGACY_ROOT_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
    ROUND123_SPREADING_ACTIVATION_COMPAT_SHIM_VERSION,
    ROUND124_COLLECT_ONLY_RECOVERY_VERIFICATION_VERSION,
    build_round122_legacy_root_import_blocker_diagnosis,
    build_round123_spreading_activation_import_compatibility_shim_decision,
    build_round124_collect_only_recovery_verification,
    write_round_report,
)
from legacy.eve_modules.spreading_activation import SpreadingActivation as LegacySpreadingActivation
from spreading_activation import SpreadingActivation


def test_round122_diagnoses_missing_root_spreading_activation_import_without_activation(tmp_path: Path) -> None:
    (tmp_path / "test_legacy.py").write_text("from spreading_activation import SpreadingActivation\n", encoding="utf-8")
    report = build_round122_legacy_root_import_blocker_diagnosis(repo_root=tmp_path, module_available=False)

    assert report["diagnosis_version"] == ROUND122_LEGACY_ROOT_IMPORT_BLOCKER_DIAGNOSIS_VERSION
    assert report["diagnosis_status"] == "legacy_root_import_blocker_active"
    assert report["root_import_sites"] == ["test_legacy.py"]
    assert report["recommended_round123_action"] == "add_minimal_compatibility_shim"
    assert report["production_persistence_enabled"] is False
    assert report["runtime_mapping_enabled_default"] is False
    assert report["enforcement_enabled_default"] is False
    assert report["agp_bypass_used"] is False


def test_round123_root_shim_reexports_retained_legacy_spreading_activation() -> None:
    assert SpreadingActivation is LegacySpreadingActivation
    sa = SpreadingActivation(base_threshold=0.3)
    sa.activate("민석", 0.5)
    assert "민석" in sa.get_active()

    diagnosis = build_round122_legacy_root_import_blocker_diagnosis(module_available=True)
    decision = build_round123_spreading_activation_import_compatibility_shim_decision(
        source_round122_diagnosis=diagnosis,
        import_check_passed=True,
    )

    assert decision["compatibility_version"] == ROUND123_SPREADING_ACTIVATION_COMPAT_SHIM_VERSION
    assert decision["decision_status"] == "minimal_compatibility_shim_applied"
    assert decision["shim_is_minimal_reexport"] is True
    assert decision["fake_behavior_markers_present"] == []
    assert decision["production_persistence_enabled"] is False
    assert decision["runtime_mapping_enabled_default"] is False
    assert decision["enforcement_enabled_default"] is False


def test_round124_records_collect_recovery_and_remaining_non_spreading_blocker() -> None:
    collect = build_round124_collect_only_recovery_verification(
        return_code=2,
        collected_tests=1267,
        remaining_errors=[
            {"path": "test_episodic.py", "missing_import": "working_memory"},
            {"path": "test_eve_main_ab.py", "missing_import": "working_memory"},
        ],
    )

    assert collect["collect_recovery_version"] == ROUND124_COLLECT_ONLY_RECOVERY_VERIFICATION_VERSION
    assert collect["collect_recovery_status"] == "collect_only_partial_new_blockers_after_spreading_activation_recovery"
    assert collect["spreading_activation_import_errors_remaining"] == 0
    assert collect["critical_blocker_improved"] is True
    assert collect["broader_validation_status"] == "blocked_partial"
    assert collect["production_persistence_enabled"] is False


def test_round122_124_export_writes_json_without_activation(tmp_path: Path) -> None:
    report = build_round124_collect_only_recovery_verification(return_code=0, collected_tests=1271)
    export = write_round_report(tmp_path / "round124.json", report)

    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert (tmp_path / "round124.json").read_text(encoding="utf-8").startswith("{\n")
