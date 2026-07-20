from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/audit/m0_c_affect_migration_check.py"
PLAN = REPO_ROOT / "docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md"
ALLOWED_STATUSES = {"MAPPED", "PROPOSED-DROP", "UNRESOLVED"}
ALLOWED_DRIVES = {
    "energy", "safety", "affiliation", "curiosity",
    "agency", "coherence", "competence", "expression",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("m0_c_affect_migration_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return _load_module().audit_repository(REPO_ROOT)


def test_required_files_exist():
    required = {
        "docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md",
        "scripts/audit/m0_c_affect_migration_check.py",
        "tests/audit/test_m0_c_affect_migration_check.py",
        "docs/EVE_IMPLEMENTATION_STATUS_v4.md",
    }
    assert all((REPO_ROOT / path).is_file() for path in required)


def test_schema_and_static_scope(report):
    assert report["schema_version"] == "1.0.0-m0-c-affect-migration"
    assert report["baseline_sha"] == "28ec113a8ee371fdc6ac13341c0d70e00db26ce4"
    assert report["validation_errors"] == []
    assert report["scope"] == {
        "static_analysis_only": True,
        "runtime_import_performed": False,
        "runtime_execution_performed": False,
        "migration_execution_performed": False,
        "projection_implementation_performed": False,
        "production_state_read_performed": False,
        "production_state_write_performed": False,
        "production_source_mutation_performed": False,
        "generated_json_committed": False,
    }


def test_axis_extraction_has_two_nonempty_source_families(report):
    summary = report["summary"]
    assert summary["legacy_mutable_hormone_axes"] > 0
    assert summary["read_only_affect_registry_axes"] > 0
    assert summary["authoritative_found_axes"] == (
        summary["legacy_mutable_hormone_axes"]
        + summary["read_only_affect_registry_axes"]
    )
    assert len(report["axes"]) == summary["authoritative_found_axes"]


def test_every_axis_has_exactly_one_mapping_without_pinning_counts(report):
    axes = [entry["axis"] for entry in report["axes"]]
    assert len(axes) == len(set(axes))
    mappings = [entry["mapping"] for entry in report["axes"]]
    assert all(mapping is not None for mapping in mappings)
    assert len(mappings) == report["summary"]["mapping_rows"]
    assert {mapping["axis"] for mapping in mappings} == set(axes)


def test_mapping_schema_and_target_drive_vocabulary(report):
    for entry in report["axes"]:
        mapping = entry["mapping"]
        assert mapping["status"] in ALLOWED_STATUSES
        assert set(mapping["target_drives"]) <= ALLOWED_DRIVES
        assert mapping["confidence"] in {"low", "medium", "high"}
        assert mapping["rationale"]
        evidence = f"{entry['path']}:{entry['line_start']}"
        assert evidence in mapping["evidence"]
        if mapping["status"] == "MAPPED":
            assert (
                mapping["target_drives"]
                or mapping["appraisal_dimensions"]
                or mapping["derived_emotion"]
            )
        elif mapping["status"] == "PROPOSED-DROP":
            assert not mapping["target_drives"]
            assert not mapping["appraisal_dimensions"]
            assert not mapping["derived_emotion"]
            assert mapping["preservation"] not in {"", "—"}
        else:
            assert mapping["open_question"] not in {"", "—"}


def test_compatibility_keys_are_reported_separately(report):
    keys = report["compatibility_keys"]
    assert keys
    assert all("key" in item and "path" in item and "line_start" in item for item in keys)
    axis_names = {entry["axis"] for entry in report["axes"]}
    assert not axis_names.intersection({item["key"] for item in keys})


def test_persistence_evidence_does_not_claim_axis_specific_snapshot_keys(report):
    evidence = report["persistence_container_evidence"]
    assert evidence
    assert all(item["axis_specific_snapshot_keys_found"] is False for item in evidence)
    assert all(item["path"] == "adapters/persistence_adapter.py" for item in evidence)


def test_required_plan_sections_and_evidence_counts_present():
    text = PLAN.read_text(encoding="utf-8")
    module = _load_module()
    for heading in module.REQUIRED_SECTIONS:
        assert heading in text
    for required in ("1,777", "54", "43"):
        assert required in text
    assert "M0_C_REQUIRED_MIGRATION_PLAN_ABSENT" in text


def test_cli_output_is_byte_identical(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [sys.executable, str(SCRIPT), "--output"]
    subprocess.check_call(command + [str(first)], cwd=REPO_ROOT)
    subprocess.check_call(command + [str(second)], cwd=REPO_ROOT)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["summary"]["validation_errors"] == 0


def test_cli_fail_on_unresolved_matches_report(report):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fail-on-unresolved", "--summary-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if report["summary"]["unresolved"]:
        assert result.returncode == 3
        assert "unresolved mappings remain" in result.stderr
    else:
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["unresolved"] == 0
