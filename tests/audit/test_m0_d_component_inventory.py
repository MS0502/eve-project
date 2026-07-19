from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/audit/m0_d_component_inventory.py"
ALLOWED_DISPOSITIONS = {"KEEP", "WRAP", "REWRITE", "EXPERIMENTAL", "DEPRECATE", "REMOVE"}
REQUIRED_ENTRY_FIELDS = {
    "path",
    "line_start",
    "line_end",
    "symbol",
    "detection",
    "evidence",
    "classification",
    "confidence",
    "unresolved",
    "manual_only",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("m0_d_component_inventory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return _load_module().audit_repository(REPO_ROOT)


def test_required_m0_d_files_exist():
    required = {
        "scripts/audit/m0_d_component_inventory.py",
        "tests/audit/test_m0_d_component_inventory.py",
        "docs/audit/M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md",
        "docs/audit/M0_D_MODULE_DISPOSITION.md",
        "docs/EVE_IMPLEMENTATION_STATUS_v4.md",
    }
    assert all((REPO_ROOT / path).is_file() for path in required)


def test_report_schema_and_static_scope(report):
    assert report["schema_version"] == "1.0.0-m0-d"
    assert report["baseline_sha"] == "fe10cd954bdf445400ea6aa9708dd214ed761114"
    assert report["scope"] == {
        "static_analysis_only": True,
        "runtime_execution_performed": False,
        "loop_start_performed": False,
        "vector_or_model_load_performed": False,
        "production_source_mutation_performed": False,
        "frozen_pr_mutation_performed": False,
        "generated_json_committed": False,
    }
    assert set(report) == {
        "schema_version",
        "baseline_sha",
        "root",
        "scope",
        "summary",
        "component_entries",
        "life_loops",
        "module_dispositions",
        "frozen_pr_recommendations",
        "v4_runtime_conflicts",
        "unresolved_items",
        "source_audit_summaries",
    }


def test_all_entries_have_required_evidence_fields(report):
    groups = (
        report["component_entries"],
        report["life_loops"],
        report["module_dispositions"],
        report["unresolved_items"],
    )
    for entries in groups:
        assert entries
        for entry in entries:
            assert REQUIRED_ENTRY_FIELDS <= set(entry)
            assert isinstance(entry["path"], str) and entry["path"]
            assert isinstance(entry["line_start"], int) and entry["line_start"] >= 1
            assert isinstance(entry["line_end"], int) and entry["line_end"] >= entry["line_start"]
            assert entry["confidence"] in {"low", "medium", "high"}
            assert isinstance(entry["unresolved"], bool)
            assert isinstance(entry["manual_only"], bool)


def test_every_runtime_module_has_exactly_one_disposition(report):
    module = _load_module()
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in module.iter_python_files(REPO_ROOT)
        if module._is_runtime_module(path.relative_to(REPO_ROOT))
    }
    entries = report["module_dispositions"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    assert set(paths) == expected
    assert {entry["classification"] for entry in entries} <= ALLOWED_DISPOSITIONS
    assert sum(report["summary"]["module_disposition_counts"].values()) == len(entries)


def test_remove_and_deprecate_entries_are_evidence_backed(report):
    for entry in report["module_dispositions"]:
        if entry["classification"] not in {"REMOVE", "DEPRECATE"}:
            continue
        assert entry["reason"]
        assert entry["evidence_references"]
        if entry["classification"] == "REMOVE":
            assert entry["unresolved"] is False
            assert entry["confidence"] == "high"


def test_life_loop_taxonomy_and_cross_references(report):
    allowed = {
        "Vital",
        "Cognitive",
        "Goal",
        "Activity",
        "Learning",
        "Memory",
        "Social",
        "Expression",
        "no-v4-equivalent",
    }
    for entry in report["life_loops"]:
        assert entry["v4_loop_taxonomy"]
        assert set(entry["v4_loop_taxonomy"]) <= allowed
        assert entry["trigger"]
        assert isinstance(entry["mutates"], list)
        assert isinstance(entry["evidence_references"], list)


def test_frozen_pr_recommendations_are_complete_and_non_mutating(report):
    expected = {109, 97, 86, 84, 82, 11, 7, 4, 1}
    recommendations = report["frozen_pr_recommendations"]
    assert {entry["pr"] for entry in recommendations} == expected
    assert all(entry["manual_only"] is True for entry in recommendations)
    assert all(entry["unresolved"] is False for entry in recommendations)
    assert {
        entry["recommendation"] for entry in recommendations
    } <= {
        "CLOSE-PRESERVE-EVIDENCE",
        "REWRITE-AS-V4-CONTRACT",
        "ABSORB-INTO-M1",
        "KEEP-FROZEN-PENDING-M1",
    }
    assert report["scope"]["frozen_pr_mutation_performed"] is False


def test_m0_c_migration_plan_gap_is_prominent(report):
    gaps = [
        entry
        for entry in report["unresolved_items"]
        if entry["classification"] == "M0_C_REQUIRED_MIGRATION_PLAN_ABSENT"
    ]
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["confidence"] == "high"
    assert gap["unresolved"] is True
    assert gap["manual_only"] is True
    conflicts = {entry["id"]: entry for entry in report["v4_runtime_conflicts"]}
    assert conflicts["affect-migration-plan-missing"]["unresolved"] is True


def test_cli_output_is_byte_identical(tmp_path):
    first = tmp_path / "d1.json"
    second = tmp_path / "d2.json"
    subprocess.check_call([sys.executable, str(SCRIPT), "--output", str(first)], cwd=REPO_ROOT)
    subprocess.check_call([sys.executable, str(SCRIPT), "--output", str(second)], cwd=REPO_ROOT)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["scope"]["runtime_execution_performed"] is False
