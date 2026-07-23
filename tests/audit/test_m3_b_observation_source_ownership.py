from __future__ import annotations

import ast
from pathlib import Path

from scripts.audit.m3_b_observation_source_ownership import (
    BASELINE_SHA,
    EXPECTED_BLOCKERS,
    REGISTRY_PATH,
    ROOT,
    audit_repository,
    extract_legacy_axes,
    extract_registry_axes,
    inspect_legacy_read_surface,
    inspect_persistence_container,
)

SCRIPT = ROOT / "scripts/audit/m3_b_observation_source_ownership.py"
REGISTRY = ROOT / REGISTRY_PATH


def test_authoritative_source_catalog_is_exactly_26_plus_37():
    legacy = extract_legacy_axes(ROOT)
    registry = extract_registry_axes(ROOT)
    assert len(legacy) == len({row["axis"] for row in legacy}) == 26
    assert len(registry) == len({row["axis"] for row in registry}) == 37
    assert not ({row["axis"] for row in legacy} & {row["axis"] for row in registry})
    assert all(row["source_kind"] == "schema_definition_only" for row in registry)


def test_legacy_axes_are_readable_but_lack_the_m3_b_immutable_source_envelope():
    surface = inspect_legacy_read_surface(ROOT)
    assert surface["owns_hormone_system_reference"] is True
    assert surface["iterates_all_hormones"] is True
    assert surface["reads_current_level"] is True
    assert surface["axis_count_readable"] == 26
    assert surface["immutable_source_envelope_complete"] is False
    assert set(surface["missing_immutable_envelope_fields"]) == set(
        surface["required_observation_fields"]
    )
    assert surface["derived_compatibility_keys_are_not_axes"] == [
        "stress",
        "energy",
        "curiosity",
    ]


def test_legacy_persistence_owns_only_the_whole_container_not_axis_snapshot_keys():
    legacy = extract_legacy_axes(ROOT)
    evidence = inspect_persistence_container(ROOT, (row["axis"] for row in legacy))
    assert evidence["whole_hormone_system_reference"] is True
    assert evidence["axis_specific_snapshot_keys"] == []
    assert evidence["axis_specific_snapshot_contract_found"] is False
    assert evidence["persistence_authority_changed"] is False


def test_registry_factory_returns_definition_defaults_not_observed_runtime_values():
    tree = ast.parse(REGISTRY.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    entry = functions["_axis_entry"]
    factory = functions["affect_hormone_axis_registry"]
    entry_literals = {
        node.value
        for node in ast.walk(entry)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    factory_calls = {
        node.func.id
        for node in ast.walk(factory)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"default", "baseline", "min", "max"} <= entry_literals
    assert "_axis_entry" in factory_calls
    assert "source_snapshot_id" not in entry_literals
    assert "source_integrity_digest" not in entry_literals


def test_repository_preflight_reports_the_exact_two_blockers_without_claiming_completion():
    report = audit_repository(ROOT)
    assert report["baseline_sha"] == BASELINE_SHA
    assert report["axis_counts"] == {
        "legacy_mutable_hormone": 26,
        "read_only_affect_registry": 37,
        "total": 63,
    }
    assert tuple(report["blockers"]) == EXPECTED_BLOCKERS
    assert report["errors"] == []
    assert report["source_family_readiness"] == {
        "legacy_mutable_hormone": "READABLE_UNVERSIONED_LEGACY_CONTAINER",
        "read_only_affect_registry": "DEFINITION_ONLY_NO_OBSERVED_VALUE_OWNER",
    }
    assert report["strict_63_axis_observation_ready"] is False
    assert report["observation_ready_axis_count"] == 0
    assert report["observation_window_started"] is False
    assert report["observation_window_satisfied"] is False
    assert report["m3_b_complete"] is False
    assert report["m3_c_open"] is False


def test_registry_scan_does_not_treat_defaults_or_projection_rules_as_value_ownership():
    report = audit_repository(ROOT)
    usage = report["registry_usage"]
    assert usage["observed_value_owner_found"] is False
    assert usage["production_value_store_candidate_count"] == 0
    assert usage["production_value_store_candidates"] == []
    assert usage["tracked_parse_errors_are_not_source_ownership_evidence"] is True


def test_preflight_is_deterministic_and_digest_is_recalculable():
    first = audit_repository(ROOT)
    second = audit_repository(ROOT)
    assert first == second
    assert len(first["report_digest"]) == 64


def test_preflight_has_no_runtime_import_or_live_action_surface():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not imported & {
        "adapters",
        "core",
        "hormone_system",
        "language",
        "main",
        "persistence",
        "sqlite3",
        "threading",
        "time",
    }
    assert not calls & {
        "append_event",
        "build_full_engine",
        "connect",
        "save",
        "start",
        "stimulate",
        "tick",
        "update",
    }
