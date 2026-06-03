from __future__ import annotations

import json
from pathlib import Path

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_mapping_limited_persistence_sandbox import (
    ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_VERSION,
    ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_VERSION,
    ROUND115_BROADER_VALIDATION_TRIAGE_VERSION,
    ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_VERSION,
    ROUND117_OPERATOR_GO_NO_GO_PACKAGE_VERSION,
    build_round113_state_debug_audit_replay_viewer,
    build_round114_legacy_root_blocker_isolation,
    build_round115_broader_validation_triage_report,
    build_round117_operator_go_no_go_package,
    run_round110_runtime_mapping_limited_persistence_sandbox,
    run_round111_sandbox_rollback_cleanup_verification,
    run_round112_post_sandbox_focused_validation_audit_replay,
    run_round116_runtime_mapping_sandbox_replay_regression_guard,
    write_round113_state_debug_audit_replay_viewer,
    write_round114_legacy_root_blocker_isolation,
    write_round115_broader_validation_triage_report,
    write_round116_runtime_mapping_sandbox_replay_regression_guard,
    write_round117_operator_go_no_go_package,
)
from adapters.state_debug_adapter import StateDebugAdapter


class _DummyStore:
    def stats(self) -> dict:
        return {"count": 1, "dimension": 300, "read_only": True}


class _DummyWrapper:
    def telemetry(self) -> dict:
        return {"total_calls": 0, "primary_hits": 0, "fallback_uses": 0, "eve_specific_hits": 0}


class _DummySA:
    def __init__(self) -> None:
        self.activations = {"concept_category::lex::민석": 1.0}


class _DummyActivation:
    def __init__(self) -> None:
        self.sa = _DummySA()


class _DummyStats:
    def __init__(self, name: str) -> None:
        self.name = name

    def stats(self) -> dict:
        return {"name": self.name, "read_only": True}


class _DummyEngine:
    def __init__(self) -> None:
        self.lex_concept_mapping = LexConceptMappingAdapter(self)
        self.eve_specific_vector_store = _DummyStore()
        self.self_embedding = _DummyWrapper()
        self.activation_adapter = _DummyActivation()
        self.concept_memory = _DummyStats("concept_memory")
        self.agp = _DummyStats("agp")


def _prepare_round110_112(engine: _DummyEngine, tmp_path: Path) -> None:
    run_round110_runtime_mapping_limited_persistence_sandbox(engine=engine, artifact_dir=tmp_path)
    run_round111_sandbox_rollback_cleanup_verification(engine=engine, artifact_dir=tmp_path)
    run_round112_post_sandbox_focused_validation_audit_replay(engine=engine, artifact_dir=tmp_path)


def test_round113_state_debug_audit_replay_viewer_is_read_only(tmp_path: Path) -> None:
    engine = _DummyEngine()
    _prepare_round110_112(engine, tmp_path)
    before_categories = engine.lex_concept_mapping.concept_categories_snapshot()

    viewer = build_round113_state_debug_audit_replay_viewer(engine=engine, artifact_dir=tmp_path)

    assert viewer["viewer_version"] == ROUND113_STATE_DEBUG_AUDIT_REPLAY_VIEWER_VERSION
    assert viewer["viewer_status"] == "state_debug_audit_replay_viewer_ready"
    assert viewer["source_rounds"] == [110, 111, 112]
    assert viewer["debug_timeline"][1]["runtime_mapping_enabled"] is True
    assert viewer["debug_timeline"][2]["runtime_mapping_enabled"] is False
    assert viewer["viewer_checks"]["production_persistence_enabled"] is False
    assert viewer["vectors_npy_committed"] is False
    assert viewer["read_only"] is True
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False
    assert engine.lex_concept_mapping.concept_categories_snapshot() == before_categories


def test_round114_isolates_legacy_root_collection_blocker(tmp_path: Path) -> None:
    (tmp_path / "test_legacy.py").write_text("from spreading_activation import SpreadingActivation\n", encoding="utf-8")
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    report = build_round114_legacy_root_blocker_isolation(repo_root=tmp_path)

    assert report["blocker_isolation_version"] == ROUND114_LEGACY_ROOT_BLOCKER_ISOLATION_VERSION
    assert report["isolation_status"] == "legacy_root_blocker_isolated"
    assert report["isolated_blockers"] == [
        {
            "file": "test_legacy.py",
            "missing_import": "spreading_activation",
            "blocker_type": "legacy_root_collection_import_error",
            "isolated": True,
        }
    ]
    assert report["checks"]["missing_spreading_activation_module"] is True
    assert report["production_persistence_enabled"] is False


def test_round115_triages_broader_validation_as_partial_without_weakening_tests(tmp_path: Path) -> None:
    (tmp_path / "test_legacy.py").write_text("import spreading_activation\n", encoding="utf-8")
    blocker = build_round114_legacy_root_blocker_isolation(repo_root=tmp_path)

    triage = build_round115_broader_validation_triage_report(
        focused_results=[{"command": "pytest -q tests/test_v3_round113_117_runtime_mapping_validation_loop.py", "status": "passed"}],
        blocker_report=blocker,
    )

    assert triage["triage_version"] == ROUND115_BROADER_VALIDATION_TRIAGE_VERSION
    assert triage["triage_status"] == "focused_passed_broader_blocked_or_partial"
    assert triage["blocked_partial_validation"] is True
    assert triage["broader_validation"][0]["status"] == "blocked_pre_existing"
    assert triage["runtime_mapping_enabled_default"] is False
    assert triage["enforcement_enabled_default"] is False


def test_round116_regression_guard_replays_sandbox_and_cleans_state(tmp_path: Path) -> None:
    engine = _DummyEngine()

    guard = run_round116_runtime_mapping_sandbox_replay_regression_guard(engine=engine, artifact_dir=tmp_path)

    assert guard["regression_guard_version"] == ROUND116_SANDBOX_REPLAY_REGRESSION_GUARD_VERSION
    assert guard["regression_status"] == "sandbox_replay_regression_guard_passed"
    assert guard["source_rounds_replayed"] == [110, 111, 112]
    assert guard["regression_checks"]["sandbox_state_removed"] is True
    assert guard["runtime_mapping_enabled_now"] is False
    assert guard["enforcement_enabled_now"] is False
    assert guard["forbidden_artifact_paths"] == []
    assert (tmp_path / "runtime_mapping_round110_sandbox_state.json").exists() is False
    assert (tmp_path / "runtime_mapping_round116_sandbox_replay_regression_guard.json").exists()


def test_round117_operator_package_stays_no_go_for_production_persistence(tmp_path: Path) -> None:
    engine = _DummyEngine()
    _prepare_round110_112(engine, tmp_path / "round113")
    round113 = build_round113_state_debug_audit_replay_viewer(engine=engine, artifact_dir=tmp_path / "round113")
    round114 = build_round114_legacy_root_blocker_isolation(repo_root=Path.cwd())
    round115 = build_round115_broader_validation_triage_report(blocker_report=round114)
    round116 = run_round116_runtime_mapping_sandbox_replay_regression_guard(engine=engine, artifact_dir=tmp_path / "round116")

    package = build_round117_operator_go_no_go_package(
        round113=round113,
        round114=round114,
        round115=round115,
        round116=round116,
    )

    assert package["go_no_go_package_version"] == ROUND117_OPERATOR_GO_NO_GO_PACKAGE_VERSION
    assert package["completed_rounds"] == [113, 114, 115, 116]
    assert package["operator_recommendation"] == "no_go_for_production_persistence_in_this_pr"
    assert package["production_persistence_enabled"] is False
    assert package["runtime_mapping_enabled_default"] is False
    assert package["enforcement_enabled_default"] is False
    assert package["vectors_npy_committed"] is False


def test_round113_117_state_debug_and_export_surfaces(tmp_path: Path) -> None:
    engine = _DummyEngine()
    _prepare_round110_112(engine, tmp_path / "artifacts")
    round113 = build_round113_state_debug_audit_replay_viewer(engine=engine, artifact_dir=tmp_path / "artifacts")
    round114 = build_round114_legacy_root_blocker_isolation(repo_root=tmp_path)
    round115 = build_round115_broader_validation_triage_report(blocker_report=round114)
    round116 = run_round116_runtime_mapping_sandbox_replay_regression_guard(engine=engine, artifact_dir=tmp_path / "guard")
    round117 = build_round117_operator_go_no_go_package(round113=round113, round114=round114, round115=round115, round116=round116)

    snapshot = StateDebugAdapter(engine).snapshot_state()["lex_concept_mapping"]
    assert snapshot["state_debug_audit_replay_viewer_available"] is True
    assert snapshot["legacy_root_blocker_isolation_available"] is True
    assert snapshot["broader_validation_triage_report_available"] is True
    assert snapshot["sandbox_replay_regression_guard_available"] is True
    assert snapshot["operator_go_no_go_package_available"] is True
    assert snapshot["runtime_mapping_enabled"] is False
    assert snapshot["enforcement_enabled"] is False

    exports = [
        write_round113_state_debug_audit_replay_viewer(tmp_path / "round113.json", round113),
        write_round114_legacy_root_blocker_isolation(tmp_path / "round114.json", round114),
        write_round115_broader_validation_triage_report(tmp_path / "round115.json", round115),
        write_round116_runtime_mapping_sandbox_replay_regression_guard(tmp_path / "round116.json", round116),
        write_round117_operator_go_no_go_package(tmp_path / "round117.json", round117),
    ]
    assert all(export["json_written"] for export in exports)
    assert all(export["vectors_npy_committed"] is False for export in exports)
    assert json.loads((tmp_path / "round117.json").read_text(encoding="utf-8"))["operator_recommendation"] == "no_go_for_production_persistence_in_this_pr"
