from __future__ import annotations

import json
from pathlib import Path

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_mapping_limited_persistence_sandbox import (
    ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION,
    ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION,
    ROUND112_SANDBOX_AUDIT_REPLAY_VERSION,
    run_round110_runtime_mapping_limited_persistence_sandbox,
    run_round111_sandbox_rollback_cleanup_verification,
    run_round112_post_sandbox_focused_validation_audit_replay,
    write_round110_runtime_mapping_limited_persistence_sandbox,
    write_round111_sandbox_rollback_cleanup_verification,
    write_round112_post_sandbox_focused_validation_audit_replay,
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


def test_round110_limited_persistence_sandbox_writes_json_only_and_rolls_back(tmp_path: Path) -> None:
    engine = _DummyEngine()
    before_categories = engine.lex_concept_mapping.concept_categories_snapshot()
    before_audit = engine.lex_concept_mapping.concept_commit_records()

    report = run_round110_runtime_mapping_limited_persistence_sandbox(engine=engine, artifact_dir=tmp_path)

    assert report["sandbox_version"] == ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION
    assert report["sandbox_status"] == "limited_persistence_sandbox_passed"
    assert report["mapped_tokens"] == ["민석"]
    assert report["runtime_mapping_enabled_during_sandbox"] is True
    assert report["enforcement_enabled_during_sandbox"] is False
    assert report["runtime_mapping_enabled_now"] is False
    assert report["enforcement_enabled_now"] is False
    assert report["production_persistence_enabled"] is False
    assert report["sandbox_state_written"] is True
    assert report["event_order"]["actual_event_order"] == [
        "sandbox_precheck_started",
        "sandbox_checkpoint_written",
        "sandbox_mapping_persisted_json_only",
        "sandbox_rollback_applied",
    ]
    assert report["event_order"]["event_order_verified"] is True
    assert report["rollback_verified"] is True
    assert report["forbidden_artifact_paths"] == []
    assert report["vectors_npy_committed"] is False
    assert report["agp_bypass_used"] is False
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False
    assert engine.lex_concept_mapping.concept_categories_snapshot() == before_categories
    assert engine.lex_concept_mapping.concept_commit_records() == before_audit

    sandbox_state = json.loads(Path(report["sandbox_state_path"]).read_text(encoding="utf-8"))
    assert sandbox_state["scope"] == "limited_sandbox_json_only"
    assert sandbox_state["contains_vectors_npy"] is False

    during_debug = json.loads(Path(report["state_debug_paths"]["during"]).read_text(encoding="utf-8"))
    assert during_debug["state_debug"]["lex_concept_mapping"]["runtime_mapping_enabled"] is True
    after_debug = json.loads(Path(report["state_debug_paths"]["after_rollback"]).read_text(encoding="utf-8"))
    assert after_debug["state_debug"]["lex_concept_mapping"]["runtime_mapping_enabled"] is False


def test_round110_blocks_without_operator_approval(tmp_path: Path) -> None:
    engine = _DummyEngine()
    report = run_round110_runtime_mapping_limited_persistence_sandbox(
        engine=engine,
        artifact_dir=tmp_path,
        operator_approved=False,
    )

    assert report["sandbox_status"] == "blocked_guard_failed"
    assert "operator_approved" in report["blocking_reasons"]
    assert report["sandbox_state_written"] is False
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False


def test_round111_cleanup_removes_sandbox_state_and_preserves_disabled_flags(tmp_path: Path) -> None:
    engine = _DummyEngine()
    round110 = run_round110_runtime_mapping_limited_persistence_sandbox(engine=engine, artifact_dir=tmp_path)
    assert Path(round110["sandbox_state_path"]).exists()

    cleanup = run_round111_sandbox_rollback_cleanup_verification(engine=engine, artifact_dir=tmp_path)

    assert cleanup["cleanup_version"] == ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION
    assert cleanup["cleanup_status"] == "sandbox_cleanup_verified"
    assert cleanup["checks"]["sandbox_state_removed"] is True
    assert Path(round110["sandbox_state_path"]).exists() is False
    assert cleanup["cleanup_event_order"]["actual_event_order"] == [
        "cleanup_precheck_started",
        "sandbox_state_file_removed",
        "cleanup_verification_passed",
    ]
    assert cleanup["runtime_mapping_enabled_now"] is False
    assert cleanup["enforcement_enabled_now"] is False
    assert cleanup["production_persistence_enabled"] is False
    assert cleanup["forbidden_artifact_paths"] == []
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False


def test_round112_replays_round110_and_round111_audits_read_only(tmp_path: Path) -> None:
    engine = _DummyEngine()
    run_round110_runtime_mapping_limited_persistence_sandbox(engine=engine, artifact_dir=tmp_path)
    run_round111_sandbox_rollback_cleanup_verification(engine=engine, artifact_dir=tmp_path)

    replay = run_round112_post_sandbox_focused_validation_audit_replay(engine=engine, artifact_dir=tmp_path)

    assert replay["replay_version"] == ROUND112_SANDBOX_AUDIT_REPLAY_VERSION
    assert replay["replay_status"] == "post_sandbox_audit_replay_passed"
    assert replay["source_rounds"] == [110, 111]
    assert all(
        value
        for key, value in replay["replay_checks"].items()
        if key not in {"production_persistence_enabled", "vectors_npy_committed", "agp_bypass_used"}
    )
    assert replay["replay_checks"]["production_persistence_enabled"] is False
    assert replay["replay_checks"]["vectors_npy_committed"] is False
    assert replay["replay_checks"]["agp_bypass_used"] is False
    assert replay["runtime_mapping_enabled_now"] is False
    assert replay["enforcement_enabled_now"] is False
    assert replay["read_only"] is True
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False


def test_round110_112_state_debug_surface_advertises_sandbox_without_default_enablement() -> None:
    snapshot = StateDebugAdapter(_DummyEngine()).snapshot_state()
    lcm = snapshot["lex_concept_mapping"]

    assert lcm["runtime_mapping_limited_persistence_sandbox_version"] == ROUND110_LIMITED_PERSISTENCE_SANDBOX_VERSION
    assert lcm["runtime_mapping_sandbox_rollback_cleanup_version"] == ROUND111_SANDBOX_ROLLBACK_CLEANUP_VERSION
    assert lcm["runtime_mapping_sandbox_audit_replay_version"] == ROUND112_SANDBOX_AUDIT_REPLAY_VERSION
    assert lcm["runtime_mapping_limited_persistence_sandbox_available"] is True
    assert lcm["runtime_mapping_sandbox_rollback_cleanup_available"] is True
    assert lcm["runtime_mapping_sandbox_audit_replay_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["runtime_mapping_limited_persistence_sandbox"]["production_persistence_enabled"] is False


def test_round110_112_export_helpers_write_status_json(tmp_path: Path) -> None:
    engine = _DummyEngine()
    round110 = run_round110_runtime_mapping_limited_persistence_sandbox(engine=engine, artifact_dir=tmp_path / "artifacts")
    round111 = run_round111_sandbox_rollback_cleanup_verification(engine=engine, artifact_dir=tmp_path / "artifacts")
    round112 = run_round112_post_sandbox_focused_validation_audit_replay(engine=engine, artifact_dir=tmp_path / "artifacts")

    export110 = write_round110_runtime_mapping_limited_persistence_sandbox(tmp_path / "round110.json", round110)
    export111 = write_round111_sandbox_rollback_cleanup_verification(tmp_path / "round111.json", round111)
    export112 = write_round112_post_sandbox_focused_validation_audit_replay(tmp_path / "round112.json", round112)

    assert export110["json_written"] is True
    assert export111["json_written"] is True
    assert export112["json_written"] is True
    assert export110["runtime_mapping_enabled"] is False
    assert export111["runtime_mapping_enabled"] is False
    assert export112["runtime_mapping_enabled"] is False
