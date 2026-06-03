from __future__ import annotations

import json
from pathlib import Path

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_mapping_persistence_approval_fixture import (
    ROUND109_APPROVAL_FIXTURE_VERSION,
    ROUND109_APPROVAL_SCOPE,
    ROUND109_TEST_DRY_RUN_MODE,
    ROUND109_TOKEN_ALLOWLIST,
    build_round109_operator_approval_fixture,
    run_round109_runtime_mapping_persistence_approval_fixture_drill,
    write_round109_runtime_mapping_persistence_approval_fixture,
)
from adapters.state_debug_adapter import StateDebugAdapter


class _DummyStore:
    def __init__(self) -> None:
        self._count = 1

    def stats(self) -> dict:
        return {"count": self._count, "dimension": 300, "read_only": True}


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


def test_round109_fixture_is_deterministic_limited_scope_and_blocks_extra_tokens() -> None:
    fixture = build_round109_operator_approval_fixture()

    assert fixture["fixture_version"] == ROUND109_APPROVAL_FIXTURE_VERSION
    assert fixture["approval_scope"] == ROUND109_APPROVAL_SCOPE
    assert fixture["approval_valid"] is True
    assert fixture["allowed_tokens"] == list(ROUND109_TOKEN_ALLOWLIST) == ["민석"]
    assert fixture["requested_tokens"] == ["민석"]
    assert fixture["test_dry_run_mode"] == ROUND109_TEST_DRY_RUN_MODE
    assert fixture["runtime_mapping_enabled_default"] is False
    assert fixture["enforcement_enabled_default"] is False
    assert fixture["actual_persistence_enablement"] is False

    blocked = build_round109_operator_approval_fixture(tokens=["민석", "EVE"])
    assert blocked["approval_valid"] is False
    assert blocked["approval_status"] == "fixture_blocked"
    assert "token_allowlist_explicit" in blocked["blocking_reasons"]


def test_round109_drill_approves_audits_rolls_back_and_exports_snapshots(tmp_path: Path) -> None:
    engine = _DummyEngine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_audit = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats()
    before_active = dict(engine.activation_adapter.sa.activations)
    before_concept_memory = engine.concept_memory.stats()
    before_agp = engine.agp.stats()

    report = run_round109_runtime_mapping_persistence_approval_fixture_drill(
        engine=engine,
        artifact_dir=tmp_path,
        changed_paths=["adapters/runtime_mapping_persistence_approval_fixture.py"],
    )

    assert report["drill_version"] == ROUND109_APPROVAL_FIXTURE_VERSION
    assert report["drill_status"] == "approval_fixture_drill_passed"
    assert report["fixture"]["approval_scope"] == "runtime_mapping_persistence_only"
    assert report["allowed_tokens"] == ["민석"]
    assert report["test_dry_run_mode"] == ROUND109_TEST_DRY_RUN_MODE
    assert report["checkpoint_created_before_candidate_mutation"] is True
    assert report["candidate_report"]["activation_status"] == "activation_candidate_applied_then_rolled_back"
    assert report["candidate_report"]["runtime_mapping_enabled_during_candidate"] is True
    assert report["candidate_report"]["enforcement_enabled_during_candidate"] is False
    assert report["runtime_mapping_enabled_now"] is False
    assert report["enforcement_enabled_now"] is False
    assert report["rollback_restored_disabled_flags"] == {
        "runtime_mapping_enabled_false": True,
        "enforcement_enabled_false": True,
    }
    assert report["audit_order"]["actual_event_order"] == [
        "activation_precheck_started",
        "checkpoint_written",
        "activation_candidate_applied",
        "rollback_validation_passed",
    ]
    assert report["audit_order"]["event_order_verified"] is True
    assert report["audit_order"]["checkpoint_event_precedes_candidate_mutation"] is True
    assert all(report["protected_surface_checks"].values())
    assert report["state_debug_exports"]["before"] is True
    assert report["state_debug_exports"]["after_candidate"] is True
    assert report["state_debug_exports"]["after_rollback"] is True
    assert report["vectors_npy_committed"] is False
    assert report["forbidden_artifact_paths"] == []
    assert report["forbidden_pr_diff_paths"] == []
    assert report["agp_bypass_used"] is False

    assert lcm.runtime_mapping_enabled is False
    assert lcm.enforcement_enabled is False
    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert dict(engine.activation_adapter.sa.activations) == before_active
    assert engine.concept_memory.stats() == before_concept_memory
    assert engine.agp.stats() == before_agp

    checkpoint = json.loads(Path(report["candidate_report"]["checkpoint_path"]).read_text(encoding="utf-8"))
    assert checkpoint["checkpoint_created_before_candidate_mutation"] is True
    assert checkpoint["mapped_tokens"] == ["민석"]
    assert checkpoint["runtime_mapping_enabled_before"] is False
    assert checkpoint["enforcement_enabled_before"] is False
    assert checkpoint["contains_vectors_npy"] is False

    after_debug = json.loads(Path(report["state_debug_exports"]["paths"]["after_candidate"]).read_text(encoding="utf-8"))
    assert after_debug["state_debug"]["lex_concept_mapping"]["runtime_mapping_enabled"] is True
    assert after_debug["state_debug"]["lex_concept_mapping"]["enforcement_enabled"] is False

    rollback_debug = json.loads(Path(report["state_debug_exports"]["paths"]["after_rollback"]).read_text(encoding="utf-8"))
    assert rollback_debug["state_debug"]["lex_concept_mapping"]["runtime_mapping_enabled"] is False
    assert rollback_debug["state_debug"]["lex_concept_mapping"]["enforcement_enabled"] is False
    assert Path(report["approval_fixture_path"]).exists()
    assert Path(report["drill_summary_path"]).exists()


def test_round109_state_debug_surface_advertises_fixture_without_default_enablement() -> None:
    engine = _DummyEngine()
    snapshot = StateDebugAdapter(engine).snapshot_state()
    lcm_state = snapshot["lex_concept_mapping"]

    assert lcm_state["runtime_mapping_persistence_approval_fixture_version"] == ROUND109_APPROVAL_FIXTURE_VERSION
    assert lcm_state["runtime_mapping_persistence_approval_fixture_available"] is True
    assert lcm_state["runtime_mapping_enabled"] is False
    assert lcm_state["enforcement_enabled"] is False
    fixture = lcm_state["runtime_mapping_persistence_approval_fixture"]
    assert fixture["approval_scope"] == ROUND109_APPROVAL_SCOPE
    assert fixture["allowed_tokens"] == ["민석"]
    assert fixture["test_dry_run_mode_only"] is True
    assert fixture["rollback_verification_required"] is True
    assert fixture["actual_persistence_enablement"] is False
    assert fixture["vectors_npy_committed"] is False


def test_round109_export_writes_json_without_enabling_mapping(tmp_path: Path) -> None:
    report = run_round109_runtime_mapping_persistence_approval_fixture_drill(
        engine=_DummyEngine(),
        artifact_dir=tmp_path / "drill",
    )
    export = write_round109_runtime_mapping_persistence_approval_fixture(tmp_path / "round109.json", report)

    assert export["export_version"] == ROUND109_APPROVAL_FIXTURE_VERSION
    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert export["vectors_npy_committed"] is False
    assert export["agp_bypass_used"] is False
    assert export["read_only_fixture"] is True
