from __future__ import annotations

import json
from pathlib import Path

from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_mapping_persistence_activation_candidate import (
    ROUND108_ACTIVATION_CANDIDATE_VERSION,
    ROUND108_OPERATOR_APPROVAL_TOKEN,
    build_operator_approval_guard,
    run_runtime_mapping_persistence_activation_candidate,
    verify_runtime_mapping_rollback,
    write_round108_runtime_mapping_persistence_activation_candidate,
)
from adapters.runtime_mapping_persistence_activation_dryrun import build_runtime_mapping_persistence_activation_dryrun
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


def _source_decision() -> dict:
    return {
        "decision_version": "v3_round106_runtime_mapping_persistence_decision",
        "decision_status": "persistence_ready_but_not_applied",
        "decision_ready": True,
        "mapped_tokens": ["민석"],
        "runtime_mapping_persisted_now": False,
    }


def _source_dry_run(engine: _DummyEngine) -> dict:
    return build_runtime_mapping_persistence_activation_dryrun(engine=engine, source_decision=_source_decision())


def test_round108_operator_guard_fails_closed_without_explicit_approval() -> None:
    engine = _DummyEngine()
    dry_run = _source_dry_run(engine)

    guard = build_operator_approval_guard(
        source_decision=_source_decision(),
        source_dry_run=dry_run,
        operator_approved=False,
        operator_approval_token=None,
    )

    assert guard["guard_passed"] is False
    assert "operator_approved" in guard["blocking_reasons"]
    assert "operator_token_ok" in guard["blocking_reasons"]

    report = run_runtime_mapping_persistence_activation_candidate(
        engine=engine,
        source_decision=_source_decision(),
        source_dry_run=dry_run,
        operator_approved=False,
        operator_approval_token=None,
        apply_candidate=True,
    )

    assert report["candidate_version"] == ROUND108_ACTIVATION_CANDIDATE_VERSION
    assert report["activation_status"] == "blocked_operator_guard_or_apply_flag_missing"
    assert report["checkpoint_written_before_mutation"] is False
    assert report["candidate_mutation_applied"] is False
    assert engine.lex_concept_mapping.runtime_mapping_enabled is False
    assert engine.lex_concept_mapping.enforcement_enabled is False
    assert report["vectors_npy_committed"] is False
    assert report["agp_bypass_used"] is False


def test_round108_apply_candidate_requires_checkpoint_then_rolls_back_and_audits(tmp_path: Path) -> None:
    engine = _DummyEngine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_audit = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats()
    before_active = dict(engine.activation_adapter.sa.activations)
    dry_run = _source_dry_run(engine)

    report = run_runtime_mapping_persistence_activation_candidate(
        engine=engine,
        source_decision=_source_decision(),
        source_dry_run=dry_run,
        operator_approved=True,
        operator_approval_token=ROUND108_OPERATOR_APPROVAL_TOKEN,
        artifact_dir=tmp_path,
        apply_candidate=True,
    )

    assert report["activation_status"] == "activation_candidate_applied_then_rolled_back"
    assert report["guard"]["guard_passed"] is True
    assert report["checkpoint_written_before_mutation"] is True
    assert report["runtime_mapping_enabled_during_candidate"] is True
    assert report["enforcement_enabled_during_candidate"] is False
    assert report["runtime_mapping_enabled_now"] is False
    assert report["enforcement_enabled_now"] is False
    assert report["rollback_verified"] is True
    assert lcm.runtime_mapping_enabled is False
    assert lcm.enforcement_enabled is False
    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert dict(engine.activation_adapter.sa.activations) == before_active
    assert report["vectors_npy_committed"] is False
    assert report["agp_bypass_used"] is False

    checkpoint_path = Path(report["checkpoint_path"])
    rollback_path = Path(report["rollback_path"])
    audit_path = Path(report["audit_log_path"])
    before_debug_path = Path(report["state_debug_before_path"])
    after_debug_path = Path(report["state_debug_after_path"])
    rollback_debug_path = Path(report["state_debug_after_rollback_path"])

    assert checkpoint_path.exists()
    assert rollback_path.exists()
    assert audit_path.exists()
    assert before_debug_path.exists()
    assert after_debug_path.exists()
    assert rollback_debug_path.exists()

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["checkpoint_created_before_candidate_mutation"] is True
    assert checkpoint["runtime_mapping_enabled_before"] is False
    assert checkpoint["enforcement_enabled_before"] is False
    assert checkpoint["contains_vectors_npy"] is False

    after_debug = json.loads(after_debug_path.read_text(encoding="utf-8"))
    after_lcm = after_debug["state_debug"]["lex_concept_mapping"]
    assert after_lcm["runtime_mapping_enabled"] is True
    assert after_lcm["enforcement_enabled"] is False

    rollback_debug = json.loads(rollback_debug_path.read_text(encoding="utf-8"))
    rollback_lcm = rollback_debug["state_debug"]["lex_concept_mapping"]
    assert rollback_lcm["runtime_mapping_enabled"] is False
    assert rollback_lcm["enforcement_enabled"] is False

    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == [
        "activation_precheck_started",
        "checkpoint_written",
        "activation_candidate_applied",
        "rollback_validation_passed",
    ]
    assert events[1]["mutation_boundary_checks"]["checkpoint_created_before_candidate_mutation"] is True
    assert events[2]["mutation_boundary_checks"]["vectors_npy_committed"] is False


def test_round108_rollback_verification_detects_unrestored_runtime_mapping(tmp_path: Path) -> None:
    engine = _DummyEngine()
    dry_run = _source_dry_run(engine)
    report = run_runtime_mapping_persistence_activation_candidate(
        engine=engine,
        source_decision=_source_decision(),
        source_dry_run=dry_run,
        operator_approved=True,
        operator_approval_token=ROUND108_OPERATOR_APPROVAL_TOKEN,
        artifact_dir=tmp_path,
        apply_candidate=True,
    )
    checkpoint = json.loads(Path(report["checkpoint_path"]).read_text(encoding="utf-8"))

    engine.lex_concept_mapping.runtime_mapping_enabled = True
    verification = verify_runtime_mapping_rollback(engine, checkpoint)

    assert verification["rollback_verified"] is False
    assert verification["checks"]["runtime_mapping_enabled_restored_false"] is False


def test_round108_state_debug_surface_advertises_candidate_without_default_enablement() -> None:
    engine = _DummyEngine()
    snapshot = StateDebugAdapter(engine).snapshot_state()
    lcm_state = snapshot["lex_concept_mapping"]

    assert lcm_state["runtime_mapping_persistence_activation_candidate_version"] == ROUND108_ACTIVATION_CANDIDATE_VERSION
    assert lcm_state["runtime_mapping_persistence_activation_candidate_available"] is True
    assert lcm_state["runtime_mapping_enabled"] is False
    assert lcm_state["enforcement_enabled"] is False
    candidate = lcm_state["runtime_mapping_persistence_activation_candidate"]
    assert candidate["operator_approval_required"] is True
    assert candidate["checkpoint_before_mutation_required"] is True
    assert candidate["rollback_verification_required"] is True
    assert candidate["audit_log_required"] is True
    assert candidate["actual_persistence_enablement"] is False


def test_round108_export_writes_json_without_enabling_mapping(tmp_path: Path) -> None:
    report = run_runtime_mapping_persistence_activation_candidate(
        engine=_DummyEngine(),
        source_decision=_source_decision(),
        source_dry_run=None,
        operator_approved=False,
        operator_approval_token=None,
        apply_candidate=False,
    )
    export = write_round108_runtime_mapping_persistence_activation_candidate(tmp_path / "round108.json", report)

    assert export["export_version"] == ROUND108_ACTIVATION_CANDIDATE_VERSION
    assert export["json_written"] is True
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert export["vectors_npy_committed"] is False
    assert export["agp_bypass_used"] is False
