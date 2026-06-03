"""EVE v3 rounds97-98 — controlled runtime mapping enable smoke and audit."""

from __future__ import annotations

import json
from typing import Any

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    run_round90_concept_commit_delta_replay_report,
    run_round91_concept_commit_replay_export_checkpoint,
    run_round92_runtime_mapping_gate_dry_run,
    run_round93_runtime_mapping_proposal_report,
    run_round94_runtime_mapping_enforcement_dry_run,
    run_round95_runtime_mapping_operator_acceptance_fixture,
    run_round96_runtime_mapping_enable_smoke_precheck,
    run_round97_controlled_runtime_mapping_enable_smoke,
    run_round98_runtime_mapping_persistence_gate_audit,
    write_round97_controlled_runtime_mapping_enable_smoke,
    write_round98_runtime_mapping_persistence_gate_audit,
)


def _prepare_round96_engine() -> tuple[object, dict[str, Any]]:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round97_test_a")
    learner.observe_text("민석 군대", source="round97_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    source_commit = run_round89_explicit_concept_commit_smoke(engine)
    source_replay = run_round90_concept_commit_delta_replay_report(engine, source_commit_report=source_commit)
    checkpoint = run_round91_concept_commit_replay_export_checkpoint(
        engine,
        source_commit_report=source_commit,
        source_replay_report=source_replay,
    )
    dry_run = run_round92_runtime_mapping_gate_dry_run(
        engine,
        tokens=["민석", "EVE"],
        source_checkpoint=checkpoint,
    )
    proposal = run_round93_runtime_mapping_proposal_report(engine, source_dry_run=dry_run)
    enforcement = run_round94_runtime_mapping_enforcement_dry_run(
        engine,
        tokens=["민석", "EVE"],
        source_proposal=proposal,
    )
    fixture = run_round95_runtime_mapping_operator_acceptance_fixture(
        engine,
        source_enforcement=enforcement,
        accepted_tokens=["민석"],
    )
    precheck = run_round96_runtime_mapping_enable_smoke_precheck(engine, source_fixture=fixture)
    return engine, precheck


def test_round97_controlled_enable_smoke_rolls_back_runtime_mapping() -> None:
    engine, precheck = _prepare_round96_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)

    assert report["smoke_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert report["round"] == 97
    assert report["source_precheck_version"] == "v3_round96_runtime_mapping_enable_smoke_precheck"
    assert report["runtime_mapping_enabled_before"] is False
    assert report["runtime_mapping_enabled_during_smoke"] is True
    assert report["runtime_mapping_enabled_after_rollback"] is False
    assert report["enforcement_enabled_during_smoke"] is False
    assert report["enforcement_enabled_after_rollback"] is False
    assert report["candidate_count"] == 1
    assert report["mapped_count"] == 1
    assert report["blocked_count"] == 0
    assert report["mapped_tokens"] == ["민석"]
    assert report["rollback"]["rollback_complete"] is True
    assert report["rollback"]["ephemeral_mapping_table_cleared"] is True
    assert report["post_smoke_recommendation"]["may_attempt_persistence_gate_next_round"] is True

    row = report["smoke_rows"][0]
    assert row["smoke_status"] == "controlled_runtime_mapping_success"
    assert row["runtime_mapping_applied_during_smoke"] is True
    assert row["runtime_mapping_persisted"] is False
    assert row["mapping_result"]["mapping_status"] == "ephemeral_runtime_mapping_success"
    assert row["mapping_result"]["anchor_source"] == "explicit_category_plus_sa_activation_only"
    assert row["agp_verify_called_during_runtime_mapping_enable_smoke"] is False
    assert row["embedding_lookup_called_during_enable_smoke"] is False

    checks = report["mutation_checks"]
    assert checks["category_snapshot_unchanged_during_enable_smoke"] is True
    assert checks["concept_commit_audit_unchanged_during_enable_smoke"] is True
    assert checks["eve_specific_vector_store_unchanged_during_enable_smoke"] is True
    assert checks["wrapper_telemetry_unchanged_during_enable_smoke"] is True
    assert checks["sa_active_categories_unchanged_during_enable_smoke"] is True
    assert checks["runtime_mapping_persisted"] is False
    assert report["runner_safety_checks"]["runtime_mapping_enabled_after_runner"] is False
    assert report["runner_safety_checks"]["enforcement_enabled_after_runner"] is False

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round98_persistence_gate_audit_keeps_runtime_mapping_disabled() -> None:
    engine, precheck = _prepare_round96_engine()
    smoke = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)
    audit = run_round98_runtime_mapping_persistence_gate_audit(engine, source_smoke=smoke)

    assert audit["audit_version"] == "v3_round98_runtime_mapping_persistence_gate_audit"
    assert audit["round"] == 98
    assert audit["source_smoke_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert audit["runtime_mapping_enabled"] is False
    assert audit["enforcement_enabled"] is False
    assert audit["mapped_count"] == 1
    assert audit["blocked_count"] == 0
    assert audit["hard_stop"] is False
    assert audit["persistence_gate_status"] == "ready_for_operator_persistence_decision"
    assert audit["operator_recommendation"]["persist_runtime_mapping_now"] is False
    assert audit["operator_recommendation"]["requires_split_full_suite_with_medium_vectors"] is True
    assert audit["runner_read_only_checks"]["runtime_mapping_enabled_after_runner"] is False
    assert audit["runner_read_only_checks"]["enforcement_enabled_after_runner"] is False


def test_round97_98_exports_and_state_debug_surface(tmp_path) -> None:
    engine, precheck = _prepare_round96_engine()
    smoke = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)
    audit = run_round98_runtime_mapping_persistence_gate_audit(engine, source_smoke=smoke)

    smoke_path = tmp_path / "round97.json"
    smoke_export = write_round97_controlled_runtime_mapping_enable_smoke(smoke, smoke_path)
    assert smoke_export["export_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert smoke_export["runtime_mapping_enabled_after_rollback"] is False
    assert smoke_export["runtime_mapping_persisted"] is False

    audit_path = tmp_path / "round98.json"
    audit_export = write_round98_runtime_mapping_persistence_gate_audit(audit, audit_path)
    assert audit_export["export_version"] == "v3_round98_runtime_mapping_persistence_gate_audit"
    assert audit_export["runtime_mapping_enabled"] is False
    assert audit_export["read_only"] is True

    loaded_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert loaded_smoke["mapped_tokens"] == ["민석"]
    assert loaded_audit["hard_stop"] is False

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 98
    assert lcm["controlled_runtime_mapping_enable_smoke_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert lcm["runtime_mapping_persistence_gate_audit_version"] == "v3_round98_runtime_mapping_persistence_gate_audit"
    assert lcm["controlled_runtime_mapping_enable_smoke_available"] is True
    assert lcm["runtime_mapping_persistence_gate_audit_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
