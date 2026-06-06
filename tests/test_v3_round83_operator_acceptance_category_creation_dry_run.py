"""EVE v3 round83 — operator acceptance fixture + category creation dry-run.

Round83 is still read-only. It models an operator accepting a concept proposal
and plans an explicit category record, but persists no operator decision, creates
no category, writes no concept memory/frame/hypergraph state, creates no SA
activation, and creates no AGP anchor.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round83_operator_acceptance_category_creation_dry_run,
    write_round83_operator_acceptance_category_creation_dry_run,
)


def _prepare_engine_with_eve_specific_observations():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round83_test_a")
    learner.observe_text("민석 군대", source="round83_test_b")
    assert learner.commit_audit_records() == []
    assert engine.eve_specific_vector_store.stats()["stored_count"] == 0
    return engine


def test_round83_operator_acceptance_fixture_plans_category_without_mutation() -> None:
    engine = _prepare_engine_with_eve_specific_observations()

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round83_operator_acceptance_category_creation_dry_run(
        engine,
        planning_tokens=["민석", "EVE"],
        accepted_tokens=["민석"],
    )

    assert report["dry_run_version"] == "v3_round83_operator_acceptance_category_creation_dry_run"
    assert report["round"] == 83
    assert report["source_gate_proposal_report_version"] == "v3_round82_concept_mapping_gate_proposal_report"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["proposal_count"] == 1
    assert report["accepted_fixture_count"] == 1
    assert report["category_creation_dry_run_count"] == 1
    assert report["would_pass_mapping_gate_count"] == 0
    assert report["accepted_tokens"] == ["민석"]
    assert "EVE" in report["blocked_tokens"]

    accepted = report["accepted_rows"][0]
    assert accepted["lexical_token"] == "민석"
    assert accepted["operator_acceptance_fixture"] is True
    assert accepted["operator_decision_status"] == "operator_accepted_fixture_not_persisted"
    assert accepted["resolved_reasons"] == ["operator_acceptance_required"]
    assert "operator_acceptance_required" not in accepted["remaining_blocked_reasons"]
    assert "explicit_category_creation_dry_run_only" in accepted["remaining_blocked_reasons"]
    assert accepted["would_create_explicit_category"] is True
    assert accepted["category_created"] is False
    assert accepted["concept_memory_mutation"] is False
    assert accepted["frame_hypergraph_mutation"] is False
    assert accepted["sa_activation_created"] is False
    assert accepted["agp_anchor_created"] is False
    assert accepted["agp_verify_called"] is False
    assert accepted["would_pass_mapping_gate_after_fixture"] is False

    category_plan = accepted["category_creation_dry_run"]
    assert category_plan["category_id"] == "concept_category::lex::민석"
    assert category_plan["would_create"] is True
    assert category_plan["created"] is False
    assert category_plan["persisted"] is False
    assert category_plan["activated"] is False
    assert category_plan["sa_activation_created"] is False
    assert category_plan["agp_anchor_created"] is False

    dry_row = report["category_creation_dry_run_rows"][0]
    assert dry_row["lexical_token"] == "민석"
    assert dry_row["dry_run_status"] == "would_create_explicit_category_record_later"
    assert dry_row["created"] is False
    assert dry_row["requires_future_concept_memory_or_frame_evidence"] is True
    assert dry_row["requires_future_sa_activation_path"] is True
    assert dry_row["requires_future_agp_bridge_smoke_test"] is True

    reason_counts = report["remaining_blocked_reason_counts"]
    assert reason_counts["explicit_category_creation_dry_run_only"] == 1
    assert reason_counts["sa_activation_path_required"] == 1
    assert reason_counts["concept_memory_or_frame_evidence_required"] == 1
    assert reason_counts["agp_bridge_smoke_test_required"] == 1
    assert reason_counts["blocked_insufficient_lexical_evidence"] == 1
    assert "operator_acceptance_required" not in reason_counts

    policy = report["policy"]
    assert policy["operator_acceptance_fixture_only"] is True
    assert policy["operator_decision_not_persisted"] is True
    assert policy["explicit_category_creation_dry_run_only"] is True
    assert policy["no_runtime_mapping"] is True
    assert policy["no_enforcement"] is True
    assert policy["no_category_creation"] is True
    assert policy["no_concept_memory_mutation"] is True
    assert policy["no_frame_hypergraph_mutation"] is True
    assert policy["no_sa_activation_creation"] is True
    assert policy["no_agp_anchor_creation"] is True
    assert policy["no_agp_verify_call"] is True
    assert policy["no_embedding_wrapper_lookup"] is True
    assert policy["no_vector_commit"] is True

    checks = report["read_only_checks"]
    assert checks["audit_records_unchanged_during_dry_run"] is True
    assert checks["vector_store_unchanged_during_dry_run"] is True
    assert checks["telemetry_unchanged_during_dry_run"] is True
    assert checks["self_learning_policy_unchanged_during_dry_run"] is True
    assert checks["operator_decision_persisted"] is False
    assert checks["category_created"] is False
    assert checks["concept_memory_mutation"] is False
    assert checks["frame_hypergraph_mutation"] is False
    assert checks["sa_activation_created"] is False
    assert checks["agp_anchor_created"] is False
    assert checks["agp_verify_called"] is False
    assert checks["wrapper_lookup_called"] is False
    assert checks["vector_commit_called"] is False

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round83_export_does_not_recompute_or_mutate(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_observations()
    report = run_round83_operator_acceptance_category_creation_dry_run(engine, planning_tokens=["민석"])

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()

    path = tmp_path / "round83.json"
    result = write_round83_operator_acceptance_category_creation_dry_run(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round83_operator_acceptance_category_creation_dry_run"
    assert result["recomputed"] is False
    assert result["observe_called"] is False
    assert result["commit_called"] is False
    assert result["lookup_called"] is False
    assert result["operator_decision_persisted"] is False
    assert result["category_created"] is False
    assert result["concept_memory_mutation"] is False
    assert result["frame_hypergraph_mutation"] is False
    assert result["sa_activation_created"] is False
    assert result["agp_anchor_created"] is False
    assert result["agp_verify_called"] is False
    assert result["policy_changed"] is False
    assert loaded["dry_run_version"] == "v3_round83_operator_acceptance_category_creation_dry_run"
    assert loaded["accepted_fixture_count"] == 1

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store


def test_round83_state_debug_exposes_operator_acceptance_fixture_surface() -> None:
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]

    assert lcm["module"] == "LexConceptMappingAdapter"
    assert lcm["operator_acceptance_category_creation_dry_run_version"] == "v3_round83_operator_acceptance_category_creation_dry_run"
    assert lcm["round"] == 94
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["candidate_schema_dry_run_available"] is True
    assert lcm["candidate_evidence_quality_report_available"] is True
    assert lcm["concept_proposal_report_available"] is True
    assert lcm["concept_mapping_gate_dry_run_available"] is True
    assert lcm["concept_mapping_gate_proposal_report_available"] is True
    assert lcm["operator_acceptance_category_creation_dry_run_available"] is True
    assert "round88_concept_mapping_v0_freeze_read_only" in lcm["policy"]
