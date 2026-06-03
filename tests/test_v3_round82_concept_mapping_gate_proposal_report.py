"""EVE v3 round82 — concept mapping gate proposal report.

Round82 is read-only: it consolidates Round81 gate dry-run blocks into operator
review/action items, but creates no categories, no concept memory entries, no SA
activation, and no AGP anchors.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round82_concept_mapping_gate_proposal_report,
    write_round82_concept_mapping_gate_proposal_report,
)


def _prepare_engine_with_eve_specific_commit():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round82_test_a")
    learner.observe_text("민석 군대", source="round82_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    return engine


def test_round82_gate_proposal_report_is_operator_action_only() -> None:
    engine = _prepare_engine_with_eve_specific_commit()

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round82_concept_mapping_gate_proposal_report(engine, planning_tokens=["민석", "EVE"])

    assert report["report_version"] == "v3_round82_concept_mapping_gate_proposal_report"
    assert report["round"] == 82
    assert report["source_gate_dry_run_version"] == "v3_round81_concept_mapping_gate_dry_run"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["proposal_count"] == 1
    assert report["blocked_candidate_count"] == 1
    assert report["would_pass_count"] == 0
    assert report["would_block_count"] == 2
    assert report["operator_action_item_count"] >= 5
    assert report["operator_recommendation"] == "do_not_enable_runtime_mapping_until_all_action_items_are_resolved"

    reason_counts = report["blocked_reason_counts"]
    assert reason_counts["operator_acceptance_required"] == 1
    assert reason_counts["explicit_category_creation_required"] == 1
    assert reason_counts["sa_activation_path_required"] == 1
    assert reason_counts["concept_memory_or_frame_evidence_required"] == 1
    assert reason_counts["agp_bridge_smoke_test_required"] == 1
    assert reason_counts["blocked_insufficient_lexical_evidence"] == 1

    actions = {row["blocked_reason"]: row for row in report["operator_action_items"]}
    assert actions["operator_acceptance_required"]["operator_action"] == "operator_must_explicitly_accept_the_concept_proposal"
    assert actions["explicit_category_creation_required"]["operator_action"] == "future_patch_must_create_an_explicit_category_record"
    assert actions["sa_activation_path_required"]["operator_action"] == "future_patch_must_prove_sa_activation_path_for_the_category"
    assert actions["agp_bridge_smoke_test_required"]["operator_action"] == "future_patch_must_add_agp_bridge_smoke_test"

    proposal = report["gate_proposal_rows"][0]
    assert proposal["lexical_token"] == "민석"
    assert proposal["would_pass_mapping_gate"] is False
    assert proposal["operator_decision_status"] == "operator_review_required"
    assert proposal["may_enable_mapping_now"] is False
    assert proposal["may_create_category_now"] is False
    assert proposal["may_create_agp_anchor_now"] is False
    assert proposal["category_created"] is False
    assert proposal["concept_memory_mutation"] is False
    assert proposal["frame_hypergraph_mutation"] is False
    assert proposal["sa_activation_created"] is False
    assert proposal["agp_anchor_created"] is False
    assert proposal["agp_verify_called"] is False

    blocked = report["blocked_rows"][0]
    assert blocked["lexical_token"] == "EVE"
    assert blocked["operator_decision_status"] == "not_ready_for_operator_acceptance"

    policy = report["policy"]
    assert policy["operator_proposal_report_only"] is True
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
    assert checks["audit_records_unchanged_during_report"] is True
    assert checks["vector_store_unchanged_during_report"] is True
    assert checks["telemetry_unchanged_during_report"] is True
    assert checks["self_learning_policy_unchanged_during_report"] is True
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


def test_round82_export_does_not_recompute_or_mutate(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    report = run_round82_concept_mapping_gate_proposal_report(engine, planning_tokens=["민석"])

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()

    path = tmp_path / "round82.json"
    result = write_round82_concept_mapping_gate_proposal_report(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round82_concept_mapping_gate_proposal_report"
    assert result["recomputed"] is False
    assert result["observe_called"] is False
    assert result["commit_called"] is False
    assert result["lookup_called"] is False
    assert result["category_created"] is False
    assert result["concept_memory_mutation"] is False
    assert result["frame_hypergraph_mutation"] is False
    assert result["sa_activation_created"] is False
    assert result["agp_anchor_created"] is False
    assert result["agp_verify_called"] is False
    assert result["policy_changed"] is False
    assert loaded["report_version"] == "v3_round82_concept_mapping_gate_proposal_report"
    assert loaded["would_pass_count"] == 0

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store


def test_round82_state_debug_exposes_gate_proposal_surface() -> None:
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]

    assert lcm["module"] == "LexConceptMappingAdapter"
    assert lcm["concept_mapping_gate_proposal_report_version"] == "v3_round82_concept_mapping_gate_proposal_report"
    assert lcm["round"] == 94
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["candidate_schema_dry_run_available"] is True
    assert lcm["candidate_evidence_quality_report_available"] is True
    assert lcm["concept_proposal_report_available"] is True
    assert lcm["concept_mapping_gate_dry_run_available"] is True
    assert lcm["concept_mapping_gate_proposal_report_available"] is True
    assert "round88_concept_mapping_v0_freeze_read_only" in lcm["policy"]
