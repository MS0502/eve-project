"""EVE v3 round80 — lexical→concept operator proposal report.

Round80 is read-only: it turns lexical-evidence-ready rows into operator-review
concept proposals, but creates no categories and no AGP anchors.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round80_concept_proposal_report,
    write_round80_concept_proposal_report,
)


def _prepare_engine_with_eve_specific_observations():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round80_test_a")
    learner.observe_text("민석 군대", source="round80_test_b")
    assert learner.commit_audit_records() == []
    assert engine.eve_specific_vector_store.stats()["stored_count"] == 0
    return engine


def test_round80_concept_proposal_report_creates_operator_proposal_only() -> None:
    engine = _prepare_engine_with_eve_specific_observations()

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round80_concept_proposal_report(engine, planning_tokens=["민석", "EVE"])

    assert report["report_version"] == "v3_round80_concept_proposal_report"
    assert report["round"] == 80
    assert report["source_evidence_quality_report_version"] == "v3_round79_lexical_concept_candidate_evidence_quality_report"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["proposal_count"] == 1
    assert report["blocked_candidate_count"] == 1
    assert report["concept_category_created_count"] == 0
    assert report["agp_anchor_created_count"] == 0
    assert report["ready_tokens"] == ["민석"]
    assert "EVE" in report["blocked_tokens"]
    assert report["operator_recommendation"] == "review_concept_proposals_before_mapping_gate_dry_run"

    proposal = report["proposal_rows"][0]
    assert proposal["lexical_token"] == "민석"
    assert proposal["proposal_status"] == "operator_review_required"
    assert proposal["proposed_concept_category"]["created"] is False
    assert proposal["proposed_concept_category"]["activated"] is False
    assert proposal["may_create_category_now"] is False
    assert proposal["may_create_agp_anchor_now"] is False
    assert proposal["category_created"] is False
    assert proposal["agp_anchor_created"] is False
    assert "operator_acceptance" in proposal["required_before_enforcement"]
    assert "agp_bridge_smoke_test" in proposal["required_before_enforcement"]

    policy = report["policy"]
    assert policy["operator_proposal_only"] is True
    assert policy["no_category_creation"] is True
    assert policy["no_concept_memory_mutation"] is True
    assert policy["no_frame_hypergraph_mutation"] is True
    assert policy["no_sa_activation_creation"] is True
    assert policy["no_agp_verify_call"] is True
    assert policy["no_embedding_wrapper_lookup"] is True
    assert policy["no_vector_commit"] is True
    assert policy["lexical_vector_is_evidence_only"] is True
    assert policy["eve_specific_vector_is_not_agp_anchor"] is True

    checks = report["read_only_checks"]
    assert checks["audit_records_unchanged_during_report"] is True
    assert checks["vector_store_unchanged_during_report"] is True
    assert checks["telemetry_unchanged_during_report"] is True
    assert checks["self_learning_policy_unchanged_during_report"] is True
    assert checks["category_created"] is False
    assert checks["concept_memory_mutation"] is False
    assert checks["frame_hypergraph_mutation"] is False
    assert checks["sa_activation_created"] is False
    assert checks["agp_verify_called"] is False
    assert checks["wrapper_lookup_called"] is False
    assert checks["vector_commit_called"] is False

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round80_export_does_not_recompute_or_mutate(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_observations()
    report = run_round80_concept_proposal_report(engine, planning_tokens=["민석"])

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()

    path = tmp_path / "round80.json"
    result = write_round80_concept_proposal_report(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round80_concept_proposal_report"
    assert result["recomputed"] is False
    assert result["observe_called"] is False
    assert result["commit_called"] is False
    assert result["lookup_called"] is False
    assert result["category_created"] is False
    assert result["concept_memory_mutation"] is False
    assert result["frame_hypergraph_mutation"] is False
    assert result["sa_activation_created"] is False
    assert result["agp_verify_called"] is False
    assert result["policy_changed"] is False
    assert loaded["report_version"] == "v3_round80_concept_proposal_report"
    assert loaded["proposal_count"] == 1

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store


def test_round80_state_debug_exposes_proposal_surface() -> None:
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]

    assert lcm["module"] == "LexConceptMappingAdapter"
    assert lcm["concept_proposal_report_version"] == "v3_round80_concept_proposal_report"
    assert lcm["round"] == 94
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["candidate_schema_dry_run_available"] is True
    assert lcm["candidate_evidence_quality_report_available"] is True
    assert lcm["concept_proposal_report_available"] is True
    assert "round88_concept_mapping_v0_freeze_read_only" in lcm["policy"]
