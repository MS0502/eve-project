"""EVE v3 round81 — concept mapping gate dry-run.

Round81 is read-only: it evaluates future mapping-gate readiness over Round80
operator proposals but creates no categories, no concept memory entries, no SA
activation, and no AGP anchors.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round81_concept_mapping_gate_dry_run,
    write_round81_concept_mapping_gate_dry_run,
)


def _prepare_engine_with_eve_specific_observations():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round81_test_a")
    learner.observe_text("민석 군대", source="round81_test_b")
    assert learner.commit_audit_records() == []
    assert engine.eve_specific_vector_store.stats()["stored_count"] == 0
    return engine


def test_round81_concept_mapping_gate_dry_run_blocks_until_operator_and_concept_evidence() -> None:
    engine = _prepare_engine_with_eve_specific_observations()

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round81_concept_mapping_gate_dry_run(engine, planning_tokens=["민석", "EVE"])

    assert report["dry_run_version"] == "v3_round81_concept_mapping_gate_dry_run"
    assert report["round"] == 81
    assert report["source_concept_proposal_report_version"] == "v3_round80_concept_proposal_report"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["proposal_count"] == 1
    assert report["blocked_candidate_count"] == 1
    assert report["would_pass_count"] == 0
    assert report["would_block_count"] == 2
    assert report["would_pass_tokens"] == []
    assert report["would_block_tokens"] == ["EVE", "민석"]
    assert report["operator_recommendation"] == "do_not_enable_mapping_gate_until_operator_acceptance_and_concept_evidence"

    reason_counts = report["blocked_reason_counts"]
    assert reason_counts["operator_acceptance_required"] == 1
    assert reason_counts["explicit_category_creation_required"] == 1
    assert reason_counts["sa_activation_path_required"] == 1
    assert reason_counts["concept_memory_or_frame_evidence_required"] == 1
    assert reason_counts["agp_bridge_smoke_test_required"] == 1
    assert reason_counts["blocked_insufficient_lexical_evidence"] == 1

    minseok_row = next(row for row in report["gate_rows"] if row["lexical_token"] == "민석")
    assert minseok_row["dry_run_status"] == "would_block_future_mapping_gate"
    assert minseok_row["would_pass_mapping_gate"] is False
    assert minseok_row["lexical_evidence_ready"] is True
    assert minseok_row["category_created"] is False
    assert minseok_row["concept_memory_mutation"] is False
    assert minseok_row["frame_hypergraph_mutation"] is False
    assert minseok_row["sa_activation_created"] is False
    assert minseok_row["agp_anchor_created"] is False
    assert minseok_row["agp_verify_called"] is False

    blocked_row = next(row for row in report["gate_rows"] if row["lexical_token"] == "EVE")
    assert blocked_row["dry_run_status"] == "blocked_before_mapping_gate"
    assert blocked_row["would_pass_mapping_gate"] is False
    assert blocked_row["lexical_evidence_ready"] is False

    policy = report["policy"]
    assert policy["dry_run_only"] is True
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


def test_round81_export_does_not_recompute_or_mutate(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_observations()
    report = run_round81_concept_mapping_gate_dry_run(engine, planning_tokens=["민석"])

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()

    path = tmp_path / "round81.json"
    result = write_round81_concept_mapping_gate_dry_run(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round81_concept_mapping_gate_dry_run"
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
    assert loaded["dry_run_version"] == "v3_round81_concept_mapping_gate_dry_run"
    assert loaded["would_pass_count"] == 0

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store


def test_round81_state_debug_exposes_gate_dry_run_surface() -> None:
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]

    assert lcm["module"] == "LexConceptMappingAdapter"
    assert lcm["concept_mapping_gate_dry_run_version"] == "v3_round81_concept_mapping_gate_dry_run"
    assert lcm["round"] == 94
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["candidate_schema_dry_run_available"] is True
    assert lcm["candidate_evidence_quality_report_available"] is True
    assert lcm["concept_proposal_report_available"] is True
    assert lcm["concept_mapping_gate_dry_run_available"] is True
    assert "round88_concept_mapping_v0_freeze_read_only" in lcm["policy"]
