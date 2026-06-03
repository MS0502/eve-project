"""EVE v3 round77 — lexical → concept mapping planning boundary.

Round77 starts the next axis after self_learning_v1 freeze, but it must remain
planning-only. Lexical vectors are evidence, not concept categories and not AGP
anchors.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
from adapters.runtime_smoke_runner import (
    build_round77_lexical_to_concept_mapping_plan,
    run_round77_lexical_to_concept_mapping_plan,
    write_round77_lexical_to_concept_mapping_plan,
)


def test_round77_mapping_contract_preserves_self_learning_v1_policy() -> None:
    engine = build_full_engine()

    report = run_round77_lexical_to_concept_mapping_plan(engine)

    assert report["planning_version"] == "v3_round77_lexical_to_concept_mapping_planning"
    assert report["round"] == 77
    assert report["source_baseline"] == "self_learning_v1_round76"
    assert report["status"] == "planning_only_not_enforced"

    active = report["active_self_learning_policy"]
    assert active["auto_observe_enabled"] is True
    assert active["auto_promotion_enabled"] is False
    assert active["commit_gate_enabled"] is True
    assert active["min_observations_for_commit"] == 2
    assert active["context_diversity_gate_enabled"] is True

    readiness = report["readiness_checks"]
    assert all(readiness.values())


def test_round77_lexical_vectors_are_evidence_not_agp_anchors() -> None:
    engine = build_full_engine()
    adapter = engine.lex_concept_mapping

    contract = adapter.mapping_contract_snapshot()
    boundaries = contract["hard_boundaries"]

    assert contract["runtime_mapping_enabled"] is False
    assert contract["enforcement_enabled"] is False
    assert boundaries["lexical_vector_is_not_concept"] is True
    assert boundaries["eve_specific_vector_is_not_agp_anchor"] is True
    assert boundaries["seed_vector_is_not_agp_anchor"] is True
    assert boundaries["agp_anchor_requires_explicit_category_and_sa_activation"] is True
    assert boundaries["no_auto_category_creation"] is True

    lexical_sources = {row["name"]: row for row in contract["lexical_sources"]}
    assert lexical_sources["fastText"]["may_create_agp_anchor"] is False
    assert lexical_sources["EveSpecificVectorStore"]["may_create_agp_anchor"] is False
    assert lexical_sources["PMI+SVD"]["may_create_agp_anchor"] is False


def test_round77_candidate_schema_and_token_plan_are_not_mapping_enforcement() -> None:
    adapter = LexConceptMappingAdapter()

    schema = adapter.candidate_record_schema()
    token_plan = adapter.plan_for_tokens(["민석", "민석", "EVE"])

    assert schema["schema_version"] == "v3_round77_lexical_concept_candidate_schema_plan"
    assert schema["status"] == "schema_only_no_candidates_persisted"
    assert "proposed_concept_category" in schema["required_fields"]
    assert "auto_created_category" in schema["forbidden_fields"]
    assert "eve_specific_vector_anchor_pass" in schema["forbidden_fields"]

    assert token_plan["runtime_mapping_enabled"] is False
    assert token_plan["token_count"] == 2
    for row in token_plan["candidate_rows"]:
        assert row["mapping_status"] == "not_mapped_round77_planning_only"
        assert row["may_use_lexical_vector_as_evidence"] is True
        assert row["may_use_lexical_vector_as_agp_anchor"] is False
        assert row["requires_explicit_concept_category"] is True
        assert row["requires_sa_activation_for_agp"] is True
        assert row["operator_decision"] == "undecided"


def test_round77_planning_and_write_are_read_only(tmp_path) -> None:
    engine = build_full_engine()
    audit_before = len(engine.eve_self_learning.commit_audit_records())
    store_before = engine.eve_specific_vector_store.stats().copy()
    telemetry_before = engine.self_embedding.telemetry().copy()

    report = run_round77_lexical_to_concept_mapping_plan(engine)
    checks = report["planning_read_only_checks"]

    assert checks["audit_records_unchanged_during_planning"] is True
    assert checks["vector_store_unchanged_during_planning"] is True
    assert checks["telemetry_unchanged_during_planning"] is True
    assert checks["self_learning_policy_unchanged_during_planning"] is True
    assert checks["runtime_mapping_enabled_after_planning"] is False
    assert checks["enforcement_enabled_after_planning"] is False

    assert len(engine.eve_self_learning.commit_audit_records()) == audit_before
    assert engine.eve_specific_vector_store.stats() == store_before
    assert engine.self_embedding.telemetry() == telemetry_before

    rebuilt = build_round77_lexical_to_concept_mapping_plan(
        report["contract_snapshot"],
        report["candidate_schema"],
        report["token_plan"],
    )
    assert rebuilt["planning_version"] == "v3_round77_lexical_to_concept_mapping_planning"
    assert rebuilt["round77_policy"]["does_not_call_commit"] is True
    assert rebuilt["round77_policy"]["does_not_create_categories"] is True
    assert engine.eve_specific_vector_store.stats() == store_before

    path = tmp_path / "round77_plan.json"
    result = write_round77_lexical_to_concept_mapping_plan(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert result["export_version"] == "v3_round77_lexical_to_concept_mapping_planning"
    assert result["recomputed"] is False
    assert result["observe_called"] is False
    assert result["commit_called"] is False
    assert result["lookup_called"] is False
    assert result["category_created"] is False
    assert result["concept_memory_mutation"] is False
    assert result["agp_verify_called"] is False
    assert loaded["planning_version"] == "v3_round77_lexical_to_concept_mapping_planning"
    assert len(engine.eve_self_learning.commit_audit_records()) == audit_before
    assert engine.eve_specific_vector_store.stats() == store_before
    assert engine.self_embedding.telemetry() == telemetry_before


def test_round77_state_debug_exposes_planning_boundary() -> None:
    engine = build_full_engine()

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]

    assert lcm["module"] == "LexConceptMappingAdapter"
    assert lcm["planning_version"] == "v3_round77_lexical_to_concept_mapping_planning"
    assert lcm["round"] >= 77
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["candidate_dry_run_version"] == "v3_round78_lexical_concept_candidate_schema_dry_run"
    assert lcm["evidence_quality_report_version"] == "v3_round79_lexical_concept_candidate_evidence_quality_report"
    assert "lexical_vectors_are_evidence_not_agp_anchors" in lcm["policy"]
