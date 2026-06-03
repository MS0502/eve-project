"""EVE v3 rounds 78-79 — lexical→concept candidate dry-run and evidence report.

These rounds are intentionally read-only. They convert lexical evidence into
candidate rows and quality summaries, but they do not create categories, mutate
concept memory, call AGP, or enable runtime mapping.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round78_lexical_concept_candidate_schema_dry_run,
    run_round79_lexical_concept_candidate_evidence_quality_report,
    write_round78_lexical_concept_candidate_schema_dry_run,
    write_round79_lexical_concept_candidate_evidence_quality_report,
)


def _prepare_engine_with_eve_specific_commit():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round78_test_a")
    learner.observe_text("민석 군대", source="round78_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    return engine


def test_round78_candidate_schema_dry_run_builds_rows_without_concept_mapping() -> None:
    engine = _prepare_engine_with_eve_specific_commit()

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round78_lexical_concept_candidate_schema_dry_run(engine, planning_tokens=["민석", "EVE"])

    assert report["dry_run_version"] == "v3_round78_lexical_concept_candidate_schema_dry_run"
    assert report["round"] == 78
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2

    rows = {row["lexical_token"]: row for row in report["candidate_rows"]}
    assert rows["민석"]["lexical_evidence_ready"] is True
    assert rows["민석"]["lexical_evidence"]["eve_specific_vector_present"] is True
    assert rows["민석"]["proposed_concept_category"] is None
    assert rows["민석"]["category_created"] is False
    assert rows["민석"]["agp_anchor_created"] is False
    assert rows["민석"]["agp_anchor_policy"] == "lexical_vector_is_evidence_only_explicit_category_and_sa_required"

    # A lexical token with no evidence is still represented, but blocked.
    assert rows["EVE"]["lexical_evidence_ready"] is False
    assert rows["EVE"]["candidate_status"] == "blocked_insufficient_lexical_evidence"

    checks = report["read_only_checks"]
    assert checks["audit_records_unchanged_during_dry_run"] is True
    assert checks["vector_store_unchanged_during_dry_run"] is True
    assert checks["telemetry_unchanged_during_dry_run"] is True
    assert checks["category_created"] is False
    assert checks["agp_verify_called"] is False

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round79_candidate_evidence_quality_report_remains_read_only() -> None:
    engine = _prepare_engine_with_eve_specific_commit()

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round79_lexical_concept_candidate_evidence_quality_report(engine, planning_tokens=["민석", "EVE"])

    assert report["report_version"] == "v3_round79_lexical_concept_candidate_evidence_quality_report"
    assert report["round"] == 79
    assert report["source_dry_run_version"] == "v3_round78_lexical_concept_candidate_schema_dry_run"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["lexical_evidence_ready_count"] == 1
    assert report["concept_category_created_count"] == 0
    assert report["agp_anchor_created_count"] == 0
    assert report["missing_explicit_concept_category_count"] == 2
    assert report["operator_recommendation"] == "proceed_to_concept_proposal_report"
    assert "민석" in report["ready_tokens"]
    assert "EVE" in report["blocked_tokens"]

    checks = report["read_only_checks"]
    assert checks["audit_records_unchanged_during_report"] is True
    assert checks["vector_store_unchanged_during_report"] is True
    assert checks["telemetry_unchanged_during_report"] is True
    assert checks["category_created"] is False
    assert checks["agp_verify_called"] is False

    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round78_79_exports_do_not_recompute_or_mutate(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    dry_run = run_round78_lexical_concept_candidate_schema_dry_run(engine, planning_tokens=["민석"])
    quality = run_round79_lexical_concept_candidate_evidence_quality_report(engine, planning_tokens=["민석"])

    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()

    path78 = tmp_path / "round78.json"
    path79 = tmp_path / "round79.json"
    result78 = write_round78_lexical_concept_candidate_schema_dry_run(dry_run, path78)
    result79 = write_round79_lexical_concept_candidate_evidence_quality_report(quality, path79)

    loaded78 = json.loads(path78.read_text(encoding="utf-8"))
    loaded79 = json.loads(path79.read_text(encoding="utf-8"))

    assert result78["export_version"] == "v3_round78_lexical_concept_candidate_schema_dry_run"
    assert result79["export_version"] == "v3_round79_lexical_concept_candidate_evidence_quality_report"
    for result in (result78, result79):
        assert result["recomputed"] is False
        assert result["observe_called"] is False
        assert result["commit_called"] is False
        assert result["lookup_called"] is False
        assert result["category_created"] is False
        assert result["concept_memory_mutation"] is False
        assert result["frame_hypergraph_mutation"] is False
        assert result["agp_verify_called"] is False
        assert result["policy_changed"] is False

    assert loaded78["dry_run_version"] == "v3_round78_lexical_concept_candidate_schema_dry_run"
    assert loaded79["report_version"] == "v3_round79_lexical_concept_candidate_evidence_quality_report"
    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store


def test_round78_79_state_debug_exposes_read_only_surfaces() -> None:
    engine = build_full_engine()
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]

    assert lcm["module"] == "LexConceptMappingAdapter"
    assert lcm["planning_version"] == "v3_round77_lexical_to_concept_mapping_planning"
    assert lcm["candidate_dry_run_version"] == "v3_round78_lexical_concept_candidate_schema_dry_run"
    assert lcm["evidence_quality_report_version"] == "v3_round79_lexical_concept_candidate_evidence_quality_report"
    assert lcm["round"] == 94
    assert lcm["concept_proposal_report_version"] == "v3_round80_concept_proposal_report"
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["candidate_schema_dry_run_available"] is True
    assert lcm["candidate_evidence_quality_report_available"] is True
    assert lcm["concept_proposal_report_available"] is True
    assert "evidence_not_agp_anchors" in lcm["policy"]
