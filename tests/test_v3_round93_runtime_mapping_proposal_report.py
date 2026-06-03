"""EVE v3 round93 — runtime lexical→concept mapping proposal report.

Round93 is read-only. It turns the Round92 runtime mapping gate dry-run into an
operator proposal surface. Runtime mapping and enforcement remain disabled.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    run_round90_concept_commit_delta_replay_report,
    run_round91_concept_commit_replay_export_checkpoint,
    run_round92_runtime_mapping_gate_dry_run,
    run_round93_runtime_mapping_proposal_report,
    write_round93_runtime_mapping_proposal_report,
)


def _prepare_round92_engine():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round93_test_a")
    learner.observe_text("민석 군대", source="round93_test_b")
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
    return engine, dry_run


def test_round93_runtime_mapping_proposal_report_keeps_runtime_mapping_disabled() -> None:
    engine, dry_run = _prepare_round92_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round93_runtime_mapping_proposal_report(
        engine,
        source_dry_run=dry_run,
    )

    assert report["proposal_version"] == "v3_round93_runtime_mapping_proposal_report"
    assert report["round"] == 93
    assert report["source_dry_run_version"] == "v3_round92_runtime_mapping_gate_dry_run"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["proposal_count"] == 1
    assert report["blocked_count"] == 1
    assert report["would_map_tokens"] == ["민석"]
    assert report["blocked_tokens"] == ["EVE"]
    assert report["operator_action_item_count"] == 1
    assert report["operator_summary"]["runtime_mapping_can_be_considered"] is True
    assert report["operator_summary"]["runtime_mapping_should_be_enabled_now"] is False
    assert report["operator_summary"]["next_recommended_round"] == "round94_runtime_mapping_enforcement_dry_run"
    assert report["operator_summary"]["safe_default"] == "keep_runtime_mapping_disabled"

    proposal = report["proposal_rows"][0]
    assert proposal["lexical_token"] == "민석"
    assert proposal["target_category_id"] == "concept_category::lex::민석"
    assert proposal["proposal_status"] == "operator_review_required"
    assert proposal["may_enable_now"] is False
    assert proposal["requires_separate_enforcement_round"] is True
    assert proposal["requires_split_full_suite_before_enforcement"] is True
    assert proposal["uses_eve_specific_vector_as_anchor"] is False
    assert proposal["anchor_source_required"] == "explicit_category_plus_sa_activation_only"

    blocked = report["blocked_rows"][0]
    assert blocked["lexical_token"] == "EVE"
    assert blocked["proposal_status"] == "blocked_from_runtime_mapping_proposal"
    assert "explicit_category_missing" in blocked["blocked_reasons"]

    checks = report["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_proposal"] is True
    assert checks["concept_commit_audit_unchanged_during_proposal"] is True
    assert checks["eve_specific_vector_store_unchanged_during_proposal"] is True
    assert checks["wrapper_telemetry_unchanged_during_proposal"] is True
    assert checks["sa_active_categories_unchanged_during_proposal"] is True
    assert checks["category_created_during_proposal"] is False
    assert checks["concept_memory_mutation_during_proposal"] is False
    assert checks["sa_activation_created_during_proposal"] is False
    assert checks["agp_verify_called_during_runtime_mapping_proposal"] is False
    assert checks["embedding_lookup_called_during_proposal"] is False
    assert report["runner_read_only_checks"]["concept_commit_not_called_by_round93_runner"] is True
    assert report["runner_read_only_checks"]["runtime_mapping_not_enabled_by_runner"] is True

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round93_state_debug_and_export_surface(tmp_path) -> None:
    engine, dry_run = _prepare_round92_engine()
    report = run_round93_runtime_mapping_proposal_report(engine, source_dry_run=dry_run)

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 94
    assert lcm["runtime_mapping_proposal_report_version"] == "v3_round93_runtime_mapping_proposal_report"
    assert lcm["runtime_mapping_proposal_report_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1

    path = tmp_path / "round93.json"
    export = write_round93_runtime_mapping_proposal_report(report, path)
    assert export["export_version"] == "v3_round93_runtime_mapping_proposal_report"
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["eve_specific_vector_commit_called"] is False
    assert export["read_only"] is True
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["proposal_version"] == report["proposal_version"]
