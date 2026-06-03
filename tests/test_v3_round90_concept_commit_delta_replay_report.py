"""EVE v3 round90 — concept commit delta/replay report.

Round90 is read-only. It summarizes the Round89 explicit concept commit, replays
AGP bridge checks, and verifies that replay does not create another category,
write concept memory, change SA activation, mutate EveSpecific vectors, or
turn on runtime lexical→concept mapping.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.agp_adapter import AGP_REASON_ANCHORED, AGP_REASON_UNKNOWN_CATEGORY
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    run_round90_concept_commit_delta_replay_report,
    write_round90_concept_commit_delta_replay_report,
)


def _prepare_committed_engine():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round90_test_a")
    learner.observe_text("민석 군대", source="round90_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    source_report = run_round89_explicit_concept_commit_smoke(engine)
    assert source_report["created_tokens"] == ["민석"]
    return engine, source_report


def test_round90_replays_concept_commit_delta_without_new_mutation() -> None:
    engine, source_report = _prepare_committed_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round90_concept_commit_delta_replay_report(engine, source_commit_report=source_report)

    assert report["report_version"] == "v3_round90_concept_commit_delta_replay_report"
    assert report["round"] == 90
    assert report["source_commit_version"] == "v3_round89_explicit_concept_commit_smoke"
    assert report["source_created_tokens"] == ["민석"]
    assert report["category_count"] == 1
    assert report["category_tokens"] == ["민석"]
    assert report["concept_memory_replay_found_count"] == 1
    assert report["sa_activation_replay_found_count"] == 1
    assert report["agp_replay_pass_count"] == 1
    assert report["agp_negative_without_sa_guard_count"] == 1
    assert report["delta_consistency"]["source_created_count_matches_category_count"] is True
    assert report["delta_consistency"]["source_concept_memory_delta_matches_replay"] is True
    assert report["delta_consistency"]["source_agp_pass_count_matches_replay"] is True
    row = report["rows"][0]
    assert row["lexical_token"] == "민석"
    assert row["category_id"] == "concept_category::lex::민석"
    assert row["concept_memory_present"] is True
    assert row["sa_activation_present"] is True
    assert row["agp_replay_called"] is True
    assert row["agp_replay_passed"] is True
    assert row["agp_replay_reason"] == AGP_REASON_ANCHORED
    assert row["agp_negative_without_sa_passed"] is False
    assert row["agp_negative_without_sa_reason"] == AGP_REASON_UNKNOWN_CATEGORY
    assert row["uses_lexical_vector_as_anchor"] is False
    assert row["uses_eve_specific_vector_as_anchor"] is False
    assert row["uses_seed_vector_as_anchor"] is False

    checks = report["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_replay"] is True
    assert checks["concept_commit_audit_unchanged_during_replay"] is True
    assert checks["eve_specific_vector_store_unchanged_during_replay"] is True
    assert checks["wrapper_telemetry_unchanged_during_replay"] is True
    assert checks["sa_active_categories_unchanged_during_replay"] is True
    assert checks["category_created_during_replay"] is False
    assert checks["concept_memory_mutation_during_replay"] is False
    assert checks["sa_activation_created_during_replay"] is False

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry
    assert report["runner_read_only_checks"]["concept_commit_not_called_by_round90_runner"] is True


def test_round90_state_debug_and_export_surface(tmp_path) -> None:
    engine, source_report = _prepare_committed_engine()
    report = run_round90_concept_commit_delta_replay_report(engine, source_commit_report=source_report)
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 94
    assert lcm["concept_commit_delta_replay_report_version"] == "v3_round90_concept_commit_delta_replay_report"
    assert lcm["concept_commit_delta_replay_report_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1
    assert lcm["concept_commit_audit_count"] == 1

    path = tmp_path / "round90.json"
    export = write_round90_concept_commit_delta_replay_report(report, path)
    assert export["export_version"] == "v3_round90_concept_commit_delta_replay_report"
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["eve_specific_vector_commit_called"] is False
    assert export["read_only"] is True
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["report_version"] == report["report_version"]
