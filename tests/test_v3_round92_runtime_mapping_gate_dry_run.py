"""EVE v3 round92 — runtime lexical→concept mapping gate dry-run.

Round92 is read-only. It evaluates which committed concept categories would be
eligible for runtime lexical→concept mapping if a future explicit enforcement
patch enabled the mapper. Runtime mapping remains disabled.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    run_round90_concept_commit_delta_replay_report,
    run_round91_concept_commit_replay_export_checkpoint,
    run_round92_runtime_mapping_gate_dry_run,
    write_round92_runtime_mapping_gate_dry_run,
)


def _prepare_committed_engine():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round92_test_a")
    learner.observe_text("민석 군대", source="round92_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    source_commit = run_round89_explicit_concept_commit_smoke(engine)
    source_replay = run_round90_concept_commit_delta_replay_report(engine, source_commit_report=source_commit)
    checkpoint = run_round91_concept_commit_replay_export_checkpoint(
        engine,
        source_commit_report=source_commit,
        source_replay_report=source_replay,
    )
    return engine, checkpoint


def test_round92_runtime_mapping_gate_dry_run_maps_only_committed_category_without_enabling_runtime_mapping() -> None:
    engine, checkpoint = _prepare_committed_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round92_runtime_mapping_gate_dry_run(
        engine,
        tokens=["민석", "EVE"],
        source_checkpoint=checkpoint,
    )

    assert report["dry_run_version"] == "v3_round92_runtime_mapping_gate_dry_run"
    assert report["round"] == 92
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["category_count"] == 1
    assert report["candidate_count"] == 2
    assert report["would_map_count"] == 1
    assert report["would_block_count"] == 1
    assert report["would_map_tokens"] == ["민석"]
    assert report["blocked_tokens"] == ["EVE"]
    assert report["operator_summary"]["next_recommended_round"] == "round93_runtime_mapping_proposal_report"
    assert report["operator_summary"]["runtime_mapping_should_be_enabled_now"] is False

    rows = {row["lexical_token"]: row for row in report["rows"]}
    assert rows["민석"]["would_map_if_runtime_mapping_enabled"] is True
    assert rows["민석"]["mapping_target_category_id"] == "concept_category::lex::민석"
    assert rows["민석"]["category_record_present"] is True
    assert rows["민석"]["concept_memory_present"] is True
    assert rows["민석"]["sa_activation_present"] is True
    assert rows["민석"]["agp_replay_passed"] is True
    assert rows["민석"]["agp_negative_without_sa_guard_present"] is True
    assert rows["민석"]["runtime_mapping_applied_now"] is False
    assert rows["민석"]["uses_eve_specific_vector_as_anchor"] is False

    assert rows["EVE"]["would_map_if_runtime_mapping_enabled"] is False
    assert "explicit_category_missing" in rows["EVE"]["blocked_reasons"]
    assert "concept_memory_missing" in rows["EVE"]["blocked_reasons"]

    checks = report["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_dry_run"] is True
    assert checks["concept_commit_audit_unchanged_during_dry_run"] is True
    assert checks["eve_specific_vector_store_unchanged_during_dry_run"] is True
    assert checks["wrapper_telemetry_unchanged_during_dry_run"] is True
    assert checks["sa_active_categories_unchanged_during_dry_run"] is True
    assert checks["agp_verify_called_during_runtime_mapping_dry_run"] is False
    assert checks["embedding_lookup_called_during_dry_run"] is False
    assert report["runner_read_only_checks"]["concept_commit_not_called_by_round92_runner"] is True

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round92_state_debug_and_export_surface(tmp_path) -> None:
    engine, checkpoint = _prepare_committed_engine()
    report = run_round92_runtime_mapping_gate_dry_run(
        engine,
        tokens=["민석", "EVE"],
        source_checkpoint=checkpoint,
    )

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 94
    assert lcm["runtime_mapping_gate_dry_run_version"] == "v3_round92_runtime_mapping_gate_dry_run"
    assert lcm["runtime_mapping_gate_dry_run_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1

    path = tmp_path / "round92.json"
    export = write_round92_runtime_mapping_gate_dry_run(report, path)
    assert export["export_version"] == "v3_round92_runtime_mapping_gate_dry_run"
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["eve_specific_vector_commit_called"] is False
    assert export["read_only"] is True
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["dry_run_version"] == report["dry_run_version"]
