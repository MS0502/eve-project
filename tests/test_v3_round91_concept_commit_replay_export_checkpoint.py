"""EVE v3 round91 — concept commit replay export / v0 checkpoint summary.

Round91 is read-only. It consolidates Round77~90 lexical→concept artifacts and
confirms that runtime mapping remains disabled after the first explicit concept
commit smoke.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    run_round90_concept_commit_delta_replay_report,
    run_round91_concept_commit_replay_export_checkpoint,
    write_round91_concept_commit_replay_export_checkpoint,
)


def _prepare_committed_engine():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round91_test_a")
    learner.observe_text("민석 군대", source="round91_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    source_commit = run_round89_explicit_concept_commit_smoke(engine)
    source_replay = run_round90_concept_commit_delta_replay_report(engine, source_commit_report=source_commit)
    return engine, source_commit, source_replay


def test_round91_checkpoint_consolidates_round77_to_90_without_mutation() -> None:
    engine, source_commit, source_replay = _prepare_committed_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    checkpoint = run_round91_concept_commit_replay_export_checkpoint(
        engine,
        source_commit_report=source_commit,
        source_replay_report=source_replay,
    )

    assert checkpoint["checkpoint_version"] == "v3_round91_concept_commit_replay_export_v0_checkpoint"
    assert checkpoint["round"] == 91
    assert checkpoint["source_rounds"] == [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    assert checkpoint["source_replay_version"] == "v3_round90_concept_commit_delta_replay_report"
    assert checkpoint["source_commit_version"] == "v3_round89_explicit_concept_commit_smoke"
    assert checkpoint["operator_summary"]["category_count"] == 1
    assert checkpoint["operator_summary"]["category_tokens"] == ["민석"]
    assert checkpoint["operator_summary"]["concept_commit_audit_count"] == 1
    assert checkpoint["operator_summary"]["agp_replay_pass_count"] == 1
    assert checkpoint["operator_summary"]["agp_negative_without_sa_guard_count"] == 1
    assert checkpoint["operator_summary"]["next_recommended_round"] == "round92_runtime_mapping_gate_dry_run"
    assert checkpoint["artifact_manifest"]["artifact_count"] >= 14
    assert checkpoint["replay_guards"]["explicit_category_without_sa_fails_agp"] is True
    assert checkpoint["replay_guards"]["lexical_vector_used_as_anchor"] is False
    assert checkpoint["replay_guards"]["eve_specific_vector_used_as_anchor"] is False
    assert checkpoint["replay_guards"]["seed_vector_used_as_anchor"] is False
    assert checkpoint["runtime_policy"]["runtime_mapping_enabled"] is False
    assert checkpoint["runtime_policy"]["enforcement_enabled"] is False
    assert checkpoint["runtime_policy"]["next_round_is_dry_run_only"] is True

    checks = checkpoint["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_checkpoint"] is True
    assert checks["concept_commit_audit_unchanged_during_checkpoint"] is True
    assert checks["eve_specific_vector_store_unchanged_during_checkpoint"] is True
    assert checks["wrapper_telemetry_unchanged_during_checkpoint"] is True
    assert checks["sa_active_categories_unchanged_during_checkpoint"] is True
    assert checks["category_created_during_checkpoint"] is False
    assert checks["concept_memory_mutation_during_checkpoint"] is False
    assert checks["sa_activation_created_during_checkpoint"] is False
    assert checkpoint["runner_read_only_checks"]["concept_commit_not_called_by_round91_runner"] is True

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round91_state_debug_and_export_surface(tmp_path) -> None:
    engine, source_commit, source_replay = _prepare_committed_engine()
    checkpoint = run_round91_concept_commit_replay_export_checkpoint(
        engine,
        source_commit_report=source_commit,
        source_replay_report=source_replay,
    )
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 94
    assert lcm["concept_commit_replay_export_checkpoint_version"] == "v3_round91_concept_commit_replay_export_v0_checkpoint"
    assert lcm["concept_commit_replay_export_checkpoint_available"] is True
    assert lcm["concept_category_count"] == 1
    assert lcm["concept_commit_audit_count"] == 1
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False

    path = tmp_path / "round91.json"
    export = write_round91_concept_commit_replay_export_checkpoint(checkpoint, path)
    assert export["export_version"] == "v3_round91_concept_commit_replay_export_v0_checkpoint"
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["eve_specific_vector_commit_called"] is False
    assert export["read_only"] is True
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["checkpoint_version"] == checkpoint["checkpoint_version"]
