"""EVE v3 round95 — runtime mapping operator acceptance fixture.

Round95 remains read-only. It turns the Round94 enforcement dry-run into an
explicit operator acceptance fixture for future controlled mutation, while
keeping runtime mapping and enforcement disabled.
"""

from __future__ import annotations

import json
from typing import Any

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    run_round90_concept_commit_delta_replay_report,
    run_round91_concept_commit_replay_export_checkpoint,
    run_round92_runtime_mapping_gate_dry_run,
    run_round93_runtime_mapping_proposal_report,
    run_round94_runtime_mapping_enforcement_dry_run,
    run_round95_runtime_mapping_operator_acceptance_fixture,
    write_round95_runtime_mapping_operator_acceptance_fixture,
)


def _prepare_round94_engine() -> tuple[object, dict[str, Any]]:
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round95_test_a")
    learner.observe_text("민석 군대", source="round95_test_b")
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
    proposal = run_round93_runtime_mapping_proposal_report(engine, source_dry_run=dry_run)
    enforcement = run_round94_runtime_mapping_enforcement_dry_run(
        engine,
        tokens=["민석", "EVE"],
        source_proposal=proposal,
    )
    return engine, enforcement


def test_round95_operator_acceptance_fixture_plans_mapping_without_mutation() -> None:
    engine, enforcement = _prepare_round94_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round95_runtime_mapping_operator_acceptance_fixture(
        engine,
        source_enforcement=enforcement,
        accepted_tokens=["민석"],
    )

    assert report["fixture_version"] == "v3_round95_runtime_mapping_operator_acceptance_fixture"
    assert report["round"] == 95
    assert report["source_enforcement_version"] == "v3_round94_runtime_mapping_enforcement_dry_run"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["accepted_count"] == 1
    assert report["blocked_count"] == 1
    assert report["accepted_tokens"] == ["민석"]
    assert report["blocked_tokens"] == ["EVE"]

    accepted = {row["lexical_token"]: row for row in report["accepted_rows"]}["민석"]
    assert accepted["operator_acceptance_fixture"] is True
    assert accepted["operator_decision_status"] == "operator_accepted_fixture_not_persisted"
    assert accepted["resolved_reasons"] == ["operator_acceptance_required"]
    assert accepted["remaining_blocked_reasons"] == []
    assert accepted["would_apply_runtime_mapping_after_fixture"] is True
    assert accepted["runtime_mapping_applied_now"] is False
    assert accepted["simulated_mapping_result"]["mapping_status"] == "simulated_runtime_mapping_success"
    assert accepted["simulated_mapping_result"]["anchor_source"] == "explicit_category_plus_sa_activation_only"
    assert accepted["uses_lexical_vector_as_anchor"] is False
    assert accepted["uses_eve_specific_vector_as_anchor"] is False
    assert accepted["uses_seed_vector_as_anchor"] is False

    blocked = {row["lexical_token"]: row for row in report["blocked_rows"]}["EVE"]
    assert blocked["operator_acceptance_fixture"] is False
    assert blocked["would_apply_runtime_mapping_after_fixture"] is False
    assert "explicit_category_missing" in blocked["blocked_reasons"]
    assert "concept_memory_missing" in blocked["blocked_reasons"]

    checks = report["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_operator_fixture"] is True
    assert checks["concept_commit_audit_unchanged_during_operator_fixture"] is True
    assert checks["eve_specific_vector_store_unchanged_during_operator_fixture"] is True
    assert checks["wrapper_telemetry_unchanged_during_operator_fixture"] is True
    assert checks["sa_active_categories_unchanged_during_operator_fixture"] is True
    assert checks["category_created_during_operator_fixture"] is False
    assert checks["concept_memory_mutation_during_operator_fixture"] is False
    assert checks["frame_hypergraph_mutation_during_operator_fixture"] is False
    assert checks["sa_activation_created_during_operator_fixture"] is False
    assert checks["agp_verify_called_during_operator_fixture"] is False
    assert checks["embedding_lookup_called_during_operator_fixture"] is False
    assert checks["runtime_mapping_applied"] is False
    assert report["runner_read_only_checks"]["concept_commit_not_called_by_round95_runner"] is True
    assert report["runner_read_only_checks"]["runtime_mapping_not_enabled_by_runner"] is True
    assert report["runner_read_only_checks"]["enforcement_not_enabled_by_runner"] is True

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round95_export_does_not_recompute_or_mutate(tmp_path) -> None:
    engine, enforcement = _prepare_round94_engine()
    report = run_round95_runtime_mapping_operator_acceptance_fixture(
        engine,
        source_enforcement=enforcement,
        accepted_tokens=["민석"],
    )

    path = tmp_path / "round95.json"
    export = write_round95_runtime_mapping_operator_acceptance_fixture(report, path)
    assert export["export_version"] == "v3_round95_runtime_mapping_operator_acceptance_fixture"
    assert export["recomputed"] is False
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["frame_hypergraph_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["eve_specific_vector_commit_called"] is False
    assert export["agp_bypass"] is False
    assert export["read_only"] is True

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["fixture_version"] == report["fixture_version"]
    assert loaded["accepted_tokens"] == ["민석"]


def test_round95_state_debug_exposes_operator_acceptance_fixture_surface() -> None:
    engine, _enforcement = _prepare_round94_engine()

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 96
    assert lcm["runtime_mapping_operator_acceptance_fixture_version"] == "v3_round95_runtime_mapping_operator_acceptance_fixture"
    assert lcm["runtime_mapping_operator_acceptance_fixture_available"] is True
    assert lcm["runtime_mapping_enable_smoke_precheck_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1
