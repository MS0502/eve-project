"""EVE v3 round94 — runtime mapping enforcement dry-run.

Round94 remains read-only. It simulates the API/result shape that a future
runtime lexical→concept mapping enforcement patch would expose, while keeping
runtime mapping and enforcement disabled.
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
    run_round94_runtime_mapping_enforcement_dry_run,
    write_round94_runtime_mapping_enforcement_dry_run,
)


def _prepare_round93_engine():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round94_test_a")
    learner.observe_text("민석 군대", source="round94_test_b")
    assert learner.commit_audit_records() == []
    assert engine.eve_specific_vector_store.stats()["stored_count"] == 0
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
    return engine, proposal


def test_round94_runtime_mapping_enforcement_dry_run_simulates_api_without_enabling_mapping() -> None:
    engine, proposal = _prepare_round93_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round94_runtime_mapping_enforcement_dry_run(
        engine,
        tokens=["민석", "EVE"],
        source_proposal=proposal,
    )

    assert report["dry_run_version"] == "v3_round94_runtime_mapping_enforcement_dry_run"
    assert report["round"] == 94
    assert report["source_proposal_version"] == "v3_round93_runtime_mapping_proposal_report"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 2
    assert report["would_apply_count"] == 1
    assert report["would_block_count"] == 1
    assert report["would_apply_tokens"] == ["민석"]
    assert report["would_block_tokens"] == ["EVE"]
    assert report["operator_summary"]["runtime_mapping_enforcement_can_be_considered"] is True
    assert report["operator_summary"]["runtime_mapping_should_be_enabled_now"] is False
    assert report["operator_summary"]["safe_default"] == "keep_runtime_mapping_disabled"
    assert report["operator_summary"]["next_recommended_round"] == "round95_runtime_mapping_operator_acceptance_fixture"

    rows = {row["lexical_token"]: row for row in report["enforcement_rows"]}
    ready = rows["민석"]
    assert ready["enforcement_status"] == "would_apply_if_runtime_mapping_enabled"
    assert ready["target_category_id"] == "concept_category::lex::민석"
    assert ready["runtime_mapping_applied_now"] is False
    assert ready["simulated_mapping_result"]["mapping_status"] == "simulated_runtime_mapping_success"
    assert ready["simulated_mapping_result"]["anchor_source"] == "explicit_category_plus_sa_activation_only"
    assert ready["api_path_dry_run"] == [
        "RuntimeLexConceptMapper.resolve(token)",
        "LexConceptMappingAdapter.runtime_mapping_gate",
        "ConceptCategoryRegistry.lookup(category_id)",
        "SAActivation.check(category_id)",
        "return MappingResult(category_id)",
    ]
    assert ready["uses_lexical_vector_as_anchor"] is False
    assert ready["uses_eve_specific_vector_as_anchor"] is False
    assert ready["uses_seed_vector_as_anchor"] is False

    blocked = rows["EVE"]
    assert blocked["enforcement_status"] == "blocked_from_runtime_mapping_enforcement"
    assert blocked["would_apply_if_runtime_mapping_enabled"] is False
    assert "explicit_category_missing" in blocked["blocked_reasons"]
    assert "concept_memory_missing" in blocked["blocked_reasons"]
    assert blocked["simulated_mapping_result"] is None

    checks = report["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_enforcement_dry_run"] is True
    assert checks["concept_commit_audit_unchanged_during_enforcement_dry_run"] is True
    assert checks["eve_specific_vector_store_unchanged_during_enforcement_dry_run"] is True
    assert checks["wrapper_telemetry_unchanged_during_enforcement_dry_run"] is True
    assert checks["sa_active_categories_unchanged_during_enforcement_dry_run"] is True
    assert checks["category_created_during_enforcement_dry_run"] is False
    assert checks["concept_memory_mutation_during_enforcement_dry_run"] is False
    assert checks["sa_activation_created_during_enforcement_dry_run"] is False
    assert checks["agp_verify_called_during_runtime_mapping_enforcement_dry_run"] is False
    assert checks["embedding_lookup_called_during_enforcement_dry_run"] is False
    assert checks["runtime_mapping_applied"] is False
    assert report["runner_read_only_checks"]["concept_commit_not_called_by_round94_runner"] is True
    assert report["runner_read_only_checks"]["runtime_mapping_not_enabled_by_runner"] is True

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round94_state_debug_and_export_surface(tmp_path) -> None:
    engine, proposal = _prepare_round93_engine()
    report = run_round94_runtime_mapping_enforcement_dry_run(engine, source_proposal=proposal)

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 96
    assert lcm["runtime_mapping_enforcement_dry_run_version"] == "v3_round94_runtime_mapping_enforcement_dry_run"
    assert lcm["runtime_mapping_enforcement_dry_run_available"] is True
    assert lcm["runtime_mapping_operator_acceptance_fixture_available"] is True
    assert lcm["runtime_mapping_enable_smoke_precheck_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1

    path = tmp_path / "round94.json"
    export = write_round94_runtime_mapping_enforcement_dry_run(report, path)
    assert export["export_version"] == "v3_round94_runtime_mapping_enforcement_dry_run"
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["eve_specific_vector_commit_called"] is False
    assert export["read_only"] is True
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["dry_run_version"] == report["dry_run_version"]
