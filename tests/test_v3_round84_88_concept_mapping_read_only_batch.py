"""EVE v3 rounds 84-88 — concept mapping read-only batch.

The batch advances the lexical→concept axis through evidence, SA path, AGP
bridge, readiness dashboard, and v0 proposal freeze surfaces. All surfaces are
read-only: no runtime mapping, category creation, concept memory/frame mutation,
SA activation, AGP anchor, AGP verify call, embedding lookup, or vector commit.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round84_concept_memory_frame_evidence_dry_run,
    run_round85_sa_activation_path_dry_run,
    run_round86_agp_bridge_smoke_dry_run,
    run_round87_concept_mapping_readiness_dashboard,
    run_round88_concept_mapping_v0_proposal_freeze,
    write_round88_concept_mapping_v0_proposal_freeze,
)


def _prepare_engine_with_eve_specific_commit():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round84_88_test_a")
    learner.observe_text("민석 군대", source="round84_88_test_b")
    commit = learner.commit_eve_specific_vectors(["민석"], context_words=["오늘", "군대"])
    assert "민석" in commit["created"]
    return engine


def _assert_read_only_checks(report: dict, phase_key: str) -> None:
    checks = report["read_only_checks"]
    assert checks[f"audit_records_unchanged_during_{phase_key}"] is True
    assert checks[f"vector_store_unchanged_during_{phase_key}"] is True
    assert checks[f"telemetry_unchanged_during_{phase_key}"] is True
    assert checks[f"self_learning_policy_unchanged_during_{phase_key}"] is True
    assert checks["runtime_mapping_enabled_after_run"] is False
    assert checks["enforcement_enabled_after_run"] is False
    assert checks["operator_decision_persisted"] is False
    assert checks["category_created"] is False
    assert checks["concept_memory_mutation"] is False
    assert checks["frame_hypergraph_mutation"] is False
    assert checks["sa_activation_created"] is False
    assert checks["agp_anchor_created"] is False
    assert checks["agp_verify_called"] is False
    assert checks["wrapper_lookup_called"] is False
    assert checks["vector_commit_called"] is False


def test_round84_concept_memory_frame_evidence_dry_run_is_read_only() -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round84_concept_memory_frame_evidence_dry_run(engine)

    assert report["dry_run_version"] == "v3_round84_concept_memory_frame_evidence_dry_run"
    assert report["round"] == 84
    assert report["source_round83_dry_run_version"] == "v3_round83_operator_acceptance_category_creation_dry_run"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["evidence_dry_run_count"] == 1
    assert report["evidence_tokens"] == ["민석"]
    row = report["evidence_rows"][0]
    assert row["lexical_token"] == "민석"
    assert row["concept_evidence_dry_run"]["would_attach_concept_memory_reference"] is True
    assert row["concept_evidence_dry_run"]["concept_memory_persisted"] is False
    assert row["concept_evidence_dry_run"]["frame_persisted"] is False
    assert "concept_memory_or_frame_evidence_dry_run_only" in row["remaining_blocked_reasons"]
    assert "concept_memory_or_frame_evidence_required" not in row["remaining_blocked_reasons"]
    assert report["policy"]["no_concept_memory_mutation"] is True
    assert report["policy"]["no_frame_hypergraph_mutation"] is True
    _assert_read_only_checks(report, "evidence_dry_run")
    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round85_sa_activation_path_dry_run_is_read_only() -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    report = run_round85_sa_activation_path_dry_run(engine)

    assert report["dry_run_version"] == "v3_round85_sa_activation_path_dry_run"
    assert report["round"] == 85
    assert report["source_round84_dry_run_version"] == "v3_round84_concept_memory_frame_evidence_dry_run"
    assert report["sa_activation_path_dry_run_count"] == 1
    assert report["sa_path_tokens"] == ["민석"]
    row = report["sa_activation_path_rows"][0]
    assert row["sa_activation_path_dry_run"]["would_activate_category"] is True
    assert row["sa_activation_path_dry_run"]["activation_created"] is False
    assert "sa_activation_path_dry_run_only" in row["remaining_blocked_reasons"]
    assert "sa_activation_path_required" not in row["remaining_blocked_reasons"]
    assert report["policy"]["no_sa_activation_creation"] is True
    _assert_read_only_checks(report, "sa_path_dry_run")


def test_round86_agp_bridge_smoke_dry_run_never_calls_agp_verify() -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    report = run_round86_agp_bridge_smoke_dry_run(engine)

    assert report["dry_run_version"] == "v3_round86_agp_bridge_smoke_dry_run"
    assert report["round"] == 86
    assert report["source_round85_dry_run_version"] == "v3_round85_sa_activation_path_dry_run"
    assert report["agp_bridge_dry_run_count"] == 1
    assert report["bridge_ready_if_persisted_count"] == 1
    assert report["bridge_ready_if_persisted_tokens"] == ["민석"]
    row = report["agp_bridge_rows"][0]
    bridge = row["agp_bridge_smoke_dry_run"]
    assert bridge["uses_lexical_vector_as_anchor"] is False
    assert bridge["uses_eve_specific_vector_as_anchor"] is False
    assert bridge["agp_verify_called_now"] is False
    assert bridge["agp_anchor_created"] is False
    assert bridge["would_pass_if_persisted"] is True
    assert row["would_pass_mapping_gate_after_bridge_fixture"] is False
    assert row["would_pass_if_all_dry_run_artifacts_persisted"] is True
    assert report["policy"]["no_agp_verify_call"] is True
    _assert_read_only_checks(report, "agp_bridge_dry_run")


def test_round87_readiness_dashboard_and_round88_freeze_are_read_only() -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    dashboard = run_round87_concept_mapping_readiness_dashboard(engine)

    assert dashboard["dashboard_version"] == "v3_round87_concept_mapping_readiness_dashboard"
    assert dashboard["round"] == 87
    assert dashboard["ready_for_explicit_concept_commit_smoke_count"] == 1
    assert dashboard["ready_for_explicit_concept_commit_smoke_tokens"] == ["민석"]
    assert dashboard["stage_counts"]["bridge_ready_if_persisted_count"] == 1
    row = dashboard["dashboard_rows"][0]
    assert row["ready_for_explicit_concept_commit_smoke"] is True
    assert row["category_created"] is False
    assert row["agp_verify_called"] is False
    _assert_read_only_checks(dashboard, "readiness_dashboard")

    freeze = run_round88_concept_mapping_v0_proposal_freeze(engine)
    assert freeze["freeze_version"] == "v3_round88_concept_mapping_v0_proposal_freeze"
    assert freeze["round"] == 88
    assert freeze["status"] == "frozen_for_operator_review_no_runtime_mapping"
    assert freeze["explicit_concept_commit_candidate_count"] == 1
    assert freeze["explicit_concept_commit_candidate_tokens"] == ["민석"]
    assert freeze["frozen_policy"]["no_runtime_mapping"] is True
    assert freeze["frozen_policy"]["no_category_creation"] is True
    assert freeze["frozen_policy"]["no_concept_memory_mutation"] is True
    assert freeze["frozen_policy"]["no_sa_activation_creation"] is True
    assert freeze["frozen_policy"]["no_agp_anchor_creation"] is True
    assert freeze["frozen_policy"]["no_agp_verify_call"] is True
    assert freeze["next_round_recommendation"]["round89"] == "explicit_concept_commit_smoke_single_round_only"
    _assert_read_only_checks(freeze, "v0_proposal_freeze")


def test_round88_export_does_not_recompute_or_mutate(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    report = run_round88_concept_mapping_v0_proposal_freeze(engine)
    before_audit = len(engine.eve_self_learning.commit_audit_records())
    before_store = engine.eve_specific_vector_store.stats().copy()

    path = tmp_path / "round88.json"
    result = write_round88_concept_mapping_v0_proposal_freeze(report, path)

    assert result["export_version"] == "v3_round88_concept_mapping_v0_proposal_freeze"
    assert result["recomputed"] is False
    assert result["category_created"] is False
    assert result["concept_memory_mutation"] is False
    assert result["sa_activation_created"] is False
    assert result["agp_anchor_created"] is False
    assert result["agp_verify_called"] is False
    assert len(engine.eve_self_learning.commit_audit_records()) == before_audit
    assert engine.eve_specific_vector_store.stats() == before_store
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["freeze_version"] == report["freeze_version"]


def test_round88_state_debug_exposes_batch_versions() -> None:
    engine = _prepare_engine_with_eve_specific_commit()
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 94
    assert lcm["concept_memory_frame_evidence_dry_run_version"] == "v3_round84_concept_memory_frame_evidence_dry_run"
    assert lcm["sa_activation_path_dry_run_version"] == "v3_round85_sa_activation_path_dry_run"
    assert lcm["agp_bridge_smoke_dry_run_version"] == "v3_round86_agp_bridge_smoke_dry_run"
    assert lcm["concept_mapping_readiness_dashboard_version"] == "v3_round87_concept_mapping_readiness_dashboard"
    assert lcm["concept_mapping_v0_proposal_freeze_version"] == "v3_round88_concept_mapping_v0_proposal_freeze"
    assert lcm["concept_mapping_v0_proposal_freeze_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
