"""EVE v3 round89 — explicit concept commit smoke.

Round89 is a single mutation round. It persists the first minimal concept-layer
record from the Round88 v0 proposal freeze, but it does not enable runtime
lexical→concept mapping. EveSpecific/seed vectors remain evidence only; AGP
passes only from explicit category + SA activation.
"""

from __future__ import annotations

import json

from main import build_full_engine
from adapters.agp_adapter import AGP_REASON_ANCHORED, AGP_REASON_UNKNOWN_CATEGORY
from adapters.runtime_smoke_runner import (
    run_round89_explicit_concept_commit_smoke,
    write_round89_explicit_concept_commit_smoke,
)


def _prepare_engine_with_eve_specific_observations():
    engine = build_full_engine()
    learner = engine.eve_self_learning
    learner.observe_text("민석 오늘", source="round89_test_a")
    learner.observe_text("민석 군대", source="round89_test_b")
    assert learner.commit_audit_records() == []
    assert engine.eve_specific_vector_store.stats()["stored_count"] == 0
    return engine


def test_round89_explicit_concept_commit_creates_minimal_category_and_agp_bridge() -> None:
    engine = _prepare_engine_with_eve_specific_observations()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round89_explicit_concept_commit_smoke(engine)

    assert report["commit_version"] == "v3_round89_explicit_concept_commit_smoke"
    assert report["round"] == 89
    assert report["source_freeze_version"] == "v3_round88_concept_mapping_v0_proposal_freeze"
    assert report["created_count"] == 1
    assert report["created_tokens"] == ["민석"]
    assert report["blocked_count"] == 1
    assert report["blocked_tokens"] == ["EVE"]
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["agp_bridge_pass_count"] == 1
    assert report["agp_bridge_pass_tokens"] == ["민석"]
    row = report["committed_rows"][0]
    assert row["lexical_token"] == "민석"
    assert row["category_id"] == "concept_category::lex::민석"
    assert row["category_created"] is True
    assert row["concept_memory"]["concept_memory_persisted"] is True
    assert row["sa_activation"]["sa_activation_created"] is True
    assert row["agp_bridge"]["agp_verify_called"] is True
    assert row["agp_bridge"]["agp_passed"] is True
    assert row["agp_bridge"]["agp_reason"] == AGP_REASON_ANCHORED
    assert row["agp_bridge"]["uses_lexical_vector_as_anchor"] is False
    assert row["agp_bridge"]["uses_eve_specific_vector_as_anchor"] is False
    assert row["agp_bridge"]["uses_seed_vector_as_anchor"] is False
    assert row["agp_bridge"]["anchor_source"] == "explicit_category_plus_sa_activation"
    assert report["mutation_checks"]["eve_specific_vector_store_unchanged_during_concept_commit"] is True
    assert report["mutation_checks"]["wrapper_lookup_telemetry_unchanged_during_concept_commit"] is True
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round89_agp_still_fails_without_sa_activation_even_after_vector_and_category_exist() -> None:
    engine = _prepare_engine_with_eve_specific_observations()
    report = run_round89_explicit_concept_commit_smoke(engine)
    category_id = report["committed_rows"][0]["category_id"]

    # Explicit category without SA activation still fails. This proves the
    # EveSpecific vector/category label alone is not an AGP anchor.
    fail = engine.agp_adapter.verify(
        candidate_response="민석 개념을 확인했어.",
        meaning={"meaning_categories": [category_id]},
        activated_categories=[],
        hormone_state={},
    )
    assert fail.passed is False
    assert fail.reason == AGP_REASON_UNKNOWN_CATEGORY

    # Explicit category + SA activation passes.
    passed = engine.agp_adapter.verify(
        candidate_response="민석 개념을 확인했어.",
        meaning={"meaning_categories": [category_id]},
        activated_categories={category_id: 1.0},
        hormone_state={},
    )
    assert passed.passed is True
    assert passed.reason == AGP_REASON_ANCHORED


def test_round89_state_debug_and_export_surface(tmp_path) -> None:
    engine = _prepare_engine_with_eve_specific_observations()
    report = run_round89_explicit_concept_commit_smoke(engine)
    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 94
    assert lcm["explicit_concept_commit_smoke_version"] == "v3_round89_explicit_concept_commit_smoke"
    assert lcm["explicit_concept_commit_smoke_available"] is True
    assert lcm["concept_category_count"] == 1
    assert lcm["concept_commit_audit_count"] == 1
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False

    path = tmp_path / "round89.json"
    export = write_round89_explicit_concept_commit_smoke(report, path)
    assert export["export_version"] == "v3_round89_explicit_concept_commit_smoke"
    assert export["category_created"] is True
    assert export["concept_memory_mutation"] is True
    assert export["sa_activation_created"] is True
    assert export["agp_verify_called"] is True
    assert export["eve_specific_vector_commit_called"] is False
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["commit_version"] == report["commit_version"]
