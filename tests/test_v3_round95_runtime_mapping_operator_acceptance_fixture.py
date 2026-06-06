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
    run_round95_runtime_mapping_operator_acceptance_fixture,
    write_round95_runtime_mapping_operator_acceptance_fixture,
)

ROUND94_ENFORCEMENT_FIXTURE: dict[str, Any] = {
    "dry_run_version": "v3_round94_runtime_mapping_enforcement_dry_run",
    "source_proposal_version": "v3_round93_runtime_mapping_proposal_report",
    "source_dry_run_version": "v3_round92_runtime_mapping_gate_dry_run",
    "runtime_mapping_enabled": False,
    "enforcement_enabled": False,
    "candidate_count": 2,
    "would_apply_count": 1,
    "blocked_count": 1,
    "enforcement_rows": [
        {
            "lexical_token": "EVE",
            "target_category_id": "concept_category::lex::EVE",
            "enforcement_status": "blocked_from_runtime_mapping_enforcement",
            "would_apply_if_runtime_mapping_enabled": False,
            "runtime_mapping_enabled_now": False,
            "enforcement_enabled_now": False,
            "runtime_mapping_applied_now": False,
            "simulated_mapping_result": None,
            "blocked_reasons": ["explicit_category_missing", "concept_memory_missing"],
            "uses_lexical_vector_as_anchor": False,
            "uses_eve_specific_vector_as_anchor": False,
            "uses_seed_vector_as_anchor": False,
        },
        {
            "lexical_token": "민석",
            "target_category_id": "concept_category::lex::민석",
            "enforcement_status": "would_apply_if_runtime_mapping_enabled",
            "would_apply_if_runtime_mapping_enabled": True,
            "runtime_mapping_enabled_now": False,
            "enforcement_enabled_now": False,
            "runtime_mapping_applied_now": False,
            "simulated_mapping_result": {
                "lexical_token": "민석",
                "category_id": "concept_category::lex::민석",
                "category_label": "민석",
                "mapping_status": "simulated_runtime_mapping_success",
                "anchor_source": "explicit_category_plus_sa_activation_only",
            },
            "blocked_reasons": [],
            "uses_lexical_vector_as_anchor": False,
            "uses_eve_specific_vector_as_anchor": False,
            "uses_seed_vector_as_anchor": False,
        },
    ],
    "would_apply_tokens": ["민석"],
    "would_block_tokens": ["EVE"],
    "read_only": True,
}


def _install_minseok_category(engine: Any) -> None:
    """Install explicit concept evidence without vector creation or runtime mapping."""
    lcm = engine.lex_concept_mapping
    category_id = "concept_category::lex::민석"
    lcm._concept_categories[category_id] = {
        "lexical_token": "민석",
        "category_id": category_id,
        "category_created": True,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "concept_memory": {"concept_memory_persisted": True},
        "sa_activation": {"sa_activation_created": True},
        "lexical_vector_is_evidence_only": True,
        "eve_specific_vector_is_not_agp_anchor": True,
        "seed_vector_is_not_agp_anchor": True,
    }


def _prepare_engine_with_round94_fixture() -> tuple[object, dict[str, Any]]:
    engine = build_full_engine()
    _install_minseok_category(engine)
    return engine, json.loads(json.dumps(ROUND94_ENFORCEMENT_FIXTURE, ensure_ascii=False))

def test_round95_operator_acceptance_fixture_plans_mapping_without_mutation() -> None:
    engine, enforcement = _prepare_engine_with_round94_fixture()
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
    engine, enforcement = _prepare_engine_with_round94_fixture()
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
    engine, enforcement = _prepare_engine_with_round94_fixture()
    run_round95_runtime_mapping_operator_acceptance_fixture(
        engine,
        source_enforcement=enforcement,
        accepted_tokens=["민석"],
    )

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 96
    assert lcm["runtime_mapping_operator_acceptance_fixture_version"] == "v3_round95_runtime_mapping_operator_acceptance_fixture"
    assert lcm["runtime_mapping_operator_acceptance_fixture_available"] is True
    assert lcm["runtime_mapping_enable_smoke_precheck_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1
