"""EVE v3 rounds97-98 — controlled runtime mapping enable smoke and audit."""

from __future__ import annotations

import json
from typing import Any

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round95_runtime_mapping_operator_acceptance_fixture,
    run_round96_runtime_mapping_enable_smoke_precheck,
    run_round97_controlled_runtime_mapping_enable_smoke,
    run_round98_runtime_mapping_persistence_gate_audit,
    write_round97_controlled_runtime_mapping_enable_smoke,
    write_round98_runtime_mapping_persistence_gate_audit,
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


def _prepare_round96_engine() -> tuple[object, dict[str, Any]]:
    engine, enforcement = _prepare_engine_with_round94_fixture()
    fixture = run_round95_runtime_mapping_operator_acceptance_fixture(
        engine, source_enforcement=enforcement, accepted_tokens=["민석"]
    )
    precheck = run_round96_runtime_mapping_enable_smoke_precheck(engine, source_fixture=fixture)
    return engine, precheck

def test_round97_controlled_enable_smoke_rolls_back_runtime_mapping() -> None:
    engine, precheck = _prepare_round96_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)

    assert report["smoke_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert report["round"] == 97
    assert report["source_precheck_version"] == "v3_round96_runtime_mapping_enable_smoke_precheck"
    assert report["runtime_mapping_enabled_before"] is False
    assert report["runtime_mapping_enabled_during_smoke"] is False
    assert report["policy"]["guarded_rehearsal_only"] is True
    assert report["policy"]["runtime_mapping_never_enabled"] is True
    assert report["runtime_mapping_enabled_after_rollback"] is False
    assert report["enforcement_enabled_during_smoke"] is False
    assert report["enforcement_enabled_after_rollback"] is False
    assert report["candidate_count"] == 1
    assert report["mapped_count"] == 1
    assert report["blocked_count"] == 0
    assert report["mapped_tokens"] == ["민석"]
    assert report["rollback"]["rollback_complete"] is True
    assert report["rollback"]["ephemeral_mapping_table_cleared"] is True
    assert report["post_smoke_recommendation"]["may_attempt_persistence_gate_next_round"] is False
    assert report["post_smoke_recommendation"]["production_persistence_no_go"] is True

    row = report["smoke_rows"][0]
    assert row["smoke_status"] == "guarded_runtime_mapping_rehearsal_success"
    assert row["runtime_mapping_applied_during_smoke"] is False
    assert row["runtime_mapping_rehearsed_without_enablement"] is True
    assert row["runtime_mapping_persisted"] is False
    assert row["mapping_result"]["mapping_status"] == "guarded_runtime_mapping_rehearsal_success"
    assert row["mapping_result"]["anchor_source"] == "explicit_category_plus_sa_activation_only"
    assert row["agp_verify_called_during_runtime_mapping_enable_smoke"] is False
    assert row["embedding_lookup_called_during_enable_smoke"] is False

    checks = report["mutation_checks"]
    assert checks["category_snapshot_unchanged_during_enable_smoke"] is True
    assert checks["concept_commit_audit_unchanged_during_enable_smoke"] is True
    assert checks["eve_specific_vector_store_unchanged_during_enable_smoke"] is True
    assert checks["wrapper_telemetry_unchanged_during_enable_smoke"] is True
    assert checks["sa_active_categories_unchanged_during_enable_smoke"] is True
    assert checks["runtime_mapping_persisted"] is False
    assert report["runner_safety_checks"]["runtime_mapping_enabled_after_runner"] is False
    assert report["runner_safety_checks"]["enforcement_enabled_after_runner"] is False

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round98_persistence_gate_audit_keeps_runtime_mapping_disabled() -> None:
    engine, precheck = _prepare_round96_engine()
    smoke = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)
    audit = run_round98_runtime_mapping_persistence_gate_audit(engine, source_smoke=smoke)

    assert audit["audit_version"] == "v3_round98_runtime_mapping_persistence_gate_audit"
    assert audit["round"] == 98
    assert audit["source_smoke_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert audit["runtime_mapping_enabled"] is False
    assert audit["enforcement_enabled"] is False
    assert audit["mapped_count"] == 1
    assert audit["blocked_count"] == 0
    assert audit["hard_stop"] is False
    assert audit["persistence_gate_status"] == "blocked_production_persistence_no_go_guarded_rehearsal_only"
    assert audit["operator_recommendation"]["persist_runtime_mapping_now"] is False
    assert audit["operator_recommendation"]["requires_split_full_suite_with_medium_vectors"] is True
    assert audit["operator_recommendation"]["production_persistence_no_go"] is True
    assert audit["runner_read_only_checks"]["runtime_mapping_enabled_after_runner"] is False
    assert audit["runner_read_only_checks"]["enforcement_enabled_after_runner"] is False


def test_round97_98_exports_and_state_debug_surface(tmp_path) -> None:
    engine, precheck = _prepare_round96_engine()
    smoke = run_round97_controlled_runtime_mapping_enable_smoke(engine, source_precheck=precheck)
    audit = run_round98_runtime_mapping_persistence_gate_audit(engine, source_smoke=smoke)

    smoke_path = tmp_path / "round97.json"
    smoke_export = write_round97_controlled_runtime_mapping_enable_smoke(smoke, smoke_path)
    assert smoke_export["export_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert smoke_export["runtime_mapping_enabled_after_rollback"] is False
    assert smoke_export["runtime_mapping_persisted"] is False

    audit_path = tmp_path / "round98.json"
    audit_export = write_round98_runtime_mapping_persistence_gate_audit(audit, audit_path)
    assert audit_export["export_version"] == "v3_round98_runtime_mapping_persistence_gate_audit"
    assert audit_export["runtime_mapping_enabled"] is False
    assert audit_export["read_only"] is True

    loaded_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert loaded_smoke["mapped_tokens"] == ["민석"]
    assert loaded_audit["hard_stop"] is False

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 98
    assert lcm["controlled_runtime_mapping_enable_smoke_version"] == "v3_round97_controlled_runtime_mapping_enable_smoke"
    assert lcm["runtime_mapping_persistence_gate_audit_version"] == "v3_round98_runtime_mapping_persistence_gate_audit"
    assert lcm["controlled_runtime_mapping_enable_smoke_available"] is True
    assert lcm["runtime_mapping_persistence_gate_audit_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
