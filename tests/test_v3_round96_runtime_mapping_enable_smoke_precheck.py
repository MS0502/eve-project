"""EVE v3 round96 — runtime mapping enable-smoke precheck.

Round96 in this package is intentionally a read-only pre-mutation gate. It
packages the requirements for a future controlled runtime mapping enable smoke,
but it does not enable runtime mapping or enforcement.
"""

from __future__ import annotations

import json
from typing import Any

from main import build_full_engine
from adapters.runtime_smoke_runner import (
    run_round95_runtime_mapping_operator_acceptance_fixture,
    run_round96_runtime_mapping_enable_smoke_precheck,
    write_round96_runtime_mapping_enable_smoke_precheck,
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


def _prepare_round95_engine() -> tuple[object, dict[str, Any]]:
    engine, enforcement = _prepare_engine_with_round94_fixture()
    fixture = run_round95_runtime_mapping_operator_acceptance_fixture(
        engine, source_enforcement=enforcement, accepted_tokens=["민석"]
    )
    return engine, fixture

def test_round96_enable_smoke_precheck_is_ready_without_enabling_runtime_mapping() -> None:
    engine, fixture = _prepare_round95_engine()
    lcm = engine.lex_concept_mapping
    before_categories = lcm.concept_categories_snapshot()
    before_records = lcm.concept_commit_records()
    before_store = engine.eve_specific_vector_store.stats().copy()
    before_telemetry = engine.self_embedding.telemetry().copy()

    report = run_round96_runtime_mapping_enable_smoke_precheck(engine, source_fixture=fixture)

    assert report["precheck_version"] == "v3_round96_runtime_mapping_enable_smoke_precheck"
    assert report["round"] == 96
    assert report["source_fixture_version"] == "v3_round95_runtime_mapping_operator_acceptance_fixture"
    assert report["runtime_mapping_enabled"] is False
    assert report["enforcement_enabled"] is False
    assert report["candidate_count"] == 1
    assert report["ready_count"] == 1
    assert report["blocked_count"] == 0
    assert report["ready_tokens"] == ["민석"]
    assert report["blocked_tokens"] == []
    assert report["operator_recommendation"]["may_attempt_separate_enable_smoke"] is True
    assert report["operator_recommendation"]["runtime_mapping_should_be_enabled_now"] is False
    assert report["operator_recommendation"]["safe_default"] == "do_not_enable_in_precheck_round"

    ready = report["ready_rows"][0]
    assert ready["lexical_token"] == "민석"
    assert ready["precheck_status"] == "ready_for_separate_enable_smoke_round"
    assert ready["precheck_blocked_reasons"] == []
    assert ready["precheck_checks"]["operator_accepted"] is True
    assert ready["precheck_checks"]["has_target_category_id"] is True
    assert ready["precheck_checks"]["has_simulated_mapping_result"] is True
    assert ready["precheck_checks"]["would_apply_after_fixture"] is True
    assert ready["precheck_checks"]["uses_lexical_vector_as_anchor"] is False
    assert ready["precheck_checks"]["uses_eve_specific_vector_as_anchor"] is False
    assert ready["precheck_checks"]["uses_seed_vector_as_anchor"] is False

    gate = report["pre_mutation_gate"]
    assert gate["operator_approval_required_for_actual_enable"] is True
    assert gate["pre_mutation_checkpoint_required"] is True
    assert gate["rollback_plan_required"] is True
    assert gate["split_full_suite_required_before_persistence"] is True

    checks = report["read_only_checks"]
    assert checks["category_snapshot_unchanged_during_enable_smoke_precheck"] is True
    assert checks["concept_commit_audit_unchanged_during_enable_smoke_precheck"] is True
    assert checks["eve_specific_vector_store_unchanged_during_enable_smoke_precheck"] is True
    assert checks["wrapper_telemetry_unchanged_during_enable_smoke_precheck"] is True
    assert checks["sa_active_categories_unchanged_during_enable_smoke_precheck"] is True
    assert checks["category_created_during_enable_smoke_precheck"] is False
    assert checks["concept_memory_mutation_during_enable_smoke_precheck"] is False
    assert checks["frame_hypergraph_mutation_during_enable_smoke_precheck"] is False
    assert checks["sa_activation_created_during_enable_smoke_precheck"] is False
    assert checks["agp_verify_called_during_enable_smoke_precheck"] is False
    assert checks["embedding_lookup_called_during_enable_smoke_precheck"] is False
    assert checks["runtime_mapping_applied"] is False
    assert report["runner_read_only_checks"]["concept_commit_not_called_by_round96_runner"] is True
    assert report["runner_read_only_checks"]["runtime_mapping_not_enabled_by_runner"] is True

    assert lcm.concept_categories_snapshot() == before_categories
    assert lcm.concept_commit_records() == before_records
    assert engine.eve_specific_vector_store.stats() == before_store
    assert engine.self_embedding.telemetry() == before_telemetry


def test_round96_export_and_state_debug_surface(tmp_path) -> None:
    engine, fixture = _prepare_round95_engine()
    report = run_round96_runtime_mapping_enable_smoke_precheck(engine, source_fixture=fixture)

    state = engine.state_debug.snapshot_state()
    lcm = state["lex_concept_mapping"]
    assert lcm["round"] == 96
    assert lcm["runtime_mapping_enable_smoke_precheck_version"] == "v3_round96_runtime_mapping_enable_smoke_precheck"
    assert lcm["runtime_mapping_enable_smoke_precheck_available"] is True
    assert lcm["runtime_mapping_enabled"] is False
    assert lcm["enforcement_enabled"] is False
    assert lcm["concept_category_count"] == 1

    path = tmp_path / "round96.json"
    export = write_round96_runtime_mapping_enable_smoke_precheck(report, path)
    assert export["export_version"] == "v3_round96_runtime_mapping_enable_smoke_precheck"
    assert export["recomputed"] is False
    assert export["runtime_mapping_enabled"] is False
    assert export["enforcement_enabled"] is False
    assert export["category_created"] is False
    assert export["concept_memory_mutation"] is False
    assert export["sa_activation_created"] is False
    assert export["agp_bypass"] is False
    assert export["read_only"] is True

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["precheck_version"] == report["precheck_version"]
    assert loaded["ready_tokens"] == ["민석"]
