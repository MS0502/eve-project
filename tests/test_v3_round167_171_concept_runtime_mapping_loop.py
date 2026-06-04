"""Rounds167-171 concept/runtime mapping non-artifact loop tests."""

from __future__ import annotations

from main import build_full_engine
from adapters.concept_runtime_mapping_diagnostics import (
    concept_runtime_mapping_failure_taxonomy,
    round169_state_debug_baseline_gate,
    selected_non_artifact_subcluster,
)


def test_round167_taxonomy_selects_only_non_artifact_state_debug_subcluster() -> None:
    taxonomy = concept_runtime_mapping_failure_taxonomy()

    assert taxonomy["round"] == 167
    assert taxonomy["total_observed_failures"] == 43
    selected = [row for row in taxonomy["subclusters"] if row["selected_for_fix"]]
    assert [row["id"] for row in selected] == ["state_debug_baseline_round_metadata"]
    assert selected[0]["count"] == 5
    assert selected[0]["artifact_dependent"] is False
    blocked = [row for row in taxonomy["subclusters"] if row["artifact_dependent"]]
    assert blocked[0]["count"] == 38
    assert "민석" in blocked[0]["blocker"]
    assert taxonomy["global_policy_flags"]["production_persistence_enabled"] is False
    assert taxonomy["global_policy_flags"]["runtime_mapping_enabled_default"] is False
    assert taxonomy["global_policy_flags"]["enforcement_enabled"] is False
    assert taxonomy["vectors_written"] is False


def test_round168_selection_rationale_rejects_vector_dependent_subcluster() -> None:
    selection = selected_non_artifact_subcluster()

    assert selection["round"] == 168
    assert selection["selected_subcluster"]["id"] == "state_debug_baseline_round_metadata"
    assert selection["selected_subcluster"]["artifact_dependent"] is False
    assert selection["rejected_subclusters"][0]["id"] == "artifact_dependent_eve_specific_commit_prerequisite"
    assert selection["rejected_subclusters"][0]["artifact_dependent"] is True
    assert selection["read_only"] is True


def test_round169_state_debug_baseline_stays_round94_until_explicit_later_surface() -> None:
    engine = build_full_engine()
    lcm = engine.lex_concept_mapping

    baseline = round169_state_debug_baseline_gate(engine)
    assert baseline["baseline_round_ok"] is True
    assert baseline["baseline_round"] == 94
    assert baseline["runtime_mapping_enabled"] is False
    assert baseline["enforcement_enabled"] is False

    state = engine.state_debug.snapshot_state()["lex_concept_mapping"]
    assert state["round"] == 94
    assert state["runtime_mapping_enabled"] is False
    assert state["enforcement_enabled"] is False

    lcm.runtime_mapping_operator_acceptance_fixture(
        source_enforcement={
            "dry_run_version": "v3_round94_runtime_mapping_enforcement_dry_run",
            "source_proposal_version": "v3_round93_runtime_mapping_proposal_report",
            "enforcement_rows": [],
        }
    )
    transitioned = engine.state_debug.snapshot_state()["lex_concept_mapping"]
    assert transitioned["round"] == 96
    assert transitioned["runtime_mapping_operator_acceptance_fixture_available"] is True
    assert transitioned["runtime_mapping_enable_smoke_precheck_available"] is True
    assert transitioned["runtime_mapping_enabled"] is False
    assert transitioned["enforcement_enabled"] is False
