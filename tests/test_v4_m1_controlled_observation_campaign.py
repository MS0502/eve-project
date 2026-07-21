from __future__ import annotations

import ast
from pathlib import Path

from scripts.audit.m1_controlled_observation_campaign import (
    CAMPAIGN_ID,
    CAMPAIGN_SCHEMA_VERSION,
    STATIC_SILENT_BROAD_FROZEN,
    STATIC_SILENT_BROAD_INTEGRATED,
    run_controlled_observation_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = REPO_ROOT / "scripts/audit/m1_controlled_observation_campaign.py"


def test_campaign_is_deterministic_and_non_authoritative():
    first = run_controlled_observation_campaign()
    second = run_controlled_observation_campaign()

    assert first == second
    assert first["campaign_id"] == CAMPAIGN_ID
    assert first["campaign_schema_version"] == CAMPAIGN_SCHEMA_VERSION
    assert first["authority"] == "shadow_only"
    assert first["runtime_integrated"] is False
    assert first["persistence_mode"] == "none"
    assert first["packet"]["canonical_record"]["human_accepted"] is False
    assert first["packet"]["canonical_record"]["v4_2_eligible"] is False


def test_window_scope_and_event_rate_are_explicit():
    evidence = run_controlled_observation_campaign()
    window = evidence["observation_window"]
    rate = evidence["event_rate"]

    assert window == {
        "decay_cycles": 6,
        "duration_kind": "logical_steps_only",
        "legacy_calls": 12,
        "logical_steps": 18,
        "production_runtime_integrated": False,
        "scenario_count": 3,
        "scenarios": [
            "delegated_legacy_learn_pair_success_and_failure",
            "observer_snapshot_failure_isolation",
            "selected_active_silent_handler_probes",
        ],
        "tick_dt_total": 6.0,
        "tick_steps": 6,
        "wall_clock_duration": None,
    }
    assert rate["candidate_events"] == 12
    assert rate["events_per_legacy_call"] == {
        "denominator": 12,
        "numerator": 12,
        "value": 1.0,
    }
    assert rate["events_per_logical_step"]["denominator"] == 18
    assert rate["events_per_logical_step"]["numerator"] == 12
    assert rate["events_during_tick_steps"] == 0
    assert rate["max_events_in_one_step"] == 1
    assert rate["serialized_event_bytes_total"] > 0
    assert rate["serialized_event_bytes_max"] > 0
    assert rate["serialized_packet_bytes"] > 0


def test_replay_rate_and_complete_divergence_ledger_are_explicit():
    replay = run_controlled_observation_campaign()["replay_equivalence"]

    assert replay["compared_events"] == 12
    assert replay["match_rate"] == {
        "denominator": 12,
        "numerator": 12,
        "value": 1.0,
    }
    assert replay["divergence_count"] == 0
    assert replay["divergences"] == []
    assert replay["final_equivalence_matches"] is True
    assert replay["final_mismatches"] == []
    assert len(replay["rows"]) == 12
    assert all(row["matches"] for row in replay["rows"])
    assert all(row["mismatch_codes"] == [] for row in replay["rows"])


def test_event_types_include_one_visible_legacy_failure():
    events = run_controlled_observation_campaign()["events"]

    successes = [
        event
        for event in events
        if event["event_type"] == "shadow.legacy_mutation_observed_candidate"
    ]
    failures = [
        event
        for event in events
        if event["event_type"] == "shadow.legacy_mutation_failed_candidate"
    ]
    assert len(successes) == 11
    assert len(failures) == 1
    assert failures[0]["sequence"] == 7
    assert failures[0]["payload"]["legacy_outcome"] == {
        "error_type": "RuntimeError",
        "succeeded": False,
    }


def test_legacy_preservation_is_measured_against_unobserved_baseline():
    preservation = run_controlled_observation_campaign()["legacy_preservation"]

    assert preservation["call_order_preserved"] is True
    assert preservation["exception_identity_preserved"] is True
    assert preservation["legacy_state_matches_unobserved"] is True
    assert len(preservation["source_evidence_digest"]) == 64


def test_silent_failure_candidates_have_exact_locations_and_honest_remainder():
    evidence = run_controlled_observation_campaign()["silent_failure_observation"]
    candidates = evidence["candidates"]

    assert evidence["observed_candidate_count"] == 5
    assert evidence["selected_occurrence_count"] == 5
    assert evidence["selected_occurrences_observed"] == 5
    assert evidence["frozen_static_denominator"] == STATIC_SILENT_BROAD_FROZEN
    assert evidence["integrated_static_denominator"] == STATIC_SILENT_BROAD_INTEGRATED
    assert evidence["frozen_unobserved_remainder"] == 520
    assert evidence["integrated_unobserved_remainder"] == 527
    assert [(row["line_range"], row["stage"]) for row in candidates] == [
        ("40-43", "sa.decay"),
        ("44-47", "wm.decay"),
        ("50-55", "sa.apply_hormone_modulation"),
        ("88-91", "wm.get_focus"),
        ("94-97", "wm.get_focus_set"),
    ]
    assert all(row["path"] == "adapters/activation_adapter.py" for row in candidates)
    assert all(row["observed_silent"] is True for row in candidates)
    assert all(row["outward_error_type"] is None for row in candidates)
    assert all(len(row["error_message_digest"]) == 64 for row in candidates)
    assert "controlled silent probe" not in repr(candidates)


def test_packet_machine_passes_without_crossing_human_gate():
    packet = run_controlled_observation_campaign()["packet"]["canonical_record"]

    assert packet["event_count"] == 12
    assert packet["success_count"] == 11
    assert packet["failure_count"] == 1
    assert packet["observer_failure_count"] == 1
    assert packet["machine_status"] == "machine_evidence_complete"
    assert packet["machine_passed"] is True
    assert packet["eligible_for_human_review"] is True
    assert packet["human_review_status"] == "required_not_performed"
    assert packet["human_accepted"] is False
    assert packet["v4_2_eligible"] is False
    assert packet["runtime_integrated"] is False
    assert packet["persistence_mode"] == "none"
    assert packet["unauthorized_effects_detected"] is False


def test_campaign_module_has_no_io_clock_random_thread_or_runtime_install_surface():
    source = CAMPAIGN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    called_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert not imported_roots & {
        "asyncio",
        "datetime",
        "pathlib",
        "pickle",
        "random",
        "secrets",
        "sqlite3",
        "threading",
        "time",
        "uuid",
    }
    assert not called_names & {
        "connect",
        "install",
        "load",
        "open",
        "save",
        "sleep",
        "start",
        "write_bytes",
        "write_text",
    }
