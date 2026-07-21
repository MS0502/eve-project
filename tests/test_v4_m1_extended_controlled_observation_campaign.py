from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit.m1_extended_controlled_observation_campaign import (
    BASELINE_SHA,
    CAMPAIGN_ID,
    CAMPAIGN_SCHEMA_VERSION,
    REQUIRED_MUTATION_FORMS,
    STANDALONE_TICK_STEPS,
    canonical_raw_text,
    render_evidence_markdown,
    run_extended_controlled_observation_campaign,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json"
EVIDENCE_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_EVIDENCE.md"


@pytest.fixture(scope="module")
def evidence() -> dict:
    return run_extended_controlled_observation_campaign()


def test_campaign_is_deterministic_disconnected_and_non_authoritative(evidence):
    second = run_extended_controlled_observation_campaign()

    assert evidence == second
    assert evidence["baseline_sha"] == BASELINE_SHA
    assert evidence["campaign_id"] == CAMPAIGN_ID
    assert evidence["campaign_schema_version"] == CAMPAIGN_SCHEMA_VERSION
    assert evidence["authority"] == "shadow_only"
    assert evidence["machine_gate"] == {
        "machine_passed": True,
        "status": "extended_mechanism_evidence_complete",
    }
    assert evidence["human_gate"] == {
        "eligible_for_human_review": True,
        "human_accepted": False,
        "human_review_status": "required_not_performed",
        "v4_2_eligible": False,
    }
    assert evidence["unauthorized_effects"] == {
        "defaults_changed": False,
        "external_effects_outside_temporary_roots": False,
        "legacy_authority_changed": False,
        "production_persistence_changed": False,
    }
    window = evidence["observation_window"]
    assert window["runtime_integrated"] is False
    assert window["production_observer_installed"] is False
    assert window["production_persistence_enabled"] is False
    assert window["controlled_direct_write_cleanup_verified"] is True


def test_all_m0_a_mutation_forms_are_observed_and_replay_matched(evidence):
    classification = evidence["mutation_classification"]
    rows = classification["rows"]

    assert tuple(classification["required_forms"]) == REQUIRED_MUTATION_FORMS
    assert {row["form"] for row in rows} == set(REQUIRED_MUTATION_FORMS)
    assert len(rows) == len(REQUIRED_MUTATION_FORMS)
    assert all(row["observed"] is True for row in rows)
    assert all(row["replay_matches"] is True for row in rows)
    assert all(row["event_ids"] for row in rows)
    assert {row["path"] for row in rows} == {
        "adapters/live_loop.py",
        "adapters/persistence_adapter.py",
        "legacy/eve_modules/spreading_activation.py",
    }


def test_source_rows_correspond_to_real_ast_mutation_shapes(evidence):
    live_tree = ast.parse(
        (REPO_ROOT / "adapters/live_loop.py").read_text(encoding="utf-8")
    )
    spreading_tree = ast.parse(
        (REPO_ROOT / "legacy/eve_modules/spreading_activation.py").read_text(
            encoding="utf-8"
        )
    )
    persistence_tree = ast.parse(
        (REPO_ROOT / "adapters/persistence_adapter.py").read_text(encoding="utf-8")
    )

    assert any(
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == "processed_input_count"
        and isinstance(node.op, ast.Add)
        for node in ast.walk(live_tree)
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_last_emit_time"
            for target in node.targets
        )
        for node in ast.walk(live_tree)
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "weights"
            for target in node.targets
        )
        for node in ast.walk(spreading_tree)
    )
    assert sum(
        1
        for node in ast.walk(spreading_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add"
        and isinstance(node.func.value, ast.Subscript)
        and isinstance(node.func.value.value, ast.Attribute)
        and node.func.value.value.attr == "neighbors"
    ) >= 2
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dump"
        for node in ast.walk(persistence_tree)
    )


def test_multiple_adapter_dispositions_and_call_paths_are_explicit(evidence):
    targets = evidence["observation_window"]["adapter_call_paths"]

    assert evidence["observation_window"]["adapter_count"] == 3
    assert {row["disposition"] for row in targets} == {"WRAP", "REWRITE"}
    assert {(row["callable"], row["disposition"]) for row in targets} == {
        ("ActivationAdapter.learn_pair", "WRAP"),
        ("LiveLoop._drain_user_inputs", "REWRITE"),
        ("PersistenceAdapter.save", "REWRITE"),
    }
    assert len({row["stream_id"] for row in targets}) == 3
    assert len({row["target_id"] for row in targets}) == 3


def test_concurrency_probe_mutates_while_actual_tick_thread_is_alive(evidence):
    row = evidence["raw_observations"]["activation"]

    assert row["thread_started"] is True
    assert row["thread_barrier_reached"] is True
    assert row["thread_alive_before_mutation"] is True
    assert row["thread_alive_after_mutation"] is True
    assert row["tick_count_at_barrier"] == 1
    assert row["mutation_event_delta_while_thread_alive"] == 1
    assert row["live_tick_event_delta"] == 0
    assert row["thread_stopped"] is True
    assert row["thread_trace"] == ["tick:entered", "tick:released"]


def test_replay_is_complete_for_every_event_and_final_state(evidence):
    replay = evidence["replay_equivalence"]

    assert replay["compared_events"] == 4
    assert replay["match_rate"] == {
        "denominator": 4,
        "numerator": 4,
        "value": 1.0,
    }
    assert replay["divergence_count"] == 0
    assert replay["divergences"] == []
    assert len(replay["rows"]) == 4
    assert all(row["matches"] is True for row in replay["rows"])
    assert all(row["mismatch_codes"] == [] for row in replay["rows"])
    assert len(replay["final_equivalence"]) == 3
    assert all(row["matches"] is True for row in replay["final_equivalence"])


def test_failure_visibility_and_legacy_preservation_are_reconfirmed(evidence):
    visibility = evidence["failure_visibility"]
    observer = visibility["observer_failure"]
    preservation = evidence["legacy_preservation"]

    assert visibility["legacy_failure_event_count"] == 1
    assert visibility["legacy_failure_visible"] is True
    assert evidence["success_event_count"] == 3
    failures = [
        event
        for event in evidence["events"]
        if event["event_type"] == "shadow.legacy_mutation_failed_candidate"
    ]
    assert len(failures) == 1
    assert failures[0]["payload"]["legacy_outcome"] == {
        "error_type": "RuntimeError",
        "succeeded": False,
    }
    assert observer["event_count"] == 0
    assert observer["stage"] == "before_snapshot"
    assert observer["error_type"] == "RuntimeError"
    assert observer["legacy_state_preserved"] is True
    assert observer["return_value_preserved"] is True
    assert len(observer["error_message_digest"]) == 64
    assert all(preservation.values())


def test_granularity_remains_discrete_transition_only_without_amplification(evidence):
    granularity = evidence["granularity"]

    assert granularity == {
        "candidate_events": 4,
        "discrete_observed_calls": 4,
        "events_during_live_tick_before_mutation": 0,
        "events_during_standalone_tick_steps": 0,
        "max_events_per_observed_call": 1,
        "standalone_tick_steps": STANDALONE_TICK_STEPS,
    }
    assert evidence["raw_observations"]["activation"][
        "standalone_tick_event_delta"
    ] == 0


def test_direct_write_is_real_bounded_hashed_and_cleaned_up(evidence):
    persistence = evidence["raw_observations"]["persistence"]
    files = persistence["final_snapshot"]["files"]

    assert persistence["controlled_legacy_save_replaced"] is True
    assert persistence["state_matches_unobserved"] is True
    assert persistence["temporary_roots_removed"] is True
    assert persistence["event_delta"] == 1
    assert len(files) == 1
    assert files[0]["relative_path"] == "state.v41sidecar"
    assert files[0]["size_bytes"] > 0
    assert len(files[0]["sha256"]) == 64
    assert files == persistence["baseline_snapshot"]["files"]


def test_committed_raw_artifact_recalculates_every_claim(evidence):
    committed = RAW_PATH.read_text(encoding="utf-8")
    parsed = json.loads(committed)

    assert committed == canonical_raw_text(evidence)
    assert parsed == evidence
    assert parsed["machine_gate"]["machine_passed"] is True
    assert parsed["replay_equivalence"]["match_rate"]["numerator"] == len(
        parsed["events"]
    )
    assert parsed["replay_equivalence"]["divergence_count"] == len(
        parsed["replay_equivalence"]["divergences"]
    )
    assert parsed["granularity"]["candidate_events"] == len(parsed["events"])
    assert parsed["failure_visibility"]["legacy_failure_event_count"] == sum(
        event["event_type"] == "shadow.legacy_mutation_failed_candidate"
        for event in parsed["events"]
    )


def test_committed_report_is_exact_render_of_raw_artifact(evidence):
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    expected = render_evidence_markdown(evidence, raw_sha)
    normalized = " ".join(expected.split())

    assert EVIDENCE_PATH.read_text(encoding="utf-8") == expected
    assert f"Raw observation artifact SHA-256: `{raw_sha}`" in expected
    assert "5 / 532" in expected
    assert "not a claim that all historical mutation sites are covered" in normalized
