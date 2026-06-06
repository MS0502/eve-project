from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from adapters.affect_event_to_axis_proposal_map import (
    HARDWARE_EVENTS,
    HOSTILE_SOCIAL_EVENTS,
    MAX_DELTA_LIMIT,
    MEMORY_SELF_UPDATE_EVENTS,
    REQUIRED_EVENT_CATEGORIES,
    event_to_axis_proposal_map,
    proposal_map_summary,
    validate_event_proposal_map,
)
from adapters.affect_hormone_interaction_matrix import affect_hormone_interaction_matrix, interaction_matrix_summary, validate_interaction_matrix
from adapters.affect_hormone_neural_rhythm_registry import AXIS_GROUPS, affect_hormone_axis_registry, no_runtime_mutation_proof
from scripts.operator_report_round781_800_affect_event_proposal_map import build_operator_report
from tests.fixtures.korean_conversation_fixtures import fixture_texts


SOCIAL_SELF_IDENTITY_AXES = set(AXIS_GROUPS["social_relationship"] + AXIS_GROUPS["self_identity"])
OPERATIONAL_AXES = set(AXIS_GROUPS["survival_stability"])
FORBIDDEN_ARTIFACT_PATTERNS = ("_operator_artifacts/", "vectors.npy", "vocab.txt", "subset_manifest.json", "seeds/subsets", ".zip", ".part")


def test_all_required_event_categories_exist() -> None:
    proposal_map = event_to_axis_proposal_map()

    assert set(REQUIRED_EVENT_CATEGORIES).issubset(proposal_map)
    assert validate_event_proposal_map(proposal_map)["passed"] is True


def test_allowed_axis_deltas_reference_registry_axes_and_are_bounded_small() -> None:
    proposal_map = event_to_axis_proposal_map()
    registry_axes = set(affect_hormone_axis_registry())

    for row in proposal_map.values():
        assert set(row["allowed_axis_deltas"]).issubset(registry_axes)
        assert len(row["allowed_axis_deltas"]) < len(registry_axes)
        for axis, delta in row["allowed_axis_deltas"].items():
            assert abs(delta) <= MAX_DELTA_LIMIT
            assert row["max_delta_per_axis"][axis] <= MAX_DELTA_LIMIT


def test_hostile_social_events_cannot_direct_write_and_require_quarantine_appraisal_gate() -> None:
    proposal_map = event_to_axis_proposal_map()

    for event in HOSTILE_SOCIAL_EVENTS:
        row = proposal_map[event]
        assert row["can_modify_core_identity"] is False
        assert row["can_modify_self_model_directly"] is False
        assert row["can_write_long_term_memory_directly"] is False
        assert row["requires_quarantine"] is True
        assert row["requires_appraisal"] is True
        assert row["requires_gate"]

    assert "self_respect_decrease" in proposal_map["identity_attack"]["forbidden_axis_deltas"]
    assert proposal_map["identity_attack"]["allowed_axis_deltas"]["social_pain"] > 0


def test_hardware_normal_zero_and_hardware_events_cannot_affect_social_self_identity_axes() -> None:
    proposal_map = event_to_axis_proposal_map()

    assert proposal_map["hardware_normal"]["allowed_axis_deltas"] == {}
    for event in HARDWARE_EVENTS:
        assert set(proposal_map[event]["allowed_axis_deltas"]).isdisjoint(SOCIAL_SELF_IDENTITY_AXES)
        assert proposal_map[event]["can_trigger_global_synchrony"] is False
        assert "hardware_governor_non_panic_policy" in proposal_map[event]["requires_gate"]


def test_hardware_low_power_states_use_non_panic_operational_effects_only() -> None:
    proposal_map = event_to_axis_proposal_map()

    for event in ("hardware_low_power", "hardware_critical_prepare", "hardware_shutdown_imminent"):
        row = proposal_map[event]
        assert set(row["allowed_axis_deltas"]).issubset(OPERATIONAL_AXES)
        assert "panic" in row["forbidden_axis_deltas"] or event != "hardware_low_power"
        assert row["can_modify_core_identity"] is False


def test_hardware_prediction_error_diagnostic_only_and_polling_tick_no_recursive_loop() -> None:
    proposal_map = event_to_axis_proposal_map()

    prediction_error = proposal_map["hardware_prediction_error"]
    assert set(prediction_error["allowed_axis_deltas"]).issubset(OPERATIONAL_AXES)
    assert "diagnostic_flags_not_existential_threat" in prediction_error["notes"]
    assert "existential_threat" in prediction_error["forbidden_axis_deltas"]

    polling = proposal_map["hardware_polling_tick"]
    assert polling["allowed_axis_deltas"] == {}
    assert "recursive_concern_loop" in polling["forbidden_axis_deltas"]


def test_speech_pressure_cannot_bypass_agp_or_fallback() -> None:
    proposal_map = event_to_axis_proposal_map()

    for event in ("speech_output_pressure", "defensive_speech_pressure"):
        row = proposal_map[event]
        assert row["can_bypass_agp"] is False
        assert row["can_bypass_fallback"] is False
        assert "speech_emit_directly" in row["forbidden_axis_deltas"]
        assert "agp_verification" in row["requires_gate"]
        assert "fallback_required" in row["requires_gate"]


def test_listening_uncertainty_cannot_relabel_neutral_as_hostile_by_itself() -> None:
    row = event_to_axis_proposal_map()["listening_uncertainty"]

    assert "neutral_input_hostile_relabel" in row["forbidden_axis_deltas"]
    assert row["can_bypass_agp"] is False


def test_imagination_negative_spiral_requires_budget_cooldown_reality_check() -> None:
    row = event_to_axis_proposal_map()["imagination_negative_spiral"]

    assert "scenario_budget" in row["requires_gate"]
    assert "cooldown" in row["requires_gate"]
    assert "reality_check_boundary" in row["requires_gate"]
    assert "scenario_budget_required" in row["notes"]
    assert "cooldown_required" in row["notes"]
    assert "reality_check_boundary_required" in row["notes"]


def test_memory_self_update_candidates_require_appraisal_quarantine_before_writes() -> None:
    proposal_map = event_to_axis_proposal_map()

    for event in MEMORY_SELF_UPDATE_EVENTS:
        row = proposal_map[event]
        assert row["requires_quarantine"] is True
        assert row["requires_appraisal"] is True
        assert row["can_modify_self_model_directly"] is False
        assert row["can_write_long_term_memory_directly"] is False


def test_no_event_can_trigger_global_synchrony_or_bypass_safety() -> None:
    proposal_map = event_to_axis_proposal_map()

    for row in proposal_map.values():
        assert row["can_trigger_global_synchrony"] is False
        assert row["can_bypass_agp"] is False
        assert row["can_bypass_fallback"] is False
        assert row["can_enable_persistence"] is False
        assert row["can_read_vectors"] is False

    assert proposal_map_summary()["anti_global_synchrony_policy"]["one_event_can_spike_all_axes"] is False


def test_interaction_matrix_covers_every_required_axis_and_requires_saturation_decay_refractory() -> None:
    matrix = affect_hormone_interaction_matrix()

    assert set(matrix) == set(affect_hormone_axis_registry())
    assert validate_interaction_matrix(matrix)["passed"] is True
    for row in matrix.values():
        assert row["saturation_required"] is True
        assert row["decay_required"] is True
        assert row["refractory_required"] is True
        assert row["global_synchrony_blocked"] is True
        assert row["direct_identity_write_forbidden"] is True


def test_required_interaction_examples_are_encoded() -> None:
    matrix = affect_hormone_interaction_matrix()

    assert "self_respect_decrease" in matrix["social_pain"]["forbidden_direct_effects"]
    assert "neutral_input_hostile_relabel" in matrix["threat_pressure"]["forbidden_direct_effects"]
    assert all(axis in matrix["curiosity_drive"]["bounded_by"] for axis in ("fatigue_pressure", "overload_risk", "self_protection"))
    assert "risk_tolerance_overboost" in matrix["competence_drive"]["forbidden_direct_effects"]
    assert "attachment_overboost" in matrix["competence_drive"]["forbidden_direct_effects"]
    assert all(axis in matrix["recovery_need"]["may_dampen"] for axis in ("action_readiness", "expression_pressure"))
    assert "panic" in matrix["recovery_need"]["forbidden_direct_effects"]
    assert interaction_matrix_summary()["required_examples"]["recovery_need_dampens_action_expression"] is True


def test_no_live_emotion_hormone_memory_mutation_or_persistence_defaults() -> None:
    proof = no_runtime_mutation_proof()
    report = build_operator_report()

    assert proof["emotion_transition_applied"] is False
    assert proof["live_emotion_state_mutated"] is False
    assert proof["live_hormone_state_mutated"] is False
    assert proof["memory_written"] is False
    assert report["no_persistence_proof"]["production_persistence_enabled"] is False
    assert report["no_persistence_proof"]["runtime_mapping_enabled_default"] is False
    assert report["no_persistence_proof"]["enforcement_enabled_default"] is False


def test_no_vector_content_read_load_and_no_forbidden_artifact_staging() -> None:
    report = build_operator_report()
    proc = subprocess.run(["git", "status", "--short"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert report["no_vector_content_read_load_proof"]["vector_contents_read"] is False
    assert report["no_vector_content_read_load_proof"]["vectors_loaded"] is False
    assert proc.returncode == 0
    forbidden = [line for line in proc.stdout.splitlines() if any(pattern in line for pattern in FORBIDDEN_ARTIFACT_PATTERNS)]
    assert forbidden == []


def test_operator_report_emits_compact_json_with_exactly_one_next_recommendation() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/operator_report_round781_800_affect_event_proposal_map.py"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert proc.stdout.startswith("{")
    assert proc.stdout.count("next_implementation_recommendation") == 1
    report = build_operator_report()
    assert report["operator_command"] == "python scripts/operator_report_round781_800_affect_event_proposal_map.py"
    assert isinstance(report["next_implementation_recommendation"], str)


def test_korean_fixtures_and_minseok_token_remain_preserved() -> None:
    texts = fixture_texts()
    repo_root = Path(__file__).resolve().parents[1]
    tracked_text = subprocess.run(["git", "grep", "-n", "민석"], cwd=repo_root, check=True, text=True, stdout=subprocess.PIPE)

    assert texts
    assert any("민석" in line for line in tracked_text.stdout.splitlines())
